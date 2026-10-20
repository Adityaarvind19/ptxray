// cudatrace interposition layer.
//
// Built as libcudatrace.so and injected with LD_PRELOAD. Because a preloaded
// library comes first in the dynamic linker's global lookup scope, every call
// a program (or a library such as libtorch_cuda.so) makes into the shared CUDA
// runtime (libcudart.so) through its PLT binds to the functions defined here.
// Each hook records what happened and forwards to the real function, which is
// found with dlsym(RTLD_NEXT) or, if libcudart was dlopen()ed with RTLD_LOCAL,
// with dlopen(RTLD_NOLOAD) on its soname.
//
// GPU-side durations come from CUDA events recorded on the same stream right
// before and after the operation. Events are resolved lazily (at synchronize
// calls and at exit) so the tracer never adds a synchronization of its own.
//
// Limits: programs linked against the *static* runtime (nvcc's default,
// -cudart static) cannot be interposed this way; build with -cudart shared.
// Work launched directly through the driver API (cuLaunchKernel), e.g. by
// cuBLAS/cuDNN/Triton, is not seen by runtime-API hooks.

#include <cuda_runtime_api.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core.hpp"

#define CT_EXPORT extern "C" __attribute__((visibility("default")))

// Per-thread-default-stream variants (used when a program is compiled with
// --default-stream per-thread). They are only declared by the header under a
// macro, so declare them here.
extern "C" {
// Registration entry points that nvcc-generated host code calls from static
// initializers (declared in nvcc's internal crt/host_runtime.h).
void** CUDARTAPI __cudaRegisterFatBinary(void* fatCubin);
void CUDARTAPI __cudaRegisterFunction(void** handle, const char* hostFun, char* deviceFun, const char* deviceName,
                                      int thread_limit, uint3* tid, uint3* bid, dim3* bDim, dim3* gDim, int* wSize);
// Kernel launch path used by nvcc 12.8+/13 generated code: the host stub calls
// __cudaGetKernel once to turn itself into a cudaKernel_t handle, then
// __cudaLaunchKernel(handle, ...) for every <<<...>>> (declared in nvcc's
// internal crt/device_functions.h). Older nvcc calls cudaLaunchKernel.
cudaError_t CUDARTAPI __cudaGetKernel(cudaKernel_t* kernel, const void* host_stub);
cudaError_t CUDARTAPI __cudaLaunchKernel(cudaKernel_t, dim3, dim3, void**, size_t, cudaStream_t);
cudaError_t CUDARTAPI __cudaLaunchKernel_ptsz(cudaKernel_t, dim3, dim3, void**, size_t, cudaStream_t);
cudaError_t CUDARTAPI cudaLaunchKernel_ptsz(const void*, dim3, dim3, void**, size_t, cudaStream_t);
cudaError_t CUDARTAPI cudaMemcpyAsync_ptsz(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t CUDARTAPI cudaMemcpy_ptds(void*, const void*, size_t, cudaMemcpyKind);
}

namespace {

// ---------------------------------------------------------------- clock, ids

uint64_t mono_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}
// Everything below that has dynamic initialization is created on first use
// (function-local statics): CUDA libraries loaded alongside the program can
// run their static initializers, and so call our registration hooks, before
// this library's own global constructors have run.
uint64_t t0_ns() {
  static const uint64_t t = mono_ns();
  return t;
}
__attribute__((constructor)) void start_clock() { t0_ns(); }
uint64_t now_us() { return (mono_ns() - t0_ns()) / 1000; }

uint32_t this_tid() {
  static thread_local uint32_t tid = static_cast<uint32_t>(syscall(SYS_gettid));
  return tid;
}

// ------------------------------------------------------------------ settings

struct Settings {
  bool disabled = false;
  bool gpu_timing = true;
  std::string trace_path;   // empty = do not write
  std::string report_path;  // empty = stderr
  size_t max_events = 1000000;
  size_t max_pending = 1 << 16;
  std::string ptx_dir;  // CUDATRACE_DUMP_PTX: write embedded PTX here
};

Settings load_settings() {
  Settings s;
  auto env = [](const char* k) -> const char* { const char* v = getenv(k); return v && *v ? v : nullptr; };
  auto falsy = [](const char* v) { return !strcmp(v, "0") || !strcmp(v, "off") || !strcmp(v, "false"); };
  if (const char* v = env("CUDATRACE_DISABLE")) s.disabled = !falsy(v);
  if (const char* v = env("CUDATRACE_GPU_TIMING")) s.gpu_timing = !falsy(v);
  if (const char* v = env("CUDATRACE_MAX_EVENTS")) s.max_events = strtoull(v, nullptr, 10);
  const char* t = env("CUDATRACE_TRACE");
  if (!t) s.trace_path = "cudatrace-%p.json";
  else if (!falsy(t)) s.trace_path = t;
  if (const char* v = env("CUDATRACE_REPORT")) s.report_path = v;
  if (const char* v = env("CUDATRACE_DUMP_PTX")) s.ptx_dir = v;
  // Expand %p to the pid so child processes do not overwrite each other.
  for (std::string* p : {&s.trace_path, &s.report_path}) {
    size_t i = p->find("%p");
    if (i != std::string::npos) p->replace(i, 2, std::to_string(getpid()));
  }
  return s;
}

// All process-lifetime state below is heap-allocated and never freed: function
// statics and globals with destructors would be torn down before our ELF
// destructor (which writes the report) runs.
const Settings& settings() {
  static const Settings* s = new Settings(load_settings());
  return *s;
}

ct::Tracer& tracer() {
  // Intentionally leaked: it must outlive every static destructor and atexit
  // handler that might still call into CUDA while the process exits.
  static ct::Tracer* t = [] {
    ct::Config c;
    c.max_trace_events = settings().max_events;
    return new ct::Tracer(c);
  }();
  return *t;
}

// ------------------------------------------------------- resolving real calls

void* lookup(const char* name) {
  if (void* p = dlsym(RTLD_NEXT, name)) return p;
  static const char* sonames[] = {"libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so"};
  for (const char* so : sonames)  // already loaded, but in a local scope?
    if (void* h = dlopen(so, RTLD_LAZY | RTLD_NOLOAD))
      if (void* p = dlsym(h, name)) return p;
  for (const char* so : sonames)  // last resort: load it ourselves
    if (void* h = dlopen(so, RTLD_LAZY | RTLD_GLOBAL))
      if (void* p = dlsym(h, name)) return p;
  return nullptr;
}

#define REAL(fn)                                                                    \
  ([]() -> decltype(&::fn) {                                                         \
    static decltype(&::fn) p = reinterpret_cast<decltype(&::fn)>(lookup(#fn));       \
    return p;                                                                        \
  }())

cudaError_t missing(const char* name) {
  static std::atomic<bool> warned{false};
  if (!warned.exchange(true))
    fprintf(stderr, "==cudatrace== error: cannot find the real %s in libcudart\n", name);
  return cudaErrorSharedObjectSymbolNotFound;
}

// Re-entrancy guard: only the outermost intercepted call on a thread is traced.
thread_local int t_depth = 0;
struct Scope {
  bool outer;
  Scope() : outer(t_depth++ == 0) {}
  ~Scope() { --t_depth; }
};
bool active() { return !settings().disabled; }

std::atomic<bool> g_used{false};    // anything worth reporting happened
std::atomic<bool> g_called{false};  // the program made a real CUDA runtime call
void at_exit_drain();

void after_first_call() {
  g_used = true;
  g_called = true;
  // Registered after the first real CUDA call, i.e. after libcudart registered
  // its own teardown, so (atexit being LIFO) ours runs while the context is alive.
  static std::once_flag once;
  std::call_once(once, [] { atexit(at_exit_drain); });
}

// A forked child (e.g. a PyTorch DataLoader worker) inherits our state; it
// only reports if it makes CUDA calls of its own.
__attribute__((constructor)) void install_fork_handler() {
  pthread_atfork(nullptr, nullptr, [] { g_used = false; g_called = false; });
}

// Our own internal CUDA calls must not leave an error behind for the
// application's cudaGetLastError(). Only clear when the slate was clean.
struct ErrorStateGuard {
  bool was_clean;
  ErrorStateGuard() {
    auto peek = REAL(cudaPeekAtLastError);
    was_clean = peek && peek() == cudaSuccess;
  }
  ~ErrorStateGuard() {
    auto get = REAL(cudaGetLastError);
    if (was_clean && get) get();
  }
};

std::string err_name(cudaError_t e) {
  auto f = REAL(cudaGetErrorName);
  const char* s = f ? f(e) : nullptr;
  return s ? s : ("cudaError " + std::to_string(static_cast<int>(e)));
}

// ----------------------------------------------------------- symbolization

std::string demangle(const char* s) {
  if (!s) return {};
  int st = 0;
  char* d = abi::__cxa_demangle(s, nullptr, nullptr, &st);
  if (st == 0 && d) { std::string r(d); std::free(d); return r; }
  std::free(d);
  return s;
}

std::string site_of(void* ra) {
  if (!ra) return "?";
  void* pc = static_cast<char*>(ra) - 1;  // inside the call instruction
  Dl_info info;
  char buf[512];
  if (dladdr(pc, &info) && info.dli_fname) {
    const char* obj = strrchr(info.dli_fname, '/');
    obj = obj ? obj + 1 : info.dli_fname;
    uintptr_t off = reinterpret_cast<uintptr_t>(pc) - reinterpret_cast<uintptr_t>(info.dli_fbase);
    if (info.dli_sname) {
      uintptr_t soff = reinterpret_cast<uintptr_t>(pc) - reinterpret_cast<uintptr_t>(info.dli_saddr);
      snprintf(buf, sizeof buf, "%s+0x%" PRIxPTR " (%s+0x%" PRIxPTR ")",
               ct::shorten(demangle(info.dli_sname), 80).c_str(), soff, obj, off);
    } else {
      snprintf(buf, sizeof buf, "%s+0x%" PRIxPTR, obj, off);
    }
    return buf;
  }
  snprintf(buf, sizeof buf, "%p", pc);
  return buf;
}

// ------------------------------------------------- code-object registration

struct RegisteredKernel {
  std::string mangled;
  int module = 0;
};
std::mutex g_reg_mu;
std::unordered_map<const void*, RegisteredKernel>& kernels_map() {  // host stub -> kernel
  static auto* m = new std::unordered_map<const void*, RegisteredKernel>();
  return *m;
}
std::unordered_map<void**, int>& handles_map() {  // fatbin handle -> module id
  static auto* m = new std::unordered_map<void**, int>();
  return *m;
}

std::string object_of(const void* addr) {
  Dl_info info;
  if (dladdr(addr, &info) && info.dli_fname && *info.dli_fname) {
    const char* o = strrchr(info.dli_fname, '/');
    return o ? o + 1 : info.dli_fname;
  }
  return "?";
}

void register_module(const void* wrapper, void** handle) {
  ct::Fatbin fb = ct::parse_registration(wrapper);
  std::string obj = object_of(wrapper);
  // Decode PTX headers for small modules (cheap); always when dumping.
  static int decoded = 0;
  const Settings& st = settings();
  for (auto& im : fb.images) {
    if (!im.is_ptx()) continue;
    bool want = !st.ptx_dir.empty() || (decoded < 64 && im.uncompressed_size < (4u << 20));
    if (!want) continue;
    std::string text, err;
    if (!ct::decode_ptx(im, &text, &err)) continue;
    ++decoded;
    ct::parse_ptx_header(text, &im);
    if (!st.ptx_dir.empty()) {
      char path[1024];
      snprintf(path, sizeof path, "%s/%s.m%zu.compute_%d.ptx", st.ptx_dir.c_str(), obj.c_str(),
               tracer().module_count() + 1, im.arch);
      if (FILE* f = fopen(path, "w")) {
        fwrite(text.data(), 1, text.size(), f);
        fclose(f);
        tracer().note_ptx_dump(path);
      }
    }
  }
  int id = tracer().add_module(obj, fb);
  std::lock_guard<std::mutex> lk(g_reg_mu);
  handles_map()[handle] = id;
  g_used = true;
}

void register_function(void** handle, const void* host_stub, const char* device_fun) {
  std::lock_guard<std::mutex> lk(g_reg_mu);
  auto h = handles_map().find(handle);
  int module = h == handles_map().end() ? 0 : h->second;
  kernels_map()[host_stub] = {device_fun, module};
  if (module) tracer().add_module_kernel(module, device_fun);
}

std::unordered_map<const void*, const void*>& kernel_handles() {  // cudaKernel_t -> host stub
  static auto* m = new std::unordered_map<const void*, const void*>();
  return *m;
}

const void* stub_of(cudaKernel_t k) {
  std::lock_guard<std::mutex> lk(g_reg_mu);
  auto it = kernel_handles().find(k);
  return it == kernel_handles().end() ? static_cast<const void*>(k) : it->second;
}

// Host time spent in __cudaGetKernel per kernel. With nvcc 12.8+/13 code this
// is where lazy module loading, and therefore the driver's PTX JIT, happens:
// the stub calls it once, on the kernel's first launch, before the launch call.
std::unordered_map<const void*, uint64_t>& load_us() {
  static auto* m = new std::unordered_map<const void*, uint64_t>();
  return *m;
}

uint64_t take_load_us(const void* host_stub) {
  std::lock_guard<std::mutex> lk(g_reg_mu);
  auto it = load_us().find(host_stub);
  if (it == load_us().end()) return 0;
  uint64_t us = it->second;
  load_us().erase(it);
  return us;
}

int module_of(const void* host_stub) {
  std::lock_guard<std::mutex> lk(g_reg_mu);
  auto it = kernels_map().find(host_stub);
  return it == kernels_map().end() ? 0 : it->second.module;
}

std::string kernel_name(const void* func) {
  static std::mutex mu;
  static auto& cache = *new std::unordered_map<const void*, std::string>();
  {
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(func);
    if (it != cache.end()) return it->second;
  }
  std::string name;
  {  // best source: the device function name nvcc registered for this stub
    std::lock_guard<std::mutex> lk(g_reg_mu);
    auto it = kernels_map().find(func);
    if (it != kernels_map().end()) name = demangle(it->second.mangled.c_str());
  }
  using GetName = cudaError_t (*)(const char**, const void*);
  static GetName get_name = reinterpret_cast<GetName>(lookup("cudaFuncGetName"));  // CUDA >= 12.3
  if (name.empty() && get_name) {
    ErrorStateGuard eg;
    const char* n = nullptr;
    if (get_name(&n, func) == cudaSuccess && n) name = demangle(n);
  }
  if (name.empty()) {  // fall back to the host stub's symbol (needs -rdynamic)
    Dl_info info;
    if (dladdr(func, &info) && info.dli_sname && info.dli_saddr == func) name = demangle(info.dli_sname);
  }
  if (name.empty()) {
    char buf[32];
    snprintf(buf, sizeof buf, "kernel@%p", func);
    name = buf;
  }
  std::lock_guard<std::mutex> lk(mu);
  return cache.emplace(func, name).first->second;
}

// ------------------------------------------------------------ stream lanes

uint32_t lane_of(cudaStream_t s) {
  static std::mutex mu;
  static auto& lanes = *new std::unordered_map<cudaStream_t, uint32_t>();
  static uint32_t next = 2;
  if (s == nullptr || s == cudaStreamLegacy) {
    static std::once_flag f;
    std::call_once(f, [] { tracer().name_lane(0, "stream 0 (legacy default)"); });
    return 0;
  }
  if (s == cudaStreamPerThread) {
    static std::once_flag f;
    std::call_once(f, [] { tracer().name_lane(1, "per-thread default stream"); });
    return 1;
  }
  std::lock_guard<std::mutex> lk(mu);
  auto it = lanes.find(s);
  if (it != lanes.end()) return it->second;
  uint32_t id = next++;
  lanes[s] = id;
  char buf[64];
  snprintf(buf, sizeof buf, "stream %u (%p)", id, static_cast<void*>(s));
  tracer().name_lane(id, buf);
  return id;
}

// ------------------------------------------------------ event-based timing

struct Pending {
  cudaEvent_t start = nullptr, stop = nullptr;
  ct::Cat cat = ct::Cat::Kernel;
  std::string name;
  uint32_t lane = 0;
  uint64_t bytes = 0;
  std::string args;
};

std::mutex g_pend_mu;
std::vector<Pending>& pending_list() {
  static auto* v = new std::vector<Pending>();
  return *v;
}
std::vector<cudaEvent_t>& event_pool() {
  static auto* v = new std::vector<cudaEvent_t>();
  return *v;
}

std::mutex g_clock_mu;
bool g_clock_ok = false;
cudaEvent_t g_base = nullptr;  // GPU timestamp that corresponds to host time g_base_us
uint64_t g_base_us = 0;

bool ensure_clock() {
  std::lock_guard<std::mutex> lk(g_clock_mu);
  if (g_clock_ok) return true;
  ErrorStateGuard eg;
  cudaEvent_t e;
  if (REAL(cudaEventCreate)(&e) != cudaSuccess) return false;
  if (REAL(cudaEventRecord)(e, nullptr) != cudaSuccess || REAL(cudaEventSynchronize)(e) != cudaSuccess) {
    REAL(cudaEventDestroy)(e);
    return false;
  }
  g_base_us = now_us();
  g_base = e;
  g_clock_ok = true;
  return true;
}

cudaEvent_t acquire_event() {  // requires g_pend_mu
  if (!event_pool().empty()) { cudaEvent_t e = event_pool().back(); event_pool().pop_back(); return e; }
  cudaEvent_t e = nullptr;
  if (REAL(cudaEventCreate)(&e) != cudaSuccess) return nullptr;
  return e;
}

void release(Pending& p) {  // requires g_pend_mu
  if (p.start) event_pool().push_back(p.start);
  if (p.stop) event_pool().push_back(p.stop);
  p.start = p.stop = nullptr;
}

// Records the start event. Returns false (and the op stays untimed) when timing
// is off, the stream is being captured into a CUDA graph, or the pending budget
// is exhausted (we never force a sync to make room).
bool begin_timed(cudaStream_t s, Pending& p) {
  if (!settings().gpu_timing || !ensure_clock()) return false;
  ErrorStateGuard eg;
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  if (REAL(cudaStreamIsCapturing)(s, &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone) return false;
  std::lock_guard<std::mutex> lk(g_pend_mu);
  if (pending_list().size() >= settings().max_pending) return false;
  p.start = acquire_event();
  p.stop = acquire_event();
  if (!p.start || !p.stop || REAL(cudaEventRecord)(p.start, s) != cudaSuccess) { release(p); return false; }
  return true;
}

void end_timed(cudaStream_t s, Pending& p, bool op_ok) {
  ErrorStateGuard eg;
  std::lock_guard<std::mutex> lk(g_pend_mu);
  if (!op_ok || REAL(cudaEventRecord)(p.stop, s) != cudaSuccess) { release(p); return; }
  pending_list().push_back(std::move(p));
}

// Turn completed event pairs into GPU timeline events. Non-blocking unless
// `block` (used at exit).
void drain(bool block) {
  if (!g_clock_ok) return;
  std::vector<Pending> todo, keep;
  {
    std::lock_guard<std::mutex> lk(g_pend_mu);
    todo.swap(pending_list());
  }
  if (todo.empty()) return;
  ErrorStateGuard eg;
  std::vector<Pending> done;
  for (auto& p : todo) {
    cudaError_t st = block ? REAL(cudaEventSynchronize)(p.stop) : REAL(cudaEventQuery)(p.stop);
    if (st == cudaErrorNotReady) { keep.push_back(std::move(p)); continue; }
    float at_ms = 0, dur_ms = 0;
    if (st == cudaSuccess && REAL(cudaEventElapsedTime)(&at_ms, g_base, p.start) == cudaSuccess &&
        REAL(cudaEventElapsedTime)(&dur_ms, p.start, p.stop) == cudaSuccess) {
      int64_t ts = static_cast<int64_t>(g_base_us) + static_cast<int64_t>(at_ms * 1000.0);
      tracer().gpu_op(p.name, p.cat, p.lane, ts < 0 ? 0 : ts, static_cast<uint64_t>(dur_ms * 1000.0 + 0.5),
                      p.bytes, p.args);
    } else {
      tracer().untimed_op(p.cat);
    }
    done.push_back(std::move(p));
  }
  std::lock_guard<std::mutex> lk(g_pend_mu);
  for (auto& p : done) release(p);
  for (auto& p : keep) pending_list().push_back(std::move(p));
}

void at_exit_drain() {
  if (!g_called) return;
  drain(true);
  // Record the device's compute capability for the image-selection report.
  int dev = 0, major = 0, minor = 0;
  ErrorStateGuard eg;
  if (REAL(cudaGetDevice)(&dev) == cudaSuccess &&
      REAL(cudaDeviceGetAttribute)(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
      REAL(cudaDeviceGetAttribute)(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess)
    tracer().set_device(major * 10 + minor, "");
}

// ------------------------------------------------------------ hook helpers

void record_call(const char* api, ct::Cat cat, uint64_t t0, uint64_t t1, cudaError_t r, void* ra,
                 const std::string& args = "") {
  std::string a = args;
  if (r != cudaSuccess) {
    std::string e = err_name(r);
    tracer().api_error(api, e, site_of(ra));
    a += std::string(a.empty() ? "" : ",") + "\"error\":\"" + ct::json_escape(e) + "\"";
  }
  tracer().host_call(api, cat, t0, t1 - t0, this_tid(), a);
}

cudaError_t on_alloc(const char* api, cudaError_t r, void* ptr, uint64_t size, const char* kind,
                     bool device_mem, uint64_t t0, uint64_t t1, void* ra) {
  char args[96];
  snprintf(args, sizeof args, "\"bytes\":%" PRIu64 ",\"ptr\":\"%p\"", size, ptr);
  record_call(api, ct::Cat::Alloc, t0, t1, r, ra, args);
  if (r == cudaSuccess && ptr)
    tracer().alloc(reinterpret_cast<uint64_t>(ptr), size, kind, device_mem, site_of(ra), t1);
  after_first_call();
  return r;
}

cudaError_t on_free(const char* api, cudaError_t r, void* ptr, uint64_t t0, uint64_t t1, void* ra) {
  char args[48];
  snprintf(args, sizeof args, "\"ptr\":\"%p\"", ptr);
  // During process teardown libcudart refuses calls but the context (and all
  // its memory) is being destroyed anyway, so count that as a successful free.
  bool ok = r == cudaSuccess || r == cudaErrorCudartUnloading;
  if (r == cudaErrorCudartUnloading) r = cudaSuccess;  // do not report as an error
  record_call(api, ct::Cat::Free, t0, t1, ok ? cudaSuccess : r, ra, args);
  if (ptr) tracer().free(reinterpret_cast<uint64_t>(ptr), api, ok, site_of(ra), t1);
  after_first_call();
  return r;
}

enum class Mem { Host, Device, Unknown };
struct PtrInfo { Mem where; bool pinned; };

PtrInfo classify(const void* p) {
  cudaPointerAttributes a;
  ErrorStateGuard eg;
  if (!p || REAL(cudaPointerGetAttributes)(&a, p) != cudaSuccess) return {Mem::Unknown, false};
  switch (a.type) {
    case cudaMemoryTypeDevice:
    case cudaMemoryTypeManaged: return {Mem::Device, false};
    case cudaMemoryTypeHost: return {Mem::Host, true};
    default: return {Mem::Host, false};  // cudaMemoryTypeUnregistered: pageable
  }
}

template <class Call>
cudaError_t memcpy_common(const char* api, bool async, void* dst, const void* src, size_t n,
                          cudaMemcpyKind kind, cudaStream_t s, void* ra, Call call) {
  uint64_t t0 = now_us();
  // Direction and, for host buffers, whether they are pinned.
  const void* host = nullptr;
  const char* dir = "Memcpy";
  switch (kind) {
    case cudaMemcpyHostToDevice: dir = "HtoD"; host = src; break;
    case cudaMemcpyDeviceToHost: dir = "DtoH"; host = dst; break;
    case cudaMemcpyDeviceToDevice: dir = "DtoD"; break;
    case cudaMemcpyHostToHost: dir = "HtoH"; break;
    default: {
      PtrInfo d = classify(dst), sr = classify(src);
      bool dd = d.where == Mem::Device, sd = sr.where == Mem::Device;
      dir = sd ? (dd ? "DtoD" : "DtoH") : (dd ? "HtoD" : "HtoH");
      host = (sd && !dd) ? dst : (!sd && dd) ? src : nullptr;
    }
  }
  bool pageable = false;
  std::string name = std::string("Memcpy ") + dir;
  if (host) {
    pageable = !classify(host).pinned;
    name += pageable ? " (pageable)" : " (pinned)";
  }
  Pending p;
  p.cat = ct::Cat::Memcpy;
  p.name = name;
  p.lane = lane_of(s);
  p.bytes = n;
  p.args = "\"bytes\":" + std::to_string(n);
  bool timed = begin_timed(s, p);
  cudaError_t r = call();
  if (timed) end_timed(s, p, r == cudaSuccess);
  else if (r == cudaSuccess) tracer().untimed_op(ct::Cat::Memcpy);
  uint64_t t1 = now_us();
  if (!async) tracer().host_blocked(t1 - t0);
  if (r == cudaSuccess) tracer().memcpy_issued(n, async, pageable, site_of(ra));
  record_call(api, ct::Cat::Memcpy, t0, t1, r, ra, "\"bytes\":" + std::to_string(n) + ",\"kind\":\"" + name + "\"");
  after_first_call();
  return r;
}

// host_stub identifies the kernel (name, module); call() performs the real launch.
template <class Call>
cudaError_t launch_common(const char* api, const void* host_stub, dim3 g, dim3 b, size_t smem, cudaStream_t s,
                          void* ra, Call call) {
  uint64_t t0 = now_us();
  Pending p;
  p.cat = ct::Cat::Kernel;
  p.name = kernel_name(host_stub);
  p.lane = lane_of(s);
  char a[160];
  snprintf(a, sizeof a, "\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"smem\":%zu", g.x, g.y, g.z, b.x, b.y,
           b.z, smem);
  p.args = a;
  const std::string name = p.name;  // p is moved into the pending list below
  bool timed = begin_timed(s, p);
  cudaError_t r = call();
  if (timed) end_timed(s, p, r == cudaSuccess);
  else if (r == cudaSuccess) tracer().untimed_op(ct::Cat::Kernel);
  uint64_t t1 = now_us();
  // A kernel's first launch also pays for loading its module (and JIT-compiling
  // it from PTX if needed), whether that happened inside this call (older nvcc)
  // or just before it in __cudaGetKernel (nvcc 12.8+/13).
  if (r == cudaSuccess) tracer().kernel_launch(name, t1 - t0 + take_load_us(host_stub), module_of(host_stub));
  record_call(api, ct::Cat::Kernel, t0, t1, r, ra, "\"kernel\":\"" + ct::json_escape(name) + "\"," + a);
  after_first_call();
  return r;
}

template <class Call>
cudaError_t sync_common(const char* api, void* ra, Call call) {
  uint64_t t0 = now_us();
  cudaError_t r = call();
  uint64_t t1 = now_us();
  tracer().host_blocked(t1 - t0);
  record_call(api, ct::Cat::Sync, t0, t1, r, ra);
  after_first_call();
  drain(false);  // work has finished; cheap to resolve now
  return r;
}

}  // namespace

// ================================================================= the hooks

#define RA __builtin_return_address(0)
#define PASSTHRU(fn, ...)                       \
  auto real = REAL(fn);                         \
  if (!real) return missing(#fn);               \
  Scope sc;                                     \
  if (!sc.outer || !active()) return real(__VA_ARGS__)

// ---- code-object registration (runs from static initializers, before main)

CT_EXPORT void** CUDARTAPI __cudaRegisterFatBinary(void* fat) {
  auto real = REAL(__cudaRegisterFatBinary);
  if (!real) { missing("__cudaRegisterFatBinary"); return nullptr; }
  void** handle = real(fat);
  if (active()) register_module(fat, handle);
  return handle;
}

CT_EXPORT void CUDARTAPI __cudaRegisterFunction(void** handle, const char* host_fun, char* device_fun,
                                                const char* device_name, int thread_limit, uint3* tid,
                                                uint3* bid, dim3* bdim, dim3* gdim, int* wsize) {
  auto real = REAL(__cudaRegisterFunction);
  if (!real) { missing("__cudaRegisterFunction"); return; }
  real(handle, host_fun, device_fun, device_name, thread_limit, tid, bid, bdim, gdim, wsize);
  if (active() && device_fun) register_function(handle, host_fun, device_fun);
}

// ---- allocation

CT_EXPORT cudaError_t CUDARTAPI cudaMalloc(void** p, size_t n) {
  PASSTHRU(cudaMalloc, p, n);
  uint64_t t0 = now_us();
  cudaError_t r = real(p, n);
  return on_alloc("cudaMalloc", r, p ? *p : nullptr, n, "device", true, t0, now_us(), RA);
}

CT_EXPORT cudaError_t CUDARTAPI cudaMallocManaged(void** p, size_t n, unsigned int flags) {
  PASSTHRU(cudaMallocManaged, p, n, flags);
  uint64_t t0 = now_us();
  cudaError_t r = real(p, n, flags);
  return on_alloc("cudaMallocManaged", r, p ? *p : nullptr, n, "managed", true, t0, now_us(), RA);
}

CT_EXPORT cudaError_t CUDARTAPI cudaMallocPitch(void** p, size_t* pitch, size_t w, size_t h) {
  PASSTHRU(cudaMallocPitch, p, pitch, w, h);
  uint64_t t0 = now_us();
  cudaError_t r = real(p, pitch, w, h);
  uint64_t bytes = (r == cudaSuccess && pitch) ? static_cast<uint64_t>(*pitch) * h : 0;
  return on_alloc("cudaMallocPitch", r, p ? *p : nullptr, bytes, "device", true, t0, now_us(), RA);
}

CT_EXPORT cudaError_t CUDARTAPI cudaMalloc3D(cudaPitchedPtr* pp, cudaExtent e) {
  PASSTHRU(cudaMalloc3D, pp, e);
  uint64_t t0 = now_us();
  cudaError_t r = real(pp, e);
  uint64_t bytes = (r == cudaSuccess && pp) ? static_cast<uint64_t>(pp->pitch) * e.height * e.depth : 0;
  return on_alloc("cudaMalloc3D", r, pp ? pp->ptr : nullptr, bytes, "device", true, t0, now_us(), RA);
}

CT_EXPORT cudaError_t CUDARTAPI cudaMallocAsync(void** p, size_t n, cudaStream_t s) {
  PASSTHRU(cudaMallocAsync, p, n, s);
  uint64_t t0 = now_us();
  cudaError_t r = real(p, n, s);
  return on_alloc("cudaMallocAsync", r, p ? *p : nullptr, n, "device-async", true, t0, now_us(), RA);
}

CT_EXPORT cudaError_t CUDARTAPI cudaMallocFromPoolAsync(void** p, size_t n, cudaMemPool_t pool, cudaStream_t s) {
  PASSTHRU(cudaMallocFromPoolAsync, p, n, pool, s);
  uint64_t t0 = now_us();
  cudaError_t r = real(p, n, pool, s);
  return on_alloc("cudaMallocFromPoolAsync", r, p ? *p : nullptr, n, "device-async", true, t0, now_us(), RA);
}

CT_EXPORT cudaError_t CUDARTAPI cudaMallocHost(void** p, size_t n) {
  PASSTHRU(cudaMallocHost, p, n);
  uint64_t t0 = now_us();
  cudaError_t r = real(p, n);
  return on_alloc("cudaMallocHost", r, p ? *p : nullptr, n, "pinned-host", false, t0, now_us(), RA);
}

CT_EXPORT cudaError_t CUDARTAPI cudaHostAlloc(void** p, size_t n, unsigned int flags) {
  PASSTHRU(cudaHostAlloc, p, n, flags);
  uint64_t t0 = now_us();
  cudaError_t r = real(p, n, flags);
  return on_alloc("cudaHostAlloc", r, p ? *p : nullptr, n, "pinned-host", false, t0, now_us(), RA);
}

CT_EXPORT cudaError_t CUDARTAPI cudaFree(void* p) {
  PASSTHRU(cudaFree, p);
  uint64_t t0 = now_us();
  cudaError_t r = real(p);
  return on_free("cudaFree", r, p, t0, now_us(), RA);
}

CT_EXPORT cudaError_t CUDARTAPI cudaFreeAsync(void* p, cudaStream_t s) {
  PASSTHRU(cudaFreeAsync, p, s);
  uint64_t t0 = now_us();
  cudaError_t r = real(p, s);
  return on_free("cudaFreeAsync", r, p, t0, now_us(), RA);
}

CT_EXPORT cudaError_t CUDARTAPI cudaFreeHost(void* p) {
  PASSTHRU(cudaFreeHost, p);
  uint64_t t0 = now_us();
  cudaError_t r = real(p);
  return on_free("cudaFreeHost", r, p, t0, now_us(), RA);
}

// ---- transfers

CT_EXPORT cudaError_t CUDARTAPI cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind k) {
  PASSTHRU(cudaMemcpy, d, s, n, k);
  return memcpy_common("cudaMemcpy", false, d, s, n, k, nullptr, RA, [&] { return real(d, s, n, k); });
}

CT_EXPORT cudaError_t CUDARTAPI cudaMemcpy_ptds(void* d, const void* s, size_t n, cudaMemcpyKind k) {
  PASSTHRU(cudaMemcpy_ptds, d, s, n, k);
  return memcpy_common("cudaMemcpy", false, d, s, n, k, cudaStreamPerThread, RA,
                       [&] { return real(d, s, n, k); });
}

CT_EXPORT cudaError_t CUDARTAPI cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind k,
                                                cudaStream_t st) {
  PASSTHRU(cudaMemcpyAsync, d, s, n, k, st);
  return memcpy_common("cudaMemcpyAsync", true, d, s, n, k, st, RA, [&] { return real(d, s, n, k, st); });
}

CT_EXPORT cudaError_t CUDARTAPI cudaMemcpyAsync_ptsz(void* d, const void* s, size_t n, cudaMemcpyKind k,
                                                     cudaStream_t st) {
  PASSTHRU(cudaMemcpyAsync_ptsz, d, s, n, k, st);
  return memcpy_common("cudaMemcpyAsync", true, d, s, n, k, st ? st : cudaStreamPerThread, RA,
                       [&] { return real(d, s, n, k, st); });
}

template <class Call>
static cudaError_t memset_common(const char* api, size_t n, cudaStream_t s, void* ra, Call call) {
  uint64_t t0 = now_us();
  Pending p;
  p.cat = ct::Cat::Memset;
  p.name = "Memset";
  p.lane = lane_of(s);
  p.bytes = n;
  bool timed = begin_timed(s, p);
  cudaError_t r = call();
  if (timed) end_timed(s, p, r == cudaSuccess);
  record_call(api, ct::Cat::Memset, t0, now_us(), r, ra, "\"bytes\":" + std::to_string(n));
  after_first_call();
  return r;
}

CT_EXPORT cudaError_t CUDARTAPI cudaMemset(void* d, int v, size_t n) {
  PASSTHRU(cudaMemset, d, v, n);
  return memset_common("cudaMemset", n, nullptr, RA, [&] { return real(d, v, n); });
}

CT_EXPORT cudaError_t CUDARTAPI cudaMemsetAsync(void* d, int v, size_t n, cudaStream_t s) {
  PASSTHRU(cudaMemsetAsync, d, v, n, s);
  return memset_common("cudaMemsetAsync", n, s, RA, [&] { return real(d, v, n, s); });
}

// ---- kernel launches (what <<<...>>> compiles to)

CT_EXPORT cudaError_t CUDARTAPI cudaLaunchKernel(const void* f, dim3 g, dim3 b, void** a, size_t smem,
                                                 cudaStream_t s) {
  PASSTHRU(cudaLaunchKernel, f, g, b, a, smem, s);
  return launch_common("cudaLaunchKernel", f, g, b, smem, s, RA, [&] { return real(f, g, b, a, smem, s); });
}

CT_EXPORT cudaError_t CUDARTAPI cudaLaunchKernel_ptsz(const void* f, dim3 g, dim3 b, void** a, size_t smem,
                                                      cudaStream_t s) {
  PASSTHRU(cudaLaunchKernel_ptsz, f, g, b, a, smem, s);
  return launch_common("cudaLaunchKernel", f, g, b, smem, s ? s : cudaStreamPerThread, RA,
                       [&] { return real(f, g, b, a, smem, s); });
}

// nvcc 12.8+/13 launch path.
CT_EXPORT cudaError_t CUDARTAPI __cudaGetKernel(cudaKernel_t* k, const void* host_stub) {
  PASSTHRU(__cudaGetKernel, k, host_stub);
  uint64_t t0 = now_us();
  cudaError_t r = real(k, host_stub);  // may load the module and JIT-compile its PTX
  uint64_t t1 = now_us();
  if (r == cudaSuccess && k && *k) {
    std::lock_guard<std::mutex> lk(g_reg_mu);
    kernel_handles()[*k] = host_stub;
    load_us()[host_stub] += t1 - t0;
  }
  record_call("__cudaGetKernel (module load)", ct::Cat::Other, t0, t1, r, RA,
              "\"kernel\":\"" + ct::json_escape(kernel_name(host_stub)) + "\"");
  after_first_call();
  return r;
}

CT_EXPORT cudaError_t CUDARTAPI __cudaLaunchKernel(cudaKernel_t k, dim3 g, dim3 b, void** a, size_t smem,
                                                   cudaStream_t s) {
  PASSTHRU(__cudaLaunchKernel, k, g, b, a, smem, s);
  return launch_common("cudaLaunchKernel", stub_of(k), g, b, smem, s, RA, [&] { return real(k, g, b, a, smem, s); });
}

CT_EXPORT cudaError_t CUDARTAPI __cudaLaunchKernel_ptsz(cudaKernel_t k, dim3 g, dim3 b, void** a, size_t smem,
                                                        cudaStream_t s) {
  PASSTHRU(__cudaLaunchKernel_ptsz, k, g, b, a, smem, s);
  return launch_common("cudaLaunchKernel", stub_of(k), g, b, smem, s ? s : cudaStreamPerThread, RA,
                       [&] { return real(k, g, b, a, smem, s); });
}

// ---- synchronization

CT_EXPORT cudaError_t CUDARTAPI cudaDeviceSynchronize(void) {
  PASSTHRU(cudaDeviceSynchronize);
  return sync_common("cudaDeviceSynchronize", RA, [&] { return real(); });
}

CT_EXPORT cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t s) {
  PASSTHRU(cudaStreamSynchronize, s);
  return sync_common("cudaStreamSynchronize", RA, [&] { return real(s); });
}

CT_EXPORT cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t e) {
  PASSTHRU(cudaEventSynchronize, e);
  return sync_common("cudaEventSynchronize", RA, [&] { return real(e); });
}

// ================================================================ shutdown

// Runs from the ELF destructor list, i.e. after main() returned, after atexit
// handlers (including at_exit_drain and libcudart's teardown) and after static
// destructors, so frees made during teardown are already accounted for.
__attribute__((destructor)) static void cudatrace_fini() {
  if (!g_used || settings().disabled) return;
  ct::Tracer& t = tracer();
  uint64_t wall = now_us();
  const Settings& s = settings();

  std::string rep = t.report(wall, getpid());
  if (!s.trace_path.empty()) {
    if (FILE* f = fopen(s.trace_path.c_str(), "w")) {
      t.write_trace(f, getpid());
      fclose(f);
      rep += "==cudatrace== timeline written to " + s.trace_path + " (open in https://ui.perfetto.dev)\n";
    } else {
      rep += "==cudatrace== could not write " + s.trace_path + ": " + strerror(errno) + "\n";
    }
  }
  FILE* out = stderr;
  if (!s.report_path.empty()) out = fopen(s.report_path.c_str(), "w");
  if (!out) out = stderr;
  fputs(rep.c_str(), out);
  fflush(out);
  if (out != stderr) fclose(out);
}
