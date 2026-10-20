#include "core.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdlib>
#include <cxxabi.h>

namespace ct {

namespace {

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, f);
  int n = vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  if (n < 0) return {};
  if (static_cast<size_t>(n) < sizeof buf) return std::string(buf, n);
  std::string s(n, '\0');
  va_start(ap, f);
  vsnprintf(&s[0], n + 1, f, ap);
  va_end(ap);
  return s;
}

template <class M>
std::vector<std::pair<std::string, OpStat>> by_total_desc(const M& m) {
  std::vector<std::pair<std::string, OpStat>> v(m.begin(), m.end());
  std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
    if (a.second.total_us != b.second.total_us) return a.second.total_us > b.second.total_us;
    return a.first < b.first;
  });
  return v;
}

}  // namespace

const char* cat_name(Cat c) {
  switch (c) {
    case Cat::Kernel: return "kernel";
    case Cat::Memcpy: return "memcpy";
    case Cat::Memset: return "memset";
    case Cat::Sync: return "sync";
    case Cat::Alloc: return "alloc";
    case Cat::Free: return "free";
    case Cat::Other: return "api";
  }
  return "api";
}

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) out += fmt("\\u%04x", c);
        else out += static_cast<char>(c);
    }
  }
  return out;
}

std::string demangle(const std::string& s) {
  int st = 0;
  char* d = abi::__cxa_demangle(s.c_str(), nullptr, nullptr, &st);
  std::string r = (st == 0 && d) ? std::string(d) : s;
  std::free(d);
  return r;
}

std::string human_bytes(uint64_t b) {
  const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double v = static_cast<double>(b);
  int u = 0;
  while (v >= 1024.0 && u < 4) { v /= 1024.0; ++u; }
  if (u == 0) return fmt("%" PRIu64 " B", b);
  return fmt("%.1f %s", v, units[u]);
}

std::string shorten(const std::string& s, size_t n) {
  if (s.size() <= n) return s;
  if (n < 4) return s.substr(0, n);
  return s.substr(0, n - 3) + "...";
}

Tracer::Tracer(Config c) : cfg_(c) {}

void Tracer::push_event(TraceEvent&& e) {
  if (events_.size() >= cfg_.max_trace_events) {
    truncated_ = true;
    return;
  }
  events_.push_back(std::move(e));
}

void Tracer::host_call(const std::string& api, Cat cat, uint64_t ts_us, uint64_t dur_us,
                       uint32_t tid, const std::string& args) {
  std::lock_guard<std::mutex> lk(mu_);
  host_stats_[api].add(static_cast<double>(dur_us));
  push_event({api, cat, false, ts_us, dur_us, tid, args});
}

void Tracer::gpu_op(const std::string& name, Cat cat, uint32_t lane, uint64_t ts_us,
                    uint64_t dur_us, uint64_t bytes, const std::string& args) {
  std::lock_guard<std::mutex> lk(mu_);
  if (cat == Cat::Kernel) kernel_stats_[name].add(static_cast<double>(dur_us));
  else copy_stats_[name].add(static_cast<double>(dur_us), bytes);
  push_event({name, cat, true, ts_us, dur_us, lane, args});
}

void Tracer::untimed_op(Cat cat) {
  std::lock_guard<std::mutex> lk(mu_);
  ++untimed_[static_cast<int>(cat)];
}

void Tracer::alloc(uint64_t ptr, uint64_t size, const std::string& kind, bool device_mem,
                   const std::string& site, uint64_t ts_us) {
  std::lock_guard<std::mutex> lk(mu_);
  ++n_allocs_;
  bytes_allocated_ += size;
  freed_.erase(ptr);
  auto it = live_.find(ptr);
  if (it != live_.end() && it->second.device_mem) dev_in_use_ -= it->second.size;  // stale entry
  live_[ptr] = Allocation{size, kind, device_mem, site, ts_us};
  if (device_mem) {
    dev_in_use_ += size;
    dev_peak_ = std::max(dev_peak_, dev_in_use_);
    mem_counter_.emplace_back(ts_us, dev_in_use_);
  }
}

FreeResult Tracer::free(uint64_t ptr, const std::string& api, bool real_ok,
                        const std::string& site, uint64_t ts_us) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = live_.find(ptr);
  if (it != live_.end()) {
    if (!real_ok) return FreeResult::Failed;  // allocation stays live
    ++n_frees_;
    if (it->second.device_mem) {
      dev_in_use_ -= it->second.size;
      mem_counter_.emplace_back(ts_us, dev_in_use_);
    }
    live_.erase(it);
    freed_.insert(ptr);
    return FreeResult::Freed;
  }
  FreeResult r;
  std::string kind;
  if (freed_.count(ptr)) { r = FreeResult::DoubleFree; kind = "double free"; ++double_frees_; }
  else if (!real_ok) { r = FreeResult::InvalidFree; kind = "invalid free"; ++invalid_frees_; }
  else { r = FreeResult::Untracked; kind = "untracked free"; ++untracked_frees_; }
  if (bad_free_examples_.size() < 8) bad_free_examples_.push_back({kind, api, site, ptr});
  return r;
}

void Tracer::memcpy_issued(uint64_t bytes, bool async, bool pageable_host, const std::string& site) {
  std::lock_guard<std::mutex> lk(mu_);
  ++total_copies_;
  if (bytes < cfg_.small_copy_bytes) ++small_copies_;
  if (async && pageable_host) {
    if (pageable_async_ == 0) pageable_site_ = site;
    ++pageable_async_;
    pageable_async_bytes_ += bytes;
  }
}

void Tracer::api_error(const std::string& api, const std::string& err, const std::string& site) {
  std::lock_guard<std::mutex> lk(mu_);
  auto& e = errors_[{api, err}];
  if (e.count++ == 0) e.first_site = site;
}

void Tracer::host_blocked(uint64_t us) {
  std::lock_guard<std::mutex> lk(mu_);
  blocked_us_ += us;
}

void Tracer::name_lane(uint32_t lane, const std::string& name) {
  std::lock_guard<std::mutex> lk(mu_);
  lane_names_[lane] = name;
}

bool Tracer::empty() const {
  std::lock_guard<std::mutex> lk(mu_);
  return host_stats_.empty();
}

uint64_t Tracer::device_bytes_in_use() const { std::lock_guard<std::mutex> lk(mu_); return dev_in_use_; }
uint64_t Tracer::peak_device_bytes() const { std::lock_guard<std::mutex> lk(mu_); return dev_peak_; }
size_t Tracer::outstanding_allocations() const { std::lock_guard<std::mutex> lk(mu_); return live_.size(); }
uint64_t Tracer::double_frees() const { std::lock_guard<std::mutex> lk(mu_); return double_frees_; }
uint64_t Tracer::invalid_frees() const { std::lock_guard<std::mutex> lk(mu_); return invalid_frees_; }
uint64_t Tracer::untracked_frees() const { std::lock_guard<std::mutex> lk(mu_); return untracked_frees_; }
uint64_t Tracer::pageable_async_copies() const { std::lock_guard<std::mutex> lk(mu_); return pageable_async_; }
uint64_t Tracer::small_copies() const { std::lock_guard<std::mutex> lk(mu_); return small_copies_; }
size_t Tracer::trace_event_count() const { std::lock_guard<std::mutex> lk(mu_); return events_.size(); }
bool Tracer::trace_truncated() const { std::lock_guard<std::mutex> lk(mu_); return truncated_; }
uint64_t Tracer::kernel_launches_timed() const {
  std::lock_guard<std::mutex> lk(mu_);
  uint64_t n = 0;
  for (auto& kv : kernel_stats_) n += kv.second.calls;
  return n;
}
uint64_t Tracer::error_count() const {
  std::lock_guard<std::mutex> lk(mu_);
  uint64_t n = 0;
  for (auto& kv : errors_) n += kv.second.count;
  return n;
}

int Tracer::add_module(const std::string& object, const Fatbin& fb) {
  std::lock_guard<std::mutex> lk(mu_);
  ModuleRecord m;
  m.id = static_cast<int>(modules_.size()) + 1;
  m.object = object;
  m.fatbin = fb;
  for (auto& im : m.fatbin.images) im.payload = nullptr;  // do not keep pointers into the binary
  modules_.push_back(std::move(m));
  return modules_.back().id;
}

void Tracer::add_module_kernel(int module, const std::string& kernel) {
  std::lock_guard<std::mutex> lk(mu_);
  if (module < 1 || module > static_cast<int>(modules_.size())) return;
  modules_[module - 1].kernels.push_back(kernel);
}

void Tracer::set_device(int cc, const std::string& name) {
  std::lock_guard<std::mutex> lk(mu_);
  device_cc_ = cc;
  device_name_ = name;
}

void Tracer::kernel_launch(const std::string& name, uint64_t host_us, int module) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = launches_.find(name);
  if (it == launches_.end()) {
    LaunchStat& l = launches_[name];
    l.first_us = host_us;
    l.module = module;
    return;
  }
  ++it->second.later;
  it->second.later_total_us += host_us;
}

void Tracer::note_ptx_dump(const std::string& path) {
  std::lock_guard<std::mutex> lk(mu_);
  ptx_dumps_.push_back(path);
}

size_t Tracer::module_count() const { std::lock_guard<std::mutex> lk(mu_); return modules_.size(); }
int Tracer::device_cc() const { std::lock_guard<std::mutex> lk(mu_); return device_cc_; }
LaunchStat Tracer::launch_stat(const std::string& k) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = launches_.find(k);
  return it == launches_.end() ? LaunchStat{} : it->second;
}

// The "which GPU code actually runs, and how did it get there" section.
std::string Tracer::code_object_section() const {
  const std::string P = "==ptxray== ";
  std::string r;
  auto line = [&](const std::string& s) { r += P + s + "\n"; };
  if (modules_.empty()) return r;

  std::string dev = !device_cc_ ? "unknown (no GPU context)"
                    : device_name_.empty() ? fmt("sm_%d", device_cc_)
                                           : fmt("sm_%d (%s)", device_cc_, device_name_.c_str());
  line(fmt("GPU code objects: %zu fatbinar%s registered; device %s", modules_.size(),
           modules_.size() == 1 ? "y" : "ies", dev.c_str()));

  // Group modules by the binary that registered them (libtorch_cuda.so alone
  // registers hundreds).
  std::map<std::string, std::vector<const ModuleRecord*>> by_obj;
  for (auto& m : modules_) by_obj[m.object].push_back(&m);
  for (auto& [obj, mods] : by_obj) {
    size_t nk = 0, sass = 0, jit = 0, none = 0, empty = 0;
    std::map<std::string, int> archs;
    for (auto* m : mods) {
      nk += m->kernels.size();
      for (auto& im : m->fatbin.images) archs[(im.is_ptx() ? "compute_" : "sm_") + std::to_string(im.arch)]++;
      if (!device_cc_) continue;
      if (m->kernels.empty()) { ++empty; continue; }
      Pick p = select_image(m->fatbin, device_cc_).pick;
      (p == Pick::Sass ? sass : p == Pick::PtxJit ? jit : none)++;
    }
    std::string as;
    for (auto& kv : archs) as += (as.empty() ? "" : " ") + kv.first;
    line(fmt("  %s: %zu module(s), %zu kernel(s); images: %s", obj.c_str(), mods.size(), nk, as.c_str()));
    if (device_cc_)
      line(fmt("    on sm_%d: %zu native SASS, %zu PTX JIT, %zu no compatible image%s", device_cc_, sass, jit,
               none, empty ? fmt(", %zu without kernels", empty).c_str() : ""));
    if (mods.size() > 6) continue;  // details only for small binaries
    for (auto* m : mods) {
      std::string imgs;
      for (auto& im : m->fatbin.images) {
        imgs += (imgs.empty() ? "" : ", ") + im.label() + " " + human_bytes(im.uncompressed_size ? im.uncompressed_size : im.size);
        if (im.compression != "none") imgs += " [" + im.compression + "]";
      }
      if (!m->fatbin.ok) imgs = "unparsed: " + m->fatbin.error;
      line(fmt("    module %d: %s", m->id, imgs.c_str()));
      if (!m->kernels.empty()) {
        std::string ks;
        for (size_t i = 0; i < m->kernels.size() && i < 4; ++i)
          ks += (i ? ", " : "") + shorten(demangle(m->kernels[i]), 40);
        if (m->kernels.size() > 4) ks += fmt(", +%zu more", m->kernels.size() - 4);
        line(fmt("      kernels: %s", ks.c_str()));
        if (device_cc_) line("      -> " + select_image(m->fatbin, device_cc_).describe(m->fatbin, device_cc_));
      }
      for (auto& im : m->fatbin.images)
        if (im.decoded)
          line(fmt("      PTX: .version %s, .target %s, %zu .entry", im.ptx_version.c_str(), im.ptx_target.c_str(),
                   im.ptx_entries.size()));
    }
  }
  for (auto& d : ptx_dumps_) line("  PTX written to " + d);
  return r;
}

// Chrome Trace Event Format (opens in ui.perfetto.dev and chrome://tracing).
// pid 1 = host CUDA API calls, one track per thread.
// pid 2 = GPU, one track per CUDA stream, plus a device-memory counter.
void Tracer::write_trace(FILE* f, int pid) const {
  std::lock_guard<std::mutex> lk(mu_);
  fputs("{\"displayTimeUnit\":\"ms\",\"otherData\":{\"tool\":\"ptxray\"", f);
  fprintf(f, ",\"pid\":%d,\"truncated\":%s},\n\"traceEvents\":[\n", pid, truncated_ ? "true" : "false");
  fprintf(f, "{\"ph\":\"M\",\"pid\":1,\"name\":\"process_name\",\"args\":{\"name\":\"Host: CUDA API (pid %d)\"}},\n", pid);
  fputs("{\"ph\":\"M\",\"pid\":2,\"name\":\"process_name\",\"args\":{\"name\":\"GPU\"}},\n", f);
  for (auto& kv : lane_names_)
    fprintf(f, "{\"ph\":\"M\",\"pid\":2,\"tid\":%u,\"name\":\"thread_name\",\"args\":{\"name\":\"%s\"}},\n",
            kv.first, json_escape(kv.second).c_str());
  for (auto& [ts, bytes] : mem_counter_)
    fprintf(f, "{\"ph\":\"C\",\"pid\":2,\"name\":\"Device memory (MiB)\",\"ts\":%" PRIu64
               ",\"args\":{\"MiB\":%.3f}},\n",
            ts, bytes / (1024.0 * 1024.0));
  bool first = true;
  for (auto& e : events_) {
    if (!first) fputs(",\n", f);
    first = false;
    fprintf(f, "{\"ph\":\"X\",\"pid\":%d,\"tid\":%u,\"cat\":\"%s\",\"name\":\"%s\",\"ts\":%" PRIu64
               ",\"dur\":%" PRIu64,
            e.gpu ? 2 : 1, e.lane, cat_name(e.cat), json_escape(e.name).c_str(), e.ts_us,
            e.dur_us);
    if (!e.args.empty()) fprintf(f, ",\"args\":{%s}", e.args.c_str());
    fputc('}', f);
  }
  if (first) fputs("{\"ph\":\"M\",\"pid\":1,\"name\":\"process_sort_index\",\"args\":{\"sort_index\":0}}", f);
  fputs("\n]}\n", f);
}

std::string Tracer::report(uint64_t wall_us, int pid) const {
  std::lock_guard<std::mutex> lk(mu_);
  const std::string P = "==ptxray== ";
  std::string r;
  auto line = [&](const std::string& s) { r += P + s + "\n"; };

  line(fmt("pid %d, wall time %.2f ms", pid, wall_us / 1000.0));
  if (!modules_.empty()) { line(""); r += code_object_section(); }

  // Kernels.
  uint64_t kcalls = 0;
  double ktotal = 0;
  for (auto& kv : kernel_stats_) { kcalls += kv.second.calls; ktotal += kv.second.total_us; }
  line("");
  line(fmt("GPU kernels: %" PRIu64 " timed launches, %.3f ms total GPU time", kcalls, ktotal / 1000.0));
  if (!kernel_stats_.empty()) {
    line(fmt("  %6s %9s %10s %10s %10s  %s", "time%", "calls", "total ms", "avg us", "max us", "kernel"));
    int shown = 0;
    for (auto& [name, s] : by_total_desc(kernel_stats_)) {
      if (shown++ == 15) { line(fmt("  ... %zu more kernels in the trace", kernel_stats_.size() - 15)); break; }
      line(fmt("  %5.1f%% %9" PRIu64 " %10.3f %10.2f %10.2f  %s",
               ktotal > 0 ? 100.0 * s.total_us / ktotal : 0.0, s.calls, s.total_us / 1000.0,
               s.total_us / s.calls, s.max_us, shorten(name, 70).c_str()));
    }
  }
  if (untimed_[static_cast<int>(Cat::Kernel)])
    line(fmt("  (%" PRIu64 " launches not timed: stream capture or timing disabled)",
             untimed_[static_cast<int>(Cat::Kernel)]));

  // Copies and memsets.
  line("");
  line(fmt("Memory transfers: %" PRIu64 " copies issued", total_copies_));
  if (!copy_stats_.empty()) {
    line(fmt("  %-26s %8s %12s %10s %10s", "operation", "calls", "bytes", "GPU ms", "GB/s"));
    for (auto& [name, s] : by_total_desc(copy_stats_)) {
      double gbps = s.total_us > 0 ? (s.bytes / 1e9) / (s.total_us / 1e6) : 0.0;
      line(fmt("  %-26s %8" PRIu64 " %12s %10.3f %10.2f", shorten(name, 26).c_str(), s.calls,
               human_bytes(s.bytes).c_str(), s.total_us / 1000.0, gbps));
    }
  }

  // Host API.
  line("");
  line(fmt("Host time blocked on the GPU (synchronize + synchronous memcpy): %.3f ms (%.1f%% of wall)",
           blocked_us_ / 1000.0, wall_us ? 100.0 * blocked_us_ / wall_us : 0.0));
  if (!host_stats_.empty()) {
    line(fmt("  %-28s %9s %12s %10s", "CUDA API (host side)", "calls", "total ms", "avg us"));
    int shown = 0;
    for (auto& [name, s] : by_total_desc(host_stats_)) {
      if (shown++ == 10) break;
      line(fmt("  %-28s %9" PRIu64 " %12.3f %10.2f", name.c_str(), s.calls, s.total_us / 1000.0,
               s.total_us / s.calls));
    }
  }

  // Memory.
  uint64_t live_dev = 0, live_bytes = 0;
  for (auto& kv : live_) { live_bytes += kv.second.size; if (kv.second.device_mem) live_dev += kv.second.size; }
  line("");
  line(fmt("Allocations: %" PRIu64 " allocs (%s), %" PRIu64 " frees, peak device memory %s",
           n_allocs_, human_bytes(bytes_allocated_).c_str(), n_frees_, human_bytes(dev_peak_).c_str()));

  // Diagnostics.
  std::vector<std::string> diags;
  if (device_cc_) {
    size_t jit = 0, none = 0;
    std::string jit_example, none_example;
    for (auto& m : modules_) {
      if (m.kernels.empty()) continue;
      Selection sel = select_image(m.fatbin, device_cc_);
      if (sel.pick == Pick::PtxJit) { if (!jit++) jit_example = m.object; }
      if (sel.pick == Pick::None) { if (!none++) none_example = m.object; }
    }
    if (jit)
      diags.push_back(fmt("[ptx-jit] %zu module(s) (e.g. in %s) carry no SASS for sm_%d, so the driver "
                          "JIT-compiles their PTX at load time; build with -gencode arch=compute_%d,code=sm_%d "
                          "to ship native code",
                          jit, jit_example.c_str(), device_cc_, device_cc_, device_cc_));
    if (none)
      diags.push_back(fmt("[no-image] %zu module(s) (e.g. in %s) have no image that can run on sm_%d", none,
                          none_example.c_str(), device_cc_));
  }
  {
    std::vector<std::pair<std::string, LaunchStat>> slow;
    // Lazy loading of a native-SASS module costs ~ms; only call that out when
    // it is large (>= 20 ms), but always when the module needed PTX JIT.
    for (auto& kv : launches_) {
      const LaunchStat& l = kv.second;
      double later_avg = l.later ? l.later_total_us / l.later : 0.0;
      bool jit = device_cc_ && l.module >= 1 && l.module <= static_cast<int>(modules_.size()) &&
                 select_image(modules_[l.module - 1].fatbin, device_cc_).pick == Pick::PtxJit;
      uint64_t floor_us = jit ? 1000 : 20000;
      if (l.first_us >= floor_us && (l.later == 0 || l.first_us > 20 * later_avg)) slow.push_back(kv);
    }
    std::sort(slow.begin(), slow.end(), [](auto& a, auto& b) { return a.second.first_us > b.second.first_us; });
    if (!slow.empty()) {
      std::string d = "[first-launch] kernels whose first launch was much slower than later ones "
                      "(lazy module loading, plus PTX JIT where noted):";
      for (size_t i = 0; i < slow.size() && i < 5; ++i) {
        const LaunchStat& l = slow[i].second;
        std::string later = l.later ? fmt("%.1f us afterwards", l.later_total_us / l.later) : "launched once";
        std::string why;
        if (l.module >= 1 && l.module <= static_cast<int>(modules_.size()) && device_cc_) {
          const ModuleRecord& m = modules_[l.module - 1];
          if (select_image(m.fatbin, device_cc_).pick == Pick::PtxJit) why = fmt(" [module %d: PTX JIT]", m.id);
        }
        d += fmt("\n%s      %s: %.2f ms first, %s%s", P.c_str(), shorten(slow[i].first, 50).c_str(),
                 l.first_us / 1000.0, later.c_str(), why.c_str());
      }
      diags.push_back(d);
    }
  }
  if (!live_.empty()) {
    std::vector<std::pair<uint64_t, Allocation>> v(live_.begin(), live_.end());
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) {
      if (a.second.size != b.second.size) return a.second.size > b.second.size;
      return a.second.ts_us < b.second.ts_us;
    });
    std::string d = fmt("[outstanding] %zu allocation(s) totalling %s were never freed "
                        "(a leak, or memory held by a caching allocator such as PyTorch's):",
                        live_.size(), human_bytes(live_bytes).c_str());
    for (size_t i = 0; i < v.size() && i < 5; ++i)
      d += fmt("\n%s      %s %s at %s", P.c_str(), human_bytes(v[i].second.size).c_str(),
               v[i].second.kind.c_str(), v[i].second.site.c_str());
    if (v.size() > 5) d += fmt("\n%s      ... and %zu more", P.c_str(), v.size() - 5);
    diags.push_back(d);
  }
  if (double_frees_ || invalid_frees_ || untracked_frees_) {
    std::string d = fmt("[bad-free] %" PRIu64 " double free(s), %" PRIu64 " invalid free(s), %" PRIu64
                        " free(s) of pointers not allocated through traced APIs:",
                        double_frees_, invalid_frees_, untracked_frees_);
    for (auto& b : bad_free_examples_)
      d += fmt("\n%s      %s: %s(0x%" PRIx64 ") at %s", P.c_str(), b.kind.c_str(), b.api.c_str(),
               b.ptr, b.site.c_str());
    diags.push_back(d);
  }
  if (pageable_async_)
    diags.push_back(fmt("[pageable] %" PRIu64 " cudaMemcpyAsync call(s) (%s) used pageable host memory; "
                        "these are staged through a driver buffer and are not truly asynchronous. "
                        "Use cudaMallocHost/cudaHostRegister. First at %s",
                        pageable_async_, human_bytes(pageable_async_bytes_).c_str(), pageable_site_.c_str()));
  if (small_copies_ >= cfg_.small_copy_warn)
    diags.push_back(fmt("[small-copy] %" PRIu64 " of %" PRIu64 " copies moved < %s; each pays fixed "
                        "launch/latency cost, consider batching",
                        small_copies_, total_copies_, human_bytes(cfg_.small_copy_bytes).c_str()));
  if (wall_us && blocked_us_ > cfg_.sync_warn_fraction * wall_us)
    diags.push_back(fmt("[sync-heavy] host spent %.1f%% of wall time blocked on the GPU; look for "
                        "per-iteration synchronization (e.g. .item(), cudaDeviceSynchronize in loops)",
                        100.0 * blocked_us_ / wall_us));
  for (auto& [key, e] : errors_)
    diags.push_back(fmt("[error] %s returned %s %" PRIu64 " time(s); first at %s", key.first.c_str(),
                        key.second.c_str(), e.count, e.first_site.c_str()));
  if (truncated_)
    diags.push_back(fmt("[trace] timeline truncated at %zu events (statistics above are complete); "
                        "raise PTXRAY_MAX_EVENTS", events_.size()));

  line("");
  if (diags.empty()) line("Diagnostics: none");
  else {
    line(fmt("Diagnostics: %zu", diags.size()));
    for (auto& d : diags) line("  " + d);
  }
  return r;
}

}  // namespace ct
