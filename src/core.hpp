// cudatrace core: event recording, aggregation, diagnostics and output.
//
// This file has no CUDA dependency on purpose. The interposition layer
// (hooks.cpp) translates intercepted CUDA calls into calls on Tracer, and
// everything below can be unit-tested on a machine without a GPU.
#pragma once

#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fatbin.hpp"

namespace ct {

enum class Cat : uint8_t { Kernel, Memcpy, Memset, Sync, Alloc, Free, Other };
const char* cat_name(Cat c);

struct Config {
  size_t max_trace_events = 1000000;  // stop recording timeline events past this
  uint64_t small_copy_bytes = 4096;    // copies below this count as "small"
  uint64_t small_copy_warn = 32;       // warn when at least this many small copies
  double sync_warn_fraction = 0.20;    // warn when host is blocked > 20% of wall time
};

// One timeline event (Chrome trace "X" event). Host events live on lane=tid,
// GPU events on lane=stream index.
struct TraceEvent {
  std::string name;
  Cat cat;
  bool gpu;
  uint64_t ts_us;
  uint64_t dur_us;
  uint32_t lane;
  std::string args;  // JSON object body without braces, may be empty
};

struct OpStat {
  uint64_t calls = 0;
  double total_us = 0;
  double max_us = 0;
  uint64_t bytes = 0;
  void add(double us, uint64_t b = 0) {
    ++calls;
    total_us += us;
    if (us > max_us) max_us = us;
    bytes += b;
  }
};

struct Allocation {
  uint64_t size;
  std::string kind;  // "device", "managed", "pinned-host", "device-async", ...
  bool device_mem;   // counts toward device-memory usage
  std::string site;  // where it was allocated (symbolized return address)
  uint64_t ts_us;
};

// One registered fatbinary (one per translation unit / CUDA library module).
struct ModuleRecord {
  int id = 0;
  std::string object;  // executable or shared library that registered it
  Fatbin fatbin;       // image metadata (payload pointers are not used after registration)
  std::vector<std::string> kernels;
};

struct LaunchStat {
  uint64_t first_us = 0;  // host time of the first launch (includes lazy load / JIT)
  uint64_t later = 0;     // number of launches after the first
  double later_total_us = 0;
  int module = 0;         // registering fatbinary, 0 if unknown
};

enum class FreeResult { Freed, Failed, DoubleFree, InvalidFree, Untracked };

class Tracer {
 public:
  explicit Tracer(Config c = {});

  // Any intercepted host-side API call.
  void host_call(const std::string& api, Cat cat, uint64_t ts_us, uint64_t dur_us,
                 uint32_t tid, const std::string& args = "");
  // A completed GPU operation (timed with CUDA events).
  void gpu_op(const std::string& name, Cat cat, uint32_t lane, uint64_t ts_us,
              uint64_t dur_us, uint64_t bytes, const std::string& args = "");
  // An operation that ran on the GPU but could not be timed (e.g. inside
  // stream capture, or the pending-event budget was exhausted).
  void untimed_op(Cat cat);

  void alloc(uint64_t ptr, uint64_t size, const std::string& kind, bool device_mem,
             const std::string& site, uint64_t ts_us);
  // real_ok: whether the underlying CUDA call succeeded.
  FreeResult free(uint64_t ptr, const std::string& api, bool real_ok,
                  const std::string& site, uint64_t ts_us);

  void memcpy_issued(uint64_t bytes, bool async, bool pageable_host, const std::string& site);
  void api_error(const std::string& api, const std::string& err, const std::string& site);
  void host_blocked(uint64_t us);
  void name_lane(uint32_t lane, const std::string& name);

  // Code objects and the Driver Compiler Interface view.
  int add_module(const std::string& object, const Fatbin& fb);
  void add_module_kernel(int module, const std::string& kernel);
  void set_device(int cc, const std::string& name);
  void kernel_launch(const std::string& name, uint64_t host_us, int module);  // host-side launch time
  void note_ptx_dump(const std::string& path);

  void write_trace(FILE* f, int pid) const;
  std::string report(uint64_t wall_us, int pid) const;
  bool empty() const;

  // Introspection (used by tests).
  uint64_t device_bytes_in_use() const;
  uint64_t peak_device_bytes() const;
  size_t outstanding_allocations() const;
  uint64_t double_frees() const;
  uint64_t invalid_frees() const;
  uint64_t untracked_frees() const;
  uint64_t pageable_async_copies() const;
  uint64_t small_copies() const;
  uint64_t kernel_launches_timed() const;
  uint64_t error_count() const;
  size_t trace_event_count() const;
  size_t module_count() const;
  int device_cc() const;
  LaunchStat launch_stat(const std::string& kernel) const;
  bool trace_truncated() const;

 private:
  void push_event(TraceEvent&& e);  // requires mu_ held

  Config cfg_;
  mutable std::mutex mu_;

  std::vector<TraceEvent> events_;
  bool truncated_ = false;
  std::vector<std::pair<uint64_t, uint64_t>> mem_counter_;  // (ts_us, device bytes)
  std::map<uint32_t, std::string> lane_names_;

  std::map<std::string, OpStat> host_stats_;    // by API name
  std::map<std::string, OpStat> kernel_stats_;  // by kernel name (GPU time)
  std::map<std::string, OpStat> copy_stats_;    // by "Memcpy HtoD (pinned)" etc. (GPU time)
  uint64_t untimed_[7] = {};

  std::unordered_map<uint64_t, Allocation> live_;
  std::unordered_set<uint64_t> freed_;  // freed and not re-allocated since
  uint64_t dev_in_use_ = 0, dev_peak_ = 0;
  uint64_t n_allocs_ = 0, n_frees_ = 0, bytes_allocated_ = 0;

  struct BadFree { std::string kind, api, site; uint64_t ptr; };
  std::vector<BadFree> bad_free_examples_;
  uint64_t double_frees_ = 0, invalid_frees_ = 0, untracked_frees_ = 0;

  uint64_t pageable_async_ = 0, pageable_async_bytes_ = 0;
  std::string pageable_site_;
  uint64_t small_copies_ = 0, total_copies_ = 0;
  uint64_t blocked_us_ = 0;

  std::vector<ModuleRecord> modules_;
  std::map<std::string, LaunchStat> launches_;
  std::vector<std::string> ptx_dumps_;
  int device_cc_ = 0;
  std::string device_name_;

  std::string code_object_section() const;  // requires mu_ held

  struct ErrInfo { uint64_t count = 0; std::string first_site; };
  std::map<std::pair<std::string, std::string>, ErrInfo> errors_;  // (api, error)
};

std::string json_escape(const std::string& s);
std::string human_bytes(uint64_t b);
std::string demangle(const std::string& s);
std::string shorten(const std::string& s, size_t n);

}  // namespace ct
