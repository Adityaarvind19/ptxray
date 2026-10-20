// Unit tests for the CUDA-independent core. Runs anywhere: make test
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "core.hpp"
#include "fatbin.hpp"

#include <fstream>
#include <iterator>
#include <vector>

static int g_fail = 0, g_pass = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    if (cond) ++g_pass;                                                      \
    else { ++g_fail; fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
  } while (0)

static bool contains(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }

static void test_alloc_accounting() {
  ct::Tracer t;
  t.alloc(0x1000, 100, "device", true, "a", 1);
  t.alloc(0x2000, 50, "device", true, "b", 2);
  t.alloc(0x3000, 1000, "pinned-host", false, "c", 3);
  CHECK(t.device_bytes_in_use() == 150);
  CHECK(t.peak_device_bytes() == 150);
  CHECK(t.free(0x1000, "cudaFree", true, "x", 4) == ct::FreeResult::Freed);
  CHECK(t.device_bytes_in_use() == 50);
  CHECK(t.peak_device_bytes() == 150);  // peak is sticky
  t.alloc(0x4000, 10, "device", true, "d", 5);
  CHECK(t.peak_device_bytes() == 150);
  CHECK(t.outstanding_allocations() == 3);  // 0x2000, 0x3000, 0x4000
}

static void test_bad_frees() {
  ct::Tracer t;
  t.alloc(0x1000, 8, "device", true, "a", 1);
  CHECK(t.free(0x1000, "cudaFree", true, "s", 2) == ct::FreeResult::Freed);
  CHECK(t.free(0x1000, "cudaFree", false, "s", 3) == ct::FreeResult::DoubleFree);
  CHECK(t.free(0xdead, "cudaFree", false, "s", 4) == ct::FreeResult::InvalidFree);
  CHECK(t.free(0xbeef, "cudaFree", true, "s", 5) == ct::FreeResult::Untracked);
  CHECK(t.double_frees() == 1 && t.invalid_frees() == 1 && t.untracked_frees() == 1);
  // Address reuse: freed then re-allocated is a normal allocation again.
  t.alloc(0x1000, 8, "device", true, "a2", 6);
  CHECK(t.free(0x1000, "cudaFree", true, "s", 7) == ct::FreeResult::Freed);
  // A failed free of a live pointer keeps it live.
  t.alloc(0x5000, 8, "device", true, "a3", 8);
  CHECK(t.free(0x5000, "cudaFree", false, "s", 9) == ct::FreeResult::Failed);
  CHECK(t.outstanding_allocations() == 1);
}

static void test_copy_diagnostics() {
  ct::Config c;
  c.small_copy_warn = 3;
  ct::Tracer t(c);
  for (int i = 0; i < 3; ++i) t.memcpy_issued(16, false, false, "s");
  t.memcpy_issued(1 << 20, true, true, "pg");
  t.memcpy_issued(1 << 20, true, false, "pinned");
  CHECK(t.small_copies() == 3);
  CHECK(t.pageable_async_copies() == 1);
  std::string r = t.report(1000, 1);
  CHECK(contains(r, "[small-copy] 3 of 5"));
  CHECK(contains(r, "[pageable] 1 cudaMemcpyAsync"));
}

static void test_report_tables() {
  ct::Tracer t;
  t.host_call("cudaLaunchKernel", ct::Cat::Kernel, 0, 5, 7);
  t.gpu_op("saxpy(float*, int)", ct::Cat::Kernel, 0, 10, 30, 0);
  t.gpu_op("saxpy(float*, int)", ct::Cat::Kernel, 0, 50, 10, 0);
  t.gpu_op("Memcpy HtoD (pinned)", ct::Cat::Memcpy, 0, 0, 1000, 1000000000);  // 1 GB in 1 ms
  std::string r = t.report(10000, 42);
  CHECK(contains(r, "2 timed launches"));
  CHECK(contains(r, "saxpy(float*, int)"));
  CHECK(contains(r, "20.00"));     // avg us
  CHECK(contains(r, "1000.00"));   // 1 GB / 1 ms = 1000 GB/s
  CHECK(contains(r, "Diagnostics: none"));
  CHECK(t.kernel_launches_timed() == 2);
}

static void test_sync_and_errors() {
  ct::Tracer t;
  t.host_blocked(600);
  t.api_error("cudaMalloc", "cudaErrorMemoryAllocation", "main+0x10");
  t.api_error("cudaMalloc", "cudaErrorMemoryAllocation", "main+0x20");
  std::string r = t.report(1000, 1);
  CHECK(contains(r, "[sync-heavy]"));
  CHECK(contains(r, "cudaMalloc returned cudaErrorMemoryAllocation 2 time(s); first at main+0x10"));
  CHECK(t.error_count() == 2);
}

static void test_trace_cap_and_json() {
  ct::Config c;
  c.max_trace_events = 2;
  ct::Tracer t(c);
  for (int i = 0; i < 5; ++i) t.host_call("cudaFree", ct::Cat::Free, i, 1, 1);
  CHECK(t.trace_event_count() == 2);
  CHECK(t.trace_truncated());
  CHECK(ct::json_escape("a\"b\\c\n\x01") == "a\\\"b\\\\c\\n\\u0001");
  t.name_lane(3, "stream \"3\"");
  t.alloc(0x10, 1 << 20, "device", true, "s", 1);
  const char* path = "build/core_test_trace.json";
  FILE* f = fopen(path, "w");
  CHECK(f != nullptr);
  if (f) { t.write_trace(f, 99); fclose(f); }
  // Validate with Python if available (CI and Colab both have it).
  int rc = system("python3 -c \"import json,sys; d=json.load(open('build/core_test_trace.json')); "
                  "sys.exit(0 if len(d['traceEvents'])>=4 and d['otherData']['truncated'] else 1)\"");
  CHECK(rc == 0);
}

// ---- fatbinary parsing, against real nvcc output (tests/data, built with
// nvcc 13 for sm_75: t_basic = SASS sm_75 + PTX compute_75, jit_ptx = PTX only)

static std::vector<uint8_t> slurp(const char* path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

// The .nv_fatbin section can hold several containers; return the last one
// (the one carrying the program's kernels in these fixtures).
static ct::Fatbin last_container(const std::vector<uint8_t>& sec) {
  ct::Fatbin last;
  for (size_t off = 0; off + 16 <= sec.size(); off += 8) {
    uint32_t magic;
    std::memcpy(&magic, &sec[off], 4);
    if (magic != ct::kFatbinMagic) continue;
    ct::Fatbin fb = ct::parse_fatbin(&sec[off], sec.size() - off);
    if (!fb.ok) continue;
    last = fb;
    off += fb.total_bytes - 8;
  }
  return last;
}

static void test_fatbin_sass_plus_ptx() {
  auto sec = slurp("tests/data/t_basic.nv_fatbin");
  CHECK(!sec.empty());
  ct::Fatbin fb = last_container(sec);
  CHECK(fb.ok);
  CHECK(fb.images.size() == 2);
  if (fb.images.size() != 2) return;
  CHECK(fb.images[0].is_sass() && fb.images[0].arch == 75);
  CHECK(fb.images[1].is_ptx() && fb.images[1].arch == 75);
  CHECK(fb.images[0].payload[0] == 0x7f && fb.images[0].payload[1] == 'E');  // ELF cubin
  std::string ptx, err;
  bool ok = ct::decode_ptx(fb.images[1], &ptx, &err);
  CHECK(ok);
  if (ok) {
    ct::parse_ptx_header(ptx, &fb.images[1]);
    CHECK(fb.images[1].ptx_target == "sm_75");
    CHECK(!fb.images[1].ptx_version.empty());
    CHECK(fb.images[1].ptx_entries.size() == 1);
    CHECK(fb.images[1].ptx_entries.size() == 1 && contains(fb.images[1].ptx_entries[0], "vector_add"));
  }
  // Driver image selection.
  CHECK(ct::select_image(fb, 75).pick == ct::Pick::Sass);    // exact match
  CHECK(ct::select_image(fb, 86).pick == ct::Pick::PtxJit);  // sm_75 SASS cannot run on 8.x
  CHECK(ct::select_image(fb, 70).pick == ct::Pick::None);    // PTX too new, no SASS
  CHECK(contains(ct::select_image(fb, 86).describe(fb, 86), "JIT-compiles PTX compute_75"));
}

static void test_fatbin_ptx_only() {
  auto sec = slurp("tests/data/jit_ptx.nv_fatbin");
  ct::Fatbin fb = last_container(sec);
  CHECK(fb.ok && fb.images.size() == 1);
  if (fb.images.size() != 1) return;
  CHECK(fb.images[0].is_ptx() && fb.images[0].compression == "zstd");
  CHECK(fb.images[0].uncompressed_size > 10 * fb.images[0].payload_bytes);  // PTX compresses well
  std::string ptx, err;
  CHECK(ct::decode_ptx(fb.images[0], &ptx, &err));
  ct::parse_ptx_header(ptx, &fb.images[0]);
  CHECK(fb.images[0].ptx_entries.size() == 24);
  CHECK(ct::select_image(fb, 75).pick == ct::Pick::PtxJit);
  CHECK(ct::select_image(fb, 89).pick == ct::Pick::PtxJit);
}

static void test_fatbin_rules_synthetic() {
  ct::Fatbin fb;
  fb.ok = true;
  auto img = [](int kind, int arch) { ct::FatbinImage i; i.kind = kind; i.arch = arch; return i; };
  fb.images = {img(2, 70), img(2, 80), img(2, 86), img(1, 80)};
  CHECK(ct::select_image(fb, 75).pick == ct::Pick::Sass);   // sm_70 runs on 7.5
  CHECK(fb.images[ct::select_image(fb, 75).index].arch == 70);
  CHECK(fb.images[ct::select_image(fb, 89).index].arch == 86);  // highest compatible minor
  CHECK(ct::select_image(fb, 90).pick == ct::Pick::PtxJit);  // no 9.x SASS -> PTX 8.0
  CHECK(ct::select_image(fb, 61).pick == ct::Pick::None);
}

static void test_fatbin_rejects_garbage() {
  uint8_t junk[64] = {1, 2, 3};
  CHECK(!ct::parse_fatbin(junk, sizeof junk).ok);
  CHECK(!ct::parse_fatbin(nullptr, 0).ok);
  auto sec = slurp("tests/data/t_basic.nv_fatbin");
  if (sec.size() > 1600) {
    // Truncate inside the second container: must fail cleanly, not overread.
    CHECK(!ct::parse_fatbin(&sec[1552], 200).ok);
  }
  uint32_t wrapper[6] = {0x12345678};
  CHECK(!ct::parse_registration(wrapper).ok);
}

static void test_module_report() {
  auto sec = slurp("tests/data/jit_ptx.nv_fatbin");
  ct::Fatbin fb = last_container(sec);
  ct::Tracer t;
  int id = t.add_module("jit_ptx", fb);
  t.add_module_kernel(id, "_Z4polyILi0EEvPfi");
  t.set_device(75, "Tesla T4");
  t.kernel_launch("void poly<0>(float*, int)", 250000, id);  // 250 ms first launch
  t.kernel_launch("void poly<0>(float*, int)", 5, id);
  std::string r = t.report(1000000, 1);
  CHECK(contains(r, "device sm_75 (Tesla T4)"));
  CHECK(contains(r, "on sm_75: 0 native SASS, 1 PTX JIT, 0 no compatible image"));
  CHECK(contains(r, "[ptx-jit] 1 module(s)"));
  CHECK(contains(r, "250.00 ms first, 5.0 us afterwards [module 1: PTX JIT]"));
  CHECK(contains(r, "kernels: void poly<0>(float*, int)"));
}

static void test_helpers() {
  CHECK(ct::human_bytes(512) == "512 B");
  CHECK(ct::human_bytes(1536) == "1.5 KiB");
  CHECK(ct::human_bytes(3ull << 30) == "3.0 GiB");
  CHECK(ct::shorten("abcdefgh", 6) == "abc...");
  CHECK(ct::shorten("abc", 6) == "abc");
}

int main() {
  test_alloc_accounting();
  test_bad_frees();
  test_copy_diagnostics();
  test_report_tables();
  test_sync_and_errors();
  test_trace_cap_and_json();
  test_helpers();
  test_fatbin_sass_plus_ptx();
  test_fatbin_ptx_only();
  test_fatbin_rules_synthetic();
  test_fatbin_rejects_garbage();
  test_module_report();
  printf("core_test: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
