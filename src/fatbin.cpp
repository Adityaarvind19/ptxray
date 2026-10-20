#include "fatbin.hpp"

#include <dlfcn.h>

#include <cstring>

namespace ct {

namespace {

template <class T>
bool read(const uint8_t* base, size_t len, size_t off, T* out) {
  if (off > len || len - off < sizeof(T)) return false;
  std::memcpy(out, base + off, sizeof(T));
  return true;
}

std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r");
  size_t b = s.find_last_not_of(" \t\r");
  return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

}  // namespace

std::string FatbinImage::label() const {
  std::string s = is_ptx() ? "PTX compute_" : (is_sass() ? "SASS sm_" : "kind" + std::to_string(kind) + " ");
  s += std::to_string(arch);
  if (is_ptx() && (ptx_major || ptx_minor))
    s += " (ISA " + std::to_string(ptx_major) + "." + std::to_string(ptx_minor) + ")";
  return s;
}

Fatbin parse_fatbin(const void* data, size_t max_len) {
  Fatbin fb;
  const uint8_t* p = static_cast<const uint8_t*>(data);
  uint32_t magic;
  uint16_t version, hsize;
  uint64_t payload;
  if (!p || !read(p, max_len, 0, &magic) || !read(p, max_len, 4, &version) ||
      !read(p, max_len, 6, &hsize) || !read(p, max_len, 8, &payload)) {
    fb.error = "truncated container header";
    return fb;
  }
  if (magic != kFatbinMagic) { fb.error = "bad container magic"; return fb; }
  if (hsize < 16 || payload > max_len - hsize) { fb.error = "container size out of range"; return fb; }
  fb.total_bytes = hsize + payload;

  size_t off = hsize, end = hsize + payload;
  for (int guard = 0; off < end && guard < 4096; ++guard) {
    FatbinImage img;
    uint16_t kind, ever;
    uint32_t ehs, csize, opt, arch;
    uint16_t minor, major;
    uint64_t size, flags, usize;
    if (!read(p, end, off + 0, &kind) || !read(p, end, off + 2, &ever) || !read(p, end, off + 4, &ehs) ||
        !read(p, end, off + 8, &size) || !read(p, end, off + 16, &csize) || !read(p, end, off + 20, &opt) ||
        !read(p, end, off + 24, &minor) || !read(p, end, off + 26, &major) || !read(p, end, off + 28, &arch) ||
        !read(p, end, off + 40, &flags) || !read(p, end, off + 56, &usize)) {
      fb.error = "truncated entry header";
      return fb;
    }
    (void)ever; (void)opt;
    if (ehs < 64 || size > end - off - ehs) { fb.error = "entry size out of range"; return fb; }
    img.kind = kind;
    img.arch = static_cast<int>(arch);
    img.size = size;
    img.flags = flags;
    img.uncompressed_size = usize;
    if (kind == 1) { img.ptx_major = major; img.ptx_minor = minor; }
    if (flags & 0x8000) img.compression = "zstd";
    else if (flags & 0x2000) img.compression = "legacy";
    else img.compression = "none";
    img.payload = p + off + ehs;
    img.payload_bytes = (img.compression != "none" && csize && csize <= size) ? csize : size;
    fb.images.push_back(img);
    off += ehs + size;
  }
  fb.ok = true;
  return fb;
}

Fatbin parse_registration(const void* wrapper) {
  struct Wrapper { uint32_t magic, version; const void* data; const void* extra; };
  Fatbin fb;
  if (!wrapper) { fb.error = "null wrapper"; return fb; }
  Wrapper w;
  std::memcpy(&w, wrapper, sizeof w);
  if (w.magic != kFatbinWrapperMagic) { fb.error = "bad wrapper magic"; return fb; }
  // The container is in the binary's mapped .nv_fatbin section; read its
  // header first to learn the bound, then parse within it.
  const uint8_t* d = static_cast<const uint8_t*>(w.data);
  uint16_t hsize;
  uint64_t payload;
  if (!d) { fb.error = "null data"; return fb; }
  std::memcpy(&hsize, d + 6, 2);
  std::memcpy(&payload, d + 8, 8);
  if (payload > (uint64_t(1) << 34)) { fb.error = "implausible container size"; return fb; }
  return parse_fatbin(d, hsize + payload);
}

bool decode_ptx(const FatbinImage& img, std::string* text, std::string* err) {
  if (!img.is_ptx()) { *err = "not a PTX entry"; return false; }
  if (img.compression == "none") {
    size_t n = strnlen(reinterpret_cast<const char*>(img.payload), img.payload_bytes);
    text->assign(reinterpret_cast<const char*>(img.payload), n);
    return true;
  }
  if (img.compression == "zstd") {
    using Decompress = size_t (*)(void*, size_t, const void*, size_t);
    using IsError = unsigned (*)(size_t);
    static void* h = dlopen("libzstd.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!h) { *err = "zstd-compressed and libzstd.so.1 is not available"; return false; }
    static auto dec = reinterpret_cast<Decompress>(dlsym(h, "ZSTD_decompress"));
    static auto is_err = reinterpret_cast<IsError>(dlsym(h, "ZSTD_isError"));
    if (!dec || !is_err || !img.uncompressed_size || img.uncompressed_size > (uint64_t(1) << 31)) {
      *err = "cannot decompress";
      return false;
    }
    std::string out(img.uncompressed_size, '\0');
    size_t r = dec(&out[0], out.size(), img.payload, img.payload_bytes);
    if (is_err(r)) { *err = "zstd error"; return false; }
    out.resize(strnlen(out.data(), r));
    *text = std::move(out);
    return true;
  }
  *err = "compressed with the legacy fatbinary format (rebuild with -Xfatbin -compress=false, or use cuobjdump -ptx)";
  return false;
}

void parse_ptx_header(const std::string& ptx, FatbinImage* img) {
  size_t pos = 0;
  while (pos < ptx.size()) {
    size_t nl = ptx.find('\n', pos);
    std::string line = trim(ptx.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos));
    pos = nl == std::string::npos ? ptx.size() : nl + 1;
    if (line.rfind(".version", 0) == 0) img->ptx_version = trim(line.substr(8));
    else if (line.rfind(".target", 0) == 0) img->ptx_target = trim(line.substr(7));
    else {
      // ".visible .entry _Z5saxpyfPfS0_i(" or ".entry name("
      size_t e = line.find(".entry ");
      if (e != std::string::npos && (e == 0 || line[e - 1] == ' ')) {
        std::string rest = trim(line.substr(e + 7));
        size_t stop = rest.find_first_of("( \t");
        img->ptx_entries.push_back(rest.substr(0, stop));
      }
    }
  }
  img->decoded = true;
}

Selection select_image(const Fatbin& fb, int cc) {
  Selection s;
  int best_sass = -1, best_ptx = -1;
  for (size_t i = 0; i < fb.images.size(); ++i) {
    const FatbinImage& im = fb.images[i];
    if (im.is_sass() && im.arch / 10 == cc / 10 && im.arch % 10 <= cc % 10 &&
        (best_sass < 0 || im.arch > fb.images[best_sass].arch))
      best_sass = static_cast<int>(i);
    if (im.is_ptx() && im.arch <= cc && (best_ptx < 0 || im.arch > fb.images[best_ptx].arch))
      best_ptx = static_cast<int>(i);
  }
  if (best_sass >= 0) { s.pick = Pick::Sass; s.index = best_sass; }
  else if (best_ptx >= 0) { s.pick = Pick::PtxJit; s.index = best_ptx; }
  return s;
}

std::string Selection::describe(const Fatbin& fb, int cc) const {
  std::string dev = "sm_" + std::to_string(cc);
  switch (pick) {
    case Pick::Sass: return "loads " + fb.images[index].label() + " natively on " + dev;
    case Pick::PtxJit:
      return "no SASS for " + dev + ": driver JIT-compiles " + fb.images[index].label() + " (ptxas at run time)";
    case Pick::None: break;
  }
  return "NO compatible image for " + dev + " (launches fail with cudaErrorNoKernelImageForDevice)";
}

}  // namespace ct
