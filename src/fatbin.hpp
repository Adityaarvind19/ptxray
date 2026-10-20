// Parser for NVIDIA fatbinary containers (the .nv_fatbin section that nvcc
// embeds in every CUDA executable or library), and the driver's image
// selection rule.
//
// Layout, as observed in nvcc 12.x/13.x output (little endian):
//
//   container header (16 bytes)
//     u32 magic = 0xBA55ED50   u16 version   u16 header_size   u64 payload_size
//   followed by payload_size bytes of entries, each:
//     u16 kind        1 = PTX text, 2 = ELF cubin (SASS)
//     u16 version
//     u32 header_size (64 for cubin, 80 for PTX)
//     u64 size        bytes of payload after the header (padded)
//     u32 compressed_size        (0 when not compressed)
//     u32 options_offset
//     u16 minor, u16 major       (PTX ISA version for PTX entries)
//     u32 arch                   (75 = sm_75 / compute_75)
//     u32 obj_name_offset, u32 obj_name_len
//     u64 flags                  0x1 = 64-bit, 0x10 = Linux,
//                                0x2000 = compressed (legacy LZ format),
//                                0x8000 = compressed (zstd, CUDA 12.8+)
//     u64 reserved
//     u64 uncompressed_size
//
// The registration wrapper nvcc passes to __cudaRegisterFatBinary is
//   struct { u32 magic = 0x466243B1; u32 version; const void* data; void* extra; }
// where data points at a container header.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ct {

constexpr uint32_t kFatbinMagic = 0xBA55ED50u;
constexpr uint32_t kFatbinWrapperMagic = 0x466243B1u;

struct FatbinImage {
  int kind = 0;   // 1 = PTX, 2 = cubin
  int arch = 0;   // e.g. 75
  int ptx_major = 0, ptx_minor = 0;  // from the header (PTX entries)
  uint64_t size = 0;                 // stored payload bytes
  uint64_t uncompressed_size = 0;
  uint64_t flags = 0;
  std::string compression;           // "none", "zstd", "legacy"
  const uint8_t* payload = nullptr;  // valid only while the binary is mapped
  uint64_t payload_bytes = 0;        // exact compressed/raw byte count

  // Filled in for PTX entries when the text could be decoded.
  bool decoded = false;
  std::string ptx_version, ptx_target;
  std::vector<std::string> ptx_entries;

  bool is_ptx() const { return kind == 1; }
  bool is_sass() const { return kind == 2; }
  std::string label() const;  // "SASS sm_75" / "PTX compute_75 (ISA 8.5)"
};

struct Fatbin {
  bool ok = false;
  std::string error;
  uint64_t total_bytes = 0;
  std::vector<FatbinImage> images;
};

// Parse one container starting at `data`. `max_len` bounds every read.
Fatbin parse_fatbin(const void* data, size_t max_len);
// Parse what __cudaRegisterFatBinary receives. Returns !ok if the wrapper
// magic does not match.
Fatbin parse_registration(const void* wrapper);

// Decode a PTX entry's payload into text (uncompressed or zstd via a
// dlopen()ed libzstd). Returns false and sets *err otherwise.
bool decode_ptx(const FatbinImage& img, std::string* text, std::string* err);
// Extract .version, .target and .entry names from PTX text.
void parse_ptx_header(const std::string& ptx, FatbinImage* img);

// Which image the driver will load on a device of compute capability `cc`
// (e.g. 75), following the CUDA compatibility rules:
//  * a cubin for sm_XY runs on devices of the same major version X with
//    minor >= Y; the highest such cubin is preferred;
//  * otherwise the driver JIT-compiles the highest PTX with compute_XY <= cc;
//  * otherwise nothing can run (cudaErrorNoKernelImageForDevice).
// Arch-specific targets (sm_90a, ...) are exact-match only; the header does
// not mark them, so they are treated like ordinary cubins.
enum class Pick { Sass, PtxJit, None };
struct Selection {
  Pick pick = Pick::None;
  int index = -1;  // into Fatbin::images
  std::string describe(const Fatbin& fb, int cc) const;
};
Selection select_image(const Fatbin& fb, int cc);

}  // namespace ct
