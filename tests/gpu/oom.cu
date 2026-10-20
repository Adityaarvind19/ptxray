// An allocation that cannot succeed. The error must be reported with its site
// and passed back to the program unchanged.
#include "common.cuh"

int main() {
  void* p = nullptr;
  cudaError_t e = cudaMalloc(&p, size_t(1) << 50);  // 1 PiB
  cudaGetLastError();
  printf("oom: cudaMalloc(1 PiB) -> %s\n", cudaGetErrorName(e));
  return e == cudaErrorMemoryAllocation ? 0 : 1;
}
