// Host-only program that calls the CUDA runtime. Used to check that the
// interposer loads, resolves the real functions, forwards results unchanged,
// and reports errors -- on a machine with no GPU (e.g. CI).
#include <cuda_runtime_api.h>
#include <cstdio>

int main() {
  void* p = nullptr;
  cudaError_t r1 = cudaMalloc(&p, 1 << 20);  // fails without a GPU/driver
  cudaError_t r2 = cudaFree(nullptr);
  cudaError_t r3 = cudaDeviceSynchronize();
  printf("smoke: cudaMalloc=%s cudaFree=%s cudaDeviceSynchronize=%s\n", cudaGetErrorName(r1),
         cudaGetErrorName(r2), cudaGetErrorName(r3));
  return 0;
}
