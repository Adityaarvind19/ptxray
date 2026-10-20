// 1000 tiny synchronous copies in a loop: the classic "batch these" pattern.
#include "common.cuh"

int main() {
  float* d;
  CK(cudaMalloc(&d, 1000 * 16 * sizeof(float)));
  float h[16] = {};
  for (int i = 0; i < 1000; ++i) CK(cudaMemcpy(d + 16 * i, h, sizeof h, cudaMemcpyHostToDevice));
  CK(cudaDeviceSynchronize());
  CK(cudaFree(d));
  printf("small_copies: done\n");
  return 0;
}
