// The same 64 MiB transfer from pageable and from pinned host memory with
// cudaMemcpyAsync. The tracer must flag the pageable copies and its per-row
// bandwidth should show pinned memory is faster.
#include "common.cuh"
#include <cstring>

int main() {
  const size_t bytes = 64u << 20;
  void* pageable = malloc(bytes);
  void* pinned;
  CK(cudaMallocHost(&pinned, bytes));
  memset(pageable, 1, bytes);
  memset(pinned, 1, bytes);
  void* dev;
  CK(cudaMalloc(&dev, bytes));
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  for (int i = 0; i < 5; ++i) {
    CK(cudaMemcpyAsync(dev, pageable, bytes, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(dev, pinned, bytes, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(pageable, dev, bytes, cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(pinned, dev, bytes, cudaMemcpyDeviceToHost, s));
  }
  CK(cudaStreamSynchronize(s));
  CK(cudaStreamDestroy(s));
  CK(cudaFree(dev));
  CK(cudaFreeHost(pinned));
  free(pageable);
  printf("pageable: done\n");
  return 0;
}
