// Four host threads, each with its own stream, launching concurrently. Checks
// thread safety of the tracer and that each stream gets its own GPU track.
#include "common.cuh"
#include <thread>
#include <vector>

__global__ void busy(float* x, int iters) {
  float v = x[threadIdx.x];
  for (int i = 0; i < iters; ++i) v = v * 0.999f + 0.001f;
  x[threadIdx.x] = v;
}

int main() {
  const int kThreads = 4, kLaunches = 50;
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t)
    ts.emplace_back([&] {
      cudaStream_t s;
      float* x;
      CK(cudaStreamCreate(&s));
      CK(cudaMalloc(&x, 256 * sizeof(float)));
      CK(cudaMemsetAsync(x, 0, 256 * sizeof(float), s));
      for (int i = 0; i < kLaunches; ++i) busy<<<1, 256, 0, s>>>(x, 2000);
      CK(cudaGetLastError());
      CK(cudaStreamSynchronize(s));
      CK(cudaFree(x));
      CK(cudaStreamDestroy(s));
    });
  for (auto& t : ts) t.join();
  printf("streams: %d threads x %d launches done\n", kThreads, kLaunches);
  return 0;
}
