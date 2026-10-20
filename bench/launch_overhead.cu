// Measures host-side cost per kernel launch. Run it with and without the
// tracer (bench/run_bench.sh) to get the tracer's per-launch overhead.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void empty_kernel() {}

int main(int argc, char** argv) {
  int n = argc > 1 ? atoi(argv[1]) : 100000;
  for (int i = 0; i < 1000; ++i) empty_kernel<<<1, 32>>>();  // warm up
  cudaDeviceSynchronize();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i) empty_kernel<<<1, 32>>>();
  cudaDeviceSynchronize();
  auto t1 = std::chrono::steady_clock::now();
  double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  printf("launches=%d total_ms=%.3f us_per_launch=%.3f\n", n, us / 1000.0, us / n);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
