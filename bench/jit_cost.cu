// PTX JIT cost experiment.
//
// One module with many heavily unrolled kernels, so that compiling its PTX to
// SASS is real work for ptxas. The same source is built twice:
//   jit_sass: -gencode arch=compute_XY,code=sm_XY       (native SASS shipped)
//   jit_ptx:  -gencode arch=compute_XY,code=compute_XY  (PTX only)
// For the PTX-only build the driver must JIT-compile the module through its
// built-in ptxas the first time it is loaded (or reuse its JIT cache in
// ~/.nv/ComputeCache). With lazy module loading (CUDA 12 default) that cost
// lands on the first kernel launch.
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

template <int Salt>
__global__ void poly(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
#pragma unroll
  for (int k = 0; k < 96; ++k) v = fmaf(v, 1.0001f + 1e-6f * (k + Salt), __sinf(v * (Salt + 1)) * 0.5f);
  x[i] = v;
}

#define K(s) poly<s><<<blocks, 256>>>(d, n)

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

int main() {
  const int n = 1 << 16, blocks = (n + 255) / 256;
  auto t = Clock::now();
  cudaFree(nullptr);  // create the context
  double init_ms = ms_since(t);

  float* d;
  cudaMalloc(&d, n * sizeof(float));
  cudaMemset(d, 0, n * sizeof(float));
  cudaDeviceSynchronize();

  t = Clock::now();
  K(0);
  cudaDeviceSynchronize();
  double first_ms = ms_since(t);

  t = Clock::now();
  K(0);
  cudaDeviceSynchronize();
  double second_ms = ms_since(t);

  // The other 23 kernels of the same module.
  t = Clock::now();
  K(1); K(2); K(3); K(4); K(5); K(6); K(7); K(8); K(9); K(10); K(11); K(12);
  K(13); K(14); K(15); K(16); K(17); K(18); K(19); K(20); K(21); K(22); K(23);
  cudaDeviceSynchronize();
  double rest_ms = ms_since(t);

  cudaError_t e = cudaGetLastError();
  cudaFree(d);
  printf("init_ms=%.3f first_launch_ms=%.3f second_launch_ms=%.3f other_23_kernels_ms=%.3f status=%s\n",
         init_ms, first_ms, second_ms, rest_ms, cudaGetErrorName(e));
  return e == cudaSuccess ? 0 : 1;
}
