// Happy path: allocate, copy in, launch the same kernel 10 times, copy out,
// verify, free everything. The tracer must see 10 launches and no leaks, and
// must not change the program's results.
#include "common.cuh"
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 20;
  std::vector<float> a(n), b(n), c(n);
  for (int i = 0; i < n; ++i) { a[i] = float(i); b[i] = 2.0f * i; }
  float *da, *db, *dc;
  CK(cudaMalloc(&da, n * sizeof(float)));
  CK(cudaMalloc(&db, n * sizeof(float)));
  CK(cudaMalloc(&dc, n * sizeof(float)));
  CK(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  for (int it = 0; it < 10; ++it) vector_add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
  CK(cudaGetLastError());
  CK(cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i)
    if (c[i] != 3.0f * i) { printf("basic: MISMATCH at %d\n", i); return 1; }
  CK(cudaFree(da));
  CK(cudaFree(db));
  CK(cudaFree(dc));
  printf("basic: OK (checksum c[n-1]=%.1f)\n", c[n - 1]);
  return 0;
}
