// A double free and a free of a host pointer. Both must be reported, and the
// program's own view of the errors must be unchanged by the tracer.
#include "common.cuh"

int main() {
  void* p;
  CK(cudaMalloc(&p, 4096));
  CK(cudaFree(p));
  cudaError_t e1 = cudaFree(p);  // double free
  cudaGetLastError();
  int host_var = 0;
  cudaError_t e2 = cudaFree(&host_var);  // not a device pointer
  cudaGetLastError();
  printf("bad_free: double=%s host=%s\n", cudaGetErrorName(e1), cudaGetErrorName(e2));
  return 0;
}
