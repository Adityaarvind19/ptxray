// Three allocations, only one freed: the report must list two outstanding
// allocations and name make_buffers() as the allocation site.
#include "common.cuh"

__attribute__((noinline)) void make_buffers(void** a, void** b, void** c) {
  CK(cudaMalloc(a, 1 << 20));
  CK(cudaMalloc(b, 2 << 20));
  CK(cudaMalloc(c, 4 << 20));
}

int main() {
  void *a, *b, *c;
  make_buffers(&a, &b, &c);
  CK(cudaFree(b));
  printf("leak: done (leaked 1 MiB + 4 MiB on purpose)\n");
  return 0;
}
