#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(call)                                                                  \
  do {                                                                            \
    cudaError_t e_ = (call);                                                      \
    if (e_ != cudaSuccess) {                                                      \
      fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call,             \
              cudaGetErrorString(e_));                                            \
      exit(2);                                                                    \
    }                                                                             \
  } while (0)
