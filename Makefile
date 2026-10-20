# ptxray build
#   make            -> build/libptxray.so
#   make test       -> CPU unit tests (no GPU needed)
#   make smoke      -> interposition smoke test (no GPU needed, needs libcudart.so)
#   make gpu        -> GPU test programs + benchmark (needs nvcc)
#   make gpu-test   -> run the GPU test suite (needs an NVIDIA GPU)
#   make bench      -> tracer overhead per kernel launch
#   make jit        -> PTX JIT experiment: SASS vs PTX-only, JIT cache, lazy vs eager loading

CUDA_HOME ?= /usr/local/cuda
CUDA_INC  ?= $(CUDA_HOME)/include
CUDA_LIB  ?= $(CUDA_HOME)/lib64
CXX       ?= g++
NVCC      ?= $(CUDA_HOME)/bin/nvcc

CXXFLAGS  ?= -O2 -g
override CXXFLAGS += -std=c++17 -Wall -Wextra -Werror -fPIC -fvisibility=hidden -pthread
# Compute capability of the local GPU (75 for a T4); override with GPU_ARCH=86 etc.
GPU_ARCH  ?= $(or $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .),75)
# -cudart shared is essential: the static runtime cannot be interposed.
# -rdynamic (-export-dynamic) lets leak reports name the allocating function.
NVBASE    := -O2 -g -lineinfo -cudart shared -Xlinker -export-dynamic -std=c++17
NVFLAGS   ?= $(NVBASE) -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH) -gencode arch=compute_$(GPU_ARCH),code=compute_$(GPU_ARCH)

B := build
LIB := $(B)/libptxray.so
GPU_TESTS := basic leak bad_free pageable streams small_copies oom
GPU_BINS := $(addprefix $(B)/t_,$(GPU_TESTS)) $(B)/bench_launch $(B)/jit_sass $(B)/jit_ptx

.PHONY: all test smoke gpu gpu-test bench jit clean
all: $(LIB)

$(B):
	mkdir -p $(B)

$(B)/core.o: src/core.cpp src/core.hpp src/fatbin.hpp | $(B)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(B)/fatbin.o: src/fatbin.cpp src/fatbin.hpp | $(B)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(B)/hooks.o: src/hooks.cpp src/core.hpp src/fatbin.hpp | $(B)
	$(CXX) $(CXXFLAGS) -I$(CUDA_INC) -c $< -o $@

# Not linked against libcudart on purpose: the real functions are looked up at
# run time from whichever libcudart the traced program loaded.
$(LIB): $(B)/core.o $(B)/fatbin.o $(B)/hooks.o
	$(CXX) -shared -pthread -o $@ $^ -ldl

$(B)/core_test: tests/core_test.cpp $(B)/core.o $(B)/fatbin.o
	$(CXX) $(CXXFLAGS) -Isrc $^ -o $@ -ldl

test: $(B)/core_test
	./$(B)/core_test

$(B)/smoke: tests/smoke_nogpu.cpp | $(B)
	$(CXX) -std=c++17 -O1 -g -I$(CUDA_INC) $< -o $@ -L$(CUDA_LIB) -lcudart -Wl,-rpath,$(CUDA_LIB)

smoke: $(LIB) $(B)/smoke
	bash tests/run_smoke.sh

$(B)/t_%: tests/gpu/%.cu | $(B)
	$(NVCC) $(NVFLAGS) $< -o $@ -lpthread

$(B)/bench_launch: bench/launch_overhead.cu | $(B)
	$(NVCC) $(NVFLAGS) $< -o $@

# Same source, two code-object layouts: native SASS vs PTX only (driver JIT).
$(B)/jit_sass: bench/jit_cost.cu | $(B)
	$(NVCC) $(NVBASE) -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH) $< -o $@
$(B)/jit_ptx: bench/jit_cost.cu | $(B)
	$(NVCC) $(NVBASE) -gencode arch=compute_$(GPU_ARCH),code=compute_$(GPU_ARCH) $< -o $@

gpu: $(LIB) $(GPU_BINS)

gpu-test: gpu
	bash tests/run_gpu_tests.sh

bench: gpu
	bash bench/run_bench.sh

jit: gpu
	bash bench/run_jit.sh

clean:
	rm -rf $(B)
