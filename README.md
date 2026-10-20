# cudatrace

**A loader-level inspector for CUDA programs: it shows which GPU code objects a binary carries (SASS cubins and PTX), predicts whether the driver will load native SASS or JIT-compile PTX, measures what that JIT costs, and traces every CUDA runtime call, kernel, copy and allocation, all without recompiling the program.**

```sh
make && ./bin/cudatrace -- ./my_cuda_app
```

cudatrace is a C++17 shared library injected with `LD_PRELOAD`. It sits between an unmodified program and the CUDA runtime (`libcudart.so`) and intercepts two layers:

| Layer | What is intercepted | What cudatrace reports |
|---|---|---|
| **Code objects** (static init, before `main`) | `__cudaRegisterFatBinary`, `__cudaRegisterFunction`: how nvcc-generated host code hands its embedded GPU code to the runtime | Every embedded image (SASS `sm_XY` ELF cubin or PTX `compute_XY`), its size and compression, the PTX `.version`/`.target`/`.entry` directives, which image the driver will pick for the actual GPU, and whether it must be **JIT-compiled from PTX** by the driver's built-in ptxas |
| **Runtime API** | `cudaLaunchKernel`, `cudaMalloc*`/`cudaFree*`, `cudaMemcpy*`/`cudaMemset*`, `cudaHostAlloc`, `*Synchronize`, including the per-thread-default-stream `_ptsz`/`_ptds` variants | GPU time per kernel and per copy (CUDA events, no added synchronization), first-launch latency, memory leaks, double/invalid frees, pageable-memory async copies, small-copy and sync-heavy patterns, API errors with call sites, and a Perfetto timeline |

## Why the code-object layer matters

A CUDA binary is a host ELF with a `.nv_fatbin` section holding one or more **fatbinary containers**. Each container carries SASS cubins for specific architectures and/or PTX. At load time the driver must pick one image per module for the GPU it is running on:

- a **cubin for `sm_XY`** runs on devices with the same major version and minor ≥ Y;
- otherwise the driver **JIT-compiles the newest compatible PTX** (the Driver Compiler Interface invoking ptxas at run time), caching the result in `~/.nv/ComputeCache`;
- otherwise launches fail with `cudaErrorNoKernelImageForDevice`.

A missing `-gencode ...,code=sm_XY` therefore does not fail. It silently moves ptxas from build time to the user's first kernel launch. cudatrace makes that visible:

```
==cudatrace== GPU code objects: 1 fatbinary registered; device sm_75
==cudatrace==   jit_ptx: 1 module(s), 24 kernel(s); images: compute_75
==cudatrace==     on sm_75: 0 native SASS, 1 PTX JIT, 0 no compatible image
==cudatrace==     module 1: PTX compute_75 (ISA 9.4) 373.8 KiB [zstd]
==cudatrace==       kernels: void poly<23>(float*, int), void poly<22>(float*, int), ... +20 more
==cudatrace==       -> no SASS for sm_75: driver JIT-compiles PTX compute_75 (ISA 9.4) (ptxas at run time)
==cudatrace==       PTX: .version 9.4, .target sm_75, 24 .entry
...
==cudatrace==   [ptx-jit] 1 module(s) (e.g. in jit_ptx) carry no SASS for sm_75, so the driver JIT-compiles
                their PTX at load time; build with -gencode arch=compute_75,code=sm_75 to ship native code
==cudatrace==   [first-launch] kernels whose first launch was much slower than later ones ...
==cudatrace==       void poly<0>(float*, int): <first-launch ms> first, <steady us> afterwards [module 1: PTX JIT]
```

`--dump-ptx DIR` writes each embedded PTX module to disk (zstd-compressed PTX is decoded through a `dlopen`ed libzstd).

## Results (NVIDIA T4, Google Colab)

Reproduce everything with [`notebooks/cudatrace_colab.ipynb`](notebooks/cudatrace_colab.ipynb) (free T4 runtime, "Run all").

### PTX JIT experiment: `make jit`

The same source (24 heavily unrolled kernels in one module) built with native SASS and with PTX only:

Measured on a Tesla T4 (sm_75), driver 580.82.07, CUDA 13 toolchain (PTX ISA 9.0), Google Colab. Median of 5 runs.

| build / loading mode | context init ms | 1st launch ms | 2nd launch ms | other 23 kernels ms |
|---|---|---|---|---|
| SASS sm_75, lazy loading (default) | 150.8 | **0.74** | 0.015 | 0.53 |
| PTX only, lazy loading, JIT cache off | 149.8 | **318.4** | 0.019 | 0.55 |
| PTX only, lazy loading, JIT cache warm | 151.4 | **2.84** | 0.016 | 0.44 |
| PTX only, eager loading, JIT cache off | **475.1** | 0.03 | 0.015 | 0.23 |

- Shipping only PTX makes the first launch **~430× slower** (318 ms vs 0.74 ms). The driver runs ptxas on the whole module at that point; the other 23 kernels in the module then cost nothing extra.
- The driver's JIT cache (`~/.nv/ComputeCache`) brings a repeat run down to 2.8 ms. Every fresh container or new driver pays the full cost again.
- With `CUDA_MODULE_LOADING=EAGER` the same ~325 ms moves from the first launch into context creation (475 ms vs 151 ms).
- cudatrace predicts this from the fatbinary alone (`no SASS for sm_75: driver JIT-compiles PTX compute_75`) and flags the missing `-gencode ...,code=sm_75`.

### Tracer overhead: `make bench`

Host-side cost per empty-kernel launch, median of 5 runs of 100,000 launches:

| configuration | us / launch | overhead us / launch |
|---|---|---|
| no tracer | 2.57 | - |
| cudatrace, API trace only (`CUDATRACE_GPU_TIMING=0`) | 4.00 | +1.43 |
| cudatrace, API trace + GPU event timing (default) | 7.87 | +5.30 |

This is the worst case (empty kernels, launch-bound). The default mode pays for two `cudaEventRecord` calls per launch. The API-only mode costs 1.4 µs per call.

### Test suite

- `make test`: unit tests for the fatbinary parser (against real nvcc output in `tests/data/`), the image selection rules, memory accounting, diagnostics and trace output. No GPU needed.
- `make smoke`: interposition test that runs on machines without a GPU (CI). It checks that results are forwarded unchanged and that calls are recorded.
- `make gpu-test`: test programs on a real GPU (leaks, double free, pageable vs pinned copies, 4 threads × 4 streams, OOM, SASS-only vs PTX-only builds), each checked against the report.

## How it works

```
 unmodified program (nvcc -cudart shared, or PyTorch, ...)
   │  static init: __cudaRegisterFatBinary / __cudaRegisterFunction ─┐
   │  runtime:     cudaLaunchKernel / cudaMemcpyAsync / cudaFree ... ─┤  PLT calls bind to the
   ▼                                                                  │  preloaded library first
 libcudatrace.so (LD_PRELOAD) ◄───────────────────────────────────────┘
   ├─ fatbin.cpp  parse containers → images → PTX header → driver image selection
   ├─ hooks.cpp   record, time with cudaEventRecord on the same stream, forward
   │              real symbols: dlsym(RTLD_NEXT) or dlopen(soname, RTLD_NOLOAD)
   └─ core.cpp    aggregation, diagnostics, report, Chrome-trace JSON (Perfetto)
   ▼
 libcudart.so ─► libcuda.so (driver: module load, PTX JIT, launch)
```

Design points (details and trade-offs in [NOTES.md](NOTES.md)):

- **No added synchronization.** Each timed operation gets a start/stop event pair on its own stream. Pairs are resolved with `cudaEventQuery` at the application's own sync points and with `cudaEventSynchronize` only at exit. GPU timestamps are mapped to host time through one base event.
- **Safe inside CUDA graphs.** Operations on a stream under capture are not timed, because recording an event would alter the graph.
- **Does not perturb the application.** The tracer's internal CUDA calls never leave an error for the application's `cudaGetLastError()`, and all results are passed through untouched (`make smoke` checks this).
- **Correct at exit.** The GPU-event drain runs from an `atexit` handler registered after libcudart's, so it runs while the context is alive. The report is written from an ELF destructor, after static destructors, so frees during teardown are counted. All process-lifetime state is intentionally leaked so it outlives those destructors.
- **`fork`-aware.** Children (e.g. DataLoader workers) only report if they use CUDA themselves.

## Usage

```sh
make                      # build/libcudatrace.so (needs CUDA headers; does not link libcudart)
./bin/cudatrace [-o trace.json] [-r report.txt] [--dump-ptx DIR] [--no-gpu-timing] -- program args...
```

| Variable | Default | Meaning |
|---|---|---|
| `CUDATRACE_TRACE` | `cudatrace-<pid>.json` | Perfetto/Chrome trace path, `off` to disable |
| `CUDATRACE_REPORT` | stderr | report path |
| `CUDATRACE_DUMP_PTX` | unset | directory to write embedded PTX into |
| `CUDATRACE_GPU_TIMING` | `1` | `0` = API tracing only (lower overhead) |
| `CUDATRACE_MAX_EVENTS` | `1000000` | cap on timeline events (statistics stay complete) |

## Limitations

- **Static CUDA runtime.** Programs linked with nvcc's default `-cudart static` contain their own copy of the runtime, so there is nothing to interpose. Build with `-cudart shared`; the launcher warns when it detects a static build. PyTorch and most frameworks use the shared runtime.
- **Driver API launches.** Kernels launched directly through the driver API (`cuLaunchKernel`), as cuBLAS, cuDNN and Triton do, are not seen by runtime hooks. The natural extension is to interpose `cuGetProcAddress`, which is how libcudart itself obtains driver entry points, and wrap `cuModuleLoadDataEx`/`cuLibraryLoadData` to also capture JIT options and the JIT info/error log.
- **Image selection** follows the documented compatibility rules. Arch-specific targets (`sm_90a`) are exact-match only, but the fatbinary header does not mark them, so they are treated like ordinary cubins.
- **Legacy fatbin compression.** PTX compressed with the pre-12.8 format is listed but not decoded (build with `-Xfatbin -compress=false`, or use `cuobjdump -ptx`).
- Single-GPU processes are assumed for the timeline clock and the device query.

## Layout

```
src/fatbin.{hpp,cpp}   fatbinary container parser, PTX header parser, image selection (no CUDA dependency)
src/core.{hpp,cpp}     tracer state, diagnostics, report, trace writer (no CUDA dependency)
src/hooks.cpp          LD_PRELOAD interposition layer
tests/                 unit tests, nvcc fixtures, smoke test, GPU test programs + runner
bench/                 launch-overhead benchmark, PTX JIT experiment
examples/              PyTorch training loop with two classic mistakes
notebooks/             Colab notebook that reproduces every result
```
