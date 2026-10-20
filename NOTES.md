# Design notes and interview prep

These notes cover why each piece is built the way it is. Each answer should be explainable on a whiteboard without looking at the code.

## 1. How interposition works

**Q: What does `LD_PRELOAD` actually do?**
The dynamic linker (`ld.so`) loads the listed libraries right after the executable and before its `DT_NEEDED` dependencies, and puts them in the **global lookup scope** in that order. When any object resolves an undefined symbol through its PLT/GOT (for example the program calling `cudaMalloc`), `ld.so` searches the global scope in order. Our library defines `cudaMalloc`, so it wins over `libcudart.so`. This also applies to libraries loaded later with `dlopen` (e.g. `libtorch_cuda.so`), because the global scope is always searched before a library's local scope.

**Q: How do you call the real function?**
`dlsym(RTLD_NEXT, "cudaMalloc")` searches the objects *after* ours in the lookup order. One case breaks this: when libcudart was `dlopen`ed with `RTLD_LOCAL` (common in Python), it is not in the global scope, and `RTLD_NEXT` returns NULL. The fallback is `dlopen("libcudart.so.12", RTLD_NOLOAD)`, which returns a handle to the already-loaded copy without loading a second one, followed by `dlsym` on that handle. Pointers are resolved once and cached in function-local statics.

**Q: Why doesn't the library link against libcudart?**
If it did, it would pull in its own copy (e.g. the system `/usr/local/cuda` one) while PyTorch uses the pip-installed one. Two runtimes in one process means two sets of state. Resolving at run time always targets whatever the program actually loaded.

**Q: Why doesn't it work with `-cudart static`?**
With the static runtime, `cudaMalloc` is a local function inside the executable. Calls to it are direct, not through the PLT, so the dynamic linker is never consulted. There is nothing to interpose.

**Q: What is the re-entrancy guard for?**
A thread-local depth counter means only the outermost intercepted call is traced. If the tracer's own work ever triggered a hooked function, it would be forwarded untouched instead of recursing.

**Q: Why `-fvisibility=hidden`?**
Only the hook symbols are exported. Internal helpers (the tracer, parser, `std::` instantiations) must not interpose on, or be interposed by, symbols in the traced program.

## 2. Code objects, fatbinaries and the Driver Compiler Interface

**Q: What is in a CUDA executable?**
It's a host ELF. nvcc embeds device code in a `.nv_fatbin` section, and a small wrapper struct in `.nvFatBinSegment`: magic `0x466243B1`, version, a pointer to the fatbin, and an extra pointer. At static-init time nvcc's generated code calls `__cudaRegisterFatBinary(&wrapper)`, then `__cudaRegisterFunction(handle, hostStub, deviceName, ...)` for every kernel. That call is how `kernel<<<...>>>` (a call to a host stub) is later mapped to the device function. nvcc also registers `__cudaUnregisterFatBinary` with `atexit`.

**Q: Describe the fatbin layout.**
A container header holds magic `0xBA55ED50`, version, header size and payload size. It's followed by entries, each with its own header: kind (1 = PTX, 2 = ELF cubin), header size, payload size, compressed size, PTX ISA version (for PTX), the arch number (75 = sm_75/compute_75), flags, and the uncompressed size. Flag `0x8000` means zstd compression (nvcc 12.8+), `0x2000` the older format. I worked this out by dumping real nvcc output byte by byte, and the unit tests run against those dumps (`tests/data/`).

**Q: SASS vs PTX?**
SASS is the native machine code for one architecture family, stored as an ELF cubin produced by ptxas. PTX is NVIDIA's virtual ISA: a stable, forward-compatible target that ptxas lowers to SASS for a specific GPU. Shipping PTX buys forward compatibility (it runs on GPUs that didn't exist at build time), and the price is compiling at load time.

**Q: Which image does the driver pick?**
A cubin for sm_XY runs on devices of the same major version with minor ≥ Y (sm_70 runs on 7.5, not on 8.0), and the newest compatible cubin is preferred. If no cubin fits, the driver JIT-compiles the newest PTX whose compute_XY ≤ the device's capability. If neither exists, launches fail with `cudaErrorNoKernelImageForDevice`. `select_image()` implements this, and the unit tests cover each case (including "PTX too new" and "SASS from another major").

**Q: What is the Driver Compiler Interface, and where does JIT happen?**
The driver contains a JIT compiler: ptxas built into `libcuda`/`libnvidia-ptxjitcompiler`. It's reached explicitly through `cuModuleLoadDataEx` (with `CU_JIT_*` options such as optimization level, info/error log buffers and wall time), `cuLink*`, or `cuLibraryLoadData`. It's reached implicitly when the runtime loads a module that has only PTX for the current GPU. Results are cached in `~/.nv/ComputeCache` (`CUDA_CACHE_DISABLE`, `CUDA_CACHE_PATH`, `CUDA_CACHE_MAXSIZE`).

**Q: When does the JIT cost appear?**
With **lazy loading** (`CUDA_MODULE_LOADING=LAZY`, the default since CUDA 12.2), a module is loaded, and JIT-compiled if needed, when one of its kernels is first used, so the cost lands on the first launch. With **eager** loading it lands on context creation. `bench/run_jit.sh` measures both, plus the warm-cache case. The tracer attributes first-launch latency to modules it predicted would be JIT-compiled.

**Q: How did you make the JIT measurable?**
The benchmark puts 24 kernels, each with a 96-iteration fully unrolled loop, into one module, so its PTX is about 400 KB. Offline `ptxas` takes a few hundred milliseconds on it, and the driver's JIT does comparable work on first use.

## 3. Timing without adding synchronization

**Q: How do you time a kernel without a sync?**
Record event A on the kernel's stream before the launch and event B after it. Both are executed by the GPU in stream order, so `cudaEventElapsedTime(A, B)` is the kernel's GPU time (plus a sub-microsecond gap). The pairs sit in a pending list. They're resolved with `cudaEventQuery` (non-blocking) whenever the *application* synchronizes, and with `cudaEventSynchronize` only at exit. The tracer never adds a sync point of its own, and events are recycled through a pool.

**Q: How do GPU timestamps line up with host time?**
On first use the tracer records a base event, synchronizes on it once, and notes the host clock. Every GPU event's timestamp is `host_base + elapsed(base, event)`.

**Q: CUDA graphs?**
Recording an event on a stream under capture would insert a node into the user's graph. So `cudaStreamIsCapturing` is checked first and captured work stays untimed (it's counted in the report). Draining only happens at sync points, which are illegal during global-mode capture anyway, so the tracer never makes a capture-unsafe call itself.

**Q: Why can't the tracer change the app's error state?**
`cudaGetLastError` is sticky per thread. If an internal `cudaEventCreate` failed, the application would later see an error it never caused. `ErrorStateGuard` peeks before internal calls and clears afterwards, but only if the state was clean before, so a pending application error is never consumed.

**Q: What is the overhead?**
Per launch: two `cudaEventRecord` calls, a capture check, a mutex-protected append and a hash lookup for the kernel name. `make bench` measures it, and `PTXRAY_GPU_TIMING=0` drops the event part.

## 4. Process lifetime

**Q: When is the report written, and why there?**
From an ELF destructor of the preloaded library. That runs after `main` returns, after all `atexit` handlers and after static destructors, so `cudaFree` calls made during teardown are still counted. During teardown libcudart returns `cudaErrorCudartUnloading`; that's treated as a successful free, because destroying the context releases the memory.

**Q: But the GPU events must be read while the context is still alive?**
Yes. That's why the drain is a separate `atexit` handler, registered *after* the first real CUDA call, which is after libcudart registered its own teardown. `atexit` is LIFO, so the drain runs first.

**Q: Why is all global state leaked (`*new T`)?**
Function-local statics and globals with destructors are destroyed before the ELF destructor runs. An early version printed a garbage trace path at exit for exactly this reason: the `std::string` in a static `Settings` had already been destroyed. Heap objects that are never freed outlive everything.

**Q: `fork`?**
`pthread_atfork` resets the "has activity" flags in the child, so DataLoader workers that never touch CUDA don't print duplicate reports.

## 5. Diagnostics: what they mean

- **outstanding**: allocations live at exit, with the symbolized allocation site (`dladdr` on the return address, demangled; build with `-rdynamic` for function names, or use `addr2line` with the printed offset). PyTorch's caching allocator keeps memory by design, so the report says "leak *or* caching allocator".
- **bad-free**: a double free (the pointer is in the freed set, not re-allocated since), an invalid free (unknown pointer and the real call failed), or untracked (unknown pointer but the real call succeeded, i.e. allocated by an API not hooked).
- **pageable**: `cudaMemcpyAsync` from pageable memory is staged through a pinned driver buffer and does not overlap with host work. Detected with `cudaPointerGetAttributes` (`cudaMemoryTypeUnregistered`).
- **small-copy / sync-heavy**: many tiny transfers, or a large share of wall time blocked in synchronize calls or synchronous copies (the classic per-step `.item()`).

## 6. What I'd do next

1. Interpose `cuGetProcAddress` to see driver-API launches (cuBLAS/cuDNN/Triton) and wrap `cuModuleLoadDataEx`/`cuLibraryLoadData`. That gives JIT time per module directly, plus the JIT info log (registers and spills reported by the driver's ptxas).
2. Parse the cubin ELF itself (`.nv.info.*` sections) for per-kernel register count, shared memory and max threads. That's what `cuobjdump -elf` shows.
3. A `--check` mode that fails CI when a binary would hit the PTX JIT path on a list of target GPUs.
4. CUPTI Activity API as an alternative timing back end, to compare against event timing.
