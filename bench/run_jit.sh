#!/usr/bin/env bash
# PTX JIT experiment: what does shipping PTX instead of SASS cost at run time?
# Median of 5 runs per row (ranked by first-launch time).
set -euo pipefail
cd "$(dirname "$0")/.."
CACHE=$(mktemp -d)

row() {  # row LABEL BINARY ENV... -> prints a markdown table row
  local label=$1 bin=$2; shift 2
  local line
  line=$(for i in 1 2 3 4 5; do env "$@" "./build/$bin"; done |
         awk -F'[ =]' '{print $4, $0}' | sort -n | sed -n 3p | cut -d' ' -f2-)
  local init first second rest
  init=$(sed -n 's/.*init_ms=\([0-9.]*\).*/\1/p' <<<"$line")
  first=$(sed -n 's/.*first_launch_ms=\([0-9.]*\).*/\1/p' <<<"$line")
  second=$(sed -n 's/.*second_launch_ms=\([0-9.]*\).*/\1/p' <<<"$line")
  rest=$(sed -n 's/.*other_23_kernels_ms=\([0-9.]*\).*/\1/p' <<<"$line")
  printf "| %s | %s | %s | %s | %s |\n" "$label" "$init" "$first" "$second" "$rest"
}

gpu=$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
echo "PTX JIT experiment on: $gpu   (24 unrolled kernels in one module, median of 5)"
echo
echo "| build / loading mode | context init ms | 1st launch ms | 2nd launch ms | other 23 kernels ms |"
echo "|---|---|---|---|---|"
row "SASS sm_XY, lazy loading (default)"          jit_sass CUDA_CACHE_DISABLE=1
row "PTX only, lazy loading, JIT cache off"        jit_ptx  CUDA_CACHE_DISABLE=1
env CUDA_CACHE_PATH="$CACHE" ./build/jit_ptx >/dev/null   # warm the JIT cache
row "PTX only, lazy loading, JIT cache warm"       jit_ptx  CUDA_CACHE_PATH="$CACHE"
row "PTX only, eager loading, JIT cache off"       jit_ptx  CUDA_CACHE_DISABLE=1 CUDA_MODULE_LOADING=EAGER
rm -rf "$CACHE"
echo
echo "What ptxray says about each binary:"
for b in jit_sass jit_ptx; do
  CUDA_CACHE_DISABLE=1 PTXRAY_TRACE=off PTXRAY_REPORT=/dev/stdout LD_PRELOAD="$PWD/build/libptxray.so" \
    "./build/$b" | grep -E -- "->|\[ptx-jit\]|\[first-launch\]|ms first" | sed "s/^/  [$b] /"
done
