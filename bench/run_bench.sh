#!/usr/bin/env bash
# Per-launch overhead of the tracer: median of 5 runs for each configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
LIB="$PWD/build/libcudatrace.so"
N=${1:-100000}

median_us() {  # median us_per_launch over 5 runs of the given env prefix
  for i in 1 2 3 4 5; do
    env "$@" ./build/bench_launch "$N" 2>/dev/null | sed -n 's/.*us_per_launch=\([0-9.]*\).*/\1/p'
  done | sort -n | sed -n 3p
}

base=$(median_us X=1)
full=$(median_us LD_PRELOAD="$LIB" CUDATRACE_TRACE=off CUDATRACE_REPORT=/dev/null)
light=$(median_us LD_PRELOAD="$LIB" CUDATRACE_TRACE=off CUDATRACE_REPORT=/dev/null CUDATRACE_GPU_TIMING=0)

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
echo "Kernel launch overhead, $N empty-kernel launches, median of 5 runs, GPU: $gpu"
echo
echo "| configuration | us / launch | overhead us / launch |"
echo "|---|---|---|"
printf "| no tracer | %.2f | - |\n" "$base"
awk -v b="$base" -v l="$light" -v f="$full" 'BEGIN {
  printf "| cudatrace, API trace only (CUDATRACE_GPU_TIMING=0) | %.2f | %+.2f |\n", l, l - b
  printf "| cudatrace, API trace + GPU event timing (default) | %.2f | %+.2f |\n", f, f - b }'
