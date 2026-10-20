#!/usr/bin/env bash
# GPU test suite: runs each test program under the tracer and checks the report.
# Needs an NVIDIA GPU (e.g. a Colab/Kaggle T4). Build first: make gpu
set -uo pipefail
cd "$(dirname "$0")/.."
LIB="$PWD/build/libptxray.so"
OUT="$PWD/build/gpu-test-out"
rm -rf "$OUT" && mkdir -p "$OUT"
pass=0; fail=0

run() {  # run NAME -> runs build/t_NAME under the tracer, report in $OUT/NAME.txt
  local name=$1
  (cd "$OUT" && PTXRAY_TRACE="$OUT/$name.json" PTXRAY_REPORT="$OUT/$name.txt" \
     LD_PRELOAD="$LIB" "$OLDPWD/build/t_$name" > "$OUT/$name.stdout" 2>&1)
}
expect() {  # expect NAME "pattern" "description"
  if grep -qE -- "$2" "$OUT/$1.txt"; then ((pass++)); echo "  ok   $1: $3"
  else ((fail++)); echo "  FAIL $1: $3 (pattern: $2)"; fi
}
check() {  # check NAME CONDITION_EXIT_STATUS "description"
  if [[ $2 == 0 ]]; then ((pass++)); echo "  ok   $1: $3"; else ((fail++)); echo "  FAIL $1: $3"; fi
}
valid_json() { python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$OUT/$1.json"; }

echo "== basic"
run basic; check basic $? "program succeeds under the tracer"
plain=$(./build/t_basic); check basic $([[ "$plain" == "$(cat "$OUT/basic.stdout")" ]]; echo $?) "output identical with and without tracer"
expect basic "GPU kernels: 10 timed launches" "all 10 launches timed"
expect basic "vector_add\(float const\*, float const\*, float\*, int\)" "kernel name demangled"
expect basic "Memcpy HtoD \(pageable\) +2 " "2 host-to-device copies classified"
expect basic "Memcpy DtoH \(pageable\) +1 " "1 device-to-host copy classified"
expect basic "peak device memory 12.0 MiB" "peak device memory tracked"
for d in outstanding bad-free pageable error ptx-jit no-image small-copy; do
  if grep -q "\[$d\]" "$OUT/basic.txt"; then ((fail++)); echo "  FAIL basic: false positive [$d]"; else ((pass++)); fi
done; echo "  ok   basic: no false-positive diagnostics"
valid_json basic; check basic $? "trace is valid JSON"

echo "== leak"
run leak; check leak $? "program succeeds"
expect leak "\[outstanding\] 2 allocation\(s\) totalling 5.0 MiB" "2 outstanding allocations found"
expect leak "4.0 MiB device at make_buffers" "allocation site symbolized"

echo "== bad_free"
run bad_free; check bad_free $? "program succeeds"
expect bad_free "1 double free\(s\), 1 invalid free\(s\)" "double and invalid free detected"

echo "== pageable"
run pageable; check pageable $? "program succeeds"
expect pageable "\[pageable\] 10 cudaMemcpyAsync" "10 pageable async copies flagged"
expect pageable "Memcpy HtoD \(pinned\)" "pinned copies identified"
expect pageable "Memcpy DtoH \(pageable\)" "pageable copies identified"

echo "== streams"
run streams; check streams $? "program succeeds"
expect streams "200 +[0-9.]+ +[0-9.]+ +[0-9.]+ +busy\(float\*, int\)" "all 200 launches from 4 threads timed"
lanes=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(len({e['tid'] for e in d['traceEvents'] if e.get('ph')=='X' and e['pid']==2 and e['cat']=='kernel'}))" "$OUT/streams.json")
check streams $([[ $lanes == 4 ]]; echo $?) "kernels on 4 separate stream tracks (got $lanes)"

echo "== small_copies"
run small_copies; check small_copies $? "program succeeds"
expect small_copies "\[small-copy\] 1000 of 1000" "small copies flagged"

echo "== code objects and the PTX JIT path"
expect basic "-> loads SASS sm_[0-9]+ natively on sm_[0-9]+" "native SASS selected for a -gencode sm build"
expect basic "PTX compute_[0-9]+" "embedded PTX image listed"
run_env() { local name=$1 bin=$2; shift 2
  (cd "$OUT" && env "$@" PTXRAY_TRACE="$OUT/$name.json" PTXRAY_REPORT="$OUT/$name.txt" \
     LD_PRELOAD="$LIB" "$OLDPWD/build/$bin" > "$OUT/$name.stdout" 2>&1); }
run_env jit_sass jit_sass CUDA_CACHE_DISABLE=1; check jit_sass $? "SASS build runs"
expect jit_sass "-> loads SASS sm_" "SASS-only build: native load predicted"
if grep -q "\[ptx-jit\]" "$OUT/jit_sass.txt"; then ((fail++)); echo "  FAIL jit_sass: unexpected [ptx-jit]"; else ((pass++)); echo "  ok   jit_sass: no JIT warning"; fi
run_env jit_ptx jit_ptx CUDA_CACHE_DISABLE=1; check jit_ptx $? "PTX-only build runs (driver JIT)"
expect jit_ptx "driver JIT-compiles PTX compute_" "PTX-only build: JIT predicted"
expect jit_ptx "\[ptx-jit\] 1 module" "missing -gencode SASS target flagged"
expect jit_ptx "\[first-launch\]" "JIT cost visible as first-launch latency"
expect jit_ptx "\[module 1: PTX JIT\]" "first-launch latency attributed to the JIT'd module"

echo "== oom"
run oom; check oom $? "error passed through unchanged"
expect oom "cudaMalloc returned cudaErrorMemoryAllocation 1 time\(s\); first at main" "error reported with site"

echo
echo "GPU tests: $pass passed, $fail failed  (reports and traces in build/gpu-test-out/)"
[[ $fail == 0 ]]
