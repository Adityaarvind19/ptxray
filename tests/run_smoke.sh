#!/usr/bin/env bash
# Interposition smoke test. Works with or without a GPU:
#  * the program's output must be identical with and without the tracer
#    (results are forwarded unchanged), and
#  * the tracer must have intercepted the calls and printed its report.
set -euo pipefail
cd "$(dirname "$0")/.."
LIB="$PWD/build/libcudatrace.so"
TMP=$(mktemp -d)

plain=$(./build/smoke)
traced=$(cd "$TMP" && CUDATRACE_REPORT="$TMP/report.txt" LD_PRELOAD="$LIB" "$OLDPWD/build/smoke")

echo "without tracer: $plain"
echo "with tracer:    $traced"
[[ "$plain" == "$traced" ]] || { echo "FAIL: tracer changed program behaviour"; exit 1; }

cat "$TMP/report.txt"
grep -q "cudaMalloc" "$TMP/report.txt" || { echo "FAIL: cudaMalloc was not intercepted"; exit 1; }
grep -q "cudaDeviceSynchronize" "$TMP/report.txt" || { echo "FAIL: sync not intercepted"; exit 1; }
python3 -c "import json,glob,sys; [json.load(open(f)) for f in glob.glob('$TMP/cudatrace-*.json')] or sys.exit(1)" \
  || { echo "FAIL: no valid trace JSON written"; exit 1; }
echo "smoke: PASS"
