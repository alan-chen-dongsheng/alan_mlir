#!/bin/bash
set -euo pipefail

# Alan Eltwise CPU Execution via C++ source code generation
# Pipeline: Alan → Linalg → SCF → EmitC → C++ source → clang++ -std=c++17 → run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"
ALAN_OPT="${BUILD_DIR}/tools/alan-opt/alan-opt"
MLIR_OPT="${MLIR_OPT:-mlir-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"
CLANG="${CLANG:-clang++}"

TEST_MLIR="$1"
if [ -z "$TEST_MLIR" ] || [ ! -f "$TEST_MLIR" ]; then
  echo "Usage: $0 <test.mlir>"
  exit 1
fi

echo "=== Alan Eltwise CPU (C++ source) Execution ==="
echo "Input: $TEST_MLIR"

TMP_DIR="${TMP_DIR:-/tmp/alan_cpu_cpp_$$}"
mkdir -p "$TMP_DIR"
echo "Temp dir: $TMP_DIR"

echo ""
echo "1. Lowering Alan to Linalg..."
"$ALAN_OPT" "$TEST_MLIR" --convert-alan-to-linalg -o "$TMP_DIR/step1_linalg.mlir"
echo "   Done"

echo ""
echo "2. Bufferizing + lowering to loops..."
"$MLIR_OPT" "$TMP_DIR/step1_linalg.mlir" \
  --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs-from-loops" \
  --convert-bufferization-to-memref \
  --convert-linalg-to-loops \
  --buffer-results-to-out-params="modify-public-functions hoist-static-allocs" \
  -o "$TMP_DIR/step2_loops.mlir"
echo "   Done"

echo ""
echo "3. Normalizing strided memrefs..."
"$ALAN_OPT" "$TMP_DIR/step2_loops.mlir" --normalize-strided-memref \
  -o "$TMP_DIR/step3_normalized.mlir"
echo "   Done"

echo ""
echo "4. Converting to EmitC dialect..."
"$MLIR_OPT" "$TMP_DIR/step3_normalized.mlir" \
  --arith-expand \
  --convert-arith-to-emitc \
  --convert-scf-to-emitc \
  --convert-memref-to-emitc \
  --convert-func-to-emitc \
  -o "$TMP_DIR/step4_emitc_raw.mlir"
echo "   Done"

echo ""
echo "5. Fixing up EmitC function boundary types..."
# NOTE: This Python script is a workaround for incomplete EmitC dialect conversion.
# It fixes: memref->ptr conversion, unrealized casts, array->ptr, subscript index types.
# See docs/emitc_workaround.md for details.
python3 "${SCRIPT_DIR}/fix_emitc.py" < "$TMP_DIR/step4_emitc_raw.mlir" \
  > "$TMP_DIR/step5_emitc.mlir"
"$MLIR_OPT" "$TMP_DIR/step5_emitc.mlir" --reconcile-unrealized-casts \
  -o "$TMP_DIR/step6_emitc_clean.mlir"
echo "   Done"

echo ""
echo "6. Translating to C++ source..."
"$MLIR_TRANSLATE" --mlir-to-cpp "$TMP_DIR/step6_emitc_clean.mlir" \
  -o "$TMP_DIR/generated.cpp"
printf '#include <cstddef>\n#include <cstdint>\n#include <cmath>\n' > "$TMP_DIR/generated_with_header.cpp"
cat "$TMP_DIR/generated.cpp" >> "$TMP_DIR/generated_with_header.cpp"
echo "   Done: $TMP_DIR/generated.cpp"

echo ""
echo "7. Compiling with clang++ (C++17)..."
"$CLANG" -std=c++17 -O2 -lm \
  "$TMP_DIR/generated_with_header.cpp" \
  "${PROJECT_ROOT}/runtime/cpu/eltwise_runner_cpp.cpp" \
  -o "$TMP_DIR/test_runner"
echo "   Done"

echo ""
echo "8. Running test..."
"$TMP_DIR/test_runner"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
  echo "SUCCESS: All tests passed!"
else
  echo "FAILURE: Tests failed with exit code $EXIT_CODE"
fi

if [ -z "${KEEP_TEMP:-}" ]; then
  rm -rf "$TMP_DIR"
else
  echo "Keeping temp dir: $TMP_DIR"
fi

exit $EXIT_CODE
