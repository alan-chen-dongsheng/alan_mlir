#!/bin/bash
set -euo pipefail

# Alan Eltwise CPU Execution Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"
ALAN_OPT="${BUILD_DIR}/tools/alan-opt/alan-opt"
MLIR_OPT="${MLIR_OPT:-mlir-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"
LLC="${LLC:-llc}"
CLANG="${CLANG:-clang}"

# Test MLIR file
TEST_MLIR="$1"

if [ -z "$TEST_MLIR" ] || [ ! -f "$TEST_MLIR" ]; then
  echo "Usage: $0 <test.mlir>"
  exit 1
fi

echo "=== Alan Eltwise CPU Execution ==="
echo "Input: $TEST_MLIR"

# Create temp directory
TMP_DIR="${TMP_DIR:-/tmp/alan_cpu_$$}"
mkdir -p "$TMP_DIR"
echo "Temp dir: $TMP_DIR"

echo ""
echo "1. Lowering Alan to Linalg..."
"$ALAN_OPT" "$TEST_MLIR" \
  --convert-alan-to-linalg \
  -o "$TMP_DIR/step1_linalg.mlir"
echo "   Done: $TMP_DIR/step1_linalg.mlir"

echo ""
echo "2. Bufferizing to memrefs..."
"$MLIR_OPT" "$TMP_DIR/step1_linalg.mlir" \
  --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs-from-loops" \
  --convert-bufferization-to-memref \
  -o "$TMP_DIR/step2_bufferized.mlir"
echo "   Done: $TMP_DIR/step2_bufferized.mlir"

echo ""
echo "3. Lowering Linalg to loops..."
"$MLIR_OPT" "$TMP_DIR/step2_bufferized.mlir" \
  --convert-linalg-to-parallel-loops \
  --convert-scf-to-cf \
  -o "$TMP_DIR/step3_loops.mlir"
echo "   Done: $TMP_DIR/step3_loops.mlir"

echo ""
echo "4. Lowering to LLVM dialect..."
"$MLIR_OPT" "$TMP_DIR/step3_loops.mlir" \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts \
  -o "$TMP_DIR/step4_llvm.mlir"
echo "   Done: $TMP_DIR/step4_llvm.mlir"

echo ""
echo "5. Translating MLIR to LLVM IR..."
"$MLIR_TRANSLATE" "$TMP_DIR/step4_llvm.mlir" --mlir-to-llvmir -o "$TMP_DIR/module.ll"
echo "   Done: $TMP_DIR/module.ll"

echo ""
echo "6. Compiling LLVM IR to object..."
"$LLC" -filetype=obj "$TMP_DIR/module.ll" -o "$TMP_DIR/module.o"
echo "   Done: $TMP_DIR/module.o"

echo ""
echo "7. Linking with runtime..."
"$CLANG" -O2 -lm \
  "$TMP_DIR/module.o" \
  "${PROJECT_ROOT}/runtime/cpu/eltwise_runner.c" \
  -o "$TMP_DIR/test_runner"
echo "   Done: $TMP_DIR/test_runner"

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
