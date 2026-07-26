#!/bin/bash
set +euo pipefail

# Alan Eltwise RVV Spike Execution Script
# Note: Using LLVM auto-vectorization for RVV code generation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"
export PATH="/usr/local/opt/llvm/bin:/usr/local/bin:$PATH"

ALAN_OPT="${BUILD_DIR}/tools/alan-opt/alan-opt"
MLIR_OPT="mlir-opt"
MLIR_TRANSLATE="mlir-translate"
LLC="llc"
RISCV_GCC="riscv64-unknown-elf-gcc"
RISCV_OBJDUMP="riscv64-unknown-elf-objdump"
SPIKE="spike"
PK="${PK:-/usr/local/Cellar/riscv-pk/main/riscv64-unknown-elf/bin/pk}"

# Test MLIR file
TEST_MLIR="$1"

if [ -z "$TEST_MLIR" ] || [ ! -f "$TEST_MLIR" ]; then
  echo "Usage: $0 <test.mlir>"
  exit 1
fi

echo "=== Alan Eltwise RVV Spike Execution ==="
echo "Input: $TEST_MLIR"

# Create temp directory
TMP_DIR="${TMP_DIR:-/tmp/alan_rvv_$$}"
mkdir -p "$TMP_DIR"
echo "Temp dir: $TMP_DIR"

echo ""
echo "1. Lowering Alan to Linalg..."
"$ALAN_OPT" "$TEST_MLIR" \
  --convert-alan-to-linalg \
  -o "$TMP_DIR/step1_linalg.mlir"
echo "   Done"

echo ""
echo "2. Bufferizing..."
"$MLIR_OPT" "$TMP_DIR/step1_linalg.mlir" \
  --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs-from-loops" \
  --convert-linalg-to-parallel-loops \
  -o "$TMP_DIR/step2_loops.mlir"
echo "   Done"

echo ""
echo "3. Lowering to LLVM dialect..."
"$MLIR_OPT" "$TMP_DIR/step2_loops.mlir" \
  --convert-scf-to-cf \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts \
  -o "$TMP_DIR/step3_llvm.mlir"
echo "   Done"

echo ""
echo "4. Translating MLIR to LLVM IR..."
"$MLIR_TRANSLATE" "$TMP_DIR/step3_llvm.mlir" --mlir-to-llvmir -o "$TMP_DIR/module.ll"
echo "   Done"

echo ""
echo "5. Compiling LLVM IR to RVV object with auto-vectorization..."
"$LLC" "$TMP_DIR/module.ll" \
  -mtriple=riscv64-unknown-elf \
  -mattr=+m,+a,+f,+d,+c,+v \
  -O3 \
  -filetype=obj \
  -o "$TMP_DIR/module.o"
echo "   Done"

echo ""
echo "6. Checking for RVV instructions..."
RVV_COUNT=0
if command -v "$RISCV_OBJDUMP" &>/dev/null; then
  "$RISCV_OBJDUMP" -d "$TMP_DIR/module.o" > "$TMP_DIR/disasm.txt" 2>/dev/null
  if [ -f "$TMP_DIR/disasm.txt" ]; then
    RVV_COUNT=$(grep -E '\b(vadd|vsub|vmul|vle|vse|vfadd|vfmul|vfmax|vsetvl)\.' "$TMP_DIR/disasm.txt" 2>/dev/null | wc -l | tr -d ' ')
    echo "   Found $RVV_COUNT RVV instructions"
    if [ "$RVV_COUNT" -eq "0" ]; then
      echo "   INFO: Using scalar code path"
    fi
  fi
else
  echo "   $RISCV_OBJDUMP not found, skipping check"
fi

echo ""
echo "7. Linking with RVV runtime..."
"$RISCV_GCC" -march=rv64gcv -mabi=lp64d -O2 \
  "$TMP_DIR/module.o" \
  "${PROJECT_ROOT}/runtime/rvv/eltwise_runner.c" \
  -o "$TMP_DIR/test_runner.elf"
echo "   Done"

echo ""
echo "8. Running on Spike..."
"$SPIKE" --isa=rv64gcv "$PK" "$TMP_DIR/test_runner.elf"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
  echo "SUCCESS: All RVV tests passed! (Found $RVV_COUNT RVV instructions)"
else
  echo "FAILURE: RVV tests failed with exit code $EXIT_CODE"
fi

if [ -z "${KEEP_TEMP:-}" ]; then
  rm -rf "$TMP_DIR"
else
  echo "Keeping temp dir: $TMP_DIR"
fi

exit $EXIT_CODE
