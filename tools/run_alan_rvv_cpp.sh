#!/bin/bash
set -euo pipefail

# Alan Eltwise RVV Execution via C++ source code generation
# Pipeline: Alan → Linalg → SCF → EmitC → C++ source → riscv64-g++ -O3 (auto-vectorize) → spike

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"
export PATH="/usr/local/opt/llvm/bin:/usr/local/bin:$PATH"

ALAN_OPT="${BUILD_DIR}/tools/alan-opt/alan-opt"
MLIR_OPT="mlir-opt"
MLIR_TRANSLATE="mlir-translate"
RISCV_GCC="riscv64-unknown-elf-g++"
RISCV_OBJDUMP="riscv64-unknown-elf-objdump"
SPIKE="spike"
PK="${PK:-/usr/local/Cellar/riscv-pk/main/riscv64-unknown-elf/bin/pk}"

TEST_MLIR="$1"
if [ -z "$TEST_MLIR" ] || [ ! -f "$TEST_MLIR" ]; then
  echo "Usage: $0 <test.mlir>"
  exit 1
fi

echo "=== Alan Eltwise RVV (C++ source) Spike Execution ==="
echo "Input: $TEST_MLIR"

TMP_DIR="${TMP_DIR:-/tmp/alan_rvv_cpp_$$}"
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
python3 "${SCRIPT_DIR}/fix_emitc.py" < "$TMP_DIR/step4_emitc_raw.mlir" \
  > "$TMP_DIR/step5_emitc.mlir"
"$MLIR_OPT" "$TMP_DIR/step5_emitc.mlir" --reconcile-unrealized-casts \
  -o "$TMP_DIR/step6_emitc_clean.mlir"
echo "   Done"

echo ""
echo "6. Translating to C++ source..."
"$MLIR_TRANSLATE" --mlir-to-cpp "$TMP_DIR/step6_emitc_clean.mlir" \
  -o "$TMP_DIR/generated.cpp"
printf '#include <cstddef>\n#include <cstdint>\n#include <cmath>\n' > "$TMP_DIR/generated_full.cpp"
cat "$TMP_DIR/generated.cpp" >> "$TMP_DIR/generated_full.cpp"
echo "   Done: $TMP_DIR/generated.cpp"

echo ""
echo "7. Cross-compiling with riscv64-unknown-elf-g++ -O3 -march=rv64gcv..."
"$RISCV_GCC" -std=c++17 -march=rv64gcv -mabi=lp64d -O3 -lm \
  "$TMP_DIR/generated_full.cpp" \
  "${PROJECT_ROOT}/runtime/rvv/eltwise_runner_cpp.cpp" \
  -o "$TMP_DIR/test_runner.elf"
echo "   Done"

echo ""
echo "8. Checking for RVV instructions..."
RVV_COUNT=0
if command -v "$RISCV_OBJDUMP" &>/dev/null; then
  "$RISCV_OBJDUMP" -d "$TMP_DIR/test_runner.elf" > "$TMP_DIR/disasm.txt" 2>/dev/null
  if [ -f "$TMP_DIR/disasm.txt" ]; then
    RVV_COUNT=$(grep -E '\b(vadd|vsub|vmul|vle|vse|vfadd|vfmul|vfmax|vsetvl|vfmv|vl)\.' "$TMP_DIR/disasm.txt" 2>/dev/null | wc -l | tr -d ' ')
    echo "   Found $RVV_COUNT RVV instructions in binary"
    if [ "$RVV_COUNT" -eq "0" ]; then
      echo "   INFO: Using scalar code path (auto-vectorization did not trigger)"
    fi
  fi
else
  echo "   $RISCV_OBJDUMP not found, skipping check"
fi

echo ""
echo "9. Running on Spike..."
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
