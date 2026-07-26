# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Alan MLIR is a custom MLIR-based compiler that implements element-wise tensor operations with dual-backend support: CPU execution and RISC-V RVV (Vector Extension) execution via Spike simulator.

## Build Commands

```bash
# Initial build setup
mkdir -p build && cd build
cmake .. -DMLIR_DIR=$(brew --prefix llvm)/lib/cmake/mlir \
         -DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm
make -j$(sysctl -n hw.ncpu)

# Rebuild after changes
cmake --build build -j8

# Run alan-opt tool
./build/tools/alan-opt/alan-opt <input.mlir> [options]
```

## Testing

```bash
# End-to-end CPU test
./tools/run_alan_cpu.sh test/Execution/Alan/eltwise_test.mlir

# End-to-end RVV/Spike test
./tools/run_alan_rvv_spike.sh test/Execution/Alan/eltwise_rvv_test.mlir

# Preserve intermediate files for debugging
KEEP_TEMP=1 ./tools/run_alan_cpu.sh test/Execution/Alan/eltwise_test.mlir
```

## Architecture

### Compilation Pipeline

The project implements two parallel lowering paths from the Alan dialect:

**CPU Path:**
```
alan.eltwise → linalg.generic → bufferization → parallel loops →
SCF → CF → Arith → LLVM IR → native executable
```

**RVV Path:**
```
alan.eltwise → linalg.generic → bufferization → parallel loops →
SCF → CF → Arith → LLVM IR → RISC-V RVV ELF → Spike simulator
```

Both paths share the Alan → Linalg conversion, then diverge for target-specific code generation.

### TableGen Integration

All passes are defined in `lib/Conversion/Passes.td` using MLIR's TableGen framework:

- `ConvertAlanToLinalg`: Converts alan.eltwise to linalg.generic
- `AlanCPULoweringPipeline`: Complete CPU lowering pipeline
- `AlanVectorization`: RVV vectorization pipeline

**Critical:** When adding new passes:
1. Define the pass in `Passes.td` with constructor and dependent dialects
2. Include the generated `Passes.h.inc` in the implementation file with `#define GEN_PASS_DEF_<PASSNAME>`
3. Inherit from `::impl::<PassName>Base<DerivedPass>` (note the `::impl::` namespace prefix to avoid ambiguity with `mlir::impl`)
4. Implement only `runOnOperation()` - other methods like `getArgument()` and `getDependentDialects()` are auto-generated
5. Register in `lib/Conversion/Passes.cpp` by including `Passes.h.inc` with `#define GEN_PASS_REGISTRATION`

### Dialect Structure

The Alan dialect defines a single operation `alan.eltwise` that performs element-wise binary operations on tensors:

```mlir
%result = alan.eltwise %lhs, %rhs {kind = "add"} 
          : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
```

Supported operations: `add`, `sub`, `mul`, `max`, `min`

The operation is defined in `include/Dialect/Alan/AlanOps.td` and verified in `lib/Dialect/Alan/AlanDialect.cpp`.

### Key Files

- `include/Conversion/Passes.h`: Public header for pass declarations
- `lib/Conversion/Passes.cpp`: Pass registration (calls generated registration functions)
- `lib/Conversion/AlanToLinalg/AlanToLinalg.cpp`: Core conversion pattern using `OpConversionPattern`
- `lib/Conversion/AlanToLLVM/AlanCPUPipeline.cpp`: CPU pipeline orchestrating multiple passes
- `tools/alan-opt/alan-opt.cpp`: Main driver registering dialects and passes

### Runtime Integration

The `runtime/` directory contains C test harnesses that:
1. Allocate and initialize input tensors
2. Call the compiled function using MemRef descriptor ABI
3. Verify results against expected output
4. Return 0 on success, non-zero on failure

The execution scripts (`tools/run_alan_*.sh`) orchestrate the full pipeline: MLIR → LLVM IR → object file → linking with runtime → execution.

## Common Tasks

### Adding a New Eltwise Operation

1. Add case in `lib/Conversion/AlanToLinalg/AlanToLinalg.cpp` lowering pattern (e.g., `kind == "div"`)
2. Update verifier in `lib/Dialect/Alan/AlanDialect.cpp` to accept the new kind
3. Add test case in `test/Execution/Alan/eltwise_test.mlir`

### Adding a New Lowering Pass

1. Define pass in `lib/Conversion/Passes.td`
2. Create implementation in `lib/Conversion/<PassName>/<PassName>.cpp`
3. Add CMakeLists.txt for the new pass library
4. Update `lib/Conversion/CMakeLists.txt` to include the new subdirectory
5. Link the pass library in `tools/alan-opt/CMakeLists.txt`

### Debugging Pass Registration Issues

If passes don't appear in `alan-opt --help`:
1. Verify the pass is defined in `Passes.td`
2. Check that `Passes.cpp` includes `Passes.h.inc` with `GEN_PASS_REGISTRATION`
3. Ensure `tools/alan-opt/alan-opt.cpp` calls `registerAlanConversionPasses()`
4. Verify CMakeLists.txt links `MLIRAlanConversionPasses` library

## Known Limitations

- RVV path uses LLVM auto-vectorization (-O3), not explicit vector instructions
- Only f32 floating point type is currently supported
- Primary testing is on Rank-1 tensors, though arbitrary Rank is supported
