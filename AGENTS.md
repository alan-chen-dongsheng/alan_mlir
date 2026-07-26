# Repository Guidelines

## Project Structure & Module Organization

This MLIR-based compiler project follows a standard LLVM/MLIR directory layout:

```
alan_mlir/
├── include/Dialect/Alan/          # Dialect TableGen definitions (.td) and headers
├── lib/Dialect/Alan/              # Dialect C++ implementation
├── lib/Conversion/                # Conversion passes (Alan→Linalg, CPU pipeline)
│   ├── AlanToLinalg/              # Alan dialect to Linalg lowering
│   └── AlanToLLVM/                # CPU lowering pipeline
├── tools/alan-opt/                # Alan MLIR optimizer driver
├── tools/*.sh                      # Execution scripts (CPU, RVV Spike)
├── runtime/                        # C runtime wrappers for execution
├── test/                           # MLIR test files
│   ├── Dialect/Alan/              # Parser/verifier tests
│   ├── Conversion/AlanToLinalg/   # Lowering tests
│   └── Execution/Alan/            # End-to-end tests
└── docs/                           # Documentation
```

## Build, Test, and Development Commands

| Command | Description |
|---------|-------------|
| `cd build && cmake ..` | Configure CMake with LLVM/MLIR |
| `make -j$(nproc)` | Build all targets in `build/` |
| `build/tools/alan-opt/alan-opt file.mlir` | Parse and verify Alan MLIR |
| `build/tools/alan-opt/alan-opt --convert-alan-to-linalg file.mlir` | Run Alan→Linalg conversion |
| `./tools/run_alan_cpu.sh test.mlir` | End-to-end CPU execution |
| `./tools/run_alan_rvv_spike.sh test.mlir` | End-to-end RVV Spike execution |

## Coding Style & Naming Conventions

- **Indentation**: 2 spaces for CMake, 4 spaces for C++
- **C++**: Follow LLVM/MLIR coding standards (camelCase for functions, PascalCase for classes)
  - Dialect classes: `AlanDialect`
  - Pass names: `ConvertAlanToLinalgPass`
  - File names: `AlanDialect.cpp`, `AlanToLinalg.td`
- **MLIR/TableGen**: Use `.td` extension for TableGen files
- **CMake**: Targets prefixed with `MLIR` (e.g., `MLIRAlanDialect`)

## Testing Guidelines

- **Test files**: Place `.mlir` tests in `test/` with appropriate subdirectories
- **End-to-end tests**: Use `eltwise` prefix (e.g., `eltwise_add.mlir`)
- Run CPU and RVV scripts after modifying lowering passes to verify correctness
- All operations (add, sub, mul, max, min) must have corresponding test coverage

## Commit & Pull Request Guidelines

- **Commit messages**: Format as `area: description`
  - Example: `feat(alan): add eltwise verifier for shape checking`
  - Example: `fix(cpu): resolve memref descriptor alignment issue`
- Keep commits atomic: one logical change per commit
- Include test coverage with feature additions
- Reference issues in description when applicable
- Run both CPU and RVV end-to-end tests before submission

## Architecture Notes

- **Dialect first**: Always extend the Alan dialect before adding new conversions
- **Pipeline separation**: Keep CPU and RVV pipeline logic in distinct modules
- **TableGen preferred**: Define dialects, ops, and passes via TableGen when possible
