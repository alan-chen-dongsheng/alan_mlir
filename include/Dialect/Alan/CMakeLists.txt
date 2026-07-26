set(LLVM_TARGET_DEFINITIONS AlanDialect.td)
mlir_tablegen(AlanDialect.h.inc -gen-dialect-decls)
mlir_tablegen(AlanDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(AlanDialectIncGen)

set(LLVM_TARGET_DEFINITIONS AlanOps.td)
mlir_tablegen(AlanOps.h.inc -gen-op-decls)
mlir_tablegen(AlanOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(AlanOpsIncGen)
