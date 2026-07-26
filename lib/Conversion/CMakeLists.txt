set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AlanConversion)
add_public_tablegen_target(MLIRAlanConversionPassIncGen)

add_subdirectory(AlanToLinalg)
add_subdirectory(AlanToLLVM)
