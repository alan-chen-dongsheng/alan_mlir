set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AlanConversion)
add_public_tablegen_target(MLIRAlanConversionPassIncGen)

# Add include directory for generated headers
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Create a library for pass registration
add_mlir_library(MLIRAlanConversionPasses
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Conversion
  ${CMAKE_CURRENT_BINARY_DIR}

  LINK_LIBS PUBLIC
  MLIRAlanToLinalg
  MLIRAlanToLLVM
  MLIRAlanToVector
  MLIRPass
)

add_subdirectory(AlanToLinalg)
add_subdirectory(AlanToLLVM)
add_subdirectory(AlanToVector)
