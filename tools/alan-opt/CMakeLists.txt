get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRAlanDialect
        MLIRAlanToLinalg
        MLIROptLib
        )

add_llvm_executable(alan-opt alan-opt.cpp)
llvm_update_compile_flags(alan-opt)
target_link_libraries(alan-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(alan-opt)
