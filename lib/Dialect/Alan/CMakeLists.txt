add_mlir_dialect_library(MLIRAlanDialect
        AlanDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dialect/Alan
        ${PROJECT_BINARY_DIR}/include/Dialect/Alan

        DEPENDS
        AlanDialectIncGen
        AlanOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRDialect
        MLIRSupport
        )
