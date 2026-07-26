add_mlir_conversion_library(MLIRAlanToLinalg
        AlanToLinalg.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Conversion
        ${PROJECT_BINARY_DIR}/lib/Conversion

        LINK_LIBS PUBLIC
        MLIRAlanDialect
        MLIRArithDialect
        MLIRLinalgDialect
        MLIRPass
        MLIRTensorDialect
        MLIRTransforms
        )
