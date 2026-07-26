add_mlir_conversion_library(MLIRAlanNormalizeStridedMemref
        NormalizeStridedMemref.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Conversion
        ${PROJECT_BINARY_DIR}/lib/Conversion

        LINK_LIBS PUBLIC
        MLIRArithDialect
        MLIRFuncDialect
        MLIRFuncTransforms
        MLIRMemRefDialect
        MLIRPass
        MLIRSCFDialect
        MLIRTransforms
        )
