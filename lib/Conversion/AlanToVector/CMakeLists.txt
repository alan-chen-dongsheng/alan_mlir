add_mlir_conversion_library(MLIRAlanToVector
        AlanVectorization.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/lib/Conversion

        LINK_LIBS PUBLIC
        MLIRAlanDialect
        MLIRAlanToLinalg
        MLIRArithDialect
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRSupport
        MLIRTensorDialect
        MLIRTransforms
        MLIRVectorDialect
        )
