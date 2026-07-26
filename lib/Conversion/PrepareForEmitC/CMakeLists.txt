add_mlir_conversion_library(MLIRAlanPrepareForEmitC
        PrepareForEmitC.cpp

        LINK_LIBS PUBLIC
        MLIREmitCDialect
        MLIRFuncDialect
        MLIRMemRefDialect
        MLIRPass
        MLIRTransforms
        )
