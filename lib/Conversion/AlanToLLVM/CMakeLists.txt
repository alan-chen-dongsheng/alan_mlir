add_mlir_conversion_library(MLIRAlanToLLVM
        AlanCPUPipeline.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Conversion
        ${PROJECT_BINARY_DIR}/lib/Conversion

        LINK_LIBS PUBLIC
        MLIRAlanDialect
        MLIRAlanToLinalg
        MLIRArithDialect
        MLIRControlFlowDialect
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRLLVMDialect
        MLIRMemRefDialect
        MLIRPass
        MLIRSCFDialect
        MLIRSupport
        MLIRTensorDialect
        MLIRTransforms
        )
