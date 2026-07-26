add_mlir_conversion_library(MLIRAlanToLLVM
        AlanCPUPipeline.cpp

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
