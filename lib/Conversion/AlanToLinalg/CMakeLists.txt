add_mlir_conversion_library(MLIRAlanToLinalg
        AlanToLinalg.cpp

        LINK_LIBS PUBLIC
        MLIRAlanDialect
        MLIRArithDialect
        MLIRLinalgDialect
        MLIRPass
        MLIRTensorDialect
        MLIRTransforms
        )
