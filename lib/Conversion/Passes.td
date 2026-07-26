#ifndef ALAN_CONVERSION_PASSES
#define ALAN_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertAlanToLinalg : Pass<"convert-alan-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert Alan dialect operations to Linalg dialect";
  let description = [{
    Converts operations from the Alan dialect to operations in the Linalg dialect.
    Specifically, alan.eltwise operations are converted to linalg.generic operations.
  }];
  let constructor = "mlir::alan::createConvertAlanToLinalgPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def AlanCPULoweringPipeline : Pass<"alan-cpu-lowering-pipeline", "mlir::ModuleOp"> {
  let summary = "Lower Alan dialect through Linalg to LLVM for CPU execution";
  let description = [{
    Runs the complete lowering pipeline from Alan dialect through Linalg,
    bufferization, SCF, and finally to LLVM dialect for CPU execution.
  }];
  let constructor = "mlir::alan::createAlanCPULoweringPipelinePass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def AlanVectorization : Pass<"alan-vectorize", "mlir::ModuleOp"> {
  let summary = "Vectorize Alan eltwise operations for RVV";
  let description = [{
    Vectorizes Alan eltwise operations for RISC-V RVV execution.
    Combines Alan to Linalg conversion with Linalg vectorization.
  }];
  let constructor = "mlir::alan::createAlanVectorizationPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect"
  ];
}

def NormalizeStridedMemref : Pass<"normalize-strided-memref", "mlir::ModuleOp"> {
  let summary = "Normalize strided memrefs to static identity layout";
  let description = [{
    Converts strided memref types to static identity-layout memrefs.
    This is useful for EmitC lowering which does not support strided memrefs.
    Note: this pass assumes that all strided memrefs actually have identity
    layout (stride=1, offset=0) at runtime; it is unsound otherwise.
  }];
  let constructor = "mlir::alan::createNormalizeStridedMemrefPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];
}

#endif // ALAN_CONVERSION_PASSES
