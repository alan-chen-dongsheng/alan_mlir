#include "Conversion/Passes.h"
#include "Dialect/Alan/AlanDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// Include the generated base class for this pass
#define GEN_PASS_DEF_ALANCPULOWERINGPIPELINE
#include "Passes.h.inc"

#define DEBUG_TYPE "alan-cpu-pipeline"

using namespace mlir;
using namespace mlir::alan;

namespace {

struct AlanCPULoweringPipelinePass
    : public ::impl::AlanCPULoweringPipelineBase<AlanCPULoweringPipelinePass> {
  using Base::Base;

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    PassManager pm(context);

    // Step 1: Convert Alan to Linalg
    pm.addPass(createConvertAlanToLinalgPass());

    // Step 2: Convert Linalg to parallel loops
    pm.addPass(createConvertLinalgToParallelLoopsPass());

    // Step 3: Canonicalize and CSE
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (failed(pm.run(module))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::alan::createAlanCPULoweringPipelinePass() {
  return std::make_unique<AlanCPULoweringPipelinePass>();
}
