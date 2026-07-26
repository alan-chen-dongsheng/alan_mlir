#include "Dialect/Alan/AlanDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "alan-cpu-pipeline"

using namespace mlir;
using namespace mlir::alan;

namespace {

struct AlanCPULoweringPipelinePass
    : public PassWrapper<AlanCPULoweringPipelinePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AlanCPULoweringPipelinePass)

  StringRef getArgument() const override { return "alan-cpu-lowering-pipeline"; }
  StringRef getDescription() const override {
    return "Lower Alan dialect through Linalg to LLVM for CPU execution";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<tensor::TensorDialect>();
  }

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

void mlir::alan::registerAlanCPUPasses() {
  PassRegistration<AlanCPULoweringPipelinePass>();
}
