#include "Dialect/Alan/AlanDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::alan;

namespace {

struct AlanVectorizationPass
    : public PassWrapper<AlanVectorizationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AlanVectorizationPass)

  StringRef getArgument() const override { return "alan-vectorize"; }
  StringRef getDescription() const override {
    return "Vectorize Alan eltwise operations for RVV";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    PassManager pm(context);

    // Step 1: Convert Alan to Linalg first if not already done
    pm.addPass(createConvertAlanToLinalgPass());

    // Step 2: Vectorize Linalg ops
    pm.addPass(createLinalgVectorizePass());

    // Step 3: Canonicalize vector ops
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (failed(pm.run(module))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::alan::createAlanVectorizationPass() {
  return std::make_unique<AlanVectorizationPass>();
}

void mlir::alan::registerAlanVectorizationPass() {
  PassRegistration<AlanVectorizationPass>();
}
