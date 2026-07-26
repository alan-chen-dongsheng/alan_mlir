#include "Conversion/Passes.h"
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

// Include the generated base class for this pass
#define GEN_PASS_DEF_ALANVECTORIZATION
#include "Passes.h.inc"

using namespace mlir;
using namespace mlir::alan;

namespace {

struct AlanVectorizationPass
    : public ::impl::AlanVectorizationBase<AlanVectorizationPass> {
  using Base::Base;

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    PassManager pm(context);

    // Step 1: Convert Alan to Linalg first if not already done
    pm.addPass(createConvertAlanToLinalgPass());

    // Step 2: Convert Linalg to parallel loops
    // Note: Vectorization is handled by LLVM backend with -O3 and RVV attributes
    // The RVV execution script uses LLC with +v attribute for auto-vectorization
    pm.addPass(createConvertLinalgToParallelLoopsPass());

    // Step 3: Canonicalize
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
