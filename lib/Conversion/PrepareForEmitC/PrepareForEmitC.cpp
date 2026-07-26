#include "Conversion/Passes.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

#define GEN_PASS_DEF_PREPAREFOREMITC
#include "Passes.h.inc"

using namespace mlir;

namespace {

// Pattern 1: Remove unrealized_conversion_cast operations
struct RemoveUnrealizedCasts : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    // Replace all uses of the cast results with the source values
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      op.getResult(i).replaceAllUsesWith(op.getOperand(i));
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct PrepareForEmitCPass
    : public ::impl::PrepareForEmitCBase<PrepareForEmitCPass> {
  using Base::Base;

  void runOnOperation() override {
    auto *context = &getContext();
    RewritePatternSet patterns(context);

    // Add all patterns
    patterns.add<RemoveUnrealizedCasts>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::alan::createPrepareForEmitCPass() {
  return std::make_unique<PrepareForEmitCPass>();
}
