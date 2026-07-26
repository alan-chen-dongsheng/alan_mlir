#include "Conversion/Passes.h"
#include "Dialect/Alan/AlanDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

// Include the generated base class for this pass
#define GEN_PASS_DEF_CONVERTALANTOLINALG
#include "Passes.h.inc"

#define DEBUG_TYPE "alan-to-linalg"

using namespace mlir;
using namespace mlir::alan;

namespace {

/// Conversion pattern for EltwiseOp to linalg.generic.
struct EltwiseOpLowering : public OpConversionPattern<EltwiseOp> {
  using OpConversionPattern<EltwiseOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(EltwiseOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto eltType = resultType.getElementType();
    auto rank = resultType.getRank();

    // Create an empty tensor for output
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), eltType);

    // Create indexing maps: identity for each input and output
    SmallVector<AffineMap, 3> indexingMaps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    indexingMaps.push_back(identityMap);  // lhs
    indexingMaps.push_back(identityMap);  // rhs
    indexingMaps.push_back(identityMap);  // output

    // All iterators are parallel
    SmallVector<utils::IteratorType, 4> iteratorTypes(
        rank, utils::IteratorType::parallel);

    // Create linalg.generic
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        TypeRange{resultType},
        ValueRange{lhs, rhs},
        ValueRange{emptyTensor.getResult()},
        indexingMaps,
        iteratorTypes);

    // Create the region body
    auto &region = genericOp.getRegion();
    Block *body = rewriter.createBlock(&region);
    for (auto i = 0; i < 2; ++i) {
      body->addArgument(eltType, loc);
    }
    body->addArgument(eltType, loc);

    rewriter.setInsertionPointToStart(body);
    Value a = body->getArgument(0);
    Value bVal = body->getArgument(1);
    Value result;
    StringRef kind = op.getKind();

    if (kind == "add") {
      if (mlir::isa<FloatType>(eltType)) {
        result = rewriter.create<arith::AddFOp>(loc, a, bVal);
      } else {
        result = rewriter.create<arith::AddIOp>(loc, a, bVal);
      }
    } else if (kind == "sub") {
      if (mlir::isa<FloatType>(eltType)) {
        result = rewriter.create<arith::SubFOp>(loc, a, bVal);
      } else {
        result = rewriter.create<arith::SubIOp>(loc, a, bVal);
      }
    } else if (kind == "mul") {
      if (mlir::isa<FloatType>(eltType)) {
        result = rewriter.create<arith::MulFOp>(loc, a, bVal);
      } else {
        result = rewriter.create<arith::MulIOp>(loc, a, bVal);
      }
    } else if (kind == "max") {
      if (mlir::isa<FloatType>(eltType)) {
        result = rewriter.create<arith::MaximumFOp>(loc, a, bVal);
      } else {
        result = rewriter.create<arith::MaxSIOp>(loc, a, bVal);
      }
    } else if (kind == "min") {
      if (mlir::isa<FloatType>(eltType)) {
        result = rewriter.create<arith::MinimumFOp>(loc, a, bVal);
      } else {
        result = rewriter.create<arith::MinSIOp>(loc, a, bVal);
      }
    } else {
      rewriter.replaceOp(op, lhs);
      return success();
    }

    rewriter.create<linalg::YieldOp>(loc, TypeRange{}, ValueRange{result});
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct ConvertAlanToLinalgPass
    : public ::impl::ConvertAlanToLinalgBase<ConvertAlanToLinalgPass> {
  using Base::Base;

  void runOnOperation() override {
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addIllegalDialect<AlanDialect>();

    patterns.add<EltwiseOpLowering>(context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::alan::createConvertAlanToLinalgPass() {
  return std::make_unique<ConvertAlanToLinalgPass>();
}
