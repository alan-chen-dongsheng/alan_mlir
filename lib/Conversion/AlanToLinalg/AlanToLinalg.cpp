#include "Conversion/Passes.h"
#include "Dialect/Alan/AlanDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"

// Include the generated base class for this pass
#define GEN_PASS_DEF_CONVERTALANTOLINALG
#include "Passes.h.inc"

#define DEBUG_TYPE "alan-to-linalg"

using namespace mlir;
using namespace mlir::alan;

namespace {

/// Helper to create a binary elementwise op inside a linalg.generic body.
/// Dispatches to the correct arith op based on element type and kind string.
static Value buildEltwiseOp(ConversionPatternRewriter &rewriter,
                            Location loc, StringRef kind,
                            Type eltType, Value a, Value b) {
  if (kind == "add") {
    if (mlir::isa<FloatType>(eltType))
      return arith::AddFOp::create(rewriter, loc, a, b);
    return arith::AddIOp::create(rewriter, loc, a, b);
  }
  if (kind == "sub") {
    if (mlir::isa<FloatType>(eltType))
      return arith::SubFOp::create(rewriter, loc, a, b);
    return arith::SubIOp::create(rewriter, loc, a, b);
  }
  if (kind == "mul") {
    if (mlir::isa<FloatType>(eltType))
      return arith::MulFOp::create(rewriter, loc, a, b);
    return arith::MulIOp::create(rewriter, loc, a, b);
  }
  if (kind == "max") {
    if (mlir::isa<FloatType>(eltType))
      return arith::MaximumFOp::create(rewriter, loc, a, b);
    return arith::MaxSIOp::create(rewriter, loc, a, b);
  }
  if (kind == "min") {
    if (mlir::isa<FloatType>(eltType))
      return arith::MinimumFOp::create(rewriter, loc, a, b);
    return arith::MinSIOp::create(rewriter, loc, a, b);
  }
  return nullptr;
}

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
    auto emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), eltType);

    // Create indexing maps: identity for each input and output
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<AffineMap, 3> indexingMaps = {identityMap, identityMap, identityMap};

    // All iterators are parallel
    SmallVector<utils::IteratorType, 4> iteratorTypes(
        rank, utils::IteratorType::parallel);

    // Create linalg.generic
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc,
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
    StringRef kind = op.getKind();

    Value result = buildEltwiseOp(rewriter, loc, kind, eltType, a, bVal);
    if (!result) {
      rewriter.replaceOp(op, lhs);
      return success();
    }

    linalg::YieldOp::create(rewriter, loc, TypeRange{}, ValueRange{result});
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

/// Conversion pattern for ReluOp to linalg.generic.
struct ReluOpLowering : public OpConversionPattern<ReluOp> {
  using OpConversionPattern<ReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ReluOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = adaptor.getInput();
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto eltType = resultType.getElementType();
    auto rank = resultType.getRank();

    // Create an empty tensor for output
    auto emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), eltType);

    // Create indexing maps: identity for input and output
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<AffineMap, 2> indexingMaps = {identityMap, identityMap};

    // All iterators are parallel
    SmallVector<utils::IteratorType, 4> iteratorTypes(
        rank, utils::IteratorType::parallel);

    // Create linalg.generic
    auto genericOp = linalg::GenericOp::create(
        rewriter, loc,
        TypeRange{resultType},
        ValueRange{input},
        ValueRange{emptyTensor.getResult()},
        indexingMaps,
        iteratorTypes);

    // Create the region body
    auto &region = genericOp.getRegion();
    Block *body = rewriter.createBlock(&region);
    body->addArgument(eltType, loc);  // input
    body->addArgument(eltType, loc);  // output

    rewriter.setInsertionPointToStart(body);
    Value x = body->getArgument(0);

    // ReLU: max(x, 0)
    Value zero;
    Value result;
    if (mlir::isa<FloatType>(eltType)) {
      zero = arith::ConstantOp::create(
          rewriter, loc, rewriter.getFloatAttr(eltType, 0.0));
      result = arith::MaximumFOp::create(rewriter, loc, x, zero);
    } else {
      zero = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(eltType, 0));
      result = arith::MaxSIOp::create(rewriter, loc, x, zero);
    }

    linalg::YieldOp::create(rewriter, loc, TypeRange{}, ValueRange{result});
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
    patterns.add<ReluOpLowering>(context);

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
