#include "Dialect/Alan/AlanDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::alan;

//===----------------------------------------------------------------------===//
// Alan Dialect
//===----------------------------------------------------------------------===//

void AlanDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Alan/AlanOps.cpp.inc"
      >();
}

Attribute AlanDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  // No custom attributes yet
  parser.emitError(parser.getNameLoc(), "unknown alan attribute");
  return Attribute();
}

void AlanDialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const {
  // No custom attributes yet
}

Type AlanDialect::parseType(DialectAsmParser &parser) const {
  // No custom types yet
  parser.emitError(parser.getNameLoc(), "unknown alan type");
  return Type();
}

void AlanDialect::printType(Type type, DialectAsmPrinter &printer) const {
  // No custom types yet
}

//===----------------------------------------------------------------------===//
// EltwiseOp
//===----------------------------------------------------------------------===//

LogicalResult EltwiseOp::verify() {
  auto lhsType = mlir::cast<RankedTensorType>(getLhs().getType());
  auto rhsType = mlir::cast<RankedTensorType>(getRhs().getType());
  auto resultType = mlir::cast<RankedTensorType>(getResult().getType());

  // Check ranks match
  if (lhsType.getRank() != rhsType.getRank()) {
    return emitError("input tensor ranks must match, got ")
           << lhsType.getRank() << " and " << rhsType.getRank();
  }

  // Check shapes match
  if (lhsType.getShape() != rhsType.getShape()) {
    return emitError("input tensor shapes must match, got ")
           << lhsType.getShape() << " and " << rhsType.getShape();
  }

  // Check result shape matches input
  if (lhsType.getShape() != resultType.getShape()) {
    return emitError("result tensor shape must match input shape, got ")
           << resultType.getShape() << " vs input " << lhsType.getShape();
  }

  // Check element types match
  if (lhsType.getElementType() != rhsType.getElementType()) {
    return emitError("input tensor element types must match, got ")
           << lhsType.getElementType() << " and " << rhsType.getElementType();
  }

  // Check result element type matches
  if (lhsType.getElementType() != resultType.getElementType()) {
    return emitError("result element type must match input, got ")
           << resultType.getElementType() << " vs input "
           << lhsType.getElementType();
  }

  // Verify operation kind is valid
  StringRef kind = getKind();
  if (kind != "add" && kind != "sub" && kind != "mul" &&
      kind != "max" && kind != "min") {
    return emitError("unsupported operation kind: ") << kind;
  }

  return success();
}

#define GET_OP_CLASSES
#include "Dialect/Alan/AlanOps.cpp.inc"

#include "Dialect/Alan/AlanDialect.cpp.inc"
