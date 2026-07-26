#ifndef ALAN_DIALECT_ALAN_DIALECT_H_
#define ALAN_DIALECT_ALAN_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Alan Dialect
//===----------------------------------------------------------------------===//
#include "Dialect/Alan/AlanDialect.h.inc"

//===----------------------------------------------------------------------===//
// Alan Operations
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "Dialect/Alan/AlanOps.h.inc"

#endif // ALAN_DIALECT_ALAN_DIALECT_H_
