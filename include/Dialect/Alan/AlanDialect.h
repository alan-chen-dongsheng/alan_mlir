#ifndef ALAN_DIALECT_ALAN_DIALECT_H_
#define ALAN_DIALECT_ALAN_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

//===----------------------------------------------------------------------===//
// Alan Dialect
//===----------------------------------------------------------------------===//
#include "Dialect/Alan/AlanDialect.h.inc"

//===----------------------------------------------------------------------===//
// Alan Operations
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "Dialect/Alan/AlanOps.h.inc"

namespace mlir {
namespace alan {

// Alan to Linalg pass
std::unique_ptr<Pass> createConvertAlanToLinalgPass();
void registerAlanToLinalgPass();

// CPU Lowering Pipeline
std::unique_ptr<Pass> createAlanCPULoweringPipelinePass();
void registerAlanCPUPasses();

} // namespace alan
} // namespace mlir

#endif // ALAN_DIALECT_ALAN_DIALECT_H_
