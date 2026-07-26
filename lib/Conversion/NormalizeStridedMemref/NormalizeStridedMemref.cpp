#include "Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DEF_NORMALIZESTRIDEDMEMREF
#include "Passes.h.inc"

using namespace mlir;
using namespace mlir::alan;

namespace {

static MemRefType normalizeMemRefType(MemRefType type) {
  if (type.getLayout().isIdentity())
    return type;
  return MemRefType::get(type.getShape(), type.getElementType());
}

static Type normalizeType(Type type) {
  if (auto memrefTy = dyn_cast<MemRefType>(type))
    return normalizeMemRefType(memrefTy);
  return type;
}

static Type normalizeFunctionType(FunctionType type) {
  SmallVector<Type> inputs;
  for (Type t : type.getInputs())
    inputs.push_back(normalizeType(t));
  SmallVector<Type> results;
  for (Type t : type.getResults())
    results.push_back(normalizeType(t));
  return FunctionType::get(type.getContext(), inputs, results);
}

struct NormalizeStridedMemrefPass
    : public ::impl::NormalizeStridedMemrefBase<NormalizeStridedMemrefPass> {
  using Base::Base;

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    module.walk([&](func::FuncOp funcOp) {
      auto funcTy = funcOp.getFunctionType();
      auto newFuncTy = normalizeFunctionType(funcTy);
      if (newFuncTy == funcTy)
        return;

      funcOp.setType(newFuncTy);

      for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
        BlockArgument arg = funcOp.getArgument(i);
        Type newTy = normalizeType(arg.getType());
        if (newTy != arg.getType())
          arg.setType(newTy);
      }

      // Walk all ops in the function body and fix up memref types.
      funcOp.walk([&](Operation *op) {
        for (unsigned i = 0; i < op->getNumResults(); ++i) {
          Value result = op->getResult(i);
          Type newTy = normalizeType(result.getType());
          if (newTy != result.getType())
            result.setType(newTy);
        }
      });
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::alan::createNormalizeStridedMemrefPass() {
  return std::make_unique<NormalizeStridedMemrefPass>();
}
