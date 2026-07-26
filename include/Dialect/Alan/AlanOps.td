#ifndef ALAN_DIALECT_ALANOPS_
#define ALAN_DIALECT_ALANOPS_

include "AlanDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class Alan_Op<string mnemonic, list<Trait> traits = []> :
    Op<AlanDialect, mnemonic, traits>;

def EltwiseOp : Alan_Op<"eltwise", [
    InferTypeOpInterface,
    SameOperandsAndResultType]> {
  let summary = "Element-wise operation on tensors";
  let description = [{
    Performs an element-wise binary operation on two tensors of the same shape
    and element type.
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    StrAttr:$kind
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];

  let hasVerifier = 1;
  let hasFolder = 0;
}

def ReluOp : Alan_Op<"relu", [
    InferTypeOpInterface,
    SameOperandsAndResultType]> {
  let summary = "ReLU activation function";
  let description = [{
    Applies the Rectified Linear Unit function element-wise:
    output = max(input, 0)
  }];

  let arguments = (ins AnyRankedTensor:$input);

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];

  let hasVerifier = 1;
  let hasFolder = 0;
}

#endif // ALAN_DIALECT_ALANOPS_
