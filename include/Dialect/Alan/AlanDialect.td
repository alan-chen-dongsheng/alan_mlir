#ifndef ALAN_DIALECT_ALAN_
#define ALAN_DIALECT_ALAN_

include "mlir/IR/DialectBase.td"

def AlanDialect : Dialect {
  let name = "alan";
  let cppNamespace = "::mlir::alan";

  let description = [{
    The Alan dialect is a custom MLIR dialect for representing element-wise
    operations on tensors.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 0;
}

#endif // ALAN_DIALECT_ALAN_
