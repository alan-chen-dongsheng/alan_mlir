#include "Conversion/Passes.h"

// Include the generated pass registration functions
// This defines registerAlanConversionPasses() in the global namespace
#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

// Provide the mlir::alan:: namespace wrapper
namespace mlir {
namespace alan {

void registerAlanConversionPasses() {
  // Call the generated registration function (in global namespace)
  ::registerAlanConversionPasses();
}

} // namespace alan
} // namespace mlir
