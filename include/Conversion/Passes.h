#ifndef ALAN_CONVERSION_PASSES_H
#define ALAN_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace alan {

// Forward declarations of pass creation functions
std::unique_ptr<Pass> createConvertAlanToLinalgPass();
std::unique_ptr<Pass> createAlanCPULoweringPipelinePass();
std::unique_ptr<Pass> createAlanVectorizationPass();
std::unique_ptr<Pass> createNormalizeStridedMemrefPass();

// Register all Alan conversion passes
void registerAlanConversionPasses();

} // namespace alan
} // namespace mlir

// NOTE: Do NOT include Passes.h.inc here to avoid multiple definitions.
// Each pass implementation file should include it with the appropriate GEN_PASS_DEF_* macro.

#endif // ALAN_CONVERSION_PASSES_H
