#ifndef LOOM_CONVERSION_LLVMTOSCF_H
#define LOOM_CONVERSION_LLVMTOSCF_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace loom {

std::unique_ptr<mlir::Pass> createLowerLLVMToSCFPass();

} // namespace loom

#endif // LOOM_CONVERSION_LLVMTOSCF_H
