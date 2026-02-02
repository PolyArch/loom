//===- SCFPostProcess.h - Loom SCF post-processing -----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_CONVERSION_SCFPOSTPROCESS_H
#define LOOM_CONVERSION_SCFPOSTPROCESS_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace loom {

std::unique_ptr<mlir::Pass> createUpliftWhileToForPass();
std::unique_ptr<mlir::Pass> createAttachLoopAnnotationsPass();

} // namespace loom

#endif // LOOM_CONVERSION_SCFPOSTPROCESS_H
