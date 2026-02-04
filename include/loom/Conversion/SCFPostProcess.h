//===-- SCFPostProcess.h - Loom SCF post-processing -------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This header declares passes for post-processing SCF IR, including uplifting
// scf.while loops to scf.for when possible, attaching loop annotations from
// Loom pragma markers to their corresponding loop operations, and annotating
// streamable scf.while loops for dataflow lowering.
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
std::unique_ptr<mlir::Pass> createMarkWhileStreamablePass();

} // namespace loom

#endif // LOOM_CONVERSION_SCFPOSTPROCESS_H
