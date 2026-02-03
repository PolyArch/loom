//===- SCFToHandshake.h - Loom SCF to Handshake pass ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_CONVERSION_SCFTOHANDSHAKE_H
#define LOOM_CONVERSION_SCFTOHANDSHAKE_H

#include <memory>

namespace mlir {
class Pass;
}

namespace loom {

std::unique_ptr<mlir::Pass> createSCFToHandshakeDataflowPass();

} // namespace loom

#endif // LOOM_CONVERSION_SCFTOHANDSHAKE_H
