//===-- SCFToHandshake.h - SCF to Handshake conversion ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This header declares the SCF-to-Handshake dataflow conversion pass. The pass
// transforms accelerator-marked functions from SCF dialect to CIRCT's Handshake
// dialect, enabling dataflow-style hardware generation.
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
