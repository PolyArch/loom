//===- HandshakeToDot.h - Handshake MLIR to DOT Export ----------*- C++ -*-===//
//
// Exports handshake MLIR modules to Graphviz DOT format for visualization.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_VISUALIZATION_HANDSHAKETODOT_H
#define LOOM_VISUALIZATION_HANDSHAKETODOT_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

namespace loom {
namespace visualization {

/// Write DOT representation for all handshake.func ops in module to \p os.
/// Uses subgraphs if multiple functions are present.
void exportModuleToDot(mlir::ModuleOp module, llvm::raw_ostream &os);

} // namespace visualization
} // namespace loom

#endif // LOOM_VISUALIZATION_HANDSHAKETODOT_H
