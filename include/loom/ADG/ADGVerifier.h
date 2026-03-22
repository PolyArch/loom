#ifndef LOOM_ADG_ADGVERIFIER_H
#define LOOM_ADG_ADGVERIFIER_H

#include "mlir/IR/BuiltinOps.h"

namespace loom {

/// Verify a top-level MLIR module containing a fabric.module.
///
/// This performs two layers of checking:
/// - regular MLIR verification for Fabric op-local invariants
/// - fabric.module graph checks shared by builder-emitted ADGs and
///   hand-authored ADGs consumed by the mapper
///
/// The graph-level checks currently include:
/// - no dangling top-level hardware outputs
/// - no dangling fabric.module input ports
/// - no dangling inline hardware input ports
/// - hardware graph connections must preserve tag-kind compatibility
mlir::LogicalResult verifyFabricModule(mlir::ModuleOp topModule);

} // namespace loom

#endif // LOOM_ADG_ADGVERIFIER_H
