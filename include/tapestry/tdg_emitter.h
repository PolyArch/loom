#ifndef TAPESTRY_TDG_EMITTER_H
#define TAPESTRY_TDG_EMITTER_H

#include <memory>
#include <string>

namespace mlir {
class MLIRContext;
class ModuleOp;
template <typename T> class OwningOpRef;
} // namespace mlir

namespace tapestry {

class TaskGraph;

/// Convert a TaskGraph (user-side C++ objects) into a TDG MLIR module.
///
/// The emitter:
///   1. Registers the TDG dialect on the provided context.
///   2. Creates a `tdg.graph` op wrapping the entire graph.
///   3. For each kernel: emits a `tdg.kernel` op (body left empty for
///      kernel_compiler from C08 to fill later).
///   4. For each edge: fills contract defaults (ordering=FIFO,
///      data_type="f32" when unset) and emits a `tdg.contract` op.
///
/// \param graph  The user-constructed TaskGraph.
/// \param ctx    An MLIRContext (TDG dialect will be loaded if needed).
/// \returns An OwningOpRef<ModuleOp> containing the TDG MLIR.
mlir::OwningOpRef<mlir::ModuleOp> emitTDG(const TaskGraph &graph,
                                           mlir::MLIRContext &ctx);

/// Serialize a TDG MLIR module to a text file.
///
/// \param module  The MLIR module to write.
/// \param path    Output file path.
/// \returns true on success.
bool writeTDGToFile(mlir::ModuleOp module, const std::string &path);

} // namespace tapestry

#endif // TAPESTRY_TDG_EMITTER_H
