#ifndef LOOM_FRONTEND_LOWERING_GRAPH_REGION_LOWERING_H
#define LOOM_FRONTEND_LOWERING_GRAPH_REGION_LOWERING_H

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace loom {
namespace lowering {

// What graph-region lowering can do with one leaf operation. The cases name
// the capabilities `GraphRegionLowerer::lowerOperations` actually implements,
// so a leaf is supported because some capability covers it, never because it
// escaped an exclusion list. This is the one authority that preflight, the
// lowering fallback, and the parallel completion check all consult.
enum class GraphLeafLowering {
  // Regionless and pure, and eligible under the graph contract as a registered
  // canonical Dataflow actor or as one of the pure address leaves the memory
  // root resolution walks. Hoisting it into the graph frontier is a sound
  // action, so this also covers a leaf whose in-place rewrite still leaves the
  // move to the frontier fallback.
  Movable,
  // An effectful leaf whose action is dedicated to the position it already
  // occupies, so it never reaches the frontier fallback: the memref and
  // dataflow accesses and the stream endpoints `lowerOperations` rewrites or
  // consumes in place, the LLVM memory operations graph normalization turns
  // into dataflow memory actors before region lowering runs, with
  // `checkResidualMemoryEffects` failing closed on any it could not convert,
  // and the fresh allocation root standing in the graph frontier, whose action
  // is to preserve it there.
  Implemented,
  // No implemented action. An effectful canonical actor with neither a rewrite
  // nor a proof of movability lands here, so lowering never has to relocate an
  // operation whose semantics it cannot reproduce.
  Unsupported,
};

// Whether lowering covers `op` and how, judged where `op` stands: a leaf whose
// only implemented action is to stay put is classified by its position.
GraphLeafLowering classifyGraphLoweringLeaf(::mlir::Operation *op);

::mlir::LogicalResult
checkGraphRegionLoweringPreconditions(::mlir::ModuleOp module);

// `indexBits` is the canonical index width the caller's pass boundary already
// resolved; region lowering never resolves it again.
::mlir::LogicalResult lowerGraphRegions(::dataflow::GraphOp graph,
                                        unsigned indexBits);

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_GRAPH_REGION_LOWERING_H
