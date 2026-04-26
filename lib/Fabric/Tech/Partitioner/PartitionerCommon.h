#ifndef FABRIC_TECH_PARTITIONER_PARTITIONERCOMMON_H
#define FABRIC_TECH_PARTITIONER_PARTITIONERCOMMON_H

// Internal shared helpers for the per-algorithm partitioner implementations.
// This header is library-internal: it lives under lib/, not include/, so it
// is never exposed to downstream targets.
//
// The helpers here implement the structural pieces that every partitioning
// algorithm needs:
//   * canonical-attr stripping for op equality checks,
//   * yield-driven reverse topological visitation,
//   * multi-op candidate collection along the operand[0] backbone,
//   * a compact reachability matrix for inter-block cycle detection,
//   * marginal partition-cost evaluation through the standard CostModel.
//
// Greedy and List share these without modification; algorithms that need
// different cycle / cost semantics may build on top of them.

#include "Common/Config.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/Tech/Partitioner/Partitioner.h"
#include "Fabric/Tech/TemplateLibrary.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <utility>

namespace fabric {

// One pending block under construction during partitioning search. Shared
// across algorithms so the cost / cycle helpers below operate on a single
// uniform representation.
struct PendingBlock {
  ::llvm::SmallVector<::mlir::Operation *> ops;
  const FuTemplate *tpl = nullptr;
};

// Print one attribute to a canonical string suitable for value-equality
// comparison.
std::string canonAttr(::mlir::Attribute a);

// Strip `loom.*` annotations, then return a sorted vector of (key, canonical
// value) pairs. Used to compare two ops' attribute sets while ignoring
// loom-internal metadata.
::llvm::SmallVector<std::pair<std::string, std::string>, 4>
stripLoomAttrs(::mlir::ArrayRef<::mlir::NamedAttribute> attrs);

// Compute a reverse-topological visitation order for the body of `graph`,
// driven by the yield: ops whose results feed the yield come first. Ops not
// reachable from the yield are appended at the end in graph body program
// order, so every op is still visited exactly once.
::llvm::SmallVector<::mlir::Operation *>
reverseTopoOrder(::dataflow::GraphOp graph);

// Match a multi-op template against a candidate rooted at `root`.
//
// The template body must be a linear chain along operand[0]: T[N-1] is the
// root, T[i-1] = T[i].operand[0].defining_op. Returns the candidate's ops
// in body program order on success, or an empty vector on failure.
::llvm::SmallVector<::mlir::Operation *>
collectMultiOpCandidate(::mlir::Operation *root, const FuTemplate &tpl);

// Compact reachability matrix used for cycle detection over the inter-block
// SSA graph.
struct ReachMatrix {
  ::llvm::SmallVector<::llvm::BitVector> rows;

  void ensureSize(unsigned n);

  void rebuild(unsigned n,
               const ::llvm::SmallVector<::llvm::SmallVector<unsigned>> &edges);
};

// Build the per-block direct out-edge list from the current partition state.
// Only blocks whose op set is tracked in `opToBlock` (i.e. blocks with a
// bound template) contribute edges.
::llvm::SmallVector<::llvm::SmallVector<unsigned>> collectBlockEdges(
    const ::llvm::SmallVector<PendingBlock> &blocks,
    const ::llvm::DenseMap<::mlir::Operation *, unsigned> &opToBlock);

// Check whether accepting `candOps` under the current partition would create
// a multi-block SSA cycle.
bool wouldFormMultiBlockCycle(
    const ::llvm::SmallVector<::mlir::Operation *> &candOps,
    const ::llvm::SmallVector<PendingBlock> &blocks,
    const ::llvm::DenseMap<::mlir::Operation *, unsigned> &opToBlock,
    const ReachMatrix &reach);

// Update the reach matrix incrementally after adding the candidate as a new
// block with id `newId`. `outBlocks` are direct successors and `inBlocks`
// are direct predecessors. Caller guarantees the addition does not introduce
// a multi-block cycle.
void addBlockToReach(unsigned newId,
                     const ::llvm::DenseSet<unsigned> &outBlocks,
                     const ::llvm::DenseSet<unsigned> &inBlocks,
                     ReachMatrix &reach);

// Compute the marginal cost of the current pending partition by mirroring
// it into a public PartitionResult and calling the standard CostModel.
double computePendingCost(const ::llvm::SmallVector<PendingBlock> &blocks,
                          const TemplateLibrary &lib,
                          const ::loom::TechMapConfig &cfg);

// Returns true iff the bound blocks of `result` (those with `tpl != nullptr`)
// form a multi-block SSA cycle — i.e. there exists a non-trivial strongly
// connected component of size >= 2 in the block-condensation graph induced
// by inter-block SSA edges. Self-loops on a single block are NOT considered
// a cycle (graph-region semantics permit self-feedback inside a single
// dataflow.subgraph). Used by post-solve repair paths that need to verify
// AC-CORR-3 after an algorithm has produced a candidate partition.
bool partitionHasMultiBlockCycle(const PartitionResult &result);

} // namespace fabric

#endif // FABRIC_TECH_PARTITIONER_PARTITIONERCOMMON_H
