#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_INCREMENTALEXTENSIONS_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_INCREMENTALEXTENSIONS_H

// Internal helper interfaces for `Incremental.cpp`. The candidate
// generators (op-list widening, mux/demux insertion, structural
// extension) live in their own translation units so the main
// `IncrementalSynthesizer::run` loop in `Incremental.cpp` stays
// focused on the left-fold control flow.
//
// Tier-A and tier-B candidate generators live in
// `IncrementalExtensions.cpp`; the tier-C SCC handling (back-edge
// alignment, fabric.op[@dataflow.carry] grafting, trivial FU build
// for tier-C inputs) lives in `IncrementalExtensionsTierC.cpp`.
//
// This header is *not* part of the public Synthesizer API; it is only
// included by `Incremental.cpp` and the IncrementalExtensions* TUs.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, sections
// "Strategy: incremental > extend_to_cover" and "SCC handling for
// tier C".

#include "Common/SynthConfig.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace loom::fabric::tech::detail {

// Op-list widening: for each diff site between `curWrapper` and `sg`
// where the FU's fabric.op and sg's op share a hardware share-group +
// width, return a candidate wrapper whose op_list at that position is
// the sorted union (current ∪ {new op name}).
::llvm::SmallVector<::mlir::OwningOpRef<::fabric::ModuleOp>, 4>
widenOplistCandidates(::fabric::ModuleOp curWrapper,
                      ::dataflow::SubgraphOp sg);

// Mux/demux insert: tier B baseline generator. Targets the spec's
// "extend by one tail op" / "FU has one extra head op" cases. Handles
// both the case where sg is one op longer than the FU's chain at a
// yield position (insert demux between FU's existing producer and the
// new tail op + mux at yield) and the symmetric case where the FU is
// one op longer than sg (insert demux at the FU's chain pre-head + mux
// at yield to skip the extra op).
//
// When `cfg.subgraphShareRecurse == true`, the generator additionally
// emits a recursive-compression candidate that widens an existing FU
// body fabric.op's op_list to absorb the new tail op when the two
// share a hardware share-group + width (spec Q12). When the flag is
// false, only the baseline candidates are emitted.
::llvm::SmallVector<::mlir::OwningOpRef<::fabric::ModuleOp>, 4>
insertMuxDemuxCandidates(::fabric::ModuleOp curWrapper,
                         ::dataflow::SubgraphOp sg,
                         const ::loom::SynthConfig &cfg);

// Tier-C structural extension. Generates one candidate that grafts a
// new sub-FU for the diff region, including
// `fabric.op[@dataflow.carry]` SCC bodies if needed. Uses the flow-
// signature heuristic from `pre_align_sccs` to decide which carry
// heads merge with existing carries in the FU and which are new.
// Returns an empty vector when the heuristic cannot align the SCCs
// (caller should consult `classifyTierCConflict` to distinguish a
// `feedback_align_conflict` from a generic structural mismatch).
::llvm::SmallVector<::mlir::OwningOpRef<::fabric::ModuleOp>, 4>
structuralExtendCandidates(::fabric::ModuleOp curWrapper,
                           ::dataflow::SubgraphOp sg,
                           const ::loom::SynthConfig &cfg);

// True iff `sg`'s body contains a graph-region back-edge (and the
// incremental main loop should therefore consider invoking the
// structural extension hook).
bool hasBackEdgeInDiff(::fabric::ModuleOp curWrapper,
                       ::dataflow::SubgraphOp sg);

// Build the trivial FU for a single tier-C input subgraph: a 1:1 mirror
// of `first`'s body emitted as fabric.ops, with graph-region back-edges
// resolved through a build-then-rewire placeholder scheme. Used by the
// Incremental main loop when the input subgraph contains a back-edge
// (the Anchor strategy refuses such inputs as `topology_mismatch`).
::mlir::OwningOpRef<::fabric::ModuleOp>
buildTrivialFuTierC(::mlir::MLIRContext *ctx, ::llvm::StringRef groupName,
                    ::dataflow::SubgraphOp first);

// Classify why `structuralExtendCandidates` returned empty: returns
// `FeedbackAlignConflict` when the flow-signature heuristic refused to
// align the SCCs (e.g. one input has two carry heads in the same
// equivalence class, or two inputs disagree on the upstream stream
// signature), and `std::nullopt` otherwise (which the main loop
// surfaces as `topology_mismatch`).
::std::optional<SynthFailureReason>
classifyTierCConflict(::fabric::ModuleOp curWrapper,
                      ::dataflow::SubgraphOp sg,
                      const ::loom::SynthConfig &cfg);

} // namespace loom::fabric::tech::detail

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_INCREMENTALEXTENSIONS_H
