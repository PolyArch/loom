#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_INCREMENTALEXTENSIONS_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_INCREMENTALEXTENSIONS_H

// Internal helper interfaces for `Incremental.cpp`. The candidate
// generators (op-list widening, mux/demux insertion, structural
// extension hook) live in their own translation unit so the main
// `IncrementalSynthesizer::run` loop in `Incremental.cpp` stays
// focused on the left-fold control flow.
//
// This header is *not* part of the public Synthesizer API; it is only
// included by `Incremental.cpp` and `IncrementalExtensions.cpp`.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, section
// "Strategy: incremental > extend_to_cover".

#include "Common/SynthConfig.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/SmallVector.h"

namespace loom::fabric::tech::detail {

// Op-list widening: for each diff site between `curWrapper` and `sg`
// where the FU's fabric.op and sg's op share a hardware share-group +
// width, return a candidate wrapper whose op_list at that position is
// the sorted union (current ∪ {new op name}).
::llvm::SmallVector<::mlir::OwningOpRef<::mlir::func::FuncOp>, 4>
widenOplistCandidates(::mlir::func::FuncOp curWrapper,
                      ::dataflow::SubgraphOp sg);

// Mux/demux insert: tier B baseline generator. Targets the spec's
// "extend by one tail op" / "FU has one extra head op" cases. Handles
// both the case where sg is one op longer than the FU's chain at a
// yield position (insert demux between FU's existing producer and the
// new tail op + mux at yield) and the symmetric case where the FU is
// one op longer than sg (insert demux at the FU's chain pre-head + mux
// at yield to skip the extra op).
::llvm::SmallVector<::mlir::OwningOpRef<::mlir::func::FuncOp>, 4>
insertMuxDemuxCandidates(::mlir::func::FuncOp curWrapper,
                         ::dataflow::SubgraphOp sg);

// Tier C structural extension hook. The current iteration returns an
// empty candidate set; the back-edge / SCC implementation lands in the
// follow-up tier-C task. Defined here so the main loop can invoke it
// uniformly with the other generators.
::llvm::SmallVector<::mlir::OwningOpRef<::mlir::func::FuncOp>, 4>
structuralExtendCandidates(::mlir::func::FuncOp curWrapper,
                           ::dataflow::SubgraphOp sg,
                           const ::loom::SynthConfig &cfg);

// True iff `sg`'s body contains a graph-region back-edge (and the
// incremental main loop should therefore consider invoking the
// structural extension hook).
bool hasBackEdgeInDiff(::mlir::func::FuncOp curWrapper,
                       ::dataflow::SubgraphOp sg);

} // namespace loom::fabric::tech::detail

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_INCREMENTALEXTENSIONS_H
