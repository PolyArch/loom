#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_ANCHOR_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_ANCHOR_H

// Anchor strategy: lock-step BFS from yield anchors across all input
// `dataflow.subgraph`s in a synth group. Designed to handle tier A
// inputs (topology-isomorphic; only the op identity at each node
// position varies) plus the restricted tier B case where local
// `fabric.mux` legalizes a cross-share-group node position.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, section
// "Strategy: anchor (tier A by default)".
//
// Threading: the strategy must build its candidate wrapper inside the
// worker-local `MLIRContext` provided via `SynthInputs.context`. The
// pass's main thread re-homes the returned wrapper into the user's
// module context (see `GeneralizeSubgraphsToFuPass`'s splice loop).

#include "Common/SynthConfig.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace loom::fabric::tech {

// Anchor strategy: lock-step BFS from yield anchors. Handles tier-A
// topology-isomorphic input groups; cross-share-group decisions go
// through a CostModel-ranked `decide_op_node` that may insert a local
// `fabric.mux` when `SynthConfig.anchorAllowIntraPositionMux` is true.
class AnchorSynthesizer final : public Synthesizer {
public:
  explicit AnchorSynthesizer(const ::loom::SynthConfig &cfg);
  SynthResult run(const SynthInputs &) override;

private:
  const ::loom::SynthConfig &cfg;
};

// Lifted wrapper-port type lists for one synth group. `inputs` carries
// one `fabric.bits<N>` type per block-argument index of the canonical
// (input #0) subgraph; `outputs` carries one `fabric.bits<N>` type per
// `dataflow.yield` operand of the canonical subgraph. All inputs in a
// tier-A group must agree on per-index block-arg width and yield-arity
// for the lift to succeed.
struct WrapperPorts {
  ::llvm::SmallVector<::mlir::Type, 4> inputs;
  ::llvm::SmallVector<::mlir::Type, 4> outputs;
};

// Compute the wrapper's expected `(inputs, outputs)` signature from one
// synth group's input subgraphs. Returns `std::nullopt` when any input
// subgraph disagrees with its peers on block-arg shape, block-arg
// width, yield arity, or yield operand width, or when any port type is
// not lift-able to `fabric.bits<N>` (block-arg / yield-operand types
// must be `iN`, `fN`, `index`, or `none`). The `MLIRContext *` is the
// uniquer used to construct the lifted `fabric.bits<N>` types; pass the
// caller's owning context (the user's module context for the symbol
// precheck).
::std::optional<WrapperPorts>
collectWrapperPorts(::llvm::ArrayRef<::dataflow::SubgraphOp> sgs,
                    ::mlir::MLIRContext *ctx);

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_ANCHOR_H
