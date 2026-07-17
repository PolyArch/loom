#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_ANCHOR_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_ANCHOR_H

// Anchor aligns canonical ConfiguredFunctions from their ordered outputs.
// Topology-compatible node positions may share one physical operation or use
// explicit demux/mux routing across distinct hardware share groups.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, heading
// "Anchor Synthesis".
//
// Threading: the Anchor producer builds its candidate wrapper inside the
// worker-local `MLIRContext` provided via `SynthInputs.context`. The
// pass's main thread re-homes the returned wrapper into the user's
// module context (see the configured-function synthesis pass splice loop).

#include "Fabric/IR/ConfiguredFunction.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace loom::fabric::tech {

// Physical wrapper-port type lists for one canonical function group.
struct WrapperPorts {
  ::llvm::SmallVector<::mlir::Type, 4> inputs;
  ::llvm::SmallVector<::mlir::Type, 4> outputs;
};

// Compute the physical wrapper signature. Returns `std::nullopt` when boundary
// shapes differ or a software type cannot be represented by `fabric.bits<N>`.
::std::optional<WrapperPorts>
collectWrapperPorts(::llvm::ArrayRef<::fabric::ConfiguredFunction> functions,
                    ::mlir::MLIRContext *ctx);

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_ANCHOR_H
