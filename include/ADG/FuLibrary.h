#ifndef LOOM_ADG_FULIBRARY_H
#define LOOM_ADG_FULIBRARY_H

#include "ADG/Builder.h"
#include "ADG/SpecialMathCapabilityProfile.h"
#include "Dataflow/IR/DataflowEnums.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::adg {

/// Adds the catalog's scalar ALU FU to one PE. Inputs are data0, data1, and
/// condition, in that order. The helper constructs only ordinary Fabric
/// resources and closes the FU before returning.
llvm::Error addCoreAluFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                         ::fabric::ResolvedIndexWidthSet resolvedIndexWidths);

/// Adds scalar multiply, fused and non-fused multiply-add, and local carry
/// recurrence graphs. Inputs are data0, data1, data2, and phase.
llvm::Error addMacFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs);

/// Adds two fixed-step stream resources plus carry, invariant, and gate
/// resources. Inputs are data0, data1, data2, and phase. The two stream step
/// kinds identify distinct physical resources and must differ.
llvm::Error addLoopControlFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                             ::dataflow::StreamStepKind firstStep,
                             ::dataflow::StreamStepKind secondStep);

/// Transient typed widths for one vector-compute FU expansion. The emitted
/// physical ports and capability records remain the only persistent owner.
struct VectorComputeFuParameters final {
  std::uint32_t outerPayloadBits;
  std::uint32_t vectorPayloadBits;
};

/// Adds the fixed-vector compute FU. Inputs are data0, data1, data2, and
/// vector condition, in that order.
llvm::Error addVectorComputeFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                               VectorComputeFuParameters parameters);

/// Transient typed inputs for one vector-structure FU expansion. The emitted
/// Fabric ports and capability records are the only persistent authority.
struct VectorStructuralFuParameters final {
  std::uint32_t outerPayloadBits;
  std::uint32_t vectorPayloadBits;
  std::uint32_t indexPayloadBits;
  ::fabric::FixedVectorSliceAlignMergeParams sliceCapability;
  ::fabric::FixedVectorShuffleParams shuffleCapability;
};

/// Adds fixed-vector leading-slice alignment/merge and shuffle resources.
/// Inputs are two vector/value roles followed by the slice capability's
/// maximum number of dynamic-position roles.
llvm::Error
addVectorStructuralFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                      const VectorStructuralFuParameters &parameters);

/// Adds fixed-vector representation and stream-group adapters. Inputs are
/// data/vector, mask, and phase. Results are data/vector, mask, and phase.
llvm::Error addVectorAdapterFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs);

/// Adds constant, rendezvous, and runtime token-routing resources. Inputs are
/// selector/control followed by four payload lanes. Four payload lanes are
/// exposed as results.
struct TokenControlFuParameters final {
  std::uint32_t outerPayloadBits;
  std::uint32_t selectorPayloadBits;
};

llvm::Error addTokenControlFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                              TokenControlFuParameters parameters);

/// Adds the low-density scalar divide, remainder, and elementary math FU.
/// Inputs are data0 and data1, in that order.
llvm::Error addSpecialMathFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                             BuiltinSpecialMathCapabilityProfile profile);

/// One `fabric.op` resource of a composite FU: the implementation family and
/// typed capability envelope its operation set needs, the operations it
/// enables, and the Fabric types of its ports.
struct CompositeFuNodeSpec final {
  ::fabric::ImplementationFamilyId implementationFamily;
  ::fabric::FamilyCapabilityParams hardwareParameters;
  std::vector<::dataflow::OperationSchemaId> enabledOperations;
  std::vector<PortType> inputTypes;
  std::vector<PortType> outputTypes;
};

/// One edge between two operation resources of a composite FU.
struct CompositeFuEdgeSpec final {
  std::uint32_t producerNode = 0;
  std::uint64_t producerResult = 0;
  std::uint32_t consumerNode = 0;
  std::uint64_t consumerOperand = 0;
};

/// One FU boundary port of a composite FU, named by the node port it carries.
/// The port's Fabric type is that node port's type, so the boundary has no
/// second type declaration.
struct CompositeFuPortSpec final {
  std::uint32_t node = 0;
  std::uint64_t portOrdinal = 0;
};

/// One composite FU: a connected graph of operation resources with an explicit
/// ordered boundary. This is the hardware request. Whatever relation produced
/// it, a mined common software subgraph or a catalog decision, stays with its
/// own owner; this structure is the only description the Builder authors from.
struct CompositeFuSpec final {
  std::vector<CompositeFuNodeSpec> nodes;
  std::vector<CompositeFuEdgeSpec> internalEdges;
  std::vector<CompositeFuPortSpec> inputs;
  std::vector<CompositeFuPortSpec> outputs;
};

/// The authoring handles of one placed composite FU, in spec node order.
/// Canonical finalization relabels FU graph nodes and capability rows, so a
/// caller that must bind exact software actors resolves these handles through
/// FinalizedFabricDesign instead of assuming the authoring order survived.
struct CompositeFuPlacement final {
  std::vector<FuNode> nodes;
  FuCapabilityTemplateHandle capability;
};

/// Adds one composite FU to an open PE and closes the FU. `inputs` are the PE
/// values that carry the FU's ordered boundary inputs, one per
/// `spec.inputs` entry. Every operand of every node must be either an internal
/// edge destination or a boundary input, and a spec whose internal relation
/// contains a cycle is rejected: a recurrence needs an explicit FU backedge
/// and is not authored implicitly.
llvm::Expected<CompositeFuPlacement>
addCompositeFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
               const CompositeFuSpec &spec);

} // namespace loom::adg

#endif // LOOM_ADG_FULIBRARY_H
