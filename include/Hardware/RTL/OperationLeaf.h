#ifndef LOOM_HARDWARE_RTL_OPERATIONLEAF_H
#define LOOM_HARDWARE_RTL_OPERATIONLEAF_H

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Hardware/Configuration/ConfigurationABI.h"

#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <vector>

namespace mlir {
class OpBuilder;
}

namespace loom::hardware::rtl {

/// The structural protocol exposed by one abstract operation leaf. Ordinary
/// stateless families remain combinational behind the common elastic slot.
/// One-cycle control/stream families use that same common slot. Managed token
/// families retain their exact resource timing inside the provider transform,
/// while transparent loops commit directly without a hidden result register.
enum class FabricOperationLeafProtocol {
  Combinational,
  ElasticToken,
  OrderedCardinalityToken,
  ManagedToken,
  TransparentToken,
};

/// The transient structural interface derived from one exact sealed Fabric
/// capability. It has no persistent identity and does not duplicate family
/// semantics or resource timing.
struct FabricOperationLeafInterface final {
  FabricOperationLeafProtocol protocol =
      FabricOperationLeafProtocol::Combinational;

  bool hasTokenHandshake() const {
    return protocol != FabricOperationLeafProtocol::Combinational;
  }

  bool hasElasticResultStorage() const {
    return protocol == FabricOperationLeafProtocol::ElasticToken ||
           protocol == FabricOperationLeafProtocol::OrderedCardinalityToken;
  }

  bool hasOrderedProductionGroups() const {
    return protocol == FabricOperationLeafProtocol::OrderedCardinalityToken;
  }

  bool hasDirectTokenPublication() const {
    return protocol == FabricOperationLeafProtocol::ManagedToken ||
           protocol == FabricOperationLeafProtocol::TransparentToken;
  }
};

/// Derives the protocol only from the exact implementation family, registered
/// operation schema, and physical port inventory carried by the sealed
/// capability. Invalid or incomplete control/stream shapes reject instead of
/// falling back to a combinational convention.
llvm::Expected<FabricOperationLeafInterface> deriveFabricOperationLeafInterface(
    const fabric::ResolvedFabricOpCapabilityView &capability);

/// Named fields in the structural owner's packed selected-context state. The
/// operation schema remains the semantic owner of every transition; these
/// names only make one capability-derived storage layout reusable by the
/// common shell and its provider.
enum class FabricOperationLeafStateFieldKind {
  Mode,
  RetainedValue,
  Current,
  Limit,
  Step,
  BufferedValue,
  BufferedMask,
};

struct FabricOperationLeafStateFieldLayout final {
  FabricOperationLeafStateFieldKind kind;
  unsigned bitOffset = 0;
  unsigned bitCount = 0;
};

struct FabricOperationLeafStateLayout final {
  std::vector<FabricOperationLeafStateFieldLayout> fields;
  unsigned bitCount = 0;

  const FabricOperationLeafStateFieldLayout *
  find(FabricOperationLeafStateFieldKind kind) const {
    for (const FabricOperationLeafStateFieldLayout &field : fields)
      if (field.kind == kind)
        return &field;
    return nullptr;
  }

  unsigned encodedBitCount() const { return bitCount; }
  llvm::APInt resetValue() const { return llvm::APInt(bitCount, 0); }
};

/// Derives the opaque state-bank shape needed by one exact stateful schema.
/// Stateless families return no layout. The packed representation is a
/// transient lowering agreement and never becomes Fabric identity.
llvm::Expected<std::optional<FabricOperationLeafStateLayout>>
deriveFabricOperationLeafStateLayout(
    const fabric::ResolvedFabricOpCapabilityView &capability);

/// The transient packed-state encoding shared by the LoopCarry provider and
/// its structural state-bank owner. Dataflow remains the semantic state owner.
llvm::APInt encodeLoopCarryOperationLeafState(
    ::dataflow::semantics::CarrySemanticState state);

/// The transient selected-context state boundary shared only by the initial
/// elastic-transparent loop providers. Bit zero is the logical mode. An
/// invariant appends its retained payload at the next bit; carry and gate have
/// no retained payload in this boundary.
struct TransparentLoopOperationLeafStateLayout final {
  static constexpr unsigned modeBit = 0;
  static constexpr unsigned invariantPayloadOffset = 1;

  unsigned payloadWidthBits = 0;

  unsigned encodedBitCount() const {
    return invariantPayloadOffset + payloadWidthBits;
  }

  llvm::APInt resetValue() const { return llvm::APInt(encodedBitCount(), 0); }
};

/// Derives the selected-context state shape from one exact sealed capability.
/// Stateless operations and stateful families with a different boundary return
/// no layout. The result is transient and never participates in Fabric or
/// HardwareImplementation identity.
llvm::Expected<std::optional<TransparentLoopOperationLeafStateLayout>>
deriveTransparentLoopOperationLeafStateLayout(
    const fabric::ResolvedFabricOpCapabilityView &capability);

/// Derives the transient provider boundary from the exact Fabric capability
/// and ConfigurationABI. Nonzero physical payloads and encoded configuration
/// fields form the ordinary combinational boundary. Control/stream families
/// additionally expose ready/valid and, where required, an opaque
/// selected-context state transform. Ordered production exposes the structural
/// continuation state and the leaf's final-production decision. Context
/// selection, state storage, clock, reset, and elastic result storage remain
/// structural-owner responsibilities.
llvm::Expected<std::vector<circt::hw::PortInfo>> deriveFabricOperationLeafPorts(
    mlir::OpBuilder &builder,
    const fabric::FabricPhysicalOccurrenceOwnerRef &occurrence,
    const fabric::ResolvedFabricOpCapabilityView &capability,
    const ConfigurationABI &configurationAbi);

llvm::Error verifyFabricOperationLeafPorts(
    circt::hw::HWModuleGeneratedOp leaf,
    const fabric::FabricPhysicalOccurrenceOwnerRef &occurrence,
    const fabric::ResolvedFabricOpCapabilityView &capability,
    const ConfigurationABI &configurationAbi);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_OPERATIONLEAF_H
