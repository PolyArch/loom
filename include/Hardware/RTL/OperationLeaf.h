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
/// fields form the ordinary combinational boundary. The initial transparent
/// loop providers additionally expose ready/valid and a selected-context state
/// transform. Context selection, state storage, clock, and reset remain
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
