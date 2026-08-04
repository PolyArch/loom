#ifndef LOOM_HARDWARE_RTL_OPERATIONLEAF_H
#define LOOM_HARDWARE_RTL_OPERATIONLEAF_H

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Hardware/Configuration/ConfigurationABI.h"

#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace mlir {
class OpBuilder;
}

namespace loom::hardware::rtl {

/// The transient packed-state encoding shared by the LoopCarry provider and
/// its structural state-bank owner. Dataflow remains the semantic state owner.
llvm::APInt encodeLoopCarryOperationLeafState(
    ::dataflow::semantics::CarrySemanticState state);

/// Derives the transient provider boundary from the exact Fabric capability
/// and ConfigurationABI. Nonzero physical payloads and encoded configuration
/// fields form the ordinary combinational boundary. The LoopCarry provider
/// additionally exposes ready/valid and a selected-context state transform.
/// Context selection, state storage, clock, and reset remain structural-owner
/// responsibilities.
llvm::Expected<std::vector<circt::hw::PortInfo>> deriveFabricOperationLeafPorts(
    mlir::OpBuilder &builder,
    const fabric::ResolvedFabricOpCapabilityView &capability,
    const ConfigurationABI &configurationAbi);

llvm::Error verifyFabricOperationLeafPorts(
    circt::hw::HWModuleGeneratedOp leaf,
    const fabric::ResolvedFabricOpCapabilityView &capability,
    const ConfigurationABI &configurationAbi);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_OPERATIONLEAF_H
