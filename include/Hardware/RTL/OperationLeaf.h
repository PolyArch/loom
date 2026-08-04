#ifndef LOOM_HARDWARE_RTL_OPERATIONLEAF_H
#define LOOM_HARDWARE_RTL_OPERATIONLEAF_H

#include "Fabric/Identity/FabricRefImport.h"
#include "Hardware/Configuration/ConfigurationABI.h"

#include "circt/Dialect/HW/HWOps.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace mlir {
class OpBuilder;
}

namespace loom::hardware::rtl {

/// Derives the transient provider boundary from the exact Fabric capability
/// and ConfigurationABI. Nonzero physical input payloads precede encoded
/// configuration fields; nonzero physical output payloads follow them.
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
