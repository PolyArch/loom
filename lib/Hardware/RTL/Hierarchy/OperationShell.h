#ifndef LOOM_LIB_HARDWARE_RTL_HIERARCHY_OPERATIONSHELL_H
#define LOOM_LIB_HARDWARE_RTL_HIERARCHY_OPERATIONSHELL_H

#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/PhysicalOperation.h"

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::hardware::rtl::hierarchy {

struct ClockResetPlan;

} // namespace loom::hardware::rtl::hierarchy

namespace loom::hardware::rtl {
struct ConfigurationTransportLayout;
}

namespace loom::hardware::rtl::hierarchy {

struct OperationEndpointPlan final {
  fabric::FabricPortDirection direction = fabric::FabricPortDirection::Input;
  fabric::FabricOrdinal ordinal = 0;
  std::uint32_t payloadWidthBits = 0;
  std::optional<circt::hw::PortInfo> data;
  circt::hw::PortInfo valid;
  circt::hw::PortInfo ready;
};

struct OperationShellModule final {
  ResolvedFabricPhysicalOperation operation;
  circt::hw::HWModuleOp module;
  std::vector<OperationEndpointPlan> endpoints;
};

llvm::Expected<std::vector<OperationShellModule>> buildOperationShellModules(
    mlir::OpBuilder &builder, mlir::Location location,
    fabric::SpatialCoreOccurrenceRef spatialCore,
    const ConfigurationABI &configurationAbi,
    const ConfigurationTransportLayout &transportLayout,
    llvm::ArrayRef<ResolvedFabricPhysicalOperation> operations,
    std::vector<FabricOperationLeafAssociation> &associations,
    const ClockResetPlan &clockReset);

} // namespace loom::hardware::rtl::hierarchy

#endif // LOOM_LIB_HARDWARE_RTL_HIERARCHY_OPERATIONSHELL_H
