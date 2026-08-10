#ifndef LOOM_LIB_HARDWARE_RTL_HIERARCHY_CONFIGURATIONCONTROLLER_H
#define LOOM_LIB_HARDWARE_RTL_HIERARCHY_CONFIGURATIONCONTROLLER_H

#include "Support.h"

namespace loom::hardware::rtl::hierarchy {

struct ConfigurationControllerModule final {
  circt::hw::HWModuleOp module;
};

void appendAxiLiteConfigurationPorts(
    mlir::OpBuilder &builder,
    llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs,
    llvm::SmallVectorImpl<circt::hw::PortInfo> &outputs);

llvm::Expected<ConfigurationControllerModule>
buildConfigurationControllerModule(
    mlir::OpBuilder &builder, mlir::Location location,
    const ConfigurationABI &configurationAbi,
    const ConfigurationTransportLayout &transportLayout,
    const ClockResetPlan &clockReset);

} // namespace loom::hardware::rtl::hierarchy

#endif // LOOM_LIB_HARDWARE_RTL_HIERARCHY_CONFIGURATIONCONTROLLER_H
