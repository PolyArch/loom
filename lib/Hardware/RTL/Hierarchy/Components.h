#ifndef LOOM_LIB_HARDWARE_RTL_HIERARCHY_COMPONENTS_H
#define LOOM_LIB_HARDWARE_RTL_HIERARCHY_COMPONENTS_H

#include "OperationShell.h"
#include "Support.h"

#include "circt/Dialect/HW/HWOps.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace loom::hardware::rtl::hierarchy {

template <typename Reference> struct ComponentModule final {
  Reference reference;
  circt::hw::HWModuleOp module;
  std::vector<EndpointPlan> endpoints;
};

using FuModule = ComponentModule<fabric::FabricFuOccurrenceRef>;
using PeModule = ComponentModule<fabric::FabricPeOccurrenceRef>;
using SwitchModule = ComponentModule<fabric::FabricSwitchOccurrenceRef>;
using FifoModule = ComponentModule<fabric::FabricFifoOccurrenceRef>;
using BoundaryModule = ComponentModule<fabric::FabricBoundaryOccurrenceRef>;
using MemoryModule = ComponentModule<fabric::FabricMemoryOccurrenceRef>;

llvm::Expected<std::vector<FuModule>>
buildFuModules(mlir::OpBuilder &builder, mlir::Location location,
               fabric::SpatialCoreOccurrenceRef spatialCore,
               const fabric::FabricArtifactView &fabric,
               const ConfigurationABI &configurationAbi,
               llvm::ArrayRef<OperationShellModule> operationShells,
               const ClockResetPlan &clockReset);

llvm::Expected<std::vector<PeModule>>
buildPeModules(mlir::OpBuilder &builder, mlir::Location location,
               fabric::SpatialCoreOccurrenceRef spatialCore,
               const fabric::FabricArtifactView &fabric,
               const ConfigurationABI &configurationAbi,
               llvm::ArrayRef<FuModule> fuModules,
               const ClockResetPlan &clockReset);

llvm::Expected<std::vector<SwitchModule>>
buildSwitchModules(mlir::OpBuilder &builder, mlir::Location location,
                   fabric::SpatialCoreOccurrenceRef spatialCore,
                   const fabric::FabricArtifactView &fabric,
                   const ConfigurationABI &configurationAbi);

llvm::Expected<std::vector<FifoModule>>
buildFifoModules(mlir::OpBuilder &builder, mlir::Location location,
                 fabric::SpatialCoreOccurrenceRef spatialCore,
                 const fabric::FabricArtifactView &fabric,
                 const ConfigurationABI &configurationAbi,
                 const ClockResetPlan &clockReset);

llvm::Expected<std::vector<BoundaryModule>>
buildBoundaryModules(mlir::OpBuilder &builder, mlir::Location location,
                     fabric::SpatialCoreOccurrenceRef spatialCore,
                     const fabric::FabricArtifactView &fabric,
                     const ConfigurationABI &configurationAbi);

} // namespace loom::hardware::rtl::hierarchy

#endif // LOOM_LIB_HARDWARE_RTL_HIERARCHY_COMPONENTS_H
