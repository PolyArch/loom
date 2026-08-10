#ifndef LOOM_LIB_HARDWARE_RTL_HIERARCHY_COMPONENTS_H
#define LOOM_LIB_HARDWARE_RTL_HIERARCHY_COMPONENTS_H

#include "MemoryService.h"
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

struct FuModule final {
  fabric::FabricFuOccurrenceRef reference;
  circt::hw::HWModuleOp module;
  std::vector<EndpointPlan> endpoints;
  std::optional<unsigned> contextWidthBits;
};
using PeModule = ComponentModule<fabric::FabricPeOccurrenceRef>;
using SwitchModule = ComponentModule<fabric::FabricSwitchOccurrenceRef>;
using FifoModule = ComponentModule<fabric::FabricFifoOccurrenceRef>;
using BoundaryModule = ComponentModule<fabric::FabricBoundaryOccurrenceRef>;
struct MemoryModule final {
  fabric::FabricMemoryOccurrenceRef reference;
  circt::hw::HWModuleOp module;
  std::vector<EndpointPlan> endpoints;
  std::vector<MemoryEndpointPortPlan> memoryEndpoints;
};

llvm::Expected<std::vector<FuModule>>
buildFuModules(mlir::OpBuilder &builder, mlir::Location location,
               fabric::SpatialCoreOccurrenceRef spatialCore,
               const fabric::FabricArtifactView &fabric,
               const ConfigurationABI &configurationAbi,
               const ConfigurationTransportLayout &transportLayout,
               llvm::ArrayRef<OperationShellModule> operationShells,
               const ClockResetPlan &clockReset);

llvm::Expected<std::vector<PeModule>>
buildPeModules(mlir::OpBuilder &builder, mlir::Location location,
               fabric::SpatialCoreOccurrenceRef spatialCore,
               const fabric::FabricArtifactView &fabric,
               const ConfigurationABI &configurationAbi,
               const ConfigurationTransportLayout &transportLayout,
               llvm::ArrayRef<FuModule> fuModules,
               const ClockResetPlan &clockReset);

llvm::Expected<PeModule>
buildTemporalPeModule(mlir::OpBuilder &builder, mlir::Location location,
                      fabric::SpatialCoreOccurrenceRef spatialCore,
                      const fabric::FabricArtifactView &fabric,
                      const ConfigurationABI &configurationAbi,
                      const ConfigurationTransportLayout &transportLayout,
                      llvm::ArrayRef<FuModule> fuModules,
                      const ClockResetPlan &clockReset,
                      fabric::FabricPeOccurrenceRef pe);

llvm::Expected<std::vector<SwitchModule>>
buildSwitchModules(mlir::OpBuilder &builder, mlir::Location location,
                   fabric::SpatialCoreOccurrenceRef spatialCore,
                   const fabric::FabricArtifactView &fabric,
                   const ConfigurationABI &configurationAbi,
                   const ConfigurationTransportLayout &transportLayout,
                   const ClockResetPlan &clockReset);

llvm::Expected<std::vector<FifoModule>>
buildFifoModules(mlir::OpBuilder &builder, mlir::Location location,
                 fabric::SpatialCoreOccurrenceRef spatialCore,
                 const fabric::FabricArtifactView &fabric,
                 const ConfigurationABI &configurationAbi,
                 const ConfigurationTransportLayout &transportLayout,
                 const ClockResetPlan &clockReset);

llvm::Expected<std::vector<BoundaryModule>>
buildBoundaryModules(mlir::OpBuilder &builder, mlir::Location location,
                     fabric::SpatialCoreOccurrenceRef spatialCore,
                     const fabric::FabricArtifactView &fabric,
                     const ConfigurationABI &configurationAbi,
                     const ConfigurationTransportLayout &transportLayout);

llvm::Expected<std::vector<MemoryModule>>
buildMemoryModules(mlir::OpBuilder &builder, mlir::Location location,
                   fabric::SpatialCoreOccurrenceRef spatialCore,
                   const fabric::FabricArtifactView &fabric,
                   const ConfigurationABI &configurationAbi,
                   const ConfigurationTransportLayout &transportLayout,
                   const ClockResetPlan &clockReset,
                   const PortableMemoryServiceLayout &memoryServiceLayout);

} // namespace loom::hardware::rtl::hierarchy

#endif // LOOM_LIB_HARDWARE_RTL_HIERARCHY_COMPONENTS_H
