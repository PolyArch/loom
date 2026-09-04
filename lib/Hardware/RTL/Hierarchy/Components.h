#ifndef LOOM_LIB_HARDWARE_RTL_HIERARCHY_COMPONENTS_H
#define LOOM_LIB_HARDWARE_RTL_HIERARCHY_COMPONENTS_H

#include "MemoryService.h"
#include "OperationShell.h"
#include "Support.h"

#include "circt/Dialect/HW/HWOps.h"
#include "llvm/Support/Error.h"

#include <map>
#include <string>
#include <vector>

namespace loom::hardware::rtl::hierarchy {

inline constexpr llvm::StringLiteral resultPresentationPriorityPortName =
    "result_presentation_priority";
inline constexpr llvm::StringLiteral resultRequesterOffersPortName =
    "result_requester_offers";

/// Availability of a result before any downstream admission.
inline std::string fuOutputOfferPortName(const EndpointPlan &endpoint) {
  return "output_" + std::to_string(endpoint.localOrdinal) + "_offer";
}

inline std::string fuOutputRequesterPortName(const EndpointPlan &endpoint) {
  return "output_" + std::to_string(endpoint.localOrdinal) + "_requester";
}

template <typename Reference> struct ComponentModule final {
  Reference reference;
  circt::hw::HWModuleOp module;
  std::vector<EndpointPlan> endpoints;
  ConfigurationBundlePlan configuration;
};

struct FuModule final {
  fabric::FabricFuOccurrenceRef reference;
  circt::hw::HWModuleOp module;
  std::vector<EndpointPlan> endpoints;
  std::optional<unsigned> contextWidthBits;
  std::vector<bool> resultRequesterDirectPublication;
  ConfigurationBundlePlan configuration;
};
using PeModule = ComponentModule<fabric::FabricPeOccurrenceRef>;
struct SwitchModule final {
  fabric::FabricSwitchOccurrenceRef reference;
  circt::hw::HWModuleOp module;
  std::vector<EndpointPlan> endpoints;
  std::vector<std::uint8_t> implementationKey;
  ConfigurationBundlePlan configuration;
  FieldDecoderPlan configurationDecoder;
};
struct FifoModule final {
  fabric::FabricFifoOccurrenceRef reference;
  circt::hw::HWModuleOp module;
  std::vector<EndpointPlan> endpoints;
  std::vector<std::uint8_t> implementationKey;
  ConfigurationBundlePlan configuration;
  FieldDecoderPlan configurationDecoder;
};
using BoundaryModule = ComponentModule<fabric::FabricBoundaryOccurrenceRef>;
struct MemoryModule final {
  fabric::FabricMemoryOccurrenceRef reference;
  circt::hw::HWModuleOp module;
  std::vector<EndpointPlan> endpoints;
  std::vector<MemoryEndpointPortPlan> memoryEndpoints;
  std::string implementationKey;
  ConfigurationBundlePlan configuration;
  FieldDecoderPlan configurationDecoder;
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
               const ClockResetPlan &clockReset, mlir::ModuleOp container,
               llvm::StringRef materializationKey = {});

llvm::Expected<PeModule> buildTemporalPeModule(
    mlir::OpBuilder &builder, mlir::Location location,
    fabric::SpatialCoreOccurrenceRef spatialCore,
    const fabric::FabricArtifactView &fabric,
    const ConfigurationABI &configurationAbi,
    const ConfigurationTransportLayout &transportLayout,
    llvm::ArrayRef<FuModule> fuModules, const ClockResetPlan &clockReset,
    fabric::FabricPeOccurrenceRef pe, llvm::StringRef materializationKey);

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
                   const PortableMemoryServiceLayout &memoryServiceLayout,
                   llvm::StringRef materializationKey);

} // namespace loom::hardware::rtl::hierarchy

#endif // LOOM_LIB_HARDWARE_RTL_HIERARCHY_COMPONENTS_H
