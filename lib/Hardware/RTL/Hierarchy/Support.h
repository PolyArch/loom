#ifndef LOOM_LIB_HARDWARE_RTL_HIERARCHY_SUPPORT_H
#define LOOM_LIB_HARDWARE_RTL_HIERARCHY_SUPPORT_H

#include "Fabric/Identity/FabricRefImport.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/RTL/ConfigurationTransport.h"
#include "Hardware/RTL/Transport.h"

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace loom::hardware::rtl::hierarchy {

struct FieldDecoderPlan final {
  const ProgrammingUnit *unit = nullptr;
  std::size_t transportUnitOrdinal = 0;
  std::uint64_t encodedBitCount = 0;
  std::vector<std::uint64_t> destinationBits;
};

struct ClockResetPlan final {
  bool asynchronousReset = false;
  bool activeLowReset = false;
};

struct EndpointPlan final {
  fabric::FabricTransportEndpointRef endpoint;
  fabric::FabricPortDirection direction = fabric::FabricPortDirection::Input;
  fabric::FabricOrdinal localOrdinal = 0;
  ::fabric::DataPathType dataPath;
  std::optional<circt::hw::PortInfo> data;
  std::optional<circt::hw::PortInfo> tag;
  circt::hw::PortInfo valid;
  circt::hw::PortInfo ready;
};

struct ChannelSignals final {
  std::optional<mlir::Value> data;
  std::optional<mlir::Value> tag;
  mlir::Value valid;
  mlir::Value ready;
};

llvm::Error invalid(const llvm::Twine &message);
llvm::Error unsupported(const llvm::Twine &message);

std::string configurationPortName(std::size_t transportUnitOrdinal);
std::string endpointKey(const fabric::FabricTransportEndpointRef &endpoint);

llvm::Expected<fabric::FabricPhysicalConfigurationFieldRef>
qualifyConfigurationField(fabric::SpatialCoreOccurrenceRef spatialCore,
                          const fabric::FabricSemanticConfigFieldRef &field);

llvm::Expected<FieldDecoderPlan>
prepareFieldDecoder(fabric::SpatialCoreOccurrenceRef spatialCore,
                    const fabric::FabricSemanticConfigFieldRef &field,
                    const ConfigurationABI &configurationAbi,
                    const ConfigurationTransportLayout &transportLayout);

llvm::Expected<FieldDecoderPlan>
prepareFieldDecoder(fabric::SpatialCoreOccurrenceRef spatialCore,
                    const fabric::FabricSemanticConfigFieldRef &field,
                    const fabric::FabricConfigurationResidency &residency,
                    const ConfigurationABI &configurationAbi,
                    const ConfigurationTransportLayout &transportLayout);

llvm::Expected<std::pair<FieldDecoderPlan, const FiniteCodebookEncoding *>>
prepareFiniteField(fabric::SpatialCoreOccurrenceRef spatialCore,
                   const fabric::FabricSemanticConfigFieldRef &field,
                   const ConfigurationABI &configurationAbi,
                   const ConfigurationTransportLayout &transportLayout);

llvm::Expected<std::pair<FieldDecoderPlan, const FiniteCodebookEncoding *>>
prepareFiniteField(fabric::SpatialCoreOccurrenceRef spatialCore,
                   const fabric::FabricSemanticConfigFieldRef &field,
                   const fabric::FabricConfigurationResidency &residency,
                   const ConfigurationABI &configurationAbi,
                   const ConfigurationTransportLayout &transportLayout);

llvm::Expected<llvm::APInt>
physicalCode(const FiniteCodebookEncoding &codebook,
             llvm::ArrayRef<std::uint8_t> semanticValue);

llvm::Expected<ClockResetPlan>
prepareClockReset(const fabric::FabricSystemRootView &system,
                  fabric::SpatialCoreOccurrenceRef spatialCore);

llvm::Expected<std::vector<EndpointPlan>>
deriveEndpointPlans(mlir::OpBuilder &builder,
                    const fabric::FabricArtifactView &fabric,
                    const fabric::FabricTransportEndpointOwnerRef &owner);

void appendEndpointPorts(llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs,
                         llvm::SmallVectorImpl<circt::hw::PortInfo> &outputs,
                         const EndpointPlan &endpoint);

void appendClockResetAndConfigurationPorts(
    mlir::OpBuilder &builder, const ConfigurationABI &configurationAbi,
    const ConfigurationTransportLayout &transportLayout,
    llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs);

mlir::Value bitConstant(mlir::OpBuilder &builder, mlir::Location location,
                        bool value);
mlir::Value andValues(mlir::OpBuilder &builder, mlir::Location location,
                      llvm::ArrayRef<mlir::Value> values);
mlir::Value orValues(mlir::OpBuilder &builder, mlir::Location location,
                     llvm::ArrayRef<mlir::Value> values);
mlir::Value decodeFieldSignal(mlir::OpBuilder &builder, mlir::Location location,
                              circt::hw::HWModulePortAccessor &accessor,
                              const FieldDecoderPlan &decoder);
mlir::Value matchesCode(mlir::OpBuilder &builder, mlir::Location location,
                        mlir::Value field, const llvm::APInt &code);
mlir::Value selectedBit(mlir::OpBuilder &builder, mlir::Location location,
                        mlir::Value field, std::uint64_t bit);

llvm::Expected<ForwardTransportSignals>
adaptForward(mlir::OpBuilder &builder, mlir::Location location,
             const EndpointPlan &source, const EndpointPlan &destination,
             const ChannelSignals &signals);

mlir::Value createRegister(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Value next, mlir::Value clock,
                           mlir::Value reset, const llvm::APInt &resetValue,
                           llvm::StringRef name, bool asynchronousReset);

llvm::Expected<std::map<std::string, mlir::Value>>
instantiateModule(mlir::OpBuilder &builder, mlir::Location location,
                  circt::hw::HWModuleOp module, llvm::StringRef instanceName,
                  const std::map<std::string, mlir::Value> &inputs);

llvm::Expected<mlir::Operation *>
findCanonicalEntityOperation(const fabric::FabricArtifactView &fabric,
                             fabric::FabricEntityId id);

llvm::Expected<mlir::Operation *>
findCanonicalFuNodeOperation(const fabric::FabricArtifactView &fabric,
                             fabric::FabricFuOccurrenceNodeRef node);

} // namespace loom::hardware::rtl::hierarchy

#endif // LOOM_LIB_HARDWARE_RTL_HIERARCHY_SUPPORT_H
