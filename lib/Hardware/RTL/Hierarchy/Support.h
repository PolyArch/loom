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
#include <tuple>
#include <vector>

namespace loom::hardware::rtl::hierarchy {

struct FieldDecoderPlan final {
  const ProgrammingUnit *unit = nullptr;
  std::size_t transportUnitOrdinal = 0;
  std::size_t fieldOrdinal = 0;
  std::uint64_t encodedBitCount = 0;
  std::vector<DestinationSlice> destinationSlices;
};

struct ConfigurationFieldKey final {
  std::size_t transportUnitOrdinal = 0;
  std::size_t fieldOrdinal = 0;

  friend bool operator<(ConfigurationFieldKey lhs,
                        ConfigurationFieldKey rhs) {
    return std::tie(lhs.transportUnitOrdinal, lhs.fieldOrdinal) <
           std::tie(rhs.transportUnitOrdinal, rhs.fieldOrdinal);
  }

  friend bool operator==(ConfigurationFieldKey lhs,
                         ConfigurationFieldKey rhs) {
    return lhs.transportUnitOrdinal == rhs.transportUnitOrdinal &&
           lhs.fieldOrdinal == rhs.fieldOrdinal;
  }
};

struct ConfigurationWordKey final {
  std::size_t transportUnitOrdinal = 0;
  std::uint64_t wordOrdinal = 0;

  friend bool operator<(ConfigurationWordKey lhs, ConfigurationWordKey rhs) {
    return std::tie(lhs.transportUnitOrdinal, lhs.wordOrdinal) <
           std::tie(rhs.transportUnitOrdinal, rhs.wordOrdinal);
  }

  friend bool operator==(ConfigurationWordKey lhs, ConfigurationWordKey rhs) {
    return lhs.transportUnitOrdinal == rhs.transportUnitOrdinal &&
           lhs.wordOrdinal == rhs.wordOrdinal;
  }
};

struct ConfigurationBundleWord final {
  ConfigurationWordKey key;
  std::uint32_t usedBitMask = 0;

  friend bool operator==(ConfigurationBundleWord lhs,
                         ConfigurationBundleWord rhs) {
    return lhs.key == rhs.key && lhs.usedBitMask == rhs.usedBitMask;
  }
};

/// Transient canonical packing of exact ConfigurationABI fields consumed by
/// one hierarchy subtree. Membership and widths remain derived cache state.
struct ConfigurationBundlePlan final {
  std::vector<ConfigurationBundleWord> words;

  const ConfigurationBundleWord *find(ConfigurationWordKey key) const;
  bool empty() const { return words.empty(); }
};

struct ConfigurationBundleSignals final {
  const ConfigurationBundlePlan *plan = nullptr;
  mlir::Value bundle;
  std::vector<mlir::Value> cachedWords;
};

inline constexpr llvm::StringLiteral configurationBundlePortName =
    "configuration_bundle";
inline constexpr llvm::StringLiteral configurationValuePortName =
    "configuration_value";
/// The InstructionContextRef a Temporal PE grants to one child FU for the
/// current clock cycle. The FU and its operation shells evaluate that context's
/// configuration, operand heads, and state bank; a boundary token belongs to it
/// by construction, while an FU-internal result names its own context.
inline constexpr llvm::StringLiteral dispatchContextPortName =
    "dispatch_context";
/// Whether that context owns this cycle's evaluation service grant. A default
/// context index without this grant cannot start an operation transition.
inline constexpr llvm::StringLiteral dispatchEnablePortName = "dispatch_enable";

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

std::string endpointKey(const fabric::FabricTransportEndpointRef &endpoint);

ConfigurationFieldKey configurationFieldKey(const FieldDecoderPlan &decoder);
mlir::Type configurationBundleType(mlir::MLIRContext *context,
                                   const ConfigurationBundlePlan &plan);
llvm::Error verifyConfigurationBundlePort(
    circt::hw::HWModuleOp module, const ConfigurationBundlePlan &plan);
llvm::Error verifyConfigurationValuePort(circt::hw::HWModuleOp module,
                                         const FieldDecoderPlan &decoder);

llvm::Expected<ConfigurationBundlePlan> deriveConfigurationBundlePlan(
    llvm::ArrayRef<FieldDecoderPlan> decoders,
    llvm::ArrayRef<ConfigurationBundlePlan> childBundles = {});

llvm::Expected<std::vector<FieldDecoderPlan>>
prepareFieldDecoders(const ConfigurationABI &configurationAbi,
                     const ConfigurationTransportLayout &transportLayout);

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
    mlir::OpBuilder &builder, const ConfigurationBundlePlan &configuration,
    llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs);

llvm::Expected<mlir::Value> projectConfigurationBundle(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value parentValue,
    const ConfigurationBundlePlan &parent,
    const ConfigurationBundlePlan &child);

llvm::Error addConfigurationInstanceInput(
    mlir::OpBuilder &builder, mlir::Location location,
    circt::hw::HWModulePortAccessor &accessor,
    const ConfigurationBundlePlan &parent,
    const ConfigurationBundlePlan &child, circt::hw::HWModuleOp childModule,
    std::map<std::string, mlir::Value> &inputs);

ConfigurationBundleSignals configurationBundleSignals(
    circt::hw::HWModulePortAccessor &accessor,
    const ConfigurationBundlePlan &configuration);

/// The ordinal width of a closed domain of `count` members (at least one bit).
unsigned indexWidth(std::uint64_t count);

mlir::Value bitConstant(mlir::OpBuilder &builder, mlir::Location location,
                        bool value);
mlir::Value andValues(mlir::OpBuilder &builder, mlir::Location location,
                      llvm::ArrayRef<mlir::Value> values);
mlir::Value orValues(mlir::OpBuilder &builder, mlir::Location location,
                     llvm::ArrayRef<mlir::Value> values);
mlir::Value decodeFieldSignal(mlir::OpBuilder &builder, mlir::Location location,
                              ConfigurationBundleSignals &configuration,
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
