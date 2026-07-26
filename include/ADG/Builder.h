#ifndef LOOM_ADG_BUILDER_H
#define LOOM_ADG_BUILDER_H

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricHardwareDomainContracts.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/IR/FabricEnums.h"
#include "Fabric/IR/ImplementationFamily.h"
#include "Fabric/IR/MemoryOperationPort.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/IR/SystemServiceContract.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "mlir/IR/Value.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace loom::adg {
namespace detail {
class DesignState;
struct SystemHandleAccess;
} // namespace detail

class FuBuilder;
class PeBuilder;
class HardwareDomainBuilder;
class ServiceTransformBuilder;
class SystemBuilder;

/// A typed authoring description of one legal fabric.module port type.
class PortType final {
public:
  enum class Kind : std::uint8_t { Bits, TaggedBits, Memory };

  static constexpr std::int64_t kDynamicExtent =
      std::numeric_limits<std::int64_t>::min();

  static llvm::Expected<PortType> bits(std::uint32_t width);
  static llvm::Expected<PortType> taggedBits(std::uint32_t width,
                                             std::uint32_t tagWidth);
  static llvm::Expected<PortType> memory(llvm::ArrayRef<std::int64_t> shape,
                                         const PortType &elementType);

  Kind kind() const { return kind_; }
  std::uint32_t width() const { return width_; }
  std::uint32_t tagWidth() const { return tagWidth_; }
  llvm::ArrayRef<std::int64_t> shape() const { return shape_; }

  friend bool operator==(const PortType &left, const PortType &right) {
    return left.kind_ == right.kind_ && left.width_ == right.width_ &&
           left.tagWidth_ == right.tagWidth_ && left.shape_ == right.shape_;
  }

private:
  PortType(Kind kind, std::uint32_t width, std::uint32_t tagWidth,
           std::vector<std::int64_t> shape)
      : kind_(kind), width_(width), tagWidth_(tagWidth),
        shape_(std::move(shape)) {}

  Kind kind_;
  std::uint32_t width_;
  std::uint32_t tagWidth_;
  std::vector<std::int64_t> shape_;
};

/// An owner-checked SSA transport or memory capability in one SpatialCore.
class SpatialValue final {
public:
  SpatialValue() = default;

private:
  SpatialValue(const std::shared_ptr<detail::DesignState> &state,
               std::size_t rootOrdinal, mlir::Value value)
      : state_(state), rootOrdinal_(rootOrdinal), value_(value) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  mlir::Value value_;

  friend class SpatialCoreBuilder;
  friend class PeBuilder;
};

/// An owner-checked value on one PE's internal untagged boundary.
class PeValue final {
public:
  PeValue() = default;

private:
  PeValue(const std::shared_ptr<detail::DesignState> &state,
          std::size_t rootOrdinal, std::size_t peOrdinal, mlir::Value value)
      : state_(state), rootOrdinal_(rootOrdinal), peOrdinal_(peOrdinal),
        value_(value) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  std::size_t peOrdinal_ = 0;
  mlir::Value value_;

  friend class PeBuilder;
};

/// An owner-checked value in one FU graph region.
class FuValue final {
public:
  FuValue() = default;

private:
  FuValue(const std::shared_ptr<detail::DesignState> &state,
          std::size_t rootOrdinal, std::size_t peOrdinal, std::size_t fuOrdinal,
          mlir::Value value)
      : state_(state), rootOrdinal_(rootOrdinal), peOrdinal_(peOrdinal),
        fuOrdinal_(fuOrdinal), value_(value) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  std::size_t peOrdinal_ = 0;
  std::size_t fuOrdinal_ = 0;
  mlir::Value value_;

  friend class FuBuilder;
};

enum class FuConfigurationMode : std::uint8_t { PerInstruction, PerFu };

struct TemporalRegisterFifoParameters final {
  std::uint32_t count;
  std::uint32_t depth;
  std::uint32_t ports;
};

struct TemporalPeParameters final {
  std::uint32_t instructionCapacity;
  FuConfigurationMode fuConfigurationMode;
  ::fabric::OperandBufferMode operandBufferMode;
  std::uint32_t operandBufferSize;
  std::optional<TemporalRegisterFifoParameters> registerFifos;
};

/// A closed typed description of one anonymous PE boundary and schedule.
class PeSpec final {
public:
  static PeSpec spatial(std::vector<PortType> inputTypes,
                        std::vector<PortType> outputTypes);
  static PeSpec temporal(std::vector<PortType> inputTypes,
                         std::vector<PortType> outputTypes,
                         TemporalPeParameters parameters);

private:
  PeSpec(::fabric::Schedule schedule, std::vector<PortType> inputTypes,
         std::vector<PortType> outputTypes,
         std::optional<TemporalPeParameters> temporal)
      : schedule_(schedule), inputTypes_(std::move(inputTypes)),
        outputTypes_(std::move(outputTypes)), temporal_(std::move(temporal)) {}

  ::fabric::Schedule schedule_;
  std::vector<PortType> inputTypes_;
  std::vector<PortType> outputTypes_;
  std::optional<TemporalPeParameters> temporal_;

  friend class SpatialCoreBuilder;
};

struct FuSpec final {
  std::vector<PortType> inputTypes;
  std::vector<PortType> outputTypes;
};

/// One concrete fabric.op capability using the generated semantic owners.
struct OperationCapabilitySpec final {
  ::fabric::ImplementationFamilyId implementationFamily;
  ::fabric::FamilyCapabilityParams hardwareParameters;
  std::vector<::dataflow::OperationSchemaId> enabledOperations;
  std::vector<PortType> outputTypes;
};

struct FifoSpec final {
  PortType outputType;
  std::uint32_t maxDepth;
  bool bypassable;
};

struct BoundarySpec final {
  ::fabric::BoundaryDirection direction;
  std::vector<PortType> inputTypes;
  std::vector<PortType> outputTypes;

  static BoundarySpec s2t(const PortType &dataInput, const PortType &tagInput,
                          const PortType &taggedOutput);
  static BoundarySpec t2s(const PortType &taggedInput,
                          llvm::ArrayRef<PortType> outputs);
};

struct SwitchSpec final {
  ::fabric::Schedule schedule;
  std::vector<PortType> inputTypes;
  std::vector<PortType> outputTypes;
  std::vector<std::vector<std::uint32_t>> sourcesByOutput;
  std::optional<std::uint32_t> routeTableSize;

  static SwitchSpec
  spatial(std::vector<PortType> inputTypes, std::vector<PortType> outputTypes,
          std::vector<std::vector<std::uint32_t>> sourcesByOutput);

  static SwitchSpec
  temporal(std::vector<PortType> inputTypes, std::vector<PortType> outputTypes,
           std::vector<std::vector<std::uint32_t>> sourcesByOutput,
           std::uint32_t routeTableSize);
};

/// One exact spatial fabric.mem Operation Engine declaration.
class MemorySpec final {
public:
  static MemorySpec
  spatial(std::vector<PortType> inputTypes, std::vector<PortType> outputTypes,
          std::vector<std::uint32_t> managerInputOrdinals,
          std::vector<std::uint32_t> subordinateOutputOrdinals,
          std::vector<::fabric::MemoryOperationPortDeclaration> operationPorts);

private:
  MemorySpec(
      std::vector<PortType> inputTypes, std::vector<PortType> outputTypes,
      std::vector<std::uint32_t> managerInputOrdinals,
      std::vector<std::uint32_t> subordinateOutputOrdinals,
      std::vector<::fabric::MemoryOperationPortDeclaration> operationPorts)
      : inputTypes_(std::move(inputTypes)),
        outputTypes_(std::move(outputTypes)),
        managerInputOrdinals_(std::move(managerInputOrdinals)),
        subordinateOutputOrdinals_(std::move(subordinateOutputOrdinals)),
        operationPorts_(std::move(operationPorts)) {}

  std::vector<PortType> inputTypes_;
  std::vector<PortType> outputTypes_;
  std::vector<std::uint32_t> managerInputOrdinals_;
  std::vector<std::uint32_t> subordinateOutputOrdinals_;
  std::vector<::fabric::MemoryOperationPortDeclaration> operationPorts_;

  friend class SpatialCoreBuilder;
};

class FuBuilder final {
public:
  llvm::Expected<FuValue> input(std::size_t ordinal) const;

  llvm::Expected<std::vector<FuValue>>
  addOperation(llvm::ArrayRef<FuValue> inputs,
               const OperationCapabilitySpec &spec);

  llvm::Expected<FuValue> addMux(llvm::ArrayRef<FuValue> inputs);

  llvm::Expected<std::vector<FuValue>> addDemux(FuValue input,
                                                std::uint32_t outputCount);

  llvm::Error close(llvm::ArrayRef<FuValue> outputs);

private:
  llvm::Expected<mlir::Value>
  resolveValue(const std::shared_ptr<detail::DesignState> &state,
               const FuValue &value) const;

  FuBuilder(const std::shared_ptr<detail::DesignState> &state,
            std::size_t rootOrdinal, std::size_t peOrdinal,
            std::size_t fuOrdinal)
      : state_(state), rootOrdinal_(rootOrdinal), peOrdinal_(peOrdinal),
        fuOrdinal_(fuOrdinal) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_;
  std::size_t peOrdinal_;
  std::size_t fuOrdinal_;

  friend class PeBuilder;
};

class PeBuilder final {
public:
  llvm::Expected<PeValue> input(std::size_t ordinal) const;
  llvm::Expected<SpatialValue> output(std::size_t ordinal) const;

  llvm::Expected<FuBuilder> addFu(llvm::ArrayRef<PeValue> inputs,
                                  const FuSpec &spec);

  llvm::Error close();

private:
  llvm::Expected<mlir::Value>
  resolveValue(const std::shared_ptr<detail::DesignState> &state,
               const PeValue &value) const;

  PeBuilder(const std::shared_ptr<detail::DesignState> &state,
            std::size_t rootOrdinal, std::size_t peOrdinal)
      : state_(state), rootOrdinal_(rootOrdinal), peOrdinal_(peOrdinal) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_;
  std::size_t peOrdinal_;

  friend class SpatialCoreBuilder;
};

class SpatialCoreBuilder final {
public:
  llvm::Expected<SpatialValue> input(std::size_t ordinal) const;

  llvm::Expected<SpatialValue> addFifo(SpatialValue input,
                                       const FifoSpec &spec);

  llvm::Expected<std::vector<SpatialValue>>
  addBoundary(llvm::ArrayRef<SpatialValue> inputs, const BoundarySpec &spec);

  llvm::Expected<std::vector<SpatialValue>>
  addSwitch(llvm::ArrayRef<SpatialValue> inputs, const SwitchSpec &spec);

  llvm::Expected<std::vector<SpatialValue>>
  addMemory(llvm::ArrayRef<SpatialValue> inputs, const MemorySpec &spec);

  llvm::Expected<PeBuilder> addPe(llvm::ArrayRef<SpatialValue> inputs,
                                  const PeSpec &spec);

  /// Closes this root with the exact declared result sequence.
  llvm::Error close(llvm::ArrayRef<SpatialValue> outputs);

private:
  llvm::Expected<mlir::Value>
  resolveValue(const std::shared_ptr<detail::DesignState> &state,
               const SpatialValue &value) const;

  SpatialCoreBuilder(const std::shared_ptr<detail::DesignState> &state,
                     std::size_t rootOrdinal)
      : state_(state), rootOrdinal_(rootOrdinal) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_;

  friend class DesignBuilder;
};

/// One exact published Module dependency selected for a System root.
class ImportedSpatialCore final {
public:
  ImportedSpatialCore() = default;

private:
  ImportedSpatialCore(const std::shared_ptr<detail::DesignState> &state,
                      std::size_t rootOrdinal, std::size_t importOrdinal)
      : state_(state), rootOrdinal_(rootOrdinal),
        importOrdinal_(importOrdinal) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  std::size_t importOrdinal_ = 0;

  friend class SystemBuilder;
  friend struct detail::SystemHandleAccess;
};

/// One role-specific projection admitted as a hardware-domain member.
class HardwareDomainMember final {
public:
  HardwareDomainMember() = default;

private:
  HardwareDomainMember(const std::shared_ptr<detail::DesignState> &state,
                       std::size_t rootOrdinal,
                       loom::fabric::FabricInventoryOwnerRef owner)
      : state_(state), rootOrdinal_(rootOrdinal), owner_(std::move(owner)) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  loom::fabric::FabricInventoryOwnerRef owner_;

  friend class HostCore;
  friend class AccCore;
  friend class SystemMemoryService;
  friend class ExternalBoundary;
  friend class SystemServiceEndpoint;
  friend class SystemTransportResource;
  friend class SystemTransferPattern;
  friend class HardwareDomainBuilder;
  friend class ServiceTransformBuilder;
  friend struct detail::SystemHandleAccess;
};

class SystemTransportEndpoint final {
public:
  SystemTransportEndpoint() = default;

private:
  SystemTransportEndpoint(const std::shared_ptr<detail::DesignState> &state,
                          std::size_t rootOrdinal,
                          loom::fabric::FabricTransportEndpointRef reference,
                          loom::fabric::FabricPortDirection direction)
      : state_(state), rootOrdinal_(rootOrdinal),
        reference_(std::move(reference)), direction_(direction) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  loom::fabric::FabricTransportEndpointRef reference_;
  loom::fabric::FabricPortDirection direction_ =
      loom::fabric::FabricPortDirection::Input;

  friend class SystemBuilder;
  friend class SystemTransportResource;
  friend class SystemServiceEndpoint;
  friend class AccCore;
  friend struct detail::SystemHandleAccess;
};

class SystemMemoryEndpoint final {
public:
  SystemMemoryEndpoint() = default;

private:
  SystemMemoryEndpoint(const std::shared_ptr<detail::DesignState> &state,
                       std::size_t rootOrdinal,
                       loom::fabric::FabricMemoryEndpointRef reference,
                       loom::fabric::FabricMemoryEndpointRole role)
      : state_(state), rootOrdinal_(rootOrdinal),
        reference_(std::move(reference)), role_(role) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  loom::fabric::FabricMemoryEndpointRef reference_;
  loom::fabric::FabricMemoryEndpointRole role_ =
      loom::fabric::FabricMemoryEndpointRole::Manager;

  friend class ServiceTransformBuilder;
  friend class SystemServiceEndpoint;
  friend class AccCore;
  friend struct detail::SystemHandleAccess;
};

class HostCore final {
public:
  HardwareDomainMember domainMember() const;

private:
  HostCore(const std::shared_ptr<detail::DesignState> &state,
           std::size_t rootOrdinal, loom::fabric::FabricEntityId entity)
      : state_(state), rootOrdinal_(rootOrdinal), entity_(entity) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  loom::fabric::FabricEntityId entity_ = 0;

  friend class SystemBuilder;
  friend struct detail::SystemHandleAccess;
};

class AccCore final {
public:
  HardwareDomainMember domainMember() const;
  HardwareDomainMember instructionCoreDomainMember() const;
  HardwareDomainMember spatialCoreDomainMember() const;

  llvm::Expected<SystemTransportEndpoint>
  spatialTransportInput(std::size_t ordinal) const;
  llvm::Expected<SystemTransportEndpoint>
  spatialTransportOutput(std::size_t ordinal) const;
  llvm::Expected<SystemMemoryEndpoint>
  spatialMemoryManager(std::size_t ordinal) const;
  llvm::Expected<SystemMemoryEndpoint>
  spatialMemorySubordinate(std::size_t ordinal) const;

private:
  AccCore(const std::shared_ptr<detail::DesignState> &state,
          std::size_t rootOrdinal, loom::fabric::FabricEntityId entity)
      : state_(state), rootOrdinal_(rootOrdinal), entity_(entity) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  loom::fabric::FabricEntityId entity_ = 0;

  friend class SystemBuilder;
  friend struct detail::SystemHandleAccess;
};

class SystemMemoryService final {
public:
  HardwareDomainMember domainMember() const;

private:
  SystemMemoryService(const std::shared_ptr<detail::DesignState> &state,
                      std::size_t rootOrdinal,
                      loom::fabric::FabricEntityId entity)
      : state_(state), rootOrdinal_(rootOrdinal), entity_(entity) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  loom::fabric::FabricEntityId entity_ = 0;

  friend class SystemBuilder;
  friend struct detail::SystemHandleAccess;
};

class ExternalBoundary final {
public:
  HardwareDomainMember domainMember() const;

private:
  ExternalBoundary(const std::shared_ptr<detail::DesignState> &state,
                   std::size_t rootOrdinal, loom::fabric::FabricEntityId entity)
      : state_(state), rootOrdinal_(rootOrdinal), entity_(entity) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  loom::fabric::FabricEntityId entity_ = 0;

  friend class SystemBuilder;
  friend struct detail::SystemHandleAccess;
};

class SystemServiceEndpoint final {
public:
  HardwareDomainMember domainMember() const;
  llvm::Expected<SystemTransportEndpoint> transport() const;
  llvm::Expected<SystemMemoryEndpoint> memory() const;

private:
  SystemServiceEndpoint(const std::shared_ptr<detail::DesignState> &state,
                        std::size_t rootOrdinal,
                        loom::fabric::FabricEntityId entity)
      : state_(state), rootOrdinal_(rootOrdinal), entity_(entity) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  loom::fabric::FabricEntityId entity_ = 0;

  friend class SystemBuilder;
  friend struct detail::SystemHandleAccess;
};

struct SystemTransportResourceSpec final {
  std::vector<PortType> inputTypes;
  std::vector<PortType> outputTypes;
  ::fabric::ResourceContract resourceContract;
};

class SystemTransportResource final {
public:
  HardwareDomainMember domainMember() const;
  llvm::Expected<SystemTransportEndpoint> input(std::size_t ordinal) const;
  llvm::Expected<SystemTransportEndpoint> output(std::size_t ordinal) const;

private:
  SystemTransportResource(const std::shared_ptr<detail::DesignState> &state,
                          std::size_t rootOrdinal,
                          loom::fabric::FabricEntityId entity)
      : state_(state), rootOrdinal_(rootOrdinal), entity_(entity) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  loom::fabric::FabricEntityId entity_ = 0;

  friend class SystemBuilder;
  friend struct detail::SystemHandleAccess;
};

class SystemTransferPattern final {
public:
  HardwareDomainMember domainMember() const;

private:
  SystemTransferPattern(const std::shared_ptr<detail::DesignState> &state,
                        std::size_t rootOrdinal,
                        loom::fabric::FabricTransferPatternRef reference)
      : state_(state), rootOrdinal_(rootOrdinal),
        reference_(std::move(reference)) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  loom::fabric::FabricTransferPatternRef reference_;

  friend class SystemBuilder;
  friend struct detail::SystemHandleAccess;
};

/// Scoped definition used where a domain's members may refer back to it.
class HardwareDomainBuilder final {
public:
  HardwareDomainMember domainMember() const;
  llvm::Error close(llvm::ArrayRef<HardwareDomainMember> members,
                    loom::fabric::HardwareDomainContract contract);

private:
  HardwareDomainBuilder(const std::shared_ptr<detail::DesignState> &state,
                        std::size_t rootOrdinal,
                        loom::fabric::FabricEntityId entity)
      : state_(state), rootOrdinal_(rootOrdinal), entity_(entity) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  loom::fabric::FabricEntityId entity_ = 0;

  friend class SystemBuilder;
  friend struct detail::SystemHandleAccess;
};

/// Scoped definition used where owned endpoints name their transform.
class ServiceTransformBuilder final {
public:
  HardwareDomainMember domainMember() const;
  llvm::Error close(llvm::ArrayRef<SystemMemoryEndpoint> inputs,
                    llvm::ArrayRef<SystemMemoryEndpoint> outputs,
                    loom::fabric::ServiceTransformContract contract);

private:
  ServiceTransformBuilder(const std::shared_ptr<detail::DesignState> &state,
                          std::size_t rootOrdinal,
                          loom::fabric::FabricEntityId entity)
      : state_(state), rootOrdinal_(rootOrdinal), entity_(entity) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  loom::fabric::FabricEntityId entity_ = 0;

  friend class SystemBuilder;
  friend struct detail::SystemHandleAccess;
};

/// Typed authoring view over one fabric.system root.
class SystemBuilder final {
public:
  llvm::Expected<ImportedSpatialCore>
  importSpatialCore(const loom::fabric::FinalizedFabricRoot &module);

  llvm::Expected<HostCore> addHostCore(
      const loom::fabric::InstructionCoreArchitecturalContract &architecture,
      const loom::fabric::InstructionCoreMicroarchitecturalRealization
          &microarchitecture);
  llvm::Expected<AccCore> addAccCore(
      const loom::fabric::InstructionCoreArchitecturalContract &architecture,
      const loom::fabric::InstructionCoreMicroarchitecturalRealization
          &microarchitecture,
      const ImportedSpatialCore &spatialCore);
  llvm::Expected<SystemMemoryService>
  addMemoryService(const ::fabric::MemoryServiceContractRecord &contract);
  llvm::Expected<ExternalBoundary> addExternalBoundary();
  llvm::Expected<HardwareDomainBuilder> createHardwareDomain();
  llvm::Expected<ServiceTransformBuilder> createServiceTransform();

  llvm::Expected<loom::fabric::ServiceRateContractRecord>
  createServiceRate(const HardwareDomainBuilder &clock,
                    std::uint64_t operationsPerWindow,
                    std::uint64_t windowTicks, std::uint64_t maxOutstanding,
                    loom::fabric::ServiceProgress progress) const;

  llvm::Expected<SystemServiceEndpoint> addServiceEndpoint(
      const HostCore &owner,
      const loom::fabric::CanonicalServiceCapabilitySet &capabilities,
      std::optional<PortType> carrier = std::nullopt);
  llvm::Expected<SystemServiceEndpoint> addServiceEndpoint(
      const AccCore &owner,
      const loom::fabric::CanonicalServiceCapabilitySet &capabilities,
      std::optional<PortType> carrier = std::nullopt);
  llvm::Expected<SystemServiceEndpoint> addServiceEndpoint(
      const SystemMemoryService &owner,
      const loom::fabric::CanonicalServiceCapabilitySet &capabilities,
      std::optional<PortType> carrier = std::nullopt);
  llvm::Expected<SystemServiceEndpoint> addServiceEndpoint(
      const ServiceTransformBuilder &owner,
      const loom::fabric::CanonicalServiceCapabilitySet &capabilities,
      std::optional<PortType> carrier = std::nullopt);
  llvm::Expected<SystemServiceEndpoint> addServiceEndpoint(
      const ExternalBoundary &owner,
      const loom::fabric::CanonicalServiceCapabilitySet &capabilities,
      std::optional<PortType> carrier = std::nullopt);

  llvm::Expected<SystemTransportResource>
  addTransportResource(const SystemTransportResourceSpec &spec);
  llvm::Expected<SystemTransferPattern>
  addTransferPattern(const SystemTransportResource &resource,
                     std::size_t inputOrdinal,
                     llvm::ArrayRef<std::uint32_t> outputOrdinals,
                     std::uint32_t usePatternOrdinal);
  llvm::Error addClockCrossing(const SystemTransportResource &resource,
                               const SystemTransferPattern &pattern,
                               const HardwareDomainBuilder &sourceClock,
                               const HardwareDomainBuilder &destinationClock,
                               std::uint32_t depth,
                               std::uint32_t synchronizerStages);
  llvm::Error connect(const SystemTransportEndpoint &source,
                      const SystemTransportEndpoint &destination);

  llvm::Error close();

private:
  SystemBuilder(const std::shared_ptr<detail::DesignState> &state,
                std::size_t rootOrdinal)
      : state_(state), rootOrdinal_(rootOrdinal) {}

  llvm::Expected<SystemServiceEndpoint> addServiceEndpoint(
      loom::fabric::FabricInventoryOwnerRef owner,
      const loom::fabric::CanonicalServiceCapabilitySet &capabilities,
      std::optional<PortType> carrier);

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;

  friend class DesignBuilder;
};

/// Immutable transient closure over finalized Fabric roots.
class FinalizedFabricDesign final {
public:
  llvm::ArrayRef<loom::fabric::FinalizedFabricRoot> roots() const {
    return roots_;
  }

private:
  explicit FinalizedFabricDesign(
      std::vector<loom::fabric::FinalizedFabricRoot> roots)
      : roots_(std::move(roots)) {}

  std::vector<loom::fabric::FinalizedFabricRoot> roots_;

  friend class DesignBuilder;
};

/// The sole owner of one typed Fabric authoring draft.
class DesignBuilder final {
public:
  explicit DesignBuilder(const loom::ArtifactStore &store);
  ~DesignBuilder();

  DesignBuilder(const DesignBuilder &) = delete;
  DesignBuilder &operator=(const DesignBuilder &) = delete;
  DesignBuilder(DesignBuilder &&) noexcept;
  DesignBuilder &operator=(DesignBuilder &&) noexcept;

  llvm::Expected<SpatialCoreBuilder>
  createSpatialCore(llvm::StringRef label, llvm::ArrayRef<PortType> inputs,
                    llvm::ArrayRef<PortType> outputs);

  llvm::Expected<SystemBuilder> createSystem(llvm::StringRef label);

  llvm::Expected<FinalizedFabricDesign> finalize() &&;

private:
  std::shared_ptr<detail::DesignState> state_;
};

} // namespace loom::adg

#endif // LOOM_ADG_BUILDER_H
