#ifndef LOOM_ADG_BUILDER_H
#define LOOM_ADG_BUILDER_H

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricHardwareDomainContracts.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/IR/FabricEnums.h"
#include "Fabric/IR/ImplementationFamily.h"
#include "Fabric/IR/MemoryConnectivityContract.h"
#include "Fabric/IR/MemoryOperationPort.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/IR/ModuleDomain.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/IR/SwitchResourceContract.h"
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
struct MeshSwitchNetworkState;
struct SystemHandleAccess;
} // namespace detail

class FuBuilder;
class FuNode;
class PeBuilder;
class ModuleDomainMemberHandle;
class HardwareDomainBuilder;
class ServiceTransformBuilder;
class SystemBuilder;
class MeshCellAttachment;
class MeshSwitchNetwork;

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

/// A move-only placeholder for one SpatialCore-local feedback edge. The
/// placeholder must be resolved to an exact SpatialValue before the root is
/// closed and never becomes part of finalized Fabric IR.
class SpatialBackedge final {
public:
  SpatialBackedge(const SpatialBackedge &) = delete;
  SpatialBackedge &operator=(const SpatialBackedge &) = delete;

  SpatialBackedge(SpatialBackedge &&other) noexcept
      : value_(std::move(other.value_)),
        placeholder_(std::exchange(other.placeholder_, nullptr)) {}
  SpatialBackedge &operator=(SpatialBackedge &&other) noexcept {
    if (this == &other)
      return *this;
    value_ = std::move(other.value_);
    placeholder_ = std::exchange(other.placeholder_, nullptr);
    return *this;
  }

  SpatialValue value() const { return value_; }

private:
  SpatialBackedge(SpatialValue value, mlir::Operation *placeholder)
      : value_(std::move(value)), placeholder_(placeholder) {}

  SpatialValue value_;
  mlir::Operation *placeholder_ = nullptr;

  friend class SpatialCoreBuilder;
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
  friend class FuNode;
};

/// An owner-checked physical node in one FU graph. Values are addressed by
/// ordered output port; the handle also names operation activation or one
/// static mux/demux route in an FU capability-template row.
class FuNode final {
public:
  FuNode() = default;

  llvm::Expected<FuValue> output(std::size_t ordinal) const;

  /// This node's own occurrence as one unified domain member handle.
  ModuleDomainMemberHandle domainMember() const;

private:
  FuNode(const std::shared_ptr<detail::DesignState> &state,
         std::size_t rootOrdinal, std::size_t peOrdinal, std::size_t fuOrdinal,
         mlir::Operation *operation)
      : state_(state), rootOrdinal_(rootOrdinal), peOrdinal_(peOrdinal),
        fuOrdinal_(fuOrdinal), operation_(operation) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  std::size_t peOrdinal_ = 0;
  std::size_t fuOrdinal_ = 0;
  mlir::Operation *operation_ = nullptr;

  friend class FuBuilder;
};

/// A move-only placeholder for one FU-local feedback edge. The placeholder
/// must be resolved to an exact FuValue before the FU is closed and never
/// becomes part of finalized Fabric IR.
class FuBackedge final {
public:
  FuBackedge(const FuBackedge &) = delete;
  FuBackedge &operator=(const FuBackedge &) = delete;

  FuBackedge(FuBackedge &&other) noexcept
      : value_(std::move(other.value_)),
        placeholder_(std::exchange(other.placeholder_, nullptr)) {}
  FuBackedge &operator=(FuBackedge &&other) noexcept {
    if (this == &other)
      return *this;
    value_ = std::move(other.value_);
    placeholder_ = std::exchange(other.placeholder_, nullptr);
    return *this;
  }

  FuValue value() const { return value_; }

private:
  FuBackedge(FuValue value, mlir::Operation *placeholder)
      : value_(std::move(value)), placeholder_(placeholder) {}

  FuValue value_;
  mlir::Operation *placeholder_ = nullptr;

  friend class FuBuilder;
};

/// One opaque, owner-checked handle for a Module's symbolic Clock or Reset
/// slot. The handle is local to its open SpatialCoreBuilder, is not a
/// persistent Fabric reference, and never binds by ordinal alone: the
/// instantiate call site supplies the exact builder context.
class ModuleDomainSlotHandle final {
public:
  ModuleDomainSlotHandle() = default;

private:
  ModuleDomainSlotHandle(const std::weak_ptr<detail::DesignState> &state,
                         std::size_t rootOrdinal,
                         loom::fabric::FabricClockResetKind kind,
                         loom::fabric::FabricOrdinal ordinal)
      : state_(state), rootOrdinal_(rootOrdinal), kind_(kind),
        ordinal_(ordinal) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  loom::fabric::FabricClockResetKind kind_ =
      loom::fabric::FabricClockResetKind::Clock;
  loom::fabric::FabricOrdinal ordinal_ = 0;

  friend class SpatialCoreBuilder;
};

/// One opaque, owner-checked authoring handle local to an open
/// SpatialCoreBuilder. It mechanically projects the closed member wire
/// `Boundary(direction, endpoint) | Internal(owner, role, subOrdinal)` owned
/// by the Fabric identity catalog and is consumed only by
/// `SpatialCoreBuilder::assignDomainSlot`.
class ModuleDomainMemberHandle final {
public:
  ModuleDomainMemberHandle() = default;

private:
  using InternalRole =
      ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole;

  static ModuleDomainMemberHandle
  boundary(const std::weak_ptr<detail::DesignState> &state,
           std::size_t rootOrdinal, loom::fabric::FabricPortDirection direction,
           loom::fabric::FabricOrdinal endpointOrdinal) {
    ModuleDomainMemberHandle handle;
    handle.state_ = state;
    handle.rootOrdinal_ = rootOrdinal;
    handle.direction_ = direction;
    handle.ordinal_ = endpointOrdinal;
    return handle;
  }

  static ModuleDomainMemberHandle
  internal(const std::weak_ptr<detail::DesignState> &state,
           std::size_t rootOrdinal, mlir::Operation *owner, InternalRole role,
           loom::fabric::FabricOrdinal subOrdinal) {
    ModuleDomainMemberHandle handle;
    handle.state_ = state;
    handle.rootOrdinal_ = rootOrdinal;
    handle.internal_ = true;
    handle.owner_ = owner;
    handle.role_ = role;
    handle.ordinal_ = subOrdinal;
    return handle;
  }

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  bool internal_ = false;
  loom::fabric::FabricPortDirection direction_ =
      loom::fabric::FabricPortDirection::Input;
  mlir::Operation *owner_ = nullptr;
  InternalRole role_ = InternalRole::Occurrence;
  loom::fabric::FabricOrdinal ordinal_ = 0;

  friend class SpatialCoreBuilder;
  friend class PeBuilder;
  friend class FuBuilder;
  friend class FuNode;
};

/// One authoring-only child-to-parent slot row for a Module instance edge.
/// Both handles must name the same Clock/Reset kind; the child handle must
/// belong to the instantiate target and the parent handle to the receiving
/// SpatialCoreBuilder.
struct ModuleInstanceDomainSlotBinding final {
  ModuleDomainSlotHandle childSlot;
  ModuleDomainSlotHandle parentSlot;
};

/// The typed result of one FIFO construction: its connectivity value plus
/// its occurrence member handle.
class FifoResult final {
public:
  SpatialValue value() const { return value_; }
  ModuleDomainMemberHandle domainMember() const { return member_; }

private:
  FifoResult(SpatialValue value, ModuleDomainMemberHandle member)
      : value_(value), member_(member) {}

  SpatialValue value_;
  ModuleDomainMemberHandle member_;

  friend class SpatialCoreBuilder;
};

/// The typed result of one construction that owns exactly one physical
/// occurrence plus its connectivity values.
class SingleOccurrenceResult final {
public:
  llvm::ArrayRef<SpatialValue> values() const { return values_; }
  SpatialValue front() const { return values_.front(); }
  SpatialValue operator[](std::size_t ordinal) const {
    return values_[ordinal];
  }
  std::size_t size() const { return values_.size(); }
  bool empty() const { return values_.empty(); }
  ModuleDomainMemberHandle domainMember() const { return member_; }

private:
  SingleOccurrenceResult(std::vector<SpatialValue> values,
                         ModuleDomainMemberHandle member)
      : values_(std::move(values)), member_(member) {}

  std::vector<SpatialValue> values_;
  ModuleDomainMemberHandle member_;

  friend class SpatialCoreBuilder;
};

using BoundaryResult = SingleOccurrenceResult;
using SwitchResult = SingleOccurrenceResult;

/// The typed result of one memory construction: its connectivity values, its
/// occurrence, every Operation Engine port, and its Local Memory Service when
/// the declaration carries one.
class MemoryResult final {
public:
  llvm::ArrayRef<SpatialValue> values() const { return values_; }
  ModuleDomainMemberHandle domainMember() const { return occurrence_; }
  llvm::Expected<ModuleDomainMemberHandle>
  operationPortMember(std::size_t ordinal) const;
  std::optional<ModuleDomainMemberHandle> localServiceMember() const {
    return localService_;
  }

private:
  MemoryResult(std::vector<SpatialValue> values,
               ModuleDomainMemberHandle occurrence,
               std::vector<ModuleDomainMemberHandle> operationPorts,
               std::optional<ModuleDomainMemberHandle> localService)
      : values_(std::move(values)), occurrence_(occurrence),
        operationPorts_(std::move(operationPorts)),
        localService_(localService) {}

  std::vector<SpatialValue> values_;
  ModuleDomainMemberHandle occurrence_;
  std::vector<ModuleDomainMemberHandle> operationPorts_;
  std::optional<ModuleDomainMemberHandle> localService_;

  friend class SpatialCoreBuilder;
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
  ::fabric::ResourceContract resourceContract;
};

/// One static route selected in a coherent FU capability-template row.
/// selectedPort is a mux input ordinal or demux output ordinal.
struct FuRouteSelection final {
  FuNode selector;
  std::uint32_t selectedPort;
};

/// One normalized semantic row in an FU's finite topology domain. Exact
/// software parameters are bound later by TechMapping and are not enumerated
/// here.
struct FuCapabilityTemplateSpec final {
  std::vector<FuNode> activeOperations;
  std::vector<FuRouteSelection> routes;
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
  static BoundarySpec s2tWithConfiguredTag(const PortType &dataInput,
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
  std::optional<::fabric::TemporalSwitchGrantPolicy> grantPolicy;

  static SwitchSpec
  spatial(std::vector<PortType> inputTypes, std::vector<PortType> outputTypes,
          std::vector<std::vector<std::uint32_t>> sourcesByOutput);

  static SwitchSpec
  temporal(std::vector<PortType> inputTypes, std::vector<PortType> outputTypes,
           std::vector<std::vector<std::uint32_t>> sourcesByOutput,
           std::uint32_t routeTableSize,
           std::optional<::fabric::TemporalSwitchGrantPolicy> grantPolicy);
};

/// The closed arbitration choice applied to helper-generated Temporal
/// switches that have physical fan-in.
enum class MeshSwitchGrantPolicyKind : std::uint8_t {
  FixedPriority,
  RoundRobin,
};

/// One ordered local ingress/egress bank placed at an authoring-only cell.
struct MeshCellAttachmentSpec final {
  std::uint32_t x;
  std::uint32_t y;
  std::vector<PortType> inputTypes;
  std::vector<PortType> outputTypes;
};

/// A validated typed recipe for one bounded rectangular switch network.
class MeshSwitchNetworkSpec final {
public:
  static llvm::Expected<MeshSwitchNetworkSpec>
  spatial(std::uint32_t width, std::uint32_t height,
          std::uint32_t lanesPerDirection, const PortType &linkType,
          std::vector<MeshCellAttachmentSpec> attachments);

  static llvm::Expected<MeshSwitchNetworkSpec>
  temporal(std::uint32_t width, std::uint32_t height,
           std::uint32_t lanesPerDirection, const PortType &linkType,
           std::uint32_t routeTableSize,
           MeshSwitchGrantPolicyKind grantPolicyKind,
           std::vector<MeshCellAttachmentSpec> attachments);

private:
  MeshSwitchNetworkSpec(
      ::fabric::Schedule schedule, std::uint32_t width, std::uint32_t height,
      std::uint32_t lanesPerDirection, PortType linkType,
      std::optional<std::uint32_t> routeTableSize,
      std::optional<MeshSwitchGrantPolicyKind> grantPolicyKind,
      std::vector<MeshCellAttachmentSpec> attachments)
      : schedule_(schedule), width_(width), height_(height),
        lanesPerDirection_(lanesPerDirection), linkType_(std::move(linkType)),
        routeTableSize_(routeTableSize), grantPolicyKind_(grantPolicyKind),
        attachments_(std::move(attachments)) {}

  ::fabric::Schedule schedule_;
  std::uint32_t width_;
  std::uint32_t height_;
  std::uint32_t lanesPerDirection_;
  PortType linkType_;
  std::optional<std::uint32_t> routeTableSize_;
  std::optional<MeshSwitchGrantPolicyKind> grantPolicyKind_;
  std::vector<MeshCellAttachmentSpec> attachments_;

  friend class SpatialCoreBuilder;
};

/// One owner-checked local bank returned by MeshSwitchNetwork authoring.
class MeshCellAttachment final {
public:
  llvm::ArrayRef<SpatialValue> inputs() const;
  llvm::Error connectOutputs(llvm::ArrayRef<SpatialValue> outputs);

private:
  MeshCellAttachment(std::shared_ptr<detail::MeshSwitchNetworkState> state,
                     std::size_t ordinal)
      : state_(std::move(state)), ordinal_(ordinal) {}

  std::shared_ptr<detail::MeshSwitchNetworkState> state_;
  std::size_t ordinal_ = 0;

  friend class MeshSwitchNetwork;
};

/// Authoring-only access to the ordered local banks of one expanded network.
class MeshSwitchNetwork final {
public:
  llvm::Expected<MeshCellAttachment> attachment(std::size_t ordinal) const;
  llvm::ArrayRef<ModuleDomainMemberHandle> domainMembers() const;
  std::size_t size() const;

private:
  explicit MeshSwitchNetwork(
      std::shared_ptr<detail::MeshSwitchNetworkState> state)
      : state_(std::move(state)) {}

  std::shared_ptr<detail::MeshSwitchNetworkState> state_;

  friend class SpatialCoreBuilder;
};

/// One exact optional fabric.mem Operation Engine declaration.
class MemoryEngineSpec final {
public:
  static MemoryEngineSpec
  spatial(std::vector<::fabric::MemoryOperationPortDeclaration> operationPorts);
  static MemoryEngineSpec temporal(
      std::uint64_t residentContextCount,
      std::vector<::fabric::MemoryOperationPortDeclaration> operationPorts);

private:
  MemoryEngineSpec(
      ::fabric::Schedule schedule,
      std::optional<std::uint64_t> residentContextCount,
      std::vector<::fabric::MemoryOperationPortDeclaration> operationPorts)
      : schedule_(schedule), residentContextCount_(residentContextCount),
        operationPorts_(std::move(operationPorts)) {}

  ::fabric::Schedule schedule_;
  std::optional<std::uint64_t> residentContextCount_;
  std::vector<::fabric::MemoryOperationPortDeclaration> operationPorts_;

  friend class MemorySpec;
  friend class SpatialCoreBuilder;
};

/// One exact occurrence-level memory dispatch and internal-connectivity
/// contract in its canonical owner wire.
class MemoryConnectivitySpec final {
public:
  static llvm::Expected<MemoryConnectivitySpec>
  create(::fabric::MemoryConnectivityDeclaration declaration);

private:
  explicit MemoryConnectivitySpec(std::vector<std::uint8_t> canonicalBytes)
      : canonicalBytes_(std::move(canonicalBytes)) {}

  std::vector<std::uint8_t> canonicalBytes_;

  friend class MemorySpec;
  friend class SpatialCoreBuilder;
};

/// One exact optional fabric.mem Local Memory Service declaration.
class LocalMemoryServiceSpec final {
public:
  static llvm::Expected<LocalMemoryServiceSpec>
  create(std::uint64_t capacityBytes,
         const ::fabric::MemoryServiceContractRecord &contract);

private:
  LocalMemoryServiceSpec(std::uint64_t capacityBytes,
                         std::vector<std::uint8_t> contractBytes)
      : capacityBytes_(capacityBytes),
        contractBytes_(std::move(contractBytes)) {}

  std::uint64_t capacityBytes_;
  std::vector<std::uint8_t> contractBytes_;

  friend class MemorySpec;
  friend class SpatialCoreBuilder;
};

/// One fabric.mem declaration composed from its two orthogonal resources.
class MemorySpec final {
public:
  static llvm::Expected<MemorySpec>
  create(std::vector<PortType> inputTypes, std::vector<PortType> outputTypes,
         std::vector<std::uint32_t> managerInputOrdinals,
         std::vector<std::uint32_t> subordinateOutputOrdinals,
         std::optional<MemoryEngineSpec> engine,
         std::optional<LocalMemoryServiceSpec> localService,
         MemoryConnectivitySpec connectivity);

  llvm::ArrayRef<PortType> inputTypes() const { return inputTypes_; }
  llvm::ArrayRef<PortType> outputTypes() const { return outputTypes_; }

private:
  MemorySpec(std::vector<PortType> inputTypes,
             std::vector<PortType> outputTypes,
             std::vector<std::uint32_t> managerInputOrdinals,
             std::vector<std::uint32_t> subordinateOutputOrdinals,
             std::optional<MemoryEngineSpec> engine,
             std::optional<LocalMemoryServiceSpec> localService,
             MemoryConnectivitySpec connectivity)
      : inputTypes_(std::move(inputTypes)),
        outputTypes_(std::move(outputTypes)),
        managerInputOrdinals_(std::move(managerInputOrdinals)),
        subordinateOutputOrdinals_(std::move(subordinateOutputOrdinals)),
        engine_(std::move(engine)), localService_(std::move(localService)),
        connectivity_(std::move(connectivity)) {}

  std::vector<PortType> inputTypes_;
  std::vector<PortType> outputTypes_;
  std::vector<std::uint32_t> managerInputOrdinals_;
  std::vector<std::uint32_t> subordinateOutputOrdinals_;
  std::optional<MemoryEngineSpec> engine_;
  std::optional<LocalMemoryServiceSpec> localService_;
  MemoryConnectivitySpec connectivity_;

  friend class SpatialCoreBuilder;
};

class FuBuilder final {
public:
  llvm::Expected<FuValue> input(std::size_t ordinal) const;

  /// The FU occurrence as one unified domain member handle.
  ModuleDomainMemberHandle domainMember() const;

  llvm::Expected<FuBackedge> createBackedge(const PortType &type);

  llvm::Error resolveBackedge(FuBackedge &&backedge, FuValue source);

  llvm::Expected<FuNode> addOperation(llvm::ArrayRef<FuValue> inputs,
                                      const OperationCapabilitySpec &spec);

  llvm::Expected<FuNode> addMux(llvm::ArrayRef<FuValue> inputs);

  llvm::Expected<FuNode> addDemux(FuValue input, std::uint32_t outputCount);

  llvm::Error addCapabilityTemplate(const FuCapabilityTemplateSpec &spec);

  llvm::Error close(llvm::ArrayRef<FuValue> outputs);

private:
  llvm::Expected<mlir::Value>
  resolveValue(const std::shared_ptr<detail::DesignState> &state,
               const FuValue &value) const;

  llvm::Expected<mlir::Operation *>
  resolveNode(const std::shared_ptr<detail::DesignState> &state,
              const FuNode &node) const;

  FuBuilder(const std::shared_ptr<detail::DesignState> &state,
            std::size_t rootOrdinal, std::size_t peOrdinal,
            std::size_t fuOrdinal, mlir::Operation *operation)
      : state_(state), rootOrdinal_(rootOrdinal), peOrdinal_(peOrdinal),
        fuOrdinal_(fuOrdinal), operation_(operation) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_;
  std::size_t peOrdinal_;
  std::size_t fuOrdinal_;
  mlir::Operation *operation_ = nullptr;

  friend class PeBuilder;
};

class PeBuilder final {
public:
  llvm::Expected<PeValue> input(std::size_t ordinal) const;
  llvm::Expected<SpatialValue> output(std::size_t ordinal) const;

  /// The PE occurrence as one unified domain member handle.
  ModuleDomainMemberHandle domainMember() const;

  /// One resident instruction context as a domain member handle. A spatial
  /// PE admits only ordinal zero; a temporal PE admits exactly its
  /// instruction-capacity range.
  llvm::Expected<ModuleDomainMemberHandle>
  instructionContextMember(std::size_t ordinal) const;

  llvm::Expected<FuBuilder> addFu(llvm::ArrayRef<PeValue> inputs,
                                  const FuSpec &spec);

  llvm::Error close();

private:
  llvm::Expected<mlir::Value>
  resolveValue(const std::shared_ptr<detail::DesignState> &state,
               const PeValue &value) const;

  PeBuilder(const std::shared_ptr<detail::DesignState> &state,
            std::size_t rootOrdinal, std::size_t peOrdinal,
            mlir::Operation *operation, std::size_t instructionContexts)
      : state_(state), rootOrdinal_(rootOrdinal), peOrdinal_(peOrdinal),
        operation_(operation), instructionContexts_(instructionContexts) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_;
  std::size_t peOrdinal_;
  mlir::Operation *operation_ = nullptr;
  std::size_t instructionContexts_ = 0;

  friend class SpatialCoreBuilder;
};

class SpatialCoreBuilder final {
public:
  llvm::Expected<SpatialValue> input(std::size_t ordinal) const;

  llvm::Expected<SpatialBackedge> createBackedge(const PortType &type);

  llvm::Error resolveBackedge(SpatialBackedge &&backedge, SpatialValue source);

  /// Declares one symbolic Clock or Reset slot on this Module and returns
  /// its opaque authoring handle.
  llvm::Expected<ModuleDomainSlotHandle>
  declareDomainSlot(loom::fabric::FabricClockResetKind kind);

  /// Returns the closed Module's effective slot inventory for one kind.
  /// Omitted single-domain authoring is materialized by close(), so its
  /// Clock and Reset slots are addressable for explicit instance bindings.
  llvm::Expected<std::vector<ModuleDomainSlotHandle>>
  domainSlots(loom::fabric::FabricClockResetKind kind) const;

  /// Selects one Module boundary face directly as a domain member handle.
  llvm::Expected<ModuleDomainMemberHandle>
  inputDomainMember(std::size_t ordinal) const;
  llvm::Expected<ModuleDomainMemberHandle>
  outputDomainMember(std::size_t ordinal) const;

  /// Authors one row of the Module-owned domain_assignments relation. The
  /// member and slot must belong to this same open SpatialCore.
  llvm::Error assignDomainSlot(const ModuleDomainMemberHandle &member,
                               const ModuleDomainSlotHandle &slot);

  /// Instantiates one closed SpatialCore from this design as a module
  /// template. Fabric finalization expands the instance into fresh physical
  /// occurrences. domainBindings is the exact total child-to-parent slot
  /// correspondence and must contain every effective child slot.
  llvm::Expected<std::vector<SpatialValue>>
  instantiate(const SpatialCoreBuilder &target,
              llvm::ArrayRef<SpatialValue> inputs,
              llvm::ArrayRef<ModuleInstanceDomainSlotBinding> domainBindings);

  llvm::Expected<FifoResult> addFifo(SpatialValue input, const FifoSpec &spec);

  llvm::Expected<BoundaryResult>
  addBoundary(llvm::ArrayRef<SpatialValue> inputs, const BoundarySpec &spec);

  llvm::Expected<SwitchResult> addSwitch(llvm::ArrayRef<SpatialValue> inputs,
                                         const SwitchSpec &spec);

  llvm::Expected<MeshSwitchNetwork>
  addMeshSwitchNetwork(const MeshSwitchNetworkSpec &spec);

  llvm::Expected<MemoryResult> addMemory(llvm::ArrayRef<SpatialValue> inputs,
                                         const MemorySpec &spec);

  llvm::Expected<PeBuilder> addPe(llvm::ArrayRef<SpatialValue> inputs,
                                  const PeSpec &spec);

  /// Clones or removes one exact finalized occurrence in this fresh derived
  /// draft. The existing Fabric physical-owner union is the typed selector;
  /// definition nodes and nested inventory members are rejected.
  llvm::Error
  cloneOccurrence(const loom::fabric::FabricModulePhysicalOwnerRef &prototype);
  llvm::Error
  eraseOccurrence(const loom::fabric::FabricModulePhysicalOwnerRef &target);

  /// Replaces one or several exact point connections. Every destination must
  /// already be connected in the finalized parent and may occur only once in
  /// the replacement set.
  llvm::Error replacePointConnection(
      const loom::fabric::FabricTransportEndpointRef &destination,
      const loom::fabric::FabricTransportEndpointRef &source);
  llvm::Error replaceParallelConnections(
      llvm::ArrayRef<loom::fabric::FabricPointConnectionPayload> connections);

  /// Replaces the Module signature with a prefix-preserving input inventory
  /// and exact physical output sources. This operation never invents domain
  /// assignments: boundary growth is rejected.
  llvm::Error changeBoundaryInventory(
      std::size_t inputCount,
      llvm::ArrayRef<loom::fabric::FabricTransportEndpointRef> outputSources);

  llvm::Error replacePeKind(loom::fabric::FabricPeOccurrenceRef target,
                            loom::fabric::FabricPeOccurrenceRef prototype);
  llvm::Error resizeInstructionStore(loom::fabric::FabricPeOccurrenceRef target,
                                     std::uint32_t instructionCapacity);
  llvm::Error replaceFuInventory(
      loom::fabric::FabricPeOccurrenceRef target,
      llvm::ArrayRef<loom::fabric::FabricFuOccurrenceRef> prototypes);
  llvm::Error
  replaceFuCapability(loom::fabric::FabricFuOccurrenceRef target,
                      loom::fabric::FabricFuOccurrenceRef prototype);
  llvm::Error replaceSwitchModeOrScheduleCapacity(
      loom::fabric::FabricSwitchOccurrenceRef target,
      loom::fabric::FabricSwitchOccurrenceRef prototype);
  llvm::Error resizeMemory(loom::fabric::FabricMemoryOccurrenceRef target,
                           std::uint64_t capacityBytes);
  llvm::Error replaceMemoryOperationTable(
      loom::fabric::FabricMemoryOccurrenceRef target,
      loom::fabric::FabricMemoryOccurrenceRef prototype);
  llvm::Error resizeFifo(loom::fabric::FabricFifoOccurrenceRef target,
                         std::uint32_t depth);
  llvm::Error
  changeFifoBypassCapability(loom::fabric::FabricFifoOccurrenceRef target,
                             bool bypassable);

  /// Closes a root returned by DesignBuilder::deriveSpatialCore with the
  /// preserved or explicitly replaced output sequence.
  llvm::Error closeDerived();

  /// Closes this root with the exact declared result sequence. When domain
  /// authoring is active, every member of the complete boundary and internal
  /// inventory must carry exactly one Clock and one Reset assignment before
  /// any output mutation is published.
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
  friend class MeshCellAttachment;
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

  friend class SystemBuilder;
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
  loom::fabric::SystemMemoryServiceRef reference() const;
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
  loom::fabric::HardwareDomainRef reference() const;
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
  llvm::Expected<HostCore> hostCore(std::size_t ordinal) const;

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
  llvm::Error attachSpatialMemory(const SystemMemoryEndpoint &spatialEndpoint,
                                  const SystemServiceEndpoint &serviceEndpoint);
  llvm::Error
  attachServiceLegCarriers(const SystemMemoryEndpoint &endpoint,
                           dataflow::semantics::ServiceKind kind,
                           dataflow::StructuralOrdinal legOrdinal,
                           llvm::ArrayRef<SystemTransportEndpoint> carriers);
  llvm::Error addClockCrossing(const SystemTransportResource &resource,
                               const SystemTransferPattern &pattern,
                               const HardwareDomainBuilder &sourceClock,
                               const HardwareDomainBuilder &destinationClock,
                               std::uint32_t depth,
                               std::uint32_t synchronizerStages);
  llvm::Error connect(const SystemTransportEndpoint &source,
                      const SystemTransportEndpoint &destination);
  llvm::Error connect(const SystemMemoryEndpoint &manager,
                      const SystemMemoryEndpoint &subordinate);

  llvm::Expected<AccCore>
  addAccCoreFromPrototype(loom::fabric::AccCoreOccurrenceRef prototype,
                          const loom::fabric::FinalizedFabricRoot &spatialCore);
  llvm::Error removeAccCore(loom::fabric::AccCoreOccurrenceRef target);
  llvm::Error replaceSpatialAttachment(
      loom::fabric::AccCoreOccurrenceRef target,
      const loom::fabric::FinalizedFabricRoot &spatialCore);
  llvm::Error selectInstructionCoreRealization(
      loom::fabric::InstructionCoreContextRef target,
      loom::fabric::InstructionCoreContextRef prototype);
  llvm::Error
  replaceTransportResource(loom::fabric::SystemTransportResourceRef target,
                           loom::fabric::SystemTransportResourceRef prototype);
  llvm::Error replaceTransportConnection(
      const loom::fabric::FabricTransportEndpointRef &destination,
      const loom::fabric::FabricTransportEndpointRef &source);
  llvm::Error replaceSpatialMemoryAttachment(
      const loom::fabric::FabricMemoryEndpointRef &spatialEndpoint,
      loom::fabric::SystemServiceEndpointRef serviceEndpoint);
  llvm::Error replaceMemoryServiceConnection(
      const loom::fabric::FabricMemoryEndpointRef &destination,
      const loom::fabric::FabricMemoryEndpointRef &source);

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

  /// Rebuilds one exact finalized root as a fresh ordinary Builder draft. The
  /// parent remains immutable and every result still passes through
  /// DesignBuilder::finalize().
  llvm::Expected<SpatialCoreBuilder>
  deriveSpatialCore(const loom::fabric::FinalizedFabricRoot &parent);

  llvm::Expected<SystemBuilder> deriveSystem(
      const loom::fabric::FinalizedFabricRoot &parent,
      llvm::ArrayRef<loom::fabric::FinalizedFabricRoot> admissibleModules);

  llvm::Expected<FinalizedFabricDesign> finalize() &&;

private:
  std::shared_ptr<detail::DesignState> state_;
};

} // namespace loom::adg

#endif // LOOM_ADG_BUILDER_H
