#ifndef LOOM_DSE_HARDWAREDECISION_H
#define LOOM_DSE_HARDWAREDECISION_H

#include "Common/Artifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/IR/FabricEnums.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace loom::dse {

struct AddOccurrence final {
  loom::fabric::FabricModulePhysicalOwnerRef prototype;
};

struct RemoveOccurrence final {
  loom::fabric::FabricModulePhysicalOwnerRef target;
};

struct ReplacePointConnection final {
  loom::fabric::FabricTransportEndpointRef destination;
  loom::fabric::FabricTransportEndpointRef source;
};

struct AdjustParallelConnectionCount final {
  std::vector<loom::fabric::FabricPointConnectionPayload> connections;
};

struct BoundaryInventoryValue final {
  std::uint64_t inputCount = 0;
  std::vector<loom::fabric::FabricTransportEndpointRef> outputSources;
};

struct ChangeBoundaryInventory final {
  BoundaryInventoryValue value;
};

using SpatialTopologyDecision =
    std::variant<AddOccurrence, RemoveOccurrence, ReplacePointConnection,
                 AdjustParallelConnectionCount, ChangeBoundaryInventory>;

struct AddOccurrenceDomain final {
  std::vector<loom::fabric::FabricModulePhysicalOwnerRef> prototypes;
};

struct RemoveOccurrenceDomain final {
  std::vector<loom::fabric::FabricModulePhysicalOwnerRef> targets;
};

struct ReplacePointConnectionDomain final {
  loom::fabric::FabricTransportEndpointRef destination;
  std::vector<loom::fabric::FabricTransportEndpointRef> sources;
};

struct AdjustParallelConnectionCountDomain final {
  std::vector<std::vector<loom::fabric::FabricPointConnectionPayload>> values;
};

struct ChangeBoundaryInventoryDomain final {
  std::vector<BoundaryInventoryValue> values;
};

using SpatialTopologyDecisionDomain = std::variant<
    AddOccurrenceDomain, RemoveOccurrenceDomain, ReplacePointConnectionDomain,
    AdjustParallelConnectionCountDomain, ChangeBoundaryInventoryDomain>;

struct ChangePeKind final {
  loom::fabric::FabricPeOccurrenceRef target;
  loom::fabric::FabricPeOccurrenceRef prototype;
};

struct ResizeInstructionStore final {
  loom::fabric::FabricPeOccurrenceRef target;
  std::uint32_t instructionCapacity = 0;
};

struct ResizeInstructionStores final {
  std::vector<ResizeInstructionStore> stores;
};

struct ChangeFuInventory final {
  loom::fabric::FabricPeOccurrenceRef target;
  std::vector<loom::fabric::FabricFuOccurrenceRef> prototypes;
};

struct ChangeFuCapability final {
  loom::fabric::FabricFuOccurrenceRef target;
  loom::fabric::FabricFuOccurrenceRef prototype;
};

struct ChangeSwitchModeOrScheduleCapacity final {
  loom::fabric::FabricSwitchOccurrenceRef target;
  loom::fabric::FabricSwitchOccurrenceRef prototype;
};

struct ResizeSwitchRouteTable final {
  loom::fabric::FabricSwitchOccurrenceRef target;
  std::uint32_t entries = 0;
};

struct ResizeMemory final {
  loom::fabric::FabricMemoryOccurrenceRef target;
  std::uint64_t capacityBytes = 0;
};

struct ChangeMemoryOperationTable final {
  loom::fabric::FabricMemoryOccurrenceRef target;
  loom::fabric::FabricMemoryOccurrenceRef prototype;
};

struct ResizeFifo final {
  loom::fabric::FabricFifoOccurrenceRef target;
  std::uint32_t depth = 0;
};

struct ChangeFifoBypassCapability final {
  loom::fabric::FabricFifoOccurrenceRef target;
  bool bypassable = false;
};

struct ChangeFifoQueueDiscipline final {
  loom::fabric::FabricFifoOccurrenceRef target;
  ::fabric::FifoQueueDiscipline discipline =
      ::fabric::FifoQueueDiscipline::StrictFifo;
};

struct ChangeTemporalOperandBufferMode final {
  loom::fabric::FabricPeOccurrenceRef target;
  ::fabric::OperandBufferMode mode =
      ::fabric::OperandBufferMode::PerInstruction;
};

struct ResizeTemporalOperandBuffer final {
  loom::fabric::FabricPeOccurrenceRef target;
  std::uint32_t entriesPerAllocationUnit = 0;
};

using SpatialMicroarchitectureDecision =
    std::variant<ChangePeKind, ResizeInstructionStore, ChangeFuInventory,
                 ChangeFuCapability, ChangeSwitchModeOrScheduleCapacity,
                 ResizeMemory, ChangeMemoryOperationTable, ResizeFifo,
                 ChangeFifoBypassCapability, ResizeInstructionStores,
                 ChangeTemporalOperandBufferMode, ResizeTemporalOperandBuffer,
                 ResizeSwitchRouteTable, ChangeFifoQueueDiscipline>;

struct ChangePeKindDomain final {
  loom::fabric::FabricPeOccurrenceRef target;
  std::vector<loom::fabric::FabricPeOccurrenceRef> prototypes;
};

struct ResizeInstructionStoreDomain final {
  loom::fabric::FabricPeOccurrenceRef target;
  std::vector<std::uint32_t> capacities;
};

struct ResizeInstructionStoresDomain final {
  std::vector<ResizeInstructionStore> stores;
};

struct ChangeFuInventoryDomain final {
  loom::fabric::FabricPeOccurrenceRef target;
  std::vector<std::vector<loom::fabric::FabricFuOccurrenceRef>> values;
};

struct ChangeFuCapabilityDomain final {
  loom::fabric::FabricFuOccurrenceRef target;
  std::vector<loom::fabric::FabricFuOccurrenceRef> prototypes;
};

struct ChangeSwitchModeOrScheduleCapacityDomain final {
  loom::fabric::FabricSwitchOccurrenceRef target;
  std::vector<loom::fabric::FabricSwitchOccurrenceRef> prototypes;
};

struct ResizeSwitchRouteTableDomain final {
  loom::fabric::FabricSwitchOccurrenceRef target;
  std::vector<std::uint32_t> entries;
};

struct ResizeMemoryDomain final {
  loom::fabric::FabricMemoryOccurrenceRef target;
  std::vector<std::uint64_t> capacitiesBytes;
};

struct ChangeMemoryOperationTableDomain final {
  loom::fabric::FabricMemoryOccurrenceRef target;
  std::vector<loom::fabric::FabricMemoryOccurrenceRef> prototypes;
};

struct ResizeFifoDomain final {
  loom::fabric::FabricFifoOccurrenceRef target;
  std::vector<std::uint32_t> depths;
};

struct ChangeFifoBypassCapabilityDomain final {
  loom::fabric::FabricFifoOccurrenceRef target;
  std::vector<bool> values;
};

struct ChangeFifoQueueDisciplineDomain final {
  loom::fabric::FabricFifoOccurrenceRef target;
  std::vector<::fabric::FifoQueueDiscipline> disciplines;
};

struct ChangeTemporalOperandBufferModeDomain final {
  loom::fabric::FabricPeOccurrenceRef target;
  std::vector<::fabric::OperandBufferMode> modes;
};

struct ResizeTemporalOperandBufferDomain final {
  loom::fabric::FabricPeOccurrenceRef target;
  std::vector<std::uint32_t> entriesPerAllocationUnit;
};

using SpatialMicroarchitectureDecisionDomain = std::variant<
    ChangePeKindDomain, ResizeInstructionStoreDomain, ChangeFuInventoryDomain,
    ChangeFuCapabilityDomain, ChangeSwitchModeOrScheduleCapacityDomain,
    ResizeMemoryDomain, ChangeMemoryOperationTableDomain, ResizeFifoDomain,
    ChangeFifoBypassCapabilityDomain, ResizeInstructionStoresDomain,
    ChangeTemporalOperandBufferModeDomain, ResizeTemporalOperandBufferDomain,
    ResizeSwitchRouteTableDomain, ChangeFifoQueueDisciplineDomain>;

struct AddAccCore final {
  loom::fabric::AccCoreOccurrenceRef prototype;
  ArtifactRootReference module;
};

struct RemoveAccCore final {
  loom::fabric::AccCoreOccurrenceRef target;
};

struct ReplaceSpatialAttachment final {
  loom::fabric::AccCoreOccurrenceRef target;
  ArtifactRootReference module;
};

struct SelectInstructionCoreRealization final {
  loom::fabric::InstructionCoreContextRef target;
  loom::fabric::InstructionCoreContextRef prototype;
};

struct ChangeTransportResource final {
  loom::fabric::SystemTransportResourceRef target;
  loom::fabric::SystemTransportResourceRef prototype;
};

struct ChangeTransportConnection final {
  loom::fabric::FabricTransportEndpointRef destination;
  loom::fabric::FabricTransportEndpointRef source;
};

struct SwapTransportConnectionSources final {
  loom::fabric::FabricTransportEndpointRef firstDestination;
  loom::fabric::FabricTransportEndpointRef secondDestination;
};

struct ChangeSpatialMemoryAttachment final {
  loom::fabric::FabricMemoryEndpointRef spatialEndpoint;
  loom::fabric::SystemServiceEndpointRef serviceEndpoint;
};

struct ChangeMemoryServiceConnection final {
  loom::fabric::FabricMemoryEndpointRef destination;
  loom::fabric::FabricMemoryEndpointRef source;
};

using ServiceOrMemoryAttachmentDecision =
    std::variant<ChangeSpatialMemoryAttachment, ChangeMemoryServiceConnection>;

struct ChangeServiceOrMemoryAttachment final {
  ServiceOrMemoryAttachmentDecision value;
};

struct ResizeSystemMemoryRegion final {
  loom::fabric::SystemMemoryServiceRef service;
  loom::fabric::FabricOrdinal regionOrdinal = 0;
  std::uint64_t sizeBytes = 0;
};

using SystemCompositionDecision =
    std::variant<AddAccCore, RemoveAccCore, ReplaceSpatialAttachment,
                 SelectInstructionCoreRealization, ChangeTransportResource,
                 ChangeTransportConnection, ChangeServiceOrMemoryAttachment,
                 SwapTransportConnectionSources, ResizeSystemMemoryRegion>;

struct AddAccCoreDomain final {
  loom::fabric::AccCoreOccurrenceRef prototype;
  std::vector<ArtifactRootReference> modules;
};

struct RemoveAccCoreDomain final {
  std::vector<loom::fabric::AccCoreOccurrenceRef> targets;
};

struct ReplaceSpatialAttachmentDomain final {
  loom::fabric::AccCoreOccurrenceRef target;
  std::vector<ArtifactRootReference> modules;
};

struct SelectInstructionCoreRealizationDomain final {
  loom::fabric::InstructionCoreContextRef target;
  std::vector<loom::fabric::InstructionCoreContextRef> prototypes;
};

struct ChangeTransportResourceDomain final {
  loom::fabric::SystemTransportResourceRef target;
  std::vector<loom::fabric::SystemTransportResourceRef> prototypes;
};

struct ChangeTransportConnectionDomain final {
  loom::fabric::FabricTransportEndpointRef destination;
  std::vector<loom::fabric::FabricTransportEndpointRef> sources;
};

struct SwapTransportConnectionSourcesDomain final {
  loom::fabric::FabricTransportEndpointRef firstDestination;
  std::vector<loom::fabric::FabricTransportEndpointRef> secondDestinations;
};

struct ChangeSpatialMemoryAttachmentDomain final {
  loom::fabric::FabricMemoryEndpointRef spatialEndpoint;
  std::vector<loom::fabric::SystemServiceEndpointRef> serviceEndpoints;
};

struct ChangeMemoryServiceConnectionDomain final {
  loom::fabric::FabricMemoryEndpointRef destination;
  std::vector<loom::fabric::FabricMemoryEndpointRef> sources;
};

struct ResizeSystemMemoryRegionDomain final {
  loom::fabric::SystemMemoryServiceRef service;
  loom::fabric::FabricOrdinal regionOrdinal = 0;
  std::vector<std::uint64_t> sizesBytes;
};

using ChangeServiceOrMemoryAttachmentDomain =
    std::variant<ChangeSpatialMemoryAttachmentDomain,
                 ChangeMemoryServiceConnectionDomain>;

using SystemCompositionDecisionDomain = std::variant<
    AddAccCoreDomain, RemoveAccCoreDomain, ReplaceSpatialAttachmentDomain,
    SelectInstructionCoreRealizationDomain, ChangeTransportResourceDomain,
    ChangeTransportConnectionDomain, ChangeServiceOrMemoryAttachmentDomain,
    SwapTransportConnectionSourcesDomain, ResizeSystemMemoryRegionDomain>;

llvm::Expected<std::vector<SpatialTopologyDecision>>
expandSpatialTopologyDecisionDomains(
    llvm::ArrayRef<SpatialTopologyDecisionDomain> domains);

llvm::Expected<std::vector<SpatialMicroarchitectureDecision>>
expandSpatialMicroarchitectureDecisionDomains(
    llvm::ArrayRef<SpatialMicroarchitectureDecisionDomain> domains);

/// Canonical global-depth comparison for a shared-pool capacity suggestion.
/// The domain contains depth one, depth two, the sufficient bound, and one
/// deeper control, with semantic duplicates removed.
llvm::Expected<ResizeFifoDomain>
deriveFifoCapacityDepthDomain(loom::fabric::FabricFifoOccurrenceRef owner,
                              std::uint64_t selectedCapacity,
                              std::uint64_t sufficientCapacity);

llvm::Expected<std::vector<SystemCompositionDecision>>
expandSystemCompositionDecisionDomains(
    llvm::ArrayRef<SystemCompositionDecisionDomain> domains);

std::vector<std::uint8_t> encodeSpatialTopologyRewriteConfig(
    llvm::ArrayRef<SpatialTopologyDecision> decisions,
    std::uint64_t maxChildrenPerParent);
std::vector<std::uint8_t> encodeSpatialMicroarchitectureRewriteConfig(
    llvm::ArrayRef<SpatialMicroarchitectureDecision> decisions,
    std::uint64_t maxChildrenPerParent);
std::vector<std::uint8_t> encodeSystemCompositionRewriteConfig(
    llvm::ArrayRef<SystemCompositionDecision> decisions,
    std::uint64_t maxChildrenPerParent);

llvm::Expected<std::pair<std::vector<SpatialTopologyDecision>, std::uint64_t>>
adoptSpatialTopologyRewriteConfig(llvm::ArrayRef<std::uint8_t> bytes);
llvm::Expected<
    std::pair<std::vector<SpatialMicroarchitectureDecision>, std::uint64_t>>
adoptSpatialMicroarchitectureRewriteConfig(llvm::ArrayRef<std::uint8_t> bytes);
llvm::Expected<std::pair<std::vector<SystemCompositionDecision>, std::uint64_t>>
adoptSystemCompositionRewriteConfig(llvm::ArrayRef<std::uint8_t> bytes);

std::vector<std::uint8_t>
encodeSpatialTopologyDecision(const ArtifactRootReference &parent,
                              const SpatialTopologyDecision &decision);
std::vector<std::uint8_t> encodeSpatialMicroarchitectureDecision(
    const ArtifactRootReference &parent,
    const SpatialMicroarchitectureDecision &decision,
    llvm::ArrayRef<loom::fabric::FabricModuleEntityCorrespondence> entities =
        {});
std::vector<std::uint8_t> encodeSystemCompositionDecision(
    const ArtifactRootReference &parent,
    const SystemCompositionDecision &decision,
    llvm::ArrayRef<loom::fabric::FabricSystemEntityCorrespondence> entities,
    llvm::ArrayRef<loom::fabric::FabricSystemTransferPatternCorrespondence>
        transferPatterns);

struct SpatialTopologyCandidateDecision final {
  ArtifactRootReference parent;
  SpatialTopologyDecision decision;
};

struct SpatialMicroarchitectureCandidateDecision final {
  ArtifactRootReference parent;
  SpatialMicroarchitectureDecision decision;
  std::vector<loom::fabric::FabricModuleEntityCorrespondence> entities;
};

struct SystemCompositionCandidateDecision final {
  ArtifactRootReference parent;
  SystemCompositionDecision decision;
  std::vector<loom::fabric::FabricSystemEntityCorrespondence> entities;
  std::vector<loom::fabric::FabricSystemTransferPatternCorrespondence>
      transferPatterns;
};

/// Hardware transformation lineage is the sole owner of the Mapping layers
/// that can be retained across one child. `Rebase` means the selections are
/// semantically unchanged but must be reconstructed against the child owner;
/// `Reopen` means only the dependency cone rooted below may be repaired.
enum class HardwareMappingImpactKind : std::uint8_t {
  Unchanged,
  Rebase,
  Reopen,
};

llvm::StringRef
hardwareMappingImpactKindSpelling(HardwareMappingImpactKind kind);

struct TechMappingImpactProjection final {
  HardwareMappingImpactKind kind = HardwareMappingImpactKind::Unchanged;
  std::vector<loom::fabric::FabricModulePhysicalOwnerRef> realizationRoots;
};

struct SpatialMappingImpactProjection final {
  HardwareMappingImpactKind kind = HardwareMappingImpactKind::Unchanged;
  std::vector<loom::fabric::FabricModulePhysicalOwnerRef> placementRoots;
  std::vector<loom::fabric::FabricTransportEndpointRef> routeRoots;
};

struct SystemMappingImpactProjection final {
  HardwareMappingImpactKind kind = HardwareMappingImpactKind::Unchanged;
  std::vector<loom::fabric::AccCoreOccurrenceRef> executionRoots;
  std::vector<loom::fabric::InstructionCoreContextRef> instructionContextRoots;
  std::vector<loom::fabric::SystemTransportResourceRef> transportRoots;
  std::vector<loom::fabric::FabricTransportEndpointRef> routeRoots;
  std::vector<loom::fabric::SystemServiceEndpointRef> serviceRoots;
  std::vector<loom::fabric::SystemMemoryServiceRef> memoryServiceRoots;
  std::vector<loom::fabric::FabricMemoryEndpointRef> memoryRoots;
};

/// Stable mutation classes used by the hardware feedback matrix. A class is
/// an accounting label, not a second candidate identity or a promise that a
/// particular provider can already materialize every member.
enum class HardwareMutationFamily : std::uint8_t {
  SpatialTopology,
  InstructionCapacity,
  FuCapability,
  SpatialMemory,
  SpatialFifo,
  TemporalOperandBuffer,
  SpatialSwitch,
  SystemSpatialCoreAttachment,
  SystemAccCore,
  SystemInstructionContext,
  SystemTransport,
  SystemMemoryService,
};

llvm::StringRef hardwareMutationFamilySpelling(HardwareMutationFamily family);

enum class HardwareMutationLocality : std::uint8_t {
  Unchanged,
  LocalCone,
  GlobalReopen,
};

llvm::StringRef
hardwareMutationLocalitySpelling(HardwareMutationLocality locality);

struct HardwareImpactProjection final {
  ArtifactRootReference parent;
  std::optional<ArtifactRootReference> child;
  std::vector<loom::fabric::FabricModuleEntityCorrespondence> moduleEntities;
  TechMappingImpactProjection tech;
  SpatialMappingImpactProjection spatial;
  SystemMappingImpactProjection system;
  HardwareMutationFamily family = HardwareMutationFamily::SpatialTopology;
  HardwareMutationLocality locality = HardwareMutationLocality::Unchanged;
};

HardwareImpactProjection projectHardwareImpact(
    const SpatialTopologyCandidateDecision &decision,
    std::optional<ArtifactRootReference> child = std::nullopt);
HardwareImpactProjection projectHardwareImpact(
    const SpatialMicroarchitectureCandidateDecision &decision,
    std::optional<ArtifactRootReference> child = std::nullopt);
HardwareImpactProjection projectHardwareImpact(
    const SystemCompositionCandidateDecision &decision,
    std::optional<ArtifactRootReference> child = std::nullopt);

llvm::Expected<SpatialTopologyCandidateDecision>
adoptSpatialTopologyDecision(llvm::ArrayRef<std::uint8_t> bytes);
llvm::Expected<SpatialMicroarchitectureCandidateDecision>
adoptSpatialMicroarchitectureDecision(llvm::ArrayRef<std::uint8_t> bytes);
llvm::Expected<SystemCompositionCandidateDecision>
adoptSystemCompositionDecision(llvm::ArrayRef<std::uint8_t> bytes);

llvm::ArrayRef<std::uint8_t> spatialTopologyDecisionSchemaBytes();
llvm::ArrayRef<std::uint8_t> spatialMicroarchitectureDecisionSchemaBytes();
llvm::ArrayRef<std::uint8_t> systemCompositionDecisionSchemaBytes();

} // namespace loom::dse

#endif // LOOM_DSE_HARDWAREDECISION_H
