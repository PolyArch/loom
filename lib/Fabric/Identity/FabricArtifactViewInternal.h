#ifndef LOOM_LIB_FABRIC_IDENTITY_FABRICARTIFACTVIEWINTERNAL_H
#define LOOM_LIB_FABRIC_IDENTITY_FABRICARTIFACTVIEWINTERNAL_H

#include "Common/Artifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/MemoryConnectivityContract.h"
#include "Fabric/IR/MemoryOperationPort.h"
#include "Fabric/Identity/FabricFuCapabilityTemplate.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::fabric::detail {

struct FabricTransportEndpointViewData {
  FabricPortDirection direction = FabricPortDirection::Input;
  std::vector<std::uint8_t> canonicalType;
};

struct FabricMemoryEndpointViewData {
  FabricMemoryEndpointRole role = FabricMemoryEndpointRole::Manager;
  std::vector<std::uint8_t> canonicalType;
};

struct FabricNestedOwnerViewData {
  std::vector<FabricTransportEndpointViewData> transportEndpoints;
  std::vector<FabricMemoryEndpointViewData> memoryEndpoints;
  std::vector<std::uint64_t> inventoryCounts;
  std::optional<::fabric::ResourceContract> resourceContract;
};

struct FabricModuleBoundaryEndpointViewData {
  FabricSpatialAttachmentEndpointRef::Plane plane =
      FabricSpatialAttachmentEndpointRef::Plane::Transport;
  FabricOrdinal occurrenceOrdinal = 0;
  std::vector<std::uint8_t> canonicalType;
};

struct FabricFuNodeViewData {
  FabricFuNodeKind kind = FabricFuNodeKind::Op;
  FabricNestedOwnerViewData owner;
  std::optional<std::size_t> operationCapabilityIndex;
};

struct FabricMemoryOperationPortViewData {
  FabricNestedOwnerViewData owner;
  ::fabric::MemoryOperationPortRecord record;
};

struct FabricEntityViewData {
  FabricEntityKind kind = FabricEntityKind::FabricModuleTemplate;
  FabricNestedOwnerViewData owner;
  std::vector<FabricFuNodeViewData> fuNodes;
  std::vector<ResolvedFabricOpCapabilityView> operationCapabilities;
  std::vector<FabricFuCapabilityTemplateRecord> fuCapabilityTemplates;
  std::vector<FabricMemoryOperationPortViewData> memoryOperationPorts;
  std::optional<::fabric::Schedule> memorySchedule;
  std::optional<std::uint64_t> memoryResidentContextCount;
  std::optional<::fabric::MemoryConnectivityContractRecord> memoryConnectivity;
  std::vector<FabricNestedOwnerViewData> instructionContexts;
  std::vector<FabricNestedOwnerViewData> transferPatterns;
  std::vector<FabricTransferPatternRef> transferPatternRefs;
  std::vector<SystemTransferPatternRecord> transferPatternRecords;
  std::optional<FabricNestedOwnerViewData> spatialCore;
  std::optional<FabricImportedModuleTargetRef> spatialCoreTarget;
  std::optional<FabricNestedOwnerViewData> instructionCore;
  std::optional<InstructionCoreArchitecturalContract>
      instructionCoreArchitecture;
  std::optional<InstructionCoreMicroarchitecturalRealization>
      instructionCoreMicroarchitecture;
  std::optional<FabricNestedOwnerViewData> localMemoryService;
  std::optional<::fabric::Schedule> peSchedule;
  std::optional<FabricPeOccurrenceRef> parentPe;
  std::optional<FabricFuTemplateRef> fuTemplate;
  std::optional<FabricMemoryEngineTemplateRef> memoryEngineTemplate;
  std::optional<FabricMemoryEngineTemplateRecord> memoryEngineTemplateRecord;
  std::optional<std::vector<std::uint8_t>> memoryEngineTemplateProjection;
  std::optional<FabricHardwareDomainKind> hardwareDomainKind;
  std::optional<HardwareDomainContractRecord> hardwareDomainContract;
  std::optional<ClockCrossingContractRecord> clockCrossing;
  std::vector<FabricModuleBoundaryEndpointViewData> moduleBoundaryInputs;
  std::vector<FabricModuleBoundaryEndpointViewData> moduleBoundaryOutputs;
};

struct FabricArtifactViewData {
  FabricArtifactViewData(ArtifactIdentity identity, FabricRootKind rootKind)
      : identity(std::move(identity)), rootKind(rootKind) {}

  ArtifactIdentity identity;
  FabricRootKind rootKind = FabricRootKind::Module;
  std::vector<FabricEntityViewData> entities;
  std::vector<FabricPointConnectionPayload> pointConnections;
  std::vector<FabricPhysicalTraversalRef> admittedTraversals;
  std::vector<FabricModuleBoundaryTransportAttachmentView>
      moduleBoundaryTransportAttachments;
  std::vector<FabricArtifactView> importedModules;
  std::vector<FabricSpatialAttachmentRecordView> spatialAttachments;
  std::vector<HardwareDomainRef> hardwareDomains;
  std::vector<SystemTransportResourceRef> transportResources;
};

llvm::Expected<FabricArtifactView>
buildFabricArtifactView(FabricArtifactViewData data);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_IDENTITY_FABRICARTIFACTVIEWINTERNAL_H
