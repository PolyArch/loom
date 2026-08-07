#ifndef LOOM_LIB_FABRIC_IDENTITY_FABRICARTIFACTVIEWSTORAGE_H
#define LOOM_LIB_FABRIC_IDENTITY_FABRICARTIFACTVIEWSTORAGE_H

#include "Fabric/Identity/FabricRefImport.h"

#include "FabricArtifactViewInternal.h"

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace loom::fabric {

struct FabricArtifactView::Storage {
  explicit Storage(detail::FabricArtifactViewData data)
      : data(std::move(data)) {}

  detail::FabricArtifactViewData data;
  std::vector<FabricPeOccurrenceRef> peOccurrences;
  std::vector<FabricFuOccurrenceRef> fuOccurrences;
  std::vector<FabricMemoryOccurrenceRef> memoryOccurrences;
  std::vector<FabricSwitchOccurrenceRef> switchOccurrences;
  std::vector<FabricFifoOccurrenceRef> fifoOccurrences;
  std::vector<FabricBoundaryOccurrenceRef> boundaryOccurrences;
  std::vector<HostCoreOccurrenceRef> hostCoreOccurrences;
  std::vector<AccCoreOccurrenceRef> accCoreOccurrences;
  std::vector<SystemMemoryServiceRef> systemMemoryServices;
  std::vector<SystemServiceEndpointRef> systemServiceEndpoints;
  std::vector<SystemServiceTransformRef> systemServiceTransforms;
  std::vector<ExternalBoundaryRef> externalBoundaries;
  std::vector<FabricTransportEndpointRef> transportEndpoints;
  std::vector<FabricPhysicalTagMatchDomainView> physicalTagMatchDomains;
  std::map<std::vector<std::uint8_t>, FabricOrdinal> tagMatchDomainByEndpoint;
  std::vector<FabricPhysicalTagAssignmentPointView> physicalTagAssignmentPoints;
  std::map<std::vector<std::uint8_t>, FabricOrdinal>
      tagAssignmentPointByEndpoint;
  std::vector<FabricPhysicalTraversalView> physicalTraversalViews;
  std::vector<FabricInventoryOwnerRef> moduleResourceOwners;
  std::vector<FabricModuleDomainMemberRef> moduleDomainMembers;
  std::vector<FabricFuTemplateRef> fuTemplates;
  std::vector<FabricMemoryEngineTemplateRef> memoryEngineTemplates;
  std::vector<std::vector<FabricMemoryOperationPortRef>> memoryPortRefs;
  std::vector<std::vector<std::uint8_t>> pointConnectionKeys;
  std::vector<std::vector<std::uint8_t>> memoryServiceConnectionKeys;
  std::vector<std::vector<std::uint8_t>> traversalKeys;
  std::map<std::vector<std::uint8_t>, std::vector<FabricFuPortAttachmentView>>
      fuPortAttachments;

  const detail::FabricEntityViewData *entity(FabricEntityId id,
                                             FabricEntityKind expected) const {
    if (id >= data.entities.size())
      return nullptr;
    const detail::FabricEntityViewData &record = data.entities[id];
    return record.kind == expected ? &record : nullptr;
  }

  template <FabricEntityKind Kind>
  const detail::FabricEntityViewData *
  entity(FabricTypedEntityRef<Kind> ref) const {
    return entity(ref.id(), Kind);
  }

  const detail::FabricNestedOwnerViewData *
  spatialCore(SpatialCoreOccurrenceRef ref) const {
    const detail::FabricEntityViewData *record = entity(ref.core);
    return record && record->spatialCore ? &*record->spatialCore : nullptr;
  }

  const detail::FabricNestedOwnerViewData *
  instructionCore(InstructionCoreContextRef ref) const {
    const detail::FabricEntityViewData *record = entity(ref.core);
    return record && record->instructionCore ? &*record->instructionCore
                                             : nullptr;
  }

  const detail::FabricNestedOwnerViewData *
  instructionContext(InstructionContextRef ref) const {
    const detail::FabricEntityViewData *record = entity(ref.pe);
    if (!record || ref.ordinal >= record->instructionContexts.size())
      return nullptr;
    return &record->instructionContexts[ref.ordinal];
  }

  const std::vector<detail::FabricFuNodeViewData> *
  fuNodes(FabricFuTemplateRef ref) const {
    const detail::FabricEntityViewData *record = entity(ref);
    return record ? &record->fuNodes : nullptr;
  }

  const std::vector<FabricFuCapabilityTemplateRecord> *
  fuCapabilityTemplates(FabricFuTemplateRef ref) const {
    const detail::FabricEntityViewData *record = entity(ref);
    return record ? &record->fuCapabilityTemplates : nullptr;
  }

  const std::vector<detail::FabricFuNodeViewData> *
  fuNodes(FabricFuOccurrenceRef ref) const {
    const detail::FabricEntityViewData *record = entity(ref);
    if (!record)
      return nullptr;
    if (!record->fuNodes.empty())
      return &record->fuNodes;
    return record->fuTemplate ? fuNodes(*record->fuTemplate) : nullptr;
  }

  const detail::FabricNestedOwnerViewData *
  fuNode(FabricFuTemplateNodeRef ref) const {
    const auto *nodes = fuNodes(ref.fu);
    if (!nodes || ref.ordinal >= nodes->size() ||
        (*nodes)[ref.ordinal].kind != ref.node)
      return nullptr;
    return &(*nodes)[ref.ordinal].owner;
  }

  const detail::FabricNestedOwnerViewData *
  fuNode(FabricFuOccurrenceNodeRef ref) const {
    const auto *nodes = fuNodes(ref.fu);
    if (!nodes || ref.ordinal >= nodes->size() ||
        (*nodes)[ref.ordinal].kind != ref.node)
      return nullptr;
    return &(*nodes)[ref.ordinal].owner;
  }

  const detail::FabricFuNodeViewData *
  fuNodeRecord(FabricFuTemplateNodeRef ref) const {
    const auto *nodes = fuNodes(ref.fu);
    if (!nodes || ref.ordinal >= nodes->size() ||
        (*nodes)[ref.ordinal].kind != ref.node)
      return nullptr;
    return &(*nodes)[ref.ordinal];
  }

  const ResolvedFabricOpCapabilityView *
  operationCapability(FabricFuTemplateNodeRef ref) const {
    const detail::FabricEntityViewData *record = entity(ref.fu);
    const detail::FabricFuNodeViewData *node = fuNodeRecord(ref);
    if (!record || !node || !node->operationCapabilityIndex ||
        *node->operationCapabilityIndex >= record->operationCapabilities.size())
      return nullptr;
    return &record->operationCapabilities[*node->operationCapabilityIndex];
  }

  const detail::FabricMemoryOperationPortViewData *
  memoryPortRecord(FabricMemoryOperationPortRef ref) const {
    const detail::FabricEntityViewData *record = entity(ref.memory);
    if (!record || ref.ordinal >= record->memoryOperationPorts.size())
      return nullptr;
    return &record->memoryOperationPorts[ref.ordinal];
  }

  const detail::FabricNestedOwnerViewData *
  memoryPort(FabricMemoryOperationPortRef ref) const {
    const detail::FabricMemoryOperationPortViewData *record =
        memoryPortRecord(ref);
    return record ? &record->owner : nullptr;
  }

  const detail::FabricNestedOwnerViewData *
  memoryService(const FabricMemoryServiceRef &ref) const {
    switch (ref.kind()) {
    case FabricMemoryServiceKind::Local: {
      const detail::FabricEntityViewData *record =
          entity(std::get<FabricMemoryOccurrenceRef>(ref.payload));
      return record && record->localMemoryService
                 ? &record->localMemoryService->owner
                 : nullptr;
    }
    case FabricMemoryServiceKind::System:
      return nullptr;
    }
    return nullptr;
  }

  const detail::FabricNestedOwnerViewData *
  transferPattern(FabricTransferPatternRef ref) const {
    const detail::FabricEntityViewData *record = entity(ref.resource);
    if (!record || ref.ordinal >= record->transferPatterns.size())
      return nullptr;
    return &record->transferPatterns[ref.ordinal];
  }

  const detail::FabricTransportEndpointViewData *
  transportEndpoint(const FabricTransportEndpointRef &endpoint) const {
    const detail::FabricNestedOwnerViewData *owner = nullptr;
    switch (endpoint.owner.kind()) {
    case FabricTransportEndpointOwnerKind::SpatialCoreOccurrence:
      owner = spatialCore(
          std::get<SpatialCoreOccurrenceRef>(endpoint.owner.payload));
      break;
    case FabricTransportEndpointOwnerKind::FabricPeOccurrence:
      if (const auto *record =
              entity(std::get<FabricPeOccurrenceRef>(endpoint.owner.payload)))
        owner = &record->owner;
      break;
    case FabricTransportEndpointOwnerKind::FabricFuOccurrence:
      if (const auto *record =
              entity(std::get<FabricFuOccurrenceRef>(endpoint.owner.payload)))
        owner = &record->owner;
      break;
    case FabricTransportEndpointOwnerKind::FabricMemoryOccurrence:
      if (const auto *record = entity(
              std::get<FabricMemoryOccurrenceRef>(endpoint.owner.payload)))
        owner = &record->owner;
      break;
    case FabricTransportEndpointOwnerKind::FabricSwitchOccurrence:
      if (const auto *record = entity(
              std::get<FabricSwitchOccurrenceRef>(endpoint.owner.payload)))
        owner = &record->owner;
      break;
    case FabricTransportEndpointOwnerKind::FabricFifoOccurrence:
      if (const auto *record =
              entity(std::get<FabricFifoOccurrenceRef>(endpoint.owner.payload)))
        owner = &record->owner;
      break;
    case FabricTransportEndpointOwnerKind::FabricBoundaryOccurrence:
      if (const auto *record = entity(
              std::get<FabricBoundaryOccurrenceRef>(endpoint.owner.payload)))
        owner = &record->owner;
      break;
    case FabricTransportEndpointOwnerKind::SystemServiceEndpoint:
      if (const auto *record = entity(
              std::get<SystemServiceEndpointRef>(endpoint.owner.payload)))
        owner = &record->owner;
      break;
    case FabricTransportEndpointOwnerKind::SystemTransportResource:
      if (const auto *record = entity(
              std::get<SystemTransportResourceRef>(endpoint.owner.payload)))
        owner = &record->owner;
      break;
    }
    if (!owner || endpoint.ordinal >= owner->transportEndpoints.size())
      return nullptr;
    return &owner->transportEndpoints[endpoint.ordinal];
  }

  const detail::FabricMemoryEndpointViewData *
  memoryEndpoint(const FabricMemoryEndpointRef &endpoint) const {
    const detail::FabricNestedOwnerViewData *owner = nullptr;
    switch (endpoint.owner.kind()) {
    case FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence:
      owner = spatialCore(
          std::get<SpatialCoreOccurrenceRef>(endpoint.owner.payload));
      break;
    case FabricMemoryEndpointOwnerKind::FabricMemoryOccurrence:
      if (const auto *record = entity(
              std::get<FabricMemoryOccurrenceRef>(endpoint.owner.payload)))
        owner = &record->owner;
      break;
    case FabricMemoryEndpointOwnerKind::SystemServiceEndpoint:
      if (const auto *record = entity(
              std::get<SystemServiceEndpointRef>(endpoint.owner.payload)))
        owner = &record->owner;
      break;
    }
    if (!owner || endpoint.ordinal >= owner->memoryEndpoints.size())
      return nullptr;
    return &owner->memoryEndpoints[endpoint.ordinal];
  }

  const detail::FabricModuleBoundaryEndpointViewData *moduleBoundaryEndpoint(
      const FabricModuleBoundaryEndpointRef &endpoint) const {
    const detail::FabricEntityViewData *module = entity(endpoint.module);
    if (!module)
      return nullptr;
    const auto &endpoints = endpoint.direction == FabricPortDirection::Input
                                ? module->moduleBoundaryInputs
                                : module->moduleBoundaryOutputs;
    if (endpoint.ordinal >= endpoints.size())
      return nullptr;
    return &endpoints[endpoint.ordinal];
  }

  const detail::FabricNestedOwnerViewData *
  inventoryOwner(const FabricInventoryOwnerRef &owner) const {
    const detail::FabricEntityViewData *record = nullptr;
    switch (owner.kind()) {
    case FabricInventoryOwnerKind::ModuleTemplate:
      record = entity(std::get<FabricModuleTemplateRef>(owner.payload));
      break;
    case FabricInventoryOwnerKind::PeOccurrence:
      record = entity(std::get<FabricPeOccurrenceRef>(owner.payload));
      break;
    case FabricInventoryOwnerKind::FuTemplate:
      record = entity(std::get<FabricFuTemplateRef>(owner.payload));
      break;
    case FabricInventoryOwnerKind::FuOccurrence:
      record = entity(std::get<FabricFuOccurrenceRef>(owner.payload));
      break;
    case FabricInventoryOwnerKind::FuTemplateNode:
      return fuNode(std::get<FabricFuTemplateNodeRef>(owner.payload));
    case FabricInventoryOwnerKind::FuOccurrenceNode:
      return fuNode(std::get<FabricFuOccurrenceNodeRef>(owner.payload));
    case FabricInventoryOwnerKind::MemoryOccurrence:
      record = entity(std::get<FabricMemoryOccurrenceRef>(owner.payload));
      break;
    case FabricInventoryOwnerKind::MemoryOperationPort:
      return memoryPort(std::get<FabricMemoryOperationPortRef>(owner.payload));
    case FabricInventoryOwnerKind::MemoryService: {
      const FabricMemoryServiceRef &service =
          std::get<FabricMemoryServiceRef>(owner.payload);
      if (service.kind() == FabricMemoryServiceKind::Local)
        return memoryService(service);
      record = entity(std::get<SystemMemoryServiceRef>(service.payload));
      break;
    }
    case FabricInventoryOwnerKind::SwitchOccurrence:
      record = entity(std::get<FabricSwitchOccurrenceRef>(owner.payload));
      break;
    case FabricInventoryOwnerKind::FifoOccurrence:
      record = entity(std::get<FabricFifoOccurrenceRef>(owner.payload));
      break;
    case FabricInventoryOwnerKind::BoundaryOccurrence:
      record = entity(std::get<FabricBoundaryOccurrenceRef>(owner.payload));
      break;
    case FabricInventoryOwnerKind::InstructionContext:
      return instructionContext(std::get<InstructionContextRef>(owner.payload));
    case FabricInventoryOwnerKind::InstructionCoreContext:
      return instructionCore(
          std::get<InstructionCoreContextRef>(owner.payload));
    case FabricInventoryOwnerKind::HostCoreOccurrence:
      record = entity(std::get<HostCoreOccurrenceRef>(owner.payload));
      break;
    case FabricInventoryOwnerKind::AccCoreOccurrence:
      record = entity(std::get<AccCoreOccurrenceRef>(owner.payload));
      break;
    case FabricInventoryOwnerKind::SystemServiceEndpoint:
      record = entity(std::get<SystemServiceEndpointRef>(owner.payload));
      break;
    case FabricInventoryOwnerKind::SystemServiceTransform:
      record = entity(std::get<SystemServiceTransformRef>(owner.payload));
      break;
    case FabricInventoryOwnerKind::SystemTransportResource:
      record = entity(std::get<SystemTransportResourceRef>(owner.payload));
      break;
    case FabricInventoryOwnerKind::TransferPattern:
      return transferPattern(std::get<FabricTransferPatternRef>(owner.payload));
    case FabricInventoryOwnerKind::HardwareDomain:
      record = entity(std::get<HardwareDomainRef>(owner.payload));
      break;
    case FabricInventoryOwnerKind::ExternalBoundary:
      record = entity(std::get<ExternalBoundaryRef>(owner.payload));
      break;
    case FabricInventoryOwnerKind::SpatialCoreOccurrence:
      return spatialCore(std::get<SpatialCoreOccurrenceRef>(owner.payload));
    }
    return record ? &record->owner : nullptr;
  }
};

} // namespace loom::fabric

#endif // LOOM_LIB_FABRIC_IDENTITY_FABRICARTIFACTVIEWSTORAGE_H
