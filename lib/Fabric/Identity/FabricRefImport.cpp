#include "Fabric/Identity/FabricRefImport.h"

#include "Fabric/Artifact/FabricSystemRootView.h"

#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefText.h"
#include "FabricArtifactViewInternal.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::fabric;

struct FabricArtifactView::Storage {
  detail::FabricArtifactViewData data;
  std::vector<std::vector<FabricMemoryOperationPortRef>> memoryPortRefs;
  std::vector<std::vector<std::uint8_t>> pointConnectionKeys;
  std::vector<std::vector<std::uint8_t>> traversalKeys;

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
      return record && record->localMemoryService ? &*record->localMemoryService
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

FabricArtifactView::~FabricArtifactView() = default;

namespace {

std::uint64_t inventoryCount(llvm::ArrayRef<std::uint64_t> counts,
                             FabricInventoryKind kind) {
  const std::size_t index = static_cast<std::size_t>(kind);
  return index < counts.size() ? counts[index] : 0;
}

template <typename Row>
bool containsCanonicalRow(const std::vector<Row> &rows, const Row &needle) {
  return std::binary_search(rows.begin(), rows.end(), needle);
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::vector<std::uint8_t>
pointConnectionKey(const FabricTransportEndpointRef &source,
                   const FabricTransportEndpointRef &destination) {
  std::vector<std::uint8_t> sourceBytes = canonicalFabricBytes(source);
  std::vector<std::uint8_t> destinationBytes =
      canonicalFabricBytes(destination);
  std::vector<std::uint8_t> key;
  key.reserve(16 + sourceBytes.size() + destinationBytes.size());
  appendU64(key, sourceBytes.size());
  key.insert(key.end(), sourceBytes.begin(), sourceBytes.end());
  appendU64(key, destinationBytes.size());
  key.insert(key.end(), destinationBytes.begin(), destinationBytes.end());
  return key;
}

bool haveSameTransportKind(llvm::ArrayRef<std::uint8_t> left,
                           llvm::ArrayRef<std::uint8_t> right) {
  constexpr std::size_t kindBytes = sizeof(std::uint32_t);
  return left.size() >= kindBytes && right.size() >= kindBytes &&
         left.take_front(kindBytes) == right.take_front(kindBytes);
}

llvm::Error invalidView(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

llvm::Error
validateInventoryShape(llvm::ArrayRef<std::uint64_t> inventoryCounts,
                       llvm::StringRef ownerDescription) {
  if (inventoryCounts.size() != fabricClosedBound(FabricInventoryKind{}))
    return invalidView(llvm::Twine(ownerDescription) +
                       " has an incomplete canonical inventory catalog");
  return llvm::Error::success();
}

llvm::Error validateNestedOwner(const detail::FabricNestedOwnerViewData &owner,
                                llvm::StringRef ownerDescription) {
  if (llvm::Error error =
          validateInventoryShape(owner.inventoryCounts, ownerDescription))
    return error;
  if (inventoryCount(owner.inventoryCounts,
                     FabricInventoryKind::ResourceState) != 0 ||
      inventoryCount(owner.inventoryCounts, FabricInventoryKind::UsePattern) !=
          0)
    return invalidView(llvm::Twine(ownerDescription) +
                       " duplicates ResourceContract-owned inventories");
  for (const detail::FabricTransportEndpointViewData &endpoint :
       owner.transportEndpoints) {
    if (endpoint.canonicalType.empty())
      return invalidView(llvm::Twine(ownerDescription) +
                         " has a token endpoint without a physical type");
    if (endpoint.direction != FabricPortDirection::Input &&
        endpoint.direction != FabricPortDirection::Output)
      return invalidView(llvm::Twine(ownerDescription) +
                         " has an unknown token endpoint direction");
  }
  for (const detail::FabricMemoryEndpointViewData &endpoint :
       owner.memoryEndpoints)
    if (endpoint.role != FabricMemoryEndpointRole::Manager &&
        endpoint.role != FabricMemoryEndpointRole::Subordinate)
      return invalidView(llvm::Twine(ownerDescription) +
                         " has an unknown memory endpoint role");
  return llvm::Error::success();
}

} // namespace

const ArtifactIdentity &FabricArtifactView::identity() const {
  return storage_->data.identity;
}

FabricRootKind FabricArtifactView::rootKind() const {
  return storage_->data.rootKind;
}

std::optional<FabricEntityKind>
FabricArtifactView::entityKind(FabricEntityId id) const {
  if (id >= storage_->data.entities.size())
    return std::nullopt;
  return storage_->data.entities[id].kind;
}

std::uint64_t FabricArtifactView::transportEndpointCount(
    const FabricTransportEndpointOwnerRef &owner) const {
  const detail::FabricEntityViewData *entity = nullptr;
  switch (owner.kind()) {
  case FabricTransportEndpointOwnerKind::SpatialCoreOccurrence: {
    const auto *nested = storage_->spatialCore(
        std::get<SpatialCoreOccurrenceRef>(owner.payload));
    return nested ? nested->transportEndpoints.size() : 0;
  }
  case FabricTransportEndpointOwnerKind::FabricPeOccurrence:
    entity = storage_->entity(std::get<FabricPeOccurrenceRef>(owner.payload));
    break;
  case FabricTransportEndpointOwnerKind::FabricFuOccurrence:
    entity = storage_->entity(std::get<FabricFuOccurrenceRef>(owner.payload));
    break;
  case FabricTransportEndpointOwnerKind::FabricMemoryOccurrence:
    entity =
        storage_->entity(std::get<FabricMemoryOccurrenceRef>(owner.payload));
    break;
  case FabricTransportEndpointOwnerKind::FabricSwitchOccurrence:
    entity =
        storage_->entity(std::get<FabricSwitchOccurrenceRef>(owner.payload));
    break;
  case FabricTransportEndpointOwnerKind::FabricFifoOccurrence:
    entity = storage_->entity(std::get<FabricFifoOccurrenceRef>(owner.payload));
    break;
  case FabricTransportEndpointOwnerKind::FabricBoundaryOccurrence:
    entity =
        storage_->entity(std::get<FabricBoundaryOccurrenceRef>(owner.payload));
    break;
  case FabricTransportEndpointOwnerKind::SystemServiceEndpoint:
    entity =
        storage_->entity(std::get<SystemServiceEndpointRef>(owner.payload));
    break;
  case FabricTransportEndpointOwnerKind::SystemTransportResource:
    entity =
        storage_->entity(std::get<SystemTransportResourceRef>(owner.payload));
    break;
  }
  return entity ? entity->owner.transportEndpoints.size() : 0;
}

std::optional<FabricPortDirection>
FabricArtifactView::transportEndpointDirection(
    const FabricTransportEndpointRef &endpoint) const {
  const detail::FabricTransportEndpointViewData *record =
      storage_->transportEndpoint(endpoint);
  return record ? std::optional<FabricPortDirection>(record->direction)
                : std::nullopt;
}

llvm::ArrayRef<std::uint8_t> FabricArtifactView::transportEndpointType(
    const FabricTransportEndpointRef &endpoint) const {
  const detail::FabricTransportEndpointViewData *record =
      storage_->transportEndpoint(endpoint);
  return record ? llvm::ArrayRef<std::uint8_t>(record->canonicalType)
                : llvm::ArrayRef<std::uint8_t>();
}

std::uint64_t FabricArtifactView::memoryEndpointCount(
    const FabricMemoryEndpointOwnerRef &owner) const {
  const detail::FabricEntityViewData *entity = nullptr;
  switch (owner.kind()) {
  case FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence: {
    const auto *nested = storage_->spatialCore(
        std::get<SpatialCoreOccurrenceRef>(owner.payload));
    return nested ? nested->memoryEndpoints.size() : 0;
  }
  case FabricMemoryEndpointOwnerKind::FabricMemoryOccurrence:
    entity =
        storage_->entity(std::get<FabricMemoryOccurrenceRef>(owner.payload));
    break;
  case FabricMemoryEndpointOwnerKind::SystemServiceEndpoint:
    entity =
        storage_->entity(std::get<SystemServiceEndpointRef>(owner.payload));
    break;
  }
  return entity ? entity->owner.memoryEndpoints.size() : 0;
}

std::uint64_t
FabricArtifactView::inventorySize(const FabricInventoryOwnerRef &owner,
                                  FabricInventoryKind inventory) const {
  const detail::FabricNestedOwnerViewData *resolved =
      storage_->inventoryOwner(owner);
  if (!resolved)
    return 0;
  if (inventory == FabricInventoryKind::ResourceState)
    return resolved->resourceContract ? resolved->resourceContract->stateCount()
                                      : 0;
  if (inventory == FabricInventoryKind::UsePattern)
    return resolved->resourceContract
               ? resolved->resourceContract->usePatternCount()
               : 0;
  return inventoryCount(resolved->inventoryCounts, inventory);
}

const ::fabric::ResourceContract *FabricArtifactView::resourceContract(
    const FabricInventoryOwnerRef &owner) const {
  const detail::FabricNestedOwnerViewData *resolved =
      storage_->inventoryOwner(owner);
  return resolved && resolved->resourceContract ? &*resolved->resourceContract
                                                : nullptr;
}

std::optional<FabricFuNodeKind>
FabricArtifactView::fuNodeKind(const FabricInventoryOwnerRef &owner,
                               FabricOrdinal ordinal) const {
  const std::vector<detail::FabricFuNodeViewData> *nodes = nullptr;
  if (owner.kind() == FabricInventoryOwnerKind::FuTemplate)
    nodes = storage_->fuNodes(std::get<FabricFuTemplateRef>(owner.payload));
  else if (owner.kind() == FabricInventoryOwnerKind::FuOccurrence)
    nodes = storage_->fuNodes(std::get<FabricFuOccurrenceRef>(owner.payload));
  if (!nodes || ordinal >= nodes->size())
    return std::nullopt;
  return (*nodes)[ordinal].kind;
}

bool FabricArtifactView::declaresLocalMemoryService(
    FabricMemoryOccurrenceRef memory) const {
  const detail::FabricEntityViewData *record = storage_->entity(memory);
  return record && record->localMemoryService.has_value();
}

std::optional<FabricMemoryEndpointRole> FabricArtifactView::memoryEndpointRole(
    const FabricMemoryEndpointRef &endpoint) const {
  const detail::FabricMemoryEndpointViewData *record =
      storage_->memoryEndpoint(endpoint);
  return record ? std::optional<FabricMemoryEndpointRole>(record->role)
                : std::nullopt;
}

llvm::ArrayRef<std::uint8_t> FabricArtifactView::memoryEndpointType(
    const FabricMemoryEndpointRef &endpoint) const {
  const detail::FabricMemoryEndpointViewData *record =
      storage_->memoryEndpoint(endpoint);
  return record ? llvm::ArrayRef<std::uint8_t>(record->canonicalType)
                : llvm::ArrayRef<std::uint8_t>();
}

std::uint64_t FabricArtifactView::moduleBoundaryEndpointCount(
    FabricModuleTemplateRef module, FabricPortDirection direction) const {
  const detail::FabricEntityViewData *record = storage_->entity(module);
  if (!record)
    return 0;
  return direction == FabricPortDirection::Input
             ? record->moduleBoundaryInputs.size()
             : record->moduleBoundaryOutputs.size();
}

std::optional<FabricSpatialAttachmentEndpointRef::Plane>
FabricArtifactView::moduleBoundaryEndpointPlane(
    const FabricModuleBoundaryEndpointRef &endpoint) const {
  const detail::FabricModuleBoundaryEndpointViewData *record =
      storage_->moduleBoundaryEndpoint(endpoint);
  return record ? std::optional<FabricSpatialAttachmentEndpointRef::Plane>(
                      record->plane)
                : std::nullopt;
}

std::optional<FabricOrdinal>
FabricArtifactView::moduleBoundaryEndpointOccurrenceOrdinal(
    const FabricModuleBoundaryEndpointRef &endpoint) const {
  const detail::FabricModuleBoundaryEndpointViewData *record =
      storage_->moduleBoundaryEndpoint(endpoint);
  return record ? std::optional<FabricOrdinal>(record->occurrenceOrdinal)
                : std::nullopt;
}

llvm::ArrayRef<std::uint8_t> FabricArtifactView::moduleBoundaryEndpointType(
    const FabricModuleBoundaryEndpointRef &endpoint) const {
  const detail::FabricModuleBoundaryEndpointViewData *record =
      storage_->moduleBoundaryEndpoint(endpoint);
  return record ? llvm::ArrayRef<std::uint8_t>(record->canonicalType)
                : llvm::ArrayRef<std::uint8_t>();
}

std::optional<FabricHardwareDomainKind>
FabricArtifactView::hardwareDomainKind(HardwareDomainRef domain) const {
  const detail::FabricEntityViewData *record = storage_->entity(domain);
  return record ? record->hardwareDomainKind : std::nullopt;
}

std::optional<FabricFuTemplateRef>
FabricArtifactView::fuTemplateOf(FabricFuOccurrenceRef occurrence) const {
  const detail::FabricEntityViewData *record = storage_->entity(occurrence);
  return record ? record->fuTemplate : std::nullopt;
}

llvm::ArrayRef<FabricFuCapabilityTemplateRecord>
FabricArtifactView::fuCapabilityTemplates(
    FabricFuTemplateRef definition) const {
  const auto *records = storage_->fuCapabilityTemplates(definition);
  return records ? llvm::ArrayRef<FabricFuCapabilityTemplateRecord>(*records)
                 : llvm::ArrayRef<FabricFuCapabilityTemplateRecord>();
}

llvm::ArrayRef<FabricMemoryOperationPortRef>
FabricArtifactView::memoryOperationPorts(
    FabricMemoryOccurrenceRef memory) const {
  if (memory.id() >= storage_->memoryPortRefs.size())
    return {};
  return storage_->memoryPortRefs[memory.id()];
}

const MemoryOperationPortView *FabricArtifactView::memoryOperationPort(
    FabricMemoryOperationPortRef port) const {
  const detail::FabricMemoryOperationPortViewData *record =
      storage_->memoryPortRecord(port);
  return record ? &record->record : nullptr;
}

const MemoryCapabilityAlternativeView *
FabricArtifactView::memoryCapabilityAlternative(
    FabricMemoryCapabilityAlternativeRef alternative) const {
  const MemoryOperationPortView *port = memoryOperationPort(alternative.port);
  if (!port || alternative.ordinal >= port->capabilityAlternatives().size())
    return nullptr;
  return &port->capabilityAlternatives()[alternative.ordinal];
}

std::optional<::fabric::Schedule>
FabricArtifactView::memorySchedule(FabricMemoryOccurrenceRef memory) const {
  const detail::FabricEntityViewData *record = storage_->entity(memory);
  return record ? record->memorySchedule : std::nullopt;
}

std::uint64_t FabricArtifactView::memoryResidentContextCount(
    FabricMemoryOccurrenceRef memory) const {
  const detail::FabricEntityViewData *record = storage_->entity(memory);
  return record && record->memoryResidentContextCount
             ? *record->memoryResidentContextCount
             : 0;
}

const ::fabric::MemoryConnectivityContractRecord *
FabricArtifactView::memoryConnectivity(FabricMemoryOccurrenceRef memory) const {
  const detail::FabricEntityViewData *record = storage_->entity(memory);
  return record && record->memoryConnectivity ? &*record->memoryConnectivity
                                              : nullptr;
}

bool FabricArtifactView::hasPointConnection(
    const FabricTransportEndpointRef &source,
    const FabricTransportEndpointRef &destination) const {
  return containsCanonicalRow(storage_->pointConnectionKeys,
                              pointConnectionKey(source, destination));
}

llvm::ArrayRef<FabricPointConnectionPayload>
FabricArtifactView::pointConnections() const {
  return storage_->data.pointConnections;
}

bool FabricArtifactView::admitsTraversal(
    const FabricPhysicalTraversalRef &traversal) const {
  return containsCanonicalRow(storage_->traversalKeys,
                              canonicalFabricBytes(traversal));
}

llvm::ArrayRef<FabricPhysicalTraversalRef>
FabricArtifactView::admittedTraversals() const {
  return storage_->data.admittedTraversals;
}

llvm::ArrayRef<FabricSpatialAttachmentRecordView>
FabricSystemRootView::spatialAttachments() const {
  return artifact_.storage_->data.spatialAttachments;
}

llvm::ArrayRef<HardwareDomainRef>
FabricSystemRootView::hardwareDomains() const {
  return artifact_.storage_->data.hardwareDomains;
}

const HardwareDomainContractRecord *
FabricSystemRootView::hardwareDomainContract(HardwareDomainRef domain) const {
  const detail::FabricEntityViewData *entity =
      artifact_.storage_->entity(domain);
  return entity && entity->hardwareDomainContract
             ? &*entity->hardwareDomainContract
             : nullptr;
}

llvm::ArrayRef<FabricInventoryOwnerRef>
FabricSystemRootView::hardwareDomainMembers(HardwareDomainRef domain) const {
  const HardwareDomainContractRecord *contract = hardwareDomainContract(domain);
  return contract ? contract->members()
                  : llvm::ArrayRef<FabricInventoryOwnerRef>();
}

llvm::ArrayRef<SystemTransportResourceRef>
FabricSystemRootView::transportResources() const {
  return artifact_.storage_->data.transportResources;
}

llvm::ArrayRef<FabricTransferPatternRef> FabricSystemRootView::transferPatterns(
    SystemTransportResourceRef resource) const {
  if (resource.id() >= artifact_.storage_->data.entities.size())
    return {};
  const detail::FabricEntityViewData *entity =
      artifact_.storage_->entity(resource);
  return entity ? llvm::ArrayRef<FabricTransferPatternRef>(
                      entity->transferPatternRefs)
                : llvm::ArrayRef<FabricTransferPatternRef>();
}

const SystemTransferPatternRecord *
FabricSystemRootView::transferPattern(FabricTransferPatternRef pattern) const {
  const detail::FabricEntityViewData *entity =
      artifact_.storage_->entity(pattern.resource);
  if (!entity || pattern.ordinal >= entity->transferPatternRecords.size())
    return nullptr;
  const SystemTransferPatternRecord &record =
      entity->transferPatternRecords[pattern.ordinal];
  return record.pattern() == pattern ? &record : nullptr;
}

const ClockCrossingContractRecord *
FabricSystemRootView::clockCrossing(SystemTransportResourceRef resource) const {
  const detail::FabricEntityViewData *entity =
      artifact_.storage_->entity(resource);
  return entity && entity->clockCrossing ? &*entity->clockCrossing : nullptr;
}

llvm::Expected<FabricSystemRootView>
loom::fabric::requireSystemRoot(const FabricArtifactView &view) {
  if (view.rootKind() != FabricRootKind::System)
    return makeFabricRefError(FabricRefErrorKind::WrongRootKind,
                              "Fabric root is not a System");
  return FabricSystemRootView(view);
}

llvm::Expected<FabricArtifactView>
loom::fabric::detail::buildFabricArtifactView(FabricArtifactViewData data) {
  auto validClosedValue = [](auto value) {
    return static_cast<std::uint32_t>(value) < fabricClosedBound(value);
  };

  for (std::size_t index = 0; index < data.entities.size(); ++index) {
    FabricEntityViewData &entity = data.entities[index];
    if (!validClosedValue(entity.kind))
      return invalidView("entity has an unknown kind");
    if (llvm::Error error = validateNestedOwner(entity.owner, "Fabric entity"))
      return std::move(error);
    for (const FabricMemoryEndpointViewData &endpoint :
         entity.owner.memoryEndpoints)
      if (!validClosedValue(endpoint.role))
        return invalidView("memory endpoint has an unknown role");

    for (FabricFuNodeViewData &node : entity.fuNodes) {
      if (!validClosedValue(node.kind))
        return invalidView("FU node has an unknown kind");
      if (llvm::Error error = validateNestedOwner(node.owner, "FU node"))
        return std::move(error);
    }
    if (entity.kind == FabricEntityKind::FabricFuTemplate) {
      auto normalized = normalizeFabricFuCapabilityTemplateInventory(
          entity.fuCapabilityTemplates);
      if (!normalized)
        return normalized.takeError();
      if (*normalized != entity.fuCapabilityTemplates)
        return invalidView("FU capability-template inventory is not canonical");
    } else if (!entity.fuCapabilityTemplates.empty()) {
      return invalidView(
          "only an FU template may own capability-template records");
    }
    for (FabricMemoryOperationPortViewData &port : entity.memoryOperationPorts)
      if (llvm::Error error =
              validateNestedOwner(port.owner, "memory operation port"))
        return std::move(error);
    const bool isMemory =
        entity.kind == FabricEntityKind::FabricMemoryOccurrence;
    if (!isMemory &&
        (entity.memorySchedule || entity.memoryResidentContextCount ||
         entity.memoryConnectivity || !entity.memoryOperationPorts.empty() ||
         entity.localMemoryService))
      return invalidView("non-memory entity owns memory occurrence state");
    if (isMemory) {
      if (!entity.memoryConnectivity)
        return invalidView("memory occurrence has no connectivity contract");
      if (entity.memoryConnectivity->operationPorts().size() !=
          entity.memoryOperationPorts.size())
        return invalidView(
            "memory connectivity operation ports do not match the view");
      const std::uint64_t subordinateCount = llvm::count_if(
          entity.owner.memoryEndpoints,
          [](const FabricMemoryEndpointViewData &endpoint) {
            return endpoint.role == FabricMemoryEndpointRole::Subordinate;
          });
      if (entity.memoryConnectivity->subordinateEndpoints().size() !=
          subordinateCount)
        return invalidView(
            "memory connectivity subordinate endpoints do not match the "
            "view");
      if (!entity.memorySchedule) {
        if (entity.memoryResidentContextCount ||
            !entity.memoryOperationPorts.empty())
          return invalidView(
              "storage-only memory occurrence owns operation-engine state");
      } else if (*entity.memorySchedule == ::fabric::Schedule::Spatial) {
        if (entity.memoryResidentContextCount)
          return invalidView(
              "spatial memory occurrence owns resident contexts");
      } else if (*entity.memorySchedule == ::fabric::Schedule::Temporal) {
        if (!entity.memoryResidentContextCount ||
            *entity.memoryResidentContextCount == 0)
          return invalidView(
              "temporal memory occurrence has no resident contexts");
      } else {
        return invalidView("memory occurrence has an unknown schedule");
      }
      const std::uint64_t expectedContexts =
          entity.memoryResidentContextCount.value_or(0);
      for (const FabricMemoryOperationPortViewData &port :
           entity.memoryOperationPorts)
        if (inventoryCount(port.owner.inventoryCounts,
                           FabricInventoryKind::MemoryOperationContext) !=
            expectedContexts)
          return invalidView(
              "memory operation-context inventory does not match its "
              "engine");
    }
    for (FabricNestedOwnerViewData &context : entity.instructionContexts)
      if (llvm::Error error =
              validateNestedOwner(context, "instruction context"))
        return std::move(error);
    for (FabricNestedOwnerViewData &pattern : entity.transferPatterns)
      if (llvm::Error error = validateNestedOwner(pattern, "transfer pattern"))
        return std::move(error);
    if (entity.spatialCore)
      if (llvm::Error error =
              validateNestedOwner(*entity.spatialCore, "spatial core"))
        return std::move(error);
    if (entity.instructionCore)
      if (llvm::Error error =
              validateNestedOwner(*entity.instructionCore, "instruction core"))
        return std::move(error);
    if (entity.localMemoryService)
      if (llvm::Error error = validateNestedOwner(*entity.localMemoryService,
                                                  "local memory service"))
        return std::move(error);

    if (inventoryCount(entity.owner.inventoryCounts,
                       FabricInventoryKind::FuNode) != entity.fuNodes.size())
      return invalidView("FU node inventory does not match its records");
    if (inventoryCount(entity.owner.inventoryCounts,
                       FabricInventoryKind::MemoryOperationPort) !=
        entity.memoryOperationPorts.size())
      return invalidView(
          "memory operation-port inventory does not match its records");
    if (inventoryCount(entity.owner.inventoryCounts,
                       FabricInventoryKind::InstructionContext) !=
        entity.instructionContexts.size())
      return invalidView(
          "instruction-context inventory does not match its records");
    if (inventoryCount(entity.owner.inventoryCounts,
                       FabricInventoryKind::TransferPattern) !=
        entity.transferPatterns.size())
      return invalidView(
          "transfer-pattern inventory does not match its records");
    if (entity.transferPatterns.size() !=
            entity.transferPatternRecords.size() ||
        entity.transferPatterns.size() != entity.transferPatternRefs.size())
      return invalidView(
          "transfer-pattern inventory does not match its semantic records");

    if (entity.fuTemplate) {
      if (entity.kind != FabricEntityKind::FabricFuOccurrence)
        return invalidView("only an FU occurrence may select an FU template");
      const FabricEntityId templateId = entity.fuTemplate->id();
      if (templateId >= data.entities.size() ||
          data.entities[templateId].kind != FabricEntityKind::FabricFuTemplate)
        return invalidView("FU occurrence selects an invalid FU template");
    } else if (entity.kind == FabricEntityKind::FabricFuOccurrence) {
      return invalidView("FU occurrence has no FU template relation");
    }

    if (entity.hardwareDomainKind &&
        (entity.kind != FabricEntityKind::HardwareDomain ||
         !validClosedValue(*entity.hardwareDomainKind)))
      return invalidView("invalid hardware-domain kind projection");
    if (entity.kind == FabricEntityKind::HardwareDomain &&
        !entity.hardwareDomainKind)
      return invalidView("hardware-domain entity has no domain kind");
    (void)index;
  }

  using PointConnectionRow =
      std::pair<std::vector<std::uint8_t>, FabricPointConnectionPayload>;
  std::vector<PointConnectionRow> pointConnectionRows;
  pointConnectionRows.reserve(data.pointConnections.size());
  for (FabricPointConnectionPayload &connection : data.pointConnections)
    pointConnectionRows.emplace_back(
        pointConnectionKey(connection.source, connection.destination),
        std::move(connection));
  std::sort(pointConnectionRows.begin(), pointConnectionRows.end(),
            [](const PointConnectionRow &lhs, const PointConnectionRow &rhs) {
              return lhs.first < rhs.first;
            });
  for (std::size_t index = 1; index < pointConnectionRows.size(); ++index)
    if (pointConnectionRows[index - 1].first ==
        pointConnectionRows[index].first)
      return invalidView("point connections contain a duplicate");

  data.pointConnections.clear();
  std::vector<std::vector<std::uint8_t>> pointConnectionKeys;
  std::set<std::vector<std::uint8_t>> connectedSources;
  std::set<std::vector<std::uint8_t>> connectedDestinations;
  data.pointConnections.reserve(pointConnectionRows.size());
  pointConnectionKeys.reserve(pointConnectionRows.size());
  for (PointConnectionRow &row : pointConnectionRows) {
    if (!connectedSources.insert(canonicalFabricBytes(row.second.source))
             .second)
      return invalidView("point connection source is connected more than once");
    if (!connectedDestinations
             .insert(canonicalFabricBytes(row.second.destination))
             .second)
      return invalidView(
          "point connection destination is connected more than once");
    pointConnectionKeys.push_back(std::move(row.first));
    data.pointConnections.push_back(std::move(row.second));
  }

  using TraversalRow =
      std::pair<std::vector<std::uint8_t>, FabricPhysicalTraversalRef>;
  std::vector<TraversalRow> traversalRows;
  traversalRows.reserve(data.admittedTraversals.size());
  for (FabricPhysicalTraversalRef &traversal : data.admittedTraversals)
    traversalRows.emplace_back(canonicalFabricBytes(traversal),
                               std::move(traversal));
  std::sort(traversalRows.begin(), traversalRows.end(),
            [](const TraversalRow &lhs, const TraversalRow &rhs) {
              return lhs.first < rhs.first;
            });
  for (std::size_t index = 1; index < traversalRows.size(); ++index)
    if (traversalRows[index - 1].first == traversalRows[index].first)
      return invalidView("admitted traversals contain a duplicate");

  data.admittedTraversals.clear();
  std::vector<std::vector<std::uint8_t>> traversalKeys;
  data.admittedTraversals.reserve(traversalRows.size());
  traversalKeys.reserve(traversalRows.size());
  for (TraversalRow &row : traversalRows) {
    traversalKeys.push_back(std::move(row.first));
    data.admittedTraversals.push_back(std::move(row.second));
  }

  std::vector<std::vector<FabricMemoryOperationPortRef>> memoryPortRefs(
      data.entities.size());
  for (auto [entityId, entity] : llvm::enumerate(data.entities)) {
    if (entity.kind != FabricEntityKind::FabricMemoryOccurrence)
      continue;
    auto &refs = memoryPortRefs[entityId];
    refs.reserve(entity.memoryOperationPorts.size());
    for (std::uint64_t ordinal = 0;
         ordinal < entity.memoryOperationPorts.size(); ++ordinal)
      refs.push_back(FabricMemoryOperationPortRef{
          FabricMemoryOccurrenceRef(entityId), ordinal});
  }

  auto storage =
      std::make_shared<FabricArtifactView::Storage>(FabricArtifactView::Storage{
          std::move(data), std::move(memoryPortRefs),
          std::move(pointConnectionKeys), std::move(traversalKeys)});
  FabricArtifactView view(std::move(storage));
  for (const FabricPointConnectionPayload &connection :
       view.pointConnections()) {
    if (llvm::Error error = validateFabricRef(view, connection.source))
      return std::move(error);
    if (llvm::Error error = validateFabricRef(view, connection.destination))
      return std::move(error);
    if (view.transportEndpointDirection(connection.source) !=
        FabricPortDirection::Output)
      return invalidView("point connection source is not an output endpoint");
    if (view.transportEndpointDirection(connection.destination) !=
        FabricPortDirection::Input)
      return invalidView(
          "point connection destination is not an input endpoint");
    if (!haveSameTransportKind(
            view.transportEndpointType(connection.source),
            view.transportEndpointType(connection.destination)))
      return invalidView("point connection changes transport port kind");
  }
  for (const FabricPhysicalTraversalRef &traversal : view.admittedTraversals())
    if (llvm::Error error = validateFabricRef(view, traversal))
      return std::move(error);
  for (std::size_t id = 0; id < view.storage_->data.entities.size(); ++id) {
    if (view.entityKind(id) == FabricEntityKind::FabricFuTemplate) {
      const FabricFuTemplateRef owner(id);
      for (auto [ordinal, record] :
           llvm::enumerate(view.fuCapabilityTemplates(owner))) {
        FabricFuCapabilityTemplateRef ref{owner,
                                          static_cast<FabricOrdinal>(ordinal)};
        if (llvm::Error error = validateFabricRef(view, ref))
          return std::move(error);
        for (const FabricFuTemplateNodeRef &node : record.activeNodes)
          if (llvm::Error error = validateFabricRef(view, node))
            return std::move(error);
        for (const FabricFuCapabilityTemplateEdge &edge : record.activeEdges) {
          llvm::Error error = std::visit(
              [&](const auto &endpoint) {
                return validateFabricRef(view, endpoint);
              },
              edge.source.payload);
          if (error)
            return std::move(error);
          error = std::visit(
              [&](const auto &endpoint) {
                return validateFabricRef(view, endpoint);
              },
              edge.destination.payload);
          if (error)
            return std::move(error);
        }
      }
      continue;
    }

    if (view.entityKind(id) != FabricEntityKind::FabricMemoryOccurrence)
      continue;
    const FabricMemoryOccurrenceRef memory(id);
    for (FabricMemoryOperationPortRef port :
         view.memoryOperationPorts(memory)) {
      if (llvm::Error error = validateFabricRef(view, port))
        return std::move(error);
      const MemoryOperationPortView *record = view.memoryOperationPort(port);
      if (!record)
        return invalidView("memory operation-port record cannot be resolved");
      for (std::uint64_t ordinal = 0;
           ordinal < record->capabilityAlternatives().size(); ++ordinal) {
        FabricMemoryCapabilityAlternativeRef alternative{port, ordinal};
        if (llvm::Error error = validateFabricRef(view, alternative))
          return std::move(error);
        if (!view.memoryCapabilityAlternative(alternative))
          return invalidView(
              "memory capability-alternative record cannot be resolved");
      }
    }
  }
  return view;
}

namespace {

llvm::Error ordinalOutOfRange(llvm::StringRef what, FabricOrdinal ordinal,
                              std::uint64_t bound) {
  return makeFabricRefError(FabricRefErrorKind::OrdinalOutOfRange,
                            llvm::Twine(what) + " ordinal " +
                                llvm::Twine(ordinal) + " is outside [0, " +
                                llvm::Twine(bound) + ")");
}

/// Range-checks one owner-relative ordinal against the canonical inventory the
/// consuming family selects. The owner itself is validated first, so an
/// absent inventory and an invalid owner never blur together.
llvm::Error checkInventory(const FabricArtifactView &view,
                           const FabricInventoryOwnerRef &owner,
                           FabricInventoryKind inventory,
                           FabricOrdinal ordinal) {
  if (llvm::Error error = validateFabricRef(view, owner))
    return error;
  const std::uint64_t bound = view.inventorySize(owner, inventory);
  if (ordinal >= bound)
    return ordinalOutOfRange(fabricRefKeyword(inventory), ordinal, bound);
  return llvm::Error::success();
}

/// An in-range node ordinal still names exactly the node kind its owner's
/// configured graph declares there.
llvm::Error checkNode(const FabricArtifactView &view,
                      const FabricInventoryOwnerRef &owner,
                      FabricFuNodeKind node, FabricOrdinal ordinal) {
  if (llvm::Error error =
          checkInventory(view, owner, FabricInventoryKind::FuNode, ordinal))
    return error;
  const std::optional<FabricFuNodeKind> declared =
      view.fuNodeKind(owner, ordinal);
  if (!declared || *declared != node)
    return makeFabricRefError(FabricRefErrorKind::WrongEntityKind,
                              llvm::Twine("node ordinal ") +
                                  llvm::Twine(ordinal) + " is not a " +
                                  fabricRefKeyword(node) + " node");
  return llvm::Error::success();
}

FabricInventoryKind portInventory(FabricPortDirection direction) {
  return direction == FabricPortDirection::Input
             ? FabricInventoryKind::InputPort
             : FabricInventoryKind::OutputPort;
}

} // namespace

llvm::Error
loom::fabric::checkFabricBinding(const FabricArtifactView &view,
                                 const FabricImportBinding &binding) {
  if (view.identity() != binding.artifact)
    return makeFabricRefError(FabricRefErrorKind::ForeignArtifact,
                              "the bound Fabric artifact is not the one this "
                              "view resolves");
  if (view.rootKind() != binding.rootKind)
    return makeFabricRefError(
        FabricRefErrorKind::WrongRootKind,
        llvm::Twine("the bound Fabric root is ") +
            fabricRefKeyword(view.rootKind()) + " where " +
            fabricRefKeyword(binding.rootKind) + " is required");
  return llvm::Error::success();
}

llvm::Error loom::fabric::checkFabricBinding(const FabricArtifactView &view,
                                             const FabricImportBinding &binding,
                                             const ArtifactIdentity &encoded) {
  if (encoded != binding.artifact)
    return makeFabricRefError(FabricRefErrorKind::ForeignArtifact,
                              "the reference names a foreign Fabric artifact");
  return checkFabricBinding(view, binding);
}

llvm::Error loom::fabric::validateFabricEntity(const FabricArtifactView &view,
                                               FabricEntityKind kind,
                                               FabricEntityId id) {
  const std::optional<FabricEntityKind> actual = view.entityKind(id);
  if (!actual)
    return makeFabricRefError(FabricRefErrorKind::UnknownEntity,
                              llvm::Twine("no entity ") + llvm::Twine(id) +
                                  " in this Fabric artifact");
  if (*actual != kind)
    return makeFabricRefError(FabricRefErrorKind::WrongEntityKind,
                              llvm::Twine("entity ") + llvm::Twine(id) +
                                  " is " + fabricRefKeyword(*actual) +
                                  " where " + fabricRefKeyword(kind) +
                                  " is required");
  return llvm::Error::success();
}

//===---------------------------------------------------------------------===//
// Closed owner unions
//===---------------------------------------------------------------------===//

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricTransportEndpointOwnerRef &owner) {
  switch (owner.kind()) {
#define LOOM_FABRIC_TRANSPORT_OWNER(Ordinal, Name, Type)                       \
  case FabricTransportEndpointOwnerKind::Name:                                 \
    return validateFabricRef(view, std::get<Type>(owner.payload));
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::Error::success();
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricMemoryEndpointOwnerRef &owner) {
  switch (owner.kind()) {
#define LOOM_FABRIC_MEMORY_OWNER(Ordinal, Name, Type)                          \
  case FabricMemoryEndpointOwnerKind::Name:                                    \
    return validateFabricRef(view, std::get<Type>(owner.payload));
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::Error::success();
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricInventoryOwnerRef &owner) {
  switch (owner.kind()) {
#define LOOM_FABRIC_INVENTORY_OWNER(Name, Type)                                \
  case FabricInventoryOwnerKind::Name:                                         \
    return validateFabricRef(view, std::get<Type>(owner.payload));
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::Error::success();
}

//===---------------------------------------------------------------------===//
// Structural families
//===---------------------------------------------------------------------===//

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const SpatialCoreOccurrenceRef &ref) {
  return validateFabricRef(view, ref.core);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const InstructionCoreContextRef &ref) {
  return validateFabricRef(view, ref.core);
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const InstructionContextRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.pe),
                        FabricInventoryKind::InstructionContext, ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricModuleBoundaryEndpointRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.module),
                        portInventory(ref.direction), ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricFuTemplateNodeRef &ref) {
  return checkNode(view, FabricInventoryOwnerRef::of(ref.fu), ref.node,
                   ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricFuOccurrenceNodeRef &ref) {
  return checkNode(view, FabricInventoryOwnerRef::of(ref.fu), ref.node,
                   ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricFuTemplatePortRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.fu),
                        portInventory(ref.direction), ref.ordinal);
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const FabricFuNodePortRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.node))
    return error;
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.node),
                        portInventory(ref.direction), ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricFuOccurrencePortRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.fu),
                        portInventory(ref.direction), ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricTransportEndpointRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.owner))
    return error;
  const std::uint64_t bound = view.transportEndpointCount(ref.owner);
  if (ref.ordinal >= bound)
    return ordinalOutOfRange("transport endpoint", ref.ordinal, bound);
  return llvm::Error::success();
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricMemoryEndpointRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.owner))
    return error;
  const std::uint64_t bound = view.memoryEndpointCount(ref.owner);
  if (ref.ordinal >= bound)
    return ordinalOutOfRange("memory endpoint", ref.ordinal, bound);
  return llvm::Error::success();
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricMemoryOperationPortRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.memory),
                        FabricInventoryKind::MemoryOperationPort, ref.ordinal);
}

llvm::Error loom::fabric::validateFabricRef(
    const FabricArtifactView &view,
    const FabricMemoryCapabilityAlternativeRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.port))
    return error;
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.port),
                        FabricInventoryKind::MemoryCapabilityAlternative,
                        ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricMemoryOperationContextRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.port))
    return error;
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.port),
                        FabricInventoryKind::MemoryOperationContext,
                        ref.ordinal);
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const FabricMemoryServiceRef &ref) {
  switch (ref.kind()) {
  case FabricMemoryServiceKind::Local: {
    // The Local variant exists only where the memory occurrence declares its
    // optional Local Memory Service. This is the one place that rule lives,
    // so every nested region, owner, and refined use inherits it.
    const FabricMemoryOccurrenceRef memory =
        std::get<FabricMemoryOccurrenceRef>(ref.payload);
    if (llvm::Error error = validateFabricRef(view, memory))
      return error;
    if (!view.declaresLocalMemoryService(memory))
      return makeFabricRefError(FabricRefErrorKind::WrongEntityKind,
                                llvm::Twine("memory occurrence ") +
                                    llvm::Twine(memory.id()) +
                                    " declares no Local Memory Service");
    return llvm::Error::success();
  }
  case FabricMemoryServiceKind::System:
    return validateFabricRef(view,
                             std::get<SystemMemoryServiceRef>(ref.payload));
  }
  return llvm::Error::success();
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricMemoryServiceRegionRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.service),
                        FabricInventoryKind::MemoryServiceRegion, ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricTransferPatternRef &ref) {
  return checkInventory(view, FabricInventoryOwnerRef::of(ref.resource),
                        FabricInventoryKind::TransferPattern, ref.ordinal);
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricFuCapabilityTemplateRef &ref) {
  if (llvm::Error error = validateFabricRef(view, ref.fu))
    return error;
  const std::uint64_t bound = view.fuCapabilityTemplates(ref.fu).size();
  if (ref.ordinal >= bound)
    return ordinalOutOfRange("FU capability template", ref.ordinal, bound);
  return llvm::Error::success();
}

#define LOOM_FABRIC_OWNER_ROLE(Alias, Inventory, Family, Keyword)              \
  llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,  \
                                              const Family &ref) {             \
    return checkInventory(view, ref.owner.catalog(),                           \
                          FabricInventoryKind::Inventory, ref.ordinal);        \
  }
#include "Fabric/Identity/FabricRefs.def"

//===---------------------------------------------------------------------===//
// Directed physical traversals
//===---------------------------------------------------------------------===//

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const FabricPhysicalTraversalRef &ref) {
  // Every traversal first resolves its own structural fields, so an
  // out-of-range ordinal is never reported as a resource-contract failure.
  switch (ref.kind()) {
  case FabricPhysicalTraversalKind::PointConnection: {
    const FabricPointConnectionPayload &payload =
        std::get<FabricPointConnectionPayload>(ref.payload);
    if (llvm::Error error = validateFabricRef(view, payload.source))
      return error;
    if (llvm::Error error = validateFabricRef(view, payload.destination))
      return error;
    if (!view.hasPointConnection(payload.source, payload.destination))
      return makeFabricRefError(
          FabricRefErrorKind::AbsentPointConnection,
          "no unique directed fixed connection between these endpoints");
    return llvm::Error::success();
  }
  case FabricPhysicalTraversalKind::PeSelectorTraversal: {
    const FabricPeSelectorPayload &payload =
        std::get<FabricPeSelectorPayload>(ref.payload);
    if (llvm::Error error = validateFabricRef(view, payload.owner))
      return error;
    if (llvm::Error error = validateFabricRef(view, payload.source))
      return error;
    if (llvm::Error error = validateFabricRef(view, payload.destination))
      return error;
    break;
  }
  case FabricPhysicalTraversalKind::PeRegisterFifoTraversal: {
    const FabricPeRegisterFifoPayload &payload =
        std::get<FabricPeRegisterFifoPayload>(ref.payload);
    if (llvm::Error error = checkInventory(
            view, FabricInventoryOwnerRef::of(payload.owner),
            FabricInventoryKind::RegisterFifo, payload.registerFifo))
      return error;
    break;
  }
  case FabricPhysicalTraversalKind::SwitchTraversal: {
    const FabricSwitchTraversalPayload &payload =
        std::get<FabricSwitchTraversalPayload>(ref.payload);
    const FabricInventoryOwnerRef owner =
        FabricInventoryOwnerRef::of(payload.owner);
    if (llvm::Error error = checkInventory(
            view, owner, FabricInventoryKind::SwitchInput, payload.input))
      return error;
    if (llvm::Error error = checkInventory(
            view, owner, FabricInventoryKind::SwitchOutput, payload.output))
      return error;
    break;
  }
  case FabricPhysicalTraversalKind::FifoTraversal:
    if (llvm::Error error = validateFabricRef(
            view, std::get<FabricFifoTraversalPayload>(ref.payload).owner))
      return error;
    break;
  case FabricPhysicalTraversalKind::BoundaryTraversal: {
    const FabricBoundaryTraversalPayload &payload =
        std::get<FabricBoundaryTraversalPayload>(ref.payload);
    if (llvm::Error error =
            checkInventory(view, FabricInventoryOwnerRef::of(payload.owner),
                           FabricInventoryKind::BoundaryOutput, payload.output))
      return error;
    break;
  }
  case FabricPhysicalTraversalKind::SystemTransferPatternLeg: {
    const FabricTransferPatternLegPayload &payload =
        std::get<FabricTransferPatternLegPayload>(ref.payload);
    if (llvm::Error error = validateFabricRef(view, payload.owner))
      return error;
    if (llvm::Error error = checkInventory(
            view, FabricInventoryOwnerRef::of(payload.owner),
            FabricInventoryKind::TransferPatternEgress, payload.egress))
      return error;
    break;
  }
  }
  // The owning resource contract closes the remaining traversal alternatives:
  // a nonexistent switch turn, a bypass on a non-bypassable FIFO, or a
  // selector pair the PE does not expose.
  if (!view.admitsTraversal(ref))
    return makeFabricRefError(
        FabricRefErrorKind::TraversalNotAdmitted,
        llvm::Twine("the owning resource contract does not admit this ") +
            fabricRefKeyword(ref.kind()) + " traversal");
  return llvm::Error::success();
}

llvm::Expected<FabricFuOccurrenceNodeRef>
loom::fabric::deriveFabricFuOccurrenceNode(const FabricArtifactView &view,
                                           const FabricFuTemplateNodeRef &node,
                                           FabricFuOccurrenceRef occurrence) {
  if (llvm::Error error = validateFabricRef(view, node))
    return std::move(error);
  if (llvm::Error error = validateFabricRef(view, occurrence))
    return std::move(error);
  const std::optional<FabricFuTemplateRef> elaborated =
      view.fuTemplateOf(occurrence);
  if (!elaborated || *elaborated != node.fu)
    return makeFabricRefError(FabricRefErrorKind::WrongOwner,
                              llvm::Twine("FU occurrence ") +
                                  llvm::Twine(occurrence.id()) +
                                  " was not elaborated from FU template " +
                                  llvm::Twine(node.fu.id()));
  return FabricFuOccurrenceNodeRef{node.node, occurrence, node.ordinal};
}

//===---------------------------------------------------------------------===//
// Typed refinements
//===---------------------------------------------------------------------===//

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const LocalMemoryServiceRef &ref) {
  // The refined name only narrows the accepted variant; presence of the
  // service remains the generic reference's rule.
  if (ref.underlying().kind() != FabricMemoryServiceKind::Local)
    return makeFabricRefError(FabricRefErrorKind::WrongEntityKind,
                              "a local memory service reference selects the "
                              "System variant");
  return validateFabricRef(view, ref.underlying());
}

/// The owner inventory decides which refined endpoint name applies; the
/// reference never carries a copied role field.
static llvm::Error checkEndpointRole(const FabricArtifactView &view,
                                     const FabricMemoryEndpointRef &endpoint,
                                     FabricMemoryEndpointRole required) {
  if (llvm::Error error = validateFabricRef(view, endpoint))
    return error;
  const std::optional<FabricMemoryEndpointRole> declared =
      view.memoryEndpointRole(endpoint);
  if (!declared || *declared != required)
    return makeFabricRefError(FabricRefErrorKind::WrongEntityKind,
                              llvm::Twine("the owner inventory does not "
                                          "declare this endpoint ") +
                                  fabricRefKeyword(required));
  return llvm::Error::success();
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const ManagerEndpointRef &ref) {
  return checkEndpointRole(view, ref.underlying(),
                           FabricMemoryEndpointRole::Manager);
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const SubordinateEndpointRef &ref) {
  return checkEndpointRole(view, ref.underlying(),
                           FabricMemoryEndpointRole::Subordinate);
}

static llvm::Error checkHardwareDomainKind(const FabricArtifactView &view,
                                           HardwareDomainRef domain,
                                           FabricHardwareDomainKind required) {
  if (llvm::Error error = validateFabricRef(view, domain))
    return error;
  const std::optional<FabricHardwareDomainKind> declared =
      view.hardwareDomainKind(domain);
  if (!declared || *declared != required)
    return makeFabricRefError(FabricRefErrorKind::WrongEntityKind,
                              llvm::Twine("hardware domain ") +
                                  llvm::Twine(domain.id()) + " is not a " +
                                  fabricRefKeyword(required) + " domain");
  return llvm::Error::success();
}

llvm::Error
loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                const MemoryConsistencyDomainRef &ref) {
  return checkHardwareDomainKind(view, ref.underlying(),
                                 FabricHardwareDomainKind::MemoryConsistency);
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const ClockDomainRef &ref) {
  return checkHardwareDomainKind(view, ref.underlying(),
                                 FabricHardwareDomainKind::Clock);
}

llvm::Error loom::fabric::validateFabricRef(const FabricArtifactView &view,
                                            const ResetDomainRef &ref) {
  return checkHardwareDomainKind(view, ref.underlying(),
                                 FabricHardwareDomainKind::Reset);
}
