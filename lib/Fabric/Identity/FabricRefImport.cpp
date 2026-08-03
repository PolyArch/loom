#include "Fabric/Identity/FabricRefImport.h"

#include "Fabric/Artifact/FabricSystemRootView.h"

#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefText.h"
#include "FabricArtifactViewInternal.h"
#include "FabricArtifactViewStorage.h"
#include "FabricTraversalProjection.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::fabric;

FabricArtifactView::~FabricArtifactView() = default;

namespace {

std::uint64_t inventoryCount(llvm::ArrayRef<std::uint64_t> counts,
                             FabricInventoryKind kind) {
  const std::size_t index = static_cast<std::size_t>(kind);
  return index < counts.size() ? counts[index] : 0;
}

std::optional<std::uint32_t>
encodedUntaggedPayloadWidth(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 2 * sizeof(std::uint32_t))
    return std::nullopt;
  const auto readU32 = [](llvm::ArrayRef<std::uint8_t> value) {
    return (static_cast<std::uint32_t>(value[0]) << 24) |
           (static_cast<std::uint32_t>(value[1]) << 16) |
           (static_cast<std::uint32_t>(value[2]) << 8) |
           static_cast<std::uint32_t>(value[3]);
  };
  if (readU32(bytes.take_front(sizeof(std::uint32_t))) != 0)
    return std::nullopt;
  return readU32(bytes.drop_front(sizeof(std::uint32_t)));
}

std::optional<::fabric::DataPathType>
decodeTransportDataPath(llvm::ArrayRef<std::uint8_t> bytes) {
  constexpr std::size_t wordBytes = sizeof(std::uint32_t);
  const auto readU32 = [](llvm::ArrayRef<std::uint8_t> value) {
    return (static_cast<std::uint32_t>(value[0]) << 24) |
           (static_cast<std::uint32_t>(value[1]) << 16) |
           (static_cast<std::uint32_t>(value[2]) << 8) |
           static_cast<std::uint32_t>(value[3]);
  };
  if (bytes.size() != 2 * wordBytes && bytes.size() != 3 * wordBytes)
    return std::nullopt;
  const std::uint32_t kind = readU32(bytes.take_front(wordBytes));
  const std::uint32_t payload = readU32(bytes.slice(wordBytes, wordBytes));
  if (kind == 0 && bytes.size() == 2 * wordBytes)
    return ::fabric::DataPathType{::fabric::DataPathKind::Bits, payload, 0};
  if (kind == 1 && bytes.size() == 3 * wordBytes) {
    const std::uint32_t tag = readU32(bytes.drop_front(2 * wordBytes));
    ::fabric::DataPathType result{::fabric::DataPathKind::BitsTag, payload,
                                  tag};
    if (result.isWellFormed())
      return result;
  }
  return std::nullopt;
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
    if (!decodeTransportDataPath(endpoint.canonicalType))
      return invalidView(llvm::Twine(ownerDescription) +
                         " has a token endpoint without a canonical physical "
                         "data path");
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

llvm::ArrayRef<FabricArtifactView> FabricArtifactView::importedModules() const {
  return storage_->data.importedModules;
}

std::optional<FabricEntityKind>
FabricArtifactView::entityKind(FabricEntityId id) const {
  if (id >= storage_->data.entities.size())
    return std::nullopt;
  return storage_->data.entities[id].kind;
}

llvm::ArrayRef<FabricPeOccurrenceRef>
FabricArtifactView::peOccurrences() const {
  return storage_->peOccurrences;
}

llvm::ArrayRef<FabricFuOccurrenceRef>
FabricArtifactView::fuOccurrences() const {
  return storage_->fuOccurrences;
}

llvm::ArrayRef<FabricMemoryOccurrenceRef>
FabricArtifactView::memoryOccurrences() const {
  return storage_->memoryOccurrences;
}

llvm::ArrayRef<FabricSwitchOccurrenceRef>
FabricArtifactView::switchOccurrences() const {
  return storage_->switchOccurrences;
}

llvm::ArrayRef<FabricFifoOccurrenceRef>
FabricArtifactView::fifoOccurrences() const {
  return storage_->fifoOccurrences;
}

llvm::ArrayRef<FabricBoundaryOccurrenceRef>
FabricArtifactView::boundaryOccurrences() const {
  return storage_->boundaryOccurrences;
}

std::optional<FabricBoundaryTagContinuityPointView>
FabricArtifactView::boundaryTagContinuityPoint(
    FabricBoundaryOccurrenceRef boundary) const {
  const detail::FabricEntityViewData *entity = storage_->entity(boundary);
  if (!entity || entity->kind != FabricEntityKind::FabricBoundaryOccurrence)
    return std::nullopt;

  std::array<::fabric::DataPathType, 2> inputs{};
  std::array<::fabric::DataPathType, 2> outputs{};
  std::size_t inputCount = 0;
  std::size_t outputCount = 0;
  for (const detail::FabricTransportEndpointViewData &endpoint :
       entity->owner.transportEndpoints) {
    const auto dataPath = decodeTransportDataPath(endpoint.canonicalType);
    if (!dataPath)
      return std::nullopt;
    if (endpoint.direction == FabricPortDirection::Input) {
      if (inputCount == inputs.size())
        return std::nullopt;
      inputs[inputCount++] = *dataPath;
    } else if (endpoint.direction == FabricPortDirection::Output) {
      if (outputCount == outputs.size())
        return std::nullopt;
      outputs[outputCount++] = *dataPath;
    } else {
      return std::nullopt;
    }
  }
  if (inputCount == 0 || outputCount == 0)
    return std::nullopt;

  const ::fabric::DataPathType &input = inputs.front();
  const ::fabric::DataPathType &output = outputs.front();
  if (input.kind == ::fabric::DataPathKind::Bits &&
      output.kind == ::fabric::DataPathKind::BitsTag) {
    if (outputCount != 1)
      return std::nullopt;
    if (inputCount == 2 && (inputs[1].kind != ::fabric::DataPathKind::Bits ||
                            inputs[1].payloadWidthBits != output.tagWidthBits))
      return std::nullopt;
    return FabricBoundaryTagContinuityPointView{
        inputCount == 2 ? FabricBoundaryTagContinuityKind::TokenWriter
                        : FabricBoundaryTagContinuityKind::ConfigurableWriter,
        0, output.tagWidthBits};
  }
  if (input.kind == ::fabric::DataPathKind::BitsTag &&
      output.kind == ::fabric::DataPathKind::BitsTag) {
    if (inputCount != 1 || outputCount != 1)
      return std::nullopt;
    return FabricBoundaryTagContinuityPointView{
        FabricBoundaryTagContinuityKind::Rewriter, input.tagWidthBits,
        output.tagWidthBits};
  }
  if (input.kind == ::fabric::DataPathKind::BitsTag &&
      output.kind == ::fabric::DataPathKind::Bits) {
    if (inputCount != 1)
      return std::nullopt;
    if (outputCount == 2 && (outputs[1].kind != ::fabric::DataPathKind::Bits ||
                             outputs[1].payloadWidthBits != input.tagWidthBits))
      return std::nullopt;
    return FabricBoundaryTagContinuityPointView{
        FabricBoundaryTagContinuityKind::Remover, input.tagWidthBits, 0};
  }
  return std::nullopt;
}

llvm::ArrayRef<FabricTransportEndpointRef>
FabricArtifactView::transportEndpoints() const {
  return storage_->transportEndpoints;
}

std::optional<::fabric::DataPathType>
FabricArtifactView::transportEndpointDataPath(
    const FabricTransportEndpointRef &endpoint) const {
  const detail::FabricTransportEndpointViewData *record =
      storage_->transportEndpoint(endpoint);
  return record ? decodeTransportDataPath(record->canonicalType) : std::nullopt;
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

std::optional<FabricTransportEndpointRef>
FabricArtifactView::fuOccurrenceTransportEndpoint(
    FabricFuOccurrencePortRef port) const {
  const FabricTransportEndpointOwnerRef owner =
      FabricTransportEndpointOwnerRef::of(port.fu);
  std::uint64_t directionOrdinal = 0;
  for (std::uint64_t ordinal = 0; ordinal < transportEndpointCount(owner);
       ++ordinal) {
    const FabricTransportEndpointRef endpoint{owner, ordinal};
    const auto direction = transportEndpointDirection(endpoint);
    if (!direction || *direction != port.direction)
      continue;
    if (directionOrdinal++ == port.ordinal)
      return endpoint;
  }
  return std::nullopt;
}

llvm::ArrayRef<FabricFuPortAttachmentView>
FabricArtifactView::fuOccurrencePortAttachments(
    FabricFuOccurrencePortRef port) const {
  const auto fixed = fuOccurrenceTransportEndpoint(port);
  if (!fixed)
    return {};
  auto found = storage_->fuPortAttachments.find(canonicalFabricBytes(*fixed));
  return found == storage_->fuPortAttachments.end()
             ? llvm::ArrayRef<FabricFuPortAttachmentView>()
             : llvm::ArrayRef<FabricFuPortAttachmentView>(found->second);
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
  if (inventory == FabricInventoryKind::FuNode &&
      owner.kind() == FabricInventoryOwnerKind::FuOccurrence) {
    const auto *nodes =
        storage_->fuNodes(std::get<FabricFuOccurrenceRef>(owner.payload));
    return nodes ? nodes->size() : 0;
  }
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

llvm::ArrayRef<FabricInventoryOwnerRef>
FabricArtifactView::moduleResourceOwners() const {
  return storage_->moduleResourceOwners;
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

const ::fabric::MemoryServiceContractRecord *
FabricArtifactView::localMemoryService(FabricMemoryOccurrenceRef memory) const {
  const detail::FabricEntityViewData *record = storage_->entity(memory);
  return record && record->localMemoryService
             ? &record->localMemoryService->record
             : nullptr;
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

llvm::ArrayRef<FabricModuleBoundaryTransportAttachmentView>
FabricArtifactView::moduleBoundaryTransportAttachments() const {
  return storage_->data.moduleBoundaryTransportAttachments;
}

std::optional<FabricHardwareDomainKind>
FabricArtifactView::hardwareDomainKind(HardwareDomainRef domain) const {
  const detail::FabricEntityViewData *record = storage_->entity(domain);
  return record ? record->hardwareDomainKind : std::nullopt;
}

std::optional<FabricPeOccurrenceRef>
FabricArtifactView::parentPeOf(FabricFuOccurrenceRef occurrence) const {
  const detail::FabricEntityViewData *record = storage_->entity(occurrence);
  return record ? record->parentPe : std::nullopt;
}

std::optional<::fabric::Schedule>
FabricArtifactView::peSchedule(FabricPeOccurrenceRef occurrence) const {
  const detail::FabricEntityViewData *record = storage_->entity(occurrence);
  return record ? record->peSchedule : std::nullopt;
}

std::uint64_t FabricArtifactView::peResidentContextCount(
    FabricPeOccurrenceRef occurrence) const {
  const detail::FabricEntityViewData *record = storage_->entity(occurrence);
  return record ? record->instructionContexts.size() : 0;
}

std::optional<FabricFuTemplateRef>
FabricArtifactView::fuTemplateOf(FabricFuOccurrenceRef occurrence) const {
  const detail::FabricEntityViewData *record = storage_->entity(occurrence);
  return record ? record->fuTemplate : std::nullopt;
}

llvm::ArrayRef<FabricFuTemplateRef> FabricArtifactView::fuTemplates() const {
  return storage_->fuTemplates;
}

llvm::ArrayRef<FabricMemoryEngineTemplateRef>
FabricArtifactView::memoryEngineTemplates() const {
  return storage_->memoryEngineTemplates;
}

llvm::ArrayRef<FabricFuCapabilityTemplateRecord>
FabricArtifactView::fuCapabilityTemplates(
    FabricFuTemplateRef definition) const {
  const auto *records = storage_->fuCapabilityTemplates(definition);
  return records ? llvm::ArrayRef<FabricFuCapabilityTemplateRecord>(*records)
                 : llvm::ArrayRef<FabricFuCapabilityTemplateRecord>();
}

const ResolvedFabricOpCapabilityView *
FabricArtifactView::resolvedFabricOpCapability(
    const FabricFuTemplateNodeRef &operation) const {
  return storage_->operationCapability(operation);
}

const ResolvedFabricOpCapabilityView *
FabricArtifactView::resolvedFabricOpCapability(
    const FabricFuOccurrenceNodeRef &operation) const {
  const detail::FabricEntityViewData *occurrence =
      storage_->entity(operation.fu);
  if (!occurrence || !occurrence->fuTemplate)
    return nullptr;
  return storage_->operationCapability(FabricFuTemplateNodeRef{
      operation.node, *occurrence->fuTemplate, operation.ordinal});
}

llvm::ArrayRef<ResolvedFabricOpCapabilityView>
FabricArtifactView::resolvedFabricOpCapabilities(
    FabricFuTemplateRef definition) const {
  const detail::FabricEntityViewData *record = storage_->entity(definition);
  return record ? llvm::ArrayRef<ResolvedFabricOpCapabilityView>(
                      record->operationCapabilities)
                : llvm::ArrayRef<ResolvedFabricOpCapabilityView>();
}

std::optional<FabricMemoryEngineTemplateRef>
FabricArtifactView::memoryEngineTemplateOf(
    FabricMemoryOccurrenceRef occurrence) const {
  const detail::FabricEntityViewData *record = storage_->entity(occurrence);
  return record ? record->memoryEngineTemplate : std::nullopt;
}

const FabricMemoryEngineTemplateRecord *
FabricArtifactView::memoryEngineTemplate(
    FabricMemoryEngineTemplateRef definition) const {
  const detail::FabricEntityViewData *record = storage_->entity(definition);
  return record && record->memoryEngineTemplateRecord
             ? &*record->memoryEngineTemplateRecord
             : nullptr;
}

const MemoryOperationPortView *
FabricArtifactView::memoryEngineTemplateOperationPort(
    FabricMemoryEngineTemplateOperationPortRef port) const {
  const FabricMemoryEngineTemplateRecord *engine =
      memoryEngineTemplate(port.engine);
  if (!engine || port.ordinal >= engine->operationPorts.size())
    return nullptr;
  return &engine->operationPorts[port.ordinal];
}

const MemoryCapabilityAlternativeView *
FabricArtifactView::memoryEngineTemplateCapabilityAlternative(
    FabricMemoryEngineTemplateCapabilityAlternativeRef alternative) const {
  const MemoryOperationPortView *port =
      memoryEngineTemplateOperationPort(alternative.port);
  if (!port || alternative.ordinal >= port->capabilityAlternatives().size())
    return nullptr;
  return &port->capabilityAlternatives()[alternative.ordinal];
}

const ::fabric::MemoryTransportEndpointDescriptor *
FabricArtifactView::memoryEngineTemplateEndpoint(
    FabricMemoryEngineTemplateEndpointRef endpoint) const {
  const FabricMemoryEngineTemplateRecord *engine =
      memoryEngineTemplate(endpoint.engine);
  if (!engine || endpoint.ordinal >= engine->tokenEndpoints.size())
    return nullptr;
  return &engine->tokenEndpoints[endpoint.ordinal];
}

bool FabricArtifactView::hasMemoryEngineTemplateInternalConnection(
    const FabricMemoryEngineTemplateInternalConnectionRef &connection) const {
  const FabricMemoryEngineTemplateRecord *engine =
      memoryEngineTemplate(connection.engine);
  if (!engine || connection.source.engine != connection.engine ||
      connection.sink.engine != connection.engine)
    return false;
  return llvm::any_of(
      engine->internalConnections,
      [&](const ::fabric::MemoryInternalConnectionDeclaration &candidate) {
        return candidate.sourceEndpointOrdinal == connection.source.ordinal &&
               candidate.sinkEndpointOrdinal == connection.sink.ordinal;
      });
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

llvm::ArrayRef<FabricPhysicalTraversalView>
FabricArtifactView::physicalTraversals() const {
  return storage_->physicalTraversalViews;
}

llvm::ArrayRef<FabricSpatialAttachmentRecordView>
FabricSystemRootView::spatialAttachments() const {
  return artifact_.storage_->data.spatialAttachments;
}

const InstructionCoreArchitecturalContract *
FabricSystemRootView::instructionCoreArchitecture(
    HostCoreOccurrenceRef core) const {
  const detail::FabricEntityViewData *entity = artifact_.storage_->entity(core);
  return entity && entity->instructionCoreArchitecture
             ? &*entity->instructionCoreArchitecture
             : nullptr;
}

const InstructionCoreArchitecturalContract *
FabricSystemRootView::instructionCoreArchitecture(
    InstructionCoreContextRef core) const {
  const detail::FabricEntityViewData *entity =
      artifact_.storage_->entity(core.core);
  return entity && entity->instructionCoreArchitecture
             ? &*entity->instructionCoreArchitecture
             : nullptr;
}

const InstructionCoreMicroarchitecturalRealization *
FabricSystemRootView::instructionCoreMicroarchitecture(
    HostCoreOccurrenceRef core) const {
  const detail::FabricEntityViewData *entity = artifact_.storage_->entity(core);
  return entity && entity->instructionCoreMicroarchitecture
             ? &*entity->instructionCoreMicroarchitecture
             : nullptr;
}

const InstructionCoreMicroarchitecturalRealization *
FabricSystemRootView::instructionCoreMicroarchitecture(
    InstructionCoreContextRef core) const {
  const detail::FabricEntityViewData *entity =
      artifact_.storage_->entity(core.core);
  return entity && entity->instructionCoreMicroarchitecture
             ? &*entity->instructionCoreMicroarchitecture
             : nullptr;
}

std::optional<FabricImportedModuleTargetRef>
FabricSystemRootView::spatialCoreTarget(AccCoreOccurrenceRef core) const {
  const detail::FabricEntityViewData *entity = artifact_.storage_->entity(core);
  return entity ? entity->spatialCoreTarget : std::nullopt;
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

    std::vector<bool> referencedOperationCapabilities(
        entity.operationCapabilities.size(), false);
    for (auto [nodeOrdinal, node] : llvm::enumerate(entity.fuNodes)) {
      if (!validClosedValue(node.kind))
        return invalidView("FU node has an unknown kind");
      if (llvm::Error error = validateNestedOwner(node.owner, "FU node"))
        return std::move(error);
      const bool operationNode = node.kind == FabricFuNodeKind::Op;
      if (operationNode != node.operationCapabilityIndex.has_value())
        return invalidView(
            "FU operation node and resolved capability do not correspond");
      if (!operationNode)
        continue;
      if (*node.operationCapabilityIndex >= entity.operationCapabilities.size())
        return invalidView("FU operation capability index is out of range");
      if (referencedOperationCapabilities[*node.operationCapabilityIndex])
        return invalidView("FU operation capability is referenced twice");
      referencedOperationCapabilities[*node.operationCapabilityIndex] = true;

      const ResolvedFabricOpCapabilityView &capability =
          entity.operationCapabilities[*node.operationCapabilityIndex];
      const FabricFuTemplateNodeRef expectedReference{
          FabricFuNodeKind::Op, FabricFuTemplateRef(index), nodeOrdinal};
      if (capability.occurrence != expectedReference)
        return invalidView("FU operation capability has the wrong owner");
      if (capability.enabledOperationSchemas.empty())
        return invalidView("FU operation capability has no enabled schema");
      std::set<::dataflow::OperationSchemaId> enabled;
      for (::dataflow::OperationSchemaId schema :
           capability.enabledOperationSchemas) {
        if (!enabled.insert(schema).second)
          return invalidView(
              "FU operation capability has a duplicate enabled schema");
        if (!::fabric::admitsOperationSchema(capability.implementationFamily,
                                             schema))
          return invalidView(
              "FU operation capability escapes its implementation family");
      }
      if (::fabric::capabilityParamsSchema(
              capability.parameterizedCapability) !=
          ::fabric::implementationFamily(capability.implementationFamily)
              .capabilityParamsSchema)
        return invalidView(
            "FU operation capability has the wrong parameter schema");
      if (capability.physicalPorts.size() !=
          node.owner.transportEndpoints.size())
        return invalidView(
            "FU operation capability has an incomplete physical port view");
      FabricOrdinal expectedInputOrdinal = 0;
      FabricOrdinal expectedOutputOrdinal = 0;
      for (auto [portOrdinal, port] :
           llvm::enumerate(capability.physicalPorts)) {
        const FabricTransportEndpointViewData &endpoint =
            node.owner.transportEndpoints[portOrdinal];
        std::optional<std::uint32_t> encodedWidth =
            encodedUntaggedPayloadWidth(endpoint.canonicalType);
        const FabricOrdinal expectedPortOrdinal =
            endpoint.direction == FabricPortDirection::Input
                ? expectedInputOrdinal++
                : expectedOutputOrdinal++;
        if (port.reference.node != expectedReference ||
            port.reference.direction != endpoint.direction ||
            port.reference.ordinal != expectedPortOrdinal ||
            port.canonicalType != endpoint.canonicalType || !encodedWidth ||
            port.payloadWidthBits != *encodedWidth)
          return invalidView(
              "FU operation capability physical port does not match Fabric");
      }
      const FabricInventoryOwnerRef expectedOwner =
          FabricInventoryOwnerRef::of(expectedReference);
      for (auto [ordinal, field] :
           llvm::enumerate(capability.configurationFieldSchema))
        if (field.owner.catalog() != expectedOwner || field.ordinal != ordinal)
          return invalidView(
              "FU operation capability has a noncanonical configuration "
              "field reference");
      for (auto [ordinal, refinement] :
           llvm::enumerate(capability.physicalRefinementDomains))
        if (refinement.owner.catalog() != expectedOwner ||
            refinement.ordinal != ordinal)
          return invalidView(
              "FU operation capability has a noncanonical refinement "
              "reference");
      auto nodeContract =
          ::fabric::encodeResourceContractRecord(*node.owner.resourceContract);
      if (!nodeContract)
        return nodeContract.takeError();
      auto capabilityContract = ::fabric::encodeResourceContractRecord(
          capability.resourceStateAndTimingContract);
      if (!capabilityContract)
        return capabilityContract.takeError();
      if (*nodeContract != *capabilityContract)
        return invalidView(
            "FU operation capability has a different resource contract");
    }
    if (llvm::is_contained(referencedOperationCapabilities, false))
      return invalidView("FU operation capability is unreachable");
    if (entity.kind == FabricEntityKind::FabricFuTemplate) {
      auto normalized = normalizeFabricFuCapabilityTemplateInventory(
          entity.fuCapabilityTemplates);
      if (!normalized)
        return normalized.takeError();
      if (*normalized != entity.fuCapabilityTemplates)
        return invalidView("FU capability-template inventory is not canonical");
    } else if (!entity.fuCapabilityTemplates.empty() ||
               !entity.operationCapabilities.empty()) {
      return invalidView(
          "only an FU template may own operation capability records");
    }
    for (FabricMemoryOperationPortViewData &port : entity.memoryOperationPorts)
      if (llvm::Error error =
              validateNestedOwner(port.owner, "memory operation port"))
        return std::move(error);
    const bool isMemory =
        entity.kind == FabricEntityKind::FabricMemoryOccurrence;
    const bool isMemoryTemplate =
        entity.kind == FabricEntityKind::FabricMemoryEngineTemplate;
    if (entity.memoryEngineTemplate && !isMemory)
      return invalidView(
          "only a memory occurrence may select a memory engine template");
    if (entity.memoryEngineTemplateRecord.has_value() != isMemoryTemplate)
      return invalidView(
          "memory engine template entity and record do not correspond");
    if (entity.memoryEngineTemplateProjection && !isMemory && !isMemoryTemplate)
      return invalidView(
          "unrelated entity owns a memory engine template projection");
    if (isMemoryTemplate) {
      if (!entity.memoryEngineTemplateProjection)
        return invalidView(
            "memory engine template has no canonical projection");
      const FabricMemoryEngineTemplateRecord &engine =
          *entity.memoryEngineTemplateRecord;
      if (engine.schedule != ::fabric::Schedule::Spatial &&
          engine.schedule != ::fabric::Schedule::Temporal)
        return invalidView("memory engine template has an unknown schedule");
      if (engine.schedule == ::fabric::Schedule::Spatial) {
        if (engine.residentContextCount)
          return invalidView(
              "spatial memory engine template owns resident contexts");
      } else if (!engine.residentContextCount ||
                 *engine.residentContextCount == 0) {
        return invalidView(
            "temporal memory engine template has no resident contexts");
      }
      if (engine.operationPorts.empty())
        return invalidView("memory engine template has no operation ports");
      for (const ::fabric::MemoryTransportEndpointDescriptor &endpoint :
           engine.tokenEndpoints)
        if (!validClosedValue(endpoint.direction))
          return invalidView(
              "memory engine template endpoint has an unknown direction");
      std::optional<std::pair<std::uint64_t, std::uint64_t>> previous;
      for (const ::fabric::MemoryInternalConnectionDeclaration &connection :
           engine.internalConnections) {
        if (connection.sourceEndpointOrdinal >= engine.tokenEndpoints.size() ||
            connection.sinkEndpointOrdinal >= engine.tokenEndpoints.size())
          return invalidView(
              "memory engine template internal connection is out of range");
        const auto &source =
            engine.tokenEndpoints[connection.sourceEndpointOrdinal];
        const auto &sink =
            engine.tokenEndpoints[connection.sinkEndpointOrdinal];
        if (source.direction != FabricPortDirection::Output ||
            sink.direction != FabricPortDirection::Input)
          return invalidView(
              "memory engine template internal connection has the wrong "
              "direction");
        if (source.tagWidth.has_value() != sink.tagWidth.has_value())
          return invalidView(
              "memory engine template internal connection crosses token "
              "kinds");
        if (source.payloadWidth < sink.payloadWidth)
          return invalidView(
              "memory engine template internal connection narrows its "
              "source capacity");
        const std::pair<std::uint64_t, std::uint64_t> current{
            connection.sourceEndpointOrdinal, connection.sinkEndpointOrdinal};
        if (previous && *previous >= current)
          return invalidView(
              "memory engine template internal connections are not canonical");
        previous = current;
      }
    }
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
      const bool hasEngine = entity.memorySchedule.has_value();
      if (entity.memoryEngineTemplate.has_value() != hasEngine ||
          entity.memoryEngineTemplateProjection.has_value() != hasEngine)
        return invalidView(
            "memory occurrence and engine template relation do not "
            "correspond");
      if (entity.memoryEngineTemplate) {
        const FabricEntityId templateId = entity.memoryEngineTemplate->id();
        if (templateId >= data.entities.size() ||
            data.entities[templateId].kind !=
                FabricEntityKind::FabricMemoryEngineTemplate ||
            !data.entities[templateId].memoryEngineTemplateRecord ||
            !data.entities[templateId].memoryEngineTemplateProjection)
          return invalidView(
              "memory occurrence selects an invalid engine template");
        if (*entity.memoryEngineTemplateProjection !=
            *data.entities[templateId].memoryEngineTemplateProjection)
          return invalidView(
              "memory occurrence disagrees with its engine template");
      }
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
    if (entity.spatialCoreTarget) {
      if (entity.kind != FabricEntityKind::AccCoreOccurrence ||
          !entity.spatialCore)
        return invalidView(
            "only an AccCore SpatialCore may select an imported Module");
      if (entity.spatialCoreTarget->dependencyOrdinal >=
          data.importedModules.size())
        return invalidView(
            "AccCore SpatialCore target has no imported Module dependency");
      const FabricArtifactView &module =
          data.importedModules[entity.spatialCoreTarget->dependencyOrdinal];
      if (module.rootKind() != FabricRootKind::Module ||
          module.entityKind(entity.spatialCoreTarget->target.id()) !=
              FabricEntityKind::FabricModuleTemplate)
        return invalidView(
            "AccCore SpatialCore target is not an imported Module template");
    } else if (entity.kind == FabricEntityKind::AccCoreOccurrence) {
      return invalidView("AccCore has no imported SpatialCore target");
    }
    if (entity.instructionCore)
      if (llvm::Error error =
              validateNestedOwner(*entity.instructionCore, "instruction core"))
        return std::move(error);
    if (entity.localMemoryService) {
      if (llvm::Error error = validateNestedOwner(
              entity.localMemoryService->owner, "local memory service"))
        return std::move(error);
      if (inventoryCount(entity.localMemoryService->owner.inventoryCounts,
                         FabricInventoryKind::MemoryServiceRegion) !=
          entity.localMemoryService->record.regions().size())
        return invalidView(
            "local memory service region inventory does not match its record");
      if (!entity.localMemoryService->owner.resourceContract)
        return invalidView(
            "local memory service has no derived resource contract");
      auto derivedResource = ::fabric::encodeResourceContractRecord(
          *entity.localMemoryService->owner.resourceContract);
      if (!derivedResource)
        return derivedResource.takeError();
      auto ownerResource = ::fabric::encodeResourceContractRecord(
          entity.localMemoryService->record.resourceContract());
      if (!ownerResource)
        return ownerResource.takeError();
      if (*derivedResource != *ownerResource)
        return invalidView(
            "local memory service resource contract is not owner-exact");
    }

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

    const bool isPe = entity.kind == FabricEntityKind::FabricPeOccurrence;
    if (entity.peSchedule.has_value() != isPe)
      return invalidView(
          "PE occurrence and scheduling projection do not correspond");
    if (isPe) {
      if (*entity.peSchedule == ::fabric::Schedule::Spatial) {
        if (entity.instructionContexts.size() != 1)
          return invalidView(
              "spatial PE occurrence does not own its sole context");
      } else if (*entity.peSchedule == ::fabric::Schedule::Temporal) {
        if (entity.instructionContexts.empty())
          return invalidView("temporal PE occurrence has no resident contexts");
      } else {
        return invalidView("PE occurrence has an unknown schedule");
      }
    }

    const bool isFuOccurrence =
        entity.kind == FabricEntityKind::FabricFuOccurrence;
    if (entity.parentPe.has_value() != isFuOccurrence)
      return invalidView(
          "FU occurrence and owning PE relation do not correspond");
    if (entity.parentPe) {
      const FabricEntityId parentId = entity.parentPe->id();
      if (parentId >= data.entities.size() ||
          data.entities[parentId].kind != FabricEntityKind::FabricPeOccurrence)
        return invalidView("FU occurrence selects an invalid owning PE");
    }

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

  if (data.rootKind != FabricRootKind::Module &&
      !data.moduleBoundaryTransportAttachments.empty())
    return invalidView(
        "only a Module root may expose boundary transport attachments");
  std::sort(data.moduleBoundaryTransportAttachments.begin(),
            data.moduleBoundaryTransportAttachments.end(),
            [](const auto &lhs, const auto &rhs) {
              const auto lhsBoundary = canonicalFabricBytes(lhs.boundary);
              const auto rhsBoundary = canonicalFabricBytes(rhs.boundary);
              if (lhsBoundary != rhsBoundary)
                return lhsBoundary < rhsBoundary;
              return canonicalFabricBytes(lhs.endpoint) <
                     canonicalFabricBytes(rhs.endpoint);
            });
  for (std::size_t index = 1;
       index < data.moduleBoundaryTransportAttachments.size(); ++index)
    if (data.moduleBoundaryTransportAttachments[index - 1].boundary ==
        data.moduleBoundaryTransportAttachments[index].boundary)
      return invalidView(
          "a Module boundary has more than one transport attachment");

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
  std::vector<FabricPeOccurrenceRef> peOccurrences;
  std::vector<FabricFuOccurrenceRef> fuOccurrences;
  std::vector<FabricMemoryOccurrenceRef> memoryOccurrences;
  std::vector<FabricSwitchOccurrenceRef> switchOccurrences;
  std::vector<FabricFifoOccurrenceRef> fifoOccurrences;
  std::vector<FabricBoundaryOccurrenceRef> boundaryOccurrences;
  std::vector<FabricTransportEndpointRef> transportEndpoints;
  std::vector<FabricFuTemplateRef> fuTemplates;
  std::vector<FabricMemoryEngineTemplateRef> memoryEngineTemplates;
  for (auto [entityId, entity] : llvm::enumerate(data.entities)) {
    std::optional<FabricTransportEndpointOwnerRef> transportOwner;
    switch (entity.kind) {
    case FabricEntityKind::FabricPeOccurrence:
      peOccurrences.emplace_back(entityId);
      transportOwner =
          FabricTransportEndpointOwnerRef::of(FabricPeOccurrenceRef(entityId));
      break;
    case FabricEntityKind::FabricFuOccurrence:
      fuOccurrences.emplace_back(entityId);
      transportOwner =
          FabricTransportEndpointOwnerRef::of(FabricFuOccurrenceRef(entityId));
      break;
    case FabricEntityKind::FabricMemoryOccurrence:
      memoryOccurrences.emplace_back(entityId);
      transportOwner = FabricTransportEndpointOwnerRef::of(
          FabricMemoryOccurrenceRef(entityId));
      break;
    case FabricEntityKind::FabricSwitchOccurrence:
      switchOccurrences.emplace_back(entityId);
      transportOwner = FabricTransportEndpointOwnerRef::of(
          FabricSwitchOccurrenceRef(entityId));
      break;
    case FabricEntityKind::FabricFifoOccurrence:
      fifoOccurrences.emplace_back(entityId);
      transportOwner = FabricTransportEndpointOwnerRef::of(
          FabricFifoOccurrenceRef(entityId));
      break;
    case FabricEntityKind::FabricBoundaryOccurrence:
      boundaryOccurrences.emplace_back(entityId);
      transportOwner = FabricTransportEndpointOwnerRef::of(
          FabricBoundaryOccurrenceRef(entityId));
      break;
    case FabricEntityKind::SystemServiceEndpoint:
      transportOwner = FabricTransportEndpointOwnerRef::of(
          SystemServiceEndpointRef(entityId));
      break;
    case FabricEntityKind::SystemTransportResource:
      transportOwner = FabricTransportEndpointOwnerRef::of(
          SystemTransportResourceRef(entityId));
      break;
    default:
      break;
    }
    if (transportOwner)
      for (FabricOrdinal ordinal = 0;
           ordinal < entity.owner.transportEndpoints.size(); ++ordinal)
        transportEndpoints.push_back({*transportOwner, ordinal});
    if (entity.spatialCore) {
      const auto owner = FabricTransportEndpointOwnerRef::of(
          SpatialCoreOccurrenceRef{AccCoreOccurrenceRef(entityId)});
      for (FabricOrdinal ordinal = 0;
           ordinal < entity.spatialCore->transportEndpoints.size(); ++ordinal)
        transportEndpoints.push_back({owner, ordinal});
    }

    if (entity.kind == FabricEntityKind::FabricFuTemplate)
      fuTemplates.emplace_back(entityId);
    if (entity.kind == FabricEntityKind::FabricMemoryEngineTemplate)
      memoryEngineTemplates.emplace_back(entityId);
    if (entity.kind != FabricEntityKind::FabricMemoryOccurrence)
      continue;
    auto &refs = memoryPortRefs[entityId];
    refs.reserve(entity.memoryOperationPorts.size());
    for (std::uint64_t ordinal = 0;
         ordinal < entity.memoryOperationPorts.size(); ++ordinal)
      refs.push_back(FabricMemoryOperationPortRef{
          FabricMemoryOccurrenceRef(entityId), ordinal});
  }
  std::sort(transportEndpoints.begin(), transportEndpoints.end(),
            [](const FabricTransportEndpointRef &lhs,
               const FabricTransportEndpointRef &rhs) {
              return canonicalFabricBytes(lhs) < canonicalFabricBytes(rhs);
            });

  auto storage = std::make_shared<FabricArtifactView::Storage>(std::move(data));
  storage->peOccurrences = std::move(peOccurrences);
  storage->fuOccurrences = std::move(fuOccurrences);
  storage->memoryOccurrences = std::move(memoryOccurrences);
  storage->switchOccurrences = std::move(switchOccurrences);
  storage->fifoOccurrences = std::move(fifoOccurrences);
  storage->boundaryOccurrences = std::move(boundaryOccurrences);
  storage->transportEndpoints = std::move(transportEndpoints);
  storage->fuTemplates = std::move(fuTemplates);
  storage->memoryEngineTemplates = std::move(memoryEngineTemplates);
  storage->memoryPortRefs = std::move(memoryPortRefs);
  storage->pointConnectionKeys = std::move(pointConnectionKeys);
  storage->traversalKeys = std::move(traversalKeys);
  FabricArtifactView view(storage);
  auto moduleResourceOwners = detail::projectModuleResourceOwners(view);
  if (!moduleResourceOwners)
    return moduleResourceOwners.takeError();
  storage->moduleResourceOwners = std::move(*moduleResourceOwners);
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
  std::set<std::vector<std::uint8_t>> attachedTransportEndpoints;
  for (const FabricModuleBoundaryTransportAttachmentView &attachment :
       view.moduleBoundaryTransportAttachments()) {
    if (llvm::Error error = validateFabricRef(view, attachment.boundary))
      return std::move(error);
    if (llvm::Error error = validateFabricRef(view, attachment.endpoint))
      return std::move(error);
    if (view.moduleBoundaryEndpointPlane(attachment.boundary) !=
        FabricSpatialAttachmentEndpointRef::Plane::Transport)
      return invalidView("a memory Module boundary has a transport attachment");
    if (view.transportEndpointDirection(attachment.endpoint) !=
        attachment.boundary.direction)
      return invalidView("a Module boundary attachment changes direction");
    if (!haveSameTransportKind(
            view.moduleBoundaryEndpointType(attachment.boundary),
            view.transportEndpointType(attachment.endpoint)))
      return invalidView("a Module boundary attachment changes transport kind");
    if (!attachedTransportEndpoints
             .insert(canonicalFabricBytes(attachment.endpoint))
             .second)
      return invalidView(
          "an occurrence endpoint is attached to multiple Module boundaries");
  }
  for (const FabricPhysicalTraversalRef &traversal :
       view.admittedTraversals()) {
    if (llvm::Error error = validateFabricRef(view, traversal))
      return std::move(error);
    auto projected = detail::projectFabricTraversal(view, traversal);
    if (!projected)
      return projected.takeError();
    for (const FabricTransportEndpointRef &endpoint : projected->sources)
      if (llvm::Error error = validateFabricRef(view, endpoint))
        return std::move(error);
    for (const FabricTransportEndpointRef &endpoint : projected->destinations)
      if (llvm::Error error = validateFabricRef(view, endpoint))
        return std::move(error);
    storage->physicalTraversalViews.push_back(std::move(*projected));
  }
  for (const FabricPhysicalTraversalView &traversal :
       storage->physicalTraversalViews) {
    const auto *selector =
        std::get_if<FabricPeSelectorPayload>(&traversal.reference.payload);
    if (!selector)
      continue;
    const auto append =
        [&](const FabricTransportEndpointRef &fixed,
            const FabricTransportEndpointRef &attachment) -> llvm::Error {
      if (fixed.owner.kind() !=
          FabricTransportEndpointOwnerKind::FabricFuOccurrence)
        return llvm::Error::success();
      const auto fu = std::get<FabricFuOccurrenceRef>(fixed.owner.payload);
      if (view.parentPeOf(fu) != selector->owner)
        return invalidView(
            "PE selector endpoint belongs to an FU outside its owner");
      auto &domain = storage->fuPortAttachments[canonicalFabricBytes(fixed)];
      domain.push_back({attachment, traversal.reference});
      return llvm::Error::success();
    };
    if (llvm::Error error = append(selector->source, selector->destination))
      return std::move(error);
    if (llvm::Error error = append(selector->destination, selector->source))
      return std::move(error);
  }
  for (auto &[fixed, domain] : storage->fuPortAttachments) {
    (void)fixed;
    llvm::sort(domain, [](const auto &lhs, const auto &rhs) {
      const auto lhsEndpoint = canonicalFabricBytes(lhs.endpoint);
      const auto rhsEndpoint = canonicalFabricBytes(rhs.endpoint);
      if (lhsEndpoint != rhsEndpoint)
        return lhsEndpoint < rhsEndpoint;
      return canonicalFabricBytes(lhs.localTraversal) <
             canonicalFabricBytes(rhs.localTraversal);
    });
    for (std::size_t index = 1; index < domain.size(); ++index)
      if (domain[index - 1].endpoint == domain[index].endpoint)
        return invalidView(
            "one FU port has ambiguous local traversals to an attachment");
  }
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

    if (view.entityKind(id) == FabricEntityKind::FabricMemoryEngineTemplate) {
      const FabricMemoryEngineTemplateRef owner(id);
      const FabricMemoryEngineTemplateRecord *engine =
          view.memoryEngineTemplate(owner);
      if (!engine)
        return invalidView("memory engine template cannot be resolved");
      for (std::uint64_t portOrdinal = 0;
           portOrdinal < engine->operationPorts.size(); ++portOrdinal) {
        FabricMemoryEngineTemplateOperationPortRef port{owner, portOrdinal};
        if (llvm::Error error = validateFabricRef(view, port))
          return std::move(error);
        for (std::uint64_t alternativeOrdinal = 0;
             alternativeOrdinal < engine->operationPorts[portOrdinal]
                                      .capabilityAlternatives()
                                      .size();
             ++alternativeOrdinal)
          if (llvm::Error error = validateFabricRef(
                  view, FabricMemoryEngineTemplateCapabilityAlternativeRef{
                            port, alternativeOrdinal}))
            return std::move(error);
      }
      for (std::uint64_t endpointOrdinal = 0;
           endpointOrdinal < engine->tokenEndpoints.size(); ++endpointOrdinal)
        if (llvm::Error error = validateFabricRef(
                view,
                FabricMemoryEngineTemplateEndpointRef{owner, endpointOrdinal}))
          return std::move(error);
      for (const ::fabric::MemoryInternalConnectionDeclaration &connection :
           engine->internalConnections) {
        FabricMemoryEngineTemplateInternalConnectionRef reference{
            owner,
            FabricMemoryEngineTemplateEndpointRef{
                owner, connection.sourceEndpointOrdinal},
            FabricMemoryEngineTemplateEndpointRef{
                owner, connection.sinkEndpointOrdinal}};
        if (llvm::Error error = validateFabricRef(view, reference))
          return std::move(error);
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
