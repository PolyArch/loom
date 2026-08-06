#include "FabricSystemCanonicalLabeling.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/CanonicalRelation.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/Artifact/FabricHardwareDomainContracts.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/IR/SystemServiceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace mlir;

namespace loom::fabric::detail {
namespace {

enum class RelationKind : std::uint32_t {
  RootChild,
  MemoryServiceCapability,
  ServiceEndpointOwner,
  ServiceEndpointCapability,
  ServiceCapabilityConsistencyDomain,
  ServiceCapabilityRateClock,
  ServiceCapabilityProgressClock,
  ServiceTransformInput,
  ServiceTransformOutput,
  ServiceTransformConsistencyDomain,
  ServiceTransformRegionPair,
  CoherentRegionInput,
  CoherentRegionOutput,
  HardwareDomainMember,
  ResetSynchronousClock,
  MemoryConsistencyParticipant,
  MemoryConsistencyProgressClock,
  TransportCrossingPattern,
  TransportCrossingSourceClock,
  TransportCrossingDestinationClock,
  TransferPatternOwner,
  TransferPatternIngress,
  TransferPatternEgress,
  TransferPatternUse,
  ConnectionSource,
  ConnectionDestination,
  SpatialAttachmentEndpoint,
  ServiceLegAttachmentEndpoint,
  ServiceLegAttachmentCarrier,
};

struct ProvisionalEntity {
  std::uint32_t vertex = 0;
  FabricEntityKind kind = FabricEntityKind::FabricModuleTemplate;
};

struct ReferencedEntity {
  FabricEntityId id = 0;
  FabricEntityKind kind = FabricEntityKind::FabricModuleTemplate;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

void appendU32(std::string &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<char>(value >> 24));
  bytes.push_back(static_cast<char>(value >> 16));
  bytes.push_back(static_cast<char>(value >> 8));
  bytes.push_back(static_cast<char>(value));
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64(std::string &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<char>(value >> shift));
}

void appendI64(std::string &bytes, std::int64_t value) {
  appendU64(bytes, static_cast<std::uint64_t>(value));
}

void appendBytes(std::string &bytes, llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.append(reinterpret_cast<const char *>(value.data()), value.size());
}

void appendText(std::string &bytes, llvm::StringRef value) {
  appendU64(bytes, value.size());
  bytes.append(value.data(), value.size());
}

template <typename Enum> void appendEnum(std::string &bytes, Enum value) {
  appendU32(bytes, static_cast<std::uint32_t>(value));
}

llvm::Error
appendExpectedBytes(std::string &bytes,
                    llvm::Expected<std::vector<std::uint8_t>> encoded) {
  if (!encoded)
    return encoded.takeError();
  appendBytes(bytes, *encoded);
  return llvm::Error::success();
}

llvm::Error
appendExpectedBytes(std::string &bytes,
                    llvm::Expected<CanonicalSemanticBytes> encoded) {
  if (!encoded)
    return encoded.takeError();
  appendBytes(bytes, encoded->bytes());
  return llvm::Error::success();
}

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr attribute) {
  std::vector<std::uint8_t> result;
  result.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

std::optional<FabricEntityKind> entityKind(Operation *operation) {
  if (isa<::fabric::SystemHostCoreOp>(operation))
    return FabricEntityKind::HostCoreOccurrence;
  if (isa<::fabric::SystemAccCoreOp>(operation))
    return FabricEntityKind::AccCoreOccurrence;
  if (isa<::fabric::SystemMemoryServiceOp>(operation))
    return FabricEntityKind::SystemMemoryService;
  if (isa<::fabric::SystemServiceEndpointOp>(operation))
    return FabricEntityKind::SystemServiceEndpoint;
  if (isa<::fabric::SystemServiceTransformOp>(operation))
    return FabricEntityKind::SystemServiceTransform;
  if (isa<::fabric::SystemExternalBoundaryOp>(operation))
    return FabricEntityKind::ExternalBoundary;
  if (isa<::fabric::SystemTransportResourceOp>(operation))
    return FabricEntityKind::SystemTransportResource;
  if (isa<::fabric::SystemHardwareDomainOp>(operation))
    return FabricEntityKind::HardwareDomain;
  return std::nullopt;
}

std::optional<FabricEntityId> provisionalEntityId(Operation *operation) {
  auto id = operation->getAttrOfType<::fabric::EntityIdAttr>("entity_id");
  return id ? std::optional<FabricEntityId>(id.getId()) : std::nullopt;
}

template <FabricEntityKind Kind>
ReferencedEntity entityReference(const FabricTypedEntityRef<Kind> &reference) {
  return {reference.id(), Kind};
}

ReferencedEntity entityReference(const SpatialCoreOccurrenceRef &reference) {
  return entityReference(reference.core);
}

ReferencedEntity entityReference(const InstructionCoreContextRef &reference) {
  return entityReference(reference.core);
}

ReferencedEntity entityReference(const InstructionContextRef &reference) {
  return entityReference(reference.pe);
}

ReferencedEntity entityReference(const FabricFuTemplateNodeRef &reference) {
  return entityReference(reference.fu);
}

ReferencedEntity entityReference(const FabricFuOccurrenceNodeRef &reference) {
  return entityReference(reference.fu);
}

ReferencedEntity
entityReference(const FabricMemoryOperationPortRef &reference) {
  return entityReference(reference.memory);
}

ReferencedEntity entityReference(const FabricTransferPatternRef &reference) {
  return entityReference(reference.resource);
}

ReferencedEntity entityReference(const FabricMemoryServiceRef &reference) {
  return std::visit(
      [](const auto &payload) { return entityReference(payload); },
      reference.payload);
}

ReferencedEntity
entityReference(const FabricTransportEndpointOwnerRef &reference) {
  return std::visit(
      [](const auto &payload) { return entityReference(payload); },
      reference.payload);
}

ReferencedEntity
entityReference(const FabricMemoryEndpointOwnerRef &reference) {
  return std::visit(
      [](const auto &payload) { return entityReference(payload); },
      reference.payload);
}

ReferencedEntity entityReference(const FabricInventoryOwnerRef &reference) {
  return std::visit(
      [](const auto &payload) { return entityReference(payload); },
      reference.payload);
}

std::string relationLabel(RelationKind kind,
                          llvm::ArrayRef<std::uint64_t> fields) {
  std::string label;
  appendU32(label, static_cast<std::uint32_t>(kind));
  appendU64(label, fields.size());
  for (std::uint64_t field : fields)
    appendU64(label, field);
  return label;
}

std::string relationLabel(RelationKind kind,
                          std::initializer_list<std::uint64_t> fields = {}) {
  return relationLabel(kind, llvm::ArrayRef<std::uint64_t>(fields));
}

std::vector<std::uint64_t>
inventoryOwnerLabelFields(const FabricInventoryOwnerRef &owner) {
  std::vector<std::uint64_t> fields{static_cast<std::uint32_t>(owner.kind())};
  switch (owner.kind()) {
  case FabricInventoryOwnerKind::FuTemplateNode: {
    const auto &node = std::get<FabricFuTemplateNodeRef>(owner.payload);
    fields.push_back(static_cast<std::uint32_t>(node.node));
    fields.push_back(node.ordinal);
    break;
  }
  case FabricInventoryOwnerKind::FuOccurrenceNode: {
    const auto &node = std::get<FabricFuOccurrenceNodeRef>(owner.payload);
    fields.push_back(static_cast<std::uint32_t>(node.node));
    fields.push_back(node.ordinal);
    break;
  }
  case FabricInventoryOwnerKind::MemoryOperationPort:
    fields.push_back(
        std::get<FabricMemoryOperationPortRef>(owner.payload).ordinal);
    break;
  case FabricInventoryOwnerKind::MemoryService:
    fields.push_back(static_cast<std::uint32_t>(
        std::get<FabricMemoryServiceRef>(owner.payload).kind()));
    break;
  case FabricInventoryOwnerKind::InstructionContext:
    fields.push_back(std::get<InstructionContextRef>(owner.payload).ordinal);
    break;
  case FabricInventoryOwnerKind::TransferPattern:
    fields.push_back(std::get<FabricTransferPatternRef>(owner.payload).ordinal);
    break;
  default:
    break;
  }
  return fields;
}

llvm::Expected<std::string>
memoryServiceIntrinsic(const ::fabric::MemoryServiceContractRecord &record) {
  std::string intrinsic = "SYSTEM_MEMORY_SERVICE";
  appendU64(intrinsic, record.regions().size());
  for (const ::fabric::MemoryServiceRegionDeclaration &region :
       record.regions()) {
    appendU64(intrinsic, region.addressBaseBytes);
    appendU64(intrinsic, region.sizeBytes);
    appendEnum(intrinsic, region.behavior);
    appendU32(intrinsic, region.mmioAcceptedAccessDomain.has_value());
    if (region.mmioAcceptedAccessDomain)
      if (llvm::Error error = appendExpectedBytes(
              intrinsic, ::fabric::encodeParameterizedMemoryAccessDomain(
                             *region.mmioAcceptedAccessDomain)))
        return std::move(error);
  }
  if (llvm::Error error = appendExpectedBytes(
          intrinsic,
          ::fabric::encodeResourceContractRecord(record.resourceContract())))
    return std::move(error);
  appendU64(intrinsic, record.capabilities().size());
  return intrinsic;
}

llvm::Expected<std::string> memoryServiceCapabilityIntrinsic(
    const ::fabric::MemoryServiceCapabilityDeclaration &capability) {
  std::string intrinsic = "SYSTEM_MEMORY_SERVICE_CAPABILITY";
  if (llvm::Error error = appendExpectedBytes(
          intrinsic, ::fabric::encodeMemoryActorContractDomain(
                         capability.actorContractDomain)))
    return std::move(error);
  appendU32(intrinsic, capability.accessDomain.has_value());
  if (capability.accessDomain)
    if (llvm::Error error = appendExpectedBytes(
            intrinsic, ::fabric::encodeParameterizedMemoryAccessDomain(
                           *capability.accessDomain)))
      return std::move(error);
  appendU64(intrinsic, capability.serviceRegionOrdinals.size());
  for (std::uint64_t ordinal : capability.serviceRegionOrdinals)
    appendU64(intrinsic, ordinal);
  appendU64(intrinsic, capability.serviceBeatWidthBits);
  appendU64(intrinsic, capability.admissibleUsePatterns.size());
  for (::fabric::UsePatternKey pattern : capability.admissibleUsePatterns)
    appendU32(intrinsic, pattern.ordinal());
  appendU32(intrinsic,
            static_cast<std::uint32_t>(capability.consistencyBinding.index()));
  if (const auto *local = std::get_if<::fabric::LocalProviderConsistency>(
          &capability.consistencyBinding)) {
    appendEnum(intrinsic, local->releaseVisibilityPoint);
    appendU32(intrinsic, static_cast<std::uint32_t>(local->progress.index()));
    if (const auto *bounded =
            std::get_if<::fabric::LocalBoundedCompletionCycles>(
                &local->progress))
      appendU64(intrinsic, bounded->maxIssueToRetireCycles);
  }
  return intrinsic;
}

llvm::Expected<std::string>
serviceCapabilityIntrinsic(const CanonicalServiceCapabilityRecord &capability) {
  std::string intrinsic = "SYSTEM_SERVICE_CAPABILITY";
  appendEnum(intrinsic, capability.kind());
  appendEnum(intrinsic, capability.role());
  appendU32(intrinsic, static_cast<std::uint32_t>(capability.domain().index()));
  if (const auto *message =
          std::get_if<MessageTransferCapabilityDomain>(&capability.domain())) {
    appendU64(intrinsic, message->payloadTypes().size());
    for (Type type : message->payloadTypes())
      if (llvm::Error error = appendExpectedBytes(
              intrinsic, dataflow::encodeCanonicalType(type)))
        return std::move(error);
  } else if (const auto *addressed =
                 std::get_if<AddressedMemoryCapabilityDomain>(
                     &capability.domain())) {
    if (llvm::Error error = appendExpectedBytes(
            intrinsic, ::fabric::encodeMemoryActorContractDomain(
                           addressed->actorContracts())))
      return std::move(error);
    if (llvm::Error error = appendExpectedBytes(
            intrinsic, ::fabric::encodeParameterizedMemoryAccessDomain(
                           addressed->accesses())))
      return std::move(error);
    if (llvm::Error error = appendExpectedBytes(
            intrinsic,
            ::fabric::encodeUnsignedDomain(addressed->addressBytes())))
      return std::move(error);
    appendU64(intrinsic, addressed->serviceBeatWidthBits());
    appendU32(intrinsic, addressed->consistencyDomain().has_value());
  } else {
    const auto &fence = std::get<FenceCapabilityDomain>(capability.domain());
    if (llvm::Error error = appendExpectedBytes(
            intrinsic,
            ::fabric::encodeMemoryActorContractDomain(fence.actorContracts())))
      return std::move(error);
  }
  const ServiceRateContractRecord &rate = capability.rate();
  appendU64(intrinsic, rate.operationsPerWindow());
  appendU64(intrinsic, rate.windowTicks());
  appendU64(intrinsic, rate.maxOutstanding());
  appendU32(intrinsic, static_cast<std::uint32_t>(rate.progress().index()));
  if (const auto *bounded =
          std::get_if<::fabric::BoundedCompletion>(&rate.progress()))
    appendU64(intrinsic, bounded->maxIssueToRetireTicks);
  return intrinsic;
}

std::string
serviceTransformIntrinsic(const SystemServiceTransformRecord &record) {
  std::string intrinsic = "SYSTEM_SERVICE_TRANSFORM";
  appendU64(intrinsic, record.inputs().size());
  appendU64(intrinsic, record.outputs().size());
  appendU32(intrinsic, static_cast<std::uint32_t>(record.contract().index()));
  if (const auto *offset =
          std::get_if<AddressOffsetTransform>(&record.contract())) {
    appendU32(intrinsic, offset->addressWidth);
    appendI64(intrinsic, offset->signedOffset);
  } else if (const auto *mask =
                 std::get_if<AddressMaskXorTransform>(&record.contract())) {
    appendU32(intrinsic, mask->addressWidth);
    appendU64(intrinsic, mask->andMask);
    appendU64(intrinsic, mask->xorMask);
  } else if (const auto *interleave =
                 std::get_if<StaticInterleaveTransform>(&record.contract())) {
    appendU64(intrinsic, interleave->granuleBytes);
    appendU64(intrinsic, interleave->outputCount);
  } else {
    const auto &coherent = std::get<CoherentMemoryTransform>(record.contract());
    appendU64(intrinsic, coherent.regions.size());
  }
  return intrinsic;
}

llvm::Expected<std::string>
hardwareDomainIntrinsic(const HardwareDomainContractRecord &record) {
  std::string intrinsic = "SYSTEM_HARDWARE_DOMAIN";
  appendEnum(intrinsic, record.kind());
  appendU64(intrinsic, record.members().size());
  if (const auto *clock =
          std::get_if<ClockDomainContractRecord>(&record.contract())) {
    if (llvm::Error error = appendExpectedBytes(
            intrinsic, encodeClockDomainContractRecord(*clock)))
      return std::move(error);
  } else if (const auto *reset =
                 std::get_if<ResetDomainContractRecord>(&record.contract())) {
    appendEnum(intrinsic, reset->polarity());
    appendEnum(intrinsic, reset->assertion());
    appendEnum(intrinsic, reset->deassertion());
    appendEnum(intrinsic, reset->initialState());
    appendU32(intrinsic, reset->synchronousTo().has_value());
    appendU32(intrinsic, reset->releaseLatencyCycles());
  } else if (const auto *power =
                 std::get_if<PowerDomainContractRecord>(&record.contract())) {
    appendU64(intrinsic, power->nominalVoltageUv());
  } else if (const auto *address =
                 std::get_if<AddressDomainContractRecord>(&record.contract())) {
    appendU32(intrinsic, address->addressWidth());
    appendU64(intrinsic, address->ranges().size());
    for (const AddressDomainRange &range : address->ranges()) {
      llvm::SmallString<64> lower;
      llvm::SmallString<64> upper;
      range.lower.toString(lower, 16, false);
      range.upperExclusive.toString(upper, 16, false);
      appendText(intrinsic, lower);
      appendText(intrinsic, upper);
    }
  } else {
    const auto &consistency =
        std::get<::fabric::MemoryConsistencyContract>(record.contract());
    appendEnum(intrinsic, consistency.releaseVisibilityPoint());
    appendU32(intrinsic,
              static_cast<std::uint32_t>(consistency.progress().index()));
    if (const auto *bounded =
            std::get_if<::fabric::BoundedCompletion>(&consistency.progress()))
      appendU64(intrinsic, bounded->maxIssueToRetireTicks);
    if (llvm::Error error = appendExpectedBytes(
            intrinsic, ::fabric::encodeResourceContractRecord(
                           consistency.resourceContract())))
      return std::move(error);
    appendU64(intrinsic, consistency.participants().size());
  }
  return intrinsic;
}

class SystemSemanticGraph {
public:
  static llvm::Expected<SystemSemanticGraph>
  build(::fabric::SystemOp root,
        llvm::ArrayRef<FabricDirectDependency> sourceDependencies) {
    SystemSemanticGraph graph(root, sourceDependencies);
    if (llvm::Error error = graph.collect())
      return std::move(error);
    if (llvm::Error error = graph.buildRelations())
      return std::move(error);
    return graph;
  }

  llvm::Expected<FabricSystemCanonicalLabeling> canonicalize() {
    auto canonical = canonicalizeRelationGraph(intrinsics_, edges_);
    if (!canonical)
      return canonical.takeError();

    std::vector<FabricEntityId> idByVertex(
        intrinsics_.size(), std::numeric_limits<FabricEntityId>::max());
    FabricEntityId nextId = 0;
    for (std::uint32_t vertex : canonical->canonicalOrder)
      if (entityCarrierByVertex_.count(vertex))
        idByVertex[vertex] = nextId++;

    std::vector<FabricSystemEntityCarrier> carriers;
    carriers.reserve(entityCarrierByVertex_.size());
    for (std::uint32_t vertex : canonical->canonicalOrder) {
      auto found = entityCarrierByVertex_.find(vertex);
      if (found == entityCarrierByVertex_.end())
        continue;
      FabricSystemEntityCarrier carrier = found->second;
      carrier.id = idByVertex[vertex];
      carriers.push_back(carrier);
    }

    llvm::DenseMap<std::uint32_t, Operation *> operationByVertex;
    for (const auto &entry : vertexByOperation_)
      operationByVertex[entry.second] = entry.first;
    std::vector<Operation *> operationOrder;
    llvm::DenseMap<Operation *, FabricOrdinal> patternOrdinals;
    llvm::DenseMap<std::uint32_t, FabricOrdinal> nextPatternOrdinal;
    for (std::uint32_t vertex : canonical->canonicalOrder)
      if (Operation *operation = operationByVertex.lookup(vertex)) {
        operationOrder.push_back(operation);
        auto resource = patternResourceVertexByOperation_.find(operation);
        if (resource != patternResourceVertexByOperation_.end())
          patternOrdinals[operation] = nextPatternOrdinal[resource->second]++;
      }

    return FabricSystemCanonicalLabeling{
        std::move(canonical->bytes), std::move(carriers),
        std::move(operationOrder), sourceDependencyToCanonical_,
        std::move(patternOrdinals)};
  }

private:
  SystemSemanticGraph(::fabric::SystemOp root,
                      llvm::ArrayRef<FabricDirectDependency> sourceDependencies)
      : root_(root), sourceDependencies_(sourceDependencies.begin(),
                                         sourceDependencies.end()) {}

  std::uint32_t addVertex(std::string intrinsic) {
    const std::uint32_t vertex = intrinsics_.size();
    intrinsics_.push_back(std::move(intrinsic));
    return vertex;
  }

  llvm::Expected<std::uint64_t>
  canonicalDependencyOrdinal(std::uint64_t sourceOrdinal) const {
    if (sourceOrdinal >= sourceDependencyToCanonical_.size())
      return invalid("System field references a dependency outside its table");
    return sourceDependencyToCanonical_[sourceOrdinal];
  }

  llvm::Expected<std::string> importedModuleTargetIntrinsic(
      const FabricImportedModuleTargetRef &target) const {
    auto dependency = canonicalDependencyOrdinal(target.dependencyOrdinal);
    if (!dependency)
      return dependency.takeError();
    std::string intrinsic;
    appendU64(intrinsic, *dependency);
    appendBytes(intrinsic, canonicalFabricBytes(target.target));
    return intrinsic;
  }

  llvm::Expected<std::string> importedModuleEndpointIntrinsic(
      const FabricImportedModuleBoundaryEndpointRef &endpoint) const {
    auto dependency = canonicalDependencyOrdinal(endpoint.dependencyOrdinal);
    if (!dependency)
      return dependency.takeError();
    std::string intrinsic;
    appendU64(intrinsic, *dependency);
    appendBytes(intrinsic, canonicalFabricBytes(endpoint.target));
    return intrinsic;
  }

  llvm::Expected<std::string> operationIntrinsic(Operation *operation) {
    std::string intrinsic;
    if (auto host = dyn_cast<::fabric::SystemHostCoreOp>(operation)) {
      appendText(intrinsic, "system.host_core");
      appendBytes(intrinsic, unsignedBytes(host.getArchitectureAttr()));
      appendBytes(intrinsic, unsignedBytes(host.getMicroarchitectureAttr()));
      return intrinsic;
    }
    if (auto core = dyn_cast<::fabric::SystemAccCoreOp>(operation)) {
      appendText(intrinsic, "system.acc_core");
      appendBytes(intrinsic, unsignedBytes(core.getArchitectureAttr()));
      appendBytes(intrinsic, unsignedBytes(core.getMicroarchitectureAttr()));
      auto target = decodeFabricImportedModuleTargetRef(
          unsignedBytes(core.getSpatialCoreAttr()));
      if (!target)
        return target.takeError();
      auto targetIntrinsic = importedModuleTargetIntrinsic(*target);
      if (!targetIntrinsic)
        return targetIntrinsic.takeError();
      appendText(intrinsic, *targetIntrinsic);
      return intrinsic;
    }
    if (auto service = dyn_cast<::fabric::SystemMemoryServiceOp>(operation)) {
      auto record = ::fabric::decodeMemoryServiceContractRecord(
          unsignedBytes(service.getServiceContractAttr().getRecord()),
          operation->getContext(), ::fabric::MemoryServiceOwnerKind::System);
      if (!record)
        return record.takeError();
      return memoryServiceIntrinsic(*record);
    }
    if (auto endpoint =
            dyn_cast<::fabric::SystemServiceEndpointOp>(operation)) {
      appendText(intrinsic, "system.service_endpoint");
      if (TypeAttr carrier = endpoint.getCarrierTypeAttr()) {
        appendU32(intrinsic, 1);
        if (llvm::Error error = appendExpectedBytes(
                intrinsic,
                ::fabric::encodeFabricTransportType(carrier.getValue())))
          return std::move(error);
      } else {
        appendU32(intrinsic, 0);
      }
      return intrinsic;
    }
    if (auto transform =
            dyn_cast<::fabric::SystemServiceTransformOp>(operation)) {
      auto record = decodeSystemServiceTransformRecord(
          unsignedBytes(transform.getContractAttr()));
      if (!record)
        return record.takeError();
      return serviceTransformIntrinsic(*record);
    }
    if (auto attachment =
            dyn_cast<::fabric::SystemServiceLegCarrierAttachmentOp>(
                operation)) {
      auto record = decodeServiceLegCarrierAttachmentRecord(
          unsignedBytes(attachment.getRecordAttr()));
      if (!record)
        return record.takeError();
      appendText(intrinsic, "system.service_leg_carrier_attachment");
      appendEnum(intrinsic, record->kind());
      appendU64(intrinsic, record->legOrdinal());
      return intrinsic;
    }
    if (isa<::fabric::SystemExternalBoundaryOp>(operation)) {
      appendText(intrinsic, "system.external_boundary");
      return intrinsic;
    }
    if (auto domain = dyn_cast<::fabric::SystemHardwareDomainOp>(operation)) {
      auto record = decodeHardwareDomainContractRecord(
          unsignedBytes(domain.getContractAttr()));
      if (!record)
        return record.takeError();
      return hardwareDomainIntrinsic(*record);
    }
    if (auto resource =
            dyn_cast<::fabric::SystemTransportResourceOp>(operation)) {
      appendText(intrinsic, "system.transport_resource");
      if (llvm::Error error = appendExpectedBytes(
              intrinsic,
              ::fabric::encodeFabricTransportFunctionType(cast<FunctionType>(
                  resource.getFunctionTypeAttr().getValue()))))
        return std::move(error);
      appendBytes(intrinsic, unsignedBytes(resource.getResourceContractAttr()));
      if (DenseI8ArrayAttr crossing = resource.getClockCrossingAttr()) {
        appendU32(intrinsic, 1);
        auto record =
            decodeClockCrossingContractRecord(unsignedBytes(crossing));
        if (!record)
          return record.takeError();
        appendU32(intrinsic, record->depth());
        appendU32(intrinsic, record->synchronizerStages());
      } else {
        appendU32(intrinsic, 0);
      }
      return intrinsic;
    }
    if (isa<::fabric::SystemTransferPatternOp>(operation)) {
      appendText(intrinsic, "system.transfer_pattern");
      return intrinsic;
    }
    if (isa<::fabric::SystemConnectionOp>(operation)) {
      appendText(intrinsic, "system.connection");
      return intrinsic;
    }
    if (auto attachment =
            dyn_cast<::fabric::SystemSpatialAttachmentOp>(operation)) {
      appendText(intrinsic, "system.spatial_attachment");
      auto endpoint = decodeFabricImportedModuleBoundaryEndpointRef(
          unsignedBytes(attachment.getModuleEndpointAttr()));
      if (!endpoint)
        return endpoint.takeError();
      auto endpointIntrinsic = importedModuleEndpointIntrinsic(*endpoint);
      if (!endpointIntrinsic)
        return endpointIntrinsic.takeError();
      appendText(intrinsic, *endpointIntrinsic);
      return intrinsic;
    }
    return invalid("System canonical labeling does not yet project " +
                   operation->getName().getStringRef());
  }

  llvm::Error collect() {
    sourceDependencyToCanonical_.resize(sourceDependencies_.size());
    llvm::SmallVector<std::pair<std::vector<std::uint8_t>, std::uint64_t>>
        dependencyRows;
    dependencyRows.reserve(sourceDependencies_.size());
    for (auto [ordinal, dependency] : llvm::enumerate(sourceDependencies_)) {
      if (dependency.role != FabricDependencyRole::ImportedModule)
        return invalid("fabric.system admits only ImportedModule dependencies");
      std::vector<std::uint8_t> bytes =
          encodeArtifactRootReference(dependency.root);
      std::vector<std::uint8_t> row;
      appendU32(row, static_cast<std::uint32_t>(dependency.role));
      row.insert(row.end(), bytes.begin(), bytes.end());
      dependencyRows.emplace_back(std::move(row), ordinal);
    }
    llvm::sort(dependencyRows, [](const auto &left, const auto &right) {
      return left.first < right.first;
    });
    for (std::size_t index = 1; index < dependencyRows.size(); ++index)
      if (dependencyRows[index - 1].first == dependencyRows[index].first)
        return invalid("System authoring root repeats a direct dependency");
    for (auto [canonical, row] : llvm::enumerate(dependencyRows))
      sourceDependencyToCanonical_[row.second] = canonical;

    rootVertex_ = addVertex("FABRIC_SYSTEM_ROOT");
    for (Operation &operation : root_.getBody().front()) {
      auto intrinsic = operationIntrinsic(&operation);
      if (!intrinsic)
        return intrinsic.takeError();
      const std::uint32_t vertex = addVertex(std::move(*intrinsic));
      vertexByOperation_[&operation] = vertex;
      edges_.push_back(
          {rootVertex_, vertex, relationLabel(RelationKind::RootChild)});

      std::optional<FabricEntityKind> kind = entityKind(&operation);
      if (!kind)
        continue;
      entityCarrierByVertex_[vertex] = {*kind, 0, &operation};
      std::optional<FabricEntityId> authored = provisionalEntityId(&operation);
      if (!authored)
        continue;
      if (!entityByProvisionalId_
               .try_emplace(*authored, ProvisionalEntity{vertex, *kind})
               .second)
        return invalid("System authoring root repeats a provisional EntityId");
    }

    for (Operation &operation : root_.getBody().front()) {
      auto pattern = dyn_cast<::fabric::SystemTransferPatternOp>(&operation);
      if (!pattern)
        continue;
      auto record = decodeSystemTransferPatternRecord(
          unsignedBytes(pattern.getContractAttr()));
      if (!record)
        return record.takeError();
      std::vector<std::uint8_t> key = canonicalFabricBytes(record->pattern());
      if (!patternVertexByProvisionalRef_
               .emplace(std::move(key), vertexByOperation_.lookup(&operation))
               .second)
        return invalid("System authoring root repeats a transfer-pattern ref");
    }
    return llvm::Error::success();
  }

  llvm::Error
  addEntityRelation(std::uint32_t source, const ReferencedEntity &referenced,
                    RelationKind role,
                    std::initializer_list<std::uint64_t> labelFields = {}) {
    auto target = entityByProvisionalId_.find(referenced.id);
    if (target == entityByProvisionalId_.end())
      return invalid("System relation references an unknown entity");
    if (target->second.kind != referenced.kind)
      return invalid("System relation names the wrong entity kind");
    edges_.push_back(
        {source, target->second.vertex, relationLabel(role, labelFields)});
    return llvm::Error::success();
  }

  llvm::Error
  addTransportEndpointRelation(std::uint32_t source,
                               const FabricTransportEndpointRef &endpoint,
                               RelationKind role) {
    return addEntityRelation(
        source, entityReference(endpoint.owner), role,
        {static_cast<std::uint32_t>(endpoint.owner.kind()), endpoint.ordinal});
  }

  llvm::Error addMemoryEndpointRelation(std::uint32_t source,
                                        const FabricMemoryEndpointRef &endpoint,
                                        RelationKind role) {
    return addEntityRelation(
        source, entityReference(endpoint.owner), role,
        {static_cast<std::uint32_t>(endpoint.owner.kind()), endpoint.ordinal});
  }

  llvm::Error addInventoryOwnerRelation(std::uint32_t source,
                                        const FabricInventoryOwnerRef &owner,
                                        RelationKind role) {
    if (owner.kind() == FabricInventoryOwnerKind::TransferPattern) {
      const auto &pattern = std::get<FabricTransferPatternRef>(owner.payload);
      const std::vector<std::uint8_t> key = canonicalFabricBytes(pattern);
      auto target = patternVertexByProvisionalRef_.find(key);
      if (target == patternVertexByProvisionalRef_.end())
        return invalid(
            "System relation references an unknown transfer pattern");
      edges_.push_back(
          {source, target->second,
           relationLabel(role, {static_cast<std::uint32_t>(owner.kind())})});
      return llvm::Error::success();
    }
    const ReferencedEntity referenced = entityReference(owner);
    auto target = entityByProvisionalId_.find(referenced.id);
    if (target == entityByProvisionalId_.end())
      return invalid("System relation references an unknown inventory owner");
    if (target->second.kind != referenced.kind)
      return invalid("System inventory owner names the wrong entity kind");
    edges_.push_back({source, target->second.vertex,
                      relationLabel(role, inventoryOwnerLabelFields(owner))});
    return llvm::Error::success();
  }

  llvm::Error addClockRelation(std::uint32_t source,
                               const ClockDomainRef &clock, RelationKind role) {
    return addEntityRelation(source, entityReference(clock.underlying()), role);
  }

  llvm::Error
  addConsistencyDomainRelation(std::uint32_t source,
                               const MemoryConsistencyDomainRef &domain,
                               RelationKind role) {
    return addEntityRelation(source, entityReference(domain.underlying()),
                             role);
  }

  llvm::Error addUsePatternRelation(std::uint32_t source,
                                    const FabricUsePatternRef &pattern,
                                    RelationKind role) {
    const FabricInventoryOwnerRef &owner = pattern.owner.catalog();
    return addEntityRelation(
        source, entityReference(owner), role,
        {static_cast<std::uint32_t>(owner.kind()), pattern.ordinal});
  }

  llvm::Error
  addMemoryServiceRegionRelation(std::uint32_t source,
                                 const FabricMemoryServiceRegionRef &region,
                                 RelationKind role) {
    return addEntityRelation(
        source, entityReference(region.service), role,
        {static_cast<std::uint32_t>(region.service.kind()), region.ordinal});
  }

  llvm::Error
  addTransferPatternRelation(std::uint32_t source,
                             const FabricTransferPatternRef &pattern,
                             RelationKind role) {
    const std::vector<std::uint8_t> key = canonicalFabricBytes(pattern);
    auto target = patternVertexByProvisionalRef_.find(key);
    if (target == patternVertexByProvisionalRef_.end())
      return invalid("System relation references an unknown transfer pattern");
    edges_.push_back({source, target->second, relationLabel(role)});
    return llvm::Error::success();
  }

  std::uint32_t addAuxiliary(std::uint32_t owner, std::string intrinsic,
                             RelationKind relation) {
    const std::uint32_t vertex = addVertex(std::move(intrinsic));
    edges_.push_back({owner, vertex, relationLabel(relation)});
    return vertex;
  }

  llvm::Error addMemoryServiceRelations(::fabric::SystemMemoryServiceOp service,
                                        std::uint32_t vertex) {
    auto record = ::fabric::decodeMemoryServiceContractRecord(
        unsignedBytes(service.getServiceContractAttr().getRecord()),
        service.getContext(), ::fabric::MemoryServiceOwnerKind::System);
    if (!record)
      return record.takeError();
    for (const ::fabric::MemoryServiceCapabilityDeclaration &capability :
         record->capabilities()) {
      auto intrinsic = memoryServiceCapabilityIntrinsic(capability);
      if (!intrinsic)
        return intrinsic.takeError();
      const std::uint32_t capabilityVertex = addAuxiliary(
          vertex, std::move(*intrinsic), RelationKind::MemoryServiceCapability);
      if (const auto *domain = std::get_if<MemoryConsistencyDomainRef>(
              &capability.consistencyBinding))
        if (llvm::Error error = addConsistencyDomainRelation(
                capabilityVertex, *domain,
                RelationKind::ServiceCapabilityConsistencyDomain))
          return error;
    }
    return llvm::Error::success();
  }

  llvm::Error
  addServiceEndpointRelations(::fabric::SystemServiceEndpointOp endpoint,
                              std::uint32_t vertex) {
    auto owner = decodeSystemServiceEndpointOwnerRef(
        unsignedBytes(endpoint.getOwnerAttr()));
    if (!owner)
      return owner.takeError();
    if (llvm::Error error = addInventoryOwnerRelation(
            vertex, owner->owner(), RelationKind::ServiceEndpointOwner))
      return error;

    auto capabilities = decodeCanonicalServiceCapabilitySet(
        unsignedBytes(endpoint.getCapabilitiesAttr()), endpoint.getContext());
    if (!capabilities)
      return capabilities.takeError();
    for (const CanonicalServiceCapabilityRecord &capability :
         capabilities->capabilities()) {
      auto intrinsic = serviceCapabilityIntrinsic(capability);
      if (!intrinsic)
        return intrinsic.takeError();
      const std::uint32_t capabilityVertex =
          addAuxiliary(vertex, std::move(*intrinsic),
                       RelationKind::ServiceEndpointCapability);
      if (const auto *addressed = std::get_if<AddressedMemoryCapabilityDomain>(
              &capability.domain())) {
        if (addressed->consistencyDomain())
          if (llvm::Error error = addConsistencyDomainRelation(
                  capabilityVertex, *addressed->consistencyDomain(),
                  RelationKind::ServiceCapabilityConsistencyDomain))
            return error;
      } else if (const auto *fence =
                     std::get_if<FenceCapabilityDomain>(&capability.domain())) {
        if (llvm::Error error = addConsistencyDomainRelation(
                capabilityVertex, fence->consistencyDomain(),
                RelationKind::ServiceCapabilityConsistencyDomain))
          return error;
      }
      if (llvm::Error error =
              addClockRelation(capabilityVertex, capability.rate().rateClock(),
                               RelationKind::ServiceCapabilityRateClock))
        return error;
      if (const auto *bounded = std::get_if<::fabric::BoundedCompletion>(
              &capability.rate().progress()))
        if (llvm::Error error =
                addClockRelation(capabilityVertex, bounded->progressClock,
                                 RelationKind::ServiceCapabilityProgressClock))
          return error;
    }
    return llvm::Error::success();
  }

  llvm::Error
  addServiceTransformRelations(::fabric::SystemServiceTransformOp transform,
                               std::uint32_t vertex) {
    auto record = decodeSystemServiceTransformRecord(
        unsignedBytes(transform.getContractAttr()));
    if (!record)
      return record.takeError();
    for (auto [ordinal, input] : llvm::enumerate(record->inputs()))
      if (llvm::Error error =
              addEntityRelation(vertex, entityReference(input.owner),
                                RelationKind::ServiceTransformInput,
                                {static_cast<std::uint32_t>(input.owner.kind()),
                                 input.ordinal, ordinal}))
        return error;
    for (auto [ordinal, output] : llvm::enumerate(record->outputs()))
      if (llvm::Error error = addEntityRelation(
              vertex, entityReference(output.owner),
              RelationKind::ServiceTransformOutput,
              {static_cast<std::uint32_t>(output.owner.kind()), output.ordinal,
               ordinal}))
        return error;
    const auto *coherent =
        std::get_if<CoherentMemoryTransform>(&record->contract());
    if (!coherent)
      return llvm::Error::success();
    if (llvm::Error error = addConsistencyDomainRelation(
            vertex, coherent->consistencyDomain,
            RelationKind::ServiceTransformConsistencyDomain))
      return error;
    for (const CoherentMemoryRegionCorrespondence &region : coherent->regions) {
      const std::uint32_t pair =
          addAuxiliary(vertex, "COHERENT_MEMORY_REGION_PAIR",
                       RelationKind::ServiceTransformRegionPair);
      if (llvm::Error error = addMemoryServiceRegionRelation(
              pair, region.input, RelationKind::CoherentRegionInput))
        return error;
      if (llvm::Error error = addMemoryServiceRegionRelation(
              pair, region.output, RelationKind::CoherentRegionOutput))
        return error;
    }
    return llvm::Error::success();
  }

  llvm::Error
  addHardwareDomainRelations(::fabric::SystemHardwareDomainOp domain,
                             std::uint32_t vertex) {
    auto record = decodeHardwareDomainContractRecord(
        unsignedBytes(domain.getContractAttr()));
    if (!record)
      return record.takeError();
    for (const FabricInventoryOwnerRef &member : record->members())
      if (llvm::Error error = addInventoryOwnerRelation(
              vertex, member, RelationKind::HardwareDomainMember))
        return error;
    if (const auto *reset =
            std::get_if<ResetDomainContractRecord>(&record->contract())) {
      if (reset->synchronousTo())
        return addClockRelation(vertex, *reset->synchronousTo(),
                                RelationKind::ResetSynchronousClock);
      return llvm::Error::success();
    }
    const auto *consistency =
        std::get_if<::fabric::MemoryConsistencyContract>(&record->contract());
    if (!consistency)
      return llvm::Error::success();
    for (const ::fabric::MemoryConsistencyParticipant &participant :
         consistency->participants()) {
      if (const auto *service =
              std::get_if<FabricMemoryServiceRef>(&participant.payload)) {
        if (llvm::Error error = addEntityRelation(
                vertex, entityReference(*service),
                RelationKind::MemoryConsistencyParticipant,
                {static_cast<std::uint32_t>(participant.kind()),
                 static_cast<std::uint32_t>(service->kind())}))
          return error;
        continue;
      }
      const FabricMemoryEndpointRef &provider =
          std::get<SubordinateEndpointRef>(participant.payload).underlying();
      if (llvm::Error error = addEntityRelation(
              vertex, entityReference(provider.owner),
              RelationKind::MemoryConsistencyParticipant,
              {static_cast<std::uint32_t>(participant.kind()),
               static_cast<std::uint32_t>(provider.owner.kind()),
               provider.ordinal}))
        return error;
    }
    if (const auto *bounded =
            std::get_if<::fabric::BoundedCompletion>(&consistency->progress()))
      return addClockRelation(vertex, bounded->progressClock,
                              RelationKind::MemoryConsistencyProgressClock);
    return llvm::Error::success();
  }

  llvm::Error
  addTransportResourceRelations(::fabric::SystemTransportResourceOp resource,
                                std::uint32_t vertex) {
    DenseI8ArrayAttr crossing = resource.getClockCrossingAttr();
    if (!crossing)
      return llvm::Error::success();
    auto record = decodeClockCrossingContractRecord(unsignedBytes(crossing));
    if (!record)
      return record.takeError();
    if (llvm::Error error =
            addTransferPatternRelation(vertex, record->transferPattern(),
                                       RelationKind::TransportCrossingPattern))
      return error;
    if (llvm::Error error =
            addClockRelation(vertex, record->sourceClock(),
                             RelationKind::TransportCrossingSourceClock))
      return error;
    return addClockRelation(vertex, record->destinationClock(),
                            RelationKind::TransportCrossingDestinationClock);
  }

  llvm::Error
  addTransferPatternRelations(::fabric::SystemTransferPatternOp pattern,
                              std::uint32_t vertex) {
    auto record = decodeSystemTransferPatternRecord(
        unsignedBytes(pattern.getContractAttr()));
    if (!record)
      return record.takeError();
    const ReferencedEntity owner = entityReference(record->pattern().resource);
    if (llvm::Error error = addEntityRelation(
            vertex, owner, RelationKind::TransferPatternOwner))
      return error;
    patternResourceVertexByOperation_[pattern.getOperation()] =
        entityByProvisionalId_.lookup(owner.id).vertex;
    if (llvm::Error error = addTransportEndpointRelation(
            vertex, record->ingress(), RelationKind::TransferPatternIngress))
      return error;
    for (const FabricTransportEndpointRef &egress : record->egresses())
      if (llvm::Error error = addTransportEndpointRelation(
              vertex, egress, RelationKind::TransferPatternEgress))
        return error;
    return addUsePatternRelation(vertex, record->usePattern(),
                                 RelationKind::TransferPatternUse);
  }

  llvm::Error
  addSpatialAttachmentRelations(::fabric::SystemSpatialAttachmentOp attachment,
                                std::uint32_t vertex) {
    auto endpoint = decodeFabricSpatialAttachmentEndpointRef(
        unsignedBytes(attachment.getSpatialEndpointAttr()));
    if (!endpoint)
      return endpoint.takeError();
    if (const FabricTransportEndpointRef *transport = endpoint->transport())
      return addEntityRelation(
          vertex, entityReference(transport->owner),
          RelationKind::SpatialAttachmentEndpoint,
          {static_cast<std::uint32_t>(endpoint->plane()),
           static_cast<std::uint32_t>(transport->owner.kind()),
           transport->ordinal});
    const FabricMemoryEndpointRef &memory = *endpoint->memory();
    return addEntityRelation(vertex, entityReference(memory.owner),
                             RelationKind::SpatialAttachmentEndpoint,
                             {static_cast<std::uint32_t>(endpoint->plane()),
                              static_cast<std::uint32_t>(memory.owner.kind()),
                              memory.ordinal});
  }

  llvm::Error addServiceLegCarrierAttachmentRelations(
      ::fabric::SystemServiceLegCarrierAttachmentOp attachment,
      std::uint32_t vertex) {
    auto record = decodeServiceLegCarrierAttachmentRecord(
        unsignedBytes(attachment.getRecordAttr()));
    if (!record)
      return record.takeError();
    if (llvm::Error error = addMemoryEndpointRelation(
            vertex, record->endpoint(),
            RelationKind::ServiceLegAttachmentEndpoint))
      return error;
    for (const FabricTransportEndpointRef &carrier : record->carriers())
      if (llvm::Error error = addTransportEndpointRelation(
              vertex, carrier, RelationKind::ServiceLegAttachmentCarrier))
        return error;
    return llvm::Error::success();
  }

  llvm::Error buildRelations() {
    for (Operation &operation : root_.getBody().front()) {
      const std::uint32_t vertex = vertexByOperation_.lookup(&operation);
      if (auto service =
              dyn_cast<::fabric::SystemMemoryServiceOp>(&operation)) {
        if (llvm::Error error = addMemoryServiceRelations(service, vertex))
          return error;
        continue;
      }
      if (auto endpoint =
              dyn_cast<::fabric::SystemServiceEndpointOp>(&operation)) {
        if (llvm::Error error = addServiceEndpointRelations(endpoint, vertex))
          return error;
        continue;
      }
      if (auto transform =
              dyn_cast<::fabric::SystemServiceTransformOp>(&operation)) {
        if (llvm::Error error = addServiceTransformRelations(transform, vertex))
          return error;
        continue;
      }
      if (auto domain =
              dyn_cast<::fabric::SystemHardwareDomainOp>(&operation)) {
        if (llvm::Error error = addHardwareDomainRelations(domain, vertex))
          return error;
        continue;
      }
      if (auto resource =
              dyn_cast<::fabric::SystemTransportResourceOp>(&operation)) {
        if (llvm::Error error = addTransportResourceRelations(resource, vertex))
          return error;
        continue;
      }
      if (auto pattern =
              dyn_cast<::fabric::SystemTransferPatternOp>(&operation)) {
        if (llvm::Error error = addTransferPatternRelations(pattern, vertex))
          return error;
        continue;
      }
      if (auto attachment =
              dyn_cast<::fabric::SystemSpatialAttachmentOp>(&operation)) {
        if (llvm::Error error =
                addSpatialAttachmentRelations(attachment, vertex))
          return error;
        continue;
      }
      if (auto attachment =
              dyn_cast<::fabric::SystemServiceLegCarrierAttachmentOp>(
                  &operation)) {
        if (llvm::Error error =
                addServiceLegCarrierAttachmentRelations(attachment, vertex))
          return error;
        continue;
      }
      auto connection = dyn_cast<::fabric::SystemConnectionOp>(&operation);
      if (!connection)
        continue;
      auto source = decodeFabricRef<FabricTransportEndpointRef>(
          unsignedBytes(connection.getSourceAttr()));
      if (!source)
        return source.takeError();
      auto destination = decodeFabricRef<FabricTransportEndpointRef>(
          unsignedBytes(connection.getDestinationAttr()));
      if (!destination)
        return destination.takeError();
      if (llvm::Error error = addTransportEndpointRelation(
              vertex, *source, RelationKind::ConnectionSource))
        return error;
      if (llvm::Error error = addTransportEndpointRelation(
              vertex, *destination, RelationKind::ConnectionDestination))
        return error;
    }
    return llvm::Error::success();
  }

  ::fabric::SystemOp root_;
  std::vector<FabricDirectDependency> sourceDependencies_;
  std::vector<std::uint64_t> sourceDependencyToCanonical_;
  std::uint32_t rootVertex_ = 0;
  std::vector<std::string> intrinsics_;
  std::vector<CanonicalRelationEdge> edges_;
  llvm::DenseMap<Operation *, std::uint32_t> vertexByOperation_;
  llvm::DenseMap<FabricEntityId, ProvisionalEntity> entityByProvisionalId_;
  std::map<std::vector<std::uint8_t>, std::uint32_t>
      patternVertexByProvisionalRef_;
  llvm::DenseMap<Operation *, std::uint32_t> patternResourceVertexByOperation_;
  llvm::DenseMap<std::uint32_t, FabricSystemEntityCarrier>
      entityCarrierByVertex_;
};

} // namespace

llvm::Expected<FabricSystemCanonicalLabeling>
computeFabricSystemCanonicalLabeling(
    ::fabric::SystemOp root,
    llvm::ArrayRef<FabricDirectDependency> sourceDependencies) {
  auto graph = SystemSemanticGraph::build(root, sourceDependencies);
  if (!graph)
    return graph.takeError();
  return graph->canonicalize();
}

} // namespace loom::fabric::detail
