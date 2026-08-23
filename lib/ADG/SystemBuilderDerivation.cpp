#include "ADG/Builder.h"

#include "BuilderInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Fabric/Artifact/FabricHardwareDomainContracts.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/SystemServiceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::adg {
namespace {

using detail::invalid;

std::vector<std::uint8_t> unsignedBytes(mlir::DenseI8ArrayAttr attribute) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    bytes.push_back(static_cast<std::uint8_t>(byte));
  return bytes;
}

mlir::DenseI8ArrayAttr denseBytes(mlir::MLIRContext &context,
                                  llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(&context, signedBytes);
}

template <typename RootOp>
llvm::Expected<RootOp> singleFabricRoot(mlir::ModuleOp module) {
  if (!llvm::hasSingleElement(module.getBody()->getOperations()))
    return invalid("finalized parent does not contain exactly one Fabric root");
  auto root = mlir::dyn_cast<RootOp>(&module.getBody()->front());
  if (!root)
    return invalid("finalized parent contains the wrong Fabric root operation");
  return root;
}

llvm::Expected<detail::SystemRootState *>
derivedSystem(const std::shared_ptr<detail::DesignState> &state,
              std::size_t rootOrdinal) {
  if (rootOrdinal >= state->systemRoots.size())
    return invalid("System handle has an invalid owner ordinal");
  detail::SystemRootState &root = state->systemRoots[rootOrdinal];
  if (root.closed)
    return invalid("System is already closed");
  if (!root.derivedParent)
    return invalid("System mutation requires a finalized parent draft");
  return &root;
}

template <typename Op>
llvm::Expected<Op> systemEntity(detail::SystemRootState &root,
                                loom::fabric::FabricEntityId id,
                                llvm::StringRef description) {
  if (id >= root.entities.size() || !root.entities[id].operation)
    return invalid(description + " does not resolve in the parent System");
  auto operation = mlir::dyn_cast<Op>(root.entities[id].operation);
  if (!operation)
    return invalid(description + " has the wrong typed System owner");
  return operation;
}

template <typename Op>
llvm::Expected<loom::fabric::FabricEntityId> canonicalEntityId(Op operation) {
  auto id = operation.getEntityIdAttr();
  if (!id)
    return invalid("canonical System entity has no EntityId");
  return id.getId();
}

llvm::Expected<
    std::pair<loom::fabric::FabricEntityId, detail::SystemEntityState>>
decodeEntityState(mlir::Operation *operation, mlir::MLIRContext &context) {
  using loom::fabric::FabricEntityKind;
  if (auto entity = mlir::dyn_cast<::fabric::SystemHostCoreOp>(operation)) {
    auto id = canonicalEntityId(entity);
    if (!id)
      return id.takeError();
    return std::make_pair(
        *id, detail::SystemEntityState{FabricEntityKind::HostCoreOccurrence,
                                       operation});
  }
  if (auto entity = mlir::dyn_cast<::fabric::SystemAccCoreOp>(operation)) {
    auto id = canonicalEntityId(entity);
    if (!id)
      return id.takeError();
    auto target = loom::fabric::decodeFabricImportedModuleTargetRef(
        unsignedBytes(entity.getSpatialCoreAttr()));
    if (!target)
      return target.takeError();
    detail::SystemEntityState state{FabricEntityKind::AccCoreOccurrence,
                                    operation};
    state.importedModule = target->dependencyOrdinal;
    return std::make_pair(*id, std::move(state));
  }
  if (auto entity =
          mlir::dyn_cast<::fabric::SystemMemoryServiceOp>(operation)) {
    auto id = canonicalEntityId(entity);
    if (!id)
      return id.takeError();
    return std::make_pair(
        *id, detail::SystemEntityState{FabricEntityKind::SystemMemoryService,
                                       operation});
  }
  if (auto entity =
          mlir::dyn_cast<::fabric::SystemServiceEndpointOp>(operation)) {
    auto id = canonicalEntityId(entity);
    if (!id)
      return id.takeError();
    auto capabilities = loom::fabric::decodeCanonicalServiceCapabilitySet(
        unsignedBytes(entity.getCapabilitiesAttr()), &context);
    if (!capabilities)
      return capabilities.takeError();
    detail::SystemEntityState state{FabricEntityKind::SystemServiceEndpoint,
                                    operation};
    state.endpointPlane = capabilities->plane();
    state.endpointRole = capabilities->role();
    return std::make_pair(*id, std::move(state));
  }
  if (auto entity =
          mlir::dyn_cast<::fabric::SystemServiceTransformOp>(operation)) {
    auto id = canonicalEntityId(entity);
    if (!id)
      return id.takeError();
    return std::make_pair(
        *id, detail::SystemEntityState{FabricEntityKind::SystemServiceTransform,
                                       operation});
  }
  if (auto entity =
          mlir::dyn_cast<::fabric::SystemTransportResourceOp>(operation)) {
    auto id = canonicalEntityId(entity);
    if (!id)
      return id.takeError();
    detail::SystemEntityState state{FabricEntityKind::SystemTransportResource,
                                    operation};
    state.inputCount = entity.getFunctionType().getNumInputs();
    state.outputCount = entity.getFunctionType().getNumResults();
    state.crossingDeclared = static_cast<bool>(entity.getClockCrossingAttr());
    return std::make_pair(*id, std::move(state));
  }
  if (auto entity =
          mlir::dyn_cast<::fabric::SystemHardwareDomainOp>(operation)) {
    auto id = canonicalEntityId(entity);
    if (!id)
      return id.takeError();
    return std::make_pair(
        *id,
        detail::SystemEntityState{FabricEntityKind::HardwareDomain, operation});
  }
  if (auto entity =
          mlir::dyn_cast<::fabric::SystemExternalBoundaryOp>(operation)) {
    auto id = canonicalEntityId(entity);
    if (!id)
      return id.takeError();
    return std::make_pair(
        *id, detail::SystemEntityState{FabricEntityKind::ExternalBoundary,
                                       operation});
  }
  return invalid("operation is not a canonical System entity");
}

bool isSystemEntity(mlir::Operation *operation) {
  return mlir::isa<
      ::fabric::SystemHostCoreOp, ::fabric::SystemAccCoreOp,
      ::fabric::SystemMemoryServiceOp, ::fabric::SystemServiceEndpointOp,
      ::fabric::SystemServiceTransformOp, ::fabric::SystemTransportResourceOp,
      ::fabric::SystemHardwareDomainOp, ::fabric::SystemExternalBoundaryOp>(
      operation);
}

llvm::Error rebuildEntityStates(detail::SystemRootState &root,
                                mlir::MLIRContext &context) {
  std::vector<
      std::pair<loom::fabric::FabricEntityId, detail::SystemEntityState>>
      entities;
  for (mlir::Operation &operation : root.operation.getBody().front()) {
    if (!isSystemEntity(&operation))
      continue;
    auto state = decodeEntityState(&operation, context);
    if (!state)
      return state.takeError();
    entities.push_back(std::move(*state));
  }
  llvm::sort(entities, [](const auto &left, const auto &right) {
    return left.first < right.first;
  });
  root.entities.clear();
  root.entities.reserve(entities.size());
  for (auto &entry : entities) {
    if (entry.first != root.entities.size())
      return invalid("canonical System EntityIds are not dense");
    if (entry.second.importedModule &&
        *entry.second.importedModule >= root.importedModules.size())
      return invalid("AccCore selects an unknown imported Module");
    root.entities.push_back(std::move(entry.second));
  }

  for (mlir::Operation &operation : root.operation.getBody().front()) {
    auto pattern = mlir::dyn_cast<::fabric::SystemTransferPatternOp>(operation);
    if (!pattern)
      continue;
    auto record = loom::fabric::decodeSystemTransferPatternRecord(
        unsignedBytes(pattern.getContractAttr()));
    if (!record)
      return record.takeError();
    const auto resource = record->pattern().resource.id();
    if (resource >= root.entities.size() ||
        root.entities[resource].kind !=
            loom::fabric::FabricEntityKind::SystemTransportResource)
      return invalid("transfer pattern selects an unknown transport resource");
    root.entities[resource].nextTransferPatternOrdinal =
        std::max(root.entities[resource].nextTransferPatternOrdinal,
                 record->pattern().ordinal + 1);
  }
  return llvm::Error::success();
}

llvm::Error
requireAdmissibleModule(const detail::SystemRootState &root,
                        const loom::fabric::FinalizedFabricRoot &module) {
  if (module.view().rootKind() != loom::fabric::FabricRootKind::Module)
    return invalid("System composition candidate is not a Module root");
  if (llvm::none_of(root.admissibleModules, [&](const auto &reference) {
        return reference == module.reference();
      }))
    return invalid("Module is outside the admitted System composition set");
  return llvm::Error::success();
}

mlir::OpBuilder insertionBuilder(detail::DesignState &state,
                                 detail::SystemRootState &root) {
  mlir::OpBuilder builder(&state.context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  return builder;
}

using BoundaryKey =
    std::pair<loom::fabric::FabricPortDirection, loom::fabric::FabricOrdinal>;

llvm::Expected<std::map<BoundaryKey, loom::fabric::SystemServiceEndpointRef>>
memoryAttachmentServices(detail::SystemRootState &root,
                         loom::fabric::AccCoreOccurrenceRef core) {
  std::map<BoundaryKey, loom::fabric::SystemServiceEndpointRef> services;
  for (mlir::Operation &operation : root.operation.getBody().front()) {
    auto attachment =
        mlir::dyn_cast<::fabric::SystemSpatialAttachmentOp>(operation);
    if (!attachment || !attachment.getServiceEndpointAttr())
      continue;
    auto spatial = loom::fabric::decodeFabricSpatialAttachmentEndpointRef(
        unsignedBytes(attachment.getSpatialEndpointAttr()));
    if (!spatial)
      return spatial.takeError();
    const auto *memory = spatial->memory();
    if (!memory ||
        memory->owner.kind() !=
            loom::fabric::FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence)
      continue;
    const auto owner =
        std::get<loom::fabric::SpatialCoreOccurrenceRef>(memory->owner.payload);
    if (owner.core != core)
      continue;
    auto module = loom::fabric::decodeFabricImportedModuleBoundaryEndpointRef(
        unsignedBytes(attachment.getModuleEndpointAttr()));
    if (!module)
      return module.takeError();
    auto service =
        loom::fabric::decodeFabricRef<loom::fabric::SystemServiceEndpointRef>(
            unsignedBytes(attachment.getServiceEndpointAttr()));
    if (!service)
      return service.takeError();
    BoundaryKey key{module->target.direction, module->target.ordinal};
    if (!services.emplace(key, *service).second)
      return invalid("SpatialCore repeats a memory boundary attachment");
  }
  return services;
}

llvm::Expected<loom::fabric::FabricSpatialAttachmentEndpointRef>
spatialEndpoint(loom::fabric::AccCoreOccurrenceRef core,
                const detail::ImportedModuleBoundary &boundary) {
  const loom::fabric::SpatialCoreOccurrenceRef spatial{core};
  if (boundary.plane ==
      loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Transport) {
    return loom::fabric::FabricSpatialAttachmentEndpointRef::create(
        loom::fabric::FabricTransportEndpointRef{
            loom::fabric::FabricTransportEndpointOwnerRef::of(spatial),
            boundary.occurrenceOrdinal});
  }
  return loom::fabric::FabricSpatialAttachmentEndpointRef::create(
      loom::fabric::FabricMemoryEndpointRef{
          loom::fabric::FabricMemoryEndpointOwnerRef::of(spatial),
          boundary.occurrenceOrdinal});
}

llvm::Error emitSpatialAttachments(
    detail::DesignState &state, detail::SystemRootState &root,
    loom::fabric::AccCoreOccurrenceRef core, std::size_t importedOrdinal,
    const std::map<BoundaryKey, loom::fabric::SystemServiceEndpointRef>
        &memoryServices,
    bool includeTransport) {
  if (importedOrdinal >= root.importedModules.size())
    return invalid("SpatialCore attachment selects an unknown Module");
  const detail::ImportedModuleState &module =
      root.importedModules[importedOrdinal];
  mlir::OpBuilder builder = insertionBuilder(state, root);
  auto emit = [&](llvm::ArrayRef<detail::ImportedModuleBoundary> boundaries,
                  loom::fabric::FabricPortDirection direction) -> llvm::Error {
    for (auto indexed : llvm::enumerate(boundaries)) {
      const BoundaryKey key{direction, indexed.index()};
      const detail::ImportedModuleBoundary &boundary = indexed.value();
      if (!includeTransport &&
          boundary.plane == loom::fabric::FabricSpatialAttachmentEndpointRef::
                                Plane::Transport)
        continue;
      std::optional<loom::fabric::SystemServiceEndpointRef> service;
      if (boundary.plane ==
          loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Memory) {
        auto found = memoryServices.find(key);
        if (found == memoryServices.end())
          return invalid("replacement Module has an unmatched memory boundary");
        service = found->second;
      }
      auto occurrenceEndpoint = spatialEndpoint(core, boundary);
      if (!occurrenceEndpoint)
        return occurrenceEndpoint.takeError();
      const loom::fabric::FabricImportedModuleBoundaryEndpointRef
          moduleEndpoint{
              importedOrdinal,
              {module.module, direction,
               static_cast<loom::fabric::FabricOrdinal>(indexed.index())}};
      mlir::DenseI8ArrayAttr serviceAttribute;
      if (service)
        serviceAttribute = denseBytes(
            state.context, loom::fabric::canonicalFabricBytes(*service));
      auto attachment = ::fabric::SystemSpatialAttachmentOp::create(
          builder, root.operation.getLoc(),
          denseBytes(
              state.context,
              loom::fabric::encodeFabricImportedModuleBoundaryEndpointRef(
                  moduleEndpoint)),
          denseBytes(state.context,
                     loom::fabric::encodeFabricSpatialAttachmentEndpointRef(
                         *occurrenceEndpoint)),
          serviceAttribute);
      if (mlir::failed(mlir::verify(attachment))) {
        attachment.erase();
        return invalid("Fabric rejected the derived SpatialCore attachment");
      }
    }
    return llvm::Error::success();
  };
  if (llvm::Error error =
          emit(module.inputs, loom::fabric::FabricPortDirection::Input))
    return error;
  return emit(module.outputs, loom::fabric::FabricPortDirection::Output);
}

bool inventoryMemberNamesEndpoint(
    const loom::fabric::FabricInventoryOwnerRef &owner,
    const std::set<loom::fabric::FabricEntityId> &endpoints) {
  const auto *endpoint =
      std::get_if<loom::fabric::SystemServiceEndpointRef>(&owner.payload);
  return endpoint && endpoints.count(endpoint->id());
}

bool transportEndpointNamesCore(
    const loom::fabric::FabricTransportEndpointRef &endpoint,
    loom::fabric::AccCoreOccurrenceRef core) {
  return std::visit(
      [&](const auto &owner) {
        using Owner = std::decay_t<decltype(owner)>;
        if constexpr (std::is_same_v<Owner,
                                     loom::fabric::SpatialCoreOccurrenceRef> ||
                      std::is_same_v<Owner,
                                     loom::fabric::InstructionCoreContextRef>)
          return owner.core == core;
        return false;
      },
      endpoint.owner.payload);
}

bool memoryEndpointNamesCore(
    const loom::fabric::FabricMemoryEndpointRef &endpoint,
    loom::fabric::AccCoreOccurrenceRef core) {
  const auto *owner = std::get_if<loom::fabric::SpatialCoreOccurrenceRef>(
      &endpoint.owner.payload);
  return owner && owner->core == core;
}

bool transportEndpointNamesService(
    const loom::fabric::FabricTransportEndpointRef &endpoint,
    const std::set<loom::fabric::FabricEntityId> &services) {
  const auto *owner = std::get_if<loom::fabric::SystemServiceEndpointRef>(
      &endpoint.owner.payload);
  return owner && services.count(owner->id());
}

bool memoryEndpointNamesService(
    const loom::fabric::FabricMemoryEndpointRef &endpoint,
    const std::set<loom::fabric::FabricEntityId> &services) {
  const auto *owner = std::get_if<loom::fabric::SystemServiceEndpointRef>(
      &endpoint.owner.payload);
  return owner && services.count(owner->id());
}

std::optional<loom::fabric::FabricInventoryOwnerRef>
remapCoreMember(const loom::fabric::FabricInventoryOwnerRef &owner,
                loom::fabric::AccCoreOccurrenceRef source,
                loom::fabric::AccCoreOccurrenceRef destination) {
  return std::visit(
      [&](const auto &member)
          -> std::optional<loom::fabric::FabricInventoryOwnerRef> {
        using Member = std::decay_t<decltype(member)>;
        if constexpr (std::is_same_v<Member,
                                     loom::fabric::AccCoreOccurrenceRef>) {
          if (member == source)
            return loom::fabric::FabricInventoryOwnerRef::of(destination);
        } else if constexpr (std::is_same_v<
                                 Member,
                                 loom::fabric::InstructionCoreContextRef>) {
          if (member.core == source)
            return loom::fabric::FabricInventoryOwnerRef::of(
                loom::fabric::InstructionCoreContextRef{destination});
        } else if constexpr (std::is_same_v<
                                 Member,
                                 loom::fabric::SpatialCoreOccurrenceRef>) {
          if (member.core == source)
            return loom::fabric::FabricInventoryOwnerRef::of(
                loom::fabric::SpatialCoreOccurrenceRef{destination});
        }
        return std::nullopt;
      },
      owner.payload);
}

llvm::Error
copyHardwareDomainMembership(detail::DesignState &state,
                             detail::SystemRootState &root,
                             loom::fabric::AccCoreOccurrenceRef source,
                             loom::fabric::AccCoreOccurrenceRef destination) {
  for (mlir::Operation &operation : root.operation.getBody().front()) {
    auto domain = mlir::dyn_cast<::fabric::SystemHardwareDomainOp>(operation);
    if (!domain)
      continue;
    auto record = loom::fabric::decodeHardwareDomainContractRecord(
        unsignedBytes(domain.getContractAttr()));
    if (!record)
      return record.takeError();
    std::vector<loom::fabric::FabricInventoryOwnerRef> members(
        record->members().begin(), record->members().end());
    for (const auto &member : record->members())
      if (auto mapped = remapCoreMember(member, source, destination))
        members.push_back(std::move(*mapped));
    auto updated = loom::fabric::HardwareDomainContractRecord::create(
        std::move(members), record->contract());
    if (!updated)
      return updated.takeError();
    auto bytes = loom::fabric::encodeHardwareDomainContractRecord(*updated);
    if (!bytes)
      return bytes.takeError();
    domain.setContractAttr(denseBytes(state.context, *bytes));
  }
  return llvm::Error::success();
}

struct SpatialEndpointOrdinalMap final {
  std::map<loom::fabric::FabricOrdinal, loom::fabric::FabricOrdinal> transport;
  std::map<loom::fabric::FabricOrdinal, loom::fabric::FabricOrdinal> memory;
};

llvm::Expected<SpatialEndpointOrdinalMap>
buildSpatialEndpointOrdinalMap(const detail::ImportedModuleState &source,
                               const detail::ImportedModuleState &destination) {
  SpatialEndpointOrdinalMap result;
  const auto mapDirection =
      [&](llvm::ArrayRef<detail::ImportedModuleBoundary> sourceBoundaries,
          llvm::ArrayRef<detail::ImportedModuleBoundary> destinationBoundaries)
      -> llvm::Error {
    const std::size_t count =
        std::min(sourceBoundaries.size(), destinationBoundaries.size());
    for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
      const detail::ImportedModuleBoundary &sourceBoundary =
          sourceBoundaries[ordinal];
      const detail::ImportedModuleBoundary &destinationBoundary =
          destinationBoundaries[ordinal];
      if (sourceBoundary.plane != destinationBoundary.plane)
        continue;
      auto &mapping = sourceBoundary.plane ==
                              loom::fabric::FabricSpatialAttachmentEndpointRef::
                                  Plane::Transport
                          ? result.transport
                          : result.memory;
      auto [position, inserted] =
          mapping.emplace(sourceBoundary.occurrenceOrdinal,
                          destinationBoundary.occurrenceOrdinal);
      if (!inserted &&
          position->second != destinationBoundary.occurrenceOrdinal)
        return invalid(
            "Module boundary correspondence maps one occurrence endpoint "
            "inconsistently");
    }
    return llvm::Error::success();
  };
  if (llvm::Error error = mapDirection(source.inputs, destination.inputs))
    return std::move(error);
  if (llvm::Error error = mapDirection(source.outputs, destination.outputs))
    return std::move(error);
  return result;
}

llvm::Error
copySpatialServiceLegAttachments(detail::DesignState &state,
                                 detail::SystemRootState &root,
                                 loom::fabric::AccCoreOccurrenceRef source,
                                 loom::fabric::AccCoreOccurrenceRef destination,
                                 const SpatialEndpointOrdinalMap &mapping) {
  std::vector<loom::fabric::ServiceLegCarrierAttachmentRecord> records;
  for (mlir::Operation &operation : root.operation.getBody().front()) {
    auto attachment =
        mlir::dyn_cast<::fabric::SystemServiceLegCarrierAttachmentOp>(
            operation);
    if (!attachment)
      continue;
    auto record = loom::fabric::decodeServiceLegCarrierAttachmentRecord(
        unsignedBytes(attachment.getRecordAttr()));
    if (!record)
      return record.takeError();
    if (record->endpoint().owner.kind() !=
        loom::fabric::FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence)
      continue;
    const auto endpointOwner = std::get<loom::fabric::SpatialCoreOccurrenceRef>(
        record->endpoint().owner.payload);
    if (endpointOwner.core != source)
      continue;

    auto mappedEndpoint = mapping.memory.find(record->endpoint().ordinal);
    if (mappedEndpoint == mapping.memory.end())
      continue;

    const loom::fabric::FabricMemoryEndpointRef endpoint{
        loom::fabric::FabricMemoryEndpointOwnerRef::of(
            loom::fabric::SpatialCoreOccurrenceRef{destination}),
        mappedEndpoint->second};
    std::vector<loom::fabric::FabricTransportEndpointRef> carriers;
    carriers.reserve(record->carriers().size());
    for (const auto &carrier : record->carriers()) {
      loom::fabric::FabricTransportEndpointRef remapped = carrier;
      if (carrier.owner.kind() ==
          loom::fabric::FabricTransportEndpointOwnerKind::
              SpatialCoreOccurrence) {
        const auto owner = std::get<loom::fabric::SpatialCoreOccurrenceRef>(
            carrier.owner.payload);
        if (owner.core == source) {
          auto mappedCarrier = mapping.transport.find(carrier.ordinal);
          if (mappedCarrier == mapping.transport.end())
            return invalid(
                "replacement Module has no matching service-leg carrier");
          remapped.owner = loom::fabric::FabricTransportEndpointOwnerRef::of(
              loom::fabric::SpatialCoreOccurrenceRef{destination});
          remapped.ordinal = mappedCarrier->second;
        }
      }
      carriers.push_back(std::move(remapped));
    }
    auto remapped = loom::fabric::ServiceLegCarrierAttachmentRecord::create(
        endpoint, record->kind(), record->legOrdinal(), std::move(carriers));
    if (!remapped)
      return remapped.takeError();
    records.push_back(std::move(*remapped));
  }

  mlir::OpBuilder builder = insertionBuilder(state, root);
  for (const auto &record : records) {
    auto bytes = loom::fabric::encodeServiceLegCarrierAttachmentRecord(record);
    if (!bytes)
      return bytes.takeError();
    auto operation = ::fabric::SystemServiceLegCarrierAttachmentOp::create(
        builder, root.operation.getLoc(), denseBytes(state.context, *bytes));
    if (mlir::failed(mlir::verify(operation))) {
      operation.erase();
      return invalid("Fabric rejected the derived service-leg attachment");
    }
  }
  return llvm::Error::success();
}

llvm::Error
remapSpatialServiceLegAttachments(detail::DesignState &state,
                                  detail::SystemRootState &root,
                                  loom::fabric::AccCoreOccurrenceRef core,
                                  const SpatialEndpointOrdinalMap &mapping) {
  for (auto iterator = root.operation.getBody().front().begin();
       iterator != root.operation.getBody().front().end();) {
    mlir::Operation &operation = *iterator++;
    auto attachment =
        mlir::dyn_cast<::fabric::SystemServiceLegCarrierAttachmentOp>(
            operation);
    if (!attachment)
      continue;
    auto record = loom::fabric::decodeServiceLegCarrierAttachmentRecord(
        unsignedBytes(attachment.getRecordAttr()));
    if (!record)
      return record.takeError();

    loom::fabric::FabricMemoryEndpointRef endpoint = record->endpoint();
    if (memoryEndpointNamesCore(endpoint, core)) {
      auto mapped = mapping.memory.find(endpoint.ordinal);
      if (mapped == mapping.memory.end()) {
        attachment.erase();
        continue;
      }
      endpoint.ordinal = mapped->second;
    }

    std::vector<loom::fabric::FabricTransportEndpointRef> carriers;
    carriers.reserve(record->carriers().size());
    for (loom::fabric::FabricTransportEndpointRef carrier :
         record->carriers()) {
      if (transportEndpointNamesCore(carrier, core)) {
        auto mapped = mapping.transport.find(carrier.ordinal);
        if (mapped == mapping.transport.end())
          continue;
        carrier.ordinal = mapped->second;
      }
      carriers.push_back(std::move(carrier));
    }
    auto remapped = loom::fabric::ServiceLegCarrierAttachmentRecord::create(
        endpoint, record->kind(), record->legOrdinal(), std::move(carriers));
    if (!remapped)
      return remapped.takeError();
    auto bytes =
        loom::fabric::encodeServiceLegCarrierAttachmentRecord(*remapped);
    if (!bytes)
      return bytes.takeError();
    attachment.setRecordAttr(denseBytes(state.context, *bytes));
  }
  return llvm::Error::success();
}

llvm::Error removeHardwareDomainMembership(
    detail::DesignState &state, detail::SystemRootState &root,
    loom::fabric::AccCoreOccurrenceRef target,
    const std::set<loom::fabric::FabricEntityId> &removedEndpoints) {
  for (mlir::Operation &operation : root.operation.getBody().front()) {
    auto domain = mlir::dyn_cast<::fabric::SystemHardwareDomainOp>(operation);
    if (!domain)
      continue;
    auto record = loom::fabric::decodeHardwareDomainContractRecord(
        unsignedBytes(domain.getContractAttr()));
    if (!record)
      return record.takeError();
    std::vector<loom::fabric::FabricInventoryOwnerRef> members;
    for (const auto &member : record->members())
      if (!loom::fabric::inventoryOwnerBelongsToAccCore(member, target) &&
          !inventoryMemberNamesEndpoint(member, removedEndpoints))
        members.push_back(member);
    auto updated = loom::fabric::HardwareDomainContractRecord::create(
        std::move(members), record->contract());
    if (!updated)
      return updated.takeError();
    auto bytes = loom::fabric::encodeHardwareDomainContractRecord(*updated);
    if (!bytes)
      return bytes.takeError();
    domain.setContractAttr(denseBytes(state.context, *bytes));
  }
  return llvm::Error::success();
}

llvm::Expected<bool>
attachmentNamesCore(::fabric::SystemSpatialAttachmentOp attachment,
                    loom::fabric::AccCoreOccurrenceRef core) {
  auto endpoint = loom::fabric::decodeFabricSpatialAttachmentEndpointRef(
      unsignedBytes(attachment.getSpatialEndpointAttr()));
  if (!endpoint)
    return endpoint.takeError();
  if (const auto *transport = endpoint->transport()) {
    if (transport->owner.kind() !=
        loom::fabric::FabricTransportEndpointOwnerKind::SpatialCoreOccurrence)
      return false;
    return std::get<loom::fabric::SpatialCoreOccurrenceRef>(
               transport->owner.payload)
               .core == core;
  }
  const auto *memory = endpoint->memory();
  if (!memory ||
      memory->owner.kind() !=
          loom::fabric::FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence)
    return false;
  return std::get<loom::fabric::SpatialCoreOccurrenceRef>(memory->owner.payload)
             .core == core;
}

llvm::Error compactImportedModules(detail::DesignState &state,
                                   detail::SystemRootState &root) {
  std::vector<bool> used(root.importedModules.size(), false);
  for (mlir::Operation &operation : root.operation.getBody().front()) {
    auto core = mlir::dyn_cast<::fabric::SystemAccCoreOp>(operation);
    if (!core)
      continue;
    auto target = loom::fabric::decodeFabricImportedModuleTargetRef(
        unsignedBytes(core.getSpatialCoreAttr()));
    if (!target)
      return target.takeError();
    if (target->dependencyOrdinal >= used.size())
      return invalid("AccCore selects an unknown imported Module");
    used[target->dependencyOrdinal] = true;
  }

  std::vector<std::size_t> remapped(used.size(), used.size());
  if (llvm::all_of(used, [](bool value) { return value; }))
    return llvm::Error::success();
  std::vector<detail::ImportedModuleState> compact;
  compact.reserve(root.importedModules.size());
  for (std::size_t ordinal = 0; ordinal < used.size(); ++ordinal) {
    if (!used[ordinal])
      continue;
    remapped[ordinal] = compact.size();
    compact.push_back(std::move(root.importedModules[ordinal]));
  }
  for (mlir::Operation &operation : root.operation.getBody().front()) {
    if (auto core = mlir::dyn_cast<::fabric::SystemAccCoreOp>(operation)) {
      auto target = loom::fabric::decodeFabricImportedModuleTargetRef(
          unsignedBytes(core.getSpatialCoreAttr()));
      if (!target)
        return target.takeError();
      target->dependencyOrdinal = remapped[target->dependencyOrdinal];
      core.setSpatialCoreAttr(denseBytes(
          state.context,
          loom::fabric::encodeFabricImportedModuleTargetRef(*target)));
      continue;
    }
    auto attachment =
        mlir::dyn_cast<::fabric::SystemSpatialAttachmentOp>(operation);
    if (!attachment)
      continue;
    auto endpoint = loom::fabric::decodeFabricImportedModuleBoundaryEndpointRef(
        unsignedBytes(attachment.getModuleEndpointAttr()));
    if (!endpoint)
      return endpoint.takeError();
    if (endpoint->dependencyOrdinal >= remapped.size() ||
        remapped[endpoint->dependencyOrdinal] == remapped.size())
      return invalid("Spatial attachment selects an unused imported Module");
    endpoint->dependencyOrdinal = remapped[endpoint->dependencyOrdinal];
    attachment.setModuleEndpointAttr(
        denseBytes(state.context,
                   loom::fabric::encodeFabricImportedModuleBoundaryEndpointRef(
                       *endpoint)));
  }
  root.importedModules = std::move(compact);
  for (detail::SystemEntityState &entity : root.entities)
    if (entity.importedModule)
      entity.importedModule = remapped[*entity.importedModule];
  return llvm::Error::success();
}

llvm::Error verifySystemDraft(detail::SystemRootState &root,
                              llvm::StringRef description) {
  if (mlir::failed(mlir::verify(root.operation)))
    return invalid("Fabric rejected the derived " + description);
  return llvm::Error::success();
}

} // namespace

llvm::Expected<SystemBuilder> DesignBuilder::deriveSystem(
    const loom::fabric::FinalizedFabricRoot &parent,
    llvm::ArrayRef<loom::fabric::FinalizedFabricRoot> admissibleModules) {
  if (!state_ || state_->consumed)
    return invalid("DesignBuilder is already consumed");
  if (!state_->spatialRoots.empty() || !state_->systemRoots.empty())
    return invalid("derived Fabric draft requires an empty DesignBuilder");
  if (parent.view().rootKind() != loom::fabric::FabricRootKind::System)
    return invalid("deriveSystem requires a finalized System parent");

  std::vector<std::pair<std::vector<std::uint8_t>, ArtifactRootReference>>
      admitted;
  admitted.reserve(admissibleModules.size());
  for (const auto &module : admissibleModules) {
    if (module.view().rootKind() != loom::fabric::FabricRootKind::Module)
      return invalid("admissible System composition root is not a Module");
    admitted.emplace_back(encodeArtifactRootReference(module.reference()),
                          module.reference());
  }
  llvm::sort(admitted, [](const auto &left, const auto &right) {
    return left.first < right.first;
  });
  for (std::size_t index = 1; index < admitted.size(); ++index)
    if (admitted[index - 1].first == admitted[index].first)
      return invalid("admissible System composition set repeats a Module");

  auto module = detail::loadCanonicalFabricModule(
      parent, *state_, loom::fabric::FabricRootKind::System);
  if (!module)
    return module.takeError();
  auto rootOperation = singleFabricRoot<::fabric::SystemOp>(*module);
  if (!rootOperation)
    return rootOperation.takeError();
  state_->labels.insert(rootOperation->getSymName());
  state_->systemRoots.emplace_back(*rootOperation,
                                   rootOperation->getSymName().str());
  detail::SystemRootState &root = state_->systemRoots.back();
  root.derivedParent = parent.view();
  for (auto &entry : admitted)
    root.admissibleModules.push_back(std::move(entry.second));

  SystemBuilder builder(state_, 0);
  for (const loom::fabric::FabricDirectDependency &dependency :
       parent.directDependencies()) {
    if (dependency.role != loom::fabric::FabricDependencyRole::ImportedModule)
      return invalid("derived System parent has a non-Module dependency");
    auto imported =
        loom::fabric::importEntireFabricRoot(dependency.root, state_->store);
    if (!imported)
      return imported.takeError();
    auto handle = builder.importSpatialCore(*imported);
    if (!handle)
      return handle.takeError();
  }
  if (llvm::Error error = rebuildEntityStates(root, state_->context))
    return std::move(error);
  return builder;
}

llvm::Expected<AccCore> SystemBuilder::addAccCoreFromPrototype(
    loom::fabric::AccCoreOccurrenceRef prototype,
    const loom::fabric::FinalizedFabricRoot &spatialCore) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  if (llvm::Error error = requireAdmissibleModule(**root, spatialCore))
    return std::move(error);
  auto prototypeOp = systemEntity<::fabric::SystemAccCoreOp>(
      **root, prototype.id(), "AccCore prototype");
  if (!prototypeOp)
    return prototypeOp.takeError();
  auto architecture = loom::fabric::decodeInstructionCoreArchitecturalContract(
      unsignedBytes(prototypeOp->getArchitectureAttr()));
  if (!architecture)
    return architecture.takeError();
  auto microarchitecture =
      loom::fabric::decodeInstructionCoreMicroarchitecturalRealization(
          unsignedBytes(prototypeOp->getMicroarchitectureAttr()));
  if (!microarchitecture)
    return microarchitecture.takeError();
  auto services = memoryAttachmentServices(**root, prototype);
  if (!services)
    return services.takeError();
  const std::optional<std::size_t> prototypeImport =
      (*root)->entities[prototype.id()].importedModule;
  if (!prototypeImport || *prototypeImport >= (*root)->importedModules.size())
    return invalid("AccCore prototype has no imported SpatialCore Module");
  auto imported = importSpatialCore(spatialCore);
  if (!imported)
    return imported.takeError();
  auto endpointMapping = buildSpatialEndpointOrdinalMap(
      (*root)->importedModules[*prototypeImport],
      (*root)->importedModules[imported->importOrdinal_]);
  if (!endpointMapping)
    return endpointMapping.takeError();
  auto created = addAccCore(*architecture, *microarchitecture, *imported);
  if (!created)
    return created.takeError();
  const loom::fabric::AccCoreOccurrenceRef destination(created->entity_);
  if (llvm::Error error =
          emitSpatialAttachments(**state, **root, destination,
                                 imported->importOrdinal_, *services, false))
    return std::move(error);
  if (llvm::Error error = copySpatialServiceLegAttachments(
          **state, **root, prototype, destination, *endpointMapping))
    return std::move(error);
  if (llvm::Error error =
          copyHardwareDomainMembership(**state, **root, prototype, destination))
    return std::move(error);
  if (llvm::Error error = verifySystemDraft(**root, "AccCore addition"))
    return std::move(error);
  return *created;
}

llvm::Error
SystemBuilder::removeAccCore(loom::fabric::AccCoreOccurrenceRef target) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto targetOp = systemEntity<::fabric::SystemAccCoreOp>(**root, target.id(),
                                                          "removed AccCore");
  if (!targetOp)
    return targetOp.takeError();

  std::set<loom::fabric::FabricEntityId> removedEndpoints;
  for (mlir::Operation &operation : (*root)->operation.getBody().front()) {
    auto endpoint =
        mlir::dyn_cast<::fabric::SystemServiceEndpointOp>(operation);
    if (!endpoint)
      continue;
    auto owner = loom::fabric::decodeSystemServiceEndpointOwnerRef(
        unsignedBytes(endpoint.getOwnerAttr()));
    if (!owner)
      return owner.takeError();
    if (loom::fabric::inventoryOwnerBelongsToAccCore(owner->owner(), target)) {
      auto id = canonicalEntityId(endpoint);
      if (!id)
        return id.takeError();
      removedEndpoints.insert(*id);
    }
  }

  for (auto iterator = (*root)->operation.getBody().front().begin();
       iterator != (*root)->operation.getBody().front().end();) {
    mlir::Operation &operation = *iterator++;
    if (auto endpoint =
            mlir::dyn_cast<::fabric::SystemServiceEndpointOp>(operation)) {
      auto id = canonicalEntityId(endpoint);
      if (!id)
        return id.takeError();
      if (removedEndpoints.count(*id))
        endpoint.erase();
      continue;
    }
    if (auto attachment =
            mlir::dyn_cast<::fabric::SystemSpatialAttachmentOp>(operation)) {
      auto namesCore = attachmentNamesCore(attachment, target);
      if (!namesCore)
        return namesCore.takeError();
      if (*namesCore) {
        attachment.erase();
        continue;
      }
      if (auto serviceAttribute = attachment.getServiceEndpointAttr()) {
        auto service = loom::fabric::decodeFabricRef<
            loom::fabric::SystemServiceEndpointRef>(
            unsignedBytes(serviceAttribute));
        if (!service)
          return service.takeError();
        if (removedEndpoints.count(service->id()))
          return invalid(
              "removed AccCore service endpoint is attached to another core");
      }
      continue;
    }
    if (auto connection =
            mlir::dyn_cast<::fabric::SystemConnectionOp>(operation)) {
      bool remove = false;
      if (connection.getMemoryServiceAttr()) {
        auto source = loom::fabric::decodeFabricRef<
            loom::fabric::FabricMemoryEndpointRef>(
            unsignedBytes(connection.getSourceAttr()));
        if (!source)
          return source.takeError();
        auto destination = loom::fabric::decodeFabricRef<
            loom::fabric::FabricMemoryEndpointRef>(
            unsignedBytes(connection.getDestinationAttr()));
        if (!destination)
          return destination.takeError();
        remove = memoryEndpointNamesCore(*source, target) ||
                 memoryEndpointNamesCore(*destination, target) ||
                 memoryEndpointNamesService(*source, removedEndpoints) ||
                 memoryEndpointNamesService(*destination, removedEndpoints);
      } else {
        auto source = loom::fabric::decodeFabricRef<
            loom::fabric::FabricTransportEndpointRef>(
            unsignedBytes(connection.getSourceAttr()));
        if (!source)
          return source.takeError();
        auto destination = loom::fabric::decodeFabricRef<
            loom::fabric::FabricTransportEndpointRef>(
            unsignedBytes(connection.getDestinationAttr()));
        if (!destination)
          return destination.takeError();
        remove = transportEndpointNamesCore(*source, target) ||
                 transportEndpointNamesCore(*destination, target) ||
                 transportEndpointNamesService(*source, removedEndpoints) ||
                 transportEndpointNamesService(*destination, removedEndpoints);
      }
      if (remove)
        connection.erase();
      continue;
    }
    auto leg = mlir::dyn_cast<::fabric::SystemServiceLegCarrierAttachmentOp>(
        operation);
    if (!leg)
      continue;
    auto record = loom::fabric::decodeServiceLegCarrierAttachmentRecord(
        unsignedBytes(leg.getRecordAttr()));
    if (!record)
      return record.takeError();
    if (memoryEndpointNamesCore(record->endpoint(), target) ||
        memoryEndpointNamesService(record->endpoint(), removedEndpoints)) {
      leg.erase();
      continue;
    }
    std::vector<loom::fabric::FabricTransportEndpointRef> carriers;
    for (const auto &carrier : record->carriers())
      if (!transportEndpointNamesCore(carrier, target) &&
          !transportEndpointNamesService(carrier, removedEndpoints))
        carriers.push_back(carrier);
    if (carriers.size() == record->carriers().size())
      continue;
    auto filtered = loom::fabric::ServiceLegCarrierAttachmentRecord::create(
        record->endpoint(), record->kind(), record->legOrdinal(),
        std::move(carriers));
    if (!filtered)
      return filtered.takeError();
    auto bytes =
        loom::fabric::encodeServiceLegCarrierAttachmentRecord(*filtered);
    if (!bytes)
      return bytes.takeError();
    leg.setRecordAttr(denseBytes((*state)->context, *bytes));
  }
  if (llvm::Error error = removeHardwareDomainMembership(
          **state, **root, target, removedEndpoints))
    return error;
  targetOp->erase();
  (*root)->entities[target.id()].operation = nullptr;
  (*root)->entities[target.id()].importedModule.reset();
  for (loom::fabric::FabricEntityId endpoint : removedEndpoints) {
    (*root)->entities[endpoint].operation = nullptr;
    (*root)->entities[endpoint].importedModule.reset();
  }
  if (llvm::Error error = compactImportedModules(**state, **root))
    return error;
  return verifySystemDraft(**root, "AccCore removal");
}

llvm::Error SystemBuilder::replaceSpatialAttachment(
    loom::fabric::AccCoreOccurrenceRef target,
    const loom::fabric::FinalizedFabricRoot &spatialCore) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  if (llvm::Error error = requireAdmissibleModule(**root, spatialCore))
    return error;
  auto targetOp = systemEntity<::fabric::SystemAccCoreOp>(
      **root, target.id(), "AccCore SpatialCore target");
  if (!targetOp)
    return targetOp.takeError();
  const std::optional<std::size_t> previousImport =
      (*root)->entities[target.id()].importedModule;
  if (!previousImport || *previousImport >= (*root)->importedModules.size())
    return invalid("AccCore has no imported SpatialCore Module");
  auto services = memoryAttachmentServices(**root, target);
  if (!services)
    return services.takeError();
  auto imported = importSpatialCore(spatialCore);
  if (!imported)
    return imported.takeError();
  auto endpointMapping = buildSpatialEndpointOrdinalMap(
      (*root)->importedModules[*previousImport],
      (*root)->importedModules[imported->importOrdinal_]);
  if (!endpointMapping)
    return endpointMapping.takeError();

  for (auto iterator = (*root)->operation.getBody().front().begin();
       iterator != (*root)->operation.getBody().front().end();) {
    mlir::Operation &operation = *iterator++;
    auto attachment =
        mlir::dyn_cast<::fabric::SystemSpatialAttachmentOp>(operation);
    if (!attachment)
      continue;
    auto namesCore = attachmentNamesCore(attachment, target);
    if (!namesCore)
      return namesCore.takeError();
    if (*namesCore)
      attachment.erase();
  }
  const detail::ImportedModuleState &module =
      (*root)->importedModules[imported->importOrdinal_];
  targetOp->setSpatialCoreAttr(denseBytes(
      (*state)->context, loom::fabric::encodeFabricImportedModuleTargetRef(
                             {imported->importOrdinal_, module.module})));
  (*root)->entities[target.id()].importedModule = imported->importOrdinal_;
  if (llvm::Error error = emitSpatialAttachments(
          **state, **root, target, imported->importOrdinal_, *services, true))
    return error;
  if (llvm::Error error = remapSpatialServiceLegAttachments(
          **state, **root, target, *endpointMapping))
    return error;
  if (llvm::Error error = compactImportedModules(**state, **root))
    return error;
  return verifySystemDraft(**root, "SpatialCore replacement");
}

llvm::Error SystemBuilder::selectInstructionCoreRealization(
    loom::fabric::InstructionCoreContextRef target,
    loom::fabric::InstructionCoreContextRef prototype) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto targetOp = systemEntity<::fabric::SystemAccCoreOp>(
      **root, target.core.id(), "InstructionCore target");
  if (!targetOp)
    return targetOp.takeError();
  auto prototypeOp = systemEntity<::fabric::SystemAccCoreOp>(
      **root, prototype.core.id(), "InstructionCore prototype");
  if (!prototypeOp)
    return prototypeOp.takeError();
  if (targetOp->getArchitectureAttr() != prototypeOp->getArchitectureAttr())
    return invalid(
        "InstructionCore realization changes the architectural contract");
  targetOp->setMicroarchitectureAttr(prototypeOp->getMicroarchitectureAttr());
  return verifySystemDraft(**root, "InstructionCore realization");
}

llvm::Error SystemBuilder::replaceTransportResource(
    loom::fabric::SystemTransportResourceRef target,
    loom::fabric::SystemTransportResourceRef prototype) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  auto targetOp = systemEntity<::fabric::SystemTransportResourceOp>(
      **root, target.id(), "transport resource target");
  if (!targetOp)
    return targetOp.takeError();
  auto prototypeOp = systemEntity<::fabric::SystemTransportResourceOp>(
      **root, prototype.id(), "transport resource prototype");
  if (!prototypeOp)
    return prototypeOp.takeError();
  if (targetOp->getFunctionType() != prototypeOp->getFunctionType())
    return invalid("transport resource replacement changes endpoint types");

  std::vector<loom::fabric::SystemTransferPatternRecord> prototypePatterns;
  for (mlir::Operation &operation : (*root)->operation.getBody().front()) {
    auto pattern = mlir::dyn_cast<::fabric::SystemTransferPatternOp>(operation);
    if (!pattern)
      continue;
    auto record = loom::fabric::decodeSystemTransferPatternRecord(
        unsignedBytes(pattern.getContractAttr()));
    if (!record)
      return record.takeError();
    if (record->pattern().resource == prototype)
      prototypePatterns.push_back(std::move(*record));
  }
  llvm::sort(prototypePatterns, [](const auto &left, const auto &right) {
    return left.pattern().ordinal < right.pattern().ordinal;
  });
  for (std::size_t ordinal = 0; ordinal < prototypePatterns.size(); ++ordinal)
    if (prototypePatterns[ordinal].pattern().ordinal != ordinal)
      return invalid("transport prototype has a non-dense pattern inventory");

  std::optional<loom::fabric::ClockCrossingContractRecord> crossing;
  if (auto crossingAttribute = prototypeOp->getClockCrossingAttr()) {
    auto decoded = loom::fabric::decodeClockCrossingContractRecord(
        unsignedBytes(crossingAttribute));
    if (!decoded)
      return decoded.takeError();
    crossing = std::move(*decoded);
  }

  for (auto iterator = (*root)->operation.getBody().front().begin();
       iterator != (*root)->operation.getBody().front().end();) {
    mlir::Operation &operation = *iterator++;
    auto pattern = mlir::dyn_cast<::fabric::SystemTransferPatternOp>(operation);
    if (!pattern)
      continue;
    auto record = loom::fabric::decodeSystemTransferPatternRecord(
        unsignedBytes(pattern.getContractAttr()));
    if (!record)
      return record.takeError();
    if (record->pattern().resource == target)
      pattern.erase();
  }

  targetOp->setResourceContractAttr(prototypeOp->getResourceContractAttr());
  const auto targetOwner =
      loom::fabric::FabricTransportEndpointOwnerRef::of(target);
  mlir::OpBuilder builder = insertionBuilder(**state, **root);
  for (const auto &pattern : prototypePatterns) {
    const auto remapEndpoint = [&](const auto &endpoint)
        -> llvm::Expected<loom::fabric::FabricTransportEndpointRef> {
      const auto *owner = std::get_if<loom::fabric::SystemTransportResourceRef>(
          &endpoint.owner.payload);
      if (!owner || *owner != prototype)
        return invalid("transport prototype pattern has a foreign endpoint");
      return loom::fabric::FabricTransportEndpointRef{targetOwner,
                                                      endpoint.ordinal};
    };
    auto ingress = remapEndpoint(pattern.ingress());
    if (!ingress)
      return ingress.takeError();
    std::vector<loom::fabric::FabricTransportEndpointRef> egresses;
    for (const auto &egress : pattern.egresses()) {
      auto mapped = remapEndpoint(egress);
      if (!mapped)
        return mapped.takeError();
      egresses.push_back(*mapped);
    }
    const loom::fabric::FabricInventoryOwnerRef &useOwner =
        pattern.usePattern().owner.catalog();
    const auto *useResource =
        std::get_if<loom::fabric::SystemTransportResourceRef>(
            &useOwner.payload);
    if (!useResource || *useResource != prototype)
      return invalid("transport prototype pattern has a foreign use pattern");
    auto mapped = loom::fabric::SystemTransferPatternRecord::create(
        {target, pattern.pattern().ordinal}, *ingress, std::move(egresses),
        {loom::fabric::FabricUsePatternOwnerRef(
             loom::fabric::FabricInventoryOwnerRef::of(target)),
         pattern.usePattern().ordinal});
    if (!mapped)
      return mapped.takeError();
    auto operation = ::fabric::SystemTransferPatternOp::create(
        builder, (*root)->operation.getLoc(),
        denseBytes((*state)->context,
                   loom::fabric::encodeSystemTransferPatternRecord(*mapped)));
    if (mlir::failed(mlir::verify(operation))) {
      operation.erase();
      return invalid("Fabric rejected the derived transfer pattern");
    }
  }

  if (crossing) {
    if (crossing->transferPattern().resource != prototype)
      return invalid("transport prototype crossing has a foreign pattern");
    auto mapped = loom::fabric::ClockCrossingContractRecord::createAsyncFifo(
        {target, crossing->transferPattern().ordinal}, crossing->sourceClock(),
        crossing->destinationClock(), crossing->depth(),
        crossing->synchronizerStages());
    if (!mapped)
      return mapped.takeError();
    auto bytes = loom::fabric::encodeClockCrossingContractRecord(*mapped);
    if (!bytes)
      return bytes.takeError();
    targetOp->setClockCrossingAttr(denseBytes((*state)->context, *bytes));
  } else {
    targetOp->removeClockCrossingAttr();
  }
  (*root)->entities[target.id()].nextTransferPatternOrdinal =
      prototypePatterns.size();
  (*root)->entities[target.id()].crossingDeclared = crossing.has_value();
  return verifySystemDraft(**root, "transport resource replacement");
}

llvm::Error SystemBuilder::replaceTransportConnection(
    const loom::fabric::FabricTransportEndpointRef &destination,
    const loom::fabric::FabricTransportEndpointRef &source) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  ::fabric::SystemConnectionOp selected;
  for (mlir::Operation &operation : (*root)->operation.getBody().front()) {
    auto connection = mlir::dyn_cast<::fabric::SystemConnectionOp>(operation);
    if (!connection || connection.getMemoryServiceAttr())
      continue;
    auto existing =
        loom::fabric::decodeFabricRef<loom::fabric::FabricTransportEndpointRef>(
            unsignedBytes(connection.getDestinationAttr()));
    if (!existing)
      return existing.takeError();
    if (*existing == destination) {
      if (selected)
        return invalid("transport destination has multiple connections");
      selected = connection;
    }
  }
  if (!selected)
    return invalid("transport destination has no parent connection");
  selected.setSourceAttr(denseBytes(
      (*state)->context, loom::fabric::canonicalFabricBytes(source)));
  return verifySystemDraft(**root, "transport connection replacement");
}

llvm::Error SystemBuilder::replaceSpatialMemoryAttachment(
    const loom::fabric::FabricMemoryEndpointRef &spatialEndpoint,
    loom::fabric::SystemServiceEndpointRef serviceEndpoint) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  ::fabric::SystemSpatialAttachmentOp selected;
  for (mlir::Operation &operation : (*root)->operation.getBody().front()) {
    auto attachment =
        mlir::dyn_cast<::fabric::SystemSpatialAttachmentOp>(operation);
    if (!attachment)
      continue;
    auto existing = loom::fabric::decodeFabricSpatialAttachmentEndpointRef(
        unsignedBytes(attachment.getSpatialEndpointAttr()));
    if (!existing)
      return existing.takeError();
    if (existing->memory() && *existing->memory() == spatialEndpoint) {
      if (selected)
        return invalid("spatial memory endpoint has multiple attachments");
      selected = attachment;
    }
  }
  if (!selected)
    return invalid("spatial memory endpoint has no parent attachment");
  selected.setServiceEndpointAttr(denseBytes(
      (*state)->context, loom::fabric::canonicalFabricBytes(serviceEndpoint)));
  return verifySystemDraft(**root, "memory attachment replacement");
}

llvm::Error SystemBuilder::replaceMemoryServiceConnection(
    const loom::fabric::FabricMemoryEndpointRef &destination,
    const loom::fabric::FabricMemoryEndpointRef &source) {
  auto state = detail::activeState(state_);
  if (!state)
    return state.takeError();
  auto root = derivedSystem(*state, rootOrdinal_);
  if (!root)
    return root.takeError();
  ::fabric::SystemConnectionOp selected;
  for (mlir::Operation &operation : (*root)->operation.getBody().front()) {
    auto connection = mlir::dyn_cast<::fabric::SystemConnectionOp>(operation);
    if (!connection || !connection.getMemoryServiceAttr())
      continue;
    auto existing =
        loom::fabric::decodeFabricRef<loom::fabric::FabricMemoryEndpointRef>(
            unsignedBytes(connection.getDestinationAttr()));
    if (!existing)
      return existing.takeError();
    if (*existing == destination) {
      if (selected)
        return invalid("memory destination has multiple connections");
      selected = connection;
    }
  }
  if (!selected)
    return invalid("memory destination has no parent connection");
  selected.setSourceAttr(denseBytes(
      (*state)->context, loom::fabric::canonicalFabricBytes(source)));
  return verifySystemDraft(**root, "memory connection replacement");
}

} // namespace loom::adg
