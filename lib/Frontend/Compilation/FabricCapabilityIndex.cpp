#include "Frontend/Compilation/FabricCapabilityIndex.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/MemoryOperationPort.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"

#include <utility>

using namespace loom;

namespace {

llvm::Expected<std::optional<::loom::PointerLayout>> resolveActorPointerLayout(
    mlir::Operation *actor,
    const dataflow::CanonicalActorSchemaProjection &projection) {
  auto addressSpace = dataflow::projectActorPointerAddressSpace(projection);
  if (!addressSpace)
    return addressSpace.takeError();
  if (!*addressSpace)
    return std::optional<::loom::PointerLayout>{};
  auto layout = ::loom::resolvePointerLayout(actor, **addressSpace);
  if (!layout)
    return layout.takeError();
  return std::optional<::loom::PointerLayout>(*layout);
}

} // namespace

frontend::FabricCapabilityIndex::FabricCapabilityIndex(
    fabric::FabricArtifactView fabric)
    : fabric_(std::move(fabric)),
      operationsBySchema_(dataflow::operationSchemaCount()),
      memoryPortsBySchema_(dataflow::operationSchemaCount()) {
  index(fabric_, 1);
  if (fabric_.rootKind() != fabric::FabricRootKind::System) {
    for (const fabric::FabricArtifactView &module : fabric_.importedModules())
      index(module, 1);
    return;
  }

  auto system = fabric::requireSystemRoot(fabric_);
  if (!system)
    llvm_unreachable("a finalized System Fabric must admit its typed view");
  std::vector<std::uint64_t> moduleOccurrences(fabric_.importedModules().size(),
                                               0);
  for (fabric::FabricEntityId id = 0;; ++id) {
    const std::optional<fabric::FabricEntityKind> kind = fabric_.entityKind(id);
    if (!kind)
      break;
    if (*kind != fabric::FabricEntityKind::AccCoreOccurrence)
      continue;
    const std::optional<fabric::FabricImportedModuleTargetRef> target =
        system->spatialCoreTarget(fabric::AccCoreOccurrenceRef(id));
    if (!target || target->dependencyOrdinal >= moduleOccurrences.size())
      llvm_unreachable("a finalized AccCore has no imported SpatialCore");
    ++moduleOccurrences[target->dependencyOrdinal];
  }

  for (std::size_t ordinal = 0; ordinal < fabric_.importedModules().size();
       ++ordinal)
    index(fabric_.importedModules()[ordinal], moduleOccurrences[ordinal]);
}

void frontend::FabricCapabilityIndex::index(
    const fabric::FabricArtifactView &fabric,
    std::uint64_t rootOccurrenceCount) {
  const std::size_t ownerOrdinal = owners_.size();
  owners_.push_back(fabric);

  llvm::DenseMap<fabric::FabricEntityId, std::uint64_t> fuOccurrences;
  for (fabric::FabricEntityId id = 0;; ++id) {
    std::optional<fabric::FabricEntityKind> kind = fabric.entityKind(id);
    if (!kind)
      break;
    if (*kind != fabric::FabricEntityKind::FabricFuOccurrence)
      continue;
    std::optional<fabric::FabricFuTemplateRef> definition =
        fabric.fuTemplateOf(fabric::FabricFuOccurrenceRef(id));
    if (!definition)
      llvm_unreachable("a finalized FU occurrence has no template");
    ++fuOccurrences[definition->id()];
  }

  for (fabric::FabricEntityId id = 0;; ++id) {
    std::optional<fabric::FabricEntityKind> kind = fabric.entityKind(id);
    if (!kind)
      break;
    if (*kind == fabric::FabricEntityKind::FabricMemoryOccurrence) {
      fabric::FabricMemoryOccurrenceRef memory(id);
      for (fabric::FabricMemoryOperationPortRef port :
           fabric.memoryOperationPorts(memory)) {
        const fabric::MemoryOperationPortView *record =
            fabric.memoryOperationPort(port);
        if (!record)
          continue;
        llvm::SmallBitVector indexed(dataflow::operationSchemaCount());
        for (const ::fabric::MemoryCapabilityAlternativeRecord &alternative :
             record->capabilityAlternatives()) {
          const std::uint32_t schemaOrdinal = static_cast<std::uint32_t>(
              alternative.actorContractDomain.actorSchema());
          if (schemaOrdinal >= memoryPortsBySchema_.size() ||
              indexed.test(schemaOrdinal))
            continue;
          indexed.set(schemaOrdinal);
          memoryPortsBySchema_[schemaOrdinal].push_back(
              MemoryResource{ownerOrdinal, port, rootOccurrenceCount});
        }
      }
      continue;
    }
    if (*kind == fabric::FabricEntityKind::FabricFuTemplate) {
      fabric::FabricFuTemplateRef definition(id);
      const auto occurrences = fuOccurrences.find(id);
      if (occurrences == fuOccurrences.end())
        continue;
      for (const fabric::ResolvedFabricOpCapabilityView &capability :
           fabric.resolvedFabricOpCapabilities(definition)) {
        for (dataflow::OperationSchemaId schema :
             capability.enabledOperationSchemas) {
          const std::uint32_t schemaOrdinal =
              static_cast<std::uint32_t>(schema);
          if (schemaOrdinal >= operationsBySchema_.size())
            continue;
          operationsBySchema_[schemaOrdinal].push_back(
              OperationResource{ownerOrdinal, capability.occurrence,
                                rootOccurrenceCount, occurrences->second});
        }
      }
    }
  }
}

llvm::SmallVector<
    loom::ArtifactReference<loom::fabric::FabricFuTemplateNodeRef>, 4>
frontend::FabricCapabilityIndex::admittingOperationResources(
    const dataflow::CanonicalActorSchemaProjection &actor,
    unsigned indexBitWidth, const ::loom::PointerLayout *pointerLayout) const {
  llvm::SmallVector<ArtifactReference<fabric::FabricFuTemplateNodeRef>, 4>
      result;
  const std::uint32_t schemaOrdinal = static_cast<std::uint32_t>(actor.schema);
  if (schemaOrdinal >= operationsBySchema_.size())
    return result;
  for (const OperationResource &operation :
       operationsBySchema_[schemaOrdinal]) {
    const fabric::FabricArtifactView &owner = owners_[operation.ownerOrdinal];
    const fabric::ResolvedFabricOpCapabilityView *capability =
        owner.resolvedFabricOpCapability(operation.reference);
    if (!capability)
      continue;
    if (llvm::Error error =
            capability->admit(actor, indexBitWidth, pointerLayout)) {
      llvm::consumeError(std::move(error));
      continue;
    }
    result.push_back(ArtifactReference<fabric::FabricFuTemplateNodeRef>{
        owner.identity(), operation.reference});
  }
  return result;
}

llvm::Expected<std::uint64_t>
frontend::FabricCapabilityIndex::admittingOperationResourceCount(
    const dataflow::CanonicalActorSchemaProjection &actor,
    unsigned indexBitWidth, const ::loom::PointerLayout *pointerLayout) const {
  std::uint64_t result = 0;
  const std::uint32_t schemaOrdinal = static_cast<std::uint32_t>(actor.schema);
  if (schemaOrdinal >= operationsBySchema_.size())
    return result;
  for (const OperationResource &operation :
       operationsBySchema_[schemaOrdinal]) {
    const fabric::FabricArtifactView &owner = owners_[operation.ownerOrdinal];
    const fabric::ResolvedFabricOpCapabilityView *capability =
        owner.resolvedFabricOpCapability(operation.reference);
    if (!capability)
      continue;
    if (llvm::Error error =
            capability->admit(actor, indexBitWidth, pointerLayout)) {
      llvm::consumeError(std::move(error));
      continue;
    }
    const std::optional<std::uint64_t> concrete = llvm::checkedMulUnsigned(
        operation.rootOccurrenceCount, operation.localOccurrenceCount);
    if (!concrete)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "fabric_capability_count_overflow: operation occurrence product");
    const std::optional<std::uint64_t> total =
        llvm::checkedAddUnsigned(result, *concrete);
    if (!total)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "fabric_capability_count_overflow: operation occurrence sum");
    result = *total;
  }
  return result;
}

llvm::Expected<llvm::SmallVector<
    loom::ArtifactReference<loom::fabric::FabricFuTemplateNodeRef>, 4>>
frontend::FabricCapabilityIndex::admittingOperationResources(
    mlir::Operation *actor) const {
  auto projection = dataflow::projectRegisteredActorSchemaProjection(actor);
  if (!projection)
    return projection.takeError();
  auto indexBitWidth = loom::getIndexBitWidth(actor);
  if (!indexBitWidth)
    return indexBitWidth.takeError();
  auto pointerLayout = resolveActorPointerLayout(actor, *projection);
  if (!pointerLayout)
    return pointerLayout.takeError();
  return admittingOperationResources(
      *projection, *indexBitWidth, *pointerLayout ? &**pointerLayout : nullptr);
}

llvm::Expected<std::uint64_t>
frontend::FabricCapabilityIndex::admittingOperationResourceCount(
    mlir::Operation *actor) const {
  auto projection = dataflow::projectRegisteredActorSchemaProjection(actor);
  if (!projection)
    return projection.takeError();
  auto indexBitWidth = loom::getIndexBitWidth(actor);
  if (!indexBitWidth)
    return indexBitWidth.takeError();
  auto pointerLayout = resolveActorPointerLayout(actor, *projection);
  if (!pointerLayout)
    return pointerLayout.takeError();
  return admittingOperationResourceCount(
      *projection, *indexBitWidth, *pointerLayout ? &**pointerLayout : nullptr);
}

llvm::Expected<llvm::SmallVector<
    loom::ArtifactReference<loom::fabric::FabricMemoryCapabilityAlternativeRef>,
    4>>
frontend::FabricCapabilityIndex::admittingMemoryResources(
    mlir::Operation *actor) const {
  auto projection = dataflow::projectRegisteredActorSchemaProjection(actor);
  if (!projection)
    return projection.takeError();
  if (dataflow::actorKind(projection->schema) !=
      dataflow::CanonicalDataflowActorKind::Memory)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "fabric_capability_query_invalid: actor is not a memory actor");

  auto service = dataflow::semantics::CanonicalService::forActor(actor);
  if (!service)
    return service.takeError();
  std::optional<dataflow::semantics::CanonicalMemoryAccessView> access;
  if (service->kind() != dataflow::semantics::ServiceKind::MemoryFence) {
    auto projected = dataflow::semantics::getCanonicalMemoryAccessView(actor);
    if (!projected)
      return projected.takeError();
    access.emplace(*projected);
  }

  llvm::SmallVector<
      ArtifactReference<fabric::FabricMemoryCapabilityAlternativeRef>, 4>
      result;
  const std::uint32_t schemaOrdinal =
      static_cast<std::uint32_t>(projection->schema);
  if (schemaOrdinal >= memoryPortsBySchema_.size())
    return result;
  for (const MemoryResource &resource : memoryPortsBySchema_[schemaOrdinal]) {
    const fabric::FabricArtifactView &owner = owners_[resource.ownerOrdinal];
    const fabric::MemoryOperationPortView *port =
        owner.memoryOperationPort(resource.reference);
    if (!port)
      continue;
    auto matches = port->matchingCapabilities(*projection, *service, access);
    if (!matches)
      return matches.takeError();
    for (const ::fabric::MemoryCapabilityMatch &match : *matches) {
      result.push_back(
          ArtifactReference<fabric::FabricMemoryCapabilityAlternativeRef>{
              owner.identity(),
              fabric::FabricMemoryCapabilityAlternativeRef{
                  resource.reference, match.alternativeOrdinal}});
    }
  }
  return result;
}

llvm::Expected<std::uint64_t>
frontend::FabricCapabilityIndex::admittingMemoryResourceCount(
    mlir::Operation *actor) const {
  auto projection = dataflow::projectRegisteredActorSchemaProjection(actor);
  if (!projection)
    return projection.takeError();
  if (dataflow::actorKind(projection->schema) !=
      dataflow::CanonicalDataflowActorKind::Memory)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "fabric_capability_query_invalid: actor is not a memory actor");

  auto service = dataflow::semantics::CanonicalService::forActor(actor);
  if (!service)
    return service.takeError();
  std::optional<dataflow::semantics::CanonicalMemoryAccessView> access;
  if (service->kind() != dataflow::semantics::ServiceKind::MemoryFence) {
    auto projected = dataflow::semantics::getCanonicalMemoryAccessView(actor);
    if (!projected)
      return projected.takeError();
    access.emplace(*projected);
  }

  std::uint64_t result = 0;
  const std::uint32_t schemaOrdinal =
      static_cast<std::uint32_t>(projection->schema);
  if (schemaOrdinal >= memoryPortsBySchema_.size())
    return result;
  for (const MemoryResource &resource : memoryPortsBySchema_[schemaOrdinal]) {
    const fabric::FabricArtifactView &owner = owners_[resource.ownerOrdinal];
    const fabric::MemoryOperationPortView *port =
        owner.memoryOperationPort(resource.reference);
    if (!port)
      continue;
    auto matches = port->matchingCapabilities(*projection, *service, access);
    if (!matches)
      return matches.takeError();
    if (matches->empty())
      continue;
    const std::optional<std::uint64_t> total =
        llvm::checkedAddUnsigned(result, resource.rootOccurrenceCount);
    if (!total)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "fabric_capability_count_overflow: memory occurrence sum");
    result = *total;
  }
  return result;
}

llvm::Expected<std::optional<frontend::ExactFabricCapabilityMiss>>
frontend::FabricCapabilityIndex::firstInadmissibleActor(
    const dataflow::CanonicalDataflowArtifact &program) const {
  auto view = program.view();
  if (!view)
    return view.takeError();
  for (const dataflow::CanonicalActorView &actor : view->actors()) {
    auto projection =
        dataflow::projectRegisteredActorSchemaProjection(actor.op);
    if (!projection)
      return projection.takeError();
    if (actor.kind == dataflow::CanonicalDataflowActorKind::Memory) {
      auto resources = admittingMemoryResources(actor.op);
      if (!resources)
        return resources.takeError();
      if (resources->empty())
        return std::optional<frontend::ExactFabricCapabilityMiss>{
            frontend::ExactFabricCapabilityMiss{
                actor.ref, actor.kind, projection->schema, projection->type}};
      continue;
    }
    auto resources = admittingOperationResources(actor.op);
    if (!resources)
      return resources.takeError();
    if (resources->empty())
      return std::optional<frontend::ExactFabricCapabilityMiss>{
          frontend::ExactFabricCapabilityMiss{
              actor.ref, actor.kind, projection->schema, projection->type}};
  }
  return std::optional<frontend::ExactFabricCapabilityMiss>{};
}
