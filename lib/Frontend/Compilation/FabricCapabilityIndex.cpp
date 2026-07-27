#include "Frontend/Compilation/FabricCapabilityIndex.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/IR/MemoryOperationPort.h"

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Error.h"

using namespace loom;

frontend::FabricCapabilityIndex::FabricCapabilityIndex(
    fabric::FabricArtifactView fabric)
    : fabric_(std::move(fabric)),
      operationsBySchema_(dataflow::operationSchemaCount()),
      memoryPortsBySchema_(dataflow::operationSchemaCount()) {
  index(fabric_);
  for (const fabric::FabricArtifactView &module : fabric_.importedModules())
    index(module);
}

void frontend::FabricCapabilityIndex::index(
    const fabric::FabricArtifactView &fabric) {
  const std::size_t ownerOrdinal = owners_.size();
  owners_.push_back(fabric);
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
              MemoryResource{ownerOrdinal, port});
        }
      }
      continue;
    }
    if (*kind == fabric::FabricEntityKind::FabricFuTemplate) {
      fabric::FabricFuTemplateRef definition(id);
      for (const fabric::ResolvedFabricOpCapabilityView &capability :
           fabric.resolvedFabricOpCapabilities(definition)) {
        for (dataflow::OperationSchemaId schema :
             capability.enabledOperationSchemas) {
          const std::uint32_t schemaOrdinal =
              static_cast<std::uint32_t>(schema);
          if (schemaOrdinal >= operationsBySchema_.size())
            continue;
          operationsBySchema_[schemaOrdinal].push_back(
              OperationResource{ownerOrdinal, capability.occurrence});
        }
      }
    }
  }
}

llvm::SmallVector<
    loom::ArtifactReference<loom::fabric::FabricFuTemplateNodeRef>, 4>
frontend::FabricCapabilityIndex::admittingOperationResources(
    const dataflow::CanonicalActorSchemaProjection &actor,
    unsigned indexBitWidth) const {
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
    if (llvm::Error error = capability->admit(actor, indexBitWidth)) {
      llvm::consumeError(std::move(error));
      continue;
    }
    result.push_back(ArtifactReference<fabric::FabricFuTemplateNodeRef>{
        owner.identity(), operation.reference});
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
  return admittingOperationResources(*projection, *indexBitWidth);
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
