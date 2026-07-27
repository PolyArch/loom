#include "Frontend/Compilation/FabricCapabilityIndex.h"

#include "llvm/Support/Error.h"

using namespace loom;

frontend::FabricCapabilityIndex::FabricCapabilityIndex(
    fabric::FabricArtifactView fabric)
    : fabric_(std::move(fabric)),
      operationsBySchema_(dataflow::operationSchemaCount()) {
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
    if (*kind != fabric::FabricEntityKind::FabricFuTemplate)
      continue;
    fabric::FabricFuTemplateRef definition(id);
    for (const fabric::ResolvedFabricOpCapabilityView &capability :
         fabric.resolvedFabricOpCapabilities(definition)) {
      for (dataflow::OperationSchemaId schema :
           capability.enabledOperationSchemas) {
        const std::uint32_t schemaOrdinal = static_cast<std::uint32_t>(schema);
        if (schemaOrdinal >= operationsBySchema_.size())
          continue;
        operationsBySchema_[schemaOrdinal].push_back(
            OperationResource{ownerOrdinal, capability.occurrence});
      }
    }
  }
}

llvm::SmallVector<
    loom::ArtifactReference<loom::fabric::FabricFuTemplateNodeRef>, 4>
frontend::FabricCapabilityIndex::admittingOperationResources(
    const dataflow::CanonicalActorSchemaProjection &actor) const {
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
    if (llvm::Error error = capability->admit(actor)) {
      llvm::consumeError(std::move(error));
      continue;
    }
    result.push_back(ArtifactReference<fabric::FabricFuTemplateNodeRef>{
        owner.identity(), operation.reference});
  }
  return result;
}
