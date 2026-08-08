#include "Fabric/Artifact/FabricSystemRootView.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <type_traits>
#include <utility>
#include <variant>

namespace loom::fabric {
namespace {

llvm::Error invalidPhysicalOwner(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

FabricInventoryOwnerRef
inventoryOwner(const FabricModulePhysicalOwnerRef &owner) {
  return std::visit(
      [](const auto &value) -> FabricInventoryOwnerRef {
        using Type = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Type, LocalMemoryServiceRef>)
          return FabricInventoryOwnerRef::of(value.underlying());
        else
          return FabricInventoryOwnerRef::of(value);
      },
      owner.payload());
}

} // namespace

llvm::Expected<ResolvedFabricPhysicalOwnerView>
FabricSystemRootView::resolvePhysicalOwner(
    const FabricPhysicalOccurrenceOwnerRef &owner) const {
  if (owner.kind() == FabricPhysicalOccurrenceOwnerKind::DirectSystemOwner) {
    const auto &local = std::get<FabricInventoryOwnerRef>(owner.payload());
    if (llvm::Error error = validateFabricRef(artifact_, local))
      return std::move(error);
    return ResolvedFabricPhysicalOwnerView{artifact_, local};
  }

  const auto &internal =
      std::get<SpatialCoreInternalOccurrenceRef>(owner.payload());
  const std::optional<FabricImportedModuleTargetRef> target =
      spatialCoreTarget(internal.spatialCore.core);
  if (!target)
    return invalidPhysicalOwner("SpatialCore has no imported Module target");
  const llvm::ArrayRef<FabricArtifactView> modules =
      artifact_.importedModules();
  if (target->dependencyOrdinal >= modules.size())
    return invalidPhysicalOwner(
        "SpatialCore imported Module dependency is out of range");
  FabricArtifactView module =
      modules[static_cast<std::size_t>(target->dependencyOrdinal)];
  if (module.moduleRootTemplate() != target->target)
    return invalidPhysicalOwner(
        "SpatialCore imported Module target does not match");
  const auto &local =
      std::get<FabricModulePhysicalOwnerRef>(internal.target.payload());
  FabricInventoryOwnerRef resolved = inventoryOwner(local);
  if (llvm::Error error = validateFabricRef(module, resolved))
    return std::move(error);
  return ResolvedFabricPhysicalOwnerView{std::move(module),
                                         std::move(resolved)};
}

} // namespace loom::fabric
