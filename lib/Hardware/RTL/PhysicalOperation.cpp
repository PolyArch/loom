#include "Hardware/RTL/PhysicalOperation.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <utility>

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "rtl_physical_operation_invalid: " + message);
}

llvm::Expected<fabric::FabricPhysicalOccurrenceOwnerRef>
qualifyOperation(fabric::SpatialCoreOccurrenceRef spatialCore,
                 fabric::FabricFuOccurrenceNodeRef localOccurrence) {
  auto owner = fabric::FabricModulePhysicalOwnerRef::create(localOccurrence);
  if (!owner)
    return owner.takeError();
  auto target = fabric::FabricModulePhysicalTargetRef::create(*owner);
  if (!target)
    return target.takeError();
  return fabric::FabricPhysicalOccurrenceOwnerRef::create(
      fabric::SpatialCoreInternalOccurrenceRef{spatialCore,
                                               std::move(*target)});
}

} // namespace

llvm::Expected<fabric::FabricArtifactView>
resolveFabricSpatialCoreModule(const fabric::FabricSystemRootView &system,
                               fabric::SpatialCoreOccurrenceRef spatialCore) {
  auto target = system.spatialCoreTarget(spatialCore.core);
  if (!target)
    return invalid("SpatialCore occurrence has no imported Module target");
  llvm::ArrayRef<fabric::FabricArtifactView> modules =
      system.artifact().importedModules();
  if (target->dependencyOrdinal >= modules.size())
    return invalid("SpatialCore imported Module ordinal is out of range");
  const fabric::FabricArtifactView &module = modules[target->dependencyOrdinal];
  if (module.moduleRootTemplate() != target->target)
    return invalid("SpatialCore imported Module target is inconsistent");
  return module;
}

llvm::Expected<ResolvedFabricPhysicalOperation> resolveFabricPhysicalOperation(
    const fabric::FabricSystemRootView &system,
    const fabric::FabricPhysicalOccurrenceOwnerRef &occurrence) {
  if (occurrence.kind() !=
      fabric::FabricPhysicalOccurrenceOwnerKind::SpatialCoreInternal)
    return invalid("operation occurrence is not SpatialCore-internal");
  const auto &internal =
      std::get<fabric::SpatialCoreInternalOccurrenceRef>(occurrence.payload());
  if (internal.target.kind() != fabric::FabricModulePhysicalTargetKind::Owner)
    return invalid("operation occurrence does not name a Module owner");
  const auto &owner =
      std::get<fabric::FabricModulePhysicalOwnerRef>(internal.target.payload());
  if (owner.kind() != fabric::FabricModulePhysicalOwnerKind::FuOccurrenceNode)
    return invalid("operation occurrence does not name an FU node");
  const auto localOccurrence =
      std::get<fabric::FabricFuOccurrenceNodeRef>(owner.payload());
  auto module = resolveFabricSpatialCoreModule(system, internal.spatialCore);
  if (!module)
    return module.takeError();
  if (llvm::Error error = fabric::validateFabricRef(*module, localOccurrence))
    return std::move(error);
  const auto *capability = module->resolvedFabricOpCapability(localOccurrence);
  if (!capability)
    return invalid("operation occurrence has no resolved capability");
  return ResolvedFabricPhysicalOperation{occurrence, std::move(*module),
                                         localOccurrence, capability};
}

llvm::Expected<std::vector<ResolvedFabricPhysicalOperation>>
enumerateFabricPhysicalOperations(const fabric::FabricSystemRootView &system) {
  std::vector<ResolvedFabricPhysicalOperation> result;
  for (fabric::AccCoreOccurrenceRef core :
       system.artifact().accCoreOccurrences()) {
    const fabric::SpatialCoreOccurrenceRef spatialCore{core};
    auto module = resolveFabricSpatialCoreModule(system, spatialCore);
    if (!module)
      return module.takeError();
    for (fabric::FabricFuOccurrenceRef occurrence : module->fuOccurrences()) {
      const auto definition = module->fuTemplateOf(occurrence);
      if (!definition)
        return invalid("FU occurrence has no exact template owner");
      for (const fabric::ResolvedFabricOpCapabilityView &capability :
           module->resolvedFabricOpCapabilities(*definition)) {
        auto localOccurrence = fabric::deriveFabricFuOccurrenceNode(
            *module, capability.occurrence, occurrence);
        if (!localOccurrence)
          return localOccurrence.takeError();
        auto physical = qualifyOperation(spatialCore, *localOccurrence);
        if (!physical)
          return physical.takeError();
        auto resolved = resolveFabricPhysicalOperation(system, *physical);
        if (!resolved)
          return resolved.takeError();
        result.push_back(std::move(*resolved));
      }
    }
  }
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return fabric::canonicalFabricBytes(lhs.physicalOccurrence) <
           fabric::canonicalFabricBytes(rhs.physicalOccurrence);
  });
  if (std::adjacent_find(
          result.begin(), result.end(), [](const auto &lhs, const auto &rhs) {
            return lhs.physicalOccurrence == rhs.physicalOccurrence;
          }) != result.end())
    return invalid("physical operation occurrence inventory is not unique");
  return result;
}

} // namespace loom::hardware::rtl
