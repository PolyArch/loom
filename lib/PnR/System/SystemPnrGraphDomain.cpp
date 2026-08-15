#include "../StaticSchedulePressure.h"
#include "../SpatialPhysicalTiming.h"
#include "PnR/SpatialRecurrenceTiming.h"
#include "SystemPnrSearchDomainInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <optional>
#include <utility>
#include <vector>

namespace loom::pnr::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_pnr_search_domain_invalid: " +
                                     message);
}

} // namespace

llvm::Expected<std::vector<SpatialCatalogEntry>>
importSpatialCatalog(llvm::ArrayRef<ArtifactRootReference> references,
                     const ::dataflow::CanonicalDataflowProgramView &dataflow,
                     const ::loom::fabric::FabricSystemRootView &system,
                     const ArtifactStore &store) {
  std::vector<ArtifactRootReference> canonical(references.begin(),
                                               references.end());
  llvm::sort(canonical, artifactRootReferenceLess);
  canonical.erase(std::unique(canonical.begin(), canonical.end()),
                  canonical.end());

  std::vector<ArtifactIdentity> attachedModules;
  for (::loom::fabric::AccCoreOccurrenceRef core :
       system.artifact().accCoreOccurrences()) {
    const auto target = system.spatialCoreTarget(core);
    if (!target ||
        target->dependencyOrdinal >= system.artifact().importedModules().size())
      return invalid("AccCore SpatialCore target does not resolve");
    attachedModules.push_back(system.artifact()
                                  .importedModules()[target->dependencyOrdinal]
                                  .identity());
  }

  std::vector<SpatialCatalogEntry> result;
  result.reserve(canonical.size());
  for (const ArtifactRootReference &reference : canonical) {
    auto spatial = ::loom::mapping::importSpatialMapping(reference, store);
    if (!spatial)
      return spatial.takeError();
    if (spatial->view().dataflowIdentity() != dataflow.identity())
      return invalid(
          "SpatialMapping catalog contains a foreign Dataflow owner");
    if (!llvm::is_contained(attachedModules, spatial->view().fabricIdentity()))
      return invalid(
          "SpatialMapping Fabric is not attached to a System AccCore");

    std::optional<std::uint64_t> moduleDependencyOrdinal;
    const ::loom::fabric::FabricArtifactView *spatialModule = nullptr;
    for (auto [ordinal, module] :
         llvm::enumerate(system.artifact().importedModules())) {
      if (module.identity() != spatial->view().fabricIdentity())
        continue;
      if (moduleDependencyOrdinal)
        return invalid(
            "System imports one SpatialMapping Module more than once");
      moduleDependencyOrdinal = ordinal;
      spatialModule = &module;
    }
    if (!moduleDependencyOrdinal)
      return invalid("SpatialMapping Module dependency does not resolve");

    ArtifactRootReference techReference{
        ::loom::mapping::mappingArtifactSchema.identity.str(),
        ::loom::mapping::mappingArtifactSchema.version,
        spatial->view().techMappingIdentity()};
    auto tech = ::loom::mapping::importTechMapping(techReference, store);
    if (!tech)
      return tech.takeError();
    if (tech->view().dataflowIdentity() != dataflow.identity() ||
        tech->view().fabricIdentity() != spatial->view().fabricIdentity())
      return invalid(
          "SpatialMapping catalog has inconsistent TechMapping lineage");
    auto pressures = projectStaticSchedulePressureByGraph(
        dataflow, tech->view(), *spatialModule, spatial->view());
    if (!pressures)
      return pressures.takeError();
    std::vector<SpatialCatalogGraphProgress> graphProgress;
    std::vector<SpatialRecurrenceTimingProjection> graphRecurrenceTimings;
    graphProgress.reserve(tech->view().covers().size());
    graphRecurrenceTimings.reserve(tech->view().covers().size());
    for (const ::dataflow::GraphRef graph : tech->view().covers()) {
      const std::array<::dataflow::GraphRef, 1> selected{graph};
      auto progress = ::loom::mapping::projectSpatialMappingProgress(
          dataflow, tech->view(), *spatialModule,
          spatial->view().computeBindings(), spatial->view().routeTrees(),
          selected);
      if (!progress)
        return progress.takeError();
      graphProgress.push_back(
          {graph, std::move(progress->routeObligations)});
      auto recurrence = projectSpatialMappingGraphRecurrenceTiming(
          dataflow, tech->view(), *spatialModule, spatial->view(), graph);
      if (!recurrence)
        return recurrence.takeError();
      graphRecurrenceTimings.push_back(std::move(*recurrence));
    }
    result.push_back(
        {reference, std::move(*spatial), *moduleDependencyOrdinal,
         std::vector<::dataflow::GraphRef>(tech->view().covers().begin(),
                                           tech->view().covers().end()),
         std::move(graphProgress), std::move(*pressures),
         std::move(graphRecurrenceTimings), 0, 0, {},
         ::loom::fabric::FabricPhysicalTimingProfileKind::
             NormalizedHeuristic});
  }
  return result;
}

std::vector<::loom::fabric::AccCoreOccurrenceRef>
canonicalSystemAccCores(const ::loom::fabric::FabricSystemRootView &system) {
  std::vector<::loom::fabric::AccCoreOccurrenceRef> cores(
      system.artifact().accCoreOccurrences().begin(),
      system.artifact().accCoreOccurrences().end());
  llvm::sort(cores, [](auto left, auto right) {
    return ::loom::fabric::canonicalFabricBytes(left) <
           ::loom::fabric::canonicalFabricBytes(right);
  });
  return cores;
}

llvm::Error validateSystemBindingDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<SystemSearchBindingDomain> bindings,
    const SystemFrozenConstraintIndex &constraints,
    llvm::ArrayRef<ArtifactRootReference> constraintSpatialMappings,
    const ArtifactStore &store) {
  const auto cores = canonicalSystemAccCores(fabric);
  struct GraphBinding final {
    const SystemSearchBindingDomain *binding = nullptr;
    ::dataflow::GraphRef graph;
  };
  std::vector<GraphBinding> graphBindings;
  std::vector<ArtifactRootReference> hierarchicalReferences(
      constraintSpatialMappings.begin(), constraintSpatialMappings.end());

  for (const SystemSearchBindingDomain &binding : bindings) {
    if (std::holds_alternative<::dataflow::RootThreadLaunchRef>(binding.key)) {
      std::vector<::loom::fabric::AccCoreOccurrenceRef> expected = cores;
      applySystemConstraintRestriction(
          expected, constraints,
          ::mapping::SystemConstraintProjection::ThreadTargetAccCore,
          ::loom::mapping::SystemConstraintSubject{
              std::get<::dataflow::RootThreadLaunchRef>(binding.key)});
      for (const SystemSearchAtom &atom : binding.atoms) {
        const auto *thread =
            std::get_if<SystemThreadBindingDomain>(&atom.domain);
        if (!thread || thread->compatibleAccCores != expected)
          return invalid(
              "thread binding atom has a noncanonical AccCore domain");
      }
      continue;
    }

    const auto launch = std::get<::dataflow::RootedGraphLaunchRef>(binding.key);
    auto graph = dataflow.resolve(launch);
    if (!graph)
      return graph.takeError();
    graphBindings.push_back({&binding, *graph});
    for (const SystemSearchAtom &atom : binding.atoms) {
      const auto *hierarchical =
          std::get_if<SystemHierarchicalGraphBindingDomain>(&atom.domain);
      if (!hierarchical)
        return invalid("graph binding atom has the wrong domain variant");
      hierarchicalReferences.insert(
          hierarchicalReferences.end(),
          hierarchical->compatibleSpatialMappings.begin(),
          hierarchical->compatibleSpatialMappings.end());
    }
  }

  if (graphBindings.empty())
    return llvm::Error::success();

  auto catalog =
      importSpatialCatalog(hierarchicalReferences, dataflow, fabric, store);
  if (!catalog)
    return catalog.takeError();
  for (const GraphBinding &entry : graphBindings) {
    std::vector<ArtifactRootReference> expected;
    for (const SpatialCatalogEntry &mapping : *catalog)
      if (llvm::is_contained(mapping.covers, entry.graph))
        expected.push_back(mapping.reference);
    applySystemConstraintRestriction(
        expected, constraints,
        ::mapping::SystemConstraintProjection::GraphSelectedSpatialMapping,
        ::loom::mapping::SystemConstraintSubject{
            std::get<::dataflow::RootedGraphLaunchRef>(entry.binding->key)});
    for (const SystemSearchAtom &atom : entry.binding->atoms) {
      const auto *actual =
          std::get_if<SystemHierarchicalGraphBindingDomain>(&atom.domain);
      if (!actual || actual->compatibleSpatialMappings != expected)
        return invalid("hierarchical graph binding atom domain is not exact");
    }
  }
  return llvm::Error::success();
}

} // namespace loom::pnr::detail
