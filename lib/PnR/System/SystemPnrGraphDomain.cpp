#include "../SpatialPhysicalTiming.h"
#include "../StaticSchedulePressure.h"
#include "PnR/SpatialRecurrenceTiming.h"
#include "SystemPnrSearchDomainInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <map>
#include <memory>
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

llvm::Expected<std::vector<SpatialCatalogEntry>> importSpatialCatalog(
    llvm::ArrayRef<ArtifactRootReference> references,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &system,
    const ArtifactStore &store,
    const ::loom::mapping::SpatialMappingImportContext *imports,
    SpatialCatalogImportStatistics *statistics) {
  std::vector<ArtifactRootReference> canonical(references.begin(),
                                               references.end());
  llvm::sort(canonical, artifactRootReferenceLess);
  canonical.erase(std::unique(canonical.begin(), canonical.end()),
                  canonical.end());

  std::optional<::loom::mapping::SpatialMappingImportContext> ownedImports;
  if (!imports) {
    auto built =
        ::loom::mapping::buildSpatialMappingImportContext(canonical, store);
    if (!built)
      return built.takeError();
    ownedImports.emplace(std::move(*built));
    imports = &*ownedImports;
  }
  if (llvm::ArrayRef<ArtifactRootReference>(canonical) != imports->references())
    return invalid(
        "SpatialMapping import context does not match the exact catalog");

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
  std::map<ArtifactRootReference,
           std::unique_ptr<::loom::mapping::FinalizedTechMapping>,
           decltype(&artifactRootReferenceLess)>
      techMappings(&artifactRootReferenceLess);
  for (const ArtifactRootReference &reference : canonical) {
    auto spatial =
        ::loom::mapping::resolveSpatialMappingImportHandle(*imports, reference);
    if (!spatial)
      return spatial.takeError();
    if ((*spatial)->view().dataflowIdentity() != dataflow.identity())
      return invalid(
          "SpatialMapping catalog contains a foreign Dataflow owner");
    if (!llvm::is_contained(attachedModules,
                            (*spatial)->view().fabricIdentity()))
      return invalid(
          "SpatialMapping Fabric is not attached to a System AccCore");

    std::optional<std::uint64_t> moduleDependencyOrdinal;
    const ::loom::fabric::FabricArtifactView *spatialModule = nullptr;
    for (auto [ordinal, module] :
         llvm::enumerate(system.artifact().importedModules())) {
      if (module.identity() != (*spatial)->view().fabricIdentity())
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
        (*spatial)->view().techMappingIdentity()};
    if (statistics)
      ++statistics->techMappingImportRequests;
    auto tech = techMappings.find(techReference);
    if (tech == techMappings.end()) {
      auto imported = ::loom::mapping::importTechMapping(techReference, store);
      if (!imported)
        return imported.takeError();
      tech =
          techMappings
              .emplace(techReference,
                       std::make_unique<::loom::mapping::FinalizedTechMapping>(
                           std::move(*imported)))
              .first;
      if (statistics)
        ++statistics->techMappingImportMisses;
    } else if (statistics) {
      ++statistics->techMappingImportHits;
    }
    const ::loom::mapping::TechMappingView &techView = tech->second->view();
    if (techView.identity() != techReference.artifact ||
        techView.dataflowIdentity() != dataflow.identity() ||
        techView.fabricIdentity() != (*spatial)->view().fabricIdentity())
      return invalid(
          "SpatialMapping catalog has inconsistent TechMapping lineage");
    auto pressures = projectStaticSchedulePressureByGraph(
        dataflow, techView, *spatialModule, (*spatial)->view());
    if (!pressures)
      return pressures.takeError();
    std::vector<SpatialCatalogGraphProgress> graphProgress;
    std::vector<std::shared_ptr<const FrozenSpatialRecurrenceTimingDemand>>
        graphRecurrenceDemands;
    graphProgress.reserve(techView.covers().size());
    graphRecurrenceDemands.reserve(techView.covers().size());
    for (const ::dataflow::GraphRef graph : techView.covers()) {
      const std::array<::dataflow::GraphRef, 1> selected{graph};
      auto progress = ::loom::mapping::projectSpatialMappingProgress(
          dataflow, techView, *spatialModule,
          (*spatial)->view().computeBindings(),
          (*spatial)->view().registerFifoTransfers(),
          (*spatial)->view().routeTrees(), selected);
      if (!progress)
        return progress.takeError();
      graphProgress.push_back({graph, std::move(progress->routeObligations)});
      auto recurrence = freezeSpatialMappingGraphRecurrenceTimingDemand(
          dataflow, techView, *spatialModule, (*spatial)->view(), graph);
      if (!recurrence)
        return recurrence.takeError();
      graphRecurrenceDemands.push_back(std::move(*recurrence));
    }
    result.push_back(
        {reference,
         std::move(*spatial),
         *moduleDependencyOrdinal,
         std::vector<::dataflow::GraphRef>(techView.covers().begin(),
                                           techView.covers().end()),
         std::move(graphProgress),
         std::move(*pressures),
         std::move(graphRecurrenceDemands),
         0,
         0,
         {},
         ::loom::fabric::FabricPhysicalTimingProfileKind::NormalizedHeuristic});
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
    llvm::ArrayRef<SpatialCatalogEntry> spatialCatalog) {
  const auto cores = canonicalSystemAccCores(fabric);
  struct GraphBinding final {
    const SystemSearchBindingDomain *binding = nullptr;
    ::dataflow::GraphRef graph;
  };
  std::vector<GraphBinding> graphBindings;
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
    }
  }

  if (graphBindings.empty())
    return llvm::Error::success();

  for (const GraphBinding &entry : graphBindings) {
    std::vector<ArtifactRootReference> expected;
    for (const SpatialCatalogEntry &mapping : spatialCatalog)
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
