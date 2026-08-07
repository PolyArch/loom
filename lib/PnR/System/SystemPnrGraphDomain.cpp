#include "SystemPnrSearchDomainInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ComponentViewDigest.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingConstraintSet.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
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

bool sameProblem(const FlatSpatialReopenProblem &left,
                 const FlatSpatialReopenProblem &right) {
  return !flatSpatialReopenProblemLess(left, right) &&
         !flatSpatialReopenProblemLess(right, left);
}

bool coversAny(llvm::ArrayRef<::dataflow::GraphRef> covers,
               llvm::ArrayRef<::dataflow::GraphRef> required) {
  return llvm::any_of(covers, [&](::dataflow::GraphRef graph) {
    return llvm::is_contained(required, graph);
  });
}

bool sameFlatDomain(const SystemFlatGraphBindingDomain &left,
                    const SystemFlatGraphBindingDomain &right) {
  if (left.exactSpatialReopenProblems.size() !=
          right.exactSpatialReopenProblems.size() ||
      left.compatibleImmutableSeeds != right.compatibleImmutableSeeds)
    return false;
  return std::equal(left.exactSpatialReopenProblems.begin(),
                    left.exactSpatialReopenProblems.end(),
                    right.exactSpatialReopenProblems.begin(), sameProblem);
}

llvm::Expected<::loom::fabric::FabricArtifactView>
resolveAttachedModule(const ::loom::fabric::FabricSystemRootView &system,
                      const ArtifactIdentity &identity) {
  std::optional<std::uint64_t> dependencyOrdinal;
  for (auto [ordinal, module] :
       llvm::enumerate(system.artifact().importedModules())) {
    if (module.identity() != identity)
      continue;
    if (dependencyOrdinal)
      return invalid("System imports one flat problem Module more than once");
    dependencyOrdinal = ordinal;
  }
  if (!dependencyOrdinal)
    return invalid("flat problem Fabric is not imported by the System");
  const bool attached = llvm::any_of(
      system.artifact().accCoreOccurrences(), [&](const auto core) {
        const auto target = system.spatialCoreTarget(core);
        return target && target->dependencyOrdinal == *dependencyOrdinal;
      });
  if (!attached)
    return invalid("flat problem Fabric is not attached to an AccCore");
  return system.artifact().importedModules()[*dependencyOrdinal];
}

struct ValidatedProblem final {
  FlatSpatialReopenCatalogEntry catalog;
  ::loom::mapping::FinalizedTechMapping tech;
  ::loom::mapping::FinalizedSpatialMappingConstraintSet constraints;
  ::loom::fabric::FabricArtifactView module;
};

llvm::Expected<ValidatedProblem>
validateProblem(const FlatSpatialReopenProblem &problem,
                const ::dataflow::CanonicalDataflowProgramView &dataflow,
                const ::loom::fabric::FabricSystemRootView &system,
                const ArtifactStore &store) {
  if (problem.spatialConfig.domain() != PnrConfigDomain::Spatial)
    return invalid("flat problem has a non-Spatial resolved config");
  if (llvm::Error error = validateComponentViewDigest(
          problem.spatialConfig.schemaDescriptorBytes(),
          problem.spatialConfig.canonicalViewBytes(),
          problem.spatialConfig.digest()))
    return llvm::joinErrors(invalid("flat problem config digest is invalid"),
                            std::move(error));

  auto tech =
      ::loom::mapping::importTechMapping(problem.techMappingReference, store);
  if (!tech)
    return tech.takeError();
  if (tech->view().dataflowIdentity() != dataflow.identity())
    return invalid("flat problem TechMapping has a foreign D owner");
  auto module = resolveAttachedModule(system, tech->view().fabricIdentity());
  if (!module)
    return module.takeError();

  auto constraints = ::loom::mapping::importSpatialMappingConstraintSet(
      problem.spatialConstraintReference, store);
  if (!constraints)
    return constraints.takeError();
  if (constraints->view().dataflowIdentity() != dataflow.identity() ||
      constraints->view().techMappingIdentity() != tech->view().identity() ||
      constraints->view().fabricIdentity() != module->identity())
    return invalid("Spatial MappingConstraintSet has foreign T/F owners");

  FlatSpatialReopenCatalogEntry entry{
      problem, std::vector<::dataflow::GraphRef>(tech->view().covers().begin(),
                                                 tech->view().covers().end())};
  return ValidatedProblem{std::move(entry), std::move(*tech),
                          std::move(*constraints), std::move(*module)};
}

llvm::Expected<FlatSpatialSeedCatalogEntry>
validateSeed(const ArtifactRootReference &reference,
             const ::dataflow::CanonicalDataflowProgramView &dataflow,
             llvm::ArrayRef<ValidatedProblem> problems,
             const ArtifactStore &store) {
  auto seed = ::loom::mapping::importSpatialMapping(reference, store);
  if (!seed)
    return seed.takeError();
  if (seed->view().dataflowIdentity() != dataflow.identity())
    return invalid("flat seed has a foreign D owner");

  bool matchedTech = false;
  bool admitted = false;
  std::vector<::dataflow::GraphRef> covers;
  for (const ValidatedProblem &problem : problems) {
    if (seed->view().techMappingIdentity() != problem.tech.view().identity())
      continue;
    matchedTech = true;
    if (seed->view().fabricIdentity() != problem.module.identity())
      return invalid("flat seed has inconsistent T/F lineage");
    if (covers.empty())
      covers.assign(problem.tech.view().covers().begin(),
                    problem.tech.view().covers().end());
    llvm::Error admission = ::loom::mapping::admitSpatialMappingConstraints(
        dataflow, problem.tech.view(), problem.module,
        problem.constraints.view(), seed->view());
    if (!admission) {
      admitted = true;
      break;
    }
    bool rejected = false;
    llvm::Error remaining = llvm::handleErrors(
        std::move(admission),
        [&](const ::loom::mapping::SpatialMappingConstraintRejection &) {
          rejected = true;
        });
    if (remaining)
      return std::move(remaining);
    if (!rejected)
      return invalid("flat seed constraint admission returned no outcome");
  }
  if (!matchedTech)
    return invalid("flat seed does not match a listed reopen problem");
  if (!admitted)
    return invalid("flat seed is rejected by every matching reopen problem");
  return FlatSpatialSeedCatalogEntry{reference, std::move(covers)};
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
    for (auto [ordinal, module] :
         llvm::enumerate(system.artifact().importedModules())) {
      if (module.identity() != spatial->view().fabricIdentity())
        continue;
      if (moduleDependencyOrdinal)
        return invalid(
            "System imports one SpatialMapping Module more than once");
      moduleDependencyOrdinal = ordinal;
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
    result.push_back(
        {reference, std::move(*spatial), *moduleDependencyOrdinal,
         std::vector<::dataflow::GraphRef>(tech->view().covers().begin(),
                                           tech->view().covers().end())});
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

bool flatSpatialReopenProblemLess(const FlatSpatialReopenProblem &left,
                                  const FlatSpatialReopenProblem &right) {
  if (left.techMappingReference != right.techMappingReference)
    return artifactRootReferenceLess(left.techMappingReference,
                                     right.techMappingReference);
  if (left.spatialConfig.canonicalViewBytes() !=
      right.spatialConfig.canonicalViewBytes())
    return left.spatialConfig.canonicalViewBytes() <
           right.spatialConfig.canonicalViewBytes();
  return artifactRootReferenceLess(left.spatialConstraintReference,
                                   right.spatialConstraintReference);
}

llvm::Expected<CanonicalFlatGraphCatalog>
canonicalizeAndValidateFlatGraphCatalog(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<::dataflow::GraphRef> requiredGraphs,
    const SystemFlatGraphSearchInput &input, const ArtifactStore &store) {
  if (requiredGraphs.empty()) {
    if (!input.reopenProblems.empty() || !input.verifiedSeeds.empty())
      return invalid("instruction-only flat search has an unused graph domain");
    return CanonicalFlatGraphCatalog{};
  }
  if (input.reopenProblems.empty())
    return invalid("flat graph search requires at least one reopen problem");

  std::vector<ValidatedProblem> validated;
  validated.reserve(input.reopenProblems.size());
  for (const FlatSpatialReopenProblem &problem : input.reopenProblems) {
    auto entry = validateProblem(problem, dataflow, fabric, store);
    if (!entry)
      return entry.takeError();
    validated.push_back(std::move(*entry));
  }
  llvm::sort(validated, [](const auto &left, const auto &right) {
    return flatSpatialReopenProblemLess(left.catalog.problem,
                                        right.catalog.problem);
  });
  validated.erase(std::unique(validated.begin(), validated.end(),
                              [](const auto &left, const auto &right) {
                                return sameProblem(left.catalog.problem,
                                                   right.catalog.problem);
                              }),
                  validated.end());
  for (const ValidatedProblem &problem : validated)
    if (!coversAny(problem.catalog.covers, requiredGraphs))
      return invalid("flat reopen problem covers no rooted graph in H");
  for (::dataflow::GraphRef graph : requiredGraphs)
    if (!llvm::any_of(validated, [&](const ValidatedProblem &problem) {
          return llvm::is_contained(problem.catalog.covers, graph);
        }))
      return invalid("flat rooted graph has no compatible reopen problem");

  std::vector<ArtifactRootReference> canonicalSeeds = input.verifiedSeeds;
  llvm::sort(canonicalSeeds, artifactRootReferenceLess);
  canonicalSeeds.erase(
      std::unique(canonicalSeeds.begin(), canonicalSeeds.end()),
      canonicalSeeds.end());
  std::vector<FlatSpatialSeedCatalogEntry> seeds;
  seeds.reserve(canonicalSeeds.size());
  for (const ArtifactRootReference &seed : canonicalSeeds) {
    auto entry = validateSeed(seed, dataflow, validated, store);
    if (!entry)
      return entry.takeError();
    if (!coversAny(entry->covers, requiredGraphs))
      return invalid("flat seed covers no rooted graph in H");
    seeds.push_back(std::move(*entry));
  }

  CanonicalFlatGraphCatalog result;
  result.problems.reserve(validated.size());
  for (ValidatedProblem &problem : validated)
    result.problems.push_back(std::move(problem.catalog));
  result.seeds = std::move(seeds);
  return result;
}

llvm::Expected<SystemFlatGraphBindingDomain>
projectFlatGraphBindingDomain(const CanonicalFlatGraphCatalog &catalog,
                              ::dataflow::GraphRef graph) {
  SystemFlatGraphBindingDomain result;
  for (const FlatSpatialReopenCatalogEntry &problem : catalog.problems)
    if (llvm::is_contained(problem.covers, graph))
      result.exactSpatialReopenProblems.push_back(problem.problem);
  if (result.exactSpatialReopenProblems.empty())
    return invalid("flat rooted graph has no compatible reopen problem");
  for (const FlatSpatialSeedCatalogEntry &seed : catalog.seeds)
    if (llvm::is_contained(seed.covers, graph))
      result.compatibleImmutableSeeds.push_back(seed.reference);
  return result;
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
  std::vector<::dataflow::GraphRef> requiredGraphs;
  std::optional<bool> flatMode;
  std::vector<ArtifactRootReference> hierarchicalReferences(
      constraintSpatialMappings.begin(), constraintSpatialMappings.end());
  SystemFlatGraphSearchInput flatInput;

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
    if (!llvm::is_contained(requiredGraphs, *graph))
      requiredGraphs.push_back(*graph);
    for (const SystemSearchAtom &atom : binding.atoms) {
      const bool atomIsFlat =
          std::holds_alternative<SystemFlatGraphBindingDomain>(atom.domain);
      if (!atomIsFlat &&
          !std::holds_alternative<SystemHierarchicalGraphBindingDomain>(
              atom.domain))
        return invalid("graph binding atom has the wrong domain variant");
      if (flatMode && *flatMode != atomIsFlat)
        return invalid("graph binding atoms mix hierarchical and flat domains");
      flatMode = atomIsFlat;
      if (atomIsFlat) {
        const auto &flat = std::get<SystemFlatGraphBindingDomain>(atom.domain);
        flatInput.reopenProblems.insert(flatInput.reopenProblems.end(),
                                        flat.exactSpatialReopenProblems.begin(),
                                        flat.exactSpatialReopenProblems.end());
        flatInput.verifiedSeeds.insert(flatInput.verifiedSeeds.end(),
                                       flat.compatibleImmutableSeeds.begin(),
                                       flat.compatibleImmutableSeeds.end());
      } else {
        const auto &hierarchical =
            std::get<SystemHierarchicalGraphBindingDomain>(atom.domain);
        hierarchicalReferences.insert(
            hierarchicalReferences.end(),
            hierarchical.compatibleSpatialMappings.begin(),
            hierarchical.compatibleSpatialMappings.end());
      }
    }
  }

  if (graphBindings.empty())
    return llvm::Error::success();
  if (!flatMode)
    return invalid("graph binding domain mode is absent");
  if (*flatMode) {
    auto catalog = canonicalizeAndValidateFlatGraphCatalog(
        dataflow, fabric, requiredGraphs, flatInput, store);
    if (!catalog)
      return catalog.takeError();
    for (const GraphBinding &entry : graphBindings) {
      auto expected = projectFlatGraphBindingDomain(*catalog, entry.graph);
      if (!expected)
        return expected.takeError();
      for (const SystemSearchAtom &atom : entry.binding->atoms) {
        const auto *actual =
            std::get_if<SystemFlatGraphBindingDomain>(&atom.domain);
        if (!actual || !sameFlatDomain(*actual, *expected))
          return invalid("flat graph binding atom domain is not exact");
      }
    }
    return llvm::Error::success();
  }

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
