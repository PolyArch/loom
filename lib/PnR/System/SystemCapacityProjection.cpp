#include "SystemCapacityProjection.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SpatialMappingCapacityVerification.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::pnr::detail {
namespace {

using namespace ::loom::mapping::detail;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_capacity_projection_invalid: " +
                                     message);
}

void appendU32(std::string &bytes, std::uint32_t value) {
  for (unsigned shift = 24;; shift -= 8) {
    bytes.push_back(static_cast<char>(value >> shift));
    if (shift == 0)
      break;
  }
}

void appendU64(std::string &bytes, std::uint64_t value) {
  for (unsigned shift = 56;; shift -= 8) {
    bytes.push_back(static_cast<char>(value >> shift));
    if (shift == 0)
      break;
  }
}

void appendSized(std::string &bytes, llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.append(reinterpret_cast<const char *>(value.data()), value.size());
}

void appendSized(std::string &bytes, llvm::StringRef value) {
  appendU64(bytes, value.size());
  bytes.append(value.data(), value.size());
}

std::vector<std::vector<::loom::fabric::FabricPhysicalTraversalRef>>
spatialRouteTraversals(const ::loom::mapping::SpatialMappingView &mapping) {
  std::vector<std::vector<::loom::fabric::FabricPhysicalTraversalRef>> result;
  result.reserve(mapping.routeTrees().size());
  for (const auto &route : mapping.routeTrees()) {
    std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversals;
    if (route.localTraversal)
      traversals.push_back(*route.localTraversal);
    for (const auto &node : route.nodes)
      if (node.incomingTraversal)
        traversals.push_back(*node.incomingTraversal);
    for (const auto &sink : route.sinks)
      if (sink.localTraversal)
        traversals.push_back(*sink.localTraversal);
    result.push_back(std::move(traversals));
  }
  return result;
}

template <typename Ref> std::string fabricRefKey(const Ref &reference) {
  const auto bytes = ::loom::fabric::canonicalFabricBytes(reference);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

llvm::Expected<std::string>
instructionActivationKey(const ArtifactIdentity &dataflowIdentity,
                         const SystemInstructionResourceUseSelection &use) {
  auto root = ::dataflow::encodeDataflowReference(dataflowIdentity, use.root);
  if (!root)
    return root.takeError();
  const auto context = ::loom::fabric::canonicalFabricBytes(use.context);
  std::string result;
  appendU32(result, 0);
  appendSized(result, *root);
  appendSized(result, context);
  return result;
}

std::string serviceActivationKey(const SystemServiceResourceUseSelection &use) {
  std::string result;
  appendU32(result, 1);
  appendU32(result, use.context);
  appendU32(result, use.subject);
  appendU32(result, use.branch);
  return result;
}

} // namespace

llvm::Expected<std::unique_ptr<SystemCapacityModel>> buildSystemCapacityModel(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<::loom::fabric::AccCoreOccurrenceRef> cores,
    llvm::ArrayRef<PnrIndex> coreTargetClasses,
    llvm::ArrayRef<PnrIndex> mappingTargetClasses,
    llvm::ArrayRef<SpatialCatalogEntry> spatialCatalog,
    llvm::ArrayRef<FrozenSystemGraphExecutionDecision> graphDecisions,
    llvm::ArrayRef<FrozenSystemInstructionUsePatternDomain>
        instructionUsePatterns,
    llvm::ArrayRef<FrozenSystemMemoryServiceBinding> memoryBindings,
    llvm::ArrayRef<FrozenSystemConsistencyUsePatternDomain>
        consistencyUsePatterns,
    const FrozenEndpointRoutingTopology &routingTopology) {
  if (cores.size() != coreTargetClasses.size() ||
      spatialCatalog.size() != mappingTargetClasses.size())
    return invalid("System capacity catalogs have inconsistent widths");
  if (cores.size() == std::numeric_limits<std::size_t>::max())
    return invalid("System capacity namespace count exceeds size_t");
  if (cores.size() >
      static_cast<std::size_t>(std::numeric_limits<PnrIndex>::max()))
    return invalid("System capacity core catalog exceeds PnrIndex");
  if (spatialCatalog.size() >
      static_cast<std::size_t>(std::numeric_limits<PnrIndex>::max()))
    return invalid("System capacity mapping catalog exceeds PnrIndex");

  std::vector<ResourceCapacityNamespaceView> namespaces;
  namespaces.reserve(cores.size() + 1);
  namespaces.push_back(
      {&fabric.artifact(), rootResourceCapacityQualifier(fabric.artifact())});
  std::map<PnrIndex, const ::loom::fabric::FabricArtifactView *>
      modulesByTargetClass;
  for (const auto &[coreOrdinal, core] : llvm::enumerate(cores)) {
    const auto target = fabric.spatialCoreTarget(core);
    if (!target ||
        target->dependencyOrdinal >= fabric.artifact().importedModules().size())
      return invalid("AccCore capacity namespace has no imported Module");
    const auto &module =
        fabric.artifact().importedModules()[target->dependencyOrdinal];
    const auto [position, inserted] = modulesByTargetClass.try_emplace(
        coreTargetClasses[coreOrdinal], &module);
    if (!inserted && position->second->identity() != module.identity())
      return invalid(
          "one System target class names different imported Modules");
    namespaces.push_back(
        {&module, occurrenceResourceCapacityQualifier(
                      fabric.artifact(),
                      ::loom::fabric::SpatialCoreOccurrenceRef{core})});
  }

  std::vector<ResourceCapacityPatternSource> patternSources;
  for (const auto &domain : instructionUsePatterns)
    for (const auto &pattern : domain.patterns)
      patternSources.push_back({0, pattern});
  for (const auto &binding : memoryBindings)
    for (const auto &domain : binding.usePatternDomains)
      for (const auto &pattern : domain.patterns)
        patternSources.push_back({0, pattern});
  for (const auto &domain : consistencyUsePatterns)
    for (const auto &pattern : domain.patterns)
      patternSources.push_back({0, pattern});

  std::vector<ResourceCapacityTraversalSource> traversalSources;
  traversalSources.reserve(routingTopology.traversals().size());
  for (const auto &traversal : routingTopology.traversals())
    traversalSources.push_back({0, traversal.reference});

  using PatternCatalog =
      std::map<std::string, ::loom::fabric::FabricUsePatternRef>;
  using TraversalCatalog =
      std::map<std::string, ::loom::fabric::FabricPhysicalTraversalRef>;
  std::map<PnrIndex, PatternCatalog> patternsByTargetClass;
  std::map<PnrIndex, TraversalCatalog> traversalsByTargetClass;
  std::vector<SystemCapacityModel::ImportedProjection> importedProjections;
  importedProjections.reserve(spatialCatalog.size());
  for (const auto &[mappingOrdinal, entry] : llvm::enumerate(spatialCatalog)) {
    const PnrIndex targetClass = mappingTargetClasses[mappingOrdinal];
    const auto module = modulesByTargetClass.find(targetClass);
    if (module == modulesByTargetClass.end())
      return invalid("SpatialMapping target class has no System occurrence");
    const auto &mapping = entry.mapping.view();
    if (mapping.fabricIdentity() != module->second->identity())
      return invalid("SpatialMapping target class has the wrong Module");

    SystemCapacityModel::ImportedProjection projection{
        mapping.identity(), {}, {}};
    projection.uses.reserve(mapping.resourceUses().size());
    for (const auto &use : mapping.resourceUses()) {
      auto activation = deriveSpatialCapacityActivationKey(
          *module->second, dataflow.identity(), use);
      if (!activation)
        return activation.takeError();
      projection.uses.push_back({use.useSite, std::move(*activation)});
      patternsByTargetClass[targetClass].try_emplace(fabricRefKey(use.useSite),
                                                     use.useSite);
    }
    projection.routes = spatialRouteTraversals(mapping);
    for (const auto &route : projection.routes)
      for (const auto &traversal : route)
        traversalsByTargetClass[targetClass].try_emplace(
            fabricRefKey(traversal), traversal);
    importedProjections.push_back(std::move(projection));
  }

  for (const auto &[coreOrdinal, targetClass] :
       llvm::enumerate(coreTargetClasses)) {
    const std::size_t namespaceOrdinal = coreOrdinal + 1;
    const auto patterns = patternsByTargetClass.find(targetClass);
    if (patterns != patternsByTargetClass.end())
      for (const auto &[key, pattern] : patterns->second) {
        (void)key;
        patternSources.push_back({namespaceOrdinal, pattern});
      }
    const auto traversals = traversalsByTargetClass.find(targetClass);
    if (traversals != traversalsByTargetClass.end())
      for (const auto &[key, traversal] : traversals->second) {
        (void)key;
        traversalSources.push_back({namespaceOrdinal, traversal});
      }
  }

  auto resources =
      freezeResourceCapacityIndex(namespaces, patternSources, traversalSources);
  if (!resources)
    return resources.takeError();
  auto result = std::make_unique<SystemCapacityModel>();
  result->resources_ = std::move(*resources);
  result->importedProjections_ = std::move(importedProjections);
  result->coreTargetClasses_.assign(coreTargetClasses.begin(),
                                    coreTargetClasses.end());
  result->mappingTargetClasses_.assign(mappingTargetClasses.begin(),
                                       mappingTargetClasses.end());
  result->rootTraversalOrdinals_.reserve(routingTopology.traversals().size());
  for (const auto &traversal : routingTopology.traversals()) {
    auto ordinal = result->resources_.traversalOrdinal(0, traversal.reference);
    if (!ordinal)
      return ordinal.takeError();
    result->rootTraversalOrdinals_.push_back(*ordinal);
  }

  result->graphKeys_.reserve(graphDecisions.size());
  for (const auto &decision : graphDecisions) {
    auto encoded = ::dataflow::encodeDataflowReference(dataflow.identity(),
                                                       decision.launch);
    if (!encoded)
      return encoded.takeError();
    result->graphKeys_.emplace_back(
        reinterpret_cast<const char *>(encoded->data()), encoded->size());
  }
  return result;
}

llvm::Expected<ResourceCapacityOveruseProjection> SystemCapacityModel::project(
    const FrozenSystemPnrProblem &problem,
    SystemCandidateCapacityProjectionView candidate) const {
  std::vector<FrozenResourceCapacityUseSelection> uses;
  uses.reserve(candidate.instructionResourceUses.size() +
               candidate.serviceResourceUses.size());
  for (const auto &use : candidate.instructionResourceUses) {
    auto pattern = resources_.patternOrdinal(0, use.pattern);
    if (!pattern)
      return pattern.takeError();
    auto activation = instructionActivationKey(problem.dataflowIdentity(), use);
    if (!activation)
      return activation.takeError();
    uses.push_back({*pattern, std::move(*activation)});
  }
  for (const auto &use : candidate.serviceResourceUses) {
    auto pattern = resources_.patternOrdinal(0, use.pattern);
    if (!pattern)
      return pattern.takeError();
    uses.push_back({*pattern, serviceActivationKey(use)});
  }

  std::vector<FrozenResourceCapacityRouteSelection> routes;
  routes.reserve(candidate.serviceRoutes.size());
  for (const SystemServiceRouteSelection &route : candidate.serviceRoutes) {
    if (route.nodeOffset > candidate.serviceRouteNodes.size() ||
        route.nodeCount > candidate.serviceRouteNodes.size() - route.nodeOffset)
      return invalid("System route capacity node range is out of bounds");
    FrozenResourceCapacityRouteSelection selected;
    for (const SystemServiceRouteNodeSelection &node :
         candidate.serviceRouteNodes.slice(route.nodeOffset, route.nodeCount)) {
      if (node.incomingTraversal == getInvalidPnrIndex())
        continue;
      if (node.incomingTraversal >= rootTraversalOrdinals_.size())
        return invalid("System route capacity traversal is out of bounds");
      selected.traversalOrdinals.push_back(
          rootTraversalOrdinals_[node.incomingTraversal]);
    }
    routes.push_back(std::move(selected));
  }

  if (candidate.graphChoices.size() != problem.graphDecisions().size() ||
      graphKeys_.size() != problem.graphDecisions().size())
    return invalid("System capacity graph choice closure is incomplete");
  std::set<std::tuple<std::string, PnrIndex, PnrIndex>> spatialContexts;
  std::set<std::pair<PnrIndex, PnrIndex>> routedMappings;
  for (PnrIndex graph = 0; graph < problem.graphDecisions().size(); ++graph) {
    const auto mappings = problem.graphChoiceCatalogOrdinals(graph);
    if (candidate.graphChoices[graph] >= mappings.size())
      return invalid("System capacity graph choice is out of range");
    const PnrIndex mapping = mappings[candidate.graphChoices[graph]];
    for (PnrIndex thread : problem.graphThreadOverlaps(graph)) {
      if (thread >= candidate.threadChoices.size())
        return invalid("System capacity graph overlap is out of range");
      const auto cores = problem.threadChoiceCatalogOrdinals(thread);
      if (candidate.threadChoices[thread] >= cores.size())
        return invalid("System capacity thread choice is out of range");
      const PnrIndex core = cores[candidate.threadChoices[thread]];
      spatialContexts.emplace(graphKeys_[graph], core, mapping);
      routedMappings.emplace(core, mapping);
    }
  }

  const auto projection =
      [&](PnrIndex core,
          PnrIndex mapping) -> llvm::Expected<const ImportedProjection *> {
    if (mapping >= importedProjections_.size() ||
        mapping >= mappingTargetClasses_.size() ||
        core >= coreTargetClasses_.size())
      return invalid("imported capacity projection is out of range");
    if (coreTargetClasses_[core] != mappingTargetClasses_[mapping])
      return invalid("selected System execution has no capacity projection");
    return &importedProjections_[mapping];
  };
  for (const auto &[graphKey, core, mapping] : spatialContexts) {
    auto imported = projection(core, mapping);
    if (!imported)
      return imported.takeError();
    for (const auto &use : (*imported)->uses) {
      const std::size_t namespaceOrdinal = static_cast<std::size_t>(core) + 1;
      auto pattern = resources_.patternOrdinal(namespaceOrdinal, use.pattern);
      if (!pattern)
        return pattern.takeError();
      std::string activation;
      appendSized(activation, graphKey);
      appendSized(activation, (*imported)->mappingIdentity.bytes());
      appendSized(activation, use.activationKey);
      uses.push_back({*pattern, std::move(activation)});
    }
  }
  for (const auto &[core, mapping] : routedMappings) {
    auto imported = projection(core, mapping);
    if (!imported)
      return imported.takeError();
    const std::size_t namespaceOrdinal = static_cast<std::size_t>(core) + 1;
    for (const auto &route : (*imported)->routes) {
      FrozenResourceCapacityRouteSelection selected;
      selected.traversalOrdinals.reserve(route.size());
      for (const auto &traversal : route) {
        auto ordinal = resources_.traversalOrdinal(namespaceOrdinal, traversal);
        if (!ordinal)
          return ordinal.takeError();
        selected.traversalOrdinals.push_back(*ordinal);
      }
      routes.push_back(std::move(selected));
    }
  }
  return deriveResourceCapacityOveruse(resources_, uses, routes);
}

} // namespace loom::pnr::detail
