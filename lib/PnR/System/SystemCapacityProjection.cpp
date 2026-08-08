#include "SystemCapacityProjection.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Mapping/Artifact/SpatialMappingCapacityVerification.h"

#include "llvm/ADT/STLExtras.h"

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

  std::vector<ResourceCapacityNamespaceView> namespaces;
  namespaces.reserve(cores.size() + 1);
  namespaces.push_back(
      {&fabric.artifact(), rootResourceCapacityQualifier(fabric.artifact())});
  std::vector<const ::loom::fabric::FabricArtifactView *> modules;
  modules.reserve(cores.size());
  for (const auto core : cores) {
    const auto target = fabric.spatialCoreTarget(core);
    if (!target ||
        target->dependencyOrdinal >= fabric.artifact().importedModules().size())
      return invalid("AccCore capacity namespace has no imported Module");
    const auto &module =
        fabric.artifact().importedModules()[target->dependencyOrdinal];
    modules.push_back(&module);
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

  for (const auto &[coreOrdinal, module] : llvm::enumerate(modules)) {
    const std::size_t namespaceOrdinal = coreOrdinal + 1;
    for (const auto &[mappingOrdinal, entry] :
         llvm::enumerate(spatialCatalog)) {
      if (coreTargetClasses[coreOrdinal] !=
          mappingTargetClasses[mappingOrdinal])
        continue;
      const auto &mapping = entry.mapping.view();
      if (mapping.fabricIdentity() != module->identity())
        return invalid(
            "SpatialMapping capacity namespace has the wrong Module");
      for (const auto &use : mapping.resourceUses())
        patternSources.push_back({namespaceOrdinal, use.useSite});
      for (const auto &route : spatialRouteTraversals(mapping))
        for (const auto &traversal : route)
          traversalSources.push_back({namespaceOrdinal, traversal});
    }
  }

  auto resources =
      freezeResourceCapacityIndex(namespaces, patternSources, traversalSources);
  if (!resources)
    return resources.takeError();
  auto result = std::make_unique<SystemCapacityModel>();
  result->resources_ = std::move(*resources);
  result->coreCount_ = cores.size();
  result->mappingCount_ = spatialCatalog.size();
  result->rootTraversalOrdinals_.reserve(routingTopology.traversals().size());
  for (const auto &traversal : routingTopology.traversals()) {
    auto ordinal = result->resources_.traversalOrdinal(0, traversal.reference);
    if (!ordinal)
      return ordinal.takeError();
    result->rootTraversalOrdinals_.push_back(*ordinal);
  }

  result->importedProjections_.resize(cores.size() * spatialCatalog.size());
  for (const auto &[coreOrdinal, module] : llvm::enumerate(modules)) {
    const std::size_t namespaceOrdinal = coreOrdinal + 1;
    for (const auto &[mappingOrdinal, entry] :
         llvm::enumerate(spatialCatalog)) {
      if (coreTargetClasses[coreOrdinal] !=
          mappingTargetClasses[mappingOrdinal])
        continue;
      const auto &mapping = entry.mapping.view();
      SystemCapacityModel::ImportedProjection projection{
          mapping.identity(), {}, {}};
      projection.uses.reserve(mapping.resourceUses().size());
      for (const auto &use : mapping.resourceUses()) {
        auto pattern =
            result->resources_.patternOrdinal(namespaceOrdinal, use.useSite);
        if (!pattern)
          return pattern.takeError();
        auto activation = deriveSpatialCapacityActivationKey(
            *module, dataflow.identity(), use);
        if (!activation)
          return activation.takeError();
        projection.uses.push_back({*pattern, std::move(*activation)});
      }
      for (const auto &route : spatialRouteTraversals(mapping)) {
        FrozenResourceCapacityRouteSelection selected;
        selected.traversalOrdinals.reserve(route.size());
        for (const auto &traversal : route) {
          auto ordinal =
              result->resources_.traversalOrdinal(namespaceOrdinal, traversal);
          if (!ordinal)
            return ordinal.takeError();
          selected.traversalOrdinals.push_back(*ordinal);
        }
        projection.routes.push_back(std::move(selected));
      }
      result->importedProjections_[coreOrdinal * spatialCatalog.size() +
                                   mappingOrdinal] = std::move(projection);
    }
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
  for (const auto &[ordinal, use] :
       llvm::enumerate(candidate.instructionResourceUses)) {
    auto pattern = resources_.patternOrdinal(0, use.pattern);
    if (!pattern)
      return pattern.takeError();
    std::string activation;
    appendU32(activation, 0);
    appendU64(activation, ordinal);
    uses.push_back({*pattern, std::move(activation)});
  }
  for (const auto &[ordinal, use] :
       llvm::enumerate(candidate.serviceResourceUses)) {
    auto pattern = resources_.patternOrdinal(0, use.pattern);
    if (!pattern)
      return pattern.takeError();
    std::string activation;
    appendU32(activation, 1);
    appendU64(activation, ordinal);
    uses.push_back({*pattern, std::move(activation)});
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
    if (mapping >= mappingCount_ || core >= coreCount_)
      return invalid("imported capacity projection is out of range");
    const std::size_t ordinal = core * mappingCount_ + mapping;
    if (ordinal >= importedProjections_.size() ||
        !importedProjections_[ordinal])
      return invalid("selected System execution has no capacity projection");
    return &*importedProjections_[ordinal];
  };
  for (const auto &[graphKey, core, mapping] : spatialContexts) {
    auto imported = projection(core, mapping);
    if (!imported)
      return imported.takeError();
    for (const auto &use : (*imported)->uses) {
      std::string activation;
      appendSized(activation, graphKey);
      appendSized(activation, (*imported)->mappingIdentity.bytes());
      appendSized(activation, use.activationKey);
      uses.push_back({use.patternOrdinal, std::move(activation)});
    }
  }
  for (const auto &[core, mapping] : routedMappings) {
    auto imported = projection(core, mapping);
    if (!imported)
      return imported.takeError();
    routes.insert(routes.end(), (*imported)->routes.begin(),
                  (*imported)->routes.end());
  }
  return deriveResourceCapacityOveruse(resources_, uses, routes);
}

} // namespace loom::pnr::detail
