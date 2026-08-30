#include "SystemCapacityProjection.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "Mapping/Artifact/SpatialMappingCapacityVerification.h"
#include "Mapping/Artifact/SystemMappingCapacityVerification.h"
#include "Mapping/Artifact/SystemMappingClosureProjection.h"

#include "llvm/ADT/STLExtras.h"

#include <array>
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

llvm::Expected<std::vector<SystemImportedRouteProjection>>
spatialRouteTraversals(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                       const ::loom::mapping::SpatialMappingView &mapping) {
  std::vector<SystemImportedRouteProjection> result;
  result.reserve(mapping.routeTrees().size() +
                 mapping.registerFifoTransfers().size());
  const auto append =
      [&](const auto &logicalNet,
          std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversals)
      -> llvm::Error {
    auto type = dataflow.tokenType(logicalNet);
    if (!type)
      return type.takeError();
    auto width = dataflow.transportPayloadBitWidth(*type);
    if (!width)
      return width.takeError();
    llvm::sort(traversals, [](const auto &left, const auto &right) {
      return ::loom::fabric::canonicalFabricBytes(left) <
             ::loom::fabric::canonicalFabricBytes(right);
    });
    traversals.erase(std::unique(traversals.begin(), traversals.end()),
                     traversals.end());
    result.push_back({std::move(traversals), *width});
    return llvm::Error::success();
  };
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
    if (llvm::Error error = append(route.logicalNet, std::move(traversals)))
      return std::move(error);
  }
  for (const auto &transfer : mapping.registerFifoTransfers())
    if (llvm::Error error =
            append(transfer.logicalNet,
                   {transfer.writeTraversal, transfer.readTraversal}))
      return std::move(error);
  return result;
}

template <typename Ref> std::string fabricRefKey(const Ref &reference) {
  const auto bytes = ::loom::fabric::canonicalFabricBytes(reference);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

llvm::Expected<std::string> instructionActivationKey(
    const ArtifactIdentity &dataflowIdentity,
    ::dataflow::RootThreadLaunchRef rootReference,
    const ::loom::fabric::InstructionCoreContextRef &contextReference) {
  auto root =
      ::dataflow::encodeDataflowReference(dataflowIdentity, rootReference);
  if (!root)
    return root.takeError();
  const auto context = ::loom::fabric::canonicalFabricBytes(contextReference);
  std::string result;
  appendU32(result, 0);
  appendSized(result, *root);
  appendSized(result, context);
  return result;
}

llvm::Expected<std::string>
instructionActivationKey(const ArtifactIdentity &dataflowIdentity,
                         const SystemInstructionResourceUseSelection &use) {
  return instructionActivationKey(dataflowIdentity, use.root, use.context);
}

std::string serviceActivationKey(const SystemServiceResourceUseSelection &use) {
  std::string result;
  appendU32(result, 1);
  appendU32(result, use.context);
  appendU32(result, use.subject);
  appendU32(result, use.branch);
  return result;
}

template <typename Ref>
llvm::Expected<std::string> dataflowRefKey(const ArtifactIdentity &identity,
                                           const Ref &reference) {
  auto encoded = ::dataflow::encodeDataflowReference(identity, reference);
  if (!encoded)
    return encoded.takeError();
  return std::string(reinterpret_cast<const char *>(encoded->data()),
                     encoded->size());
}

struct InstructionProgressEvents final {
  ::dataflow::EventFamilyKey trigger;
  ::dataflow::EventFamilyKey release;
};

InstructionProgressEvents
instructionProgressEvents(::dataflow::RootThreadLaunchRef root) {
  const ::dataflow::RootThreadBoundaryTransferRef startTransfer(
      ::dataflow::RootThreadStartTransferRef{root});
  const ::dataflow::RootThreadBoundaryTransferRef completionTransfer(
      ::dataflow::RootThreadCompletionTransferRef{root});
  return {::dataflow::EventFamilyKey(::dataflow::StaticTransferEventRef(
              ::dataflow::ConsumedTransferEventRef{
                  ::dataflow::CanonicalSinkTerminalRef(
                      ::dataflow::RootThreadBoundarySinkRef{startTransfer})})),
          ::dataflow::EventFamilyKey(::dataflow::StaticTransferEventRef(
              ::dataflow::ProducedTransferEventRef{
                  ::dataflow::CanonicalProducerTerminalRef(
                      ::dataflow::RootThreadBoundarySourceRef{
                          completionTransfer})}))};
}

llvm::Expected<::dataflow::EventFamilyKey>
serviceProgressTrigger(const SystemServiceTargetSubject &subject) {
  const auto *member = std::get_if<SystemServiceMemberTargetSubject>(&subject);
  if (!member)
    return invalid("memory exposure cannot own a progress ResourceUse");
  const ::dataflow::ContextualActorRef *actor = nullptr;
  if (const auto *addressed =
          std::get_if<::dataflow::AddressedMemoryActorMemberRef>(
              &member->member))
    actor = &addressed->actor;
  else if (const auto *fence =
               std::get_if<::dataflow::FenceActorMemberRef>(&member->member))
    actor = &fence->actor;
  if (!actor)
    return invalid("service ResourceUse member has no contextual actor");
  return ::dataflow::EventFamilyKey(
      ::dataflow::ContextualActorTransitionEventRef{*actor, 0});
}

void canonicalizeCells(
    std::vector<::loom::mapping::SystemPresburgerCell> &cells) {
  llvm::sort(cells, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.dimensionCount, lhs.symbolCount, lhs.localCount,
                    lhs.equalities, lhs.inequalities) <
           std::tie(rhs.dimensionCount, rhs.symbolCount, rhs.localCount,
                    rhs.equalities, rhs.inequalities);
  });
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
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
    llvm::ArrayRef<FrozenSystemServiceContext> serviceContexts,
    llvm::ArrayRef<FrozenSystemServiceLeg> serviceLegs,
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
  std::vector<::dataflow::EventFamilyKey> progressEvents;
  std::vector<SystemCapacityModel::ImportedProjection> importedProjections;
  importedProjections.reserve(spatialCatalog.size());
  for (const auto &[mappingOrdinal, entry] : llvm::enumerate(spatialCatalog)) {
    const PnrIndex targetClass = mappingTargetClasses[mappingOrdinal];
    const auto module = modulesByTargetClass.find(targetClass);
    if (module == modulesByTargetClass.end())
      return invalid("SpatialMapping target class has no System occurrence");
    const auto &mapping = entry.mapping->view();
    if (mapping.fabricIdentity() != module->second->identity())
      return invalid("SpatialMapping target class has the wrong Module");

    SystemCapacityModel::ImportedProjection projection{
        mapping.identity(), {}, {}};
    std::map<std::string,
             std::vector<SystemCapacityModel::ImportedProgressUseProjection>>
        progressByGraph;
    std::map<
        std::string,
        std::vector<::loom::mapping::MappingRouteProgressObligationProjection>>
        routeProgressByGraph;
    for (const auto &use : mapping.resourceUses()) {
      auto activation = deriveSpatialCapacityActivationKey(
          *module->second, dataflow.identity(), use);
      if (!activation)
        return activation.takeError();
      patternsByTargetClass[targetClass].try_emplace(fabricRefKey(use.useSite),
                                                     use.useSite);
      auto ownerGraph = ::loom::mapping::resolveSystemSpatialActivityEventGraph(
          dataflow, use.activation.trigger.event);
      if (!ownerGraph)
        return ownerGraph.takeError();
      std::set<std::string> projectedLaunches;
      for (const FrozenSystemGraphExecutionDecision &decision :
           graphDecisions) {
        auto launchedGraph = dataflow.resolve(decision.launch);
        if (!launchedGraph)
          return launchedGraph.takeError();
        if (*launchedGraph != *ownerGraph)
          continue;
        auto graphKey = dataflowRefKey(dataflow.identity(), decision.launch);
        if (!graphKey)
          return graphKey.takeError();
        if (!projectedLaunches.insert(*graphKey).second)
          continue;
        auto triggers = ::loom::mapping::projectSystemSpatialActivityEvent(
            dataflow, decision.launch, use.activation.trigger.event);
        if (!triggers)
          return triggers.takeError();
        auto releases = ::loom::mapping::projectSystemSpatialCausalRelease(
            dataflow, decision.launch, use.activation.release);
        if (!releases)
          return releases.takeError();
        SystemCapacityModel::ImportedProgressUseProjection projected{
            use.useSite, *activation, std::move(*triggers), {}};
        projected.causalRelease.reserve(releases->size());
        for (const auto &release : *releases)
          projected.causalRelease.push_back({release.alternatives});
        progressEvents.insert(progressEvents.end(),
                              projected.triggerAlternatives.begin(),
                              projected.triggerAlternatives.end());
        for (const auto &release : projected.causalRelease)
          progressEvents.insert(progressEvents.end(),
                                release.alternatives.begin(),
                                release.alternatives.end());
        progressByGraph[*graphKey].push_back(std::move(projected));
      }
    }
    for (const SpatialCatalogGraphProgress &graphProgress : entry.graphProgress)
      for (const FrozenSystemGraphExecutionDecision &decision :
           graphDecisions) {
        auto launchedGraph = dataflow.resolve(decision.launch);
        if (!launchedGraph)
          return launchedGraph.takeError();
        if (*launchedGraph != graphProgress.graph)
          continue;
        auto graphKey = dataflowRefKey(dataflow.identity(), decision.launch);
        if (!graphKey)
          return graphKey.takeError();
        progressByGraph.try_emplace(*graphKey);
        auto [position, inserted] = routeProgressByGraph.try_emplace(
            *graphKey, graphProgress.routeObligations);
        if (!inserted &&
            position->second.size() != graphProgress.routeObligations.size())
          return invalid("one rooted graph has inconsistent route progress");
      }
    projection.graphProgress.reserve(progressByGraph.size());
    for (auto &[graphKey, uses] : progressByGraph) {
      auto routesForGraph = routeProgressByGraph.find(graphKey);
      std::vector<::loom::mapping::MappingRouteProgressObligationProjection>
          routeObligations;
      if (routesForGraph != routeProgressByGraph.end())
        routeObligations = std::move(routesForGraph->second);
      projection.graphProgress.push_back(
          {std::move(graphKey), std::move(routeObligations), std::move(uses)});
    }
    auto routes = spatialRouteTraversals(dataflow, mapping);
    if (!routes)
      return routes.takeError();
    projection.routes = std::move(*routes);
    for (const auto &route : projection.routes)
      for (const auto &traversal : route.traversals)
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

  result->patternPhysicalOwnerKeys_.reserve(
      result->resources_.patterns().size());
  for (const FrozenResourceCapacityPattern &pattern :
       result->resources_.patterns()) {
    std::optional<::loom::fabric::SpatialCoreOccurrenceRef> spatialCore;
    if (pattern.namespaceOrdinal != 0) {
      const std::size_t core = pattern.namespaceOrdinal - 1;
      if (core >= cores.size())
        return invalid("UsePattern capacity namespace has no AccCore");
      spatialCore = ::loom::fabric::SpatialCoreOccurrenceRef{cores[core]};
    }
    auto physical = ::loom::mapping::detail::qualifySystemResourceOwner(
        pattern.reference.owner.catalog(), spatialCore);
    if (!physical)
      return physical.takeError();
    const auto bytes = ::loom::fabric::canonicalFabricBytes(*physical);
    result->patternPhysicalOwnerKeys_.emplace_back(
        reinterpret_cast<const char *>(bytes.data()), bytes.size());
  }
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
  std::vector<::loom::mapping::CanonicalServiceLegKey> progressLegs;
  progressLegs.reserve(serviceLegs.size());
  for (const FrozenSystemServiceLeg &leg : serviceLegs)
    progressLegs.push_back(leg.key);
  auto serviceRouteProgress =
      ::loom::mapping::deriveSystemTransferRouteProgressDependencies(
          dataflow, progressLegs);
  if (!serviceRouteProgress)
    return serviceRouteProgress.takeError();
  result->serviceRouteProgressDependencies_ = std::move(*serviceRouteProgress);
  for (const auto &root : dataflow.rootThreadLaunches()) {
    const InstructionProgressEvents events =
        instructionProgressEvents(root.ref);
    progressEvents.push_back(events.trigger);
    progressEvents.push_back(events.release);
  }
  for (const FrozenSystemServiceContext &context : serviceContexts)
    for (const SystemServiceTargetSubject &subject : context.subjects) {
      const auto *member =
          std::get_if<SystemServiceMemberTargetSubject>(&subject);
      if (!member ||
          std::holds_alternative<::dataflow::MessageTransferMemberRef>(
              member->member))
        continue;
      auto trigger = serviceProgressTrigger(subject);
      if (!trigger)
        return trigger.takeError();
      progressEvents.push_back(std::move(*trigger));
    }
  auto progressModel =
      ::loom::mapping::freezeMappingProgressModel(dataflow, progressEvents);
  if (!progressModel)
    return progressModel.takeError();
  result->progressModel_ = std::move(*progressModel);
  return result;
}

namespace {

struct SystemServiceRouteDemandProjection final {
  std::vector<FrozenResourceCapacityRouteSelection> routes;
  std::vector<::loom::mapping::MappingRouteProgressObligationProjection>
      progressObligations;
};

struct SystemMutableResourceDemandProjection final {
  std::vector<FrozenResourceCapacityUseSelection> uses;
  std::vector<::loom::mapping::MappingProgressActivationProjection>
      progressActivations;
};

struct SystemNonRouteDemandProjection final {
  std::vector<PnrIndex> threadChoices;
  std::vector<PnrIndex> graphChoices;
  std::vector<SystemInstructionResourceUseSelection> instructionResourceUses;
  std::vector<SystemServiceResourceUseSelection> serviceResourceUses;
  std::vector<FrozenResourceCapacityUseSelection> uses;
  std::size_t mutableUseCount = 0;
  std::vector<FrozenResourceCapacityRouteSelection> importedRoutes;
  ::loom::mapping::MappingDataflowProgressBasis basis;
  std::vector<::loom::mapping::MappingRouteProgressObligationProjection>
      progressObligations;
  std::vector<::loom::mapping::MappingProgressActivationProjection>
      progressActivations;
  std::size_t mutableProgressActivationCount = 0;
};

bool sameInstructionUse(const SystemInstructionResourceUseSelection &lhs,
                        const SystemInstructionResourceUseSelection &rhs) {
  return lhs.root == rhs.root && lhs.context == rhs.context &&
         lhs.pattern == rhs.pattern;
}

bool sameServiceUse(const SystemServiceResourceUseSelection &lhs,
                    const SystemServiceResourceUseSelection &rhs) {
  return lhs.context == rhs.context && lhs.subject == rhs.subject &&
         lhs.branch == rhs.branch && lhs.pattern == rhs.pattern;
}

bool hasSameNonRouteSelections(
    const SystemNonRouteDemandProjection &projection,
    SystemCandidateCapacityProjectionView candidate) {
  return llvm::equal(projection.threadChoices, candidate.threadChoices) &&
         llvm::equal(projection.graphChoices, candidate.graphChoices) &&
         llvm::equal(projection.instructionResourceUses,
                     candidate.instructionResourceUses, sameInstructionUse) &&
         llvm::equal(projection.serviceResourceUses,
                     candidate.serviceResourceUses, sameServiceUse);
}

bool hasSameExecutionSelections(
    const SystemNonRouteDemandProjection &projection,
    SystemCandidateCapacityProjectionView candidate) {
  return llvm::equal(projection.threadChoices, candidate.threadChoices) &&
         llvm::equal(projection.graphChoices, candidate.graphChoices);
}

llvm::Expected<SystemServiceRouteDemandProjection> projectServiceRouteDemand(
    const FrozenSystemPnrProblem &problem,
    SystemCandidateCapacityProjectionView candidate,
    llvm::ArrayRef<std::size_t> rootTraversalOrdinals,
    llvm::ArrayRef<::loom::mapping::SystemTransferRouteProgressDependency>
        progressDependencies) {
  SystemServiceRouteDemandProjection result;
  result.routes.reserve(candidate.serviceRoutes.size());
  std::vector<::loom::mapping::SystemTransferLegView> transferLegs;
  transferLegs.reserve(candidate.serviceRoutes.size());
  for (const SystemServiceRouteSelection &route : candidate.serviceRoutes) {
    if (route.nodeOffset > candidate.serviceRouteNodes.size() ||
        route.nodeCount >
            candidate.serviceRouteNodes.size() - route.nodeOffset ||
        route.sinkOffset > candidate.serviceRouteSinks.size() ||
        route.sinkCount > candidate.serviceRouteSinks.size() - route.sinkOffset)
      return invalid("System route capacity node range is out of bounds");
    if (route.leg >= problem.serviceLegs().size() || route.nodeCount == 0 ||
        route.rootEndpoint >= problem.routingTopology().endpoints().size())
      return invalid("System progress route selection is out of range");

    FrozenResourceCapacityRouteSelection capacityRoute;
    capacityRoute.payloadWidthBits =
        problem.serviceLegs()[route.leg].requiredPayloadWidthBits;
    const auto nodes =
        candidate.serviceRouteNodes.slice(route.nodeOffset, route.nodeCount);
    for (const SystemServiceRouteNodeSelection &node : nodes) {
      if (node.incomingTraversal == getInvalidPnrIndex())
        continue;
      if (node.incomingTraversal >= rootTraversalOrdinals.size())
        return invalid("System route capacity traversal is out of bounds");
      capacityRoute.traversalOrdinals.push_back(
          rootTraversalOrdinals[node.incomingTraversal]);
    }
    result.routes.push_back(std::move(capacityRoute));

    const FrozenSystemServiceLeg &leg = problem.serviceLegs()[route.leg];
    ::loom::mapping::SystemTransferLegView progressRoute{
        leg.key,
        problem.routingTopology().endpoints()[route.rootEndpoint].reference,
        {},
        {}};
    progressRoute.nodes.reserve(nodes.size() - 1);
    for (PnrIndex nodeOrdinal = 1; nodeOrdinal < nodes.size(); ++nodeOrdinal) {
      const SystemServiceRouteNodeSelection &node = nodes[nodeOrdinal];
      if (node.incomingTraversal >=
          problem.routingTopology().traversals().size())
        return invalid("System progress route traversal is out of range");
      progressRoute.nodes.push_back({nodeOrdinal, node.parentNode,
                                     problem.routingTopology()
                                         .traversals()[node.incomingTraversal]
                                         .reference});
    }
    for (const SystemServiceRouteSinkSelection &sink :
         candidate.serviceRouteSinks.slice(route.sinkOffset, route.sinkCount)) {
      if (sink.terminal >= problem.serviceTerminals().size() ||
          sink.node >= nodes.size())
        return invalid("System progress route sink is out of range");
      progressRoute.sinks.push_back(
          {problem.serviceTerminals()[sink.terminal].key, sink.node});
    }
    transferLegs.push_back(std::move(progressRoute));
  }
  auto progress = ::loom::mapping::projectSystemTransferRouteProgress(
      transferLegs, progressDependencies);
  if (!progress)
    return progress.takeError();
  result.progressObligations = std::move(*progress);
  return result;
}

llvm::Error appendResourceProgressActivation(
    const FrozenResourceCapacityIndex &resources,
    llvm::ArrayRef<std::string> patternPhysicalOwnerKeys,
    std::size_t patternOrdinal, ::loom::mapping::ExecutionContextKey context,
    ::dataflow::RootThreadLaunchRef relationRoot,
    std::vector<::loom::mapping::SystemPresburgerCell> relationDomain,
    std::vector<::dataflow::EventFamilyKey> triggerAlternatives,
    std::vector<::loom::mapping::MappingProgressCausalReleaseProjection>
        causalRelease,
    std::vector<::loom::mapping::MappingProgressActivationProjection>
        &activations) {
  if (patternOrdinal >= resources.patterns().size() ||
      patternOrdinal >= patternPhysicalOwnerKeys.size())
    return invalid("progress activation names a foreign UsePattern");
  canonicalizeCells(relationDomain);
  const FrozenResourceCapacityPattern &pattern =
      resources.patterns()[patternOrdinal];
  ::loom::mapping::MappingResourceProgressUse arbitration = pattern.progressUse;
  arbitration.physicalOwnerKey = patternPhysicalOwnerKeys[patternOrdinal];
  ::loom::mapping::MappingProgressActivationProjection activation{
      std::move(context),
      relationRoot,
      std::move(relationDomain),
      std::move(triggerAlternatives),
      {},
      std::move(causalRelease),
      std::move(arbitration)};
  activation.capacityClaims.reserve(pattern.claims.size());
  for (const FrozenResourceCapacityClaim &claim : pattern.claims)
    activation.capacityClaims.push_back({claim.cell, claim.amount});
  activations.push_back(std::move(activation));
  return llvm::Error::success();
}

llvm::Expected<SystemMutableResourceDemandProjection>
projectMutableResourceDemand(
    const FrozenResourceCapacityIndex &resources,
    llvm::ArrayRef<std::string> patternPhysicalOwnerKeys,
    const FrozenSystemPnrProblem &problem,
    SystemCandidateCapacityProjectionView candidate) {
  SystemMutableResourceDemandProjection result;
  result.uses.reserve(candidate.instructionResourceUses.size() +
                      candidate.serviceResourceUses.size());
  result.progressActivations.reserve(candidate.instructionResourceUses.size() +
                                     candidate.serviceResourceUses.size());

  struct InstructionContextProjection final {
    ::dataflow::RootThreadLaunchRef root;
    ::loom::fabric::InstructionCoreContextRef context;
    std::vector<::loom::mapping::SystemPresburgerCell> cells;
  };
  std::map<std::string, InstructionContextProjection> instructionContexts;
  for (PnrIndex decision = 0; decision < problem.threadDecisions().size();
       ++decision) {
    const auto choices = problem.threadChoiceCatalogOrdinals(decision);
    if (candidate.threadChoices[decision] >= choices.size())
      return invalid("System thread choice is out of range");
    const PnrIndex core = choices[candidate.threadChoices[decision]];
    if (core >= problem.accCores().size())
      return invalid("System thread choice names a foreign AccCore");
    const auto &thread = problem.threadDecisions()[decision];
    const ::loom::fabric::InstructionCoreContextRef context{
        problem.accCores()[core]};
    auto key = instructionActivationKey(problem.dataflowIdentity(), thread.root,
                                        context);
    if (!key)
      return key.takeError();
    auto [position, inserted] = instructionContexts.try_emplace(
        *key, InstructionContextProjection{thread.root, context, {}});
    position->second.cells.push_back(thread.cell);
  }
  for (const auto &use : candidate.instructionResourceUses) {
    auto key = instructionActivationKey(problem.dataflowIdentity(), use);
    if (!key)
      return key.takeError();
    const auto domain = instructionContexts.find(*key);
    if (domain == instructionContexts.end())
      return invalid("Instruction ResourceUse has no selected context");
    auto pattern = resources.patternOrdinal(0, use.pattern);
    if (!pattern)
      return pattern.takeError();
    result.uses.push_back({*pattern, *key});
    const InstructionProgressEvents events =
        instructionProgressEvents(use.root);
    if (llvm::Error error = appendResourceProgressActivation(
            resources, patternPhysicalOwnerKeys, *pattern,
            ::loom::mapping::InstructionExecutionContextKey{use.context.core},
            use.root, domain->second.cells, {events.trigger},
            {{{events.release}}}, result.progressActivations))
      return std::move(error);
  }

  for (const SystemServiceResourceUseSelection &use :
       candidate.serviceResourceUses) {
    if (use.context >= problem.serviceContexts().size())
      return invalid("service ResourceUse context is out of range");
    const FrozenSystemServiceContext &serviceContext =
        problem.serviceContexts()[use.context];
    if (use.subject >= serviceContext.subjects.size() ||
        serviceContext.threadDecision >= problem.threadDecisions().size())
      return invalid("service ResourceUse selection is out of range");
    const auto threadCores =
        problem.threadChoiceCatalogOrdinals(serviceContext.threadDecision);
    if (candidate.threadChoices[serviceContext.threadDecision] >=
        threadCores.size())
      return invalid("service ResourceUse thread choice is out of range");
    const PnrIndex core =
        threadCores[candidate.threadChoices[serviceContext.threadDecision]];
    if (core >= problem.accCores().size())
      return invalid("service ResourceUse names a foreign AccCore");
    ::loom::mapping::ExecutionContextKey context;
    if (serviceContext.graphDecision != getInvalidPnrIndex()) {
      if (serviceContext.graphDecision >= problem.graphDecisions().size())
        return invalid("service ResourceUse graph choice is out of range");
      const auto mappings =
          problem.graphChoiceCatalogOrdinals(serviceContext.graphDecision);
      if (candidate.graphChoices[serviceContext.graphDecision] >=
          mappings.size())
        return invalid("service ResourceUse Mapping choice is out of range");
      const PnrIndex mapping =
          mappings[candidate.graphChoices[serviceContext.graphDecision]];
      if (mapping >= problem.spatialMappings().size())
        return invalid("service ResourceUse names a foreign SpatialMapping");
      context = ::loom::mapping::SpatialExecutionContextKey{
          problem.accCores()[core],
          problem.spatialMappings()[mapping].artifact};
    } else {
      context = ::loom::mapping::InstructionExecutionContextKey{
          problem.accCores()[core]};
    }
    auto trigger = serviceProgressTrigger(serviceContext.subjects[use.subject]);
    if (!trigger)
      return trigger.takeError();
    auto pattern = resources.patternOrdinal(0, use.pattern);
    if (!pattern)
      return pattern.takeError();
    result.uses.push_back({*pattern, serviceActivationKey(use)});
    if (llvm::Error error = appendResourceProgressActivation(
            resources, patternPhysicalOwnerKeys, *pattern, std::move(context),
            problem.threadDecisions()[serviceContext.threadDecision].root,
            serviceContext.cells, {std::move(*trigger)}, {},
            result.progressActivations))
      return std::move(error);
  }
  return result;
}

} // namespace

class SystemCandidateProjectionCache final {
public:
  std::shared_ptr<const SystemNonRouteDemandProjection> nonRoute;
  std::shared_ptr<const SystemServiceRouteDemandProjection> serviceRoutes;
};

llvm::Expected<ResourceCapacityOveruseProjection>
SystemCapacityModel::projectImportedCapacity(
    const FrozenSystemPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices) const {
  if (threadChoices.size() != problem.threadDecisions().size() ||
      graphChoices.size() != problem.graphDecisions().size() ||
      graphKeys_.size() != problem.graphDecisions().size())
    return invalid("System imported-capacity choice closure is incomplete");

  struct SpatialContextProjection final {
    std::string graphKey;
    PnrIndex core = getInvalidPnrIndex();
    PnrIndex mapping = getInvalidPnrIndex();
  };
  std::map<std::tuple<std::string, PnrIndex, PnrIndex>,
           SpatialContextProjection>
      spatialContexts;
  std::set<std::pair<PnrIndex, PnrIndex>> routedMappings;
  for (PnrIndex graph = 0; graph < problem.graphDecisions().size(); ++graph) {
    const auto mappings = problem.graphChoiceCatalogOrdinals(graph);
    if (graphChoices[graph] >= mappings.size())
      return invalid("System imported-capacity graph choice is out of range");
    const PnrIndex mapping = mappings[graphChoices[graph]];
    for (PnrIndex thread : problem.graphThreadOverlaps(graph)) {
      if (thread >= threadChoices.size())
        return invalid(
            "System imported-capacity graph overlap is out of range");
      const auto cores = problem.threadChoiceCatalogOrdinals(thread);
      if (threadChoices[thread] >= cores.size())
        return invalid(
            "System imported-capacity thread choice is out of range");
      const PnrIndex core = cores[threadChoices[thread]];
      spatialContexts.try_emplace(
          std::make_tuple(graphKeys_[graph], core, mapping),
          SpatialContextProjection{graphKeys_[graph], core, mapping});
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

  std::vector<FrozenResourceCapacityUseSelection> uses;
  for (const auto &[key, context] : spatialContexts) {
    (void)key;
    auto imported = projection(context.core, context.mapping);
    if (!imported)
      return imported.takeError();
    const std::string &selectedGraphKey = context.graphKey;
    const auto graphProgress = llvm::find_if(
        (*imported)->graphProgress, [&](const auto &candidateProgress) {
          return candidateProgress.graphKey == selectedGraphKey;
        });
    if (graphProgress == (*imported)->graphProgress.end())
      return invalid(
          "selected SpatialMapping has no rooted capacity projection");
    const std::size_t namespaceOrdinal =
        static_cast<std::size_t>(context.core) + 1;
    for (const auto &use : graphProgress->uses) {
      auto pattern = resources_.patternOrdinal(namespaceOrdinal, use.pattern);
      if (!pattern)
        return pattern.takeError();
      std::string activation;
      appendSized(activation, context.graphKey);
      appendSized(activation, (*imported)->mappingIdentity.bytes());
      appendSized(activation, use.activationKey);
      uses.push_back({*pattern, std::move(activation)});
    }
  }

  std::vector<FrozenResourceCapacityRouteSelection> routes;
  for (const auto &[core, mapping] : routedMappings) {
    auto imported = projection(core, mapping);
    if (!imported)
      return imported.takeError();
    const std::size_t namespaceOrdinal = static_cast<std::size_t>(core) + 1;
    for (const auto &route : (*imported)->routes) {
      FrozenResourceCapacityRouteSelection selected;
      selected.payloadWidthBits = route.payloadWidthBits;
      selected.traversalOrdinals.reserve(route.traversals.size());
      for (const auto &traversal : route.traversals) {
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

namespace {

llvm::Expected<SystemCandidatePhysicalDemandProjection>
finalizeSystemCandidateProjection(
    const FrozenResourceCapacityIndex &resources,
    const ::loom::mapping::FrozenMappingProgressModel &progressModel,
    const SystemCandidateProjectionCache &cache) {
  if (!cache.nonRoute || !cache.serviceRoutes)
    return invalid("System candidate demand projection is incomplete");
  const std::array<llvm::ArrayRef<FrozenResourceCapacityRouteSelection>, 2>
      routeSegments{cache.serviceRoutes->routes,
                    cache.nonRoute->importedRoutes};
  auto demand = deriveResourcePhysicalDemand(resources, cache.nonRoute->uses,
                                             routeSegments);
  if (!demand)
    return demand.takeError();
  if (demand->baselineOccupancy.size() != resources.cells().size())
    return invalid("System progress capacity baseline has the wrong width");

  std::vector<::loom::mapping::MappingRouteProgressObligationProjection>
      routeObligations = cache.serviceRoutes->progressObligations;
  routeObligations.insert(routeObligations.end(),
                          cache.nonRoute->progressObligations.begin(),
                          cache.nonRoute->progressObligations.end());
  std::vector<::loom::mapping::MappingProgressCapacityCellProjection>
      capacityCells;
  capacityCells.reserve(resources.cells().size());
  for (const auto &[ordinal, cell] : llvm::enumerate(resources.cells()))
    capacityCells.push_back(
        {cell.capacity, demand->baselineOccupancy[ordinal]});
  // This System PnR demand path projects service and imported-route
  // obligations only; the buffer-dependency edge set stays engaged-empty,
  // which the progress kernel treats as a no-op.
  const std::optional<
      std::vector<::loom::mapping::MappingBufferDependencyEdge>>
      bufferDependencyEdges =
          std::vector<::loom::mapping::MappingBufferDependencyEdge>{};
  auto progress = ::loom::mapping::deriveMappingProgressClosure(
      progressModel, ::loom::mapping::MappingProgressProjectionView{
                         cache.nonRoute->basis, routeObligations, capacityCells,
                         cache.nonRoute->progressActivations,
                         bufferDependencyEdges});
  if (!progress)
    return progress.takeError();
  return SystemCandidatePhysicalDemandProjection{std::move(demand->capacity),
                                                 *progress, demand->timing};
}

} // namespace

llvm::Expected<SystemCandidateProjectionResult>
SystemCapacityModel::projectWithCache(
    const FrozenSystemPnrProblem &problem,
    SystemCandidateCapacityProjectionView candidate) const {
  if (candidate.threadChoices.size() != problem.threadDecisions().size() ||
      candidate.graphChoices.size() != problem.graphDecisions().size() ||
      graphKeys_.size() != problem.graphDecisions().size())
    return invalid("System physical-demand choice closure is incomplete");
  if (!progressModel_)
    return invalid("System progress model is absent");
  auto serviceRoutes =
      projectServiceRouteDemand(problem, candidate, rootTraversalOrdinals_,
                                serviceRouteProgressDependencies_);
  if (!serviceRoutes)
    return serviceRoutes.takeError();

  auto nonRoute = std::make_shared<SystemNonRouteDemandProjection>();
  nonRoute->threadChoices.assign(candidate.threadChoices.begin(),
                                 candidate.threadChoices.end());
  nonRoute->graphChoices.assign(candidate.graphChoices.begin(),
                                candidate.graphChoices.end());
  nonRoute->instructionResourceUses.assign(
      candidate.instructionResourceUses.begin(),
      candidate.instructionResourceUses.end());
  nonRoute->serviceResourceUses.assign(candidate.serviceResourceUses.begin(),
                                       candidate.serviceResourceUses.end());
  nonRoute->basis = problem.progressBasis();
  auto mutableDemand = projectMutableResourceDemand(
      resources_, patternPhysicalOwnerKeys_, problem, candidate);
  if (!mutableDemand)
    return mutableDemand.takeError();
  nonRoute->uses = std::move(mutableDemand->uses);
  nonRoute->mutableUseCount = nonRoute->uses.size();
  nonRoute->progressActivations = std::move(mutableDemand->progressActivations);
  nonRoute->mutableProgressActivationCount =
      nonRoute->progressActivations.size();

  struct SpatialContextProjection final {
    std::string graphKey;
    ::dataflow::RootThreadLaunchRef relationRoot;
    PnrIndex core = getInvalidPnrIndex();
    PnrIndex mapping = getInvalidPnrIndex();
    std::vector<::loom::mapping::SystemPresburgerCell> cells;
  };
  std::map<std::tuple<std::string, PnrIndex, PnrIndex>,
           SpatialContextProjection>
      spatialContexts;
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
      auto [position, inserted] = spatialContexts.try_emplace(
          std::make_tuple(graphKeys_[graph], core, mapping),
          SpatialContextProjection{
              graphKeys_[graph],
              problem.graphDecisions()[graph].launch.rootThreadLaunch,
              core,
              mapping,
              {}});
      if (!inserted &&
          position->second.relationRoot !=
              problem.graphDecisions()[graph].launch.rootThreadLaunch)
        return invalid("one Spatial context crosses relation-root spaces");
      position->second.cells.push_back(problem.graphDecisions()[graph].cell);
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
  for (auto &[key, context] : spatialContexts) {
    (void)key;
    canonicalizeCells(context.cells);
    auto imported = projection(context.core, context.mapping);
    if (!imported)
      return imported.takeError();
    const std::string &selectedGraphKey = context.graphKey;
    const auto graphProgress = llvm::find_if(
        (*imported)->graphProgress, [&](const auto &candidateProgress) {
          return candidateProgress.graphKey == selectedGraphKey;
        });
    if (graphProgress == (*imported)->graphProgress.end())
      return invalid("selected SpatialMapping has no rooted progress "
                     "projection");
    nonRoute->progressObligations.insert(
        nonRoute->progressObligations.end(),
        graphProgress->routeObligations.begin(),
        graphProgress->routeObligations.end());
    for (const auto &use : graphProgress->uses) {
      const std::size_t namespaceOrdinal =
          static_cast<std::size_t>(context.core) + 1;
      auto pattern = resources_.patternOrdinal(namespaceOrdinal, use.pattern);
      if (!pattern)
        return pattern.takeError();
      std::string activation;
      appendSized(activation, context.graphKey);
      appendSized(activation, (*imported)->mappingIdentity.bytes());
      appendSized(activation, use.activationKey);
      nonRoute->uses.push_back({*pattern, std::move(activation)});
      if (context.mapping >= problem.spatialMappings().size() ||
          context.core >= problem.accCores().size())
        return invalid("selected Spatial context is out of range");
      if (llvm::Error error = appendResourceProgressActivation(
              resources_, patternPhysicalOwnerKeys_, *pattern,
              ::loom::mapping::SpatialExecutionContextKey{
                  problem.accCores()[context.core],
                  problem.spatialMappings()[context.mapping].artifact},
              context.relationRoot, context.cells, use.triggerAlternatives,
              use.causalRelease, nonRoute->progressActivations))
        return std::move(error);
    }
  }
  for (const auto &[core, mapping] : routedMappings) {
    auto imported = projection(core, mapping);
    if (!imported)
      return imported.takeError();
    const std::size_t namespaceOrdinal = static_cast<std::size_t>(core) + 1;
    for (const auto &route : (*imported)->routes) {
      FrozenResourceCapacityRouteSelection selected;
      selected.payloadWidthBits = route.payloadWidthBits;
      selected.traversalOrdinals.reserve(route.traversals.size());
      for (const auto &traversal : route.traversals) {
        auto ordinal = resources_.traversalOrdinal(namespaceOrdinal, traversal);
        if (!ordinal)
          return ordinal.takeError();
        selected.traversalOrdinals.push_back(*ordinal);
      }
      nonRoute->importedRoutes.push_back(std::move(selected));
    }
  }
  auto cache = std::make_shared<SystemCandidateProjectionCache>();
  cache->nonRoute = std::move(nonRoute);
  cache->serviceRoutes = std::make_shared<SystemServiceRouteDemandProjection>(
      std::move(*serviceRoutes));
  auto demand =
      finalizeSystemCandidateProjection(resources_, *progressModel_, *cache);
  if (!demand)
    return demand.takeError();
  return SystemCandidateProjectionResult{std::move(*demand), std::move(cache)};
}

llvm::Expected<SystemCandidateProjectionResult>
SystemCapacityModel::projectRouteDelta(
    const FrozenSystemPnrProblem &problem,
    SystemCandidateCapacityProjectionView candidate,
    const SystemCandidateProjectionCache &previous) const {
  if (!progressModel_)
    return invalid("System progress model is absent");
  if (!previous.nonRoute ||
      !hasSameNonRouteSelections(*previous.nonRoute, candidate))
    return invalid("System route delta changed a non-route selection");
  auto serviceRoutes =
      projectServiceRouteDemand(problem, candidate, rootTraversalOrdinals_,
                                serviceRouteProgressDependencies_);
  if (!serviceRoutes)
    return serviceRoutes.takeError();
  auto cache = std::make_shared<SystemCandidateProjectionCache>();
  cache->nonRoute = previous.nonRoute;
  cache->serviceRoutes = std::make_shared<SystemServiceRouteDemandProjection>(
      std::move(*serviceRoutes));
  auto demand =
      finalizeSystemCandidateProjection(resources_, *progressModel_, *cache);
  if (!demand)
    return demand.takeError();
  return SystemCandidateProjectionResult{std::move(*demand), std::move(cache)};
}

llvm::Expected<SystemCandidateProjectionResult>
SystemCapacityModel::projectResourceDelta(
    const FrozenSystemPnrProblem &problem,
    SystemCandidateCapacityProjectionView candidate,
    const SystemCandidateProjectionCache &previous) const {
  if (!progressModel_)
    return invalid("System progress model is absent");
  if (!previous.nonRoute || !previous.serviceRoutes ||
      !hasSameExecutionSelections(*previous.nonRoute, candidate))
    return invalid("System resource delta changed an execution selection");
  if (previous.nonRoute->mutableUseCount > previous.nonRoute->uses.size() ||
      previous.nonRoute->mutableProgressActivationCount >
          previous.nonRoute->progressActivations.size())
    return invalid("System resource delta cache is inconsistent");
  auto mutableDemand = projectMutableResourceDemand(
      resources_, patternPhysicalOwnerKeys_, problem, candidate);
  if (!mutableDemand)
    return mutableDemand.takeError();

  auto nonRoute =
      std::make_shared<SystemNonRouteDemandProjection>(*previous.nonRoute);
  nonRoute->instructionResourceUses.assign(
      candidate.instructionResourceUses.begin(),
      candidate.instructionResourceUses.end());
  nonRoute->serviceResourceUses.assign(candidate.serviceResourceUses.begin(),
                                       candidate.serviceResourceUses.end());
  std::vector<FrozenResourceCapacityUseSelection> uses =
      std::move(mutableDemand->uses);
  nonRoute->mutableUseCount = uses.size();
  uses.insert(uses.end(),
              previous.nonRoute->uses.begin() +
                  previous.nonRoute->mutableUseCount,
              previous.nonRoute->uses.end());
  nonRoute->uses = std::move(uses);
  std::vector<::loom::mapping::MappingProgressActivationProjection>
      activations = std::move(mutableDemand->progressActivations);
  nonRoute->mutableProgressActivationCount = activations.size();
  activations.insert(activations.end(),
                     previous.nonRoute->progressActivations.begin() +
                         previous.nonRoute->mutableProgressActivationCount,
                     previous.nonRoute->progressActivations.end());
  nonRoute->progressActivations = std::move(activations);

  auto cache = std::make_shared<SystemCandidateProjectionCache>();
  cache->nonRoute = std::move(nonRoute);
  cache->serviceRoutes = previous.serviceRoutes;
  auto demand =
      finalizeSystemCandidateProjection(resources_, *progressModel_, *cache);
  if (!demand)
    return demand.takeError();
  return SystemCandidateProjectionResult{std::move(*demand), std::move(cache)};
}

llvm::Expected<SystemCandidatePhysicalDemandProjection>
SystemCapacityModel::project(
    const FrozenSystemPnrProblem &problem,
    SystemCandidateCapacityProjectionView candidate) const {
  auto projected = projectWithCache(problem, candidate);
  if (!projected)
    return projected.takeError();
  return std::move(projected->demand);
}

} // namespace loom::pnr::detail
