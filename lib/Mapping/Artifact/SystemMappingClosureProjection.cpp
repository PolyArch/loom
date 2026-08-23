#include "Mapping/Artifact/SystemMappingClosureProjection.h"

#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "ResourceCapacityVerification.h"
#include "SpatialMappingCapacityVerification.h"
#include "SystemMappingCapacityVerification.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>

namespace loom::mapping {
namespace {

using detail::FrozenResourceCapacityIndex;
using detail::FrozenResourceCapacityRouteSelection;
using detail::ResourceCapacityNamespaceView;
using detail::ResourceCapacityPatternSource;
using detail::ResourceCapacityTraversalSource;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_mapping_closure_projection_invalid: " +
                                     message);
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
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

void appendI64(std::string &bytes, std::int64_t value) {
  appendU64(bytes, static_cast<std::uint64_t>(value));
}

void appendSized(std::string &bytes, llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.append(reinterpret_cast<const char *>(value.data()), value.size());
}

void appendSized(std::string &bytes, llvm::StringRef value) {
  appendU64(bytes, value.size());
  bytes.append(value.data(), value.size());
}

template <typename Ref>
llvm::Expected<std::string>
dataflowKey(const ::dataflow::CanonicalDataflowProgramView &dataflow,
            const Ref &reference) {
  auto encoded =
      ::dataflow::encodeDataflowReference(dataflow.identity(), reference);
  if (!encoded)
    return encoded.takeError();
  return byteKey(*encoded);
}

template <typename Ref> std::string fabricKey(const Ref &reference) {
  return byteKey(::loom::fabric::canonicalFabricBytes(reference));
}

void appendCell(std::string &bytes, const SystemPresburgerCell &cell) {
  appendU32(bytes, cell.dimensionCount);
  appendU32(bytes, cell.symbolCount);
  appendU32(bytes, cell.localCount);
  const auto appendRows = [&](const auto &rows) {
    appendU64(bytes, rows.size());
    for (const auto &row : rows) {
      appendU64(bytes, row.size());
      for (std::int64_t value : row)
        appendI64(bytes, value);
    }
  };
  appendRows(cell.equalities);
  appendRows(cell.inequalities);
}

void canonicalizeCells(std::vector<SystemPresburgerCell> &cells) {
  llvm::sort(cells, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.dimensionCount, lhs.symbolCount, lhs.localCount,
                    lhs.equalities, lhs.inequalities) <
           std::tie(rhs.dimensionCount, rhs.symbolCount, rhs.localCount,
                    rhs.equalities, rhs.inequalities);
  });
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
}

struct NamespaceMetadata final {
  std::optional<::loom::fabric::SpatialCoreOccurrenceRef> spatialCore;
};

struct PendingActivation final {
  std::size_t namespaceOrdinal = 0;
  ::loom::fabric::FabricUsePatternRef usePattern;
  ExecutionContextKey context;
  std::vector<SystemPresburgerCell> relationDomain;
  std::vector<::dataflow::EventFamilyKey> triggerAlternatives;
  std::vector<SystemCausalReleasePointProjection> causalRelease;
  std::vector<::fabric::UsePatternValue> parameters;
  std::vector<::fabric::UsePatternValue> sharingAssignments;
};

const ::loom::fabric::FabricArtifactView *
resolveOccurrenceModule(const ::loom::fabric::FabricSystemRootView &fabric,
                        ::loom::fabric::AccCoreOccurrenceRef core,
                        const SpatialMappingView &mapping) {
  const auto target = fabric.spatialCoreTarget(core);
  if (!target ||
      target->dependencyOrdinal >= fabric.artifact().importedModules().size())
    return nullptr;
  const auto &module =
      fabric.artifact().importedModules()[target->dependencyOrdinal];
  const auto root = module.moduleRootTemplate();
  if (!root || *root != target->target ||
      module.identity() != mapping.fabricIdentity())
    return nullptr;
  return &module;
}

llvm::Error collectSpatialRouteTraversals(
    const SpatialMappingView &mapping, std::size_t namespaceOrdinal,
    std::vector<ResourceCapacityTraversalSource> &sources,
    std::vector<FrozenResourceCapacityRouteSelection> &routes,
    const FrozenResourceCapacityIndex *index = nullptr) {
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
    if (!index) {
      for (const auto &traversal : traversals)
        sources.push_back({namespaceOrdinal, traversal});
      continue;
    }
    FrozenResourceCapacityRouteSelection selected;
    for (const auto &traversal : traversals) {
      auto ordinal = index->traversalOrdinal(namespaceOrdinal, traversal);
      if (!ordinal)
        return ordinal.takeError();
      selected.traversalOrdinals.push_back(*ordinal);
    }
    routes.push_back(std::move(selected));
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<::dataflow::EventFamilyKey>>
projectSpatialEvent(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                    ::dataflow::RootedGraphLaunchRef graph,
                    const SpatialActivityEventRef &event) {
  return std::visit(
      [&](const auto &typed)
          -> llvm::Expected<std::vector<::dataflow::EventFamilyKey>> {
        using Event = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Event, SpatialActorTransitionEventRef>) {
          ::dataflow::EventFamilyKey projected(
              ::dataflow::ContextualActorTransitionEventRef{
                  ::dataflow::ContextualActorRef{graph, typed.actor},
                  typed.transition});
          if (llvm::Error error = dataflow.validate(projected))
            return std::move(error);
          return std::vector<::dataflow::EventFamilyKey>{std::move(projected)};
        } else {
          return dataflow.projectRootedGraphEndpointEventFamilies(graph, typed);
        }
      },
      event);
}

llvm::Expected<::dataflow::GraphRef>
spatialEventGraph(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                  const SpatialActivityEventRef &event) {
  return std::visit(
      [&](const auto &typed) -> llvm::Expected<::dataflow::GraphRef> {
        using Event = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Event, SpatialActorTransitionEventRef>) {
          auto actor = dataflow.resolve(typed.actor);
          if (!actor)
            return actor.takeError();
          return actor->graph;
        } else {
          return dataflow.graphOf(typed);
        }
      },
      event);
}

llvm::Expected<std::vector<SystemCausalReleasePointProjection>>
projectSpatialRelease(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                      ::dataflow::RootedGraphLaunchRef graph,
                      llvm::ArrayRef<SpatialEventPointView> release) {
  auto launchedGraph = dataflow.resolve(graph);
  if (!launchedGraph)
    return launchedGraph.takeError();
  std::vector<SystemCausalReleasePointProjection> result;
  result.reserve(release.size());
  for (const auto &point : release) {
    auto ownerGraph = spatialEventGraph(dataflow, point.event);
    if (!ownerGraph)
      return ownerGraph.takeError();
    if (*ownerGraph != *launchedGraph)
      continue;
    auto alternatives = projectSpatialEvent(dataflow, graph, point.event);
    if (!alternatives)
      return alternatives.takeError();
    result.push_back({std::move(*alternatives), point.guaranteedOffset});
  }
  if (!release.empty() && result.empty())
    return invalid("Spatial activation has no release in its rooted graph");
  return result;
}

llvm::Expected<std::vector<SystemPresburgerCell>>
selectedPlanCells(const SystemServicePlanSelectionView &selection,
                  std::uint64_t planOrdinal,
                  llvm::ArrayRef<SystemPresburgerCell> contextDomain) {
  std::vector<SystemPresburgerCell> explicitCells;
  std::vector<SystemPresburgerCell> selected;
  for (const auto &clause : selection.clauses) {
    explicitCells.insert(explicitCells.end(), clause.cells.begin(),
                         clause.cells.end());
    if (clause.target == planOrdinal)
      selected.insert(selected.end(), clause.cells.begin(), clause.cells.end());
  }
  if (selection.defaultPlanOrdinal == planOrdinal) {
    auto complement = splitSystemPresburgerSet(contextDomain, explicitCells);
    if (!complement)
      return complement.takeError();
    selected.insert(selected.end(),
                    std::make_move_iterator(complement->outside.begin()),
                    std::make_move_iterator(complement->outside.end()));
  }
  canonicalizeCells(selected);
  return selected;
}

llvm::Expected<ServicePlanSelectionAnchor>
resourceSelectionAnchor(const SystemResourceUseView &use) {
  const auto *service =
      std::get_if<SystemServicePlanResourceOwnerView>(&use.owner);
  if (!service)
    return invalid("Instruction ResourceUse has no service selection anchor");
  const auto *transition =
      std::get_if<::dataflow::ContextualActorTransitionEventRef>(
          &use.activation.trigger.event);
  if (!transition)
    return invalid("service ResourceUse trigger is not an actor transition");
  return std::visit(
      [&](const auto &element) -> llvm::Expected<ServicePlanSelectionAnchor> {
        using Element = std::decay_t<decltype(element)>;
        if constexpr (std::is_same_v<Element, SystemMemoryRegionElementView>) {
          return ServicePlanSelectionAnchor(
              ServiceMemberPlanSelectionAnchor{::dataflow::ServiceMemberRef(
                  ::dataflow::AddressedMemoryActorMemberRef{
                      transition->actor})});
        } else if constexpr (std::is_same_v<Element,
                                            SystemConsistencyElementView>) {
          return ServicePlanSelectionAnchor(
              ServiceMemberPlanSelectionAnchor{::dataflow::ServiceMemberRef(
                  ::dataflow::FenceActorMemberRef{transition->actor})});
        } else {
          return invalid("transfer route cannot own a System ResourceUse");
        }
      },
      service->element);
}

const SystemServiceRealizationView *
findService(llvm::ArrayRef<SystemServiceRealizationView> services,
            const SystemServiceObligationKey &key) {
  const auto found = llvm::find_if(
      services, [&](const auto &service) { return service.key == key; });
  return found == services.end() ? nullptr : &*found;
}

llvm::Expected<std::vector<SystemPresburgerCell>>
contextDomain(const SystemExecutionContextProjection &contexts,
              const ::dataflow::ContextualActorRef &actor,
              const ExecutionContextKey &context) {
  const auto *spatial = std::get_if<SpatialExecutionContextKey>(&context);
  if (!spatial)
    return invalid("actor service selection has an Instruction context");
  for (const auto &domain : contexts.spatialDomains)
    if (domain.graph == actor.launch && domain.context == *spatial)
      return domain.cells;
  return invalid("service selection has no reachable Spatial context");
}

llvm::Error
appendDirectActivation(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                       const SystemExecutionContextProjection &contexts,
                       llvm::ArrayRef<SystemServiceRealizationView> services,
                       const SystemResourceUseView &use,
                       std::vector<PendingActivation> &pending) {
  if (const auto *instruction =
          std::get_if<SystemInstructionResourceOwnerView>(&use.owner)) {
    bool found = false;
    for (const auto &domain : contexts.instructionDomains) {
      if (domain.root != instruction->root ||
          domain.context.accCore != instruction->instructionContext.core)
        continue;
      found = true;
      pending.push_back(PendingActivation{0,
                                          use.useSite,
                                          ExecutionContextKey(domain.context),
                                          domain.cells,
                                          {use.activation.trigger.event},
                                          {},
                                          use.parameters,
                                          use.sharingAssignments});
      for (const auto &release : use.activation.release)
        pending.back().causalRelease.push_back(
            {{release.event}, release.guaranteedOffset});
    }
    if (!found)
      return invalid("Instruction ResourceUse has no reachable context");
    return llvm::Error::success();
  }

  const auto &owner = std::get<SystemServicePlanResourceOwnerView>(use.owner);
  const auto *service = findService(services, owner.service);
  if (!service)
    return invalid("service ResourceUse has no ServiceRealization");
  auto anchor = resourceSelectionAnchor(use);
  if (!anchor)
    return anchor.takeError();
  const auto *transition =
      std::get_if<::dataflow::ContextualActorTransitionEventRef>(
          &use.activation.trigger.event);
  if (!transition)
    return invalid("service ResourceUse trigger is not contextual");

  bool found = false;
  for (const auto &selection : service->selections) {
    if (!(selection.key.anchor == *anchor))
      continue;
    auto domain =
        contextDomain(contexts, transition->actor, selection.key.context);
    if (!domain)
      return domain.takeError();
    auto cells = selectedPlanCells(selection, owner.planOrdinal, *domain);
    if (!cells)
      return cells.takeError();
    if (cells->empty())
      continue;
    found = true;
    pending.push_back(PendingActivation{0,
                                        use.useSite,
                                        selection.key.context,
                                        std::move(*cells),
                                        {use.activation.trigger.event},
                                        {},
                                        use.parameters,
                                        use.sharingAssignments});
    for (const auto &release : use.activation.release)
      pending.back().causalRelease.push_back(
          {{release.event}, release.guaranteedOffset});
  }
  if (!found)
    return invalid("service ResourceUse has no selected plan domain");
  return llvm::Error::success();
}

llvm::Expected<std::string>
activationKey(const ::dataflow::CanonicalDataflowProgramView &dataflow,
              const ::loom::fabric::FabricSystemRootView &fabric,
              const SystemResourceActivationProjection &activation) {
  std::string result;
  auto context = encodeExecutionContextKey(activation.context);
  if (!context)
    return context.takeError();
  appendSized(result, *context);
  appendU64(result, activation.relationDomain.size());
  for (const auto &cell : activation.relationDomain)
    appendCell(result, cell);
  appendU64(result, activation.triggerAlternatives.size());
  for (const auto &event : activation.triggerAlternatives) {
    auto encoded = dataflowKey(dataflow, event);
    if (!encoded)
      return encoded.takeError();
    appendSized(result, *encoded);
  }
  appendSized(result,
              ::loom::fabric::canonicalFabricBytes(activation.physicalOwner));
  appendU64(result, activation.usePatternOrdinal);

  auto resolved = fabric.resolvePhysicalOwner(activation.physicalOwner);
  if (!resolved)
    return resolved.takeError();
  const auto *contract =
      resolved->artifact.resourceContract(resolved->localOwner);
  if (!contract || activation.usePatternOrdinal >= contract->usePatternCount())
    return invalid("activation UsePattern is absent from its physical owner");
  const auto pattern = contract->usePattern(
      ::fabric::UsePatternKey(activation.usePatternOrdinal));
  const auto appendValues =
      [&](llvm::ArrayRef<::fabric::UsePatternValueSchema> schemas,
          llvm::ArrayRef<::fabric::UsePatternValue> values) -> llvm::Error {
    if (schemas.size() != values.size())
      return invalid("activation value count disagrees with UsePattern");
    appendU64(result, values.size());
    for (const auto &[schema, value] : llvm::zip_equal(schemas, values)) {
      auto encoded = ::fabric::encodeUsePatternValue(schema, value);
      if (!encoded)
        return encoded.takeError();
      appendSized(result, *encoded);
    }
    return llvm::Error::success();
  };
  if (llvm::Error error =
          appendValues(pattern.parameters, activation.parameters))
    return std::move(error);
  if (llvm::Error error = appendValues(pattern.sharingAssignments,
                                       activation.sharingAssignments))
    return std::move(error);

  appendU64(result, activation.capacityClaims.size());
  for (const auto &claim : activation.capacityClaims) {
    appendU64(result, claim.capacityCellOrdinal);
    appendU64(result, claim.amount);
  }
  appendU64(result, activation.causalRelease.size());
  for (const auto &point : activation.causalRelease) {
    appendU64(result, point.alternatives.size());
    for (const auto &event : point.alternatives) {
      auto encoded = dataflowKey(dataflow, event);
      if (!encoded)
        return encoded.takeError();
      appendSized(result, *encoded);
    }
    appendU32(result, point.guaranteedOffset ? 1 : 0);
    if (point.guaranteedOffset)
      appendSized(result, *point.guaranteedOffset);
  }
  return result;
}

} // namespace

llvm::Expected<std::vector<::dataflow::EventFamilyKey>>
projectSystemSpatialActivityEvent(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::RootedGraphLaunchRef graph,
    const SpatialActivityEventRef &event) {
  return projectSpatialEvent(dataflow, graph, event);
}

llvm::Expected<::dataflow::GraphRef> resolveSystemSpatialActivityEventGraph(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const SpatialActivityEventRef &event) {
  return spatialEventGraph(dataflow, event);
}

llvm::Expected<std::vector<SystemCausalReleasePointProjection>>
projectSystemSpatialCausalRelease(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::RootedGraphLaunchRef graph,
    llvm::ArrayRef<SpatialEventPointView> release) {
  return projectSpatialRelease(dataflow, graph, release);
}

llvm::Expected<SystemMappingClosureProjection> projectSystemMappingClosure(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemMappingView &mapping, const ArtifactStore &store,
    const SpatialMappingImportContext *spatialMappings,
    ExecutionControlView executionControl) {
  const auto interrupted = [&]() -> llvm::Error {
    return llvm::createStringError(std::errc::timed_out,
                                   "System closure projection was "
                                   "interrupted");
  };
  if (executionControl.stopRequested())
    return interrupted();
  if (mapping.dataflowIdentity() != dataflow.identity() ||
      mapping.fabricIdentity() != fabric.artifact().identity())
    return invalid("projection inputs disagree with SystemMapping lineage");

  auto contexts =
      projectSystemExecutionContexts(dataflow, mapping.executionBindings());
  if (!contexts)
    return contexts.takeError();

  std::optional<SpatialMappingImportContext> ownedSpatialMappings;
  if (!spatialMappings) {
    auto built = buildSpatialMappingImportContext(
        mapping.executionBindings().spatialMappingImports(), store);
    if (!built)
      return built.takeError();
    ownedSpatialMappings.emplace(std::move(*built));
    spatialMappings = &*ownedSpatialMappings;
  }
  std::vector<ArtifactRootReference> canonicalImports(
      mapping.executionBindings().spatialMappingImports().begin(),
      mapping.executionBindings().spatialMappingImports().end());
  llvm::sort(canonicalImports, artifactRootReferenceLess);
  if (std::adjacent_find(canonicalImports.begin(), canonicalImports.end()) !=
      canonicalImports.end())
    return invalid("SystemMapping import table contains a duplicate");
  for (const ArtifactRootReference &reference : canonicalImports) {
    if (executionControl.stopRequested())
      return interrupted();
    if (!spatialMappings->find(reference))
      return invalid("SpatialMapping import context does not cover the exact "
                     "SystemMapping import table");
  }

  std::vector<ResourceCapacityNamespaceView> namespaces{
      {&fabric.artifact(),
       detail::rootResourceCapacityQualifier(fabric.artifact())}};
  std::vector<NamespaceMetadata> metadata{{std::nullopt}};
  std::map<std::string, std::size_t> namespaceByOccurrence;
  std::map<std::string, const FinalizedSpatialMapping *> importedMappings;
  std::map<std::string, FinalizedTechMapping> importedTechMappings;
  std::map<std::string, std::size_t> namespaceByContext;

  for (const auto &domain : contexts->spatialDomains) {
    if (executionControl.stopRequested())
      return interrupted();
    const std::string mappingKey =
        byteKey(encodeArtifactRootReference(domain.spatialMapping));
    auto imported = importedMappings.find(mappingKey);
    if (imported == importedMappings.end()) {
      auto value =
          resolveSpatialMappingImport(*spatialMappings, domain.spatialMapping);
      if (!value)
        return value.takeError();
      imported = importedMappings.emplace(mappingKey, *value).first;
      ArtifactRootReference techReference{
          mappingArtifactSchema.identity.str(), mappingArtifactSchema.version,
          imported->second->view().techMappingIdentity()};
      auto tech = importTechMapping(techReference, store);
      if (!tech)
        return tech.takeError();
      if (tech->view().dataflowIdentity() != dataflow.identity() ||
          tech->view().fabricIdentity() !=
              imported->second->view().fabricIdentity())
        return invalid("Spatial context has inconsistent TechMapping lineage");
      importedTechMappings.emplace(mappingKey, std::move(*tech));
    }
    const auto spatialCore =
        ::loom::fabric::SpatialCoreOccurrenceRef{domain.context.accCore};
    const auto *module = resolveOccurrenceModule(fabric, domain.context.accCore,
                                                 imported->second->view());
    if (!module)
      return invalid("Spatial context does not match its AccCore Module");
    const std::string occurrenceKey = fabricKey(spatialCore);
    auto [position, inserted] =
        namespaceByOccurrence.try_emplace(occurrenceKey, namespaces.size());
    if (inserted) {
      namespaces.push_back({module, detail::occurrenceResourceCapacityQualifier(
                                        fabric.artifact(), spatialCore)});
      metadata.push_back({spatialCore});
    } else if (namespaces[position->second].fabric->identity() !=
               module->identity()) {
      return invalid("one SpatialCore occurrence resolves two Modules");
    }
    auto encodedContext = encodeExecutionContextKey(domain.context);
    if (!encodedContext)
      return encodedContext.takeError();
    namespaceByContext[byteKey(*encodedContext)] = position->second;
  }

  std::vector<PendingActivation> pending;
  for (const auto &use : mapping.resourceUses())
    if (llvm::Error error = appendDirectActivation(
            dataflow, *contexts, mapping.serviceRealizations(), use, pending))
      return std::move(error);

  std::set<std::string> routedMappings;
  for (const auto &domain : contexts->spatialDomains) {
    auto launchedGraph = dataflow.resolve(domain.graph);
    if (!launchedGraph)
      return launchedGraph.takeError();
    auto encodedContext = encodeExecutionContextKey(domain.context);
    if (!encodedContext)
      return encodedContext.takeError();
    const auto namespacePosition =
        namespaceByContext.find(byteKey(*encodedContext));
    if (namespacePosition == namespaceByContext.end())
      return invalid("Spatial context has no capacity namespace");
    const std::string mappingKey =
        byteKey(encodeArtifactRootReference(domain.spatialMapping));
    const auto imported = importedMappings.find(mappingKey);
    if (imported == importedMappings.end())
      return invalid("Spatial context lost its imported Mapping");
    for (const auto &use : imported->second->view().resourceUses()) {
      auto ownerGraph =
          spatialEventGraph(dataflow, use.activation.trigger.event);
      if (!ownerGraph)
        return ownerGraph.takeError();
      if (*ownerGraph != *launchedGraph)
        continue;
      auto trigger = projectSpatialEvent(dataflow, domain.graph,
                                         use.activation.trigger.event);
      if (!trigger)
        return trigger.takeError();
      auto release =
          projectSpatialRelease(dataflow, domain.graph, use.activation.release);
      if (!release)
        return release.takeError();
      pending.push_back(
          PendingActivation{namespacePosition->second, use.useSite,
                            ExecutionContextKey(domain.context), domain.cells,
                            std::move(*trigger), std::move(*release),
                            use.parameters, use.sharingAssignments});
    }
    std::string routedKey = fabricKey(domain.context.accCore);
    appendSized(routedKey, domain.context.spatialMapping.bytes());
    routedMappings.insert(std::move(routedKey));
  }

  std::vector<ResourceCapacityPatternSource> patternSources;
  patternSources.reserve(pending.size());
  for (const auto &activation : pending)
    patternSources.push_back(
        {activation.namespaceOrdinal, activation.usePattern});

  std::vector<ResourceCapacityTraversalSource> traversalSources;
  for (const auto &service : mapping.serviceRealizations())
    for (const auto &plan : service.plans)
      for (const auto &leg : plan.transferLegs)
        for (const auto &node : leg.nodes)
          traversalSources.push_back({0, node.incomingTraversal});
  for (const auto &domain : contexts->spatialDomains) {
    std::string routedKey = fabricKey(domain.context.accCore);
    appendSized(routedKey, domain.context.spatialMapping.bytes());
    if (routedMappings.erase(routedKey) == 0)
      continue;
    auto encodedContext = encodeExecutionContextKey(domain.context);
    if (!encodedContext)
      return encodedContext.takeError();
    const auto namespacePosition =
        namespaceByContext.find(byteKey(*encodedContext));
    const std::string mappingKey =
        byteKey(encodeArtifactRootReference(domain.spatialMapping));
    const auto imported = importedMappings.find(mappingKey);
    if (namespacePosition == namespaceByContext.end() ||
        imported == importedMappings.end())
      return invalid("routed SpatialMapping lost its capacity namespace");
    std::vector<FrozenResourceCapacityRouteSelection> unused;
    if (llvm::Error error = collectSpatialRouteTraversals(
            imported->second->view(), namespacePosition->second,
            traversalSources, unused))
      return std::move(error);
  }

  auto capacity = detail::freezeResourceCapacityIndex(
      namespaces, patternSources, traversalSources);
  if (!capacity)
    return capacity.takeError();

  std::vector<FrozenResourceCapacityRouteSelection> selectedRoutes;
  for (const auto &service : mapping.serviceRealizations())
    for (const auto &plan : service.plans)
      for (const auto &leg : plan.transferLegs) {
        FrozenResourceCapacityRouteSelection route;
        for (const auto &node : leg.nodes) {
          auto ordinal = capacity->traversalOrdinal(0, node.incomingTraversal);
          if (!ordinal)
            return ordinal.takeError();
          route.traversalOrdinals.push_back(*ordinal);
        }
        selectedRoutes.push_back(std::move(route));
      }
  routedMappings.clear();
  for (const auto &domain : contexts->spatialDomains) {
    std::string routedKey = fabricKey(domain.context.accCore);
    appendSized(routedKey, domain.context.spatialMapping.bytes());
    if (!routedMappings.insert(routedKey).second)
      continue;
    auto encodedContext = encodeExecutionContextKey(domain.context);
    if (!encodedContext)
      return encodedContext.takeError();
    const auto namespacePosition =
        namespaceByContext.find(byteKey(*encodedContext));
    const std::string mappingKey =
        byteKey(encodeArtifactRootReference(domain.spatialMapping));
    const auto imported = importedMappings.find(mappingKey);
    if (namespacePosition == namespaceByContext.end() ||
        imported == importedMappings.end())
      return invalid("selected route lost its SpatialMapping namespace");
    std::vector<ResourceCapacityTraversalSource> unused;
    if (llvm::Error error = collectSpatialRouteTraversals(
            imported->second->view(), namespacePosition->second, unused,
            selectedRoutes, &*capacity))
      return std::move(error);
  }
  auto baseline = detail::deriveResourceCapacityBaselineOccupancy(
      *capacity, selectedRoutes);
  if (!baseline)
    return baseline.takeError();

  struct KeyedCapacityCell final {
    std::size_t oldOrdinal = 0;
    std::string key;
    ::loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOwner;
  };
  std::vector<KeyedCapacityCell> cellOrder;
  cellOrder.reserve(capacity->cells().size());
  for (const auto &[oldOrdinal, cell] : llvm::enumerate(capacity->cells())) {
    if (cell.namespaceOrdinal >= metadata.size())
      return invalid("capacity cell has no physical namespace");
    auto physical = detail::qualifySystemResourceOwner(
        cell.owner, metadata[cell.namespaceOrdinal].spatialCore);
    if (!physical)
      return physical.takeError();
    std::string key = fabricKey(*physical);
    appendU32(key, cell.state.ordinal());
    appendU32(key, cell.dimension.ordinal());
    cellOrder.push_back({oldOrdinal, std::move(key), std::move(*physical)});
  }
  llvm::sort(cellOrder, [](const auto &lhs, const auto &rhs) {
    return lhs.key < rhs.key;
  });
  for (auto adjacent : llvm::zip(cellOrder, llvm::drop_begin(cellOrder)))
    if (std::get<0>(adjacent).key == std::get<1>(adjacent).key)
      return invalid("capacity catalog contains a duplicate physical cell");

  std::vector<std::uint64_t> canonicalCellOrdinal(cellOrder.size());
  SystemMappingClosureProjection result;
  result.executionContexts = std::move(*contexts);
  result.serviceRealizations.assign(mapping.serviceRealizations().begin(),
                                    mapping.serviceRealizations().end());
  std::vector<SystemTransferLegView> serviceLegs;
  for (const SystemServiceRealizationView &service : result.serviceRealizations)
    for (const SystemServicePlanView &plan : service.plans)
      serviceLegs.insert(serviceLegs.end(), plan.transferLegs.begin(),
                         plan.transferLegs.end());
  auto serviceProgress =
      projectSystemTransferRouteProgress(dataflow, serviceLegs);
  if (!serviceProgress)
    return serviceProgress.takeError();
  result.routeObligations = std::move(*serviceProgress);

  std::set<std::pair<std::string, std::uint64_t>> projectedSpatialGraphs;
  std::vector<::dataflow::GraphRef> selectedProgressGraphs;
  for (const SystemSpatialContextDomain &domain :
       result.executionContexts.spatialDomains) {
    auto graph = dataflow.resolve(domain.graph);
    if (!graph)
      return graph.takeError();
    const std::string mappingKey =
        byteKey(encodeArtifactRootReference(domain.spatialMapping));
    if (!projectedSpatialGraphs.emplace(mappingKey, graph->entity.value())
             .second)
      continue;
    const auto spatial = importedMappings.find(mappingKey);
    const auto tech = importedTechMappings.find(mappingKey);
    const auto *module =
        spatial == importedMappings.end()
            ? nullptr
            : resolveOccurrenceModule(fabric, domain.context.accCore,
                                      spatial->second->view());
    if (spatial == importedMappings.end() ||
        tech == importedTechMappings.end() || !module)
      return invalid("Spatial progress projection lost an imported owner");
    const std::array<::dataflow::GraphRef, 1> selected{*graph};
    selectedProgressGraphs.push_back(*graph);
    auto progress = projectSpatialMappingProgress(
        dataflow, tech->second.view(), *module,
        spatial->second->view().computeBindings(),
        spatial->second->view().registerFifoTransfers(),
        spatial->second->view().routeTrees(), selected);
    if (!progress)
      return progress.takeError();
    result.routeObligations.insert(result.routeObligations.end(),
                                   progress->routeObligations.begin(),
                                   progress->routeObligations.end());
  }
  llvm::sort(selectedProgressGraphs, [](const auto lhs, const auto rhs) {
    return lhs.entity.value() < rhs.entity.value();
  });
  selectedProgressGraphs.erase(
      std::unique(selectedProgressGraphs.begin(), selectedProgressGraphs.end()),
      selectedProgressGraphs.end());
  auto progressBasis =
      deriveMappingDataflowProgressBasis(dataflow, selectedProgressGraphs);
  if (!progressBasis)
    return progressBasis.takeError();
  result.progressBasis = std::move(*progressBasis);
  result.capacityCells.reserve(cellOrder.size());
  for (const auto &[canonicalOrdinal, keyed] : llvm::enumerate(cellOrder)) {
    const auto &cell = capacity->cells()[keyed.oldOrdinal];
    if (keyed.oldOrdinal >= baseline->size())
      return invalid("capacity cell has no physical namespace");
    if ((*baseline)[keyed.oldOrdinal] > cell.capacity)
      return invalid("static route baseline exceeds Fabric capacity");
    canonicalCellOrdinal[keyed.oldOrdinal] = canonicalOrdinal;
    result.capacityCells.push_back({keyed.physicalOwner, cell.state,
                                    cell.dimension, cell.capacity,
                                    (*baseline)[keyed.oldOrdinal]});
  }

  std::vector<std::pair<std::string, SystemResourceActivationProjection>>
      keyedActivations;
  keyedActivations.reserve(pending.size());
  for (PendingActivation &source : pending) {
    if (source.namespaceOrdinal >= metadata.size())
      return invalid("activation has no physical namespace");
    auto physical = detail::qualifySystemResourceOwner(
        source.usePattern.owner.catalog(),
        metadata[source.namespaceOrdinal].spatialCore);
    if (!physical)
      return physical.takeError();
    auto patternOrdinal =
        capacity->patternOrdinal(source.namespaceOrdinal, source.usePattern);
    if (!patternOrdinal)
      return patternOrdinal.takeError();
    if (*patternOrdinal >= capacity->patterns().size())
      return invalid("activation has a foreign frozen UsePattern");
    canonicalizeCells(source.relationDomain);
    SystemResourceActivationProjection activation{
        std::move(source.context),
        std::move(source.relationDomain),
        std::move(source.triggerAlternatives),
        std::move(*physical),
        source.usePattern.ordinal,
        std::move(source.parameters),
        std::move(source.sharingAssignments),
        {},
        std::move(source.causalRelease)};
    for (const auto &claim : capacity->patterns()[*patternOrdinal].claims) {
      if (claim.cell >= canonicalCellOrdinal.size())
        return invalid("activation claim names a foreign capacity cell");
      activation.capacityClaims.push_back(
          {canonicalCellOrdinal[claim.cell], claim.amount});
    }
    llvm::sort(activation.capacityClaims, [](const auto &lhs, const auto &rhs) {
      return std::tie(lhs.capacityCellOrdinal, lhs.amount) <
             std::tie(rhs.capacityCellOrdinal, rhs.amount);
    });
    auto key = activationKey(dataflow, fabric, activation);
    if (!key)
      return key.takeError();
    keyedActivations.emplace_back(std::move(*key), std::move(activation));
  }
  llvm::sort(keyedActivations, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  result.resourceActivations.reserve(keyedActivations.size());
  for (auto &entry : keyedActivations)
    result.resourceActivations.push_back(std::move(entry.second));
  return result;
}

} // namespace loom::mapping
