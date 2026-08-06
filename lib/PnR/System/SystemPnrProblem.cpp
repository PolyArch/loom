#include "PnR/System/SystemPnrProblem.h"

#include "PnR/InitializerRelationSolver.h"
#include "SystemPnrSearchDomainInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ComponentViewDigest.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::pnr;

char SystemPnrFreezeFailure::ID;

void SystemPnrFreezeFailure::log(llvm::raw_ostream &stream) const {
  stream << (kind_ == SystemPnrFreezeFailureKind::Invalid
                 ? "system_pnr_freeze_invalid: "
                 : "system_pnr_proven_infeasible: ")
         << message_;
}

std::error_code SystemPnrFreezeFailure::convertToErrorCode() const {
  return std::make_error_code(std::errc::invalid_argument);
}

namespace {

constexpr llvm::StringLiteral frozenArtifact = "FrozenSystemPnrProblem";
constexpr PnrCapacityContext catalogIndexContext{
    frozenArtifact, "target_catalog", "target", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext choiceOffsetContext{
    frozenArtifact, "execution_decisions", "choice",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext choiceCountContext{
    frozenArtifact, "execution_decisions", "choice", PnrCapacityMeasure::Count};
constexpr PnrCapacityContext decisionIndexContext{
    frozenArtifact, "execution_decisions", "decision",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext overlapOffsetContext{
    frozenArtifact, "graph_thread_overlap", "overlap",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext serviceTerminalContext{
    frozenArtifact, "service_routing", "terminal", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext serviceEndpointChoiceContext{
    frozenArtifact, "service_routing", "endpoint_choice",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext serviceLegContext{
    frozenArtifact, "service_routing", "leg", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext serviceLegSinkContext{
    frozenArtifact, "service_routing", "sink", PnrCapacityMeasure::Offset};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<SystemPnrFreezeFailure>(
      SystemPnrFreezeFailureKind::Invalid, message.str());
}

llvm::Error infeasible(const llvm::Twine &message) {
  return llvm::make_error<SystemPnrFreezeFailure>(
      SystemPnrFreezeFailureKind::ProvenInfeasible, message.str());
}

llvm::Expected<PnrIndex> checked(PnrCapacityContext context,
                                 std::size_t value) {
  return checkedPnrIndex(context, static_cast<std::uint64_t>(value));
}

std::string bytesKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

std::string coreKey(::loom::fabric::AccCoreOccurrenceRef core) {
  return bytesKey(::loom::fabric::canonicalFabricBytes(core));
}

std::string targetClassKey(const FrozenSystemSpatialTargetClass &targetClass) {
  std::string key = bytesKey(targetClass.moduleIdentity.bytes());
  const auto moduleBytes =
      ::loom::fabric::canonicalFabricBytes(targetClass.moduleTemplate);
  key.append(reinterpret_cast<const char *>(moduleBytes.data()),
             moduleBytes.size());
  return key;
}

llvm::Expected<FrozenSystemSpatialTargetClass>
targetClassForModule(const ::loom::fabric::FabricArtifactView &module) {
  auto root = module.moduleRootTemplate();
  if (module.rootKind() != ::loom::fabric::FabricRootKind::Module || !root)
    return invalid("System SpatialCore dependency is not an exact Module root");
  return FrozenSystemSpatialTargetClass{module.identity(), *root};
}

struct FrozenSystemRoutingData final {
  FrozenEndpointRoutingTopology topology;
  std::vector<FrozenSystemTransferTerminal> terminals;
  std::vector<PnrIndex> endpointChoices;
  std::vector<FrozenSystemServiceLeg> legs;
  std::vector<PnrIndex> legSinks;
};

struct ServiceLegDraft final {
  ::loom::mapping::CanonicalServiceLegKey key;
  const SystemSearchTransferTerminalDomain *source = nullptr;
  std::vector<const SystemSearchTransferTerminalDomain *> sinks;
};

llvm::Expected<FrozenSystemRoutingData>
freezeSystemRouting(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                    const ::loom::fabric::FabricSystemRootView &fabric,
                    const SystemPnrSearchDomainView &searchDomain) {
  FrozenSystemRoutingData result;
  auto topology = freezeEndpointRoutingTopology(fabric.artifact());
  if (!topology)
    return topology.takeError();
  result.topology = std::move(*topology);

  llvm::StringMap<PnrIndex> endpointOrdinals;
  for (auto [ordinal, endpoint] :
       llvm::enumerate(result.topology.endpoints())) {
    auto index = checked(serviceEndpointChoiceContext, ordinal);
    if (!index)
      return index.takeError();
    const std::string key =
        bytesKey(::loom::fabric::canonicalFabricBytes(endpoint.reference));
    if (!endpointOrdinals.try_emplace(key, *index).second)
      return invalid("System routing topology has a duplicate endpoint");
  }

  std::map<std::string, std::uint32_t> producerWidths;
  for (const ::dataflow::RootThreadLaunchRef &root :
       searchDomain.rootThreadLaunches())
    if (llvm::Error error = dataflow.forEachProducerTerminal(
            root,
            [&](const ::dataflow::CanonicalProducerTerminalView &view)
                -> llvm::Error {
              auto key = ::dataflow::encodeDataflowReference(
                  dataflow.identity(), view.terminal);
              if (!key)
                return key.takeError();
              auto width = dataflow.transportPayloadBitWidth(view.payloadType);
              if (!width)
                return width.takeError();
              auto [found, inserted] =
                  producerWidths.emplace(bytesKey(*key), *width);
              if (!inserted && found->second != *width)
                return invalid(
                    "one producer terminal has inconsistent payload widths");
              return llvm::Error::success();
            }))
      return std::move(error);

  auto appendTerminal = [&](const SystemSearchTransferTerminalDomain &domain)
      -> llvm::Expected<PnrIndex> {
    auto terminal = checked(serviceTerminalContext, result.terminals.size());
    if (!terminal)
      return terminal.takeError();
    auto offset =
        checked(serviceEndpointChoiceContext, result.endpointChoices.size());
    if (!offset)
      return offset.takeError();
    std::vector<PnrIndex> choices;
    choices.reserve(domain.compatibleTransportEndpoints.size());
    for (const auto &endpoint : domain.compatibleTransportEndpoints) {
      auto found = endpointOrdinals.find(
          bytesKey(::loom::fabric::canonicalFabricBytes(endpoint)));
      if (found == endpointOrdinals.end())
        return invalid("H service terminal names an endpoint outside F");
      choices.push_back(found->second);
    }
    if (!llvm::is_sorted(choices) ||
        std::adjacent_find(choices.begin(), choices.end()) != choices.end())
      return invalid(
          "H service terminal endpoint domain is not canonical in F");
    auto count = checked(serviceEndpointChoiceContext, choices.size());
    if (!count)
      return count.takeError();
    result.endpointChoices.insert(result.endpointChoices.end(), choices.begin(),
                                  choices.end());
    result.terminals.push_back({domain.key, *offset, *count});
    return *terminal;
  };

  for (const SystemSearchServiceDomain &service :
       searchDomain.serviceObligations()) {
    const auto *producer =
        std::get_if<::loom::mapping::TransferObligationFamilyKey>(&service.key);
    if (!producer)
      return invalid(
          "operation-service target selection must precede System routing");
    auto producerKey =
        ::dataflow::encodeDataflowReference(dataflow.identity(), *producer);
    if (!producerKey)
      return producerKey.takeError();
    auto payloadWidth = producerWidths.find(bytesKey(*producerKey));
    if (payloadWidth == producerWidths.end())
      return invalid("H transfer obligation has no Dataflow producer");

    std::map<std::string, ServiceLegDraft> drafts;
    for (const SystemSearchTransferTerminalDomain &terminal :
         service.transferTerminals) {
      const ::loom::mapping::CanonicalServiceLegKey &leg =
          std::holds_alternative<
              ::loom::mapping::SystemTransferSourceTerminalKey>(terminal.key)
              ? std::get<::loom::mapping::SystemTransferSourceTerminalKey>(
                    terminal.key)
                    .leg
              : std::get<::loom::mapping::SystemTransferSinkTerminalKey>(
                    terminal.key)
                    .leg;
      if (leg.obligation != service.key)
        return invalid("H transfer terminal belongs to a foreign obligation");
      auto key = ::loom::mapping::encodeCanonicalServiceLegKey(
          dataflow.identity(), leg);
      if (!key)
        return key.takeError();
      auto [found, inserted] =
          drafts.try_emplace(bytesKey(*key), ServiceLegDraft{leg, nullptr, {}});
      ServiceLegDraft &draft = found->second;
      if (std::holds_alternative<
              ::loom::mapping::SystemTransferSourceTerminalKey>(terminal.key)) {
        if (draft.source)
          return invalid("H service leg has duplicate source terminals");
        draft.source = &terminal;
      } else {
        draft.sinks.push_back(&terminal);
      }
    }

    for (auto &[key, draft] : drafts) {
      if (draft.sinks.empty())
        continue;
      if (!draft.source)
        return invalid("H service leg with sinks has no source terminal");
      if (draft.source->compatibleTransportEndpoints.empty())
        return infeasible(
            "a service source terminal has no compatible endpoint");
      for (const auto *sink : draft.sinks)
        if (sink->compatibleTransportEndpoints.empty())
          return infeasible(
              "a service sink terminal has no compatible endpoint");

      auto source = appendTerminal(*draft.source);
      if (!source)
        return source.takeError();
      auto sinkOffset = checked(serviceLegSinkContext, result.legSinks.size());
      if (!sinkOffset)
        return sinkOffset.takeError();
      for (const auto *sink : draft.sinks) {
        auto terminal = appendTerminal(*sink);
        if (!terminal)
          return terminal.takeError();
        result.legSinks.push_back(*terminal);
      }
      auto sinkCount = checked(serviceLegSinkContext, draft.sinks.size());
      if (!sinkCount)
        return sinkCount.takeError();
      if (llvm::Error error = preflightPnrIndexCapacity(serviceLegContext,
                                                        result.legs.size() + 1))
        return std::move(error);
      result.legs.push_back({std::move(draft.key), *source, *sinkOffset,
                             *sinkCount, payloadWidth->second});
    }
  }
  return result;
}

llvm::Error validateInputs(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemPnrSearchDomainView &searchDomain,
    const ResolvedPnrConfigView &config,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints) {
  if (config.domain() != PnrConfigDomain::System)
    return invalid("System PnR received a non-System resolved config view");
  if (llvm::Error error = validateComponentViewDigest(
          config.schemaDescriptorBytes(), config.canonicalViewBytes(),
          config.digest()))
    return llvm::joinErrors(invalid("System PnR config digest is invalid"),
                            std::move(error));
  if (llvm::Error error = validateSystemPnrSearchDomainDigest(
          systemPnrSearchDomainSchemaDescriptorBytes(),
          searchDomain.canonicalViewBytes(), searchDomain.digest()))
    return llvm::joinErrors(invalid("System search-domain digest is invalid"),
                            std::move(error));
  if (searchDomain.dataflowReference().artifact != dataflow.identity() ||
      searchDomain.fabricReference().artifact != fabric.artifact().identity())
    return invalid("System search domain has foreign D/F owners");
  if (searchDomain.constraintReference() != constraints.reference())
    return invalid("System search domain has a foreign K owner");
  if (constraints.view().dataflowIdentity() != dataflow.identity() ||
      constraints.view().fabricIdentity() != fabric.artifact().identity())
    return invalid("System MappingConstraintSet has foreign D/F owners");
  if (searchDomain.rootThreadLaunches() !=
          constraints.view().rootThreadLaunches() ||
      searchDomain.rootThreadLaunches().empty())
    return invalid("System root launch closure differs between H and K");
  return llvm::Error::success();
}

struct Catalogs final {
  std::vector<FrozenSystemSpatialTargetClass> targetClasses;
  std::vector<::loom::fabric::AccCoreOccurrenceRef> cores;
  std::vector<PnrIndex> coreTargetClasses;
  std::vector<ArtifactRootReference> mappings;
  std::vector<PnrIndex> mappingTargetClasses;
};

llvm::Expected<Catalogs>
buildCatalogs(const ::dataflow::CanonicalDataflowProgramView &dataflow,
              const ::loom::fabric::FabricSystemRootView &system,
              const SystemPnrSearchDomainView &searchDomain,
              const ArtifactStore &store) {
  Catalogs result;
  result.cores.assign(system.artifact().accCoreOccurrences().begin(),
                      system.artifact().accCoreOccurrences().end());
  llvm::sort(result.cores, [](auto lhs, auto rhs) {
    return ::loom::fabric::canonicalFabricBytes(lhs) <
           ::loom::fabric::canonicalFabricBytes(rhs);
  });

  std::vector<FrozenSystemSpatialTargetClass> coreClasses;
  coreClasses.reserve(result.cores.size());
  for (auto core : result.cores) {
    auto target = system.spatialCoreTarget(core);
    if (!target ||
        target->dependencyOrdinal >= system.artifact().importedModules().size())
      return invalid("AccCore has no exact imported SpatialCore target");
    const auto &module =
        system.artifact().importedModules()[target->dependencyOrdinal];
    auto targetClass = targetClassForModule(module);
    if (!targetClass)
      return targetClass.takeError();
    if (targetClass->moduleTemplate != target->target)
      return invalid(
          "AccCore SpatialCore target disagrees with its Module root");
    coreClasses.push_back(std::move(*targetClass));
  }

  for (const SystemSearchBindingDomain &binding : searchDomain.bindings())
    if (std::holds_alternative<::dataflow::RootedGraphLaunchRef>(binding.key))
      for (const SystemSearchAtom &atom : binding.atoms)
        if (atom.domains.compatibleSpatialMappings)
          result.mappings.insert(
              result.mappings.end(),
              atom.domains.compatibleSpatialMappings->begin(),
              atom.domains.compatibleSpatialMappings->end());
  llvm::sort(result.mappings, artifactRootReferenceLess);
  result.mappings.erase(
      std::unique(result.mappings.begin(), result.mappings.end()),
      result.mappings.end());

  std::vector<FrozenSystemSpatialTargetClass> mappingClasses;
  mappingClasses.reserve(result.mappings.size());
  std::set<std::string> attachedTargetClasses;
  for (const auto &targetClass : coreClasses)
    attachedTargetClasses.insert(targetClassKey(targetClass));
  for (const ArtifactRootReference &reference : result.mappings) {
    auto mapping = ::loom::mapping::importSpatialMapping(reference, store);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().dataflowIdentity() != dataflow.identity())
      return invalid("SpatialMapping catalog has a foreign Dataflow owner");
    const ::loom::fabric::FabricArtifactView *module = nullptr;
    for (const auto &candidate : system.artifact().importedModules())
      if (candidate.identity() == mapping->view().fabricIdentity()) {
        module = &candidate;
        break;
      }
    if (!module)
      return invalid("SpatialMapping target Module is not imported by System");
    auto targetClass = targetClassForModule(*module);
    if (!targetClass)
      return targetClass.takeError();
    if (!attachedTargetClasses.count(targetClassKey(*targetClass)))
      return invalid(
          "SpatialMapping target class is not attached to a System AccCore");
    mappingClasses.push_back(std::move(*targetClass));
  }

  result.targetClasses = coreClasses;
  result.targetClasses.insert(result.targetClasses.end(),
                              mappingClasses.begin(), mappingClasses.end());
  llvm::sort(result.targetClasses, [](const auto &lhs, const auto &rhs) {
    return targetClassKey(lhs) < targetClassKey(rhs);
  });
  result.targetClasses.erase(
      std::unique(result.targetClasses.begin(), result.targetClasses.end(),
                  [](const auto &lhs, const auto &rhs) {
                    return targetClassKey(lhs) == targetClassKey(rhs);
                  }),
      result.targetClasses.end());
  std::map<std::string, PnrIndex> classOrdinals;
  for (const auto &[ordinal, targetClass] :
       llvm::enumerate(result.targetClasses)) {
    auto index = checked(catalogIndexContext, ordinal);
    if (!index)
      return index.takeError();
    classOrdinals.emplace(targetClassKey(targetClass), *index);
  }
  for (const auto &targetClass : coreClasses)
    result.coreTargetClasses.push_back(
        classOrdinals.at(targetClassKey(targetClass)));
  for (const auto &targetClass : mappingClasses) {
    auto found = classOrdinals.find(targetClassKey(targetClass));
    if (found == classOrdinals.end())
      return invalid("SpatialMapping target class is absent from the System");
    result.mappingTargetClasses.push_back(found->second);
  }
  return result;
}

struct Decisions final {
  std::vector<FrozenSystemThreadExecutionDecision> threads;
  std::vector<PnrIndex> threadChoices;
  std::vector<FrozenSystemGraphExecutionDecision> graphs;
  std::vector<PnrIndex> graphChoices;
};

llvm::Expected<Decisions>
buildDecisions(const SystemPnrSearchDomainView &searchDomain,
               const Catalogs &catalogs) {
  Decisions result;
  std::map<std::string, PnrIndex> coreOrdinals;
  for (const auto &[ordinal, core] : llvm::enumerate(catalogs.cores)) {
    auto index = checked(catalogIndexContext, ordinal);
    if (!index)
      return index.takeError();
    coreOrdinals.emplace(coreKey(core), *index);
  }
  std::map<ArtifactRootReference, PnrIndex,
           decltype(&artifactRootReferenceLess)>
      mappingOrdinals(&artifactRootReferenceLess);
  for (const auto &[ordinal, mapping] : llvm::enumerate(catalogs.mappings)) {
    auto index = checked(catalogIndexContext, ordinal);
    if (!index)
      return index.takeError();
    mappingOrdinals.emplace(mapping, *index);
  }

  for (const SystemSearchBindingDomain &binding : searchDomain.bindings()) {
    if (!std::holds_alternative<::dataflow::RootThreadLaunchRef>(binding.key))
      continue;
    const auto root = std::get<::dataflow::RootThreadLaunchRef>(binding.key);
    for (const SystemSearchAtom &atom : binding.atoms) {
      if (!atom.domains.compatibleAccCores ||
          atom.domains.compatibleSpatialMappings ||
          atom.domains.compatibleServiceRegions ||
          atom.domains.compatibleTransportEndpoints)
        return invalid("thread atom has an ill-typed H target domain");
      if (atom.domains.compatibleAccCores->empty())
        return infeasible("thread atom has no compatible AccCore");
      auto offset = checked(choiceOffsetContext, result.threadChoices.size());
      auto count =
          checked(choiceCountContext, atom.domains.compatibleAccCores->size());
      auto decision = checked(decisionIndexContext, result.threads.size());
      if (!offset)
        return offset.takeError();
      if (!count)
        return count.takeError();
      if (!decision)
        return decision.takeError();
      for (auto core : *atom.domains.compatibleAccCores) {
        auto found = coreOrdinals.find(coreKey(core));
        if (found == coreOrdinals.end())
          return invalid("thread atom names an AccCore outside F");
        result.threadChoices.push_back(found->second);
      }
      result.threads.push_back({root, atom.cell, *offset, *count, *decision});
    }
  }

  for (const SystemSearchBindingDomain &binding : searchDomain.bindings()) {
    if (!std::holds_alternative<::dataflow::RootedGraphLaunchRef>(binding.key))
      continue;
    const auto launch = std::get<::dataflow::RootedGraphLaunchRef>(binding.key);
    for (const SystemSearchAtom &atom : binding.atoms) {
      if (!atom.domains.compatibleSpatialMappings ||
          atom.domains.compatibleAccCores ||
          atom.domains.compatibleServiceRegions ||
          atom.domains.compatibleTransportEndpoints)
        return invalid("graph atom has an ill-typed H target domain");
      if (atom.domains.compatibleSpatialMappings->empty())
        return infeasible("graph atom has no compatible SpatialMapping");
      auto offset = checked(choiceOffsetContext, result.graphChoices.size());
      auto count = checked(choiceCountContext,
                           atom.domains.compatibleSpatialMappings->size());
      auto decision = checked(decisionIndexContext,
                              result.threads.size() + result.graphs.size());
      if (!offset)
        return offset.takeError();
      if (!count)
        return count.takeError();
      if (!decision)
        return decision.takeError();
      for (const ArtifactRootReference &mapping :
           *atom.domains.compatibleSpatialMappings) {
        auto found = mappingOrdinals.find(mapping);
        if (found == mappingOrdinals.end())
          return invalid("graph atom names a SpatialMapping outside H");
        result.graphChoices.push_back(found->second);
      }
      result.graphs.push_back({launch, atom.cell, *offset, *count, *decision});
    }
  }
  return result;
}

llvm::ArrayRef<PnrIndex> choiceSlice(llvm::ArrayRef<PnrIndex> choices,
                                     PnrIndex offset, PnrIndex count) {
  return choices.slice(offset, count);
}

llvm::Expected<std::unique_ptr<detail::InitializerRelationModel>>
buildRelations(const Catalogs &catalogs, const Decisions &decisions,
               std::vector<PnrIndex> &overlapOffsets,
               std::vector<PnrIndex> &overlaps) {
  std::vector<PnrIndex> choiceCounts;
  choiceCounts.reserve(decisions.threads.size() + decisions.graphs.size());
  for (const auto &thread : decisions.threads)
    choiceCounts.push_back(thread.choiceCount);
  for (const auto &graph : decisions.graphs)
    choiceCounts.push_back(graph.choiceCount);

  std::vector<detail::InitializerRelationInput> relations;
  std::map<std::uint64_t, std::vector<PnrIndex>> threadsByRoot;
  for (const auto &[threadOrdinal, thread] : llvm::enumerate(decisions.threads))
    threadsByRoot[thread.root.entity.value()].push_back(
        static_cast<PnrIndex>(threadOrdinal));
  overlapOffsets.reserve(decisions.graphs.size() + 1);
  overlapOffsets.push_back(0);
  for (const auto &graph : decisions.graphs) {
    std::vector<PnrIndex> intersecting;
    const auto rootThreads =
        threadsByRoot.find(graph.launch.rootThreadLaunch.entity.value());
    if (rootThreads == threadsByRoot.end())
      return invalid("graph atom has no parent thread domain");
    std::optional<std::size_t> exactThread;
    for (PnrIndex threadOrdinal : rootThreads->second) {
      const auto &thread = decisions.threads[threadOrdinal];
      if (thread.root == graph.launch.rootThreadLaunch &&
          thread.cell == graph.cell) {
        exactThread = threadOrdinal;
        break;
      }
    }
    for (PnrIndex threadOrdinal : rootThreads->second) {
      const auto &thread = decisions.threads[threadOrdinal];
      if (thread.root != graph.launch.rootThreadLaunch)
        continue;
      if (exactThread && threadOrdinal != *exactThread)
        continue;
      bool intersects = thread.cell == graph.cell;
      if (!intersects) {
        auto result =
            detail::systemPresburgerCellsIntersect(thread.cell, graph.cell);
        if (!result)
          return result.takeError();
        intersects = *result;
      }
      if (!intersects)
        continue;
      auto threadIndex = checked(decisionIndexContext, threadOrdinal);
      if (!threadIndex)
        return threadIndex.takeError();
      intersecting.push_back(*threadIndex);

      detail::InitializerRelationInput relation;
      relation.kind = detail::InitializerRelationKind::Equal;
      detail::InitializerRelationMemberInput threadMember;
      threadMember.decision = thread.relationDecision;
      for (PnrIndex core : choiceSlice(decisions.threadChoices,
                                       thread.choiceOffset, thread.choiceCount))
        threadMember.projectedValues.push_back(
            catalogs.coreTargetClasses[core]);
      detail::InitializerRelationMemberInput graphMember;
      graphMember.decision = graph.relationDecision;
      for (PnrIndex mapping : choiceSlice(
               decisions.graphChoices, graph.choiceOffset, graph.choiceCount))
        graphMember.projectedValues.push_back(
            catalogs.mappingTargetClasses[mapping]);
      relation.members.push_back(std::move(threadMember));
      relation.members.push_back(std::move(graphMember));
      relations.push_back(std::move(relation));
    }
    if (intersecting.empty())
      return invalid("graph atom does not intersect its parent thread domain");
    overlaps.insert(overlaps.end(), intersecting.begin(), intersecting.end());
    auto offset = checked(overlapOffsetContext, overlaps.size());
    if (!offset)
      return offset.takeError();
    overlapOffsets.push_back(*offset);
  }

  auto model = detail::InitializerRelationModel::create(std::move(choiceCounts),
                                                        std::move(relations));
  if (!model)
    return model.takeError();
  return std::make_unique<detail::InitializerRelationModel>(std::move(*model));
}

} // namespace

FrozenSystemPnrProblem::FrozenSystemPnrProblem(
    ArtifactIdentity dataflowIdentity, ArtifactIdentity fabricIdentity,
    ArtifactIdentity constraintIdentity,
    SystemPnrSearchDomainDigest searchDomainDigest,
    ResolvedPnrConfigView config,
    std::vector<::dataflow::RootThreadLaunchRef> rootThreadLaunches,
    std::vector<FrozenSystemSpatialTargetClass> targetClasses,
    std::vector<::loom::fabric::AccCoreOccurrenceRef> accCores,
    std::vector<PnrIndex> accCoreTargetClasses,
    std::vector<ArtifactRootReference> spatialMappings,
    std::vector<PnrIndex> spatialMappingTargetClasses,
    std::vector<FrozenSystemThreadExecutionDecision> threadDecisions,
    std::vector<PnrIndex> threadChoiceCatalogOrdinals,
    std::vector<FrozenSystemGraphExecutionDecision> graphDecisions,
    std::vector<PnrIndex> graphChoiceCatalogOrdinals,
    std::vector<PnrIndex> graphThreadOverlapOffsets,
    std::vector<PnrIndex> graphThreadOverlaps,
    FrozenEndpointRoutingTopology routingTopology,
    std::vector<FrozenSystemTransferTerminal> serviceTerminals,
    std::vector<PnrIndex> serviceTerminalEndpointChoices,
    std::vector<FrozenSystemServiceLeg> serviceLegs,
    std::vector<PnrIndex> serviceLegSinkTerminals,
    std::unique_ptr<detail::InitializerRelationModel> initializerRelations)
    : dataflowIdentity_(std::move(dataflowIdentity)),
      fabricIdentity_(std::move(fabricIdentity)),
      constraintIdentity_(std::move(constraintIdentity)),
      searchDomainDigest_(std::move(searchDomainDigest)),
      config_(std::move(config)),
      rootThreadLaunches_(std::move(rootThreadLaunches)),
      targetClasses_(std::move(targetClasses)), accCores_(std::move(accCores)),
      accCoreTargetClasses_(std::move(accCoreTargetClasses)),
      spatialMappings_(std::move(spatialMappings)),
      spatialMappingTargetClasses_(std::move(spatialMappingTargetClasses)),
      threadDecisions_(std::move(threadDecisions)),
      threadChoiceCatalogOrdinals_(std::move(threadChoiceCatalogOrdinals)),
      graphDecisions_(std::move(graphDecisions)),
      graphChoiceCatalogOrdinals_(std::move(graphChoiceCatalogOrdinals)),
      graphThreadOverlapOffsets_(std::move(graphThreadOverlapOffsets)),
      graphThreadOverlaps_(std::move(graphThreadOverlaps)),
      routingTopology_(std::move(routingTopology)),
      serviceTerminals_(std::move(serviceTerminals)),
      serviceTerminalEndpointChoices_(
          std::move(serviceTerminalEndpointChoices)),
      serviceLegs_(std::move(serviceLegs)),
      serviceLegSinkTerminals_(std::move(serviceLegSinkTerminals)),
      initializerRelations_(std::move(initializerRelations)) {}

FrozenSystemPnrProblem::~FrozenSystemPnrProblem() = default;

llvm::ArrayRef<PnrIndex>
FrozenSystemPnrProblem::threadChoiceCatalogOrdinals(PnrIndex decision) const {
  assert(decision < threadDecisions_.size());
  const auto &record = threadDecisions_[decision];
  return choiceSlice(threadChoiceCatalogOrdinals_, record.choiceOffset,
                     record.choiceCount);
}

llvm::ArrayRef<PnrIndex>
FrozenSystemPnrProblem::graphChoiceCatalogOrdinals(PnrIndex decision) const {
  assert(decision < graphDecisions_.size());
  const auto &record = graphDecisions_[decision];
  return choiceSlice(graphChoiceCatalogOrdinals_, record.choiceOffset,
                     record.choiceCount);
}

llvm::ArrayRef<PnrIndex> FrozenSystemPnrProblem::serviceTerminalEndpointChoices(
    PnrIndex terminal) const {
  assert(terminal < serviceTerminals_.size());
  const auto &record = serviceTerminals_[terminal];
  return choiceSlice(serviceTerminalEndpointChoices_,
                     record.endpointChoiceOffset, record.endpointChoiceCount);
}

llvm::ArrayRef<PnrIndex>
FrozenSystemPnrProblem::serviceLegSinkTerminals(PnrIndex leg) const {
  assert(leg < serviceLegs_.size());
  const auto &record = serviceLegs_[leg];
  return choiceSlice(serviceLegSinkTerminals_, record.sinkOffset,
                     record.sinkCount);
}

PnrIndex FrozenSystemPnrProblem::accCoreTargetClass(PnrIndex core) const {
  assert(core < accCoreTargetClasses_.size());
  return accCoreTargetClasses_[core];
}

PnrIndex
FrozenSystemPnrProblem::spatialMappingTargetClass(PnrIndex mapping) const {
  assert(mapping < spatialMappingTargetClasses_.size());
  return spatialMappingTargetClasses_[mapping];
}

llvm::Expected<FrozenSystemPnrProblemHandle> loom::pnr::freezeSystemPnrProblem(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemPnrSearchDomainView &searchDomain,
    const ResolvedPnrConfigView &config,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    const ArtifactStore &store) {
  if (llvm::Error error =
          validateInputs(dataflow, fabric, searchDomain, config, constraints))
    return std::move(error);
  auto catalogs = buildCatalogs(dataflow, fabric, searchDomain, store);
  if (!catalogs)
    return catalogs.takeError();
  auto decisions = buildDecisions(searchDomain, *catalogs);
  if (!decisions)
    return decisions.takeError();
  std::vector<PnrIndex> overlapOffsets;
  std::vector<PnrIndex> overlaps;
  auto relations =
      buildRelations(*catalogs, *decisions, overlapOffsets, overlaps);
  if (!relations)
    return relations.takeError();
  auto routing = freezeSystemRouting(dataflow, fabric, searchDomain);
  if (!routing)
    return routing.takeError();

  return FrozenSystemPnrProblemHandle(new FrozenSystemPnrProblem(
      dataflow.identity(), fabric.artifact().identity(),
      constraints.view().identity(), searchDomain.digest(), config,
      std::vector<::dataflow::RootThreadLaunchRef>(
          searchDomain.rootThreadLaunches().begin(),
          searchDomain.rootThreadLaunches().end()),
      std::move(catalogs->targetClasses), std::move(catalogs->cores),
      std::move(catalogs->coreTargetClasses), std::move(catalogs->mappings),
      std::move(catalogs->mappingTargetClasses), std::move(decisions->threads),
      std::move(decisions->threadChoices), std::move(decisions->graphs),
      std::move(decisions->graphChoices), std::move(overlapOffsets),
      std::move(overlaps), std::move(routing->topology),
      std::move(routing->terminals), std::move(routing->endpointChoices),
      std::move(routing->legs), std::move(routing->legSinks),
      std::move(*relations)));
}
