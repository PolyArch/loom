#include "PnR/System/SystemPnrProblem.h"

#include "PnR/InitializerRelationSolver.h"
#include "SystemPnrSearchDomainInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ComponentViewDigest.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
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
constexpr PnrCapacityContext serviceContextIndexContext{
    frozenArtifact, "service_context", "context", PnrCapacityMeasure::Index};

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

struct MergedServiceTerminal final {
  ::loom::mapping::SystemTransferTerminalKey key;
  std::vector<::loom::fabric::FabricTransportEndpointRef> endpoints;
};

struct ServiceLegDraft final {
  ::loom::mapping::CanonicalServiceLegKey key;
  const MergedServiceTerminal *source = nullptr;
  std::vector<const MergedServiceTerminal *> sinks;
};

llvm::Expected<std::uint32_t> operationServiceLegPayloadWidth(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::CanonicalServiceLegKey &leg) {
  const ::dataflow::ContextualActorRef *contextual = nullptr;
  if (const auto *addressed =
          std::get_if<::dataflow::AddressedMemoryActorMemberRef>(&leg.member))
    contextual = &addressed->actor;
  else if (const auto *fence =
               std::get_if<::dataflow::FenceActorMemberRef>(&leg.member))
    contextual = &fence->actor;
  if (!contextual)
    return invalid("operation-service leg has no contextual actor");
  if (llvm::Error error = dataflow.validate(*contextual))
    return std::move(error);
  auto actor = dataflow.resolve(contextual->actor);
  if (!actor)
    return actor.takeError();
  auto service = ::dataflow::semantics::CanonicalService::forActor(actor->op);
  if (!service)
    return service.takeError();
  if (leg.ordinal >= service->legCount())
    return invalid("operation-service leg ordinal is out of range");

  std::uint32_t result = 0;
  for (const auto &value : service->legPayload(leg.ordinal)) {
    auto width = dataflow.transportPayloadBitWidth(value.type);
    if (!width)
      return width.takeError();
    result = std::max(result, *width);
  }
  return result;
}

llvm::Expected<FrozenSystemRoutingData> freezeSystemRouting(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemPnrSearchDomainView &searchDomain,
    llvm::ArrayRef<FrozenSystemServiceContext> serviceContexts,
    llvm::ArrayRef<FrozenSystemGraphExecutionDecision> graphDecisions) {
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

  auto appendTerminal =
      [&](const MergedServiceTerminal &domain) -> llvm::Expected<PnrIndex> {
    auto terminal = checked(serviceTerminalContext, result.terminals.size());
    if (!terminal)
      return terminal.takeError();
    auto offset =
        checked(serviceEndpointChoiceContext, result.endpointChoices.size());
    if (!offset)
      return offset.takeError();
    std::vector<PnrIndex> choices;
    choices.reserve(domain.endpoints.size());
    for (const auto &endpoint : domain.endpoints) {
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

  for (const auto &[serviceOrdinal, service] :
       llvm::enumerate(searchDomain.serviceObligations())) {
    const auto *producer =
        std::get_if<::loom::mapping::TransferObligationFamilyKey>(&service.key);
    std::uint32_t payloadWidthBits = 0;
    if (producer) {
      auto producerKey =
          ::dataflow::encodeDataflowReference(dataflow.identity(), *producer);
      if (!producerKey)
        return producerKey.takeError();
      auto payloadWidth = producerWidths.find(bytesKey(*producerKey));
      if (payloadWidth == producerWidths.end())
        return invalid("H transfer obligation has no Dataflow producer");
      payloadWidthBits = payloadWidth->second;
    }

    std::map<std::string, MergedServiceTerminal> terminals;
    for (const SystemSearchTransferTerminalCompatibility &row :
         service.transferTerminalCompatibility) {
      if (const auto *bound =
              std::get_if<SystemMessageTerminalEndpoint>(&row.boundEndpoint)) {
        if (!producer)
          return invalid("operation-service obligation has a message row");
        if (row.compatibleTransportEndpoints.size() > 1 ||
            (!row.compatibleTransportEndpoints.empty() &&
             row.compatibleTransportEndpoints.front() != bound->endpoint))
          return invalid("message terminal row is not factorized by its exact "
                         "bound endpoint");
      } else if (producer) {
        return invalid("transfer obligation has a memory terminal row");
      }
      auto terminalBytes = ::loom::mapping::encodeSystemTransferTerminalKey(
          dataflow.identity(), row.terminal);
      if (!terminalBytes)
        return terminalBytes.takeError();
      auto [position, inserted] = terminals.try_emplace(
          bytesKey(*terminalBytes), MergedServiceTerminal{row.terminal, {}});
      position->second.endpoints.insert(
          position->second.endpoints.end(),
          row.compatibleTransportEndpoints.begin(),
          row.compatibleTransportEndpoints.end());
    }
    for (auto &[key, terminal] : terminals) {
      llvm::sort(terminal.endpoints, [](const auto &left, const auto &right) {
        return ::loom::fabric::canonicalFabricBytes(left) <
               ::loom::fabric::canonicalFabricBytes(right);
      });
      terminal.endpoints.erase(
          std::unique(terminal.endpoints.begin(), terminal.endpoints.end()),
          terminal.endpoints.end());
    }

    std::map<std::string, ServiceLegDraft> drafts;
    for (const auto &[terminalKey, terminal] : terminals) {
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

      std::uint32_t legPayloadWidthBits = payloadWidthBits;
      if (!producer) {
        auto width = operationServiceLegPayloadWidth(dataflow, draft.key);
        if (!width)
          return width.takeError();
        legPayloadWidthBits = *width;
      }

      std::vector<PnrIndex> contexts;
      if (producer) {
        contexts.push_back(getInvalidPnrIndex());
      } else {
        const ::dataflow::RootedGraphLaunchRef *launch = nullptr;
        if (const auto *addressed =
                std::get_if<::dataflow::AddressedMemoryActorMemberRef>(
                    &draft.key.member))
          launch = &addressed->actor.launch;
        else if (const auto *fence =
                     std::get_if<::dataflow::FenceActorMemberRef>(
                         &draft.key.member))
          launch = &fence->actor.launch;
        if (!launch)
          return invalid("operation-service leg has no graph-backed member");
        for (const auto &[contextOrdinal, context] :
             llvm::enumerate(serviceContexts))
          if (context.service == serviceOrdinal &&
              context.graphDecision < graphDecisions.size() &&
              graphDecisions[context.graphDecision].launch == *launch)
            contexts.push_back(static_cast<PnrIndex>(contextOrdinal));
      }
      if (contexts.empty())
        return invalid("operation-service leg has no execution context");

      for (PnrIndex context : contexts) {
        auto source = appendTerminal(*draft.source);
        if (!source)
          return source.takeError();
        auto sinkOffset =
            checked(serviceLegSinkContext, result.legSinks.size());
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
        if (llvm::Error error = preflightPnrIndexCapacity(
                serviceLegContext, result.legs.size() + 1))
          return std::move(error);
        result.legs.push_back({draft.key, context, *source, *sinkOffset,
                               *sinkCount, legPayloadWidthBits});
      }
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
  std::vector<detail::SpatialCatalogEntry> spatialCatalog;
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
        if (const auto *domain =
                std::get_if<SystemHierarchicalGraphBindingDomain>(&atom.domain))
          result.mappings.insert(result.mappings.end(),
                                 domain->compatibleSpatialMappings.begin(),
                                 domain->compatibleSpatialMappings.end());
  llvm::sort(result.mappings, artifactRootReferenceLess);
  result.mappings.erase(
      std::unique(result.mappings.begin(), result.mappings.end()),
      result.mappings.end());
  auto spatialCatalog =
      detail::importSpatialCatalog(result.mappings, dataflow, system, store);
  if (!spatialCatalog)
    return spatialCatalog.takeError();
  result.spatialCatalog = std::move(*spatialCatalog);

  std::vector<FrozenSystemSpatialTargetClass> mappingClasses;
  mappingClasses.reserve(result.mappings.size());
  std::set<std::string> attachedTargetClasses;
  for (const auto &targetClass : coreClasses)
    attachedTargetClasses.insert(targetClassKey(targetClass));
  for (const detail::SpatialCatalogEntry &entry : result.spatialCatalog) {
    const ::loom::fabric::FabricArtifactView *module = nullptr;
    for (const auto &candidate : system.artifact().importedModules())
      if (candidate.identity() == entry.mapping.view().fabricIdentity()) {
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
      const auto *domain = std::get_if<SystemThreadBindingDomain>(&atom.domain);
      if (!domain)
        return invalid("thread atom has an ill-typed H target domain");
      if (domain->compatibleAccCores.empty())
        return infeasible("thread atom has no compatible AccCore");
      auto offset = checked(choiceOffsetContext, result.threadChoices.size());
      auto count =
          checked(choiceCountContext, domain->compatibleAccCores.size());
      auto decision = checked(decisionIndexContext, result.threads.size());
      if (!offset)
        return offset.takeError();
      if (!count)
        return count.takeError();
      if (!decision)
        return decision.takeError();
      for (auto core : domain->compatibleAccCores) {
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
      const auto *domain =
          std::get_if<SystemHierarchicalGraphBindingDomain>(&atom.domain);
      if (!domain)
        return invalid("graph atom has an ill-typed H target domain");
      if (domain->compatibleSpatialMappings.empty())
        return infeasible("graph atom has no compatible SpatialMapping");
      auto offset = checked(choiceOffsetContext, result.graphChoices.size());
      auto count =
          checked(choiceCountContext, domain->compatibleSpatialMappings.size());
      auto decision = checked(decisionIndexContext,
                              result.threads.size() + result.graphs.size());
      if (!offset)
        return offset.takeError();
      if (!count)
        return count.takeError();
      if (!decision)
        return decision.takeError();
      for (const ArtifactRootReference &mapping :
           domain->compatibleSpatialMappings) {
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

llvm::Expected<std::vector<FrozenSystemServiceContext>>
buildServiceContexts(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                     llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
                     llvm::ArrayRef<SystemSearchServiceDomain> services,
                     const Decisions &decisions,
                     llvm::ArrayRef<PnrIndex> overlapOffsets,
                     llvm::ArrayRef<PnrIndex> overlaps) {
  auto obligations =
      ::loom::mapping::projectSystemServiceObligations(dataflow, roots);
  if (!obligations)
    return obligations.takeError();
  std::vector<FrozenSystemServiceContext> result;
  for (const auto &[serviceOrdinal, service] : llvm::enumerate(services)) {
    if (!std::holds_alternative<
            ::loom::mapping::OperationServiceObligationFamilyKey>(service.key))
      continue;
    const auto serviceKey = service.key;
    const auto projection =
        llvm::find_if(*obligations, [&](const auto &candidate) {
          return candidate.key == serviceKey;
        });
    if (projection == obligations->end())
      return invalid("H operation service has no Dataflow obligation");

    std::vector<::dataflow::RootedGraphLaunchRef> launches;
    for (const auto &member : projection->members) {
      const ::dataflow::RootedGraphLaunchRef *launch = nullptr;
      if (const auto *addressed =
              std::get_if<::dataflow::AddressedMemoryActorMemberRef>(&member))
        launch = &addressed->actor.launch;
      else if (const auto *fence =
                   std::get_if<::dataflow::FenceActorMemberRef>(&member))
        launch = &fence->actor.launch;
      if (!launch)
        return invalid("operation service has a non-graph member");
      if (!llvm::is_contained(launches, *launch))
        launches.push_back(*launch);
    }
    for (const auto &exposure : projection->exposures)
      if (!llvm::is_contained(launches, exposure.launch))
        launches.push_back(exposure.launch);

    for (const auto &launch : launches) {
      std::vector<SystemServiceTargetSubject> subjects;
      for (const auto &member : projection->members) {
        const ::dataflow::RootedGraphLaunchRef *memberLaunch = nullptr;
        if (const auto *addressed =
                std::get_if<::dataflow::AddressedMemoryActorMemberRef>(&member))
          memberLaunch = &addressed->actor.launch;
        else if (const auto *fence =
                     std::get_if<::dataflow::FenceActorMemberRef>(&member))
          memberLaunch = &fence->actor.launch;
        if (memberLaunch && *memberLaunch == launch)
          subjects.push_back(SystemServiceMemberTargetSubject{member});
      }
      for (const auto &exposure : projection->exposures)
        if (exposure.launch == launch)
          subjects.push_back(SystemMemoryExposureTargetSubject{exposure});
      if (subjects.empty())
        return invalid("operation-service context has no target subject");
      bool covered = false;
      for (const auto &[graphOrdinal, graph] :
           llvm::enumerate(decisions.graphs)) {
        if (graph.launch != launch)
          continue;
        if (graphOrdinal + 1 >= overlapOffsets.size())
          return invalid("graph-thread overlap index is incomplete");
        const PnrIndex begin = overlapOffsets[graphOrdinal];
        const PnrIndex end = overlapOffsets[graphOrdinal + 1];
        if (begin > end || end > overlaps.size())
          return invalid("graph-thread overlap range is invalid");
        auto graphIndex = checked(serviceContextIndexContext, graphOrdinal);
        auto serviceIndex = checked(serviceContextIndexContext, serviceOrdinal);
        if (!graphIndex)
          return graphIndex.takeError();
        if (!serviceIndex)
          return serviceIndex.takeError();
        for (PnrIndex thread : overlaps.slice(begin, end - begin)) {
          result.push_back({*serviceIndex, *graphIndex, thread, subjects});
          covered = true;
        }
      }
      if (!covered)
        return invalid("operation-service subject has no execution atom");
    }
  }
  if (llvm::Error error =
          preflightPnrIndexCapacity(serviceContextIndexContext, result.size()))
    return std::move(error);
  return result;
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
    std::vector<SystemSearchServiceDomain> serviceDomains,
    std::vector<FrozenSystemServiceContext> serviceContexts,
    std::vector<FrozenSystemMemoryServiceBinding> memoryServiceBindings,
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
      serviceDomains_(std::move(serviceDomains)),
      serviceContexts_(std::move(serviceContexts)),
      memoryServiceBindings_(std::move(memoryServiceBindings)),
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
  auto serviceContexts = buildServiceContexts(
      dataflow, searchDomain.rootThreadLaunches(),
      searchDomain.serviceObligations(), *decisions, overlapOffsets, overlaps);
  if (!serviceContexts)
    return serviceContexts.takeError();
  auto constraintIndex = detail::buildFrozenConstraintIndex(constraints.view());
  if (!constraintIndex)
    return constraintIndex.takeError();
  auto memoryBindings = detail::projectSystemMemoryServiceBindings(
      dataflow, fabric, searchDomain.rootThreadLaunches(),
      catalogs->spatialCatalog, *constraintIndex);
  if (!memoryBindings)
    return memoryBindings.takeError();
  auto routing = freezeSystemRouting(dataflow, fabric, searchDomain,
                                     *serviceContexts, decisions->graphs);
  if (!routing)
    return routing.takeError();

  std::vector<SystemSearchServiceDomain> serviceDomains(
      searchDomain.serviceObligations().begin(),
      searchDomain.serviceObligations().end());

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
      std::move(serviceDomains), std::move(*serviceContexts),
      std::move(*memoryBindings), std::move(routing->legs),
      std::move(routing->legSinks), std::move(*relations)));
}
