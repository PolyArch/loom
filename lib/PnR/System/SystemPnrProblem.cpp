#include "PnR/System/SystemPnrProblem.h"

#include "../SpatialPhysicalTiming.h"
#include "PnR/InitializerRelationSolver.h"
#include "SystemCapacityProjection.h"
#include "SystemPnrDerivedContextInternal.h"
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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
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
constexpr PnrCapacityContext serviceTerminalOwnerDomainContext{
    frozenArtifact, "service_routing", "terminal_owner_domain",
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

struct FrozenSystemRoutingData final {
  std::vector<FrozenSystemTransferTerminal> terminals;
  std::vector<FrozenSystemTransferTerminalOwnerDomain> ownerDomains;
  std::vector<PnrIndex> endpointChoices;
  std::vector<FrozenSystemServiceLeg> legs;
  std::vector<PnrIndex> legSinks;
};

struct MergedServiceTerminal final {
  ::loom::mapping::SystemTransferTerminalKey key;
  std::vector<::loom::fabric::FabricTransportEndpointRef> boundEndpoints;
  std::vector<::loom::fabric::FabricTransportEndpointRef> endpoints;
};

struct TerminalOwnerDependency final {
  bool fixedHost = false;
  PnrIndex threadDecision = getInvalidPnrIndex();
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

llvm::Expected<TerminalOwnerDependency> terminalOwnerDependency(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::SystemTransferTerminalKey &terminal,
    const FrozenSystemServiceContext &context,
    llvm::ArrayRef<FrozenSystemThreadExecutionDecision> threadDecisions,
    std::optional<PnrIndex> exactOwnerThreadDecision = std::nullopt) {
  const bool source =
      std::holds_alternative<::loom::mapping::SystemTransferSourceTerminalKey>(
          terminal);
  const auto &leg =
      source
          ? std::get<::loom::mapping::SystemTransferSourceTerminalKey>(terminal)
                .leg
          : std::get<::loom::mapping::SystemTransferSinkTerminalKey>(terminal)
                .leg;
  const auto *producer =
      std::get_if<::loom::mapping::TransferObligationFamilyKey>(
          &leg.obligation);
  if (!producer)
    return TerminalOwnerDependency{};

  std::optional<::dataflow::CanonicalSinkTerminalRef> selectedSink;
  if (!source) {
    const auto ordinal =
        std::get<::loom::mapping::SystemTransferSinkTerminalKey>(terminal)
            .sinkOrdinal;
    ::dataflow::StructuralOrdinal cursor = 0;
    if (llvm::Error error = dataflow.pairedSinks(
            *producer,
            [&](const ::dataflow::CanonicalSinkTerminalRef &candidate) {
              if (cursor++ == ordinal) {
                selectedSink = candidate;
              }
            }))
      return std::move(error);
    if (!selectedSink)
      return invalid("message terminal has an out-of-range sink ordinal");
  }

  const auto rootBoundary =
      [&](const ::dataflow::RootThreadBoundaryTransferRef &transfer)
      -> TerminalOwnerDependency {
    const bool completion =
        std::holds_alternative<::dataflow::RootThreadCompletionTransferRef>(
            transfer);
    const bool host = completion != source;
    return host ? TerminalOwnerDependency{true, getInvalidPnrIndex()}
                : TerminalOwnerDependency{false, context.threadDecision};
  };
  if (source) {
    if (const auto *root =
            std::get_if<::dataflow::RootThreadBoundarySourceRef>(producer))
      return rootBoundary(root->transfer);
    if (std::holds_alternative<::dataflow::GraphLaunchBoundarySourceRef>(
            *producer))
      return TerminalOwnerDependency{false, context.threadDecision};
  } else {
    if (const auto *root =
            std::get_if<::dataflow::RootThreadBoundarySinkRef>(&*selectedSink))
      return rootBoundary(root->transfer);
    if (std::holds_alternative<::dataflow::GraphLaunchBoundarySinkRef>(
            *selectedSink))
      return TerminalOwnerDependency{false, context.threadDecision};
  }

  std::optional<::dataflow::RootThreadLaunchRef> ownerRoot;
  if (source) {
    const auto &channel =
        std::get<::dataflow::ChannelProducerTerminalRef>(*producer).producer;
    ownerRoot = std::visit(
        [](const auto &value) {
          if constexpr (std::is_same_v<
                            std::decay_t<decltype(value)>,
                            ::dataflow::GraphStreamOutputProducerRef>)
            return value.launch.rootThreadLaunch;
          else
            return value.launch;
        },
        channel);
  } else {
    const auto &channel =
        std::get<::dataflow::ChannelConsumerTerminalRef>(*selectedSink)
            .consumer;
    ownerRoot = std::visit(
        [](const auto &value) {
          if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                       ::dataflow::GraphStreamInputConsumerRef>)
            return value.launch.rootThreadLaunch;
          else
            return value.launch;
        },
        channel);
  }
  if (exactOwnerThreadDecision) {
    if (*exactOwnerThreadDecision >= threadDecisions.size() ||
        threadDecisions[*exactOwnerThreadDecision].root != *ownerRoot)
      return invalid("message sink applicability has a foreign owner decision");
    return TerminalOwnerDependency{false, *exactOwnerThreadDecision};
  }

  if (context.threadDecision < threadDecisions.size() &&
      threadDecisions[context.threadDecision].root == *ownerRoot)
    return TerminalOwnerDependency{false, context.threadDecision};

  std::optional<PnrIndex> selected;
  for (const auto &[ordinal, decision] : llvm::enumerate(threadDecisions)) {
    if (decision.root != *ownerRoot)
      continue;
    if (selected)
      return invalid("channel consumer owner requires source_map partition "
                     "projection before routing freeze");
    selected = static_cast<PnrIndex>(ordinal);
  }
  if (!selected)
    return invalid("message terminal has no execution-owner decision");
  return TerminalOwnerDependency{false, *selected};
}

llvm::Expected<FrozenSystemRoutingData> freezeSystemRouting(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const FrozenEndpointRoutingTopology &topology,
    const SystemPnrSearchDomainView &searchDomain,
    llvm::ArrayRef<FrozenSystemServiceContext> serviceContexts,
    llvm::ArrayRef<FrozenSystemThreadExecutionDecision> threadDecisions,
    llvm::ArrayRef<PnrIndex> threadChoiceCatalogOrdinals,
    llvm::ArrayRef<::loom::fabric::AccCoreOccurrenceRef> accCores,
    llvm::ArrayRef<FrozenSystemGraphExecutionDecision> graphDecisions) {
  FrozenSystemRoutingData result;
  llvm::StringMap<PnrIndex> endpointOrdinals;
  for (auto [ordinal, endpoint] : llvm::enumerate(topology.endpoints())) {
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
      [&](const MergedServiceTerminal &domain,
          std::optional<TerminalOwnerDependency> ownerDependency)
      -> llvm::Expected<PnrIndex> {
    auto terminal = checked(serviceTerminalContext, result.terminals.size());
    if (!terminal)
      return terminal.takeError();
    auto ownerOffset =
        checked(serviceTerminalOwnerDomainContext, result.ownerDomains.size());
    if (!ownerOffset)
      return ownerOffset.takeError();

    struct OwnerGroup final {
      FrozenSystemTransferTerminalOwner owner;
      std::vector<PnrIndex> choices;
    };
    std::map<std::string, OwnerGroup> groups;
    const auto addOwner = [&](FrozenSystemTransferTerminalOwner owner) {
      const auto ownerBytes = std::visit(
          [](const auto value) {
            return ::loom::fabric::canonicalFabricBytes(value);
          },
          owner);
      groups.try_emplace(bytesKey(ownerBytes),
                         OwnerGroup{std::move(owner), {}});
    };
    if (ownerDependency) {
      if (ownerDependency->fixedHost) {
        if (fabric.artifact().hostCoreOccurrences().size() != 1)
          return invalid(
              "message runtime terminal has no unique HostCore owner");
        addOwner(fabric.artifact().hostCoreOccurrences().front());
      } else {
        if (ownerDependency->threadDecision >= threadDecisions.size())
          return invalid("message terminal has a foreign owner decision");
        const auto &decision = threadDecisions[ownerDependency->threadDecision];
        if (decision.choiceOffset > threadChoiceCatalogOrdinals.size() ||
            decision.choiceCount >
                threadChoiceCatalogOrdinals.size() - decision.choiceOffset)
          return invalid("message terminal owner domain is out of range");
        for (PnrIndex core : threadChoiceCatalogOrdinals.slice(
                 decision.choiceOffset, decision.choiceCount)) {
          if (core >= accCores.size())
            return invalid("message terminal owner names a foreign AccCore");
          addOwner(accCores[core]);
        }
      }
    }
    for (const auto &endpoint : domain.boundEndpoints) {
      auto found = endpointOrdinals.find(
          bytesKey(::loom::fabric::canonicalFabricBytes(endpoint)));
      if (found == endpointOrdinals.end())
        return invalid("H service terminal names an endpoint outside F");
      if (!ownerDependency)
        continue;
      const auto *serviceEndpoint =
          std::get_if<::loom::fabric::SystemServiceEndpointRef>(
              &endpoint.owner.payload);
      if (!serviceEndpoint)
        return invalid("message H row is not a direct service endpoint");
      const auto *owner = fabric.serviceEndpointOwner(*serviceEndpoint);
      if (!owner)
        return invalid("message H row has no service endpoint owner");
      FrozenSystemTransferTerminalOwner executionOwner;
      if (const auto *host = std::get_if<::loom::fabric::HostCoreOccurrenceRef>(
              &owner->owner().payload))
        executionOwner = *host;
      else if (const auto *core =
                   std::get_if<::loom::fabric::AccCoreOccurrenceRef>(
                       &owner->owner().payload))
        executionOwner = *core;
      else
        return invalid("message H row has a nonexecution endpoint owner");
      const auto ownerBytes = std::visit(
          [](const auto value) {
            return ::loom::fabric::canonicalFabricBytes(value);
          },
          executionOwner);
      auto group = groups.find(bytesKey(ownerBytes));
      if (ownerDependency && group == groups.end())
        continue;
      if (group == groups.end())
        group = groups
                    .try_emplace(bytesKey(ownerBytes),
                                 OwnerGroup{executionOwner, {}})
                    .first;
      if (llvm::is_contained(domain.endpoints, endpoint))
        group->second.choices.push_back(found->second);
    }
    for (auto &[key, group] : groups) {
      (void)key;
      llvm::sort(group.choices);
      group.choices.erase(
          std::unique(group.choices.begin(), group.choices.end()),
          group.choices.end());
      auto choiceOffset =
          checked(serviceEndpointChoiceContext, result.endpointChoices.size());
      auto choiceCount =
          checked(serviceEndpointChoiceContext, group.choices.size());
      if (!choiceOffset)
        return choiceOffset.takeError();
      if (!choiceCount)
        return choiceCount.takeError();
      result.endpointChoices.insert(result.endpointChoices.end(),
                                    group.choices.begin(), group.choices.end());
      result.ownerDomains.push_back({group.owner, *choiceOffset, *choiceCount});
    }
    auto ownerCount = checked(serviceTerminalOwnerDomainContext, groups.size());
    if (!ownerCount)
      return ownerCount.takeError();
    result.terminals.push_back(
        {domain.key, ownerDependency && ownerDependency->fixedHost,
         ownerDependency ? ownerDependency->threadDecision
                         : getInvalidPnrIndex(),
         *ownerOffset, *ownerCount});
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
      auto [position, inserted] =
          terminals.try_emplace(bytesKey(*terminalBytes),
                                MergedServiceTerminal{row.terminal, {}, {}});
      if (const auto *bound =
              std::get_if<SystemMessageTerminalEndpoint>(&row.boundEndpoint))
        position->second.boundEndpoints.push_back(bound->endpoint);
      position->second.endpoints.insert(
          position->second.endpoints.end(),
          row.compatibleTransportEndpoints.begin(),
          row.compatibleTransportEndpoints.end());
    }
    for (auto &[key, terminal] : terminals) {
      llvm::sort(terminal.boundEndpoints,
                 [](const auto &left, const auto &right) {
                   return ::loom::fabric::canonicalFabricBytes(left) <
                          ::loom::fabric::canonicalFabricBytes(right);
                 });
      terminal.boundEndpoints.erase(std::unique(terminal.boundEndpoints.begin(),
                                                terminal.boundEndpoints.end()),
                                    terminal.boundEndpoints.end());
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
        for (const auto &[contextOrdinal, context] :
             llvm::enumerate(serviceContexts))
          if (context.service == serviceOrdinal)
            contexts.push_back(static_cast<PnrIndex>(contextOrdinal));
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
        return invalid("service leg has no execution context");

      for (PnrIndex context : contexts) {
        struct ApplicableSink final {
          const MergedServiceTerminal *terminal = nullptr;
          std::optional<PnrIndex> ownerThreadDecision;
        };
        std::vector<ApplicableSink> applicableSinks;
        if (producer &&
            std::holds_alternative<::dataflow::ChannelProducerTerminalRef>(
                *producer)) {
          for (const FrozenSystemApplicableMessageSink &applicable :
               serviceContexts[context].applicableMessageSinks) {
            const auto found = llvm::find_if(
                draft.sinks, [&](const MergedServiceTerminal *sink) {
                  const auto *key = std::get_if<
                      ::loom::mapping::SystemTransferSinkTerminalKey>(
                      &sink->key);
                  return key && key->sinkOrdinal == applicable.sinkOrdinal;
                });
            if (found == draft.sinks.end())
              return invalid(
                  "channel applicability names an absent sink terminal");
            applicableSinks.push_back({*found, applicable.ownerThreadDecision});
          }
        } else {
          for (const MergedServiceTerminal *sink : draft.sinks)
            applicableSinks.push_back({sink, std::nullopt});
        }
        if (applicableSinks.empty())
          continue;

        std::optional<TerminalOwnerDependency> sourceOwner;
        if (producer) {
          auto dependency = terminalOwnerDependency(dataflow, draft.source->key,
                                                    serviceContexts[context],
                                                    threadDecisions);
          if (!dependency)
            return dependency.takeError();
          sourceOwner = *dependency;
        }
        auto source = appendTerminal(*draft.source, sourceOwner);
        if (!source)
          return source.takeError();
        auto sinkOffset =
            checked(serviceLegSinkContext, result.legSinks.size());
        if (!sinkOffset)
          return sinkOffset.takeError();
        for (const ApplicableSink &sink : applicableSinks) {
          std::optional<TerminalOwnerDependency> sinkOwner;
          if (producer) {
            auto dependency = terminalOwnerDependency(
                dataflow, sink.terminal->key, serviceContexts[context],
                threadDecisions, sink.ownerThreadDecision);
            if (!dependency)
              return dependency.takeError();
            sinkOwner = *dependency;
          }
          auto terminal = appendTerminal(*sink.terminal, sinkOwner);
          if (!terminal)
            return terminal.takeError();
          result.legSinks.push_back(*terminal);
        }
        auto sinkCount = checked(serviceLegSinkContext, applicableSinks.size());
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

llvm::StringLiteral pointerLayoutKindName(::loom::PointerLayoutKind kind) {
  switch (kind) {
  case ::loom::PointerLayoutKind::StableIntegral:
    return "stable_integral";
  case ::loom::PointerLayoutKind::NonIntegral:
    return "non_integral";
  case ::loom::PointerLayoutKind::Unstable:
    return "unstable";
  case ::loom::PointerLayoutKind::ExternalState:
    return "external_state";
  }
  llvm_unreachable("unknown pointer layout kind");
}

llvm::Expected<std::string> describeMessagePayload(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::SystemTransferTerminalKey &terminal) {
  const auto &leg =
      std::holds_alternative<::loom::mapping::SystemTransferSourceTerminalKey>(
          terminal)
          ? std::get<::loom::mapping::SystemTransferSourceTerminalKey>(terminal)
                .leg
          : std::get<::loom::mapping::SystemTransferSinkTerminalKey>(terminal)
                .leg;
  const auto *producer =
      std::get_if<::loom::mapping::TransferObligationFamilyKey>(
          &leg.obligation);
  if (!producer)
    return invalid("message terminal belongs to a non-transfer obligation");
  auto resolved = dataflow.resolve(*producer);
  if (!resolved)
    return resolved.takeError();

  std::string description;
  llvm::raw_string_ostream stream(description);
  resolved->payloadType.print(stream);
  if (const auto pointer =
          mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(resolved->payloadType)) {
    auto layout = dataflow.pointerLayout(pointer.getAddressSpace());
    if (!layout)
      return layout.takeError();
    stream << " [pointer_layout={address_space=" << layout->addressSpace
           << ", representation_bits=" << layout->representationBits
           << ", address_bits=" << layout->addressBits
           << ", kind=" << pointerLayoutKindName(layout->kind) << "}]";
  }
  stream.flush();
  return description;
}

llvm::Error validateStaticMessageTerminalSupport(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const FrozenSystemRoutingData &routing,
    llvm::ArrayRef<FrozenSystemThreadExecutionDecision> threadDecisions,
    llvm::ArrayRef<PnrIndex> threadChoiceCatalogOrdinals,
    llvm::ArrayRef<::loom::fabric::AccCoreOccurrenceRef> accCores) {
  std::vector<std::vector<std::uint8_t>> supportedChoices;
  supportedChoices.reserve(threadDecisions.size());
  for (const FrozenSystemThreadExecutionDecision &decision : threadDecisions)
    supportedChoices.emplace_back(decision.choiceCount, 1);
  std::vector<std::set<std::string>> constrainingPayloads(
      threadDecisions.size());

  std::vector<std::uint8_t> referenced(routing.terminals.size(), 0);
  for (const FrozenSystemServiceLeg &leg : routing.legs) {
    if (!std::holds_alternative<::loom::mapping::TransferObligationFamilyKey>(
            leg.key.obligation))
      continue;
    if (leg.sourceTerminal >= referenced.size() ||
        leg.sinkOffset > routing.legSinks.size() ||
        leg.sinkCount > routing.legSinks.size() - leg.sinkOffset)
      return invalid("message service leg has an invalid terminal range");
    referenced[leg.sourceTerminal] = 1;
    for (PnrIndex terminal : llvm::ArrayRef(routing.legSinks)
                                 .slice(leg.sinkOffset, leg.sinkCount)) {
      if (terminal >= referenced.size())
        return invalid("message service leg names a foreign sink terminal");
      referenced[terminal] = 1;
    }
  }

  for (const auto &[terminalOrdinal, terminal] :
       llvm::enumerate(routing.terminals)) {
    if (!referenced[terminalOrdinal])
      continue;
    auto payload = describeMessagePayload(dataflow, terminal.key);
    if (!payload)
      return payload.takeError();
    if (terminal.ownerDomainOffset > routing.ownerDomains.size() ||
        terminal.ownerDomainCount >
            routing.ownerDomains.size() - terminal.ownerDomainOffset)
      return invalid("message terminal has an invalid owner-domain range");
    const auto ownerDomains =
        llvm::ArrayRef(routing.ownerDomains)
            .slice(terminal.ownerDomainOffset, terminal.ownerDomainCount);
    const auto hasEndpoint = [&](const auto &owner) {
      const auto found = llvm::find_if(ownerDomains, [&](const auto &domain) {
        return domain.owner == owner;
      });
      return found != ownerDomains.end() && found->endpointChoiceCount != 0;
    };
    if (terminal.fixedHostOwner) {
      if (!llvm::any_of(ownerDomains, [&](const auto &domain) {
            return std::holds_alternative<
                       ::loom::fabric::HostCoreOccurrenceRef>(domain.owner) &&
                   domain.endpointChoiceCount != 0;
          }))
        return infeasible(llvm::Twine("message terminal payload ") + *payload +
                          " has no compatible HostCore endpoint");
      continue;
    }
    if (terminal.ownerThreadDecision >= threadDecisions.size())
      return invalid("message terminal has a foreign owner decision");
    const auto &decision = threadDecisions[terminal.ownerThreadDecision];
    if (decision.choiceOffset > threadChoiceCatalogOrdinals.size() ||
        decision.choiceCount >
            threadChoiceCatalogOrdinals.size() - decision.choiceOffset)
      return invalid("message terminal owner choice range is invalid");
    auto support =
        llvm::MutableArrayRef(supportedChoices[terminal.ownerThreadDecision]);
    for (PnrIndex choice = 0; choice < decision.choiceCount; ++choice) {
      const PnrIndex core =
          threadChoiceCatalogOrdinals[decision.choiceOffset + choice];
      if (core >= accCores.size())
        return invalid("message terminal owner choice names a foreign core");
      const bool admitted =
          hasEndpoint(FrozenSystemTransferTerminalOwner{accCores[core]});
      support[choice] &= admitted;
      if (!admitted)
        constrainingPayloads[terminal.ownerThreadDecision].insert(*payload);
    }
  }

  for (const auto &[decision, support] : llvm::enumerate(supportedChoices))
    if (!support.empty() &&
        llvm::none_of(support, [](std::uint8_t value) { return value != 0; })) {
      std::string diagnostic;
      llvm::raw_string_ostream stream(diagnostic);
      stream << "message terminal capability support eliminates every AccCore "
                "choice for thread decision "
             << decision << "; payloads constraining AccCore choices: ";
      llvm::interleaveComma(constrainingPayloads[decision], stream);
      stream.flush();
      return infeasible(diagnostic);
    }
  return llvm::Error::success();
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
  std::shared_ptr<const std::vector<FrozenSystemSpatialTargetClass>>
      targetClasses;
  std::shared_ptr<const std::vector<::loom::fabric::AccCoreOccurrenceRef>>
      cores;
  std::shared_ptr<const std::vector<PnrIndex>> coreTargetClasses;
  std::vector<ArtifactRootReference> mappings;
  std::vector<PnrIndex> mappingTargetClasses;
  std::shared_ptr<const std::vector<detail::SpatialCatalogEntry>>
      spatialCatalog;
};

std::vector<ArtifactRootReference> activeSpatialMappings(
    const SystemPnrSearchDomainView &searchDomain,
    const ::loom::mapping::SystemMappingConstraintSetView &constraints) {
  std::vector<ArtifactRootReference> result(
      constraints.spatialMappingReferences().begin(),
      constraints.spatialMappingReferences().end());
  for (const SystemSearchBindingDomain &binding : searchDomain.bindings())
    if (std::holds_alternative<::dataflow::RootedGraphLaunchRef>(binding.key))
      for (const SystemSearchAtom &atom : binding.atoms)
        if (const auto *domain =
                std::get_if<SystemHierarchicalGraphBindingDomain>(&atom.domain))
          result.insert(result.end(), domain->compatibleSpatialMappings.begin(),
                        domain->compatibleSpatialMappings.end());
  llvm::sort(result, artifactRootReferenceLess);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

llvm::Expected<Catalogs>
buildCatalogs(const detail::SystemStaticContextStorage &staticContext,
              const detail::SystemActiveContextStorage &activeContext,
              const SystemPnrSearchDomainView &searchDomain) {
  Catalogs result;
  result.targetClasses = staticContext.targetClasses;
  result.cores = staticContext.accCores;
  result.coreTargetClasses = staticContext.accCoreTargetClasses;
  result.mappings = activeContext.spatialMappings;
  result.mappingTargetClasses.assign(
      activeContext.spatialMappingTargetClasses->begin(),
      activeContext.spatialMappingTargetClasses->end());
  result.spatialCatalog = activeContext.spatialCatalog;

  std::vector<ArtifactRootReference> searchMappings;
  for (const SystemSearchBindingDomain &binding : searchDomain.bindings())
    if (std::holds_alternative<::dataflow::RootedGraphLaunchRef>(binding.key))
      for (const SystemSearchAtom &atom : binding.atoms)
        if (const auto *domain =
                std::get_if<SystemHierarchicalGraphBindingDomain>(&atom.domain))
          searchMappings.insert(searchMappings.end(),
                                domain->compatibleSpatialMappings.begin(),
                                domain->compatibleSpatialMappings.end());
  llvm::sort(searchMappings, artifactRootReferenceLess);
  searchMappings.erase(
      std::unique(searchMappings.begin(), searchMappings.end()),
      searchMappings.end());
  for (const ArtifactRootReference &reference : searchMappings)
    if (!std::binary_search(result.mappings.begin(), result.mappings.end(),
                            reference, artifactRootReferenceLess))
      return invalid("System search domain names a SpatialMapping outside its "
                     "active context");
  return result;
}

struct Decisions final {
  std::vector<FrozenSystemThreadExecutionDecision> threads;
  std::vector<PnrIndex> threadChoices;
  std::vector<FrozenSystemGraphExecutionDecision> graphs;
  std::vector<PnrIndex> graphChoices;
};

llvm::Expected<Decisions>
buildDecisions(const ::dataflow::CanonicalDataflowProgramView &dataflow,
               const SystemPnrSearchDomainView &searchDomain,
               const Catalogs &catalogs) {
  Decisions result;
  std::map<std::string, PnrIndex> coreOrdinals;
  for (const auto &[ordinal, core] : llvm::enumerate(*catalogs.cores)) {
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
    auto logicalDomain = dataflow.projectRootThreadLogicalDomain(root);
    if (!logicalDomain)
      return logicalDomain.takeError();
    const auto relationKind =
        logicalDomain->kind == ::dataflow::ThreadDomainKind::DynamicWork
            ? ::mapping::SystemBindingRelationKind::StableKeyLookup
            : ::mapping::SystemBindingRelationKind::PresburgerPartition;
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
      result.threads.push_back(
          {root, atom.cell, *offset, *count, *decision, relationKind});
    }
  }

  for (const SystemSearchBindingDomain &binding : searchDomain.bindings()) {
    if (!std::holds_alternative<::dataflow::RootedGraphLaunchRef>(binding.key))
      continue;
    const auto launch = std::get<::dataflow::RootedGraphLaunchRef>(binding.key);
    auto logicalDomain =
        dataflow.projectRootThreadLogicalDomain(launch.rootThreadLaunch);
    if (!logicalDomain)
      return logicalDomain.takeError();
    const auto relationKind =
        logicalDomain->kind == ::dataflow::ThreadDomainKind::DynamicWork
            ? ::mapping::SystemBindingRelationKind::StableKeyLookup
            : ::mapping::SystemBindingRelationKind::PresburgerPartition;
    for (const SystemSearchAtom &atom : binding.atoms) {
      const auto *domain =
          std::get_if<SystemHierarchicalGraphBindingDomain>(&atom.domain);
      if (!domain)
        return invalid("graph atom has an ill-typed H target domain");
      if (domain->compatibleSpatialMappings.empty()) {
        auto graph = dataflow.resolve(launch);
        if (!graph)
          return graph.takeError();
        return infeasible(
            "graph atom has no compatible SpatialMapping for root launch " +
            llvm::Twine(launch.rootThreadLaunch.entity.value()) +
            ", static graph launch " +
            llvm::Twine(launch.staticGraphLaunch.entity.value()) + ", graph " +
            llvm::Twine(graph->entity.value()));
      }
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
      result.graphs.push_back(
          {launch, atom.cell, *offset, *count, *decision, relationKind});
    }
  }
  return result;
}

struct GraphChoicePressures final {
  std::vector<std::uint64_t> schedule;
  std::vector<std::uint64_t> operandIngress;
};

llvm::Expected<GraphChoicePressures> buildGraphChoicePressures(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const Catalogs &catalogs, const Decisions &decisions) {
  if (catalogs.spatialCatalog->size() != catalogs.mappings.size())
    return invalid("SpatialMapping pressure catalog has the wrong width");
  GraphChoicePressures result;
  result.schedule.reserve(decisions.graphChoices.size());
  result.operandIngress.reserve(decisions.graphChoices.size());
  for (const FrozenSystemGraphExecutionDecision &decision : decisions.graphs) {
    auto graph = dataflow.resolve(decision.launch);
    if (!graph)
      return graph.takeError();
    for (PnrIndex mapping :
         llvm::ArrayRef(decisions.graphChoices)
             .slice(decision.choiceOffset, decision.choiceCount)) {
      if (mapping >= catalogs.spatialCatalog->size())
        return invalid("graph choice pressure names a foreign mapping");
      const detail::SpatialCatalogEntry &entry =
          (*catalogs.spatialCatalog)[mapping];
      if (entry.covers.size() != entry.graphStaticSchedulePressures.size() ||
          entry.covers.size() !=
              entry.graphSharedOperandIngressPressures.size())
        return invalid("SpatialMapping graph pressure has the wrong width");
      const auto covered = llvm::find(entry.covers, *graph);
      if (covered == entry.covers.end())
        return invalid("graph choice pressure is absent from its mapping");
      const std::size_t graphOrdinal =
          static_cast<std::size_t>(covered - entry.covers.begin());
      result.schedule.push_back(
          entry.graphStaticSchedulePressures[graphOrdinal]);
      result.operandIngress.push_back(
          entry.graphSharedOperandIngressPressures[graphOrdinal]);
    }
  }
  if (result.schedule.size() != decisions.graphChoices.size() ||
      result.operandIngress.size() != decisions.graphChoices.size())
    return invalid("graph choice pressure projection is incomplete");
  return result;
}

llvm::Expected<std::vector<
    std::shared_ptr<const detail::FrozenSpatialRecurrenceTimingDemand>>>
buildGraphChoiceRecurrenceDemands(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const Catalogs &catalogs, const Decisions &decisions) {
  if (catalogs.spatialCatalog->size() != catalogs.mappings.size())
    return invalid("SpatialMapping recurrence catalog has the wrong width");
  std::vector<
      std::shared_ptr<const detail::FrozenSpatialRecurrenceTimingDemand>>
      result;
  result.reserve(decisions.graphChoices.size());
  for (const FrozenSystemGraphExecutionDecision &decision : decisions.graphs) {
    auto graph = dataflow.resolve(decision.launch);
    if (!graph)
      return graph.takeError();
    for (PnrIndex mapping :
         llvm::ArrayRef(decisions.graphChoices)
             .slice(decision.choiceOffset, decision.choiceCount)) {
      if (mapping >= catalogs.spatialCatalog->size())
        return invalid("graph choice recurrence names a foreign mapping");
      const detail::SpatialCatalogEntry &entry =
          (*catalogs.spatialCatalog)[mapping];
      if (entry.covers.size() != entry.graphRecurrenceDemands.size())
        return invalid("SpatialMapping graph recurrence has the wrong width");
      const auto covered = llvm::find(entry.covers, *graph);
      if (covered == entry.covers.end())
        return invalid("graph choice recurrence is absent from its mapping");
      const auto &demand =
          entry.graphRecurrenceDemands[static_cast<std::size_t>(
              covered - entry.covers.begin())];
      if (!demand)
        return invalid("SpatialMapping graph recurrence demand is null");
      result.push_back(demand);
    }
  }
  if (result.size() != decisions.graphChoices.size())
    return invalid("graph choice recurrence demand is incomplete");
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
            (*catalogs.coreTargetClasses)[core]);
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

llvm::Expected<::loom::mapping::SystemPresburgerCell>
serviceContextCell(const Decisions &decisions, PnrIndex graphDecision,
                   PnrIndex threadDecision) {
  if (threadDecision >= decisions.threads.size())
    return invalid("service context has an invalid thread decision");
  if (graphDecision == getInvalidPnrIndex())
    return decisions.threads[threadDecision].cell;
  if (graphDecision >= decisions.graphs.size())
    return invalid("service context has an invalid graph decision");
  auto intersection = ::loom::mapping::intersectSystemPresburgerCells(
      decisions.threads[threadDecision].cell,
      decisions.graphs[graphDecision].cell);
  if (!intersection)
    return intersection.takeError();
  if (!*intersection)
    return invalid("service context execution cells do not intersect");
  return std::move(**intersection);
}

llvm::Expected<::loom::mapping::SystemPresburgerCell>
liftImageSymbols(::loom::mapping::SystemPresburgerCell image,
                 std::uint32_t symbolCount) {
  if (image.symbolCount == symbolCount)
    return image;
  if (image.symbolCount != 0)
    return invalid("channel source_map image has a foreign symbol signature");
  const std::size_t insertion = image.dimensionCount;
  for (auto &row : image.equalities)
    row.insert(row.begin() + insertion, symbolCount, 0);
  for (auto &row : image.inequalities)
    row.insert(row.begin() + insertion, symbolCount, 0);
  image.symbolCount = symbolCount;
  return ::loom::mapping::canonicalizeSystemPresburgerCell(image);
}

::dataflow::RootThreadLaunchRef
channelConsumerRoot(const ::dataflow::ChannelConsumerRef &consumer) {
  return std::visit(
      [](const auto &value) {
        using Consumer = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Consumer,
                                     ::dataflow::GraphStreamInputConsumerRef>)
          return value.launch.rootThreadLaunch;
        else
          return value.launch;
      },
      consumer);
}

llvm::Expected<std::vector<FrozenSystemServiceContext>>
partitionChannelServiceContext(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::dataflow::ChannelProducerTerminalRef &producer,
    FrozenSystemServiceContext base, const Decisions &decisions) {
  auto consumers = dataflow.channelConsumers(producer.producer);
  if (!consumers)
    return consumers.takeError();
  if (base.cells.size() != 1)
    return invalid(
        "channel producer context must start from one execution cell");

  std::vector<FrozenSystemServiceContext> partitions{std::move(base)};
  for (const auto &[sinkOrdinal, consumer] : llvm::enumerate(*consumers)) {
    const auto consumerRoot = channelConsumerRoot(consumer.consumer);
    bool covered = false;
    for (const auto &[ownerOrdinal, owner] :
         llvm::enumerate(decisions.threads)) {
      if (owner.root != consumerRoot)
        continue;
      covered = true;
      ::loom::mapping::SystemPresburgerCell image;
      if (consumer.sourceMap) {
        auto projected = ::loom::mapping::imageSystemPresburgerCell(
            owner.cell, *consumer.sourceMap);
        if (!projected)
          return projected.takeError();
        auto lifted =
            liftImageSymbols(std::move(*projected),
                             partitions.front().cells.front().symbolCount);
        if (!lifted)
          return lifted.takeError();
        image = std::move(*lifted);
      } else {
        if (owner.cell.dimensionCount != 0 ||
            partitions.front().cells.front().dimensionCount != 0)
          return invalid("ranked direct channel receive has no source_map");
        image = partitions.front().cells.front();
      }

      std::vector<FrozenSystemServiceContext> refined;
      for (FrozenSystemServiceContext &partition : partitions) {
        auto split = ::loom::mapping::splitSystemPresburgerSet(
            partition.cells, llvm::ArrayRef(image));
        if (!split)
          return split.takeError();
        if (!split->outside.empty()) {
          FrozenSystemServiceContext outside = partition;
          outside.cells = std::move(split->outside);
          refined.push_back(std::move(outside));
        }
        if (!split->inside.empty()) {
          partition.cells = std::move(split->inside);
          FrozenSystemApplicableMessageSink applicable{
              static_cast<::dataflow::StructuralOrdinal>(sinkOrdinal),
              static_cast<PnrIndex>(ownerOrdinal)};
          if (!llvm::is_contained(partition.applicableMessageSinks, applicable))
            partition.applicableMessageSinks.push_back(applicable);
          llvm::sort(partition.applicableMessageSinks, [](const auto &lhs,
                                                          const auto &rhs) {
            return std::tie(lhs.sinkOrdinal, lhs.ownerThreadDecision) <
                   std::tie(rhs.sinkOrdinal, rhs.ownerThreadDecision);
          });
          refined.push_back(std::move(partition));
        }
      }
      partitions = std::move(refined);
    }
    if (!covered)
      return invalid("channel consumer has no execution-owner decision");
  }

  std::map<std::vector<std::pair<std::uint64_t, PnrIndex>>, std::size_t>
      groupOrdinals;
  std::vector<FrozenSystemServiceContext> grouped;
  for (FrozenSystemServiceContext &partition : partitions) {
    std::vector<std::pair<std::uint64_t, PnrIndex>> key;
    key.reserve(partition.applicableMessageSinks.size());
    for (const auto &sink : partition.applicableMessageSinks)
      key.emplace_back(sink.sinkOrdinal, sink.ownerThreadDecision);
    auto [found, inserted] = groupOrdinals.try_emplace(key, grouped.size());
    if (inserted) {
      grouped.push_back(std::move(partition));
      continue;
    }
    auto &cells = grouped[found->second].cells;
    cells.insert(cells.end(), std::make_move_iterator(partition.cells.begin()),
                 std::make_move_iterator(partition.cells.end()));
  }
  return grouped;
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
    if (const auto *producer =
            std::get_if<::loom::mapping::TransferObligationFamilyKey>(
                &service.key)) {
      auto serviceIndex = checked(serviceContextIndexContext, serviceOrdinal);
      if (!serviceIndex)
        return serviceIndex.takeError();
      const std::vector<SystemServiceTargetSubject> subjects = {
          SystemServiceMemberTargetSubject{::dataflow::ServiceMemberRef(
              ::dataflow::MessageTransferMemberRef{})}};
      const ::dataflow::RootThreadLaunchRef *threadRoot = nullptr;
      const ::dataflow::RootedGraphLaunchRef *graphLaunch = nullptr;
      if (const auto *root =
              std::get_if<::dataflow::RootThreadBoundarySourceRef>(producer))
        threadRoot = &std::visit(
            [](const auto &value) -> const ::dataflow::RootThreadLaunchRef & {
              return value.launch;
            },
            root->transfer);
      else if (const auto *graph =
                   std::get_if<::dataflow::GraphLaunchBoundarySourceRef>(
                       producer))
        graphLaunch = &std::visit(
            [](const auto &value) -> const ::dataflow::RootedGraphLaunchRef & {
              return value.launch;
            },
            graph->transfer);
      else {
        const auto &channel =
            std::get<::dataflow::ChannelProducerTerminalRef>(*producer)
                .producer;
        if (const auto *graph =
                std::get_if<::dataflow::GraphStreamOutputProducerRef>(&channel))
          graphLaunch = &graph->launch;
        else
          threadRoot =
              &std::get<::dataflow::ThreadChannelSendSiteRef>(channel).launch;
      }

      bool covered = false;
      if (threadRoot) {
        for (const auto &[threadOrdinal, thread] :
             llvm::enumerate(decisions.threads)) {
          if (thread.root != *threadRoot)
            continue;
          auto cell = serviceContextCell(decisions, getInvalidPnrIndex(),
                                         static_cast<PnrIndex>(threadOrdinal));
          if (!cell)
            return cell.takeError();
          FrozenSystemServiceContext context{
              *serviceIndex,
              getInvalidPnrIndex(),
              static_cast<PnrIndex>(threadOrdinal),
              {std::move(*cell)},
              subjects,
              {}};
          if (const auto *channel =
                  std::get_if<::dataflow::ChannelProducerTerminalRef>(
                      producer)) {
            auto partitions = partitionChannelServiceContext(
                dataflow, *channel, std::move(context), decisions);
            if (!partitions)
              return partitions.takeError();
            result.insert(result.end(),
                          std::make_move_iterator(partitions->begin()),
                          std::make_move_iterator(partitions->end()));
          } else {
            result.push_back(std::move(context));
          }
          covered = true;
        }
      } else if (graphLaunch) {
        for (const auto &[graphOrdinal, graph] :
             llvm::enumerate(decisions.graphs)) {
          if (graph.launch != *graphLaunch)
            continue;
          if (graphOrdinal + 1 >= overlapOffsets.size())
            return invalid("message graph-thread overlap index is incomplete");
          const PnrIndex begin = overlapOffsets[graphOrdinal];
          const PnrIndex end = overlapOffsets[graphOrdinal + 1];
          if (begin > end || end > overlaps.size())
            return invalid("message graph-thread overlap range is invalid");
          for (PnrIndex thread : overlaps.slice(begin, end - begin)) {
            auto cell = serviceContextCell(
                decisions, static_cast<PnrIndex>(graphOrdinal), thread);
            if (!cell)
              return cell.takeError();
            FrozenSystemServiceContext context{
                *serviceIndex, static_cast<PnrIndex>(graphOrdinal),
                thread,        {std::move(*cell)},
                subjects,      {}};
            if (const auto *channel =
                    std::get_if<::dataflow::ChannelProducerTerminalRef>(
                        producer)) {
              auto partitions = partitionChannelServiceContext(
                  dataflow, *channel, std::move(context), decisions);
              if (!partitions)
                return partitions.takeError();
              result.insert(result.end(),
                            std::make_move_iterator(partitions->begin()),
                            std::make_move_iterator(partitions->end()));
            } else {
              result.push_back(std::move(context));
            }
            covered = true;
          }
        }
      }
      if (!covered)
        return invalid("message service has no producer execution atom");
      continue;
    }
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
          auto cell = serviceContextCell(decisions, *graphIndex, thread);
          if (!cell)
            return cell.takeError();
          result.push_back({*serviceIndex,
                            *graphIndex,
                            thread,
                            {std::move(*cell)},
                            subjects,
                            {}});
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

llvm::Expected<FrozenSystemPnrProblemHandle> loom::pnr::freezeSystemPnrProblem(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<::loom::fabric::FabricPhysicalTimingProfileView>
        physicalTimingProfiles,
    const SystemPnrSearchDomainView &searchDomain,
    const ResolvedPnrConfigView &config,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    const ArtifactStore &store, const SystemStaticContext *staticContext,
    const SystemActiveContext *activeContext) {
  if (llvm::Error error =
          validateInputs(dataflow, fabric, searchDomain, config, constraints))
    return std::move(error);
  std::optional<SystemStaticContext> ownedStaticContext;
  if (!staticContext) {
    auto built = buildSystemStaticContext(fabric);
    if (!built)
      return built.takeError();
    ownedStaticContext.emplace(std::move(*built));
    staticContext = &*ownedStaticContext;
  } else if (llvm::Error error =
                 revalidateSystemStaticContext(*staticContext, fabric)) {
    return std::move(error);
  }
  const detail::SystemStaticContextStorage &staticStorage =
      detail::systemStaticContextStorage(*staticContext);
  std::optional<SystemActiveContext> ownedActiveContext;
  if (!activeContext) {
    auto mappings = activeSpatialMappings(searchDomain, constraints.view());
    auto built = buildSystemActiveContext(*staticContext, dataflow, fabric,
                                          physicalTimingProfiles, constraints,
                                          mappings, store);
    if (!built)
      return built.takeError();
    ownedActiveContext.emplace(std::move(*built));
    activeContext = &*ownedActiveContext;
  } else if (llvm::Error error = revalidateSystemActiveContext(
                 *activeContext, *staticContext, dataflow, fabric,
                 physicalTimingProfiles, constraints,
                 activeContext->spatialMappings())) {
    return std::move(error);
  }
  const detail::SystemActiveContextStorage &activeStorage =
      detail::systemActiveContextStorage(*activeContext);
  auto objectiveProgram = MappingObjectiveProgram::get(
      config.selectedObjectiveCatalogs(), config.policy().objectiveSelection);
  if (!objectiveProgram)
    return objectiveProgram.takeError();
  auto catalogs = buildCatalogs(staticStorage, activeStorage, searchDomain);
  if (!catalogs)
    return catalogs.takeError();
  auto decisions = buildDecisions(dataflow, searchDomain, *catalogs);
  if (!decisions)
    return decisions.takeError();
  auto graphChoicePressures =
      buildGraphChoicePressures(dataflow, *catalogs, *decisions);
  if (!graphChoicePressures)
    return graphChoicePressures.takeError();
  auto graphChoiceRecurrenceDemands =
      buildGraphChoiceRecurrenceDemands(dataflow, *catalogs, *decisions);
  if (!graphChoiceRecurrenceDemands)
    return graphChoiceRecurrenceDemands.takeError();
  std::vector<::dataflow::GraphRef> coveredGraphs;
  std::set<std::uint64_t> coveredGraphOrdinals;
  coveredGraphs.reserve(decisions->graphs.size());
  for (const FrozenSystemGraphExecutionDecision &decision : decisions->graphs) {
    auto graph = dataflow.resolve(decision.launch);
    if (!graph)
      return graph.takeError();
    if (coveredGraphOrdinals.insert(graph->entity.value()).second)
      coveredGraphs.push_back(*graph);
  }
  auto progressBasis = ::loom::mapping::deriveMappingDataflowProgressBasis(
      dataflow, coveredGraphs);
  if (!progressBasis)
    return progressBasis.takeError();
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
      *catalogs->spatialCatalog, *constraintIndex);
  if (!memoryBindings)
    return memoryBindings.takeError();
  auto routing = freezeSystemRouting(
      dataflow, fabric, *staticStorage.routingTopology, searchDomain,
      *serviceContexts, decisions->threads, decisions->threadChoices,
      *catalogs->cores, decisions->graphs);
  if (!routing)
    return routing.takeError();
  if (llvm::Error error = validateStaticMessageTerminalSupport(
          dataflow, *routing, decisions->threads, decisions->threadChoices,
          *catalogs->cores))
    return std::move(error);
  auto capacityModel = detail::buildSystemCapacityModel(
      dataflow, fabric, *catalogs->cores, *catalogs->coreTargetClasses,
      catalogs->mappingTargetClasses, *catalogs->spatialCatalog,
      decisions->graphs, *staticStorage.instructionUsePatterns, *memoryBindings,
      *staticStorage.consistencyUsePatterns, *serviceContexts, routing->legs,
      *staticStorage.routingTopology);
  if (!capacityModel)
    return capacityModel.takeError();

  std::vector<SystemSearchServiceDomain> serviceDomains(
      searchDomain.serviceObligations().begin(),
      searchDomain.serviceObligations().end());

  std::vector<std::uint64_t> spatialMappingWorstRouteArrivalDelayQuanta;
  std::vector<std::uint64_t> spatialMappingTotalRouteNegativeSlackQuanta;
  std::vector<ComponentViewDigest::Storage>
      spatialMappingPhysicalTimingProfileDigests;
  std::vector<::loom::fabric::FabricPhysicalTimingProfileKind>
      spatialMappingPhysicalTimingProfileKinds;
  spatialMappingWorstRouteArrivalDelayQuanta.reserve(
      catalogs->spatialCatalog->size());
  spatialMappingTotalRouteNegativeSlackQuanta.reserve(
      catalogs->spatialCatalog->size());
  spatialMappingPhysicalTimingProfileDigests.reserve(
      catalogs->spatialCatalog->size());
  spatialMappingPhysicalTimingProfileKinds.reserve(
      catalogs->spatialCatalog->size());
  for (const detail::SpatialCatalogEntry &entry : *catalogs->spatialCatalog) {
    spatialMappingWorstRouteArrivalDelayQuanta.push_back(
        entry.worstRouteArrivalDelayQuanta);
    spatialMappingTotalRouteNegativeSlackQuanta.push_back(
        entry.totalRouteNegativeSlackQuanta);
    spatialMappingPhysicalTimingProfileDigests.push_back(
        entry.physicalTimingProfileDigest);
    spatialMappingPhysicalTimingProfileKinds.push_back(
        entry.physicalTimingProfileKind);
  }

  return FrozenSystemPnrProblemHandle(new FrozenSystemPnrProblem(
      dataflow.identity(), fabric.artifact().identity(),
      constraints.view().identity(), searchDomain.digest(), config,
      std::move(*objectiveProgram), deriveDeterministicWorkBudgetView(config),
      *progressBasis,
      std::vector<::dataflow::RootThreadLaunchRef>(
          searchDomain.rootThreadLaunches().begin(),
          searchDomain.rootThreadLaunches().end()),
      catalogs->targetClasses, catalogs->cores, catalogs->coreTargetClasses,
      std::move(catalogs->mappings), activeStorage.spatialMappingImports,
      std::move(catalogs->mappingTargetClasses),
      std::move(spatialMappingWorstRouteArrivalDelayQuanta),
      std::move(spatialMappingTotalRouteNegativeSlackQuanta),
      std::move(spatialMappingPhysicalTimingProfileDigests),
      std::move(spatialMappingPhysicalTimingProfileKinds),
      std::move(decisions->threads), std::move(decisions->threadChoices),
      std::move(decisions->graphs), std::move(decisions->graphChoices),
      std::move(graphChoicePressures->schedule),
      std::move(graphChoicePressures->operandIngress),
      std::move(*graphChoiceRecurrenceDemands), std::move(overlapOffsets),
      std::move(overlaps), staticStorage.routingTopology,
      std::move(routing->terminals), std::move(routing->ownerDomains),
      std::move(routing->endpointChoices), std::move(serviceDomains),
      std::move(*serviceContexts), std::move(*memoryBindings),
      staticStorage.instructionUsePatterns,
      staticStorage.consistencyUsePatterns, std::move(routing->legs),
      std::move(routing->legSinks), std::move(*capacityModel),
      std::move(*relations)));
}

llvm::Expected<FrozenSystemPnrProblemHandle>
loom::pnr::freezeSystemPnrProblemWithNormalizedTiming(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemPnrSearchDomainView &searchDomain,
    const ResolvedPnrConfigView &config,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    const ArtifactStore &store, const SystemStaticContext *staticContext,
    const SystemActiveContext *activeContext) {
  auto profiles =
      ::loom::fabric::projectNormalizedSystemPhysicalTimingProfiles(fabric);
  if (!profiles)
    return profiles.takeError();
  return freezeSystemPnrProblem(dataflow, fabric, *profiles, searchDomain,
                                config, constraints, store, staticContext,
                                activeContext);
}
