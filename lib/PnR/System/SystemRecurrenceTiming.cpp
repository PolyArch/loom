#include "SystemRecurrenceTiming.h"

#include "../SpatialRecurrenceTimingPersistent.h"
#include "SystemCandidateServiceResolver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

using namespace loom;
using namespace loom::pnr;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_recurrence_timing_invalid: " +
                                     message);
}

llvm::Expected<std::uint64_t>
checkedAdd(std::uint64_t left, std::uint64_t right, llvm::StringRef subject) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    return invalid(subject + " exceeds u64");
  return left + right;
}

const ::dataflow::ContextualActorRef *
contextualActor(const SystemServiceTargetSubject &subject) {
  const auto *member = std::get_if<SystemServiceMemberTargetSubject>(&subject);
  if (!member)
    return nullptr;
  if (const auto *addressed =
          std::get_if<::dataflow::AddressedMemoryActorMemberRef>(
              &member->member))
    return &addressed->actor;
  if (const auto *fence =
          std::get_if<::dataflow::FenceActorMemberRef>(&member->member))
    return &fence->actor;
  return nullptr;
}

llvm::Expected<std::uint64_t>
routeLatency(const FrozenSystemPnrProblem &problem, PnrIndex leg,
             llvm::ArrayRef<SystemServiceRouteSelection> routes,
             llvm::ArrayRef<SystemServiceRouteNodeSelection> nodes,
             llvm::ArrayRef<SystemServiceRouteSinkSelection> sinks) {
  const SystemServiceRouteSelection *selected = nullptr;
  for (const SystemServiceRouteSelection &route : routes) {
    if (route.leg != leg)
      continue;
    if (selected)
      return invalid("one service leg has multiple route trees");
    selected = &route;
  }
  if (!selected || selected->nodeOffset > nodes.size() ||
      selected->nodeCount > nodes.size() - selected->nodeOffset ||
      selected->sinkOffset > sinks.size() ||
      selected->sinkCount > sinks.size() - selected->sinkOffset ||
      selected->nodeCount == 0)
    return invalid("service recurrence route tree is incomplete");
  const auto routeNodes =
      nodes.slice(selected->nodeOffset, selected->nodeCount);
  std::uint64_t maximum = 0;
  for (const SystemServiceRouteSinkSelection &sink :
       sinks.slice(selected->sinkOffset, selected->sinkCount)) {
    if (sink.node >= routeNodes.size())
      return invalid("service recurrence sink node is out of range");
    std::uint64_t latency = 0;
    PnrIndex node = sink.node;
    for (std::size_t depth = 0; node != 0; ++depth) {
      if (depth >= routeNodes.size())
        return invalid("service recurrence route tree contains a cycle");
      const SystemServiceRouteNodeSelection &record = routeNodes[node];
      if (record.parentNode >= node ||
          record.incomingTraversal >=
              problem.routingTopology().traversals().size())
        return invalid("service recurrence route node is malformed");
      auto next = checkedAdd(latency,
                             problem.routingTopology()
                                 .traversals()[record.incomingTraversal]
                                 .architecturalLatencyCycles,
                             "service recurrence route latency");
      if (!next)
        return next.takeError();
      latency = *next;
      node = record.parentNode;
    }
    maximum = std::max(maximum, latency);
  }
  return maximum;
}

llvm::Expected<std::optional<std::uint64_t>> boundaryCompletion(
    const FrozenSystemPnrProblem &problem, PnrIndex graphDecision,
    const ::dataflow::ContextualActorRef &actor,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices,
    llvm::ArrayRef<SystemServiceRouteSelection> serviceRoutes,
    llvm::ArrayRef<SystemServiceRouteNodeSelection> serviceRouteNodes,
    llvm::ArrayRef<SystemServiceRouteSinkSelection> serviceRouteSinks) {
  bool found = false;
  std::uint64_t maximum = 0;
  for (const auto &[contextOrdinalValue, context] :
       llvm::enumerate(problem.serviceContexts())) {
    if (context.graphDecision != graphDecision)
      continue;
    const PnrIndex contextOrdinal = static_cast<PnrIndex>(contextOrdinalValue);
    for (const SystemServiceTargetSubject &subject : context.subjects) {
      const auto *candidate = contextualActor(subject);
      if (!candidate || *candidate != actor)
        continue;
      found = true;
      auto binding = loom::pnr::detail::resolveSystemMemoryServiceBinding(
          problem, contextOrdinal, subject, threadChoices, graphChoices);
      if (!binding)
        return binding.takeError();
      if (!(*binding)->maxIssueToRetireCycles)
        return std::optional<std::uint64_t>{};
      const auto *member =
          std::get_if<SystemServiceMemberTargetSubject>(&subject);
      if (!member)
        return invalid("memory recurrence subject is not a service member");
      std::uint64_t transport = 0;
      bool foundLeg = false;
      for (const auto &[legOrdinalValue, leg] :
           llvm::enumerate(problem.serviceLegs())) {
        if (leg.serviceContext != contextOrdinal ||
            leg.key.member != member->member)
          continue;
        foundLeg = true;
        auto latency =
            routeLatency(problem, static_cast<PnrIndex>(legOrdinalValue),
                         serviceRoutes, serviceRouteNodes, serviceRouteSinks);
        if (!latency)
          return latency.takeError();
        auto next = checkedAdd(transport, *latency,
                               "service recurrence transport latency");
        if (!next)
          return next.takeError();
        transport = *next;
      }
      if (!foundLeg)
        return invalid("memory recurrence service has no transfer leg");
      auto completion =
          checkedAdd(*(*binding)->maxIssueToRetireCycles, transport,
                     "service recurrence completion latency");
      if (!completion)
        return completion.takeError();
      maximum = std::max(maximum, *completion);
    }
  }
  if (!found)
    return invalid("boundary memory actor has no System service context");
  return std::optional<std::uint64_t>{maximum};
}

} // namespace

llvm::Expected<SpatialRecurrenceTimingProjection>
loom::pnr::detail::projectSystemRecurrenceTiming(
    const FrozenSystemPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices,
    llvm::ArrayRef<SystemServiceRouteSelection> serviceRoutes,
    llvm::ArrayRef<SystemServiceRouteNodeSelection> serviceRouteNodes,
    llvm::ArrayRef<SystemServiceRouteSinkSelection> serviceRouteSinks) {
  if (threadChoices.size() != problem.threadDecisions().size() ||
      graphChoices.size() != problem.graphDecisions().size())
    return invalid("System recurrence execution choices are incomplete");
  SpatialRecurrenceTimingProjection result;
  for (PnrIndex decision = 0; decision < graphChoices.size(); ++decision) {
    const auto domain = problem.graphChoiceCatalogOrdinals(decision);
    if (graphChoices[decision] >= domain.size())
      return invalid("System recurrence graph choice is out of range");
    const auto &demand =
        problem.graphChoiceRecurrenceDemand(decision, graphChoices[decision]);
    if (!demand)
      return invalid("System recurrence demand is null");
    auto projection = projectFrozenSpatialRecurrenceTimingDemand(
        *demand, problem.graphDecisions()[decision].launch,
        [&](const ::dataflow::ContextualActorRef &actor) {
          return boundaryCompletion(problem, decision, actor, threadChoices,
                                    graphChoices, serviceRoutes,
                                    serviceRouteNodes, serviceRouteSinks);
        });
    if (!projection)
      return projection.takeError();
    if (projection->kind ==
        SpatialRecurrenceTimingProofKind::ProofNotEstablished)
      return std::move(*projection);
    result.recurrenceMinimumInitiationIntervalCycles =
        std::max(result.recurrenceMinimumInitiationIntervalCycles,
                 projection->recurrenceMinimumInitiationIntervalCycles);
    result.witnesses.insert(
        result.witnesses.end(),
        std::make_move_iterator(projection->witnesses.begin()),
        std::make_move_iterator(projection->witnesses.end()));
  }
  return result;
}
