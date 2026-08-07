#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <variant>
#include <vector>

namespace dataflow {
namespace {

using EventDomainRef = std::variant<RootThreadLaunchRef, RootedGraphLaunchRef>;

RootThreadLaunchRef rootOf(const RootThreadBoundaryTransferRef &transfer) {
  return std::visit([](const auto &reference) { return reference.launch; },
                    transfer);
}

RootedGraphLaunchRef rootedOf(const GraphLaunchBoundaryTransferRef &transfer) {
  return std::visit([](const auto &reference) { return reference.launch; },
                    transfer);
}

EventDomainRef domainOf(const ChannelProducerRef &producer) {
  return std::visit(
      [](const auto &reference) -> EventDomainRef {
        using Ref = std::decay_t<decltype(reference)>;
        if constexpr (std::is_same_v<Ref, GraphStreamOutputProducerRef>)
          return reference.launch;
        else
          return reference.launch;
      },
      producer);
}

EventDomainRef domainOf(const ChannelConsumerRef &consumer) {
  return std::visit(
      [](const auto &reference) -> EventDomainRef {
        using Ref = std::decay_t<decltype(reference)>;
        if constexpr (std::is_same_v<Ref, GraphStreamInputConsumerRef>)
          return reference.launch;
        else
          return reference.launch;
      },
      consumer);
}

EventDomainRef domainOf(const CanonicalProducerTerminalRef &terminal) {
  return std::visit(
      [](const auto &reference) -> EventDomainRef {
        using Ref = std::decay_t<decltype(reference)>;
        if constexpr (std::is_same_v<Ref, RootThreadBoundarySourceRef>)
          return rootOf(reference.transfer);
        else if constexpr (std::is_same_v<Ref, GraphLaunchBoundarySourceRef>)
          return rootedOf(reference.transfer);
        else
          return domainOf(reference.producer);
      },
      terminal);
}

EventDomainRef domainOf(const CanonicalSinkTerminalRef &terminal) {
  return std::visit(
      [](const auto &reference) -> EventDomainRef {
        using Ref = std::decay_t<decltype(reference)>;
        if constexpr (std::is_same_v<Ref, RootThreadBoundarySinkRef>)
          return rootOf(reference.transfer);
        else if constexpr (std::is_same_v<Ref, GraphLaunchBoundarySinkRef>)
          return rootedOf(reference.transfer);
        else
          return domainOf(reference.consumer);
      },
      terminal);
}

EventDomainRef domainOf(const StaticTransferEventRef &event) {
  return std::visit(
      [](const auto &reference) -> EventDomainRef {
        return domainOf(reference.terminal);
      },
      event);
}

EventDomainRef domainOf(const EventFamilyKey &event) {
  return std::visit(
      [](const auto &reference) -> EventDomainRef {
        using Ref = std::decay_t<decltype(reference)>;
        if constexpr (std::is_same_v<Ref, StaticTransferEventRef>)
          return domainOf(reference);
        else
          return reference.actor.launch;
      },
      event);
}

} // namespace

llvm::Expected<CanonicalRootThreadLogicalDomainView>
CanonicalDataflowProgramView::projectRootThreadLogicalDomain(
    RootThreadLaunchRef ref) const {
  auto resolved = resolve(ref);
  if (!resolved)
    return resolved.takeError();
  auto launch = llvm::dyn_cast<ThreadLaunchOp>(resolved->op);
  auto thread = llvm::dyn_cast<ThreadOp>(resolved->callee);
  if (!launch || !thread || thread.isExternal())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dataflow_logical_domain_invalid: root launch does not resolve a "
        "body-owning thread");

  const std::size_t inputCount = thread.getFunctionType().getNumInputs();
  const std::size_t bodyArgumentCount =
      thread.getBody().front().getNumArguments();
  if (bodyArgumentCount < inputCount + 1)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dataflow_logical_domain_invalid: thread body lacks its control slot");
  const std::size_t rank = bodyArgumentCount - inputCount - 1;
  if (rank > std::numeric_limits<std::uint32_t>::max())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dataflow_logical_domain_invalid: coordinate rank exceeds u32");
  if (launch.getGridUpperBounds().size() != rank)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dataflow_logical_domain_invalid: launch extent count differs from "
        "thread coordinate rank");

  std::vector<mlir::Value> parameters;
  parameters.reserve(launch.getGridUpperBounds().size() +
                     launch.getBodyOperands().size());
  parameters.insert(parameters.end(), launch.getGridUpperBounds().begin(),
                    launch.getGridUpperBounds().end());

  const std::optional<std::uint64_t> workItemOrdinal =
      thread.getDomain().getWorkItemArgOrdinal();
  for (auto item : llvm::enumerate(launch.getBodyOperands())) {
    if (thread.getDomain().getKind() == ThreadDomainKind::DynamicWork &&
        workItemOrdinal && item.index() == *workItemOrdinal)
      continue;
    mlir::Type type = item.value().getType();
    auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
    if (llvm::isa<mlir::IndexType>(type) || (integer && integer.isSignless()))
      parameters.push_back(item.value());
  }

  return CanonicalRootThreadLogicalDomainView{ref, thread.getDomain().getKind(),
                                              static_cast<std::uint32_t>(rank),
                                              std::move(parameters)};
}

llvm::Expected<std::optional<CanonicalRootThreadLogicalDomainView>>
CanonicalDataflowProgramView::projectWholeRootedGraphLogicalDomain(
    RootedGraphLaunchRef ref) const {
  auto rooted = resolve(ref);
  if (!rooted)
    return rooted.takeError();
  auto root = resolve(ref.rootThreadLaunch);
  if (!root)
    return root.takeError();
  auto graphLaunch = resolve(ref.staticGraphLaunch);
  if (!graphLaunch)
    return graphLaunch.takeError();

  auto thread = llvm::dyn_cast<ThreadOp>(root->callee);
  auto launch = llvm::dyn_cast<GraphLaunchOp>(graphLaunch->op);
  if (!thread || thread.isExternal() || !launch)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dataflow_logical_domain_invalid: rooted graph launch does not "
        "resolve inside a body-owning thread");

  if (launch->getParentOp() != thread.getOperation() ||
      launch->getBlock() != &thread.getBody().front())
    return std::optional<CanonicalRootThreadLogicalDomainView>{};

  auto domain = projectRootThreadLogicalDomain(ref.rootThreadLaunch);
  if (!domain)
    return domain.takeError();
  return std::optional<CanonicalRootThreadLogicalDomainView>(
      std::move(*domain));
}

llvm::Expected<EventLogicalProjection>
CanonicalDataflowProgramView::eventLogicalProjection(
    const EventFamilyKey &event) const {
  if (llvm::Error error = validate(event))
    return std::move(error);

  EventDomainRef owner = domainOf(event);
  llvm::Expected<CanonicalRootThreadLogicalDomainView> logical = std::visit(
      [&](const auto &reference)
          -> llvm::Expected<CanonicalRootThreadLogicalDomainView> {
        using Ref = std::decay_t<decltype(reference)>;
        if constexpr (std::is_same_v<Ref, RootThreadLaunchRef>) {
          return projectRootThreadLogicalDomain(reference);
        } else {
          auto whole = projectWholeRootedGraphLogicalDomain(reference);
          if (!whole)
            return whole.takeError();
          if (!*whole)
            return llvm::createStringError(
                llvm::inconvertibleErrorCode(),
                "dataflow_logical_domain_invalid: exact rooted event "
                "may-domain is not published");
          return std::move(**whole);
        }
      },
      owner);
  if (!logical)
    return logical.takeError();

  EventLogicalProjection projection;
  projection.reserve(logical->coordinateRank +
                     logical->launchParameters.size());
  for (StructuralOrdinal ordinal = 0; ordinal < logical->coordinateRank;
       ++ordinal)
    projection.emplace_back(CoordinateSlot{ordinal});
  for (StructuralOrdinal ordinal = 0;
       ordinal < logical->launchParameters.size(); ++ordinal)
    projection.emplace_back(LaunchParameterSlot{ordinal});
  return projection;
}

} // namespace dataflow
