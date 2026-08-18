#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
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

class ClosedIntegerExpressionEvaluator final {
public:
  ClosedIntegerExpressionEvaluator() = default;

  explicit ClosedIntegerExpressionEvaluator(
      const CanonicalDirectInvocationPathView &path) {
    for (mlir::Operation *operation : path.calls) {
      auto call = llvm::dyn_cast_or_null<mlir::LLVM::CallOp>(operation);
      if (!call || !call.getCalleeAttr()) {
        validPath = false;
        return;
      }
      auto callee =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
              call, call.getCalleeAttr());
      if (!callee ||
          !incomingCalls.try_emplace(callee.getOperation(), call).second) {
        validPath = false;
        return;
      }
    }
  }

  std::optional<mlir::Attribute> evaluate(mlir::Value value) {
    if (!validPath)
      return std::nullopt;
    if (auto found = values.find(value); found != values.end())
      return found->second;
    if (unavailable.contains(value) || !active.insert(value).second)
      return std::nullopt;

    std::optional<mlir::Attribute> result = evaluateImpl(value);
    active.erase(value);
    if (result)
      values.try_emplace(value, *result);
    else
      unavailable.insert(value);
    return result;
  }

private:
  std::optional<mlir::Attribute> evaluateImpl(mlir::Value value) {
    mlir::Attribute direct;
    if (mlir::matchPattern(value, mlir::m_Constant(&direct)))
      return direct;

    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value)) {
      auto function = argument.getOwner()->getParentOp()
                          ? llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(
                                argument.getOwner()->getParentOp())
                          : mlir::LLVM::LLVMFuncOp{};
      if (!function || function.isExternal() ||
          argument.getOwner() != &function.getBody().front())
        return std::nullopt;
      auto incoming = incomingCalls.find(function.getOperation());
      if (incoming == incomingCalls.end() ||
          argument.getArgNumber() >=
              incoming->second.getCalleeOperands().size())
        return std::nullopt;
      return evaluate(
          incoming->second.getCalleeOperands()[argument.getArgNumber()]);
    }

    auto result = llvm::dyn_cast<mlir::OpResult>(value);
    if (!result)
      return std::nullopt;
    mlir::Operation *definition = result.getOwner();
    if (definition->getNumRegions() != 0 ||
        !mlir::isMemoryEffectFree(definition))
      return std::nullopt;

    llvm::SmallVector<mlir::Attribute> operands;
    operands.reserve(definition->getNumOperands());
    for (mlir::Value operand : definition->getOperands()) {
      std::optional<mlir::Attribute> constant = evaluate(operand);
      if (!constant)
        return std::nullopt;
      operands.push_back(*constant);
    }

    mlir::Operation *probe = definition->clone();
    llvm::SmallVector<mlir::OpFoldResult> folded;
    const mlir::LogicalResult status = probe->fold(operands, folded);
    std::optional<mlir::Attribute> replacement;
    if (mlir::succeeded(status) &&
        folded.size() == definition->getNumResults()) {
      mlir::OpFoldResult selected = folded[result.getResultNumber()];
      if (auto attribute = llvm::dyn_cast<mlir::Attribute>(selected)) {
        replacement = attribute;
      } else if (mlir::Value foldedValue =
                     llvm::dyn_cast<mlir::Value>(selected);
                 foldedValue.getDefiningOp() != probe) {
        replacement = evaluate(foldedValue);
      }
    }
    probe->erase();
    if (!replacement)
      return std::nullopt;
    auto typed = llvm::dyn_cast<mlir::TypedAttr>(*replacement);
    if (!typed || typed.getType() != value.getType())
      return std::nullopt;
    return replacement;
  }

  llvm::DenseMap<mlir::Value, mlir::Attribute> values;
  llvm::DenseSet<mlir::Value> unavailable;
  llvm::DenseSet<mlir::Value> active;
  llvm::DenseMap<mlir::Operation *, mlir::LLVM::CallOp> incomingCalls;
  bool validPath = true;
};

std::optional<std::vector<std::uint64_t>>
projectDenseExtents(dataflow::ThreadLaunchOp launch,
                    const CanonicalDirectInvocationPathView *path) {
  std::vector<std::uint64_t> extents;
  extents.reserve(launch.getGridUpperBounds().size());
  ClosedIntegerExpressionEvaluator evaluator =
      path ? ClosedIntegerExpressionEvaluator(*path)
           : ClosedIntegerExpressionEvaluator();
  for (mlir::Value bound : launch.getGridUpperBounds()) {
    std::optional<mlir::Attribute> constant = evaluator.evaluate(bound);
    if (!constant)
      return std::nullopt;
    auto integer = llvm::dyn_cast<mlir::IntegerAttr>(*constant);
    if (!integer || integer.getValue().isNegative() ||
        integer.getValue().getActiveBits() > 64)
      return std::nullopt;
    extents.push_back(integer.getValue().getZExtValue());
  }
  return extents;
}

llvm::Expected<std::optional<std::vector<std::vector<std::uint64_t>>>>
enumerateDenseCoordinates(
    llvm::Expected<std::optional<std::vector<std::uint64_t>>> extents,
    std::uint64_t maximumPoints) {
  if (!extents)
    return extents.takeError();
  if (!*extents)
    return std::optional<std::vector<std::vector<std::uint64_t>>>{};

  std::uint64_t pointCount = 1;
  for (std::uint64_t extent : **extents) {
    if (extent == 0)
      return std::vector<std::vector<std::uint64_t>>{};
    if (pointCount > maximumPoints / extent)
      return std::optional<std::vector<std::vector<std::uint64_t>>>{};
    pointCount *= extent;
  }

  std::vector<std::vector<std::uint64_t>> coordinates;
  coordinates.reserve(static_cast<std::size_t>(pointCount));
  for (std::uint64_t linear = 0; linear != pointCount; ++linear) {
    std::uint64_t remainder = linear;
    std::vector<std::uint64_t> point((*extents)->size(), 0);
    for (std::size_t dimension = (*extents)->size(); dimension != 0;
         --dimension) {
      point[dimension - 1] = remainder % (**extents)[dimension - 1];
      remainder /= (**extents)[dimension - 1];
    }
    coordinates.push_back(std::move(point));
  }
  return std::optional<std::vector<std::vector<std::uint64_t>>>(
      std::move(coordinates));
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

llvm::Expected<std::optional<std::vector<std::uint64_t>>>
CanonicalDataflowProgramView::projectStaticDenseExtents(
    RootedGraphLaunchRef ref) const {
  auto logical = projectWholeRootedGraphLogicalDomain(ref);
  if (!logical)
    return logical.takeError();
  if (!*logical ||
      (*logical)->kind != dataflow::ThreadDomainKind::DenseRectangular)
    return std::optional<std::vector<std::uint64_t>>{};
  auto root = resolve(ref.rootThreadLaunch);
  if (!root)
    return root.takeError();
  auto launch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(root->op);
  if (!launch)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dataflow_logical_domain_invalid: root reference does not resolve a "
        "thread launch");

  return projectDenseExtents(launch, nullptr);
}

llvm::Expected<std::optional<std::vector<std::uint64_t>>>
CanonicalDataflowProgramView::projectStaticDenseExtents(
    RootedGraphLaunchRef ref, llvm::StringRef entrySymbol) const {
  auto logical = projectWholeRootedGraphLogicalDomain(ref);
  if (!logical)
    return logical.takeError();
  if (!*logical ||
      (*logical)->kind != dataflow::ThreadDomainKind::DenseRectangular)
    return std::optional<std::vector<std::uint64_t>>{};
  auto root = resolve(ref.rootThreadLaunch);
  if (!root)
    return root.takeError();
  auto launch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(root->op);
  if (!launch)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dataflow_logical_domain_invalid: root reference does not resolve a "
        "thread launch");
  auto paths = projectRootThreadInvocationPathsFromAbiEntry(
      entrySymbol, ref.rootThreadLaunch);
  if (!paths)
    return paths.takeError();

  std::optional<std::vector<std::uint64_t>> common;
  for (const CanonicalDirectInvocationPathView &path : *paths) {
    auto extents = projectDenseExtents(launch, &path);
    if (!extents)
      return std::optional<std::vector<std::uint64_t>>{};
    if (!common)
      common = std::move(*extents);
    else if (*common != *extents)
      return std::optional<std::vector<std::uint64_t>>{};
  }
  return common;
}

llvm::Expected<std::optional<std::vector<std::vector<std::uint64_t>>>>
CanonicalDataflowProgramView::enumerateStaticDenseCoordinates(
    RootedGraphLaunchRef ref, std::uint64_t maximumPoints) const {
  if (maximumPoints == 0)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dataflow_logical_domain_invalid: coordinate enumeration bound is "
        "zero");
  return enumerateDenseCoordinates(projectStaticDenseExtents(ref),
                                   maximumPoints);
}

llvm::Expected<std::optional<std::vector<std::vector<std::uint64_t>>>>
CanonicalDataflowProgramView::enumerateStaticDenseCoordinates(
    RootedGraphLaunchRef ref, std::uint64_t maximumPoints,
    llvm::StringRef entrySymbol) const {
  if (maximumPoints == 0)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dataflow_logical_domain_invalid: coordinate enumeration bound is "
        "zero");
  return enumerateDenseCoordinates(projectStaticDenseExtents(ref, entrySymbol),
                                   maximumPoints);
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

llvm::Expected<RootThreadLaunchRef>
CanonicalDataflowProgramView::eventRootThreadLaunch(
    const EventFamilyKey &event) const {
  if (llvm::Error error = validate(event))
    return std::move(error);
  return std::visit(
      [](const auto &owner) -> RootThreadLaunchRef {
        using Owner = std::decay_t<decltype(owner)>;
        if constexpr (std::is_same_v<Owner, RootThreadLaunchRef>)
          return owner;
        else
          return owner.rootThreadLaunch;
      },
      domainOf(event));
}

} // namespace dataflow
