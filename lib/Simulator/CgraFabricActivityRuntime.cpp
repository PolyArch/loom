#include "CgraFabricActivityRuntime.h"

#include "CGRAExecutionPlan.h"
#include "CGRAResourceRuntime.h"

#include "Fabric/IR/FifoResourceContract.h"
#include "Fabric/IR/TemporalPeResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <system_error>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "CGRA Fabric activity: " + message);
}

fabric::FabricResourceStateRef
stateRef(const fabric::FabricInventoryOwnerRef &owner, ::fabric::StateKey key) {
  return {fabric::FabricResourceStateOwnerRef(owner), key.ordinal()};
}

template <typename Dimension>
constexpr ::fabric::CapacityDimensionKey dimensionKey(Dimension dimension) {
  return ::fabric::CapacityDimensionKey(static_cast<std::uint32_t>(dimension));
}

} // namespace

llvm::Expected<std::unique_ptr<CgraFabricActivityRuntime>>
CgraFabricActivityRuntime::create(const fabric::FabricArtifactView &view,
                                  const CgraFrozenExecutionPlan &plan,
                                  ActivityWindow window,
                                  const SpatialEventCoordinate &launch) {
  if (window != ActivityWindow::LaunchToGraphRetirement &&
      window != ActivityWindow::LaunchToTerminal)
    return invalid("requested window is outside the progress domain");
  auto result = std::unique_ptr<CgraFabricActivityRuntime>(
      new CgraFabricActivityRuntime(plan, window));
  auto zero = evaluation::ExactRatio::get(0, 1);
  if (!zero)
    return zero.takeError();
  for (const auto &owner : view.moduleResourceOwners()) {
    const auto *contract = view.resourceContract(owner);
    if (!contract)
      return invalid("physical owner has no ResourceContract");
    for (std::uint32_t state = 0; state != contract->stateCount(); ++state) {
      const auto resource = stateRef(owner, ::fabric::StateKey(state));
      const auto key = fabric::canonicalFabricBytes(resource);
      for (auto [ordinal, capacity] : llvm::enumerate(
               contract->capacityDimensions(::fabric::StateKey(state)))) {
        const auto dense = result->dimensions_.size();
        if (dense >= plan.resources.dimensions.size() ||
            plan.resources.dimensions[dense].capacity !=
                capacity.capacity.value() ||
            plan.resources.dimensions[dense].initialOccupancy !=
                capacity.initialOccupancy.value())
          return invalid(
              "resource projection differs from the Fabric inventory");
        const auto dimension =
            ::fabric::CapacityDimensionKey(static_cast<std::uint32_t>(ordinal));
        result->dimensionOrdinals_.emplace(
            std::make_pair(key, dimension.ordinal()), dense);
        result->dimensions_.push_back(
            {resource, dimension, launch.referenceCycle, *zero,
             capacity.initialOccupancy.value(),
             capacity.initialOccupancy.value(), false});
      }
    }
    for (std::uint32_t pattern = 0; pattern != contract->usePatternCount();
         ++pattern) {
      const fabric::FabricUsePatternRef reference{
          fabric::FabricUsePatternOwnerRef(owner), pattern};
      result->useOrdinals_.emplace(fabric::canonicalFabricBytes(reference),
                                   result->useCounts_.size());
      result->useCounts_.push_back({reference, 0});
    }
  }
  if (result->dimensions_.size() != plan.resources.dimensions.size())
    return invalid("resource projection omits Fabric capacity dimensions");

  result->actionUses_.reserve(plan.physicalUses.size());
  for (const auto &use : plan.physicalUses) {
    std::vector<std::uint64_t> counters;
    for (const auto &pattern :
         llvm::ArrayRef(plan.physicalUsePatterns)
             .slice(use.patternOffset, use.patternCount)) {
      const auto found =
          result->useOrdinals_.find(fabric::canonicalFabricBytes(pattern));
      if (found == result->useOrdinals_.end())
        return invalid("selected use is outside the Fabric inventory");
      counters.push_back(found->second);
    }
    llvm::sort(counters);
    counters.erase(std::unique(counters.begin(), counters.end()),
                   counters.end());
    result->actionUses_.push_back(std::move(counters));
  }

  for (const auto fifo : view.fifoOccurrences())
    if (llvm::Error error = result->markDurable(
            stateRef(fabric::FabricInventoryOwnerRef::of(fifo),
                     ::fabric::fifoResourceState(
                         ::fabric::FifoResourceState::BufferedQueue)),
            ::fabric::fifoBufferedCapacity(
                ::fabric::FifoBufferedCapacity::QueueSlot)))
      return std::move(error);

  for (const auto pe : view.peOccurrences()) {
    if (view.peSchedule(pe) != ::fabric::Schedule::Temporal)
      continue;
    auto operand = ::fabric::deriveTemporalPeOperandBufferContract(view, pe);
    if (!operand)
      return operand.takeError();
    const auto owner = fabric::FabricInventoryOwnerRef::of(pe);
    auto &operandDimensions = result->operandDimensions_[pe.id()];
    for (std::uint32_t unit = 0; unit != operand->allocationUnitCount();
         ++unit) {
      const auto resource = stateRef(owner, operand->entryPoolState(unit));
      const auto dimension =
          dimensionKey(::fabric::OperandEntryPoolDimension::OccupiedEntry);
      if (llvm::Error error = result->markDurable(resource, dimension))
        return std::move(error);
      auto ordinal = result->dimensionOrdinal(resource, dimension);
      if (!ordinal)
        return ordinal.takeError();
      operandDimensions.pools.push_back(*ordinal);
    }
    for (std::uint32_t queue = 0; queue != operand->logicalQueues().size();
         ++queue) {
      const auto resource = stateRef(owner, operand->queueState(queue));
      const auto dimension =
          dimensionKey(::fabric::OperandQueueDimension::QueuedOperand);
      if (llvm::Error error = result->markDurable(resource, dimension))
        return std::move(error);
      auto ordinal = result->dimensionOrdinal(resource, dimension);
      if (!ordinal)
        return ordinal.takeError();
      operandDimensions.queues.push_back(*ordinal);
    }
    const auto *contract = view.resourceContract(owner);
    const auto fifoCount =
        view.inventorySize(owner, fabric::FabricInventoryKind::RegisterFifo);
    if (!contract || fifoCount > std::numeric_limits<std::uint32_t>::max())
      return invalid("temporal PE has no complete register FIFO inventory");
    for (std::uint32_t fifo = 0; fifo != fifoCount; ++fifo) {
      auto state = ::fabric::resolveTemporalPeRegisterFifoState(
          *contract, static_cast<std::uint32_t>(fifoCount), fifo);
      if (!state)
        return state.takeError();
      if (llvm::Error error = result->markDurable(
              stateRef(owner, *state),
              ::fabric::temporalRegisterFifoCapacity(
                  ::fabric::TemporalRegisterFifoCapacity::OccupiedEntry)))
        return std::move(error);
    }
  }

  result->storageDimensions_.resize(plan.transport.traversalStorages.size());
  for (const auto &traversal : plan.transport.traversals) {
    if (traversal.storageOrdinal == invalidCgraTransportOrdinal)
      continue;
    if (traversal.storageOrdinal >= result->storageDimensions_.size())
      return invalid("traversal storage has no physical inventory entry");
    std::optional<fabric::FabricResourceStateRef> resource;
    std::optional<::fabric::CapacityDimensionKey> dimension;
    if (const auto *fifo = std::get_if<fabric::FabricFifoTraversalPayload>(
            &traversal.reference.payload)) {
      resource = stateRef(fabric::FabricInventoryOwnerRef::of(fifo->owner),
                          ::fabric::fifoResourceState(
                              ::fabric::FifoResourceState::BufferedQueue));
      dimension = ::fabric::fifoBufferedCapacity(
          ::fabric::FifoBufferedCapacity::QueueSlot);
    } else if (const auto *fifo =
                   std::get_if<fabric::FabricPeRegisterFifoPayload>(
                       &traversal.reference.payload)) {
      const auto owner = fabric::FabricInventoryOwnerRef::of(fifo->owner);
      const auto *contract = view.resourceContract(owner);
      const auto count =
          view.inventorySize(owner, fabric::FabricInventoryKind::RegisterFifo);
      if (!contract || count > std::numeric_limits<std::uint32_t>::max() ||
          fifo->registerFifo >= count)
        return invalid("traversal register FIFO has no exact contract");
      auto state = ::fabric::resolveTemporalPeRegisterFifoState(
          *contract, static_cast<std::uint32_t>(count),
          static_cast<std::uint32_t>(fifo->registerFifo));
      if (!state)
        return state.takeError();
      resource = stateRef(owner, *state);
      dimension = ::fabric::temporalRegisterFifoCapacity(
          ::fabric::TemporalRegisterFifoCapacity::OccupiedEntry);
    } else {
      return invalid("traversal storage has no typed occupancy provider");
    }
    auto ordinal = result->dimensionOrdinal(*resource, *dimension);
    if (!ordinal)
      return ordinal.takeError();
    auto &selected = result->storageDimensions_[traversal.storageOrdinal];
    if (selected && *selected != *ordinal)
      return invalid("traversal storage names different capacity owners");
    selected = *ordinal;
  }
  for (const auto &dimension : result->storageDimensions_)
    if (!dimension)
      return invalid("a selected storage has no observed Fabric dimension");
  return result;
}

llvm::Expected<std::uint64_t> CgraFabricActivityRuntime::dimensionOrdinal(
    const fabric::FabricResourceStateRef &resource,
    ::fabric::CapacityDimensionKey dimension) const {
  const auto found = dimensionOrdinals_.find(
      {fabric::canonicalFabricBytes(resource), dimension.ordinal()});
  if (found == dimensionOrdinals_.end())
    return invalid("capacity reference is outside the exact Fabric inventory");
  return found->second;
}

llvm::Error CgraFabricActivityRuntime::markDurable(
    const fabric::FabricResourceStateRef &resource,
    ::fabric::CapacityDimensionKey dimension) {
  auto ordinal = dimensionOrdinal(resource, dimension);
  if (!ordinal)
    return ordinal.takeError();
  dimensions_[*ordinal].durable = true;
  return llvm::Error::success();
}

llvm::Error
CgraFabricActivityRuntime::integrate(std::uint64_t dimension,
                                     evaluation::ExactRatio throughCycle) {
  auto &observation = dimensions_[dimension];
  const int order =
      evaluation::compareExactRatio(throughCycle, observation.lastCycle);
  if (order < 0)
    return invalid("observation moved backward in reference cycles");
  if (order == 0)
    return llvm::Error::success();
  if (observation.occupancy == 0) {
    observation.lastCycle = throughCycle;
    return llvm::Error::success();
  }
  auto duration = throughCycle.subtract(observation.lastCycle);
  if (!duration)
    return duration.takeError();
  auto contribution = duration->multiplyInteger(observation.occupancy);
  if (!contribution)
    return contribution.takeError();
  auto integral = observation.integral.add(*contribution);
  if (!integral)
    return integral.takeError();
  observation.integral = *integral;
  observation.lastCycle = throughCycle;
  return llvm::Error::success();
}

llvm::Error
CgraFabricActivityRuntime::observe(std::uint64_t dimension,
                                   std::uint32_t occupancy,
                                   const SpatialEventCoordinate &coordinate) {
  if (closed_)
    return llvm::Error::success();
  if (dimension >= dimensions_.size() ||
      occupancy > plan_.resources.dimensions[dimension].capacity)
    return invalid("observed occupancy exceeds its exact capacity");
  if (occupancy == dimensions_[dimension].occupancy)
    return llvm::Error::success();
  if (llvm::Error error = integrate(dimension, coordinate.referenceCycle))
    return error;
  auto &observation = dimensions_[dimension];
  observation.occupancy = occupancy;
  observation.peak = std::max(observation.peak, occupancy);
  return llvm::Error::success();
}

llvm::Error CgraFabricActivityRuntime::observeClaims(
    std::uint64_t actionOrdinal, const CgraResourceRuntime &resources,
    const SpatialEventCoordinate &coordinate) {
  if (closed_)
    return llvm::Error::success();
  if (actionOrdinal >= plan_.physicalUseTimings.size())
    return invalid("physical lifecycle names an unknown action");
  const auto selected =
      plan_.physicalUseTimings[actionOrdinal].selectedUseOrdinal;
  if (selected >= plan_.resources.selectedUses.size())
    return invalid("physical lifecycle names an unknown claim envelope");
  const auto &use = plan_.resources.selectedUses[selected];
  for (const auto &claim : llvm::ArrayRef(plan_.resources.claims)
                               .slice(use.claimOffset, use.claimCount)) {
    if (claim.dimensionOrdinal >= dimensions_.size())
      return invalid("claim is outside the observed capacity inventory");
    if (dimensions_[claim.dimensionOrdinal].durable)
      continue;
    if (llvm::Error error =
            observe(claim.dimensionOrdinal,
                    resources.occupancy(claim.dimensionOrdinal), coordinate))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error CgraFabricActivityRuntime::observeGranted(
    std::uint64_t actionOrdinal, const CgraResourceRuntime &resources,
    const SpatialEventCoordinate &coordinate) {
  if (closed_)
    return llvm::Error::success();
  if (actionOrdinal >= actionUses_.size())
    return invalid("grant names an unknown physical action");
  for (std::uint64_t ordinal : actionUses_[actionOrdinal]) {
    auto &count = useCounts_[ordinal].activations;
    if (count == std::numeric_limits<std::uint64_t>::max())
      return invalid("use activation count overflows u64");
    ++count;
  }
  return observeClaims(actionOrdinal, resources, coordinate);
}

llvm::Error CgraFabricActivityRuntime::observeReleased(
    std::uint64_t actionOrdinal, const CgraResourceRuntime &resources,
    const SpatialEventCoordinate &coordinate) {
  return observeClaims(actionOrdinal, resources, coordinate);
}

llvm::Error CgraFabricActivityRuntime::observeTraversalStorage(
    std::uint64_t storageOrdinal, std::uint32_t occupancy,
    const SpatialEventCoordinate &coordinate) {
  if (closed_)
    return llvm::Error::success();
  if (storageOrdinal >= storageDimensions_.size() ||
      !storageDimensions_[storageOrdinal])
    return invalid("storage commit has no observed capacity dimension");
  return observe(*storageDimensions_[storageOrdinal], occupancy, coordinate);
}

llvm::Error CgraFabricActivityRuntime::observeOperandQueue(
    fabric::FabricPeOccurrenceRef pe,
    const ::fabric::TemporalOperandBufferContract &contract,
    std::uint32_t queue, std::uint32_t allocationUnit,
    std::uint32_t queueOccupancy, std::uint32_t poolOccupancy,
    const SpatialEventCoordinate &coordinate) {
  if (closed_)
    return llvm::Error::success();
  const auto found = operandDimensions_.find(pe.id());
  if (found == operandDimensions_.end() ||
      queue >= found->second.queues.size() ||
      allocationUnit >= found->second.pools.size() ||
      contract.allocationUnitOf(queue) != allocationUnit)
    return invalid(
        "operand commit is outside its exact Fabric queue inventory");
  if (llvm::Error error =
          observe(found->second.queues[queue], queueOccupancy, coordinate))
    return error;
  return observe(found->second.pools[allocationUnit], poolOccupancy,
                 coordinate);
}

llvm::Error
CgraFabricActivityRuntime::close(ActivityWindow window,
                                 const SpatialEventCoordinate &coordinate) {
  if (closed_ || window != window_)
    return llvm::Error::success();
  for (std::uint64_t ordinal = 0; ordinal != dimensions_.size(); ++ordinal)
    if (llvm::Error error = integrate(ordinal, coordinate.referenceCycle))
      return error;
  closed_ = true;
  return llvm::Error::success();
}

std::vector<ActivitySummary> CgraFabricActivityRuntime::takeSummaries() {
  if (!closed_ || (dimensions_.empty() && useCounts_.empty()))
    return {};
  FabricResourcesActivity activity;
  for (const auto &[key, ordinal] : useOrdinals_)
    activity.useCounts.push_back(std::move(useCounts_[ordinal]));
  std::vector<std::uint8_t> previous;
  for (const auto &[key, ordinal] : dimensionOrdinals_) {
    const auto &observation = dimensions_[ordinal];
    if (activity.resourceOccupancy.empty() || key.first != previous) {
      activity.resourceOccupancy.push_back({observation.resource, {}});
      previous = key.first;
    }
    activity.resourceOccupancy.back().dimensions.push_back(
        {observation.dimension, observation.integral, observation.peak});
  }
  return {{window_, ActivityCoverage::Complete, std::move(activity)}};
}

} // namespace loom::sim::detail
