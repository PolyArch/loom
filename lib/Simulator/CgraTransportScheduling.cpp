#include "CgraTransportRuntime.h"

#include <system_error>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

} // namespace

llvm::Expected<bool>
CgraTransportRuntime::markDirectSinksReady(std::uint64_t slot) {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA direct sink readiness names an inactive token");
  InFlight &inFlight = inFlight_[slot];
  const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
  if (inFlight.readySinks.size() != binding.sinkCount ||
      inFlight.permittedSinkTerminals.size() != binding.sinkCount)
    return invalid("CGRA sink readiness state has the wrong domain");
  bool changed = false;
  for (std::uint32_t sink = 0; sink != binding.sinkCount; ++sink) {
    const SinkBinding &selected = sinks_[binding.sinkOffset + sink];
    if (selected.traversalTerminalCount != 0 || inFlight.readySinks[sink])
      continue;
    inFlight.readySinks[sink] = true;
    ++inFlight.readySinkCount;
    changed = true;
  }
  return changed;
}

llvm::Expected<bool>
CgraTransportRuntime::markTerminalSinksReady(std::uint64_t slot,
                                             std::uint64_t nodeOrdinal) {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA terminal sink readiness names an inactive token");
  InFlight &inFlight = inFlight_[slot];
  const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
  if (nodeOrdinal < binding.traversalNodeOffset ||
      nodeOrdinal >= binding.traversalNodeOffset + binding.traversalNodeCount)
    return invalid("CGRA terminal sink readiness names another route");
  if (inFlight.readySinks.size() != binding.sinkCount ||
      inFlight.permittedSinkTerminals.size() != binding.sinkCount)
    return invalid("CGRA sink readiness state has the wrong domain");
  bool changed = false;
  for (std::uint32_t sink : traversalNodes_[nodeOrdinal].terminalSinks) {
    if (sink >= binding.sinkCount)
      return invalid("CGRA terminal traversal names an unknown sink");
    const std::uint32_t required =
        sinks_[binding.sinkOffset + sink].traversalTerminalCount;
    std::uint32_t &permitted = inFlight.permittedSinkTerminals[sink];
    if (required == 0 || permitted >= required)
      return invalid("CGRA sink received too many terminal permissions");
    ++permitted;
    if (permitted != required)
      continue;
    inFlight.readySinks[sink] = true;
    ++inFlight.readySinkCount;
    changed = true;
  }
  return changed;
}

llvm::Error CgraTransportRuntime::scheduleArrival(
    std::uint64_t slot, const SpatialEventCoordinate &coordinate) {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA transport arrival names an inactive token");
  InFlight &inFlight = inFlight_[slot];
  if (inFlight.arrivalScheduled)
    return invalid("CGRA transport arrival was scheduled twice");
  arrivalEvents_.schedule(
      {{coordinate, inFlight.bindingOrdinal, inFlight.occurrenceOrdinal, 0},
       slot});
  inFlight.arrivalScheduled = true;
  return llvm::Error::success();
}

llvm::Expected<bool> CgraTransportRuntime::scheduleReadyTraversals(
    std::uint64_t slot, const SpatialEventCoordinate &coordinate) {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA traversal event names an inactive token");
  const InFlight &inFlight = inFlight_[slot];
  const TransferBinding &binding = bindings_[inFlight.bindingOrdinal];
  bool scheduled = false;
  for (std::uint64_t nodeOrdinal = binding.traversalNodeOffset;
       nodeOrdinal != binding.traversalNodeOffset + binding.traversalNodeCount;
       ++nodeOrdinal) {
    if (traversalNodeStates_[nodeOrdinal] != TraversalNodeState::Idle ||
        traversalRemainingPredecessors_[nodeOrdinal] != 0)
      continue;
    const TraversalNodeBinding &node = traversalNodes_[nodeOrdinal];
    if (node.kind != TraversalNodeKind::PhysicalAction) {
      if (node.storageOrdinal >= storages_.size())
        return invalid("CGRA traversal storage ordinal is out of range");
      StorageBinding &storage = storages_[node.storageOrdinal];
      const bool buffered =
          node.kind == TraversalNodeKind::BufferedStorage &&
          storage.kind == CgraTraversalStorageKind::BufferedFifo;
      const bool registerWrite =
          node.kind == TraversalNodeKind::RegisterStorageWrite &&
          storage.kind != CgraTraversalStorageKind::BufferedFifo;
      const bool registerRead =
          node.kind == TraversalNodeKind::RegisterStorageRead &&
          storage.kind != CgraTraversalStorageKind::BufferedFifo;
      if (!buffered && !registerWrite && !registerRead)
        return invalid("CGRA traversal disagrees with its storage owner");
      if (registerRead)
        storage.pendingDequeueNodes.push_back(nodeOrdinal);
      else
        storage.pendingEnqueueNodes.push_back(nodeOrdinal);
      traversalNodeStates_[nodeOrdinal] = TraversalNodeState::WaitingStorage;
      if (llvm::Error error = scheduleStorage(node.storageOrdinal, coordinate))
        return std::move(error);
      scheduled = true;
      continue;
    }
    traversalEvents_.schedule(
        {{coordinate, nodeOrdinal, inFlight.occurrenceOrdinal,
          static_cast<std::uint32_t>(nodeOrdinal -
                                     binding.traversalNodeOffset)},
         nodeOrdinal});
    traversalNodeStates_[nodeOrdinal] = TraversalNodeState::Scheduled;
    scheduled = true;
  }
  return scheduled;
}

llvm::Error CgraTransportRuntime::scheduleStorage(
    std::uint64_t storageOrdinal, const SpatialEventCoordinate &coordinate) {
  if (storageOrdinal >= storages_.size())
    return invalid("CGRA storage event names an unknown queue");
  StorageBinding &storage = storages_[storageOrdinal];
  if (storage.eventScheduled || storage.activeActionCount != 0)
    return llvm::Error::success();
  if (storage.pendingEnqueueNodes.empty() &&
      storage.pendingDequeueNodes.empty() && storage.queue.empty())
    return llvm::Error::success();
  storageEvents_.schedule({{coordinate, storageOrdinal, 0, 0}, storageOrdinal});
  storage.eventScheduled = true;
  return llvm::Error::success();
}

llvm::Error CgraTransportRuntime::schedulePublication(
    std::uint64_t slot, const SpatialEventCoordinate &coordinate) {
  if (slot >= inFlight_.size() || !inFlight_[slot].active)
    return invalid("CGRA transport publication names an inactive token");
  InFlight &inFlight = inFlight_[slot];
  if (inFlight.publicationScheduled || inFlight.published)
    return invalid("CGRA transport publication was scheduled twice");
  scheduleAt(slot, coordinate);
  inFlight.publicationScheduled = true;
  return llvm::Error::success();
}

} // namespace loom::sim::detail
