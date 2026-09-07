#include "CgraTransportRuntime.h"

#include <limits>
#include <system_error>
#include <utility>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

} // namespace

CgraTransportRuntime::CgraTransportRuntime(const CgraFrozenExecutionPlan &plan,
                                           const CgraTransportGraph &graph,
                                           SimulatorState &state,
                                           CgraPhysicalActionRuntime &physical)
    : plan_(&plan), state_(&state), physical_(&physical), graph_(graph),
      producerStates_(graph.bindings.size()),
      storageFrameCommits_(graph.storages.size()),
      channelArrivalCounts_(state.channelSlots.size(), 0),
      nextActionOccurrence_(plan.physicalUseTimings.size(), 0) {
  storages_.reserve(graph.storages.size());
  touchedStorageFrameCommits_.reserve(graph.storages.size());
  operandQueueUnits_.reserve(graph.operandQueueUnits.size());
  for (const auto &binding : graph.operandQueueUnits)
    operandQueueUnits_.emplace_back(binding);
  operandQueues_.reserve(graph.operandQueues.size());
  for (const auto &binding : graph.operandQueues)
    operandQueues_.emplace_back(binding);
}

llvm::Expected<CgraTransportRuntime> CgraTransportRuntime::create(
    const CgraFrozenExecutionPlan &plan, const CgraTransportGraph &graph,
    SimulatorState &state, CgraPhysicalActionRuntime &physical) {
  CgraTransportRuntime runtime(plan, graph, state, physical);
  for (const auto &binding : graph.storages) {
    const bool fullReplacementAllowed =
        binding.kind != CgraTraversalStorageKind::BufferedFifo &&
        binding.independentReadWriteServices;
    auto queue = CgraTransportStorageRuntime::create(
        binding.capacity, fullReplacementAllowed, binding.queueDiscipline);
    if (!queue)
      return queue.takeError();
    runtime.storages_.emplace_back(binding, std::move(*queue));
  }

  // Occupancy belongs to this invocation. The prepared graph holds only each
  // queue's consumer channels and its shared Fabric allocation unit.
  for (OperandQueueState &queue : runtime.operandQueues_) {
    const auto &binding = queue.binding;
    assert(!binding.consumers.empty() &&
           "prepared CGRA operand queue must have a consumer");
    const std::size_t initialOccupancy =
        state.channelSlots[binding.consumers.front().channel].ready.size();
    if (initialOccupancy > std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA PE operand queue occupancy exceeds u32");
    for (const auto &consumer : binding.consumers)
      if (state.channelSlots[consumer.channel].ready.size() != initialOccupancy)
        return invalid("CGRA PE operand broadcast consumers disagree on state");
    OperandQueueUnitState &unit =
        runtime.operandQueueUnits_[binding.unitBinding];
    if (unit.occupancy > unit.binding.capacity ||
        initialOccupancy > unit.binding.capacity - unit.occupancy)
      return invalid("CGRA PE operand allocation unit starts overfull");
    queue.occupancy = static_cast<std::uint32_t>(initialOccupancy);
    queue.entries.resize(initialOccupancy);
    unit.occupancy += queue.occupancy;
  }
  return runtime;
}

} // namespace loom::sim::detail
