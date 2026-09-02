#include "Simulator/CGRASimulator.h"

#include "CGRAPreparedExecutionInternal.h"
#include "CgraGraphActivationRuntime.h"
#include "SimulationWireInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::sim {

char CgraExecutionUnsupported::ID = 0;

void CgraExecutionUnsupported::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code CgraExecutionUnsupported::convertToErrorCode() const {
  return std::make_error_code(std::errc::not_supported);
}

namespace {

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

llvm::Expected<SpatialEventCoordinate> launchCoordinate() {
  auto cycle = evaluation::ExactRatio::get(0, 1);
  if (!cycle)
    return cycle.takeError();
  return SpatialEventCoordinate{std::move(*cycle), 0};
}

std::optional<std::uint64_t>
integralReferenceCycleDistance(const SpatialEventCoordinate &from,
                               const SpatialEventCoordinate &to) {
  if (compareSpatialEventCoordinates(to, from) < 0)
    return std::nullopt;
  using u128 = unsigned __int128;
  const u128 fromValue = static_cast<u128>(from.referenceCycle.numerator()) *
                         to.referenceCycle.denominator();
  const u128 toValue = static_cast<u128>(to.referenceCycle.numerator()) *
                       from.referenceCycle.denominator();
  const u128 commonDenominator =
      static_cast<u128>(from.referenceCycle.denominator()) *
      to.referenceCycle.denominator();
  const u128 difference = toValue - fromValue;
  if (commonDenominator == 0 || difference % commonDenominator != 0 ||
      difference / commonDenominator >
          std::numeric_limits<std::uint64_t>::max())
    return std::nullopt;
  return static_cast<std::uint64_t>(difference / commonDenominator);
}

std::vector<std::uint64_t> findTransferWaitCycle(
    llvm::ArrayRef<CgraClosedWaitSetDiagnostic::Transfer> transfers) {
  const std::uint64_t absent = std::numeric_limits<std::uint64_t>::max();
  std::vector<std::vector<std::uint64_t>> edges(transfers.size());
  for (std::uint64_t waiting = 0; waiting != transfers.size(); ++waiting) {
    const auto &transfer = transfers[waiting];
    if (!transfer.blocked)
      continue;
    for (std::uint64_t blocking = 0; blocking != transfers.size(); ++blocking) {
      const auto &candidate = transfers[blocking];
      const bool actorWait =
          transfer.blockingActorOrdinal != absent &&
          candidate.producerActorOrdinal == transfer.blockingActorOrdinal;
      const auto ownsStorageHead = [&](const auto &head) {
        return head && candidate.bindingOrdinal == head->bindingOrdinal &&
               candidate.occurrenceOrdinal == head->occurrenceOrdinal &&
               blocking != waiting;
      };
      const bool waitingForStorageCapacity =
          transfer.blockingTraversalWaitingForStorage;
      if (actorWait ||
          (waitingForStorageCapacity &&
           ownsStorageHead(transfer.blockingStorageHead)) ||
          ownsStorageHead(transfer.blockingDownstreamStorageHead))
        edges[waiting].push_back(blocking);
    }
    llvm::sort(edges[waiting]);
    edges[waiting].erase(
        std::unique(edges[waiting].begin(), edges[waiting].end()),
        edges[waiting].end());
  }

  std::vector<std::uint8_t> state(edges.size(), 0);
  std::vector<std::uint64_t> stack;
  std::vector<std::uint64_t> stackPosition(edges.size(), absent);
  std::vector<std::uint64_t> cycle;
  std::function<bool(std::uint64_t)> visit = [&](std::uint64_t node) {
    state[node] = 1;
    stackPosition[node] = stack.size();
    stack.push_back(node);
    for (std::uint64_t sink : edges[node]) {
      if (state[sink] == 0) {
        if (visit(sink))
          return true;
        continue;
      }
      if (state[sink] != 1)
        continue;
      cycle.assign(stack.begin() + stackPosition[sink], stack.end());
      cycle.push_back(sink);
      return true;
    }
    stack.pop_back();
    stackPosition[node] = absent;
    state[node] = 2;
    return false;
  };
  for (std::uint64_t node = 0; node != edges.size(); ++node)
    if (state[node] == 0 && visit(node))
      break;
  return cycle;
}

CgraClosedWaitSetDiagnostic::TransferWaitKind
transferWaitKind(const CgraClosedWaitSetDiagnostic::Transfer &waiting,
                 const CgraClosedWaitSetDiagnostic::Transfer &blocking) {
  const auto matches = [&](const auto &head) {
    return head && blocking.bindingOrdinal == head->bindingOrdinal &&
           blocking.occurrenceOrdinal == head->occurrenceOrdinal;
  };
  if (waiting.blockingTraversalWaitingForStorage &&
      matches(waiting.blockingStorageHead))
    return CgraClosedWaitSetDiagnostic::TransferWaitKind::StorageHead;
  if (matches(waiting.blockingDownstreamStorageHead))
    return CgraClosedWaitSetDiagnostic::TransferWaitKind::DownstreamStorageHead;
  return CgraClosedWaitSetDiagnostic::TransferWaitKind::ActorPublication;
}

struct ActorWaitCase final {
  std::vector<std::uint64_t> internalProducers;
};

struct ActorWaitState final {
  std::vector<std::uint64_t> outputBackpressure;
  std::vector<ActorWaitCase> missingInputCases;
  bool usesOutputBackpressure = false;
  bool eligible = false;
};

using ActorTransitionProbeTable =
    std::vector<std::optional<detail::ActorTransitionProbeResult>>;

ActorTransitionProbeTable
deriveActorTransitionProbes(const detail::PreparedGraphExecution &execution,
                            const detail::SimulatorState &state) {
  ActorTransitionProbeTable probes(execution.actorPlans.size());
  for (const auto [ordinal, actor] : llvm::enumerate(execution.actorPlans)) {
    if (actor.transitionProbe == detail::ActorTransitionProbeKind::Unavailable)
      continue;
    auto probe = detail::probeActorTransition(actor, state);
    if (!probe) {
      llvm::consumeError(probe.takeError());
      continue;
    }
    probes[ordinal] = std::move(*probe);
  }
  return probes;
}

/// Projects the unified causal certificate of one closed wait. The wait-for
/// graph joins typed dynamic owners — exact actor firing occurrences and the
/// queue classes of physical storages — through the wait facts the runtime
/// observed: missing inputs, output backpressure, queue order, downstream
/// capacity, and terminal consumption. The certificate is the single closed
/// sink strongly connected component of that graph; every node in it waits
/// inside the component. When a required occurrence or owner is
/// indeterminate, or no closed component exists, the outcome is a typed
/// proof failure rather than a forged certificate.
void buildWaitCertificate(const detail::CgraGraphActivationRuntime &runtime,
                          CgraClosedWaitSetDiagnostic &closedWait) {
  using Diagnostic = CgraClosedWaitSetDiagnostic;
  using ActorKey = Diagnostic::WaitActorFiringKey;
  using QueueClass = Diagnostic::WaitQueueClass;
  using StorageKey = Diagnostic::WaitStorageQueueKey;
  using OwnerKey = Diagnostic::WaitOwnerKey;
  using EdgeKind = Diagnostic::WaitEdgeKind;
  constexpr std::uint64_t absent = detail::invalidCgraTransportOrdinal;

  closedWait.waitCertificate.clear();
  closedWait.waitProofFailure.reset();

  const bool debugCertificate =
      std::getenv("LOOM_DEBUG_WAIT_CERTIFICATE") != nullptr;
  if (debugCertificate) {
    for (const auto &input : closedWait.blockedActorInputs)
      llvm::errs() << "wait-certificate debug: blocked-input actor="
                   << input.semanticActorOrdinal << " input="
                   << input.inputOrdinal << " channel=" << input.channelOrdinal
                   << " source=" << static_cast<unsigned>(input.sourceKind)
                   << " defining=" << input.definingActorOrdinal
                   << " expected=" << input.expectedProducerOccurrenceOrdinal
                   << '\n';
    for (const auto &firing : closedWait.actorFirings)
      llvm::errs() << "wait-certificate debug: firing actor="
                   << firing.semanticActorOrdinal << " occurrence="
                   << firing.occurrenceOrdinal << '\n';
  }

  const auto fail = [&](Diagnostic::WaitProofFailure reason) {
    closedWait.waitProofFailure = reason;
  };

  llvm::DenseMap<std::uint64_t, std::uint64_t> activeOccurrence;
  for (const Diagnostic::ActorFiring &firing : closedWait.actorFirings)
    activeOccurrence[firing.semanticActorOrdinal] = firing.occurrenceOrdinal;
  // Input-side waits belong to the firing that will consume the awaited
  // input: the actor's next occurrence, even when an earlier occurrence is
  // still completing its outputs.
  const auto nextFiring = [&](std::uint64_t actor) -> std::uint64_t {
    return runtime.nextActorOccurrenceOrdinal(actor).value_or(absent);
  };
  const auto actorNode = [&](std::uint64_t actor, std::uint64_t occurrence) {
    return OwnerKey{ActorKey{actor, occurrence}};
  };
  const auto storageNode = [&](std::uint64_t ordinal, QueueClass queueClass) {
    return OwnerKey{StorageKey{Diagnostic::WaitStorageDomain::TraversalStorage,
                               ordinal, queueClass}};
  };

  // Residency per storage: queue order with exact tag values, grouped into
  // the queue classes the discipline presents.
  struct ResidencyEntry {
    std::uint64_t bindingOrdinal;
    std::uint64_t occurrenceOrdinal;
    llvm::APInt tagValue;
    bool tagged;
    std::uint32_t queuePosition;
    std::uint64_t producerActorOrdinal;
    std::vector<std::uint64_t> destinationActorOrdinals;
    std::vector<std::uint64_t> destinationChannelOrdinals;
    std::vector<std::uint32_t> destinationInputOrdinals;
  };
  struct StorageResidency {
    std::vector<ResidencyEntry> entries;
    ::fabric::FifoQueueDiscipline discipline =
        ::fabric::FifoQueueDiscipline::StrictFifo;
  };
  std::vector<StorageResidency> residencies;
  const std::uint64_t storageCount = runtime.traversalStorageCount();
  residencies.reserve(storageCount);
  for (std::uint64_t storage = 0; storage != storageCount; ++storage) {
    StorageResidency residency;
    residency.discipline =
        runtime.traversalStorageQueueDiscipline(storage).value_or(
            ::fabric::FifoQueueDiscipline::StrictFifo);
    for (const auto &entry : runtime.storageResidencyDiagnostics(storage))
      residency.entries.push_back(
          {entry.bindingOrdinal, entry.occurrenceOrdinal, entry.physicalTagValue,
           entry.physicalTagOrdinal != absent, entry.queuePosition,
           entry.producerActorOrdinal, entry.destinationActorOrdinals,
           entry.destinationChannelOrdinals, entry.destinationInputOrdinals});
    residencies.push_back(std::move(residency));
  }
  const auto queueClassOf = [&](const StorageResidency &storage,
                                const llvm::APInt &tagValue,
                                bool tagged) {
    return storage.discipline ==
                   ::fabric::FifoQueueDiscipline::PerTagVirtualChannel &&
                   tagged
               ? QueueClass::tag(tagValue)
               : QueueClass::global();
  };
  // The class head is the first resident entry of the class in queue order.
  const auto classHead = [&](const StorageResidency &storage,
                             const QueueClass &queueClass)
      -> const ResidencyEntry * {
    for (const ResidencyEntry &entry : storage.entries)
      if (queueClassOf(storage, entry.tagValue,
                       true) == queueClass)
        return &entry;
    return nullptr;
  };
  const auto classPositionOf = [&](const StorageResidency &storage,
                                   const QueueClass &queueClass,
                                   std::uint64_t bindingOrdinal,
                                   std::uint64_t occurrenceOrdinal)
      -> std::optional<std::uint32_t> {
    std::uint32_t position = 0;
    for (const ResidencyEntry &entry : storage.entries) {
      if (queueClassOf(storage, entry.tagValue, true) != queueClass)
        continue;
      if (entry.bindingOrdinal == bindingOrdinal &&
          entry.occurrenceOrdinal == occurrenceOrdinal)
        return position;
      ++position;
    }
    return std::nullopt;
  };

  // Transfers indexed for awaited-token lookup.
  const auto findTransfer = [&](std::uint64_t producerActor,
                                std::uint64_t occurrence,
                                std::uint64_t consumerActor,
                                std::uint32_t consumerInput)
      -> const Diagnostic::Transfer * {
    for (const Diagnostic::Transfer &transfer : closedWait.transfers) {
      if (transfer.producerActorOrdinal != producerActor ||
          transfer.occurrenceOrdinal != occurrence)
        continue;
      for (std::size_t sink = 0; sink != transfer.unpublishedActorOrdinals.size();
           ++sink)
        if (transfer.unpublishedActorOrdinals[sink] == consumerActor &&
            transfer.unpublishedInputOrdinals[sink] == consumerInput)
          return &transfer;
    }
    return nullptr;
  };
  const auto findTransferByBinding = [&](std::uint64_t bindingOrdinal,
                                         std::uint64_t occurrence)
      -> const Diagnostic::Transfer * {
    for (const Diagnostic::Transfer &transfer : closedWait.transfers)
      if (transfer.bindingOrdinal == bindingOrdinal &&
          transfer.occurrenceOrdinal == occurrence)
        return &transfer;
    return nullptr;
  };
  const auto findResidency = [&](std::uint64_t bindingOrdinal,
                                 std::uint64_t occurrence)
      -> std::pair<std::uint64_t, const ResidencyEntry *> {
    for (std::uint64_t storage = 0; storage != storageCount; ++storage)
      for (const ResidencyEntry &entry : residencies[storage].entries)
        if (entry.bindingOrdinal == bindingOrdinal &&
            entry.occurrenceOrdinal == occurrence)
          return {storage, &entry};
    return {absent, nullptr};
  };

  std::vector<Diagnostic::WaitEdge> edges;
  const auto appendEdge = [&](Diagnostic::WaitEdge edge) {
    edges.push_back(std::move(edge));
  };

  // Operand queue bindings of one actor input, when a queue feeds the channel.
  const auto operandQueueOfInput = [&](std::uint64_t actor, std::uint32_t input)
      -> std::optional<std::uint64_t> {
    for (std::size_t ordinal = 0;
         ordinal != closedWait.operandQueueHeads.size(); ++ordinal)
      for (const auto &[consumerActor, consumerInput] :
           closedWait.operandQueueHeads[ordinal].consumers)
        if (consumerActor == actor && consumerInput == input)
          return ordinal;
    return std::nullopt;
  };

  // Actor input waits: a resident awaited token orders behind its queue-class
  // head; an absent awaited token waits on the producer firing.
  for (const Diagnostic::BlockedActorInput &input : closedWait.blockedActorInputs) {
    if (input.sourceKind != Diagnostic::ActorInputSourceKind::ActorResult ||
        input.definingActorOrdinal == absent)
      continue;
    const std::uint64_t waitingOccurrence =
        nextFiring(input.semanticActorOrdinal);
    const std::uint64_t expectedOccurrence =
        input.expectedProducerOccurrenceOrdinal;
    if (waitingOccurrence == absent || expectedOccurrence == absent)
      return fail(Diagnostic::WaitProofFailure::IndeterminateDynamicOwner);
    const OwnerKey waiting =
        actorNode(input.semanticActorOrdinal, waitingOccurrence);

    if (const auto queueOrdinal =
            operandQueueOfInput(input.semanticActorOrdinal,
                                input.inputOrdinal)) {
      Diagnostic::WaitEdge edge;
      edge.from = waiting;
      edge.to = OwnerKey{StorageKey{
          Diagnostic::WaitStorageDomain::OperandQueue, *queueOrdinal,
          QueueClass::global()}};
      edge.kind = EdgeKind::OperandQueueWait;
      edge.waitingInputOrdinal = input.inputOrdinal;
      edge.waitingChannelOrdinal = input.channelOrdinal;
      edge.occurrenceOrdinal = expectedOccurrence;
      appendEdge(std::move(edge));
      continue;
    }

    const Diagnostic::Transfer *awaited =
        findTransfer(input.definingActorOrdinal, expectedOccurrence,
                     input.semanticActorOrdinal, input.inputOrdinal);
    if (!awaited) {
      Diagnostic::WaitEdge edge;
      edge.from = waiting;
      edge.to = actorNode(input.definingActorOrdinal, expectedOccurrence);
      edge.kind = EdgeKind::ActorMissingInput;
      edge.waitingInputOrdinal = input.inputOrdinal;
      edge.waitingChannelOrdinal = input.channelOrdinal;
      edge.occurrenceOrdinal = expectedOccurrence;
      appendEdge(std::move(edge));
      continue;
    }
    const auto [storageOrdinal, entry] =
        findResidency(awaited->bindingOrdinal, awaited->occurrenceOrdinal);
    if (!entry) {
      Diagnostic::WaitEdge edge;
      edge.from = waiting;
      edge.to = actorNode(input.definingActorOrdinal, expectedOccurrence);
      edge.kind = EdgeKind::ActorMissingInput;
      edge.waitingInputOrdinal = input.inputOrdinal;
      edge.waitingChannelOrdinal = input.channelOrdinal;
      edge.bindingOrdinal = awaited->bindingOrdinal;
      edge.occurrenceOrdinal = expectedOccurrence;
      appendEdge(std::move(edge));
      continue;
    }
    const StorageResidency &storage = residencies[storageOrdinal];
    const bool tagged = awaited->physicalTagOrdinal != absent;
    const QueueClass queueClass =
        queueClassOf(storage, awaited->physicalTagValue, tagged);
    const auto position = classPositionOf(storage, queueClass,
                                          awaited->bindingOrdinal,
                                          awaited->occurrenceOrdinal);
    const ResidencyEntry *head = classHead(storage, queueClass);
    if (!position || !head)
      return fail(Diagnostic::WaitProofFailure::IndeterminateDynamicOwner);
    // The awaited token waits for its queue class to advance. Position zero
    // means it is itself the class head: the queue must deliver it while the
    // queue's own out-edges state what the head waits behind.
    Diagnostic::WaitEdge edge;
    edge.from = waiting;
    edge.to = storageNode(storageOrdinal, queueClass);
    edge.kind = EdgeKind::StorageOrder;
    edge.waitingInputOrdinal = input.inputOrdinal;
    edge.waitingChannelOrdinal = input.channelOrdinal;
    edge.bindingOrdinal = awaited->bindingOrdinal;
    edge.occurrenceOrdinal = awaited->occurrenceOrdinal;
    edge.storageOrdinal = storageOrdinal;
    edge.fifoOccurrence = awaited->blockingFifoOccurrence;
    edge.awaitedClassPosition = *position;
    if (tagged)
      edge.awaitedTagValue = awaited->physicalTagValue;
    if (head->tagged)
      edge.headTagValue = head->tagValue;
    edge.headBindingOrdinal = head->bindingOrdinal;
    edge.headOccurrenceOrdinal = head->occurrenceOrdinal;
    if (!head->destinationActorOrdinals.empty()) {
      edge.headDestinationActorOrdinal = head->destinationActorOrdinals.front();
      edge.headDestinationChannelOrdinal =
          head->destinationChannelOrdinals.front();
      edge.headDestinationInputOrdinal = head->destinationInputOrdinals.front();
    }
    appendEdge(std::move(edge));
  }

  // Transfer-side waits: output backpressure toward a full queue, downstream
  // capacity between queues, and terminal consumption of a class head.
  for (const Diagnostic::Transfer &transfer : closedWait.transfers) {
    const bool tagged = transfer.physicalTagOrdinal != absent;
    if (transfer.producerActorOrdinal == absent)
      continue;
    const OwnerKey producer =
        actorNode(transfer.producerActorOrdinal, transfer.occurrenceOrdinal);
    // Output backpressure: the transfer has not been durably accepted by the
    // storage it waits to enter.
    if (transfer.blockingTraversalWaitingForStorage &&
        transfer.blockingStorageOrdinal != absent) {
      const auto discipline = transfer.blockingStorageOrdinal < storageCount
                                  ? residencies[transfer.blockingStorageOrdinal]
                                        .discipline
                                  : ::fabric::FifoQueueDiscipline::StrictFifo;
      Diagnostic::WaitEdge edge;
      edge.from = producer;
      edge.to = storageNode(
          transfer.blockingStorageOrdinal,
          discipline == ::fabric::FifoQueueDiscipline::PerTagVirtualChannel &&
                  tagged
              ? QueueClass::tag(transfer.physicalTagValue)
              : QueueClass::global());
      edge.kind = EdgeKind::ActorOutputBackpressure;
      edge.bindingOrdinal = transfer.bindingOrdinal;
      edge.occurrenceOrdinal = transfer.occurrenceOrdinal;
      edge.storageOrdinal = transfer.blockingStorageOrdinal;
      edge.fifoOccurrence = transfer.blockingFifoOccurrence;
      edge.storageCapacity = transfer.blockingStorageCapacity;
      edge.storageOccupancy = transfer.blockingStorageOccupancy;
      appendEdge(std::move(edge));
    }
    // The transfer cannot publish into a channel the consumer has not
    // drained: the channel slot is the durable acceptance point, and the
    // consumer's next firing owns the outstanding token.
    if (transfer.blockingActorOrdinal != absent) {
      const std::uint64_t consumerOccurrence =
          nextFiring(transfer.blockingActorOrdinal);
      if (consumerOccurrence == absent)
        return fail(Diagnostic::WaitProofFailure::IndeterminateDynamicOwner);
      Diagnostic::WaitEdge channelEdge;
      channelEdge.from = producer;
      channelEdge.to =
          actorNode(transfer.blockingActorOrdinal, consumerOccurrence);
      channelEdge.kind = EdgeKind::ActorOutputBackpressure;
      channelEdge.bindingOrdinal = transfer.bindingOrdinal;
      channelEdge.occurrenceOrdinal = transfer.occurrenceOrdinal;
      appendEdge(std::move(channelEdge));
    }
    // A transfer blocked on operand-queue capacity waits on the queue that
    // cannot admit it.
    if (transfer.operandCapacityBlocked)
      for (const auto &wait : transfer.operandQueueWaits) {
        for (std::size_t ordinal = 0;
             ordinal != closedWait.operandQueueHeads.size(); ++ordinal) {
          const auto &queueHead = closedWait.operandQueueHeads[ordinal];
          if (!(queueHead.queue == wait.queue) ||
              queueHead.allocationUnit != wait.allocationUnit)
            continue;
          Diagnostic::WaitEdge queueEdge;
          queueEdge.from = producer;
          queueEdge.to = OwnerKey{StorageKey{
              Diagnostic::WaitStorageDomain::OperandQueue, ordinal,
              QueueClass::global()}};
          queueEdge.kind = EdgeKind::ActorOutputBackpressure;
          queueEdge.bindingOrdinal = transfer.bindingOrdinal;
          queueEdge.occurrenceOrdinal = transfer.occurrenceOrdinal;
          appendEdge(std::move(queueEdge));
          break;
        }
      }
    // A resident token whose publication is incomplete holds its producer
    // firing open: behind its queue-class head the firing waits on the queue
    // order, and at an unconsumed head it waits on the queue's delivery.
    if (transfer.publishedSinkCount >= transfer.sinkCount ||
        transfer.sinkCount == 0)
      continue;
    const auto [residentStorage, residentEntry] =
        findResidency(transfer.bindingOrdinal, transfer.occurrenceOrdinal);
    if (!residentEntry)
      continue;
    const StorageResidency &resident = residencies[residentStorage];
    const QueueClass residentClass =
        queueClassOf(resident, transfer.physicalTagValue, tagged);
    const auto position = classPositionOf(resident, residentClass,
                                          transfer.bindingOrdinal,
                                          transfer.occurrenceOrdinal);
    if (!position)
      return fail(Diagnostic::WaitProofFailure::IndeterminateDynamicOwner);
    Diagnostic::WaitEdge edge;
    edge.from = producer;
    edge.to = storageNode(residentStorage, residentClass);
    edge.kind = *position == 0 ? EdgeKind::ActorOutputBackpressure
                               : EdgeKind::StorageOrder;
    edge.bindingOrdinal = transfer.bindingOrdinal;
    edge.occurrenceOrdinal = transfer.occurrenceOrdinal;
    edge.storageOrdinal = residentStorage;
    edge.fifoOccurrence = transfer.blockingFifoOccurrence;
    edge.awaitedClassPosition = *position;
    if (tagged)
      edge.awaitedTagValue = transfer.physicalTagValue;
    if (const ResidencyEntry *head = classHead(resident, residentClass)) {
      if (head->tagged)
        edge.headTagValue = head->tagValue;
      edge.headBindingOrdinal = head->bindingOrdinal;
      edge.headOccurrenceOrdinal = head->occurrenceOrdinal;
    }
    appendEdge(std::move(edge));
  }

  // Queue-side waits of every resident class head: continue downstream when
  // the next storage is full at cycle start, or wait on the terminal consumer
  // that has not taken the head token.
  for (std::uint64_t storageOrdinal = 0; storageOrdinal != storageCount;
       ++storageOrdinal) {
    const StorageResidency &storage = residencies[storageOrdinal];
    llvm::SmallVector<QueueClass, 4> classes;
    for (const ResidencyEntry &entry : storage.entries) {
      const QueueClass queueClass =
          queueClassOf(storage, entry.tagValue, true);
      if (!llvm::is_contained(classes, queueClass))
        classes.push_back(queueClass);
    }
    for (const QueueClass &queueClass : classes) {
      const ResidencyEntry *head = classHead(storage, queueClass);
      if (!head || head->bindingOrdinal == absent)
        continue;
      const Diagnostic::Transfer *transfer =
          findTransferByBinding(head->bindingOrdinal, head->occurrenceOrdinal);
      if (!transfer)
        return fail(Diagnostic::WaitProofFailure::IndeterminateDynamicOwner);
      const OwnerKey from = storageNode(storageOrdinal, queueClass);
      // The head continues into a downstream storage.
      if (transfer->blockingDownstreamStorageOrdinal != absent &&
          transfer->blockingDownstreamStorageOrdinal < storageCount &&
          transfer->blockingDownstreamStorageOccupancy +
                  transfer->blockingDownstreamStorageReservations >=
              transfer->blockingDownstreamStorageCapacity) {
        const std::uint64_t downstream =
            transfer->blockingDownstreamStorageOrdinal;
        Diagnostic::WaitEdge edge;
        edge.from = from;
        edge.to = storageNode(
            downstream, queueClassOf(residencies[downstream],
                                     transfer->physicalTagValue,
                                     transfer->physicalTagOrdinal != absent));
        edge.kind = EdgeKind::StorageDownstream;
        edge.bindingOrdinal = transfer->bindingOrdinal;
        edge.occurrenceOrdinal = transfer->occurrenceOrdinal;
        edge.storageOrdinal = downstream;
        edge.storageCapacity = transfer->blockingDownstreamStorageCapacity;
        edge.storageOccupancy = transfer->blockingDownstreamStorageOccupancy;
        if (head->tagged)
          edge.headTagValue = head->tagValue;
        edge.headBindingOrdinal = head->bindingOrdinal;
        edge.headOccurrenceOrdinal = head->occurrenceOrdinal;
        appendEdge(std::move(edge));
        continue;
      }
      // The head reached its route terminal and waits on its consumers. A
      // resident head is unconsumed by construction: the exact consumer
      // firing that must take it is the consumer's next occurrence.
      for (std::size_t destination = 0;
           destination != head->destinationActorOrdinals.size();
           ++destination) {
        const std::uint64_t consumer =
            head->destinationActorOrdinals[destination];
        const std::uint64_t channel =
            head->destinationChannelOrdinals[destination];
        const std::uint64_t consumerOccurrence = nextFiring(consumer);
        if (consumerOccurrence == absent)
          return fail(
              Diagnostic::WaitProofFailure::IndeterminateDynamicOwner);
        Diagnostic::WaitEdge edge;
        edge.from = from;
        edge.to = actorNode(consumer, consumerOccurrence);
        edge.kind = EdgeKind::StorageConsumer;
        edge.bindingOrdinal = head->bindingOrdinal;
        edge.occurrenceOrdinal = head->occurrenceOrdinal;
        edge.storageOrdinal = storageOrdinal;
        if (head->tagged)
          edge.headTagValue = head->tagValue;
        edge.headBindingOrdinal = head->bindingOrdinal;
        edge.headOccurrenceOrdinal = head->occurrenceOrdinal;
        edge.headDestinationActorOrdinal = consumer;
        edge.headDestinationChannelOrdinal = channel;
        edge.headDestinationInputOrdinal =
            head->destinationInputOrdinals[destination];
        appendEdge(std::move(edge));
      }
    }
  }

  // Operand queue out-edges: a non-empty queue waits on the consumer of its
  // head. The supply side reuses the consumer's own blocked-input facts: the
  // queue's next awaited token is exactly the token its consumer awaits.
  for (std::size_t ordinal = 0; ordinal != closedWait.operandQueueHeads.size();
       ++ordinal) {
    const auto &queueHead = closedWait.operandQueueHeads[ordinal];
    const OwnerKey queueNode =
        OwnerKey{StorageKey{Diagnostic::WaitStorageDomain::OperandQueue,
                            ordinal, QueueClass::global()}};
    if (queueHead.occupancy != 0)
      for (const auto &[consumer, input] : queueHead.consumers) {
        const std::uint64_t consumerOccurrence = nextFiring(consumer);
        if (consumerOccurrence == absent)
          return fail(
              Diagnostic::WaitProofFailure::IndeterminateDynamicOwner);
        Diagnostic::WaitEdge edge;
        edge.from = queueNode;
        edge.to = actorNode(consumer, consumerOccurrence);
        edge.kind = EdgeKind::StorageConsumer;
        edge.bindingOrdinal = queueHead.headBindingOrdinal;
        edge.occurrenceOrdinal = queueHead.headOccurrenceOrdinal;
        edge.headDestinationActorOrdinal = consumer;
        edge.headDestinationInputOrdinal = input;
        appendEdge(std::move(edge));
      }
    for (const auto &[consumer, input] : queueHead.consumers) {
      const Diagnostic::BlockedActorInput *blockedInput = nullptr;
      for (const auto &input_ : closedWait.blockedActorInputs)
        if (input_.semanticActorOrdinal == consumer &&
            input_.inputOrdinal == input)
          blockedInput = &input_;
      if (!blockedInput ||
          blockedInput->definingActorOrdinal == absent ||
          blockedInput->expectedProducerOccurrenceOrdinal == absent)
        continue;
      const std::uint64_t awaitedOccurrence =
          blockedInput->expectedProducerOccurrenceOrdinal;
      const Diagnostic::Transfer *awaited =
          findTransfer(blockedInput->definingActorOrdinal, awaitedOccurrence,
                       consumer, input);
      if (awaited) {
        const auto [storageOrdinal, entry] =
            findResidency(awaited->bindingOrdinal, awaited->occurrenceOrdinal);
        if (entry) {
          const StorageResidency &storage = residencies[storageOrdinal];
          const bool tagged = awaited->physicalTagOrdinal != absent;
          Diagnostic::WaitEdge edge;
          edge.from = queueNode;
          edge.to = storageNode(
              storageOrdinal,
              queueClassOf(storage, awaited->physicalTagValue, tagged));
          edge.kind = EdgeKind::StorageOrder;
          edge.bindingOrdinal = awaited->bindingOrdinal;
          edge.occurrenceOrdinal = awaited->occurrenceOrdinal;
          edge.storageOrdinal = storageOrdinal;
          appendEdge(std::move(edge));
          continue;
        }
      }
      Diagnostic::WaitEdge edge;
      edge.from = queueNode;
      edge.to = actorNode(blockedInput->definingActorOrdinal,
                          awaitedOccurrence);
      edge.kind = EdgeKind::ActorMissingInput;
      edge.occurrenceOrdinal = awaitedOccurrence;
      appendEdge(std::move(edge));
    }
  }

  if (closedWait.waitProofFailure)
    return;

  if (debugCertificate) {
    llvm::errs() << "wait-certificate debug: edges=" << edges.size() << '\n';
    for (const Diagnostic::WaitEdge &edge : edges) {
      llvm::errs() << "  edge ";
      if (const auto *firing = std::get_if<0>(&edge.from.owner))
        llvm::errs() << "a" << firing->semanticActorOrdinal << "/"
                     << firing->occurrenceOrdinal;
      else {
        const auto &queue = std::get<1>(edge.from.owner);
        llvm::errs() << "s" << queue.ordinal
                     << (queue.queueClass.tagLocal ? "/t" : "/g");
        if (queue.queueClass.tagLocal) {
          llvm::SmallString<24> text;
          queue.queueClass.tagValue.toStringUnsigned(text, 10);
          llvm::errs() << text;
        }
      }
      llvm::errs() << " -> ";
      if (const auto *firing = std::get_if<0>(&edge.to.owner))
        llvm::errs() << "a" << firing->semanticActorOrdinal << "/"
                     << firing->occurrenceOrdinal;
      else {
        const auto &queue = std::get<1>(edge.to.owner);
        llvm::errs() << "s" << queue.ordinal
                     << (queue.queueClass.tagLocal ? "/t" : "/g");
        if (queue.queueClass.tagLocal) {
          llvm::SmallString<24> text;
          queue.queueClass.tagValue.toStringUnsigned(text, 10);
          llvm::errs() << text;
        }
      }
      llvm::errs() << " kind=" << static_cast<unsigned>(edge.kind)
                   << " binding=" << edge.bindingOrdinal << '\n';
    }
  }

  // Canonical node/edge order, then the closed sink strongly connected
  // component: every node of the certificate waits inside the component and
  // no edge leaves it. Among candidates the one with the smallest minimum
  // node key is selected, so the certificate is a function of the graph.
  llvm::sort(edges, [](const Diagnostic::WaitEdge &lhs,
                       const Diagnostic::WaitEdge &rhs) {
    return std::tie(lhs.from, lhs.to, lhs.kind, lhs.bindingOrdinal,
                    lhs.occurrenceOrdinal, lhs.waitingChannelOrdinal) <
           std::tie(rhs.from, rhs.to, rhs.kind, rhs.bindingOrdinal,
                    rhs.occurrenceOrdinal, rhs.waitingChannelOrdinal);
  });
  edges.erase(std::unique(edges.begin(),
                          edges.end(),
                          [](const Diagnostic::WaitEdge &lhs,
                             const Diagnostic::WaitEdge &rhs) {
                            return lhs.from == rhs.from && lhs.to == rhs.to &&
                                   lhs.kind == rhs.kind &&
                                   lhs.bindingOrdinal == rhs.bindingOrdinal &&
                                   lhs.occurrenceOrdinal ==
                                       rhs.occurrenceOrdinal;
                          }),
              edges.end());

  std::vector<OwnerKey> nodes;
  for (const Diagnostic::WaitEdge &edge : edges) {
    nodes.push_back(edge.from);
    nodes.push_back(edge.to);
  }
  llvm::sort(nodes);
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  if (nodes.empty())
    return fail(Diagnostic::WaitProofFailure::NoClosedComponent);

  std::map<OwnerKey, std::uint32_t> indexOf;
  for (std::uint32_t index = 0; index != nodes.size(); ++index)
    indexOf[nodes[index]] = index;
  std::vector<std::vector<std::uint32_t>> outgoing(nodes.size());
  for (const Diagnostic::WaitEdge &edge : edges)
    outgoing[indexOf[edge.from]].push_back(indexOf[edge.to]);

  // Iterative Tarjan over the canonically ordered graph.
  constexpr std::uint32_t absent32 =
      std::numeric_limits<std::uint32_t>::max();
  std::vector<std::vector<std::uint32_t>> components;
  {
    std::vector<std::uint32_t> index(nodes.size(), absent32), low(nodes.size());
    std::vector<bool> onStack(nodes.size(), false);
    std::vector<std::uint32_t> stack;
    std::uint32_t nextIndex = 0;
    for (std::uint32_t root = 0; root != nodes.size(); ++root) {
      if (index[root] != absent32)
        continue;
      std::vector<std::pair<std::uint32_t, std::uint32_t>> work{{root, 0}};
      std::vector<std::uint32_t> callStack;
      while (!work.empty()) {
        auto &[node, child] = work.back();
        if (child == 0) {
          index[node] = low[node] = nextIndex++;
          stack.push_back(node);
          onStack[node] = true;
          callStack.push_back(node);
        }
        if (child < outgoing[node].size()) {
          const std::uint32_t target = outgoing[node][child];
          ++work.back().second;
          if (index[target] == absent32) {
            work.push_back({target, 0});
            continue;
          }
          if (onStack[target])
            low[node] = std::min(low[node], index[target]);
          continue;
        }
        if (callStack.size() >= 2) {
          const std::uint32_t parent = callStack[callStack.size() - 2];
          low[parent] = std::min(low[parent], low[node]);
        }
        callStack.pop_back();
        if (low[node] == index[node]) {
          std::vector<std::uint32_t> component;
          std::uint32_t member;
          do {
            member = stack.back();
            stack.pop_back();
            onStack[member] = false;
            component.push_back(member);
          } while (member != node);
          llvm::sort(component);
          components.push_back(std::move(component));
        }
        work.pop_back();
      }
    }
  }

  const auto isClosedSink = [&](const std::vector<std::uint32_t> &component) {
    llvm::SmallBitVector member(nodes.size(), false);
    for (std::uint32_t node : component)
      member.set(node);
    for (std::uint32_t node : component) {
      bool internalWait = false;
      for (std::uint32_t target : outgoing[node]) {
        if (!member.test(target))
          return false;
        internalWait = true;
      }
      if (!internalWait)
        return false;
    }
    return true;
  };
  std::optional<std::vector<std::uint32_t>> selected;
  for (const std::vector<std::uint32_t> &component : components) {
    if (!isClosedSink(component))
      continue;
    if (!selected || component.front() < selected->front())
      selected = component;
  }
  if (!selected) {
    if (debugCertificate) {
      llvm::errs() << "wait-certificate debug: nodes=" << nodes.size()
                   << " components=" << components.size() << '\n';
      for (const auto &component : components) {
        llvm::errs() << "  component size=" << component.size() << ":";
        for (std::uint32_t node : component) {
          llvm::errs() << " ";
          const OwnerKey &key = nodes[node];
          if (const auto *firing = std::get_if<0>(&key.owner))
            llvm::errs() << "a" << firing->semanticActorOrdinal << "/"
                         << firing->occurrenceOrdinal;
          else
            llvm::errs() << "s" << std::get<1>(key.owner).ordinal;
        }
        llvm::errs() << '\n';
      }
    }
    return fail(Diagnostic::WaitProofFailure::NoClosedComponent);
  }

  llvm::SmallBitVector member(nodes.size(), false);
  for (std::uint32_t node : *selected)
    member.set(node);
  for (const Diagnostic::WaitEdge &edge : edges)
    if (member.test(indexOf[edge.from]))
      closedWait.waitCertificate.push_back(edge);
}

llvm::Expected<std::vector<CgraClosedWaitSetDiagnostic::ActorWaitCycleEdge>>
deriveActorWaitCycle(
    const detail::PreparedGraphExecution &execution,
    const detail::SimulatorState &state,
    llvm::ArrayRef<std::optional<detail::ActorTransitionProbeResult>> probes,
    const CgraClosedWaitSetDiagnostic &closedWait) {
  const std::uint64_t absent = std::numeric_limits<std::uint64_t>::max();
  const std::size_t actorCount = execution.actorPlans.size();
  if (probes.size() != actorCount)
    return invalid("CGRA actor probe table does not cover the graph");
  llvm::DenseMap<mlir::Operation *, std::uint64_t> actorByOperation;
  actorByOperation.reserve(actorCount);
  for (const auto [ordinal, actor] : llvm::enumerate(execution.actorPlans))
    actorByOperation.try_emplace(actor.operation, ordinal);

  std::vector<bool> active(actorCount, false);
  for (const auto &firing : closedWait.actorFirings)
    if (firing.semanticActorOrdinal < active.size())
      active[firing.semanticActorOrdinal] = true;

  std::vector<ActorWaitState> waits(actorCount);
  for (const auto &transfer : closedWait.transfers) {
    if (!transfer.blocked || transfer.producerActorOrdinal >= actorCount)
      continue;
    ActorWaitState &wait = waits[transfer.producerActorOrdinal];
    if (transfer.blockingActorOrdinal < actorCount)
      wait.outputBackpressure.push_back(transfer.blockingActorOrdinal);
    const auto appendStorageOwner = [&](const auto &head) {
      if (!head || (head->bindingOrdinal == transfer.bindingOrdinal &&
                    head->occurrenceOrdinal == transfer.occurrenceOrdinal))
        return;
      const auto owner =
          llvm::find_if(closedWait.transfers, [&](const auto &c) {
            return c.bindingOrdinal == head->bindingOrdinal &&
                   c.occurrenceOrdinal == head->occurrenceOrdinal;
          });
      if (owner != closedWait.transfers.end() &&
          owner->producerActorOrdinal < actorCount)
        wait.outputBackpressure.push_back(owner->producerActorOrdinal);
    };
    if (transfer.blockingTraversalWaitingForStorage)
      appendStorageOwner(transfer.blockingStorageHead);
    appendStorageOwner(transfer.blockingDownstreamStorageHead);
  }
  for (ActorWaitState &wait : waits) {
    llvm::sort(wait.outputBackpressure);
    wait.outputBackpressure.erase(std::unique(wait.outputBackpressure.begin(),
                                              wait.outputBackpressure.end()),
                                  wait.outputBackpressure.end());
  }

  for (std::size_t ordinal = 0; ordinal != actorCount; ++ordinal) {
    ActorWaitState &wait = waits[ordinal];
    if (active[ordinal] || !wait.outputBackpressure.empty()) {
      wait.usesOutputBackpressure = true;
      wait.eligible = !wait.outputBackpressure.empty();
      continue;
    }
    const auto &actor = execution.actorPlans[ordinal];
    if (!probes[ordinal])
      continue;
    const detail::ActorTransitionProbeResult &selected = *probes[ordinal];
    if (selected.readiness != detail::ActorTransitionReadiness::Blocked)
      continue;

    ActorWaitCase blockedCase;
    bool hasUnownedMissingInput = false;
    bool hasMissingInput = false;
    for (std::uint32_t input : selected.shape.requiredInputs) {
      if (input >= actor.inputChannelCount)
        return invalid("CGRA actor probe names an unknown required input");
      const std::uint64_t channel = actor.firstInputChannel + input;
      if (channel >= state.channelSlots.size())
        return invalid("CGRA actor input channel is outside runtime state");
      if (!state.channelSlots[channel].ready.empty())
        continue;
      mlir::Value value = actor.operation->getOperand(input);
      if (actor.memory && input == actor.memory->memoryOperandOrdinal &&
          (state.memoryViews.contains(value) || state.memories.contains(value)))
        continue;
      hasMissingInput = true;
      mlir::Operation *producer = value.getDefiningOp();
      const auto found =
          producer ? actorByOperation.find(producer) : actorByOperation.end();
      if (found == actorByOperation.end()) {
        hasUnownedMissingInput = true;
        continue;
      }
      blockedCase.internalProducers.push_back(found->second);
    }
    llvm::sort(blockedCase.internalProducers);
    blockedCase.internalProducers.erase(
        std::unique(blockedCase.internalProducers.begin(),
                    blockedCase.internalProducers.end()),
        blockedCase.internalProducers.end());
    wait.eligible = hasMissingInput && !hasUnownedMissingInput &&
                    !blockedCase.internalProducers.empty();
    wait.missingInputCases.push_back(std::move(blockedCase));
  }

  std::vector<bool> closed(actorCount, false);
  for (std::size_t actor = 0; actor != actorCount; ++actor)
    closed[actor] = waits[actor].eligible;
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t actor = 0; actor != actorCount; ++actor) {
      if (!closed[actor])
        continue;
      const ActorWaitState &wait = waits[actor];
      bool internallyBlocked = false;
      if (wait.usesOutputBackpressure) {
        internallyBlocked =
            llvm::any_of(wait.outputBackpressure,
                         [&](std::uint64_t target) { return closed[target]; });
      } else {
        internallyBlocked = llvm::all_of(
            wait.missingInputCases, [&](const ActorWaitCase &blockedCase) {
              return llvm::any_of(
                  blockedCase.internalProducers,
                  [&](std::uint64_t producer) { return closed[producer]; });
            });
      }
      if (!internallyBlocked) {
        closed[actor] = false;
        changed = true;
      }
    }
  }

  using Edge =
      std::pair<std::uint64_t, CgraClosedWaitSetDiagnostic::ActorWaitKind>;
  std::vector<std::vector<Edge>> edges(actorCount);
  for (std::size_t actor = 0; actor != actorCount; ++actor) {
    if (!closed[actor])
      continue;
    const ActorWaitState &wait = waits[actor];
    if (wait.usesOutputBackpressure) {
      for (std::uint64_t target : wait.outputBackpressure)
        if (closed[target])
          edges[actor].push_back(
              {target,
               CgraClosedWaitSetDiagnostic::ActorWaitKind::OutputBackpressure});
    } else {
      for (const ActorWaitCase &blockedCase : wait.missingInputCases)
        for (std::uint64_t producer : blockedCase.internalProducers)
          if (closed[producer])
            edges[actor].push_back(
                {producer,
                 CgraClosedWaitSetDiagnostic::ActorWaitKind::MissingInput});
    }
    llvm::sort(edges[actor], [](const Edge &lhs, const Edge &rhs) {
      return std::tie(lhs.first, lhs.second) < std::tie(rhs.first, rhs.second);
    });
    edges[actor].erase(std::unique(edges[actor].begin(), edges[actor].end()),
                       edges[actor].end());
  }

  std::vector<std::uint8_t> visitState(actorCount, 0);
  std::vector<std::uint64_t> stack;
  std::vector<std::uint64_t> stackPosition(actorCount, absent);
  std::vector<std::uint64_t> cycle;
  std::function<bool(std::uint64_t)> visit = [&](std::uint64_t actor) {
    visitState[actor] = 1;
    stackPosition[actor] = stack.size();
    stack.push_back(actor);
    for (const Edge &edge : edges[actor]) {
      const std::uint64_t target = edge.first;
      if (visitState[target] == 0) {
        if (visit(target))
          return true;
        continue;
      }
      if (visitState[target] != 1)
        continue;
      cycle.assign(stack.begin() + stackPosition[target], stack.end());
      cycle.push_back(target);
      return true;
    }
    stack.pop_back();
    stackPosition[actor] = absent;
    visitState[actor] = 2;
    return false;
  };
  for (std::uint64_t actor = 0; actor != actorCount; ++actor)
    if (closed[actor] && visitState[actor] == 0 && visit(actor))
      break;

  std::vector<CgraClosedWaitSetDiagnostic::ActorWaitCycleEdge> result;
  for (std::size_t index = 1; index < cycle.size(); ++index) {
    const std::uint64_t waiting = cycle[index - 1];
    const std::uint64_t blocking = cycle[index];
    const auto selected = llvm::find_if(edges[waiting], [&](const Edge &edge) {
      return edge.first == blocking;
    });
    if (selected == edges[waiting].end())
      return invalid("CGRA actor wait cycle lost its dependency edge");
    result.push_back({waiting, blocking, selected->second});
  }
  return result;
}

} // namespace

/// Independent closure check of one emitted certificate: every owner it names
/// waits inside the certificate, and the certificate is one strongly
/// connected component. Anchor tests call this; the builder's own selection
/// already guarantees these invariants, so this verifier never trusts the
/// builder's internal state.
bool verifyClosedWaitCertificateClosure(
    const CgraClosedWaitSetDiagnostic &closedWait) {
  if (closedWait.waitProofFailure)
    return false;
  return verifyClosedWaitCertificateClosure(closedWait.waitCertificate);
}

bool verifyClosedWaitCertificateClosure(
    llvm::ArrayRef<CgraClosedWaitSetDiagnostic::WaitEdge> edges) {
  using OwnerKey = CgraClosedWaitSetDiagnostic::WaitOwnerKey;
  if (edges.empty())
    return false;
  std::vector<OwnerKey> nodes;
  for (const auto &edge : edges) {
    nodes.push_back(edge.from);
    nodes.push_back(edge.to);
  }
  llvm::sort(nodes);
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  std::map<OwnerKey, std::uint32_t> indexOf;
  for (std::uint32_t index = 0; index != nodes.size(); ++index)
    indexOf[nodes[index]] = index;
  std::vector<std::vector<std::uint32_t>> outgoing(nodes.size());
  std::vector<std::uint32_t> indegree(nodes.size(), 0);
  for (const auto &edge : edges) {
    outgoing[indexOf[edge.from]].push_back(indexOf[edge.to]);
    ++indegree[indexOf[edge.to]];
  }
  for (std::uint32_t node = 0; node != nodes.size(); ++node)
    if (outgoing[node].empty() || indegree[node] == 0)
      return false;
  std::vector<bool> reached(nodes.size(), false);
  std::vector<std::uint32_t> work{0};
  reached[0] = true;
  while (!work.empty()) {
    const std::uint32_t node = work.back();
    work.pop_back();
    for (std::uint32_t target : outgoing[node])
      if (!reached[target]) {
        reached[target] = true;
        work.push_back(target);
      }
  }
  return llvm::all_of(reached, [](bool value) { return value; });
}

struct PreparedCgraWorkloadExecution::Impl final {
  std::shared_ptr<const PreparedCgraExecution::Impl> prepared;
  const detail::PreparedCgraGraph *graphExecution = nullptr;
  detail::ResolvedLaunchContext context;
  ArtifactIdentity workload;
  ArtifactIdentity runtimeInput;

  Impl(std::shared_ptr<const PreparedCgraExecution::Impl> prepared,
       const detail::PreparedCgraGraph &graphExecution,
       detail::ResolvedLaunchContext context, ArtifactIdentity workload,
       ArtifactIdentity runtimeInput)
      : prepared(std::move(prepared)), graphExecution(&graphExecution),
        context(std::move(context)), workload(workload),
        runtimeInput(runtimeInput) {}
};

PreparedCgraWorkloadExecution::PreparedCgraWorkloadExecution(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PreparedCgraWorkloadExecution::PreparedCgraWorkloadExecution(
    PreparedCgraWorkloadExecution &&) noexcept = default;
PreparedCgraWorkloadExecution &PreparedCgraWorkloadExecution::operator=(
    PreparedCgraWorkloadExecution &&) noexcept = default;
PreparedCgraWorkloadExecution::~PreparedCgraWorkloadExecution() = default;

struct CgraExecutionSession::Impl final {
  std::shared_ptr<const PreparedCgraExecution::Impl> preparedOwner;
  const PreparedCgraExecution::Impl *prepared = nullptr;
  const CanonicalSimulationWorkload *workload = nullptr;
  const CanonicalSimulationRuntimeInput *runtimeInput = nullptr;
  const detail::PreparedCgraGraph *graphExecution = nullptr;
  detail::ResolvedLaunchContext context;
  detail::SimulatorState dynamicState;
  std::optional<detail::CgraGraphActivationRuntime> runtime;
  SpatialExecutionSessionState lifecycle =
      SpatialExecutionSessionState::Runnable;
  CgraSimulationCounters counters;
  std::optional<SpatialEventCoordinate> graphRetirement;
  std::optional<SpatialEventCoordinate> lastCoordinate;
  std::optional<CgraClosedWaitSetDiagnostic> closedWait;
  std::optional<SpatialDiagnosticTrace> trace;
  std::map<std::pair<std::uint64_t, std::uint64_t>, SpatialEventCoordinate>
      physicalRequestCoordinates;
  std::map<std::pair<std::uint64_t, std::uint64_t>, SpatialEventCoordinate>
      physicalGrantCoordinates;
  bool resultTaken = false;

  Impl(std::shared_ptr<const PreparedCgraExecution::Impl> prepared,
       const CanonicalSimulationWorkload &workload,
       const CanonicalSimulationRuntimeInput &runtimeInput,
       const detail::PreparedCgraGraph &graphExecution,
       detail::ResolvedLaunchContext context,
       std::optional<TraceCaptureLevel> traceLevel)
      : preparedOwner(std::move(prepared)), prepared(preparedOwner.get()),
        workload(&workload), runtimeInput(&runtimeInput),
        graphExecution(&graphExecution), context(std::move(context)) {
    if (traceLevel)
      trace.emplace(SpatialDiagnosticTrace{*traceLevel, {}});
  }

  llvm::Expected<ActorTransitionOccurrenceRef>
  transitionOccurrence(const detail::CgraActorLifecycleEvent &event) const {
    if (event.semanticActorOrdinal >= graphExecution->actors.size())
      return invalid("CGRA trace actor ordinal is out of range");
    return ActorTransitionOccurrenceRef{
        GraphInvocationOccurrenceRef{0},
        graphExecution->actors[event.semanticActorOrdinal],
        event.occurrenceOrdinal};
  }

  llvm::Expected<TokenOccurrenceRef>
  tokenOccurrence(const detail::CgraTokenPublication &publication) const {
    if (const auto *ingress = std::get_if<::dataflow::GraphIngressTokenRef>(
            &publication.producer)) {
      if (publication.occurrenceOrdinal != publication.producerSequenceOrdinal)
        return invalid("CGRA graph-ingress trace sequence is not dense");
      return TokenOccurrenceRef{GraphIngressTokenOccurrenceRef{
          GraphInvocationOccurrenceRef{0}, *ingress,
          publication.producerSequenceOrdinal}};
    }
    const auto &result =
        std::get<::dataflow::ActorTokenResultRef>(publication.producer);
    return TokenOccurrenceRef{ActorResultTokenOccurrenceRef{
        ActorTransitionOccurrenceRef{GraphInvocationOccurrenceRef{0},
                                     result.actor,
                                     publication.occurrenceOrdinal},
        result.ordinal, publication.producerSequenceOrdinal}};
  }

  llvm::Error captureFrame(const detail::CgraGraphActivationFrame &frame) {
    if (!trace)
      return llvm::Error::success();
    SpatialTraceFrame projected{frame.coordinate, {}};
    projected.events.reserve(
        frame.actorEvents.size() + frame.publications.size() +
        frame.memoryLinearizations.size() + frame.physicalTraceEvents.size());
    for (const detail::CgraActorLifecycleEvent &event : frame.actorEvents) {
      auto transition = transitionOccurrence(event);
      if (!transition)
        return transition.takeError();
      if (event.kind == detail::CgraActorLifecycleKind::Committed)
        projected.events.push_back(ActorCommittedTraceEvent{*transition});
      else
        projected.events.push_back(ActorRetiredTraceEvent{*transition});
    }
    if (trace->level >= TraceCaptureLevel::Semantic) {
      projected.events.insert(projected.events.end(),
                              frame.memoryLinearizations.begin(),
                              frame.memoryLinearizations.end());
      for (const detail::CgraTokenPublication &publication :
           frame.publications) {
        auto occurrence = tokenOccurrence(publication);
        if (!occurrence)
          return occurrence.takeError();
        auto type = prepared->dataflowView.tokenType(publication.producer);
        if (!type)
          return type.takeError();
        auto value = detail::canonicalValueSequenceFromTokens(
            llvm::ArrayRef(publication.token), *type,
            context.graphOp.getOperation());
        if (!value)
          return value.takeError();
        projected.events.push_back(TokenPublishedTraceEvent{
            std::move(*occurrence), std::move(*value)});
      }
    }
    if (trace->level >= TraceCaptureLevel::Microarchitecture)
      projected.events.insert(projected.events.end(),
                              frame.physicalTraceEvents.begin(),
                              frame.physicalTraceEvents.end());
    if (projected.events.empty())
      return llvm::Error::success();
    return appendSpatialTraceFrame(*trace, std::move(projected));
  }

  llvm::Error observeGraphRetirement(const SpatialEventCoordinate &coordinate) {
    if (graphRetirement ||
        !detail::graphCompletionReady(graphExecution->execution, dynamicState))
      return llvm::Error::success();
    graphRetirement = coordinate;
    return llvm::Error::success();
  }

  llvm::Error settleQuiescence() {
    if (runtime->nextCoordinate())
      return invalid("CGRA session quiesced with a scheduled event");
    if (dynamicState.failure != detail::RunFailure::None ||
        !dynamicState.diagnostics.empty()) {
      lifecycle = SpatialExecutionSessionState::Failed;
      return llvm::createStringError(
          std::errc::state_not_recoverable,
          "CGRA execution ended with a semantic provider failure");
    }
    if (graphRetirement && !runtime->hasPendingEvents()) {
      if (llvm::Error error = detail::validateGraphRetirementBoundary(
              context.graphOp, graphExecution->execution, dynamicState)) {
        lifecycle = SpatialExecutionSessionState::Failed;
        return error;
      }
      if (detail::hasPendingVectorGroups(dynamicState)) {
        lifecycle = SpatialExecutionSessionState::Failed;
        return invalid("CGRA execution retired with incomplete vector state");
      }
      lifecycle = SpatialExecutionSessionState::Retired;
      return llvm::Error::success();
    }

    lifecycle = SpatialExecutionSessionState::Halted;
    closedWait.emplace();
    closedWait->pendingActorFirings = runtime->pendingActorFiringCount();
    closedWait->pendingTransfers = runtime->pendingTransferCount();
    closedWait->pendingPhysicalActions = runtime->pendingPhysicalActionCount();
    closedWait->graphRetirementVisible = graphRetirement.has_value();
    closedWait->ownerReferences = CgraExecutionOwnerReferences{
        {::dataflow::canonicalDataflowSchema.identity.str(),
         ::dataflow::canonicalDataflowSchema.version,
         prepared->dataflow.identity()},
        prepared->fabric.reference(),
        prepared->tech.reference(),
        prepared->spatial.reference()};
    const auto &operandProgress = runtime->operandQueueProgress();
    closedWait->operandQueueGroupCount = operandProgress.groupCount;
    closedWait->operandQueuePotentiallyBlockingGroupCount =
        operandProgress.potentiallyBlockingGroupCount;
    closedWait->operandQueueSharedIngressPressure =
        operandProgress.sharedIngressPressure;
    closedWait->operandQueueDistinctIngressCount =
        operandProgress.distinctIngressCount;
    closedWait->operandQueuePairingKeyCount = operandProgress.pairingKeyCount;
    closedWait->operandQueueProgressStatus =
        static_cast<std::uint8_t>(operandProgress.status);
    closedWait->operandQueueProgressSupport =
        static_cast<std::uint8_t>(operandProgress.support);
    closedWait->operandQueueProjectionDigest = operandProgress.projectionDigest;
    for (const auto &head : runtime->pendingOperandQueueHeadDiagnostics())
      closedWait->operandQueueHeads.push_back(
          {head.queue, head.fu, head.allocationUnit, head.capacity,
           head.occupancy, head.reservations, head.headBindingOrdinal,
           head.headOccurrenceOrdinal, head.headProducerSequenceOrdinal,
           head.headTag, head.exactHead, head.consumers});
    for (auto &rotation : runtime->exhaustedOfferRotationDiagnostics())
      closedWait->exhaustedOfferRotations.push_back(
          {rotation.storageOrdinal, rotation.fifoOccurrence,
           rotation.residentChannelCount, rotation.refusedOffersSinceCommit,
           rotation.occupancy, rotation.capacity,
           std::move(rotation.residentTagValues)});
    for (const auto &firing : runtime->pendingActorFiringDiagnostics())
      closedWait->actorFirings.push_back(
          {firing.semanticActorOrdinal, firing.occurrenceOrdinal,
           firing.transitionCaseOrdinal, firing.expectedTransfers,
           firing.completedTransfers, firing.physicalComplete,
           firing.causalReleaseSatisfied});
    const auto projectStorageHead =
        [](const std::optional<
            detail::CgraPendingTransferDiagnostic::StorageHead> &head)
        -> std::optional<CgraClosedWaitSetDiagnostic::Transfer::StorageHead> {
      if (!head)
        return std::nullopt;
      return CgraClosedWaitSetDiagnostic::Transfer::StorageHead{
          head->storageOrdinal, head->bindingOrdinal, head->occurrenceOrdinal,
          head->traversalNodeOrdinal};
    };
    for (const auto &transfer : runtime->pendingTransferDiagnostics()) {
      std::vector<CgraClosedWaitSetDiagnostic::Transfer::OperandQueueWait>
          operandQueueWaits;
      operandQueueWaits.reserve(transfer.operandQueueWaits.size());
      for (const auto &wait : transfer.operandQueueWaits)
        operandQueueWaits.push_back(
            {wait.queue, wait.fu, wait.ingress, wait.tag, wait.allocationUnit,
             wait.occupancy, wait.reservations, wait.capacity});
      closedWait->transfers.push_back(
          {transfer.bindingOrdinal,
           transfer.occurrenceOrdinal,
           transfer.producerActorOrdinal,
           transfer.producerResultOrdinal,
           transfer.physicalTagOrdinal,
           transfer.physicalTagValue,
           transfer.blocked,
           transfer.arrivalScheduled,
           transfer.publicationReady,
           transfer.published,
           transfer.consumedRequested,
           transfer.operandCapacityReserved,
           transfer.operandCapacityBlocked,
           transfer.producedPermitted,
           transfer.producedRetired,
           transfer.traversalPermitted,
           transfer.traversalRetired,
           transfer.traversalTerminalsPermitted,
           transfer.consumedPermitted,
           transfer.consumedRetired,
           transfer.readySinkCount,
           transfer.publishedSinkCount,
           transfer.sinkCount,
           transfer.publicationCount,
           transfer.requestedPublicationCount,
           transfer.publishedPublicationCount,
           transfer.unpublishedActorOrdinals,
           transfer.unpublishedInputOrdinals,
           transfer.unpublishedReadyTokenCounts,
           transfer.blockingTraversalNodeOrdinal,
           transfer.blockingStorageOrdinal,
           transfer.blockingFifoOccurrence,
           transfer.blockingStorageOccupancy,
           transfer.blockingStorageReservations,
           transfer.blockingStorageCapacity,
           projectStorageHead(transfer.blockingStorageHead),
           transfer.blockingTraversalWaitingForStorage,
           transfer.blockingDownstreamStorageCount,
           transfer.blockingUnbufferedSinkCount,
           transfer.blockingDownstreamStorageOrdinal,
           transfer.blockingDownstreamStorageOccupancy,
           transfer.blockingDownstreamStorageReservations,
           transfer.blockingDownstreamStorageCapacity,
           transfer.blockingDownstreamStorageReserved,
           projectStorageHead(transfer.blockingDownstreamStorageHead),
           transfer.blockingActorOrdinal,
           transfer.blockingReadyTokenCount,
           transfer.blockingQueueOccupancy,
           transfer.blockingQueueReservations,
           transfer.blockingQueueCapacity,
           std::move(operandQueueWaits),
           transfer.producer,
           transfer.blockingTraversals,
           transfer.blockingDownstreamTraversals,
           transfer.physicalTagOwner});
    }
    const std::size_t actorCount = graphExecution->execution.actorPlans.size();
    ActorTransitionProbeTable probes =
        deriveActorTransitionProbes(graphExecution->execution, dynamicState);
    llvm::DenseMap<mlir::Operation *, std::uint64_t> actorByOperation;
    actorByOperation.reserve(actorCount);
    for (const auto [ordinal, actor] :
         llvm::enumerate(graphExecution->execution.actorPlans))
      if (actor.operation)
        actorByOperation.try_emplace(actor.operation, ordinal);
    for (std::size_t actor = 0; actor != actorCount; ++actor) {
      if (!probes[actor] ||
          probes[actor]->readiness != detail::ActorTransitionReadiness::Blocked)
        continue;
      const auto &plan = graphExecution->execution.actorPlans[actor];
      if (!plan.operation)
        continue;
      const auto actorEntity =
          plan.operation->getAttrOfType<dataflow::EntityIdAttr>(
              dataflow::kEntityIdAttrName);
      const std::uint64_t actorEntityId =
          actorEntity ? actorEntity.getId()
                      : detail::invalidCgraTransportOrdinal;
      for (std::uint32_t input : probes[actor]->shape.requiredInputs) {
        const std::uint64_t channel = plan.firstInputChannel + input;
        if (channel >= dynamicState.channelSlots.size())
          continue;
        if (!dynamicState.channelSlots[channel].ready.empty())
          continue;
        mlir::Value value = plan.operation->getOperand(input);
        CgraClosedWaitSetDiagnostic::ActorInputSourceKind sourceKind =
            CgraClosedWaitSetDiagnostic::ActorInputSourceKind::Unknown;
        std::uint64_t definingActor = detail::invalidCgraTransportOrdinal;
        std::uint64_t definingActorEntity = detail::invalidCgraTransportOrdinal;
        bool definingActorTerminal = false;
        const bool memoryCapability =
            plan.memory && input == plan.memory->memoryOperandOrdinal &&
            (dynamicState.memoryViews.contains(value) ||
             dynamicState.memories.contains(value));
        if (memoryCapability)
          continue;
        if (mlir::Operation *producer = value.getDefiningOp()) {
          auto found = actorByOperation.find(producer);
          if (found != actorByOperation.end()) {
            sourceKind =
                CgraClosedWaitSetDiagnostic::ActorInputSourceKind::ActorResult;
            definingActor = found->second;
            if (auto entity = producer->getAttrOfType<dataflow::EntityIdAttr>(
                    dataflow::kEntityIdAttrName))
              definingActorEntity = entity.getId();
            definingActorTerminal =
                probes[definingActor] &&
                probes[definingActor]->readiness ==
                    detail::ActorTransitionReadiness::Terminal;
          }
        } else if (llvm::isa<mlir::BlockArgument>(value)) {
          sourceKind =
              CgraClosedWaitSetDiagnostic::ActorInputSourceKind::GraphInput;
        }
        const std::uint64_t expectedProducerOccurrence =
            runtime->channelArrivalCount(channel).value_or(
                detail::invalidCgraTransportOrdinal);
        closedWait->blockedActorInputs.push_back(
            {static_cast<std::uint64_t>(actor), actorEntityId, input, channel,
             sourceKind, definingActor, definingActorEntity,
             definingActorTerminal, expectedProducerOccurrence});
      }
    }
    for (const auto &action : runtime->pendingPhysicalActionDiagnostics())
      closedWait->physicalActions.push_back(
          {action.action.actionOrdinal, action.action.occurrenceOrdinal,
           static_cast<std::uint8_t>(action.client),
           action.semanticActorOrdinal, action.action.granted,
           action.action.hasCommit, action.action.requiresCausalRelease,
           action.action.intrinsicReleaseReached,
           action.action.causalReleaseReached});
    const std::vector<std::uint64_t> transferCycle =
        findTransferWaitCycle(closedWait->transfers);
    for (std::size_t edge = 1; edge < transferCycle.size(); ++edge) {
      const auto &waiting = closedWait->transfers[transferCycle[edge - 1]];
      const auto &blocking = closedWait->transfers[transferCycle[edge]];
      closedWait->transferWaitCycle.push_back(
          {waiting.bindingOrdinal, waiting.occurrenceOrdinal,
           waiting.blockingActorOrdinal, blocking.bindingOrdinal,
           blocking.occurrenceOrdinal, transferWaitKind(waiting, blocking)});
    }
    auto actorCycle = deriveActorWaitCycle(graphExecution->execution,
                                           dynamicState, probes, *closedWait);
    if (!actorCycle)
      return actorCycle.takeError();
    closedWait->actorWaitCycle = std::move(*actorCycle);
    buildWaitCertificate(*runtime, *closedWait);
    return llvm::Error::success();
  }
};

CgraExecutionSession::CgraExecutionSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CgraExecutionSession::CgraExecutionSession(CgraExecutionSession &&) noexcept =
    default;
CgraExecutionSession &
CgraExecutionSession::operator=(CgraExecutionSession &&) noexcept = default;
CgraExecutionSession::~CgraExecutionSession() = default;

SpatialExecutionSessionState CgraExecutionSession::state() const {
  return impl_ ? impl_->lifecycle : SpatialExecutionSessionState::Failed;
}

const CgraSimulationCounters &CgraExecutionSession::counters() const {
  static const CgraSimulationCounters empty;
  return impl_ ? impl_->counters : empty;
}

const std::optional<CgraClosedWaitSetDiagnostic> &
CgraExecutionSession::closedWaitSet() const {
  static const std::optional<CgraClosedWaitSetDiagnostic> empty;
  return impl_ ? impl_->closedWait : empty;
}

const std::optional<SpatialDiagnosticTrace> &
CgraExecutionSession::diagnosticTrace() const {
  static const std::optional<SpatialDiagnosticTrace> empty;
  return impl_ ? impl_->trace : empty;
}

llvm::Expected<SpatialExecutionSessionState> CgraExecutionSession::advance(
    std::uint64_t maxEventFrames,
    std::optional<std::chrono::steady_clock::time_point> executionDeadline) {
  if (!impl_)
    return invalid("CGRA execution session is empty");
  if (impl_->resultTaken)
    return invalid("CGRA execution result was already taken");
  if (impl_->lifecycle != SpatialExecutionSessionState::Runnable)
    return impl_->lifecycle;
  if (maxEventFrames == 0)
    return invalid("CGRA execution advance requires a positive frame budget");

  std::uint64_t advanced = 0;
  while (advanced != maxEventFrames) {
    if (executionDeadline &&
        std::chrono::steady_clock::now() >= *executionDeadline) {
      impl_->lifecycle = SpatialExecutionSessionState::StoppedByLimit;
      return impl_->lifecycle;
    }

    auto frame = impl_->runtime->advance();
    if (!frame) {
      impl_->lifecycle = SpatialExecutionSessionState::Failed;
      return frame.takeError();
    }
    if (!*frame) {
      if (llvm::Error error = impl_->settleQuiescence())
        return std::move(error);
      return impl_->lifecycle;
    }

    impl_->lastCoordinate = (**frame).coordinate;
    ++impl_->counters.eventFrameCount;
    ++advanced;
    impl_->counters.maximumReferenceCycleNumerator =
        std::max(impl_->counters.maximumReferenceCycleNumerator,
                 (**frame).coordinate.referenceCycle.numerator());
    impl_->counters.maximumEventDelta =
        std::max(impl_->counters.maximumEventDelta, (**frame).coordinate.delta);
    impl_->counters.emptyEventFrameCount +=
        (**frame).physicalEvents.empty() && (**frame).actorEvents.empty() &&
        (**frame).publications.empty() &&
        (**frame).memoryLinearizations.empty();
    impl_->counters.computeSourceFrameCount += ((**frame).sourceMask & 1) != 0;
    impl_->counters.memorySourceFrameCount += ((**frame).sourceMask & 2) != 0;
    impl_->counters.transportSourceFrameCount +=
        ((**frame).sourceMask & 4) != 0;
    impl_->counters.physicalSourceFrameCount += ((**frame).sourceMask & 8) != 0;
    for (const detail::CgraActorLifecycleEvent &event : (**frame).actorEvents) {
      if (event.kind == detail::CgraActorLifecycleKind::Committed)
        ++impl_->counters.actorCommitCount;
      else
        ++impl_->counters.actorRetirementCount;
    }
    impl_->counters.tokenPublicationCount += (**frame).publications.size();
    impl_->counters.memoryLinearizationCount +=
        (**frame).memoryLinearizations.size();
    for (const detail::CgraPhysicalLifecycleEvent &event :
         (**frame).physicalEvents) {
      const auto key =
          std::make_pair(event.actionOrdinal, event.occurrenceOrdinal);
      switch (event.kind) {
      case detail::CgraPhysicalLifecycleKind::Requested:
        ++impl_->counters.physicalRequestCount;
        if (!impl_->physicalRequestCoordinates.emplace(key, event.coordinate)
                 .second) {
          impl_->lifecycle = SpatialExecutionSessionState::Failed;
          return invalid("CGRA physical request was observed twice");
        }
        break;
      case detail::CgraPhysicalLifecycleKind::Granted:
        ++impl_->counters.physicalGrantCount;
        if (auto request = impl_->physicalRequestCoordinates.find(key);
            request == impl_->physicalRequestCoordinates.end()) {
          impl_->lifecycle = SpatialExecutionSessionState::Failed;
          return invalid("CGRA physical grant has no request observation");
        } else {
          auto wait =
              integralReferenceCycleDistance(request->second, event.coordinate);
          if (!wait) {
            ++impl_->counters.nonIntegralTimingObservationCount;
          } else if (*wait == 0) {
            ++impl_->counters.physicalGrantSameCycleCount;
          } else {
            ++impl_->counters.physicalGrantDelayedCount;
            impl_->counters.physicalGrantWaitCycleSum += *wait;
            impl_->counters.physicalGrantWaitCycleMax =
                std::max(impl_->counters.physicalGrantWaitCycleMax, *wait);
          }
        }
        impl_->physicalGrantCoordinates.insert_or_assign(key, event.coordinate);
        break;
      case detail::CgraPhysicalLifecycleKind::Committed:
        break;
      case detail::CgraPhysicalLifecycleKind::Retired:
        ++impl_->counters.physicalRetirementCount;
        if (auto request = impl_->physicalRequestCoordinates.find(key);
            request == impl_->physicalRequestCoordinates.end()) {
          impl_->lifecycle = SpatialExecutionSessionState::Failed;
          return invalid("CGRA physical retirement has no request observation");
        } else if (auto lifetime = integralReferenceCycleDistance(
                       request->second, event.coordinate)) {
          impl_->counters.physicalActionLifetimeCycleSum += *lifetime;
          impl_->counters.physicalActionLifetimeCycleMax = std::max(
              impl_->counters.physicalActionLifetimeCycleMax, *lifetime);
        } else {
          ++impl_->counters.nonIntegralTimingObservationCount;
        }
        if (auto grant = impl_->physicalGrantCoordinates.find(key);
            grant != impl_->physicalGrantCoordinates.end()) {
          if (auto active = integralReferenceCycleDistance(grant->second,
                                                           event.coordinate)) {
            impl_->counters.physicalGrantedLifetimeCycleSum += *active;
            impl_->counters.physicalGrantedLifetimeCycleMax = std::max(
                impl_->counters.physicalGrantedLifetimeCycleMax, *active);
          } else {
            ++impl_->counters.nonIntegralTimingObservationCount;
          }
        } else {
          impl_->lifecycle = SpatialExecutionSessionState::Failed;
          return invalid("CGRA physical retirement has no grant observation");
        }
        impl_->physicalRequestCoordinates.erase(key);
        impl_->physicalGrantCoordinates.erase(key);
        break;
      }
    }
    if (llvm::Error error = impl_->captureFrame(**frame)) {
      impl_->lifecycle = SpatialExecutionSessionState::Failed;
      return std::move(error);
    }
    if (llvm::Error error =
            impl_->observeGraphRetirement((**frame).coordinate)) {
      impl_->lifecycle = SpatialExecutionSessionState::Failed;
      return std::move(error);
    }
    if (!impl_->runtime->hasPendingEvents()) {
      if (llvm::Error error = impl_->settleQuiescence())
        return std::move(error);
      return impl_->lifecycle;
    }
  }
  return impl_->lifecycle;
}

llvm::Expected<RetiredCgraSimulation>
CgraExecutionSession::takeRetiredSimulation() {
  if (!impl_)
    return invalid("CGRA execution session is empty");
  if (impl_->resultTaken)
    return invalid("CGRA execution result was already taken");
  if (impl_->lifecycle != SpatialExecutionSessionState::Retired ||
      !impl_->graphRetirement || !impl_->lastCoordinate)
    return llvm::createStringError(
        std::errc::state_not_recoverable,
        "CGRA execution session has not retired successfully");

  auto observations = detail::projectRetiredFunctionalObservations(
      impl_->context.graphOp, impl_->dynamicState, *impl_->workload,
      *impl_->runtimeInput, impl_->context, impl_->prepared->dataflowView);
  if (!observations)
    return observations.takeError();
  impl_->resultTaken = true;
  auto launch = launchCoordinate();
  if (!launch)
    return launch.takeError();
  return RetiredCgraSimulation{
      std::move(*observations),
      SpatialProgressObservations{std::move(*launch), impl_->graphRetirement,
                                  *impl_->lastCoordinate},
      impl_->counters};
}

llvm::Expected<HaltedCgraSimulation>
CgraExecutionSession::takeHaltedSimulation() {
  if (!impl_)
    return invalid("CGRA execution session is empty");
  if (impl_->resultTaken)
    return invalid("CGRA execution result was already taken");
  if (impl_->lifecycle != SpatialExecutionSessionState::Halted ||
      !impl_->closedWait)
    return llvm::createStringError(
        std::errc::state_not_recoverable,
        "CGRA execution session has no proven Halted result");

  auto observations = detail::projectHaltedFunctionalObservations(
      impl_->context.graphOp, impl_->dynamicState, *impl_->workload,
      *impl_->runtimeInput, impl_->context, impl_->prepared->dataflowView);
  if (!observations)
    return observations.takeError();
  auto launch = launchCoordinate();
  if (!launch)
    return launch.takeError();
  const SpatialEventCoordinate terminal =
      impl_->lastCoordinate ? *impl_->lastCoordinate : *launch;
  impl_->resultTaken = true;
  return HaltedCgraSimulation{
      std::move(*observations),
      SpatialProgressObservations{std::move(*launch), impl_->graphRetirement,
                                  terminal},
      impl_->counters};
}

llvm::Expected<PreparedCgraWorkloadExecution> prepareCgraWorkloadExecution(
    const PreparedCgraExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput) {
  if (!prepared.impl_)
    return invalid("prepared CGRA execution is empty");
  const SpatialSimulationWorkload *spatial = workload.spatial();
  if (!spatial)
    return invalid("CGRA execution session requires a Spatial workload");
  auto graph = admitCgraSpatialSimulation(prepared, workload, runtimeInput);
  if (!graph)
    return graph.takeError();
  auto graphFound = llvm::find_if(
      prepared.impl_->graphs, [&](const detail::PreparedCgraGraph &candidate) {
        return candidate.graph == *graph;
      });
  const detail::PreparedCgraGraph *graphExecution =
      graphFound == prepared.impl_->graphs.end() ? nullptr : &*graphFound;
  if (!graphExecution)
    return invalid("CGRA session has no prepared semantic graph");
  auto context = detail::resolveLaunchContext(prepared.impl_->dataflowView,
                                              spatial->launchRef);
  if (!context)
    return context.takeError();
  if (std::optional<std::string> reason =
          detail::unsupportedTypedDfgInput(workload, runtimeInput, *context))
    return llvm::createStringError(std::errc::not_supported, "%s",
                                   reason->c_str());

  return PreparedCgraWorkloadExecution(
      std::make_unique<PreparedCgraWorkloadExecution::Impl>(
          prepared.impl_, *graphExecution, std::move(*context),
          workload.identity(), runtimeInput.identity()));
}

llvm::Expected<CgraExecutionSession>
startCgraExecutionSession(const PreparedCgraExecution &prepared,
                          const CanonicalSimulationWorkload &workload,
                          const CanonicalSimulationRuntimeInput &runtimeInput,
                          std::optional<TraceCaptureLevel> traceLevel,
                          CgraExternalMemoryProvider *externalMemoryProvider) {
  auto preparedWorkload =
      prepareCgraWorkloadExecution(prepared, workload, runtimeInput);
  if (!preparedWorkload)
    return preparedWorkload.takeError();
  return startCgraExecutionSession(*preparedWorkload, workload, runtimeInput,
                                   traceLevel, externalMemoryProvider);
}

llvm::Expected<CgraExecutionSession>
startCgraExecutionSession(const PreparedCgraWorkloadExecution &prepared,
                          const CanonicalSimulationWorkload &workload,
                          const CanonicalSimulationRuntimeInput &runtimeInput,
                          std::optional<TraceCaptureLevel> traceLevel,
                          CgraExternalMemoryProvider *externalMemoryProvider) {
  if (!prepared.impl_ || !prepared.impl_->prepared ||
      !prepared.impl_->graphExecution)
    return invalid("prepared CGRA workload execution is empty");
  if (prepared.impl_->workload != workload.identity() ||
      prepared.impl_->runtimeInput != runtimeInput.identity())
    return invalid("prepared CGRA workload execution has foreign inputs");
  const SpatialSimulationWorkload *spatial = workload.spatial();
  if (!spatial)
    return invalid("prepared CGRA workload execution is not Spatial");
  const detail::PreparedCgraGraph &graphExecution =
      *prepared.impl_->graphExecution;

  auto impl = std::make_unique<CgraExecutionSession::Impl>(
      prepared.impl_->prepared, workload, runtimeInput, graphExecution,
      prepared.impl_->context, traceLevel);

  llvm::SmallVector<detail::GraphIngressEmission, 4> ingress;
  impl->dynamicState.graphIngressCapture = &ingress;
  llvm::scope_exit clearCapture(
      [&] { impl->dynamicState.graphIngressCapture = nullptr; });
  if (llvm::Error error = detail::initializeTypedGraphExecutionState(
          impl->dynamicState, graphExecution.execution, impl->context.graphOp,
          workload, runtimeInput, impl->context))
    return std::move(error);
  clearCapture.release();
  impl->dynamicState.graphIngressCapture = nullptr;

  auto runtime = detail::CgraGraphActivationRuntime::create(
      prepared.impl_->prepared->executionPlan,
      prepared.impl_->prepared->dataflowView, spatial->launchRef,
      graphExecution.graph, graphExecution.execution, impl->dynamicState,
      traceLevel == TraceCaptureLevel::Microarchitecture,
      externalMemoryProvider);
  if (!runtime)
    return runtime.takeError();
  impl->runtime.emplace(std::move(*runtime));
  auto launch = launchCoordinate();
  if (!launch)
    return launch.takeError();
  if (llvm::Error error = impl->runtime->start(*launch, ingress))
    return std::move(error);
  if (llvm::Error error = impl->observeGraphRetirement(*launch))
    return std::move(error);
  return CgraExecutionSession(std::move(impl));
}

llvm::Expected<CgraSimulationOutcome> simulateCgraWorkload(
    const PreparedCgraExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxEventFrames,
    std::optional<std::chrono::steady_clock::time_point> executionDeadline,
    CgraExternalMemoryProvider *externalMemoryProvider) {
  if (maxEventFrames == 0)
    return invalid("CGRA simulation requires a positive event-frame limit");
  auto preparedWorkload =
      prepareCgraWorkloadExecution(prepared, workload, runtimeInput);
  if (!preparedWorkload)
    return preparedWorkload.takeError();
  return simulateCgraWorkload(*preparedWorkload, workload, runtimeInput,
                              maxEventFrames, executionDeadline,
                              externalMemoryProvider);
}

llvm::Expected<CgraSimulationOutcome> simulateCgraWorkload(
    const PreparedCgraWorkloadExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxEventFrames,
    std::optional<std::chrono::steady_clock::time_point> executionDeadline,
    CgraExternalMemoryProvider *externalMemoryProvider) {
  if (maxEventFrames == 0)
    return invalid("CGRA simulation requires a positive event-frame limit");
  auto session = startCgraExecutionSession(
      prepared, workload, runtimeInput, std::nullopt, externalMemoryProvider);
  if (!session)
    return session.takeError();
  auto advanced = session->advance(maxEventFrames, executionDeadline);
  if (!advanced)
    return advanced.takeError();
  SpatialExecutionSessionState state = *advanced;
  if (state == SpatialExecutionSessionState::Runnable) {
    session->impl_->lifecycle = SpatialExecutionSessionState::StoppedByLimit;
    state = session->impl_->lifecycle;
  }

  CgraSimulationOutcome result;
  result.state = state;
  result.counters = session->counters();
  result.closedWaitSet = session->closedWaitSet();
  if (state == SpatialExecutionSessionState::Retired) {
    auto retired = session->takeRetiredSimulation();
    if (!retired)
      return retired.takeError();
    result.retired = std::move(*retired);
  } else if (state == SpatialExecutionSessionState::Halted) {
    auto halted = session->takeHaltedSimulation();
    if (!halted)
      return halted.takeError();
    result.halted = std::move(*halted);
  }
  return result;
}

} // namespace loom::sim
