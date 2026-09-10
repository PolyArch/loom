#include "Gem5SystemExecutionInternal.h"

#include "Dataflow/IR/DataflowEventDerivation.h"
#include "llvm/Support/JSON.h"

namespace loom::runtime::gem5_system {
namespace {

std::uint32_t readBigEndianU32(llvm::StringRef bytes, std::size_t offset) {
  std::uint32_t value = 0;
  for (std::size_t index = 0; index != 4; ++index)
    value = (value << 8) | static_cast<unsigned char>(bytes[offset + index]);
  return value;
}

std::uint64_t readBigEndianU64(llvm::StringRef bytes, std::size_t offset) {
  std::uint64_t value = 0;
  for (std::size_t index = 0; index != 8; ++index)
    value = (value << 8) | static_cast<unsigned char>(bytes[offset + index]);
  return value;
}

} // namespace

llvm::Expected<std::vector<sim::SystemRootLifecycleObservation>>
parseRootLifecycleResult(llvm::StringRef bytes, const Gem5SystemFacts &facts) {
  constexpr std::size_t headerBytes = 4;
  constexpr std::size_t recordBytes = 64;
  if (bytes.size() < headerBytes ||
      readBigEndianU32(bytes, 0) != gem5RootLifecycleTraceMagic)
    return invalid("gem5 root lifecycle result has the wrong header");
  if ((bytes.size() - headerBytes) % recordBytes != 0)
    return invalid("gem5 root lifecycle result has a partial record");
  const std::size_t recordCount = (bytes.size() - headerBytes) / recordBytes;

  std::vector<sim::SystemRootLifecycleObservation> observations;
  observations.reserve(recordCount);
  std::optional<bool> acknowledgedMode;
  std::uint64_t lastAcknowledgementGeneration = 0;
  std::uint64_t currentEndpoint = 0;
  for (std::size_t offset = headerBytes; offset != bytes.size();
       offset += recordBytes) {
    const std::uint64_t entity = readBigEndianU64(bytes, offset);
    const std::uint64_t occurrence = readBigEndianU64(bytes, offset + 8);
    const std::uint32_t action = readBigEndianU32(bytes, offset + 16);
    const std::uint64_t tick = readBigEndianU64(bytes, offset + 20);
    const std::uint64_t delta = readBigEndianU64(bytes, offset + 28);
    const std::uint64_t acknowledgementGeneration =
        readBigEndianU64(bytes, offset + 36);
    const std::uint32_t decision = readBigEndianU32(bytes, offset + 44);
    const std::uint64_t endpoint = readBigEndianU64(bytes, offset + 48);
    const std::uint64_t memoryOccupiedTicks =
        readBigEndianU64(bytes, offset + 56);
    if (action >
        static_cast<std::uint32_t>(Gem5RootLifecycleAction::Completion))
      return invalid("gem5 root lifecycle result has an unknown action");
    const bool acknowledged = acknowledgementGeneration != 0;
    if (!acknowledgedMode)
      acknowledgedMode = acknowledged;
    if (*acknowledgedMode != acknowledged)
      return invalid("gem5 root lifecycle result mixes control modes");
    if (acknowledged &&
        acknowledgementGeneration <= lastAcknowledgementGeneration)
      return invalid("gem5 root lifecycle acknowledgements are not increasing");
    if (decision >
        static_cast<std::uint32_t>(Gem5RootEventControlDecision::Reject))
      return invalid("gem5 root lifecycle control decision is invalid");
    const auto controlDecision =
        static_cast<Gem5RootEventControlDecision>(decision);
    if (controlDecision == Gem5RootEventControlDecision::Reject ||
        endpoint >= gem5MaximumStaticDispatchEntries)
      return invalid("gem5 root lifecycle records a rejected endpoint");
    if (action == static_cast<std::uint32_t>(Gem5RootLifecycleAction::Start)) {
      if (controlDecision != Gem5RootEventControlDecision::Continue ||
          endpoint != currentEndpoint)
        return invalid("gem5 root start has a noncanonical control decision");
    } else if (controlDecision == Gem5RootEventControlDecision::Stay) {
      if (endpoint != currentEndpoint)
        return invalid("gem5 root stay changes the active endpoint");
    } else if (controlDecision ==
               Gem5RootEventControlDecision::ActivateEndpoint) {
      currentEndpoint = endpoint;
    } else {
      return invalid(
          "gem5 root completion has a noncanonical control decision");
    }
    if (!acknowledged &&
        (endpoint != 0 ||
         (action == static_cast<std::uint32_t>(Gem5RootLifecycleAction::Start)
              ? controlDecision != Gem5RootEventControlDecision::Continue
              : controlDecision != Gem5RootEventControlDecision::Stay)))
      return invalid("uncontrolled gem5 root lifecycle changed endpoint");
    lastAcknowledgementGeneration = acknowledgementGeneration;
    if (!facts.dataflow)
      return invalid("host-only gem5 execution contains root lifecycle events");
    const dataflow::RootThreadLaunchRef root{
        facts.dataflow->artifact, dataflow::RootThreadLaunchId(entity)};
    const dataflow::EventFamilyKey event =
        action == static_cast<std::uint32_t>(Gem5RootLifecycleAction::Start)
            ? dataflow::rootThreadStartEventFamily(root)
            : dataflow::rootThreadCompletionEventFamily(root);
    observations.push_back(
        {event, occurrence, {tick, delta}, memoryOccupiedTicks});
  }
  return observations;
}

llvm::Expected<Gem5AttemptResult> parseAttemptResult(llvm::StringRef text) {
  auto value = llvm::json::parse(text);
  if (!value)
    return invalid("gem5 result is not valid JSON");
  const llvm::json::Object *object = value->getAsObject();
  if (!object || object->size() != 6)
    return invalid("gem5 result does not have the exact result shape");
  const auto schema = object->getString("schema");
  const auto entry = object->getInteger("entry_tick");
  const auto exit = object->getInteger("exit_tick");
  const auto cause = object->getString("cause");
  if (!schema || *schema != "loom.gem5_system_attempt.3" || !entry || !exit ||
      !cause || *entry < 0 || *exit < 0 || *entry > *exit)
    return invalid("gem5 result fields are invalid");
  const auto *activity = object->getObject("memory_activity");
  if (!activity || activity->size() != 1)
    return invalid("gem5 result has no exact native memory counter shape");
  const auto occupied = activity->getInteger("occupied_ticks");
  if (!occupied || *occupied < 0)
    return invalid("gem5 native memory service counter is invalid");
  const auto *intervals = object->getArray("computation_interval");
  if (!intervals || (!intervals->empty() && intervals->size() != 4))
    return invalid("gem5 computation interval array is malformed");
  std::optional<sim::SystemComputationInterval> computation;
  if (!intervals->empty()) {
    sim::SystemComputationInterval interval;
    std::uint64_t *fields[] = {&interval.beginTick, &interval.endTick,
                               &interval.beginMemoryOccupiedTicks,
                               &interval.endMemoryOccupiedTicks};
    for (std::size_t field = 0; field != 4; ++field) {
      auto value = (*intervals)[field].getAsInteger();
      if (!value || *value < 0)
        return invalid(
            "gem5 computation observation is not a nonnegative integer");
      *fields[field] = static_cast<std::uint64_t>(*value);
    }
    computation = interval;
  }
  return Gem5AttemptResult{static_cast<std::uint64_t>(*entry),
                           static_cast<std::uint64_t>(*exit),
                           cause->str(),
                           {static_cast<std::uint64_t>(*occupied)},
                           std::move(computation)};
}

} // namespace loom::runtime::gem5_system
