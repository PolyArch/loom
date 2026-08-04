#include "Simulator/SpatialTrace.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <system_error>

namespace loom::sim {
namespace {

template <typename... Ts> struct Overloaded : Ts... {
  using Ts::operator()...;
};
template <typename... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

template <typename Ref>
llvm::Error appendDataflowRef(std::vector<std::uint8_t> &bytes,
                              const Ref &reference) {
  auto encoded = ::dataflow::encodeDataflowReference(reference);
  if (!encoded)
    return encoded.takeError();
  bytes.insert(bytes.end(), encoded->begin(), encoded->end());
  return llvm::Error::success();
}

llvm::Error appendTransitionKey(std::vector<std::uint8_t> &bytes,
                                const ActorTransitionOccurrenceRef &ref) {
  appendU64(bytes, ref.invocation.invocationOrdinal);
  if (llvm::Error error = appendDataflowRef(bytes, ref.actor))
    return error;
  appendU64(bytes, ref.transitionOrdinal);
  return llvm::Error::success();
}

llvm::Error appendTokenKey(std::vector<std::uint8_t> &bytes,
                           const TokenOccurrenceRef &ref) {
  appendU32(bytes, static_cast<std::uint32_t>(ref.index()));
  return std::visit(
      Overloaded{
          [&](const GraphIngressTokenOccurrenceRef &ingress) -> llvm::Error {
            appendU64(bytes, ingress.invocation.invocationOrdinal);
            if (llvm::Error error = appendDataflowRef(bytes, ingress.ingress))
              return error;
            appendU64(bytes, ingress.producerSequenceOrdinal);
            return llvm::Error::success();
          },
          [&](const ActorResultTokenOccurrenceRef &result) -> llvm::Error {
            if (llvm::Error error =
                    appendTransitionKey(bytes, result.transition))
              return error;
            appendU64(bytes, result.resultOrdinal);
            appendU64(bytes, result.producerSequenceOrdinal);
            return llvm::Error::success();
          }},
      ref);
}

llvm::Error appendMemoryActionKey(std::vector<std::uint8_t> &bytes,
                                  const MemoryActionOccurrenceRef &ref) {
  if (llvm::Error error = appendTransitionKey(bytes, ref.transition))
    return error;
  appendU32(bytes, static_cast<std::uint32_t>(ref.granularity.index()));
  if (const auto *lane = std::get_if<LaneMemoryActionRef>(&ref.granularity))
    appendU64(bytes, lane->rowMajorOrdinal);
  return llvm::Error::success();
}

llvm::Error appendPhysicalActionKey(std::vector<std::uint8_t> &bytes,
                                    const PhysicalActionOccurrenceRef &ref) {
  appendU32(bytes, static_cast<std::uint32_t>(ref.parent.index()));
  if (const auto *transition =
          std::get_if<TransitionPhysicalActionParent>(&ref.parent)) {
    if (llvm::Error error =
            appendTransitionKey(bytes, transition->transition))
      return error;
  } else if (llvm::Error error = appendTokenKey(
                 bytes, std::get<TokenPhysicalActionParent>(ref.parent).token)) {
    return error;
  }
  appendU64(bytes, ref.localActionOrdinal);
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::uint8_t>>
buildEventKey(const SpatialTraceEvent &event) {
  std::vector<std::uint8_t> bytes;
  appendU32(bytes, static_cast<std::uint32_t>(event.index()));
  llvm::Error error = std::visit(
      Overloaded{
          [&](const ActorCommittedTraceEvent &value) {
            return appendTransitionKey(bytes, value.transition);
          },
          [&](const ActorRetiredTraceEvent &value) {
            return appendTransitionKey(bytes, value.transition);
          },
          [&](const TokenPublishedTraceEvent &value) {
            return appendTokenKey(bytes, value.token);
          },
          [&](const MemoryLinearizedTraceEvent &value) {
            return appendMemoryActionKey(bytes, value.action);
          },
          [&](const PhysicalRequestedTraceEvent &value) {
            return appendPhysicalActionKey(bytes, value.action);
          },
          [&](const PhysicalGrantedTraceEvent &value) {
            return appendPhysicalActionKey(bytes, value.action);
          },
          [&](const PhysicalRetiredTraceEvent &value) {
            return appendPhysicalActionKey(bytes, value.action);
          }},
      event);
  if (error)
    return std::move(error);
  return bytes;
}

llvm::Error validateEvent(const SpatialTraceEvent &event) {
  const auto *requested = std::get_if<PhysicalRequestedTraceEvent>(&event);
  if (!requested)
    return llvm::Error::success();
  const auto *transfer =
      std::get_if<PhysicalTransferTarget>(&requested->target);
  if (!transfer)
    return llvm::Error::success();
  if (transfer->traversals.empty())
    return invalid("physical transfer target must contain a traversal");
  std::optional<std::vector<std::uint8_t>> previous;
  for (const auto &traversal : transfer->traversals) {
    auto current = ::loom::fabric::canonicalFabricBytes(traversal);
    if (previous && *previous >= current)
      return invalid(
          "physical transfer target is not a canonical traversal set");
    previous = std::move(current);
  }
  return llvm::Error::success();
}

} // namespace

TraceCaptureLevel minimumTraceCaptureLevel(const SpatialTraceEvent &event) {
  if (event.index() <= 1)
    return TraceCaptureLevel::Firing;
  if (event.index() <= 3)
    return TraceCaptureLevel::Semantic;
  return TraceCaptureLevel::Microarchitecture;
}

llvm::Expected<std::vector<std::uint8_t>>
canonicalSpatialTraceEventKey(const SpatialTraceEvent &event) {
  return buildEventKey(event);
}

llvm::Error canonicalizeSpatialTraceFrame(SpatialTraceFrame &frame,
                                          TraceCaptureLevel level) {
  if (frame.events.empty())
    return invalid("spatial trace frame must contain at least one event");

  struct KeyedEvent final {
    std::vector<std::uint8_t> key;
    SpatialTraceEvent event;
  };
  std::vector<KeyedEvent> keyed;
  keyed.reserve(frame.events.size());
  for (SpatialTraceEvent &event : frame.events) {
    if (minimumTraceCaptureLevel(event) > level)
      return invalid("spatial trace event exceeds the selected capture level");
    if (llvm::Error error = validateEvent(event))
      return error;
    auto key = canonicalSpatialTraceEventKey(event);
    if (!key)
      return key.takeError();
    keyed.push_back({std::move(*key), std::move(event)});
  }
  llvm::sort(keyed, [](const KeyedEvent &lhs, const KeyedEvent &rhs) {
    return lhs.key < rhs.key;
  });
  for (std::size_t index = 1; index != keyed.size(); ++index)
    if (keyed[index - 1].key == keyed[index].key)
      return invalid("spatial trace frame contains a duplicate event key");
  frame.events.clear();
  frame.events.reserve(keyed.size());
  for (KeyedEvent &entry : keyed)
    frame.events.push_back(std::move(entry.event));
  return llvm::Error::success();
}

llvm::Error appendSpatialTraceFrame(SpatialDiagnosticTrace &trace,
                                    SpatialTraceFrame frame) {
  if (!trace.frames.empty() &&
      compareSpatialEventCoordinates(trace.frames.back().coordinate,
                                     frame.coordinate) >= 0)
    return invalid("spatial trace frame coordinates must strictly increase");
  if (llvm::Error error = canonicalizeSpatialTraceFrame(frame, trace.level))
    return error;
  trace.frames.push_back(std::move(frame));
  return llvm::Error::success();
}

} // namespace loom::sim
