#ifndef LOOM_SIMULATOR_SPATIALTRACE_H
#define LOOM_SIMULATOR_SPATIALTRACE_H

#include "Simulator/SimulationExecution.h"

#include "Dataflow/IR/DataflowStructuralRefs.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace loom::sim {

/// Trace capture is an invocation-local diagnostic until the detailed-bundle
/// owner and SimulationExecution schema minor are available.
enum class TraceCaptureLevel : std::uint32_t {
  Firing = 0,
  Semantic = 1,
  Microarchitecture = 2,
};

struct GraphInvocationOccurrenceRef final {
  std::uint64_t invocationOrdinal = 0;
};

struct ActorTransitionOccurrenceRef final {
  GraphInvocationOccurrenceRef invocation;
  ::dataflow::ActorRef actor;
  std::uint64_t transitionOrdinal = 0;
};

struct GraphIngressTokenOccurrenceRef final {
  GraphInvocationOccurrenceRef invocation;
  ::dataflow::GraphIngressTokenRef ingress;
  std::uint64_t producerSequenceOrdinal = 0;
};

struct ActorResultTokenOccurrenceRef final {
  ActorTransitionOccurrenceRef transition;
  std::uint64_t resultOrdinal = 0;
  std::uint64_t producerSequenceOrdinal = 0;
};

using TokenOccurrenceRef =
    std::variant<GraphIngressTokenOccurrenceRef,
                 ActorResultTokenOccurrenceRef>;

struct ActorWideMemoryActionRef final {};
struct LaneMemoryActionRef final {
  std::uint64_t rowMajorOrdinal = 0;
};
using MemoryActionGranularity =
    std::variant<ActorWideMemoryActionRef, LaneMemoryActionRef>;

struct MemoryActionOccurrenceRef final {
  ActorTransitionOccurrenceRef transition;
  MemoryActionGranularity granularity;
};

struct InitialMemoryVersionRef final {};
struct WrittenMemoryVersionRef final {
  MemoryActionOccurrenceRef action;
};
using MemoryVersionRef =
    std::variant<InitialMemoryVersionRef, WrittenMemoryVersionRef>;

struct TransitionPhysicalActionParent final {
  ActorTransitionOccurrenceRef transition;
};
struct TokenPhysicalActionParent final {
  TokenOccurrenceRef token;
};
using PhysicalActionParent =
    std::variant<TransitionPhysicalActionParent, TokenPhysicalActionParent>;

struct PhysicalActionOccurrenceRef final {
  PhysicalActionParent parent;
  std::uint64_t localActionOrdinal = 0;
};

struct PhysicalUseTarget final {
  ::loom::fabric::FabricUsePatternRef usePattern;
};

struct PhysicalTransferTarget final {
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversals;
  std::optional<::loom::fabric::FabricUsePatternRef> usePattern;
};

using PhysicalActionTarget =
    std::variant<PhysicalUseTarget, PhysicalTransferTarget>;

struct ActorCommittedTraceEvent final {
  ActorTransitionOccurrenceRef transition;
};
struct ActorRetiredTraceEvent final {
  ActorTransitionOccurrenceRef transition;
};
struct TokenPublishedTraceEvent final {
  TokenOccurrenceRef token;
  CanonicalValueSequence value;
};
struct MemoryLinearizedTraceEvent final {
  MemoryActionOccurrenceRef action;
  std::optional<MemoryVersionRef> readsFrom;
  std::optional<MemoryVersionRef> modificationPredecessor;
  std::optional<MemoryActionOccurrenceRef> sequentiallyConsistentPredecessor;
};
struct PhysicalRequestedTraceEvent final {
  PhysicalActionOccurrenceRef action;
  PhysicalActionTarget target;
};
struct PhysicalGrantedTraceEvent final {
  PhysicalActionOccurrenceRef action;
};
struct PhysicalRetiredTraceEvent final {
  PhysicalActionOccurrenceRef action;
};

using SpatialTraceEvent =
    std::variant<ActorCommittedTraceEvent, ActorRetiredTraceEvent,
                 TokenPublishedTraceEvent, MemoryLinearizedTraceEvent,
                 PhysicalRequestedTraceEvent, PhysicalGrantedTraceEvent,
                 PhysicalRetiredTraceEvent>;

struct SpatialTraceFrame final {
  SpatialEventCoordinate coordinate;
  std::vector<SpatialTraceEvent> events;
};

struct SpatialDiagnosticTrace final {
  TraceCaptureLevel level = TraceCaptureLevel::Firing;
  std::vector<SpatialTraceFrame> frames;
};

TraceCaptureLevel minimumTraceCaptureLevel(const SpatialTraceEvent &event);

/// Returns the closed canonical ordering key for one diagnostic event. The key
/// is shared by frame canonicalization and capture-level inclusion checks; it
/// is not the deferred persistent trace record encoding.
llvm::Expected<std::vector<std::uint8_t>>
canonicalSpatialTraceEventKey(const SpatialTraceEvent &event);

/// Sorts one frame by the future trace event key and rejects duplicate keys or
/// events above the selected level. This is diagnostic validation only; it is
/// not the deferred persistent trace finalizer.
llvm::Error canonicalizeSpatialTraceFrame(SpatialTraceFrame &frame,
                                          TraceCaptureLevel level);

/// Adds one nonempty, strictly later frame after canonical validation.
llvm::Error appendSpatialTraceFrame(SpatialDiagnosticTrace &trace,
                                    SpatialTraceFrame frame);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SPATIALTRACE_H
