#ifndef LOOM_PNR_SPATIALRECURRENCETIMING_H
#define LOOM_PNR_SPATIALRECURRENCETIMING_H

#include "Dataflow/IR/DataflowStructuralRefs.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dataflow {
class CanonicalDataflowProgramView;
}

namespace loom::fabric {
class FabricArtifactView;
}

namespace loom::mapping {
class SpatialMappingView;
class TechMappingView;
}

namespace loom::pnr {

enum class SpatialRecurrenceTimingProofKind : std::uint8_t {
  Proven,
  ProofNotEstablished,
};

enum class SpatialRecurrenceEdgeDisposition : std::uint8_t {
  ComputeInternal,
  MemoryInternal,
  ExternalRouteTree,
  RegisterFifo,
};

struct SpatialRecurrenceTimingEdgeWitness final {
  ::dataflow::ActorTokenResultRef producer;
  ::dataflow::ActorTokenOperandRef consumer;
  SpatialRecurrenceEdgeDisposition disposition =
      SpatialRecurrenceEdgeDisposition::ComputeInternal;
  std::uint64_t publicationLatencyCycles = 0;
  std::uint64_t transportLatencyCycles = 0;
  std::uint64_t nextStateLatencyCycles = 0;
  std::uint64_t totalLatencyCycles = 0;

  friend bool operator==(const SpatialRecurrenceTimingEdgeWitness &lhs,
                         const SpatialRecurrenceTimingEdgeWitness &rhs) {
    return lhs.producer == rhs.producer && lhs.consumer == rhs.consumer &&
           lhs.disposition == rhs.disposition &&
           lhs.publicationLatencyCycles == rhs.publicationLatencyCycles &&
           lhs.transportLatencyCycles == rhs.transportLatencyCycles &&
           lhs.nextStateLatencyCycles == rhs.nextStateLatencyCycles &&
           lhs.totalLatencyCycles == rhs.totalLatencyCycles;
  }
};

/// One canonical loop-carried cycle. `edges` starts at the carry actor, follows
/// the non-feedback DAG to the feedback producer, then closes with the exact
/// `dataflow.carry` Next edge.
struct SpatialRecurrenceTimingWitness final {
  ::dataflow::GraphRef graph;
  ::dataflow::ActorTokenResultRef feedbackProducer;
  ::dataflow::ActorTokenOperandRef feedbackConsumer;
  std::uint64_t dependenceDistance = 1;
  std::uint64_t latencyCycles = 0;
  std::uint64_t recurrenceMinimumInitiationIntervalCycles = 1;
  std::vector<SpatialRecurrenceTimingEdgeWitness> edges;

  friend bool operator==(const SpatialRecurrenceTimingWitness &lhs,
                         const SpatialRecurrenceTimingWitness &rhs) {
    return lhs.graph == rhs.graph &&
           lhs.feedbackProducer == rhs.feedbackProducer &&
           lhs.feedbackConsumer == rhs.feedbackConsumer &&
           lhs.dependenceDistance == rhs.dependenceDistance &&
           lhs.latencyCycles == rhs.latencyCycles &&
           lhs.recurrenceMinimumInitiationIntervalCycles ==
               rhs.recurrenceMinimumInitiationIntervalCycles &&
           lhs.edges == rhs.edges;
  }
};

/// Closed Mapping-derived recurrence proof. The diagnostic is populated only
/// for ProofNotEstablished and names the first canonical missing proof fact.
struct SpatialRecurrenceTimingProjection final {
  SpatialRecurrenceTimingProofKind kind =
      SpatialRecurrenceTimingProofKind::Proven;
  std::uint64_t recurrenceMinimumInitiationIntervalCycles = 1;
  std::vector<SpatialRecurrenceTimingWitness> witnesses;
  std::string diagnostic;

  friend bool operator==(const SpatialRecurrenceTimingProjection &lhs,
                         const SpatialRecurrenceTimingProjection &rhs) {
    return lhs.kind == rhs.kind &&
           lhs.recurrenceMinimumInitiationIntervalCycles ==
               rhs.recurrenceMinimumInitiationIntervalCycles &&
           lhs.witnesses == rhs.witnesses &&
           lhs.diagnostic == rhs.diagnostic;
  }
};

/// Cold reconstruction from one persistent SpatialMapping and its exact
/// immutable D/T/F dependency closure. This is the System catalog and replay
/// oracle; the Mapping does not persist a second timing authority.
llvm::Expected<SpatialRecurrenceTimingProjection>
projectSpatialMappingRecurrenceTiming(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &mapping);

llvm::Expected<SpatialRecurrenceTimingProjection>
projectSpatialMappingGraphRecurrenceTiming(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &mapping,
    ::dataflow::GraphRef graph);

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALRECURRENCETIMING_H
