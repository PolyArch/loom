#ifndef LOOM_DATAFLOW_IR_DATAFLOWSTATICSCHEDULEANALYSIS_H
#define LOOM_DATAFLOW_IR_DATAFLOWSTATICSCHEDULEANALYSIS_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace dataflow {

struct StaticActorCriticality final {
  ActorRef actor;
  GraphRef graph;
  std::uint64_t graphCriticalLength = 0;
  std::uint64_t recurrenceCriticalLength = 0;
  bool temporalStateCarrier = false;
};

struct StaticActorEdgeCriticality final {
  ActorTokenResultRef producer;
  ActorTokenOperandRef consumer;
  GraphRef graph;
  std::uint64_t weight = 0;
  bool initializedFeedback = false;
};

struct StaticRecurrenceFeedback final {
  ActorTokenResultRef producer;
  ActorTokenOperandRef consumer;
  GraphRef graph;
  std::uint64_t dependenceDistance = 1;
};

struct StaticGraphRecurrenceTopology final {
  GraphRef graph;
  bool postInitializationAcyclic = true;
};

/// Dataflow-owned immutable projection of graph critical paths and explicit
/// initialized recurrences. Placement and timing remain downstream concerns.
class StaticScheduleAnalysis final {
public:
  llvm::ArrayRef<StaticActorCriticality> actors() const { return actors_; }
  llvm::ArrayRef<StaticActorEdgeCriticality> edges() const { return edges_; }
  llvm::ArrayRef<StaticRecurrenceFeedback> feedbacks() const {
    return feedbacks_;
  }
  llvm::ArrayRef<StaticGraphRecurrenceTopology> recurrenceTopologies() const {
    return recurrenceTopologies_;
  }

  const StaticActorCriticality *findActor(ActorRef actor) const;
  std::uint64_t edgeWeight(const ActorTokenResultRef &producer,
                           const ActorTokenOperandRef &consumer) const;
  std::uint64_t graphCriticalLength(GraphRef graph) const;

  std::vector<StaticActorCriticality> actors_;
  std::vector<StaticActorEdgeCriticality> edges_;
  std::vector<StaticRecurrenceFeedback> feedbacks_;
  std::vector<StaticGraphRecurrenceTopology> recurrenceTopologies_;
};

llvm::Expected<StaticScheduleAnalysis>
deriveStaticScheduleAnalysis(const CanonicalDataflowProgramView &dataflow,
                             llvm::ArrayRef<GraphRef> covers);

} // namespace dataflow

#endif // LOOM_DATAFLOW_IR_DATAFLOWSTATICSCHEDULEANALYSIS_H
