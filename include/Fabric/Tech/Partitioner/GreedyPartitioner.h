#ifndef FABRIC_TECH_PARTITIONER_GREEDY_PARTITIONER_H
#define FABRIC_TECH_PARTITIONER_GREEDY_PARTITIONER_H

#include "Fabric/Tech/Partitioner/Partitioner.h"

namespace fabric {

// Greedy partitioner. The current implementation is a placeholder that
// emits one Block per op via `buildSingletonPartition`; it exists so the
// algorithm-dispatch path can be wired and tested end-to-end.
class GreedyPartitioner : public IPartitioner {
public:
  PartitionResult run(::dataflow::GraphOp graph, const TemplateLibrary &lib,
                      const ::loom::TechMapConfig &cfg) override;
};

} // namespace fabric

#endif // FABRIC_TECH_PARTITIONER_GREEDY_PARTITIONER_H
