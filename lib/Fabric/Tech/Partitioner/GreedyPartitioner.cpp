#include "Fabric/Tech/Partitioner/GreedyPartitioner.h"

namespace fabric {

PartitionResult GreedyPartitioner::run(::dataflow::GraphOp graph,
                                       const TemplateLibrary &lib,
                                       const ::loom::TechMapConfig &cfg) {
  (void)cfg;
  return buildSingletonPartition(graph, lib);
}

} // namespace fabric
