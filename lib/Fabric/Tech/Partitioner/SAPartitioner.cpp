#include "Fabric/Tech/Partitioner/SAPartitioner.h"

namespace fabric {

PartitionResult SAPartitioner::run(::dataflow::GraphOp graph,
                                   const TemplateLibrary &lib,
                                   const ::loom::TechMapConfig &cfg) {
  (void)cfg;
  return buildSingletonPartition(graph, lib);
}

} // namespace fabric
