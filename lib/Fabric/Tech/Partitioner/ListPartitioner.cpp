#include "Fabric/Tech/Partitioner/ListPartitioner.h"

namespace fabric {

PartitionResult ListPartitioner::run(::dataflow::GraphOp graph,
                                     const TemplateLibrary &lib,
                                     const ::loom::TechMapConfig &cfg) {
  (void)cfg;
  return buildSingletonPartition(graph, lib);
}

} // namespace fabric
