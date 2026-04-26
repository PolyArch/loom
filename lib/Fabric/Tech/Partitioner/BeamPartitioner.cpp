#include "Fabric/Tech/Partitioner/BeamPartitioner.h"

namespace fabric {

PartitionResult BeamPartitioner::run(::dataflow::GraphOp graph,
                                     const TemplateLibrary &lib,
                                     const ::loom::TechMapConfig &cfg) {
  (void)cfg;
  return buildSingletonPartition(graph, lib);
}

} // namespace fabric
