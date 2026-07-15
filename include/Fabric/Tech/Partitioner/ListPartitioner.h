#ifndef FABRIC_TECH_PARTITIONER_LIST_PARTITIONER_H
#define FABRIC_TECH_PARTITIONER_LIST_PARTITIONER_H

#include "Fabric/Tech/Partitioner/Partitioner.h"

namespace fabric {

// List-priority partitioner.
class ListPartitioner : public IPartitioner {
public:
  PartitionResult run(::dataflow::GraphOp graph, const TemplateLibrary &lib,
                      const ::loom::ResolvedFabricTechMapConfig &cfg) override;
};

} // namespace fabric

#endif // FABRIC_TECH_PARTITIONER_LIST_PARTITIONER_H
