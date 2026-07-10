#ifndef FABRIC_TECH_PARTITIONER_BEAM_PARTITIONER_H
#define FABRIC_TECH_PARTITIONER_BEAM_PARTITIONER_H

#include "Fabric/Tech/Partitioner/Partitioner.h"

namespace fabric {

// Beam-search partitioner.
class BeamPartitioner : public IPartitioner {
public:
  PartitionResult run(::dataflow::GraphOp graph, const TemplateLibrary &lib,
                      const ::loom::TechMapConfig &cfg) override;
};

} // namespace fabric

#endif // FABRIC_TECH_PARTITIONER_BEAM_PARTITIONER_H
