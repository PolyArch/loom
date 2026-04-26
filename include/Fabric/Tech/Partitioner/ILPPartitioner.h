#ifndef FABRIC_TECH_PARTITIONER_ILP_PARTITIONER_H
#define FABRIC_TECH_PARTITIONER_ILP_PARTITIONER_H

#include "Fabric/Tech/Partitioner/Partitioner.h"

namespace fabric {

// ILP-backed partitioner.
//
// On small inputs (graph body <= `kILPMaxOps` ops) and when the build was
// configured with `LOOM_ENABLE_ILP`, the partitioner formulates a small mixed
// integer program over single-op template assignments and dispatches it to
// the HiGHS MIP solver. When the input is too large, when any candidate
// template is multi-op (so the simplified single-op MIP cannot model it), or
// when the build was configured with `LOOM_ENABLE_ILP=OFF`, the partitioner
// emits a one-line module-level warning and delegates to the greedy
// partitioner so the pass still produces a valid partition.
class ILPPartitioner : public IPartitioner {
public:
  // Maximum graph-body op count for which the MIP path is taken. Beyond this
  // threshold the partitioner falls back to greedy.
  static constexpr unsigned kILPMaxOps = 200;

  PartitionResult run(::dataflow::GraphOp graph, const TemplateLibrary &lib,
                      const ::loom::TechMapConfig &cfg) override;
};

} // namespace fabric

#endif // FABRIC_TECH_PARTITIONER_ILP_PARTITIONER_H
