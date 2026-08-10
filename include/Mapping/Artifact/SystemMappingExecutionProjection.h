#ifndef LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGEXECUTIONPROJECTION_H
#define LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGEXECUTIONPROJECTION_H

#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/Support/Error.h"

#include <vector>

namespace loom::mapping {

struct SystemInstructionContextDomain final {
  ::dataflow::RootThreadLaunchRef root;
  InstructionExecutionContextKey context;
  std::vector<SystemPresburgerCell> cells;
};

struct SystemSpatialContextDomain final {
  ::dataflow::RootedGraphLaunchRef graph;
  ArtifactRootReference spatialMapping;
  SpatialExecutionContextKey context;
  std::vector<SystemPresburgerCell> cells;
};

/// The unique nonpersistent execution-context projection of one already
/// verified SystemMapping. Deployment, Runtime, and simulator bridges consume
/// this projection instead of independently evaluating B_thread or B_graph.
struct SystemExecutionContextProjection final {
  std::vector<SystemInstructionContextDomain> instructionDomains;
  std::vector<SystemSpatialContextDomain> spatialDomains;
};

llvm::Expected<SystemExecutionContextProjection> projectSystemExecutionContexts(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const SystemExecutionBindingView &execution);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGEXECUTIONPROJECTION_H
