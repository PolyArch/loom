#ifndef LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGEXECUTIONPROJECTION_H
#define LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGEXECUTIONPROJECTION_H

#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/Support/Error.h"

#include <cstdint>
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

struct SelectedSystemSpatialContext final {
  ArtifactRootReference spatialMapping;
  SpatialExecutionContextKey context;
};

llvm::Expected<SystemExecutionContextProjection> projectSystemExecutionContexts(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const SystemExecutionBindingView &execution);

/// Selects the one Spatial execution context for a concrete graph-launch
/// coordinate. Launch-parameter symbols remain existential: if legal symbol
/// valuations select different contexts, the point is ambiguous.
llvm::Expected<SelectedSystemSpatialContext>
selectSystemSpatialExecutionContext(
    const SystemExecutionContextProjection &projection,
    ::dataflow::RootedGraphLaunchRef graph,
    llvm::ArrayRef<std::uint64_t> denseCoordinates);

/// Resolves one already verified ServicePlanSelection at a concrete logical
/// point. Launch-parameter symbols remain existential, matching execution
/// context selection: a point that can select different plans for legal
/// symbol valuations is rejected as ambiguous.
llvm::Expected<std::uint64_t> selectSystemServicePlanOrdinal(
    const SystemServiceRealizationView &realization,
    const ServicePlanSelectionAnchor &anchor,
    const ExecutionContextKey &context,
    llvm::ArrayRef<SystemPresburgerCell> contextDomain,
    llvm::ArrayRef<std::uint64_t> denseCoordinates);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGEXECUTIONPROJECTION_H
