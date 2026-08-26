#ifndef LOOM_DSE_FUREVERSESYNTHESISWORKFLOW_H
#define LOOM_DSE_FUREVERSESYNTHESISWORKFLOW_H

#include "Common/Artifact.h"
#include "Config/ResolvedConfig.h"
#include "DSE/Plan.h"

#include "llvm/Support/Error.h"

#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::dse {

/// Ordinary DSE Plan composition for the bounded reverse-FU domain. Every
/// node retains its existing descriptor and resolved configuration owner.
class FuReverseSynthesisCandidateWorkflow final {
public:
  const ArtifactRootReference &dataflow() const { return dataflow_; }
  const ResolvedConfig &resolvedConfig() const { return resolvedConfig_; }
  PlanOutputRef module() const { return module_; }
  PlanOutputRef techMappings() const { return techMappings_; }
  PlanOutputRef jointTechMapping() const { return jointTechMapping_; }
  PlanOutputRef system() const { return system_; }
  PlanOutputRef physicalTimingProfiles() const {
    return physicalTimingProfiles_;
  }
  PlanOutputRef configurationAbi() const { return configurationAbi_; }
  PlanOutputRef spatialMappings() const { return spatialMappings_; }
  PlanOutputRef jointSpatialMappings() const { return jointSpatialMappings_; }
  PlanOutputRef systemMappings() const { return systemMappings_; }
  PlanOutputRef portableRtlImplementations() const {
    return portableRtlImplementations_;
  }

private:
  FuReverseSynthesisCandidateWorkflow(
      ArtifactRootReference dataflow, ResolvedConfig resolvedConfig,
      PlanOutputRef module, PlanOutputRef techMappings,
      PlanOutputRef jointTechMapping, PlanOutputRef system,
      PlanOutputRef physicalTimingProfiles, PlanOutputRef configurationAbi,
      PlanOutputRef spatialMappings, PlanOutputRef jointSpatialMappings,
      PlanOutputRef systemMappings, PlanOutputRef portableRtlImplementations)
      : dataflow_(std::move(dataflow)),
        resolvedConfig_(std::move(resolvedConfig)), module_(module),
        techMappings_(techMappings), jointTechMapping_(jointTechMapping),
        system_(system), physicalTimingProfiles_(physicalTimingProfiles),
        configurationAbi_(configurationAbi), spatialMappings_(spatialMappings),
        jointSpatialMappings_(jointSpatialMappings),
        systemMappings_(systemMappings),
        portableRtlImplementations_(portableRtlImplementations) {}

  ArtifactRootReference dataflow_;
  ResolvedConfig resolvedConfig_;
  PlanOutputRef module_;
  PlanOutputRef techMappings_;
  PlanOutputRef jointTechMapping_;
  PlanOutputRef system_;
  PlanOutputRef physicalTimingProfiles_;
  PlanOutputRef configurationAbi_;
  PlanOutputRef spatialMappings_;
  PlanOutputRef jointSpatialMappings_;
  PlanOutputRef systemMappings_;
  PlanOutputRef portableRtlImplementations_;

  friend llvm::Expected<FuReverseSynthesisCandidateWorkflow>
  buildFuReverseSynthesisCandidateWorkflow(const ArtifactRootReference &,
                                           const ResolvedConfig &,
                                           const ArtifactStore &);
};

struct FuReverseSynthesisWorkflowArtifacts final {
  ArtifactRootReference dataflow;
  ArtifactRootReference module;
  std::vector<ArtifactRootReference> techMappings;
  ArtifactRootReference jointTechMapping;
  ArtifactRootReference system;
  std::vector<ArtifactRootReference> physicalTimingProfiles;
  ArtifactRootReference configurationAbi;
  std::vector<ArtifactRootReference> spatialMappings;
  std::vector<ArtifactRootReference> jointSpatialMappings;
  std::vector<ArtifactRootReference> systemMappings;
  std::vector<ArtifactRootReference> portableRtlImplementations;
};

enum class FuReverseSynthesisWorkflowDisposition : std::uint8_t {
  CompleteCandidate,
  NoFeasibleCandidate,
};

/// Requires every admitted graph to be reachable from at least one root
/// thread, then composes the existing reverse synthesis, Spatial PnR, System
/// PnR, and portable RTL candidate generators into one resolved DSE Plan.
llvm::Expected<FuReverseSynthesisCandidateWorkflow>
buildFuReverseSynthesisCandidateWorkflow(const ArtifactRootReference &dataflow,
                                         const ResolvedConfig &baseConfig,
                                         const ArtifactStore &store);

/// Classifies completion against this workflow's required product. A generic
/// DSE Plan can complete with independent terminal outputs even when one
/// required Mapping branch is empty, so terminal-root presence alone does not
/// establish a complete reverse-synthesis candidate.
llvm::Expected<FuReverseSynthesisWorkflowDisposition>
classifyFuReverseSynthesisWorkflow(
    const FuReverseSynthesisCandidateWorkflow &workflow,
    const CompletedDsePlanExecution &execution);

/// Resolves a completed plan into its exact artifact closure and independently
/// checks graph coverage, System attachment, Mapping ownership, packed ABI,
/// and portable RTL HardwareImplementation lineage.
llvm::Expected<FuReverseSynthesisWorkflowArtifacts>
projectFuReverseSynthesisWorkflowArtifacts(
    const FuReverseSynthesisCandidateWorkflow &workflow,
    const CompletedDsePlanExecution &execution, const ArtifactStore &artifacts,
    const BlobStore &blobs);

llvm::Error verifyFuReverseSynthesisWorkflowArtifacts(
    const FuReverseSynthesisWorkflowArtifacts &artifacts,
    const ArtifactStore &store, const BlobStore &blobs);

} // namespace loom::dse

#endif // LOOM_DSE_FUREVERSESYNTHESISWORKFLOW_H
