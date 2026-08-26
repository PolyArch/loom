#ifndef LOOM_APPLICATION_APPLICATIONRUNTIMEVALIDATIONINTERNAL_H
#define LOOM_APPLICATION_APPLICATIONRUNTIMEVALIDATIONINTERNAL_H

#include "Application/Build.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/JointHardwareReopen.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::application::detail {

struct ImportedApplicationMapping final {
  mapping::FinalizedSystemMapping mapping;
  dataflow::CanonicalDataflowArtifact dataflow;
  dataflow::CanonicalDataflowProgramView dataflowView;
  fabric::FinalizedFabricRoot system;
};

struct ApplicationRuntimeValidation final {
  ApplicationMappingRuntimeDisposition disposition =
      ApplicationMappingRuntimeDisposition::ProofNotEstablished;
  std::vector<ArtifactRootReference> evidence;
  std::optional<std::uint64_t> dfgCycles;
  std::optional<std::uint64_t> cgraCycles;
  std::optional<dse::SpatialFifoRuntimeFeedback> spatialFifoFeedback;
  std::optional<dse::SpatialOperandQueueRuntimeFeedback>
      spatialOperandQueueFeedback;
  std::optional<dse::SpatialTransportRuntimeFeedback> spatialTransportFeedback;
  std::vector<ArtifactRootReference> oracleEvidence;
  std::optional<std::uint64_t> resourceCoreCost;
};

llvm::Expected<ImportedApplicationMapping>
importApplicationMapping(const dse::JointDesignExecution &execution,
                         const ArtifactStore &artifacts);

llvm::Expected<const PreparedApplicationSoftware *>
findPreparedSoftware(const PreparedApplicationBuild &prepared,
                     const ArtifactIdentity &dataflowIdentity);

llvm::Expected<ApplicationRuntimeValidation> validateApplicationMappingRuntime(
    const PreparedApplicationBuild &prepared,
    const PreparedApplicationMappingAlternative &alternative,
    const dse::JointDesignExecution &execution,
    const dse::PlanExecutionPolicy &executionPolicy,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::application::detail

#endif // LOOM_APPLICATION_APPLICATIONRUNTIMEVALIDATIONINTERNAL_H
