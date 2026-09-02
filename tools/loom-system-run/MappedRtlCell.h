#ifndef LOOM_TOOLS_LOOM_SYSTEM_RUN_MAPPEDRTLCELL_H
#define LOOM_TOOLS_LOOM_SYSTEM_RUN_MAPPEDRTLCELL_H

#include "SpatialInvocationCase.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Deployment/Deployment.h"
#include "Evaluation/Case.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Request.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::system_run {

/// Provider options of the Verilator-backed mapped RTL simulation, resolved by
/// the driver's command line and handed to the provider as local tool config.
struct MappedRtlProviderOptions final {
  std::uint64_t buildJobs = 0;
  std::uint64_t buildWorkers = 0;
  std::uint64_t modelThreads = 0;
};

/// The published Request, its artifact resolution, and the imported Evidence
/// of one mapped RTL cell; the driver completes and compares the run like
/// every other Spatial engine.
struct MappedRtlCellEvidence final {
  evaluation::EvaluationRequest request;
  evaluation::CaseArtifactResolution resolution;
  evaluation::EvaluationEvidence evidence;
};

/// Rebuilds every hardware binding of the source Deployment as a portable
/// Spatial-core RTL implementation with its mapped RTL runtime platform
/// binding, keeping the System Mapping, host program, instruction-core
/// binaries, and static memory images.
llvm::Expected<deployment::FinalizedDeployment>
deriveMappedRtlDeployment(const deployment::FinalizedDeployment &source,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs);

/// Executes one materialized Spatial invocation as mapped RTL under
/// `bundleRoot`: resolves the launch selection against the mapped RTL
/// Deployment, binds the Verilator provider, publishes the Request, runs the
/// frozen invocation bundle, and imports its Evidence.
llvm::Expected<MappedRtlCellEvidence>
executeMappedRtlCell(const SpatialInvocationCase &invocation,
                     const deployment::FinalizedDeployment &rtlDeployment,
                     llvm::StringRef bundleRoot,
                     const MappedRtlProviderOptions &options,
                     const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::system_run

#endif // LOOM_TOOLS_LOOM_SYSTEM_RUN_MAPPEDRTLCELL_H
