#include "Deployment/DeploymentSpatialLaunchSelection.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"

#include <system_error>

namespace loom::deployment {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("deployment_spatial_launch_invalid: ") + message);
}

} // namespace

llvm::Expected<DeploymentSpatialLaunchSelection>
resolveDeploymentSpatialLaunchSelection(
    const FinalizedDeployment &finalized, dataflow::RootedGraphLaunchRef graph,
    llvm::ArrayRef<std::uint64_t> denseCoordinates,
    const ArtifactStore &artifacts) {
  const Deployment &deployment = finalized.deployment();
  if (deployment.hardwareBindings().size() != 1)
    return invalid("Deployment does not contain one hardware binding");

  auto systemMapping =
      mapping::importSystemMapping(deployment.systemMapping(), artifacts);
  if (!systemMapping)
    return systemMapping.takeError();
  const mapping::SystemMappingView &mappingView = systemMapping->view();

  ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      mappingView.dataflowIdentity()};
  auto dataflowArtifact =
      dataflow::importCanonicalDataflow(dataflowReference, artifacts);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflowView = dataflowArtifact->view();
  if (!dataflowView)
    return dataflowView.takeError();

  auto contexts = mapping::projectSystemExecutionContexts(
      *dataflowView, mappingView.executionBindings());
  if (!contexts)
    return contexts.takeError();
  auto selected = mapping::selectSystemSpatialExecutionContext(
      *contexts, graph, denseCoordinates);
  if (!selected)
    return selected.takeError();

  return DeploymentSpatialLaunchSelection{
      deployment.hardwareBindings().front().hardwareImplementation,
      std::move(dataflowReference), selected->spatialMapping, selected->context,
      std::vector<ArtifactRootReference>(
          deployment.configurationImages().begin(),
          deployment.configurationImages().end())};
}

} // namespace loom::deployment
