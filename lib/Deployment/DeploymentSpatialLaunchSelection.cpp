#include "Deployment/DeploymentSpatialLaunchSelection.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
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
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  const Deployment &deployment = finalized.deployment();

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

  const fabric::SpatialCoreOccurrenceRef subject{selected->context.accCore};
  std::optional<hardware::FinalizedHardwareImplementation> implementation;
  for (const DeploymentHardwareBinding &binding :
       deployment.hardwareBindings()) {
    auto candidate = hardware::importHardwareImplementation(
        binding.hardwareImplementation, artifacts, blobs);
    if (!candidate)
      return candidate.takeError();
    if (candidate->implementation().subject() != subject)
      continue;
    if (implementation)
      return invalid("Deployment repeats the selected SpatialCore binding");
    implementation = std::move(*candidate);
  }
  if (!implementation)
    return invalid("Deployment omits the selected SpatialCore binding");

  auto abi = hardware::importConfigurationABI(
      implementation->implementation().configurationAbi(), artifacts);
  if (!abi)
    return abi.takeError();
  std::vector<ArtifactRootReference> configurationImages;
  for (const ArtifactRootReference &reference :
       deployment.configurationImages()) {
    auto image = importHardwareConfigurationImage(reference, artifacts);
    if (!image)
      return image.takeError();
    const hardware::ProgrammingUnit *unit =
        abi->abi().findProgrammingUnit(image->image().programmingUnitId());
    if (!unit)
      return invalid("configuration image names a missing programming unit");
    const hardware::ProgrammingUnitOccurrenceScope scope =
        hardware::deriveProgrammingUnitOccurrenceScope(*unit);
    if (!scope.includesDirectSystemResources && scope.spatialCores.size() == 1 &&
        scope.spatialCores.front() == subject)
      configurationImages.push_back(reference);
  }

  return DeploymentSpatialLaunchSelection{
      implementation->reference(),
      std::move(dataflowReference), selected->spatialMapping, selected->context,
      std::move(configurationImages)};
}

} // namespace loom::deployment
