#include "Deployment/DeploymentSpatialLaunchSelection.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"

#include <cassert>
#include <system_error>

namespace loom::deployment {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("deployment_spatial_launch_invalid: ") + message);
}

} // namespace

llvm::Expected<DeploymentSpatialLaunchProjection>
DeploymentSpatialLaunchProjection::get(const FinalizedDeployment &finalized,
                                       const ArtifactStore &artifacts,
                                       const BlobStore &blobs) {
  const Deployment &deployment = finalized.deployment();

  if (!deployment.systemMapping())
    return invalid("host-only Deployment has no Spatial launch");
  auto systemMapping =
      mapping::importSystemMapping(*deployment.systemMapping(), artifacts);
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
  const auto &dataflowView = dataflowArtifact->view();

  auto contexts = mapping::projectSystemExecutionContexts(
      dataflowView, mappingView.executionBindings());
  if (!contexts)
    return contexts.takeError();
  std::vector<fabric::SpatialCoreOccurrenceRef> subjects;
  for (const auto &domain : contexts->spatialDomains) {
    const fabric::SpatialCoreOccurrenceRef subject{domain.context.accCore};
    if (!llvm::is_contained(subjects, subject))
      subjects.push_back(subject);
  }

  if (subjects.empty())
    return DeploymentSpatialLaunchProjection(std::move(dataflowReference),
                                             std::move(*contexts), {});

  std::vector<hardware::FinalizedHardwareImplementation> implementations;
  for (const DeploymentHardwareBinding &binding :
       deployment.hardwareBindings()) {
    auto candidate = hardware::importHardwareImplementation(
        binding.hardwareImplementation, artifacts, blobs);
    if (!candidate)
      return candidate.takeError();
    if (llvm::is_contained(subjects, candidate->implementation().subject()))
      implementations.push_back(std::move(*candidate));
  }
  std::vector<FinalizedHardwareConfigurationImage> images;
  for (const ArtifactRootReference &reference :
       deployment.configurationImages()) {
    auto image = importHardwareConfigurationImage(reference, artifacts);
    if (!image)
      return image.takeError();
    images.push_back(std::move(*image));
  }

  std::vector<SpatialCoreBinding> bindings;
  for (fabric::SpatialCoreOccurrenceRef subject : subjects) {
    const hardware::FinalizedHardwareImplementation *implementation = nullptr;
    for (const auto &candidate : implementations) {
      if (candidate.implementation().subject() != subject)
        continue;
      if (implementation)
        return invalid("Deployment repeats the selected SpatialCore binding");
      implementation = &candidate;
    }
    if (!implementation)
      return invalid("Deployment omits the selected SpatialCore binding");
    auto abi = hardware::importConfigurationABI(
        implementation->implementation().configurationAbi(), artifacts);
    if (!abi)
      return abi.takeError();
    SpatialCoreBinding binding{subject, implementation->reference(), {}};
    // Direct System images remain Deployment-global and require an
    // independent System provider.
    for (const auto &image : images) {
      const hardware::ProgrammingUnit *unit =
          abi->abi().findProgrammingUnit(image.image().programmingUnitId());
      if (!unit)
        return invalid("configuration image names a missing programming unit");
      const hardware::ProgrammingUnitOccurrenceScope scope =
          hardware::deriveProgrammingUnitOccurrenceScope(*unit);
      if (!scope.includesDirectSystemResources &&
          scope.spatialCores.size() == 1 &&
          scope.spatialCores.front() == subject)
        binding.configurationImages.push_back(image.reference());
    }
    bindings.push_back(std::move(binding));
  }
  return DeploymentSpatialLaunchProjection(
      std::move(dataflowReference), std::move(*contexts), std::move(bindings));
}

llvm::Expected<DeploymentSpatialLaunchSelection>
DeploymentSpatialLaunchProjection::select(
    dataflow::RootedGraphLaunchRef graph,
    llvm::ArrayRef<std::uint64_t> denseCoordinates) const {
  auto selected = mapping::selectSystemSpatialExecutionContext(
      contexts_, graph, denseCoordinates);
  if (!selected)
    return selected.takeError();
  const fabric::SpatialCoreOccurrenceRef subject{selected->context.accCore};
  const auto binding =
      llvm::find_if(bindings_, [&](const SpatialCoreBinding &binding) {
        return binding.subject == subject;
      });
  assert(binding != bindings_.end() &&
         "selected context has no verified binding");
  return DeploymentSpatialLaunchSelection{
      binding->implementation, dataflow_, selected->spatialMapping,
      selected->context, binding->configurationImages};
}

llvm::Expected<DeploymentSpatialLaunchSelection>
resolveDeploymentSpatialLaunchSelection(
    const FinalizedDeployment &finalized, dataflow::RootedGraphLaunchRef graph,
    llvm::ArrayRef<std::uint64_t> denseCoordinates,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto projection =
      DeploymentSpatialLaunchProjection::get(finalized, artifacts, blobs);
  if (!projection)
    return projection.takeError();
  return projection->select(graph, denseCoordinates);
}

} // namespace loom::deployment
