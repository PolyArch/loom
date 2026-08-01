#include "Simulator/SimulationArtifacts.h"

#include "SimulationWireInternal.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "llvm/Support/Error.h"

#include <system_error>
#include <utility>

namespace loom::sim {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("simulation_artifact_store_invalid: ") + message);
}

bool hasSchema(const ArtifactRootReference &reference,
               const ArtifactSchemaDescriptor &schema) {
  return reference.schemaIdentity == schema.identity &&
         reference.schemaVersion == schema.version;
}

llvm::Expected<ArtifactIdentity>
spatialWorkloadOwnerIdentity(llvm::ArrayRef<std::uint8_t> bytes) {
  detail::WireReader reader(bytes);
  llvm::Expected<std::uint32_t> root = reader.u32();
  if (!root)
    return root.takeError();
  if (*root != static_cast<std::uint32_t>(SimulationWorkloadKind::Spatial))
    return invalid("workload is not a Spatial root");
  return reader.identity();
}

} // namespace

llvm::Expected<ArtifactRootReference>
publishSimulationWorkload(const CanonicalSimulationWorkload &workload,
                          const ArtifactStore &store) {
  llvm::Expected<ArtifactIdentity> stored =
      store.put(simulationWorkloadSchema, workload.canonicalBytes());
  if (!stored)
    return stored.takeError();
  if (*stored != workload.identity())
    return invalid("ArtifactStore returned a different workload identity");
  return ArtifactRootReference{simulationWorkloadSchema.identity.str(),
                               simulationWorkloadSchema.version, *stored};
}

llvm::Expected<ArtifactRootReference> publishSimulationRuntimeInput(
    const CanonicalSimulationRuntimeInput &runtimeInput,
    const ArtifactStore &store) {
  llvm::Expected<ArtifactIdentity> stored =
      store.put(simulationRuntimeInputSchema, runtimeInput.canonicalBytes());
  if (!stored)
    return stored.takeError();
  if (*stored != runtimeInput.identity())
    return invalid("ArtifactStore returned a different runtime-input identity");
  return ArtifactRootReference{simulationRuntimeInputSchema.identity.str(),
                               simulationRuntimeInputSchema.version, *stored};
}

llvm::Expected<ImportedSpatialSimulationInputs> importSpatialSimulationInputs(
    const ArtifactRootReference &workloadReference,
    const ArtifactRootReference &runtimeInputReference,
    const ArtifactStore &store) {
  if (!hasSchema(workloadReference, simulationWorkloadSchema))
    return invalid("foreign SimulationWorkload reference schema");
  if (!hasSchema(runtimeInputReference, simulationRuntimeInputSchema))
    return invalid("foreign SimulationRuntimeInput reference schema");

  llvm::Expected<CanonicalSemanticBytes> workloadBytes =
      store.get(workloadReference);
  if (!workloadBytes)
    return workloadBytes.takeError();
  llvm::Expected<ArtifactIdentity> dataflowIdentity =
      spatialWorkloadOwnerIdentity(workloadBytes->bytes());
  if (!dataflowIdentity)
    return dataflowIdentity.takeError();
  ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version, *dataflowIdentity};
  llvm::Expected<dataflow::CanonicalDataflowArtifact> dataflow =
      dataflow::importCanonicalDataflow(dataflowReference, store);
  if (!dataflow)
    return dataflow.takeError();
  llvm::Expected<dataflow::CanonicalDataflowProgramView> view =
      dataflow->view();
  if (!view)
    return view.takeError();
  llvm::Expected<CanonicalSimulationWorkload> workload =
      importSimulationWorkload(workloadBytes->bytes(), *view,
                               workloadReference.artifact);
  if (!workload)
    return workload.takeError();

  llvm::Expected<CanonicalSemanticBytes> runtimeBytes =
      store.get(runtimeInputReference);
  if (!runtimeBytes)
    return runtimeBytes.takeError();
  llvm::Expected<CanonicalSimulationRuntimeInput> runtimeInput =
      importSimulationRuntimeInput(runtimeBytes->bytes(), *workload, *view,
                                   runtimeInputReference.artifact);
  if (!runtimeInput)
    return runtimeInput.takeError();
  return ImportedSpatialSimulationInputs{
      std::move(*dataflow), std::move(*workload), std::move(*runtimeInput)};
}

llvm::Expected<ImportedStructuredProgramSimulationInputs>
importStructuredProgramSimulationInputs(
    const ArtifactRootReference &workloadReference,
    const ArtifactRootReference &runtimeInputReference,
    const ArtifactStore &store) {
  if (!hasSchema(workloadReference, simulationWorkloadSchema))
    return invalid("foreign SimulationWorkload reference schema");
  if (!hasSchema(runtimeInputReference, simulationRuntimeInputSchema))
    return invalid("foreign SimulationRuntimeInput reference schema");

  llvm::Expected<CanonicalSemanticBytes> workloadBytes =
      store.get(workloadReference);
  if (!workloadBytes)
    return workloadBytes.takeError();
  llvm::Expected<ArtifactIdentity> structuredIdentity =
      detail::structuredProgramWorkloadOwnerIdentity(workloadBytes->bytes());
  if (!structuredIdentity)
    return structuredIdentity.takeError();
  ArtifactRootReference structuredReference{
      frontend::structuredProgramArtifactSchema.identity.str(),
      frontend::structuredProgramArtifactSchema.version, *structuredIdentity};
  llvm::Expected<frontend::StructuredProgramCandidate> structured =
      frontend::importStructuredProgram(structuredReference, store);
  if (!structured)
    return structured.takeError();
  llvm::Expected<frontend::StructuredProgramCandidateView> view =
      structured->view();
  if (!view)
    return view.takeError();
  llvm::Expected<CanonicalSimulationWorkload> workload =
      importSimulationWorkload(workloadBytes->bytes(), *view,
                               workloadReference.artifact);
  if (!workload)
    return workload.takeError();

  llvm::Expected<CanonicalSemanticBytes> runtimeBytes =
      store.get(runtimeInputReference);
  if (!runtimeBytes)
    return runtimeBytes.takeError();
  llvm::Expected<CanonicalSimulationRuntimeInput> runtimeInput =
      importSimulationRuntimeInput(runtimeBytes->bytes(), *workload, *view,
                                   runtimeInputReference.artifact);
  if (!runtimeInput)
    return runtimeInput.takeError();
  return ImportedStructuredProgramSimulationInputs{
      std::move(*structured), std::move(*workload), std::move(*runtimeInput)};
}

} // namespace loom::sim
