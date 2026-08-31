#include "Evaluation/Models/CgraClosedWait.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/Request.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Simulator/CgraClosedWaitCertificate.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>

namespace loom::evaluation::models {
namespace {

constexpr std::uint64_t kClosedWaitReplayEventFrameLimit = 1'000'000;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::errc::invalid_argument,
                                 "cgra_closed_wait_finding_invalid: %s",
                                 message.str().c_str());
}

llvm::Expected<std::vector<std::uint8_t>>
encodeWitness(const OwnerValue &value) {
  const auto *certificate = value.getIf<sim::CgraClosedWaitCertificate>();
  if (!certificate)
    return invalid("terminal witness has the wrong owner type");
  return sim::encodeCgraClosedWaitCertificate(*certificate);
}

llvm::Expected<OwnerValue> decodeWitness(llvm::ArrayRef<std::uint8_t> bytes) {
  auto certificate = sim::decodeCgraClosedWaitCertificate(bytes);
  if (!certificate)
    return certificate.takeError();
  return OwnerValue::get(std::move(*certificate));
}

llvm::Error validateWitness(const OwnerValue &value,
                            const FindingTerminalWitnessContext &context) {
  const auto *certificate = value.getIf<sim::CgraClosedWaitCertificate>();
  if (!certificate)
    return invalid("terminal witness has the wrong owner type");
  if (llvm::Error error = sim::verifyCgraClosedWaitCertificate(*certificate))
    return error;
  const EvaluationRequest &request = context.request();
  if (request.modelBinding().descriptorRef() !=
      cgraSimulationModelDescriptorRef())
    return invalid("terminal witness Request selects a foreign model");
  const auto exactRole = [&](CaseSubjectRoleRef role,
                             const ArtifactRootReference &expected) {
    const auto subjects = request.subjectBindings().subjects(role);
    return subjects.size() == 1 && subjects.front() == expected;
  };
  if (!exactRole(cgraSimulationProgramRole(), certificate->owners.dataflow) ||
      !exactRole(cgraSimulationHardwareRole(), certificate->owners.fabric) ||
      !exactRole(cgraSimulationSpatialMappingRole(),
                 certificate->owners.spatialMapping))
    return invalid("terminal witness owners do not match exact Request roles");

  auto mapping = mapping::importSpatialMapping(
      certificate->owners.spatialMapping, context.artifactStore());
  if (!mapping)
    return mapping.takeError();
  if (mapping->view().dataflowIdentity() !=
          certificate->owners.dataflow.artifact ||
      mapping->view().fabricIdentity() !=
          certificate->owners.fabric.artifact ||
      mapping->view().techMappingIdentity() !=
          certificate->owners.techMapping.artifact)
    return invalid("terminal witness Mapping closure disagrees with its owners");
  return llvm::Error::success();
}

const ScopeFormDescriptor scopeForms[] = {
    {ScopeFormRef(0),
     "the exact CGRA simulation case",
     {},
     WholeExactCaseScope{},
     nullptr}};

const FindingDescriptor descriptor{
    CgraClosedWait,
    "cgra_closed_wait",
    "One runtime-proven closed strongly connected wait component for an exact "
    "CGRA simulation Request.",
    scopeForms,
    {},
    sim::terminalWitnessRefOccurrenceCodec(),
    FindingTerminalWitnessCodec{{"loom.cgra_closed_wait_certificate", {1, 0}},
                                &encodeWitness, &decodeWitness,
                                &validateWitness}};

} // namespace

llvm::Error registerCgraClosedWaitFinding() {
  return registerFindingDescriptor(descriptor);
}

llvm::Expected<VerifiedCgraClosedWaitEvidence>
importVerifiedCgraClosedWaitEvidence(
    const ArtifactRootReference &evidenceReference,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  if (llvm::Error error = registerCgraSimulationModel())
    return error;
  auto projection = importEvaluationEvidenceDependencyProjection(
      evidenceReference, artifactStore);
  if (!projection)
    return projection.takeError();
  if (projection->outcomeKind != EvidenceOutcomeKind::Completed)
    return invalid("Evidence outcome is not Completed");

  auto requestDependencies = importEvaluationRequestArtifactReferences(
      projection->request, artifactStore);
  if (!requestDependencies)
    return requestDependencies.takeError();
  std::optional<ArtifactRootReference> spatialMapping;
  std::optional<ArtifactRootReference> workload;
  std::optional<ArtifactRootReference> runtimeInput;
  for (const ArtifactRootReference &dependency : *requestDependencies) {
    const auto rememberUnique = [&](std::optional<ArtifactRootReference> &slot,
                                    const char *name) -> llvm::Error {
      if (slot)
        return invalid(llvm::Twine("Request names multiple ") + name +
                       " roots");
      slot = dependency;
      return llvm::Error::success();
    };
    if (dependency.schemaIdentity == mapping::mappingArtifactSchema.identity &&
        dependency.schemaVersion == mapping::mappingArtifactSchema.version) {
      if (llvm::Error error = rememberUnique(spatialMapping, "Mapping"))
        return error;
    } else if (dependency.schemaIdentity == sim::simulationWorkloadSchema.identity &&
               dependency.schemaVersion == sim::simulationWorkloadSchema.version) {
      if (llvm::Error error = rememberUnique(workload, "workload"))
        return error;
    } else if (dependency.schemaIdentity ==
                   sim::simulationRuntimeInputSchema.identity &&
               dependency.schemaVersion ==
                   sim::simulationRuntimeInputSchema.version) {
      if (llvm::Error error = rememberUnique(runtimeInput, "runtime-input"))
        return error;
    }
  }
  if (!spatialMapping || !workload || !runtimeInput)
    return invalid("Evidence Request has no exact CGRA input closure");

  auto resolved = resolveCgraSimulationCase(*spatialMapping, *workload,
                                            *runtimeInput, artifactStore);
  if (!resolved)
    return resolved.takeError();
  auto request = importEvaluationRequest(projection->request,
                                         resolved->resolution, artifactStore,
                                         blobStore);
  if (!request)
    return request.takeError();
  if (request->modelBinding().descriptorRef() !=
      cgraSimulationModelDescriptorRef())
    return invalid("Evidence Request selects a foreign model");
  auto evidence = importEvaluationEvidence(
      evidenceReference, resolved->resolution, artifactStore, blobStore);
  if (!evidence)
    return evidence.takeError();
  if (evidence->requestRef() != projection->request ||
      evidence->outcomeKind() != EvidenceOutcomeKind::Completed)
    return invalid("strict Evidence differs from its dependency projection");

  const auto *completed = std::get_if<CompletedEvidence>(&evidence->outcome());
  if (!completed)
    return invalid("Completed Evidence has no completed payload");
  std::optional<std::size_t> findingOrdinal;
  for (const auto indexed : llvm::enumerate(request->findingRequests())) {
    if (indexed.value().query().kind != CgraClosedWait)
      continue;
    if (findingOrdinal)
      return invalid("Request repeats the CGRA closed-wait finding");
    findingOrdinal = indexed.index();
  }
  if (!findingOrdinal || *findingOrdinal >= completed->findingResults.size())
    return invalid("Request has no CGRA closed-wait finding result");
  const auto *present = std::get_if<PresentFinding>(
      &completed->findingResults[*findingOrdinal].result);
  if (!present || present->occurrences.size() != 1)
    return invalid("CGRA closed-wait finding is not uniquely Present");
  const auto *witnessRef =
      present->occurrences.front().getIf<sim::TerminalWitnessRef>();
  if (!witnessRef)
    return invalid("CGRA closed-wait occurrence has a foreign owner type");

  const ModelOutputBinding *output = nullptr;
  for (const ModelOutputBinding &candidate : evidence->outputBindings())
    if (candidate.slot == witnessRef->executionOutputSlot) {
      if (output)
        return invalid("Evidence repeats the terminal execution output slot");
      output = &candidate;
    }
  if (!output || witnessRef->executionOutputOrdinal >= output->artifacts.size())
    return invalid("terminal witness reference is outside its output binding");
  const ArtifactRootReference executionReference =
      output->artifacts[witnessRef->executionOutputOrdinal];
  auto execution = sim::importSimulationExecution(
      executionReference, resolved->resolution, artifactStore, blobStore);
  if (!execution)
    return execution.takeError();
  if (execution->request() != projection->request || !execution->spatial())
    return invalid("terminal execution has a foreign Request or form");
  const auto *halted = std::get_if<sim::HaltedExecution>(&execution->terminal());
  if (!halted || halted->findingKind != CgraClosedWait)
    return invalid("terminal execution is not a CGRA closed wait");
  const auto *certificate =
      halted->witness.getIf<sim::CgraClosedWaitCertificate>();
  if (!certificate)
    return invalid("Halted execution has a foreign certificate owner type");
  auto digest = sim::digestCgraClosedWaitCertificate(*certificate);
  if (!digest)
    return digest.takeError();

  // Persistent feedback trusts neither an authored Halted execution nor an
  // authored Evidence wrapper. Rebuild the exact deterministic CGRA Request
  // from its stored roots, replay it to a terminal, and require byte-identical
  // Evidence identity. The model's config view is empty, so unrelated default
  // ResolvedConfig fields cannot enter the reconstructed Request identity.
  auto replayPrepared = prepareCgraSimulationEvaluation(
      resolved->canonicalDataflow, resolved->fabric, *spatialMapping,
      *workload, *runtimeInput, defaultResolvedConfig(), artifactStore,
      blobStore);
  if (!replayPrepared)
    return replayPrepared.takeError();
  if (evaluationRequestReference(replayPrepared->request) !=
      projection->request)
    return invalid("stored Request cannot be deterministically reconstructed");
  auto replay = evaluateCgraSimulationWithDiagnostics(
      *replayPrepared, {kClosedWaitReplayEventFrameLimit, std::nullopt},
      artifactStore, blobStore);
  if (!replay)
    return replay.takeError();
  if (evaluationEvidenceReference(replay->evidence) != evidenceReference)
    return invalid("stored closed-wait Evidence differs from deterministic "
                   "replay");
  return VerifiedCgraClosedWaitEvidence(
      evidenceReference, projection->request, executionReference, *certificate,
      std::move(*digest));
}

llvm::Expected<CgraSimulationEvidenceTerminal>
classifyCompletedCgraSimulationEvidence(
    const EvaluationEvidence &evidence,
    const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  auto request = importEvaluationRequest(evidence.requestRef(), resolution,
                                         artifactStore, blobStore);
  if (!request)
    return request.takeError();
  if (request->modelBinding().descriptorRef() !=
      cgraSimulationModelDescriptorRef())
    return invalid("Evidence Request selects a foreign model");
  const auto *completed = std::get_if<CompletedEvidence>(&evidence.outcome());
  if (!completed)
    return invalid("CGRA Evidence is not Completed");

  std::optional<std::size_t> findingOrdinal;
  for (const auto indexed : llvm::enumerate(request->findingRequests())) {
    if (indexed.value().query().kind != CgraClosedWait)
      continue;
    if (findingOrdinal)
      return invalid("Request repeats the CGRA closed-wait finding");
    findingOrdinal = indexed.index();
  }
  if (!findingOrdinal || *findingOrdinal >= completed->findingResults.size())
    return invalid("Completed CGRA Evidence omits its closed-wait result");

  std::optional<ArtifactRootReference> executionReference;
  for (const ModelOutputBinding &binding : evidence.outputBindings())
    for (const ArtifactRootReference &output : binding.artifacts) {
      if (output.schemaIdentity != sim::simulationExecutionSchema.identity ||
          output.schemaVersion != sim::simulationExecutionSchema.version)
        continue;
      if (executionReference)
        return invalid("CGRA Evidence repeats its SimulationExecution output");
      executionReference = output;
    }
  if (!executionReference)
    return invalid("CGRA Evidence has no SimulationExecution output");
  auto execution = sim::importSimulationExecution(
      *executionReference, resolution, artifactStore, blobStore);
  if (!execution)
    return execution.takeError();
  if (execution->request() != evidence.requestRef() || !execution->spatial())
    return invalid("CGRA execution has a foreign Request or form");

  const FindingResultValue &finding =
      completed->findingResults[*findingOrdinal].result;
  if (std::holds_alternative<sim::RetiredExecution>(execution->terminal())) {
    if (!std::holds_alternative<AbsentFinding>(finding))
      return invalid("Retired CGRA execution reports a closed wait");
    return CgraSimulationEvidenceTerminal::Retired;
  }
  const auto *halted =
      std::get_if<sim::HaltedExecution>(&execution->terminal());
  if (!halted || halted->findingKind != CgraClosedWait ||
      !std::holds_alternative<PresentFinding>(finding))
    return invalid("Completed CGRA terminal and finding result disagree");
  return CgraSimulationEvidenceTerminal::ClosedWait;
}

} // namespace loom::evaluation::models
