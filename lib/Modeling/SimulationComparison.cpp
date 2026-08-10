#include "Evaluation/Models/SimulationComparison.h"
#include "Evaluation/ProductionRegistry.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/Models/DfgSimulation.h"
#include "Evaluation/StandardFindings.h"
#include "Simulator/SimulationExecution.h"
#include "Simulator/SpatialObservationComparison.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr BuiltinEvaluationCase kCase =
    BuiltinEvaluationCase::SimulationExecutionComparison;
constexpr BuiltinEvaluationModel kModel =
    BuiltinEvaluationModel::SimulationExecutionComparison;
constexpr CaseSubjectRoleRef kReferenceExecutionRole(0);
constexpr CaseSubjectRoleRef kCandidateExecutionRole(1);
constexpr ScopeFormRef kWholeExactCaseScope(0);

EvaluationCaseSignatureRef caseSignatureRef() {
  return llvm::cantFail(EvaluationCaseSignatureRef::get(
      evaluationSchemaVersion(), builtinEvaluationCaseKind(kCase)));
}

const ArtifactSchemaDescriptor *const kExecutionSchemas[] = {
    &sim::simulationExecutionSchema};

std::optional<ArtifactRootReference>
uniqueDependencyWithSchema(const CaseArtifactResolution::Entry &entry,
                           const ArtifactSchemaDescriptor &schema) {
  std::optional<ArtifactRootReference> result;
  for (const ArtifactRootReference &reference : entry.dependencyClosure) {
    if (reference.schemaIdentity != schema.identity ||
        reference.schemaVersion != schema.version)
      continue;
    if (result)
      return std::nullopt;
    result = reference;
  }
  return result;
}

llvm::Error
verifyExecutionCompatibility(const ArtifactRootReference &subject,
                             const EvaluationCase &,
                             const EvaluationSubjectBindings &bindings,
                             const CaseArtifactResolution &resolution,
                             const ArtifactStore &, const BlobStore &) {
  const auto references = bindings.subjects(kReferenceExecutionRole);
  const auto candidates = bindings.subjects(kCandidateExecutionRole);
  if (references.size() != 1 || candidates.size() != 1)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "simulation_comparison_model_invalid: execution roles are not total");
  if (subject != candidates.front())
    return llvm::Error::success();
  const auto *referenceEntry = resolution.find(references.front());
  const auto *candidateEntry = resolution.find(candidates.front());
  if (!referenceEntry || !candidateEntry)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "simulation_comparison_model_invalid: execution is unresolved");
  const auto referenceWorkload = uniqueDependencyWithSchema(
      *referenceEntry, sim::simulationWorkloadSchema);
  const auto candidateWorkload = uniqueDependencyWithSchema(
      *candidateEntry, sim::simulationWorkloadSchema);
  const auto referenceInput = uniqueDependencyWithSchema(
      *referenceEntry, sim::simulationRuntimeInputSchema);
  const auto candidateInput = uniqueDependencyWithSchema(
      *candidateEntry, sim::simulationRuntimeInputSchema);
  if (!referenceWorkload || !candidateWorkload || !referenceInput ||
      !candidateInput || *referenceWorkload != *candidateWorkload ||
      *referenceInput != *candidateInput)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "simulation_comparison_model_invalid: executions do not share one "
        "exact workload and runtime input");
  return llvm::Error::success();
}

const CaseSubjectRoleDescriptor kSubjectRoles[] = {
    {kReferenceExecutionRole, "reference_execution",
     SubjectRoleCardinality::ExactlyOne, kExecutionSchemas, nullptr},
    {kCandidateExecutionRole, "candidate_execution",
     SubjectRoleCardinality::ExactlyOne, kExecutionSchemas,
     &verifyExecutionCompatibility}};

const EvaluationCaseSignatureDescriptor kCaseSignature{
    builtinEvaluationCaseKind(kCase),
    "simulation_execution_comparison",
    "Two exact SimulationExecution roots with one compatible workload and "
    "runtime input.",
    kSubjectRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbsentReferenceCycle{},
    {}};

const ScopeFormRef kWholeCaseScopeForms[] = {kWholeExactCaseScope};
const FindingCapability kFindingCapabilities[] = {{
    standard_findings::FunctionalMismatch,
    kWholeCaseScopeForms,
    findingResultFormMask(FindingResultForm::Absent) |
        findingResultFormMask(FindingResultForm::Present) |
        findingResultFormMask(FindingResultForm::NotApplicable),
}};
const ModeledPhenomenon kModeledPhenomena[] = {
    ModeledPhenomenon::CanonicalDataflow};

struct EmptySimulationComparisonConfig final {};

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.simulation_comparison.config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptySimulationComparisonConfig{});
}

llvm::Expected<std::vector<std::uint8_t>>
encodeConfig(const OwnerValue &value) {
  if (!value.getIf<EmptySimulationComparisonConfig>())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "simulation_comparison_model_invalid: config has the wrong owner "
        "type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue> adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                       const ComponentViewDigest &) {
  if (!bytes.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "simulation_comparison_model_invalid: config view must be empty");
  return OwnerValue::get(EmptySimulationComparisonConfig{});
}

const ResolvedModelConfigViewContract kConfigView{
    configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig};

const EvaluationModelDescriptor kModelDescriptor{
    builtinEvaluationModelKind(kModel),
    "simulation_execution_comparison",
    "loom.simulation_comparison.exact_spatial.v1",
    caseSignatureRef(),
    {},
    {},
    kFindingCapabilities,
    {},
    {},
    kConfigView,
    kModeledPhenomena,
    EvaluationExecutionMethod::Simulation,
    {},
    DeterminismContract::Deterministic,
    {},
    ProviderForm::InProcess};

llvm::Expected<bool> hasDeterministicExactRelation(
    const sim::CanonicalSimulationExecution &execution,
    const CaseArtifactResolution &resolution, const ArtifactStore &store,
    const BlobStore &blobs) {
  auto request =
      importEvaluationRequest(execution.request(), resolution, store, blobs);
  if (!request)
    return request.takeError();
  if (!request->workload() || !request->runtimeInput())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "simulation_comparison_model_invalid: execution Request has no "
        "Spatial inputs");
  auto inputs = sim::importSpatialSimulationInputs(
      *request->workload(), *request->runtimeInput(), store);
  if (!inputs)
    return inputs.takeError();
  auto view = inputs->dataflow.view();
  if (!view)
    return view.takeError();
  for (const dataflow::CanonicalActorView &actor : view->actors()) {
    auto projection =
        dataflow::projectRegisteredActorSchemaProjection(actor.op);
    if (!projection)
      return projection.takeError();
    const auto *memory =
        std::get_if<dataflow::MemoryContractPayload>(&projection->payload);
    if (!memory)
      continue;
    const auto *plain = std::get_if<dataflow::PlainAccessProjection>(memory);
    if (!plain || plain->isVolatile)
      return false;
  }
  return true;
}

FindingResult
compareExecutions(const sim::CanonicalSimulationExecution &reference,
                  const sim::CanonicalSimulationExecution &candidate,
                  bool exactRelationApplies) {
  if (!exactRelationApplies)
    return FindingResult{
        NotApplicableFinding{NotApplicableReason::UndefinedForSubject}};
  const bool sameTerminal =
      reference.terminal().index() == candidate.terminal().index();
  if (!sameTerminal)
    return FindingResult{PresentFinding{{FindingOccurrence::get(
        standard_findings::FunctionalMismatchOccurrence{})}}};
  if (!std::holds_alternative<sim::RetiredExecution>(reference.terminal()))
    return FindingResult{
        NotApplicableFinding{NotApplicableReason::UndefinedForSubject}};
  bool mismatch = reference.root().index() != candidate.root().index();
  if (!mismatch && reference.spatial())
    mismatch = !sim::haveExactlyEqualSpatialFunctionalObservations(
        reference.spatial()->functionalObservations,
        candidate.spatial()->functionalObservations);
  if (!mismatch && reference.system())
    mismatch = !sim::haveExactlyEqualSystemFunctionalObservations(
        reference.system()->functionalObservations,
        candidate.system()->functionalObservations);
  if (!mismatch)
    return FindingResult{AbsentFinding{}};
  return FindingResult{PresentFinding{{FindingOccurrence::get(
      standard_findings::FunctionalMismatchOccurrence{})}}};
}

llvm::Expected<EvaluationModelResult>
evaluate(const EvaluationRequest &request,
         const CaseArtifactResolution &resolution,
         const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  const auto references =
      request.subjectBindings().subjects(kReferenceExecutionRole);
  const auto candidates =
      request.subjectBindings().subjects(kCandidateExecutionRole);
  if (references.size() != 1 || candidates.size() != 1)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "simulation_comparison_model_invalid: Request roles are not total");
  auto reference = sim::importSimulationExecution(
      references.front(), resolution, artifactStore, blobStore);
  if (!reference)
    return reference.takeError();
  auto candidate = sim::importSimulationExecution(
      candidates.front(), resolution, artifactStore, blobStore);
  if (!candidate)
    return candidate.takeError();
  auto deterministic = hasDeterministicExactRelation(*reference, resolution,
                                                     artifactStore, blobStore);
  if (!deterministic)
    return deterministic.takeError();

  std::vector<FindingResult> findings;
  findings.reserve(request.findingRequests().size());
  for (const FindingRequest &finding : request.findingRequests()) {
    if (finding.query().kind != standard_findings::FunctionalMismatch)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "simulation_comparison_model_invalid: unsupported finding");
    findings.push_back(
        compareExecutions(*reference, *candidate, *deterministic));
  }
  return EvaluationModelResult{{}, CompletedEvidence{{}, std::move(findings)}};
}

const EvaluationModelProvider kProvider{
    kModelDescriptor.reference(), EvaluationModelInProcessProvider{&evaluate}};

using ResolutionMap =
    std::map<ArtifactRootReference, std::vector<ArtifactRootReference>,
             decltype(&artifactRootReferenceLess)>;

llvm::Error addResolutionEntry(ResolutionMap &entries,
                               const CaseArtifactResolution::Entry &entry) {
  auto [found, inserted] =
      entries.try_emplace(entry.artifact, entry.dependencyClosure);
  if (!inserted && found->second != entry.dependencyClosure)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "simulation_comparison_model_invalid: conflicting exact dependency "
        "closure");
  return llvm::Error::success();
}

std::vector<ArtifactRootReference>
resolutionClosure(const CaseArtifactResolution &resolution) {
  std::vector<ArtifactRootReference> closure;
  for (const CaseArtifactResolution::Entry &entry : resolution.entries()) {
    closure.push_back(entry.artifact);
    closure.insert(closure.end(), entry.dependencyClosure.begin(),
                   entry.dependencyClosure.end());
  }
  llvm::sort(closure, artifactRootReferenceLess);
  closure.erase(std::unique(closure.begin(), closure.end()), closure.end());
  return closure;
}

llvm::Expected<CaseArtifactResolution>
mergeResolutions(const ArtifactRootReference &referenceExecution,
                 const CaseArtifactResolution &referenceResolution,
                 const ArtifactRootReference &referenceRequest,
                 const ArtifactRootReference &candidateExecution,
                 const CaseArtifactResolution &candidateResolution,
                 const ArtifactRootReference &candidateRequest) {
  ResolutionMap entries(&artifactRootReferenceLess);
  for (const CaseArtifactResolution::Entry &entry :
       referenceResolution.entries())
    if (llvm::Error error = addResolutionEntry(entries, entry))
      return std::move(error);
  for (const CaseArtifactResolution::Entry &entry :
       candidateResolution.entries())
    if (llvm::Error error = addResolutionEntry(entries, entry))
      return std::move(error);

  const std::vector<ArtifactRootReference> referenceClosure =
      resolutionClosure(referenceResolution);
  const std::vector<ArtifactRootReference> candidateClosure =
      resolutionClosure(candidateResolution);
  if (llvm::Error error =
          addResolutionEntry(entries, {referenceRequest, referenceClosure}))
    return std::move(error);
  if (llvm::Error error =
          addResolutionEntry(entries, {candidateRequest, candidateClosure}))
    return std::move(error);
  std::vector<ArtifactRootReference> referenceExecutionClosure =
      referenceClosure;
  referenceExecutionClosure.push_back(referenceRequest);
  llvm::sort(referenceExecutionClosure, artifactRootReferenceLess);
  referenceExecutionClosure.erase(std::unique(referenceExecutionClosure.begin(),
                                              referenceExecutionClosure.end()),
                                  referenceExecutionClosure.end());
  std::vector<ArtifactRootReference> candidateExecutionClosure =
      candidateClosure;
  candidateExecutionClosure.push_back(candidateRequest);
  llvm::sort(candidateExecutionClosure, artifactRootReferenceLess);
  candidateExecutionClosure.erase(std::unique(candidateExecutionClosure.begin(),
                                              candidateExecutionClosure.end()),
                                  candidateExecutionClosure.end());
  if (llvm::Error error = addResolutionEntry(
          entries, {referenceExecution, referenceExecutionClosure}))
    return std::move(error);
  if (llvm::Error error = addResolutionEntry(
          entries, {candidateExecution, candidateExecutionClosure}))
    return std::move(error);

  std::vector<CaseArtifactResolution::Entry> result;
  result.reserve(entries.size());
  for (auto &[artifact, closure] : entries)
    result.push_back({artifact, std::move(closure)});
  return CaseArtifactResolution::get(std::move(result));
}

} // namespace

llvm::Error registerSimulationComparisonModel() {
  if (llvm::Error error = registerDfgSimulationModel())
    return error;
  if (llvm::Error error = registerCgraSimulationModel())
    return error;
  if (llvm::Error error = standard_findings::registerStandardFindings())
    return error;
  if (llvm::Error error = registerEvaluationCaseSignature(kCaseSignature))
    return error;
  if (llvm::Error error = registerEvaluationModelDescriptor(kModelDescriptor))
    return error;
  return registerEvaluationModelProvider(kProvider);
}

llvm::Expected<PreparedSimulationComparisonEvaluation>
prepareSimulationComparisonEvaluation(
    const ArtifactRootReference &referenceExecution,
    const CaseArtifactResolution &referenceResolution,
    const ArtifactRootReference &candidateExecution,
    const CaseArtifactResolution &candidateResolution,
    const ResolvedConfig &config, const ArtifactStore &artifactStore,
    const BlobStore &blobStore) {
  if (llvm::Error error = registerSimulationComparisonModel())
    return std::move(error);
  auto referenceRequest = sim::simulationExecutionRequestReference(
      referenceExecution, artifactStore);
  if (!referenceRequest)
    return referenceRequest.takeError();
  auto candidateRequest = sim::simulationExecutionRequestReference(
      candidateExecution, artifactStore);
  if (!candidateRequest)
    return candidateRequest.takeError();
  auto resolution = mergeResolutions(referenceExecution, referenceResolution,
                                     *referenceRequest, candidateExecution,
                                     candidateResolution, *candidateRequest);
  if (!resolution)
    return resolution.takeError();
  if (auto imported = sim::importSimulationExecution(
          referenceExecution, *resolution, artifactStore, blobStore);
      !imported)
    return imported.takeError();
  if (auto imported = sim::importSimulationExecution(
          candidateExecution, *resolution, artifactStore, blobStore);
      !imported)
    return imported.takeError();

  auto bindings = EvaluationSubjectBindings::get(
      {{kReferenceExecutionRole, {referenceExecution}},
       {kCandidateExecutionRole, {candidateExecution}}});
  if (!bindings)
    return bindings.takeError();
  auto evaluationCase = EvaluationCase::get(
      caseSignatureRef(), std::move(*bindings), std::nullopt, std::nullopt, {},
      *resolution, artifactStore, blobStore);
  if (!evaluationCase)
    return evaluationCase.takeError();
  auto finding = FindingRequest::get(
      FindingQuery{standard_findings::FunctionalMismatch,
                   EvaluationScope{kWholeExactCaseScope, {}}},
      {}, *evaluationCase, *resolution, artifactStore);
  if (!finding)
    return finding.takeError();
  auto modelBinding =
      ResolvedModelBinding::project(kModelDescriptor.reference(), {}, config);
  if (!modelBinding)
    return modelBinding.takeError();
  auto request = EvaluationRequest::get(*evaluationCase, {}, {*finding},
                                        std::move(*modelBinding), 0,
                                        *resolution, artifactStore, blobStore);
  if (!request)
    return request.takeError();
  auto published = publishEvaluationRequest(*request, artifactStore);
  if (!published)
    return published.takeError();
  return PreparedSimulationComparisonEvaluation{
      std::move(*request), std::move(*resolution), FindingRequestOrdinal(0)};
}

llvm::Expected<EvaluationEvidence> evaluateSimulationComparison(
    const PreparedSimulationComparisonEvaluation &prepared,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  RequestVerifier verifier(prepared.resolution, artifactStore, blobStore);
  if (llvm::Error error = verifier.verify(prepared.request))
    return std::move(error);
  auto result =
      evaluate(prepared.request, prepared.resolution, artifactStore, blobStore);
  if (!result)
    return result.takeError();
  return EvaluationEvidence::get(prepared.request,
                                 std::move(result->outputBindings),
                                 std::move(result->outcome),
                                 prepared.resolution, artifactStore, blobStore);
}

} // namespace loom::evaluation::models
