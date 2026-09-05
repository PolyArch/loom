#include "Application/ProductOracleEvaluation.h"

#include "Application/Package.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/ProductionRegistry.h"
#include "Evaluation/Request.h"
#include "Evaluation/StandardFindings.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::application {
namespace {

constexpr evaluation::BuiltinEvaluationCase kCase =
    evaluation::BuiltinEvaluationCase::ApplicationProductOracle;
constexpr evaluation::BuiltinEvaluationModel kModel =
    evaluation::BuiltinEvaluationModel::ApplicationProductOracle;
constexpr evaluation::CaseSubjectRoleRef kManifestRole(0);
constexpr evaluation::CaseSubjectRoleRef kExecutionRole(1);
constexpr evaluation::ScopeFormRef kWholeExactCaseScope(0);

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "application_product_oracle_invalid: " + message);
}

evaluation::EvaluationCaseSignatureRef caseSignatureRef() {
  return llvm::cantFail(evaluation::EvaluationCaseSignatureRef::get(
      evaluation::evaluationSchemaVersion(),
      evaluation::builtinEvaluationCaseKind(kCase)));
}

const ArtifactSchemaDescriptor *const kManifestSchemas[] = {
    &applicationRuntimeManifestSchema};
const ArtifactSchemaDescriptor *const kExecutionSchemas[] = {
    &sim::simulationExecutionSchema};

llvm::Error verifyExecutionCompatibility(
    const ArtifactRootReference &subject, const evaluation::EvaluationCase &,
    const evaluation::EvaluationSubjectBindings &bindings,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  const auto manifests = bindings.subjects(kManifestRole);
  const auto executions = bindings.subjects(kExecutionRole);
  if (manifests.size() != 1 || executions.size() != 1)
    return invalid("case roles are not total");
  if (subject != executions.front())
    return llvm::Error::success();
  auto manifest =
      importApplicationRuntimeManifest(manifests.front(), artifacts, blobs);
  if (!manifest)
    return manifest.takeError();
  if (!manifest->manifest().productOracle())
    return invalid("runtime manifest has no product oracle contract");
  auto execution =
      sim::importSimulationExecution(subject, resolution, artifacts, blobs);
  if (!execution)
    return execution.takeError();
  if (!execution->system())
    return invalid("execution is not a System execution");
  auto request = evaluation::importEvaluationRequest(
      execution->request(), resolution, artifacts, blobs);
  if (!request)
    return request.takeError();
  if (!request->workload() ||
      *request->workload() != manifest->manifest().activationWorkload() ||
      !request->runtimeInput() ||
      *request->runtimeInput() !=
          manifest->manifest().activationRuntimeInput())
    return invalid("execution does not use the manifest activation inputs");
  return llvm::Error::success();
}

const evaluation::CaseSubjectRoleDescriptor kSubjectRoles[] = {
    {kManifestRole, "application_runtime_manifest",
     evaluation::SubjectRoleCardinality::ExactlyOne, kManifestSchemas,
     nullptr},
    {kExecutionRole, "system_execution",
     evaluation::SubjectRoleCardinality::ExactlyOne, kExecutionSchemas,
     &verifyExecutionCompatibility}};

const evaluation::EvaluationCaseSignatureDescriptor kCaseSignature{
    evaluation::builtinEvaluationCaseKind(kCase),
    "application_product_oracle",
    "One Application runtime manifest and one exact System execution of its "
    "activation inputs.",
    kSubjectRoles,
    evaluation::ArtifactRequirement::Forbidden,
    {},
    evaluation::ArtifactRequirement::Forbidden,
    {},
    nullptr,
    evaluation::AbsentReferenceCycle{},
    {}};

const evaluation::ScopeFormRef kWholeCaseScopeForms[] = {
    kWholeExactCaseScope};
const evaluation::FindingCapability kFindingCapabilities[] = {{
    evaluation::standard_findings::FunctionalMismatch,
    kWholeCaseScopeForms,
    evaluation::findingResultFormMask(evaluation::FindingResultForm::Absent) |
        evaluation::findingResultFormMask(
            evaluation::FindingResultForm::Present),
}};
const evaluation::ModeledPhenomenon kModeledPhenomena[] = {
    evaluation::ModeledPhenomenon::StructuredProgram,
    evaluation::ModeledPhenomenon::SystemMemoryHierarchy};

struct EmptyProductOracleConfig final {};

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.application.product_oracle.config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<evaluation::OwnerValue>
projectConfig(const ResolvedConfig &) {
  return evaluation::OwnerValue::get(EmptyProductOracleConfig{});
}

llvm::Expected<std::vector<std::uint8_t>>
encodeConfig(const evaluation::OwnerValue &value) {
  if (!value.getIf<EmptyProductOracleConfig>())
    return invalid("config has the wrong owner type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<evaluation::OwnerValue>
adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
            const ComponentViewDigest &) {
  if (!bytes.empty())
    return invalid("config view must be empty");
  return evaluation::OwnerValue::get(EmptyProductOracleConfig{});
}

const evaluation::ResolvedModelConfigViewContract kConfigView{
    configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig};

const evaluation::EvaluationModelDescriptor kModelDescriptor{
    evaluation::builtinEvaluationModelKind(kModel),
    "application_product_oracle",
    "loom.application.product_oracle.exact_output.v1",
    caseSignatureRef(),
    {},
    {},
    kFindingCapabilities,
    {},
    {},
    kConfigView,
    kModeledPhenomena,
    evaluation::EvaluationExecutionMethod::Simulation,
    {},
    evaluation::DeterminismContract::Deterministic,
    {},
    ProviderForm::InProcess};

bool matchesProductOracle(
    const ApplicationRuntimeManifest &manifest,
    const sim::CanonicalSimulationExecution &execution,
    const BlobStore &blobs) {
  const ProductOracleContract &contract = *manifest.productOracle();
  if (!std::holds_alternative<sim::RetiredExecution>(execution.terminal()) ||
      !execution.system())
    return false;
  const sim::SystemFunctionalObservations &observations =
      execution.system()->functionalObservations;
  if (observations.valueResults.size() != 1 ||
      observations.memories.size() != 1)
    return false;
  const auto *result =
      std::get_if<sim::PublishedValueResult>(&observations.valueResults.front());
  if (!result || result->value.tokenCount != 1 ||
      result->value.lanes.size() != 1 ||
      result->value.lanes.front().state != sim::SemanticState::Defined ||
      result->value.lanes.front().pointerTarget ||
      result->value.lanes.front().bits.getBitWidth() != 32 ||
      !result->value.lanes.front().bits.isZero())
    return false;
  const auto *memory =
      std::get_if<sim::FullMemoryObservation>(&observations.memories.front());
  if (!memory)
    return false;
  auto expected = blobs.get(contract.expectedOutput);
  if (!expected) {
    llvm::consumeError(expected.takeError());
    return false;
  }
  if (memory->bytes.size() != expected->size())
    return false;
  return llvm::equal(
      memory->bytes, *expected,
      [](const sim::SemanticMemoryByte &actual, std::uint8_t expectedByte) {
        return actual.state == sim::SemanticState::Defined &&
               actual.value == expectedByte;
      });
}

llvm::Expected<evaluation::EvaluationModelResult> evaluateModel(
    const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  const auto manifests = request.subjectBindings().subjects(kManifestRole);
  const auto executions = request.subjectBindings().subjects(kExecutionRole);
  if (manifests.size() != 1 || executions.size() != 1)
    return invalid("Request roles are not total");
  auto manifest =
      importApplicationRuntimeManifest(manifests.front(), artifacts, blobs);
  if (!manifest)
    return manifest.takeError();
  auto execution = sim::importSimulationExecution(
      executions.front(), resolution, artifacts, blobs);
  if (!execution)
    return execution.takeError();
  if (std::holds_alternative<sim::StoppedByLimitExecution>(
          execution->terminal()))
    return evaluation::EvaluationModelResult{
        {}, evaluation::CancelledOrTimeoutEvidence{
                evaluation::OutcomeReason::ExecutionLimitReached}};

  std::vector<evaluation::FindingResult> findings;
  findings.reserve(request.findingRequests().size());
  for (const evaluation::FindingRequest &finding : request.findingRequests()) {
    if (finding.query().kind !=
        evaluation::standard_findings::FunctionalMismatch)
      return invalid("Request contains an unsupported finding");
    if (matchesProductOracle(manifest->manifest(), *execution, blobs))
      findings.push_back({evaluation::AbsentFinding{}});
    else
      findings.push_back({evaluation::PresentFinding{{
          evaluation::FindingOccurrence::get(
              evaluation::standard_findings::FunctionalMismatchOccurrence{})}}});
  }
  return evaluation::EvaluationModelResult{
      {}, evaluation::CompletedEvidence{{}, std::move(findings)}};
}

const evaluation::EvaluationModelProvider kProvider{
    kModelDescriptor.reference(),
    evaluation::EvaluationModelInProcessProvider{&evaluateModel}};

using ResolutionMap =
    std::map<ArtifactRootReference, std::vector<ArtifactRootReference>,
             decltype(&artifactRootReferenceLess)>;

llvm::Error addResolutionEntry(
    ResolutionMap &entries,
    const evaluation::CaseArtifactResolution::Entry &entry) {
  auto [found, inserted] =
      entries.try_emplace(entry.artifact, entry.dependencyClosure);
  if (!inserted && found->second != entry.dependencyClosure)
    return invalid("conflicting exact dependency closure");
  return llvm::Error::success();
}

std::vector<ArtifactRootReference> resolutionClosure(
    const evaluation::CaseArtifactResolution &resolution) {
  std::vector<ArtifactRootReference> closure;
  for (const evaluation::CaseArtifactResolution::Entry &entry :
       resolution.entries()) {
    closure.push_back(entry.artifact);
    closure.insert(closure.end(), entry.dependencyClosure.begin(),
                   entry.dependencyClosure.end());
  }
  llvm::sort(closure, artifactRootReferenceLess);
  closure.erase(std::unique(closure.begin(), closure.end()), closure.end());
  return closure;
}

llvm::Expected<evaluation::CaseArtifactResolution> extendResolution(
    const FinalizedApplicationRuntimeManifest &manifest,
    const ArtifactRootReference &execution,
    const evaluation::CaseArtifactResolution &executionResolution,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  ResolutionMap entries(&artifactRootReferenceLess);
  for (const evaluation::CaseArtifactResolution::Entry &entry :
       executionResolution.entries())
    if (llvm::Error error = addResolutionEntry(entries, entry))
      return std::move(error);

  auto entryDeployment = deployment::importDeployment(
      manifest.manifest().deployment(), artifacts, blobs);
  if (!entryDeployment)
    return entryDeployment.takeError();
  auto packageClosure = deriveApplicationPackageClosure(
      manifest, *entryDeployment, artifacts, blobs);
  if (!packageClosure)
    return packageClosure.takeError();
  std::vector<ArtifactRootReference> manifestClosure =
      std::move(packageClosure->artifacts);
  manifestClosure.erase(
      std::remove(manifestClosure.begin(), manifestClosure.end(),
                  manifest.reference()),
      manifestClosure.end());
  if (llvm::Error error = addResolutionEntry(
          entries, {manifest.reference(), std::move(manifestClosure)}))
    return std::move(error);

  if (!entries.count(execution)) {
    auto executionRequest =
        sim::simulationExecutionRequestReference(execution, artifacts);
    if (!executionRequest)
      return executionRequest.takeError();
    const std::vector<ArtifactRootReference> requestClosure =
        resolutionClosure(executionResolution);
    if (llvm::Error error =
            addResolutionEntry(entries, {*executionRequest, requestClosure}))
      return std::move(error);
    std::vector<ArtifactRootReference> executionClosure = requestClosure;
    executionClosure.push_back(*executionRequest);
    llvm::sort(executionClosure, artifactRootReferenceLess);
    executionClosure.erase(
        std::unique(executionClosure.begin(), executionClosure.end()),
        executionClosure.end());
    if (llvm::Error error = addResolutionEntry(
            entries, {execution, std::move(executionClosure)}))
      return std::move(error);
  }

  std::vector<evaluation::CaseArtifactResolution::Entry> result;
  result.reserve(entries.size());
  for (auto &[artifact, closure] : entries)
    result.push_back({artifact, std::move(closure)});
  return evaluation::CaseArtifactResolution::get(std::move(result));
}

} // namespace

llvm::Error registerProductOracleEvaluationModel() {
  if (llvm::Error error = evaluation::registerProductionEvaluationRegistry())
    return error;
  if (llvm::Error error =
          evaluation::standard_findings::registerStandardFindings())
    return error;
  if (llvm::Error error =
          evaluation::registerEvaluationCaseSignature(kCaseSignature))
    return error;
  if (llvm::Error error =
          evaluation::registerEvaluationModelDescriptor(kModelDescriptor))
    return error;
  return evaluation::registerEvaluationModelProvider(kProvider);
}

llvm::Expected<PreparedProductOracleEvaluation>
prepareProductOracleEvaluation(
    const FinalizedApplicationRuntimeManifest &manifest,
    const ArtifactRootReference &execution,
    const evaluation::CaseArtifactResolution &executionResolution,
    const ResolvedConfig &config, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  if (!manifest.manifest().productOracle())
    return invalid("runtime manifest has no product oracle contract");
  if (llvm::Error error = registerProductOracleEvaluationModel())
    return std::move(error);
  auto resolution = extendResolution(manifest, execution, executionResolution,
                                     artifacts, blobs);
  if (!resolution)
    return resolution.takeError();
  auto bindings = evaluation::EvaluationSubjectBindings::get(
      {{kManifestRole, {manifest.reference()}},
       {kExecutionRole, {execution}}});
  if (!bindings)
    return bindings.takeError();
  auto evaluationCase = evaluation::EvaluationCase::get(
      caseSignatureRef(), std::move(*bindings), std::nullopt, std::nullopt, {},
      *resolution, artifacts, blobs);
  if (!evaluationCase)
    return evaluationCase.takeError();
  auto finding = evaluation::FindingRequest::get(
      {evaluation::standard_findings::FunctionalMismatch,
       evaluation::EvaluationScope{kWholeExactCaseScope, {}}},
      {}, *evaluationCase, *resolution, artifacts);
  if (!finding)
    return finding.takeError();
  auto model = evaluation::ResolvedModelBinding::project(
      kModelDescriptor.reference(), {}, config);
  if (!model)
    return model.takeError();
  auto request = evaluation::EvaluationRequest::get(
      *evaluationCase, {}, {*finding}, std::move(*model), 0, *resolution,
      artifacts, blobs);
  if (!request)
    return request.takeError();
  auto published = evaluation::publishEvaluationRequest(*request, artifacts);
  if (!published)
    return published.takeError();
  return PreparedProductOracleEvaluation{
      std::move(*request), std::move(*resolution),
      evaluation::FindingRequestOrdinal(0)};
}

} // namespace loom::application
