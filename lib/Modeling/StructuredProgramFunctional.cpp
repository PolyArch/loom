#include "Evaluation/Models/StructuredProgramFunctional.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/ResolvedConfig.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/StandardFindings.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SourceBackedDfgValidation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr EvaluationCaseKind kCaseKind(2);
constexpr EvaluationModelKind kModelKind(4);
constexpr CaseSubjectRoleRef kCandidateRole(0);
constexpr ScopeFormRef kWholeExactCaseScope(0);

EvaluationCaseSignatureRef caseSignatureRef() {
  return llvm::cantFail(
      EvaluationCaseSignatureRef::get(evaluationSchemaVersion(), kCaseKind));
}

const ArtifactSchemaDescriptor *const kStructuredSchemas[] = {
    &frontend::structuredProgramArtifactSchema};
const ArtifactSchemaDescriptor *const kWorkloadSchemas[] = {
    &sim::simulationWorkloadSchema};
const ArtifactSchemaDescriptor *const kRuntimeInputSchemas[] = {
    &sim::simulationRuntimeInputSchema};

const CaseSubjectRoleDescriptor kSubjectRoles[] = {
    {kCandidateRole, "selected_structured_program",
     SubjectRoleCardinality::ExactlyOne, kStructuredSchemas, nullptr}};

llvm::Error verifyWorkloadCompatibility(
    const EvaluationSubjectBindings &,
    const std::optional<ArtifactRootReference> &workload,
    const std::optional<ArtifactRootReference> &runtimeInput,
    const CaseArtifactResolution &resolution) {
  if (!workload || !runtimeInput)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: workload inputs are not total");
  const CaseArtifactResolution::Entry *workloadEntry =
      resolution.find(*workload);
  const CaseArtifactResolution::Entry *runtimeEntry =
      resolution.find(*runtimeInput);
  if (!workloadEntry || !runtimeEntry ||
      !CaseArtifactResolution::reaches(*runtimeEntry, *workload))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: runtime input does not reach "
        "its exact workload");
  const bool hasSource = llvm::any_of(
      workloadEntry->dependencyClosure,
      [](const ArtifactRootReference &reference) {
        return reference.schemaIdentity ==
                   frontend::structuredProgramArtifactSchema.identity &&
               reference.schemaVersion ==
                   frontend::structuredProgramArtifactSchema.version;
      });
  if (!hasSource)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: workload has no exact source "
        "program");
  return llvm::Error::success();
}

const EvaluationCaseSignatureDescriptor kCaseSignature{
    kCaseKind,
    "structured_program_functional_comparison",
    "One exact Structured candidate compared with its workload-owned source.",
    kSubjectRoles,
    ArtifactRequirement::Required,
    kWorkloadSchemas,
    ArtifactRequirement::Required,
    kRuntimeInputSchemas,
    &verifyWorkloadCompatibility,
    AbsentReferenceCycle{},
    {}};

const ScopeFormRef kWholeCaseScopeForms[] = {kWholeExactCaseScope};
const FindingCapability kFindingCapabilities[] = {
    {standard_findings::FunctionalMismatch, kWholeCaseScopeForms,
     findingResultFormMask(FindingResultForm::Absent) |
         findingResultFormMask(FindingResultForm::Present) |
         findingResultFormMask(FindingResultForm::NotApplicable)}};
const ModeledPhenomenon kModeledPhenomena[] = {
    ModeledPhenomenon::StructuredProgram};

struct EmptyFunctionalConfig final {};

struct SourceObservationCacheEntry final {
  ArtifactRootReference source;
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;
  sim::NativeStructuredProgramObservations observations;
};

std::optional<SourceObservationCacheEntry> &sourceObservationCache() {
  thread_local std::optional<SourceObservationCacheEntry> cache;
  return cache;
}

bool haveEquivalentSourceObservations(
    const sim::NativeStructuredProgramObservations &lhs,
    const sim::NativeStructuredProgramObservations &rhs) {
  if (!sim::haveEquivalentFunctionalObservations(lhs, rhs) ||
      lhs.blockActivations.size() != rhs.blockActivations.size())
    return false;
  return llvm::equal(
      lhs.blockActivations, rhs.blockActivations,
      [](const sim::NativeStructuredBlockActivation &lhsActivation,
         const sim::NativeStructuredBlockActivation &rhsActivation) {
        return lhsActivation.block == rhsActivation.block &&
               lhsActivation.activations == rhsActivation.activations;
      });
}

llvm::Error primeSourceObservationCache(
    const ArtifactRootReference &source,
    const ArtifactRootReference &workloadReference,
    const ArtifactRootReference &runtimeInputReference,
    const sim::NativeStructuredProgramObservations &observations) {
  std::optional<SourceObservationCacheEntry> &cache = sourceObservationCache();
  if (cache && cache->source == source &&
      cache->workload == workloadReference &&
      cache->runtimeInput == runtimeInputReference) {
    if (!haveEquivalentSourceObservations(cache->observations, observations))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_functional_model_invalid: nondeterministic source "
          "observations");
    return llvm::Error::success();
  }
  cache.emplace(SourceObservationCacheEntry{
      source, workloadReference, runtimeInputReference, observations});
  return llvm::Error::success();
}

enum class ReplayResultKind : std::uint8_t {
  Equivalent,
  Mismatch,
  Inapplicable,
  Unsupported,
  ExecutionFailed,
};

struct CachedReplayResult final {
  ReplayResultKind kind = ReplayResultKind::Unsupported;
  std::uint64_t dynamicActivations = 0;
  std::uint64_t wavefrontSteps = 0;
  std::uint64_t eventCount = 0;

  friend bool operator==(const CachedReplayResult &lhs,
                         const CachedReplayResult &rhs) {
    return lhs.kind == rhs.kind &&
           lhs.dynamicActivations == rhs.dynamicActivations &&
           lhs.wavefrontSteps == rhs.wavefrontSteps &&
           lhs.eventCount == rhs.eventCount;
  }
};

std::map<std::vector<std::uint8_t>, CachedReplayResult> &replayCache() {
  static std::map<std::vector<std::uint8_t>, CachedReplayResult> cache;
  return cache;
}

std::mutex &replayCacheMutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<std::uint8_t>
replayCacheKey(const ArtifactRootReference &candidate,
               const ArtifactRootReference &workload,
               const ArtifactRootReference &runtimeInput) {
  std::vector<std::uint8_t> key = encodeArtifactRootReference(candidate);
  for (const ArtifactRootReference *reference : {&workload, &runtimeInput}) {
    std::vector<std::uint8_t> bytes = encodeArtifactRootReference(*reference);
    key.insert(key.end(), bytes.begin(), bytes.end());
  }
  return key;
}

llvm::Expected<CachedReplayResult> classifyReplayResult(
    llvm::Expected<sim::SourceBackedDfgValidationResult> replay) {
  if (replay) {
    ReplayResultKind kind = ReplayResultKind::Unsupported;
    switch (replay->status) {
    case sim::SourceBackedDfgValidationStatus::Equivalent:
      kind = ReplayResultKind::Equivalent;
      break;
    case sim::SourceBackedDfgValidationStatus::Mismatch:
      kind = ReplayResultKind::Mismatch;
      break;
    case sim::SourceBackedDfgValidationStatus::Inapplicable:
      kind = ReplayResultKind::Inapplicable;
      break;
    }
    return CachedReplayResult{kind, replay->dynamicActivations,
                              replay->wavefrontSteps, replay->eventCount};
  }

  std::error_code code;
  std::string message;
  llvm::raw_string_ostream stream(message);
  llvm::handleAllErrors(std::move(replay).takeError(),
                        [&](const llvm::ErrorInfoBase &failure) {
                          code = failure.convertToErrorCode();
                          failure.log(stream);
                        });
  stream.flush();
  if (code == std::make_error_code(std::errc::not_supported))
    return CachedReplayResult{ReplayResultKind::Unsupported};
  if (code == std::make_error_code(std::errc::io_error))
    return CachedReplayResult{ReplayResultKind::ExecutionFailed};
  return llvm::createStringError(code ? code : llvm::inconvertibleErrorCode(),
                                 "%s", message.c_str());
}

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.structured_functional.config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptyFunctionalConfig{});
}

llvm::Expected<std::vector<std::uint8_t>>
encodeConfig(const OwnerValue &value) {
  if (!value.getIf<EmptyFunctionalConfig>())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured functional config has the wrong owner type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue> adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                       const ComponentViewDigest &) {
  if (!bytes.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured functional config must be empty");
  return OwnerValue::get(EmptyFunctionalConfig{});
}

const ResolvedModelConfigViewContract kConfigView{
    configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig};

const EvaluationModelDescriptor kModelDescriptor{
    kModelKind,
    "structured_program_functional",
    "loom.structured_program.functional.v1",
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
    {}};

llvm::Expected<EvaluationModelResult> classifyNativeFailure(llvm::Error error) {
  std::error_code code;
  std::string message;
  llvm::raw_string_ostream stream(message);
  llvm::handleAllErrors(std::move(error),
                        [&](const llvm::ErrorInfoBase &failure) {
                          code = failure.convertToErrorCode();
                          failure.log(stream);
                        });
  stream.flush();
  if (code == std::make_error_code(std::errc::not_supported))
    return EvaluationModelResult{
        {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  if (code == std::make_error_code(std::errc::io_error))
    return EvaluationModelResult{
        {}, ExecutionFailedEvidence{OutcomeReason::ToolFailure}};
  return llvm::createStringError(code ? code : llvm::inconvertibleErrorCode(),
                                 "%s", message.c_str());
}

llvm::Expected<const sim::NativeStructuredProgramObservations *>
sourceObservationsFor(
    const ArtifactRootReference &source,
    const ArtifactRootReference &workloadReference,
    const ArtifactRootReference &runtimeInputReference,
    const sim::ImportedStructuredProgramSimulationInputs &inputs) {
  std::optional<SourceObservationCacheEntry> &cache = sourceObservationCache();
  if (cache && cache->source == source &&
      cache->workload == workloadReference &&
      cache->runtimeInput == runtimeInputReference)
    return &cache->observations;

  auto observations = sim::executeNativeStructuredProgram(
      inputs.structuredProgram, inputs.workload, inputs.runtimeInput);
  if (!observations)
    return observations.takeError();
  cache.emplace(SourceObservationCacheEntry{source, workloadReference,
                                            runtimeInputReference,
                                            std::move(*observations)});
  return &cache->observations;
}

llvm::Expected<EvaluationModelResult>
evaluate(const EvaluationRequest &request, const CaseArtifactResolution &,
         const ArtifactStore &artifactStore) {
  llvm::ArrayRef<ArtifactRootReference> candidates =
      request.subjectBindings().subjects(kCandidateRole);
  if (candidates.size() != 1 || !request.workload() || !request.runtimeInput())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: exact case inputs are not "
        "total");
  auto candidate =
      frontend::importStructuredProgram(candidates.front(), artifactStore);
  if (!candidate)
    return candidate.takeError();
  auto inputs = sim::importStructuredProgramSimulationInputs(
      *request.workload(), *request.runtimeInput(), artifactStore);
  if (!inputs)
    return inputs.takeError();
  const ArtifactRootReference source{
      frontend::structuredProgramArtifactSchema.identity.str(),
      frontend::structuredProgramArtifactSchema.version,
      inputs->structuredProgram.identity()};
  bool mismatch = false;
  if (candidate->identity() != inputs->structuredProgram.identity()) {
    auto sourceObservations = sourceObservationsFor(
        source, *request.workload(), *request.runtimeInput(), *inputs);
    if (!sourceObservations)
      return classifyNativeFailure(sourceObservations.takeError());
    auto selectedObservations = sim::executeSelectedStructuredProgram(
        *candidate, inputs->structuredProgram, inputs->workload,
        inputs->runtimeInput);
    if (!selectedObservations)
      return classifyNativeFailure(selectedObservations.takeError());
    mismatch = !sim::haveEquivalentFunctionalObservations(
        **sourceObservations, *selectedObservations);
  }

  std::optional<CachedReplayResult> replay;
  if (!mismatch &&
      candidate->identity() != inputs->structuredProgram.identity()) {
    const std::vector<std::uint8_t> key = replayCacheKey(
        candidates.front(), *request.workload(), *request.runtimeInput());
    std::lock_guard<std::mutex> lock(replayCacheMutex());
    auto found = replayCache().find(key);
    if (found == replayCache().end())
      return EvaluationModelResult{
          {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
    replay = found->second;
    if (replay->kind == ReplayResultKind::Unsupported)
      return EvaluationModelResult{
          {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
    if (replay->kind == ReplayResultKind::ExecutionFailed)
      return EvaluationModelResult{
          {}, ExecutionFailedEvidence{OutcomeReason::ToolFailure}};
    mismatch = replay->kind == ReplayResultKind::Mismatch;
  }

  std::vector<FindingResult> findings;
  findings.reserve(request.findingRequests().size());
  for (const FindingRequest &finding : request.findingRequests()) {
    if (finding.query().kind != standard_findings::FunctionalMismatch)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_functional_model_invalid: unsupported finding");
    if (mismatch) {
      findings.push_back(FindingResult{PresentFinding{{FindingOccurrence::get(
          standard_findings::FunctionalMismatchOccurrence{})}}});
    } else if (replay && replay->kind == ReplayResultKind::Inapplicable) {
      findings.push_back(FindingResult{
          NotApplicableFinding{NotApplicableReason::UndefinedForSubject}});
    } else {
      findings.push_back(FindingResult{AbsentFinding{}});
    }
  }
  return EvaluationModelResult{{}, CompletedEvidence{{}, std::move(findings)}};
}

const EvaluationModelProvider kProvider{kModelDescriptor.reference(),
                                        &evaluate};

llvm::Expected<CaseArtifactResolution>
resolveCase(const ArtifactRootReference &candidate,
            const ArtifactRootReference &source,
            const ArtifactRootReference &workload,
            const ArtifactRootReference &runtimeInput,
            const ArtifactStore &artifactStore) {
  std::map<ArtifactRootReference, std::vector<ArtifactRootReference>,
           decltype(&artifactRootReferenceLess)>
      entries(&artifactRootReferenceLess);
  entries[candidate];
  entries[source];
  entries[workload] = {source};
  entries[runtimeInput] = {source, workload};
  std::vector<CaseArtifactResolution::Entry> resolved;
  resolved.reserve(entries.size());
  for (auto &[reference, closure] : entries) {
    if (auto bytes = artifactStore.get(reference); !bytes)
      return bytes.takeError();
    llvm::sort(closure, artifactRootReferenceLess);
    closure.erase(std::unique(closure.begin(), closure.end()), closure.end());
    resolved.push_back({reference, std::move(closure)});
  }
  return CaseArtifactResolution::get(std::move(resolved));
}

} // namespace

llvm::Error registerStructuredProgramFunctionalModel() {
  if (llvm::Error error = standard_findings::registerStandardFindings())
    return error;
  if (llvm::Error error = registerEvaluationCaseSignature(kCaseSignature))
    return error;
  if (llvm::Error error = registerEvaluationModelDescriptor(kModelDescriptor))
    return error;
  return registerEvaluationModelProvider(kProvider);
}

llvm::Error primeStructuredProgramFunctionalReplay(
    const ArtifactRootReference &candidateReference,
    const StructuredProgramFunctionalReplayInvocation &invocation,
    const ArtifactStore &artifactStore) {
  if (llvm::Error error = registerStructuredProgramFunctionalModel())
    return error;
  if (candidateReference.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      candidateReference.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      candidateReference.artifact !=
          invocation.candidate.structuredProgram.identity())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: replay candidate mismatch");
  if (auto stored = artifactStore.get(candidateReference); !stored)
    return stored.takeError();
  auto inputs = sim::importStructuredProgramSimulationInputs(
      invocation.workload, invocation.runtimeInput, artifactStore);
  if (!inputs)
    return inputs.takeError();
  if (inputs->structuredProgram.identity() !=
          invocation.sourceProgram.identity() ||
      inputs->workload.identity() != invocation.simulationWorkload.identity() ||
      inputs->runtimeInput.identity() !=
          invocation.simulationRuntimeInput.identity() ||
      invocation.workload.artifact !=
          invocation.simulationWorkload.identity() ||
      invocation.runtimeInput.artifact !=
          invocation.simulationRuntimeInput.identity())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: replay invocation mismatch");

  const ArtifactRootReference sourceReference{
      frontend::structuredProgramArtifactSchema.identity.str(),
      frontend::structuredProgramArtifactSchema.version,
      invocation.sourceProgram.identity()};
  if (llvm::Error error = primeSourceObservationCache(
          sourceReference, invocation.workload, invocation.runtimeInput,
          invocation.sourceObservations))
    return error;

  auto classified = classifyReplayResult(sim::validateSourceBackedDfgReplay(
      invocation.sourceProgram, invocation.scope, invocation.decision,
      invocation.candidate, invocation.simulationWorkload,
      invocation.simulationRuntimeInput));
  if (!classified)
    return classified.takeError();
  const std::vector<std::uint8_t> key = replayCacheKey(
      candidateReference, invocation.workload, invocation.runtimeInput);
  std::lock_guard<std::mutex> lock(replayCacheMutex());
  auto [found, inserted] = replayCache().try_emplace(key, *classified);
  if (!inserted && !(found->second == *classified))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: nondeterministic replay");
  return llvm::Error::success();
}

llvm::Expected<PreparedStructuredProgramFunctionalEvaluation>
prepareStructuredProgramFunctionalEvaluation(
    const ArtifactRootReference &candidate,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &artifactStore) {
  if (llvm::Error error = registerStructuredProgramFunctionalModel())
    return std::move(error);
  auto inputs = sim::importStructuredProgramSimulationInputs(
      workload, runtimeInput, artifactStore);
  if (!inputs)
    return inputs.takeError();
  const ArtifactRootReference source{
      frontend::structuredProgramArtifactSchema.identity.str(),
      frontend::structuredProgramArtifactSchema.version,
      inputs->structuredProgram.identity()};
  auto resolution =
      resolveCase(candidate, source, workload, runtimeInput, artifactStore);
  if (!resolution)
    return resolution.takeError();
  auto bindings =
      EvaluationSubjectBindings::get({{kCandidateRole, {candidate}}});
  if (!bindings)
    return bindings.takeError();
  auto evaluationCase =
      EvaluationCase::get(caseSignatureRef(), std::move(*bindings), workload,
                          runtimeInput, {}, *resolution, artifactStore);
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
                                        *resolution, artifactStore);
  if (!request)
    return request.takeError();
  auto published = publishEvaluationRequest(*request, artifactStore);
  if (!published)
    return published.takeError();
  return PreparedStructuredProgramFunctionalEvaluation{
      std::move(*request), std::move(*resolution), kCandidateRole,
      FindingRequestOrdinal(0)};
}

} // namespace loom::evaluation::models
