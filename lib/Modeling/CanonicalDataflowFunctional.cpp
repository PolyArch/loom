#include "Evaluation/Models/CanonicalDataflowFunctional.h"
#include "Evaluation/ProductionRegistry.h"
#include "StructuredEvaluationInvocationCacheInternal.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/StructuredProgramFunctional.h"
#include "Evaluation/StandardFindings.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr BuiltinEvaluationCase kCase =
    BuiltinEvaluationCase::CanonicalDataflowSourceFunctionalComparison;
constexpr BuiltinEvaluationModel kModel =
    BuiltinEvaluationModel::CanonicalDataflowSourceFunctional;
constexpr CaseSubjectRoleRef kCandidateRole(0);
constexpr CaseSubjectRoleRef kStructuredParentRole(1);
constexpr ScopeFormRef kWholeExactCaseScope(0);

EvaluationCaseSignatureRef caseSignatureRef() {
  return llvm::cantFail(EvaluationCaseSignatureRef::get(
      evaluationSchemaVersion(), builtinEvaluationCaseKind(kCase)));
}

const ArtifactSchemaDescriptor *const kDataflowSchemas[] = {
    &dataflow::canonicalDataflowSchema};
const ArtifactSchemaDescriptor *const kStructuredSchemas[] = {
    &frontend::structuredProgramArtifactSchema};
const ArtifactSchemaDescriptor *const kWorkloadSchemas[] = {
    &sim::simulationWorkloadSchema};
const ArtifactSchemaDescriptor *const kRuntimeInputSchemas[] = {
    &sim::simulationRuntimeInputSchema};

const CaseSubjectRoleDescriptor kSubjectRoles[] = {
    {kCandidateRole, "canonical_dataflow", SubjectRoleCardinality::ExactlyOne,
     kDataflowSchemas, nullptr},
    {kStructuredParentRole, "selected_structured_parent",
     SubjectRoleCardinality::ExactlyOne, kStructuredSchemas, nullptr},
};

llvm::Error verifyWorkloadCompatibility(
    const EvaluationCase &, const EvaluationSubjectBindings &,
    const std::optional<ArtifactRootReference> &workload,
    const std::optional<ArtifactRootReference> &runtimeInput,
    const CaseArtifactResolution &resolution, const ArtifactStore &,
    const BlobStore &) {
  if (!workload || !runtimeInput)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical_dataflow_functional_model_invalid: workload inputs are "
        "not total");
  const CaseArtifactResolution::Entry *workloadEntry =
      resolution.find(*workload);
  const CaseArtifactResolution::Entry *runtimeEntry =
      resolution.find(*runtimeInput);
  if (!workloadEntry || !runtimeEntry ||
      !CaseArtifactResolution::reaches(*runtimeEntry, *workload))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical_dataflow_functional_model_invalid: runtime input does not "
        "reach its exact workload");
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
        "canonical_dataflow_functional_model_invalid: workload has no exact "
        "source program");
  return llvm::Error::success();
}

const EvaluationCaseSignatureDescriptor kCaseSignature{
    builtinEvaluationCaseKind(kCase),
    "canonical_dataflow_source_functional_comparison",
    "One exact Canonical Dataflow candidate and its selected Structured "
    "parent compared with the workload-owned source.",
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
    ModeledPhenomenon::CanonicalDataflow};

struct EmptyFunctionalConfig final {};

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.canonical_dataflow_functional.config.1.0";
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
        "canonical Dataflow functional config has the wrong owner type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue> adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                       const ComponentViewDigest &) {
  if (!bytes.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical Dataflow functional config must be empty");
  return OwnerValue::get(EmptyFunctionalConfig{});
}

const ResolvedModelConfigViewContract kConfigView{
    configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig};

const EvaluationModelDescriptor kModelDescriptor{
    builtinEvaluationModelKind(kModel),
    "canonical_dataflow_source_functional",
    "loom.canonical_dataflow.source_functional.v1",
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

using ReplayResultKind = detail::StructuredReplayResultKind;
using CachedReplayResult = detail::StructuredCachedReplayResult;

detail::CanonicalDataflowFunctionalCacheKey
replayCacheKey(const ArtifactRootReference &candidate,
               const ArtifactRootReference &structuredParent,
               const ArtifactRootReference &workload,
               const ArtifactRootReference &runtimeInput) {
  return {candidate, structuredParent, workload, runtimeInput};
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
    return CachedReplayResult{kind, std::move(*replay)};
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
    return CachedReplayResult{ReplayResultKind::Unsupported, std::nullopt};
  return llvm::createStringError(code ? code : llvm::inconvertibleErrorCode(),
                                 "%s", message.c_str());
}

llvm::Expected<std::shared_ptr<const CachedReplayResult>>
cachedReplay(const ArtifactRootReference &candidate,
             const ArtifactRootReference &structuredParent,
             const ArtifactRootReference &workload,
             const ArtifactRootReference &runtimeInput) {
  StructuredEvaluationInvocationCache *cache =
      detail::currentStructuredEvaluationCache();
  if (!cache)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical_dataflow_functional_model_invalid: replay lookup requires "
        "an active invocation cache");
  auto &impl = detail::StructuredEvaluationCacheAccess::impl(*cache);
  const auto key =
      replayCacheKey(candidate, structuredParent, workload, runtimeInput);
  std::lock_guard<std::mutex> lock(impl.mutex);
  auto found = impl.dataflowFunctionalResults.find(key);
  if (found == impl.dataflowFunctionalResults.end()) {
    impl.functionalMissCount.fetch_add(1, std::memory_order_relaxed);
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical_dataflow_functional_model_invalid: replay was not primed");
  }
  impl.functionalHitCount.fetch_add(1, std::memory_order_relaxed);
  return found->second;
}

llvm::Error storeReplay(const ArtifactRootReference &candidate,
                        const ArtifactRootReference &structuredParent,
                        const ArtifactRootReference &workload,
                        const ArtifactRootReference &runtimeInput,
                        CachedReplayResult replay) {
  StructuredEvaluationInvocationCache *cache =
      detail::currentStructuredEvaluationCache();
  if (!cache)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical_dataflow_functional_model_invalid: replay priming requires "
        "an active invocation cache");
  const auto key =
      replayCacheKey(candidate, structuredParent, workload, runtimeInput);
  auto value = std::make_shared<const CachedReplayResult>(std::move(replay));
  auto &impl = detail::StructuredEvaluationCacheAccess::impl(*cache);
  std::lock_guard<std::mutex> lock(impl.mutex);
  auto [found, inserted] =
      impl.dataflowFunctionalResults.try_emplace(key, value);
  if (!inserted && !(*found->second == *value))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical_dataflow_functional_model_invalid: nondeterministic "
        "replay");
  if (inserted)
    impl.functionalPrimeCount.fetch_add(1, std::memory_order_relaxed);
  return llvm::Error::success();
}

llvm::Expected<EvaluationModelResult>
evaluate(const EvaluationRequest &request, const CaseArtifactResolution &,
         const ArtifactStore &artifactStore, const BlobStore &) {
  llvm::ArrayRef<ArtifactRootReference> candidates =
      request.subjectBindings().subjects(kCandidateRole);
  llvm::ArrayRef<ArtifactRootReference> parents =
      request.subjectBindings().subjects(kStructuredParentRole);
  if (candidates.size() != 1 || parents.size() != 1 || !request.workload() ||
      !request.runtimeInput())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical_dataflow_functional_model_invalid: exact case inputs are "
        "not total");
  if (auto candidate =
          dataflow::importCanonicalDataflow(candidates.front(), artifactStore);
      !candidate)
    return candidate.takeError();
  if (auto parent =
          frontend::importStructuredProgram(parents.front(), artifactStore);
      !parent)
    return parent.takeError();

  auto replay = cachedReplay(candidates.front(), parents.front(),
                             *request.workload(), *request.runtimeInput());
  if (!replay) {
    llvm::consumeError(replay.takeError());
    return EvaluationModelResult{
        {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  }
  if ((*replay)->kind == ReplayResultKind::Unsupported)
    return EvaluationModelResult{
        {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  std::vector<FindingResult> findings;
  findings.reserve(request.findingRequests().size());
  for (const FindingRequest &finding : request.findingRequests()) {
    if (finding.query().kind != standard_findings::FunctionalMismatch)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "canonical_dataflow_functional_model_invalid: unsupported finding");
    switch ((*replay)->kind) {
    case ReplayResultKind::Equivalent:
      findings.push_back(FindingResult{AbsentFinding{}});
      break;
    case ReplayResultKind::Mismatch:
      findings.push_back(FindingResult{PresentFinding{{FindingOccurrence::get(
          standard_findings::FunctionalMismatchOccurrence{})}}});
      break;
    case ReplayResultKind::Inapplicable:
      findings.push_back(FindingResult{
          NotApplicableFinding{NotApplicableReason::UndefinedForSubject}});
      break;
    case ReplayResultKind::Unsupported:
      llvm_unreachable("unsupported replay returned above");
    }
  }
  return EvaluationModelResult{{}, CompletedEvidence{{}, std::move(findings)}};
}

const EvaluationModelProvider kProvider{
    kModelDescriptor.reference(), EvaluationModelInProcessProvider{&evaluate}};

} // namespace

llvm::Error registerCanonicalDataflowFunctionalModel() {
  if (llvm::Error error = standard_findings::registerStandardFindings())
    return error;
  if (llvm::Error error = registerEvaluationCaseSignature(kCaseSignature))
    return error;
  if (llvm::Error error = registerEvaluationModelDescriptor(kModelDescriptor))
    return error;
  return registerEvaluationModelProvider(kProvider);
}

EvaluationModelDescriptorRef canonicalDataflowFunctionalModelDescriptorRef() {
  return kModelDescriptor.reference();
}

CaseSubjectRoleRef canonicalDataflowFunctionalCandidateRole() {
  return kCandidateRole;
}

CaseSubjectRoleRef canonicalDataflowFunctionalStructuredParentRole() {
  return kStructuredParentRole;
}

llvm::Error primeCanonicalDataflowFunctionalReplay(
    const ArtifactRootReference &candidateReference,
    const ArtifactRootReference &structuredParent,
    const CanonicalDataflowFunctionalReplayInvocation &invocation,
    const ArtifactStore &artifactStore) {
  if (llvm::Error error = registerCanonicalDataflowFunctionalModel())
    return error;
  StructuredEvaluationInvocationCache *cache =
      detail::currentStructuredEvaluationCache();
  if (!cache)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical_dataflow_functional_model_invalid: replay priming requires "
        "an active invocation cache");
  if (candidateReference.schemaIdentity !=
          dataflow::canonicalDataflowSchema.identity ||
      candidateReference.schemaVersion !=
          dataflow::canonicalDataflowSchema.version ||
      candidateReference.artifact !=
          invocation.candidate.canonicalDataflow.identity() ||
      structuredParent.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      structuredParent.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      structuredParent.artifact !=
          invocation.candidate.structuredProgram.identity())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical_dataflow_functional_model_invalid: replay candidate or "
        "Structured parent mismatch");
  if (auto stored = artifactStore.get(candidateReference); !stored)
    return stored.takeError();
  if (auto stored = artifactStore.get(structuredParent); !stored)
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
        "canonical_dataflow_functional_model_invalid: replay invocation "
        "mismatch");

  const ArtifactRootReference sourceReference{
      frontend::structuredProgramArtifactSchema.identity.str(),
      frontend::structuredProgramArtifactSchema.version,
      invocation.sourceProgram.identity()};
  if (llvm::Error error = primeStructuredProgramSourceObservations(
          sourceReference, invocation.workload, invocation.runtimeInput,
          invocation.sourceObservations))
    return error;

  auto classified = classifyReplayResult(sim::validateSourceBackedDfgReplay(
      invocation.sourceProgram, invocation.candidate,
      invocation.simulationWorkload, invocation.simulationRuntimeInput,
      invocation.limits, &invocation.sourceObservations));
  if (!classified)
    return classified.takeError();

  return storeReplay(candidateReference, structuredParent, invocation.workload,
                     invocation.runtimeInput, std::move(*classified));
}

llvm::Error primeCanonicalDataflowFunctionalReplayResult(
    const ArtifactRootReference &candidate,
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput,
    const sim::SourceBackedDfgValidationResult &replay) {
  if (llvm::Error error = registerCanonicalDataflowFunctionalModel())
    return error;
  if (candidate.schemaIdentity != dataflow::canonicalDataflowSchema.identity ||
      candidate.schemaVersion != dataflow::canonicalDataflowSchema.version ||
      structuredParent.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      structuredParent.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      workload.schemaIdentity != sim::simulationWorkloadSchema.identity ||
      workload.schemaVersion != sim::simulationWorkloadSchema.version ||
      runtimeInput.schemaIdentity !=
          sim::simulationRuntimeInputSchema.identity ||
      runtimeInput.schemaVersion != sim::simulationRuntimeInputSchema.version)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical_dataflow_functional_model_invalid: primed replay has a "
        "foreign root");
  ReplayResultKind kind = ReplayResultKind::Unsupported;
  switch (replay.status) {
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
  return storeReplay(candidate, structuredParent, workload, runtimeInput,
                     CachedReplayResult{kind, replay});
}

llvm::Expected<sim::SourceBackedDfgValidationResult>
getPrimedCanonicalDataflowFunctionalReplay(
    const ArtifactRootReference &candidate,
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput) {
  auto replay =
      cachedReplay(candidate, structuredParent, workload, runtimeInput);
  if (!replay)
    return replay.takeError();
  if (!(*replay)->replay)
    return llvm::createStringError(
        std::make_error_code(std::errc::not_supported),
        "canonical_dataflow_functional_model_unsupported: replay provider "
        "unavailable");
  return *(*replay)->replay;
}

llvm::Expected<CaseArtifactResolution>
resolveCanonicalDataflowFunctionalEvaluationCase(
    const ArtifactRootReference &candidate,
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput,
    const ArtifactStore &artifactStore) {
  return resolveCanonicalDataflowFunctionalEvaluationCases(
      {candidate}, structuredParent, workload, runtimeInput, artifactStore);
}

llvm::Expected<CaseArtifactResolution>
resolveCanonicalDataflowFunctionalEvaluationCases(
    llvm::ArrayRef<ArtifactRootReference> candidateReferences,
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput,
    const ArtifactStore &artifactStore) {
  if (candidateReferences.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "canonical_dataflow_functional_model_invalid: invocation has no "
        "Canonical Dataflow candidates");
  std::vector<ArtifactRootReference> candidates(candidateReferences.begin(),
                                                candidateReferences.end());
  llvm::sort(candidates, artifactRootReferenceLess);
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  auto inputs = sim::importStructuredProgramSimulationInputs(
      workload, runtimeInput, artifactStore);
  if (!inputs)
    return inputs.takeError();
  const ArtifactRootReference source{
      frontend::structuredProgramArtifactSchema.identity.str(),
      frontend::structuredProgramArtifactSchema.version,
      inputs->structuredProgram.identity()};
  std::map<ArtifactRootReference, std::vector<ArtifactRootReference>,
           decltype(&artifactRootReferenceLess)>
      entries(&artifactRootReferenceLess);
  for (const ArtifactRootReference &candidate : candidates) {
    if (candidate.schemaIdentity !=
            dataflow::canonicalDataflowSchema.identity ||
        candidate.schemaVersion != dataflow::canonicalDataflowSchema.version)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "canonical_dataflow_functional_model_invalid: invocation contains "
          "a foreign candidate");
    entries[candidate];
  }
  entries[structuredParent];
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

llvm::Expected<PreparedCanonicalDataflowFunctionalEvaluation>
prepareCanonicalDataflowFunctionalEvaluation(
    const ArtifactRootReference &candidate,
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  if (llvm::Error error = registerCanonicalDataflowFunctionalModel())
    return std::move(error);
  auto resolution = resolveCanonicalDataflowFunctionalEvaluationCase(
      candidate, structuredParent, workload, runtimeInput, artifactStore);
  if (!resolution)
    return resolution.takeError();
  auto bindings = EvaluationSubjectBindings::get(
      {{kCandidateRole, {candidate}},
       {kStructuredParentRole, {structuredParent}}});
  if (!bindings)
    return bindings.takeError();
  auto evaluationCase = EvaluationCase::get(
      caseSignatureRef(), std::move(*bindings), workload, runtimeInput, {},
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
  return PreparedCanonicalDataflowFunctionalEvaluation{
      std::move(*request), std::move(*resolution), kCandidateRole,
      FindingRequestOrdinal(0)};
}

} // namespace loom::evaluation::models
