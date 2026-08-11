#include "Evaluation/Request.h"
#include "Evaluation/ModelParameterBundle.h"

#include "CanonicalSupport.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"

#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::evaluation {
namespace {

using detail::appendFramedBytes;
using detail::appendFramedString;
using detail::appendSchemaVersion;
using detail::appendU32Be;
using detail::appendU64Be;
using detail::evaluationError;

constexpr llvm::StringLiteral baseCaseKeyDomain =
    "loom.evaluation.base_case_key.v1";
constexpr llvm::StringLiteral metricCaseKeyDomain =
    "loom.evaluation.metric_case_key.v1";

constexpr std::uint64_t absentReference = 0;
constexpr std::uint64_t presentReference = 1;

void appendOptionalReference(
    std::vector<std::uint8_t> &preimage,
    const std::optional<ArtifactRootReference> &reference) {
  if (!reference) {
    appendU64Be(preimage, absentReference);
    return;
  }
  appendU64Be(preimage, presentReference);
  appendFramedBytes(preimage, encodeArtifactRootReference(*reference));
}

void appendConditions(std::vector<std::uint8_t> &preimage,
                      llvm::ArrayRef<EvaluationCondition> conditions) {
  appendU64Be(preimage, conditions.size());
  for (const EvaluationCondition &condition : conditions)
    appendFramedBytes(preimage, conditionPayloadKey(condition));
}

llvm::Expected<std::vector<MetricRequest>>
canonicalizeMetricRequests(llvm::ArrayRef<MetricRequest> requests) {
  std::vector<MetricRequest> canonical(requests.begin(), requests.end());
  std::sort(canonical.begin(), canonical.end(),
            [](const MetricRequest &lhs, const MetricRequest &rhs) {
              return canonicalMetricRequestKey(lhs.query(), lhs.conditions()) <
                     canonicalMetricRequestKey(rhs.query(), rhs.conditions());
            });
  for (std::size_t index = 1; index < canonical.size(); ++index)
    if (canonical[index - 1] == canonical[index])
      return evaluationError("duplicate metric request");
  return canonical;
}

llvm::Expected<std::vector<FindingRequest>>
canonicalizeFindingRequests(llvm::ArrayRef<FindingRequest> requests) {
  std::vector<FindingRequest> canonical(requests.begin(), requests.end());
  std::sort(canonical.begin(), canonical.end(),
            [](const FindingRequest &lhs, const FindingRequest &rhs) {
              return canonicalFindingRequestKey(lhs.query(), lhs.conditions()) <
                     canonicalFindingRequestKey(rhs.query(), rhs.conditions());
            });
  for (std::size_t index = 1; index < canonical.size(); ++index)
    if (canonical[index - 1] == canonical[index])
      return evaluationError("duplicate finding request");
  return canonical;
}

bool containsFindingQuery(llvm::ArrayRef<FindingRequest> requests,
                          const FindingQuery &query) {
  return std::any_of(
      requests.begin(), requests.end(),
      [&](const FindingRequest &request) { return request.query() == query; });
}

bool hasRequiredCompletedOutput(const EvaluationModelDescriptor &descriptor) {
  return std::any_of(
      descriptor.outputSlots.begin(), descriptor.outputSlots.end(),
      [](const ModelOutputSlotDescriptor &slot) {
        const ArtifactCollectionCardinality cardinality =
            slot.cardinality(EvidenceOutcomeKind::Completed);
        return cardinality == ArtifactCollectionCardinality::ExactlyOne ||
               cardinality == ArtifactCollectionCardinality::OneOrMore;
      });
}

void appendSubjectBindings(std::vector<std::uint8_t> &bytes,
                           const EvaluationSubjectBindings &bindings) {
  appendU64Be(bytes, bindings.roleBindings().size());
  for (const CaseRoleBinding &binding : bindings.roleBindings()) {
    appendU32Be(bytes, binding.role.ordinal());
    appendU64Be(bytes, binding.subjects.size());
    for (const ArtifactRootReference &subject : binding.subjects)
      appendFramedBytes(bytes, encodeArtifactRootReference(subject));
  }
}

void appendTargetDependencies(std::vector<ArtifactRootReference> &dependencies,
                              const SubjectTargetRef &target) {
  dependencies.push_back(target.anchorSubjectArtifact);
  dependencies.push_back(target.targetArtifact());
}

void appendConditionDependencies(
    std::vector<ArtifactRootReference> &dependencies,
    const EvaluationCondition &condition) {
  for (const SubjectTargetRef *target : conditionOrderedTargets(condition))
    appendTargetDependencies(dependencies, *target);
  if (const auto *corner =
          std::get_if<ProcessCornerCondition>(&condition.payload))
    dependencies.push_back(
        platform::encodeTechnologyCornerRef(corner->corner).artifact);
  if (const auto *activity =
          std::get_if<ActivityBindingCondition>(&condition.payload))
    if (const auto *execution =
            std::get_if<ExecutionActivitySource>(&activity->source))
      dependencies.push_back(execution->simulationExecution);
}

std::vector<ArtifactRootReference>
evaluationRequestDirectDependencies(const EvaluationRequest &request) {
  std::vector<ArtifactRootReference> dependencies;
  for (const CaseRoleBinding &binding :
       request.subjectBindings().roleBindings())
    dependencies.insert(dependencies.end(), binding.subjects.begin(),
                        binding.subjects.end());
  if (request.workload())
    dependencies.push_back(*request.workload());
  if (request.runtimeInput())
    dependencies.push_back(*request.runtimeInput());
  for (const ModelInputBinding &input : request.modelBinding().inputBindings())
    dependencies.insert(dependencies.end(), input.artifacts.begin(),
                        input.artifacts.end());
  for (const EvaluationCondition &condition : request.baseConditions())
    appendConditionDependencies(dependencies, condition);
  for (const MetricRequest &metric : request.metricRequests()) {
    for (const SubjectTargetRef &target : metric.query().scope.targets)
      appendTargetDependencies(dependencies, target);
    for (const EvaluationCondition &condition : metric.conditions())
      appendConditionDependencies(dependencies, condition);
  }
  for (const FindingRequest &finding : request.findingRequests()) {
    for (const SubjectTargetRef &target : finding.query().scope.targets)
      appendTargetDependencies(dependencies, target);
    for (const EvaluationCondition &condition : finding.conditions())
      appendConditionDependencies(dependencies, condition);
  }
  std::sort(dependencies.begin(), dependencies.end(),
            artifactRootReferenceLess);
  dependencies.erase(std::unique(dependencies.begin(), dependencies.end()),
                     dependencies.end());
  return dependencies;
}

llvm::Error validateEvaluationRequestDirectDependencies(
    const EvaluationRequest &request, const ArtifactStore &artifactStore) {
  for (const ArtifactRootReference &dependency :
       evaluationRequestDirectDependencies(request)) {
    auto bytes = artifactStore.get(dependency);
    if (!bytes)
      return bytes.takeError();
  }
  return llvm::Error::success();
}

/// The static RelativeClockSchedule payload check only rejects two identical
/// SubjectTargetRef values. Request verification additionally rejects a
/// schedule whose reference and dependent clocks resolve to the same exact
/// SubjectTarget identity after EvaluationCase has validated binding, closure,
/// and owner codecs. Only the role and anchor provenance are ignored, so the
/// same root or the same complete canonical local target is rejected across
/// roles or anchors while two distinct local targets in one Artifact remain
/// distinct.
llvm::Error rejectAliasedClockDomains(const EvaluationRequest &request) {
  for (const EvaluationCondition &condition : request.baseConditions()) {
    if (condition.kind() != EvaluationConditionKind::RelativeClockSchedule)
      continue;
    const auto &schedule =
        std::get<RelativeClockScheduleCondition>(condition.payload);
    if (schedule.referenceClock.target == schedule.dependentClock.target)
      return evaluationError(
          "relative clock schedule binds one clock domain as "
          "both the reference and the dependent clock");
  }
  return llvm::Error::success();
}

} // namespace

std::vector<std::uint8_t>
canonicalMetricRequestKey(const MetricQuery &query,
                          llvm::ArrayRef<EvaluationCondition> conditions) {
  std::vector<std::uint8_t> key;
  appendU32Be(key, static_cast<std::uint32_t>(query.metric));
  appendFramedBytes(key, canonicalScopeKey(query.scope));
  appendConditions(key, conditions);
  return key;
}

std::vector<std::uint8_t>
canonicalFindingRequestKey(const FindingQuery &query,
                           llvm::ArrayRef<EvaluationCondition> conditions) {
  std::vector<std::uint8_t> key = canonicalFindingQueryKey(query);
  appendConditions(key, conditions);
  return key;
}

const ArtifactSchemaDescriptor EvaluationRequest::artifactSchema{
    "evaluation.request", {1, 0}};

llvm::Expected<MetricRequest>
MetricRequest::get(MetricQuery query,
                   llvm::ArrayRef<EvaluationCondition> conditions,
                   const EvaluationCase &evaluationCase,
                   const CaseArtifactResolution &resolution,
                   const ArtifactStore &artifactStore) {
  if (evaluationCase.signature().schemaVersion() == SchemaVersion{2, 0} &&
      static_cast<std::uint32_t>(query.metric) >
          static_cast<std::uint32_t>(MetricKind::MaximumVoltageDrop))
    return evaluationError(
        "Evaluation registry 2.0 does not contain the requested metric");
  if (llvm::Error error = validateMetricQuery(query))
    return std::move(error);

  const CaseTargetContext context =
      evaluationCase.targetContext(resolution, artifactStore);
  if (llvm::Error error = validateEvaluationScopeCase(
          query.scope, metricDescriptor(query.metric).scopeForms, context))
    return std::move(error);
  // Descriptor-context static admission only; RequestVerifier owns the exact
  // request-context reference-cycle basis and resolves it once per request.
  if (llvm::Error error = validateMetricScopeAdmissibility(
          query.metric, query.scope.form,
          *evaluationCase.signature().descriptor()))
    return std::move(error);

  const MetricDescriptor &descriptor = metricDescriptor(query.metric);
  llvm::Expected<std::vector<EvaluationCondition>> canonicalConditions =
      canonicalizeEvaluationConditions(
          conditions, ConditionLocation::MetricRequest, descriptor.spelling,
          descriptor.permittedRequestConditionPatterns, context);
  if (!canonicalConditions)
    return canonicalConditions.takeError();

  std::vector<ConditionApplicabilityPattern> presentPatterns;
  presentPatterns.reserve(canonicalConditions->size());
  for (const EvaluationCondition &condition : *canonicalConditions)
    presentPatterns.push_back(deriveConditionApplicabilityPattern(
        condition, evaluationCase.signature()));
  for (const ConditionApplicabilityPattern &required :
       descriptor.requiredRequestConditionPatterns)
    if (std::find(presentPatterns.begin(), presentPatterns.end(), required) ==
        presentPatterns.end())
      return evaluationError("metric '" + descriptor.spelling +
                             "' requires condition '" +
                             toString(required.kind) + "'");

  return MetricRequest(std::move(query), std::move(*canonicalConditions));
}

llvm::Expected<FindingRequest>
FindingRequest::get(FindingQuery query,
                    llvm::ArrayRef<EvaluationCondition> conditions,
                    const EvaluationCase &evaluationCase,
                    const CaseArtifactResolution &resolution,
                    const ArtifactStore &artifactStore) {
  if (llvm::Error error = validateFindingQuery(query))
    return std::move(error);
  const FindingDescriptor *descriptor = findFindingDescriptor(query.kind);
  const CaseTargetContext context =
      evaluationCase.targetContext(resolution, artifactStore);
  if (llvm::Error error = validateEvaluationScopeCase(
          query.scope, descriptor->scopeForms, context))
    return std::move(error);

  llvm::Expected<std::vector<EvaluationCondition>> canonicalConditions =
      canonicalizeEvaluationConditions(
          conditions, ConditionLocation::FindingRequest, descriptor->spelling,
          descriptor->permittedRequestConditionPatterns, context);
  if (!canonicalConditions)
    return canonicalConditions.takeError();
  return FindingRequest(std::move(query), std::move(*canonicalConditions));
}

llvm::Expected<EvaluationRequest> EvaluationRequest::get(
    const EvaluationCase &evaluationCase,
    llvm::ArrayRef<MetricRequest> metricRequests,
    llvm::ArrayRef<FindingRequest> findingRequests,
    ResolvedModelBinding modelBinding, std::uint64_t replicateIndex,
    const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  const EvaluationModelDescriptor *descriptor =
      modelBinding.descriptorRef().descriptor();
  if (!descriptor)
    return evaluationError("EvaluationRequest references an unregistered model "
                           "descriptor");
  if (evaluationCase.signature() != descriptor->caseSignature)
    return evaluationError(
        "EvaluationCase signature does not match the model descriptor");
  return get(evaluationCase.subjectBindings(), evaluationCase.workload(),
             evaluationCase.runtimeInput(), evaluationCase.baseConditions(),
             metricRequests, findingRequests, std::move(modelBinding),
             replicateIndex, resolution, artifactStore, blobStore);
}

llvm::Expected<EvaluationRequest> EvaluationRequest::get(
    EvaluationSubjectBindings subjectBindings,
    std::optional<ArtifactRootReference> workload,
    std::optional<ArtifactRootReference> runtimeInput,
    llvm::ArrayRef<EvaluationCondition> baseConditions,
    llvm::ArrayRef<MetricRequest> metricRequests,
    llvm::ArrayRef<FindingRequest> findingRequests,
    ResolvedModelBinding modelBinding, std::uint64_t replicateIndex,
    const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  auto canonicalMetrics = canonicalizeMetricRequests(metricRequests);
  if (!canonicalMetrics)
    return canonicalMetrics.takeError();
  auto canonicalFindings = canonicalizeFindingRequests(findingRequests);
  if (!canonicalFindings)
    return canonicalFindings.takeError();
  const EvaluationModelDescriptor *descriptor =
      modelBinding.descriptorRef().descriptor();
  if (!descriptor)
    return evaluationError("EvaluationRequest references an unregistered model "
                           "descriptor");
  if (canonicalMetrics->empty() && canonicalFindings->empty() &&
      !hasRequiredCompletedOutput(*descriptor))
    return evaluationError(
        "an EvaluationRequest requires a metric, finding, or required "
        "completed output");
  auto evaluationCase =
      EvaluationCase::get(descriptor->caseSignature, std::move(subjectBindings),
                          std::move(workload), std::move(runtimeInput),
                          baseConditions, resolution, artifactStore, blobStore);
  if (!evaluationCase)
    return evaluationCase.takeError();

  EvaluationRequest request(
      evaluationCase->subjectBindings(), evaluationCase->workload(),
      evaluationCase->runtimeInput(),
      std::vector<EvaluationCondition>(evaluationCase->baseConditions().begin(),
                                       evaluationCase->baseConditions().end()),
      std::move(*canonicalMetrics), std::move(*canonicalFindings),
      std::move(modelBinding), replicateIndex);
  RequestVerifier verifier(resolution, artifactStore, blobStore);
  if (llvm::Error error = verifier.verify(request))
    return std::move(error);
  return request;
}

const MetricRequest *
EvaluationRequest::resolve(MetricRequestOrdinal ordinal) const {
  if (ordinal.ordinal() >= metricRequests_.size())
    return nullptr;
  return &metricRequests_[ordinal.ordinal()];
}

const FindingRequest *
EvaluationRequest::resolve(FindingRequestOrdinal ordinal) const {
  if (ordinal.ordinal() >= findingRequests_.size())
    return nullptr;
  return &findingRequests_[ordinal.ordinal()];
}

const EvaluationModelDescriptor *
resolveEvaluationModelDescriptor(const EvaluationRequest &request) {
  return request.modelBinding().descriptorRef().descriptor();
}

llvm::Error RequestVerifier::verify(const EvaluationRequest &request) const {
  const EvaluationModelDescriptor *descriptor =
      resolveEvaluationModelDescriptor(request);
  if (!descriptor)
    return evaluationError("EvaluationRequest model descriptor is unresolved");
  if (request.metricRequests().empty() && request.findingRequests().empty() &&
      !hasRequiredCompletedOutput(*descriptor))
    return evaluationError("EvaluationRequest has no requested result");
  if (llvm::Error error = validateResolvedModelBinding(request.modelBinding()))
    return error;
  if (descriptor->determinism == DeterminismContract::Deterministic &&
      request.replicateIndex() != 0)
    return evaluationError("deterministic model requires replicate_index zero");

  auto evaluationCase = EvaluationCase::get(
      descriptor->caseSignature, request.subjectBindings(), request.workload(),
      request.runtimeInput(), request.baseConditions(), resolution_,
      artifactStore_, blobStore_);
  if (!evaluationCase)
    return evaluationCase.takeError();

  for (const ModelInputSlotDescriptor &slot : descriptor->inputSlots) {
    const ModelInputBinding *input =
        request.modelBinding().findInputBinding(slot.slot);
    if (slot.modelParameterContract) {
      for (const ArtifactRootReference &artifact : input->artifacts) {
        auto bundle =
            importModelParameterBundle(artifact, artifactStore_, blobStore_);
        if (!bundle)
          return bundle.takeError();
        if (bundle->bundle().parameterContract() !=
            *slot.modelParameterContract)
          return evaluationError(
              "model input bundle has the wrong parameter contract");
      }
    }
    if (slot.verifyCompatibility)
      if (llvm::Error error =
              slot.verifyCompatibility(input->artifacts, *evaluationCase,
                                       resolution_, artifactStore_, blobStore_))
        return error;
  }
  if (llvm::Error error = rejectAliasedClockDomains(request))
    return error;

  for (const MetricRequest &requestItem : request.metricRequests()) {
    auto reverified =
        MetricRequest::get(requestItem.query(), requestItem.conditions(),
                           *evaluationCase, resolution_, artifactStore_);
    if (!reverified)
      return reverified.takeError();
    if (!(*reverified == requestItem))
      return evaluationError("metric request is not canonical");
    // RequestVerifier owns exact request-context admission: resolve the
    // reference-cycle basis exactly once per metric request and discard the
    // validated result without storing or re-resolving it.
    auto cycleBasis = validateMetricScopeAdmissibility(
        requestItem.query().metric, requestItem.query().scope.form,
        *evaluationCase, resolution_, artifactStore_, blobStore_);
    if (!cycleBasis)
      return cycleBasis.takeError();
    (void)*cycleBasis;
  }
  for (const FindingRequest &requestItem : request.findingRequests()) {
    auto reverified =
        FindingRequest::get(requestItem.query(), requestItem.conditions(),
                            *evaluationCase, resolution_, artifactStore_);
    if (!reverified)
      return reverified.takeError();
    if (!(*reverified == requestItem))
      return evaluationError("finding request is not canonical");
  }

  if (llvm::Error error = validateModelCapability(*descriptor, *evaluationCase,
                                                  request.metricRequests(),
                                                  request.findingRequests()))
    return error;
  for (const FindingQuery &mandatory : descriptor->mandatoryTerminalFindings)
    if (!containsFindingQuery(request.findingRequests(), mandatory))
      return evaluationError("EvaluationRequest omits a mandatory terminal "
                             "finding");
  return llvm::Error::success();
}

CanonicalSemanticBytes
canonicalEvaluationRequestBytes(const EvaluationRequest &request) {
  const std::string json = serializeEvaluationRequest(request);
  return CanonicalSemanticBytes(
      std::vector<std::uint8_t>(json.begin(), json.end()));
}

ArtifactIdentity evaluationRequestIdentity(const EvaluationRequest &request) {
  return finalizeArtifactIdentity(EvaluationRequest::artifactSchema,
                                  canonicalEvaluationRequestBytes(request));
}

ArtifactRootReference
evaluationRequestReference(const EvaluationRequest &request) {
  return ArtifactRootReference{EvaluationRequest::artifactSchema.identity.str(),
                               EvaluationRequest::artifactSchema.version,
                               evaluationRequestIdentity(request)};
}

llvm::Expected<ArtifactRootReference>
publishEvaluationRequest(const EvaluationRequest &request,
                         const ArtifactStore &artifactStore) {
  if (llvm::Error error =
          validateEvaluationRequestDirectDependencies(request, artifactStore))
    return std::move(error);
  auto identity = artifactStore.put(EvaluationRequest::artifactSchema,
                                    canonicalEvaluationRequestBytes(request));
  if (!identity)
    return identity.takeError();
  if (*identity != evaluationRequestIdentity(request))
    return evaluationError("ArtifactStore returned a foreign EvaluationRequest "
                           "identity");
  return ArtifactRootReference{EvaluationRequest::artifactSchema.identity.str(),
                               EvaluationRequest::artifactSchema.version,
                               std::move(*identity)};
}

llvm::Expected<EvaluationRequest>
importEvaluationRequest(const ArtifactRootReference &reference,
                        const CaseArtifactResolution &resolution,
                        const ArtifactStore &artifactStore,
                        const BlobStore &blobStore) {
  if (reference.schemaIdentity != EvaluationRequest::artifactSchema.identity ||
      reference.schemaVersion != EvaluationRequest::artifactSchema.version)
    return evaluationError("foreign EvaluationRequest reference schema");
  auto bytes =
      artifactStore.get(EvaluationRequest::artifactSchema, reference.artifact);
  if (!bytes)
    return bytes.takeError();
  const llvm::ArrayRef<std::uint8_t> payload = bytes->bytes();
  llvm::StringRef json(reinterpret_cast<const char *>(payload.data()),
                       payload.size());
  auto request =
      parseEvaluationRequest(json, resolution, artifactStore, blobStore);
  if (!request)
    return request.takeError();
  if (llvm::Error error =
          validateEvaluationRequestDirectDependencies(*request, artifactStore))
    return std::move(error);
  if (evaluationRequestIdentity(*request) != reference.artifact)
    return evaluationError("stale EvaluationRequest reference identity");
  return request;
}

EvaluationCaseKey baseCaseKey(const EvaluationCase &evaluationCase) {
  std::vector<std::uint8_t> preimage;
  appendFramedString(preimage, baseCaseKeyDomain);
  appendSchemaVersion(preimage, evaluationCase.signature().schemaVersion());
  appendU32Be(preimage, evaluationCase.signature().caseKind().ordinal());
  appendSubjectBindings(preimage, evaluationCase.subjectBindings());
  appendOptionalReference(preimage, evaluationCase.workload());
  appendOptionalReference(preimage, evaluationCase.runtimeInput());
  appendConditions(preimage, evaluationCase.baseConditions());
  return EvaluationCaseKey(llvm::SHA256::hash(preimage));
}

EvaluationCaseKey metricCaseKey(const EvaluationCase &evaluationCase,
                                const MetricRequest &request) {
  const EvaluationCaseKey base = baseCaseKey(evaluationCase);

  std::vector<std::uint8_t> preimage;
  appendFramedString(preimage, metricCaseKeyDomain);
  preimage.insert(preimage.end(), base.bytes().begin(), base.bytes().end());
  appendFramedBytes(preimage, canonicalMetricRequestKey(request.query(),
                                                        request.conditions()));
  return EvaluationCaseKey(llvm::SHA256::hash(preimage));
}

} // namespace loom::evaluation
