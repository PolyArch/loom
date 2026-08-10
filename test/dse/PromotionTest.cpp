#include "DSE/Promotion.h"

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/ResolvedPnrPolicy.h"
#include "Config/ResolvedConfig.h"
#include "Evaluation/Metric.h"
#include "Evaluation/ModelDescriptor.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace loom;
using namespace loom::dse;
using namespace loom::evaluation;

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "DSE promotion test: " << message << '\n';
  std::exit(1);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

ArtifactRootReference makeReference(std::uint8_t fill) {
  std::array<std::uint8_t, ArtifactIdentity::byteSize> bytes{};
  bytes.fill(fill);
  return {"loom.test.promotion_candidate", SchemaVersion{1, 0},
          take(ArtifactIdentity::fromBytes(bytes))};
}

constexpr ArtifactSchemaDescriptor evidenceCandidateSchema{
    "loom.test.promotion_evidence_candidate", SchemaVersion{1, 0}};
constexpr EvaluationCaseKind evidenceCaseKind(0x7fff3000);
constexpr EvaluationModelKind evidenceModelKind(0x7fff3000);
constexpr CaseSubjectRoleRef evidenceCandidateRole(0);

EvaluationCaseSignatureRef evidenceSignatureRef() {
  return llvm::cantFail(EvaluationCaseSignatureRef::get(
      evaluationSchemaVersion(), evidenceCaseKind));
}

const ArtifactSchemaDescriptor *const evidenceCandidateSchemas[] = {
    &evidenceCandidateSchema};
const CaseSubjectRoleDescriptor evidenceSubjectRoles[] = {
    {evidenceCandidateRole, "candidate", SubjectRoleCardinality::ExactlyOne,
     evidenceCandidateSchemas, nullptr}};
const EvaluationCaseSignatureDescriptor evidenceCaseSignature{
    evidenceCaseKind,
    "promotion_evidence_case",
    "One exact candidate ranked only through persistent Evaluation Evidence.",
    evidenceSubjectRoles,
    ArtifactRequirement::Forbidden,
    {},
    ArtifactRequirement::Forbidden,
    {},
    nullptr,
    AbsentReferenceCycle{},
    {}};

struct EmptyPromotionModelConfig final {};

llvm::ArrayRef<std::uint8_t> promotionConfigSchema() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.test.promotion.model_config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectPromotionConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptyPromotionModelConfig{});
}

llvm::Expected<std::vector<std::uint8_t>>
encodePromotionConfig(const OwnerValue &value) {
  if (!value.getIf<EmptyPromotionModelConfig>())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "promotion config has the wrong type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue>
adoptPromotionConfig(llvm::ArrayRef<std::uint8_t> bytes,
                     const ComponentViewDigest &) {
  if (!bytes.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "promotion config is not empty");
  return OwnerValue::get(EmptyPromotionModelConfig{});
}

const ScopeFormRef promotionScopeForms[] = {ScopeFormRef(0)};
const MetricCapability promotionMetricCapabilities[] = {
    {MetricKind::Runtime, promotionScopeForms, allObservationFormsMask()}};
const EvaluationModelDescriptor promotionModelDescriptor{
    evidenceModelKind,
    "promotion_evidence_model",
    "loom.test.promotion.evidence_model.v1",
    evidenceSignatureRef(),
    {},
    promotionMetricCapabilities,
    {},
    {},
    {},
    {promotionConfigSchema(), &projectPromotionConfig, &encodePromotionConfig,
     &adoptPromotionConfig},
    {},
    EvaluationExecutionMethod::Analytic,
    {},
    DeterminismContract::Deterministic,
    {},
    ProviderForm::InProcess};

PromotionEvidence makeRuntimeEvidence(const ArtifactRootReference &candidate,
                                      MetricObservationValue observation,
                                      const ArtifactStore &store,
                                      const BlobStore &blobs) {
  CaseArtifactResolution resolution =
      take(CaseArtifactResolution::get({{candidate, {}}}));
  EvaluationSubjectBindings subjects = take(
      EvaluationSubjectBindings::get({{evidenceCandidateRole, {candidate}}}));
  EvaluationCase evaluationCase = take(EvaluationCase::get(
      evidenceSignatureRef(), std::move(subjects), std::nullopt, std::nullopt,
      {}, resolution, store, blobs));
  MetricRequest metric = take(MetricRequest::get(
      {MetricKind::Runtime, EvaluationScope{ScopeFormRef(0), {}}}, {},
      evaluationCase, resolution, store));
  ResolvedModelBinding model = take(ResolvedModelBinding::project(
      promotionModelDescriptor.reference(), {}, defaultResolvedConfig()));
  EvaluationRequest request = take(
      EvaluationRequest::get(evaluationCase, {metric}, {}, std::move(model), 0,
                             resolution, store, blobs));
  take(publishEvaluationRequest(request, store));
  EvaluationEvidence evidence = take(EvaluationEvidence::get(
      request, {},
      CompletedEvidence{{MetricResult{UncertaintyKind::ExactWithinModel,
                                      std::move(observation),
                                      {}}},
                        {}},
      resolution, store, blobs));
  return PromotionEvidence(std::move(request), std::move(evidence));
}

void metricGateUsesRepresentedSetProof() {
  MetricResult interval{UncertaintyKind::Bounded,
                        IntervalObservation{IntegerValue(0), IntegerValue(30)},
                        {}};
  require(take(evaluateMetricGate(
              MetricKind::CycleCount, interval, MetricGateComparator::LT,
              IntegerValue(40))) == GateTruth::DefinitelyTrue,
          "an interval wholly below the threshold was not proven true");
  require(take(evaluateMetricGate(
              MetricKind::CycleCount, interval, MetricGateComparator::LT,
              IntegerValue(20))) == GateTruth::Indeterminate,
          "a straddling interval did not remain indeterminate");
  require(take(evaluateMetricGate(
              MetricKind::CycleCount, interval, MetricGateComparator::GT,
              IntegerValue(40))) == GateTruth::DefinitelyFalse,
          "an interval wholly below a greater-than threshold was not false");

  MetricResult censored{
      UncertaintyKind::Bounded,
      CensoredObservation{MetricValue{IntegerValue(25)}, std::nullopt,
                          CensoredReason::SubjectDidNotComplete},
      {}};
  require(take(evaluateMetricGate(
              MetricKind::CycleCount, censored, MetricGateComparator::GE,
              IntegerValue(20))) == GateTruth::DefinitelyTrue,
          "a lower-censored set did not prove its lower-bound gate");
  require(take(evaluateMetricGate(
              MetricKind::CycleCount, censored, MetricGateComparator::LT,
              IntegerValue(30))) == GateTruth::Indeterminate,
          "an unbounded censored set was treated as a point");
}

void indeterminateAtomPrecedesBooleanSelection() {
  QualityGateClause clause;
  clause.atoms = {
      MetricGate{0, MetricRequestOrdinal(0), MetricGateComparator::LT,
                 IntegerValue(10)},
      FindingGate{1, FindingRequestOrdinal(0), RequiredFindingState::Absent},
  };
  QualityGatePolicy policy = take(QualityGatePolicy::get({std::move(clause)}));
  require(policy.atomCount() == 2,
          "quality policy did not preserve distinct canonical atoms");
  const std::array<GateTruth, 2> truths = {GateTruth::DefinitelyTrue,
                                           GateTruth::Indeterminate};
  require(take(evaluateQualityGate(policy, truths)) == GateTruth::Indeterminate,
          "a true sibling incorrectly hid an indeterminate obligation");
}

void paretoRetainsEveryNondominatedCandidate() {
  const ResolvedObjectiveCatalogs catalogs = resolvedBuiltinObjectiveCatalogs();
  const ObjectiveProgram program = take(ObjectiveProgram::get(catalogs));
  const ArtifactSchemaDescriptor schema{"loom.test.promotion_candidate",
                                        SchemaVersion{1, 0}};
  const ArtifactRootReference first = makeReference(0x11);
  const ArtifactRootReference second = makeReference(0x22);
  const ArtifactRootReference dominated = makeReference(0x33);
  const CandidateSet candidates =
      take(CandidateSet::get(schema, {dominated, second, first}));

  auto makeObjective = [&](const ArtifactRootReference &candidate,
                           std::uint64_t violation, std::uint64_t traversal) {
    std::vector<std::uint64_t> violations(resolvedPnrViolationKindCount, 0);
    violations[0] = violation;
    ObjectiveVector vector = program.makeVector();
    requireSuccess(program.evaluate({violations, {&traversal, 1}, {}}, vector));
    return CandidateObjectiveVector{candidate, std::move(vector)};
  };
  std::vector<CandidateObjectiveVector> objectives;
  objectives.push_back(makeObjective(first, 0, 5));
  objectives.push_back(makeObjective(second, 5, 0));
  objectives.push_back(makeObjective(dominated, 6, 6));
  const std::array<std::uint32_t, 2> dimensions = {
      0, resolvedPnrViolationKindCount};
  const CandidateSelectionPolicy policy = ParetoSelection{
      std::vector<std::uint32_t>(dimensions.begin(), dimensions.end())};
  const std::vector<ArtifactRootReference> selected =
      take(applyCandidateSelection(candidates, candidates.candidates(),
                                   objectives, policy, &program));
  require(selected == std::vector<ArtifactRootReference>({first, second}),
          "Pareto selection did not return the canonical nondominated set");
}

void promotionDerivesObjectivesOnlyFromPointEvidence() {
  requireSuccess(registerEvaluationCaseSignature(evidenceCaseSignature));
  requireSuccess(registerEvaluationModelDescriptor(promotionModelDescriptor));
  llvm::SmallString<128> storePath;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "loom-promotion-objective", storePath))
    fail(error.message());
  ArtifactStore store(storePath);
  BlobStore blobs(storePath);
  const auto putCandidate = [&](std::uint8_t byte) {
    ArtifactIdentity identity = take(
        store.put(evidenceCandidateSchema,
                  CanonicalSemanticBytes(std::vector<std::uint8_t>{byte})));
    return ArtifactRootReference{evidenceCandidateSchema.identity.str(),
                                 evidenceCandidateSchema.version, identity};
  };
  const ArtifactRootReference faster = putCandidate(0x11);
  const ArtifactRootReference slower = putCandidate(0x22);
  CandidateSet candidates =
      take(CandidateSet::get(evidenceCandidateSchema, {slower, faster}));

  ResolvedObjectiveCatalogs catalogs;
  catalogs.dimensions = {{ResolvedEvaluationMetricObjectiveSource{0, 0},
                          ResolvedObjectiveDirection::Minimize,
                          resolvedObjectiveDecimal(0, 0),
                          resolvedObjectiveDecimal(1, 0), 0, 100}};
  catalogs.weightedLevels = {{{{0, 1}}}};
  catalogs.totalOrderings = {{{0}}};
  ObjectiveProgram program = take(ObjectiveProgram::get(catalogs));
  QualityGatePolicy gate = take(QualityGatePolicy::get({}));

  PromotionEvidence fastEvidence = makeRuntimeEvidence(
      faster, PointObservation{take(DecimalValue::get(3, 0))}, store, blobs);
  PromotionEvidence slowEvidence = makeRuntimeEvidence(
      slower, PointObservation{take(DecimalValue::get(8, 0))}, store, blobs);
  PromotionOutcome promoted = take(promoteCandidates(
      candidates, evidenceCandidateRole, {slowEvidence, fastEvidence}, gate,
      TopKSelection{0, 1}, &program, store));
  const auto *selection = std::get_if<CompletedSelection>(&promoted);
  require(selection &&
              selection->selected == std::vector<ArtifactRootReference>{faster},
          "Promotion did not derive TopK from exact Evidence metrics");

  PromotionEvidence intervalEvidence =
      makeRuntimeEvidence(slower,
                          IntervalObservation{take(DecimalValue::get(7, 0)),
                                              take(DecimalValue::get(9, 0))},
                          store, blobs);
  PromotionOutcome unavailable = take(promoteCandidates(
      candidates, evidenceCandidateRole, {fastEvidence, intervalEvidence}, gate,
      TopKSelection{0, 1}, &program, store));
  const auto *incomplete = std::get_if<IncompleteSelection>(&unavailable);
  require(incomplete && incomplete->reason ==
                            IncompleteSelectionReason::ObjectiveUnavailable,
          "Promotion assigned a numeric value to non-Point Evidence");
  llvm::sys::fs::remove_directories(storePath);
}

} // namespace

int main() {
  metricGateUsesRepresentedSetProof();
  indeterminateAtomPrecedesBooleanSelection();
  paretoRetainsEveryNondominatedCandidate();
  promotionDerivesObjectivesOnlyFromPointEvidence();
  return 0;
}
