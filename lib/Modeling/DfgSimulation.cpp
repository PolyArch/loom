#include "Evaluation/Models/DfgSimulation.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/ModelProvider.h"
#include "Simulator/DFGSimulator.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr EvaluationCaseKind kCaseKind(3);
constexpr EvaluationModelKind kModelKind(5);
constexpr CaseSubjectRoleRef kCanonicalDataflowRole(0);
constexpr ModelOutputSlotRef kExecutionOutputSlot(0);
constexpr ScopeFormRef kWholeExactCaseScope(0);

EvaluationCaseSignatureRef caseSignatureRef() {
  return llvm::cantFail(
      EvaluationCaseSignatureRef::get(evaluationSchemaVersion(), kCaseKind));
}

const ArtifactSchemaDescriptor *const kDataflowSchemas[] = {
    &dataflow::canonicalDataflowSchema};
const ArtifactSchemaDescriptor *const kWorkloadSchemas[] = {
    &sim::simulationWorkloadSchema};
const ArtifactSchemaDescriptor *const kRuntimeInputSchemas[] = {
    &sim::simulationRuntimeInputSchema};

const CaseSubjectRoleDescriptor kSubjectRoles[] = {
    {kCanonicalDataflowRole, "canonical_dataflow",
     SubjectRoleCardinality::ExactlyOne, kDataflowSchemas, nullptr}};

llvm::Error verifyWorkloadCompatibility(
    const EvaluationSubjectBindings &bindings,
    const std::optional<ArtifactRootReference> &workload,
    const std::optional<ArtifactRootReference> &runtimeInput,
    const CaseArtifactResolution &resolution) {
  const llvm::ArrayRef<ArtifactRootReference> subjects =
      bindings.subjects(kCanonicalDataflowRole);
  if (subjects.size() != 1 || !workload || !runtimeInput)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "dfg_simulation_model_invalid: exact case inputs are not total");
  const CaseArtifactResolution::Entry *workloadEntry =
      resolution.find(*workload);
  const CaseArtifactResolution::Entry *runtimeEntry =
      resolution.find(*runtimeInput);
  if (!workloadEntry || !runtimeEntry ||
      !CaseArtifactResolution::reaches(*workloadEntry, subjects.front()) ||
      !CaseArtifactResolution::reaches(*runtimeEntry, subjects.front()) ||
      !CaseArtifactResolution::reaches(*runtimeEntry, *workload))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "dfg_simulation_model_invalid: workload lineage does not reach the "
        "exact Canonical Dataflow Program");
  return llvm::Error::success();
}

const EvaluationCaseSignatureDescriptor kCaseSignature{
    kCaseKind,
    "canonical_dataflow_simulation",
    "One exact Canonical Dataflow Program executed with one exact Spatial "
    "workload and runtime input.",
    kSubjectRoles,
    ArtifactRequirement::Required,
    kWorkloadSchemas,
    ArtifactRequirement::Required,
    kRuntimeInputSchemas,
    &verifyWorkloadCompatibility,
    AbstractCaseCycle{},
    {}};

const ScopeFormRef kWholeCaseScopeForms[] = {kWholeExactCaseScope};
const MetricCapability kMetricCapabilities[] = {
    {MetricKind::CycleCount, kWholeCaseScopeForms,
     observationFormMask(ObservationForm::Point)}};
const ModelOutputSlotDescriptor kOutputSlots[] = {{
    kExecutionOutputSlot,
    "simulation_execution",
    &sim::simulationExecutionSchema,
    {ArtifactCollectionCardinality::ExactlyOne,
     ArtifactCollectionCardinality::Forbidden,
     ArtifactCollectionCardinality::Forbidden,
     ArtifactCollectionCardinality::Forbidden},
}};
const ModeledPhenomenon kModeledPhenomena[] = {
    ModeledPhenomenon::CanonicalDataflow};

struct EmptyDfgSimulationConfig final {};

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.dfg_simulator.config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptyDfgSimulationConfig{});
}

llvm::Expected<std::vector<std::uint8_t>>
encodeConfig(const OwnerValue &value) {
  if (!value.getIf<EmptyDfgSimulationConfig>())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "dfg_simulation_model_invalid: config has the wrong owner type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue> adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                       const ComponentViewDigest &) {
  if (!bytes.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "dfg_simulation_model_invalid: config view must be empty");
  return OwnerValue::get(EmptyDfgSimulationConfig{});
}

const ResolvedModelConfigViewContract kConfigView{
    configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig};

const EvaluationModelDescriptor kModelDescriptor{
    kModelKind,
    "dfg_simulator",
    "loom.dfg_simulator.abstract_cycle.v1",
    caseSignatureRef(),
    {},
    kMetricCapabilities,
    {},
    {},
    kOutputSlots,
    kConfigView,
    kModeledPhenomena,
    EvaluationExecutionMethod::Simulation,
    {},
    DeterminismContract::Deterministic,
    {}};

llvm::Expected<EvaluationModelResult>
classifyExecutionFailure(llvm::Error error) {
  std::error_code code;
  llvm::handleAllErrors(std::move(error),
                        [&](const llvm::ErrorInfoBase &failure) {
                          code = failure.convertToErrorCode();
                        });
  if (code == std::make_error_code(std::errc::not_supported))
    return EvaluationModelResult{
        {{kExecutionOutputSlot, {}}},
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  if (code == std::make_error_code(std::errc::timed_out))
    return EvaluationModelResult{
        {{kExecutionOutputSlot, {}}},
        CancelledOrTimeoutEvidence{OutcomeReason::ExecutionLimitReached}};
  return EvaluationModelResult{
      {{kExecutionOutputSlot, {}}},
      ExecutionFailedEvidence{OutcomeReason::AdapterFailure}};
}

llvm::Expected<EvaluationModelResult> evaluateWithLimits(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, DfgSimulationAttemptLimits limits) {
  if (request.modelBinding().descriptorRef() != kModelDescriptor.reference())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "dfg_simulation_model_invalid: Request selects a foreign model");
  if (limits.maxWavefrontSteps == 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "dfg_simulation_model_invalid: attempt requires a positive "
        "wavefront limit");

  const llvm::ArrayRef<ArtifactRootReference> subjects =
      request.subjectBindings().subjects(kCanonicalDataflowRole);
  if (subjects.size() != 1 || !request.workload() || !request.runtimeInput())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "dfg_simulation_model_invalid: Request inputs are not total");
  auto inputs = sim::importSpatialSimulationInputs(
      *request.workload(), *request.runtimeInput(), artifactStore);
  if (!inputs)
    return inputs.takeError();
  if (inputs->dataflow.identity() != subjects.front().artifact)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "dfg_simulation_model_invalid: workload names a foreign Dataflow "
        "owner");

  auto retired = sim::simulateRetiredDfgWorkload(
      inputs->dataflow, inputs->workload, inputs->runtimeInput,
      limits.maxWavefrontSteps, limits.executionDeadline);
  if (!retired)
    return classifyExecutionFailure(retired.takeError());
  if (retired->report.wavefrontSteps >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return EvaluationModelResult{
        {{kExecutionOutputSlot, {}}},
        ExecutionFailedEvidence{OutcomeReason::AdapterFailure}};

  ExactRatio zero = llvm::cantFail(ExactRatio::get(0, 1));
  ExactRatio retirement =
      llvm::cantFail(ExactRatio::get(retired->report.wavefrontSteps, 1));
  const sim::SpatialEventCoordinate launch{std::move(zero), 0};
  const sim::SpatialEventCoordinate retiredAt{std::move(retirement), 0};
  sim::SpatialSimulationExecution execution{
      evaluationRequestReference(request),
      sim::RetiredExecution{},
      std::move(retired->observations),
      sim::SpatialProgressObservations{launch, retiredAt, retiredAt},
      {}};
  auto finalized =
      sim::finalizeSimulationExecution(execution, resolution, artifactStore);
  if (!finalized)
    return finalized.takeError();
  auto executionReference =
      sim::publishSimulationExecution(*finalized, artifactStore);
  if (!executionReference)
    return executionReference.takeError();

  const auto &progress = finalized->progressObservations();
  const std::uint64_t cycleCount =
      progress.graphRetirementVisible->referenceCycle.numerator();
  std::vector<MetricResult> metrics;
  metrics.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    if (metric.query().metric != MetricKind::CycleCount)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "dfg_simulation_model_invalid: unsupported metric request");
    metrics.push_back(MetricResult{
        UncertaintyKind::ExactWithinModel,
        PointObservation{IntegerValue(static_cast<std::int64_t>(cycleCount))},
        {}});
  }
  return EvaluationModelResult{
      {{kExecutionOutputSlot, {std::move(*executionReference)}}},
      CompletedEvidence{std::move(metrics), {}}};
}

llvm::Expected<EvaluationModelResult>
evaluate(const EvaluationRequest &request,
         const CaseArtifactResolution &resolution,
         const ArtifactStore &artifactStore) {
  return evaluateWithLimits(request, resolution, artifactStore,
                            DfgSimulationAttemptLimits{});
}

const EvaluationModelProvider kProvider{kModelDescriptor.reference(),
                                        &evaluate};

} // namespace

llvm::Error registerDfgSimulationModel() {
  if (llvm::Error error = registerEvaluationCaseSignature(kCaseSignature))
    return error;
  if (llvm::Error error = registerEvaluationModelDescriptor(kModelDescriptor))
    return error;
  return registerEvaluationModelProvider(kProvider);
}

llvm::Expected<PreparedDfgSimulationEvaluation>
prepareDfgSimulationEvaluation(const ArtifactRootReference &canonicalDataflow,
                               const ArtifactRootReference &workload,
                               const ArtifactRootReference &runtimeInput,
                               const ResolvedConfig &config,
                               const ArtifactStore &artifactStore) {
  if (llvm::Error error = registerDfgSimulationModel())
    return std::move(error);
  auto inputs =
      sim::importSpatialSimulationInputs(workload, runtimeInput, artifactStore);
  if (!inputs)
    return inputs.takeError();
  if (inputs->dataflow.identity() != canonicalDataflow.artifact)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "dfg_simulation_model_invalid: workload names a foreign Dataflow "
        "owner");

  auto resolution = CaseArtifactResolution::get(
      {{canonicalDataflow, {}},
       {workload, {canonicalDataflow}},
       {runtimeInput, {canonicalDataflow, workload}}});
  if (!resolution)
    return resolution.takeError();
  auto bindings = EvaluationSubjectBindings::get(
      {{kCanonicalDataflowRole, {canonicalDataflow}}});
  if (!bindings)
    return bindings.takeError();
  auto evaluationCase =
      EvaluationCase::get(caseSignatureRef(), std::move(*bindings), workload,
                          runtimeInput, {}, *resolution, artifactStore);
  if (!evaluationCase)
    return evaluationCase.takeError();
  auto cycleCount =
      MetricRequest::get(MetricQuery{MetricKind::CycleCount,
                                     EvaluationScope{kWholeExactCaseScope, {}}},
                         {}, *evaluationCase, *resolution, artifactStore);
  if (!cycleCount)
    return cycleCount.takeError();
  auto modelBinding =
      ResolvedModelBinding::project(kModelDescriptor.reference(), {}, config);
  if (!modelBinding)
    return modelBinding.takeError();
  auto request = EvaluationRequest::get(*evaluationCase, {*cycleCount}, {},
                                        std::move(*modelBinding), 0,
                                        *resolution, artifactStore);
  if (!request)
    return request.takeError();
  auto requestReference = publishEvaluationRequest(*request, artifactStore);
  if (!requestReference)
    return requestReference.takeError();
  return PreparedDfgSimulationEvaluation{std::move(*request),
                                         std::move(*resolution)};
}

llvm::Expected<EvaluationEvidence>
evaluateDfgSimulation(const PreparedDfgSimulationEvaluation &prepared,
                      DfgSimulationAttemptLimits limits,
                      const ArtifactStore &artifactStore) {
  RequestVerifier verifier(prepared.resolution, artifactStore);
  if (llvm::Error error = verifier.verify(prepared.request))
    return std::move(error);
  auto result = evaluateWithLimits(prepared.request, prepared.resolution,
                                   artifactStore, std::move(limits));
  if (!result)
    return result.takeError();
  return EvaluationEvidence::get(
      prepared.request, std::move(result->outputBindings),
      std::move(result->outcome), prepared.resolution, artifactStore);
}

} // namespace loom::evaluation::models
