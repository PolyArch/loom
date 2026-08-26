#include "Evaluation/Models/DfgSimulation.h"
#include "Evaluation/ProductionRegistry.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/InvocationDiagnosticLog.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/OperationSchema.h"
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

constexpr BuiltinEvaluationCase kCase =
    BuiltinEvaluationCase::CanonicalDataflowSimulation;
constexpr BuiltinEvaluationModel kModel = BuiltinEvaluationModel::DfgSimulator;
constexpr CaseSubjectRoleRef kCanonicalDataflowRole(0);
constexpr ModelOutputSlotRef kExecutionOutputSlot(0);
constexpr ScopeFormRef kWholeExactCaseScope(0);

EvaluationCaseSignatureRef caseSignatureRef() {
  return llvm::cantFail(EvaluationCaseSignatureRef::get(
      evaluationSchemaVersion(), builtinEvaluationCaseKind(kCase)));
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
    const EvaluationCase &, const EvaluationSubjectBindings &bindings,
    const std::optional<ArtifactRootReference> &workload,
    const std::optional<ArtifactRootReference> &runtimeInput,
    const CaseArtifactResolution &resolution, const ArtifactStore &,
    const BlobStore &) {
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
    builtinEvaluationCaseKind(kCase),
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
    builtinEvaluationModelKind(kModel),
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
    {},
    ProviderForm::InProcess};

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
    const ArtifactStore &artifactStore, const BlobStore &blobStore,
    DfgSimulationAttemptLimits limits) {
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
  auto finalized = sim::finalizeSimulationExecution(execution, resolution,
                                                    artifactStore, blobStore);
  if (!finalized)
    return finalized.takeError();
  auto executionReference =
      sim::publishSimulationExecution(*finalized, artifactStore);
  if (!executionReference)
    return executionReference.takeError();

  const auto &progress = finalized->spatialProgressObservations();
  const std::uint64_t cycleCount =
      progress.graphRetirementVisible->referenceCycle.numerator();
  std::uint64_t dynamicOperationFires = 0;
  std::uint64_t loadCount = 0;
  std::uint64_t storeCount = 0;
  std::uint64_t atomicMemoryOperationCount = 0;
  std::uint64_t fenceCount = 0;
  std::uint64_t computeOperationCount = 0;
  std::uint64_t controlOperationCount = 0;
  std::uint64_t memoryOperationCount = 0;
  std::uint64_t recurrenceCarrierCount = 0;
  std::uint64_t streamActorCount = 0;
  std::uint64_t syncActorCount = 0;
  llvm::json::Object operationFireCounts;
  for (const auto &[operation, count] : retired->report.operationFireCounts) {
    if (count >
        std::numeric_limits<std::uint64_t>::max() - dynamicOperationFires)
      return llvm::createStringError(
          std::errc::value_too_large,
          "dfg_simulation_model_invalid: operation fire count overflows");
    dynamicOperationFires += count;
    operationFireCounts[dataflow::operationSchemaSpelling(operation)] = count;
    switch (dataflow::actorKind(operation)) {
    case dataflow::CanonicalDataflowActorKind::Compute:
      computeOperationCount += count;
      break;
    case dataflow::CanonicalDataflowActorKind::Control:
      controlOperationCount += count;
      break;
    case dataflow::CanonicalDataflowActorKind::Memory:
      memoryOperationCount += count;
      break;
    }
    switch (operation) {
    case dataflow::OperationSchemaId::DataflowCarry:
      recurrenceCarrierCount += count;
      break;
    case dataflow::OperationSchemaId::DataflowStream:
      streamActorCount += count;
      break;
    case dataflow::OperationSchemaId::DataflowSync:
      syncActorCount += count;
      break;
    case dataflow::OperationSchemaId::DataflowLoad:
      loadCount += count;
      break;
    case dataflow::OperationSchemaId::DataflowStore:
      storeCount += count;
      break;
    case dataflow::OperationSchemaId::DataflowAtomicRmw:
    case dataflow::OperationSchemaId::DataflowCmpXchg:
      atomicMemoryOperationCount += count;
      break;
    case dataflow::OperationSchemaId::DataflowFence:
      fenceCount += count;
      break;
    default:
      break;
    }
  }
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::DataflowLowering,
      InvocationDiagnosticEvent::Statistics, [&] {
        llvm::json::Object fields;
        llvm::json::Object direct;
        direct["cycle_count"] = cycleCount;
        direct["wavefront_steps"] = retired->report.wavefrontSteps;
        direct["event_count"] = retired->report.eventCount;
        direct["dynamic_work_items"] = retired->report.dynamicWorkItems;
        direct["dynamic_operation_fires"] = dynamicOperationFires;
        direct["operation_kind_count"] =
            retired->report.operationFireCounts.size();
        direct["operation_fire_counts"] = std::move(operationFireCounts);
        direct["compute_operation_count"] = computeOperationCount;
        direct["control_operation_count"] = controlOperationCount;
        direct["memory_operation_count"] = memoryOperationCount;
        direct["recurrence_carrier_count"] = recurrenceCarrierCount;
        direct["stream_actor_count"] = streamActorCount;
        direct["sync_actor_count"] = syncActorCount;
        direct["load_count"] = loadCount;
        direct["store_count"] = storeCount;
        direct["atomic_memory_operation_count"] = atomicMemoryOperationCount;
        direct["fence_count"] = fenceCount;
        direct["modeled_library_call_count"] =
            retired->report.modeledLibraryCalls.size();
        llvm::json::Object derived;
        const auto rate = [](std::uint64_t numerator,
                             std::uint64_t denominator) -> llvm::json::Value {
          if (denominator == 0)
            return llvm::json::Value(nullptr);
          return llvm::json::Object{{"numerator", numerator},
                                    {"denominator", denominator}};
        };
        derived["modeled_instruction_ipc"] =
            rate(dynamicOperationFires, cycleCount);
        derived["modeled_instruction_cpi"] =
            rate(cycleCount, dynamicOperationFires);
        derived["cycles_per_dynamic_work_item"] =
            rate(cycleCount, retired->report.dynamicWorkItems);
        derived["recurrence_or_ii"] = "unsupported_single_activation";
        fields["measurement_kind"] = "direct_and_derived";
        fields["direct"] = std::move(direct);
        fields["derived"] = std::move(derived);
        fields["operation"] = "simulation_cycle_breakdown";
        fields["engine"] = "dfg";
        fields["request"] = formatArtifactRootReferenceJson(
            evaluationRequestReference(request));
        fields["cycle_count"] = cycleCount;
        fields["wavefront_steps"] = retired->report.wavefrontSteps;
        fields["event_count"] = retired->report.eventCount;
        fields["dynamic_work_items"] = retired->report.dynamicWorkItems;
        fields["dynamic_operation_fires"] = dynamicOperationFires;
        fields["operation_kind_count"] =
            retired->report.operationFireCounts.size();
        fields["compute_operation_count"] = computeOperationCount;
        fields["control_operation_count"] = controlOperationCount;
        fields["memory_operation_count"] = memoryOperationCount;
        fields["load_count"] = loadCount;
        fields["store_count"] = storeCount;
        fields["atomic_memory_operation_count"] = atomicMemoryOperationCount;
        fields["fence_count"] = fenceCount;
        fields["modeled_library_call_count"] =
            retired->report.modeledLibraryCalls.size();
        return llvm::json::Value(std::move(fields));
      });
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
         const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  return evaluateWithLimits(request, resolution, artifactStore, blobStore,
                            DfgSimulationAttemptLimits{});
}

const EvaluationModelProvider kProvider{
    kModelDescriptor.reference(), EvaluationModelInProcessProvider{&evaluate}};

} // namespace

llvm::Error registerDfgSimulationModel() {
  if (llvm::Error error = registerEvaluationCaseSignature(kCaseSignature))
    return error;
  if (llvm::Error error = registerEvaluationModelDescriptor(kModelDescriptor))
    return error;
  return registerEvaluationModelProvider(kProvider);
}

llvm::Expected<CaseArtifactResolution>
resolveDfgSimulationCase(const ArtifactRootReference &canonicalDataflow,
                         const ArtifactRootReference &workload,
                         const ArtifactRootReference &runtimeInput,
                         const ArtifactStore &artifactStore) {
  auto inputs =
      sim::importSpatialSimulationInputs(workload, runtimeInput, artifactStore);
  if (!inputs)
    return inputs.takeError();
  if (inputs->dataflow.identity() != canonicalDataflow.artifact)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "dfg_simulation_model_invalid: workload names a foreign Dataflow "
        "owner");
  return CaseArtifactResolution::get(
      {{canonicalDataflow, {}},
       {workload, {canonicalDataflow}},
       {runtimeInput, {canonicalDataflow, workload}}});
}

llvm::Expected<PreparedDfgSimulationEvaluation> prepareDfgSimulationEvaluation(
    const ArtifactRootReference &canonicalDataflow,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  if (llvm::Error error = registerDfgSimulationModel())
    return std::move(error);
  auto resolution = resolveDfgSimulationCase(canonicalDataflow, workload,
                                             runtimeInput, artifactStore);
  if (!resolution)
    return resolution.takeError();
  auto bindings = EvaluationSubjectBindings::get(
      {{kCanonicalDataflowRole, {canonicalDataflow}}});
  if (!bindings)
    return bindings.takeError();
  auto evaluationCase = EvaluationCase::get(
      caseSignatureRef(), std::move(*bindings), workload, runtimeInput, {},
      *resolution, artifactStore, blobStore);
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
                                        *resolution, artifactStore, blobStore);
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
                      const ArtifactStore &artifactStore,
                      const BlobStore &blobStore) {
  RequestVerifier verifier(prepared.resolution, artifactStore, blobStore);
  if (llvm::Error error = verifier.verify(prepared.request))
    return std::move(error);
  auto result = evaluateWithLimits(prepared.request, prepared.resolution,
                                   artifactStore, blobStore, std::move(limits));
  if (!result)
    return result.takeError();
  return EvaluationEvidence::get(prepared.request,
                                 std::move(result->outputBindings),
                                 std::move(result->outcome),
                                 prepared.resolution, artifactStore, blobStore);
}

} // namespace loom::evaluation::models
