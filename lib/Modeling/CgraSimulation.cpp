#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/ProductionRegistry.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/ModelProvider.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Simulator/CGRASimulator.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr BuiltinEvaluationCase kCase = BuiltinEvaluationCase::CgraSimulation;
constexpr BuiltinEvaluationModel kModel = BuiltinEvaluationModel::CgraSimulator;
constexpr CaseSubjectRoleRef kProgramRole(0);
constexpr CaseSubjectRoleRef kHardwareRole(1);
constexpr CaseSubjectRoleRef kSpatialMappingRole(2);
constexpr ModelOutputSlotRef kExecutionOutputSlot(0);
constexpr ScopeFormRef kWholeExactCaseScope(0);

EvaluationCaseSignatureRef caseSignatureRef() {
  return llvm::cantFail(EvaluationCaseSignatureRef::get(
      evaluationSchemaVersion(), builtinEvaluationCaseKind(kCase)));
}

SubjectReferenceType spatialMappingRootType() {
  return SubjectReferenceType{ArtifactRootType{mapping::mappingArtifactSchema}};
}

llvm::Expected<SubjectTargetRef>
resolveReferenceCycle(const EvaluationCase &evaluationCase,
                      const CaseArtifactResolution &, const ArtifactStore &,
                      const BlobStore &) {
  const auto mappings =
      evaluationCase.subjectBindings().subjects(kSpatialMappingRole);
  if (mappings.size() != 1)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "cgra_simulation_model_invalid: reference cycle requires one exact "
        "SpatialMapping");
  return SubjectTargetRef{kSpatialMappingRole, mappings.front(),
                          SubjectTarget{mappings.front()}};
}

const ArtifactSchemaDescriptor *const kDataflowSchemas[] = {
    &dataflow::canonicalDataflowSchema};
const ArtifactSchemaDescriptor *const kFabricSchemas[] = {
    &fabric::fabricArtifactSchema};
const ArtifactSchemaDescriptor *const kMappingSchemas[] = {
    &mapping::mappingArtifactSchema};
const ArtifactSchemaDescriptor *const kWorkloadSchemas[] = {
    &sim::simulationWorkloadSchema};
const ArtifactSchemaDescriptor *const kRuntimeInputSchemas[] = {
    &sim::simulationRuntimeInputSchema};

llvm::Error
verifyMappingCompatibility(const ArtifactRootReference &mapping,
                           const EvaluationCase &,
                           const EvaluationSubjectBindings &bindings,
                           const CaseArtifactResolution &resolution,
                           const ArtifactStore &, const BlobStore &) {
  const auto programs = bindings.subjects(kProgramRole);
  const auto hardware = bindings.subjects(kHardwareRole);
  const CaseArtifactResolution::Entry *entry = resolution.find(mapping);
  if (programs.size() != 1 || hardware.size() != 1 || !entry ||
      !CaseArtifactResolution::reaches(*entry, programs.front()) ||
      !CaseArtifactResolution::reaches(*entry, hardware.front()))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "cgra_simulation_model_invalid: SpatialMapping does not reach its "
        "exact Dataflow and Fabric owners");
  return llvm::Error::success();
}

const CaseSubjectRoleDescriptor kSubjectRoles[] = {
    {kProgramRole, "program", SubjectRoleCardinality::ExactlyOne,
     kDataflowSchemas, nullptr},
    {kHardwareRole, "hardware", SubjectRoleCardinality::ExactlyOne,
     kFabricSchemas, nullptr},
    {kSpatialMappingRole, "spatial_mapping", SubjectRoleCardinality::ExactlyOne,
     kMappingSchemas, &verifyMappingCompatibility}};

llvm::Error verifyWorkloadCompatibility(
    const EvaluationCase &, const EvaluationSubjectBindings &bindings,
    const std::optional<ArtifactRootReference> &workload,
    const std::optional<ArtifactRootReference> &runtimeInput,
    const CaseArtifactResolution &resolution, const ArtifactStore &,
    const BlobStore &) {
  const auto programs = bindings.subjects(kProgramRole);
  if (programs.size() != 1 || !workload || !runtimeInput)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "cgra_simulation_model_invalid: exact case inputs are not total");
  const CaseArtifactResolution::Entry *workloadEntry =
      resolution.find(*workload);
  const CaseArtifactResolution::Entry *runtimeEntry =
      resolution.find(*runtimeInput);
  if (!workloadEntry || !runtimeEntry ||
      !CaseArtifactResolution::reaches(*workloadEntry, programs.front()) ||
      !CaseArtifactResolution::reaches(*runtimeEntry, programs.front()) ||
      !CaseArtifactResolution::reaches(*runtimeEntry, *workload))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "cgra_simulation_model_invalid: workload lineage does not reach the "
        "exact Canonical Dataflow Program");
  return llvm::Error::success();
}

const EvaluationCaseSignatureDescriptor kCaseSignature{
    builtinEvaluationCaseKind(kCase),
    "cgra_simulation",
    "One exact Canonical Dataflow Program executed on one exact Fabric and "
    "complete SpatialMapping.",
    kSubjectRoles,
    ArtifactRequirement::Required,
    kWorkloadSchemas,
    ArtifactRequirement::Required,
    kRuntimeInputSchemas,
    &verifyWorkloadCompatibility,
    ExactSubjectCycle{spatialMappingRootType(), &resolveReferenceCycle},
    {}};

const ScopeFormRef kWholeCaseScopeForms[] = {kWholeExactCaseScope};
const MetricCapability kMetricCapabilities[] = {{
    MetricKind::CycleCount,
    kWholeCaseScopeForms,
    observationFormMask(ObservationForm::Point),
}};
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
    ModeledPhenomenon::CanonicalDataflow, ModeledPhenomenon::SpatialResources,
    ModeledPhenomenon::RoutedTransport,   ModeledPhenomenon::FiniteBuffering,
    ModeledPhenomenon::MemoryContention,  ModeledPhenomenon::ClockTiming};

struct EmptyCgraSimulationConfig final {};

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.cgra_simulator.config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptyCgraSimulationConfig{});
}

llvm::Expected<std::vector<std::uint8_t>>
encodeConfig(const OwnerValue &value) {
  if (!value.getIf<EmptyCgraSimulationConfig>())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "cgra_simulation_model_invalid: config has the wrong owner type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue> adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                       const ComponentViewDigest &) {
  if (!bytes.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "cgra_simulation_model_invalid: config view must be empty");
  return OwnerValue::get(EmptyCgraSimulationConfig{});
}

const ResolvedModelConfigViewContract kConfigView{
    configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig};

const EvaluationModelDescriptor kModelDescriptor{
    builtinEvaluationModelKind(kModel),
    "cgra_simulator",
    "loom.cgra_simulator.exact_mapping.v1",
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
  llvm::handleAllErrors(std::move(error), [&](const llvm::ErrorInfoBase &info) {
    code = info.convertToErrorCode();
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

llvm::Expected<CaseArtifactResolution>
buildResolution(const sim::CgraExecutionOwnerReferences &owners,
                const ArtifactRootReference &workload,
                const ArtifactRootReference &runtimeInput) {
  return CaseArtifactResolution::get(
      {{owners.dataflow, {}},
       {owners.fabric, {}},
       {owners.techMapping, {owners.dataflow, owners.fabric}},
       {owners.spatialMapping,
        {owners.dataflow, owners.fabric, owners.techMapping}},
       {workload, {owners.dataflow}},
       {runtimeInput, {owners.dataflow, workload}}});
}

class RuntimeInputCgraMemoryProvider final
    : public sim::CgraExternalMemoryProvider {
public:
  explicit RuntimeInputCgraMemoryProvider(
      const sim::SpatialSimulationRuntimeInput &input) {
    objects_.reserve(input.memoryObjects.size());
    for (const sim::RuntimeMemoryObject &object : input.memoryObjects)
      objects_.push_back(object.initialBytes);
  }

  llvm::Expected<sim::CgraExternalMemoryResponse>
  transact(const sim::CgraExternalMemoryRequest &request) override {
    if (request.elements.empty())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "CGRA runtime-input memory request has no active element");
    if (request.objectOrdinal >= objects_.size())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "CGRA runtime-input memory request names no object");
    if (lastCoordinate_ && sim::compareSpatialEventCoordinates(
                               *lastCoordinate_, request.readyCoordinate) > 0)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "CGRA runtime-input memory requests are not time ordered");

    const bool write =
        request.operation == sim::CgraExternalMemoryOperation::Write;
    std::vector<sim::SemanticMemoryByte> &object =
        objects_[request.objectOrdinal];
    for (const sim::CgraExternalMemoryElement &element : request.elements) {
      if (element.byteCount == 0 || element.byteOffset > object.size() ||
          element.byteCount > object.size() - element.byteOffset)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "CGRA runtime-input memory element exceeds its object");
      if ((write && element.writeData.size() != element.byteCount) ||
          (!write && !element.writeData.empty()))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "CGRA runtime-input memory element has the wrong payload");
      if (!write)
        for (std::uint64_t byte = 0; byte != element.byteCount; ++byte)
          if (object[element.byteOffset + byte].state !=
              sim::SemanticState::Defined)
            return llvm::createStringError(
                std::errc::not_supported,
                "CGRA external read observes an exceptional runtime byte");
    }

    sim::CgraExternalMemoryResponse response;
    if (write) {
      for (const sim::CgraExternalMemoryElement &element : request.elements)
        for (std::uint64_t byte = 0; byte != element.byteCount; ++byte)
          object[element.byteOffset + byte] = sim::SemanticMemoryByte{
              sim::SemanticState::Defined, element.writeData[byte]};
    } else {
      response.readData.reserve(request.elements.size());
      for (const sim::CgraExternalMemoryElement &element : request.elements) {
        std::vector<std::uint8_t> bytes;
        bytes.reserve(element.byteCount);
        for (std::uint64_t byte = 0; byte != element.byteCount; ++byte)
          bytes.push_back(object[element.byteOffset + byte].value);
        response.readData.push_back(std::move(bytes));
      }
    }
    lastCoordinate_ = request.readyCoordinate;
    return response;
  }

private:
  std::vector<std::vector<sim::SemanticMemoryByte>> objects_;
  std::optional<sim::SpatialEventCoordinate> lastCoordinate_;
};

llvm::Expected<EvaluationModelResult> evaluateWithPrepared(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const sim::PreparedCgraExecution &execution,
    const sim::CanonicalSimulationWorkload &workload,
    const sim::CanonicalSimulationRuntimeInput &runtimeInput,
    CgraSimulationAttemptLimits limits, const ArtifactStore &artifactStore,
    const BlobStore &blobStore) {
  if (request.modelBinding().descriptorRef() != kModelDescriptor.reference())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "cgra_simulation_model_invalid: Request selects a foreign model");
  if (limits.maxEventFrames == 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "cgra_simulation_model_invalid: attempt requires a positive event "
        "frame limit");

  const sim::SpatialSimulationRuntimeInput *spatialInput =
      runtimeInput.spatial();
  if (!spatialInput)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "cgra_simulation_model_invalid: runtime input is not Spatial");
  RuntimeInputCgraMemoryProvider externalMemory(*spatialInput);
  auto outcome = sim::simulateCgraWorkload(
      execution, workload, runtimeInput, limits.maxEventFrames,
      limits.executionDeadline, &externalMemory);
  if (!outcome)
    return classifyExecutionFailure(outcome.takeError());
  if (outcome->state == sim::SpatialExecutionSessionState::StoppedByLimit)
    return EvaluationModelResult{
        {{kExecutionOutputSlot, {}}},
        CancelledOrTimeoutEvidence{OutcomeReason::ExecutionLimitReached}};
  if (outcome->state != sim::SpatialExecutionSessionState::Retired ||
      !outcome->retired)
    return EvaluationModelResult{
        {{kExecutionOutputSlot, {}}},
        ExecutionFailedEvidence{OutcomeReason::AdapterFailure}};
  const auto &retirement = outcome->retired->progress.graphRetirementVisible;
  if (!retirement || retirement->referenceCycle.denominator() != 1 ||
      retirement->referenceCycle.numerator() >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return EvaluationModelResult{
        {{kExecutionOutputSlot, {}}},
        ExecutionFailedEvidence{OutcomeReason::AdapterFailure}};
  const std::uint64_t cycleCount = retirement->referenceCycle.numerator();

  sim::SpatialSimulationExecution model{
      evaluationRequestReference(request),
      sim::RetiredExecution{},
      std::move(outcome->retired->observations),
      std::move(outcome->retired->progress),
      {}};
  auto finalized = sim::finalizeSimulationExecution(model, resolution,
                                                    artifactStore, blobStore);
  if (!finalized)
    return finalized.takeError();
  auto reference = sim::publishSimulationExecution(*finalized, artifactStore);
  if (!reference)
    return reference.takeError();
  std::vector<MetricResult> metrics;
  metrics.reserve(request.metricRequests().size());
  for (const MetricRequest &metric : request.metricRequests()) {
    if (metric.query().metric != MetricKind::CycleCount)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "cgra_simulation_model_invalid: unsupported metric request");
    metrics.push_back(MetricResult{
        UncertaintyKind::ExactWithinModel,
        PointObservation{IntegerValue(static_cast<std::int64_t>(cycleCount))},
        {}});
  }
  return EvaluationModelResult{
      {{kExecutionOutputSlot, {std::move(*reference)}}},
      CompletedEvidence{std::move(metrics), {}}};
}

llvm::Expected<EvaluationModelResult>
evaluate(const EvaluationRequest &request,
         const CaseArtifactResolution &resolution,
         const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  const auto programs = request.subjectBindings().subjects(kProgramRole);
  const auto hardware = request.subjectBindings().subjects(kHardwareRole);
  const auto mappings = request.subjectBindings().subjects(kSpatialMappingRole);
  if (programs.size() != 1 || hardware.size() != 1 || mappings.size() != 1 ||
      !request.workload() || !request.runtimeInput())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "cgra_simulation_model_invalid: Request inputs are not total");
  auto execution = sim::prepareCgraExecution(programs.front(), hardware.front(),
                                             mappings.front(), artifactStore);
  if (!execution)
    return classifyExecutionFailure(execution.takeError());
  auto inputs = sim::importSpatialSimulationInputs(
      *request.workload(), *request.runtimeInput(), artifactStore);
  if (!inputs)
    return classifyExecutionFailure(inputs.takeError());
  return evaluateWithPrepared(request, resolution, *execution, inputs->workload,
                              inputs->runtimeInput, {}, artifactStore,
                              blobStore);
}

const EvaluationModelProvider kProvider{
    kModelDescriptor.reference(), EvaluationModelInProcessProvider{&evaluate}};

} // namespace

llvm::Error registerCgraSimulationModel() {
  if (llvm::Error error = registerEvaluationCaseSignature(kCaseSignature))
    return error;
  if (llvm::Error error = registerEvaluationModelDescriptor(kModelDescriptor))
    return error;
  return registerEvaluationModelProvider(kProvider);
}

EvaluationModelDescriptorRef cgraSimulationModelDescriptorRef() {
  return kModelDescriptor.reference();
}

CaseSubjectRoleRef cgraSimulationProgramRole() { return kProgramRole; }

CaseSubjectRoleRef cgraSimulationHardwareRole() { return kHardwareRole; }

CaseSubjectRoleRef cgraSimulationSpatialMappingRole() {
  return kSpatialMappingRole;
}

llvm::Expected<ResolvedCgraSimulationCase>
resolveCgraSimulationCase(const ArtifactRootReference &spatialMapping,
                          const ArtifactRootReference &workload,
                          const ArtifactRootReference &runtimeInput,
                          const ArtifactStore &artifactStore) {
  if (llvm::Error error = registerCgraSimulationModel())
    return std::move(error);
  auto importedMapping =
      mapping::importSpatialMapping(spatialMapping, artifactStore);
  if (!importedMapping)
    return importedMapping.takeError();
  const ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      importedMapping->view().dataflowIdentity()};
  const ArtifactRootReference fabricReference{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version,
      importedMapping->view().fabricIdentity()};
  const ArtifactRootReference techMappingReference{
      mapping::mappingArtifactSchema.identity.str(),
      mapping::mappingArtifactSchema.version,
      importedMapping->view().techMappingIdentity()};
  auto inputs =
      sim::importSpatialSimulationInputs(workload, runtimeInput, artifactStore);
  if (!inputs)
    return inputs.takeError();
  if (inputs->dataflow.identity() != dataflowReference.artifact)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "cgra_simulation_model_invalid: workload names a foreign Dataflow "
        "owner");
  auto resolution = buildResolution({dataflowReference, fabricReference,
                                     techMappingReference, spatialMapping},
                                    workload, runtimeInput);
  if (!resolution)
    return resolution.takeError();
  return ResolvedCgraSimulationCase{dataflowReference, fabricReference,
                                    std::move(*resolution)};
}

llvm::Expected<PreparedCgraSimulationEvaluation>
prepareCgraSimulationEvaluation(const ArtifactRootReference &canonicalDataflow,
                                const ArtifactRootReference &fabric,
                                const ArtifactRootReference &spatialMapping,
                                const ArtifactRootReference &workload,
                                const ArtifactRootReference &runtimeInput,
                                const ResolvedConfig &config,
                                const ArtifactStore &artifactStore,
                                const BlobStore &blobStore) {
  if (llvm::Error error = registerCgraSimulationModel())
    return std::move(error);
  auto execution = sim::prepareCgraExecution(canonicalDataflow, fabric,
                                             spatialMapping, artifactStore);
  if (!execution)
    return execution.takeError();
  auto owners = execution->ownerReferences();
  if (!owners)
    return owners.takeError();
  auto inputs =
      sim::importSpatialSimulationInputs(workload, runtimeInput, artifactStore);
  if (!inputs)
    return inputs.takeError();
  if (inputs->dataflow.identity() != canonicalDataflow.artifact)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "cgra_simulation_model_invalid: workload names a foreign Dataflow "
        "owner");
  auto resolution = buildResolution(*owners, workload, runtimeInput);
  if (!resolution)
    return resolution.takeError();
  auto bindings =
      EvaluationSubjectBindings::get({{kProgramRole, {canonicalDataflow}},
                                      {kHardwareRole, {fabric}},
                                      {kSpatialMappingRole, {spatialMapping}}});
  if (!bindings)
    return bindings.takeError();
  auto evaluationCase = EvaluationCase::get(
      caseSignatureRef(), std::move(*bindings), workload, runtimeInput, {},
      *resolution, artifactStore, blobStore);
  if (!evaluationCase)
    return evaluationCase.takeError();
  auto modelBinding =
      ResolvedModelBinding::project(kModelDescriptor.reference(), {}, config);
  if (!modelBinding)
    return modelBinding.takeError();
  auto cycleCount =
      MetricRequest::get(MetricQuery{MetricKind::CycleCount,
                                     EvaluationScope{kWholeExactCaseScope, {}}},
                         {}, *evaluationCase, *resolution, artifactStore);
  if (!cycleCount)
    return cycleCount.takeError();
  auto request = EvaluationRequest::get(*evaluationCase, {*cycleCount}, {},
                                        std::move(*modelBinding), 0,
                                        *resolution, artifactStore, blobStore);
  if (!request)
    return request.takeError();
  auto published = publishEvaluationRequest(*request, artifactStore);
  if (!published)
    return published.takeError();
  return PreparedCgraSimulationEvaluation{
      std::move(*request), std::move(*resolution), std::move(*execution),
      std::move(inputs->workload), std::move(inputs->runtimeInput)};
}

llvm::Expected<EvaluationEvidence>
evaluateCgraSimulation(const PreparedCgraSimulationEvaluation &prepared,
                       CgraSimulationAttemptLimits limits,
                       const ArtifactStore &artifactStore,
                       const BlobStore &blobStore) {
  RequestVerifier verifier(prepared.resolution, artifactStore, blobStore);
  if (llvm::Error error = verifier.verify(prepared.request))
    return std::move(error);
  auto result = evaluateWithPrepared(prepared.request, prepared.resolution,
                                     prepared.execution, prepared.workload,
                                     prepared.runtimeInput, std::move(limits),
                                     artifactStore, blobStore);
  if (!result)
    return result.takeError();
  return EvaluationEvidence::get(prepared.request,
                                 std::move(result->outputBindings),
                                 std::move(result->outcome),
                                 prepared.resolution, artifactStore, blobStore);
}

} // namespace loom::evaluation::models
