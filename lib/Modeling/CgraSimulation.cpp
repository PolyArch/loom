#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/ProductionRegistry.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/InvocationDiagnosticLog.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/ModelProvider.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Simulator/CGRASimulator.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <time.h>

#include <chrono>
#include <limits>
#include <optional>
#include <string>
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

using MonotonicClock = std::chrono::steady_clock;

std::uint64_t elapsedNanoseconds(MonotonicClock::time_point begin) {
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      MonotonicClock::now() - begin);
  return elapsed.count() <= 0 ? 0 : static_cast<std::uint64_t>(elapsed.count());
}

std::optional<std::uint64_t> processCpuNanoseconds() {
  timespec current{};
  if (::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &current) != 0 ||
      current.tv_sec < 0 || current.tv_nsec < 0 ||
      current.tv_nsec >= 1'000'000'000)
    return std::nullopt;
  constexpr std::uint64_t nanosecondsPerSecond = 1'000'000'000;
  const std::uint64_t seconds = current.tv_sec;
  if (seconds > (std::numeric_limits<std::uint64_t>::max() -
                 static_cast<std::uint64_t>(current.tv_nsec)) /
                    nanosecondsPerSecond)
    return std::nullopt;
  return seconds * nanosecondsPerSecond + current.tv_nsec;
}

std::optional<std::uint64_t>
elapsedProcessCpuNanoseconds(std::optional<std::uint64_t> begin) {
  const std::optional<std::uint64_t> end = processCpuNanoseconds();
  if (!begin || !end || *end < *begin)
    return std::nullopt;
  return *end - *begin;
}

struct AttemptIntervalStart final {
  MonotonicClock::time_point wall;
  std::optional<std::uint64_t> processCpuNanoseconds;
};

std::optional<AttemptIntervalStart> beginAttemptInterval(bool enabled) {
  if (!enabled)
    return std::nullopt;
  return AttemptIntervalStart{MonotonicClock::now(), processCpuNanoseconds()};
}

void finishAttemptInterval(
    const std::optional<AttemptIntervalStart> &begin,
    std::uint64_t &wallNanoseconds,
    std::optional<std::uint64_t> &processCpuNanoseconds) {
  if (!begin)
    return;
  wallNanoseconds = elapsedNanoseconds(begin->wall);
  processCpuNanoseconds =
      elapsedProcessCpuNanoseconds(begin->processCpuNanoseconds);
}

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

llvm::Expected<EvaluationModelResult> classifyExecutionFailure(
    llvm::Error error,
    std::optional<sim::CgraUnsupportedMemoryContract> *unsupportedMemory =
        nullptr) {
  if (unsupportedMemory)
    unsupportedMemory->reset();
  std::error_code code;
  std::string diagnostic;
  if (error.isA<sim::CgraExecutionUnsupported>()) {
    llvm::handleAllErrors(std::move(error),
                          [&](const sim::CgraExecutionUnsupported &info) {
                            code = info.convertToErrorCode();
                            llvm::raw_string_ostream stream(diagnostic);
                            info.log(stream);
                            if (unsupportedMemory)
                              *unsupportedMemory = info.memoryContract();
                          });
  } else {
    llvm::handleAllErrors(std::move(error),
                          [&](const llvm::ErrorInfoBase &info) {
                            code = info.convertToErrorCode();
                            llvm::raw_string_ostream stream(diagnostic);
                            info.log(stream);
                          });
  }
  emitInvocationDiagnostic(DiagnosticVerbosity::Summary,
                           InvocationDiagnosticStage::SystemPnr,
                           InvocationDiagnosticEvent::MappingFailure, [&] {
                             return llvm::json::Value(llvm::json::Object{
                                 {"failure_scope", "cgra_simulation_adapter"},
                                 {"diagnostic", diagnostic},
                             });
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
    const BlobStore &blobStore,
    const sim::PreparedCgraWorkloadExecution *workloadExecution = nullptr,
    std::optional<sim::CgraClosedWaitSetDiagnostic> *closedWait = nullptr,
    std::optional<sim::CgraUnsupportedMemoryContract> *unsupportedMemory =
        nullptr,
    CgraSimulationAttemptProfile *attemptProfile = nullptr) {
  if (closedWait)
    closedWait->reset();
  if (unsupportedMemory)
    unsupportedMemory->reset();
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
  const auto setupBegin = beginAttemptInterval(attemptProfile != nullptr);
  RuntimeInputCgraMemoryProvider externalMemory(*spatialInput);
  if (attemptProfile)
    finishAttemptInterval(setupBegin,
                          attemptProfile->attemptSetupWallNanoseconds,
                          attemptProfile->attemptSetupProcessCpuNanoseconds);
  const auto engineBegin = beginAttemptInterval(attemptProfile != nullptr);
  auto outcome =
      workloadExecution
          ? sim::simulateCgraWorkload(
                *workloadExecution, workload, runtimeInput,
                limits.maxEventFrames, limits.executionDeadline,
                &externalMemory)
          : sim::simulateCgraWorkload(execution, workload, runtimeInput,
                                      limits.maxEventFrames,
                                      limits.executionDeadline,
                                      &externalMemory);
  if (attemptProfile) {
    finishAttemptInterval(engineBegin,
                          attemptProfile->engineActiveWallNanoseconds,
                          attemptProfile->engineActiveProcessCpuNanoseconds);
    if (outcome)
      attemptProfile->counters = outcome->counters;
  }
  const auto projectionBegin = beginAttemptInterval(attemptProfile != nullptr);
  if (!outcome)
    return classifyExecutionFailure(outcome.takeError(), unsupportedMemory);
  if (outcome->state == sim::SpatialExecutionSessionState::StoppedByLimit)
    return EvaluationModelResult{
        {{kExecutionOutputSlot, {}}},
        CancelledOrTimeoutEvidence{OutcomeReason::ExecutionLimitReached}};
  if (outcome->state != sim::SpatialExecutionSessionState::Retired ||
      !outcome->retired) {
    if (closedWait && outcome->closedWaitSet)
      *closedWait = *outcome->closedWaitSet;
    emitInvocationDiagnostic(
        DiagnosticVerbosity::Summary, InvocationDiagnosticStage::SystemPnr,
        InvocationDiagnosticEvent::MappingFailure, [&] {
          llvm::json::Object fields;
          fields["failure_scope"] = "cgra_simulation_session";
          fields["session_state"] = static_cast<std::uint64_t>(outcome->state);
          if (outcome->closedWaitSet) {
            fields["pending_actor_firings"] =
                outcome->closedWaitSet->pendingActorFirings;
            fields["pending_transfers"] =
                outcome->closedWaitSet->pendingTransfers;
            fields["pending_physical_actions"] =
                outcome->closedWaitSet->pendingPhysicalActions;
            fields["graph_retirement_visible"] =
                outcome->closedWaitSet->graphRetirementVisible;
            if (outcome->closedWaitSet->ownerReferences) {
              const auto &owners = *outcome->closedWaitSet->ownerReferences;
              fields["closed_wait_owners"] = llvm::json::Object{
                  {"dataflow",
                   formatArtifactRootReferenceJson(owners.dataflow)},
                  {"fabric", formatArtifactRootReferenceJson(owners.fabric)},
                  {"tech_mapping",
                   formatArtifactRootReferenceJson(owners.techMapping)},
                  {"spatial_mapping",
                   formatArtifactRootReferenceJson(owners.spatialMapping)}};
            }
            fields["closed_wait_actor_count"] =
                outcome->closedWaitSet->actorFirings.size();
            fields["closed_wait_transfer_count"] =
                outcome->closedWaitSet->transfers.size();
            fields["closed_wait_transfer_cycle_edge_count"] =
                outcome->closedWaitSet->transferWaitCycle.size();
            fields["closed_wait_actor_cycle_edge_count"] =
                outcome->closedWaitSet->actorWaitCycle.size();
            fields["operand_queue_group_count"] =
                outcome->closedWaitSet->operandQueueGroupCount;
            fields["operand_queue_potentially_blocking_group_count"] =
                outcome->closedWaitSet->
                    operandQueuePotentiallyBlockingGroupCount;
            fields["operand_queue_shared_ingress_pressure"] =
                outcome->closedWaitSet->operandQueueSharedIngressPressure;
            fields["operand_queue_distinct_ingress_count"] =
                outcome->closedWaitSet->operandQueueDistinctIngressCount;
            fields["operand_queue_pairing_key_count"] =
                outcome->closedWaitSet->operandQueuePairingKeyCount;
            fields["operand_queue_progress_status"] =
                outcome->closedWaitSet->operandQueueProgressStatus;
            fields["operand_queue_progress_support"] =
                outcome->closedWaitSet->operandQueueProgressSupport;
            if (outcome->closedWaitSet->operandQueueProjectionDigest)
              fields["operand_queue_projection_digest"] =
                  formatComponentViewDigestHex(
                      *outcome->closedWaitSet->operandQueueProjectionDigest);
            else
              fields["operand_queue_projection_digest"] = nullptr;
            llvm::json::Array operandQueueHeads;
            for (const auto indexed : llvm::enumerate(
                     outcome->closedWaitSet->operandQueueHeads)) {
              if (indexed.index() == 16)
                break;
              const auto &head = indexed.value();
              llvm::json::Array consumers;
              for (const auto &[actor, input] : head.consumers)
                consumers.push_back(llvm::json::Object{
                    {"actor", actor}, {"input", input}});
              llvm::SmallString<32> tagSpelling;
              head.headTag.toStringUnsigned(tagSpelling, 16);
              operandQueueHeads.push_back(llvm::json::Object{
                  {"queue_context",
                   llvm::toHex(::loom::fabric::canonicalFabricBytes(
                       head.queue.context), true)},
                  {"queue_fu_occurrence", head.queue.fuOccurrence},
                  {"queue_fu_input", head.queue.fuInput},
                  {"fu",
                   llvm::toHex(::loom::fabric::canonicalFabricBytes(head.fu),
                               true)},
                  {"allocation_unit", head.allocationUnit},
                  {"capacity", head.capacity},
                  {"occupancy", head.occupancy},
                  {"reservations", head.reservations},
                  {"head_binding", head.headBindingOrdinal},
                  {"head_occurrence", head.headOccurrenceOrdinal},
                  {"head_producer_sequence",
                   head.headProducerSequenceOrdinal},
                  {"head_tag", tagSpelling.str().str()},
                  {"exact_head", head.exactHead},
                  {"consumers", std::move(consumers)}});
            }
            fields["closed_wait_operand_queue_heads"] =
                std::move(operandQueueHeads);
            llvm::json::Array actors;
            for (const auto indexed :
                 llvm::enumerate(outcome->closedWaitSet->actorFirings)) {
              if (indexed.index() == 4)
                break;
              const auto &actor = indexed.value();
              actors.push_back(llvm::json::Object{
                  {"actor", actor.semanticActorOrdinal},
                  {"occurrence", actor.occurrenceOrdinal},
                  {"expected_transfers", actor.expectedTransfers},
                  {"completed_transfers", actor.completedTransfers},
                  {"physical_complete", actor.physicalComplete},
              });
            }
            fields["closed_wait_actors"] = std::move(actors);
            llvm::json::Array blockedActorInputs;
            for (const auto &input :
                 outcome->closedWaitSet->blockedActorInputs)
              blockedActorInputs.push_back(llvm::json::Object{
                  {"actor", input.semanticActorOrdinal},
                  {"actor_entity", input.actorEntityId},
                  {"input", input.inputOrdinal},
                  {"channel", input.channelOrdinal},
                  {"source_kind", static_cast<std::uint64_t>(input.sourceKind)},
                  {"defining_actor", input.definingActorOrdinal},
                  {"defining_actor_entity", input.definingActorEntityId},
                  {"defining_actor_terminal", input.definingActorTerminal}});
            fields["closed_wait_blocked_actor_inputs"] =
                std::move(blockedActorInputs);
            const auto storageHeadJson = [](const auto &head) {
              if (!head)
                return llvm::json::Value(nullptr);
              return llvm::json::Value(llvm::json::Object{
                  {"storage", head->storageOrdinal},
                  {"binding", head->bindingOrdinal},
                  {"occurrence", head->occurrenceOrdinal},
                  {"traversal_node", head->traversalNodeOrdinal}});
            };
            llvm::json::Array transfers;
            for (const auto indexed :
                 llvm::enumerate(outcome->closedWaitSet->transfers)) {
              if (indexed.index() == 4)
                break;
              const auto &transfer = indexed.value();
              llvm::json::Array operandQueueWaits;
              for (const auto &wait : transfer.operandQueueWaits) {
                llvm::SmallString<32> tag;
                wait.tag.toStringUnsigned(tag, 16);
                operandQueueWaits.push_back(llvm::json::Object{
                    {"context",
                     llvm::toHex(::loom::fabric::canonicalFabricBytes(
                         wait.queue.context), true)},
                    {"fu",
                     llvm::toHex(::loom::fabric::canonicalFabricBytes(wait.fu),
                                 true)},
                    {"ingress",
                     llvm::toHex(::loom::fabric::canonicalFabricBytes(
                         wait.ingress), true)},
                    {"fu_input", wait.queue.fuInput},
                    {"tag", tag.str().str()},
                    {"allocation_unit", wait.allocationUnit},
                    {"occupancy", wait.occupancy},
                    {"reservations", wait.reservations},
                    {"capacity", wait.capacity}});
              }
              transfers.push_back(llvm::json::Object{
                  {"binding", transfer.bindingOrdinal},
                  {"occurrence", transfer.occurrenceOrdinal},
                  {"producer_actor", transfer.producerActorOrdinal},
                  {"producer_result", transfer.producerResultOrdinal},
                  {"blocked", transfer.blocked},
                  {"published", transfer.published},
                  {"ready_sinks", transfer.readySinkCount},
                  {"published_sinks", transfer.publishedSinkCount},
                  {"sink_count", transfer.sinkCount},
                  {"blocking_actor", transfer.blockingActorOrdinal},
                  {"blocking_ready_tokens", transfer.blockingReadyTokenCount},
                  {"blocking_queue_occupancy", transfer.blockingQueueOccupancy},
                  {"blocking_queue_reservations",
                   transfer.blockingQueueReservations},
                  {"blocking_queue_capacity", transfer.blockingQueueCapacity},
                  {"blocking_storage", transfer.blockingStorageOrdinal},
                  {"blocking_fifo",
                   transfer.blockingFifoOccurrence
                       ? llvm::json::Value(llvm::toHex(
                             ::loom::fabric::canonicalFabricBytes(
                                 *transfer.blockingFifoOccurrence),
                             true))
                       : llvm::json::Value(nullptr)},
                  {"blocking_storage_occupancy",
                   transfer.blockingStorageOccupancy},
                  {"blocking_storage_reservations",
                   transfer.blockingStorageReservations},
                  {"blocking_storage_capacity",
                   transfer.blockingStorageCapacity},
                  {"blocking_storage_head",
                   storageHeadJson(transfer.blockingStorageHead)},
                  {"blocking_downstream_storage_count",
                   transfer.blockingDownstreamStorageCount},
                  {"blocking_unbuffered_sink_count",
                   transfer.blockingUnbufferedSinkCount},
                  {"blocking_downstream_storage",
                   transfer.blockingDownstreamStorageOrdinal},
                  {"blocking_downstream_occupancy",
                   transfer.blockingDownstreamStorageOccupancy},
                  {"blocking_downstream_capacity",
                   transfer.blockingDownstreamStorageCapacity},
                  {"blocking_downstream_reserved",
                   transfer.blockingDownstreamStorageReserved},
                  {"blocking_downstream_head",
                   storageHeadJson(transfer.blockingDownstreamStorageHead)},
                  {"operand_queue_waits", std::move(operandQueueWaits)},
              });
            }
            fields["closed_wait_transfers"] = std::move(transfers);
            llvm::json::Array transferCycle;
            for (const auto &edge :
                 outcome->closedWaitSet->transferWaitCycle)
              transferCycle.push_back(llvm::json::Object{
                  {"waiting_binding", edge.waitingBindingOrdinal},
                  {"waiting_occurrence", edge.waitingOccurrenceOrdinal},
                  {"blocking_actor", edge.blockingActorOrdinal},
                  {"blocking_binding", edge.blockingBindingOrdinal},
                  {"blocking_occurrence", edge.blockingOccurrenceOrdinal},
                  {"kind", static_cast<std::uint64_t>(edge.kind)},
              });
            fields["closed_wait_transfer_cycle"] = std::move(transferCycle);
            llvm::json::Array actorCycle;
            for (const auto &edge : outcome->closedWaitSet->actorWaitCycle)
              actorCycle.push_back(llvm::json::Object{
                  {"waiting_actor", edge.waitingActorOrdinal},
                  {"blocking_actor", edge.blockingActorOrdinal},
                  {"kind", static_cast<std::uint64_t>(edge.kind)},
              });
            fields["closed_wait_actor_cycle"] = std::move(actorCycle);
            llvm::json::Array physicalActions;
            for (const auto indexed :
                 llvm::enumerate(outcome->closedWaitSet->physicalActions)) {
              if (indexed.index() == 4)
                break;
              const auto &action = indexed.value();
              physicalActions.push_back(llvm::json::Object{
                  {"action", action.actionOrdinal},
                  {"occurrence", action.occurrenceOrdinal},
                  {"client", action.clientKind},
                  {"semantic_actor",
                   action.semanticActorOrdinal
                       ? llvm::json::Value(*action.semanticActorOrdinal)
                       : llvm::json::Value(nullptr)},
                  {"granted", action.granted},
                  {"has_commit", action.hasCommit},
                  {"requires_causal_release",
                   action.requiresCausalRelease},
                  {"intrinsic_release_reached",
                   action.intrinsicReleaseReached},
                  {"causal_release_reached", action.causalReleaseReached},
              });
            }
            fields["closed_wait_physical_actions"] =
                std::move(physicalActions);
          }
          return llvm::json::Value(std::move(fields));
        });
    return EvaluationModelResult{
        {{kExecutionOutputSlot, {}}},
        ExecutionFailedEvidence{OutcomeReason::AdapterFailure}};
  }
  const auto &progress = outcome->retired->progress;
  const auto &retirement = progress.graphRetirementVisible;
  if (!retirement || retirement->referenceCycle.denominator() != 1 ||
      retirement->referenceCycle.numerator() >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return EvaluationModelResult{
        {{kExecutionOutputSlot, {}}},
        ExecutionFailedEvidence{OutcomeReason::AdapterFailure}};
  const std::uint64_t cycleCount = retirement->referenceCycle.numerator();
  const sim::CgraSimulationCounters &counters = outcome->retired->counters;
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::SystemPnr,
      InvocationDiagnosticEvent::Statistics, [&] {
        llvm::json::Object fields;
        llvm::json::Object direct;
        direct["cycle_count"] = cycleCount;
        direct["launch_reference_cycle_numerator"] =
            progress.launchAccepted.referenceCycle.numerator();
        direct["graph_retirement_reference_cycle_numerator"] =
            retirement->referenceCycle.numerator();
        direct["terminal_reference_cycle_numerator"] =
            progress.terminalObserved.referenceCycle.numerator();
        direct["terminal_event_delta"] = progress.terminalObserved.delta;
        direct["event_frame_count"] = counters.eventFrameCount;
        direct["empty_event_frame_count"] = counters.emptyEventFrameCount;
        direct["compute_source_frame_count"] = counters.computeSourceFrameCount;
        direct["memory_source_frame_count"] = counters.memorySourceFrameCount;
        direct["transport_source_frame_count"] =
            counters.transportSourceFrameCount;
        direct["physical_source_frame_count"] =
            counters.physicalSourceFrameCount;
        direct["maximum_reference_cycle_numerator"] =
            counters.maximumReferenceCycleNumerator;
        direct["maximum_event_delta"] = counters.maximumEventDelta;
        direct["physical_grant_wait_cycle_sum"] =
            counters.physicalGrantWaitCycleSum;
        direct["physical_grant_wait_cycle_max"] =
            counters.physicalGrantWaitCycleMax;
        direct["physical_action_lifetime_cycle_sum"] =
            counters.physicalActionLifetimeCycleSum;
        direct["physical_action_lifetime_cycle_max"] =
            counters.physicalActionLifetimeCycleMax;
        direct["physical_granted_lifetime_cycle_sum"] =
            counters.physicalGrantedLifetimeCycleSum;
        direct["physical_granted_lifetime_cycle_max"] =
            counters.physicalGrantedLifetimeCycleMax;
        direct["physical_grant_same_cycle_count"] =
            counters.physicalGrantSameCycleCount;
        direct["physical_grant_delayed_count"] =
            counters.physicalGrantDelayedCount;
        direct["non_integral_timing_observation_count"] =
            counters.nonIntegralTimingObservationCount;
        direct["actor_commit_count"] = counters.actorCommitCount;
        direct["actor_firing_count"] = counters.actorCommitCount;
        direct["actor_retirement_count"] = counters.actorRetirementCount;
        direct["token_publication_count"] = counters.tokenPublicationCount;
        direct["memory_linearization_count"] =
            counters.memoryLinearizationCount;
        direct["physical_request_count"] = counters.physicalRequestCount;
        direct["physical_grant_count"] = counters.physicalGrantCount;
        direct["physical_retirement_count"] =
            counters.physicalRetirementCount;
        direct["request_grant_gap"] =
            counters.physicalRequestCount >= counters.physicalGrantCount
                ? counters.physicalRequestCount - counters.physicalGrantCount
                : 0;
        direct["grant_retirement_gap"] =
            counters.physicalGrantCount >= counters.physicalRetirementCount
                ? counters.physicalGrantCount -
                      counters.physicalRetirementCount
                : 0;
        const sim::CgraExecutionPlanSummary plan = execution.summary();
        llvm::json::Object staticPlan{
            {"mapped_graph_count", plan.mappedGraphCount},
            {"compute_actor_count", plan.computeActorCount},
            {"actor_transition_count", plan.actorTransitionCount},
            {"semantic_configuration_field_count",
             plan.semanticConfigurationFieldCount},
            {"memory_actor_count", plan.memoryActorCount},
            {"memory_rooted_use_count", plan.memoryRootedUseCount},
            {"memory_child_transaction_count", plan.memoryChildTransactionCount},
            {"memory_result_assembly_count", plan.memoryResultAssemblyCount},
            {"compute_transition_physical_use_count",
             plan.computeTransitionPhysicalUseCount},
            {"memory_transition_physical_use_count",
             plan.memoryTransitionPhysicalUseCount},
            {"produced_physical_use_count", plan.producedPhysicalUseCount},
            {"consumed_physical_use_count", plan.consumedPhysicalUseCount},
            {"traversal_physical_use_count", plan.traversalPhysicalUseCount},
            {"physical_use_count", plan.physicalUseCount},
            {"resource_owner_count", plan.resourceOwnerCount},
            {"claim_count", plan.claimCount},
            {"route_tree_count", plan.routeTreeCount},
            {"route_node_count", plan.routeNodeCount},
            {"route_sink_count", plan.routeSinkCount},
            {"selected_traversal_count", plan.selectedTraversalCount},
            {"local_transfer_count", plan.localTransferCount},
            {"local_transfer_sink_count", plan.localTransferSinkCount},
            {"physical_tag_segment_count", plan.physicalTagSegmentCount},
            {"tagged_route_node_count", plan.taggedRouteNodeCount}};
        staticPlan["physical_use_acquire_rank_sum"] =
            plan.physicalUseAcquireRankSum;
        staticPlan["physical_use_release_rank_sum"] =
            plan.physicalUseReleaseRankSum;
        staticPlan["physical_use_max_acquire_rank"] =
            plan.physicalUseMaxAcquireRank;
        staticPlan["physical_use_max_release_rank"] =
            plan.physicalUseMaxReleaseRank;
        staticPlan["physical_use_causal_release_count"] =
            plan.physicalUseCausalReleaseCount;
        staticPlan["compute_transition_timing_count"] =
            plan.computeTransitionTimingCount;
        staticPlan["memory_transition_timing_count"] =
            plan.memoryTransitionTimingCount;
        staticPlan["produced_transport_timing_count"] =
            plan.producedTransportTimingCount;
        staticPlan["consumed_transport_timing_count"] =
            plan.consumedTransportTimingCount;
        staticPlan["traversal_transport_timing_count"] =
            plan.traversalTransportTimingCount;
        staticPlan["compute_transition_max_release_rank"] =
            plan.computeTransitionMaxReleaseRank;
        staticPlan["memory_transition_max_release_rank"] =
            plan.memoryTransitionMaxReleaseRank;
        staticPlan["produced_transport_max_release_rank"] =
            plan.producedTransportMaxReleaseRank;
        staticPlan["consumed_transport_max_release_rank"] =
            plan.consumedTransportMaxReleaseRank;
        staticPlan["traversal_transport_max_release_rank"] =
            plan.traversalTransportMaxReleaseRank;
        staticPlan["maximum_route_node_depth"] =
            plan.maximumRouteNodeDepth;
        staticPlan["temporal_compute_actor_count"] =
            plan.temporalComputeActorCount;
        staticPlan["spatial_compute_actor_count"] =
            plan.spatialComputeActorCount;
        staticPlan["temporal_dispatch_domain_count"] =
            plan.temporalDispatchDomainCount;
        staticPlan["operand_buffer_count"] = plan.operandBufferCount;
        direct["static_plan"] = std::move(staticPlan);
        llvm::json::Object derived;
        const auto rate = [](std::uint64_t numerator,
                             std::uint64_t denominator) -> llvm::json::Value {
          if (denominator == 0)
            return llvm::json::Value(nullptr);
          return llvm::json::Object{{"numerator", numerator},
                                    {"denominator", denominator}};
        };
        derived["actor_ipc"] =
            rate(counters.actorCommitCount, cycleCount);
        derived["actor_cpi"] =
            rate(cycleCount, counters.actorCommitCount);
        derived["physical_action_rate"] =
            rate(counters.physicalRetirementCount, cycleCount);
        derived["cycles_per_physical_action"] =
            rate(cycleCount, counters.physicalRetirementCount);
        derived["cycles_per_actor_retirement"] =
            rate(cycleCount, counters.actorRetirementCount);
        derived["event_frames_per_cycle"] =
            rate(counters.eventFrameCount, cycleCount);
        derived["transport_frames_per_cycle"] =
            rate(counters.transportSourceFrameCount, cycleCount);
        derived["physical_frames_per_cycle"] =
            rate(counters.physicalSourceFrameCount, cycleCount);
        if (progress.terminalObserved.referenceCycle.denominator() == 1 &&
            progress.terminalObserved.referenceCycle.numerator() >=
                retirement->referenceCycle.numerator())
          derived["post_retirement_drain_cycles"] = llvm::json::Object{
              {"numerator",
               progress.terminalObserved.referenceCycle.numerator() -
                   retirement->referenceCycle.numerator()},
              {"denominator", 1}};
        else
          derived["post_retirement_drain_cycles"] =
              llvm::json::Value(nullptr);
        derived["memory_load_store_split"] = "unsupported_by_cgra_counter";
        derived["recurrence_or_ii"] = "unsupported_single_activation";
        fields["measurement_kind"] = "direct_and_derived";
        fields["direct"] = std::move(direct);
        fields["derived"] = std::move(derived);
        fields["operation"] = "simulation_cycle_breakdown";
        fields["engine"] = "cgra";
        fields["request"] = formatArtifactRootReferenceJson(
            evaluationRequestReference(request));
        fields["cycle_count"] = cycleCount;
        fields["event_frame_count"] = counters.eventFrameCount;
        fields["empty_event_frame_count"] = counters.emptyEventFrameCount;
        fields["compute_source_frame_count"] = counters.computeSourceFrameCount;
        fields["memory_source_frame_count"] = counters.memorySourceFrameCount;
        fields["transport_source_frame_count"] =
            counters.transportSourceFrameCount;
        fields["physical_source_frame_count"] =
            counters.physicalSourceFrameCount;
        fields["actor_commit_count"] = counters.actorCommitCount;
        fields["actor_retirement_count"] = counters.actorRetirementCount;
        fields["token_publication_count"] = counters.tokenPublicationCount;
        fields["memory_linearization_count"] =
            counters.memoryLinearizationCount;
        fields["physical_request_count"] = counters.physicalRequestCount;
        fields["physical_grant_count"] = counters.physicalGrantCount;
        fields["physical_retirement_count"] =
            counters.physicalRetirementCount;
        fields["request_grant_gap"] =
            counters.physicalRequestCount >= counters.physicalGrantCount
                ? counters.physicalRequestCount - counters.physicalGrantCount
                : 0;
        fields["grant_retirement_gap"] =
            counters.physicalGrantCount >= counters.physicalRetirementCount
                ? counters.physicalGrantCount -
                      counters.physicalRetirementCount
                : 0;
        return llvm::json::Value(std::move(fields));
      });

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
  if (attemptProfile) {
    finishAttemptInterval(
        projectionBegin, attemptProfile->observationProjectionWallNanoseconds,
        attemptProfile->observationProjectionProcessCpuNanoseconds);
  }
  const auto publicationBegin = beginAttemptInterval(attemptProfile != nullptr);
  auto reference = sim::publishSimulationExecution(*finalized, artifactStore);
  if (!reference)
    return reference.takeError();
  if (attemptProfile) {
    finishAttemptInterval(
        publicationBegin, attemptProfile->artifactPublicationWallNanoseconds,
        attemptProfile->artifactPublicationProcessCpuNanoseconds);
  }
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
                              blobStore, nullptr);
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
  auto workloadExecution = sim::prepareCgraWorkloadExecution(
      *execution, inputs->workload, inputs->runtimeInput);
  if (!workloadExecution)
    return workloadExecution.takeError();
  return PreparedCgraSimulationEvaluation{
      std::move(*request), std::move(*resolution), std::move(*execution),
      std::move(inputs->workload), std::move(inputs->runtimeInput),
      std::move(*workloadExecution)};
}

llvm::Expected<EvaluationEvidence>
evaluateCgraSimulation(const PreparedCgraSimulationEvaluation &prepared,
                       CgraSimulationAttemptLimits limits,
                       const ArtifactStore &artifactStore,
                       const BlobStore &blobStore) {
  auto evaluated = evaluateCgraSimulationWithDiagnostics(
      prepared, std::move(limits), artifactStore, blobStore);
  if (!evaluated)
    return evaluated.takeError();
  return std::move(evaluated->evidence);
}

namespace {

llvm::Expected<CgraSimulationEvaluation>
evaluateCgraSimulationWithDiagnosticsImpl(
    const PreparedCgraSimulationEvaluation &prepared,
    CgraSimulationAttemptLimits limits, const ArtifactStore &artifactStore,
    const BlobStore &blobStore, bool collectAttemptProfile) {
  RequestVerifier verifier(prepared.resolution, artifactStore, blobStore);
  if (llvm::Error error = verifier.verify(prepared.request))
    return std::move(error);
  std::optional<sim::CgraClosedWaitSetDiagnostic> closedWait;
  std::optional<sim::CgraUnsupportedMemoryContract> unsupportedMemory;
  std::optional<CgraSimulationAttemptProfile> attemptProfile;
  if (collectAttemptProfile)
    attemptProfile.emplace();
  auto result = evaluateWithPrepared(
      prepared.request, prepared.resolution, prepared.execution,
      prepared.workload, prepared.runtimeInput, std::move(limits),
      artifactStore, blobStore, &prepared.workloadExecution, &closedWait,
      &unsupportedMemory, attemptProfile ? &*attemptProfile : nullptr);
  if (!result)
    return result.takeError();
  auto evidence = EvaluationEvidence::get(
      prepared.request, std::move(result->outputBindings),
      std::move(result->outcome), prepared.resolution, artifactStore,
      blobStore);
  if (!evidence)
    return evidence.takeError();
  return CgraSimulationEvaluation{std::move(*evidence), std::move(closedWait),
                                  unsupportedMemory, std::move(attemptProfile)};
}

} // namespace

llvm::Expected<CgraSimulationEvaluation> evaluateCgraSimulationWithDiagnostics(
    const PreparedCgraSimulationEvaluation &prepared,
    CgraSimulationAttemptLimits limits, const ArtifactStore &artifactStore,
    const BlobStore &blobStore) {
  return evaluateCgraSimulationWithDiagnosticsImpl(
      prepared, std::move(limits), artifactStore, blobStore, false);
}

llvm::Expected<CgraSimulationEvaluation>
evaluateCgraSimulationWithAttemptProfile(
    const PreparedCgraSimulationEvaluation &prepared,
    CgraSimulationAttemptLimits limits, const ArtifactStore &artifactStore,
    const BlobStore &blobStore) {
  return evaluateCgraSimulationWithDiagnosticsImpl(
      prepared, std::move(limits), artifactStore, blobStore, true);
}

} // namespace loom::evaluation::models
