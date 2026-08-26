#include "DeploymentTestSupport.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Evaluation/Finding.h"
#include "Evaluation/ModelDescriptor.h"
#include "Evaluation/Request.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace loom;
using namespace loom::deployment;
using namespace loom::evaluation;
using namespace loom::sim;

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    deployment::test::fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::StringRef test, llvm::Error error) {
  if (error)
    deployment::test::fail(test, llvm::toString(std::move(error)));
}

template <typename T>
void expectErrorContains(llvm::StringRef test, llvm::Expected<T> value,
                         llvm::StringRef marker) {
  if (value)
    deployment::test::fail(test, "accepted a noncanonical System artifact");
  const std::string message = llvm::toString(value.takeError());
  deployment::test::require(test, llvm::StringRef(message).contains(marker),
                            message);
}

CanonicalValueSequence scalar(unsigned width, std::uint64_t value) {
  return {1, {SemanticLane::defined(llvm::APInt(width, value))}};
}

CanonicalStreamSequence stream8(std::initializer_list<std::uint8_t> values,
                                StreamTermination termination) {
  CanonicalValueSequence sequence;
  sequence.tokenCount = values.size();
  for (std::uint8_t value : values)
    sequence.lanes.push_back(SemanticLane::defined(llvm::APInt(8, value)));
  return {std::move(sequence), termination};
}

SemanticMemoryByte byte(std::uint8_t value) {
  return {SemanticState::Defined, value};
}

constexpr EvaluationCaseKind systemCaseKind{705};
constexpr EvaluationModelKind systemModelKind{706};
constexpr FindingKind haltFindingKind{707};
constexpr CaseSubjectRoleRef deploymentRole{0};

EvaluationCaseSignatureRef systemSignatureRef() {
  return llvm::cantFail(EvaluationCaseSignatureRef::get(
      evaluationSchemaVersion(), systemCaseKind));
}

const ArtifactSchemaDescriptor *const deploymentSchemas[] = {&deploymentSchema};
const ArtifactSchemaDescriptor *const workloadSchemas[] = {
    &simulationWorkloadSchema};
const ArtifactSchemaDescriptor *const runtimeSchemas[] = {
    &simulationRuntimeInputSchema};
const CaseSubjectRoleDescriptor systemRoles[] = {
    {deploymentRole, "deployment", SubjectRoleCardinality::ExactlyOne,
     deploymentSchemas, nullptr}};

llvm::Error
verifySystemInputs(const EvaluationCase &,
                   const EvaluationSubjectBindings &bindings,
                   const std::optional<ArtifactRootReference> &workload,
                   const std::optional<ArtifactRootReference> &runtimeInput,
                   const CaseArtifactResolution &resolution,
                   const ArtifactStore &store, const BlobStore &blobs) {
  if (!workload || !runtimeInput)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "System simulation inputs are not total");
  const auto subjects = bindings.subjects(deploymentRole);
  if (subjects.size() != 1)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "System simulation has no Deployment");
  const auto *workloadEntry = resolution.find(*workload);
  const auto *runtimeEntry = resolution.find(*runtimeInput);
  if (!workloadEntry || !runtimeEntry ||
      !CaseArtifactResolution::reaches(*workloadEntry, subjects.front()) ||
      !CaseArtifactResolution::reaches(*runtimeEntry, *workload))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "System simulation closure is incomplete");
  auto imported =
      importSystemSimulationInputs(*workload, *runtimeInput, store, blobs);
  if (!imported)
    return imported.takeError();
  if (imported->deployment.reference() != subjects.front())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "System simulation names another Deployment");
  return llvm::Error::success();
}

const EvaluationCaseSignatureDescriptor systemSignature{
    systemCaseKind,
    "system_simulation_artifact_test_case",
    "One exact Deployment-owned System simulation.",
    systemRoles,
    ArtifactRequirement::Required,
    workloadSchemas,
    ArtifactRequirement::Required,
    runtimeSchemas,
    &verifySystemInputs,
    AbstractCaseCycle{},
    {}};

struct EmptyConfigView {};

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral bytes =
      "loom.test.system_simulation.config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptyConfigView{});
}

llvm::Expected<std::vector<std::uint8_t>>
encodeConfig(const OwnerValue &value) {
  if (!value.getIf<EmptyConfigView>())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "wrong System simulation config view");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue> adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                       const ComponentViewDigest &) {
  if (!bytes.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "System simulation config is not empty");
  return OwnerValue::get(EmptyConfigView{});
}

const ModelOutputSlotDescriptor executionOutputs[] = {{
    ModelOutputSlotRef(0),
    "simulation_execution",
    &simulationExecutionSchema,
    {ArtifactCollectionCardinality::ExactlyOne,
     ArtifactCollectionCardinality::Forbidden,
     ArtifactCollectionCardinality::Forbidden,
     ArtifactCollectionCardinality::ZeroOrOne},
}};
const ModeledPhenomenon systemPhenomena[] = {
    ModeledPhenomenon::StructuredProgram,
    ModeledPhenomenon::SystemMemoryHierarchy};
const EvaluationModelDescriptor systemModel{
    systemModelKind,
    "system_simulation_artifact_test_model",
    "loom.test.system_simulation.model",
    systemSignatureRef(),
    {},
    {},
    {},
    {},
    executionOutputs,
    {configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig},
    systemPhenomena,
    EvaluationExecutionMethod::Simulation,
    {},
    DeterminismContract::Deterministic,
    {},
    ProviderForm::InProcess};

struct HaltWitness {
  std::uint8_t marker = 0;
};

llvm::Expected<std::vector<std::uint8_t>>
encodeHaltWitness(const OwnerValue &value) {
  const auto *witness = value.getIf<HaltWitness>();
  if (!witness)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "wrong System halt witness");
  return std::vector<std::uint8_t>{witness->marker};
}

llvm::Expected<OwnerValue>
decodeHaltWitness(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 1)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "wrong System halt witness size");
  return OwnerValue::get(HaltWitness{bytes.front()});
}

llvm::Error validateHaltWitness(const OwnerValue &value,
                                const FindingTerminalWitnessContext &) {
  const auto *witness = value.getIf<HaltWitness>();
  if (!witness || witness->marker != 0xa5)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid System halt witness");
  return llvm::Error::success();
}

const ScopeFormDescriptor haltScopes[] = {{ScopeFormRef(0),
                                           "the exact System simulation case",
                                           {},
                                           WholeExactCaseScope{},
                                           nullptr}};
const FindingDescriptor haltFinding{
    haltFindingKind,
    "system_simulation_artifact_halt",
    "A typed System execution halt used by the artifact contract anchor.",
    haltScopes,
    {},
    terminalWitnessRefOccurrenceCodec(),
    FindingTerminalWitnessCodec{{"loom.test.system_simulation.halt", {1, 0}},
                                &encodeHaltWitness,
                                &decodeHaltWitness,
                                &validateHaltWitness}};

struct Fixture {
  FinalizedDeployment deployment;
  CanonicalSimulationWorkload workload;
  CanonicalSimulationRuntimeInput runtimeInput;
  ArtifactRootReference workloadRef;
  ArtifactRootReference runtimeRef;
  ArtifactRootReference requestRef;
  CaseArtifactResolution resolution;
};

Fixture buildFixture(llvm::StringRef test, ArtifactStore &artifacts,
                     BlobStore &blobs,
                     const deployment::test::TemporaryTree &tree) {
  FinalizedDeployment deployment =
      deployment::test::buildSystemArtifactDeployment(test, artifacts, blobs,
                                                      tree);
  const ArtifactIdentity identity = deployment.reference().artifact;
  const auto interfaceRef = [&](std::uint64_t ordinal) {
    return DeploymentExternalInterfaceRef{identity, ordinal};
  };

  SystemSimulationWorkload workloadDraft{{identity, 0}};
  workloadDraft.valueInputPlan = {RuntimeValueInput{}};
  workloadDraft.externalValueInputPlan = {
      {interfaceRef(0), RuntimeValueInput{}}};
  workloadDraft.observableContract.valueResults = {0};
  workloadDraft.observableContract.externalValueOutputs = {interfaceRef(3)};
  workloadDraft.observableContract.externalStreamOutputs = {interfaceRef(4)};
  workloadDraft.observableContract.memories = {
      {interfaceRef(2), MemoryObservationForm::FullState},
      {interfaceRef(5), MemoryObservationForm::DiffFromRuntimeInput},
      {interfaceRef(6), MemoryObservationForm::FullState}};
  CanonicalSimulationWorkload workload = take(
      test, finalizeSimulationWorkload(workloadDraft, deployment, artifacts));

  std::vector<SemanticMemoryByte> objectBytes(16, byte(0));
  RuntimeMemoryObject separateObject(objectBytes);
  RuntimeMemoryObject aliasedObject(
      objectBytes,
      {RuntimeMemoryPointer{0, 0, PointerTarget{0, llvm::APInt(64, 3)}}});
  SystemSimulationRuntimeInputDraft runtimeDraft{workload.identity()};
  runtimeDraft.runtimeEntryValues = {{0, scalar(32, 7)}};
  runtimeDraft.runtimeExternalValues = {
      {interfaceRef(0),
       {1,
        {SemanticLane::definedPointer(llvm::APInt(64, 0), 1,
                                      llvm::APInt(64, 2))}}}};
  runtimeDraft.externalStreamInputs = {
      {interfaceRef(1),
       stream8({1, 2, 3}, StreamTermination::ClosedAfterLast)}};
  runtimeDraft.memoryObjects = {std::move(separateObject),
                                std::move(aliasedObject)};
  runtimeDraft.memoryInterfaceBindings = {{interfaceRef(6), 0, 0},
                                          {interfaceRef(5), 1, 4},
                                          {interfaceRef(2), 1, 0}};
  CanonicalSimulationRuntimeInput runtimeInput =
      take(test, finalizeSimulationRuntimeInput(runtimeDraft, workload,
                                                deployment, artifacts));

  const ArtifactRootReference workloadRef =
      take(test, publishSimulationWorkload(workload, artifacts));
  const ArtifactRootReference runtimeRef =
      take(test, publishSimulationRuntimeInput(runtimeInput, artifacts));
  CaseArtifactResolution resolution =
      take(test, CaseArtifactResolution::get(
                     {{deployment.reference(), {}},
                      {workloadRef, {deployment.reference()}},
                      {runtimeRef, {deployment.reference(), workloadRef}}}));
  EvaluationSubjectBindings subjects =
      take(test, EvaluationSubjectBindings::get(
                     {{deploymentRole, {deployment.reference()}}}));
  EvaluationCase evaluationCase =
      take(test, EvaluationCase::get(systemSignatureRef(), std::move(subjects),
                                     workloadRef, runtimeRef, {}, resolution,
                                     artifacts, blobs));
  ResolvedModelBinding binding =
      take(test, ResolvedModelBinding::project(systemModel.reference(), {},
                                               defaultResolvedConfig()));
  EvaluationRequest request = take(
      test, EvaluationRequest::get(evaluationCase, {}, {}, std::move(binding),
                                   0, resolution, artifacts, blobs));
  const ArtifactRootReference requestRef =
      take(test, publishEvaluationRequest(request, artifacts));
  return {std::move(deployment), std::move(workload), std::move(runtimeInput),
          workloadRef,           runtimeRef,          requestRef,
          std::move(resolution)};
}

SystemFunctionalObservations observations(const Fixture &fixture) {
  const auto &runtime = *fixture.runtimeInput.system();
  std::vector<SemanticMemoryByte> aliased =
      runtime.memoryObjects[0].initialBytes;
  aliased[12] = byte(9);
  return {{PublishedValueResult{scalar(32, 11)}},
          {PublishedValueResult{scalar(32, 12)}},
          {stream8({4, 5}, StreamTermination::ClosedAfterLast)},
          {FullMemoryObservation{aliased},
           DiffMemoryObservation{12, {{8, {byte(9)}}}},
           FullMemoryObservation{runtime.memoryObjects[1].initialBytes}}};
}

dataflow::RootThreadLaunchRef mappedRoot(llvm::StringRef test,
                                         const Fixture &fixture,
                                         const ArtifactStore &artifacts) {
  loom::mapping::FinalizedSystemMapping systemMapping = take(
      test, loom::mapping::importSystemMapping(
                fixture.deployment.deployment().systemMapping(), artifacts));
  const auto roots =
      systemMapping.view().executionBindings().rootThreadLaunches();
  deployment::test::require(test, !roots.empty(),
                            "System Mapping has no root thread launch");
  return roots.front();
}

void systemArtifactsRoundTripWithAliasAndPointerProvenance() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  Fixture fixture = buildFixture(test, artifacts, blobs, tree);

  const auto *runtime = fixture.runtimeInput.system();
  deployment::test::require(test, runtime != nullptr,
                            "runtime input lost its System form");
  deployment::test::require(
      test,
      runtime->memoryObjects.size() == 2 &&
          runtime->memoryInterfaceBindings[0].binding.objectOrdinal == 0 &&
          runtime->memoryInterfaceBindings[1].binding.objectOrdinal == 0 &&
          runtime->memoryInterfaceBindings[2].binding.objectOrdinal == 1,
      "canonical object remap did not preserve the exact alias relation");
  deployment::test::require(
      test,
      runtime->runtimeExternalValues.front()
                  .value.lanes.front()
                  .pointerTarget->objectOrdinal == 0 &&
          runtime->memoryObjects.front()
                  .pointerValues.front()
                  .target.objectOrdinal == 1,
      "pointer provenance did not follow canonical object ordinals");

  auto imported = take(test, importSystemSimulationInputs(fixture.workloadRef,
                                                          fixture.runtimeRef,
                                                          artifacts, blobs));
  deployment::test::require(
      test,
      imported.deployment.reference() == fixture.deployment.reference() &&
          imported.workload.canonicalBytes().bytes() ==
              fixture.workload.canonicalBytes().bytes() &&
          imported.runtimeInput.canonicalBytes().bytes() ==
              fixture.runtimeInput.canonicalBytes().bytes(),
      "strict System input import changed canonical artifacts");

  const auto workloadBytes = fixture.workload.canonicalBytes().bytes();
  const auto runtimeBytes = fixture.runtimeInput.canonicalBytes().bytes();
  deployment::test::require(
      test,
      workloadBytes.take_front(4) ==
              llvm::ArrayRef<std::uint8_t>({0, 0, 0, 1}) &&
          runtimeBytes.take_front(4) ==
              llvm::ArrayRef<std::uint8_t>({0, 0, 0, 1}),
      "System root discriminant is not the fixed u32be value 1");

  std::vector<std::uint8_t> noncanonical(workloadBytes.begin(),
                                         workloadBytes.end());
  noncanonical.push_back(0);
  expectErrorContains(test,
                      importSimulationWorkload(noncanonical, fixture.deployment,
                                               artifacts,
                                               fixture.workload.identity()),
                      "trailing bytes");
  noncanonical.assign(runtimeBytes.begin(), runtimeBytes.end());
  noncanonical.push_back(0);
  expectErrorContains(test,
                      importSimulationRuntimeInput(
                          noncanonical, fixture.workload, fixture.deployment,
                          artifacts, fixture.runtimeInput.identity()),
                      "trailing bytes");
}

void systemExecutionRetainsTerminalAndTickSemantics() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  Fixture fixture = buildFixture(test, artifacts, blobs, tree);
  const dataflow::RootThreadLaunchRef root =
      mappedRoot(test, fixture, artifacts);
  const dataflow::EventFamilyKey start =
      dataflow::rootThreadStartEventFamily(root);
  const dataflow::EventFamilyKey completion =
      dataflow::rootThreadCompletionEventFamily(root);
  const std::vector<SystemRootLifecycleObservation> validLifecycle{
      {start, 7, {5, 0}},
      {start, 8, {6, 0}},
      {completion, 8, {7, 0}},
      {completion, 7, {8, 0}},
  };

  SystemSimulationExecution execution{
      fixture.requestRef,
      RetiredExecution{},
      observations(fixture),
      {{3, 0}, SystemEventCoordinate{40, 1}, {40, 2}, validLifecycle},
      {}};
  CanonicalSimulationExecution retired =
      take(test, finalizeSimulationExecution(execution, fixture.resolution,
                                             artifacts, blobs));
  const ArtifactRootReference retiredRef =
      take(test, publishSimulationExecution(retired, artifacts));
  CanonicalSimulationExecution imported =
      take(test, importSimulationExecution(retiredRef, fixture.resolution,
                                           artifacts, blobs));
  const SystemSimulationExecution *importedSystem = imported.system();
  deployment::test::require(test, importedSystem != nullptr,
                            "System execution imported as another form");
  const auto &importedLifecycle =
      importedSystem->progressObservations.rootLifecycle;
  deployment::test::require(
      test,
      imported.spatial() == nullptr &&
          std::holds_alternative<RetiredExecution>(imported.terminal()) &&
          importedLifecycle.size() == validLifecycle.size() &&
          importedLifecycle.front().event == start &&
          importedLifecycle.front().occurrence == 7 &&
          importedLifecycle.back().event == completion &&
          importedLifecycle.back().occurrence == 7,
      "System execution imported through the wrong workload-selected form");
  const auto requestPrefix = encodeArtifactRootReference(fixture.requestRef);
  deployment::test::require(
      test,
      retired.canonicalBytes().bytes().take_front(requestPrefix.size()) ==
          llvm::ArrayRef<std::uint8_t>(requestPrefix),
      "System execution serialized a duplicate engine or workload tag");

  SystemSimulationExecution malformed = execution;
  malformed.progressObservations.rootLifecycle.front().occurrence = 0;
  expectErrorContains(test,
                      finalizeSimulationExecution(malformed, fixture.resolution,
                                                  artifacts, blobs),
                      "occurrence is zero");
  malformed = execution;
  malformed.progressObservations.rootLifecycle[1].occurrence = 7;
  expectErrorContains(test,
                      finalizeSimulationExecution(malformed, fixture.resolution,
                                                  artifacts, blobs),
                      "occurrence is reused");
  malformed = execution;
  malformed.progressObservations.rootLifecycle = {{completion, 7, {5, 0}}};
  expectErrorContains(test,
                      finalizeSimulationExecution(malformed, fixture.resolution,
                                                  artifacts, blobs),
                      "completion precedes");
  malformed = execution;
  malformed.progressObservations.rootLifecycle = {
      {start, 7, {5, 0}},
      {completion, 7, {6, 0}},
      {completion, 7, {7, 0}},
  };
  expectErrorContains(test,
                      finalizeSimulationExecution(malformed, fixture.resolution,
                                                  artifacts, blobs),
                      "repeats a completion");
  malformed = execution;
  malformed.progressObservations.rootLifecycle = {{start, 7, {5, 0}}};
  expectErrorContains(test,
                      finalizeSimulationExecution(malformed, fixture.resolution,
                                                  artifacts, blobs),
                      "active root lifecycle occurrence");
  malformed = execution;
  malformed.progressObservations.rootLifecycle[1].coordinate = {5, 0};
  expectErrorContains(test,
                      finalizeSimulationExecution(malformed, fixture.resolution,
                                                  artifacts, blobs),
                      "not strictly increasing");
  malformed = execution;
  const dataflow::RootThreadLaunchRef foreignRoot{
      fixture.deployment.reference().artifact, root.entity};
  malformed.progressObservations.rootLifecycle = {
      {dataflow::rootThreadStartEventFamily(foreignRoot), 7, {5, 0}}};
  expectErrorContains(test,
                      finalizeSimulationExecution(malformed, fixture.resolution,
                                                  artifacts, blobs),
                      "not owned by a root thread launch");

  execution.progressObservations.rootLifecycle.clear();
  take(test, finalizeSimulationExecution(execution, fixture.resolution,
                                         artifacts, blobs));
  execution.progressObservations.rootLifecycle = validLifecycle;

  execution.terminal = StoppedByLimitExecution{};
  execution.progressObservations.programExitVisible.reset();
  CanonicalSimulationExecution stopped =
      take(test, finalizeSimulationExecution(execution, fixture.resolution,
                                             artifacts, blobs));
  deployment::test::require(
      test, std::holds_alternative<StoppedByLimitExecution>(stopped.terminal()),
      "StoppedByLimit terminal was not retained");

  execution.terminal =
      HaltedExecution{haltFindingKind, OwnerValue::get(HaltWitness{0xa5})};
  CanonicalSimulationExecution halted =
      take(test, finalizeSimulationExecution(execution, fixture.resolution,
                                             artifacts, blobs));
  const ArtifactRootReference haltedRef =
      take(test, publishSimulationExecution(halted, artifacts));
  imported = take(test, importSimulationExecution(haltedRef, fixture.resolution,
                                                  artifacts, blobs));
  deployment::test::require(
      test, std::holds_alternative<HaltedExecution>(imported.terminal()),
      "Halted terminal did not preserve its typed witness");

  execution.activitySummaries = {ActorTransitionsActivitySummary{}};
  expectErrorContains(test,
                      finalizeSimulationExecution(execution, fixture.resolution,
                                                  artifacts, blobs),
                      "System activity summary");

  std::vector<std::uint8_t> noncanonical(
      halted.canonicalBytes().bytes().begin(),
      halted.canonicalBytes().bytes().end());
  noncanonical.push_back(0);
  const ArtifactIdentity noncanonicalIdentity = take(
      test, artifacts.put(simulationExecutionSchema,
                          CanonicalSemanticBytes(std::move(noncanonical))));
  expectErrorContains(
      test,
      importSimulationExecution({simulationExecutionSchema.identity.str(),
                                 simulationExecutionSchema.version,
                                 noncanonicalIdentity},
                                fixture.resolution, artifacts, blobs),
      "trailing bytes");
}

} // namespace

int main() {
  requireSuccess("registration", registerFindingDescriptor(haltFinding));
  requireSuccess("registration",
                 registerEvaluationCaseSignature(systemSignature));
  requireSuccess("registration",
                 registerEvaluationModelDescriptor(systemModel));
  systemArtifactsRoundTripWithAliasAndPointerProvenance();
  systemExecutionRetainsTerminalAndTickSemantics();
  return EXIT_SUCCESS;
}
