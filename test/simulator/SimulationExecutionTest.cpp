#include "Simulator/SimulationExecution.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Evaluation/ModelDescriptor.h"
#include "Evaluation/Request.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace loom;
using namespace loom::evaluation;
using namespace loom::sim;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << "\n";
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectErrorContains(llvm::StringRef test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(llvm::StringRef test) {
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-simulation-execution", path_))
      fail(test, error.message());
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << "temporary directory cleanup failed: " << error.message()
                   << "\n";
  }

  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

mlir::MLIRContext &context() {
  static mlir::MLIRContext *instance = [] {
    mlir::DialectRegistry registry;
    registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *instance;
}

dataflow::CanonicalDataflowArtifact makeProgram(llvm::StringRef test) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @add(%ctrl: none, %lhs: i32, %rhs: i32,
                              %memory: memref<4xi8>) -> i32
      attributes {
        input_segments = array<i32: 2, 0, 1>,
        result_segments = array<i32: 1, 0, 0>
      } {
    %sum = arith.addi %lhs, %rhs : i32
    %index = arith.constant 1 : index
    %stored = arith.constant 42 : i8
    %store_done = dataflow.store %memory[%index] %stored %ctrl : memref<4xi8>
    %published:2 = dataflow.sync %ctrl, %sum
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%published#0, %store_done : none, none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)
      (%memory: memref<4xi8>)
      ctrl (%ctrl: none) {
    %lhs = arith.constant 7 : i32
    %rhs = arith.constant 9 : i32
    %value, %done = dataflow.graph.launch @add deps(%ctrl)
        values(%lhs, %rhs) stream_inputs() memories(%memory) stream_outputs()
        : (none, i32, i32, memref<4xi8>) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host(%memory: memref<4xi8>) {
    %thread = dataflow.thread.launch @worker(%memory)
        : (memref<4xi8>) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail(test, "failed to parse Dataflow fixture");
  return take(test, dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::RootedGraphLaunchRef
onlyLaunch(llvm::StringRef test,
           const dataflow::CanonicalDataflowProgramView &view) {
  require(test,
          view.rootThreadLaunches().size() == 1 &&
              view.staticGraphLaunches().size() == 1,
          "fixture must contain one rooted graph launch");
  return {view.rootThreadLaunches().front().ref,
          view.staticGraphLaunches().front().ref};
}

CanonicalValueSequence value(std::uint32_t bits) {
  return {1, {SemanticLane::defined(llvm::APInt(32, bits))}};
}

MemoryObservationPayload unchangedMemory() {
  return DiffMemoryObservation{4, {}};
}

MemoryObservationPayload changedMemory() {
  return DiffMemoryObservation{
      4,
      {{1, {{SemanticState::Defined, 42}}}},
  };
}

SpatialFunctionalObservations observations(ValueResultObservation result,
                                           bool changed) {
  return {
      {std::move(result)}, {}, {changed ? changedMemory() : unchangedMemory()}};
}

constexpr EvaluationCaseKind caseKind{701};
constexpr EvaluationModelKind modelKind{702};
constexpr EvaluationModelKind retiredOnlyModelKind{703};
constexpr FindingKind terminalFindingKind{704};
constexpr CaseSubjectRoleRef candidateRole{0};

EvaluationCaseSignatureRef signatureRef() {
  return llvm::cantFail(
      EvaluationCaseSignatureRef::get(evaluationSchemaVersion(), caseKind));
}

const ArtifactSchemaDescriptor *const candidateSchemas[] = {
    &dataflow::canonicalDataflowSchema};
const ArtifactSchemaDescriptor *const workloadSchemas[] = {
    &simulationWorkloadSchema};
const ArtifactSchemaDescriptor *const runtimeSchemas[] = {
    &simulationRuntimeInputSchema};
const CaseSubjectRoleDescriptor roles[] = {{candidateRole, "canonical_dataflow",
                                            SubjectRoleCardinality::ExactlyOne,
                                            candidateSchemas, nullptr}};

llvm::Error verifyWorkload(const EvaluationCase &,
                           const EvaluationSubjectBindings &bindings,
                           const std::optional<ArtifactRootReference> &workload,
                           const std::optional<ArtifactRootReference> &runtime,
                           const CaseArtifactResolution &resolution,
                           const ArtifactStore &, const BlobStore &) {
  if (!workload || !runtime)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "simulation test workload is not total");
  const CaseArtifactResolution::Entry *workloadEntry =
      resolution.find(*workload);
  const CaseArtifactResolution::Entry *runtimeEntry = resolution.find(*runtime);
  if (!workloadEntry || !runtimeEntry ||
      !CaseArtifactResolution::reaches(*runtimeEntry, *workload))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "runtime input does not reach workload");
  const auto subjects = bindings.subjects(candidateRole);
  if (subjects.size() != 1 ||
      !CaseArtifactResolution::reaches(*workloadEntry, subjects.front()))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "workload does not reach Dataflow owner");
  return llvm::Error::success();
}

const EvaluationCaseSignatureDescriptor signatureDescriptor{
    caseKind,
    "simulation_execution_test_case",
    "One exact Canonical Dataflow workload execution.",
    roles,
    ArtifactRequirement::Required,
    workloadSchemas,
    ArtifactRequirement::Required,
    runtimeSchemas,
    &verifyWorkload,
    AbstractCaseCycle{},
    {}};

struct TestTerminalWitness {
  std::uint32_t marker = 0;
};

llvm::Expected<std::vector<std::uint8_t>>
encodeTerminalWitness(const OwnerValue &value) {
  const auto *witness = value.getIf<TestTerminalWitness>();
  if (!witness)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "wrong terminal witness type");
  return std::vector<std::uint8_t>{
      static_cast<std::uint8_t>(witness->marker >> 24),
      static_cast<std::uint8_t>(witness->marker >> 16),
      static_cast<std::uint8_t>(witness->marker >> 8),
      static_cast<std::uint8_t>(witness->marker)};
}

llvm::Expected<OwnerValue>
decodeTerminalWitness(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 4)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "wrong terminal witness size");
  const std::uint32_t marker = (static_cast<std::uint32_t>(bytes[0]) << 24) |
                               (static_cast<std::uint32_t>(bytes[1]) << 16) |
                               (static_cast<std::uint32_t>(bytes[2]) << 8) |
                               static_cast<std::uint32_t>(bytes[3]);
  return OwnerValue::get(TestTerminalWitness{marker});
}

llvm::Error validateTerminalWitness(const OwnerValue &value,
                                    const FindingTerminalWitnessContext &) {
  const auto *witness = value.getIf<TestTerminalWitness>();
  if (!witness || witness->marker != 0x2a)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid terminal witness marker");
  return llvm::Error::success();
}

const ScopeFormDescriptor terminalScopeForms[] = {
    {ScopeFormRef(0),
     "the exact simulator test case",
     {},
     WholeExactCaseScope{},
     nullptr}};

const FindingDescriptor terminalFindingDescriptor{
    terminalFindingKind,
    "simulation_execution_test_halt",
    "A typed halt used to verify SimulationExecution witness ownership.",
    terminalScopeForms,
    {},
    terminalWitnessRefOccurrenceCodec(),
    FindingTerminalWitnessCodec{{"loom.test.simulation_execution.halt", {1, 0}},
                                &encodeTerminalWitness,
                                &decodeTerminalWitness,
                                &validateTerminalWitness}};

struct EmptyConfigView {};

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.test.simulation_execution.config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptyConfigView{});
}

llvm::Expected<std::vector<std::uint8_t>> encodeConfig(const OwnerValue &view) {
  if (!view.getIf<EmptyConfigView>())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "wrong config view type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue> adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                       const ComponentViewDigest &) {
  if (!bytes.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "config view must be empty");
  return OwnerValue::get(EmptyConfigView{});
}

const ModelOutputSlotDescriptor outputSlots[] = {
    {ModelOutputSlotRef(0),
     "simulation_execution",
     &simulationExecutionSchema,
     {ArtifactCollectionCardinality::ExactlyOne,
      ArtifactCollectionCardinality::Forbidden,
      ArtifactCollectionCardinality::Forbidden,
      ArtifactCollectionCardinality::ZeroOrOne}}};
const ModeledPhenomenon phenomena[] = {ModeledPhenomenon::CanonicalDataflow};
const ScopeFormRef wholeCaseScopeForms[] = {ScopeFormRef(0)};
const FindingCapability findingCapabilities[] = {
    {terminalFindingKind, wholeCaseScopeForms, allFindingResultFormsMask()}};
const FindingQuery mandatoryTerminalFindings[] = {
    {terminalFindingKind, EvaluationScope{ScopeFormRef(0), {}}}};
const EvaluationModelDescriptor modelDescriptor{
    modelKind,
    "simulation_execution_test_model",
    "loom.test.simulation_execution.model",
    signatureRef(),
    {},
    {},
    findingCapabilities,
    {},
    outputSlots,
    {configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig},
    phenomena,
    EvaluationExecutionMethod::Simulation,
    {},
    DeterminismContract::Deterministic,
    mandatoryTerminalFindings,
    ProviderForm::InProcess};

const ModelOutputSlotDescriptor retiredOnlyOutputSlots[] = {
    {ModelOutputSlotRef(0),
     "simulation_execution",
     &simulationExecutionSchema,
     {ArtifactCollectionCardinality::ExactlyOne,
      ArtifactCollectionCardinality::Forbidden,
      ArtifactCollectionCardinality::Forbidden,
      ArtifactCollectionCardinality::Forbidden}}};
const EvaluationModelDescriptor retiredOnlyModelDescriptor{
    retiredOnlyModelKind,
    "simulation_execution_retired_only_test_model",
    "loom.test.simulation_execution.retired_only_model",
    signatureRef(),
    {},
    {},
    findingCapabilities,
    {},
    retiredOnlyOutputSlots,
    {configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig},
    phenomena,
    EvaluationExecutionMethod::Simulation,
    {},
    DeterminismContract::Deterministic,
    mandatoryTerminalFindings,
    ProviderForm::InProcess};

struct Inputs {
  ArtifactRootReference dataflowRef;
  ArtifactRootReference workloadRef;
  ArtifactRootReference runtimeRef;
  ArtifactRootReference requestRef;
  dataflow::ActorRef actor;
  CaseArtifactResolution resolution;
};

Inputs
prepareInputs(llvm::StringRef test, const ArtifactStore &store,
              const BlobStore &blobs,
              const EvaluationModelDescriptor &descriptor = modelDescriptor) {
  llvm::cantFail(registerFindingDescriptor(terminalFindingDescriptor));
  llvm::cantFail(registerEvaluationCaseSignature(signatureDescriptor));
  llvm::cantFail(registerEvaluationModelDescriptor(descriptor));

  auto program = makeProgram(test);
  auto view = take(test, program.view());
  const ArtifactRootReference dataflowRef =
      take(test, dataflow::publishCanonicalDataflow(program, store));

  SpatialSimulationWorkload workloadDraft{onlyLaunch(test, view)};
  workloadDraft.valueInputPlan = {RuntimeValueInput{}, RuntimeValueInput{}};
  workloadDraft.observableContract.valueResults = {0};
  std::optional<dataflow::LogicalMemoryRootRef> memoryRoot;
  for (const dataflow::CanonicalLogicalMemoryRootView &root :
       view.logicalMemoryRoots()) {
    if (root.formalArgIndex && *root.formalArgIndex == 0) {
      memoryRoot = root.ref;
      break;
    }
  }
  require(test, memoryRoot.has_value(), "fixture must expose its memory root");
  workloadDraft.observableContract.memories = {
      {dataflow::LogicalMemoryRootOrViewRef{*memoryRoot},
       MemoryObservationForm::DiffFromRuntimeInput}};
  auto workload = take(test, finalizeSimulationWorkload(workloadDraft, view));
  SpatialSimulationRuntimeInputDraft runtimeDraft{workload.identity()};
  runtimeDraft.runtimeValues = {{0, value(7)}, {1, value(9)}};
  runtimeDraft.memoryObjects = {
      RuntimeMemoryObject{std::vector<SemanticMemoryByte>(
          4, SemanticMemoryByte{SemanticState::Defined, 0})}};
  runtimeDraft.memoryRootBindings = {{*memoryRoot, 0, 0}};
  auto runtime =
      take(test, finalizeSimulationRuntimeInput(runtimeDraft, workload, view));
  const ArtifactRootReference workloadRef =
      take(test, publishSimulationWorkload(workload, store));
  const ArtifactRootReference runtimeRef =
      take(test, publishSimulationRuntimeInput(runtime, store));

  CaseArtifactResolution resolution = take(
      test,
      CaseArtifactResolution::get({{dataflowRef, {}},
                                   {workloadRef, {dataflowRef}},
                                   {runtimeRef, {dataflowRef, workloadRef}}}));
  EvaluationSubjectBindings bindings = take(
      test, EvaluationSubjectBindings::get({{candidateRole, {dataflowRef}}}));
  EvaluationCase evaluationCase =
      take(test,
           EvaluationCase::get(signatureRef(), std::move(bindings), workloadRef,
                               runtimeRef, {}, resolution, store, blobs));
  ResolvedModelBinding modelBinding =
      take(test, ResolvedModelBinding::project(descriptor.reference(), {},
                                               defaultResolvedConfig()));
  FindingRequest finding = take(
      test,
      FindingRequest::get(FindingQuery{terminalFindingKind,
                                       EvaluationScope{ScopeFormRef(0), {}}},
                          {}, evaluationCase, resolution, store));
  EvaluationRequest request =
      take(test, EvaluationRequest::get(evaluationCase, {}, {finding},
                                        std::move(modelBinding), 0, resolution,
                                        store, blobs));
  const ArtifactRootReference requestRef =
      take(test, publishEvaluationRequest(request, store));
  require(test, !view.actors().empty(), "fixture must contain an actor");
  return {dataflowRef,
          workloadRef,
          runtimeRef,
          requestRef,
          view.actors().front().ref,
          std::move(resolution)};
}

evaluation::ExactRatio ratio(llvm::StringRef test, std::uint64_t numerator,
                             std::uint64_t denominator = 1) {
  return take(test, evaluation::ExactRatio::get(numerator, denominator));
}

SpatialProgressObservations retiredProgress(llvm::StringRef test) {
  return {SpatialEventCoordinate{ratio(test, 0), 0},
          SpatialEventCoordinate{ratio(test, 7, 2), 1},
          SpatialEventCoordinate{ratio(test, 4), 0}};
}

void retiredExecutionRoundTripsThroughItsExactRequest() {
  TemporaryDirectory directory(__func__);
  const ArtifactStore store(directory.path());
  const BlobStore blobs(directory.path());
  Inputs inputs = prepareInputs(__func__, store, blobs);

  SpatialSimulationExecution draft{
      inputs.requestRef,
      RetiredExecution{},
      observations(PublishedValueResult{value(16)}, true),
      retiredProgress(__func__),
      {}};
  auto finalized =
      take(__func__,
           finalizeSimulationExecution(draft, inputs.resolution, store, blobs));
  const ArtifactRootReference published =
      take(__func__, publishSimulationExecution(finalized, store));
  auto imported =
      take(__func__, importSimulationExecution(published, inputs.resolution,
                                               store, blobs));

  require(__func__, imported.request() == inputs.requestRef,
          "execution changed its exact Request reference");
  require(__func__,
          std::holds_alternative<RetiredExecution>(imported.terminal()),
          "execution changed its terminal");
  require(__func__,
          imported.canonicalBytes().bytes() ==
              finalized.canonicalBytes().bytes(),
          "strict import changed canonical execution bytes");
}

void terminalControlsRequiredCompletionFacts() {
  TemporaryDirectory directory(__func__);
  const ArtifactStore store(directory.path());
  const BlobStore blobs(directory.path());
  Inputs inputs = prepareInputs(__func__, store, blobs);
  const SpatialProgressObservations incomplete{
      SpatialEventCoordinate{ratio(__func__, 0), 0}, std::nullopt,
      SpatialEventCoordinate{ratio(__func__, 3), 0}};

  SpatialSimulationExecution retired{
      inputs.requestRef,
      RetiredExecution{},
      observations(NotPublishedValueResult{}, false),
      incomplete,
      {}};
  expectErrorContains(
      __func__,
      finalizeSimulationExecution(retired, inputs.resolution, store, blobs),
      "Retired execution requires graph retirement");

  SpatialSimulationExecution stopped{
      inputs.requestRef,
      StoppedByLimitExecution{},
      observations(NotPublishedValueResult{}, false),
      incomplete,
      {}};
  auto finalized =
      take(__func__, finalizeSimulationExecution(stopped, inputs.resolution,
                                                 store, blobs));
  require(__func__,
          std::holds_alternative<StoppedByLimitExecution>(finalized.terminal()),
          "stopped execution changed terminal");

  stopped.terminal = HaltedExecution{
      terminalFindingKind, OwnerValue::get(TestTerminalWitness{0x2a})};
  auto halted = take(__func__, finalizeSimulationExecution(
                                   stopped, inputs.resolution, store, blobs));
  const ArtifactRootReference executionRef =
      take(__func__, publishSimulationExecution(halted, store));
  auto imported =
      take(__func__, importSimulationExecution(executionRef, inputs.resolution,
                                               store, blobs));
  require(__func__,
          std::holds_alternative<HaltedExecution>(imported.terminal()),
          "Halted execution did not round-trip");

  EvaluationRequest request =
      take(__func__, importEvaluationRequest(inputs.requestRef,
                                             inputs.resolution, store, blobs));
  FindingResult present{PresentFinding{
      {FindingOccurrence::get(TerminalWitnessRef{ModelOutputSlotRef(0), 0})}}};
  auto evidence = EvaluationEvidence::get(
      request, {{ModelOutputSlotRef(0), {executionRef}}},
      CompletedEvidence{{}, {std::move(present)}}, inputs.resolution, store,
      blobs);
  take(__func__, std::move(evidence));

  FindingResult invalidPresent{PresentFinding{
      {FindingOccurrence::get(TerminalWitnessRef{ModelOutputSlotRef(0), 1})}}};
  expectErrorContains(__func__,
                      EvaluationEvidence::get(
                          request, {{ModelOutputSlotRef(0), {executionRef}}},
                          CompletedEvidence{{}, {std::move(invalidPresent)}},
                          inputs.resolution, store, blobs),
                      "does not resolve an execution");
}

void actorActivityUsesTheRootedGraphInventory() {
  TemporaryDirectory directory(__func__);
  const ArtifactStore store(directory.path());
  const BlobStore blobs(directory.path());
  Inputs inputs = prepareInputs(__func__, store, blobs);

  SpatialSimulationExecution draft{
      inputs.requestRef,
      RetiredExecution{},
      observations(PublishedValueResult{value(16)}, true),
      retiredProgress(__func__),
      {{ActivityWindow::LaunchToTerminal,
        ActivityCoverage::Partial,
        {{inputs.actor, {3, 3}}}}}};
  auto finalized =
      take(__func__,
           finalizeSimulationExecution(draft, inputs.resolution, store, blobs));
  require(__func__, finalized.spatialActivitySummaries().size() == 1,
          "execution dropped its typed actor activity summary");

  draft.activitySummaries.front().transitions.front().counts = {2, 3};
  expectErrorContains(
      __func__,
      finalizeSimulationExecution(draft, inputs.resolution, store, blobs),
      "retired count exceeds committed count");
}

void memoryDiffHasOneCanonicalRunPartition() {
  TemporaryDirectory directory(__func__);
  const ArtifactStore store(directory.path());
  const BlobStore blobs(directory.path());
  Inputs inputs = prepareInputs(__func__, store, blobs);

  SpatialFunctionalObservations functional =
      observations(PublishedValueResult{value(16)}, true);
  functional.memories = {DiffMemoryObservation{
      4,
      {{0, {{SemanticState::Defined, 1}}}, {1, {{SemanticState::Defined, 2}}}},
  }};
  SpatialSimulationExecution draft{inputs.requestRef,
                                   RetiredExecution{},
                                   std::move(functional),
                                   retiredProgress(__func__),
                                   {}};
  expectErrorContains(
      __func__,
      finalizeSimulationExecution(draft, inputs.resolution, store, blobs),
      "overlap or are adjacent");
}

void modelControlsStoppedExecutionRetention() {
  TemporaryDirectory directory(__func__);
  const ArtifactStore store(directory.path());
  const BlobStore blobs(directory.path());
  Inputs inputs =
      prepareInputs(__func__, store, blobs, retiredOnlyModelDescriptor);

  SpatialSimulationExecution execution{
      inputs.requestRef,
      RetiredExecution{},
      observations(PublishedValueResult{value(16)}, true),
      retiredProgress(__func__),
      {}};
  take(__func__,
       finalizeSimulationExecution(execution, inputs.resolution, store, blobs));

  execution.terminal = StoppedByLimitExecution{};
  execution.progressObservations.graphRetirementVisible.reset();
  expectErrorContains(
      __func__,
      finalizeSimulationExecution(execution, inputs.resolution, store, blobs),
      "does not retain StoppedByLimit execution");
}

} // namespace

int main() {
  retiredExecutionRoundTripsThroughItsExactRequest();
  terminalControlsRequiredCompletionFacts();
  actorActivityUsesTheRootedGraphInventory();
  memoryDiffHasOneCanonicalRunPartition();
  modelControlsStoppedExecutionRetention();
  return EXIT_SUCCESS;
}
