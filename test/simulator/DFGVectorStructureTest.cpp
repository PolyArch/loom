#include "DFGSimulatorInternal.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <utility>

namespace {

using namespace loom::sim;
using namespace loom::sim::detail;

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "DFGVectorStructureTest: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *instance = [] {
    mlir::DialectRegistry registry;
    registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                    mlir::DLTIDialect, mlir::func::FuncDialect,
                    mlir::LLVM::LLVMDialect, mlir::ub::UBDialect,
                    mlir::vector::VectorDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *instance;
}

dataflow::CanonicalDataflowArtifact structuralProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @structure(
      %start: none,
      %source: vector<2x2xi8>, %rhs: vector<1x2xi8>,
      %inserted: vector<2xi8>, %destination: vector<2x2xi8>,
      %position: index, %axis0: index, %axis1: index)
      -> (vector<2xi8>, vector<2x2xi8>, vector<3x2xi8>, i8)
      attributes {
        input_segments = array<i32: 7, 0, 0>,
        result_segments = array<i32: 4, 0, 0>
      } {
    %extracted = vector.extract %source[%position]
        : vector<2xi8> from vector<2x2xi8>
    %updated = vector.insert %inserted, %destination[%position]
        : vector<2xi8> into vector<2x2xi8>
    %shuffled = vector.shuffle %source, %rhs [0, 2, -1]
        : vector<2x2xi8>, vector<1x2xi8>
    %scalar = vector.extract %source[%axis0, %axis1]
        : i8 from vector<2x2xi8>
    %retired:5 = dataflow.sync %start, %extracted, %updated, %shuffled, %scalar
        : (none, vector<2xi8>, vector<2x2xi8>, vector<3x2xi8>, i8)
          -> (none, vector<2xi8>, vector<2x2xi8>, vector<3x2xi8>, i8)
    dataflow.graph.return
        values(%retired#1, %retired#2, %retired#3, %retired#4
               : vector<2xi8>, vector<2x2xi8>, vector<3x2xi8>, i8)
        streams() memories() complete(%retired#0 : none)
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("failed to parse the structural vector fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

PreparedGraphExecution prepare(dataflow::CanonicalDataflowArtifact &artifact) {
  auto graph = *artifact.module().getOps<dataflow::GraphOp>().begin();
  GraphPreparationResult result =
      take(prepareGraphExecution(artifact.module(), graph));
  if (auto *failure = std::get_if<GraphPreparationFailure>(&result))
    fail(failure->diagnostics.empty() ? "graph preparation failed"
                                      : failure->diagnostics.front());
  return std::move(std::get<PreparedGraphExecution>(result));
}

ActorExecutionPlan &findActor(PreparedGraphExecution &execution,
                              dataflow::OperationSchemaId schema) {
  for (ActorExecutionPlan &actor : execution.actorPlans)
    if (actor.projection.schema == schema)
      return actor;
  fail("prepared graph is missing a structural vector actor");
}

ActorExecutionPlan &findActor(PreparedGraphExecution &execution,
                              dataflow::OperationSchemaId schema,
                              unsigned operandCount) {
  for (ActorExecutionPlan &actor : execution.actorPlans)
    if (actor.projection.schema == schema &&
        actor.operation->getNumOperands() == operandCount)
      return actor;
  fail("prepared graph is missing the requested structural vector actor");
}

Token vectorToken(mlir::VectorType type,
                  std::initializer_list<std::uint8_t> lanes) {
  require(static_cast<std::size_t>(type.getNumElements()) == lanes.size(),
          "test vector lane count does not match its type");
  llvm::APInt bits(static_cast<unsigned>(lanes.size()) * 8, 0);
  unsigned ordinal = 0;
  for (std::uint8_t lane : lanes)
    bits.insertBits(llvm::APInt(8, lane), 8 * ordinal++);
  return take(tokenFromBitPattern(bits, type));
}

void seed(SimulatorState &state, mlir::BlockArgument argument,
          std::initializer_list<Token> tokens) {
  for (const Token &token : tokens)
    seedBlockArgument(state, argument, token);
}

Token fireOnce(ActorExecutionPlan &actor, SimulatorState &state) {
  auto transition = take(probeActorTransition(actor, state));
  require(transition.readiness == ActorTransitionReadiness::Ready &&
              transition.transitionCaseOrdinal == 0,
          "structural actor did not select its all-input transition");
  llvm::SmallVector<ActorResultEmission, 1> emissions;
  state.actorEmissionCapture = &emissions;
  const ActorTransitionCommitOutcome outcome =
      commitActorTransition(actor, state);
  state.actorEmissionCapture = nullptr;
  if (outcome != ActorTransitionCommitOutcome::Committed ||
      emissions.size() != 1 || emissions.front().resultOrdinal != 0) {
    for (const std::string &diagnostic : state.diagnostics)
      llvm::errs() << diagnostic << '\n';
    fail("structural actor did not commit exactly one result");
  }
  return std::move(emissions.front().token);
}

llvm::SmallVector<Token, 2> fireTwice(ActorExecutionPlan &actor,
                                      SimulatorState &state) {
  llvm::SmallVector<Token, 2> results;
  results.push_back(fireOnce(actor, state));
  results.push_back(fireOnce(actor, state));
  return results;
}

Token positionToken(ActorExecutionPlan &actor, std::uint64_t value) {
  auto width = take(loom::getIndexBitWidth(actor.operation));
  return indexToken(llvm::APInt(width, value));
}

void requireLane(const SemanticLane &lane, SemanticState state,
                 std::uint8_t bits = 0) {
  require(lane.state == state, "observed vector lane has the wrong state");
  if (state == SemanticState::Defined)
    require(lane.bits == llvm::APInt(8, bits),
            "observed vector lane has the wrong bits");
}

void exactStructuralActorsFireRepeatedly() {
  auto artifact = structuralProgram();
  PreparedGraphExecution execution = prepare(artifact);
  mlir::Block &entry = execution.graph.getBody().front();
  auto sourceType =
      mlir::cast<mlir::VectorType>(entry.getArgument(1).getType());
  auto rhsType = mlir::cast<mlir::VectorType>(entry.getArgument(2).getType());
  auto insertedType =
      mlir::cast<mlir::VectorType>(entry.getArgument(3).getType());
  auto destinationType =
      mlir::cast<mlir::VectorType>(entry.getArgument(4).getType());

  {
    SimulatorState state;
    initializeRunState(state, execution);
    auto &actor =
        findActor(execution, dataflow::OperationSchemaId::VectorExtract);
    seed(state, entry.getArgument(1),
         {vectorToken(sourceType, {1, 2, 3, 4}),
          vectorToken(sourceType, {11, 12, 13, 14})});
    seed(state, entry.getArgument(5),
         {positionToken(actor, 1), positionToken(actor, 0)});
    auto results = fireTwice(actor, state);
    auto sequence = take(canonicalValueSequenceFromTokens(
        results, actor.operation->getResult(0).getType(), actor.operation));
    require(sequence.tokenCount == 2 && sequence.lanes.size() == 4,
            "extract did not preserve two result occurrences");
    requireLane(sequence.lanes[0], SemanticState::Defined, 3);
    requireLane(sequence.lanes[1], SemanticState::Defined, 4);
    requireLane(sequence.lanes[2], SemanticState::Defined, 11);
    requireLane(sequence.lanes[3], SemanticState::Defined, 12);
  }

  {
    auto &actor =
        findActor(execution, dataflow::OperationSchemaId::VectorExtract, 2);
    for (PrimitiveValueState positionState :
         {PrimitiveValueState::Defined, PrimitiveValueState::Poison}) {
      SimulatorState state;
      initializeRunState(state, execution);
      seed(state, entry.getArgument(1),
           {vectorToken(sourceType, {1, 2, 3, 4})});
      Token position = positionState == PrimitiveValueState::Defined
                           ? positionToken(actor, 2)
                           : take(exceptionalValueToken(
                                 PrimitiveValueState::Poison,
                                 actor.operation->getOperand(1).getType()));
      seed(state, entry.getArgument(5), {position});
      Token result = fireOnce(actor, state);
      auto sequence = take(canonicalValueSequenceFromTokens(
          llvm::ArrayRef(result), actor.operation->getResult(0).getType(),
          actor.operation));
      const SemanticState expected =
          positionState == PrimitiveValueState::Defined ? SemanticState::Undef
                                                        : SemanticState::Poison;
      require(sequence.lanes.size() == 2,
              "exceptional extract position changed result shape");
      requireLane(sequence.lanes[0], expected);
      requireLane(sequence.lanes[1], expected);
    }
  }

  {
    SimulatorState state;
    initializeRunState(state, execution);
    auto &actor =
        findActor(execution, dataflow::OperationSchemaId::VectorExtract, 3);
    seed(state, entry.getArgument(1),
         {vectorToken(sourceType, {1, 2, 3, 4}),
          vectorToken(sourceType, {11, 12, 13, 14})});
    Token poison = take(exceptionalValueToken(
        PrimitiveValueState::Poison, actor.operation->getOperand(1).getType()));
    seed(state, entry.getArgument(6), {positionToken(actor, 2), poison});
    seed(state, entry.getArgument(7), {poison, positionToken(actor, 2)});
    auto results = fireTwice(actor, state);
    auto sequence = take(canonicalValueSequenceFromTokens(
        results, actor.operation->getResult(0).getType(), actor.operation));
    require(sequence.tokenCount == 2 && sequence.lanes.size() == 2,
            "multi-axis extract changed its result cardinality");
    requireLane(sequence.lanes[0], SemanticState::Poison);
    requireLane(sequence.lanes[1], SemanticState::Poison);
  }

  {
    SimulatorState state;
    initializeRunState(state, execution);
    auto &actor =
        findActor(execution, dataflow::OperationSchemaId::VectorInsert);
    seed(state, entry.getArgument(3),
         {vectorToken(insertedType, {9, 10}),
          vectorToken(insertedType, {19, 20})});
    seed(state, entry.getArgument(4),
         {vectorToken(destinationType, {21, 22, 23, 24}),
          vectorToken(destinationType, {29, 30, 31, 32})});
    seed(state, entry.getArgument(5),
         {positionToken(actor, 1), positionToken(actor, 0)});
    auto results = fireTwice(actor, state);
    auto sequence = take(canonicalValueSequenceFromTokens(
        results, actor.operation->getResult(0).getType(), actor.operation));
    const std::uint8_t expected[] = {21, 22, 9, 10, 19, 20, 31, 32};
    require(sequence.tokenCount == 2 && sequence.lanes.size() == 8,
            "insert did not preserve two result occurrences");
    for (auto [lane, bits] : llvm::zip_equal(sequence.lanes, expected))
      requireLane(lane, SemanticState::Defined, bits);
  }

  {
    SimulatorState state;
    initializeRunState(state, execution);
    seed(state, entry.getArgument(1),
         {vectorToken(sourceType, {1, 2, 3, 4}),
          vectorToken(sourceType, {11, 12, 13, 14})});
    seed(state, entry.getArgument(2),
         {vectorToken(rhsType, {5, 6}), vectorToken(rhsType, {15, 16})});
    auto &actor =
        findActor(execution, dataflow::OperationSchemaId::VectorShuffle);
    auto results = fireTwice(actor, state);
    auto sequence = take(canonicalValueSequenceFromTokens(
        results, actor.operation->getResult(0).getType(), actor.operation));
    require(sequence.tokenCount == 2 && sequence.lanes.size() == 12,
            "shuffle did not preserve two result occurrences");
    const std::uint8_t defined[] = {1, 2, 5, 6, 11, 12, 15, 16};
    for (unsigned token = 0; token < 2; ++token) {
      const unsigned base = token * 6;
      const unsigned values = token * 4;
      for (unsigned lane = 0; lane < 4; ++lane)
        requireLane(sequence.lanes[base + lane], SemanticState::Defined,
                    defined[values + lane]);
      requireLane(sequence.lanes[base + 4], SemanticState::Poison);
      requireLane(sequence.lanes[base + 5], SemanticState::Poison);
    }

    auto bytes = take(encodeMemoryElement(
        results.front(), actor.operation->getResult(0).getType(),
        actor.operation));
    require(bytes.size() == 6 && bytes[0].state == SemanticState::Defined &&
                bytes[4].state == SemanticState::Poison &&
                bytes[5].state == SemanticState::Poison,
            "mixed vector store did not preserve lane-local memory state");
    auto memory = std::make_shared<MemoryValue>();
    memory->bytes.assign(bytes.begin(), bytes.end());
    memory->initialized = llvm::SmallBitVector(bytes.size(), true);
    MemoryView view{memory, {}, 0, actor.operation->getResult(0).getType()};
    SimulatorState memoryState;
    auto loaded = readMemoryElement(
        view, 0, actor.operation->getResult(0).getType(), memoryState,
        actor.operation, "vector structure test");
    require(loaded.has_value(),
            "mixed vector memory value could not be loaded");
    auto roundTrip = take(canonicalValueSequenceFromTokens(
        llvm::ArrayRef(*loaded), actor.operation->getResult(0).getType(),
        actor.operation));
    require(roundTrip.lanes.size() == 6,
            "mixed vector memory round-trip changed its lane count");
    for (auto [actual, expected] : llvm::zip_equal(
             roundTrip.lanes, llvm::ArrayRef(sequence.lanes).take_front(6))) {
      require(actual.state == expected.state,
              "mixed vector memory round-trip changed a lane state");
      if (expected.state == SemanticState::Defined)
        require(actual.bits == expected.bits,
                "mixed vector memory round-trip changed lane bits");
    }
  }
}

dataflow::CanonicalDataflowArtifact indexStructureProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  dataflow.graph private @index_structure(
      %start: none, %source: vector<2xindex>, %inserted: index,
      %destination: vector<2xindex>) -> (index, vector<2xindex>)
      attributes {
        input_segments = array<i32: 3, 0, 0>,
        result_segments = array<i32: 2, 0, 0>
      } {
    %extracted = vector.extract %source[1] : index from vector<2xindex>
    %updated = vector.insert %inserted, %destination[0]
        : index into vector<2xindex>
    %retired:3 = dataflow.sync %start, %extracted, %updated
        : (none, index, vector<2xindex>)
          -> (none, index, vector<2xindex>)
    dataflow.graph.return values(%retired#1, %retired#2
                                 : index, vector<2xindex>)
        streams() memories() complete(%retired#0 : none)
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("failed to parse the vector<index> structural fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

void indexStructuralActorsUseResolvedWidth() {
  auto artifact = indexStructureProgram();
  PreparedGraphExecution execution = prepare(artifact);
  mlir::Block &entry = execution.graph.getBody().front();
  auto vectorType =
      mlir::cast<mlir::VectorType>(entry.getArgument(1).getType());
  const auto index = [](std::uint64_t value) {
    return PrimitiveValue::integer(llvm::APInt(32, value));
  };

  {
    SimulatorState state;
    initializeRunState(state, execution);
    auto &actor =
        findActor(execution, dataflow::OperationSchemaId::VectorExtract);
    seed(state, entry.getArgument(1),
         {take(tokenFromVectorPrimitiveValues({index(7), index(9)}, vectorType,
                                              actor.operation))});
    Token result = fireOnce(actor, state);
    auto sequence = take(canonicalValueSequenceFromTokens(
        llvm::ArrayRef(result), actor.operation->getResult(0).getType(),
        actor.operation));
    require(sequence.lanes.size() == 1 &&
                sequence.lanes[0].state == SemanticState::Defined &&
                sequence.lanes[0].bits == llvm::APInt(32, 9),
            "vector<index> extract did not use the resolved index width");
  }

  {
    SimulatorState state;
    initializeRunState(state, execution);
    auto &actor =
        findActor(execution, dataflow::OperationSchemaId::VectorInsert);
    seed(state, entry.getArgument(2), {indexToken(llvm::APInt(32, 5))});
    seed(state, entry.getArgument(3),
         {take(tokenFromVectorPrimitiveValues({index(1), index(2)}, vectorType,
                                              actor.operation))});
    Token result = fireOnce(actor, state);
    auto sequence = take(canonicalValueSequenceFromTokens(
        llvm::ArrayRef(result), actor.operation->getResult(0).getType(),
        actor.operation));
    require(sequence.lanes.size() == 2 &&
                sequence.lanes[0].state == SemanticState::Defined &&
                sequence.lanes[0].bits == llvm::APInt(32, 5) &&
                sequence.lanes[1].state == SemanticState::Defined &&
                sequence.lanes[1].bits == llvm::APInt(32, 2),
            "vector<index> insert did not use the resolved index width");
  }
}

dataflow::CanonicalDataflowArtifact mixedLaneProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @add(%start: none, %lhs: vector<3xi8>,
                              %rhs: vector<3xi8>)
      -> vector<3xi8>
      attributes {
        input_segments = array<i32: 2, 0, 0>,
        result_segments = array<i32: 1, 0, 0>
      } {
    %sum = arith.addi %lhs, %rhs : vector<3xi8>
    %retired:2 = dataflow.sync %start, %sum
        : (none, vector<3xi8>) -> (none, vector<3xi8>)
    dataflow.graph.return values(%retired#1 : vector<3xi8>)
        streams() memories() complete(%retired#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    %lhs = arith.constant dense<0> : vector<3xi8>
    %rhs = arith.constant dense<0> : vector<3xi8>
    %value, %done = dataflow.graph.launch @add deps(%ctrl)
        values(%lhs, %rhs) stream_inputs() memories() stream_outputs()
        : (none, vector<3xi8>, vector<3xi8>) -> (vector<3xi8>, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %thread = dataflow.thread.launch @worker()
        : () -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("failed to parse the mixed-lane fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

void mixedLaneTypedInputRoundTrips() {
  auto artifact = mixedLaneProgram();
  auto view = take(artifact.view());
  require(view.rootThreadLaunches().size() == 1 &&
              view.staticGraphLaunches().size() == 1,
          "mixed-lane fixture has the wrong rooted launch count");
  const dataflow::RootedGraphLaunchRef launch{
      view.rootThreadLaunches().front().ref,
      view.staticGraphLaunches().front().ref};

  SpatialSimulationWorkload workloadDraft{launch};
  workloadDraft.valueInputPlan = {RuntimeValueInput{}, RuntimeValueInput{}};
  workloadDraft.observableContract.valueResults = {0};
  auto workload = take(finalizeSimulationWorkload(workloadDraft, view));

  CanonicalValueSequence input;
  input.tokenCount = 1;
  input.lanes = {SemanticLane::defined(llvm::APInt(8, 0x11)),
                 SemanticLane::poison(),
                 SemanticLane::defined(llvm::APInt(8, 0x33))};
  CanonicalValueSequence rhs;
  rhs.tokenCount = 1;
  rhs.lanes = {SemanticLane::defined(llvm::APInt(8, 1)),
               SemanticLane::defined(llvm::APInt(8, 2)), SemanticLane::undef()};
  SpatialSimulationRuntimeInputDraft runtimeDraft{workload.identity()};
  runtimeDraft.runtimeValues = {{0, input}, {1, rhs}};
  auto runtime =
      take(finalizeSimulationRuntimeInput(runtimeDraft, workload, view));
  auto execution =
      take(simulateRetiredDfgWorkload(artifact, workload, runtime));
  require(execution.observations.valueResults.size() == 1,
          "mixed-lane run has the wrong observation count");
  const auto *published = std::get_if<PublishedValueResult>(
      &execution.observations.valueResults.front());
  CanonicalValueSequence expected;
  expected.tokenCount = 1;
  expected.lanes = {SemanticLane::defined(llvm::APInt(8, 0x12)),
                    SemanticLane::poison(), SemanticLane::undef()};
  require(published && published->value.tokenCount == expected.tokenCount &&
              published->value.lanes.size() == expected.lanes.size(),
          "typed DFG input/output changed the mixed-lane vector shape");
  for (auto [actual, expected] :
       llvm::zip_equal(published->value.lanes, expected.lanes)) {
    require(actual.state == expected.state,
            "typed DFG input/output changed a mixed-lane state");
    if (expected.state == SemanticState::Defined)
      require(actual.bits == expected.bits,
              "typed DFG input/output changed mixed-lane bits");
  }
}

void subByteMixedLaneMemoryFailsClosed() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @subbyte(
      %start: none, %address: index, %memory: memref<1xvector<2xi1>>)
      -> vector<2xi1>
      attributes {
        input_segments = array<i32: 1, 0, 1>,
        result_segments = array<i32: 1, 0, 0>
      } {
    %loaded, %done = dataflow.load %memory[%address] %start
        : memref<1xvector<2xi1>>
    dataflow.graph.return values(%loaded : vector<2xi1>)
        streams() memories() complete(%done : none)
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("failed to parse the sub-byte memory fixture");
  auto artifact = take(dataflow::finalizeCanonicalDataflow(module.get()));
  auto graph = *artifact.module().getOps<dataflow::GraphOp>().begin();
  GraphPreparationResult prepared =
      take(prepareGraphExecution(artifact.module(), graph));
  auto *failure = std::get_if<GraphPreparationFailure>(&prepared);
  require(failure && failure->status == "unsupported" &&
              !failure->diagnostics.empty() &&
              llvm::StringRef(failure->diagnostics.front())
                  .contains("sub-byte-aligned lanes"),
          "sub-byte mixed-lane memory did not fail closed as Unsupported");

  auto vectorType =
      mlir::VectorType::get({2}, mlir::IntegerType::get(&context(), 1));
  auto mixed = take(tokenFromVectorPrimitiveValues(
      {PrimitiveValue::integer(llvm::APInt(1, 1)), PrimitiveValue::poison()},
      vectorType, artifact.module()));
  auto encoded = encodeMemoryElement(mixed, vectorType, artifact.module());
  require(
      !encoded &&
          llvm::toString(encoded.takeError()).find("sub-byte-aligned lanes") !=
              std::string::npos,
      "sub-byte mixed-lane memory token was silently collapsed");
}

void vectorPointerMemoryFailsClosedBeforeExecution() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<!llvm.ptr, dense<[64, 64, 64]> : vector<3xi64>>>
} {
  dataflow.graph private @pointer_vector(
      %start: none, %memory: memref<2xi8>) -> vector<2xi8>
      attributes {
        input_segments = array<i32: 0, 0, 1>,
        result_segments = array<i32: 1, 0, 0>
      } {
    %addresses = ub.poison : vector<2x!llvm.ptr>
    %loaded, %done = dataflow.load %memory[%addresses] %start
        : memref<2xi8>, vector<2x!llvm.ptr>, vector<2xi8>
    dataflow.graph.return values(%loaded : vector<2xi8>) streams()
        memories() complete(%done : none)
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("failed to parse the vector-pointer memory fixture");
  auto artifact = take(dataflow::finalizeCanonicalDataflow(module.get()));
  auto graph = *artifact.module().getOps<dataflow::GraphOp>().begin();
  GraphPreparationResult prepared =
      take(prepareGraphExecution(artifact.module(), graph));
  auto *failure = std::get_if<GraphPreparationFailure>(&prepared);
  require(failure && failure->status == "unsupported" &&
              !failure->diagnostics.empty() &&
              llvm::StringRef(failure->diagnostics.front())
                  .contains("lane-local pointer provenance"),
          "vector-pointer graph preparation did not fail closed as "
          "Unsupported");
}

} // namespace

int main() {
  exactStructuralActorsFireRepeatedly();
  indexStructuralActorsUseResolvedWidth();
  mixedLaneTypedInputRoundTrips();
  subByteMixedLaneMemoryFailsClosed();
  vectorPointerMemoryFailsClosedBeforeExecution();
  return EXIT_SUCCESS;
}
