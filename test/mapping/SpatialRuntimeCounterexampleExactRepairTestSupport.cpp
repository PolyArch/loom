#include "SpatialRuntimeCounterexampleExactRepairTestSupport.h"

#include "TechMappingCandidateTestSupport.h"
#include "TemporalMappingFabricTestSupport.h"

#include "ADG/Builder.h"
#include "ADG/FuLibrary.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/UsePatternValue.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/MappingObjective.h"
#include "PnR/SpatialActionExecutor.h"
#include "PnR/SpatialActionDomain.h"
#include "PnR/SpatialAnnealingSearch.h"
#include "PnR/SpatialCanonicalSeed.h"
#include "PnR/SpatialExactRepair.h"
#include "PnR/SpatialMappingMaterializer.h"
#include "PnR/SpatialMappingSelectionProjection.h"
#include "PnR/SpatialMappingWarmSeed.h"
#include "PnR/SpatialNetRouter.h"
#include "PnR/SpatialPnrProblem.h"
#include "PnR/SpatialRouteCostState.h"

#include "SpatialBindingRelationModel.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::test {
namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "spatial runtime-counterexample exact repair test: "
               << message << '\n';
  std::exit(1);
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

class ExactRepairTemporaryDirectory final {
public:
  ExactRepairTemporaryDirectory() {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-spatial-runtime-exact-repair", path))
      fail("cannot create exact-repair ArtifactStore directory: " +
           error.message());
    path_ = path.str().str();
  }

  ~ExactRepairTemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << "cannot remove exact-repair test directory: "
                   << error.message() << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string path_;
};

mlir::MLIRContext makeExactRepairContext() {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::mapping::MappingDialect,
                  ::fabric::FabricDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact
buildRegisterFifoDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @chain(%start: none, %lhs: i32, %rhs: i32) -> i32
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %sum = arith.addi %lhs, %rhs : i32
    %retired:2 = dataflow.sync %start, %sum
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%retired#1 : i32) streams() memories()
        complete(%retired#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %lhs: i32, %rhs: i32) ctrl (%ctrl: none) {
    %result, %done = dataflow.graph.launch @chain deps(%ctrl)
        values(%lhs, %rhs) stream_inputs() memories() stream_outputs()
        : (none, i32, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %lhs = arith.constant 7 : i32
    %rhs = arith.constant 11 : i32
    %thread = dataflow.thread.launch @worker(%lhs, %rhs)
        : (i32, i32) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse register-FIFO Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

/// A chain of adds retired through the token-sync FU. Two adds on the
/// fixture's single ALU would pair through that FU's own register FIFO, which
/// closes a combinational ready cycle by itself; three adds on two ALUs admit
/// two individually acyclic pairings whose union closes a cycle through both.
dataflow::CanonicalDataflowArtifact
buildChainedAddDataflow(mlir::MLIRContext &context, std::size_t addCount) {
  std::string body = "    %sum0 = arith.addi %lhs, %rhs : i32\n";
  for (std::size_t add = 1; add < addCount; ++add)
    body += "    %sum" + std::to_string(add) + " = arith.addi %sum" +
            std::to_string(add - 1) + ", %rhs : i32\n";
  const std::string source = R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @chain(%start: none, %lhs: i32, %rhs: i32) -> i32
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
)mlir" + body +
                             "    %retired:2 = dataflow.sync %start, %sum" +
                             std::to_string(addCount - 1) + R"mlir(
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%retired#1 : i32) streams() memories()
        complete(%retired#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %lhs: i32, %rhs: i32) ctrl (%ctrl: none) {
    %result, %done = dataflow.graph.launch @chain deps(%ctrl)
        values(%lhs, %rhs) stream_inputs() memories() stream_outputs()
        : (none, i32, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %lhs = arith.constant 7 : i32
    %rhs = arith.constant 11 : i32
    %thread = dataflow.thread.launch @worker(%lhs, %rhs)
        : (i32, i32) -> !dataflow.thread_token
    return
  }
}
)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    fail("cannot parse chained-add register-FIFO Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

fabric::FinalizedFabricRoot
buildRegisterFifoFabric(const ArtifactStore &store,
                        bool feedbackBypassable = true,
                        std::size_t aluCount = 1,
                        std::uint32_t residentContexts = 3) {
  using namespace adg;

  const PortType bits128 = take(PortType::bits(128));
  const PortType tagged128 = take(PortType::taggedBits(128, 4));
  const std::vector<PortType> moduleTypes(4, tagged128);
  DesignBuilder builder(store);
  auto spatial = take(builder.createSpatialCore("runtime-exact-register-fifo",
                                                moduleTypes, moduleTypes));
  std::vector<SpatialValue> peInputs;
  peInputs.reserve(moduleTypes.size());
  std::vector<SpatialBackedge> feedback;
  feedback.reserve(moduleTypes.size());
  for (std::size_t input = 0; input < moduleTypes.size(); ++input) {
    auto backedge = take(spatial.createBackedge(tagged128));
    auto selected = take(spatial.addSwitch(
        {take(spatial.input(input)), backedge.value()},
        SwitchSpec::temporal({tagged128, tagged128}, {tagged128}, {{0, 1}}, 2,
                             ::fabric::TemporalSwitchFixedPriority{{0, 1}})));
    peInputs.push_back(selected.front());
    feedback.push_back(std::move(backedge));
  }
  auto pe = take(spatial.addPe(
      peInputs,
      PeSpec::temporal(
          std::vector<PortType>(4, bits128), moduleTypes,
          TemporalPeParameters{residentContexts,
                               FuConfigurationMode::PerInstruction,
                               ::fabric::OperandBufferMode::PerInstruction, 2,
                               TemporalRegisterFifoParameters{2, 2, 2}})));
  std::vector<PeValue> fuInputs;
  fuInputs.reserve(moduleTypes.size());
  for (std::size_t input = 0; input < moduleTypes.size(); ++input)
    fuInputs.push_back(take(pe.input(input)));
  for (std::size_t alu = 0; alu < aluCount; ++alu)
    requireSuccess(addCoreAluFu(
        pe, llvm::ArrayRef<PeValue>(fuInputs).take_front(3),
        ::fabric::ResolvedIndexWidthSet::get(
            {::fabric::ResolvedIndexWidth::I64})));
  addTokenSyncFu(pe, fuInputs, bits128,
                 ::fabric::oneCycleElasticOperationResourceContract());
  requireSuccess(pe.close());
  std::vector<SpatialValue> outputs;
  outputs.reserve(moduleTypes.size());
  for (std::size_t output = 0; output < moduleTypes.size(); ++output) {
    auto fanout = take(spatial.addSwitch(
        {take(pe.output(output))},
        SwitchSpec::temporal({tagged128}, {tagged128, tagged128}, {{0}, {0}}, 2,
                             std::nullopt)));
    SpatialValue buffered =
        take(spatial.addFifo(fanout[0],
                             FifoSpec{tagged128, 2, feedbackBypassable,
                                      std::nullopt}))
            .value();
    requireSuccess(
        spatial.resolveBackedge(std::move(feedback[output]), buffered));
    outputs.push_back(fanout[1]);
  }
  requireSuccess(spatial.close(outputs));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("register-FIFO Fabric fixture did not publish one root");
  return design.roots().front();
}

bool mappingRejected(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric,
    const mapping::SpatialMappingConstraintSetView &constraints,
    const mapping::SpatialMappingView &candidateMapping) {
  bool rejected = false;
  llvm::handleAllErrors(
      mapping::admitSpatialMappingConstraints(dataflow, techMapping, fabric,
                                              constraints, candidateMapping),
      [&](const mapping::SpatialMappingConstraintRejection &) {
        rejected = true;
      },
      [&](const llvm::ErrorInfoBase &error) { fail(error.message()); });
  return rejected;
}

using FrozenProblemFactory =
    std::function<llvm::Expected<pnr::FrozenSpatialPnrProblemHandle>(
        const mapping::SpatialMappingConstraintSetView &)>;

struct RepairEnvironment final {
  const ::dataflow::CanonicalDataflowProgramView &dataflow;
  const mapping::TechMappingView &techMapping;
  const fabric::FabricArtifactView &fabric;
  const mapping::FinalizedSpatialMappingConstraintSet &parentConstraints;
  const mapping::FinalizedSpatialMapping &parentMapping;
  const pnr::ResolvedPnrConfigView &pnrConfig;
  const ArtifactStore &store;
  FrozenProblemFactory freeze;
};

struct RepairedMapping final {
  mapping::FinalizedSpatialMapping mapping;
  pnr::SpatialExactRepairResult repair;
};

bool exactRepairResultsEqual(const pnr::SpatialExactRepairResult &lhs,
                             const pnr::SpatialExactRepairResult &rhs) {
  return lhs.kind == rhs.kind && lhs.regionDecisions == rhs.regionDecisions &&
         lhs.logicalSolverCalls == rhs.logicalSolverCalls &&
         lhs.actionCount == rhs.actionCount &&
         lhs.endpointExpansions == rhs.endpointExpansions &&
         lhs.negotiationIterations == rhs.negotiationIterations &&
         lhs.detail == rhs.detail;
}

llvm::Expected<bool>
parentSelectionsHold(const mapping::SpatialMappingView &parent,
                     const pnr::SpatialCandidateState &candidate) {
  std::vector<const pnr::RouteTreeState *> routes;
  std::vector<llvm::ArrayRef<std::optional<llvm::APInt>>> tags;
  const auto logicalNets = candidate.problem().transfers().logicalNets();
  routes.reserve(logicalNets.size());
  tags.reserve(logicalNets.size());
  for (pnr::PnrIndex logicalNet = 0; logicalNet < logicalNets.size();
       ++logicalNet) {
    routes.push_back(&candidate.routeTree(logicalNet));
    tags.push_back(candidate.tagValues(logicalNet));
  }
  return pnr::spatialMappingSelectionEqualsCandidate(parent, candidate, routes,
                                                     tags);
}

mapping::FinalizedSpatialMappingConstraintSet
finalizeNoGood(const RepairEnvironment &environment,
               llvm::ArrayRef<mapping::SpatialNoGoodLiteral> literals) {
  return take(mapping::finalizeSpatialRuntimeCounterexampleConstraintSet(
      environment.parentConstraints.reference(), literals, environment.store));
}

RepairedMapping
repairOnce(llvm::StringRef name, const RepairEnvironment &environment,
           const mapping::FinalizedSpatialMappingConstraintSet &constraints) {
  auto problem = take(environment.freeze(constraints.view()));
  auto warm = take(pnr::projectFinalizedSpatialMappingWarmSeed(
      environment.parentMapping, problem));
  if (warm.parentMappingIdentity() !=
      environment.parentMapping.reference().artifact)
    fail(name + ": warm seed lost its exact parent identity");
  auto candidate = take(warm.materializeCandidate());
  requireSuccess(candidate->verify());
  if (candidate->runtimeCounterexampleViolation() != 1 ||
      !candidate->firstRuntimeCounterexampleViolation())
    fail(name + ": warm parent did not expose one exact no-good witness");
  if (!take(parentSelectionsHold(environment.parentMapping.view(), *candidate)))
    fail(name + ": warm parent changed a persistent Mapping selection");
  if (candidate->atomicCapacityOveruse() != 0 ||
      candidate->unroutedObligationCount() != 0 ||
      candidate->routeCapacityOveruse() != 0 ||
      candidate->tagResidentCapacityOveruse() != 0 ||
      candidate->tagUnassignedCount() != 0 ||
      candidate->tagConflictCount() != 0 ||
      candidate->hardProgressViolation() != 0)
    fail(name + ": warm parent contains an unrelated hard violation");

  const pnr::PnrIndex clause =
      *candidate->firstRuntimeCounterexampleViolation();
  pnr::SpatialExactRepairScratch exactRepair;
  pnr::DeterministicPnrRandomStream stream =
      pnr::DeterministicPnrRandomStream::create(
          problem->config().policy().determinism.masterSeed, 0,
          pnr::PnrRandomStreamPurpose::ExactRepair);
  const pnr::SpatialExactRepairResult repaired = take(exactRepair.repair(
      *candidate, 0,
      problem->config().policy().search.exactRepair.maxSolverCalls, stream, {},
      clause));
  if (repaired.kind != pnr::SpatialExactRepairResultKind::Repaired ||
      repaired.actionCount == 0)
    fail(name + ": exact repair did not commit a finite literal breaker " +
         llvm::Twine(static_cast<std::uint32_t>(repaired.kind)) +
         " (solver calls " + llvm::Twine(repaired.solverCalls) + ", actions " +
         llvm::Twine(repaired.actionCount) + "): " + repaired.detail);
  if (candidate->runtimeCounterexampleViolation() != 0 ||
      candidate->firstRuntimeCounterexampleViolation())
    fail(name + ": committed repair retained its primary no-good witness");
  if (take(parentSelectionsHold(environment.parentMapping.view(), *candidate)))
    fail(name + ": repaired candidate is a semantic no-op");
  requireSuccess(candidate->verify());

  auto finalized = take(pnr::finalizeSpatialMappingCandidate(
      *candidate, environment.dataflow, environment.techMapping,
      environment.fabric, constraints.view(), environment.store));
  if (finalized.reference() == environment.parentMapping.reference())
    fail(name + ": repaired candidate republished its exact parent");
  requireSuccess(mapping::admitSpatialMappingConstraints(
      environment.dataflow, environment.techMapping, environment.fabric,
      constraints.view(), finalized.view()));
  return {std::move(finalized), repaired};
}

RepairedMapping repairDeterministically(
    llvm::StringRef name, const RepairEnvironment &environment,
    llvm::ArrayRef<mapping::SpatialNoGoodLiteral> literals) {
  const auto constraints = finalizeNoGood(environment, literals);
  if (!mappingRejected(environment.dataflow, environment.techMapping,
                       environment.fabric, constraints.view(),
                       environment.parentMapping.view()))
    fail(name + ": no-good does not reject the exact parent Mapping");
  RepairedMapping first = repairOnce(name, environment, constraints);
  const RepairedMapping repeated =
      repairOnce((name + ".replay").str(), environment, constraints);
  if (first.mapping.reference() != repeated.mapping.reference() ||
      !exactRepairResultsEqual(first.repair, repeated.repair))
    fail(
        name + ": repeated warm repair changed its result or child Mapping " +
        "(mapping equal " +
        llvm::Twine(first.mapping.reference() == repeated.mapping.reference()) +
        ", first kind " +
        llvm::Twine(static_cast<std::uint32_t>(first.repair.kind)) +
        ", repeated kind " +
        llvm::Twine(static_cast<std::uint32_t>(repeated.repair.kind)) +
        ", first region " + llvm::Twine(first.repair.regionDecisions) +
        ", repeated region " + llvm::Twine(repeated.repair.regionDecisions) +
        ", first logical solver calls " +
        llvm::Twine(first.repair.logicalSolverCalls) +
        ", repeated logical solver calls " +
        llvm::Twine(repeated.repair.logicalSolverCalls) + ", first actions " +
        llvm::Twine(first.repair.actionCount) + ", repeated actions " +
        llvm::Twine(repeated.repair.actionCount) +
        ", first endpoint expansions " +
        llvm::Twine(first.repair.endpointExpansions) +
        ", repeated endpoint expansions " +
        llvm::Twine(repeated.repair.endpointExpansions) +
        ", first negotiation iterations " +
        llvm::Twine(first.repair.negotiationIterations) +
        ", repeated negotiation iterations " +
        llvm::Twine(repeated.repair.negotiationIterations) + ")");
  return first;
}

void requireBudgetRollback(
    llvm::StringRef name, const RepairEnvironment &environment,
    llvm::ArrayRef<mapping::SpatialNoGoodLiteral> literals) {
  const auto constraints = finalizeNoGood(environment, literals);
  auto problem = take(environment.freeze(constraints.view()));
  auto warm = take(pnr::projectFinalizedSpatialMappingWarmSeed(
      environment.parentMapping, problem));
  auto candidate = take(warm.materializeCandidate());
  if (candidate->runtimeCounterexampleViolation() != 1)
    fail(name + ": rollback fixture has no live no-good witness");
  const pnr::PnrIndex clause =
      *candidate->firstRuntimeCounterexampleViolation();
  pnr::SpatialExactRepairScratch exactRepair;
  pnr::DeterministicPnrRandomStream stream =
      pnr::DeterministicPnrRandomStream::create(
          problem->config().policy().determinism.masterSeed, 0,
          pnr::PnrRandomStreamPurpose::ExactRepair);
  const auto outcome =
      take(exactRepair.repair(*candidate, 0, 1, stream, {}, clause));
  if (outcome.kind !=
          pnr::SpatialExactRepairResultKind::UnknownBudgetExhausted ||
      outcome.logicalSolverCalls != 1)
    fail(name + ": one-call breaker budget returned the wrong typed outcome");
  if (candidate->runtimeCounterexampleViolation() != 1 ||
      !take(parentSelectionsHold(environment.parentMapping.view(), *candidate)))
    fail(name + ": exhausted repair did not restore the exact warm parent");
  requireSuccess(candidate->verify());
}

bool literalHolds(const RepairEnvironment &environment,
                  const mapping::SpatialNoGoodLiteral &literal,
                  const mapping::SpatialMappingView &candidate) {
  const auto singleton = finalizeNoGood(environment, {literal});
  return mappingRejected(environment.dataflow, environment.techMapping,
                         environment.fabric, singleton.view(), candidate);
}

pnr::PnrIndex logicalNetOrdinal(
    const pnr::FrozenSpatialPnrProblem &problem,
    const ::dataflow::CanonicalGraphProducerEndpointRef &producer) {
  const auto nets = problem.transfers().logicalNets();
  const auto found =
      llvm::find_if(nets, [&](const pnr::FrozenSpatialLogicalNet &net) {
        return net.producer == producer;
      });
  if (found == nets.end())
    fail("parent RouteTree producer has no frozen logical net");
  return static_cast<pnr::PnrIndex>(found - nets.begin());
}

llvm::ArrayRef<pnr::PnrIndex>
terminalChoices(const pnr::FrozenSpatialPnrProblem &problem,
                pnr::FrozenSpatialTerminalBinding binding) {
  if (binding.kind == pnr::FrozenSpatialTerminalBindingKind::PortDemand)
    return problem.bindingRelations().portAttachmentChoices(binding.index);
  return problem.bindingRelations().graphBoundaryAttachmentChoices(
      binding.index);
}

bool terminalHasAlternative(
    const pnr::FrozenSpatialPnrProblem &problem,
    pnr::FrozenSpatialTerminalBinding binding,
    const fabric::FabricTransportEndpointRef &selected,
    const std::optional<fabric::FabricPhysicalTraversalRef> &excludedLocal =
        std::nullopt) {
  const auto options = problem.ports().attachmentOptions();
  const auto endpoints = problem.routing().routingEndpoints();
  const auto traversals = problem.routing().traversals();
  for (pnr::PnrIndex choice : terminalChoices(problem, binding)) {
    if (choice >= options.size())
      fail("terminal attachment choice is out of range");
    const auto &option = options[choice];
    if (option.endpoint >= endpoints.size() ||
        (option.localTraversal && *option.localTraversal >= traversals.size()))
      fail("terminal attachment option names a foreign route element");
    if (excludedLocal && option.localTraversal &&
        traversals[*option.localTraversal].reference == *excludedLocal)
      continue;
    if (endpoints[option.endpoint].reference != selected || excludedLocal)
      return true;
  }
  return false;
}

struct AttachmentAnchor final {
  mapping::SpatialTransferAttachmentEqualsLiteral literal;
  pnr::PnrIndex logicalNet = 0;
  pnr::FrozenSpatialTerminalBinding binding;
};

std::optional<AttachmentAnchor>
findAttachmentAnchor(const mapping::SpatialMappingView &mappingView,
                     const pnr::FrozenSpatialPnrProblem &problem,
                     pnr::FrozenSpatialTerminalBindingKind bindingKind,
                     bool source) {
  const auto &transfers = problem.transfers();
  for (const auto &route : mappingView.routeTrees()) {
    const pnr::PnrIndex logicalNet =
        logicalNetOrdinal(problem, route.logicalNet);
    if (source) {
      const auto binding = transfers.logicalNetSourceBindings()[logicalNet];
      if (binding.kind != bindingKind ||
          !terminalHasAlternative(problem, binding, route.rootEndpoint))
        continue;
      return AttachmentAnchor{{mapping::SpatialConstraintTransferTerminal{
                                   route.logicalNet, std::nullopt},
                               route.rootEndpoint},
                              logicalNet,
                              binding};
    }
    const auto &net = transfers.logicalNets()[logicalNet];
    for (const auto &sink : route.sinks) {
      const auto frozenSinks =
          transfers.logicalNetSinks().slice(net.sinkOffset, net.sinkCount);
      const auto found = llvm::find(frozenSinks, sink.sink);
      if (found == frozenSinks.end() || sink.nodeOrdinal >= route.nodes.size())
        fail("parent RouteTree sink is absent from its frozen logical net");
      const auto localSink =
          static_cast<pnr::PnrIndex>(found - frozenSinks.begin());
      const auto binding =
          transfers.logicalNetSinkBindings()[net.sinkOffset + localSink];
      const auto &endpoint = route.nodes[sink.nodeOrdinal].endpoint;
      if (binding.kind != bindingKind ||
          !terminalHasAlternative(problem, binding, endpoint))
        continue;
      return AttachmentAnchor{{mapping::SpatialConstraintTransferTerminal{
                                   route.logicalNet, sink.sink},
                               endpoint},
                              logicalNet,
                              binding};
    }
  }
  return std::nullopt;
}

struct TraversalAnchor final {
  mapping::SpatialNetUsesTraversalLiteral literal;
  enum class Position : std::uint8_t { SourceLocal, Internal, SinkLocal };
  Position position = Position::Internal;
};

std::optional<TraversalAnchor>
findTraversalAnchor(const mapping::SpatialMappingView &mappingView,
                    TraversalAnchor::Position position, bool sinkQualified) {
  for (const auto &route : mappingView.routeTrees()) {
    if (position == TraversalAnchor::Position::SourceLocal &&
        route.localTraversal) {
      std::optional<::dataflow::CanonicalGraphConsumerEndpointRef> sink;
      if (sinkQualified) {
        if (route.sinks.empty())
          continue;
        sink = route.sinks.front().sink;
      }
      return TraversalAnchor{{route.logicalNet, sink, *route.localTraversal},
                             position};
    }
    if (position == TraversalAnchor::Position::SinkLocal) {
      for (const auto &sink : route.sinks)
        if (sink.localTraversal)
          return TraversalAnchor{
              {route.logicalNet, sink.sink, *sink.localTraversal}, position};
      continue;
    }
    if (position != TraversalAnchor::Position::Internal)
      continue;
    for (const auto &node : route.nodes) {
      if (!node.incomingTraversal)
        continue;
      std::optional<::dataflow::CanonicalGraphConsumerEndpointRef> sink;
      if (sinkQualified) {
        const auto selected = llvm::find_if(
            route.sinks, [&](const mapping::SpatialRouteSinkView &candidate) {
              if (candidate.nodeOrdinal >= route.nodes.size())
                return false;
              for (std::optional<std::uint64_t> cursor = candidate.nodeOrdinal;
                   cursor; cursor = route.nodes[*cursor].parentOrdinal)
                if (*cursor == node.ordinal)
                  return true;
              return false;
            });
        if (selected == route.sinks.end())
          continue;
        sink = selected->sink;
      }
      return TraversalAnchor{{route.logicalNet, sink, *node.incomingTraversal},
                             position};
    }
  }
  return std::nullopt;
}

mapping::SpatialNetTagEqualsLiteral
findTagAnchor(const mapping::SpatialMappingView &mapping) {
  for (const auto &segment : mapping.physicalTagSegments()) {
    if (segment.routeTreeOrdinal >= mapping.routeTrees().size() ||
        segment.resourceUseOrdinal >= mapping.resourceUses().size())
      fail("parent Physical Tag segment has a foreign owner");
    const auto &assignments =
        mapping.resourceUses()[segment.resourceUseOrdinal].sharingAssignments;
    if (assignments.size() != 1)
      continue;
    const auto *tag =
        std::get_if<::fabric::PhysicalTagPatternValue>(&assignments.front());
    if (!tag)
      continue;
    return {mapping.routeTrees()[segment.routeTreeOrdinal].logicalNet,
            segment.segmentOrdinal, tag->value};
  }
  fail("tagged parent Mapping has no exact route-local Physical Tag");
}

void requireLiteralBroken(llvm::StringRef name,
                          const RepairEnvironment &environment,
                          const mapping::SpatialNoGoodLiteral &literal,
                          const mapping::SpatialMappingView &child) {
  if (literalHolds(environment, literal, child))
    fail(name + ": repaired child retained the selected literal");
}

void exerciseAttachmentCases(const RepairEnvironment &environment,
                             const pnr::FrozenSpatialPnrProblem &problem) {
  struct Case final {
    pnr::FrozenSpatialTerminalBindingKind kind;
    bool source;
    llvm::StringLiteral name;
  };
  const Case cases[] = {
      {pnr::FrozenSpatialTerminalBindingKind::PortDemand, true,
       "attachment.source.port"},
      {pnr::FrozenSpatialTerminalBindingKind::GraphBoundary, true,
       "attachment.source.boundary"},
      {pnr::FrozenSpatialTerminalBindingKind::PortDemand, false,
       "attachment.sink.port"},
      {pnr::FrozenSpatialTerminalBindingKind::GraphBoundary, false,
       "attachment.sink.boundary"},
  };
  std::vector<std::pair<const Case *, AttachmentAnchor>> anchors;
  anchors.reserve(std::size(cases));
  for (const Case &testCase : cases) {
    auto anchor = findAttachmentAnchor(environment.parentMapping.view(),
                                       problem, testCase.kind, testCase.source);
    if (!anchor)
      fail(testCase.name +
           ": real parent fixture has no alternate exact terminal choice");
    anchors.emplace_back(&testCase, std::move(*anchor));
  }
  const auto rollback = llvm::find_if(anchors, [&](const auto &entry) {
    const auto domains = problem.localTransfers().domains();
    return entry.second.logicalNet < domains.size() &&
           domains[entry.second.logicalNet].optionCount == 0;
  });
  if (rollback == anchors.end())
    fail("attachment rollback fixture has no net without a local bypass");
  requireBudgetRollback(
      "attachment.rollback", environment,
      {mapping::SpatialNoGoodLiteral{rollback->second.literal}});

  for (const auto &[testCase, anchor] : anchors) {
    const mapping::SpatialNoGoodLiteral literal = anchor.literal;
    const RepairedMapping child =
        repairDeterministically(testCase->name, environment, {literal});
    requireLiteralBroken(testCase->name, environment, literal,
                         child.mapping.view());
  }
}

void exerciseTraversalCase(llvm::StringRef name,
                           const RepairEnvironment &environment,
                           const TraversalAnchor &anchor) {
  const mapping::SpatialNoGoodLiteral literal = anchor.literal;
  const RepairedMapping child =
      repairDeterministically(name, environment, {literal});
  requireLiteralBroken(name, environment, literal, child.mapping.view());
}

} // namespace

void exerciseSpatialRuntimeCounterexampleExactRepair(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric,
    const mapping::FinalizedSpatialMappingConstraintSet &parentConstraints,
    const mapping::FinalizedSpatialMapping &parentMapping,
    const fabric::FabricPhysicalTimingProfileView &physicalTiming,
    const pnr::ResolvedPnrConfigView &pnrConfig, const ArtifactStore &store) {
  const RepairEnvironment environment{
      dataflow,
      techMapping,
      fabric,
      parentConstraints,
      parentMapping,
      pnrConfig,
      store,
      [&](const mapping::SpatialMappingConstraintSetView &constraints) {
        return pnr::freezeSpatialPnrProblem(dataflow, techMapping, fabric,
                                            physicalTiming, pnrConfig,
                                            constraints);
      }};
  const auto baseProblem = take(environment.freeze(parentConstraints.view()));
  exerciseAttachmentCases(environment, *baseProblem);

  auto wholeTree = findTraversalAnchor(
      parentMapping.view(), TraversalAnchor::Position::Internal, false);
  if (!wholeTree)
    wholeTree = findTraversalAnchor(
        parentMapping.view(), TraversalAnchor::Position::SourceLocal, false);
  if (!wholeTree)
    wholeTree = findTraversalAnchor(
        parentMapping.view(), TraversalAnchor::Position::SinkLocal, false);
  if (!wholeTree)
    fail("traversal.whole-tree: real parent has no selected traversal");
  exerciseTraversalCase("traversal.whole-tree", environment, *wholeTree);

  auto source = findAttachmentAnchor(
      parentMapping.view(), *baseProblem,
      pnr::FrozenSpatialTerminalBindingKind::GraphBoundary, true);
  auto sink = findAttachmentAnchor(
      parentMapping.view(), *baseProblem,
      pnr::FrozenSpatialTerminalBindingKind::PortDemand, false);
  if (!source || !sink)
    fail("mixed clause has no source and sink attachment anchors");
  const mapping::SpatialNoGoodLiteral traversal = wholeTree->literal;
  const mapping::SpatialNoGoodLiteral sourceAttachment = source->literal;
  const mapping::SpatialNoGoodLiteral sinkAttachment = sink->literal;
  const mapping::SpatialNoGoodLiteral exactParent =
      mapping::SpatialMappingIdentityEqualsLiteral{parentMapping.reference(),
                                                   nullptr};
  const RepairedMapping mixed = repairDeterministically(
      "mixed.traversal-attachment", environment,
      {traversal, sourceAttachment, sinkAttachment, exactParent});
  if (literalHolds(environment, traversal, mixed.mapping.view()) &&
      literalHolds(environment, sourceAttachment, mixed.mapping.view()) &&
      literalHolds(environment, sinkAttachment, mixed.mapping.view()) &&
      literalHolds(environment, exactParent, mixed.mapping.view()))
    fail("mixed clause repair did not break any exact literal");
}

void exerciseSpatialTaggedRuntimeCounterexampleExactRepair(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric,
    const mapping::FinalizedSpatialMappingConstraintSet &parentConstraints,
    const mapping::FinalizedSpatialMapping &parentMapping,
    const pnr::ResolvedPnrConfigView &pnrConfig, const ArtifactStore &store) {
  const RepairEnvironment environment{
      dataflow,
      techMapping,
      fabric,
      parentConstraints,
      parentMapping,
      pnrConfig,
      store,
      [&](const mapping::SpatialMappingConstraintSetView &constraints) {
        return pnr::freezeSpatialPnrProblem(dataflow, techMapping, fabric,
                                            pnrConfig, constraints);
      }};

  const mapping::SpatialNoGoodLiteral tag = findTagAnchor(parentMapping.view());
  const RepairedMapping tagChild =
      repairDeterministically("tag.route-local", environment, {tag});
  requireLiteralBroken("tag.route-local", environment, tag,
                       tagChild.mapping.view());

  const auto sourceLocal = findTraversalAnchor(
      parentMapping.view(), TraversalAnchor::Position::SourceLocal, false);
  const auto sinkLocal = findTraversalAnchor(
      parentMapping.view(), TraversalAnchor::Position::SinkLocal, true);
  const auto branch = findTraversalAnchor(
      parentMapping.view(), TraversalAnchor::Position::Internal, true);
  if (!sourceLocal || !sinkLocal || !branch)
    fail("tagged parent does not expose source-local, sink-local, and "
         "sink-qualified traversal positions");
  exerciseTraversalCase("traversal.source-local", environment, *sourceLocal);
  exerciseTraversalCase("traversal.sink-local", environment, *sinkLocal);
  exerciseTraversalCase("traversal.sink-qualified", environment, *branch);

  const mapping::SpatialNoGoodLiteral traversal = branch->literal;
  const mapping::SpatialNoGoodLiteral exactParent =
      mapping::SpatialMappingIdentityEqualsLiteral{parentMapping.reference(),
                                                   nullptr};
  const RepairedMapping mixed = repairDeterministically(
      "mixed.traversal-tag", environment, {traversal, tag, exactParent});
  if (literalHolds(environment, traversal, mixed.mapping.view()) &&
      literalHolds(environment, tag, mixed.mapping.view()) &&
      literalHolds(environment, exactParent, mixed.mapping.view()))
    fail("mixed traversal/tag repair retained every exact literal");
}

struct ChainedAddProblem final {
  mapping::FinalizedTechMapping tech;
  mapping::FinalizedSpatialMappingConstraintSet constraints;
  pnr::FrozenSpatialPnrProblemHandle problem;
};

ChainedAddProblem freezeChainedAddProblem(ArtifactStore &store,
                                          mlir::MLIRContext &context,
                                          const fabric::FinalizedFabricRoot &fabric,
                                          std::size_t addCount) {
  auto dataflowArtifact = buildChainedAddDataflow(context, addCount);
  (void)take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  const auto dataflow = take(dataflowArtifact.view());

  ResolvedConfig resolved = defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  const auto techConfig =
      take(mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<::dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  auto generated = mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), techConfig, store});
  const auto *techCandidates =
      std::get_if<mapping::GeneratedTechMappings>(&generated);
  if (!techCandidates || techCandidates->candidates.size() != 1)
    fail("chained-add register-FIFO fixture did not generate one TechMapping");
  auto tech = take(
      mapping::importTechMapping(techCandidates->candidates.front(), store));
  auto constraints = buildSpatialMappingConstraints(
      context, dataflow, tech.view(), fabric.view(), store);
  ResolvedConfig pnrResolved = buildSpatialPnrTestResolvedConfig();
  const auto pnrConfig =
      take(pnr::projectResolvedSpatialPnrConfigView(pnrResolved));
  auto problem = take(pnr::freezeSpatialPnrProblem(
      dataflow, tech.view(), fabric.view(), pnrConfig, constraints.view()));
  return {std::move(tech), std::move(constraints), std::move(problem)};
}

/// The frozen local-transfer domain never admits a pairing that closes a
/// combinational cycle through its own producer and consumer placements:
/// two adds on one elastic ALU keep only the external route, while the
/// cross-FU pairing into the token-sync FU survives and seeds acyclic.
void exerciseStaticallyClosedRegisterFifoDomain(
    ArtifactStore &store, mlir::MLIRContext &context,
    const fabric::FinalizedFabricRoot &fabric) {
  const ChainedAddProblem chain =
      freezeChainedAddProblem(store, context, fabric, /*addCount=*/2);
  const auto &problem = chain.problem;
  const auto &localTransfers = problem->localTransfers();
  if (localTransfers.closedOptionCount() == 0)
    fail("single-ALU chain admitted its self-closing register-FIFO pairing");
  for (const pnr::FrozenSpatialRegisterFifoTransferOption &option :
       localTransfers.options())
    if (option.writer.fu == option.reader.fu)
      fail("frozen local-transfer domain admits a same-FU pairing");
  pnr::SpatialPathFinderSeedWorkSummary work;
  auto seed = take(pnr::createCanonicalPathFinderSpatialSeed(problem, work));
  if (seed.candidate->selectedHandshakeViolation() != 0)
    fail("closed pairings left the frozen domain but the seed is cyclic");
  bool pairs = false;
  for (pnr::PnrIndex logicalNet = 0;
       logicalNet < problem->transfers().logicalNets().size(); ++logicalNet)
    pairs |= seed.candidate->usesRegisterFifo(logicalNet);
  if (!pairs)
    fail("static closure also removed the cross-FU register-FIFO pairing");
  requireSuccess(seed.candidate->verify());

}

/// Routes the register-FIFO net `selectedNet` of `candidate` externally
/// through the feedback-matched sink attachment of the fixture, so the
/// candidate keeps a legal external route where the local pairing was.
void routeRegisterFifoNetExternally(
    const pnr::FrozenSpatialPnrProblemHandle &problem,
    pnr::SpatialCandidateState &candidate, pnr::PnrIndex selectedNet) {
  constexpr pnr::PnrIndex temporalPePortCount = 4;
  const auto &selectedLogicalNet =
      problem->transfers().logicalNets()[selectedNet];
  if (selectedLogicalNet.sinkCount != 1)
    fail("register-FIFO fixture local net does not have one sink");
  const pnr::PnrIndex sourceEndpoint =
      candidate.logicalNetSourceEndpoint(selectedNet);
  const auto endpoints = problem->routing().routingEndpoints();
  if (sourceEndpoint >= endpoints.size() ||
      endpoints[sourceEndpoint].reference.ordinal < temporalPePortCount)
    fail("register-FIFO fixture source is not a PE output endpoint");
  const auto &sourceReference = endpoints[sourceEndpoint].reference;
  const auto sinkBinding =
      problem->transfers()
          .logicalNetSinkBindings()[selectedLogicalNet.sinkOffset];
  std::optional<pnr::PnrIndex> feedbackSinkOption;
  for (pnr::PnrIndex option : terminalChoices(*problem, sinkBinding)) {
    if (option >= problem->ports().attachmentOptions().size())
      fail("register-FIFO sink attachment choice is out of range");
    const pnr::PnrIndex endpoint =
        problem->ports().attachmentOptions()[option].endpoint;
    if (endpoint >= endpoints.size())
      fail("register-FIFO sink attachment endpoint is out of range");
    const auto &reference = endpoints[endpoint].reference;
    if (reference.owner == sourceReference.owner &&
        reference.ordinal == sourceReference.ordinal - temporalPePortCount) {
      feedbackSinkOption = option;
      break;
    }
  }
  if (!feedbackSinkOption)
    fail("register-FIFO fixture has no feedback-matched sink attachment");

  auto routeCosts = take(pnr::SpatialRouteCostState::create(candidate));
  pnr::SpatialNetRouterScratch router;
  requireSuccess(router.prepare(*problem));
  const std::array<pnr::PnrIndex, 1> routedNets = {selectedNet};
  requireSuccess(router.beginConstraintSweep(routedNets));
  pnr::SpatialCandidateScratch candidateScratch;
  requireSuccess(candidateScratch.prepare(*problem));
  auto externalMove = take(candidate.beginMove(candidateScratch));
  requireSuccess(routeCosts.selectLogicalNet(selectedNet));
  requireSuccess(
      externalMove.setRegisterFifoTransfer(selectedNet, std::nullopt));
  if (sinkBinding.kind == pnr::FrozenSpatialTerminalBindingKind::PortDemand)
    requireSuccess(
        externalMove.setPortAttachment(sinkBinding.index, *feedbackSinkOption));
  else
    requireSuccess(externalMove.setGraphBoundaryAttachment(
        sinkBinding.index, *feedbackSinkOption));
  take(router.routeWholeNet(
      externalMove, candidate, routeCosts, selectedNet,
      problem->config().policy().search.routing.endpointExpansionLimit));
  requireSuccess(routeCosts.acceptSelectedLogicalNet());
  requireSuccess(router.finishConstraintNet(selectedNet));
  if (!take(externalMove.close()))
    fail("external parent route closes a combinational handshake cycle");
  requireSuccess(externalMove.commit());
  if (candidate.usesRegisterFifo(selectedNet) ||
      candidate.routeTree(selectedNet).isUnrouted())
    fail("external parent transition did not materialize one routed net");
  requireSuccess(candidate.verify());
}

/// Routes the fixture's seeded register-FIFO net externally and returns the
/// net together with the option the seed had selected for it.
std::pair<pnr::PnrIndex, pnr::PnrIndex>
routeSeededRegisterFifoNetExternally(
    const pnr::FrozenSpatialPnrProblemHandle &problem,
    pnr::SpatialCandidateState &candidate) {
  pnr::PnrIndex pairedNet = pnr::getInvalidPnrIndex();
  for (pnr::PnrIndex logicalNet = 0;
       logicalNet < problem->transfers().logicalNets().size(); ++logicalNet)
    if (candidate.usesRegisterFifo(logicalNet))
      pairedNet = logicalNet;
  if (pairedNet == pnr::getInvalidPnrIndex() ||
      !take(pnr::spatialMappingViolationsAreZero(candidate)))
    fail("adoption fixture seed is not a feasible paired candidate");
  const pnr::PnrIndex pairedOption = candidate.registerFifoTransfer(pairedNet);
  routeRegisterFifoNetExternally(problem, candidate, pairedNet);
  if (!take(pnr::spatialMappingViolationsAreZero(candidate)))
    fail("external disposition did not leave a feasible routed candidate");
  return {pairedNet, pairedOption};
}

/// Routing the seeded pairing externally leaves a feasible candidate whose
/// local-transfer domain still admits it. The selected total ordering is the
/// adoption authority: with the traversal-claim measure selected, the
/// fixture's external route claims less than the pairing and the sweep
/// declines the resident alternative; with only violations selected, the
/// pairing ties the external route and the sweep adopts it under the current
/// placements without touching any other decision.
void exerciseAdmittedRegisterFifoAdoption(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &tech,
    const fabric::FabricArtifactView &fabric,
    const mapping::SpatialMappingConstraintSetView &constraints,
    const ResolvedConfig &pnrResolved) {
  const auto freeze = [&](const ResolvedConfig &config) {
    return take(pnr::freezeSpatialPnrProblem(
        dataflow, tech, fabric,
        take(pnr::projectResolvedSpatialPnrConfigView(config)), constraints));
  };
  {
    const auto problem = freeze(pnrResolved);
    pnr::SpatialPathFinderSeedWorkSummary work;
    auto seed = take(pnr::createCanonicalPathFinderSpatialSeed(problem, work));
    pnr::SpatialCandidateState &candidate = *seed.candidate;
    const auto [pairedNet, pairedOption] =
        routeSeededRegisterFifoNetExternally(problem, candidate);
    const auto externalObjective =
        take(problem->objectiveProgram().evaluate(candidate));
    pnr::SpatialAnnealingSearchScratch search;
    pnr::SpatialAnnealingStatistics statistics;
    requireSuccess(
        search.adoptAdmittedLocalTransfers(candidate, 0, statistics));
    if (statistics.localTransferAdoptionProbes != 1 ||
        statistics.adoptedLocalTransfers != 0 ||
        candidate.usesRegisterFifo(pairedNet) ||
        candidate.routeTree(pairedNet).isUnrouted())
      fail("adoption sweep adopted a pairing that worsens the selected "
           "total ordering");
    requireSuccess(candidate.verify());
    if (take(problem->objectiveProgram().evaluate(candidate)).codes() !=
        externalObjective.codes())
      fail("declined adoption changed the candidate objective");
  }
  {
    ResolvedConfig tie = pnrResolved;
    tie.dse.objectiveCatalogs.dimensions.erase(
        llvm::find_if(tie.dse.objectiveCatalogs.dimensions,
                      [](const auto &dimension) {
                        return std::holds_alternative<
                            ResolvedMappingMeasureObjectiveSource>(
                            dimension.source);
                      }),
        tie.dse.objectiveCatalogs.dimensions.end());
    tie.dse.objectiveCatalogs.weightedLevels = {{{{0, 1}, {1, 1}}}};
    const auto problem = freeze(tie);
    pnr::SpatialPathFinderSeedWorkSummary work;
    auto seed = take(pnr::createCanonicalPathFinderSpatialSeed(problem, work));
    pnr::SpatialCandidateState &candidate = *seed.candidate;
    const auto [pairedNet, pairedOption] =
        routeSeededRegisterFifoNetExternally(problem, candidate);
    const auto externalObjective =
        take(problem->objectiveProgram().evaluate(candidate));
    pnr::SpatialAnnealingSearchScratch search;
    pnr::SpatialAnnealingStatistics statistics;
    requireSuccess(
        search.adoptAdmittedLocalTransfers(candidate, 0, statistics));
    if (statistics.localTransferAdoptionProbes != 1 ||
        statistics.adoptedLocalTransfers != 1 ||
        statistics.relocatedLocalTransfers != 0 ||
        !candidate.usesRegisterFifo(pairedNet) ||
        candidate.registerFifoTransfer(pairedNet) != pairedOption ||
        !candidate.routeTree(pairedNet).isUnrouted())
      fail("adoption sweep did not restore the admitted resident pairing");
    requireSuccess(candidate.verify());
    if (take(problem->objectiveProgram().compareSelectedRank(
            take(problem->objectiveProgram().evaluate(candidate)), {},
            externalObjective, {})) != 0)
      fail("tied adoption changed the selected total ordering");
  }
}

/// Two individually acyclic pairings on two ALUs close a mutual handshake
/// cycle: the initializer pairs each chained add with its already placed
/// producer on the other ALU, so the seed carries the cycle. The witness
/// Action cuts exactly one register-FIFO disposition on the witnessed cycle,
/// routes that net externally, and leaves the other pairing in place;
/// rollback restores the cyclic selection.
void exerciseSelectedRegisterFifoHandshakeCut(
    ArtifactStore &store, mlir::MLIRContext &context,
    const fabric::FinalizedFabricRoot &fabric) {
  const ChainedAddProblem chain =
      freezeChainedAddProblem(store, context, fabric, /*addCount=*/3);
  const auto &problem = chain.problem;
  pnr::SpatialPathFinderSeedWorkSummary work;
  auto seed = take(pnr::createCanonicalPathFinderSpatialSeed(problem, work));
  pnr::SpatialCandidateState &candidate = *seed.candidate;
  std::vector<pnr::PnrIndex> cycleNets;
  for (pnr::PnrIndex logicalNet = 0;
       logicalNet < problem->transfers().logicalNets().size(); ++logicalNet)
    if (candidate.usesRegisterFifo(logicalNet) &&
        candidate.registerFifoTransferContributesToHandshakeCycle(logicalNet))
      cycleNets.push_back(logicalNet);
  if (candidate.selectedHandshakeViolation() != 1 || cycleNets.size() != 2)
    fail("alternating ALU pairings did not seed a mutual handshake cycle");
  requireSuccess(candidate.verify());

  pnr::SpatialActionDomainScratch actionDomain;
  requireSuccess(actionDomain.prepare(*problem));
  requireSuccess(actionDomain.rebuild(candidate));
  std::optional<pnr::SpatialMappingAction> witnessAction;
  for (const pnr::SpatialTransportRoutingAction &choice :
       actionDomain.view().transportChoices) {
    const auto *witness =
        std::get_if<pnr::SpatialWitnessRegionRoutingAction>(&choice);
    if (!witness ||
        witness->witnessKind !=
            ResolvedPnrViolationKind::SelectedHandshakeViolation ||
        witness->witnessOrdinal != 0)
      continue;
    witnessAction = pnr::SpatialTransportRoutingAction{*witness};
    break;
  }
  if (!witnessAction)
    fail("mutual register-FIFO cycle has no atomic handshake witness Action");

  const auto pairedNets = [&]() {
    return static_cast<int>(candidate.usesRegisterFifo(cycleNets[0])) +
           static_cast<int>(candidate.usesRegisterFifo(cycleNets[1]));
  };
  pnr::SpatialActionExecutorScratch executor;
  requireSuccess(executor.prepare(candidate));
  auto rollbackProbe = take(executor.probe(
      candidate, *witnessAction,
      pnr::SpatialActionExecutionContext::FinalClosure));
  if (rollbackProbe.isSemanticNoop() ||
      candidate.selectedHandshakeViolation() != 0 || pairedNets() != 1)
    fail("witness probe did not cut exactly one pairing of the cycle");
  requireSuccess(rollbackProbe.discard());
  if (pairedNets() != 2 || candidate.selectedHandshakeViolation() != 1)
    fail("witness rollback did not restore the mutual cycle");
  requireSuccess(candidate.verify());

  auto commitProbe = take(executor.probe(
      candidate, *witnessAction,
      pnr::SpatialActionExecutionContext::FinalClosure));
  requireSuccess(commitProbe.commit());
  const pnr::PnrIndex cut =
      candidate.usesRegisterFifo(cycleNets[0]) ? cycleNets[1] : cycleNets[0];
  if (pairedNets() != 1 || candidate.routeTree(cut).isUnrouted() ||
      candidate.selectedHandshakeViolation() != 0)
    fail("witness commit did not route exactly one cut net externally");
  requireSuccess(candidate.verify());
}

void exerciseSpatialRegisterFifoRuntimeCounterexampleExactRepair() {
  ExactRepairTemporaryDirectory directory;
  ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeExactRepairContext();

  auto dataflowArtifact = buildRegisterFifoDataflow(context);
  const auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  (void)dataflowReference;
  const auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildRegisterFifoFabric(store);
  exerciseStaticallyClosedRegisterFifoDomain(
      store, context,
      buildRegisterFifoFabric(store, /*feedbackBypassable=*/false));
  exerciseSelectedRegisterFifoHandshakeCut(
      store, context,
      buildRegisterFifoFabric(store, /*feedbackBypassable=*/false,
                              /*aluCount=*/2, /*residentContexts=*/4));

  ResolvedConfig resolved = defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  const auto techConfig =
      take(mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<::dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  auto generated = mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), techConfig, store});
  const auto *techCandidates =
      std::get_if<mapping::GeneratedTechMappings>(&generated);
  if (!techCandidates || techCandidates->candidates.size() != 1)
    fail("register-FIFO fixture did not generate one TechMapping");
  const auto tech = take(
      mapping::importTechMapping(techCandidates->candidates.front(), store));
  const auto constraints = buildSpatialMappingConstraints(
      context, dataflow, tech.view(), fabric.view(), store);

  ResolvedConfig pnrResolved = buildSpatialPnrTestResolvedConfig();
  pnrResolved.dse.spatialPnr.search.exactRepair = {
      ResolvedPnrExactRepairKind::CpSat, 256, 1024};
  const auto pnrConfig =
      take(pnr::projectResolvedSpatialPnrConfigView(pnrResolved));
  auto problem = take(pnr::freezeSpatialPnrProblem(
      dataflow, tech.view(), fabric.view(), pnrConfig, constraints.view()));
  exerciseAdmittedRegisterFifoAdoption(dataflow, tech.view(), fabric.view(),
                                       constraints.view(), pnrResolved);
  pnr::SpatialPathFinderSeedWorkSummary work;
  auto seed = take(pnr::createCanonicalPathFinderSpatialSeed(problem, work));
  pnr::PnrIndex selectedNet = pnr::getInvalidPnrIndex();
  for (pnr::PnrIndex logicalNet = 0;
       logicalNet < problem->transfers().logicalNets().size(); ++logicalNet)
    if (seed.candidate->usesRegisterFifo(logicalNet)) {
      selectedNet = logicalNet;
      break;
    }
  if (selectedNet == pnr::getInvalidPnrIndex())
    fail("real register-FIFO fixture selected no local transfer");
  const pnr::PnrIndex selectedRegisterFifoOption =
      seed.candidate->registerFifoTransfer(selectedNet);

  routeRegisterFifoNetExternally(problem, *seed.candidate, selectedNet);
  pnr::SpatialActionExecutorScratch dispositionExecutor;
  requireSuccess(dispositionExecutor.prepare(*seed.candidate));
  const pnr::SpatialMappingAction registerFifoAction =
      pnr::SpatialTransportRoutingAction{pnr::SpatialWholeNetRoutingAction{
          selectedNet, pnr::SpatialWholeNetDispositionKind::RegisterFifo,
          selectedRegisterFifoOption}};
  auto dispositionProbe = take(dispositionExecutor.probe(
      *seed.candidate, registerFifoAction,
      pnr::SpatialActionExecutionContext::FinalClosure));
  if (dispositionProbe.isSemanticNoop() ||
      !seed.candidate->usesRegisterFifo(selectedNet))
    fail("ordinary external-to-register-FIFO action did not change its net");
  requireSuccess(dispositionProbe.discard());
  if (seed.candidate->usesRegisterFifo(selectedNet) ||
      seed.candidate->routeTree(selectedNet).isUnrouted())
    fail("register-FIFO action rollback did not restore the external parent");
  requireSuccess(seed.candidate->verify());
  auto parentMapping = take(pnr::finalizeSpatialMappingCandidate(
      *seed.candidate, dataflow, tech.view(), fabric.view(), constraints.view(),
      store));
  const auto &producer =
      problem->transfers().logicalNets()[selectedNet].producer;
  const auto parentRoute =
      llvm::find_if(parentMapping.view().routeTrees(),
                    [&](const mapping::SpatialRouteTreeView &route) {
                      return route.logicalNet == producer;
                    });
  if (parentRoute == parentMapping.view().routeTrees().end())
    fail("external parent Mapping lost its selected local-transfer net");
  std::optional<fabric::FabricPhysicalTraversalRef> traversal;
  if (parentRoute->localTraversal)
    traversal = parentRoute->localTraversal;
  for (const auto &node : parentRoute->nodes)
    if (!traversal && node.incomingTraversal)
      traversal = node.incomingTraversal;
  for (const auto &sink : parentRoute->sinks)
    if (!traversal && sink.localTraversal)
      traversal = sink.localTraversal;
  if (!traversal)
    fail("external parent route selected no traversal to forbid");

  const RepairEnvironment environment{
      dataflow,
      tech.view(),
      fabric.view(),
      constraints,
      parentMapping,
      pnrConfig,
      store,
      [&](const mapping::SpatialMappingConstraintSetView &candidate) {
        return pnr::freezeSpatialPnrProblem(
            dataflow, tech.view(), fabric.view(), pnrConfig, candidate);
      }};
  const mapping::SpatialNoGoodLiteral routeLiteral =
      mapping::SpatialNetUsesTraversalLiteral{producer, std::nullopt,
                                              *traversal};
  const RepairedMapping child = repairDeterministically(
      "disposition.external-to-register-fifo", environment, {routeLiteral});
  const bool selectedRegisterFifo = llvm::any_of(
      child.mapping.view().registerFifoTransfers(),
      [&](const mapping::SpatialRegisterFifoTransferView &transfer) {
        return transfer.logicalNet == producer;
      });
  if (!selectedRegisterFifo)
    fail("exact repair did not use the real register-FIFO breaker");
  requireLiteralBroken("disposition.external-to-register-fifo", environment,
                       routeLiteral, child.mapping.view());
}

} // namespace loom::test
