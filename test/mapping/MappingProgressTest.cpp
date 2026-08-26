#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "mapping progress test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

loom::ArtifactIdentity identity(std::uint8_t value) {
  std::array<std::uint8_t, loom::ArtifactIdentity::byteSize> bytes{};
  bytes.fill(value);
  return take(loom::ArtifactIdentity::fromBytes(bytes));
}

void sharedFiniteFifoRequiresRecurrenceProof(
    const loom::mapping::FrozenMappingProgressModel &model,
    loom::mapping::MappingProgressProjection projection) {
  const loom::ArtifactIdentity owner = identity(31);
  const dataflow::CanonicalGraphProducerEndpointRef firstProducer =
      dataflow::ActorTokenResultRef{
          dataflow::ActorRef{owner, dataflow::ActorId(1)}, 0};
  const dataflow::CanonicalGraphProducerEndpointRef secondProducer =
      dataflow::ActorTokenResultRef{
          dataflow::ActorRef{owner, dataflow::ActorId(2)}, 0};
  const dataflow::CanonicalGraphConsumerEndpointRef secondConsumer =
      dataflow::ActorTokenOperandRef{
          dataflow::ActorRef{owner, dataflow::ActorId(3)}, 0};
  const loom::fabric::FabricTransportEndpointRef endpoint{
      loom::fabric::FabricTransportEndpointOwnerRef::of(
          loom::fabric::FabricFuOccurrenceRef(1)),
      0};
  const auto buffered = loom::fabric::FabricPhysicalTraversalRef::fifoTraversal(
      loom::fabric::FabricFifoOccurrenceRef(7),
      loom::fabric::FabricFifoTraversalMode::Buffered);
  const auto otherBuffered =
      loom::fabric::FabricPhysicalTraversalRef::fifoTraversal(
          loom::fabric::FabricFifoOccurrenceRef(8),
          loom::fabric::FabricFifoTraversalMode::Buffered);
  const auto bypass = loom::fabric::FabricPhysicalTraversalRef::fifoTraversal(
      loom::fabric::FabricFifoOccurrenceRef(7),
      loom::fabric::FabricFifoTraversalMode::Bypass);
  std::vector<loom::mapping::SpatialRouteTreeView> routes;
  routes.push_back(
      {firstProducer, endpoint, buffered, /*nodes=*/{}, /*sinks=*/{}});
  if (!loom::mapping::projectSpatialFiniteBufferRecurrence(routes).established)
    fail("one logical net did not establish finite FIFO independence");
  routes.push_back({secondProducer, endpoint, std::nullopt,
                    /*nodes=*/{},
                    /*sinks=*/{{secondConsumer, 0, buffered}}});
  if (!loom::mapping::spatialRouteTreeSelectsTraversal(routes.front(),
                                                       buffered) ||
      !loom::mapping::spatialRouteTreeSelectsTraversal(routes.back(), buffered))
    fail("complete RouteTree traversal domain omitted a local selection");

  const auto recurrence =
      loom::mapping::projectSpatialFiniteBufferRecurrence(routes);
  if (recurrence.kind != loom::mapping::MappingRouteProgressObligationKind::
                             FiniteBufferRecurrence ||
      recurrence.established)
    fail("shared finite FIFO recurrence was treated as established");
  routes.back().sinks.front().localTraversal = otherBuffered;
  if (!loom::mapping::projectSpatialFiniteBufferRecurrence(routes).established)
    fail("distinct finite FIFO owners were treated as shared");
  routes.back().sinks.front().localTraversal = bypass;
  if (!loom::mapping::projectSpatialFiniteBufferRecurrence(routes).established)
    fail("bypass traversal was treated as an active finite queue");
  projection.routeObligations = {recurrence};
  const auto closure =
      take(loom::mapping::deriveMappingProgressClosure(model, projection));
  if (closure.kind !=
          loom::mapping::MappingProgressClosureKind::ProofNotEstablished ||
      closure.reason != loom::mapping::MappingProgressClosureReason::
                            FiniteBufferRecurrenceNotEstablished)
    fail("shared finite FIFO produced a general liveness proof");
  projection.routeObligations.push_back(
      {loom::mapping::MappingRouteProgressObligationKind::
           DurableBoundaryAfterDivergence,
       false});
  const auto mixedClosure =
      take(loom::mapping::deriveMappingProgressClosure(model, projection));
  if (mixedClosure.kind !=
          loom::mapping::MappingProgressClosureKind::ProvenClosedWaitSet ||
      mixedClosure.reason !=
          loom::mapping::MappingProgressClosureReason::MissingDurableBoundary)
    fail("proven closed wait was hidden by an incomplete recurrence");
}

void initializedFeedbackProgressBasis() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @feedback(%start: none, %phase: i1) -> ()
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %stable = dataflow.invariant %phase, %start : none
    %carried = dataflow.carry %phase, %start, %lanes#1 : none
    %lanes:2 = dataflow.demux %phase, %carried
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams() memories()
        complete(%lanes#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %phase: i1) ctrl (%start: none) {
    %done = dataflow.graph.launch @feedback deps(%start) values(%phase)
        stream_inputs() memories() stream_outputs()
        : (none, i1) -> none
    dataflow.thread.yield %done : none
  }
  func.func private @host(%phase: i1) {
    %completion = dataflow.thread.launch @worker(%phase)
        : (i1) -> !dataflow.thread_token
    return
  }
}

)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse cyclic progress fixture");
  auto artifact = take(dataflow::finalizeCanonicalDataflow(*module));
  const auto view = take(artifact.view());
  const auto uncovered = take(loom::mapping::deriveMappingDataflowProgressBasis(
      view, /*coveredGraphs=*/{}));
  if (uncovered.kind !=
      loom::mapping::MappingDataflowProgressBasisKind::Acyclic)
    fail("progress analysis inspected a graph outside its covered set");
  const auto model =
      take(loom::mapping::freezeMappingProgressModel(view, /*events=*/{}));
  loom::mapping::MappingProgressProjection projection;
  projection.basis = uncovered;
  projection.routeObligations.push_back(
      {loom::mapping::MappingRouteProgressObligationKind::
           DurableBoundaryAfterDivergence,
       false});
  if (take(loom::mapping::deriveMappingProgressClosure(model, projection))
          .kind !=
      loom::mapping::MappingProgressClosureKind::ProvenClosedWaitSet)
    fail("post-divergence route without a durable boundary passed progress");
  projection.routeObligations.front().established = true;
  if (take(loom::mapping::deriveMappingProgressClosure(model, projection))
          .kind !=
      loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet)
    fail("post-divergence durable boundary did not close route progress");
  sharedFiniteFifoRequiresRecurrenceProof(model, projection);
  const std::array<dataflow::GraphRef, 1> covered = {view.graphs().front().ref};
  const auto basis =
      take(loom::mapping::deriveMappingDataflowProgressBasis(view, covered));
  if (basis.kind !=
      loom::mapping::MappingDataflowProgressBasisKind::InitializedFeedback)
    fail("typed initialized feedback did not produce its progress basis");
  projection.basis = basis;
  if (take(loom::mapping::deriveMappingProgressClosure(model, projection))
          .kind !=
      loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet)
    fail("durable initialized feedback did not close progress");

  std::vector<dataflow::RootedGraphLaunchRef> launches;
  view.forEachRootedGraphLaunch([&](dataflow::RootedGraphLaunchRef launch) {
    launches.push_back(launch);
  });
  if (launches.size() != 1)
    fail("initialized feedback fixture has no unique rooted launch");
  std::optional<dataflow::ActorRef> carry;
  std::optional<dataflow::ActorRef> demux;
  for (const dataflow::CanonicalActorView &actor : view.actors()) {
    const llvm::StringRef name = actor.op->getName().getStringRef();
    if (name == "dataflow.carry")
      carry = actor.ref;
    else if (name == "dataflow.demux")
      demux = actor.ref;
  }
  if (!carry || !demux)
    fail("initialized feedback fixture lost its actor pair");
  const auto transition = [&](dataflow::ActorRef actor) {
    return dataflow::EventFamilyKey(dataflow::ContextualActorTransitionEventRef{
        dataflow::ContextualActorRef{launches.front(), actor}, 0});
  };
  const auto carryTransition = transition(*carry);
  const auto demuxTransition = transition(*demux);
  const auto eventModel = take(loom::mapping::freezeMappingProgressModel(
      view, {carryTransition, demuxTransition}));
  loom::mapping::MappingProgressProjection eventProjection;
  eventProjection.basis = basis;
  eventProjection.routeObligations.push_back(
      {loom::mapping::MappingRouteProgressObligationKind::
           DurableBoundaryAfterDivergence,
       true});
  eventProjection.capacityCells.push_back({1, 0});
  const loom::mapping::InstructionExecutionContextKey contextKey{
      loom::fabric::AccCoreOccurrenceRef{}};
  eventProjection.resourceActivations.push_back(
      {contextKey,
       launches.front().rootThreadLaunch,
       {loom::mapping::SystemPresburgerCell{}},
       {carryTransition},
       {{0, 1}},
       {{{carryTransition}}},
       {"feedback-event-causality", 0,
        loom::mapping::MappingResourceGrantPolicyKind::None}});
  eventProjection.resourceActivations.push_back(
      {contextKey,
       launches.front().rootThreadLaunch,
       {loom::mapping::SystemPresburgerCell{}},
       {demuxTransition},
       {{0, 1}},
       {},
       {"feedback-event-causality", 0,
        loom::mapping::MappingResourceGrantPolicyKind::None}});
  if (take(loom::mapping::deriveMappingProgressClosure(eventModel,
                                                       eventProjection))
          .kind !=
      loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet)
    fail("initialized feedback formed false same-coordinate event causality");
  eventProjection.resourceActivations.front()
      .causalRelease.front()
      .alternatives.push_back(demuxTransition);
  if (take(loom::mapping::deriveMappingProgressClosure(eventModel,
                                                       eventProjection))
          .kind !=
      loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet)
    fail("a satisfied release alternative extended the holder interval");

  projection.basis.kind =
      loom::mapping::MappingDataflowProgressBasisKind::Cyclic;
  if (take(loom::mapping::deriveMappingProgressClosure(model, projection))
          .kind !=
      loom::mapping::MappingProgressClosureKind::ProofNotEstablished)
    fail("an unsupported actor cycle did not fail closed");
}

void completionFrontierRequiresReadyRoots() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.thread private @worker_a domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    dataflow.thread.yield
  }
  dataflow.thread private @worker_b domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    dataflow.thread.yield
  }
  dataflow.thread private @worker_c domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    dataflow.thread.yield
  }
  dataflow.thread private @worker_d domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    dataflow.thread.yield
  }
  func.func private @host() {
    %a = dataflow.thread.launch @worker_a()
        : () -> !dataflow.thread_token
    %b = dataflow.thread.launch @worker_b() wait(%a)
        : () -> !dataflow.thread_token
    %c = dataflow.thread.launch @worker_c() wait(%b)
        : () -> !dataflow.thread_token
    %d = dataflow.thread.launch @worker_d()
        : () -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse completion-frontier fixture");
  auto artifact = take(dataflow::finalizeCanonicalDataflow(*module));
  const auto view = take(artifact.view());
  std::vector<dataflow::RootThreadLaunchRef> roots;
  std::vector<dataflow::EventFamilyKey> events;
  for (const auto &root : view.rootThreadLaunches()) {
    roots.push_back(root.ref);
    events.push_back(dataflow::rootThreadStartEventFamily(root.ref));
    events.push_back(dataflow::rootThreadCompletionEventFamily(root.ref));
  }
  if (roots.size() != 4)
    fail("completion-frontier fixture lost a root launch");
  const auto model =
      take(loom::mapping::freezeMappingProgressModel(view, events));
  std::optional<dataflow::RootThreadLaunchRef> first;
  std::optional<dataflow::RootThreadLaunchRef> middle;
  std::optional<dataflow::RootThreadLaunchRef> last;
  for (const auto lhs : roots)
    for (const auto candidate : roots) {
      if (lhs == candidate ||
          !take(loom::mapping::mappingEventPrecedes(
              model, dataflow::rootThreadCompletionEventFamily(lhs),
              dataflow::rootThreadStartEventFamily(candidate))))
        continue;
      for (const auto rhs : roots) {
        if (rhs == lhs || rhs == candidate ||
            !take(loom::mapping::mappingEventPrecedes(
                model, dataflow::rootThreadCompletionEventFamily(candidate),
                dataflow::rootThreadStartEventFamily(rhs))))
          continue;
        first = lhs;
        middle = candidate;
        last = rhs;
      }
    }
  if (!first || !middle || !last)
    fail("completion-frontier fixture lost its causal chain");
  const auto independentRoot = llvm::find_if(roots, [&](const auto root) {
    return root != *first && root != *middle && root != *last;
  });
  if (independentRoot == roots.end())
    fail("completion-frontier fixture lost its independent root");
  const auto independent = *independentRoot;
  const auto admissible =
      [&](llvm::ArrayRef<dataflow::RootThreadLaunchRef> scope,
          llvm::ArrayRef<dataflow::RootThreadLaunchRef> completed,
          dataflow::RootThreadLaunchRef completing,
          llvm::ArrayRef<dataflow::RootThreadLaunchRef> active) {
        return take(loom::mapping::mappingCompletionFrontierIsAdmissible(
            view, scope, completed, completing, active));
      };
  const std::array chain = {*first, *middle, *last};
  const std::array firstSuccessor = {*middle};
  if (!admissible(chain, {}, *first, firstSuccessor))
    fail("foreign canonical root changed the exact Mapping frontier");
  const std::array sparseChainScope = {*first, *last};
  const std::array lastSuccessor = {*last};
  if (admissible(sparseChainScope, {}, *first, lastSuccessor))
    fail("frontier ignored a scope-external intermediate predecessor");
  const std::array foreignDirectScope = {independent, *last};
  if (admissible(foreignDirectScope, {}, independent, lastSuccessor))
    fail("frontier ignored a scope-external direct predecessor");
  const std::array concurrentReady = {*middle, independent};
  if (!admissible(roots, {}, *first, concurrentReady))
    fail("independent ready root was rejected at a resource boundary");
  const std::array independentlyCompleted = {independent};
  if (!admissible(roots, independentlyCompleted, *first, firstSuccessor))
    fail("independently serialized completion was rejected");
  const std::array skippedSuccessor = {*last};
  if (admissible(chain, {}, *first, skippedSuccessor))
    fail("frontier activated a root before its direct predecessor");
  const std::array mixedSuccessors = {*middle, *last};
  if (admissible(chain, {}, *first, mixedSuccessors))
    fail("frontier mixed ready and unready active roots");
  const std::array incompletePrefix = {*first};
  if (admissible(chain, incompletePrefix, *last, {}))
    fail("completing root skipped an unfinished predecessor");
  const std::array completePrefix = {*first, *middle};
  if (!admissible(chain, completePrefix, *last, {}))
    fail("complete causal prefix was rejected");

  auto externalDependencyModule = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.thread private @ready domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    dataflow.thread.yield
  }
  dataflow.thread private @externally_blocked
      domain(#dataflow.thread_domain<dense>)() ctrl (%ctrl: none) {
    dataflow.thread.yield
  }
  func.func private @host(%external: !dataflow.thread_token) {
    %ready = dataflow.thread.launch @ready()
        : () -> !dataflow.thread_token
    %blocked = dataflow.thread.launch @externally_blocked() wait(%external)
        : () -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                                  &context);
  if (!externalDependencyModule)
    fail("cannot parse external-dependency fixture");
  auto externalArtifact =
      take(dataflow::finalizeCanonicalDataflow(*externalDependencyModule));
  const auto externalView = take(externalArtifact.view());
  if (externalView.rootThreadLaunches().size() != 2)
    fail("external-dependency fixture lost a root launch");
  std::optional<dataflow::RootThreadLaunchRef> externallyBlocked;
  std::optional<dataflow::RootThreadLaunchRef> externallyReady;
  std::vector<dataflow::RootThreadLaunchRef> externalRoots;
  for (const auto &root : externalView.rootThreadLaunches()) {
    externalRoots.push_back(root.ref);
    auto launch = mlir::cast<dataflow::ThreadLaunchOp>(root.op);
    if (llvm::any_of(launch.getAsyncDependencies(), [](mlir::Value dependency) {
          return !dependency.getDefiningOp<dataflow::ThreadLaunchOp>();
        }))
      externallyBlocked = root.ref;
    else
      externallyReady = root.ref;
  }
  if (!externallyBlocked || !externallyReady)
    fail("external-dependency fixture lost its dependency classification");
  const std::array blockedAfter = {*externallyBlocked};
  if (take(loom::mapping::mappingCompletionFrontierIsAdmissible(
          externalView, externalRoots, {}, *externallyReady, blockedAfter)))
    fail("frontier activated a root with an unproved external dependency");

  auto storedWaitModule = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.thread private @waited domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    dataflow.thread.yield
  }
  dataflow.thread private @independent domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    dataflow.thread.yield
  }
  dataflow.thread private @after_wait domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    dataflow.thread.yield
  }
  func.func private @host() {
    %waited = dataflow.thread.launch @waited()
        : () -> !dataflow.thread_token
    %independent = dataflow.thread.launch @independent()
        : () -> !dataflow.thread_token
    dataflow.thread.wait %waited : !dataflow.thread_token
    %after = dataflow.thread.launch @after_wait()
        : () -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                          &context);
  if (!storedWaitModule)
    fail("cannot parse stored-wait fixture");
  auto storedWaitArtifact =
      take(dataflow::finalizeCanonicalDataflow(*storedWaitModule));
  const auto storedWaitView = take(storedWaitArtifact.view());
  std::optional<dataflow::ThreadWaitOp> storedWait;
  storedWaitArtifact.module()->walk([&](dataflow::ThreadWaitOp wait) {
    if (storedWait)
      fail("stored-wait fixture has multiple waits");
    storedWait = wait;
  });
  if (!storedWait || storedWait->getAsyncDependencies().size() != 1)
    fail("stored-wait fixture lost its wait operation");
  mlir::Operation *waitedLaunch =
      storedWait->getAsyncDependencies().front().getDefiningOp();
  std::vector<dataflow::RootThreadLaunchRef> storedWaitRoots;
  std::optional<dataflow::RootThreadLaunchRef> independentBeforeWait;
  std::optional<dataflow::RootThreadLaunchRef> afterWait;
  for (const auto &root : storedWaitView.rootThreadLaunches()) {
    storedWaitRoots.push_back(root.ref);
    auto launch = mlir::cast<dataflow::ThreadLaunchOp>(root.op);
    if (launch.getOperation() == waitedLaunch)
      continue;
    if (launch->getBlock() != storedWait->getOperation()->getBlock())
      fail("stored-wait fixture split its host block");
    if (launch->isBeforeInBlock(storedWait->getOperation()))
      independentBeforeWait = root.ref;
    else
      afterWait = root.ref;
  }
  if (!independentBeforeWait || !afterWait)
    fail("stored-wait fixture lost its launch identities");
  const std::array afterStoredWait = {*afterWait};
  if (take(loom::mapping::mappingCompletionFrontierIsAdmissible(
          storedWaitView, storedWaitRoots, {}, *independentBeforeWait,
          afterStoredWait)))
    fail("frontier ignored an explicit stored-program wait");
}

void orderedRuntimeHeadsRequireCompleteExactPairing() {
  const loom::fabric::FabricPeOccurrenceRef pe(7);
  const loom::fabric::FabricFuOccurrenceRef fu(11);
  const loom::fabric::InstructionContextRef context{pe, 2};
  const llvm::APInt tag(4, 3);
  loom::mapping::SpatialPeOperandProgressFeedback projection;
  projection.status = loom::mapping::SpatialPeOperandProgressStatus::Safe;
  projection.support = loom::mapping::SpatialPeOperandProgressSupport::Exact;
  projection.groupCount = 1;
  projection.pairingKeyCount = 2;
  projection.distinctPairingKeyCount = 1;
  projection.pairingKeys.push_back({context, fu, tag});
  projection.pairings.push_back({{context, fu, tag}, {0, 1}, {}, {0, 1}});
  constexpr std::array<std::uint8_t, 1> descriptor{1};
  constexpr std::array<std::uint8_t, 1> view{2};
  projection.projectionDigest =
      take(loom::computeComponentViewDigest(descriptor, view));
  std::vector<loom::mapping::SpatialPeOperandRuntimeHeadView> heads{
      {{context, 0, 0}, fu, tag, 0, 2, 1, 0, 10, 4, 7, true},
      {{context, 0, 1}, fu, tag, 1, 2, 1, 0, 11, 5, 7, true}};
  const auto exact = take(
      loom::mapping::deriveSpatialPeOperandRuntimeWitness(projection, heads));
  if (exact.status !=
          loom::mapping::SpatialPeOperandRuntimeWitnessStatus::Exact ||
      exact.matchedPairingKeyCount != 1 ||
      exact.unmatchedPairingKeyCount != 0 || !exact.projectionDigest)
    fail("complete ordered queue heads did not produce an exact witness");

  std::reverse(heads.begin(), heads.end());
  const auto permuted = take(
      loom::mapping::deriveSpatialPeOperandRuntimeWitness(projection, heads));
  if (!permuted.projectionDigest ||
      *permuted.projectionDigest != *exact.projectionDigest)
    fail("runtime head witness digest depends on observation order");

  heads.front().exactHead = false;
  const auto incomplete = take(
      loom::mapping::deriveSpatialPeOperandRuntimeWitness(projection, heads));
  if (incomplete.status ==
      loom::mapping::SpatialPeOperandRuntimeWitnessStatus::Exact)
    fail("an incomplete ordered head was classified as exact");

  heads.front().exactHead = true;
  heads.front().headProducerSequenceOrdinal = 8;
  const auto mismatch = take(
      loom::mapping::deriveSpatialPeOperandRuntimeWitness(projection, heads));
  if (mismatch.status ==
          loom::mapping::SpatialPeOperandRuntimeWitnessStatus::Exact ||
      mismatch.mismatchedHeadCount == 0)
    fail("mismatched ordered heads were classified as exact");

  // Potential pairings that are currently empty are dormant, not missing
  // runtime observations. They must not prevent an active exact tuple from
  // reaching the liveness verifier.
  auto dormantProjection = projection;
  const loom::fabric::InstructionContextRef dormantContext{pe, 3};
  dormantProjection.pairings.push_back(
      {{dormantContext, fu, llvm::APInt(4, 9)}, {0, 1}, {}, {0, 1}});
  auto dormantHeads = std::vector<loom::mapping::SpatialPeOperandRuntimeHeadView>{
      {{context, 0, 0}, fu, tag, 0, 2, 1, 0, 10, 4, 7, true},
      {{context, 0, 1}, fu, tag, 1, 2, 1, 0, 11, 5, 7, true},
      {{dormantContext, 0, 0}, fu, llvm::APInt(4, 9), 0, 2, 0, 0,
       std::numeric_limits<std::uint64_t>::max(),
       std::numeric_limits<std::uint64_t>::max(),
       std::numeric_limits<std::uint64_t>::max(), true},
      {{dormantContext, 0, 1}, fu, llvm::APInt(4, 9), 1, 2, 0, 0,
       std::numeric_limits<std::uint64_t>::max(),
       std::numeric_limits<std::uint64_t>::max(),
       std::numeric_limits<std::uint64_t>::max(), true}};
  const auto dormant = take(loom::mapping::deriveSpatialPeOperandRuntimeWitness(
      dormantProjection, dormantHeads));
  if (dormant.status !=
          loom::mapping::SpatialPeOperandRuntimeWitnessStatus::Exact ||
      dormant.matchedPairingKeyCount != 1 ||
      dormant.unmatchedPairingKeyCount != 0 || dormant.exactHeadCount != 4)
    fail("dormant empty pairing was treated as an incomplete head");
}

} // namespace

int main() {
  initializedFeedbackProgressBasis();
  completionFrontierRequiresReadyRoots();
  orderedRuntimeHeadsRequireCompleteExactPairing();
  llvm::outs() << "mapping progress tests passed\n";
  return EXIT_SUCCESS;
}
