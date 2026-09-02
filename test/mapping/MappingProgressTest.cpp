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

#include "llvm/ADT/STLExtras.h"
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

using loom::mapping::MappingBufferDependencyEdge;
using loom::mapping::MappingBufferDependencyEdgeKind;
using loom::mapping::MappingStaticQueueClass;
using loom::mapping::MappingStaticQueueClassKind;
using loom::mapping::MappingStaticWaitNode;
using loom::mapping::MappingStorageQueueProgressNode;

MappingStaticWaitNode globalQueueNode(std::uint64_t fifoId) {
  return MappingStorageQueueProgressNode{
      loom::fabric::FabricFifoOccurrenceRef(fifoId),
      MappingStaticQueueClass{MappingStaticQueueClassKind::Global,
                              llvm::APInt(1, 0)}};
}

MappingStaticWaitNode tagQueueNode(std::uint64_t fifoId, std::uint64_t tag) {
  return MappingStorageQueueProgressNode{
      loom::fabric::FabricFifoOccurrenceRef(fifoId),
      MappingStaticQueueClass{MappingStaticQueueClassKind::PhysicalTag,
                              llvm::APInt(4, tag)}};
}

MappingStaticWaitNode actorNode(std::uint64_t actorId) {
  return dataflow::ActorRef{identity(31), dataflow::ActorId(actorId)};
}

MappingBufferDependencyEdge
waitEdge(MappingStaticWaitNode from, MappingStaticWaitNode to,
         MappingBufferDependencyEdgeKind kind) {
  return MappingBufferDependencyEdge{std::move(from), std::move(to), kind, 0,
                                     std::nullopt};
}

/// The four-channel head-of-line fixture: a strict FIFO queue shared by two
/// nets whose consumers each forward through a second shared queue. Under one
/// global queue class the order/join edges close a cycle; splitting the shared
/// owner into per-tag classes breaks it.
struct HolFixture final {
  // Channel topology: net1 P1->C2 and net2 P2->C1 through queue 7;
  // netX C1->C2 and netY P3->C2 through queue 8.
  std::vector<MappingBufferDependencyEdge> strictEdges() const {
    const MappingStaticWaitNode q = globalQueueNode(7);
    const MappingStaticWaitNode q2 = globalQueueNode(8);
    return {
        waitEdge(actorNode(2), q, MappingBufferDependencyEdgeKind::
                                      ActorInputJoin), // C2 joins net2
        waitEdge(q, actorNode(1),
                 MappingBufferDependencyEdgeKind::
                     ActorInputJoin), // queue head release waits on C1
        waitEdge(actorNode(1), q2, MappingBufferDependencyEdgeKind::
                                       OutputCausalRelease), // C1 releases netX
        waitEdge(q2, actorNode(2),
                 MappingBufferDependencyEdgeKind::
                     ActorInputJoin), // queue 8 head release waits on C2
    };
  }

  // The same topology with queue 7 split into one class per resident tag
  // value: net1 carries tag 3 and net2 carries tag 5.
  std::vector<MappingBufferDependencyEdge> virtualChannelEdges() const {
    const MappingStaticWaitNode qT1 = tagQueueNode(7, 3);
    const MappingStaticWaitNode qT2 = tagQueueNode(7, 5);
    const MappingStaticWaitNode q2 = globalQueueNode(8);
    return {
        waitEdge(actorNode(2), qT2, MappingBufferDependencyEdgeKind::
                                        ActorInputJoin), // C2 joins net2
        waitEdge(qT1, actorNode(1),
                 MappingBufferDependencyEdgeKind::
                     ActorInputJoin), // tag-3 channel head release waits on C1
        waitEdge(actorNode(1), q2, MappingBufferDependencyEdgeKind::
                                       OutputCausalRelease), // C1 releases netX
        waitEdge(q2, actorNode(2),
                 MappingBufferDependencyEdgeKind::
                     ActorInputJoin), // queue 8 head release waits on C2
    };
  }

  // The same virtual-channel topology with one shared tag value: both nets
  // occupy one class, so the order cycle closes again.
  std::vector<MappingBufferDependencyEdge> sameTagEdges() const {
    const MappingStaticWaitNode q = tagQueueNode(7, 3);
    const MappingStaticWaitNode q2 = globalQueueNode(8);
    return {
        waitEdge(actorNode(2), q, MappingBufferDependencyEdgeKind::
                                      ActorInputJoin), // C2 joins net2
        waitEdge(q, actorNode(1),
                 MappingBufferDependencyEdgeKind::
                     ActorInputJoin), // queue head release waits on C1
        waitEdge(actorNode(1), q2, MappingBufferDependencyEdgeKind::
                                       OutputCausalRelease), // C1 releases netX
        waitEdge(q2, actorNode(2),
                 MappingBufferDependencyEdgeKind::
                     ActorInputJoin), // queue 8 head release waits on C2
    };
  }
};

void bufferDependencyClosure() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @trivial(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return values() streams() memories()
        complete(%start : none)
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse buffer-dependency fixture");
  auto artifact = take(dataflow::finalizeCanonicalDataflow(*module));
  const auto view = take(artifact.view());
  const auto model =
      take(loom::mapping::freezeMappingProgressModel(view, /*events=*/{}));
  loom::mapping::MappingProgressProjection projection;
  projection.basis.kind =
      loom::mapping::MappingDataflowProgressBasisKind::Acyclic;
  const HolFixture fixture;

  const auto closureOf = [&](std::vector<MappingBufferDependencyEdge> edges) {
    loom::mapping::MappingProgressProjection candidate = projection;
    candidate.bufferDependencyEdges = std::move(edges);
    return take(
        loom::mapping::deriveMappingProgressClosure(model, candidate));
  };
  const auto provenCycle = [](const loom::mapping::MappingProgressClosure
                                  &closure) {
    return closure.kind ==
               loom::mapping::MappingProgressClosureKind::ProvenClosedWaitSet &&
           closure.reason == loom::mapping::MappingProgressClosureReason::
                                 ClosedBufferDependencyCycle;
  };
  const auto notEstablished = [](const loom::mapping::MappingProgressClosure
                                     &closure) {
    return closure.kind ==
               loom::mapping::MappingProgressClosureKind::ProofNotEstablished &&
           closure.reason == loom::mapping::MappingProgressClosureReason::
                                 BufferDependencyNotEstablished;
  };
  const auto capacityNotEstablished =
      [](const loom::mapping::MappingProgressClosure &closure) {
        return closure.kind == loom::mapping::MappingProgressClosureKind::
                                   ProofNotEstablished &&
               closure.reason ==
                   loom::mapping::MappingProgressClosureReason::
                       ReconvergentCapacityNotEstablished;
      };

  // A cross-tag order cycle through one shared strict queue is a proven
  // closed wait.
  const auto strict = closureOf(fixture.strictEdges());
  if (!provenCycle(strict))
    fail("shared strict queue order cycle was not proven");
  if (strict.bufferDependencyCycle.size() != 4)
    fail("proven buffer-dependency cycle lost its exact component");
  for (const MappingStaticWaitNode &node :
       {globalQueueNode(7), globalQueueNode(8), actorNode(1), actorNode(2)})
    if (!llvm::is_contained(strict.bufferDependencyCycle, node))
      fail("proven buffer-dependency cycle omitted a component member");

  // The same topology with per-tag classes does not couple the two nets.
  if (closureOf(fixture.virtualChannelEdges()).kind !=
      loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet)
    fail("virtual-channel classes did not break the cross-tag order cycle");

  // One shared tag value couples the nets again.
  if (!provenCycle(closureOf(fixture.sameTagEdges())))
    fail("same-tag virtual-channel order cycle was not proven");

  // The capacity control: per-tag classes remove the order cycle, but the
  // two nets still share one physical slot pool. A proven minimum above the
  // selected pool is a closed wait by itself; a sufficient pool keeps the
  // virtual-channel liveness verdict.
  const auto virtualChannelPool = [&](std::uint64_t selectedCapacity) {
    loom::mapping::MappingProgressProjection candidate = projection;
    candidate.bufferDependencyEdges = fixture.virtualChannelEdges();
    candidate.reconvergentCapacityObligations = {
        loom::mapping::MappingReconvergentCapacityObligation{
            loom::fabric::FabricFifoOccurrenceRef(7),
            {MappingStaticQueueClass{MappingStaticQueueClassKind::PhysicalTag,
                                     llvm::APInt(4, 3)},
             MappingStaticQueueClass{MappingStaticQueueClassKind::PhysicalTag,
                                     llvm::APInt(4, 5)}},
            {},
            selectedCapacity,
            2,
            loom::mapping::MappingReconvergentCapacityProofKind::Proven}};
    return take(
        loom::mapping::deriveMappingProgressClosure(model, candidate));
  };
  const auto undersized = virtualChannelPool(1);
  if (undersized.kind !=
          loom::mapping::MappingProgressClosureKind::ProvenClosedWaitSet ||
      undersized.reason != loom::mapping::MappingProgressClosureReason::
                               ReconvergentCapacityShortfall ||
      undersized.capacityShortfall != 1)
    fail("an undersized virtual-channel pool was not a proven closed wait");
  if (virtualChannelPool(2).kind !=
      loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet)
    fail("a sufficient virtual-channel pool lost its liveness verdict");

  // A cycle carrying a capacity edge is mediated by the capacity proof: its
  // members carry no obligations here, so it stays unestablished.
  const std::vector<MappingBufferDependencyEdge> capacityCycle{
      waitEdge(globalQueueNode(7), globalQueueNode(8),
               MappingBufferDependencyEdgeKind::DownstreamCapacity),
      waitEdge(globalQueueNode(8), globalQueueNode(7),
               MappingBufferDependencyEdgeKind::DownstreamCapacity)};
  if (!capacityNotEstablished(closureOf(capacityCycle)))
    fail("capacity-only queue cycle was reported without a capacity proof");

  // A capacity edge inside an order cycle also stays unestablished.
  std::vector<MappingBufferDependencyEdge> mixed = fixture.strictEdges();
  mixed.push_back(waitEdge(globalQueueNode(7), globalQueueNode(8),
                           MappingBufferDependencyEdgeKind::DownstreamCapacity));
  if (!capacityNotEstablished(closureOf(mixed)))
    fail("capacity-carrying order cycle was reported without a capacity proof");

  // A component with a wait leaving it is not closed and never becomes a
  // witness.
  std::vector<MappingBufferDependencyEdge> open = fixture.strictEdges();
  open.push_back(waitEdge(actorNode(1), actorNode(3),
                          MappingBufferDependencyEdgeKind::ActorInputJoin));
  if (closureOf(open).kind !=
      loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet)
    fail("an open component was reported as a closed wait");

  // An indeterminate construction is unestablished, never a proven cycle.
  loom::mapping::MappingProgressProjection indeterminate = projection;
  indeterminate.bufferDependencyEdges = std::nullopt;
  if (!notEstablished(take(loom::mapping::deriveMappingProgressClosure(
          model, indeterminate))))
    fail("an indeterminate construction was not reported as unestablished");

  // The route-obligation family keeps precedence: an unestablished finite
  // buffer recurrence still wins over a proven buffer-dependency cycle.
  loom::mapping::MappingProgressProjection shared = projection;
  shared.bufferDependencyEdges = fixture.strictEdges();
  shared.routeObligations.push_back(
      {loom::mapping::MappingRouteProgressObligationKind::
           FiniteBufferRecurrence,
       false});
  const auto sharedClosure =
      take(loom::mapping::deriveMappingProgressClosure(model, shared));
  if (sharedClosure.kind !=
          loom::mapping::MappingProgressClosureKind::ProofNotEstablished ||
      sharedClosure.reason !=
          loom::mapping::MappingProgressClosureReason::
              FiniteBufferRecurrenceNotEstablished)
    fail("finite buffer recurrence lost its precedence over the buffer "
         "dependency closure");
}

void reconvergentCapacityClosure() {
  using loom::mapping::MappingReconvergentCapacityObligation;
  using loom::mapping::MappingReconvergentCapacityProofKind;
  const auto globalObligation = [](std::uint64_t fifo, std::uint64_t selected,
                                   std::optional<std::uint64_t> minimum,
                                   MappingReconvergentCapacityProofKind kind) {
    return MappingReconvergentCapacityObligation{
        loom::fabric::FabricFifoOccurrenceRef(fifo),
        {MappingStaticQueueClass{MappingStaticQueueClassKind::Global,
                                 llvm::APInt(1, 0)}},
        {},
        selected, minimum, kind};
  };
  const auto sharedTagObligation = [](
                                       std::uint64_t fifo,
                                       std::uint64_t selected,
                                       std::optional<std::uint64_t> minimum,
                                       MappingReconvergentCapacityProofKind
                                           kind) {
    return MappingReconvergentCapacityObligation{
        loom::fabric::FabricFifoOccurrenceRef(fifo),
        {MappingStaticQueueClass{MappingStaticQueueClassKind::PhysicalTag,
                                 llvm::APInt(4, 3)},
         MappingStaticQueueClass{MappingStaticQueueClassKind::PhysicalTag,
                                 llvm::APInt(4, 5)}},
        {}, selected, minimum, kind};
  };

  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @trivial(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return values() streams() memories()
        complete(%start : none)
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse reconvergent capacity fixture");
  auto artifact = take(dataflow::finalizeCanonicalDataflow(*module));
  const auto view = take(artifact.view());
  const auto model =
      take(loom::mapping::freezeMappingProgressModel(view, /*events=*/{}));
  loom::mapping::MappingProgressProjection projection;
  projection.basis.kind =
      loom::mapping::MappingDataflowProgressBasisKind::Acyclic;
  const auto closureOf =
      [&](std::vector<MappingReconvergentCapacityObligation> obligations) {
        loom::mapping::MappingProgressProjection candidate = projection;
        candidate.reconvergentCapacityObligations = std::move(obligations);
        return take(
            loom::mapping::deriveMappingProgressClosure(model, candidate));
      };

  // A proven minimum above the selected pool is a proven shortfall: the
  // single-queue bubble deadlock needs no dependency cycle.
  const auto shortfall = closureOf({globalObligation(
      7, 1, 2, MappingReconvergentCapacityProofKind::Proven)});
  if (shortfall.kind !=
          loom::mapping::MappingProgressClosureKind::ProvenClosedWaitSet ||
      shortfall.reason != loom::mapping::MappingProgressClosureReason::
                              ReconvergentCapacityShortfall)
    fail("a proven capacity shortfall was not reported");
  const auto shortfallObjective =
      loom::mapping::projectMappingProgressObjective(shortfall);
  if (shortfallObjective.hardViolationCount != 1 ||
      shortfallObjective.proofDebtWitnessCount != 0 ||
      shortfallObjective.capacityShortfall != 1)
    fail("capacity shortfall produced the wrong objective projection");

  // A proven minimum within the selected pool discharges the obligation.
  if (closureOf({globalObligation(7, 16, 2,
                                  MappingReconvergentCapacityProofKind::Proven)})
          .kind !=
      loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet)
    fail("a sufficient capacity was reported as a shortfall");

  // Tag-local order classes share one physical slot pool. They therefore
  // produce one owner obligation and one capacity comparison, never one depth
  // per tag.
  const auto sharedPoolShortfall = closureOf({sharedTagObligation(
      9, 1, 2, MappingReconvergentCapacityProofKind::Proven)});
  if (sharedPoolShortfall.kind !=
          loom::mapping::MappingProgressClosureKind::ProvenClosedWaitSet ||
      sharedPoolShortfall.reason !=
          loom::mapping::MappingProgressClosureReason::
              ReconvergentCapacityShortfall)
    fail("tag-local classes did not share their FIFO capacity owner");
  if (closureOf({sharedTagObligation(
                     9, 2, 2,
                     MappingReconvergentCapacityProofKind::Proven)})
          .kind !=
      loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet)
    fail("a sufficient shared virtual-channel pool was rejected");

  // Two obligations for one FIFO would create competing capacity owners.
  loom::mapping::MappingProgressProjection duplicateOwner = projection;
  duplicateOwner.reconvergentCapacityObligations = {
      globalObligation(7, 4, 1,
                       MappingReconvergentCapacityProofKind::Proven),
      globalObligation(7, 4, 1,
                       MappingReconvergentCapacityProofKind::Proven)};
  auto duplicateClosure =
      loom::mapping::deriveMappingProgressClosure(model, duplicateOwner);
  if (duplicateClosure)
    fail("duplicate shared-pool capacity owners were accepted");
  llvm::consumeError(duplicateClosure.takeError());

  // An unproven selected queue class is progress debt even without a known
  // dependency cycle. It cannot pass the static publication gate.
  const auto unproven = closureOf({globalObligation(
      7, 1, std::nullopt,
      MappingReconvergentCapacityProofKind::ProofNotEstablished)});
  if (unproven.kind !=
          loom::mapping::MappingProgressClosureKind::ProofNotEstablished ||
      unproven.reason != loom::mapping::MappingProgressClosureReason::
                             ReconvergentCapacityNotEstablished)
    fail("an unproven capacity obligation was not reported as progress debt");
  const auto debtObjective =
      loom::mapping::projectMappingProgressObjective(unproven);
  if (debtObjective.hardViolationCount != 0 ||
      debtObjective.proofDebtWitnessCount != 1 ||
      debtObjective.capacityShortfall != 0)
    fail("unproven capacity produced the wrong objective projection");

  // A capacity-carrying closed component resolves exactly when every member
  // class has a proven obligation within its pool.
  std::vector<MappingBufferDependencyEdge> capacityCycle{
      waitEdge(globalQueueNode(7), globalQueueNode(8),
               MappingBufferDependencyEdgeKind::DownstreamCapacity),
      waitEdge(globalQueueNode(8), globalQueueNode(7),
               MappingBufferDependencyEdgeKind::DownstreamCapacity)};
  const auto resolvedThrough = [&](std::vector<MappingReconvergentCapacityObligation>
                                       obligations) {
    loom::mapping::MappingProgressProjection candidate = projection;
    candidate.bufferDependencyEdges = capacityCycle;
    candidate.reconvergentCapacityObligations = std::move(obligations);
    return take(
        loom::mapping::deriveMappingProgressClosure(model, candidate));
  };
  if (resolvedThrough({globalObligation(7, 4, 2,
                                        MappingReconvergentCapacityProofKind::
                                            Proven),
                       globalObligation(8, 4, 1,
                                        MappingReconvergentCapacityProofKind::
                                            Proven)})
          .kind !=
      loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet)
    fail("proven-sufficient members did not resolve the capacity component");
  const auto unprovenMember = resolvedThrough({globalObligation(
      7, 4, 2, MappingReconvergentCapacityProofKind::Proven)});
  if (unprovenMember.kind !=
          loom::mapping::MappingProgressClosureKind::ProofNotEstablished ||
      unprovenMember.reason != loom::mapping::MappingProgressClosureReason::
                                   ReconvergentCapacityNotEstablished)
    fail("a capacity component without a member proof was resolved");

  // A proven shortfall on a component member still reports the shortfall.
  const auto memberShortfall = resolvedThrough({globalObligation(
      7, 1, 2, MappingReconvergentCapacityProofKind::Proven)});
  if (memberShortfall.kind !=
          loom::mapping::MappingProgressClosureKind::ProvenClosedWaitSet ||
      memberShortfall.reason != loom::mapping::MappingProgressClosureReason::
                                    ReconvergentCapacityShortfall)
    fail("a member shortfall was hidden by the component mediation");
}

} // namespace

int main() {
  initializedFeedbackProgressBasis();
  completionFrontierRequiresReadyRoots();
  orderedRuntimeHeadsRequireCompleteExactPairing();
  bufferDependencyClosure();
  reconvergentCapacityClosure();
  llvm::outs() << "mapping progress tests passed\n";
  return EXIT_SUCCESS;
}
