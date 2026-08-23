#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
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
  projection.routeObligations.push_back({false});
  if (take(loom::mapping::deriveMappingProgressClosure(model, projection))
          .kind !=
      loom::mapping::MappingProgressClosureKind::ProvenClosedWaitSet)
    fail("post-divergence route without a durable boundary passed progress");
  projection.routeObligations.front().durableBoundaryAfterDivergence = true;
  if (take(loom::mapping::deriveMappingProgressClosure(model, projection))
          .kind !=
      loom::mapping::MappingProgressClosureKind::ProvenNoClosedWaitSet)
    fail("post-divergence durable boundary did not close route progress");
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
  eventProjection.routeObligations.push_back({true});
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

void qualifiedPairingRiskIsDerivedWithoutPhysicalRepair() {
  const loom::fabric::FabricPeOccurrenceRef pe(7);
  const loom::fabric::FabricFuOccurrenceRef fu(11);
  const loom::fabric::InstructionContextRef context{pe, 2};
  const loom::fabric::FabricTransportEndpointRef ingress{
      loom::fabric::FabricTransportEndpointOwnerRef::of(pe), 0};
  std::array<std::uint8_t, loom::ArtifactIdentity::byteSize> identityBytes{};
  identityBytes.fill(9);
  const dataflow::GraphRef graph{
      take(loom::ArtifactIdentity::fromBytes(identityBytes)),
      dataflow::GraphId(1)};
  const dataflow::CanonicalGraphProducerEndpointRef producer =
      dataflow::GraphStartTokenRef{graph};
  const llvm::APInt tag(4, 3);
  loom::mapping::SpatialPeOperandQueueMatchGroupView group{
      producer,
      ingress,
      tag,
      {{{}, {context, 0, 0}, fu, 0, 2},
       {{}, {context, 0, 1}, fu, 0, 2}},
      true};
  const auto feedback = take(
      loom::mapping::deriveSpatialPeOperandProgressFeedback({group}));
  if (feedback.status !=
          loom::mapping::SpatialPeOperandProgressStatus::LikelyRisk ||
      feedback.support !=
          loom::mapping::SpatialPeOperandProgressSupport::Analytic ||
      feedback.potentiallyBlockingGroupCount != 1 ||
      feedback.pairingKeyCount != 2 || feedback.distinctPairingKeyCount != 1)
    fail("same-ingress ordered roles lost their qualified pairing risk");

  group.orderedCorrespondenceKnown = false;
  const auto fanout = take(
      loom::mapping::deriveSpatialPeOperandProgressFeedback({group}));
  if (fanout.status == loom::mapping::SpatialPeOperandProgressStatus::LikelyRisk ||
      fanout.potentiallyBlockingGroupCount != 0 ||
      fanout.unknownPairingGroupCount != 1)
    fail("same-ingress atomic fanout was treated as an automatic risk");

  auto secondIngress = ingress;
  secondIngress.ordinal = 1;
  auto secondGroup = group;
  secondGroup.ingress = secondIngress;
  secondGroup.matches.resize(1);
  const auto crossIngress = take(
      loom::mapping::deriveSpatialPeOperandProgressFeedback(
          {group, secondGroup}));
  if (crossIngress.pairingOpportunityCount == 0 ||
      crossIngress.distinctPairingKeyCount != 1 ||
      crossIngress.distinctIngressCount != 2)
    fail("cross-ingress qualified pairing opportunity was not aggregated");
}

void orderedRuntimeHeadsRequireCompleteExactPairing() {
  const loom::fabric::FabricPeOccurrenceRef pe(7);
  const loom::fabric::FabricFuOccurrenceRef fu(11);
  const loom::fabric::InstructionContextRef context{pe, 2};
  const loom::fabric::FabricTransportEndpointRef ingress{
      loom::fabric::FabricTransportEndpointOwnerRef::of(pe), 0};
  std::array<std::uint8_t, loom::ArtifactIdentity::byteSize> identityBytes{};
  identityBytes.fill(5);
  const dataflow::GraphRef graph{
      take(loom::ArtifactIdentity::fromBytes(identityBytes)),
      dataflow::GraphId(1)};
  const llvm::APInt tag(4, 3);
  const loom::mapping::SpatialPeOperandQueueMatchGroupView group{
      dataflow::GraphStartTokenRef{graph},
      ingress,
      tag,
      {{{}, {context, 0, 0}, fu, 0, 2},
       {{}, {context, 0, 1}, fu, 1, 2}},
      true};
  const auto projection = take(
      loom::mapping::deriveSpatialPeOperandProgressFeedback({group}));
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
}

} // namespace

int main() {
  initializedFeedbackProgressBasis();
  qualifiedPairingRiskIsDerivedWithoutPhysicalRepair();
  orderedRuntimeHeadsRequireCompleteExactPairing();
  llvm::outs() << "mapping progress tests passed\n";
  return EXIT_SUCCESS;
}
