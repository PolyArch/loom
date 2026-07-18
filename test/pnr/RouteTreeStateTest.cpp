#include "PnR/RouteTreeState.h"
#include "MappingCoreTestSupport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace loom::mapping;
using namespace loom::mapping::test;
using namespace loom::pnr;

namespace {

constexpr std::uint32_t unbounded = std::numeric_limits<std::uint32_t>::max();
constexpr PnrIndex sinkObligationA = 0;
constexpr PnrIndex sinkObligationB = 1;

TransportEndpointRef endpointRef(const ArtifactIdentity &fabric,
                                 std::uint64_t id) {
  return TransportEndpointRef{fabric, TransportEndpointId(id)};
}

TransportResourceRef resourceRef(const ArtifactIdentity &fabric,
                                 std::uint64_t id) {
  return TransportResourceRef{fabric, TransportResourceId(id)};
}

TransportEndpointDescriptor transportEndpoint(std::uint64_t id,
                                              PortDirection direction) {
  return TransportEndpointDescriptor{
      TransportEndpointId(id),   direction, PortKind::Value, unbounded, 0,
      fabric::DataPathKind::Bits};
}

void require(const char *test, bool condition, const char *message) {
  if (!condition)
    fail(test, message);
}

void requireSuccess(const char *test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)).c_str());
}

void requireErrorContains(const char *test, llvm::Error error,
                          std::string_view expected) {
  if (!error)
    fail(test, "expected route-tree failure");
  const std::string message = llvm::toString(std::move(error));
  if (message.find(expected) == std::string::npos)
    fail(test, ("missing diagnostic text: " + std::string(expected)).c_str());
}

template <typename T> T takeValue(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()).c_str());
  return std::move(*value);
}

FrozenRoutingGraph makeRoutingGraph(const char *test) {
  TestCase testCase = makeValidCase();
  const ArtifactIdentity &fabric = testCase.fabric.identity;
  testCase.fabric.transportResources = {
      {TransportResourceId(300),
       TransportResourceKind::Switch,
       {transportEndpoint(3000, PortDirection::Input),
        transportEndpoint(3001, PortDirection::Output),
        transportEndpoint(3002, PortDirection::Output),
        transportEndpoint(3003, PortDirection::Output)}},
      {TransportResourceId(400),
       TransportResourceKind::Switch,
       {transportEndpoint(4000, PortDirection::Input),
        transportEndpoint(4001, PortDirection::Input),
        transportEndpoint(4002, PortDirection::Output)}}};
  testCase.fabric.transportArcs = {
      {endpointRef(fabric, 2003), endpointRef(fabric, 3000)},
      {endpointRef(fabric, 3001), endpointRef(fabric, 4000)},
      {endpointRef(fabric, 3002), endpointRef(fabric, 2002)},
      {endpointRef(fabric, 3003), endpointRef(fabric, 4001)},
      {endpointRef(fabric, 4002), endpointRef(fabric, 2000)}};
  testCase.fabric.transportTraversals = {
      {resourceRef(fabric, 300), endpointRef(fabric, 3000),
       endpointRef(fabric, 3001)},
      {resourceRef(fabric, 300), endpointRef(fabric, 3000),
       endpointRef(fabric, 3002)},
      {resourceRef(fabric, 300), endpointRef(fabric, 3000),
       endpointRef(fabric, 3003)},
      {resourceRef(fabric, 400), endpointRef(fabric, 4000),
       endpointRef(fabric, 4002)},
      {resourceRef(fabric, 400), endpointRef(fabric, 4001),
       endpointRef(fabric, 4002)}};

  ValidatedTechMapping mapping = validateCase(test, testCase);
  ResolvedPnrConfigView config;
  return takeValue(test, freezeRoutingGraph(
                             makePnrProblemInputs(testCase, mapping, config)));
}

FrozenRoutingGraph makeCollisionRoutingGraph(const char *test) {
  TestCase testCase = makeValidCase();
  const ArtifactIdentity &fabric = testCase.fabric.identity;
  std::vector<TransportEndpointDescriptor> endpoints;
  for (std::uint64_t id = 5000; id <= 5002; ++id)
    endpoints.push_back(transportEndpoint(
        id, id == 5001 ? PortDirection::Input : PortDirection::Output));
  testCase.fabric.transportResources = {{TransportResourceId(300),
                                         TransportResourceKind::Switch,
                                         std::move(endpoints)}};
  testCase.fabric.transportTraversals = {
      {resourceRef(fabric, 300), endpointRef(fabric, 5001),
       endpointRef(fabric, 5000)},
      {resourceRef(fabric, 300), endpointRef(fabric, 5001),
       endpointRef(fabric, 5002)}};

  ValidatedTechMapping mapping = validateCase(test, testCase);
  ResolvedPnrConfigView config;
  return takeValue(test, freezeRoutingGraph(
                             makePnrProblemInputs(testCase, mapping, config)));
}

PnrIndex endpoint(const char *test, const FrozenRoutingGraph &graph,
                  std::uint64_t id) {
  const auto endpoints = graph.routingEndpoints();
  for (std::size_t index = 0; index < endpoints.size(); ++index)
    if (endpoints[index].id == TransportEndpointId(id))
      return static_cast<PnrIndex>(index);
  fail(test, "routing endpoint is missing");
}

PnrIndex arc(const char *test, const FrozenRoutingGraph &graph, PnrIndex source,
             PnrIndex target) {
  const PnrIndex begin = graph.adjacencyOffsets()[source];
  const PnrIndex end = graph.adjacencyOffsets()[source + 1];
  for (PnrIndex index = begin; index < end; ++index)
    if (graph.routingArcs()[index].target == target)
      return index;
  fail(test, "routing arc is missing");
}

struct Fixture {
  FrozenRoutingGraph graph;
  PnrIndex root;
  PnrIndex trunk;
  PnrIndex sinkA;
  PnrIndex sinkB;
  PnrIndex branchA;
  PnrIndex branchB;
  PnrIndex alternateA;
  PnrIndex mergeInputA;
  PnrIndex mergeInputAlternateA;
  PnrIndex mergeOutput;
  PnrIndex rootToTrunk;
  PnrIndex trunkToA;
  PnrIndex trunkToB;
  PnrIndex trunkToAlternateA;
  PnrIndex branchAToMergeInput;
  PnrIndex branchBToSink;
  PnrIndex alternateAToMergeInput;
  PnrIndex mergeInputAToOutput;
  PnrIndex mergeInputAlternateAToOutput;
  PnrIndex mergeOutputToSink;

  Fixture()
      : graph(makeRoutingGraph(__func__)),
        root(endpoint(__func__, graph, 2003)),
        trunk(endpoint(__func__, graph, 3000)),
        sinkA(endpoint(__func__, graph, 2000)),
        sinkB(endpoint(__func__, graph, 2002)),
        branchA(endpoint(__func__, graph, 3001)),
        branchB(endpoint(__func__, graph, 3002)),
        alternateA(endpoint(__func__, graph, 3003)),
        mergeInputA(endpoint(__func__, graph, 4000)),
        mergeInputAlternateA(endpoint(__func__, graph, 4001)),
        mergeOutput(endpoint(__func__, graph, 4002)),
        rootToTrunk(arc(__func__, graph, root, trunk)),
        trunkToA(arc(__func__, graph, trunk, branchA)),
        trunkToB(arc(__func__, graph, trunk, branchB)),
        trunkToAlternateA(arc(__func__, graph, trunk, alternateA)),
        branchAToMergeInput(arc(__func__, graph, branchA, mergeInputA)),
        branchBToSink(arc(__func__, graph, branchB, sinkB)),
        alternateAToMergeInput(
            arc(__func__, graph, alternateA, mergeInputAlternateA)),
        mergeInputAToOutput(arc(__func__, graph, mergeInputA, mergeOutput)),
        mergeInputAlternateAToOutput(
            arc(__func__, graph, mergeInputAlternateA, mergeOutput)),
        mergeOutputToSink(arc(__func__, graph, mergeOutput, sinkA)) {}
};

RouteTreeState makeUnrouted(const char *test, const Fixture &fixture) {
  return takeValue(test, RouteTreeState::create(fixture.graph, 2));
}

void attachInitialTree(const char *test, const Fixture &fixture,
                       RouteTreeState &state,
                       RouteTreeTransactionScratch &scratch) {
  RouteTreeTransaction transaction =
      takeValue(test, state.beginTransaction(scratch));
  const std::array<PnrIndex, 5> pathA{
      fixture.rootToTrunk, fixture.trunkToA, fixture.branchAToMergeInput,
      fixture.mergeInputAToOutput, fixture.mergeOutputToSink};
  const std::array<PnrIndex, 2> pathB{fixture.trunkToB, fixture.branchBToSink};
  requireSuccess(test, transaction.bindSource(fixture.root));
  requireSuccess(test, transaction.bindSink(sinkObligationA, fixture.sinkA));
  requireSuccess(test,
                 transaction.attachPath(fixture.root, pathA, sinkObligationA));
  requireSuccess(test, transaction.bindSink(sinkObligationB, fixture.sinkB));
  requireSuccess(test,
                 transaction.attachPath(fixture.trunk, pathB, sinkObligationB));
  requireSuccess(test, transaction.commit());
}

PnrIndex slot(const char *test, const RouteTreeState &state,
              PnrIndex endpoint) {
  const std::optional<PnrIndex> result = state.findNode(endpoint);
  if (!result)
    fail(test, "reached endpoint is missing from the route tree");
  return *result;
}

void buildsOnlyAValidRootedArborescence() {
  Fixture fixture;
  RouteTreeState state = makeUnrouted(__func__, fixture);
  RouteTreeTransactionScratch scratch;
  require(__func__, state.isUnrouted(), "new state is not explicitly unrouted");
  require(__func__, !state.sourceEndpoint().has_value(),
          "new state fixed a physical source endpoint");
  require(__func__,
          !state.sinkEndpoint(sinkObligationA).has_value() &&
              !state.sinkEndpoint(sinkObligationB).has_value(),
          "new state fixed a physical sink endpoint");

  RouteTreeTransaction transaction =
      takeValue(__func__, state.beginTransaction(scratch));
  const std::array<PnrIndex, 5> pathA{
      fixture.rootToTrunk, fixture.trunkToA, fixture.branchAToMergeInput,
      fixture.mergeInputAToOutput, fixture.mergeOutputToSink};
  requireSuccess(__func__, transaction.bindSource(fixture.root));
  requireSuccess(__func__,
                 transaction.bindSink(sinkObligationA, fixture.sinkA));
  requireSuccess(__func__,
                 transaction.attachPath(fixture.root, pathA, sinkObligationA));

  const std::vector<RouteTreeNode> beforeInvalid(state.nodeStorage().begin(),
                                                 state.nodeStorage().end());
  const std::array<PnrIndex, 3> reenteringPath{
      fixture.rootToTrunk, fixture.trunkToB, fixture.branchBToSink};
  requireSuccess(__func__,
                 transaction.bindSink(sinkObligationB, fixture.sinkB));
  requireErrorContains(
      __func__,
      transaction.attachPath(fixture.root, reenteringPath, sinkObligationB),
      "re-enters");
  require(__func__,
          std::equal(beforeInvalid.begin(), beforeInvalid.end(),
                     state.nodeStorage().begin(), state.nodeStorage().end()),
          "rejected reconvergence changed the tree");

  const std::array<PnrIndex, 2> disconnectedPath{fixture.trunkToB,
                                                 fixture.branchBToSink};
  requireErrorContains(
      __func__,
      transaction.attachPath(fixture.root, disconnectedPath, sinkObligationB),
      "does not continue");
  require(__func__,
          std::equal(beforeInvalid.begin(), beforeInvalid.end(),
                     state.nodeStorage().begin(), state.nodeStorage().end()),
          "rejected directed path changed the tree");

  requireSuccess(
      __func__,
      transaction.attachPath(fixture.trunk, disconnectedPath, sinkObligationB));
  requireSuccess(__func__, transaction.commit());
  requireSuccess(__func__, state.verify());

  const RouteTreeNode &root = state.node(slot(__func__, state, fixture.root));
  const RouteTreeNode &trunk = state.node(slot(__func__, state, fixture.trunk));
  const RouteTreeNode &sinkA = state.node(slot(__func__, state, fixture.sinkA));
  const RouteTreeNode &sinkB = state.node(slot(__func__, state, fixture.sinkB));
  require(__func__, root.parentArc == getInvalidPnrIndex(),
          "root has a parent arc");
  require(__func__, trunk.parentArc == fixture.rootToTrunk,
          "trunk lost its unique parent arc");
  require(__func__, sinkA.parentArc == fixture.mergeOutputToSink,
          "first sink lost its unique parent arc");
  require(__func__, sinkB.parentArc == fixture.branchBToSink,
          "second sink lost its unique parent arc");
  require(__func__,
          sinkA.sinkObligationCount == 1 && sinkB.sinkObligationCount == 1,
          "sink metadata does not cover both obligations");
  require(__func__,
          state.sourceEndpoint() == fixture.root &&
              state.sinkEndpoint(sinkObligationA) == fixture.sinkA &&
              state.sinkEndpoint(sinkObligationB) == fixture.sinkB,
          "committed bindings do not match the routed tree");
}

void preservesSharedPrefixAcrossPruneAndReroute() {
  Fixture fixture;
  RouteTreeState state = makeUnrouted(__func__, fixture);
  RouteTreeTransactionScratch scratch;
  attachInitialTree(__func__, fixture, state, scratch);
  const PnrIndex rootSlot = slot(__func__, state, fixture.root);
  const PnrIndex trunkSlot = slot(__func__, state, fixture.trunk);
  const PnrIndex sinkBSlot = slot(__func__, state, fixture.sinkB);

  RouteTreeTransaction transaction =
      takeValue(__func__, state.beginTransaction(scratch));
  requireSuccess(__func__, transaction.ripUpSubtree(fixture.branchA));
  require(__func__,
          state.findNode(fixture.root) == rootSlot &&
              state.findNode(fixture.trunk) == trunkSlot &&
              state.findNode(fixture.sinkB) == sinkBSlot,
          "sink prune removed a shared-prefix endpoint");
  require(__func__,
          !state.findNode(fixture.branchA) && !state.findNode(fixture.sinkA),
          "sink prune retained its unused branch");

  const std::array<PnrIndex, 4> alternatePath{
      fixture.trunkToAlternateA, fixture.alternateAToMergeInput,
      fixture.mergeInputAlternateAToOutput, fixture.mergeOutputToSink};
  requireSuccess(__func__,
                 transaction.bindSink(sinkObligationA, fixture.sinkA));
  requireSuccess(__func__, transaction.attachPath(fixture.trunk, alternatePath,
                                                  sinkObligationA));
  requireSuccess(__func__, transaction.commit());
  requireSuccess(__func__, state.verify());
  require(__func__,
          state.findNode(fixture.root) == rootSlot &&
              state.findNode(fixture.trunk) == trunkSlot &&
              state.findNode(fixture.sinkB) == sinkBSlot,
          "reroute replaced a shared-prefix endpoint");
  require(__func__, state.findNode(fixture.alternateA).has_value(),
          "reroute did not install the alternate branch");
}

void reusesScratchAndRollsBackExactTreeState() {
  Fixture fixture;
  RouteTreeState state = makeUnrouted(__func__, fixture);
  RouteTreeTransactionScratch scratch;
  attachInitialTree(__func__, fixture, state, scratch);
  const std::vector<RouteTreeNode> before(state.nodeStorage().begin(),
                                          state.nodeStorage().end());
  const PnrIndex beforeCount = state.activeNodeCount();
  const std::size_t deltaCapacity = scratch.deltaCapacity();
  const std::size_t workspaceCapacity = scratch.workspaceCapacity();
  require(__func__, deltaCapacity != 0 && workspaceCapacity != 0,
          "committed transaction did not retain scratch capacity");

  RouteTreeTransaction transaction =
      takeValue(__func__, state.beginTransaction(scratch));
  requireSuccess(__func__, transaction.ripUpSink(sinkObligationA));
  const std::array<PnrIndex, 4> alternatePath{
      fixture.trunkToAlternateA, fixture.alternateAToMergeInput,
      fixture.mergeInputAlternateAToOutput, fixture.mergeOutputToSink};
  requireSuccess(__func__,
                 transaction.bindSink(sinkObligationA, fixture.sinkA));
  requireSuccess(__func__, transaction.attachPath(fixture.trunk, alternatePath,
                                                  sinkObligationA));
  transaction.rollback();

  require(__func__, state.isRouted(), "rollback changed routed state");
  require(__func__, state.activeNodeCount() == beforeCount,
          "rollback changed the active-node count");
  require(__func__,
          std::equal(before.begin(), before.end(), state.nodeStorage().begin(),
                     state.nodeStorage().end()),
          "rollback did not restore exact node storage");
  require(__func__,
          state.findNode(fixture.branchA).has_value() &&
              !state.findNode(fixture.alternateA),
          "rollback did not restore endpoint lookup");
  require(__func__,
          state.sourceEndpoint() == fixture.root &&
              state.sinkEndpoint(sinkObligationA) == fixture.sinkA,
          "rollback did not restore physical bindings");
  require(__func__,
          scratch.deltaCapacity() >= deltaCapacity &&
              scratch.workspaceCapacity() >= workspaceCapacity,
          "rollback discarded reusable scratch capacity");
  requireSuccess(__func__, state.verify());

  RouteTreeTransaction committed =
      takeValue(__func__, state.beginTransaction(scratch));
  requireSuccess(__func__, committed.ripUpSink(sinkObligationA));
  requireSuccess(__func__, committed.bindSink(sinkObligationA, fixture.sinkA));
  requireSuccess(__func__, committed.attachPath(fixture.trunk, alternatePath,
                                                sinkObligationA));
  requireSuccess(__func__, committed.commit());
  require(__func__,
          scratch.deltaCapacity() >= deltaCapacity &&
              scratch.workspaceCapacity() >= workspaceCapacity,
          "second commit discarded reusable scratch capacity");
}

void enforcesCoverageOrExplicitUnroutedState() {
  Fixture fixture;
  RouteTreeState state = makeUnrouted(__func__, fixture);
  RouteTreeTransactionScratch scratch;
  attachInitialTree(__func__, fixture, state, scratch);

  RouteTreeTransaction incomplete =
      takeValue(__func__, state.beginTransaction(scratch));
  requireSuccess(__func__, incomplete.ripUpSink(sinkObligationA));
  requireErrorContains(__func__, incomplete.commit(), "not covered");
  incomplete.rollback();
  requireSuccess(__func__, state.verify());

  RouteTreeTransaction wholeNet =
      takeValue(__func__, state.beginTransaction(scratch));
  requireSuccess(__func__, wholeNet.ripUpWholeNet());
  requireSuccess(__func__, wholeNet.commit());
  require(__func__, state.isUnrouted() && state.activeNodeCount() == 0,
          "whole-net rip-up did not commit explicit unrouted state");
  require(__func__, state.nodeStorage().empty(),
          "whole-net rip-up retained sparse node storage");
  require(__func__,
          !state.sourceEndpoint().has_value() &&
              !state.sinkEndpoint(sinkObligationA).has_value() &&
              !state.sinkEndpoint(sinkObligationB).has_value(),
          "whole-net rip-up retained physical bindings");
}

void independentlyRebindsDuplicateSinkObligations() {
  Fixture fixture;
  RouteTreeState state = makeUnrouted(__func__, fixture);
  RouteTreeTransactionScratch scratch;
  const std::array<PnrIndex, 5> pathA{
      fixture.rootToTrunk, fixture.trunkToA, fixture.branchAToMergeInput,
      fixture.mergeInputAToOutput, fixture.mergeOutputToSink};

  RouteTreeTransaction initial =
      takeValue(__func__, state.beginTransaction(scratch));
  requireSuccess(__func__, initial.bindSource(fixture.root));
  requireSuccess(__func__, initial.bindSink(sinkObligationA, fixture.sinkA));
  requireSuccess(__func__,
                 initial.attachPath(fixture.root, pathA, sinkObligationA));
  requireSuccess(__func__, initial.bindSink(sinkObligationB, fixture.sinkA));
  requireSuccess(__func__,
                 initial.attachPath(fixture.sinkA, {}, sinkObligationB));
  requireSuccess(__func__, initial.commit());
  require(__func__,
          state.sinkEndpoint(sinkObligationA) == fixture.sinkA &&
              state.sinkEndpoint(sinkObligationB) == fixture.sinkA,
          "duplicate endpoint bindings lost obligation identity");
  require(
      __func__,
      state.node(slot(__func__, state, fixture.sinkA)).sinkObligationCount == 2,
      "shared sink node does not count both obligations");

  const std::vector<RouteTreeNode> before(state.nodeStorage().begin(),
                                          state.nodeStorage().end());
  RouteTreeTransaction rolledBack =
      takeValue(__func__, state.beginTransaction(scratch));
  requireSuccess(__func__, rolledBack.ripUpSink(sinkObligationA));
  requireSuccess(__func__, rolledBack.bindSink(sinkObligationA, fixture.sinkB));
  const std::array<PnrIndex, 2> pathB{fixture.trunkToB, fixture.branchBToSink};
  requireSuccess(__func__,
                 rolledBack.attachPath(fixture.trunk, pathB, sinkObligationA));
  require(__func__,
          state.sinkEndpoint(sinkObligationA) == fixture.sinkB &&
              state.sinkEndpoint(sinkObligationB) == fixture.sinkA,
          "single-obligation rebind changed its peer");
  rolledBack.rollback();
  require(__func__,
          state.sinkEndpoint(sinkObligationA) == fixture.sinkA &&
              state.sinkEndpoint(sinkObligationB) == fixture.sinkA,
          "rollback did not restore duplicate endpoint bindings");
  require(__func__,
          std::equal(before.begin(), before.end(), state.nodeStorage().begin(),
                     state.nodeStorage().end()),
          "duplicate-binding rollback changed the route tree");

  RouteTreeTransaction committed =
      takeValue(__func__, state.beginTransaction(scratch));
  requireSuccess(__func__, committed.ripUpSink(sinkObligationA));
  requireSuccess(__func__, committed.bindSink(sinkObligationA, fixture.sinkB));
  requireSuccess(__func__,
                 committed.attachPath(fixture.trunk, pathB, sinkObligationA));
  requireSuccess(__func__, committed.commit());
  require(__func__,
          state.sinkEndpoint(sinkObligationA) == fixture.sinkB &&
              state.sinkEndpoint(sinkObligationB) == fixture.sinkA &&
              state.node(slot(__func__, state, fixture.sinkA))
                      .sinkObligationCount == 1,
          "committed rebind did not preserve the peer obligation");

  RouteTreeTransaction sourceRollback =
      takeValue(__func__, state.beginTransaction(scratch));
  requireSuccess(__func__, sourceRollback.ripUpWholeNet());
  requireSuccess(__func__, sourceRollback.bindSource(fixture.trunk));
  sourceRollback.rollback();
  require(__func__, state.sourceEndpoint() == fixture.root,
          "source binding rollback did not restore the producer endpoint");
}

void preservesLookupCollisionChainsAfterDeletion() {
  FrozenRoutingGraph graph = makeCollisionRoutingGraph(__func__);
  const PnrIndex sinkA = endpoint(__func__, graph, 5000);
  const PnrIndex source = endpoint(__func__, graph, 5001);
  const PnrIndex sinkB = endpoint(__func__, graph, 5002);
  require(__func__, sinkA == 4 && source == 5 && sinkB == 6,
          "collision fixture endpoint ordering changed");
  const std::array<PnrIndex, 1> pathA{arc(__func__, graph, source, sinkA)};
  const std::array<PnrIndex, 1> pathB{arc(__func__, graph, source, sinkB)};

  RouteTreeState state = takeValue(__func__, RouteTreeState::create(graph, 2));
  RouteTreeTransactionScratch scratch;
  RouteTreeTransaction initial =
      takeValue(__func__, state.beginTransaction(scratch));
  requireSuccess(__func__, initial.bindSource(source));
  requireSuccess(__func__, initial.bindSink(sinkObligationA, sinkA));
  requireSuccess(__func__, initial.attachPath(source, pathA, sinkObligationA));
  requireSuccess(__func__, initial.bindSink(sinkObligationB, sinkB));
  requireSuccess(__func__, initial.attachPath(source, pathB, sinkObligationB));
  requireSuccess(__func__, initial.commit());

  RouteTreeTransaction deletion =
      takeValue(__func__, state.beginTransaction(scratch));
  requireSuccess(__func__, deletion.ripUpSink(sinkObligationA));
  require(__func__,
          state.findNode(source).has_value() &&
              state.findNode(sinkB).has_value(),
          "open-addressed deletion broke a collision chain");
  requireSuccess(__func__, deletion.bindSink(sinkObligationA, sinkB));
  requireSuccess(__func__, deletion.attachPath(sinkB, {}, sinkObligationA));
  requireSuccess(__func__, deletion.commit());
  requireSuccess(__func__, state.verify());
}

void checksPnrIndexCapacityBoundaries() {
  const std::uint64_t maximum = getPnrIndexMax();
  requireSuccess(__func__, loom::pnr::detail::preflightRouteTreeStateCapacity(
                               maximum, maximum));
  if (getPnrIndexBits() == 32) {
    requireErrorContains(__func__,
                         loom::pnr::detail::preflightRouteTreeStateCapacity(
                             maximum + 1, maximum),
                         "required_max_count=4294967296");
  }

  Fixture fixture;
  RouteTreeState state =
      takeValue(__func__, RouteTreeState::create(fixture.graph, 1));
  RouteTreeTransactionScratch scratch;
  RouteTreeTransaction transaction =
      takeValue(__func__, state.beginTransaction(scratch));
  requireErrorContains(__func__,
                       transaction.bindSource(static_cast<PnrIndex>(
                           fixture.graph.routingEndpoints().size())),
                       "source endpoint");
}

} // namespace

int main() {
  buildsOnlyAValidRootedArborescence();
  preservesSharedPrefixAcrossPruneAndReroute();
  reusesScratchAndRollsBackExactTreeState();
  enforcesCoverageOrExplicitUnroutedState();
  independentlyRebindsDuplicateSinkObligations();
  preservesLookupCollisionChainsAfterDeletion();
  checksPnrIndexCapacityBoundaries();
  return 0;
}
