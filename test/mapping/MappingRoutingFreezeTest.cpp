#include "MappingCoreTestSupport.h"

#include "Fabric/IR/FabricOps.h"
#include "PnR/FrozenRoutingGraph.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::mapping::test {
namespace {

constexpr std::uint32_t unbounded = std::numeric_limits<std::uint32_t>::max();

TransportEndpointRef endpointRef(const ArtifactIdentity &fabric,
                                 std::uint64_t id) {
  return TransportEndpointRef{fabric, TransportEndpointId(id)};
}

TransportResourceRef resourceRef(const ArtifactIdentity &fabric,
                                 std::uint64_t id) {
  return TransportResourceRef{fabric, TransportResourceId(id)};
}

TransportEndpointDescriptor transportEndpoint(
    std::uint64_t id, PortDirection direction,
    std::uint32_t payloadCapacityBits = unbounded,
    std::uint32_t tagCapacityBits = 0,
    ::fabric::DataPathKind transportKind = ::fabric::DataPathKind::Bits,
    PortKind portKind = PortKind::Value) {
  return TransportEndpointDescriptor{
      TransportEndpointId(id), direction,       portKind,
      payloadCapacityBits,     tagCapacityBits, transportKind};
}

ComputeOccurrenceDescriptor routableOccurrence(
    const ArtifactIdentity &fabric, const FuDescriptor &fu,
    std::uint64_t occurrenceId, std::uint64_t endpointId,
    PortDirection direction, std::uint32_t payloadCapacityBits = unbounded,
    std::uint32_t tagCapacityBits = 0,
    ::fabric::DataPathKind transportKind = ::fabric::DataPathKind::Bits,
    PortKind portKind = PortKind::Value) {
  const std::uint64_t endpointBase =
      direction == PortDirection::Input
          ? endpointId
          : endpointId - static_cast<std::uint64_t>(fu.inputPorts.size());
  ComputeOccurrenceDescriptor occurrence = makeSpatialComputeOccurrence(
      fabric, ComputeOccurrenceId(occurrenceId), fu, endpointBase);
  auto selected = llvm::find_if(
      occurrence.endpoints, [&](const ComputeEndpointDescriptor &endpoint) {
        return endpoint.id == ComputeEndpointId(endpointId);
      });
  if (selected == occurrence.endpoints.end())
    fail(__func__, "routable occurrence endpoint is missing");
  selected->kind = portKind;
  selected->payloadCapacityBits = payloadCapacityBits;
  selected->tagCapacityBits = tagCapacityBits;
  selected->transportKind = transportKind;
  return occurrence;
}

FrozenModelHandle validateAndFreezeRouting(const char *test,
                                           TestCase &testCase) {
  ValidatedTechMapping mapping = validateCase(test, testCase);
  ResolvedPnrConfigView config = makeSpatialPnrConfigView(__func__);
  return takeExpected(test, freezeSpatialPnrModel(makePnrProblemInputs(
                                testCase, mapping, config)));
}

std::size_t endpointIndex(const char *test, const FrozenRoutingGraph &graph,
                          TransportEndpointId endpoint) {
  for (std::size_t index = 0; index < graph.routingEndpoints().size(); ++index)
    if (graph.routingEndpoints()[index].id == endpoint)
      return index;
  fail(test, "routing endpoint is missing");
}

const FrozenRoutingArc *findArc(const FrozenRoutingGraph &graph,
                                TransportEndpointId source,
                                TransportEndpointId target) {
  const std::size_t sourceIndex = endpointIndex(__func__, graph, source);
  const std::size_t targetIndex = endpointIndex(__func__, graph, target);
  const PnrIndex begin = graph.adjacencyOffsets()[sourceIndex];
  const PnrIndex end = graph.adjacencyOffsets()[sourceIndex + 1];
  for (PnrIndex arc = begin; arc < end; ++arc)
    if (graph.routingArcs()[arc].target == targetIndex)
      return &graph.routingArcs()[arc];
  return nullptr;
}

bool reachable(const FrozenRoutingGraph &graph, TransportEndpointId source,
               TransportEndpointId target) {
  const std::size_t sourceIndex = endpointIndex(__func__, graph, source);
  const std::size_t targetIndex = endpointIndex(__func__, graph, target);
  std::vector<bool> visited(graph.routingEndpoints().size());
  std::deque<std::size_t> worklist{sourceIndex};
  visited[sourceIndex] = true;
  while (!worklist.empty()) {
    const std::size_t current = worklist.front();
    worklist.pop_front();
    if (current == targetIndex)
      return true;
    const PnrIndex begin = graph.adjacencyOffsets()[current];
    const PnrIndex end = graph.adjacencyOffsets()[current + 1];
    for (PnrIndex arc = begin; arc < end; ++arc) {
      const std::size_t next = graph.routingArcs()[arc].target;
      if (!visited[next]) {
        visited[next] = true;
        worklist.push_back(next);
      }
    }
  }
  return false;
}

void expectCanonicalTableOrdering(const char *test,
                                  const FrozenRoutingGraph &graph) {
  for (std::size_t index = 1; index < graph.transportResources().size();
       ++index)
    if (graph.transportResources()[index - 1].id.value() >=
        graph.transportResources()[index].id.value())
      fail(test, "transport resources are not ordered by typed identity");
  for (std::size_t index = 1; index < graph.routingEndpoints().size(); ++index)
    if (graph.routingEndpoints()[index - 1].id.value() >=
        graph.routingEndpoints()[index].id.value())
      fail(test, "routing endpoints are not ordered by typed identity");

  for (std::size_t resourceIndex = 0;
       resourceIndex < graph.transportResources().size(); ++resourceIndex) {
    const FrozenTransportResource &resource =
        graph.transportResources()[resourceIndex];
    const PnrIndex end = resource.endpointOffset + resource.endpointCount;
    if (end > graph.resourceEndpointVertices().size())
      fail(test, "resource endpoint range exceeds its flat table");
    std::uint64_t previousId = 0;
    for (PnrIndex offset = resource.endpointOffset; offset < end; ++offset) {
      const PnrIndex vertex = graph.resourceEndpointVertices()[offset];
      if (vertex >= graph.routingEndpoints().size())
        fail(test, "resource endpoint vertex exceeds the endpoint table");
      const FrozenRoutingEndpoint &endpoint = graph.routingEndpoints()[vertex];
      if (endpoint.ownerKind !=
              FrozenRoutingEndpointOwnerKind::TransportResource ||
          endpoint.owner != resourceIndex ||
          (offset != resource.endpointOffset &&
           endpoint.id.value() <= previousId))
        fail(test, "resource endpoint range is not canonically ordered");
      previousId = endpoint.id.value();
    }
  }

  if (graph.adjacencyOffsets().size() != graph.routingEndpoints().size() + 1 ||
      graph.adjacencyOffsets().back() != graph.routingArcs().size())
    fail(test, "routing CSR boundaries are inconsistent");
  for (std::size_t source = 0; source < graph.routingEndpoints().size();
       ++source) {
    const PnrIndex begin = graph.adjacencyOffsets()[source];
    const PnrIndex end = graph.adjacencyOffsets()[source + 1];
    if (begin > end || end > graph.routingArcs().size())
      fail(test, "routing CSR offsets are not monotonic");
    for (PnrIndex arc = begin + (begin != end); arc < end; ++arc) {
      const FrozenRoutingArc &previous = graph.routingArcs()[arc - 1];
      const FrozenRoutingArc &current = graph.routingArcs()[arc];
      if (std::tie(previous.target, previous.kind, previous.resource) >=
          std::tie(current.target, current.kind, current.resource))
        fail(test, "routing CSR arcs are not canonically ordered");
    }
  }
}

TestCase makeTraversalCase() {
  TestCase testCase = makeValidCase();
  const ArtifactIdentity &fabric = testCase.fabric.identity;
  const FuDescriptor &fu = testCase.fabric.functionalUnits.front();
  testCase.fabric.computeOccurrences = {
      routableOccurrence(fabric, fu, 100, 1000, PortDirection::Output, 64, 7,
                         ::fabric::DataPathKind::BitsTag),
      routableOccurrence(fabric, fu, 200, 2000, PortDirection::Input, 64, 9,
                         ::fabric::DataPathKind::BitsTag)};
  testCase.fabric.transportResources = {
      {TransportResourceId(300),
       TransportResourceKind::Switch,
       {transportEndpoint(3000, PortDirection::Input, 32, 9,
                          ::fabric::DataPathKind::BitsTag),
        transportEndpoint(3001, PortDirection::Output, 48, 5,
                          ::fabric::DataPathKind::BitsTag),
        transportEndpoint(3002, PortDirection::Output, 48, 5,
                          ::fabric::DataPathKind::BitsTag)}},
      {TransportResourceId(400),
       TransportResourceKind::Fifo,
       {transportEndpoint(4000, PortDirection::Input, 16, 8,
                          ::fabric::DataPathKind::BitsTag),
        transportEndpoint(4001, PortDirection::Output, 24, 6,
                          ::fabric::DataPathKind::BitsTag)}},
      {TransportResourceId(500),
       TransportResourceKind::Boundary,
       {transportEndpoint(5000, PortDirection::Input, 32, 3,
                          ::fabric::DataPathKind::BitsTag),
        transportEndpoint(5001, PortDirection::Output, 32, 1,
                          ::fabric::DataPathKind::BitsTag)},
       ::fabric::BoundaryDirection::T2t}};
  testCase.fabric.transportArcs = {
      {endpointRef(fabric, 1000), endpointRef(fabric, 3000)},
      {endpointRef(fabric, 3001), endpointRef(fabric, 4000)},
      {endpointRef(fabric, 4001), endpointRef(fabric, 5000)},
      {endpointRef(fabric, 5001), endpointRef(fabric, 2000)}};
  testCase.fabric.transportTraversals = {
      {resourceRef(fabric, 300), endpointRef(fabric, 3000),
       endpointRef(fabric, 3001)},
      {resourceRef(fabric, 400), endpointRef(fabric, 4000),
       endpointRef(fabric, 4001)},
      {resourceRef(fabric, 500), endpointRef(fabric, 5000),
       endpointRef(fabric, 5001)}};
  return testCase;
}

TestCase makeBoundaryCase(::fabric::BoundaryDirection direction,
                          ::fabric::DataPathKind inputKind,
                          std::uint32_t inputPayloadBits,
                          std::uint32_t inputTagBits,
                          ::fabric::DataPathKind outputKind,
                          std::uint32_t outputPayloadBits,
                          std::uint32_t outputTagBits) {
  TestCase testCase = makeValidCase();
  const ArtifactIdentity &fabric = testCase.fabric.identity;
  testCase.fabric.transportResources = {
      {TransportResourceId(300),
       TransportResourceKind::Boundary,
       {transportEndpoint(3000, PortDirection::Input, inputPayloadBits,
                          inputTagBits, inputKind),
        transportEndpoint(3001, PortDirection::Output, outputPayloadBits,
                          outputTagBits, outputKind)},
       direction}};
  testCase.fabric.transportTraversals = {{resourceRef(fabric, 300),
                                          endpointRef(fabric, 3000),
                                          endpointRef(fabric, 3001)}};
  return testCase;
}

TestCase makeIrregularReachabilityCase() {
  TestCase testCase = makeValidCase();
  const ArtifactIdentity &fabric = testCase.fabric.identity;
  const FuDescriptor &fu = testCase.fabric.functionalUnits.front();
  testCase.fabric.computeOccurrences = {
      routableOccurrence(fabric, fu, 100, 1000, PortDirection::Output),
      routableOccurrence(fabric, fu, 200, 2000, PortDirection::Output),
      routableOccurrence(fabric, fu, 300, 3000, PortDirection::Input),
      routableOccurrence(fabric, fu, 400, 4000, PortDirection::Input)};
  testCase.fabric.transportResources = {
      {TransportResourceId(500),
       TransportResourceKind::Switch,
       {transportEndpoint(5000, PortDirection::Input),
        transportEndpoint(5001, PortDirection::Input),
        transportEndpoint(5002, PortDirection::Output),
        transportEndpoint(5003, PortDirection::Output)}},
      {TransportResourceId(600),
       TransportResourceKind::Fifo,
       {transportEndpoint(6000, PortDirection::Input),
        transportEndpoint(6001, PortDirection::Output)}},
      {TransportResourceId(700),
       TransportResourceKind::Fifo,
       {transportEndpoint(7000, PortDirection::Input),
        transportEndpoint(7001, PortDirection::Output)}},
      {TransportResourceId(800),
       TransportResourceKind::Fifo,
       {transportEndpoint(8000, PortDirection::Input),
        transportEndpoint(8001, PortDirection::Output)}},
      {TransportResourceId(900),
       TransportResourceKind::Switch,
       {transportEndpoint(9000, PortDirection::Input),
        transportEndpoint(9001, PortDirection::Input),
        transportEndpoint(9002, PortDirection::Output),
        transportEndpoint(9003, PortDirection::Output)}}};
  testCase.fabric.transportArcs = {
      {endpointRef(fabric, 1000), endpointRef(fabric, 5000)},
      {endpointRef(fabric, 2000), endpointRef(fabric, 5001)},
      {endpointRef(fabric, 5002), endpointRef(fabric, 6000)},
      {endpointRef(fabric, 6001), endpointRef(fabric, 9000)},
      {endpointRef(fabric, 5003), endpointRef(fabric, 7000)},
      {endpointRef(fabric, 7001), endpointRef(fabric, 8000)},
      {endpointRef(fabric, 8001), endpointRef(fabric, 9001)},
      {endpointRef(fabric, 9002), endpointRef(fabric, 3000)},
      {endpointRef(fabric, 9003), endpointRef(fabric, 4000)}};
  testCase.fabric.transportTraversals = {
      {resourceRef(fabric, 500), endpointRef(fabric, 5000),
       endpointRef(fabric, 5002)},
      {resourceRef(fabric, 500), endpointRef(fabric, 5001),
       endpointRef(fabric, 5002)},
      {resourceRef(fabric, 500), endpointRef(fabric, 5000),
       endpointRef(fabric, 5003)},
      {resourceRef(fabric, 600), endpointRef(fabric, 6000),
       endpointRef(fabric, 6001)},
      {resourceRef(fabric, 700), endpointRef(fabric, 7000),
       endpointRef(fabric, 7001)},
      {resourceRef(fabric, 800), endpointRef(fabric, 8000),
       endpointRef(fabric, 8001)},
      {resourceRef(fabric, 900), endpointRef(fabric, 9000),
       endpointRef(fabric, 9002)},
      {resourceRef(fabric, 900), endpointRef(fabric, 9001),
       endpointRef(fabric, 9002)},
      {resourceRef(fabric, 900), endpointRef(fabric, 9001),
       endpointRef(fabric, 9003)}};
  return testCase;
}

void freezesExactDirectedAdjacencyAndIndependentCapacity() {
  TestCase testCase = makeTraversalCase();
  FrozenModelHandle model = validateAndFreezeRouting(__func__, testCase);
  const FrozenRoutingGraph &graph = model->routing();
  const FrozenRoutingArc *connection =
      findArc(graph, TransportEndpointId(1000), TransportEndpointId(3000));
  if (!connection || connection->kind != FrozenRoutingArcKind::PointToPoint ||
      connection->resource || connection->payloadCapacityBits != 32 ||
      connection->tagCapacityBits != 7)
    fail(__func__, "point arc lost direction, ownership, or capacity");
  const FrozenRoutingArc *switchTraversal =
      findArc(graph, TransportEndpointId(3000), TransportEndpointId(3001));
  if (!switchTraversal ||
      switchTraversal->kind != FrozenRoutingArcKind::Traversal ||
      !switchTraversal->resource ||
      switchTraversal->payloadCapacityBits != 32 ||
      switchTraversal->tagCapacityBits != 5)
    fail(__func__, "switch traversal lost resource or independent capacity");
  const FrozenRoutingArc *boundaryTraversal =
      findArc(graph, TransportEndpointId(5000), TransportEndpointId(5001));
  if (!boundaryTraversal || boundaryTraversal->payloadCapacityBits != 32 ||
      boundaryTraversal->tagCapacityBits != 1)
    fail(__func__, "boundary conversion capacity was not projected");
  if (findArc(graph, TransportEndpointId(3001), TransportEndpointId(3000)) ||
      findArc(graph, TransportEndpointId(3000), TransportEndpointId(3002)))
    fail(__func__, "routing freeze inferred an undeclared traversal");
}

void mirrorsEveryForwardArcInIncomingCsr() {
  TestCase testCase = makeTraversalCase();
  FrozenModelHandle model = validateAndFreezeRouting(__func__, testCase);
  const FrozenRoutingGraph &graph = model->routing();
  if (graph.incomingAdjacencyOffsets().size() !=
          graph.routingEndpoints().size() + 1 ||
      graph.incomingAdjacencyOffsets().back() != graph.routingArcs().size() ||
      graph.incomingSourceVertices().size() != graph.routingArcs().size() ||
      graph.incomingForwardArcIndices().size() != graph.routingArcs().size())
    fail(__func__, "incoming CSR dimensions do not mirror the forward graph");

  std::vector<bool> seenForwardArc(graph.routingArcs().size());
  for (std::size_t target = 0; target < graph.routingEndpoints().size();
       ++target) {
    const PnrIndex begin = graph.incomingAdjacencyOffsets()[target];
    const PnrIndex end = graph.incomingAdjacencyOffsets()[target + 1];
    if (begin > end || end > graph.routingArcs().size())
      fail(__func__, "incoming CSR offsets are inconsistent");
    for (PnrIndex incoming = begin; incoming < end; ++incoming) {
      const PnrIndex source = graph.incomingSourceVertices()[incoming];
      const PnrIndex forwardArc = graph.incomingForwardArcIndices()[incoming];
      if (source >= graph.routingEndpoints().size() ||
          forwardArc >= graph.routingArcs().size() ||
          seenForwardArc[forwardArc])
        fail(__func__, "incoming entry has invalid or duplicate provenance");
      const PnrIndex forwardBegin = graph.adjacencyOffsets()[source];
      const PnrIndex forwardEnd = graph.adjacencyOffsets()[source + 1];
      if (forwardArc < forwardBegin || forwardArc >= forwardEnd ||
          graph.routingArcs()[forwardArc].target != target)
        fail(__func__, "incoming entry does not identify its forward arc");
      if (incoming != begin &&
          graph.incomingForwardArcIndices()[incoming - 1] >= forwardArc)
        fail(__func__, "incoming entries are not in forward arc order");
      seenForwardArc[forwardArc] = true;
    }
  }
  if (std::find(seenForwardArc.begin(), seenForwardArc.end(), false) !=
      seenForwardArc.end())
    fail(__func__, "incoming CSR omitted a forward arc");
}

void validatesResourceTraversalStructure() {
  {
    TestCase testCase = makeTraversalCase();
    testCase.fabric.transportResources[1].endpoints.push_back(
        transportEndpoint(4002, PortDirection::Input));
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeTraversalCase();
    testCase.fabric.transportTraversals.pop_back();
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeTraversalCase();
    testCase.fabric.transportTraversals.front().target =
        endpointRef(testCase.fabric.identity, 4001);
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeTraversalCase();
    testCase.fabric.transportResources.back().boundaryDirection.reset();
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
}

void rejectsUnmediatedNativeTransportCrossing() {
  {
    TestCase testCase = makeTraversalCase();
    auto outputIt =
        llvm::find_if(testCase.fabric.computeOccurrences.front().endpoints,
                      [](const ComputeEndpointDescriptor &endpoint) {
                        return endpoint.id == ComputeEndpointId(1000);
                      });
    if (outputIt == testCase.fabric.computeOccurrences.front().endpoints.end())
      fail(__func__, "routable output endpoint is missing");
    ComputeEndpointDescriptor &output = *outputIt;
    output.transportKind = ::fabric::DataPathKind::Bits;
    output.tagCapacityBits = 0;
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeTraversalCase();
    testCase.fabric.transportArcs.clear();
    TransportEndpointDescriptor &output =
        testCase.fabric.transportResources.front().endpoints[1];
    output.transportKind = ::fabric::DataPathKind::Bits;
    output.tagCapacityBits = 0;
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeTraversalCase();
    testCase.fabric.transportArcs.clear();
    TransportEndpointDescriptor &output =
        testCase.fabric.transportResources[1].endpoints[1];
    output.transportKind = ::fabric::DataPathKind::Bits;
    output.tagCapacityBits = 0;
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
}

void rejectsMemoryTransportTraversal() {
  TestCase testCase = makeValidCase();
  const ArtifactIdentity &fabric = testCase.fabric.identity;
  testCase.fabric.transportResources = {
      {TransportResourceId(300),
       TransportResourceKind::Switch,
       {transportEndpoint(3000, PortDirection::Input, 32, 0,
                          ::fabric::DataPathKind::Bits, PortKind::Memory),
        transportEndpoint(3001, PortDirection::Output, 32, 0,
                          ::fabric::DataPathKind::Bits, PortKind::Memory)}}};
  testCase.fabric.transportTraversals = {{resourceRef(fabric, 300),
                                          endpointRef(fabric, 3000),
                                          endpointRef(fabric, 3001)}};
  expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
}

void excludesDisconnectedMemoryComputeEndpoint() {
  TestCase testCase = makeValidCase();
  const ArtifactIdentity &fabric = testCase.fabric.identity;
  testCase.fabric.functionalUnits.push_back(
      {::loom::fabric::FabricFuTemplateRef(900),
       {},
       {port(PortKind::Memory, type(9), 32)},
       {}});
  const FuDescriptor &memoryFu = testCase.fabric.functionalUnits.back();
  testCase.fabric.computeOccurrences.push_back(
      routableOccurrence(fabric, memoryFu, 901, 9000, PortDirection::Output, 32,
                         0, ::fabric::DataPathKind::Bits, PortKind::Memory));
  FrozenModelHandle model = validateAndFreezeRouting(__func__, testCase);
  const FrozenRoutingGraph &graph = model->routing();
  for (const FrozenRoutingEndpoint &endpoint : graph.routingEndpoints())
    if (endpoint.id == TransportEndpointId(9000))
      fail(__func__, "memory compute endpoint entered token routing CSR");
}

void freezesExactBoundaryConversions() {
  struct BoundaryCase {
    ::fabric::BoundaryDirection direction;
    ::fabric::DataPathKind inputKind;
    std::uint32_t inputTagBits;
    ::fabric::DataPathKind outputKind;
    std::uint32_t outputTagBits;
  };
  const BoundaryCase cases[] = {
      {::fabric::BoundaryDirection::S2t, ::fabric::DataPathKind::Bits, 0,
       ::fabric::DataPathKind::BitsTag, 4},
      {::fabric::BoundaryDirection::T2t, ::fabric::DataPathKind::BitsTag, 4,
       ::fabric::DataPathKind::BitsTag, 8},
      {::fabric::BoundaryDirection::T2s, ::fabric::DataPathKind::BitsTag, 4,
       ::fabric::DataPathKind::Bits, 0},
  };
  for (const BoundaryCase &boundary : cases) {
    TestCase testCase = makeBoundaryCase(
        boundary.direction, boundary.inputKind, 32, boundary.inputTagBits,
        boundary.outputKind, 32, boundary.outputTagBits);
    FrozenModelHandle model = validateAndFreezeRouting(__func__, testCase);
    const FrozenRoutingGraph &graph = model->routing();
    if (graph.transportResources().size() != 1 ||
        graph.transportResources().front().boundaryDirection !=
            boundary.direction)
      fail(__func__, "frozen boundary direction is not exact");
    const FrozenRoutingEndpoint &input = graph.routingEndpoints()[endpointIndex(
        __func__, graph, TransportEndpointId(3000))];
    const FrozenRoutingEndpoint &output =
        graph.routingEndpoints()[endpointIndex(__func__, graph,
                                               TransportEndpointId(3001))];
    if (input.transportKind != boundary.inputKind ||
        input.payloadCapacityBits != 32 ||
        input.tagCapacityBits != boundary.inputTagBits ||
        output.transportKind != boundary.outputKind ||
        output.payloadCapacityBits != 32 ||
        output.tagCapacityBits != boundary.outputTagBits ||
        !findArc(graph, TransportEndpointId(3000), TransportEndpointId(3001)))
      fail(__func__, "frozen boundary conversion changed its typed relation");
  }
}

void rejectsMalformedBoundaryConversions() {
  {
    TestCase testCase = makeBoundaryCase(
        ::fabric::BoundaryDirection::S2t, ::fabric::DataPathKind::BitsTag, 32,
        4, ::fabric::DataPathKind::BitsTag, 32, 4);
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeBoundaryCase(::fabric::BoundaryDirection::T2s,
                                         ::fabric::DataPathKind::BitsTag, 32, 4,
                                         ::fabric::DataPathKind::Bits, 16, 0);
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeBoundaryCase(
        ::fabric::BoundaryDirection::S2t, ::fabric::DataPathKind::Bits, 32, 1,
        ::fabric::DataPathKind::BitsTag, 32, 4);
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeBoundaryCase(
        ::fabric::BoundaryDirection::S2t, ::fabric::DataPathKind::Bits, 32, 0,
        ::fabric::DataPathKind::BitsTag, 32, 0);
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase =
        makeBoundaryCase(static_cast<::fabric::BoundaryDirection>(99),
                         ::fabric::DataPathKind::Bits, 32, 0,
                         ::fabric::DataPathKind::BitsTag, 32, 4);
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
}

void rejectsDuplicateForeignWrongKindAndWrongDirectionReferences() {
  {
    TestCase testCase = makeTraversalCase();
    testCase.fabric.transportArcs.push_back(
        testCase.fabric.transportArcs.front());
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeTraversalCase();
    testCase.fabric.transportResources.front().endpoints.front().id =
        TransportEndpointId(1000);
    expectMapError(__func__, testCase, MappingErrorCode::DuplicateEntityId);
  }
  {
    TestCase testCase = makeTraversalCase();
    testCase.fabric.transportArcs.front().source.artifact = artifact(99);
    expectMapError(__func__, testCase, MappingErrorCode::ForeignReference);
  }
  {
    TestCase testCase = makeTraversalCase();
    testCase.fabric.transportArcs.front().source.entity =
        TransportEndpointId(300);
    expectMapError(__func__, testCase, MappingErrorCode::WrongEntityKind);
  }
  {
    TestCase testCase = makeTraversalCase();
    std::swap(testCase.fabric.transportArcs.front().source,
              testCase.fabric.transportArcs.front().target);
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
}

void freezesStructurallyAcrossDescriptorPermutation() {
  TestCase baselineCase = makeIrregularReachabilityCase();
  FrozenModelHandle baselineModel =
      validateAndFreezeRouting(__func__, baselineCase);
  const FrozenRoutingGraph &baseline = baselineModel->routing();
  expectCanonicalTableOrdering(__func__, baseline);
  const std::size_t target =
      endpointIndex(__func__, baseline, TransportEndpointId(5002));
  const std::size_t firstSource =
      endpointIndex(__func__, baseline, TransportEndpointId(5000));
  const std::size_t secondSource =
      endpointIndex(__func__, baseline, TransportEndpointId(5001));
  const PnrIndex incomingBegin = baseline.incomingAdjacencyOffsets()[target];
  const PnrIndex incomingEnd = baseline.incomingAdjacencyOffsets()[target + 1];
  if (incomingEnd - incomingBegin != 2)
    fail(__func__, "multi-incoming CSR has the wrong prefix-sum range");
  const PnrIndex firstArc = baseline.incomingForwardArcIndices()[incomingBegin];
  const PnrIndex secondArc =
      baseline.incomingForwardArcIndices()[incomingBegin + 1];
  if (firstArc >= secondArc ||
      baseline.incomingSourceVertices()[incomingBegin] != firstSource ||
      baseline.incomingSourceVertices()[incomingBegin + 1] != secondSource ||
      firstArc < baseline.adjacencyOffsets()[firstSource] ||
      firstArc >= baseline.adjacencyOffsets()[firstSource + 1] ||
      secondArc < baseline.adjacencyOffsets()[secondSource] ||
      secondArc >= baseline.adjacencyOffsets()[secondSource + 1] ||
      baseline.routingArcs()[firstArc].target != target ||
      baseline.routingArcs()[secondArc].target != target)
    fail(__func__,
         "multi-incoming CSR lost canonical forward-arc identity order");
  TestCase permutedCase = makeIrregularReachabilityCase();
  std::reverse(permutedCase.fabric.computeOccurrences.begin(),
               permutedCase.fabric.computeOccurrences.end());
  for (ComputeOccurrenceDescriptor &occurrence :
       permutedCase.fabric.computeOccurrences) {
    std::reverse(occurrence.endpoints.begin(), occurrence.endpoints.end());
    std::reverse(occurrence.localArcs.begin(), occurrence.localArcs.end());
  }
  std::reverse(permutedCase.fabric.transportResources.begin(),
               permutedCase.fabric.transportResources.end());
  for (TransportResourceDescriptor &resource :
       permutedCase.fabric.transportResources)
    std::reverse(resource.endpoints.begin(), resource.endpoints.end());
  std::reverse(permutedCase.fabric.transportArcs.begin(),
               permutedCase.fabric.transportArcs.end());
  std::reverse(permutedCase.fabric.transportTraversals.begin(),
               permutedCase.fabric.transportTraversals.end());
  FrozenModelHandle permutedModel =
      validateAndFreezeRouting(__func__, permutedCase);
  const FrozenRoutingGraph &permuted = permutedModel->routing();
  expectCanonicalTableOrdering(__func__, permuted);
  if (baseline != permuted)
    fail(__func__, "descriptor permutation changed frozen routing structure");
}

void preservesIrregularReachabilityWithoutTopologyAssumptions() {
  TestCase testCase = makeIrregularReachabilityCase();
  FrozenModelHandle model = validateAndFreezeRouting(__func__, testCase);
  const FrozenRoutingGraph &graph = model->routing();
  if (!reachable(graph, TransportEndpointId(1000), TransportEndpointId(3000)) ||
      !reachable(graph, TransportEndpointId(1000), TransportEndpointId(4000)) ||
      !reachable(graph, TransportEndpointId(2000), TransportEndpointId(3000)) ||
      reachable(graph, TransportEndpointId(2000), TransportEndpointId(4000)))
    fail(__func__, "irregular directed reachability was changed");
}

void linksFactorizedComputeDomainsToRoutingVertices() {
  TestCase testCase = makeValidCase();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  ResolvedPnrConfigView config = makeSpatialPnrConfigView(__func__);
  PnrProblemInputs inputs = makePnrProblemInputs(testCase, mapping, config);
  FrozenModelHandle model =
      takeExpected(__func__, freezeSpatialPnrModel(inputs));
  const FrozenRealizationGraph &realizations = model->realizations();
  const FrozenRoutingGraph &routing = model->routing();
  if (realizations.physicalEndpoints().size() !=
      routing.computeEndpointVertices().size())
    fail(__func__, "compute endpoint projection sizes disagree");
  for (std::size_t index = 0; index < realizations.physicalEndpoints().size();
       ++index) {
    const PnrIndex vertex = routing.computeEndpointVertices()[index];
    if (routing.routingEndpoints()[vertex].id !=
        realizations.physicalEndpoints()[index].id)
      fail(__func__, "factorized endpoint domain cannot address routing graph");
  }
}

void acceptsDisconnectedTopologyAndChecksNativeCapacity() {
  TestCase testCase = makeValidCase();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  ResolvedPnrConfigView config = makeSpatialPnrConfigView(__func__);
  FrozenModelHandle model = takeExpected(
      __func__,
      freezeSpatialPnrModel(makePnrProblemInputs(testCase, mapping, config)));
  const FrozenRoutingGraph &graph = model->routing();
  if (!graph.routingArcs().empty() ||
      graph.incomingAdjacencyOffsets().size() !=
          graph.routingEndpoints().size() + 1 ||
      graph.incomingAdjacencyOffsets().back() != 0 ||
      !graph.incomingSourceVertices().empty() ||
      !graph.incomingForwardArcIndices().empty())
    fail(__func__, "disconnected topology gained implicit adjacency");

  TestCase emptyCase = makeValidCase();
  emptyCase.dataflow.graphs.front().inputPorts.clear();
  emptyCase.dataflow.graphs.front().outputPorts.clear();
  emptyCase.dataflow.actors.clear();
  emptyCase.dataflow.edges.clear();
  emptyCase.dataflow.logicalMemoryRoots.clear();
  emptyCase.fabric.computeOccurrences.clear();
  emptyCase.mapping.realizations.clear();
  emptyCase.mapping.memoryRealizations.clear();
  ValidatedTechMapping emptyMapping = validateCase(__func__, emptyCase);
  ResolvedPnrConfigView emptyConfig = makeSpatialPnrConfigView(__func__);
  FrozenModelHandle emptyModel =
      takeExpected(__func__, freezeSpatialPnrModel(makePnrProblemInputs(
                                 emptyCase, emptyMapping, emptyConfig)));
  const FrozenRoutingGraph &emptyGraph = emptyModel->routing();
  if (!emptyGraph.routingEndpoints().empty() ||
      emptyGraph.adjacencyOffsets().size() != 1 ||
      emptyGraph.adjacencyOffsets().front() != 0 ||
      !emptyGraph.routingArcs().empty() ||
      emptyGraph.incomingAdjacencyOffsets().size() != 1 ||
      emptyGraph.incomingAdjacencyOffsets().front() != 0 ||
      !emptyGraph.incomingSourceVertices().empty() ||
      !emptyGraph.incomingForwardArcIndices().empty())
    fail(__func__, "empty routing graph has nonempty adjacency");

  llvm::Error error = loom::pnr::detail::preflightFrozenRoutingGraphCapacity(
      getPnrIndexMax(), 0, 0, 0);
  if (!error)
    fail(__func__, "expected routing CSR capacity failure");
  bool sawCapacityError = false;
  llvm::handleAllErrors(
      std::move(error), [&](const PnrIndexCapacityError &capacityError) {
        sawCapacityError = true;
        std::string message;
        llvm::raw_string_ostream stream(message);
        capacityError.log(stream);
        if (message.find("table 'adjacency_offsets'") == std::string::npos)
          fail(__func__, "capacity failure named the wrong routing table");
      });
  if (!sawCapacityError)
    fail(__func__, "received a different routing capacity error category");
}

template <typename T> constexpr bool isPnrIndex = std::is_same_v<T, PnrIndex>;
static_assert(
    isPnrIndex<
        decltype(std::declval<FrozenTransportResource>().endpointOffset)> &&
    isPnrIndex<decltype(std::declval<FrozenRoutingEndpoint>().owner)> &&
    isPnrIndex<decltype(std::declval<FrozenRoutingArc>().target)>);

struct RoutingFreezeTestRunner {
  RoutingFreezeTestRunner() {
    freezesExactDirectedAdjacencyAndIndependentCapacity();
    mirrorsEveryForwardArcInIncomingCsr();
    validatesResourceTraversalStructure();
    rejectsUnmediatedNativeTransportCrossing();
    rejectsMemoryTransportTraversal();
    excludesDisconnectedMemoryComputeEndpoint();
    freezesExactBoundaryConversions();
    rejectsMalformedBoundaryConversions();
    rejectsDuplicateForeignWrongKindAndWrongDirectionReferences();
    freezesStructurallyAcrossDescriptorPermutation();
    preservesIrregularReachabilityWithoutTopologyAssumptions();
    linksFactorizedComputeDomainsToRoutingVertices();
    acceptsDisconnectedTopologyAndChecksNativeCapacity();
  }
};

const RoutingFreezeTestRunner routingFreezeTestRunner;

} // namespace
} // namespace loom::mapping::test
