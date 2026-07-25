#include "MappingCoreTestSupport.h"

#include "PnR/FrozenRoutingGraph.h"

#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace loom::mapping::test {
namespace {

template <typename T, typename = void>
struct HasLegacyRealizationAccessor : std::false_type {};

template <typename T>
struct HasLegacyRealizationAccessor<
    T, std::void_t<decltype(std::declval<const T &>().realization())>>
    : std::true_type {};

static_assert(!HasLegacyRealizationAccessor<FrozenMappingInfeasibility>::value);

bool containsExternalEdge(const FrozenRealizationGraph &graph,
                          const DataflowEdge &edge) {
  for (const FrozenLogicalNet &net : graph.logicalNets()) {
    if (net.producer != edge.source)
      continue;
    const auto sinks =
        graph.logicalNetSinks().slice(net.sinkOffset, net.sinkCount);
    if (std::any_of(sinks.begin(), sinks.end(), [&](const auto &sink) {
          return sink.consumer == edge.target;
        }))
      return true;
  }
  return false;
}
bool isGraphMemoryCapabilityPort(const FrozenTerminal &terminal) {
  const auto *graph = std::get_if<FrozenGraphBoundaryTerminal>(&terminal);
  return graph && graph->port == 0;
}
TransportEndpointRef endpointRef(const ArtifactIdentity &fabric,
                                 std::uint64_t id) {
  return TransportEndpointRef{fabric, TransportEndpointId(id)};
}
TransportResourceRef resourceRef(const ArtifactIdentity &fabric,
                                 std::uint64_t id) {
  return TransportResourceRef{fabric, TransportResourceId(id)};
}
TransportEndpointDescriptor
transportEndpoint(std::uint64_t id, PortDirection direction,
                  std::uint32_t payloadCapacityBits) {
  return {TransportEndpointId(id), direction, PortKind::Value,
          payloadCapacityBits,     0,         ::fabric::DataPathKind::Bits};
}
void selectAddressInternalMemoryGraph(TestCase &testCase) {
  selectInternalMemoryGraph(testCase);
  const ArtifactIdentity &dataflow = testCase.dataflow.identity;
  const ArtifactIdentity &fabric = testCase.fabric.identity;
  MemoryRealizationDraft &realization = testCase.mapping.memoryRealizations[0];
  const MemoryBoundaryPortCorrespondence loadControl =
      realization.boundaryPorts[1];
  const MemoryBoundaryPortCorrespondence loadResult =
      realization.boundaryPorts[2];
  const MemoryBoundaryPortCorrespondence storeData =
      realization.boundaryPorts[4];
  const MemoryBoundaryPortCorrespondence storeDone =
      realization.boundaryPorts[5];
  realization.boundaryPorts = {loadControl, loadResult, storeData, storeDone};
  testCase.fabric.memorySemanticEncodings[2].internalConnections = {
      MemoryInternalConnectionId(37), MemoryInternalConnectionId(38),
      MemoryInternalConnectionId(39)};
  const GraphRef graph{dataflow, GraphId(1)};
  const MemoryImplementationRef implementation{fabric,
                                               MemoryImplementationId(32)};
  realization.graphBoundaryPorts = {
      {GraphPortRef{graph, PortDirection::Input, 1},
       MemoryImplementationBoundaryPortRef{implementation, 0}}};
  realization.internalEdges.insert(
      realization.internalEdges.begin(),
      {{DataflowEdgeRef{
            dataflow,
            DataflowEdge{GraphPort{GraphId(1), PortDirection::Input, 1},
                         ActorPort{ActorId(2), PortDirection::Input, 0}}},
        MemoryInternalConnectionRef{fabric, MemoryInternalConnectionId(37)}},
       {DataflowEdgeRef{
            dataflow,
            DataflowEdge{GraphPort{GraphId(1), PortDirection::Input, 1},
                         ActorPort{ActorId(8), PortDirection::Input, 0}}},
        MemoryInternalConnectionRef{fabric, MemoryInternalConnectionId(38)}}});
}
TestCase makeMemoryRouteDomainCase() {
  TestCase testCase = makeMemoryAnchorCase();
  selectInternalMemoryGraph(testCase);
  const ArtifactIdentity &fabric = testCase.fabric.identity;
  testCase.fabric.computeOccurrences.push_back(
      makeSpatialComputeOccurrence(fabric, ComputeOccurrenceId(1100),
                                   testCase.fabric.functionalUnits[0], 2400));
  testCase.fabric.computeOccurrences.push_back(
      makeSpatialComputeOccurrence(fabric, ComputeOccurrenceId(1103),
                                   testCase.fabric.functionalUnits[3], 2500));

  testCase.fabric.transportResources = {
      {TransportResourceId(5000),
       TransportResourceKind::Switch,
       {transportEndpoint(50000, PortDirection::Input, 32),
        transportEndpoint(50001, PortDirection::Output, 32),
        transportEndpoint(50002, PortDirection::Output, 8),
        transportEndpoint(53000, PortDirection::Input, 64),
        transportEndpoint(53001, PortDirection::Input, 8),
        transportEndpoint(53002, PortDirection::Output, 64)}},
      {TransportResourceId(5100),
       TransportResourceKind::Switch,
       {transportEndpoint(51000, PortDirection::Input, 64),
        transportEndpoint(51001, PortDirection::Input, 8),
        transportEndpoint(51002, PortDirection::Output, 64),
        transportEndpoint(52000, PortDirection::Input, 32),
        transportEndpoint(52001, PortDirection::Output, 32),
        transportEndpoint(52002, PortDirection::Output, 8)}}};
  testCase.fabric.transportArcs = {
      {endpointRef(fabric, 30002), endpointRef(fabric, 50000)},
      {endpointRef(fabric, 50001), endpointRef(fabric, 2000)},
      {endpointRef(fabric, 50002), endpointRef(fabric, 51001)},
      {endpointRef(fabric, 40002), endpointRef(fabric, 51000)},
      {endpointRef(fabric, 51002), endpointRef(fabric, 2400)},
      {endpointRef(fabric, 2302), endpointRef(fabric, 52000)},
      {endpointRef(fabric, 52001), endpointRef(fabric, 40004)},
      {endpointRef(fabric, 52002), endpointRef(fabric, 53001)},
      {endpointRef(fabric, 2502), endpointRef(fabric, 53000)},
      {endpointRef(fabric, 53002), endpointRef(fabric, 30004)}};
  testCase.fabric.transportTraversals = {
      {resourceRef(fabric, 5000), endpointRef(fabric, 50000),
       endpointRef(fabric, 50001)},
      {resourceRef(fabric, 5000), endpointRef(fabric, 50000),
       endpointRef(fabric, 50002)},
      {resourceRef(fabric, 5100), endpointRef(fabric, 51000),
       endpointRef(fabric, 51002)},
      {resourceRef(fabric, 5100), endpointRef(fabric, 51001),
       endpointRef(fabric, 51002)},
      {resourceRef(fabric, 5100), endpointRef(fabric, 52000),
       endpointRef(fabric, 52001)},
      {resourceRef(fabric, 5100), endpointRef(fabric, 52000),
       endpointRef(fabric, 52002)},
      {resourceRef(fabric, 5000), endpointRef(fabric, 53000),
       endpointRef(fabric, 53002)},
      {resourceRef(fabric, 5000), endpointRef(fabric, 53001),
       endpointRef(fabric, 53002)}};
  return testCase;
}
const FrozenMemoryRealization &
memoryRealization(const char *test, const FrozenRealizationGraph &graph,
                  MemoryRealizationId id) {
  const auto found =
      llvm::find_if(graph.memoryRealizations(),
                    [&](const FrozenMemoryRealization &candidate) {
                      return candidate.id == id;
                    });
  if (found == graph.memoryRealizations().end())
    fail(test, "memory realization is missing");
  return *found;
}
const FrozenComputeRealization &
computeRealization(const char *test, const FrozenRealizationGraph &graph,
                   ComputeRealizationId id) {
  const auto found =
      llvm::find_if(graph.computeRealizations(),
                    [&](const FrozenComputeRealization &candidate) {
                      return candidate.id == id;
                    });
  if (found == graph.computeRealizations().end())
    fail(test, "compute realization is missing");
  return *found;
}
const FrozenMemoryImplementationOccurrence &
memoryCandidate(const char *test, const FrozenRealizationGraph &graph,
                const FrozenMemoryRealization &realization,
                MemoryOccurrenceId occurrence) {
  const auto candidates = graph.memoryImplementationOccurrences().slice(
      realization.implDomainOffset, realization.implDomainCount);
  const auto found = llvm::find_if(candidates, [&](const auto &candidate) {
    return candidate.memoryOccurrence.occurrence == occurrence;
  });
  if (found == candidates.end())
    fail(test, "memory candidate is missing");
  return *found;
}
const FrozenImplementationOccurrence &
computeCandidate(const char *test, const FrozenRealizationGraph &graph,
                 const FrozenComputeRealization &realization,
                 ComputeOccurrenceId occurrence) {
  const auto candidates = graph.implementationOccurrences().slice(
      realization.implDomainOffset, realization.implDomainCount);
  const auto found = llvm::find_if(candidates, [&](const auto &candidate) {
    return candidate.fuOccurrence.parentPe.occurrence == occurrence;
  });
  if (found == candidates.end())
    fail(test, "compute candidate is missing");
  return *found;
}
void expectMemoryInfeasibility(const char *test,
                               llvm::Expected<FrozenRealizationGraph> result,
                               FrozenMappingInfeasibilityCode expectedCode,
                               MemoryRealizationId expectedRealization) {
  if (result)
    fail(test, "expected memory mapping infeasibility");
  bool sawInfeasibility = false;
  llvm::Error remaining = llvm::handleErrors(
      result.takeError(),
      [&](const FrozenMappingInfeasibility &error) -> llvm::Error {
        sawInfeasibility = true;
        if (error.code() != expectedCode || error.computeRealization() ||
            error.memoryRealization() == nullptr ||
            *error.memoryRealization() != expectedRealization)
          fail(test, "memory infeasibility has the wrong typed identity");
        const auto *variantRealization =
            std::get_if<MemoryRealizationId>(&error.realizationId());
        if (variantRealization == nullptr ||
            *variantRealization != expectedRealization)
          fail(test, "memory infeasibility variant identity is wrong");
        return llvm::Error::success();
      });
  if (remaining) {
    llvm::consumeError(std::move(remaining));
    fail(test, "received a different freeze error category");
  }
  if (!sawInfeasibility)
    fail(test, "memory mapping infeasibility was not handled");
}
PnrIndex memoryVertex(const char *test, const FrozenRealizationGraph &graph,
                      const FrozenRoutingGraph &routing,
                      const FrozenMemoryImplementationOccurrence &candidate,
                      MemoryOperationPortTemplateId operation,
                      PortDirection direction, PnrIndex port) {
  const auto demands = graph.memoryPortDemands().slice(
      candidate.portDemandOffset, candidate.portDemandCount);
  const auto found = llvm::find_if(demands, [&](const auto &demand) {
    return demand.operation == operation && demand.direction == direction &&
           demand.port == port;
  });
  if (found == demands.end() || found->endpointCount != 1)
    fail(test, "memory endpoint domain is not a singleton");
  const PnrIndex physical =
      graph.compatibleMemoryEndpoints()[found->endpointOffset];
  return routing.memoryEndpointVertices()[physical];
}
PnrIndex computeVertex(const char *test, const FrozenRealizationGraph &graph,
                       const FrozenRoutingGraph &routing,
                       const FrozenImplementationOccurrence &candidate,
                       PortDirection direction, PnrIndex port) {
  const auto demands = graph.portDemands().slice(candidate.portDemandOffset,
                                                 candidate.portDemandCount);
  const auto found = llvm::find_if(demands, [&](const auto &demand) {
    return demand.direction == direction && demand.port == port;
  });
  if (found == demands.end() || found->endpointCount != 1)
    fail(test, "compute endpoint domain is not a singleton");
  const PnrIndex physical = graph.compatibleEndpoints()[found->endpointOffset];
  return routing.computeEndpointVertices()[physical];
}
const FrozenRoutingArc &routingArc(const char *test,
                                   const FrozenRoutingGraph &routing,
                                   TransportEndpointId source,
                                   TransportEndpointId target) {
  const auto sourceVertex = llvm::find_if(
      routing.routingEndpoints(), [&](const FrozenRoutingEndpoint &endpoint) {
        return endpoint.id == source;
      });
  const auto targetVertex = llvm::find_if(
      routing.routingEndpoints(), [&](const FrozenRoutingEndpoint &endpoint) {
        return endpoint.id == target;
      });
  if (sourceVertex == routing.routingEndpoints().end() ||
      targetVertex == routing.routingEndpoints().end())
    fail(test, "routing arc endpoint is missing");
  const PnrIndex sourceIndex =
      static_cast<PnrIndex>(sourceVertex - routing.routingEndpoints().begin());
  const PnrIndex targetIndex =
      static_cast<PnrIndex>(targetVertex - routing.routingEndpoints().begin());
  for (PnrIndex arc = routing.adjacencyOffsets()[sourceIndex];
       arc < routing.adjacencyOffsets()[sourceIndex + 1]; ++arc) {
    if (routing.routingArcs()[arc].target == targetIndex)
      return routing.routingArcs()[arc];
  }
  fail(test, "routing arc is missing");
}
void freezesInternalMemoryAnchor() {
  TestCase testCase = makeMemoryAnchorCase();
  selectInternalMemoryGraph(testCase);
  FrozenRealizationGraph graph = validateAndFreeze(__func__, testCase);
  if (graph.actorOwnerships().size() != 7)
    fail(__func__, "frozen graph has the wrong actor count");
  if (graph.computeRealizations().size() != 4)
    fail(__func__, "frozen graph has the wrong compute realization count");
  if (graph.memoryRealizations().size() != 1)
    fail(__func__, "frozen graph has the wrong memory realization count");
  const DataflowEdge xoriToPreAdd{
      ActorPort{ActorId(3), PortDirection::Output, 0},
      ActorPort{ActorId(4), PortDirection::Input, 0}};
  const DataflowEdge loadDoneToStore{
      ActorPort{ActorId(2), PortDirection::Output, 1},
      ActorPort{ActorId(8), PortDirection::Input, 2}};
  if (containsExternalEdge(graph, xoriToPreAdd) ||
      containsExternalEdge(graph, loadDoneToStore))
    fail(__func__, "internal edge escaped into the external net cache");
  const DataflowEdge loadAddress{
      GraphPort{GraphId(1), PortDirection::Input, 1},
      ActorPort{ActorId(2), PortDirection::Input, 0}};
  const DataflowEdge storeAddress{
      GraphPort{GraphId(1), PortDirection::Input, 1},
      ActorPort{ActorId(8), PortDirection::Input, 0}};
  const DataflowEdge storeDone{ActorPort{ActorId(8), PortDirection::Output, 0},
                               GraphPort{GraphId(1), PortDirection::Output, 1}};
  if (!containsExternalEdge(graph, loadAddress) ||
      !containsExternalEdge(graph, storeAddress) ||
      !containsExternalEdge(graph, storeDone))
    fail(__func__, "external memory edge was absorbed by the realization");
  const FrozenLogicalNet *addressFanout = nullptr;
  for (const FrozenLogicalNet &net : graph.logicalNets()) {
    const auto *source = std::get_if<FrozenGraphBoundaryTerminal>(&net.source);
    if (source && source->direction == PortDirection::Input &&
        source->port == 1) {
      addressFanout = &net;
      break;
    }
  }
  if (!addressFanout || addressFanout->sinkCount != 2 ||
      addressFanout->producer != loadAddress.source ||
      graph.logicalNetSinks()[addressFanout->sinkOffset].consumer !=
          loadAddress.target ||
      graph.logicalNetSinks()[addressFanout->sinkOffset + 1].consumer !=
          storeAddress.target)
    fail(__func__, "address fanout did not remain one external logical net");
  const FrozenLogicalNet *fanout = nullptr;
  for (const FrozenLogicalNet &net : graph.logicalNets()) {
    const auto *reference = std::get_if<FrozenTemplateTerminalRef>(&net.source);
    if (!reference)
      continue;
    const auto *source = std::get_if<FrozenComputeTemplateTerminal>(
        &graph.templateTerminals()[reference->terminal]);
    if (!source)
      continue;
    const FrozenComputeRealization &realization =
        graph.computeRealizations()[source->realization];
    if (realization.id == ComputeRealizationId(50) &&
        source->direction == PortDirection::Output && source->port == 0) {
      fanout = &net;
      break;
    }
  }
  if (!fanout)
    fail(__func__, "preAdd external fanout net is missing");
  if (fanout->sinkCount != 2)
    fail(__func__, "preAdd external fanout was not grouped into one net");
  const FrozenLogicalNetSink &first =
      graph.logicalNetSinks()[fanout->sinkOffset];
  const FrozenLogicalNetSink &second =
      graph.logicalNetSinks()[fanout->sinkOffset + 1];
  const DataflowEdge multiplyInput{
      ActorPort{ActorId(4), PortDirection::Output, 0},
      ActorPort{ActorId(5), PortDirection::Input, 0}};
  const DataflowEdge subtractInput{
      ActorPort{ActorId(4), PortDirection::Output, 0},
      ActorPort{ActorId(6), PortDirection::Input, 0}};
  if (fanout->producer != multiplyInput.source ||
      first.consumer != multiplyInput.target ||
      second.consumer != subtractInput.target)
    fail(__func__, "preAdd fanout sinks are not deterministic");
  if (graph.memoryServiceObligations().size() != 1)
    fail(__func__, "logical root did not deduplicate to one service");
  const FrozenMemoryServiceObligation &service =
      graph.memoryServiceObligations().front();
  if (service.root != LogicalMemoryRootId(20) ||
      service.service != MemoryServiceDomainId(30))
    fail(__func__, "logical root resolved to the wrong memory service");
  for (const FrozenLogicalNet &net : graph.logicalNets()) {
    if (isGraphMemoryCapabilityPort(net.source))
      fail(__func__, "graph memory capability port became a token source");
  }
  for (const FrozenLogicalNetSink &sink : graph.logicalNetSinks()) {
    if (isGraphMemoryCapabilityPort(sink.terminal))
      fail(__func__, "graph memory capability port became a token sink");
  }
}
void rejectsInconsistentFrozenMemoryService() {
  TestCase testCase = makeMemoryAnchorCase();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  testCase.fabric.memorySemanticEncodings[1].implementation =
      MemoryImplementationId(33);
  ResolvedPnrConfigView config;
  expectAnyError(__func__, freezeRealizationGraph(makePnrProblemInputs(
                               testCase, mapping, config)));
}
void acceptsExternalAndInternalMemoryAnchor() {
  {
    TestCase testCase = makeMemoryAnchorCase();
    auto result =
        validateTechMapping(testCase.techMappingIdentity, testCase.mapping,
                            testCase.dataflow, testCase.fabric);
    if (!result)
      fail(__func__, llvm::toString(result.takeError()).c_str());
    if (result->realizations().size() != 4 ||
        result->memoryRealizations().size() != 2)
      fail(__func__, "external memory anchor lost a realization");
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    selectInternalMemoryGraph(testCase);
    auto result =
        validateTechMapping(testCase.techMappingIdentity, testCase.mapping,
                            testCase.dataflow, testCase.fabric);
    if (!result)
      fail(__func__, llvm::toString(result.takeError()).c_str());
    if (result->memoryRealizations().size() != 1)
      fail(__func__, "internal memory anchor lost its realization");
  }
}
void rejectsInexactMemoryInternalGraph() {
  {
    TestCase testCase = makeMemoryAnchorCase();
    selectInternalMemoryGraph(testCase);
    testCase.mapping.memoryRealizations[0].internalEdges.pop_back();
    expectMapError(__func__, testCase,
                   MappingErrorCode::InvalidInternalEdgeWitness);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    selectAddressInternalMemoryGraph(testCase);
    auto &internalEdges = testCase.mapping.memoryRealizations[0].internalEdges;
    if (internalEdges[0].edge == internalEdges[1].edge)
      fail(__func__, "fixture did not provide distinct internal edges");
    internalEdges[1].connection = internalEdges[0].connection;
    expectMapError(__func__, testCase,
                   MappingErrorCode::InvalidInternalEdgeWitness);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    selectInternalMemoryGraph(testCase);
    testCase.fabric.memorySemanticEncodings[2].internalConnections.pop_back();
    expectMapError(__func__, testCase,
                   MappingErrorCode::InvalidInternalEdgeWitness);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    selectAddressInternalMemoryGraph(testCase);
    testCase.fabric.memoryImplementations[0]
        .boundaryPorts[0]
        .maxInternalFanout = 1;
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    selectAddressInternalMemoryGraph(testCase);
    testCase.mapping.memoryRealizations[0]
        .graphBoundaryPorts[0]
        .implementationPort.index = 1;
    expectMapError(__func__, testCase,
                   MappingErrorCode::IncompleteMemoryBoundaryCorrespondence);
  }
}
void rejectsForeignMemoryInternalEdgeReference() {
  TestCase testCase = makeMemoryAnchorCase();
  selectInternalMemoryGraph(testCase);
  testCase.mapping.memoryRealizations[0].internalEdges[0].edge.artifact =
      artifact(99);
  expectMapError(__func__, testCase, MappingErrorCode::ForeignReference);
}
void rejectsNoncanonicalMemoryInternalEdgeReference() {
  TestCase testCase = makeMemoryAnchorCase();
  selectInternalMemoryGraph(testCase);
  DataflowEdge &edge =
      testCase.mapping.memoryRealizations[0].internalEdges[0].edge.edge;
  edge.target = ActorPort{ActorId(2), PortDirection::Input, 1};
  expectMapError(__func__, testCase, MappingErrorCode::UnresolvedEdgeReference);
}
void freezesPhysicalMemoryOccurrenceDomains() {
  TestCase testCase = makeMemoryRouteDomainCase();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  ResolvedPnrConfigView config;
  PnrProblemInputs inputs = makePnrProblemInputs(testCase, mapping, config);
  FrozenRealizationGraph realizations =
      takeExpected(__func__, freezeRealizationGraph(inputs));
  FrozenRoutingGraph routing =
      takeExpected(__func__, freezeRoutingGraph(inputs));

  const auto memoryOccurrences = realizations.fabricMemoryOccurrences();
  if (memoryOccurrences.size() != 2 ||
      memoryOccurrences[0].ref.occurrence != MemoryOccurrenceId(3000) ||
      memoryOccurrences[1].ref.occurrence != MemoryOccurrenceId(4000) ||
      memoryOccurrences[0].ref == memoryOccurrences[1].ref)
    fail(__func__, "memory occurrence identity or ordering was lost");
  const FrozenMemoryRealization &memory =
      memoryRealization(__func__, realizations, MemoryRealizationId(60));
  if (memory.implDomainCount != 2)
    fail(__func__, "memory implementation domain has the wrong size");
  const FrozenMemoryImplementationOccurrence &memoryA =
      memoryCandidate(__func__, realizations, memory, MemoryOccurrenceId(3000));
  const FrozenMemoryImplementationOccurrence &memoryB =
      memoryCandidate(__func__, realizations, memory, MemoryOccurrenceId(4000));
  if (!memoryA.unaryEligible || !memoryB.unaryEligible)
    fail(__func__, "compatible memory occurrence was rejected");
  for (const FrozenMemoryImplementationOccurrence *candidate :
       {&memoryA, &memoryB}) {
    const auto demands = realizations.memoryPortDemands().slice(
        candidate->portDemandOffset, candidate->portDemandCount);
    if (demands.size() != 6 ||
        llvm::any_of(demands, [](const FrozenMemoryPortDemand &demand) {
          return (demand.operation == MemoryOperationPortTemplateId(34) &&
                  demand.direction == PortDirection::Output &&
                  demand.port == 3) ||
                 (demand.operation == MemoryOperationPortTemplateId(35) &&
                  demand.direction == PortDirection::Input && demand.port == 2);
        }))
      fail(__func__, "internal WAR ports escaped into physical demands");
  }

  if (realizations.memoryPhysicalEndpoints().size() !=
      routing.memoryEndpointVertices().size())
    fail(__func__, "memory endpoint projections have different sizes");
  for (std::size_t index = 0;
       index < realizations.memoryPhysicalEndpoints().size(); ++index) {
    const PnrIndex vertex = routing.memoryEndpointVertices()[index];
    if (routing.routingEndpoints()[vertex].id !=
        realizations.memoryPhysicalEndpoints()[index].id)
      fail(__func__, "memory endpoint domain cannot address routing graph");
  }
}

void derivesDirectedMemoryReachability() {
  TestCase testCase = makeMemoryRouteDomainCase();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  ResolvedPnrConfigView config;
  PnrProblemInputs inputs = makePnrProblemInputs(testCase, mapping, config);
  FrozenRealizationGraph realizations =
      takeExpected(__func__, freezeRealizationGraph(inputs));
  FrozenRoutingGraph routing =
      takeExpected(__func__, freezeRoutingGraph(inputs));

  const FrozenMemoryRealization &memory =
      memoryRealization(__func__, realizations, MemoryRealizationId(60));
  const FrozenMemoryImplementationOccurrence &memoryA =
      memoryCandidate(__func__, realizations, memory, MemoryOccurrenceId(3000));
  const FrozenMemoryImplementationOccurrence &memoryB =
      memoryCandidate(__func__, realizations, memory, MemoryOccurrenceId(4000));
  const FrozenComputeRealization &cr0 =
      computeRealization(__func__, realizations, ComputeRealizationId(50));
  const FrozenComputeRealization &cr3 =
      computeRealization(__func__, realizations, ComputeRealizationId(53));
  const FrozenImplementationOccurrence &cr0A =
      computeCandidate(__func__, realizations, cr0, ComputeOccurrenceId(1000));
  const FrozenImplementationOccurrence &cr0B =
      computeCandidate(__func__, realizations, cr0, ComputeOccurrenceId(1100));
  const FrozenImplementationOccurrence &cr3A =
      computeCandidate(__func__, realizations, cr3, ComputeOccurrenceId(1003));
  const FrozenImplementationOccurrence &cr3B =
      computeCandidate(__func__, realizations, cr3, ComputeOccurrenceId(1103));

  const PnrIndex loadA =
      memoryVertex(__func__, realizations, routing, memoryA,
                   MemoryOperationPortTemplateId(34), PortDirection::Output, 2);
  const PnrIndex loadB =
      memoryVertex(__func__, realizations, routing, memoryB,
                   MemoryOperationPortTemplateId(34), PortDirection::Output, 2);
  const PnrIndex cr0InputA = computeVertex(__func__, realizations, routing,
                                           cr0A, PortDirection::Input, 0);
  const PnrIndex cr0InputB = computeVertex(__func__, realizations, routing,
                                           cr0B, PortDirection::Input, 0);
  FrozenRoutingReachabilityScratch reachability;
  routing.computeCompatibleReachability(loadA, PortKind::Value, 16, 0,
                                        reachability);
  if (!reachability.contains(cr0InputA) || reachability.contains(cr0InputB))
    fail(__func__, "MEM_A load reachability is wrong");
  routing.computeCompatibleReachability(loadB, PortKind::Value, 16, 0,
                                        reachability);
  if (reachability.contains(cr0InputA) || !reachability.contains(cr0InputB))
    fail(__func__, "MEM_B load reachability is wrong");
  routing.computeCompatibleReachability(loadA, PortKind::Value, 8, 0,
                                        reachability);
  if (!reachability.contains(cr0InputB))
    fail(__func__, "narrow connected load path was not represented");
  routing.computeCompatibleReachability(loadA, PortKind::Stream, 8, 0,
                                        reachability);
  if (reachability.contains(cr0InputA) || reachability.contains(cr0InputB))
    fail(__func__, "route reachability ignored the required port kind");
  routing.computeCompatibleReachability(loadA, PortKind::Value, 8, 1,
                                        reachability);
  if (reachability.contains(cr0InputA) || reachability.contains(cr0InputB))
    fail(__func__, "route reachability ignored nonzero tag capacity");

  const PnrIndex cr3OutputA = computeVertex(__func__, realizations, routing,
                                            cr3A, PortDirection::Output, 0);
  const PnrIndex cr3OutputB = computeVertex(__func__, realizations, routing,
                                            cr3B, PortDirection::Output, 0);
  const PnrIndex storeA =
      memoryVertex(__func__, realizations, routing, memoryA,
                   MemoryOperationPortTemplateId(35), PortDirection::Input, 1);
  const PnrIndex storeB =
      memoryVertex(__func__, realizations, routing, memoryB,
                   MemoryOperationPortTemplateId(35), PortDirection::Input, 1);
  routing.computeCompatibleReachability(cr3OutputA, PortKind::Value, 16, 0,
                                        reachability);
  if (reachability.contains(storeA) || !reachability.contains(storeB))
    fail(__func__, "CR3_A store reachability is wrong");
  routing.computeCompatibleReachability(cr3OutputB, PortKind::Value, 16, 0,
                                        reachability);
  if (!reachability.contains(storeA) || reachability.contains(storeB))
    fail(__func__, "CR3_B store reachability is wrong");
  routing.computeCompatibleReachability(cr3OutputA, PortKind::Value, 8, 0,
                                        reachability);
  if (!reachability.contains(storeA))
    fail(__func__, "narrow connected store path was not represented");
}

void derivesMemoryRouteResourceConflicts() {
  TestCase testCase = makeMemoryRouteDomainCase();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  ResolvedPnrConfigView config;
  FrozenRoutingGraph routing = takeExpected(
      __func__,
      freezeRoutingGraph(makePnrProblemInputs(testCase, mapping, config)));

  const FrozenRoutingArc &familyALoad =
      routingArc(__func__, routing, TransportEndpointId(50000),
                 TransportEndpointId(50001));
  const FrozenRoutingArc &familyAStore =
      routingArc(__func__, routing, TransportEndpointId(53000),
                 TransportEndpointId(53002));
  const FrozenRoutingArc &familyBLoad =
      routingArc(__func__, routing, TransportEndpointId(51000),
                 TransportEndpointId(51002));
  const FrozenRoutingArc &familyBStore =
      routingArc(__func__, routing, TransportEndpointId(52000),
                 TransportEndpointId(52001));
  if (!familyALoad.resource || familyALoad.resource != familyAStore.resource ||
      !familyBLoad.resource || familyBLoad.resource != familyBStore.resource ||
      familyALoad.resource == familyBLoad.resource)
    fail(__func__, "shared route capacity did not follow resource identity");
}

void retainsDeterministicMemoryProjection() {
  TestCase testCase = makeMemoryRouteDomainCase();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  ResolvedPnrConfigView config;
  PnrProblemInputs inputs = makePnrProblemInputs(testCase, mapping, config);
  FrozenRealizationGraph realizations =
      takeExpected(__func__, freezeRealizationGraph(inputs));
  FrozenRoutingGraph routing =
      takeExpected(__func__, freezeRoutingGraph(inputs));

  testCase.fabric.memoryOccurrences.clear();
  if (realizations != takeExpected(__func__, freezeRealizationGraph(inputs)) ||
      routing != takeExpected(__func__, freezeRoutingGraph(inputs)))
    fail(__func__, "validated memory occurrence projection was not retained");

  TestCase permutedCase = makeMemoryRouteDomainCase();
  std::reverse(permutedCase.fabric.memoryOccurrences.begin(),
               permutedCase.fabric.memoryOccurrences.end());
  for (MemoryOccurrenceDescriptor &occurrence :
       permutedCase.fabric.memoryOccurrences) {
    std::reverse(occurrence.endpoints.begin(), occurrence.endpoints.end());
    std::reverse(occurrence.localArcs.begin(), occurrence.localArcs.end());
  }
  std::reverse(permutedCase.fabric.computeOccurrences.begin(),
               permutedCase.fabric.computeOccurrences.end());
  std::reverse(permutedCase.fabric.transportResources.begin(),
               permutedCase.fabric.transportResources.end());
  for (TransportResourceDescriptor &resource :
       permutedCase.fabric.transportResources)
    std::reverse(resource.endpoints.begin(), resource.endpoints.end());
  std::reverse(permutedCase.fabric.transportArcs.begin(),
               permutedCase.fabric.transportArcs.end());
  std::reverse(permutedCase.fabric.transportTraversals.begin(),
               permutedCase.fabric.transportTraversals.end());
  ValidatedTechMapping permutedMapping = validateCase(__func__, permutedCase);
  ResolvedPnrConfigView permutedConfig;
  PnrProblemInputs permutedInputs =
      makePnrProblemInputs(permutedCase, permutedMapping, permutedConfig);
  if (realizations !=
          takeExpected(__func__, freezeRealizationGraph(permutedInputs)) ||
      routing != takeExpected(__func__, freezeRoutingGraph(permutedInputs)))
    fail(__func__, "descriptor permutation changed memory route domains");
}

void rejectsInvalidMemoryOccurrence() {
  TestCase testCase = makeMemoryAnchorCase();
  auto &types =
      testCase.fabric.memoryOccurrences[0].endpoints[0].compatibleTypes;
  types.push_back(types.front());
  expectMapError(__func__, testCase, MappingErrorCode::InvalidMemoryOccurrence);
}

void reportsMemoryDomainInfeasibility() {
  {
    TestCase testCase = makeMemoryAnchorCase();
    selectInternalMemoryGraph(testCase);
    testCase.fabric.memoryOccurrences.clear();
    ValidatedTechMapping mapping = validateCase(__func__, testCase);
    ResolvedPnrConfigView config;
    expectMemoryInfeasibility(
        __func__,
        freezeRealizationGraph(makePnrProblemInputs(testCase, mapping, config)),
        FrozenMappingInfeasibilityCode::EmptyConcreteMemoryDomain,
        MemoryRealizationId(60));
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    selectInternalMemoryGraph(testCase);
    for (MemoryOccurrenceDescriptor &occurrence :
         testCase.fabric.memoryOccurrences)
      for (MemoryEndpointDescriptor &endpoint : occurrence.endpoints)
        endpoint.payloadCapacityBits = 8;
    ValidatedTechMapping mapping = validateCase(__func__, testCase);
    ResolvedPnrConfigView config;
    expectMemoryInfeasibility(
        __func__,
        freezeRealizationGraph(makePnrProblemInputs(testCase, mapping, config)),
        FrozenMappingInfeasibilityCode::EmptyMemoryUnaryEligibleDomain,
        MemoryRealizationId(60));
  }
}

void preflightsMemoryProjectionCapacity() {
#if LOOM_PNR_INDEX_BITS == 32
  llvm::Error error = loom::pnr::detail::preflightFrozenMemoryDomainsCapacity(
      0, 0, getPnrIndexMax() + std::uint64_t{1}, 0, 0);
  if (!error)
    fail(__func__, "expected memory projection capacity failure");
  bool sawCapacityError = false;
  llvm::handleAllErrors(
      std::move(error), [&](const PnrIndexCapacityError &capacityError) {
        sawCapacityError = true;
        std::string message;
        llvm::raw_string_ostream stream(message);
        capacityError.log(stream);
        if (message.find("table 'memory_physical_endpoints'") ==
            std::string::npos)
          fail(__func__, "capacity failure named the wrong memory table");
      });
  if (!sawCapacityError)
    fail(__func__, "received a different capacity error category");
#else
  if (llvm::Error error =
          loom::pnr::detail::preflightFrozenMemoryDomainsCapacity(
              0, 0, getPnrIndexMax(), 0, 0))
    fail(__func__, llvm::toString(std::move(error)).c_str());
#endif
}
void validatesCorrelatedMemoryAccessCapabilities() {
  {
    TestCase testCase = makeMemoryAnchorCase();
    testCase.dataflow.actors[0].memory->accessWidthBits = 32;
    testCase.dataflow.actors[0].memory->accessSizeBytes = 4;
    testCase.dataflow.actors[0].memory->alignmentBytes = 2;
    expectMapError(__func__, testCase,
                   MappingErrorCode::MemoryAccessIncompatible);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    auto &capabilities =
        testCase.fabric.memoryOperationPortTemplates[1].accessCapabilities;
    capabilities.erase(capabilities.begin());
    expectMapError(__func__, testCase,
                   MappingErrorCode::MemoryAccessIncompatible);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    auto &operation = testCase.fabric.memoryOperationPortTemplates[0];
    operation.physicalDataWidthBits = 16;
    operation.accessCapabilities.pop_back();
    testCase.dataflow.actors[0].memory->accessWidthBits = 32;
    testCase.dataflow.actors[0].memory->accessSizeBytes = 4;
    expectMapError(__func__, testCase,
                   MappingErrorCode::MemoryAccessIncompatible);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    auto &capabilities =
        testCase.fabric.memoryOperationPortTemplates[1].accessCapabilities;
    capabilities.push_back(capabilities.front());
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    testCase.fabric.memoryOperationPortTemplates[1]
        .accessCapabilities[0]
        .accessSizeBytes = 0;
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    testCase.fabric.memoryOperationPortTemplates[1]
        .accessCapabilities[0]
        .accessSizeBytes = 8;
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
}
void rejectsSharedMemoryOperationTemplate() {
  TestCase testCase = makeMemoryAnchorCase();
  ActorDescriptor extraLoad = testCase.dataflow.actors.front();
  extraLoad.id = ActorId(9);
  testCase.dataflow.actors.push_back(std::move(extraLoad));
  testCase.dataflow.edges.push_back(
      DataflowEdge{GraphPort{GraphId(1), PortDirection::Input, 1},
                   ActorPort{ActorId(9), PortDirection::Input, 0}});
  testCase.dataflow.edges.push_back(
      DataflowEdge{GraphPort{GraphId(1), PortDirection::Input, 2},
                   ActorPort{ActorId(9), PortDirection::Input, 1}});
  MemoryRealizationDraft &load = testCase.mapping.memoryRealizations[0];
  const ActorRef actor{testCase.dataflow.identity, ActorId(9)};
  load.actors.push_back(actor);
  ActorToMemoryOperation actorMapping = load.actorToOperations.front();
  actorMapping.actor = actor;
  load.actorToOperations.push_back(actorMapping);
  for (std::size_t index = 0; index < 2; ++index) {
    MemoryBoundaryPortCorrespondence boundary = load.boundaryPorts[index];
    boundary.actorPort.actor = actor;
    load.boundaryPorts.push_back(boundary);
  }
  expectMapError(__func__, testCase,
                 MappingErrorCode::InvalidMemoryRealization);
}
void validatesLogicalMemoryRootCapabilities() {
  {
    TestCase testCase = makeMemoryAnchorCase();
    GraphDescriptor &graph = testCase.dataflow.graphs[0];
    graph.inputPorts.push_back(port(PortKind::Memory, type(3)));
    graph.inputPorts.push_back(port(PortKind::Memory, type(4)));
    graph.outputPorts[0] = port(PortKind::Memory, type(4));
    graph.outputPorts.push_back(port(PortKind::Memory, type(5)));
    auto &roots = testCase.dataflow.logicalMemoryRoots;
    roots.push_back(LogicalMemoryRootDescriptor{
        LogicalMemoryRootId(21),
        GraphId(1),
        {GraphPort{GraphId(1), PortDirection::Input, 7},
         GraphPort{GraphId(1), PortDirection::Input, 8}},
        {}});
    roots.push_back(LogicalMemoryRootDescriptor{
        LogicalMemoryRootId(22),
        GraphId(1),
        {},
        {GraphPort{GraphId(1), PortDirection::Output, 2}}});
    roots.push_back(LogicalMemoryRootDescriptor{
        LogicalMemoryRootId(23), GraphId(1), {}, {}});
    testCase.dataflow.actors[6].memory->root = LogicalMemoryRootId(23);
    MemoryRealizationDraft &store = testCase.mapping.memoryRealizations[1];
    store.actorToOperations[0].root.entity = LogicalMemoryRootId(23);
    store.roots[0].entity = LogicalMemoryRootId(23);
    auto result =
        validateTechMapping(testCase.techMappingIdentity, testCase.mapping,
                            testCase.dataflow, testCase.fabric);
    if (!result)
      fail(__func__, llvm::toString(result.takeError()).c_str());
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    testCase.dataflow.logicalMemoryRoots.push_back(LogicalMemoryRootDescriptor{
        LogicalMemoryRootId(21),
        GraphId(1),
        {GraphPort{GraphId(1), PortDirection::Input, 0}},
        {}});
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    testCase.dataflow.edges.push_back(
        DataflowEdge{GraphPort{GraphId(1), PortDirection::Input, 0},
                     GraphPort{GraphId(1), PortDirection::Output, 0}});
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
}
void validatesAnchorMemoryServiceBindings() {
  {
    TestCase testCase = makeMemoryAnchorCase();
    MemoryRealizationDraft &store = testCase.mapping.memoryRealizations[1];
    store.actorToOperations[0].operation.entity =
        MemoryOperationPortTemplateId(36);
    store.encoding.entity = MemorySemanticEncodingId(44);
    for (MemoryBoundaryPortCorrespondence &port : store.boundaryPorts)
      port.operationPort.operation.entity = MemoryOperationPortTemplateId(36);
    expectMapError(__func__, testCase, MappingErrorCode::MemoryServiceMismatch);
  }
}
void rejectsAnchorMemoryCoverageAndReferences() {
  {
    TestCase testCase = makeMemoryAnchorCase();
    MemoryRealizationDraft duplicate =
        testCase.mapping.memoryRealizations.front();
    duplicate.id = MemoryRealizationId(62);
    testCase.mapping.memoryRealizations.push_back(std::move(duplicate));
    expectMapError(__func__, testCase,
                   MappingErrorCode::IncompleteGraphCoverage);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    testCase.mapping.memoryRealizations.pop_back();
    expectMapError(__func__, testCase,
                   MappingErrorCode::IncompleteGraphCoverage);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    testCase.mapping.realizations[0].actorToOps[0].actor.entity = ActorId(2);
    testCase.mapping.realizations[0].actorToOps[0].actor.entity = ActorId(2);
    expectMapError(__func__, testCase,
                   MappingErrorCode::WrongActorRealizationKind);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    MemoryRealizationDraft &load = testCase.mapping.memoryRealizations[0];
    load.actors[0].entity = ActorId(3);
    load.actorToOperations[0].actor.entity = ActorId(3);
    expectMapError(__func__, testCase,
                   MappingErrorCode::WrongActorRealizationKind);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    testCase.mapping.memoryRealizations[0]
        .boundaryPorts[0]
        .operationPort.index = 99;
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    testCase.mapping.memoryRealizations[0].encoding.entity =
        MemorySemanticEncodingId(10);
    expectMapError(__func__, testCase, MappingErrorCode::WrongEntityKind);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    selectInternalMemoryGraph(testCase);
    testCase.mapping.memoryRealizations[0].internalEdges[0].connection.entity =
        MemoryInternalConnectionId(34);
    expectMapError(__func__, testCase, MappingErrorCode::WrongEntityKind);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    testCase.mapping.memoryRealizations[0]
        .actorToOperations[0]
        .operation.entity = MemoryOperationPortTemplateId(35);
    expectMapError(__func__, testCase,
                   MappingErrorCode::MemoryOperationMismatch);
  }
}

} // namespace

void runMemoryMappingTests() {
  freezesInternalMemoryAnchor();
  freezesPhysicalMemoryOccurrenceDomains();
  derivesDirectedMemoryReachability();
  derivesMemoryRouteResourceConflicts();
  retainsDeterministicMemoryProjection();
  rejectsInvalidMemoryOccurrence();
  reportsMemoryDomainInfeasibility();
  preflightsMemoryProjectionCapacity();
  rejectsInconsistentFrozenMemoryService();
  acceptsExternalAndInternalMemoryAnchor();
  rejectsInexactMemoryInternalGraph();
  rejectsForeignMemoryInternalEdgeReference();
  rejectsNoncanonicalMemoryInternalEdgeReference();
  validatesCorrelatedMemoryAccessCapabilities();
  rejectsSharedMemoryOperationTemplate();
  validatesLogicalMemoryRootCapabilities();
  validatesAnchorMemoryServiceBindings();
  rejectsAnchorMemoryCoverageAndReferences();
}

} // namespace loom::mapping::test
