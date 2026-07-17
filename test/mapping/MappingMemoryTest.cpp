#include "MappingCoreTestSupport.h"

#include <algorithm>

namespace loom::mapping::test {
namespace {

bool containsExternalEdge(const FrozenRealizationGraph &graph, EdgeId edge) {
  return std::any_of(
      graph.logicalNetSinks().begin(), graph.logicalNetSinks().end(),
      [&](const FrozenLogicalNetSink &sink) { return sink.edge == edge; });
}
bool isGraphMemoryCapabilityPort(const FrozenTerminal &terminal) {
  const auto *graph = std::get_if<FrozenGraphBoundaryTerminal>(&terminal);
  return graph && graph->port == 0;
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
  for (std::uint64_t edge : {100, 104, 113, 114, 115}) {
    if (containsExternalEdge(graph, EdgeId(edge)))
      fail(__func__, "internal edge escaped into the external net cache");
  }
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
  if (first.edge != EdgeId(106) || second.edge != EdgeId(107))
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
  expectAnyError(__func__, freezeRealizationGraph(testCase.dataflow,
                                                  testCase.fabric, mapping));
}
void acceptsExternalAndInternalMemoryAnchor() {
  {
    TestCase testCase = makeMemoryAnchorCase();
    auto result = validateTechMapping(testCase.mapping, testCase.dataflow,
                                      testCase.fabric);
    if (!result)
      fail(__func__, llvm::toString(result.takeError()).c_str());
    if (result->realizations().size() != 4 ||
        result->memoryRealizations().size() != 2)
      fail(__func__, "external memory anchor lost a realization");
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    selectInternalMemoryGraph(testCase);
    auto result = validateTechMapping(testCase.mapping, testCase.dataflow,
                                      testCase.fabric);
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
    selectInternalMemoryGraph(testCase);
    testCase.mapping.memoryRealizations[0].internalEdges[1].connection =
        testCase.mapping.memoryRealizations[0].internalEdges[0].connection;
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
    selectInternalMemoryGraph(testCase);
    testCase.fabric.memoryImplementations[0]
        .boundaryPorts[0]
        .maxInternalFanout = 1;
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeMemoryAnchorCase();
    selectInternalMemoryGraph(testCase);
    testCase.mapping.memoryRealizations[0]
        .graphBoundaryPorts[0]
        .implementationPort.index = 1;
    expectMapError(__func__, testCase,
                   MappingErrorCode::IncompleteMemoryBoundaryCorrespondence);
  }
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
      DataflowEdge{EdgeId(116), GraphPort{GraphId(1), PortDirection::Input, 1},
                   ActorPort{ActorId(9), PortDirection::Input, 0}});
  testCase.dataflow.edges.push_back(
      DataflowEdge{EdgeId(117), GraphPort{GraphId(1), PortDirection::Input, 2},
                   ActorPort{ActorId(9), PortDirection::Input, 1}});
  MemoryRealizationDraft &load = testCase.mapping.memoryRealizations[0];
  const ActorRef actor{testCase.dataflow.identity, ActorId(9)};
  const MemoryOperationPortTemplateRef operation{
      testCase.fabric.identity, MemoryOperationPortTemplateId(34)};
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
    auto result = validateTechMapping(testCase.mapping, testCase.dataflow,
                                      testCase.fabric);
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
    testCase.dataflow.edges.push_back(DataflowEdge{
        EdgeId(116), GraphPort{GraphId(1), PortDirection::Input, 0},
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
    testCase.mapping.realizations[0].actors[0].entity = ActorId(2);
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
  rejectsInconsistentFrozenMemoryService();
  acceptsExternalAndInternalMemoryAnchor();
  rejectsInexactMemoryInternalGraph();
  validatesCorrelatedMemoryAccessCapabilities();
  rejectsSharedMemoryOperationTemplate();
  validatesLogicalMemoryRootCapabilities();
  validatesAnchorMemoryServiceBindings();
  rejectsAnchorMemoryCoverageAndReferences();
}

} // namespace loom::mapping::test
