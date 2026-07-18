#include "MappingCoreTestSupport.h"

namespace loom::mapping::test {
namespace {

void acceptsValidTechMapping() {
  TestCase testCase = makeValidCase();
  auto result =
      validateTechMapping(testCase.mapping, testCase.dataflow, testCase.fabric);
  if (!result)
    fail(__func__, llvm::toString(result.takeError()).c_str());
  if (result->profile() != MappingProfile::TechMapping)
    fail(__func__, "validated draft has the wrong profile");
  if (result->realizations().size() != 1)
    fail(__func__, "validated draft lost its realization");
}
void acceptsBoundaryInputFanout() {
  TestCase testCase = makeValidCase();
  const PortDescriptor value = port(PortKind::Value, type(1));
  testCase.dataflow.edges[3].source =
      GraphPort{GraphId(1), PortDirection::Input, 0};
  testCase.dataflow.actors[1].inputPorts[1] = value;
  testCase.fabric.operations[1].inputPorts[1] = value;
  testCase.fabric.encodings[0].inputs.pop_back();
  testCase.fabric.encodings[0].operations[1].inputPorts[1] = value;
  testCase.fabric.encodings[0].operations[1].operands[1] = FuInputValue{0};
  testCase.mapping.realizations[0].boundaryPorts[2].fuPort.index = 0;
  FrozenRealizationGraph graph = validateAndFreeze(__func__, testCase);
  const FrozenLogicalNet *fanout = nullptr;
  for (const FrozenLogicalNet &net : graph.logicalNets()) {
    const auto *source = std::get_if<FrozenGraphBoundaryTerminal>(&net.source);
    if (source && source->graph == GraphId(1) &&
        source->direction == PortDirection::Input && source->port == 0) {
      if (fanout)
        fail(__func__, "graph boundary source produced multiple logical nets");
      fanout = &net;
    }
  }
  if (!fanout || fanout->sinkCount != 2)
    fail(__func__, "graph boundary fanout did not produce one two-sink net");
  const FrozenLogicalNetSink &first =
      graph.logicalNetSinks()[fanout->sinkOffset];
  const FrozenLogicalNetSink &second =
      graph.logicalNetSinks()[fanout->sinkOffset + 1];
  if (first.edge != EdgeId(100) || second.edge != EdgeId(103))
    fail(__func__, "graph boundary fanout lost canonical edge provenance");
  const auto *firstTerminal =
      std::get_if<FrozenTemplateTerminalRef>(&first.terminal);
  const auto *secondTerminal =
      std::get_if<FrozenTemplateTerminalRef>(&second.terminal);
  if (!firstTerminal || !secondTerminal ||
      firstTerminal->terminal != secondTerminal->terminal)
    fail(__func__, "shared FU input did not produce one sink terminal");
  const auto *sharedTerminal = std::get_if<FrozenComputeTemplateTerminal>(
      &graph.templateTerminals()[firstTerminal->terminal]);
  if (!sharedTerminal || sharedTerminal->realization != 0 ||
      sharedTerminal->fu != FuId(10) ||
      sharedTerminal->direction != PortDirection::Input ||
      sharedTerminal->port != 0)
    fail(__func__, "fanout sinks resolved to the wrong FU input terminal");
}
void acceptsBoundaryOutputFanoutWithOneCorrespondence() {
  TestCase testCase = makeValidCase();
  const GraphId graph(1);
  const ActorId addActor(3);
  const PortDescriptor value = port(PortKind::Value, type(1));
  testCase.dataflow.graphs[0].outputPorts.push_back(value);
  testCase.dataflow.edges.push_back(
      DataflowEdge{EdgeId(200), ActorPort{addActor, PortDirection::Output, 0},
                   GraphPort{graph, PortDirection::Output, 1}});
  auto result =
      validateTechMapping(testCase.mapping, testCase.dataflow, testCase.fabric);
  if (!result)
    fail(__func__, llvm::toString(result.takeError()).c_str());
}
void rejectsDistinctBoundaryOutputsSharingFuOutput() {
  TestCase testCase = makeValidCase();
  const ArtifactIdentity &dataflowId = testCase.dataflow.identity;
  const ArtifactIdentity &fabricId = testCase.fabric.identity;
  const GraphId graph(1);
  const ActorId multiplyActor(2);
  const FuId fu(10);
  const PortDescriptor value = port(PortKind::Value, type(1));
  testCase.dataflow.graphs[0].outputPorts.push_back(value);
  testCase.dataflow.edges.push_back(DataflowEdge{
      EdgeId(200), ActorPort{multiplyActor, PortDirection::Output, 0},
      GraphPort{graph, PortDirection::Output, 1}});
  testCase.mapping.realizations[0].boundaryPorts.push_back(
      {ActorPortRef{ActorRef{dataflowId, multiplyActor}, PortDirection::Output,
                    0},
       FuPortRef{FuRef{fabricId, fu}, PortDirection::Output, 0}});
  expectError(
      __func__,
      validateTechMapping(testCase.mapping, testCase.dataflow, testCase.fabric),
      MappingErrorCode::IncompleteBoundaryCorrespondence);
}
void rejectsBoundaryInputCoalescing() {
  TestCase testCase = makeValidCase();
  const PortDescriptor value = port(PortKind::Value, type(1));
  testCase.dataflow.graphs[0].inputPorts[2] = value;
  testCase.dataflow.actors[1].inputPorts[1] = value;
  testCase.fabric.operations[1].inputPorts[1] = value;
  testCase.fabric.encodings[0].inputs.pop_back();
  testCase.fabric.encodings[0].operations[1].inputPorts[1] = value;
  testCase.fabric.encodings[0].operations[1].operands[1] = FuInputValue{0};
  testCase.mapping.realizations[0].boundaryPorts[2].fuPort.index = 0;
  expectError(
      __func__,
      validateTechMapping(testCase.mapping, testCase.dataflow, testCase.fabric),
      MappingErrorCode::ConfiguredFunctionMismatch);
}
void acceptsCrossRealizationEdgeAccounting() {
  TestCase testCase = makeValidCase();
  const ArtifactIdentity &dataflowId = testCase.dataflow.identity;
  const ArtifactIdentity &fabricId = testCase.fabric.identity;
  const ActorId multiplyActor(2);
  const ActorId addActor(3);
  const FuId multiplyFu(10);
  const FuId addFu(14);
  const FabricOpId multiplyOp(11);
  const FabricOpId addOp(12);
  const EncodingId multiplyEncoding(13);
  const EncodingId addEncoding(15);
  const PortDescriptor value = port(PortKind::Value, type(1));
  const PortDescriptor stream = port(PortKind::Stream, type(1));
  const PortDescriptor auxiliary = port(PortKind::Value, type(2));
  const SemanticKey noAttributes = semantic(10);
  testCase.fabric.functionalUnits = {
      FuDescriptor{multiplyFu, {value, stream}, {value}},
      FuDescriptor{addFu, {value, auxiliary}, {value}}};
  testCase.fabric.operations = {
      FabricOpDescriptor{multiplyOp, multiplyFu, {value, stream}, {value}},
      FabricOpDescriptor{addOp, addFu, {value, auxiliary}, {value}}};
  testCase.fabric.encodings = {
      EncodingDescriptor{
          multiplyEncoding,
          multiplyFu,
          {{0, value}, {1, stream}},
          {ConfiguredFabricOpDescriptor{multiplyOp,
                                        semantic(1),
                                        noAttributes,
                                        {value, stream},
                                        {value},
                                        {FuInputValue{0}, FuInputValue{1}}}},
          {{0, value, FabricOpResultValue{multiplyOp, 0}}}},
      EncodingDescriptor{
          addEncoding,
          addFu,
          {{0, value}, {1, auxiliary}},
          {ConfiguredFabricOpDescriptor{addOp,
                                        semantic(2),
                                        noAttributes,
                                        {value, auxiliary},
                                        {value},
                                        {FuInputValue{0}, FuInputValue{1}}}},
          {{0, value, FabricOpResultValue{addOp, 0}}}}};
  testCase.fabric.computeOccurrences = {
      makeSpatialComputeOccurrence(fabricId, ComputeOccurrenceId(1000),
                                   testCase.fabric.functionalUnits[0], 2000),
      makeSpatialComputeOccurrence(fabricId, ComputeOccurrenceId(1001),
                                   testCase.fabric.functionalUnits[1], 2100)};
  ComputeRealizationDraft multiplyRealization{
      ComputeRealizationId(20),
      {ActorRef{dataflowId, multiplyActor}},
      FuRef{fabricId, multiplyFu},
      EncodingRef{fabricId, multiplyEncoding},
      {{ActorRef{dataflowId, multiplyActor},
        FabricOpRef{fabricId, multiplyOp}}},
      {{ActorPortRef{ActorRef{dataflowId, multiplyActor}, PortDirection::Input,
                     0},
        FuPortRef{FuRef{fabricId, multiplyFu}, PortDirection::Input, 0}},
       {ActorPortRef{ActorRef{dataflowId, multiplyActor}, PortDirection::Input,
                     1},
        FuPortRef{FuRef{fabricId, multiplyFu}, PortDirection::Input, 1}},
       {ActorPortRef{ActorRef{dataflowId, multiplyActor}, PortDirection::Output,
                     0},
        FuPortRef{FuRef{fabricId, multiplyFu}, PortDirection::Output, 0}}}};
  ComputeRealizationDraft addRealization{
      ComputeRealizationId(21),
      {ActorRef{dataflowId, addActor}},
      FuRef{fabricId, addFu},
      EncodingRef{fabricId, addEncoding},
      {{ActorRef{dataflowId, addActor}, FabricOpRef{fabricId, addOp}}},
      {{ActorPortRef{ActorRef{dataflowId, addActor}, PortDirection::Input, 0},
        FuPortRef{FuRef{fabricId, addFu}, PortDirection::Input, 0}},
       {ActorPortRef{ActorRef{dataflowId, addActor}, PortDirection::Input, 1},
        FuPortRef{FuRef{fabricId, addFu}, PortDirection::Input, 1}},
       {ActorPortRef{ActorRef{dataflowId, addActor}, PortDirection::Output, 0},
        FuPortRef{FuRef{fabricId, addFu}, PortDirection::Output, 0}}}};
  testCase.mapping.realizations = {std::move(multiplyRealization),
                                   std::move(addRealization)};
  auto result =
      validateTechMapping(testCase.mapping, testCase.dataflow, testCase.fabric);
  if (!result)
    fail(__func__, llvm::toString(result.takeError()).c_str());
}
void rejectsUnsupportedSchemaProfileAndIdentity() {
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.header.schemaVersion = SchemaVersion{1, 0};
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::UnsupportedSchemaVersion);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.header.profile = MappingProfile::PhysicalMapping;
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::WrongMappingProfile);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.header.dataflowIdentity = artifact(99);
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::ArtifactIdentityMismatch);
  }
}
void rejectsForeignUnresolvedAndWrongKindReferences() {
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].actors[0].artifact = artifact(99);
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::ForeignEntityReference);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].actors[0].entity = ActorId(999);
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::UnresolvedEntityId);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].actors[0].entity = ActorId(1);
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::WrongEntityKind);
  }
}
void rejectsDuplicateEntityIdsInArtifactNamespaces() {
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.actors.push_back(ActorDescriptor{ActorId(1),
                                                       GraphId(1),
                                                       semantic(1),
                                                       semantic(1),
                                                       {},
                                                       {},
                                                       std::nullopt});
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::DuplicateEntityId);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.fabric.operations.push_back(
        FabricOpDescriptor{FabricOpId(10), FuId(10), {}, {}});
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::DuplicateEntityId);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations.push_back(
        testCase.mapping.realizations.front());
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::DuplicateEntityId);
  }
}

void rejectsInvalidDataflowEdges() {
  {
    TestCase testCase = makeValidCase();
    DataflowEdge duplicate = testCase.dataflow.edges.front();
    duplicate.id = EdgeId(200);
    testCase.dataflow.edges.push_back(std::move(duplicate));
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::DuplicateEdge);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.edges[0].source =
        ActorPort{ActorId(2), PortDirection::Input, 0};
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.actors[0].inputPorts[1].kind = PortKind::Memory;
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::InvalidPortConnection);
  }
}

void rejectsInvalidCoveredSinkAccounting() {
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.edges.erase(testCase.dataflow.edges.begin());
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::MissingSinkDriver);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.edges.pop_back();
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::MissingSinkDriver);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.edges.push_back(DataflowEdge{
        EdgeId(200), ActorPort{ActorId(2), PortDirection::Output, 0},
        GraphPort{GraphId(1), PortDirection::Output, 0}});
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::MultipleSinkDrivers);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.edges.push_back(DataflowEdge{
        EdgeId(200), ActorPort{ActorId(3), PortDirection::Output, 0},
        ActorPort{ActorId(2), PortDirection::Input, 0}});
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::MultipleSinkDrivers);
  }
}

void acceptsUncoveredActorlessPassthrough() {
  TestCase testCase = makeValidCase();
  const PortDescriptor value = port(PortKind::Value, type(1));
  testCase.dataflow.graphs.push_back(
      GraphDescriptor{GraphId(4), {value}, {value}});
  testCase.dataflow.edges.push_back(
      DataflowEdge{EdgeId(200), GraphPort{GraphId(4), PortDirection::Input, 0},
                   GraphPort{GraphId(4), PortDirection::Output, 0}});

  auto result =
      validateTechMapping(testCase.mapping, testCase.dataflow, testCase.fabric);
  if (!result)
    fail(__func__, llvm::toString(result.takeError()).c_str());
}

void rejectsCoveredActorlessPassthroughAndUnaccountedEdges() {
  {
    TestCase testCase = makeValidCase();
    const PortDescriptor value = port(PortKind::Value, type(1));
    testCase.dataflow.graphs.push_back(
        GraphDescriptor{GraphId(4), {value}, {value}});
    testCase.dataflow.edges.push_back(DataflowEdge{
        EdgeId(200), GraphPort{GraphId(4), PortDirection::Input, 0},
        GraphPort{GraphId(4), PortDirection::Output, 0}});
    testCase.mapping.coveredGraphs.push_back(
        GraphRef{testCase.dataflow.identity, GraphId(4)});
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::ActorlessGraphPassthrough);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].boundaryPorts.pop_back();
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::UnaccountedGraphEdge);
  }
}

void rejectsInvalidActorGroupsAndCoverage() {
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].actors.clear();
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::EmptyActorGroup);
  }
  {
    TestCase testCase = makeValidCase();
    const GraphId otherGraph(4);
    const ActorId otherActor(5);
    testCase.dataflow.graphs.push_back(GraphDescriptor{otherGraph, {}, {}});
    testCase.dataflow.actors.push_back(ActorDescriptor{otherActor,
                                                       otherGraph,
                                                       semantic(1),
                                                       semantic(1),
                                                       {},
                                                       {},
                                                       std::nullopt});
    testCase.mapping.coveredGraphs.push_back(
        GraphRef{testCase.dataflow.identity, otherGraph});
    testCase.mapping.realizations[0].actors.push_back(
        ActorRef{testCase.dataflow.identity, otherActor});
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::CrossGraphActorGroup);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations.clear();
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::IncompleteGraphCoverage);
  }
}

void rejectsInvalidComputeRealization() {
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].actorToOps.pop_back();
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::IncompleteActorToOpCorrespondence);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].actorToOps[1].fabricOp =
        testCase.mapping.realizations[0].actorToOps[0].fabricOp;
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::IncompleteActorToOpCorrespondence);
  }
  {
    TestCase testCase = makeValidCase();
    const FuId otherFu(14);
    const FabricOpId otherOp(16);
    testCase.fabric.functionalUnits.push_back(FuDescriptor{
        otherFu,
        {port(PortKind::Value, type(1)), port(PortKind::Value, type(2))},
        {port(PortKind::Value, type(1))}});
    testCase.fabric.operations.push_back(FabricOpDescriptor{
        otherOp,
        otherFu,
        {port(PortKind::Value, type(1)), port(PortKind::Value, type(2))},
        {port(PortKind::Value, type(1))}});
    testCase.mapping.realizations[0].actorToOps[1].fabricOp.entity = otherOp;
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::SelectedFuMismatch);
  }
  {
    TestCase testCase = makeValidCase();
    const FuId otherFu(14);
    const EncodingId otherEncoding(15);
    testCase.fabric.functionalUnits.push_back(FuDescriptor{otherFu, {}, {}});
    testCase.fabric.encodings.push_back(
        EncodingDescriptor{otherEncoding, otherFu, {}, {}, {}});
    testCase.mapping.realizations[0].encoding.entity = otherEncoding;
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::SelectedFuMismatch);
  }
}

void rejectsConfiguredFunctionMismatch() {
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.actors[1].operation = semantic(99);
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::ConfiguredFunctionMismatch);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.actors[1].attributes = semantic(99);
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::ConfiguredFunctionMismatch);
  }
  {
    TestCase testCase = makeValidCase();
    auto &add = testCase.fabric.encodings[0].operations[1];
    add.operands[0] = FuInputValue{0};
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::ConfiguredFunctionMismatch);
  }
}

void rejectsInvalidBoundaryCorrespondence() {
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].boundaryPorts.push_back(
        testCase.mapping.realizations[0].boundaryPorts.front());
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::IncompleteBoundaryCorrespondence);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].boundaryPorts[1].fuPort.index = 2;
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::ConfiguredFunctionMismatch);
  }
  {
    TestCase testCase = makeValidCase();
    const FuId otherFu(14);
    testCase.fabric.functionalUnits.push_back(
        FuDescriptor{otherFu, {port(PortKind::Value, type(1))}, {}});
    testCase.mapping.realizations[0].boundaryPorts[0].fuPort.fu.entity =
        otherFu;
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::SelectedFuMismatch);
  }
}

void rejectsInvalidComputeOccurrences() {
  auto expectInvalid = [&](auto mutate, MappingErrorCode code) {
    TestCase testCase = makeValidCase();
    mutate(testCase);
    expectMapError(__func__, testCase, code);
  };
  expectInvalid(
      [](TestCase &testCase) {
        testCase.fabric.computeOccurrences.front().schedule =
            static_cast<ComputeScheduleKind>(99);
      },
      MappingErrorCode::InvalidComputeOccurrence);
  expectInvalid(
      [](TestCase &testCase) {
        testCase.fabric.computeOccurrences.front().functionalUnits.clear();
      },
      MappingErrorCode::InvalidComputeOccurrence);
  expectInvalid(
      [](TestCase &testCase) {
        testCase.fabric.computeOccurrences.front()
            .functionalUnits.front()
            .artifact = artifact(99);
      },
      MappingErrorCode::ForeignEntityReference);
  expectInvalid(
      [](TestCase &testCase) {
        ComputeOccurrenceDescriptor &occurrence =
            testCase.fabric.computeOccurrences.front();
        occurrence.functionalUnits.push_back(
            occurrence.functionalUnits.front());
      },
      MappingErrorCode::InvalidComputeOccurrence);
  expectInvalid(
      [](TestCase &testCase) {
        testCase.fabric.computeOccurrences.front()
            .endpoints.front()
            .compatibleTypes.clear();
      },
      MappingErrorCode::InvalidComputeOccurrence);
  expectInvalid(
      [](TestCase &testCase) {
        ComputeEndpointDescriptor &endpoint =
            testCase.fabric.computeOccurrences.front().endpoints.front();
        endpoint.compatibleTypes.push_back(endpoint.compatibleTypes.front());
      },
      MappingErrorCode::InvalidComputeOccurrence);
  expectInvalid(
      [](TestCase &testCase) {
        testCase.fabric.computeOccurrences.front()
            .localArcs.front()
            .endpoint.entity = ComputeEndpointId(9999);
      },
      MappingErrorCode::UnresolvedEntityId);
  expectInvalid(
      [](TestCase &testCase) {
        ComputeOccurrenceDescriptor &occurrence =
            testCase.fabric.computeOccurrences.front();
        occurrence.localArcs.push_back(occurrence.localArcs.front());
      },
      MappingErrorCode::InvalidComputeOccurrence);
  expectInvalid(
      [](TestCase &testCase) {
        ComputeOccurrenceDescriptor other = makeSpatialComputeOccurrence(
            testCase.fabric.identity, ComputeOccurrenceId(1001),
            testCase.fabric.functionalUnits.front(), 3000);
        testCase.fabric.computeOccurrences.front().localArcs.front().endpoint =
            ComputeEndpointRef{testCase.fabric.identity,
                               other.endpoints.front().id};
        testCase.fabric.computeOccurrences.push_back(std::move(other));
      },
      MappingErrorCode::InvalidComputeOccurrence);
}
} // namespace

void runMappingVerifierTests() {
  acceptsValidTechMapping();
  acceptsBoundaryInputFanout();
  acceptsBoundaryOutputFanoutWithOneCorrespondence();
  rejectsDistinctBoundaryOutputsSharingFuOutput();
  rejectsBoundaryInputCoalescing();
  acceptsCrossRealizationEdgeAccounting();
  rejectsUnsupportedSchemaProfileAndIdentity();
  rejectsForeignUnresolvedAndWrongKindReferences();
  rejectsDuplicateEntityIdsInArtifactNamespaces();
  rejectsInvalidDataflowEdges();
  rejectsInvalidCoveredSinkAccounting();
  acceptsUncoveredActorlessPassthrough();
  rejectsCoveredActorlessPassthroughAndUnaccountedEdges();
  rejectsInvalidActorGroupsAndCoverage();
  rejectsInvalidComputeRealization();
  rejectsConfiguredFunctionMismatch();
  rejectsInvalidBoundaryCorrespondence();
  rejectsInvalidComputeOccurrences();
}

} // namespace loom::mapping::test
