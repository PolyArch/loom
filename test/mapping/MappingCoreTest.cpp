#include "Mapping/Artifact.h"
#include "Mapping/Verifier.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <utility>

using namespace loom::mapping;

namespace {

static_assert(!std::is_default_constructible_v<MappingDraftHeader>);
static_assert(!std::is_default_constructible_v<TechMappingDraft>);

struct TestCase {
  DataflowProgramView dataflow;
  FabricHardwareView fabric;
  TechMappingDraft mapping;
};

ArtifactIdentity artifact(std::uint8_t value) {
  return ArtifactIdentity({value});
}

TypeKey type(std::uint64_t value) { return TypeKey(value); }

SemanticKey semantic(std::uint8_t value) { return SemanticKey({value}); }

PortDescriptor port(PortKind kind, TypeKey typeKey) {
  return PortDescriptor{kind, typeKey};
}

void fail(const char *test, const char *message) {
  std::cerr << test << ": " << message << '\n';
  std::exit(1);
}

MappingErrorCode takeCode(llvm::Error error) {
  MappingErrorCode code = MappingErrorCode::InternalError;
  llvm::handleAllErrors(
      std::move(error),
      [&](const MappingError &mappingError) { code = mappingError.code(); });
  return code;
}

template <typename T>
void expectError(const char *test, llvm::Expected<T> result,
                 MappingErrorCode expected) {
  if (result)
    fail(test, "expected validation failure");
  if (takeCode(result.takeError()) != expected)
    fail(test, "received a different validation failure");
}

TestCase makeValidCase() {
  const ArtifactIdentity dataflowId = artifact(1);
  const ArtifactIdentity fabricId = artifact(2);
  const TypeKey word = type(1);
  const PortDescriptor value = port(PortKind::Value, word);
  const PortDescriptor stream = port(PortKind::Stream, word);
  const PortDescriptor memory = port(PortKind::Memory, word);

  const GraphId graph(1);
  const ActorId multiplyActor(2);
  const ActorId addActor(3);
  const SemanticKey multiply = semantic(1);
  const SemanticKey add = semantic(2);
  const SemanticKey noAttributes = semantic(10);

  DataflowProgramView dataflow{
      dataflowId,
      {GraphDescriptor{graph, {value, stream, memory}, {value}}},
      {ActorDescriptor{multiplyActor,
                       graph,
                       multiply,
                       noAttributes,
                       {value, stream},
                       {value}},
       ActorDescriptor{
           addActor, graph, add, noAttributes, {value, memory}, {value}}},
      {DataflowEdge{GraphPort{graph, PortDirection::Input, 0},
                    ActorPort{multiplyActor, PortDirection::Input, 0}},
       DataflowEdge{GraphPort{graph, PortDirection::Input, 1},
                    ActorPort{multiplyActor, PortDirection::Input, 1}},
       DataflowEdge{ActorPort{multiplyActor, PortDirection::Output, 0},
                    ActorPort{addActor, PortDirection::Input, 0}},
       DataflowEdge{GraphPort{graph, PortDirection::Input, 2},
                    ActorPort{addActor, PortDirection::Input, 1}},
       DataflowEdge{ActorPort{addActor, PortDirection::Output, 0},
                    GraphPort{graph, PortDirection::Output, 0}}}};

  const FuId fu(10);
  const FabricOpId multiplyOp(11);
  const FabricOpId addOp(12);
  const EncodingId encoding(13);

  FabricHardwareView fabric{
      fabricId,
      {FuDescriptor{fu, {value, stream, memory}, {value}}},
      {FabricOpDescriptor{multiplyOp, fu, {value, stream}, {value}},
       FabricOpDescriptor{addOp, fu, {value, memory}, {value}}},
      {EncodingDescriptor{
          encoding,
          fu,
          {{0, value}, {1, stream}, {2, memory}},
          {ConfiguredFabricOpDescriptor{multiplyOp,
                                        multiply,
                                        noAttributes,
                                        {value, stream},
                                        {value},
                                        {FuInputValue{0}, FuInputValue{1}}},
           ConfiguredFabricOpDescriptor{
               addOp,
               add,
               noAttributes,
               {value, memory},
               {value},
               {FabricOpResultValue{multiplyOp, 0}, FuInputValue{2}}}},
          {{0, value, FabricOpResultValue{addOp, 0}}}}}};

  ComputeRealizationDraft realization{
      ComputeRealizationId(20),
      {ActorRef{dataflowId, multiplyActor}, ActorRef{dataflowId, addActor}},
      FuRef{fabricId, fu},
      EncodingRef{fabricId, encoding},
      {{ActorRef{dataflowId, multiplyActor}, FabricOpRef{fabricId, multiplyOp}},
       {ActorRef{dataflowId, addActor}, FabricOpRef{fabricId, addOp}}},
      {{ActorPortRef{ActorRef{dataflowId, multiplyActor}, PortDirection::Input,
                     0},
        FuPortRef{FuRef{fabricId, fu}, PortDirection::Input, 0}},
       {ActorPortRef{ActorRef{dataflowId, multiplyActor}, PortDirection::Input,
                     1},
        FuPortRef{FuRef{fabricId, fu}, PortDirection::Input, 1}},
       {ActorPortRef{ActorRef{dataflowId, addActor}, PortDirection::Input, 1},
        FuPortRef{FuRef{fabricId, fu}, PortDirection::Input, 2}},
       {ActorPortRef{ActorRef{dataflowId, addActor}, PortDirection::Output, 0},
        FuPortRef{FuRef{fabricId, fu}, PortDirection::Output, 0}}}};

  TechMappingDraft mapping{MappingDraftHeader{SchemaVersion{2, 0},
                                              MappingProfile::TechMapping,
                                              dataflowId, fabricId},
                           {GraphRef{dataflowId, graph}},
                           {std::move(realization)}};

  return TestCase{std::move(dataflow), std::move(fabric), std::move(mapping)};
}

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

  auto result =
      validateTechMapping(testCase.mapping, testCase.dataflow, testCase.fabric);
  if (!result)
    fail(__func__, llvm::toString(result.takeError()).c_str());
}

void acceptsBoundaryOutputFanoutWithOneCorrespondence() {
  TestCase testCase = makeValidCase();
  const GraphId graph(1);
  const ActorId addActor(3);
  const PortDescriptor value = port(PortKind::Value, type(1));

  testCase.dataflow.graphs[0].outputPorts.push_back(value);
  testCase.dataflow.edges.push_back(
      DataflowEdge{ActorPort{addActor, PortDirection::Output, 0},
                   GraphPort{graph, PortDirection::Output, 1}});

  auto result =
      validateTechMapping(testCase.mapping, testCase.dataflow, testCase.fabric);
  if (!result)
    fail(__func__, llvm::toString(result.takeError()).c_str());
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
  const PortDescriptor memory = port(PortKind::Memory, type(1));
  const SemanticKey noAttributes = semantic(10);

  testCase.fabric.functionalUnits = {
      FuDescriptor{multiplyFu, {value, stream}, {value}},
      FuDescriptor{addFu, {value, memory}, {value}}};
  testCase.fabric.operations = {
      FabricOpDescriptor{multiplyOp, multiplyFu, {value, stream}, {value}},
      FabricOpDescriptor{addOp, addFu, {value, memory}, {value}}};
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
          {{0, value}, {1, memory}},
          {ConfiguredFabricOpDescriptor{addOp,
                                        semantic(2),
                                        noAttributes,
                                        {value, memory},
                                        {value},
                                        {FuInputValue{0}, FuInputValue{1}}}},
          {{0, value, FabricOpResultValue{addOp, 0}}}}};

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
    testCase.mapping.header.dataflowIdentity = ArtifactIdentity();
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::InvalidArtifactIdentity);
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
    testCase.dataflow.actors.push_back(ActorDescriptor{
        ActorId(1), GraphId(1), semantic(1), semantic(1), {}, {}});
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
    testCase.dataflow.edges.push_back(testCase.dataflow.edges.front());
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
                MappingErrorCode::PortSignatureMismatch);
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
    testCase.dataflow.edges.push_back(
        DataflowEdge{ActorPort{ActorId(2), PortDirection::Output, 0},
                     GraphPort{GraphId(1), PortDirection::Output, 0}});
    expectError(__func__,
                validateTechMapping(testCase.mapping, testCase.dataflow,
                                    testCase.fabric),
                MappingErrorCode::MultipleSinkDrivers);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.edges.push_back(
        DataflowEdge{ActorPort{ActorId(3), PortDirection::Output, 0},
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
      DataflowEdge{GraphPort{GraphId(4), PortDirection::Input, 0},
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
    testCase.dataflow.edges.push_back(
        DataflowEdge{GraphPort{GraphId(4), PortDirection::Input, 0},
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
    testCase.dataflow.actors.push_back(ActorDescriptor{
        otherActor, otherGraph, semantic(1), semantic(1), {}, {}});
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
        {port(PortKind::Value, type(1)), port(PortKind::Memory, type(1))},
        {port(PortKind::Value, type(1))}});
    testCase.fabric.operations.push_back(FabricOpDescriptor{
        otherOp,
        otherFu,
        {port(PortKind::Value, type(1)), port(PortKind::Memory, type(1))},
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

} // namespace

int main() {
  acceptsValidTechMapping();
  acceptsBoundaryInputFanout();
  acceptsBoundaryOutputFanoutWithOneCorrespondence();
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
  return 0;
}
