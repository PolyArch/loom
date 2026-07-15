#include "Mapping/Artifact.h"
#include "Mapping/StructureValidator.h"

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

  DataflowProgramView dataflow{
      dataflowId,
      {GraphDescriptor{graph, {value, stream, memory}, {value}}},
      {ActorDescriptor{multiplyActor, graph, {value, stream}, {value}},
       ActorDescriptor{addActor, graph, {value, memory}, {value}}},
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
      {EncodingDescriptor{encoding, fu}}};

  StructuralRealizationDraft realization{
      StructuralRealizationId(20),
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

  TechMappingDraft mapping{MappingDraftHeader{SchemaVersion{1, 0},
                                              MappingProfile::TechMapping,
                                              dataflowId, fabricId},
                           {GraphRef{dataflowId, graph}},
                           {std::move(realization)}};

  return TestCase{std::move(dataflow), std::move(fabric), std::move(mapping)};
}

void acceptsStructurallyValidDraft() {
  TestCase testCase = makeValidCase();
  auto result = validateTechMappingStructure(
      testCase.mapping, testCase.dataflow, testCase.fabric);
  if (!result)
    fail(__func__, llvm::toString(result.takeError()).c_str());
  if (result->profile() != MappingProfile::TechMapping)
    fail(__func__, "validated draft has the wrong profile");
  if (result->realizations().size() != 1)
    fail(__func__, "validated draft lost its realization");
}

void acceptsCrossRealizationEdgeAccounting() {
  TestCase testCase = makeValidCase();
  StructuralRealizationDraft second = testCase.mapping.realizations.front();
  StructuralRealizationDraft &first = testCase.mapping.realizations.front();

  first.actors.pop_back();
  first.actorToOps.pop_back();
  first.boundaryPorts.erase(first.boundaryPorts.begin() + 2,
                            first.boundaryPorts.end());
  first.boundaryPorts.push_back(
      {ActorPortRef{first.actors.front(), PortDirection::Output, 0},
       FuPortRef{first.fu, PortDirection::Output, 0}});

  second.id = StructuralRealizationId(21);
  second.actors.erase(second.actors.begin());
  second.actorToOps.erase(second.actorToOps.begin());
  second.boundaryPorts.erase(second.boundaryPorts.begin(),
                             second.boundaryPorts.begin() + 2);
  second.boundaryPorts.push_back(
      {ActorPortRef{second.actors.front(), PortDirection::Input, 0},
       FuPortRef{second.fu, PortDirection::Input, 0}});
  testCase.mapping.realizations.push_back(std::move(second));

  auto result = validateTechMappingStructure(
      testCase.mapping, testCase.dataflow, testCase.fabric);
  if (!result)
    fail(__func__, llvm::toString(result.takeError()).c_str());
}

void rejectsUnsupportedSchemaProfileAndIdentity() {
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.header.schemaVersion = SchemaVersion{1, 1};
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::UnsupportedSchemaVersion);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.header.profile = MappingProfile::PhysicalMapping;
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::WrongMappingProfile);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.header.dataflowIdentity = ArtifactIdentity();
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::InvalidArtifactIdentity);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.header.dataflowIdentity = artifact(99);
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::ArtifactIdentityMismatch);
  }
}

void rejectsForeignUnresolvedAndWrongKindReferences() {
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].actors[0].artifact = artifact(99);
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::ForeignEntityReference);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].actors[0].entity = ActorId(999);
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::UnresolvedEntityId);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].actors[0].entity = ActorId(1);
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::WrongEntityKind);
  }
}

void rejectsDuplicateEntityIdsInArtifactNamespaces() {
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.actors.push_back(
        ActorDescriptor{ActorId(1), GraphId(1), {}, {}});
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::DuplicateEntityId);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.fabric.operations.push_back(
        FabricOpDescriptor{FabricOpId(10), FuId(10), {}, {}});
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::DuplicateEntityId);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations.push_back(
        testCase.mapping.realizations.front());
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::DuplicateEntityId);
  }
}

void rejectsInvalidDataflowEdges() {
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.edges.push_back(testCase.dataflow.edges.front());
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::DuplicateEdge);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.edges[0].source =
        ActorPort{ActorId(2), PortDirection::Input, 0};
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.actors[0].inputPorts[1].kind = PortKind::Memory;
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::PortSignatureMismatch);
  }
}

void rejectsInvalidCoveredSinkAccounting() {
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.edges.erase(testCase.dataflow.edges.begin());
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::MissingSinkDriver);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.edges.pop_back();
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::MissingSinkDriver);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.edges.push_back(
        DataflowEdge{ActorPort{ActorId(2), PortDirection::Output, 0},
                     GraphPort{GraphId(1), PortDirection::Output, 0}});
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::MultipleSinkDrivers);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.edges.push_back(
        DataflowEdge{ActorPort{ActorId(3), PortDirection::Output, 0},
                     ActorPort{ActorId(2), PortDirection::Input, 0}});
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
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

  auto result = validateTechMappingStructure(
      testCase.mapping, testCase.dataflow, testCase.fabric);
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
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::ActorlessGraphPassthrough);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].boundaryPorts.pop_back();
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::UnaccountedGraphEdge);
  }
}

void rejectsInvalidActorGroupsAndCoverage() {
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].actors.clear();
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::EmptyActorGroup);
  }
  {
    TestCase testCase = makeValidCase();
    const GraphId otherGraph(4);
    const ActorId otherActor(5);
    testCase.dataflow.graphs.push_back(GraphDescriptor{otherGraph, {}, {}});
    testCase.dataflow.actors.push_back(
        ActorDescriptor{otherActor, otherGraph, {}, {}});
    testCase.mapping.coveredGraphs.push_back(
        GraphRef{testCase.dataflow.identity, otherGraph});
    testCase.mapping.realizations[0].actors.push_back(
        ActorRef{testCase.dataflow.identity, otherActor});
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::CrossGraphActorGroup);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations.clear();
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::IncompleteGraphCoverage);
  }
}

void rejectsInvalidStructuralCorrespondence() {
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].actorToOps.pop_back();
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::IncompleteActorToOpCorrespondence);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].actorToOps[1].fabricOp =
        testCase.mapping.realizations[0].actorToOps[0].fabricOp;
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::IncompleteActorToOpCorrespondence);
  }
  {
    TestCase testCase = makeValidCase();
    const FuId otherFu(14);
    testCase.fabric.functionalUnits.push_back(FuDescriptor{
        otherFu,
        {port(PortKind::Value, type(1)), port(PortKind::Memory, type(1))},
        {port(PortKind::Value, type(1))}});
    testCase.fabric.operations[1].fu = otherFu;
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::SelectedFuMismatch);
  }
  {
    TestCase testCase = makeValidCase();
    const FuId otherFu(14);
    testCase.fabric.functionalUnits.push_back(FuDescriptor{otherFu, {}, {}});
    testCase.fabric.encodings[0].fu = otherFu;
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::SelectedFuMismatch);
  }
}

void rejectsInvalidBoundaryCorrespondence() {
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations[0].boundaryPorts.push_back(
        testCase.mapping.realizations[0].boundaryPorts.front());
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::IncompleteBoundaryCorrespondence);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.fabric.functionalUnits[0].inputPorts.push_back(
        port(PortKind::Value, type(1)));
    testCase.mapping.realizations[0].boundaryPorts[1].fuPort.index = 3;
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::PortSignatureMismatch);
  }
  {
    TestCase testCase = makeValidCase();
    const FuId otherFu(14);
    testCase.fabric.functionalUnits.push_back(
        FuDescriptor{otherFu, {port(PortKind::Value, type(1))}, {}});
    testCase.mapping.realizations[0].boundaryPorts[0].fuPort.fu.entity =
        otherFu;
    expectError(__func__,
                validateTechMappingStructure(
                    testCase.mapping, testCase.dataflow, testCase.fabric),
                MappingErrorCode::SelectedFuMismatch);
  }
}

} // namespace

int main() {
  acceptsStructurallyValidDraft();
  acceptsCrossRealizationEdgeAccounting();
  rejectsUnsupportedSchemaProfileAndIdentity();
  rejectsForeignUnresolvedAndWrongKindReferences();
  rejectsDuplicateEntityIdsInArtifactNamespaces();
  rejectsInvalidDataflowEdges();
  rejectsInvalidCoveredSinkAccounting();
  acceptsUncoveredActorlessPassthrough();
  rejectsCoveredActorlessPassthroughAndUnaccountedEdges();
  rejectsInvalidActorGroupsAndCoverage();
  rejectsInvalidStructuralCorrespondence();
  rejectsInvalidBoundaryCorrespondence();
  return 0;
}
