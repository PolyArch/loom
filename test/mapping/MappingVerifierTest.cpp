#include "MappingCoreTestSupport.h"

#include "Fabric/Identity/FabricFuCapabilityTemplate.h"

namespace loom::mapping::test {
namespace {

void acceptsValidTechMapping() {
  TestCase testCase = makeValidCase();
  auto result =
      validateTechMapping(testCase.techMappingIdentity, testCase.mapping,
                          testCase.dataflow, testCase.fabric);
  if (!result)
    fail(__func__, llvm::toString(result.takeError()).c_str());
  if (result->realizations().size() != 1)
    fail(__func__, "validated draft lost its realization");
}

void rejectsMismatchedArtifactIdentity() {
  TestCase testCase = makeValidCase();
  testCase.mapping.header.dataflowIdentity = artifact(99);
  expectMapError(__func__, testCase,
                 MappingErrorCode::ArtifactIdentityMismatch);
}

void rejectsInvalidActorReferences() {
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations.front().actorToOps.front().actor.artifact =
        artifact(99);
    expectMapError(__func__, testCase, MappingErrorCode::ForeignReference);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations.front().actorToOps.front().actor.entity =
        ActorId(999);
    expectMapError(__func__, testCase, MappingErrorCode::UnresolvedEntityId);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations.front().actorToOps.front().actor.entity =
        ActorId(1);
    expectMapError(__func__, testCase, MappingErrorCode::WrongEntityKind);
  }
}

void rejectsDuplicatePersistentIdentity() {
  {
    TestCase testCase = makeValidCase();
    testCase.fabric.operations.push_back(testCase.fabric.operations.front());
    expectMapError(__func__, testCase,
                   MappingErrorCode::CapabilityTemplateMismatch);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations.push_back(
        testCase.mapping.realizations.front());
    expectMapError(__func__, testCase, MappingErrorCode::DuplicateEntityId);
  }
}

void rejectsInvalidDataflowConnectivity() {
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.edges.push_back(testCase.dataflow.edges.front());
    expectMapError(__func__, testCase, MappingErrorCode::DuplicateEdge);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.edges.front().source =
        ActorPort{ActorId(2), PortDirection::Input, 0};
    expectMapError(__func__, testCase, MappingErrorCode::InvalidPortConnection);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.dataflow.edges.erase(testCase.dataflow.edges.begin());
    expectMapError(__func__, testCase, MappingErrorCode::MissingSinkDriver);
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
  validateCase(__func__, testCase);
}

void rejectsIncompleteCoveredGraph() {
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations.clear();
    expectMapError(__func__, testCase,
                   MappingErrorCode::IncompleteGraphCoverage);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations.front().boundaryPorts.pop_back();
    expectMapError(__func__, testCase, MappingErrorCode::UnaccountedGraphEdge);
  }
}

void rejectsIncompleteActorOperationCorrespondence() {
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations.front().actorToOps.pop_back();
    expectMapError(__func__, testCase,
                   MappingErrorCode::IncompleteGraphCoverage);
  }
  {
    TestCase testCase = makeValidCase();
    auto &bindings = testCase.mapping.realizations.front().actorToOps;
    bindings.back().fabricOp = bindings.front().fabricOp;
    expectMapError(__func__, testCase,
                   MappingErrorCode::IncompleteActorToOpCorrespondence);
  }
}

void rejectsTypedFamilyMismatch() {
  TestCase testCase = makeValidCase();
  testCase.dataflow.actors.back().semantics.schema =
      ::dataflow::OperationSchemaId::ArithMulI;
  expectMapError(__func__, testCase,
                 MappingErrorCode::CapabilityTemplateMismatch);
}

void rejectsCapabilityTopologyMismatch() {
  TestCase testCase = makeValidCase();
  auto record =
      testCase.fabric.functionalUnits.front().capabilityTemplates.front();
  bool changed = false;
  for (auto &edge : record.activeEdges) {
    const auto *source = std::get_if<::loom::fabric::FabricFuTemplatePortRef>(
        &edge.source.payload);
    auto *destination = std::get_if<::loom::fabric::FabricFuNodePortRef>(
        &edge.destination.payload);
    if (!source || !destination ||
        source->direction != ::loom::fabric::FabricPortDirection::Input ||
        source->ordinal != 2)
      continue;
    destination->ordinal = 0;
    changed = true;
    break;
  }
  if (!changed)
    fail(__func__, "fixture lost the typed add-input boundary edge");
  testCase.fabric.functionalUnits.front().capabilityTemplates = llvm::cantFail(
      ::loom::fabric::normalizeFabricFuCapabilityTemplateInventory({record}));
  expectMapError(__func__, testCase,
                 MappingErrorCode::CapabilityTemplateMismatch);
}

void rejectsInvalidBoundaryCorrespondence() {
  {
    TestCase testCase = makeValidCase();
    auto &ports = testCase.mapping.realizations.front().boundaryPorts;
    ports.push_back(ports.front());
    expectMapError(__func__, testCase,
                   MappingErrorCode::IncompleteBoundaryCorrespondence);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations.front().boundaryPorts.front().fuPort.ordinal =
        1;
    expectMapError(__func__, testCase,
                   MappingErrorCode::CapabilityTemplateMismatch);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations.front().boundaryPorts.front().fuPort.fu =
        ::loom::fabric::FabricFuTemplateRef(99);
    expectMapError(__func__, testCase,
                   MappingErrorCode::MissingFuImplementation);
  }
}

void rejectsInvalidCapabilityInventory() {
  TestCase testCase = makeValidCase();
  auto &templates = testCase.fabric.functionalUnits.front().capabilityTemplates;
  templates.push_back(templates.front());
  expectMapError(__func__, testCase,
                 MappingErrorCode::InvalidCapabilityTemplateReference);
}

void rejectsInvalidComputeOccurrence() {
  {
    TestCase testCase = makeValidCase();
    testCase.fabric.computeOccurrences.front().functionalUnits.clear();
    expectMapError(__func__, testCase,
                   MappingErrorCode::InvalidComputeOccurrence);
  }
  {
    TestCase testCase = makeValidCase();
    testCase.fabric.computeOccurrences.front().functionalUnits.front() =
        ::loom::fabric::FabricFuTemplateRef(9999);
    expectMapError(__func__, testCase,
                   MappingErrorCode::MissingFuImplementation);
  }
  {
    TestCase testCase = makeValidCase();
    ComputeOccurrenceDescriptor &occurrence =
        testCase.fabric.computeOccurrences.front();
    occurrence.schedule = ComputeScheduleKind::Temporal;
    occurrence.instructionContextCapacity = 0;
    expectMapError(__func__, testCase,
                   MappingErrorCode::InvalidInstructionContextCapacity);
  }
}

} // namespace

void runMappingVerifierTests() {
  acceptsValidTechMapping();
  rejectsMismatchedArtifactIdentity();
  rejectsInvalidActorReferences();
  rejectsDuplicatePersistentIdentity();
  rejectsInvalidDataflowConnectivity();
  acceptsUncoveredActorlessPassthrough();
  rejectsIncompleteCoveredGraph();
  rejectsIncompleteActorOperationCorrespondence();
  rejectsTypedFamilyMismatch();
  rejectsCapabilityTopologyMismatch();
  rejectsInvalidBoundaryCorrespondence();
  rejectsInvalidCapabilityInventory();
  rejectsInvalidComputeOccurrence();
}

} // namespace loom::mapping::test
