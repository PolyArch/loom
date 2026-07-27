#include "MappingCoreTestSupport.h"

namespace loom::mapping::test {
namespace {

void setComputePortWidths(TestCase &testCase, std::uint32_t softwareWidth,
                          std::uint32_t physicalWidth) {
  for (GraphDescriptor &graph : testCase.dataflow.graphs) {
    for (PortDescriptor &port : graph.inputPorts)
      port.payloadWidthBits = softwareWidth;
    for (PortDescriptor &port : graph.outputPorts)
      port.payloadWidthBits = softwareWidth;
  }
  for (ActorDescriptor &actor : testCase.dataflow.actors) {
    for (PortDescriptor &port : actor.inputPorts)
      port.payloadWidthBits = softwareWidth;
    for (PortDescriptor &port : actor.outputPorts)
      port.payloadWidthBits = softwareWidth;
  }
  for (FuDescriptor &fu : testCase.fabric.functionalUnits) {
    for (PortDescriptor &port : fu.inputPorts)
      port.payloadWidthBits = physicalWidth;
    for (PortDescriptor &port : fu.outputPorts)
      port.payloadWidthBits = physicalWidth;
  }
  for (FabricOpDescriptor &operation : testCase.fabric.operations) {
    for (PortDescriptor &port : operation.inputPorts)
      port.payloadWidthBits = physicalWidth;
    for (PortDescriptor &port : operation.outputPorts)
      port.payloadWidthBits = physicalWidth;
  }
}

void acceptsExactCapabilityTemplateReference() {
  TestCase testCase = makeValidCase();
  const ComputeRealizationDraft &realization =
      testCase.mapping.realizations.front();
  if (realization.capabilityTemplate.ordinal != 0)
    fail(__func__, "valid realization lost its exact template ordinal");
  if (realization.actorToOps.size() != 2 ||
      realization.actorToOps.front().operandPorts.size() != 2 ||
      realization.actorToOps.front().resultPorts.size() != 1)
    fail(__func__, "valid realization lost ordered actor port maps");
  validateCase(__func__, testCase);
}

void rejectsInvalidCapabilityTemplateReference() {
  TestCase testCase = makeValidCase();
  testCase.mapping.realizations.front().capabilityTemplate.ordinal = 1;
  expectMapError(__func__, testCase,
                 MappingErrorCode::InvalidCapabilityTemplateReference);
}

void rejectsInactiveOperationNode() {
  TestCase testCase = makeValidCase();
  testCase.mapping.realizations.front().actorToOps.front().fabricOp.ordinal =
      99;
  expectMapError(__func__, testCase,
                 MappingErrorCode::CapabilityTemplateMismatch);
}

void rejectsIncompleteOrNonInjectivePortMap() {
  {
    TestCase testCase = makeValidCase();
    testCase.mapping.realizations.front()
        .actorToOps.front()
        .operandPorts.pop_back();
    expectMapError(__func__, testCase,
                   MappingErrorCode::CapabilityTemplateMismatch);
  }
  {
    TestCase testCase = makeValidCase();
    ActorToFabricOp &binding =
        testCase.mapping.realizations.front().actorToOps.front();
    binding.operandPorts = {0, 0};
    expectMapError(__func__, testCase,
                   MappingErrorCode::CapabilityTemplateMismatch);
  }
}

void acceptsWiderPhysicalComputePorts() {
  TestCase testCase = makeValidCase();
  setComputePortWidths(testCase, 32, 64);
  validateCase(__func__, testCase);
}

void rejectsUndersizedPhysicalComputePorts() {
  TestCase testCase = makeValidCase();
  setComputePortWidths(testCase, 32, 16);
  expectMapError(__func__, testCase,
                 MappingErrorCode::CapabilityTemplateMismatch);
}

} // namespace

void runCapabilityTemplateTests() {
  acceptsExactCapabilityTemplateReference();
  rejectsInvalidCapabilityTemplateReference();
  rejectsInactiveOperationNode();
  rejectsIncompleteOrNonInjectivePortMap();
  acceptsWiderPhysicalComputePorts();
  rejectsUndersizedPhysicalComputePorts();
}

} // namespace loom::mapping::test
