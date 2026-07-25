#include "MappingCoreTestSupport.h"

namespace loom::mapping::test {
namespace {

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

} // namespace

void runCapabilityTemplateTests() {
  acceptsExactCapabilityTemplateReference();
  rejectsInvalidCapabilityTemplateReference();
  rejectsInactiveOperationNode();
  rejectsIncompleteOrNonInjectivePortMap();
}

} // namespace loom::mapping::test
