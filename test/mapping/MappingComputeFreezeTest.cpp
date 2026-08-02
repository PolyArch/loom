#include "MappingCoreTestSupport.h"

#include <type_traits>
#include <variant>

namespace loom::mapping::test {
namespace {

void expectFreezeInfeasibility(const char *test,
                               llvm::Expected<FrozenModelHandle> result,
                               FrozenMappingInfeasibilityCode expected) {
  if (result)
    fail(test, "expected frozen mapping infeasibility");
  bool matched = false;
  llvm::handleAllErrors(result.takeError(),
                        [&](const FrozenMappingInfeasibility &error) {
                          matched = error.code() == expected;
                        });
  if (!matched)
    fail(test, "received a different frozen mapping failure");
}

void freezesOnlyOccurrencesOfTheSelectedFuDefinition() {
  TestCase testCase = makeValidCase();
  const ArtifactIdentity &fabricId = testCase.fabric.identity;
  const FuDescriptor selectedFu = testCase.fabric.functionalUnits.front();

  FuDescriptor otherFu = selectedFu;
  otherFu.id = ::loom::fabric::FabricFuTemplateRef(30);
  for (auto &record : otherFu.capabilityTemplates) {
    for (auto &node : record.activeNodes)
      node.fu = otherFu.id;
    for (auto &edge : record.activeEdges) {
      auto rewrite = [&](auto &endpoint) {
        std::visit(
            [&](auto &port) {
              if constexpr (std::is_same_v<
                                std::decay_t<decltype(port)>,
                                ::loom::fabric::FabricFuTemplatePortRef>)
                port.fu = otherFu.id;
              else
                port.node.fu = otherFu.id;
            },
            endpoint.payload);
      };
      rewrite(edge.source);
      rewrite(edge.destination);
    }
  }
  testCase.fabric.functionalUnits.push_back(std::move(otherFu));

  ComputeOccurrenceDescriptor selectedOccurrence = makeSpatialComputeOccurrence(
      fabricId, ComputeOccurrenceId(100), selectedFu, 5000);
  ComputeOccurrenceDescriptor mixedOccurrence = makeSpatialComputeOccurrence(
      fabricId, ComputeOccurrenceId(200), selectedFu, 6000);
  mixedOccurrence.functionalUnits.push_back(
      ::loom::fabric::FabricFuTemplateRef(30));
  testCase.fabric.computeOccurrences = {std::move(mixedOccurrence),
                                        std::move(selectedOccurrence)};

  FrozenModelHandle model = validateAndFreeze(__func__, testCase);
  const FrozenRealizationGraph &graph = model->realizations();
  const FrozenComputeRealization &realization =
      graph.computeRealizations().front();
  if (realization.capabilityTemplate !=
          testCase.mapping.realizations.front().capabilityTemplate ||
      realization.implDomainCount != 2)
    fail(__func__, "frozen domain lost the selected capability template");
  for (const FrozenImplementationOccurrence &implementation :
       graph.implementationOccurrences())
    if (implementation.fuOccurrence.implementation !=
        realization.capabilityTemplate.fu)
      fail(__func__, "Spatial domain admitted a different FU definition");
}

void rejectsMissingSelectedFuOccurrence() {
  TestCase testCase = makeValidCase();
  testCase.fabric.computeOccurrences.clear();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  ResolvedPnrConfigView config = makeSpatialPnrConfigView(__func__);
  expectFreezeInfeasibility(
      __func__,
      freezeSpatialPnrModel(makePnrProblemInputs(testCase, mapping, config)),
      FrozenMappingInfeasibilityCode::EmptyConcreteFuDomain);
}

void freezesExactBoundaryPortDemand() {
  TestCase testCase = makeValidCase();
  FrozenModelHandle model = validateAndFreeze(__func__, testCase);
  const FrozenRealizationGraph &graph = model->realizations();
  const FrozenComputeRealization &realization =
      graph.computeRealizations().front();
  const FrozenImplementationOccurrence &implementation =
      graph.implementationOccurrences()[realization.implDomainOffset];
  if (implementation.portDemandCount != 4)
    fail(__func__, "frozen domain lost active FU boundary ports");
  for (std::size_t index = 0; index < implementation.portDemandCount; ++index) {
    const FrozenPortDemand &demand =
        graph.portDemands()[implementation.portDemandOffset + index];
    if (demand.fu != realization.capabilityTemplate.fu)
      fail(__func__, "boundary demand names a different FU definition");
  }
}

} // namespace

void runComputeFreezeTests() {
  freezesOnlyOccurrencesOfTheSelectedFuDefinition();
  rejectsMissingSelectedFuOccurrence();
  freezesExactBoundaryPortDemand();
}

} // namespace loom::mapping::test
