#include "MappingCoreTestSupport.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace loom::mapping::test {
namespace {

void expectFreezeInfeasibility(const char *test,
                               llvm::Expected<FrozenRealizationGraph> result,
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

void freezesOrderedImplementationDomainFromExactFuMembership() {
  TestCase testCase = makeValidCase();
  const ArtifactIdentity &fabricId = testCase.fabric.identity;
  const FuDescriptor selectedFu = testCase.fabric.functionalUnits.front();
  const FuDescriptor otherFu{FuId(30), selectedFu.inputPorts,
                             selectedFu.outputPorts};
  testCase.fabric.functionalUnits.push_back(otherFu);
  testCase.fabric.computeOccurrences = {
      makeSpatialComputeOccurrence(fabricId, ComputeOccurrenceId(300),
                                   selectedFu, 3000),
      makeSpatialComputeOccurrence(fabricId, ComputeOccurrenceId(200), otherFu,
                                   4000),
      makeSpatialComputeOccurrence(fabricId, ComputeOccurrenceId(100),
                                   selectedFu, 5000)};
  testCase.fabric.computeOccurrences.front().schedule =
      ComputeScheduleKind::Temporal;
  FrozenRealizationGraph graph = validateAndFreeze(__func__, testCase);
  const FrozenComputeRealization &realization =
      graph.computeRealizations().front();
  if (realization.implDomainCount != 2)
    fail(__func__, "implementation domain has the wrong size");
  const FrozenImplementationOccurrence &first =
      graph.implementationOccurrences()[realization.implDomainOffset];
  const FrozenImplementationOccurrence &second =
      graph.implementationOccurrences()[realization.implDomainOffset + 1];
  if (graph.computeOccurrences()[first.occurrence].id !=
          ComputeOccurrenceId(100) ||
      graph.computeOccurrences()[second.occurrence].id !=
          ComputeOccurrenceId(300))
    fail(__func__, "implementation domain is not ordered by occurrence ID");
  if (!first.unaryEligible || !second.unaryEligible ||
      graph.computeOccurrences()[second.occurrence].schedule !=
          ComputeScheduleKind::Temporal)
    fail(__func__, "implementation domain lost schedule or unary eligibility");
}
void rejectsEmptyImplementationDomainAsMappingInfeasibility() {
  TestCase testCase = makeValidCase();
  testCase.fabric.computeOccurrences.clear();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  expectFreezeInfeasibility(
      __func__,
      freezeRealizationGraph(testCase.dataflow, testCase.fabric, mapping),
      FrozenMappingInfeasibilityCode::EmptyImplementationDomain);
}
void freezesFactorizedEndpointDomains() {
  TestCase testCase = makeValidCase();
  const PortDescriptor shaped = port(PortKind::Value, type(1), 16, 4, role(7));
  testCase.dataflow.graphs[0].inputPorts[0] = shaped;
  testCase.dataflow.actors[0].inputPorts[0] = shaped;
  testCase.fabric.functionalUnits[0].inputPorts[0] = shaped;
  testCase.fabric.operations[0].inputPorts[0] = shaped;
  testCase.fabric.encodings[0].inputs[0].port = shaped;
  testCase.fabric.encodings[0].operations[0].inputPorts[0] = shaped;
  testCase.fabric.computeOccurrences.front() = makeSpatialComputeOccurrence(
      testCase.fabric.identity, ComputeOccurrenceId(1000),
      testCase.fabric.functionalUnits.front(), 2000);
  ComputeOccurrenceDescriptor &occurrence =
      testCase.fabric.computeOccurrences.front();
  occurrence.endpoints[0].payloadCapacityBits = 32;
  auto addEndpoint = [&](std::uint64_t id, PortKind kind,
                         std::uint32_t endpointPayload,
                         std::uint32_t endpointTag, std::uint32_t arcPayload,
                         std::uint32_t arcTag, TypeKey endpointType,
                         PortRoleKey endpointRole, bool connected) {
    const ComputeEndpointId endpoint(id);
    occurrence.endpoints.push_back({endpoint,
                                    PortDirection::Input,
                                    kind,
                                    endpointPayload,
                                    endpointTag,
                                    {endpointType},
                                    endpointRole});
    if (connected)
      occurrence.localArcs.push_back(
          {FuPortRef{FuRef{testCase.fabric.identity, FuId(10)},
                     PortDirection::Input, 0},
           ComputeEndpointRef{testCase.fabric.identity, endpoint}, arcPayload,
           arcTag});
  };
  addEndpoint(2100, PortKind::Stream, 64, 64, 64, 64, type(1), role(7), true);
  addEndpoint(2101, PortKind::Value, 8, 64, 64, 64, type(1), role(7), true);
  addEndpoint(2102, PortKind::Value, 64, 64, 64, 64, type(1), role(7), false);
  addEndpoint(2103, PortKind::Value, 64, 64, 64, 64, type(1), role(99), true);
  addEndpoint(2104, PortKind::Value, 64, 64, 64, 64, type(99), role(7), true);
  addEndpoint(2105, PortKind::Value, 64, 2, 64, 64, type(1), role(7), true);
  addEndpoint(2106, PortKind::Value, 64, 64, 8, 64, type(1), role(7), true);
  addEndpoint(2107, PortKind::Value, 64, 64, 64, 2, type(1), role(7), true);
  FrozenRealizationGraph graph = validateAndFreeze(__func__, testCase);
  const FrozenComputeRealization &realization =
      graph.computeRealizations().front();
  const FrozenImplementationOccurrence &implementation =
      graph.implementationOccurrences()[realization.implDomainOffset];
  const FrozenPortDemand &input =
      graph.portDemands()[implementation.portDemandOffset];
  if (input.direction != PortDirection::Input || input.port != 0 ||
      input.endpointCount != 1)
    fail(__func__, "endpoint domain did not apply factorized constraints");
  const PnrIndex endpointIndex =
      graph.compatibleEndpoints()[input.endpointOffset];
  if (graph.physicalEndpoints()[endpointIndex].id != ComputeEndpointId(2000))
    fail(__func__, "endpoint domain retained the wrong physical endpoint");
}
void rejectsSpatialHallInfeasibilityWithoutEndpointVariants() {
  TestCase testCase = makeValidCase();
  ComputeOccurrenceDescriptor &occurrence =
      testCase.fabric.computeOccurrences.front();
  occurrence.endpoints[0].compatibleTypes.push_back(type(2));
  occurrence.localArcs.erase(
      std::remove_if(occurrence.localArcs.begin(), occurrence.localArcs.end(),
                     [](const ComputeLocalArcDescriptor &arc) {
                       return arc.fuPort.direction == PortDirection::Input &&
                              arc.fuPort.index == 2;
                     }),
      occurrence.localArcs.end());
  occurrence.localArcs.push_back(
      {FuPortRef{FuRef{testCase.fabric.identity, FuId(10)},
                 PortDirection::Input, 2},
       ComputeEndpointRef{testCase.fabric.identity, ComputeEndpointId(2000)},
       std::numeric_limits<std::uint32_t>::max(),
       std::numeric_limits<std::uint32_t>::max()});
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  expectFreezeInfeasibility(
      __func__,
      freezeRealizationGraph(testCase.dataflow, testCase.fabric, mapping),
      FrozenMappingInfeasibilityCode::EmptyUnaryEligibleDomain);
}
void acceptsSpatialAugmentingPathReassignment() {
  TestCase testCase = makeValidCase();
  ComputeOccurrenceDescriptor &occurrence =
      testCase.fabric.computeOccurrences.front();
  occurrence.endpoints[0].compatibleTypes.push_back(type(2));
  occurrence.endpoints[2].compatibleTypes.push_back(type(1));
  occurrence.localArcs.erase(
      std::remove_if(occurrence.localArcs.begin(), occurrence.localArcs.end(),
                     [](const ComputeLocalArcDescriptor &arc) {
                       return arc.fuPort.direction == PortDirection::Input &&
                              arc.fuPort.index == 2;
                     }),
      occurrence.localArcs.end());
  const std::uint32_t unbounded = std::numeric_limits<std::uint32_t>::max();
  occurrence.localArcs.push_back(
      {FuPortRef{FuRef{testCase.fabric.identity, FuId(10)},
                 PortDirection::Input, 0},
       ComputeEndpointRef{testCase.fabric.identity, ComputeEndpointId(2002)},
       unbounded, unbounded});
  occurrence.localArcs.push_back(
      {FuPortRef{FuRef{testCase.fabric.identity, FuId(10)},
                 PortDirection::Input, 2},
       ComputeEndpointRef{testCase.fabric.identity, ComputeEndpointId(2000)},
       unbounded, unbounded});

  FrozenRealizationGraph graph = validateAndFreeze(__func__, testCase);
  const FrozenComputeRealization &realization =
      graph.computeRealizations().front();
  const FrozenImplementationOccurrence &implementation =
      graph.implementationOccurrences()[realization.implDomainOffset];
  if (!implementation.unaryEligible)
    fail(__func__, "augmenting-path reassignment was rejected");
}
void freezesDeterministicallyAcrossInputPermutation() {
  TestCase baselineCase = makeMemoryAnchorCase();
  selectInternalMemoryGraph(baselineCase);
  FrozenRealizationGraph baseline = validateAndFreeze(__func__, baselineCase);
  TestCase permutedCase = makeMemoryAnchorCase();
  selectInternalMemoryGraph(permutedCase);
  std::reverse(permutedCase.dataflow.actors.begin(),
               permutedCase.dataflow.actors.end());
  std::reverse(permutedCase.dataflow.edges.begin(),
               permutedCase.dataflow.edges.end());
  std::reverse(permutedCase.fabric.functionalUnits.begin(),
               permutedCase.fabric.functionalUnits.end());
  std::reverse(permutedCase.fabric.operations.begin(),
               permutedCase.fabric.operations.end());
  std::reverse(permutedCase.fabric.encodings.begin(),
               permutedCase.fabric.encodings.end());
  std::reverse(permutedCase.fabric.computeOccurrences.begin(),
               permutedCase.fabric.computeOccurrences.end());
  for (ComputeOccurrenceDescriptor &occurrence :
       permutedCase.fabric.computeOccurrences) {
    std::reverse(occurrence.functionalUnits.begin(),
                 occurrence.functionalUnits.end());
    std::reverse(occurrence.endpoints.begin(), occurrence.endpoints.end());
    for (ComputeEndpointDescriptor &endpoint : occurrence.endpoints)
      std::reverse(endpoint.compatibleTypes.begin(),
                   endpoint.compatibleTypes.end());
    std::reverse(occurrence.localArcs.begin(), occurrence.localArcs.end());
  }
  std::reverse(permutedCase.fabric.memoryServiceDomains.begin(),
               permutedCase.fabric.memoryServiceDomains.end());
  std::reverse(permutedCase.fabric.memoryImplementations.begin(),
               permutedCase.fabric.memoryImplementations.end());
  std::reverse(permutedCase.fabric.memoryOperationPortTemplates.begin(),
               permutedCase.fabric.memoryOperationPortTemplates.end());
  std::reverse(permutedCase.fabric.memoryInternalConnections.begin(),
               permutedCase.fabric.memoryInternalConnections.end());
  std::reverse(permutedCase.fabric.memorySemanticEncodings.begin(),
               permutedCase.fabric.memorySemanticEncodings.end());
  std::reverse(permutedCase.mapping.realizations.begin(),
               permutedCase.mapping.realizations.end());
  MemoryRealizationDraft &memory =
      permutedCase.mapping.memoryRealizations.front();
  std::reverse(memory.actors.begin(), memory.actors.end());
  std::reverse(memory.actorToOperations.begin(),
               memory.actorToOperations.end());
  std::reverse(memory.boundaryPorts.begin(), memory.boundaryPorts.end());
  std::reverse(memory.graphBoundaryPorts.begin(),
               memory.graphBoundaryPorts.end());
  std::reverse(memory.internalEdges.begin(), memory.internalEdges.end());
  FrozenRealizationGraph permuted = validateAndFreeze(__func__, permutedCase);
  if (baseline != permuted)
    fail(__func__, "harmless vector permutation changed frozen output");
}
void enforcesFrozenInputIdentityBoundary() {
  TestCase testCase = makeValidCase();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  DataflowProgramView foreignDataflow = testCase.dataflow;
  foreignDataflow.identity = artifact(99);
  expectAnyError(__func__, freezeRealizationGraph(foreignDataflow,
                                                  testCase.fabric, mapping));
  FabricHardwareView foreignFabric = testCase.fabric;
  foreignFabric.identity = artifact(99);
  expectAnyError(__func__, freezeRealizationGraph(testCase.dataflow,
                                                  foreignFabric, mapping));
  takeExpected(__func__, freezeRealizationGraph(testCase.dataflow,
                                                testCase.fabric, mapping));
}
void preflightsFrozenCapacityPlanning() {
  TestCase testCase = makeValidCase();
  llvm::Error error =
      loom::pnr::detail::preflightFrozenRealizationGraphCapacity(
          testCase.mapping.realizations, testCase.mapping.memoryRealizations,
          getPnrIndexMax());
  if (!error)
    fail(__func__, "expected template terminal capacity failure");
  bool sawCapacityError = false;
  llvm::handleAllErrors(
      std::move(error), [&](const PnrIndexCapacityError &capacityError) {
        sawCapacityError = true;
        std::string message;
        llvm::raw_string_ostream stream(message);
        capacityError.log(stream);
        if (message.find("table 'template_terminals'") == std::string::npos)
          fail(__func__, "capacity failure named the wrong table");
      });
  if (!sawCapacityError)
    fail(__func__, "received a different capacity error category");
  PnrCapacityContext rangeContext{"FrozenRealizationGraph", "port_demands",
                                  "compatible_endpoints",
                                  PnrCapacityMeasure::Offset};
  error = loom::pnr::detail::preflightFrozenRangeCapacity(rangeContext,
                                                          getPnrIndexMax(), 1);
  if (!error)
    fail(__func__, "expected frozen range capacity failure");
  llvm::consumeError(std::move(error));
}
template <typename T> constexpr bool isPnrIndex = std::is_same_v<T, PnrIndex>;
static_assert(
    isPnrIndex<decltype(std::declval<FrozenActorOwnership>().realization)> &&
    isPnrIndex<
        decltype(std::declval<FrozenComputeTemplateTerminal>().realization)> &&
    isPnrIndex<decltype(std::declval<FrozenComputeTemplateTerminal>().port)> &&
    isPnrIndex<
        decltype(std::declval<FrozenMemoryTemplateTerminal>().realization)> &&
    isPnrIndex<decltype(std::declval<FrozenMemoryTemplateTerminal>().port)> &&
    isPnrIndex<decltype(std::declval<FrozenGraphBoundaryTerminal>().port)> &&
    isPnrIndex<decltype(std::declval<FrozenTemplateTerminalRef>().terminal)> &&
    isPnrIndex<decltype(std::declval<FrozenLogicalNet>().sinkOffset)> &&
    isPnrIndex<decltype(std::declval<FrozenLogicalNet>().sinkCount)>);

} // namespace

void runComputeFreezeTests() {
  freezesOrderedImplementationDomainFromExactFuMembership();
  rejectsEmptyImplementationDomainAsMappingInfeasibility();
  freezesFactorizedEndpointDomains();
  rejectsSpatialHallInfeasibilityWithoutEndpointVariants();
  acceptsSpatialAugmentingPathReassignment();
  freezesDeterministicallyAcrossInputPermutation();
  enforcesFrozenInputIdentityBoundary();
  preflightsFrozenCapacityPlanning();
}

} // namespace loom::mapping::test
