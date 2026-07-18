#include "MappingCoreTestSupport.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
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

void freezesCorrelatedFuAndInstructionContextDomains() {
  TestCase testCase = makeValidCase();
  const ArtifactIdentity &fabricId = testCase.fabric.identity;
  const FuDescriptor selectedFu = testCase.fabric.functionalUnits.front();
  const FuDescriptor otherFu{FuId(30), selectedFu.inputPorts,
                             selectedFu.outputPorts};
  testCase.fabric.functionalUnits.push_back(otherFu);

  ComputeOccurrenceDescriptor temporal = makeSpatialComputeOccurrence(
      fabricId, ComputeOccurrenceId(300), selectedFu, 3000);
  temporal.schedule = ComputeScheduleKind::Temporal;
  temporal.instructionContextCapacity = 3;
  ComputeOccurrenceDescriptor spatial = makeSpatialComputeOccurrence(
      fabricId, ComputeOccurrenceId(100), selectedFu, 5000);
  spatial.functionalUnits.push_back(FuRef{fabricId, otherFu.id});
  std::reverse(spatial.functionalUnits.begin(), spatial.functionalUnits.end());
  testCase.fabric.computeOccurrences = {std::move(temporal),
                                        std::move(spatial)};

  FrozenRealizationGraph graph = validateAndFreeze(__func__, testCase);
  if (graph.fabricPeOccurrences().size() != 2 ||
      graph.fabricPeOccurrences()[0].ref !=
          FabricPeOccurrenceRef{ComputeOccurrenceId(100)} ||
      graph.fabricPeOccurrences()[0].contextCount != 1 ||
      graph.fabricPeOccurrences()[1].ref !=
          FabricPeOccurrenceRef{ComputeOccurrenceId(300)} ||
      graph.fabricPeOccurrences()[1].contextCount != 3)
    fail(__func__, "PE occurrence descriptors are not canonical");

  const llvm::ArrayRef<FrozenFabricFuOccurrence> fuOccurrences =
      graph.fabricFuOccurrences();
  if (fuOccurrences.size() != 3 ||
      fuOccurrences[0].ref !=
          FabricFuOccurrenceRef{FabricPeOccurrenceRef{ComputeOccurrenceId(100)},
                                FuId(10)} ||
      fuOccurrences[1].ref !=
          FabricFuOccurrenceRef{FabricPeOccurrenceRef{ComputeOccurrenceId(100)},
                                FuId(30)} ||
      fuOccurrences[2].ref !=
          FabricFuOccurrenceRef{FabricPeOccurrenceRef{ComputeOccurrenceId(300)},
                                FuId(10)} ||
      fuOccurrences[0].ref == fuOccurrences[1].ref)
    fail(__func__, "FU occurrence descriptors lost concrete identity");
  if (graph.findFabricFuOccurrence(fuOccurrences[0].ref) != &fuOccurrences[0] ||
      graph.findFabricFuOccurrence(fuOccurrences[1].ref) != &fuOccurrences[1])
    fail(__func__, "FU occurrence lookup is not exact");
  const std::optional<InstructionContextRef> selectedSpatialContext =
      graph.instructionContext(fuOccurrences[0].ref, ContextOrdinal(0));
  const std::optional<InstructionContextRef> otherSpatialContext =
      graph.instructionContext(fuOccurrences[1].ref, ContextOrdinal(0));
  const std::optional<InstructionContextRef> temporalContext =
      graph.instructionContext(fuOccurrences[2].ref, ContextOrdinal(0));
  if (!selectedSpatialContext || !otherSpatialContext || !temporalContext ||
      *selectedSpatialContext != *otherSpatialContext ||
      selectedSpatialContext->pe == temporalContext->pe)
    fail(__func__, "instruction contexts lost parent-PE correlation");

  const FrozenComputeRealization &realization =
      graph.computeRealizations().front();
  if (realization.implDomainCount != 2)
    fail(__func__, "implementation domain has the wrong size");
  const FrozenImplementationOccurrence &first =
      graph.implementationOccurrences()[realization.implDomainOffset];
  const FrozenImplementationOccurrence &second =
      graph.implementationOccurrences()[realization.implDomainOffset + 1];
  if (first.fuOccurrence != fuOccurrences[0].ref ||
      second.fuOccurrence != fuOccurrences[2].ref)
    fail(__func__, "implementation domain is not ordered by FU occurrence");
  if (!first.unaryEligible || !second.unaryEligible)
    fail(__func__, "implementation domain lost unary eligibility");

  for (const FrozenImplementationOccurrence *implementation :
       {&first, &second}) {
    const FrozenFabricFuOccurrence *fu =
        graph.findFabricFuOccurrence(implementation->fuOccurrence);
    if (fu == nullptr)
      fail(__func__, "implementation names an unknown FU occurrence");
    const FrozenFabricPeOccurrence *parent =
        graph.findFabricPeOccurrence(fu->ref.parentPe);
    if (parent == nullptr)
      fail(__func__, "FU occurrence names an unknown parent PE");
    for (PnrIndex ordinal = 0; ordinal < parent->contextCount; ++ordinal) {
      const std::optional<InstructionContextRef> context =
          graph.instructionContext(fu->ref, ContextOrdinal(ordinal));
      if (!context || context->pe != fu->ref.parentPe ||
          context->ordinal != ContextOrdinal(ordinal))
        fail(__func__, "FU context domain crossed its parent PE");
    }
    if (graph.instructionContext(fu->ref, ContextOrdinal(parent->contextCount)))
      fail(__func__, "FU context domain accepted an invalid ordinal");
  }
}
void rejectsEmptyConcreteFuDomainAsMappingInfeasibility() {
  TestCase testCase = makeValidCase();
  testCase.fabric.computeOccurrences.clear();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  ResolvedPnrConfigView config;
  expectFreezeInfeasibility(
      __func__,
      freezeRealizationGraph(makePnrProblemInputs(testCase, mapping, config)),
      FrozenMappingInfeasibilityCode::EmptyConcreteFuDomain);
}
void freezesOnlyActiveWideSyncBoundaryPorts() {
  TestCase testCase = makeWideSyncCase();
  selectWideSyncLanes(testCase, {1, 3});
  ComputeOccurrenceDescriptor &occurrence =
      testCase.fabric.computeOccurrences.front();
  occurrence.localArcs.erase(
      std::remove_if(
          occurrence.localArcs.begin(), occurrence.localArcs.end(),
          [](const ComputeLocalArcDescriptor &arc) {
            return (arc.fuPort.direction == PortDirection::Input &&
                    (arc.fuPort.index == 2 || arc.fuPort.index == 3)) ||
                   (arc.fuPort.direction == PortDirection::Output &&
                    (arc.fuPort.index == 0 || arc.fuPort.index == 1));
          }),
      occurrence.localArcs.end());

  FrozenRealizationGraph graph = validateAndFreeze(__func__, testCase);
  std::set<std::pair<PortDirection, std::uint32_t>> demands;
  for (const FrozenPortDemand &demand : graph.portDemands()) {
    if (demand.endpointCount == 0)
      fail(__func__, "active wide-sync demand has no compatible endpoint");
    demands.emplace(demand.direction, static_cast<std::uint32_t>(demand.port));
  }
  const std::set<std::pair<PortDirection, std::uint32_t>> expected{
      {PortDirection::Input, 0},
      {PortDirection::Input, 1},
      {PortDirection::Output, 2},
      {PortDirection::Output, 3}};
  if (demands != expected || graph.portDemands().size() != expected.size())
    fail(__func__, "freeze retained inactive wide-sync boundary demands");
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
                                    endpointRole,
                                    endpointTag == 0
                                        ? fabric::DataPathKind::Bits
                                        : fabric::DataPathKind::BitsTag});
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
  ResolvedPnrConfigView config;
  expectFreezeInfeasibility(
      __func__,
      freezeRealizationGraph(makePnrProblemInputs(testCase, mapping, config)),
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
  ResolvedPnrConfigView config;
  PnrProblemInputs validInputs =
      makePnrProblemInputs(testCase, mapping, config);
  takeExpected(__func__, freezeRealizationGraph(validInputs));

  TestCase foreignDataflowCase = testCase;
  foreignDataflowCase.dataflow.identity = artifact(99);
  expectAnyError(__func__, freezeRealizationGraph(makePnrProblemInputs(
                               foreignDataflowCase, mapping, config)));

  TestCase foreignFabricCase = testCase;
  foreignFabricCase.fabric.identity = artifact(99);
  expectAnyError(__func__, freezeRealizationGraph(makePnrProblemInputs(
                               foreignFabricCase, mapping, config)));
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
static_assert(!std::is_default_constructible_v<ContextOrdinal>);
static_assert(
    std::is_same_v<decltype(std::declval<InstructionContextRef>().pe),
                   FabricPeOccurrenceRef> &&
    std::is_same_v<decltype(std::declval<InstructionContextRef>().ordinal),
                   ContextOrdinal>);
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
  freezesCorrelatedFuAndInstructionContextDomains();
  rejectsEmptyConcreteFuDomainAsMappingInfeasibility();
  freezesOnlyActiveWideSyncBoundaryPorts();
  freezesFactorizedEndpointDomains();
  rejectsSpatialHallInfeasibilityWithoutEndpointVariants();
  acceptsSpatialAugmentingPathReassignment();
  freezesDeterministicallyAcrossInputPermutation();
  enforcesFrozenInputIdentityBoundary();
  preflightsFrozenCapacityPlanning();
}

} // namespace loom::mapping::test
