#include "MappingCoreTestSupport.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>

namespace loom::mapping::test {

static_assert(!std::is_default_constructible_v<MappingDraftHeader>);
static_assert(!std::is_default_constructible_v<TechMappingDraft>);
ArtifactIdentity artifact(std::uint8_t value) {
  ArtifactIdentity::Storage bytes{};
  bytes.front() = value;
  return llvm::cantFail(ArtifactIdentity::fromBytes(bytes));
}
TypeKey type(std::uint64_t value) { return TypeKey(value); }
PortRoleKey role(std::uint64_t value) { return PortRoleKey(value); }
SemanticKey semantic(std::uint8_t value) { return SemanticKey({value}); }
PortDescriptor port(PortKind kind, TypeKey typeKey,
                    std::uint32_t payloadWidthBits, std::uint32_t tagWidthBits,
                    PortRoleKey roleKey) {
  return PortDescriptor{kind, typeKey, payloadWidthBits, tagWidthBits, roleKey};
}
ComputeOccurrenceDescriptor makeSpatialComputeOccurrence(
    const ArtifactIdentity &fabric, ComputeOccurrenceId occurrence,
    const FuDescriptor &fu, std::uint64_t endpointBase) {
  constexpr std::uint32_t unbounded = std::numeric_limits<std::uint32_t>::max();
  std::vector<ComputeEndpointDescriptor> endpoints;
  std::vector<ComputeLocalArcDescriptor> localArcs;
  auto addPort = [&](PortDirection direction, std::uint32_t index,
                     const PortDescriptor &descriptor) {
    const ComputeEndpointId endpoint(endpointBase + endpoints.size());
    const fabric::DataPathKind transportKind =
        descriptor.tagWidthBits == 0 ? fabric::DataPathKind::Bits
                                     : fabric::DataPathKind::BitsTag;
    const std::uint32_t tagCapacityBits =
        transportKind == fabric::DataPathKind::Bits ? 0 : unbounded;
    endpoints.push_back({endpoint,
                         direction,
                         descriptor.kind,
                         unbounded,
                         tagCapacityBits,
                         {descriptor.type},
                         descriptor.role,
                         transportKind});
    localArcs.push_back({FuPortRef{FuRef{fabric, fu.id}, direction, index},
                         ComputeEndpointRef{fabric, endpoint}, unbounded,
                         unbounded});
  };
  for (std::size_t index = 0; index < fu.inputPorts.size(); ++index)
    addPort(PortDirection::Input, static_cast<std::uint32_t>(index),
            fu.inputPorts[index]);
  for (std::size_t index = 0; index < fu.outputPorts.size(); ++index)
    addPort(PortDirection::Output, static_cast<std::uint32_t>(index),
            fu.outputPorts[index]);
  return ComputeOccurrenceDescriptor{occurrence,
                                     ComputeScheduleKind::Spatial,
                                     {FuRef{fabric, fu.id}},
                                     std::move(endpoints),
                                     std::move(localArcs)};
}
[[noreturn]] void fail(const char *test, const char *message) {
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
ValidatedTechMapping validateCase(const char *test, const TestCase &testCase) {
  return takeExpected(test,
                      validateTechMapping(testCase.mapping, testCase.dataflow,
                                          testCase.fabric));
}
FrozenRealizationGraph validateAndFreeze(const char *test, TestCase &testCase) {
  ValidatedTechMapping mapping = validateCase(test, testCase);
  return takeExpected(test, freezeRealizationGraph(testCase.dataflow,
                                                   testCase.fabric, mapping));
}
void expectMapError(const char *test, const TestCase &testCase,
                    MappingErrorCode expected) {
  expectError(
      test,
      validateTechMapping(testCase.mapping, testCase.dataflow, testCase.fabric),
      expected);
}
void selectWideSyncLanes(TestCase &testCase,
                         llvm::ArrayRef<std::uint32_t> laneIndices) {
  const FabricOpDescriptor &operation = testCase.fabric.operations.front();
  ActorToFabricOp &correspondence =
      testCase.mapping.realizations.front().actorToOps.front();
  correspondence.laneSelections.clear();
  testCase.mapping.realizations.front().boundaryPorts.clear();

  const ArtifactIdentity &dataflowId = testCase.dataflow.identity;
  const ArtifactIdentity &fabricId = testCase.fabric.identity;
  const ActorId actor = testCase.dataflow.actors.front().id;
  const FuId fu = testCase.fabric.functionalUnits.front().id;
  for (auto [softwareLane, laneIndex] : llvm::enumerate(laneIndices)) {
    const PairedLaneDescriptor &lane = operation.pairedLanes[laneIndex];
    correspondence.laneSelections.push_back({lane.inputPort, lane.outputPort});
    testCase.mapping.realizations.front().boundaryPorts.push_back(
        {ActorPortRef{ActorRef{dataflowId, actor}, PortDirection::Input,
                      static_cast<std::uint32_t>(softwareLane)},
         FuPortRef{FuRef{fabricId, fu}, PortDirection::Input, lane.inputPort}});
    testCase.mapping.realizations.front().boundaryPorts.push_back(
        {ActorPortRef{ActorRef{dataflowId, actor}, PortDirection::Output,
                      static_cast<std::uint32_t>(softwareLane)},
         FuPortRef{FuRef{fabricId, fu}, PortDirection::Output,
                   lane.outputPort}});
  }
}
TestCase makeWideSyncCase() {
  const ArtifactIdentity dataflowId = artifact(31);
  const ArtifactIdentity fabricId = artifact(32);
  const PortDescriptor value = port(PortKind::Value, type(31), 32);
  const GraphId graph(31);
  const ActorId syncActor(32);
  const SemanticKey sync = semantic(31);
  const SemanticKey noAttributes = semantic(32);
  const FuId fu(33);
  const FabricOpId syncOp(34);
  const EncodingId encoding(35);

  DataflowProgramView dataflow{
      dataflowId,
      {GraphDescriptor{graph, {value, value}, {value, value}}},
      {ActorDescriptor{syncActor,
                       graph,
                       sync,
                       noAttributes,
                       {value, value},
                       {value, value},
                       std::nullopt}},
      {DataflowEdge{EdgeId(310), GraphPort{graph, PortDirection::Input, 0},
                    ActorPort{syncActor, PortDirection::Input, 0}},
       DataflowEdge{EdgeId(311), GraphPort{graph, PortDirection::Input, 1},
                    ActorPort{syncActor, PortDirection::Input, 1}},
       DataflowEdge{EdgeId(312), ActorPort{syncActor, PortDirection::Output, 0},
                    GraphPort{graph, PortDirection::Output, 0}},
       DataflowEdge{EdgeId(313), ActorPort{syncActor, PortDirection::Output, 1},
                    GraphPort{graph, PortDirection::Output, 1}}},
      {}};

  FabricHardwareView fabric{
      fabricId,
      {FuDescriptor{
          fu, {value, value, value, value}, {value, value, value, value}}},
      {FabricOpDescriptor{
          syncOp,
          fu,
          {value, value, value, value},
          {value, value, value, value},
          {PairedLaneDescriptor{2, 1, 3}, PairedLaneDescriptor{0, 3, 0},
           PairedLaneDescriptor{3, 0, 2}, PairedLaneDescriptor{1, 2, 1}}}},
      {EncodingDescriptor{
          encoding,
          fu,
          {{0, value}, {1, value}, {2, value}, {3, value}},
          {ConfiguredFabricOpDescriptor{syncOp,
                                        sync,
                                        noAttributes,
                                        {value, value, value, value},
                                        {value, value, value, value},
                                        {FuInputValue{2}, FuInputValue{0},
                                         FuInputValue{3}, FuInputValue{1}}}},
          {{1, value, FabricOpResultValue{syncOp, 0}},
           {3, value, FabricOpResultValue{syncOp, 1}},
           {0, value, FabricOpResultValue{syncOp, 2}},
           {2, value, FabricOpResultValue{syncOp, 3}}}}},
      {},
      {},
      {},
      {},
      {},
      {}};
  fabric.computeOccurrences.push_back(
      makeSpatialComputeOccurrence(fabricId, ComputeOccurrenceId(3100),
                                   fabric.functionalUnits.front(), 3200));

  ComputeRealizationDraft realization{
      ComputeRealizationId(36),
      {ActorRef{dataflowId, syncActor}},
      FuRef{fabricId, fu},
      EncodingRef{fabricId, encoding},
      {{ActorRef{dataflowId, syncActor}, FabricOpRef{fabricId, syncOp}, {}}},
      {}};
  TechMappingDraft mapping{MappingDraftHeader{SchemaVersion{2, 0},
                                              MappingProfile::TechMapping,
                                              dataflowId, fabricId},
                           {GraphRef{dataflowId, graph}},
                           {std::move(realization)},
                           {}};
  TestCase testCase{std::move(dataflow), std::move(fabric), std::move(mapping)};
  selectWideSyncLanes(testCase, {0, 1});
  return testCase;
}
TestCase makeValidCase() {
  const ArtifactIdentity dataflowId = artifact(1);
  const ArtifactIdentity fabricId = artifact(2);
  const TypeKey word = type(1);
  const PortDescriptor value = port(PortKind::Value, word);
  const PortDescriptor stream = port(PortKind::Stream, word);
  const PortDescriptor auxiliary = port(PortKind::Value, type(2));
  const GraphId graph(1);
  const ActorId multiplyActor(2);
  const ActorId addActor(3);
  const SemanticKey multiply = semantic(1);
  const SemanticKey add = semantic(2);
  const SemanticKey noAttributes = semantic(10);
  DataflowProgramView dataflow{
      dataflowId,
      {GraphDescriptor{graph, {value, stream, auxiliary}, {value}}},
      {ActorDescriptor{multiplyActor,
                       graph,
                       multiply,
                       noAttributes,
                       {value, stream},
                       {value},
                       std::nullopt},
       ActorDescriptor{addActor,
                       graph,
                       add,
                       noAttributes,
                       {value, auxiliary},
                       {value},
                       std::nullopt}},
      {DataflowEdge{EdgeId(100), GraphPort{graph, PortDirection::Input, 0},
                    ActorPort{multiplyActor, PortDirection::Input, 0}},
       DataflowEdge{EdgeId(101), GraphPort{graph, PortDirection::Input, 1},
                    ActorPort{multiplyActor, PortDirection::Input, 1}},
       DataflowEdge{EdgeId(102),
                    ActorPort{multiplyActor, PortDirection::Output, 0},
                    ActorPort{addActor, PortDirection::Input, 0}},
       DataflowEdge{EdgeId(103), GraphPort{graph, PortDirection::Input, 2},
                    ActorPort{addActor, PortDirection::Input, 1}},
       DataflowEdge{EdgeId(104), ActorPort{addActor, PortDirection::Output, 0},
                    GraphPort{graph, PortDirection::Output, 0}}},
      {}};
  const FuId fu(10);
  const FabricOpId multiplyOp(11);
  const FabricOpId addOp(12);
  const EncodingId encoding(13);
  FabricHardwareView fabric{
      fabricId,
      {FuDescriptor{fu, {value, stream, auxiliary}, {value}}},
      {FabricOpDescriptor{multiplyOp, fu, {value, stream}, {value}},
       FabricOpDescriptor{addOp, fu, {value, auxiliary}, {value}}},
      {EncodingDescriptor{
          encoding,
          fu,
          {{0, value}, {1, stream}, {2, auxiliary}},
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
               {value, auxiliary},
               {value},
               {FabricOpResultValue{multiplyOp, 0}, FuInputValue{2}}}},
          {{0, value, FabricOpResultValue{addOp, 0}}}}},
      {},
      {},
      {},
      {},
      {},
      {}};
  fabric.computeOccurrences.push_back(
      makeSpatialComputeOccurrence(fabricId, ComputeOccurrenceId(1000),
                                   fabric.functionalUnits.front(), 2000));
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
                           {std::move(realization)},
                           {}};
  return TestCase{std::move(dataflow), std::move(fabric), std::move(mapping)};
}
TestCase makeMemoryAnchorCase() {
  const ArtifactIdentity dataflowId = artifact(1);
  const ArtifactIdentity fabricId = artifact(2);
  const PortDescriptor value = port(PortKind::Value, type(1));
  const PortDescriptor control = port(PortKind::Value, type(2));
  const PortDescriptor memory = port(PortKind::Memory, type(3));
  const SemanticKey noAttributes = semantic(20);
  const GraphId graph(1);
  const ActorId loadActor(2);
  const ActorId xoriActor(3);
  const ActorId preAddActor(4);
  const ActorId multiplyActor(5);
  const ActorId subtractActor(6);
  const ActorId finalAddActor(7);
  const ActorId storeActor(8);
  const LogicalMemoryRootId root(20);
  auto binaryActor = [&](ActorId actor, std::uint8_t operation) {
    return ActorDescriptor{actor,        graph,          semantic(operation),
                           noAttributes, {value, value}, {value},
                           std::nullopt};
  };
  auto graphInputEdge = [&](std::uint64_t id, std::uint32_t input,
                            ActorId target, std::uint32_t port) {
    return DataflowEdge{EdgeId(id),
                        GraphPort{graph, PortDirection::Input, input},
                        ActorPort{target, PortDirection::Input, port}};
  };
  auto actorEdge = [&](std::uint64_t id, ActorId source, std::uint32_t result,
                       ActorId target, std::uint32_t operand) {
    return DataflowEdge{EdgeId(id),
                        ActorPort{source, PortDirection::Output, result},
                        ActorPort{target, PortDirection::Input, operand}};
  };
  DataflowProgramView dataflow{
      dataflowId,
      {GraphDescriptor{graph,
                       {memory, value, control, value, value, value, value},
                       {memory, control}}},
      {ActorDescriptor{
           loadActor,
           graph,
           semantic(1),
           noAttributes,
           {value, control},
           {value, control},
           CanonicalMemoryActorView{
               MemoryOperationKind::Load,
               root,
               16,
               2,
               4,
               {{MemoryAccessPortRole::Address, PortDirection::Input, 0},
                {MemoryAccessPortRole::Control, PortDirection::Input, 1},
                {MemoryAccessPortRole::Result, PortDirection::Output, 0},
                {MemoryAccessPortRole::Done, PortDirection::Output, 1}}}},
       binaryActor(xoriActor, 2), binaryActor(preAddActor, 3),
       binaryActor(multiplyActor, 4), binaryActor(subtractActor, 5),
       binaryActor(finalAddActor, 3),
       ActorDescriptor{
           storeActor,
           graph,
           semantic(6),
           noAttributes,
           {value, value, control},
           {control},
           CanonicalMemoryActorView{
               MemoryOperationKind::Store,
               root,
               16,
               2,
               4,
               {{MemoryAccessPortRole::Address, PortDirection::Input, 0},
                {MemoryAccessPortRole::Data, PortDirection::Input, 1},
                {MemoryAccessPortRole::Control, PortDirection::Input, 2},
                {MemoryAccessPortRole::Done, PortDirection::Output, 0}}}}},
      {graphInputEdge(100, 1, loadActor, 0),
       graphInputEdge(101, 2, loadActor, 1),
       actorEdge(102, loadActor, 0, xoriActor, 0),
       graphInputEdge(103, 3, xoriActor, 1),
       actorEdge(104, xoriActor, 0, preAddActor, 0),
       graphInputEdge(105, 4, preAddActor, 1),
       actorEdge(106, preAddActor, 0, multiplyActor, 0),
       actorEdge(107, preAddActor, 0, subtractActor, 0),
       graphInputEdge(108, 5, multiplyActor, 1),
       graphInputEdge(109, 6, subtractActor, 1),
       actorEdge(110, multiplyActor, 0, finalAddActor, 0),
       actorEdge(111, subtractActor, 0, finalAddActor, 1),
       actorEdge(112, finalAddActor, 0, storeActor, 1),
       graphInputEdge(113, 1, storeActor, 0),
       actorEdge(114, loadActor, 1, storeActor, 2),
       DataflowEdge{EdgeId(115),
                    ActorPort{storeActor, PortDirection::Output, 0},
                    GraphPort{graph, PortDirection::Output, 1}}},
      {LogicalMemoryRootDescriptor{
          root,
          graph,
          {GraphPort{graph, PortDirection::Input, 0}},
          {GraphPort{graph, PortDirection::Output, 0}}}}};
  const FuId pairFu(10);
  const FuId multiplyFu(20);
  const FuId subtractFu(23);
  const FuId finalAddFu(26);
  const FabricOpId xoriOp(11);
  const FabricOpId preAddOp(12);
  const FabricOpId multiplyOp(21);
  const FabricOpId subtractOp(24);
  const FabricOpId finalAddOp(27);
  const EncodingId pairEncoding(13);
  const EncodingId multiplyEncoding(22);
  const EncodingId subtractEncoding(25);
  const EncodingId finalAddEncoding(28);
  const MemoryServiceDomainId service(30);
  const MemoryServiceDomainId otherService(31);
  const MemoryImplementationId implementation(32);
  const MemoryImplementationId otherImplementation(33);
  const MemoryOperationPortTemplateId loadTemplate(34);
  const MemoryOperationPortTemplateId storeTemplate(35);
  const MemoryOperationPortTemplateId otherStoreTemplate(36);
  const MemoryInternalConnectionId addressToLoad(37);
  const MemoryInternalConnectionId addressToStore(38);
  const MemoryInternalConnectionId doneToControl(39);
  const MemoryInternalConnectionId doneToBoundary(40);
  const MemorySemanticEncodingId loadEncoding(41);
  const MemorySemanticEncodingId storeEncoding(42);
  const MemorySemanticEncodingId groupedEncoding(43);
  const MemorySemanticEncodingId otherStoreEncoding(44);
  const std::vector<MemoryImplementationBoundaryPortDescriptor>
      implementationPorts{{PortDirection::Input, value, 2},
                          {PortDirection::Output, control, 0}};
  const std::vector<MemoryOperationPortDescriptor> loadPorts{
      {MemoryAccessPortRole::Address, PortDirection::Input, value, 0},
      {MemoryAccessPortRole::Control, PortDirection::Input, control, 0},
      {MemoryAccessPortRole::Result, PortDirection::Output, value, 0},
      {MemoryAccessPortRole::Done, PortDirection::Output, control, 1}};
  const std::vector<MemoryOperationPortDescriptor> storePorts{
      {MemoryAccessPortRole::Address, PortDirection::Input, value, 0},
      {MemoryAccessPortRole::Data, PortDirection::Input, value, 0},
      {MemoryAccessPortRole::Control, PortDirection::Input, control, 0},
      {MemoryAccessPortRole::Done, PortDirection::Output, control, 1}};
  const std::vector<MemoryAccessCapability> loadAccess{{2, 2}, {4, 4}};
  const std::vector<MemoryAccessCapability> storeAccess{{2, 2}, {4, 4}};
  auto binaryEncoding = [&](EncodingId encoding, FuId fu, FabricOpId operation,
                            std::uint8_t operationSemantics) {
    return EncodingDescriptor{
        encoding,
        fu,
        {{0, value}, {1, value}},
        {ConfiguredFabricOpDescriptor{operation,
                                      semantic(operationSemantics),
                                      noAttributes,
                                      {value, value},
                                      {value},
                                      {FuInputValue{0}, FuInputValue{1}}}},
        {{0, value, FabricOpResultValue{operation, 0}}}};
  };
  FabricHardwareView fabric{
      fabricId,
      {FuDescriptor{pairFu, {value, value, value}, {value}},
       FuDescriptor{multiplyFu, {value, value}, {value}},
       FuDescriptor{subtractFu, {value, value}, {value}},
       FuDescriptor{finalAddFu, {value, value}, {value}}},
      {FabricOpDescriptor{xoriOp, pairFu, {value, value}, {value}},
       FabricOpDescriptor{preAddOp, pairFu, {value, value}, {value}},
       FabricOpDescriptor{multiplyOp, multiplyFu, {value, value}, {value}},
       FabricOpDescriptor{subtractOp, subtractFu, {value, value}, {value}},
       FabricOpDescriptor{finalAddOp, finalAddFu, {value, value}, {value}}},
      {EncodingDescriptor{
           pairEncoding,
           pairFu,
           {{0, value}, {1, value}, {2, value}},
           {ConfiguredFabricOpDescriptor{xoriOp,
                                         semantic(2),
                                         noAttributes,
                                         {value, value},
                                         {value},
                                         {FuInputValue{0}, FuInputValue{1}}},
            ConfiguredFabricOpDescriptor{
                preAddOp,
                semantic(3),
                noAttributes,
                {value, value},
                {value},
                {FabricOpResultValue{xoriOp, 0}, FuInputValue{2}}}},
           {{0, value, FabricOpResultValue{preAddOp, 0}}}},
       binaryEncoding(multiplyEncoding, multiplyFu, multiplyOp, 4),
       binaryEncoding(subtractEncoding, subtractFu, subtractOp, 5),
       binaryEncoding(finalAddEncoding, finalAddFu, finalAddOp, 3)},
      {MemoryServiceDomainDescriptor{service},
       MemoryServiceDomainDescriptor{otherService}},
      {MemoryImplementationDescriptor{implementation, service,
                                      implementationPorts},
       MemoryImplementationDescriptor{otherImplementation, otherService,
                                      implementationPorts}},
      {MemoryOperationPortTemplateDescriptor{loadTemplate, implementation,
                                             MemoryOperationKind::Load,
                                             loadPorts, 32, loadAccess},
       MemoryOperationPortTemplateDescriptor{storeTemplate, implementation,
                                             MemoryOperationKind::Store,
                                             storePorts, 32, storeAccess},
       MemoryOperationPortTemplateDescriptor{
           otherStoreTemplate, otherImplementation, MemoryOperationKind::Store,
           storePorts, 32, storeAccess}},
      {MemoryInternalConnectionDescriptor{addressToLoad, implementation,
                                          MemoryImplementationBoundaryPort{0},
                                          MemoryOperationPort{loadTemplate, 0}},
       MemoryInternalConnectionDescriptor{
           addressToStore, implementation, MemoryImplementationBoundaryPort{0},
           MemoryOperationPort{storeTemplate, 0}},
       MemoryInternalConnectionDescriptor{
           doneToControl, implementation, MemoryOperationPort{loadTemplate, 3},
           MemoryOperationPort{storeTemplate, 2}},
       MemoryInternalConnectionDescriptor{doneToBoundary, implementation,
                                          MemoryOperationPort{storeTemplate, 3},
                                          MemoryImplementationBoundaryPort{1}}},
      {MemorySemanticEncodingDescriptor{
           loadEncoding, implementation, {loadTemplate}, {}},
       MemorySemanticEncodingDescriptor{
           storeEncoding, implementation, {storeTemplate}, {}},
       MemorySemanticEncodingDescriptor{
           groupedEncoding,
           implementation,
           {loadTemplate, storeTemplate},
           {addressToLoad, addressToStore, doneToControl, doneToBoundary}},
       MemorySemanticEncodingDescriptor{
           otherStoreEncoding, otherImplementation, {otherStoreTemplate}, {}}},
      {}};
  for (std::size_t index = 0; index < fabric.functionalUnits.size(); ++index) {
    fabric.computeOccurrences.push_back(makeSpatialComputeOccurrence(
        fabricId, ComputeOccurrenceId(1000 + index),
        fabric.functionalUnits[index], 2000 + index * 100));
  }
  auto actorRef = [&](ActorId actor) { return ActorRef{dataflowId, actor}; };
  auto actorPort = [&](ActorId actor, PortDirection direction,
                       std::uint32_t index) {
    return ActorPortRef{actorRef(actor), direction, index};
  };
  auto fuPort = [&](FuId fu, PortDirection direction, std::uint32_t index) {
    return FuPortRef{FuRef{fabricId, fu}, direction, index};
  };
  auto memoryPort = [&](MemoryOperationPortTemplateId operation,
                        std::uint32_t index) {
    return MemoryOperationPortRef{
        MemoryOperationPortTemplateRef{fabricId, operation}, index};
  };
  auto singletonRealization = [&](std::uint64_t id, ActorId actor, FuId fu,
                                  EncodingId encoding, FabricOpId operation) {
    return ComputeRealizationDraft{
        ComputeRealizationId(id),
        {actorRef(actor)},
        FuRef{fabricId, fu},
        EncodingRef{fabricId, encoding},
        {{actorRef(actor), FabricOpRef{fabricId, operation}}},
        {{actorPort(actor, PortDirection::Input, 0),
          fuPort(fu, PortDirection::Input, 0)},
         {actorPort(actor, PortDirection::Input, 1),
          fuPort(fu, PortDirection::Input, 1)},
         {actorPort(actor, PortDirection::Output, 0),
          fuPort(fu, PortDirection::Output, 0)}}};
  };
  ComputeRealizationDraft pairRealization{
      ComputeRealizationId(50),
      {actorRef(xoriActor), actorRef(preAddActor)},
      FuRef{fabricId, pairFu},
      EncodingRef{fabricId, pairEncoding},
      {{actorRef(xoriActor), FabricOpRef{fabricId, xoriOp}},
       {actorRef(preAddActor), FabricOpRef{fabricId, preAddOp}}},
      {{actorPort(xoriActor, PortDirection::Input, 0),
        fuPort(pairFu, PortDirection::Input, 0)},
       {actorPort(xoriActor, PortDirection::Input, 1),
        fuPort(pairFu, PortDirection::Input, 1)},
       {actorPort(preAddActor, PortDirection::Input, 1),
        fuPort(pairFu, PortDirection::Input, 2)},
       {actorPort(preAddActor, PortDirection::Output, 0),
        fuPort(pairFu, PortDirection::Output, 0)}}};
  ComputeRealizationDraft multiplyRealization = singletonRealization(
      51, multiplyActor, multiplyFu, multiplyEncoding, multiplyOp);
  ComputeRealizationDraft subtractRealization = singletonRealization(
      52, subtractActor, subtractFu, subtractEncoding, subtractOp);
  ComputeRealizationDraft finalAddRealization = singletonRealization(
      53, finalAddActor, finalAddFu, finalAddEncoding, finalAddOp);
  MemoryRealizationDraft loadRealization{
      MemoryRealizationId(60),
      {actorRef(loadActor)},
      {{actorRef(loadActor),
        MemoryOperationPortTemplateRef{fabricId, loadTemplate},
        LogicalMemoryRootRef{dataflowId, root}}},
      {LogicalMemoryRootRef{dataflowId, root}},
      MemorySemanticEncodingRef{fabricId, loadEncoding},
      {{actorPort(loadActor, PortDirection::Input, 0),
        memoryPort(loadTemplate, 0)},
       {actorPort(loadActor, PortDirection::Input, 1),
        memoryPort(loadTemplate, 1)},
       {actorPort(loadActor, PortDirection::Output, 0),
        memoryPort(loadTemplate, 2)},
       {actorPort(loadActor, PortDirection::Output, 1),
        memoryPort(loadTemplate, 3)}},
      {},
      {}};
  MemoryRealizationDraft storeRealization{
      MemoryRealizationId(61),
      {actorRef(storeActor)},
      {{actorRef(storeActor),
        MemoryOperationPortTemplateRef{fabricId, storeTemplate},
        LogicalMemoryRootRef{dataflowId, root}}},
      {LogicalMemoryRootRef{dataflowId, root}},
      MemorySemanticEncodingRef{fabricId, storeEncoding},
      {{actorPort(storeActor, PortDirection::Input, 0),
        memoryPort(storeTemplate, 0)},
       {actorPort(storeActor, PortDirection::Input, 1),
        memoryPort(storeTemplate, 1)},
       {actorPort(storeActor, PortDirection::Input, 2),
        memoryPort(storeTemplate, 2)},
       {actorPort(storeActor, PortDirection::Output, 0),
        memoryPort(storeTemplate, 3)}},
      {},
      {}};
  TechMappingDraft mapping{
      MappingDraftHeader{SchemaVersion{2, 0}, MappingProfile::TechMapping,
                         dataflowId, fabricId},
      {GraphRef{dataflowId, graph}},
      {std::move(pairRealization), std::move(multiplyRealization),
       std::move(subtractRealization), std::move(finalAddRealization)},
      {std::move(loadRealization), std::move(storeRealization)}};
  return TestCase{std::move(dataflow), std::move(fabric), std::move(mapping)};
}
void selectInternalMemoryGraph(TestCase &testCase) {
  const ArtifactIdentity &dataflowId = testCase.dataflow.identity;
  const ArtifactIdentity &fabricId = testCase.fabric.identity;
  const MemoryRealizationDraft &load = testCase.mapping.memoryRealizations[0];
  const MemoryRealizationDraft &store = testCase.mapping.memoryRealizations[1];
  const GraphRef graph{dataflowId, GraphId(1)};
  const MemoryImplementationRef implementation{fabricId,
                                               MemoryImplementationId(32)};
  MemoryRealizationDraft grouped{
      MemoryRealizationId(60),
      {load.actors.front(), store.actors.front()},
      {load.actorToOperations.front(), store.actorToOperations.front()},
      {LogicalMemoryRootRef{dataflowId, LogicalMemoryRootId(20)}},
      MemorySemanticEncodingRef{fabricId, MemorySemanticEncodingId(43)},
      {load.boundaryPorts[1], load.boundaryPorts[2], store.boundaryPorts[1]},
      {{GraphPortRef{graph, PortDirection::Input, 1},
        MemoryImplementationBoundaryPortRef{implementation, 0}},
       {GraphPortRef{graph, PortDirection::Output, 1},
        MemoryImplementationBoundaryPortRef{implementation, 1}}},
      {{EdgeRef{dataflowId, EdgeId(100)},
        MemoryInternalConnectionRef{fabricId, MemoryInternalConnectionId(37)}},
       {EdgeRef{dataflowId, EdgeId(113)},
        MemoryInternalConnectionRef{fabricId, MemoryInternalConnectionId(38)}},
       {EdgeRef{dataflowId, EdgeId(114)},
        MemoryInternalConnectionRef{fabricId, MemoryInternalConnectionId(39)}},
       {EdgeRef{dataflowId, EdgeId(115)},
        MemoryInternalConnectionRef{fabricId,
                                    MemoryInternalConnectionId(40)}}}};
  testCase.mapping.memoryRealizations = {std::move(grouped)};
}

} // namespace loom::mapping::test
