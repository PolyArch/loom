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
    const ::fabric::DataPathKind transportKind =
        descriptor.tagWidthBits == 0 ? ::fabric::DataPathKind::Bits
                                     : ::fabric::DataPathKind::BitsTag;
    const std::uint32_t tagCapacityBits =
        transportKind == ::fabric::DataPathKind::Bits ? 0 : unbounded;
    endpoints.push_back({endpoint,
                         direction,
                         descriptor.kind,
                         unbounded,
                         tagCapacityBits,
                         {descriptor.type},
                         descriptor.role,
                         transportKind});
    localArcs.push_back({::loom::fabric::FabricFuTemplatePortRef{
                             fu.id,
                             direction == PortDirection::Input
                                 ? ::loom::fabric::FabricPortDirection::Input
                                 : ::loom::fabric::FabricPortDirection::Output,
                             index},
                         ComputeEndpointRef{fabric, endpoint}, unbounded,
                         unbounded});
  };
  for (std::size_t index = 0; index < fu.inputPorts.size(); ++index)
    addPort(PortDirection::Input, static_cast<std::uint32_t>(index),
            fu.inputPorts[index]);
  for (std::size_t index = 0; index < fu.outputPorts.size(); ++index)
    addPort(PortDirection::Output, static_cast<std::uint32_t>(index),
            fu.outputPorts[index]);
  return ComputeOccurrenceDescriptor{
      occurrence,           ComputeScheduleKind::Spatial, {fu.id},
      std::move(endpoints), std::move(localArcs),         1};
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
  return takeExpected(
      test, validateTechMapping(testCase.techMappingIdentity, testCase.mapping,
                                testCase.dataflow, testCase.fabric));
}
PnrProblemInputs makePnrProblemInputs(TestCase &testCase,
                                      ValidatedTechMapping &mapping,
                                      ResolvedPnrConfigView &config) {
  return PnrProblemInputs{
      testCase.dataflow,
      mapping,
      testCase.fabric,
      config,
      artifact(241),
      MappingConstraintSetInput{artifact(242), testCase.dataflow.identity,
                                mapping.identity(), testCase.fabric.identity}};
}
FrozenRealizationGraph validateAndFreeze(const char *test, TestCase &testCase) {
  ValidatedTechMapping mapping = validateCase(test, testCase);
  ResolvedPnrConfigView config;
  return takeExpected(test, freezeRealizationGraph(makePnrProblemInputs(
                                testCase, mapping, config)));
}
void expectMapError(const char *test, const TestCase &testCase,
                    MappingErrorCode expected) {
  expectError(test,
              validateTechMapping(testCase.techMappingIdentity,
                                  testCase.mapping, testCase.dataflow,
                                  testCase.fabric),
              expected);
}

namespace {

::mlir::MLIRContext &testContext() {
  static ::mlir::MLIRContext context(
      ::mlir::MLIRContext::Threading::DISABLED);
  return context;
}

::dataflow::CanonicalActorSchemaProjection
integerProjection(::dataflow::OperationSchemaId schema, std::size_t inputCount,
                  std::size_t resultCount, unsigned width) {
  ::mlir::Type type = ::mlir::IntegerType::get(&testContext(), width);
  std::vector<::mlir::Type> inputs(inputCount, type);
  std::vector<::mlir::Type> results(resultCount, type);
  ::dataflow::SemanticPayload payload = ::dataflow::NoPayload{};
  if (schema == ::dataflow::OperationSchemaId::ArithAddI ||
      schema == ::dataflow::OperationSchemaId::ArithSubI ||
      schema == ::dataflow::OperationSchemaId::ArithMulI)
    payload = ::dataflow::IntegerOverflowPayload{};
  return {schema, ::mlir::FunctionType::get(&testContext(), inputs, results),
          std::move(payload)};
}

::dataflow::CanonicalActorSchemaProjection
memoryProjection(::dataflow::OperationSchemaId schema, std::size_t inputCount,
                 std::size_t resultCount, unsigned width) {
  ::mlir::Type type = ::mlir::IntegerType::get(&testContext(), width);
  return {
      schema,
      ::mlir::FunctionType::get(&testContext(),
                                std::vector<::mlir::Type>(inputCount, type),
                                std::vector<::mlir::Type>(resultCount, type)),
      ::dataflow::MemoryContractPayload{::dataflow::PlainAccessProjection{}}};
}

::fabric::FamilyCapabilityParams integerCapability(unsigned width) {
  const ::fabric::IntegerWidth admitted =
      width == 16 ? ::fabric::IntegerWidth::I16 : ::fabric::IntegerWidth::I32;
  return ::fabric::ScalarIntegerParams{
      ::fabric::IntegerWidthSet::get({admitted})};
}

::loom::fabric::FabricFuTemplateNodeRef
opNode(::loom::fabric::FabricFuTemplateRef fu, std::uint64_t ordinal) {
  return {::loom::fabric::FabricFuNodeKind::Op, fu, ordinal};
}

::loom::fabric::FabricFuCapabilityTemplateEndpointRef
boundaryEndpoint(::loom::fabric::FabricFuTemplateRef fu,
                 ::loom::fabric::FabricPortDirection direction,
                 std::uint64_t ordinal) {
  return ::loom::fabric::FabricFuCapabilityTemplateEndpointRef::boundaryPort(
      {fu, direction, ordinal});
}

::loom::fabric::FabricFuCapabilityTemplateEndpointRef
nodeEndpoint(::loom::fabric::FabricFuTemplateNodeRef node,
             ::loom::fabric::FabricPortDirection direction,
             std::uint64_t ordinal) {
  return ::loom::fabric::FabricFuCapabilityTemplateEndpointRef::nodePort(
      {node, direction, ordinal});
}

std::vector<::loom::fabric::FabricFuCapabilityTemplateRecord> templateInventory(
    std::vector<::loom::fabric::FabricFuTemplateNodeRef> nodes,
    std::vector<::loom::fabric::FabricFuCapabilityTemplateEdge> edges) {
  return llvm::cantFail(
      ::loom::fabric::normalizeFabricFuCapabilityTemplateInventory(
          {::loom::fabric::FabricFuCapabilityTemplateRecord{
              std::move(nodes), std::move(edges)}}));
}

::loom::fabric::FabricFuTemplatePortRef
fuPort(::loom::fabric::FabricFuTemplateRef fu, PortDirection direction,
       std::uint32_t ordinal) {
  return {fu,
          direction == PortDirection::Input
              ? ::loom::fabric::FabricPortDirection::Input
              : ::loom::fabric::FabricPortDirection::Output,
          ordinal};
}

} // namespace

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
  DataflowProgramView dataflow{
      dataflowId,
      32,
      {GraphDescriptor{graph, {value, stream, auxiliary}, {value}}},
      {ActorDescriptor{multiplyActor,
                       graph,
                       integerProjection(
                           ::dataflow::OperationSchemaId::ArithMulI, 2, 1, 32),
                       {value, stream},
                       {value},
                       std::nullopt},
       ActorDescriptor{addActor,
                       graph,
                       integerProjection(
                           ::dataflow::OperationSchemaId::ArithAddI, 2, 1, 32),
                       {value, auxiliary},
                       {value},
                       std::nullopt}},
      {DataflowEdge{GraphPort{graph, PortDirection::Input, 0},
                    ActorPort{multiplyActor, PortDirection::Input, 0}},
       DataflowEdge{GraphPort{graph, PortDirection::Input, 1},
                    ActorPort{multiplyActor, PortDirection::Input, 1}},
       DataflowEdge{ActorPort{multiplyActor, PortDirection::Output, 0},
                    ActorPort{addActor, PortDirection::Input, 0}},
       DataflowEdge{GraphPort{graph, PortDirection::Input, 2},
                    ActorPort{addActor, PortDirection::Input, 1}},
       DataflowEdge{ActorPort{addActor, PortDirection::Output, 0},
                    GraphPort{graph, PortDirection::Output, 0}}},
      {}};
  const ::loom::fabric::FabricFuTemplateRef fu(10);
  const auto multiplyOp = opNode(fu, 0);
  const auto addOp = opNode(fu, 1);
  const auto input = ::loom::fabric::FabricPortDirection::Input;
  const auto output = ::loom::fabric::FabricPortDirection::Output;
  FabricHardwareView fabric{
      fabricId,
      {FuDescriptor{fu,
                    {value, stream, auxiliary},
                    {value},
                    templateInventory({multiplyOp, addOp},
                                      {{boundaryEndpoint(fu, input, 0),
                                        nodeEndpoint(multiplyOp, input, 0)},
                                       {boundaryEndpoint(fu, input, 1),
                                        nodeEndpoint(multiplyOp, input, 1)},
                                       {nodeEndpoint(multiplyOp, output, 0),
                                        nodeEndpoint(addOp, input, 0)},
                                       {boundaryEndpoint(fu, input, 2),
                                        nodeEndpoint(addOp, input, 1)},
                                       {nodeEndpoint(addOp, output, 0),
                                        boundaryEndpoint(fu, output, 0)}})}},
      {FabricOpDescriptor{
           multiplyOp,
           ::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
           integerCapability(32),
           {value, stream},
           {value}},
       FabricOpDescriptor{addOp,
                          ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                          integerCapability(32),
                          {value, auxiliary},
                          {value}}},
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
      ::loom::fabric::FabricFuCapabilityTemplateRef{fu, 0},
      {{ActorRef{dataflowId, multiplyActor}, multiplyOp, {0, 1}, {0}},
       {ActorRef{dataflowId, addActor}, addOp, {0, 1}, {0}}},
      {{ActorPortRef{ActorRef{dataflowId, multiplyActor}, PortDirection::Input,
                     0},
        fuPort(fu, PortDirection::Input, 0)},
       {ActorPortRef{ActorRef{dataflowId, multiplyActor}, PortDirection::Input,
                     1},
        fuPort(fu, PortDirection::Input, 1)},
       {ActorPortRef{ActorRef{dataflowId, addActor}, PortDirection::Input, 1},
        fuPort(fu, PortDirection::Input, 2)},
       {ActorPortRef{ActorRef{dataflowId, addActor}, PortDirection::Output, 0},
        fuPort(fu, PortDirection::Output, 0)}}};
  TechMappingDraft mapping{MappingDraftHeader{dataflowId, fabricId},
                           {GraphRef{dataflowId, graph}},
                           {std::move(realization)},
                           {}};
  return TestCase{std::move(dataflow), std::move(fabric), artifact(3),
                  std::move(mapping)};
}
TestCase makeMemoryAnchorCase() {
  const ArtifactIdentity dataflowId = artifact(1);
  const ArtifactIdentity fabricId = artifact(2);
  const PortDescriptor value = port(PortKind::Value, type(1), 16);
  const PortDescriptor control = port(PortKind::Value, type(2));
  const PortDescriptor memory = port(PortKind::Memory, type(3));
  const GraphId graph(1);
  const ActorId loadActor(2);
  const ActorId xoriActor(3);
  const ActorId preAddActor(4);
  const ActorId multiplyActor(5);
  const ActorId subtractActor(6);
  const ActorId finalAddActor(7);
  const ActorId storeActor(8);
  const LogicalMemoryRootId root(20);
  auto binaryActor = [&](ActorId actor, ::dataflow::OperationSchemaId schema) {
    return ActorDescriptor{
        actor,          graph,   integerProjection(schema, 2, 1, 16),
        {value, value}, {value}, std::nullopt};
  };
  auto graphInputEdge = [&](std::uint32_t input, ActorId target,
                            std::uint32_t port) {
    return DataflowEdge{GraphPort{graph, PortDirection::Input, input},
                        ActorPort{target, PortDirection::Input, port}};
  };
  auto actorEdge = [&](ActorId source, std::uint32_t result, ActorId target,
                       std::uint32_t operand) {
    return DataflowEdge{ActorPort{source, PortDirection::Output, result},
                        ActorPort{target, PortDirection::Input, operand}};
  };
  DataflowProgramView dataflow{
      dataflowId,
      32,
      {GraphDescriptor{graph,
                       {memory, value, control, value, value, value, value},
                       {memory, control}}},
      {ActorDescriptor{
           loadActor,
           graph,
           memoryProjection(::dataflow::OperationSchemaId::DataflowLoad, 2, 2,
                            16),
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
       binaryActor(xoriActor, ::dataflow::OperationSchemaId::ArithXOrI),
       binaryActor(preAddActor, ::dataflow::OperationSchemaId::ArithAddI),
       binaryActor(multiplyActor, ::dataflow::OperationSchemaId::ArithMulI),
       binaryActor(subtractActor, ::dataflow::OperationSchemaId::ArithSubI),
       binaryActor(finalAddActor, ::dataflow::OperationSchemaId::ArithAddI),
       ActorDescriptor{
           storeActor,
           graph,
           memoryProjection(::dataflow::OperationSchemaId::DataflowStore, 3, 1,
                            16),
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
      {graphInputEdge(1, loadActor, 0), graphInputEdge(2, loadActor, 1),
       actorEdge(loadActor, 0, xoriActor, 0), graphInputEdge(3, xoriActor, 1),
       actorEdge(xoriActor, 0, preAddActor, 0),
       graphInputEdge(4, preAddActor, 1),
       actorEdge(preAddActor, 0, multiplyActor, 0),
       actorEdge(preAddActor, 0, subtractActor, 0),
       graphInputEdge(5, multiplyActor, 1), graphInputEdge(6, subtractActor, 1),
       actorEdge(multiplyActor, 0, finalAddActor, 0),
       actorEdge(subtractActor, 0, finalAddActor, 1),
       actorEdge(finalAddActor, 0, storeActor, 1),
       graphInputEdge(1, storeActor, 0), actorEdge(loadActor, 1, storeActor, 2),
       DataflowEdge{ActorPort{storeActor, PortDirection::Output, 0},
                    GraphPort{graph, PortDirection::Output, 1}}},
      {LogicalMemoryRootDescriptor{
          root,
          graph,
          {GraphPort{graph, PortDirection::Input, 0}},
          {GraphPort{graph, PortDirection::Output, 0}}}}};
  const ::loom::fabric::FabricFuTemplateRef pairFu(10);
  const ::loom::fabric::FabricFuTemplateRef multiplyFu(20);
  const ::loom::fabric::FabricFuTemplateRef subtractFu(23);
  const ::loom::fabric::FabricFuTemplateRef finalAddFu(26);
  const auto xoriOp = opNode(pairFu, 0);
  const auto preAddOp = opNode(pairFu, 1);
  const auto multiplyOp = opNode(multiplyFu, 0);
  const auto subtractOp = opNode(subtractFu, 0);
  const auto finalAddOp = opNode(finalAddFu, 0);
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
  const auto input = ::loom::fabric::FabricPortDirection::Input;
  const auto output = ::loom::fabric::FabricPortDirection::Output;
  auto binaryTemplate = [&](::loom::fabric::FabricFuTemplateRef fu,
                            ::loom::fabric::FabricFuTemplateNodeRef operation) {
    return templateInventory(
        {operation},
        {{boundaryEndpoint(fu, input, 0), nodeEndpoint(operation, input, 0)},
         {boundaryEndpoint(fu, input, 1), nodeEndpoint(operation, input, 1)},
         {nodeEndpoint(operation, output, 0),
          boundaryEndpoint(fu, output, 0)}});
  };
  FabricHardwareView fabric{
      fabricId,
      {FuDescriptor{pairFu,
                    {value, value, value},
                    {value},
                    templateInventory({xoriOp, preAddOp},
                                      {{boundaryEndpoint(pairFu, input, 0),
                                        nodeEndpoint(xoriOp, input, 0)},
                                       {boundaryEndpoint(pairFu, input, 1),
                                        nodeEndpoint(xoriOp, input, 1)},
                                       {nodeEndpoint(xoriOp, output, 0),
                                        nodeEndpoint(preAddOp, input, 0)},
                                       {boundaryEndpoint(pairFu, input, 2),
                                        nodeEndpoint(preAddOp, input, 1)},
                                       {nodeEndpoint(preAddOp, output, 0),
                                        boundaryEndpoint(pairFu, output, 0)}})},
       FuDescriptor{multiplyFu,
                    {value, value},
                    {value},
                    binaryTemplate(multiplyFu, multiplyOp)},
       FuDescriptor{subtractFu,
                    {value, value},
                    {value},
                    binaryTemplate(subtractFu, subtractOp)},
       FuDescriptor{finalAddFu,
                    {value, value},
                    {value},
                    binaryTemplate(finalAddFu, finalAddOp)}},
      {FabricOpDescriptor{xoriOp,
                          ::fabric::ImplementationFamilyId::ScalarIntegerLogic,
                          integerCapability(16),
                          {value, value},
                          {value}},
       FabricOpDescriptor{preAddOp,
                          ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                          integerCapability(16),
                          {value, value},
                          {value}},
       FabricOpDescriptor{
           multiplyOp,
           ::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
           integerCapability(16),
           {value, value},
           {value}},
       FabricOpDescriptor{subtractOp,
                          ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                          integerCapability(16),
                          {value, value},
                          {value}},
       FabricOpDescriptor{finalAddOp,
                          ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                          integerCapability(16),
                          {value, value},
                          {value}}},
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
       MemorySemanticEncodingDescriptor{groupedEncoding,
                                        implementation,
                                        {loadTemplate, storeTemplate},
                                        {doneToControl}},
       MemorySemanticEncodingDescriptor{
           otherStoreEncoding, otherImplementation, {otherStoreTemplate}, {}}},
      {}};
  for (std::size_t index = 0; index < fabric.functionalUnits.size(); ++index) {
    fabric.computeOccurrences.push_back(makeSpatialComputeOccurrence(
        fabricId, ComputeOccurrenceId(1000 + index),
        fabric.functionalUnits[index], 2000 + index * 100));
  }
  auto memoryOccurrence = [&](MemoryOccurrenceId occurrence,
                              std::uint64_t endpointBase) {
    constexpr std::uint32_t unbounded =
        std::numeric_limits<std::uint32_t>::max();
    MemoryOccurrenceDescriptor result{
        occurrence, MemoryImplementationRef{fabricId, implementation}, {}, {}};
    auto addPort = [&](MemoryOperationPortTemplateId operation,
                       std::uint32_t portIndex,
                       const MemoryOperationPortDescriptor &descriptor) {
      const MemoryEndpointId endpoint(endpointBase + result.endpoints.size());
      result.endpoints.push_back({endpoint,
                                  descriptor.direction,
                                  descriptor.port.kind,
                                  unbounded,
                                  0,
                                  {descriptor.port.type},
                                  descriptor.port.role,
                                  ::fabric::DataPathKind::Bits});
      result.localArcs.push_back(
          {MemoryOperationPortRef{
               MemoryOperationPortTemplateRef{fabricId, operation}, portIndex},
           MemoryEndpointRef{fabricId, endpoint}, unbounded, unbounded});
    };
    addPort(loadTemplate, 0, loadPorts[0]);
    addPort(loadTemplate, 1, loadPorts[1]);
    addPort(loadTemplate, 2, loadPorts[2]);
    addPort(storeTemplate, 0, storePorts[0]);
    addPort(storeTemplate, 1, storePorts[1]);
    addPort(storeTemplate, 3, storePorts[3]);
    return result;
  };
  fabric.memoryOccurrences = {
      memoryOccurrence(MemoryOccurrenceId(3000), 30000),
      memoryOccurrence(MemoryOccurrenceId(4000), 40000)};
  auto actorRef = [&](ActorId actor) { return ActorRef{dataflowId, actor}; };
  auto actorPort = [&](ActorId actor, PortDirection direction,
                       std::uint32_t index) {
    return ActorPortRef{actorRef(actor), direction, index};
  };
  auto memoryPort = [&](MemoryOperationPortTemplateId operation,
                        std::uint32_t index) {
    return MemoryOperationPortRef{
        MemoryOperationPortTemplateRef{fabricId, operation}, index};
  };
  auto singletonRealization =
      [&](std::uint64_t id, ActorId actor,
          ::loom::fabric::FabricFuTemplateRef fu,
          ::loom::fabric::FabricFuTemplateNodeRef operation) {
        return ComputeRealizationDraft{
            ComputeRealizationId(id),
            ::loom::fabric::FabricFuCapabilityTemplateRef{fu, 0},
            {{actorRef(actor), operation, {0, 1}, {0}}},
            {{actorPort(actor, PortDirection::Input, 0),
              fuPort(fu, PortDirection::Input, 0)},
             {actorPort(actor, PortDirection::Input, 1),
              fuPort(fu, PortDirection::Input, 1)},
             {actorPort(actor, PortDirection::Output, 0),
              fuPort(fu, PortDirection::Output, 0)}}};
      };
  ComputeRealizationDraft pairRealization{
      ComputeRealizationId(50),
      ::loom::fabric::FabricFuCapabilityTemplateRef{pairFu, 0},
      {{actorRef(xoriActor), xoriOp, {0, 1}, {0}},
       {actorRef(preAddActor), preAddOp, {0, 1}, {0}}},
      {{actorPort(xoriActor, PortDirection::Input, 0),
        fuPort(pairFu, PortDirection::Input, 0)},
       {actorPort(xoriActor, PortDirection::Input, 1),
        fuPort(pairFu, PortDirection::Input, 1)},
       {actorPort(preAddActor, PortDirection::Input, 1),
        fuPort(pairFu, PortDirection::Input, 2)},
       {actorPort(preAddActor, PortDirection::Output, 0),
        fuPort(pairFu, PortDirection::Output, 0)}}};
  ComputeRealizationDraft multiplyRealization =
      singletonRealization(51, multiplyActor, multiplyFu, multiplyOp);
  ComputeRealizationDraft subtractRealization =
      singletonRealization(52, subtractActor, subtractFu, subtractOp);
  ComputeRealizationDraft finalAddRealization =
      singletonRealization(53, finalAddActor, finalAddFu, finalAddOp);
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
      MappingDraftHeader{dataflowId, fabricId},
      {GraphRef{dataflowId, graph}},
      {std::move(pairRealization), std::move(multiplyRealization),
       std::move(subtractRealization), std::move(finalAddRealization)},
      {std::move(loadRealization), std::move(storeRealization)}};
  return TestCase{std::move(dataflow), std::move(fabric), artifact(4),
                  std::move(mapping)};
}
void selectInternalMemoryGraph(TestCase &testCase) {
  const ArtifactIdentity &dataflowId = testCase.dataflow.identity;
  const ArtifactIdentity &fabricId = testCase.fabric.identity;
  const MemoryRealizationDraft &load = testCase.mapping.memoryRealizations[0];
  const MemoryRealizationDraft &store = testCase.mapping.memoryRealizations[1];
  MemoryRealizationDraft grouped{
      MemoryRealizationId(60),
      {load.actors.front(), store.actors.front()},
      {load.actorToOperations.front(), store.actorToOperations.front()},
      {LogicalMemoryRootRef{dataflowId, LogicalMemoryRootId(20)}},
      MemorySemanticEncodingRef{fabricId, MemorySemanticEncodingId(43)},
      {load.boundaryPorts[0], load.boundaryPorts[1], load.boundaryPorts[2],
       store.boundaryPorts[0], store.boundaryPorts[1], store.boundaryPorts[3]},
      {},
      {{DataflowEdgeRef{
            dataflowId,
            DataflowEdge{ActorPort{ActorId(2), PortDirection::Output, 1},
                         ActorPort{ActorId(8), PortDirection::Input, 2}}},
        MemoryInternalConnectionRef{fabricId,
                                    MemoryInternalConnectionId(39)}}}};
  testCase.mapping.memoryRealizations = {std::move(grouped)};
}

} // namespace loom::mapping::test
