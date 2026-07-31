#include "MappingCoreTestSupport.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"

namespace loom::mapping::test {
namespace {

TestCase makePointerGepCase() {
  const ArtifactIdentity dataflowId = artifact(31);
  const ArtifactIdentity fabricId = artifact(32);
  const GraphId graph(1);
  const ActorId actor(2);
  const auto pointerPort = port(PortKind::Value, type(1), 64);
  const auto offsetPort = port(PortKind::Value, type(2), 64);

  static mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  context.loadDialect<mlir::LLVM::LLVMDialect>();
  mlir::Type pointer = mlir::LLVM::LLVMPointerType::get(&context);
  auto semantics = ::dataflow::CanonicalActorSchemaProjection{
      ::dataflow::OperationSchemaId::LLVMGetElementPtr,
      mlir::FunctionType::get(
          &context, {pointer, mlir::IntegerType::get(&context, 64)}, {pointer}),
      ::dataflow::GetElementPtrPayload{mlir::IntegerType::get(&context, 32),
                                       {mlir::LLVM::GEPOp::kDynamicIndex},
                                       mlir::LLVM::GEPNoWrapFlags::none}};
  DataflowProgramView dataflow{
      dataflowId,
      64,
      {GraphDescriptor{graph, {pointerPort, offsetPort}, {pointerPort}}},
      {ActorDescriptor{actor,
                       graph,
                       std::move(semantics),
                       {pointerPort, offsetPort},
                       {pointerPort},
                       std::nullopt}},
      {DataflowEdge{GraphPort{graph, PortDirection::Input, 0},
                    ActorPort{actor, PortDirection::Input, 0}},
       DataflowEdge{GraphPort{graph, PortDirection::Input, 1},
                    ActorPort{actor, PortDirection::Input, 1}},
       DataflowEdge{ActorPort{actor, PortDirection::Output, 0},
                    GraphPort{graph, PortDirection::Output, 0}}},
      {},
      {::loom::PointerLayout{0, 64, 64,
                             ::loom::PointerLayoutKind::StableIntegral}}};

  const ::loom::fabric::FabricFuTemplateRef fu(10);
  const ::loom::fabric::FabricFuTemplateNodeRef operation{
      ::loom::fabric::FabricFuNodeKind::Op, fu, 0};
  using Direction = ::loom::fabric::FabricPortDirection;
  using Endpoint = ::loom::fabric::FabricFuCapabilityTemplateEndpointRef;
  auto boundary = [&](Direction direction, std::uint64_t ordinal) {
    return Endpoint::boundaryPort({fu, direction, ordinal});
  };
  auto node = [&](Direction direction, std::uint64_t ordinal) {
    return Endpoint::nodePort({operation, direction, ordinal});
  };
  auto templates = llvm::cantFail(
      ::loom::fabric::normalizeFabricFuCapabilityTemplateInventory(
          {::loom::fabric::FabricFuCapabilityTemplateRecord{
              {operation},
              {{boundary(Direction::Input, 0), node(Direction::Input, 0)},
               {boundary(Direction::Input, 1), node(Direction::Input, 1)},
               {node(Direction::Output, 0),
                boundary(Direction::Output, 0)}}}}));
  FuDescriptor fuDescriptor{
      fu, {pointerPort, offsetPort}, {pointerPort}, std::move(templates)};
  FabricHardwareView fabric{
      fabricId,
      {fuDescriptor},
      {FabricOpDescriptor{
          operation,
          ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
          ::fabric::ScalarIntegerParams{
              ::fabric::IntegerWidthSet::get({::fabric::IntegerWidth::I64}),
              ::fabric::PointerFormatRelation::get(
                  {{0, 64, 64, ::loom::PointerLayoutKind::StableIntegral}})},
          {pointerPort, offsetPort},
          {pointerPort},
          {::dataflow::OperationSchemaId::LLVMGetElementPtr}}},
      {},
      {},
      {},
      {},
      {},
      {},
      {},
      {},
      {},
      {}};
  fabric.computeOccurrences.push_back(makeSpatialComputeOccurrence(
      fabricId, ComputeOccurrenceId(100), fabric.functionalUnits.front(), 200));

  TechMappingDraft mapping{
      MappingDraftHeader{dataflowId, fabricId},
      {GraphRef{dataflowId, graph}},
      {ComputeRealizationDraft{
          ComputeRealizationId(20),
          ::loom::fabric::FabricFuCapabilityTemplateRef{fu, 0},
          {{ActorRef{dataflowId, actor}, operation, {0, 1}, {0}}},
          {{ActorPortRef{ActorRef{dataflowId, actor}, PortDirection::Input, 0},
            {fu, Direction::Input, 0}},
           {ActorPortRef{ActorRef{dataflowId, actor}, PortDirection::Input, 1},
            {fu, Direction::Input, 1}},
           {ActorPortRef{ActorRef{dataflowId, actor}, PortDirection::Output, 0},
            {fu, Direction::Output, 0}}}}},
      {}};
  return TestCase{std::move(dataflow), std::move(fabric), artifact(33),
                  std::move(mapping)};
}

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

void rejectsDisabledConcreteOperationMember() {
  TestCase testCase = makeValidCase();
  testCase.fabric.operations.back().enabledOperationSchemas = {
      ::dataflow::OperationSchemaId::ArithSubI};
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

void requiresExactPointerLayoutForGep() {
  TestCase exact = makePointerGepCase();
  validateCase(__func__, exact);

  TestCase missing = makePointerGepCase();
  missing.dataflow.pointerLayouts.clear();
  expectMapError(__func__, missing, MappingErrorCode::InvalidPortConnection);

  TestCase mismatched = makePointerGepCase();
  mismatched.dataflow.pointerLayouts.front().representationBits = 32;
  mismatched.dataflow.pointerLayouts.front().addressBits = 32;
  expectMapError(__func__, mismatched,
                 MappingErrorCode::CapabilityTemplateMismatch);
}

} // namespace

void runCapabilityTemplateTests() {
  acceptsExactCapabilityTemplateReference();
  rejectsInvalidCapabilityTemplateReference();
  rejectsInactiveOperationNode();
  rejectsDisabledConcreteOperationMember();
  rejectsIncompleteOrNonInjectivePortMap();
  acceptsWiderPhysicalComputePorts();
  rejectsUndersizedPhysicalComputePorts();
  requiresExactPointerLayoutForGep();
}

} // namespace loom::mapping::test
