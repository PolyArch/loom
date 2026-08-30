#include "RootCompleteSpatialPnrTestSupport.h"

#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "ADG/FuLibrary.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Frontend/IR/LoomOps.h"
#include "PnR/MappingObjective.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

namespace loom::test {
namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "root-complete Spatial PnR fixture failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

} // namespace

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, loom::LoomDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @sync(%start: none, %value: i32) -> i32
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %value: i32) ctrl (%ctrl: none) {
    %result, %done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %value = arith.constant 7 : i32
    %thread = dataflow.thread.launch @worker(%value)
        : (i32) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildRootCompleteSpatialDataflow(mlir::MLIRContext &context) {
  return buildDataflow(context);
}

dataflow::CanonicalDataflowArtifact
buildAlternateDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @sync(%start: none, %value: i32) -> i32
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %value: i32) ctrl (%ctrl: none) {
    %result, %done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %value = arith.constant 8 : i32
    %thread = dataflow.thread.launch @worker(%value)
        : (i32) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse alternate Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildAlternateRootCompleteSpatialDataflow(mlir::MLIRContext &context) {
  return buildAlternateDataflow(context);
}

dataflow::CanonicalDataflowArtifact
buildVectorDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @add(%start: none, %value: vector<4xi32>)
      -> vector<4xi32>
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %sum = arith.addi %value, %value : vector<4xi32>
    %retired:2 = dataflow.sync %start, %sum
        : (none, vector<4xi32>) -> (none, vector<4xi32>)
    dataflow.graph.return values(%retired#1 : vector<4xi32>) streams()
        memories() complete(%retired#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %value: vector<4xi32>)
      ctrl (%ctrl: none) {
    %result, %done = dataflow.graph.launch @add deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, vector<4xi32>) -> (vector<4xi32>, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %value = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
    %thread = dataflow.thread.launch @worker(%value)
        : (vector<4xi32>) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse vector Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildVectorRootCompleteSpatialDataflow(mlir::MLIRContext &context) {
  return buildVectorDataflow(context);
}

fabric::FinalizedFabricRoot
buildAlternativeTechSpatialCore(ArtifactStore &store) {
  constexpr std::uint32_t payloadWidth = 128;
  const adg::PortType payloadType = take(adg::PortType::bits(payloadWidth));
  const std::vector<adg::PortType> types(8, payloadType);
  adg::DesignBuilder builder(store);
  auto spatial =
      take(builder.createSpatialCore("alternative-sync", types, types));
  std::vector<adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    spatialInputs.push_back(take(spatial.input(ordinal)));
  auto pe =
      take(spatial.addPe(spatialInputs, adg::PeSpec::spatial(types, types)));
  std::vector<adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    peInputs.push_back(take(pe.input(ordinal)));
  for (std::uint32_t ordinal = 0; ordinal != 2; ++ordinal) {
    const std::size_t laneCount = ordinal == 0 ? 4 : 8;
    const std::vector<adg::PortType> fuTypes(laneCount, payloadType);
    auto fu = take(
        pe.addFu(llvm::ArrayRef<adg::PeValue>(peInputs).take_front(laneCount),
                 adg::FuSpec{fuTypes, fuTypes}));
    std::vector<adg::FuValue> fuInputs;
    for (std::size_t input = 0; input != fuTypes.size(); ++input)
      fuInputs.push_back(take(fu.input(input)));
    auto operation = take(fu.addOperation(
        fuInputs, adg::OperationCapabilitySpec{
                      ::fabric::ImplementationFamilyId::TokenSync,
                      ::fabric::RoutedTokenParams{
                          payloadWidth, static_cast<std::uint32_t>(laneCount)},
                      {::dataflow::OperationSchemaId::DataflowSync},
                      fuTypes,
                      ::fabric::oneCycleElasticOperationResourceContract()}));
    requireSuccess(fu.addCapabilityTemplate(
        adg::FuCapabilityTemplateSpec{{operation}, {}}));
    std::vector<adg::FuValue> outputs;
    for (std::size_t output = 0; output != fuTypes.size(); ++output)
      outputs.push_back(take(operation.output(output)));
    requireSuccess(fu.close(outputs));
  }
  requireSuccess(pe.close());
  std::vector<adg::SpatialValue> outputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    outputs.push_back(take(pe.output(ordinal)));
  requireSuccess(spatial.close(outputs));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("alternative Tech fixture did not publish one Fabric root");
  return design.roots().front();
}

ResolvedConfig buildSpatialResolvedConfig() {
  ResolvedObjectiveCatalogs catalogs;
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  catalogs.dimensions = {
      {ResolvedMappingViolationObjectiveSource{
           ResolvedPnrViolationKind::UnroutedObligation},
       ResolvedObjectiveDirection::Minimize, resolvedObjectiveInteger(0),
       resolvedObjectiveInteger(1), 0, maximum},
      {ResolvedMappingViolationObjectiveSource{
           ResolvedPnrViolationKind::CapacityOveruse},
       ResolvedObjectiveDirection::Minimize, resolvedObjectiveInteger(0),
       resolvedObjectiveInteger(1), 0, maximum},
      {ResolvedMappingMeasureObjectiveSource{static_cast<std::uint32_t>(
           pnr::MappingMeasureKind::TotalSelectedTraversalClaim)},
       ResolvedObjectiveDirection::Minimize, resolvedObjectiveInteger(0),
       resolvedObjectiveInteger(1), 0, maximum},
  };
  catalogs.weightedLevels = {
      {{{0, 1}, {1, 1}, {2, 1}}},
  };
  catalogs.totalOrderings = {{{0}}};

  ResolvedConfig resolved = defaultResolvedConfig();
  resolved.dse.objectiveCatalogs = std::move(catalogs);
  resolved.dse.spatialPnr.temporaryViolations.admitted = {
      ResolvedPnrViolationKind::UnroutedObligation,
      ResolvedPnrViolationKind::CapacityOveruse,
  };
  resolved.dse.spatialPnr.objectiveSelection = {0, 0};
  auto &search = resolved.dse.spatialPnr.search;
  search.initializer.seedAttemptCount = 2;
  search.actionProposal = {0, 1, 0};
  search.annealing.calibrationProposalCount = 1;
  search.annealing.fallbackTemperature = 1;
  search.annealing.minimumTemperature = 1;
  search.annealing.coolingRatio = {1, 2};
  search.annealing.proposalsPerLevelBase = 1;
  search.annealing.proposalsPerMovableDecision = 0;
  search.exactRepair = {ResolvedPnrExactRepairKind::Disabled, 0, 0};
  return resolved;
}

pnr::ResolvedPnrConfigView buildSpatialConfig() {
  return take(
      pnr::projectResolvedSpatialPnrConfigView(buildSpatialResolvedConfig()));
}

ResolvedConfig buildSingleCandidateSpatialResolvedConfig() {
  ResolvedConfig resolved = buildSpatialResolvedConfig();
  resolved.dse.spatialPnr.search.initializer.seedAttemptCount = 1;
  return resolved;
}

pnr::ResolvedPnrConfigView buildSingleCandidateSpatialConfig() {
  return take(pnr::projectResolvedSpatialPnrConfigView(
      buildSingleCandidateSpatialResolvedConfig()));
}

pnr::ResolvedPnrConfigView buildFeedbackSpatialConfig() {
  ResolvedConfig resolved = buildSpatialResolvedConfig();
  resolved.dse.spatialPnr.search.initializer.seedAttemptCount = 8;
  resolved.dse.spatialPnr.search.routing.negotiationIterationLimit = 8;
  resolved.dse.spatialPnr.search.routing.negotiation = ResolvedPathFinderPolicy{
      ResolvedPathFinderPriceKernel::Additive, 1, {3, 2}, 1};
  resolved.dse.spatialPnr.search.actionProposal = {3, 3, 2};
  resolved.dse.spatialPnr.search.annealing.calibrationProposalCount = 16;
  resolved.dse.spatialPnr.search.annealing.proposalsPerLevelBase = 64;
  resolved.dse.spatialPnr.search.annealing.proposalsPerMovableDecision = 4;
  return take(pnr::projectResolvedSpatialPnrConfigView(resolved));
}

fabric::FinalizedFabricRoot buildSpatialCore(ArtifactStore &store,
                                             std::uint32_t payloadWidth) {
  const adg::PortType payloadType = take(adg::PortType::bits(payloadWidth));
  const std::vector<adg::PortType> types(4, payloadType);
  adg::DesignBuilder builder(store);
  auto spatial = take(builder.createSpatialCore("sync", types, types));
  std::vector<adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    spatialInputs.push_back(take(spatial.input(ordinal)));
  auto pe =
      take(spatial.addPe(spatialInputs, adg::PeSpec::spatial(types, types)));
  std::vector<adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    peInputs.push_back(take(pe.input(ordinal)));
  auto fu = take(pe.addFu(peInputs, adg::FuSpec{types, types}));
  std::vector<adg::FuValue> fuInputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    fuInputs.push_back(take(fu.input(ordinal)));
  auto operation = take(fu.addOperation(
      fuInputs, adg::OperationCapabilitySpec{
                    ::fabric::ImplementationFamilyId::TokenSync,
                    ::fabric::RoutedTokenParams{payloadWidth, 4},
                    {::dataflow::OperationSchemaId::DataflowSync},
                    types,
                    ::fabric::oneCycleElasticOperationResourceContract()}));
  requireSuccess(
      fu.addCapabilityTemplate(adg::FuCapabilityTemplateSpec{{operation}, {}}));
  std::vector<adg::FuValue> fuOutputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    fuOutputs.push_back(take(operation.output(ordinal)));
  requireSuccess(fu.close(fuOutputs));
  requireSuccess(pe.close());
  std::vector<adg::SpatialValue> outputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    outputs.push_back(take(pe.output(ordinal)));
  requireSuccess(spatial.close(outputs));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("SpatialCore fixture did not publish exactly one Fabric root");
  return design.roots().front();
}

fabric::FinalizedFabricRoot
buildLineageSpatialCore(ArtifactStore &store, std::uint32_t payloadWidth) {
  constexpr std::size_t vectorUnitCount = 8;
  constexpr std::size_t tokenUnitCount = 2;
  const adg::PortType payloadType = take(adg::PortType::bits(payloadWidth));
  const std::vector<adg::PortType> types(4, payloadType);
  const std::vector<adg::PortType> tokenInputTypes(5, payloadType);
  adg::DesignBuilder builder(store);

  auto vectorUnit =
      take(builder.createSpatialCore("lineage-vector-unit", types, types));
  std::vector<adg::SpatialValue> vectorUnitInputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    vectorUnitInputs.push_back(take(vectorUnit.input(ordinal)));
  auto vectorPe = take(vectorUnit.addPe(
      vectorUnitInputs, adg::PeSpec::spatial(types, types)));
  std::vector<adg::PeValue> vectorInputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    vectorInputs.push_back(take(vectorPe.input(ordinal)));
  requireSuccess(adg::addVectorComputeFu(vectorPe, vectorInputs,
                                         {payloadWidth, payloadWidth}));
  const ::fabric::IntegerWidthSet integerWidths =
      ::fabric::IntegerWidthSet::get(
          {::fabric::IntegerWidth::I8, ::fabric::IntegerWidth::I16,
           ::fabric::IntegerWidth::I32, ::fabric::IntegerWidth::I64});
  const ::fabric::FloatFormatSet floatFormats = ::fabric::FloatFormatSet::get(
      {::fabric::FloatFormat::F16, ::fabric::FloatFormat::BF16,
       ::fabric::FloatFormat::F32, ::fabric::FloatFormat::F64});
  const adg::VectorStructuralFuParameters structural{
      payloadWidth, payloadWidth, 64,
      ::fabric::FixedVectorSliceAlignMergeParams{
          integerWidths, floatFormats, payloadWidth, payloadWidth, 0,
          ::fabric::ResolvedIndexWidthSet::get({})},
      ::fabric::FixedVectorShuffleParams{integerWidths, floatFormats,
                                         payloadWidth, payloadWidth, 32, 8, 4}};
  requireSuccess(adg::addVectorStructuralFu(
      vectorPe, llvm::ArrayRef<adg::PeValue>(vectorInputs).take_front(2),
      structural));
  requireSuccess(vectorPe.close());
  std::vector<adg::SpatialValue> vectorUnitOutputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    vectorUnitOutputs.push_back(take(vectorPe.output(ordinal)));
  requireSuccess(vectorUnit.close(vectorUnitOutputs));
  const auto vectorClocks =
      take(vectorUnit.domainSlots(fabric::FabricClockResetKind::Clock));
  const auto vectorResets =
      take(vectorUnit.domainSlots(fabric::FabricClockResetKind::Reset));

  auto tokenUnit = take(
      builder.createSpatialCore("lineage-token-unit", tokenInputTypes, types));
  std::vector<adg::SpatialValue> tokenUnitInputs;
  for (std::size_t ordinal = 0; ordinal != tokenInputTypes.size(); ++ordinal)
    tokenUnitInputs.push_back(take(tokenUnit.input(ordinal)));
  auto tokenPe = take(tokenUnit.addPe(
      tokenUnitInputs, adg::PeSpec::spatial(tokenInputTypes, types)));
  std::vector<adg::PeValue> tokenInputs;
  for (std::size_t ordinal = 0; ordinal != tokenInputTypes.size(); ++ordinal)
    tokenInputs.push_back(take(tokenPe.input(ordinal)));
  requireSuccess(adg::addTokenControlFu(
      tokenPe, tokenInputs, {payloadWidth, std::min(payloadWidth, 64U)}));
  requireSuccess(tokenPe.close());
  std::vector<adg::SpatialValue> tokenUnitOutputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    tokenUnitOutputs.push_back(take(tokenPe.output(ordinal)));
  requireSuccess(tokenUnit.close(tokenUnitOutputs));
  const auto tokenClocks =
      take(tokenUnit.domainSlots(fabric::FabricClockResetKind::Clock));
  const auto tokenResets =
      take(tokenUnit.domainSlots(fabric::FabricClockResetKind::Reset));

  auto spatial = take(builder.createSpatialCore("lineage-sync", types, types));
  const auto spatialClock =
      take(spatial.declareDomainSlot(fabric::FabricClockResetKind::Clock));
  const auto spatialReset =
      take(spatial.declareDomainSlot(fabric::FabricClockResetKind::Reset));
  std::vector<adg::MeshCellAttachmentSpec> attachments = {
      {0, 0, {payloadType, payloadType}, {payloadType, payloadType}},
      {0, 1, {payloadType, payloadType}, {payloadType, payloadType}}};
  for (std::size_t ordinal = 0; ordinal != vectorUnitCount; ++ordinal)
    attachments.push_back(
        {1, ordinal < vectorUnitCount / 2 ? 0U : 1U, types, types});
  for (std::size_t ordinal = 0; ordinal != tokenUnitCount; ++ordinal)
    attachments.push_back({1, 1, tokenInputTypes, types});
  auto network = take(
      spatial.addMeshSwitchNetwork(take(adg::MeshSwitchNetworkSpec::spatial(
          2, 2, 2, payloadType, 1, ::fabric::FifoQueueDiscipline::StrictFifo,
          std::move(attachments)))));

  auto upperBoundary = take(network.attachment(0));
  auto lowerBoundary = take(network.attachment(1));
  requireSuccess(upperBoundary.connectOutputs(
      {take(spatial.input(0)), take(spatial.input(1))}));
  requireSuccess(lowerBoundary.connectOutputs(
      {take(spatial.input(2)), take(spatial.input(3))}));

  for (std::size_t ordinal = 0; ordinal != vectorUnitCount; ++ordinal) {
    auto attachment = take(network.attachment(2 + ordinal));
    auto outputs = take(spatial.instantiate(
        vectorUnit, attachment.inputs(),
        {{vectorClocks.front(), spatialClock},
         {vectorResets.front(), spatialReset}}));
    requireSuccess(attachment.connectOutputs(outputs));
  }
  for (std::size_t ordinal = 0; ordinal != tokenUnitCount; ++ordinal) {
    auto tokenControl =
        take(network.attachment(2 + vectorUnitCount + ordinal));
    auto tokenOutputs = take(spatial.instantiate(
        tokenUnit, tokenControl.inputs(),
        {{tokenClocks.front(), spatialClock},
         {tokenResets.front(), spatialReset}}));
    requireSuccess(tokenControl.connectOutputs(tokenOutputs));
  }

  for (const auto &member : network.domainMembers()) {
    requireSuccess(spatial.assignDomainSlot(member, spatialClock));
    requireSuccess(spatial.assignDomainSlot(member, spatialReset));
  }
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal) {
    const auto input = take(spatial.inputDomainMember(ordinal));
    const auto output = take(spatial.outputDomainMember(ordinal));
    requireSuccess(spatial.assignDomainSlot(input, spatialClock));
    requireSuccess(spatial.assignDomainSlot(input, spatialReset));
    requireSuccess(spatial.assignDomainSlot(output, spatialClock));
    requireSuccess(spatial.assignDomainSlot(output, spatialReset));
  }

  std::vector<adg::SpatialValue> outputs(upperBoundary.inputs().begin(),
                                         upperBoundary.inputs().end());
  outputs.insert(outputs.end(), lowerBoundary.inputs().begin(),
                 lowerBoundary.inputs().end());
  requireSuccess(spatial.close(outputs));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 3)
    fail("lineage SpatialCore fixture did not publish its unit and top roots");
  auto root = design.roots().back();
  if (root.view().peOccurrences().size() !=
          vectorUnitCount + tokenUnitCount ||
      root.view().fifoOccurrences().size() != 16 ||
      root.view().switchOccurrences().size() < 12)
    fail("lineage SpatialCore lost its finite multi-hop mesh topology");
  return root;
}

fabric::FinalizedFabricRoot
buildFeedbackPruningSpatialCore(ArtifactStore &store) {
  constexpr std::uint32_t payloadWidth = 256;
  const adg::PortType payloadType = take(adg::PortType::bits(payloadWidth));
  const std::vector<adg::PortType> payloadTypes(4, payloadType);
  const std::vector<adg::PortType> tokenInputTypes(5, payloadType);
  const std::vector<adg::PortType> scalarInputTypes(2, payloadType);
  const std::vector<adg::PortType> scalarOutputTypes(1, payloadType);

  adg::DesignBuilder builder(store);
  auto spatial = take(builder.createSpatialCore("feedback-pruning",
                                                payloadTypes, payloadTypes));
  auto network = take(
      spatial.addMeshSwitchNetwork(take(adg::MeshSwitchNetworkSpec::spatial(
          3, 2, 2, payloadType, 1, ::fabric::FifoQueueDiscipline::StrictFifo,
          {{0, 0, {payloadType, payloadType}, {payloadType, payloadType}},
           {0, 1, {payloadType, payloadType}, {payloadType, payloadType}},
           {1, 0, payloadTypes, payloadTypes},
           {1, 1, tokenInputTypes, payloadTypes},
           {2, 0, scalarInputTypes, scalarOutputTypes}}))));

  auto upperBoundary = take(network.attachment(0));
  auto lowerBoundary = take(network.attachment(1));
  auto vectorCompute = take(network.attachment(2));
  auto tokenControl = take(network.attachment(3));
  auto scalarCompute = take(network.attachment(4));
  requireSuccess(upperBoundary.connectOutputs(
      {take(spatial.input(0)), take(spatial.input(1))}));
  requireSuccess(lowerBoundary.connectOutputs(
      {take(spatial.input(2)), take(spatial.input(3))}));

  auto vectorPe =
      take(spatial.addPe(vectorCompute.inputs(),
                         adg::PeSpec::spatial(payloadTypes, payloadTypes)));
  std::vector<adg::PeValue> vectorInputs;
  for (std::size_t ordinal = 0; ordinal != payloadTypes.size(); ++ordinal)
    vectorInputs.push_back(take(vectorPe.input(ordinal)));
  requireSuccess(adg::addVectorComputeFu(vectorPe, vectorInputs,
                                         {payloadWidth, payloadWidth}));
  const ::fabric::IntegerWidthSet integerWidths =
      ::fabric::IntegerWidthSet::get(
          {::fabric::IntegerWidth::I8, ::fabric::IntegerWidth::I16,
           ::fabric::IntegerWidth::I32, ::fabric::IntegerWidth::I64});
  const ::fabric::FloatFormatSet floatFormats = ::fabric::FloatFormatSet::get(
      {::fabric::FloatFormat::F16, ::fabric::FloatFormat::BF16,
       ::fabric::FloatFormat::F32, ::fabric::FloatFormat::F64});
  const adg::VectorStructuralFuParameters structural{
      payloadWidth, payloadWidth, 64,
      ::fabric::FixedVectorSliceAlignMergeParams{
          integerWidths, floatFormats, payloadWidth, payloadWidth, 0,
          ::fabric::ResolvedIndexWidthSet::get({})},
      ::fabric::FixedVectorShuffleParams{integerWidths, floatFormats,
                                         payloadWidth, payloadWidth, 32, 8, 4}};
  requireSuccess(adg::addVectorStructuralFu(
      vectorPe, llvm::ArrayRef<adg::PeValue>(vectorInputs).take_front(2),
      structural));
  requireSuccess(vectorPe.close());
  std::vector<adg::SpatialValue> vectorOutputs;
  for (std::size_t ordinal = 0; ordinal != payloadTypes.size(); ++ordinal)
    vectorOutputs.push_back(take(vectorPe.output(ordinal)));
  requireSuccess(vectorCompute.connectOutputs(vectorOutputs));

  auto tokenPe =
      take(spatial.addPe(tokenControl.inputs(),
                         adg::PeSpec::spatial(tokenInputTypes, payloadTypes)));
  std::vector<adg::PeValue> tokenInputs;
  for (std::size_t ordinal = 0; ordinal != tokenInputTypes.size(); ++ordinal)
    tokenInputs.push_back(take(tokenPe.input(ordinal)));
  requireSuccess(
      adg::addTokenControlFu(tokenPe, tokenInputs, {payloadWidth, 64}));
  requireSuccess(tokenPe.close());
  std::vector<adg::SpatialValue> tokenOutputs;
  for (std::size_t ordinal = 0; ordinal != payloadTypes.size(); ++ordinal)
    tokenOutputs.push_back(take(tokenPe.output(ordinal)));
  requireSuccess(tokenControl.connectOutputs(tokenOutputs));

  auto scalarPe = take(
      spatial.addPe(scalarCompute.inputs(),
                    adg::PeSpec::spatial(scalarInputTypes, scalarOutputTypes)));
  std::vector<adg::PeValue> scalarInputs;
  for (std::size_t ordinal = 0; ordinal != scalarInputTypes.size(); ++ordinal)
    scalarInputs.push_back(take(scalarPe.input(ordinal)));
  auto scalarFu = take(scalarPe.addFu(
      scalarInputs, adg::FuSpec{scalarInputTypes, scalarOutputTypes}));
  std::vector<adg::FuValue> scalarFuInputs;
  for (std::size_t ordinal = 0; ordinal != scalarInputTypes.size(); ++ordinal)
    scalarFuInputs.push_back(take(scalarFu.input(ordinal)));
  auto scalarAdd = take(scalarFu.addOperation(
      scalarFuInputs,
      adg::OperationCapabilitySpec{
          ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
          ::fabric::ScalarIntegerParams{integerWidths},
          {::dataflow::OperationSchemaId::ArithAddI},
          {payloadType},
          ::fabric::oneCycleElasticOperationResourceContract()}));
  requireSuccess(scalarFu.addCapabilityTemplate(
      adg::FuCapabilityTemplateSpec{{scalarAdd}, {}}));
  requireSuccess(scalarFu.close({take(scalarAdd.output(0))}));
  requireSuccess(scalarPe.close());
  requireSuccess(scalarCompute.connectOutputs({take(scalarPe.output(0))}));

  std::vector<adg::SpatialValue> outputs(upperBoundary.inputs().begin(),
                                         upperBoundary.inputs().end());
  outputs.insert(outputs.end(), lowerBoundary.inputs().begin(),
                 lowerBoundary.inputs().end());
  requireSuccess(spatial.close(outputs));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("feedback-pruning SpatialCore did not publish one Fabric root");
  auto root = design.roots().front();
  if (root.view().peOccurrences().size() != 3 ||
      root.view().switchOccurrences().size() < 18)
    fail("feedback-pruning SpatialCore lost its finite distributed topology");
  return root;
}

} // namespace loom::test
