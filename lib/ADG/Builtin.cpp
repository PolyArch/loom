#include "ADG/Builtin.h"

#include "ADG/FuLibrary.h"
#include "ADG/MemoryLibrary.h"
#include "CatalogCapabilities.h"

#include "Dataflow/IR/DataflowEnums.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace loom::adg {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "adg_builtin_invalid: " + message);
}

llvm::Expected<::fabric::ResourceContract>
singleRequesterResourceContract(std::uint32_t capacity = 1) {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {
      {::fabric::StateKey(0),
       {{::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(capacity),
         ::fabric::CapacityUnits(0)}}}};
  declaration.requesters = {::fabric::RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.timingContracts = {{::fabric::TimingContractKey(0), {0, 1}}};
  declaration.usePatterns = {
      {::fabric::UsePatternKey(0),
       ::fabric::RequesterKey(0),
       ::fabric::EligibilityKey(0),
       ::fabric::EventKey(0),
       ::fabric::EventKey(1),
       std::nullopt,
       ::fabric::TimingContractKey(0),
       {{::fabric::ClaimKey(0), ::fabric::StateKey(0),
         ::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1)}},
       {{{::fabric::ClaimKey(0)}}}}};
  return ::fabric::ResourceContract::create(declaration);
}

llvm::Expected<loom::fabric::InstructionCoreArchitecturalContract>
makeBuiltinInstructionCoreArchitecture() {
  loom::fabric::RiscVArchitectureDeclaration declaration;
  declaration.xlen = loom::fabric::RiscVXLen::X64;
  declaration.base = loom::fabric::RiscVBase::I;
  declaration.extensions = {loom::fabric::RiscVExtension::M,
                            loom::fabric::RiscVExtension::A,
                            loom::fabric::RiscVExtension::F,
                            loom::fabric::RiscVExtension::D,
                            loom::fabric::RiscVExtension::C,
                            loom::fabric::RiscVExtension::Zicsr,
                            loom::fabric::RiscVExtension::Zifencei};
  declaration.endianness = loom::fabric::InstructionEndianness::Little;
  declaration.physicalAddressWidthBits = 48;
  declaration.privilegeModes = {loom::fabric::PrivilegeMode::Machine};
  declaration.abiCapabilities = {loom::fabric::RiscVAbi::Lp64d};
  declaration.memoryOrdering = loom::fabric::RiscVMemoryOrdering::Rvwmo;
  declaration.syncScopes = {loom::fabric::InstructionSyncScope::Hart};
  declaration.codeModels = {loom::fabric::RiscVCodeModel::MediumAny};
  declaration.relocationModels = {loom::fabric::RelocationModel::Static};
  declaration.runtimeServices = {
      loom::fabric::InstructionRuntimeService::ThreadDispatch,
      loom::fabric::InstructionRuntimeService::SpatialLaunch};
  return loom::fabric::InstructionCoreArchitecturalContract::create(
      std::move(declaration));
}

llvm::Expected<loom::fabric::InstructionCoreMicroarchitecturalRealization>
inOrderMicroarchitecture() {
  auto resources = singleRequesterResourceContract();
  if (!resources)
    return resources.takeError();
  loom::fabric::InstructionCoreCommonDeclaration common{
      1,
      {{loom::fabric::InstructionOperationClass::IntegerAlu, 1, 1, 1},
       {loom::fabric::InstructionOperationClass::IntegerMultiply, 1, 3, 1},
       {loom::fabric::InstructionOperationClass::LoadStore, 1, 2, 1},
       {loom::fabric::InstructionOperationClass::FloatingPointAlu, 1, 3, 1},
       {loom::fabric::InstructionOperationClass::FloatingPointMultiply, 1, 4,
        1},
       {loom::fabric::InstructionOperationClass::FloatingPointDivide, 1, 12,
        12}},
      std::move(*resources)};
  loom::fabric::InOrderMicroarchitectureDeclaration pipeline{1, 1, 1, 1,
                                                             1, 1, 4, 2};
  return loom::fabric::InstructionCoreMicroarchitecturalRealization::
      createInOrder(std::move(common), pipeline);
}

llvm::Expected<loom::fabric::InstructionCoreMicroarchitecturalRealization>
outOfOrderMicroarchitecture() {
  auto resources = singleRequesterResourceContract();
  if (!resources)
    return resources.takeError();
  loom::fabric::InstructionCoreCommonDeclaration common{
      1,
      {{loom::fabric::InstructionOperationClass::IntegerAlu, 2, 1, 1},
       {loom::fabric::InstructionOperationClass::IntegerMultiply, 1, 3, 1},
       {loom::fabric::InstructionOperationClass::IntegerDivide, 1, 12, 12},
       {loom::fabric::InstructionOperationClass::LoadStore, 2, 2, 1},
       {loom::fabric::InstructionOperationClass::FloatingPointAlu, 2, 3, 1},
       {loom::fabric::InstructionOperationClass::FloatingPointMultiply, 1, 4,
        1},
       {loom::fabric::InstructionOperationClass::FloatingPointDivide, 1, 12,
        12}},
      std::move(*resources)};
  loom::fabric::OutOfOrderMicroarchitectureDeclaration pipeline{
      2, 2, 2, 2, 2, 2, 2, 32, 16, 8, 8, 64, 64, 64};
  return loom::fabric::InstructionCoreMicroarchitecturalRealization::
      createOutOfOrder(std::move(common), pipeline);
}

std::vector<bool> distributedSites(std::uint32_t siteCount,
                                   std::uint32_t occurrenceCount,
                                   std::uint32_t offset) {
  std::vector<bool> result(siteCount, false);
  for (std::uint32_t occurrence = 0; occurrence != occurrenceCount;
       ++occurrence) {
    const std::uint32_t site =
        ((static_cast<std::uint64_t>(occurrence) * siteCount) /
             occurrenceCount +
         offset) %
        siteCount;
    result[site] = true;
  }
  return result;
}

struct FuDistribution final {
  std::vector<bool> scalarAdd;
  std::vector<bool> mac;
  std::vector<bool> vectorCompute;
  std::vector<bool> loopControl;
  std::vector<bool> tokenControl;
  std::vector<bool> vectorAdapter;
  std::vector<bool> vectorStructural;
  std::vector<bool> specialMath;
  std::vector<std::optional<std::uint32_t>> loopOrdinal;
};

struct DistributedCellCursor final {
  std::uint64_t cell = 0;
  std::uint64_t cellCount = 0;
  std::uint64_t cellStride = 0;
  std::uint64_t remainder = 0;
  std::uint64_t remainderStride = 0;
  std::uint64_t attachmentCount = 0;
  std::uint64_t offset = 0;

  DistributedCellCursor(std::uint64_t cellCount, std::uint64_t attachmentCount,
                        std::uint64_t offset = 0)
      : cellCount(cellCount), cellStride(cellCount / attachmentCount),
        remainderStride(cellCount % attachmentCount),
        attachmentCount(attachmentCount), offset(offset) {}

  std::uint64_t next() {
    const std::uint64_t result = cell >= cellCount - offset
                                     ? cell - (cellCount - offset)
                                     : cell + offset;
    cell += cellStride;
    remainder += remainderStride;
    if (remainder >= attachmentCount) {
      ++cell;
      remainder -= attachmentCount;
    }
    return result;
  }
};

FuDistribution makeFuDistribution(std::uint32_t count,
                                  const BuiltinFuOccurrenceCounts &occurrences,
                                  std::uint32_t &nextLoopOrdinal) {
  FuDistribution distribution{
      distributedSites(count, occurrences.dedicatedScalarAdd, 6),
      distributedSites(count, occurrences.mac, 0),
      distributedSites(count, occurrences.vectorCompute, 1),
      distributedSites(count, occurrences.loopControl, 2),
      distributedSites(count, occurrences.tokenControl, 3),
      distributedSites(count, occurrences.vectorAdapter, 4),
      distributedSites(count, occurrences.vectorStructural, 5),
      distributedSites(count, occurrences.specialMath, 7),
      std::vector<std::optional<std::uint32_t>>(count)};
  for (std::uint32_t site = 0; site != count; ++site)
    if (distribution.loopControl[site])
      distribution.loopOrdinal[site] = nextLoopOrdinal++;
  return distribution;
}

llvm::Error addDedicatedScalarAddFu(PeBuilder &pe,
                                    llvm::ArrayRef<PeValue> inputs) {
  if (inputs.size() != 2)
    return invalid("dedicated scalar add FU requires two data inputs");
  auto bits64 = PortType::bits(64);
  if (!bits64)
    return bits64.takeError();
  auto bits128 = PortType::bits(128);
  if (!bits128)
    return bits128.takeError();
  auto fu = pe.addFu(inputs, FuSpec{{*bits64, *bits64}, {*bits128}});
  if (!fu)
    return fu.takeError();
  auto lhs = fu->input(0);
  if (!lhs)
    return lhs.takeError();
  auto rhs = fu->input(1);
  if (!rhs)
    return rhs.takeError();
  auto operation = fu->addOperation(
      {*lhs, *rhs},
      OperationCapabilitySpec{
          ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
          ::fabric::ScalarIntegerParams{detail::catalogOrdinaryIntegerWidths()},
          {::dataflow::OperationSchemaId::ArithAddI,
           ::dataflow::OperationSchemaId::ArithSubI},
          {*bits64},
          ::fabric::oneCycleElasticOperationResourceContract()});
  if (!operation)
    return operation.takeError();
  if (llvm::Error error =
          fu->addCapabilityTemplate(FuCapabilityTemplateSpec{{*operation}, {}}))
    return error;
  auto result = operation->output(0);
  if (!result)
    return result.takeError();
  return fu->close({*result});
}

VectorStructuralFuParameters builtinVectorStructuralParameters() {
  const ::fabric::IntegerWidthSet integerWidths =
      detail::catalogOrdinaryIntegerWidths();
  const ::fabric::FloatFormatSet floatFormats = detail::catalogFloatFormats();
  return {128, 128, 64,
          ::fabric::FixedVectorSliceAlignMergeParams{
              integerWidths, floatFormats, 128, 128, 3,
              ::fabric::ResolvedIndexWidthSet::get(
                  {::fabric::ResolvedIndexWidth::I32,
                   ::fabric::ResolvedIndexWidth::I64})},
          ::fabric::FixedVectorShuffleParams{integerWidths, floatFormats, 128,
                                             128, 128, 32, 16}};
}

llvm::Expected<MemoryAccessDomainParameters> builtinMemoryAccessDomain() {
  auto indexWidths =
      ::fabric::UnsignedDomain::fromCanonical({{32, 32}, {64, 64}});
  if (!indexWidths)
    return indexWidths.takeError();
  return MemoryAccessDomainParameters{128, 128, 16, std::move(*indexWidths)};
}

llvm::Expected<MemoryInterfaceParameters> builtinMemoryInterface() {
  auto accessDomain = builtinMemoryAccessDomain();
  if (!accessDomain)
    return accessDomain.takeError();
  return MemoryInterfaceParameters{std::move(*accessDomain), 128, 128};
}

llvm::Error addFuCatalog(PeBuilder &pe, std::uint32_t site,
                         const FuDistribution &distribution) {
  std::vector<PeValue> inputs;
  inputs.reserve(5);
  for (std::size_t ordinal = 0; ordinal != 5; ++ordinal) {
    auto input = pe.input(ordinal);
    if (!input)
      return input.takeError();
    inputs.push_back(*input);
  }
  if (llvm::Error error =
          addCoreAluFu(pe, {inputs[0], inputs[1], inputs[2]},
                       ::fabric::ResolvedIndexWidthSet::get(
                           {::fabric::ResolvedIndexWidth::I32,
                            ::fabric::ResolvedIndexWidth::I64})))
    return error;
  if (distribution.scalarAdd[site])
    if (llvm::Error error = addDedicatedScalarAddFu(pe, {inputs[0], inputs[1]}))
      return error;
  if (distribution.mac[site])
    if (llvm::Error error =
            addMacFu(pe, {inputs[0], inputs[1], inputs[2], inputs[3]}))
      return error;
  if (distribution.vectorCompute[site])
    if (llvm::Error error = addVectorComputeFu(
            pe, {inputs[0], inputs[1], inputs[2], inputs[3]}, {128, 128}))
      return error;
  if (distribution.loopControl[site]) {
    static constexpr std::array<
        std::pair<::dataflow::StreamStepKind, ::dataflow::StreamStepKind>, 4>
        stepPairs = {
            {{::dataflow::StreamStepKind::Add, ::dataflow::StreamStepKind::Sub},
             {::dataflow::StreamStepKind::Mul,
              ::dataflow::StreamStepKind::SDiv},
             {::dataflow::StreamStepKind::UDiv,
              ::dataflow::StreamStepKind::ShL},
             {::dataflow::StreamStepKind::AShr,
              ::dataflow::StreamStepKind::LShr}}};
    const auto pair = stepPairs[*distribution.loopOrdinal[site] % 4];
    if (llvm::Error error =
            addLoopControlFu(pe, {inputs[0], inputs[1], inputs[2], inputs[3]},
                             pair.first, pair.second))
      return error;
  }
  if (distribution.tokenControl[site])
    if (llvm::Error error =
            addTokenControlFu(pe, inputs, TokenControlFuParameters{128, 64}))
      return error;
  if (distribution.vectorAdapter[site])
    if (llvm::Error error =
            addVectorAdapterFu(pe, {inputs[0], inputs[1], inputs[3]}))
      return error;
  if (distribution.vectorStructural[site])
    if (llvm::Error error = addVectorStructuralFu(
            pe, inputs, builtinVectorStructuralParameters()))
      return error;
  if (distribution.specialMath[site])
    if (llvm::Error error = addSpecialMathFu(pe, {inputs[0], inputs[1]}))
      return error;
  return pe.close();
}

std::uint32_t builtinTemporalTagWidth(std::uint32_t residentContexts) {
  return std::max(1U, llvm::Log2_64_Ceil(residentContexts));
}

struct MemoryMeshAttachments final {
  std::size_t first;
  std::size_t second;
  std::size_t firstInputCount;
  std::size_t firstOutputCount;
};

llvm::Expected<BuiltinSpatialCoreExpansion>
expandBuiltinSpatialCoreImpl(DesignBuilder &design,
                             const BuiltinTargetScale &scale) {
  if (!isValidBuiltinTargetScale(scale))
    return invalid("builtin target base scale is invalid or an FU occurrence "
                   "count exceeds its PE count");
  const std::uint32_t temporalTagWidth =
      builtinTemporalTagWidth(scale.temporalResidentContexts);
  auto bits128 = PortType::bits(128);
  if (!bits128)
    return bits128.takeError();
  auto tagged128 = PortType::taggedBits(128, temporalTagWidth);
  if (!tagged128)
    return tagged128.takeError();

  auto byte = PortType::bits(8);
  if (!byte)
    return byte.takeError();
  auto managerMemory =
      PortType::memory({PortType::kDynamicExtent}, std::move(*byte));
  if (!managerMemory)
    return managerMemory.takeError();
  std::vector<PortType> moduleInputs(scale.gatewayCount, *bits128);
  moduleInputs.push_back(*managerMemory);
  std::vector<PortType> moduleOutputTypes(scale.gatewayCount, *bits128);
  auto spatial = design.createSpatialCore("builtin-spatial-core", moduleInputs,
                                          moduleOutputTypes);
  if (!spatial)
    return spatial.takeError();

  auto memoryInterface = builtinMemoryInterface();
  if (!memoryInterface)
    return memoryInterface.takeError();

  auto spatialMemory = makeVariant64LocalMemory(
      {scale.memoryCapacityBytes, *memoryInterface, std::nullopt, true},
      scale.localMemoryPortVariant);
  if (!spatialMemory)
    return spatialMemory.takeError();
  auto temporalMemory = makeVariant64LocalMemory(
      {scale.memoryCapacityBytes, *memoryInterface,
       TemporalMemoryParameters{temporalTagWidth,
                                scale.temporalResidentContexts},
       true},
      scale.localMemoryPortVariant);
  if (!temporalMemory)
    return temporalMemory.takeError();

  constexpr std::uint32_t peInputPortCount = 5;
  constexpr std::uint32_t peOutputPortCount = 4;
  const std::uint32_t crossSchedulePortsPerTemporalPe =
      scale.crossScheduleBoundaryLanesPerTemporalPe;
  if (scale.temporalPeCount > std::numeric_limits<std::uint32_t>::max() /
                                  crossSchedulePortsPerTemporalPe)
    return invalid("builtin cross-schedule boundary count exceeds u32");

  const std::uint32_t meshDimension = scale.meshDimension;
  const std::uint64_t meshCellCount =
      static_cast<std::uint64_t>(meshDimension) * meshDimension;
  const std::uint64_t halfMesh = meshCellCount / 2;
  std::vector<MeshCellAttachmentSpec> spatialAttachmentSpecs;
  std::vector<MeshCellAttachmentSpec> temporalAttachmentSpecs;
  DistributedCellCursor spatialPeCells(meshCellCount, scale.spatialPeCount);
  DistributedCellCursor spatialMemoryFirstCells(meshCellCount,
                                                scale.spatialMemoryCount);
  DistributedCellCursor spatialMemorySecondCells(
      meshCellCount, scale.spatialMemoryCount, halfMesh);
  // Module gateways are the SpatialCore's external boundary width, which the
  // System interconnect owns. Spatial-to-Temporal converters are the only path
  // between the two overlaid meshes, so their count must instead track the
  // compute that generates cross-domain traffic. Sizing both from one
  // parameter left a converter pair per external port while cross-domain
  // dataflow edges grew with the mesh area, and every mixed-schedule mapping
  // funnelled through those few cells. A boundary occurrence carries one
  // statically routed logical stream, while one Temporal PE exposes several
  // independently routable inputs and outputs. Derive converter supply from
  // that physical boundary width so resident contexts do not acquire an
  // artificial one-stream-per-PE cut. Placing the pairs on the same
  // distributed phase as the PEs keeps each conversion local.
  DistributedCellCursor moduleGatewayCells(meshCellCount, scale.gatewayCount);
  DistributedCellCursor s2tSpatialCells(meshCellCount, scale.temporalPeCount);
  DistributedCellCursor t2sSpatialCells(meshCellCount, scale.temporalPeCount);
  DistributedCellCursor temporalPeCells(meshCellCount, scale.temporalPeCount);
  DistributedCellCursor temporalMemoryFirstCells(meshCellCount,
                                                 scale.temporalMemoryCount);
  DistributedCellCursor temporalMemorySecondCells(
      meshCellCount, scale.temporalMemoryCount, halfMesh);
  DistributedCellCursor temporalGatewayCells(meshCellCount,
                                             scale.temporalPeCount);
  auto appendAttachment = [&](std::vector<MeshCellAttachmentSpec> &attachments,
                              DistributedCellCursor &cellCursor,
                              std::vector<PortType> inputTypes,
                              std::vector<PortType> outputTypes) {
    const std::uint64_t cell = cellCursor.next();
    const std::size_t ordinal = attachments.size();
    attachments.push_back({static_cast<std::uint32_t>(cell % meshDimension),
                           static_cast<std::uint32_t>(cell / meshDimension),
                           std::move(inputTypes), std::move(outputTypes)});
    return ordinal;
  };
  auto appendMemoryAttachments =
      [&](std::vector<MeshCellAttachmentSpec> &attachments,
          DistributedCellCursor &firstCells, DistributedCellCursor &secondCells,
          const MemorySpec &memory,
          const PortType &linkType) -> llvm::Expected<MemoryMeshAttachments> {
    const llvm::ArrayRef<PortType> inputTypes =
        memory.inputTypes().drop_front();
    const llvm::ArrayRef<PortType> outputTypes = memory.outputTypes();
    if (inputTypes.empty() || outputTypes.empty())
      return invalid("builtin memory requires transport inputs and outputs");
    const std::size_t firstInputCount = (inputTypes.size() + 1) / 2;
    const std::size_t firstOutputCount = (outputTypes.size() + 1) / 2;
    const std::size_t first =
        appendAttachment(attachments, firstCells,
                         std::vector<PortType>(firstInputCount, linkType),
                         std::vector<PortType>(firstOutputCount, linkType));
    const std::size_t second = appendAttachment(
        attachments, secondCells,
        std::vector<PortType>(inputTypes.size() - firstInputCount, linkType),
        std::vector<PortType>(outputTypes.size() - firstOutputCount, linkType));
    return MemoryMeshAttachments{first, second, firstInputCount,
                                 firstOutputCount};
  };

  std::vector<std::size_t> spatialPeAttachments;
  for (std::uint32_t site = 0; site != scale.spatialPeCount; ++site)
    spatialPeAttachments.push_back(
        appendAttachment(spatialAttachmentSpecs, spatialPeCells,
                         std::vector<PortType>(peInputPortCount, *bits128),
                         std::vector<PortType>(peOutputPortCount, *bits128)));
  std::vector<MemoryMeshAttachments> spatialMemoryAttachments;
  for (std::uint32_t memory = 0; memory != scale.spatialMemoryCount; ++memory) {
    auto attachments = appendMemoryAttachments(
        spatialAttachmentSpecs, spatialMemoryFirstCells,
        spatialMemorySecondCells, *spatialMemory, *bits128);
    if (!attachments)
      return attachments.takeError();
    spatialMemoryAttachments.push_back(*attachments);
  }
  std::vector<std::size_t> moduleGatewayAttachments;
  std::vector<std::size_t> s2tSpatialAttachments;
  std::vector<std::size_t> t2sSpatialAttachments;
  for (std::uint32_t gateway = 0; gateway != scale.gatewayCount; ++gateway)
    moduleGatewayAttachments.push_back(appendAttachment(
        spatialAttachmentSpecs, moduleGatewayCells, {*bits128}, {*bits128}));
  for (std::uint32_t site = 0; site != scale.temporalPeCount; ++site)
    s2tSpatialAttachments.push_back(appendAttachment(
        spatialAttachmentSpecs, s2tSpatialCells,
        std::vector<PortType>(crossSchedulePortsPerTemporalPe, *bits128), {}));
  for (std::uint32_t site = 0; site != scale.temporalPeCount; ++site)
    t2sSpatialAttachments.push_back(appendAttachment(
        spatialAttachmentSpecs, t2sSpatialCells, {},
        std::vector<PortType>(crossSchedulePortsPerTemporalPe, *bits128)));

  std::vector<std::size_t> temporalPeAttachments;
  for (std::uint32_t site = 0; site != scale.temporalPeCount; ++site)
    temporalPeAttachments.push_back(
        appendAttachment(temporalAttachmentSpecs, temporalPeCells,
                         std::vector<PortType>(peInputPortCount, *tagged128),
                         std::vector<PortType>(peOutputPortCount, *tagged128)));
  std::vector<MemoryMeshAttachments> temporalMemoryAttachments;
  for (std::uint32_t memory = 0; memory != scale.temporalMemoryCount;
       ++memory) {
    auto attachments = appendMemoryAttachments(
        temporalAttachmentSpecs, temporalMemoryFirstCells,
        temporalMemorySecondCells, *temporalMemory, *tagged128);
    if (!attachments)
      return attachments.takeError();
    temporalMemoryAttachments.push_back(*attachments);
  }
  std::vector<std::size_t> temporalGatewayAttachments;
  for (std::uint32_t site = 0; site != scale.temporalPeCount; ++site)
    temporalGatewayAttachments.push_back(appendAttachment(
        temporalAttachmentSpecs, temporalGatewayCells,
        std::vector<PortType>(crossSchedulePortsPerTemporalPe, *tagged128),
        std::vector<PortType>(crossSchedulePortsPerTemporalPe, *tagged128)));

  auto spatialNetworkSpec =
      MeshSwitchNetworkSpec::spatial(meshDimension, meshDimension, 2, *bits128,
                                     std::move(spatialAttachmentSpecs));
  if (!spatialNetworkSpec)
    return spatialNetworkSpec.takeError();
  auto spatialNetwork = spatial->addMeshSwitchNetwork(*spatialNetworkSpec);
  if (!spatialNetwork)
    return spatialNetwork.takeError();
  auto temporalNetworkSpec = MeshSwitchNetworkSpec::temporal(
      meshDimension, meshDimension, 2, *tagged128,
      scale.temporalResidentContexts, MeshSwitchGrantPolicyKind::RoundRobin,
      std::move(temporalAttachmentSpecs));
  if (!temporalNetworkSpec)
    return temporalNetworkSpec.takeError();
  auto temporalNetwork = spatial->addMeshSwitchNetwork(*temporalNetworkSpec);
  if (!temporalNetwork)
    return temporalNetwork.takeError();

  std::uint32_t nextLoopOrdinal = 0;
  const FuDistribution spatialDistribution = makeFuDistribution(
      scale.spatialPeCount, scale.spatialFuOccurrences, nextLoopOrdinal);
  const FuDistribution temporalDistribution = makeFuDistribution(
      scale.temporalPeCount, scale.temporalFuOccurrences, nextLoopOrdinal);
  const std::vector<PortType> spatialPeInputs(peInputPortCount, *bits128);
  const std::vector<PortType> spatialPeOutputs(peOutputPortCount, *bits128);
  for (std::uint32_t site = 0; site != scale.spatialPeCount; ++site) {
    auto attachment = spatialNetwork->attachment(spatialPeAttachments[site]);
    if (!attachment)
      return attachment.takeError();
    auto pe =
        spatial->addPe(attachment->inputs(),
                       PeSpec::spatial(spatialPeInputs, spatialPeOutputs));
    if (!pe)
      return pe.takeError();
    if (llvm::Error error = addFuCatalog(*pe, site, spatialDistribution))
      return std::move(error);
    std::vector<SpatialValue> outputs;
    for (std::size_t output = 0; output != peOutputPortCount; ++output) {
      auto value = pe->output(output);
      if (!value)
        return value.takeError();
      outputs.push_back(*value);
    }
    if (llvm::Error error = attachment->connectOutputs(outputs))
      return std::move(error);
  }
  for (std::uint32_t memory = 0; memory != scale.spatialMemoryCount; ++memory) {
    auto first =
        spatialNetwork->attachment(spatialMemoryAttachments[memory].first);
    if (!first)
      return first.takeError();
    auto second =
        spatialNetwork->attachment(spatialMemoryAttachments[memory].second);
    if (!second)
      return second.takeError();
    std::vector<SpatialValue> inputs;
    auto manager = spatial->input(scale.gatewayCount);
    if (!manager)
      return manager.takeError();
    inputs.push_back(*manager);
    inputs.insert(inputs.end(), first->inputs().begin(), first->inputs().end());
    inputs.insert(inputs.end(), second->inputs().begin(),
                  second->inputs().end());
    auto outputs = spatial->addMemory(inputs, *spatialMemory);
    if (!outputs)
      return outputs.takeError();
    std::vector<SpatialValue> routedOutputs;
    for (SpatialValue output : outputs->values()) {
      auto fifo = spatial->addFifo(
          output, FifoSpec{*bits128, scale.temporalResidentContexts, false});
      if (!fifo)
        return fifo.takeError();
      routedOutputs.push_back(fifo->value());
    }
    const std::size_t split = spatialMemoryAttachments[memory].firstOutputCount;
    if (llvm::Error error = first->connectOutputs(
            llvm::ArrayRef<SpatialValue>(routedOutputs).take_front(split)))
      return std::move(error);
    if (llvm::Error error = second->connectOutputs(
            llvm::ArrayRef<SpatialValue>(routedOutputs).drop_front(split)))
      return std::move(error);
  }

  const std::vector<PortType> temporalPeInputs(peInputPortCount, *bits128);
  const std::vector<PortType> temporalPeOutputs(peOutputPortCount, *tagged128);
  const TemporalPeParameters temporalParameters{
      scale.temporalResidentContexts, FuConfigurationMode::PerInstruction,
      ::fabric::OperandBufferMode::PerInstruction,
      scale.temporalResidentContexts,
      TemporalRegisterFifoParameters{scale.temporalResidentContexts,
                                     scale.temporalResidentContexts, 2}};
  for (std::uint32_t site = 0; site != scale.temporalPeCount; ++site) {
    auto attachment = temporalNetwork->attachment(temporalPeAttachments[site]);
    if (!attachment)
      return attachment.takeError();
    auto pe =
        spatial->addPe(attachment->inputs(),
                       PeSpec::temporal(temporalPeInputs, temporalPeOutputs,
                                        temporalParameters));
    if (!pe)
      return pe.takeError();
    if (llvm::Error error = addFuCatalog(*pe, site, temporalDistribution))
      return std::move(error);
    std::vector<SpatialValue> outputs;
    for (std::size_t output = 0; output != peOutputPortCount; ++output) {
      auto value = pe->output(output);
      if (!value)
        return value.takeError();
      outputs.push_back(*value);
    }
    if (llvm::Error error = attachment->connectOutputs(outputs))
      return std::move(error);
  }
  for (std::uint32_t memory = 0; memory != scale.temporalMemoryCount;
       ++memory) {
    auto first =
        temporalNetwork->attachment(temporalMemoryAttachments[memory].first);
    if (!first)
      return first.takeError();
    auto second =
        temporalNetwork->attachment(temporalMemoryAttachments[memory].second);
    if (!second)
      return second.takeError();
    std::vector<SpatialValue> inputs;
    auto manager = spatial->input(scale.gatewayCount);
    if (!manager)
      return manager.takeError();
    inputs.push_back(*manager);
    inputs.insert(inputs.end(), first->inputs().begin(), first->inputs().end());
    inputs.insert(inputs.end(), second->inputs().begin(),
                  second->inputs().end());
    auto outputs = spatial->addMemory(inputs, *temporalMemory);
    if (!outputs)
      return outputs.takeError();
    std::vector<SpatialValue> routedOutputs;
    for (SpatialValue output : outputs->values()) {
      auto fifo = spatial->addFifo(
          output, FifoSpec{*tagged128, scale.temporalResidentContexts, false});
      if (!fifo)
        return fifo.takeError();
      routedOutputs.push_back(fifo->value());
    }
    const std::size_t split =
        temporalMemoryAttachments[memory].firstOutputCount;
    if (llvm::Error error = first->connectOutputs(
            llvm::ArrayRef<SpatialValue>(routedOutputs).take_front(split)))
      return std::move(error);
    if (llvm::Error error = second->connectOutputs(
            llvm::ArrayRef<SpatialValue>(routedOutputs).drop_front(split)))
      return std::move(error);
  }

  for (std::uint32_t site = 0; site != scale.temporalPeCount; ++site) {
    auto spatialAttachment =
        spatialNetwork->attachment(s2tSpatialAttachments[site]);
    if (!spatialAttachment)
      return spatialAttachment.takeError();
    auto temporalAttachment =
        temporalNetwork->attachment(temporalGatewayAttachments[site]);
    if (!temporalAttachment)
      return temporalAttachment.takeError();
    if (spatialAttachment->inputs().size() != crossSchedulePortsPerTemporalPe ||
        temporalAttachment->inputs().size() != crossSchedulePortsPerTemporalPe)
      return invalid("builtin cross-schedule attachment width changed");

    std::vector<SpatialValue> taggedOutputs;
    taggedOutputs.reserve(crossSchedulePortsPerTemporalPe);
    for (SpatialValue input : spatialAttachment->inputs()) {
      auto outputs = spatial->addBoundary(
          {input}, BoundarySpec::s2tWithConfiguredTag(*bits128, *tagged128));
      if (!outputs)
        return outputs.takeError();
      auto tagged = spatial->addFifo(
          outputs->values().front(),
          FifoSpec{*tagged128, scale.temporalResidentContexts, false});
      if (!tagged)
        return tagged.takeError();
      taggedOutputs.push_back(tagged->value());
    }
    if (llvm::Error error = temporalAttachment->connectOutputs(taggedOutputs))
      return std::move(error);

    auto t2sAttachment =
        spatialNetwork->attachment(t2sSpatialAttachments[site]);
    if (!t2sAttachment)
      return t2sAttachment.takeError();
    std::vector<SpatialValue> routedOutputs;
    routedOutputs.reserve(crossSchedulePortsPerTemporalPe);
    for (SpatialValue input : temporalAttachment->inputs()) {
      auto outputs = spatial->addBoundary(
          {input}, BoundarySpec::t2s(*tagged128, {*bits128}));
      if (!outputs)
        return outputs.takeError();
      auto fifo = spatial->addFifo(
          outputs->values().front(),
          FifoSpec{*bits128, scale.temporalResidentContexts, false});
      if (!fifo)
        return fifo.takeError();
      routedOutputs.push_back(fifo->value());
    }
    if (llvm::Error error = t2sAttachment->connectOutputs(routedOutputs))
      return std::move(error);
  }

  std::vector<SpatialValue> moduleOutputs;
  for (std::uint32_t gateway = 0; gateway != scale.gatewayCount; ++gateway) {
    auto attachment =
        spatialNetwork->attachment(moduleGatewayAttachments[gateway]);
    if (!attachment)
      return attachment.takeError();
    auto input = spatial->input(gateway);
    if (!input)
      return input.takeError();
    if (llvm::Error error = attachment->connectOutputs({*input}))
      return std::move(error);
    moduleOutputs.push_back(attachment->inputs().front());
  }
  return BuiltinSpatialCoreExpansion{std::move(*spatial),
                                     std::move(moduleOutputs)};
}

llvm::Expected<SystemBuilder>
expandBuiltinSystemImpl(DesignBuilder &design, const BuiltinTargetScale &scale,
                        const loom::fabric::FinalizedFabricRoot &module) {
  if (!isValidBuiltinTargetScale(scale))
    return invalid("builtin target base scale is invalid or an FU occurrence "
                   "count exceeds its PE count");
  auto system = design.createSystem("builtin-system");
  if (!system)
    return system.takeError();
  auto imported = system->importSpatialCore(module);
  if (!imported)
    return imported.takeError();
  auto architecture = getBuiltinInstructionCoreArchitecture();
  if (!architecture)
    return architecture.takeError();
  auto inOrder = inOrderMicroarchitecture();
  if (!inOrder)
    return inOrder.takeError();
  auto outOfOrder = outOfOrderMicroarchitecture();
  if (!outOfOrder)
    return outOfOrder.takeError();
  auto host = system->addHostCore(*architecture, *inOrder);
  if (!host)
    return host.takeError();

  std::vector<AccCore> cores;
  cores.reserve(scale.accCoreCount);
  for (std::uint32_t ordinal = 0; ordinal != scale.accCoreCount; ++ordinal) {
    auto core = system->addAccCore(
        *architecture, ordinal % 2 == 0 ? *inOrder : *outOfOrder, *imported);
    if (!core)
      return core.takeError();
    cores.push_back(*core);
  }

  auto transportContract = singleRequesterResourceContract(
      2 * (scale.accCoreCount + scale.gatewayCount) *
      scale.temporalResidentContexts);
  if (!transportContract)
    return transportContract.takeError();
  auto bits128 = PortType::bits(128);
  if (!bits128)
    return bits128.takeError();
  std::vector<HardwareDomainMember> clockMembers;
  clockMembers.reserve(cores.size() * (2 + scale.gatewayCount * 2) + 3);
  clockMembers.push_back(host->domainMember());
  for (const AccCore &core : cores) {
    clockMembers.push_back(core.instructionCoreDomainMember());
    clockMembers.push_back(core.spatialCoreDomainMember());
  }
  std::vector<SystemTransportEndpoint> memoryRequestCarriers;
  std::vector<SystemTransportEndpoint> memoryResponseCarriers;
  std::vector<std::vector<SystemTransportEndpoint>> occurrenceRequestCarriers(
      cores.size());
  std::vector<std::vector<SystemTransportEndpoint>> occurrenceResponseCarriers(
      cores.size());
  for (std::uint32_t source = 0; source != cores.size(); ++source) {
    for (std::uint32_t gateway = 0; gateway != scale.gatewayCount; ++gateway) {
      auto transport = system->addTransportResource(
          {{*bits128}, {*bits128}, *transportContract});
      if (!transport)
        return transport.takeError();
      auto pattern = system->addTransferPattern(*transport, 0, {0}, 0);
      if (!pattern)
        return pattern.takeError();
      const std::uint32_t destination =
          (source + gateway + 1) % scale.accCoreCount;
      auto sourceEndpoint = cores[source].spatialTransportOutput(gateway);
      if (!sourceEndpoint)
        return sourceEndpoint.takeError();
      occurrenceRequestCarriers[source].push_back(*sourceEndpoint);
      auto transportInput = transport->input(0);
      if (!transportInput)
        return transportInput.takeError();
      memoryRequestCarriers.push_back(*transportInput);
      if (llvm::Error error = system->connect(*sourceEndpoint, *transportInput))
        return std::move(error);
      auto transportOutput = transport->output(0);
      if (!transportOutput)
        return transportOutput.takeError();
      memoryResponseCarriers.push_back(*transportOutput);
      auto destinationEndpoint =
          cores[destination].spatialTransportInput(gateway);
      if (!destinationEndpoint)
        return destinationEndpoint.takeError();
      occurrenceResponseCarriers[destination].push_back(*destinationEndpoint);
      if (llvm::Error error =
              system->connect(*transportOutput, *destinationEndpoint))
        return std::move(error);
      clockMembers.push_back(transport->domainMember());
      clockMembers.push_back(pattern->domainMember());
    }
  }

  auto clock = system->createHardwareDomain();
  if (!clock)
    return clock.takeError();
  auto clockContract = loom::fabric::ClockDomainContractRecord::create(
      builtinSystemClockPeriodFs, 0);
  if (!clockContract)
    return clockContract.takeError();
  auto memoryServiceRate = system->createServiceRate(
      *clock, 1, 1, scale.temporalResidentContexts,
      loom::fabric::ServiceProgress(
          std::in_place_type<::fabric::BoundedCompletion>,
          ::fabric::BoundedCompletion{
              loom::fabric::ClockDomainRef(clock->reference()),
              builtinSystemMemoryCompletionCycles}));
  if (!memoryServiceRate)
    return memoryServiceRate.takeError();
  auto messageServiceRate = system->createServiceRate(
      *clock, 1, 1, scale.temporalResidentContexts,
      loom::fabric::ServiceProgress(
          std::in_place_type<::fabric::FairEventual>));
  if (!messageServiceRate)
    return messageServiceRate.takeError();
  auto systemMemoryCapacity = llvm::checkedMulUnsigned<std::uint64_t>(
      scale.memoryCapacityBytes, scale.accCoreCount);
  if (!systemMemoryCapacity)
    return invalid("builtin System memory capacity overflows u64");
  auto memoryAccessDomain = builtinMemoryAccessDomain();
  if (!memoryAccessDomain)
    return memoryAccessDomain.takeError();
  auto systemMemory = makeGeneral64SystemMemory(
      {0, *systemMemoryCapacity, std::move(*memoryAccessDomain), 128},
      std::move(*memoryServiceRate));
  if (!systemMemory)
    return systemMemory.takeError();
  auto memoryService = system->addMemoryService(systemMemory->contract);
  if (!memoryService)
    return memoryService.takeError();
  auto memoryEndpoint =
      system->addServiceEndpoint(*memoryService, systemMemory->capabilities);
  if (!memoryEndpoint)
    return memoryEndpoint.takeError();
  for (auto indexedCore : llvm::enumerate(cores)) {
    const AccCore &core = indexedCore.value();
    auto spatialMemory = core.spatialMemoryManager(0);
    if (!spatialMemory)
      return spatialMemory.takeError();
    if (llvm::Error error =
            system->attachSpatialMemory(*spatialMemory, *memoryEndpoint))
      return std::move(error);
    const auto attachOccurrenceLeg =
        [&](dataflow::semantics::ServiceKind kind,
            dataflow::StructuralOrdinal leg,
            llvm::ArrayRef<SystemTransportEndpoint> carriers) -> llvm::Error {
      return system->attachServiceLegCarriers(*spatialMemory, kind, leg,
                                              carriers);
    };
    if (llvm::Error error =
            attachOccurrenceLeg(dataflow::semantics::ServiceKind::MemoryRead, 0,
                                occurrenceRequestCarriers[indexedCore.index()]))
      return std::move(error);
    if (llvm::Error error = attachOccurrenceLeg(
            dataflow::semantics::ServiceKind::MemoryRead, 1,
            occurrenceResponseCarriers[indexedCore.index()]))
      return std::move(error);
    if (llvm::Error error = attachOccurrenceLeg(
            dataflow::semantics::ServiceKind::MemoryWrite, 0,
            occurrenceRequestCarriers[indexedCore.index()]))
      return std::move(error);
    if (llvm::Error error = attachOccurrenceLeg(
            dataflow::semantics::ServiceKind::MemoryWrite, 1,
            occurrenceResponseCarriers[indexedCore.index()]))
      return std::move(error);
  }
  auto memoryEndpointRef = memoryEndpoint->memory();
  if (!memoryEndpointRef)
    return memoryEndpointRef.takeError();
  if (llvm::Error error = system->attachServiceLegCarriers(
          *memoryEndpointRef, dataflow::semantics::ServiceKind::MemoryRead, 0,
          memoryRequestCarriers))
    return std::move(error);
  if (llvm::Error error = system->attachServiceLegCarriers(
          *memoryEndpointRef, dataflow::semantics::ServiceKind::MemoryRead, 1,
          memoryResponseCarriers))
    return std::move(error);
  if (llvm::Error error = system->attachServiceLegCarriers(
          *memoryEndpointRef, dataflow::semantics::ServiceKind::MemoryWrite, 0,
          memoryRequestCarriers))
    return std::move(error);
  if (llvm::Error error = system->attachServiceLegCarriers(
          *memoryEndpointRef, dataflow::semantics::ServiceKind::MemoryWrite, 1,
          memoryResponseCarriers))
    return std::move(error);
  clockMembers.push_back(memoryService->domainMember());
  clockMembers.push_back(memoryEndpoint->domainMember());

  mlir::MLIRContext messageTypeContext;
  auto fixedVectors = loom::fabric::FixedVectorMessagePayloadDomain::create(
      detail::catalogFixedVectorElementTypes(messageTypeContext), 128,
      dataflow::canonicalTypeMaximumRank);
  if (!fixedVectors)
    return fixedVectors.takeError();
  auto messageDomain = loom::fabric::MessageTransferCapabilityDomain::create(
      detail::catalogScalarPayloadTypes(messageTypeContext),
      std::move(*fixedVectors), detail::catalogPointerFormats());
  if (!messageDomain)
    return messageDomain.takeError();
  auto initiateCapability =
      loom::fabric::CanonicalServiceCapabilityRecord::create(
          dataflow::semantics::ServiceKind::MessageTransfer,
          loom::fabric::CanonicalServiceEndpointRole::Initiate, *messageDomain,
          *messageServiceRate);
  if (!initiateCapability)
    return initiateCapability.takeError();
  auto serveCapability = loom::fabric::CanonicalServiceCapabilityRecord::create(
      dataflow::semantics::ServiceKind::MessageTransfer,
      loom::fabric::CanonicalServiceEndpointRole::Serve, *messageDomain,
      *messageServiceRate);
  if (!serveCapability)
    return serveCapability.takeError();
  auto initiateSet = loom::fabric::CanonicalServiceCapabilitySet::create(
      {std::move(*initiateCapability)});
  if (!initiateSet)
    return initiateSet.takeError();
  auto serveSet = loom::fabric::CanonicalServiceCapabilitySet::create(
      {std::move(*serveCapability)});
  if (!serveSet)
    return serveSet.takeError();

  std::vector<SystemServiceEndpoint> messageSources;
  std::vector<SystemServiceEndpoint> messageSinks;
  messageSources.reserve(cores.size() + 1);
  messageSinks.reserve(cores.size() + 1);
  const auto appendMessageEndpoints = [&](const auto &owner) -> llvm::Error {
    auto source = system->addServiceEndpoint(owner, *initiateSet, *bits128);
    if (!source)
      return source.takeError();
    auto sink = system->addServiceEndpoint(owner, *serveSet, *bits128);
    if (!sink)
      return sink.takeError();
    clockMembers.push_back(source->domainMember());
    clockMembers.push_back(sink->domainMember());
    messageSources.push_back(*source);
    messageSinks.push_back(*sink);
    return llvm::Error::success();
  };
  if (llvm::Error error = appendMessageEndpoints(*host))
    return std::move(error);
  for (const AccCore &core : cores)
    if (llvm::Error error = appendMessageEndpoints(core))
      return std::move(error);

  std::vector<SystemTransportResource> messageRouters;
  messageRouters.reserve(messageSources.size());
  const std::array<std::vector<std::uint32_t>, 3> messagePatterns = {
      std::vector<std::uint32_t>{0}, std::vector<std::uint32_t>{1},
      std::vector<std::uint32_t>{0, 1}};
  for (std::size_t ordinal = 0; ordinal != messageSources.size(); ++ordinal) {
    auto router = system->addTransportResource(
        {{{*bits128, *bits128}}, {{*bits128, *bits128}}, *transportContract});
    if (!router)
      return router.takeError();
    clockMembers.push_back(router->domainMember());
    for (std::size_t input = 0; input != 2; ++input)
      for (const auto &outputs : messagePatterns) {
        auto pattern = system->addTransferPattern(*router, input, outputs, 0);
        if (!pattern)
          return pattern.takeError();
        clockMembers.push_back(pattern->domainMember());
      }
    auto source = messageSources[ordinal].transport();
    if (!source)
      return source.takeError();
    auto input = router->input(0);
    if (!input)
      return input.takeError();
    if (llvm::Error error = system->connect(*source, *input))
      return std::move(error);
    auto output = router->output(0);
    if (!output)
      return output.takeError();
    auto sink = messageSinks[ordinal].transport();
    if (!sink)
      return sink.takeError();
    if (llvm::Error error = system->connect(*output, *sink))
      return std::move(error);
    messageRouters.push_back(*router);
  }
  for (std::size_t ordinal = 0; ordinal != messageRouters.size(); ++ordinal) {
    auto output = messageRouters[ordinal].output(1);
    if (!output)
      return output.takeError();
    auto input = messageRouters[(ordinal + 1) % messageRouters.size()].input(1);
    if (!input)
      return input.takeError();
    if (llvm::Error error = system->connect(*output, *input))
      return std::move(error);
  }
  if (llvm::Error error = clock->close(clockMembers, *clockContract))
    return std::move(error);
  auto reset = system->createHardwareDomain();
  if (!reset)
    return reset.takeError();
  auto resetContract = loom::fabric::ResetDomainContractRecord::create(
      loom::fabric::ResetPolarity::ActiveHigh,
      loom::fabric::ResetTiming::Synchronous,
      loom::fabric::ResetTiming::Synchronous,
      loom::fabric::ResetInitialState::Asserted,
      loom::fabric::ClockDomainRef(clock->reference()), 0);
  if (!resetContract)
    return resetContract.takeError();
  if (llvm::Error error = reset->close(clockMembers, *resetContract))
    return std::move(error);
  return std::move(*system);
}

} // namespace

llvm::Expected<loom::fabric::InstructionCoreArchitecturalContract>
getBuiltinInstructionCoreArchitecture() {
  return makeBuiltinInstructionCoreArchitecture();
}

llvm::Expected<BuiltinTargetPreset>
parseBuiltinTargetPreset(llvm::StringRef spelling) {
  for (const BuiltinTargetDescriptor *descriptor :
       {&builtinSmallTarget, &builtinCoverageTarget, &builtinLargeTarget})
    if (spelling == descriptor->name)
      return descriptor->preset;
  return invalid("unknown builtin target preset '" + spelling + "'");
}

llvm::Expected<BuiltinSpatialCoreExpansion>
expandBuiltinSpatialCore(DesignBuilder &design, BuiltinTargetPreset preset) {
  const BuiltinTargetDescriptor &descriptor =
      getBuiltinTargetDescriptor(preset);
  return expandBuiltinSpatialCoreImpl(design, descriptor.scale);
}

llvm::Expected<BuiltinSpatialCoreExpansion>
expandBuiltinSpatialCore(DesignBuilder &design,
                         const BuiltinTargetScale &scale) {
  return expandBuiltinSpatialCoreImpl(design, scale);
}

llvm::Expected<SystemBuilder>
expandBuiltinSystem(DesignBuilder &design, BuiltinTargetPreset preset,
                    const loom::fabric::FinalizedFabricRoot &spatialCore) {
  const BuiltinTargetDescriptor &descriptor =
      getBuiltinTargetDescriptor(preset);
  return expandBuiltinSystemImpl(design, descriptor.scale, spatialCore);
}

llvm::Expected<SystemBuilder>
expandBuiltinSystem(DesignBuilder &design, const BuiltinTargetScale &scale,
                    const loom::fabric::FinalizedFabricRoot &spatialCore) {
  return expandBuiltinSystemImpl(design, scale, spatialCore);
}

llvm::Expected<FinalizedFabricDesign>
buildBuiltinTarget(const loom::ArtifactStore &store,
                   BuiltinTargetPreset preset) {
  return buildBuiltinTarget(store, getBuiltinTargetDescriptor(preset).scale);
}

llvm::Expected<FinalizedFabricDesign>
buildBuiltinTarget(const loom::ArtifactStore &store,
                   const BuiltinTargetScale &scale) {
  DesignBuilder moduleDesign(store);
  auto moduleExpansion = expandBuiltinSpatialCore(moduleDesign, scale);
  if (!moduleExpansion)
    return moduleExpansion.takeError();
  if (llvm::Error error =
          moduleExpansion->spatialCore.close(moduleExpansion->outputs))
    return std::move(error);
  auto modules = std::move(moduleDesign).finalize();
  if (!modules)
    return modules.takeError();
  if (modules->roots().size() != 1)
    return invalid("builtin expansion did not finalize one SpatialCore");

  DesignBuilder systemDesign(store);
  auto system =
      expandBuiltinSystem(systemDesign, scale, modules->roots().front());
  if (!system)
    return system.takeError();
  if (llvm::Error error = system->close())
    return std::move(error);
  return std::move(systemDesign).finalize();
}

llvm::Expected<FinalizedFabricDesign>
buildBuiltinTarget(const loom::ArtifactStore &store,
                   llvm::StringRef templateIdentity, std::uint32_t schemaMajor,
                   std::uint32_t schemaMinor, const BuiltinTargetScale &scale) {
  const BuiltinTargetDescriptor *descriptor =
      findBuiltinTargetDescriptor(templateIdentity, schemaMajor, schemaMinor);
  if (!descriptor)
    return invalid("resolved hardware target is not a registered builtin "
                   "template");
  return buildBuiltinTarget(store, scale);
}

} // namespace loom::adg
