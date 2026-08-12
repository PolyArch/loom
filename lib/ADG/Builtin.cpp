#include "ADG/Builtin.h"

#include "ADG/FuLibrary.h"
#include "ADG/MemoryLibrary.h"

#include "Dataflow/IR/DataflowEnums.h"
#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include <algorithm>
#include <array>
#include <cstdint>
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
      2,
      {{loom::fabric::InstructionOperationClass::IntegerAlu, 2, 1, 1},
       {loom::fabric::InstructionOperationClass::IntegerMultiply, 1, 3, 1},
       {loom::fabric::InstructionOperationClass::LoadStore, 2, 2, 1},
       {loom::fabric::InstructionOperationClass::FloatingPointAlu, 2, 3, 1},
       {loom::fabric::InstructionOperationClass::FloatingPointMultiply, 1, 4,
        1},
       {loom::fabric::InstructionOperationClass::FloatingPointDivide, 1, 12,
        12}},
      std::move(*resources)};
  loom::fabric::OutOfOrderMicroarchitectureDeclaration pipeline{
      2, 2, 2, 2, 2, 2, 2, 32, 16, 8, 8, 64, 32, 32};
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

std::uint32_t ceilDiv(std::uint32_t value, std::uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

struct FuDistribution final {
  std::vector<bool> mac;
  std::vector<bool> vectorCompute;
  std::vector<bool> loopControl;
  std::vector<bool> tokenControl;
  std::vector<bool> vectorAdapter;
  std::vector<bool> vectorStructural;
  std::vector<bool> specialMath;
  std::vector<std::optional<std::uint32_t>> loopOrdinal;
};

FuDistribution makeFuDistribution(std::uint32_t count,
                                  std::uint32_t &nextLoopOrdinal) {
  FuDistribution distribution{
      distributedSites(count, ceilDiv(count, 2), 0),
      distributedSites(count, ceilDiv(count, 4), 1),
      distributedSites(count, ceilDiv(count, 4), 2),
      distributedSites(count, ceilDiv(count, 4), 3),
      distributedSites(count, std::max(1u, ceilDiv(count, 8)), 4),
      distributedSites(count, std::max(1u, ceilDiv(count, 8)), 5),
      distributedSites(count, std::max(1u, ceilDiv(count, 16)), 7),
      std::vector<std::optional<std::uint32_t>>(count)};
  for (std::uint32_t site = 0; site != count; ++site)
    if (distribution.loopControl[site])
      distribution.loopOrdinal[site] = nextLoopOrdinal++;
  return distribution;
}

VectorStructuralFuParameters builtinVectorStructuralParameters() {
  const ::fabric::IntegerWidthSet integerWidths =
      ::fabric::IntegerWidthSet::get(
          {::fabric::IntegerWidth::I8, ::fabric::IntegerWidth::I16,
           ::fabric::IntegerWidth::I32, ::fabric::IntegerWidth::I64});
  const ::fabric::FloatFormatSet floatFormats = ::fabric::FloatFormatSet::get(
      {::fabric::FloatFormat::F16, ::fabric::FloatFormat::BF16,
       ::fabric::FloatFormat::F32, ::fabric::FloatFormat::F64});
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

std::uint32_t builtinMeshDimension(BuiltinTargetPreset preset) {
  switch (preset) {
  case BuiltinTargetPreset::Small:
    return 4;
  case BuiltinTargetPreset::Default:
    return 6;
  case BuiltinTargetPreset::Large:
    return 8;
  }
  llvm_unreachable("unknown builtin target preset");
}

struct MemoryMeshAttachments final {
  std::size_t first;
  std::size_t second;
  std::size_t firstInputCount;
  std::size_t firstOutputCount;
};

llvm::Expected<BuiltinSpatialCoreExpansion>
expandBuiltinSpatialCoreImpl(DesignBuilder &design,
                             const BuiltinTargetDescriptor &descriptor) {
  const BuiltinTargetScale &scale = descriptor.scale;
  auto tagBits = PortType::bits(scale.temporalResidentContexts);
  if (!tagBits)
    return tagBits.takeError();
  auto bits128 = PortType::bits(128);
  if (!bits128)
    return bits128.takeError();
  auto tagged128 = PortType::taggedBits(128, scale.temporalResidentContexts);
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
  auto spatial =
      design.createSpatialCore((descriptor.name + "-spatial-core").str(),
                               moduleInputs, moduleOutputTypes);
  if (!spatial)
    return spatial.takeError();

  auto memoryInterface = builtinMemoryInterface();
  if (!memoryInterface)
    return memoryInterface.takeError();

  auto spatialMemory = makeGeneral64LocalMemory(
      {scale.memoryCapacityBytes, *memoryInterface, std::nullopt, true});
  if (!spatialMemory)
    return spatialMemory.takeError();
  auto temporalMemory = makeGeneral64LocalMemory(
      {scale.memoryCapacityBytes, *memoryInterface,
       TemporalMemoryParameters{scale.temporalResidentContexts,
                                scale.temporalResidentContexts},
       true});
  if (!temporalMemory)
    return temporalMemory.takeError();

  const std::uint32_t meshDimension =
      builtinMeshDimension(descriptor.preset);
  const std::size_t meshCellCount =
      static_cast<std::size_t>(meshDimension) * meshDimension;
  std::vector<MeshCellAttachmentSpec> spatialAttachmentSpecs;
  std::vector<MeshCellAttachmentSpec> temporalAttachmentSpecs;
  std::size_t spatialCellCursor = 0;
  std::size_t temporalCellCursor = 0;
  auto appendAttachment =
      [&](std::vector<MeshCellAttachmentSpec> &attachments,
          std::size_t &cellCursor, std::vector<PortType> inputTypes,
          std::vector<PortType> outputTypes) {
        const std::size_t cell = cellCursor++ % meshCellCount;
        const std::size_t ordinal = attachments.size();
        attachments.push_back(
            {static_cast<std::uint32_t>(cell % meshDimension),
             static_cast<std::uint32_t>(cell / meshDimension),
             std::move(inputTypes), std::move(outputTypes)});
        return ordinal;
      };
  auto appendMemoryAttachments =
      [&](std::vector<MeshCellAttachmentSpec> &attachments,
          std::size_t &cellCursor, const MemorySpec &memory,
          const PortType &linkType) -> llvm::Expected<MemoryMeshAttachments> {
    const llvm::ArrayRef<PortType> inputTypes =
        memory.inputTypes().drop_front();
    const llvm::ArrayRef<PortType> outputTypes = memory.outputTypes();
    if (inputTypes.empty() || outputTypes.empty())
      return invalid(
          "builtin memory requires transport inputs and outputs");
    const std::size_t firstInputCount = (inputTypes.size() + 1) / 2;
    const std::size_t firstOutputCount = (outputTypes.size() + 1) / 2;
    const std::size_t first = appendAttachment(
        attachments, cellCursor,
        std::vector<PortType>(firstInputCount, linkType),
        std::vector<PortType>(firstOutputCount, linkType));
    const std::size_t second = appendAttachment(
        attachments, cellCursor,
        std::vector<PortType>(inputTypes.size() - firstInputCount, linkType),
        std::vector<PortType>(outputTypes.size() - firstOutputCount, linkType));
    return MemoryMeshAttachments{first, second, firstInputCount,
                                 firstOutputCount};
  };

  std::vector<std::size_t> spatialPeAttachments;
  for (std::uint32_t site = 0; site != scale.spatialPeCount; ++site)
    spatialPeAttachments.push_back(appendAttachment(
        spatialAttachmentSpecs, spatialCellCursor,
        std::vector<PortType>(5, *bits128),
        std::vector<PortType>(4, *bits128)));
  std::vector<MemoryMeshAttachments> spatialMemoryAttachments;
  for (std::uint32_t memory = 0; memory != scale.spatialMemoryCount; ++memory) {
    auto attachments = appendMemoryAttachments(
        spatialAttachmentSpecs, spatialCellCursor, *spatialMemory, *bits128);
    if (!attachments)
      return attachments.takeError();
    spatialMemoryAttachments.push_back(*attachments);
  }
  std::vector<std::size_t> moduleGatewayAttachments;
  std::vector<std::size_t> s2tSpatialAttachments;
  std::vector<std::size_t> t2sSpatialAttachments;
  for (std::uint32_t gateway = 0; gateway != scale.gatewayCount; ++gateway)
    moduleGatewayAttachments.push_back(appendAttachment(
        spatialAttachmentSpecs, spatialCellCursor, {*bits128}, {*bits128}));
  for (std::uint32_t gateway = 0; gateway != scale.gatewayCount; ++gateway)
    s2tSpatialAttachments.push_back(appendAttachment(
        spatialAttachmentSpecs, spatialCellCursor, {*bits128, *bits128}, {}));
  for (std::uint32_t gateway = 0; gateway != scale.gatewayCount; ++gateway)
    t2sSpatialAttachments.push_back(appendAttachment(
        spatialAttachmentSpecs, spatialCellCursor, {}, {*bits128, *bits128}));

  std::vector<std::size_t> temporalPeAttachments;
  for (std::uint32_t site = 0; site != scale.temporalPeCount; ++site)
    temporalPeAttachments.push_back(appendAttachment(
        temporalAttachmentSpecs, temporalCellCursor,
        std::vector<PortType>(5, *tagged128),
        std::vector<PortType>(4, *tagged128)));
  std::vector<MemoryMeshAttachments> temporalMemoryAttachments;
  for (std::uint32_t memory = 0; memory != scale.temporalMemoryCount;
       ++memory) {
    auto attachments = appendMemoryAttachments(
        temporalAttachmentSpecs, temporalCellCursor, *temporalMemory,
        *tagged128);
    if (!attachments)
      return attachments.takeError();
    temporalMemoryAttachments.push_back(*attachments);
  }
  std::vector<std::size_t> temporalGatewayAttachments;
  for (std::uint32_t gateway = 0; gateway != scale.gatewayCount; ++gateway)
    temporalGatewayAttachments.push_back(appendAttachment(
        temporalAttachmentSpecs, temporalCellCursor, {*tagged128},
        {*tagged128}));

  auto spatialNetworkSpec = MeshSwitchNetworkSpec::spatial(
      meshDimension, meshDimension, 2, *bits128,
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
  const FuDistribution spatialDistribution =
      makeFuDistribution(scale.spatialPeCount, nextLoopOrdinal);
  const FuDistribution temporalDistribution =
      makeFuDistribution(scale.temporalPeCount, nextLoopOrdinal);
  const std::vector<PortType> spatialPeInputs(5, *bits128);
  const std::vector<PortType> spatialPeOutputs(4, *bits128);
  for (std::uint32_t site = 0; site != scale.spatialPeCount; ++site) {
    auto attachment = spatialNetwork->attachment(spatialPeAttachments[site]);
    if (!attachment)
      return attachment.takeError();
    auto pe = spatial->addPe(
        attachment->inputs(),
        PeSpec::spatial(spatialPeInputs, spatialPeOutputs));
    if (!pe)
      return pe.takeError();
    if (llvm::Error error = addFuCatalog(*pe, site, spatialDistribution))
      return std::move(error);
    std::vector<SpatialValue> outputs;
    for (std::size_t output = 0; output != 4; ++output) {
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
          output,
          FifoSpec{*bits128, scale.temporalResidentContexts, true});
      if (!fifo)
        return fifo.takeError();
      routedOutputs.push_back(fifo->value());
    }
    const std::size_t split =
        spatialMemoryAttachments[memory].firstOutputCount;
    if (llvm::Error error =
            first->connectOutputs(
                llvm::ArrayRef<SpatialValue>(routedOutputs).take_front(split)))
      return std::move(error);
    if (llvm::Error error =
            second->connectOutputs(
                llvm::ArrayRef<SpatialValue>(routedOutputs).drop_front(split)))
      return std::move(error);
  }

  const std::vector<PortType> temporalPeInputs(5, *bits128);
  const std::vector<PortType> temporalPeOutputs(4, *tagged128);
  const TemporalPeParameters temporalParameters{
      scale.temporalResidentContexts, FuConfigurationMode::PerInstruction,
      ::fabric::OperandBufferMode::PerInstruction,
      scale.temporalResidentContexts,
      TemporalRegisterFifoParameters{scale.temporalResidentContexts,
                                     scale.temporalResidentContexts, 2}};
  for (std::uint32_t site = 0; site != scale.temporalPeCount; ++site) {
    auto attachment =
        temporalNetwork->attachment(temporalPeAttachments[site]);
    if (!attachment)
      return attachment.takeError();
    auto pe = spatial->addPe(
        attachment->inputs(),
        PeSpec::temporal(temporalPeInputs, temporalPeOutputs,
                         temporalParameters));
    if (!pe)
      return pe.takeError();
    if (llvm::Error error = addFuCatalog(*pe, site, temporalDistribution))
      return std::move(error);
    std::vector<SpatialValue> outputs;
    for (std::size_t output = 0; output != 4; ++output) {
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
          output,
          FifoSpec{*tagged128, scale.temporalResidentContexts, true});
      if (!fifo)
        return fifo.takeError();
      routedOutputs.push_back(fifo->value());
    }
    const std::size_t split =
        temporalMemoryAttachments[memory].firstOutputCount;
    if (llvm::Error error =
            first->connectOutputs(
                llvm::ArrayRef<SpatialValue>(routedOutputs).take_front(split)))
      return std::move(error);
    if (llvm::Error error =
            second->connectOutputs(
                llvm::ArrayRef<SpatialValue>(routedOutputs).drop_front(split)))
      return std::move(error);
  }

  for (std::uint32_t gateway = 0; gateway != scale.gatewayCount; ++gateway) {
    auto spatialAttachment =
        spatialNetwork->attachment(s2tSpatialAttachments[gateway]);
    if (!spatialAttachment)
      return spatialAttachment.takeError();
    auto temporalAttachment =
        temporalNetwork->attachment(temporalGatewayAttachments[gateway]);
    if (!temporalAttachment)
      return temporalAttachment.takeError();
    auto outputs = spatial->addBoundary(
        spatialAttachment->inputs(),
        BoundarySpec::s2t(*bits128, *tagBits, *tagged128));
    if (!outputs)
      return outputs.takeError();
    if (llvm::Error error = temporalAttachment->connectOutputs(outputs->values()))
      return std::move(error);
    auto t2sOutputs = spatial->addBoundary(
        temporalAttachment->inputs(),
        BoundarySpec::t2s(*tagged128, {*bits128, *tagBits}));
    if (!t2sOutputs)
      return t2sOutputs.takeError();
    auto t2sAttachment =
        spatialNetwork->attachment(t2sSpatialAttachments[gateway]);
    if (!t2sAttachment)
      return t2sAttachment.takeError();
    std::vector<SpatialValue> routedOutputs;
    for (SpatialValue output : t2sOutputs->values()) {
      auto fifo = spatial->addFifo(
          output,
          FifoSpec{*bits128, scale.temporalResidentContexts, true});
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
expandBuiltinSystemImpl(DesignBuilder &design,
                        const BuiltinTargetDescriptor &descriptor,
                        const loom::fabric::FinalizedFabricRoot &module) {
  auto system = design.createSystem((descriptor.name + "-system").str());
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
  cores.reserve(descriptor.scale.accCoreCount);
  for (std::uint32_t ordinal = 0; ordinal != descriptor.scale.accCoreCount;
       ++ordinal) {
    auto core = system->addAccCore(
        *architecture, ordinal % 2 == 0 ? *inOrder : *outOfOrder, *imported);
    if (!core)
      return core.takeError();
    cores.push_back(*core);
  }

  auto transportContract = singleRequesterResourceContract(
      2 * (descriptor.scale.accCoreCount + descriptor.scale.gatewayCount) *
      descriptor.scale.temporalResidentContexts);
  if (!transportContract)
    return transportContract.takeError();
  auto bits128 = PortType::bits(128);
  if (!bits128)
    return bits128.takeError();
  std::vector<HardwareDomainMember> clockMembers;
  clockMembers.reserve(cores.size() * (2 + descriptor.scale.gatewayCount * 2) +
                       3);
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
    for (std::uint32_t gateway = 0; gateway != descriptor.scale.gatewayCount;
         ++gateway) {
      auto transport = system->addTransportResource(
          {{*bits128}, {*bits128}, *transportContract});
      if (!transport)
        return transport.takeError();
      auto pattern = system->addTransferPattern(*transport, 0, {0}, 0);
      if (!pattern)
        return pattern.takeError();
      const std::uint32_t destination =
          (source + gateway + 1) % descriptor.scale.accCoreCount;
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
  auto clockContract =
      loom::fabric::ClockDomainContractRecord::create(1'000, 0);
  if (!clockContract)
    return clockContract.takeError();
  auto serviceRate = system->createServiceRate(
      *clock, 1, 1, descriptor.scale.temporalResidentContexts,
      loom::fabric::ServiceProgress(
          std::in_place_type<::fabric::FairEventual>));
  if (!serviceRate)
    return serviceRate.takeError();
  auto systemMemoryCapacity = llvm::checkedMulUnsigned<std::uint64_t>(
      descriptor.scale.memoryCapacityBytes, descriptor.scale.accCoreCount);
  if (!systemMemoryCapacity)
    return invalid("builtin System memory capacity overflows u64");
  auto memoryAccessDomain = builtinMemoryAccessDomain();
  if (!memoryAccessDomain)
    return memoryAccessDomain.takeError();
  auto systemMemory = makeGeneral64SystemMemory(
      {0, *systemMemoryCapacity, std::move(*memoryAccessDomain), 128},
      std::move(*serviceRate));
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
  auto messageDomain = loom::fabric::MessageTransferCapabilityDomain::create(
      {mlir::NoneType::get(&messageTypeContext),
       mlir::IntegerType::get(&messageTypeContext, 1),
       mlir::IntegerType::get(&messageTypeContext, 8),
       mlir::IntegerType::get(&messageTypeContext, 16),
       mlir::IntegerType::get(&messageTypeContext, 32),
       mlir::IntegerType::get(&messageTypeContext, 64),
       mlir::IndexType::get(&messageTypeContext)});
  if (!messageDomain)
    return messageDomain.takeError();
  auto initiateCapability =
      loom::fabric::CanonicalServiceCapabilityRecord::create(
          dataflow::semantics::ServiceKind::MessageTransfer,
          loom::fabric::CanonicalServiceEndpointRole::Initiate, *messageDomain,
          *serviceRate);
  if (!initiateCapability)
    return initiateCapability.takeError();
  auto serveCapability = loom::fabric::CanonicalServiceCapabilityRecord::create(
      dataflow::semantics::ServiceKind::MessageTransfer,
      loom::fabric::CanonicalServiceEndpointRole::Serve, *messageDomain,
      *serviceRate);
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
       {&builtinSmallTarget, &builtinDefaultTarget, &builtinLargeTarget})
    if (spelling == descriptor->name)
      return descriptor->preset;
  return invalid("unknown builtin target preset '" + spelling + "'");
}

llvm::Expected<BuiltinSpatialCoreExpansion>
expandBuiltinSpatialCore(DesignBuilder &design, BuiltinTargetPreset preset) {
  return expandBuiltinSpatialCoreImpl(design,
                                      getBuiltinTargetDescriptor(preset));
}

llvm::Expected<SystemBuilder>
expandBuiltinSystem(DesignBuilder &design, BuiltinTargetPreset preset,
                    const loom::fabric::FinalizedFabricRoot &spatialCore) {
  return expandBuiltinSystemImpl(design, getBuiltinTargetDescriptor(preset),
                                 spatialCore);
}

llvm::Expected<FinalizedFabricDesign>
buildBuiltinTarget(const loom::ArtifactStore &store,
                   BuiltinTargetPreset preset) {
  DesignBuilder moduleDesign(store);
  auto moduleExpansion = expandBuiltinSpatialCore(moduleDesign, preset);
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
      expandBuiltinSystem(systemDesign, preset, modules->roots().front());
  if (!system)
    return system.takeError();
  if (llvm::Error error = system->close())
    return std::move(error);
  return std::move(systemDesign).finalize();
}

} // namespace loom::adg
