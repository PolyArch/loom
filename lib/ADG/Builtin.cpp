#include "ADG/Builtin.h"

#include "ADG/FuLibrary.h"
#include "ADG/MemoryLibrary.h"

#include "Dataflow/IR/DataflowEnums.h"
#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace loom::adg {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "adg_builtin_invalid: " + message);
}

template <typename T>
llvm::Expected<T> indexed(llvm::ArrayRef<T> values, std::size_t &cursor,
                          llvm::StringRef owner) {
  if (cursor >= values.size())
    return invalid(owner + " output cursor exceeds its typed interface");
  return values[cursor++];
}

llvm::Expected<::fabric::ResourceContract> exclusiveResourceContract() {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {
      {::fabric::StateKey(0),
       {{::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1),
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
  auto resources = exclusiveResourceContract();
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
  auto resources = exclusiveResourceContract();
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

struct TypedSpatialBackedge final {
  SpatialBackedge edge;
  PortType type;
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

MemoryInterfaceParameters builtinMemoryInterface() {
  return {MemoryAccessDomainParameters{128, 128, 16}, 128, 128};
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
            pe, {inputs[0], inputs[1], inputs[2], inputs[3]}))
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
    if (llvm::Error error = addTokenControlFu(pe, inputs))
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

std::vector<std::vector<std::uint32_t>>
fullConnectivity(std::size_t inputCount, std::size_t outputCount) {
  std::vector<std::uint32_t> sources;
  sources.reserve(inputCount);
  for (std::uint32_t ordinal = 0; ordinal != inputCount; ++ordinal)
    sources.push_back(ordinal);
  return std::vector<std::vector<std::uint32_t>>(outputCount, sources);
}

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

  auto spatialMemory =
      makeGeneral64LocalMemory({scale.memoryCapacityBytes,
                                builtinMemoryInterface(), std::nullopt, true});
  if (!spatialMemory)
    return spatialMemory.takeError();
  auto temporalMemory = makeGeneral64LocalMemory(
      {scale.memoryCapacityBytes, builtinMemoryInterface(),
       TemporalMemoryParameters{scale.temporalResidentContexts,
                                scale.temporalResidentContexts},
       true});
  if (!temporalMemory)
    return temporalMemory.takeError();

  std::vector<TypedSpatialBackedge> spatialPeFeedback;
  std::vector<TypedSpatialBackedge> spatialMemoryFeedback;
  std::vector<TypedSpatialBackedge> t2sFeedback;
  std::vector<TypedSpatialBackedge> temporalPeFeedback;
  std::vector<TypedSpatialBackedge> temporalMemoryFeedback;
  std::vector<TypedSpatialBackedge> s2tFeedback;
  auto appendFeedback = [&](std::vector<TypedSpatialBackedge> &destination,
                            const PortType &type,
                            std::size_t count) -> llvm::Error {
    for (std::size_t ordinal = 0; ordinal != count; ++ordinal) {
      auto edge = spatial->createBackedge(type);
      if (!edge)
        return edge.takeError();
      destination.push_back({std::move(*edge), type});
    }
    return llvm::Error::success();
  };
  if (llvm::Error error =
          appendFeedback(spatialPeFeedback, *bits128, scale.spatialPeCount * 4))
    return std::move(error);
  for (std::uint32_t memory = 0; memory != scale.spatialMemoryCount; ++memory)
    for (std::size_t output = 0; output != spatialMemory->outputTypes().size();
         ++output)
      if (llvm::Error error =
              appendFeedback(spatialMemoryFeedback, *bits128, 1))
        return std::move(error);
  for (std::uint32_t gateway = 0; gateway != scale.gatewayCount; ++gateway) {
    if (llvm::Error error = appendFeedback(t2sFeedback, *bits128, 1))
      return std::move(error);
    if (llvm::Error error = appendFeedback(t2sFeedback, *bits128, 1))
      return std::move(error);
  }
  if (llvm::Error error = appendFeedback(temporalPeFeedback, *tagged128,
                                         scale.temporalPeCount * 4))
    return std::move(error);
  for (std::uint32_t memory = 0; memory != scale.temporalMemoryCount; ++memory)
    for (std::size_t output = 0; output != temporalMemory->outputTypes().size();
         ++output)
      if (llvm::Error error =
              appendFeedback(temporalMemoryFeedback, *tagged128, 1))
        return std::move(error);
  if (llvm::Error error =
          appendFeedback(s2tFeedback, *tagged128, scale.gatewayCount))
    return std::move(error);

  std::vector<SpatialValue> spatialSwitchInputs;
  std::vector<PortType> spatialSwitchInputTypes;
  for (std::uint32_t ordinal = 0; ordinal != scale.gatewayCount; ++ordinal) {
    auto input = spatial->input(ordinal);
    if (!input)
      return input.takeError();
    spatialSwitchInputs.push_back(*input);
    spatialSwitchInputTypes.push_back(*bits128);
  }
  auto appendValues = [](auto &values, auto &types, const auto &edges) {
    for (const auto &edge : edges) {
      values.push_back(edge.edge.value());
      types.push_back(edge.type);
    }
  };
  appendValues(spatialSwitchInputs, spatialSwitchInputTypes, spatialPeFeedback);
  appendValues(spatialSwitchInputs, spatialSwitchInputTypes,
               spatialMemoryFeedback);
  appendValues(spatialSwitchInputs, spatialSwitchInputTypes, t2sFeedback);

  std::vector<PortType> spatialSwitchOutputTypes;
  spatialSwitchOutputTypes.insert(spatialSwitchOutputTypes.end(),
                                  scale.spatialPeCount * 5, *bits128);
  for (std::uint32_t memory = 0; memory != scale.spatialMemoryCount; ++memory)
    spatialSwitchOutputTypes.insert(spatialSwitchOutputTypes.end(),
                                    spatialMemory->inputTypes().size() - 1,
                                    *bits128);
  for (std::uint32_t gateway = 0; gateway != scale.gatewayCount; ++gateway) {
    spatialSwitchOutputTypes.push_back(*bits128);
    spatialSwitchOutputTypes.push_back(*bits128);
  }
  spatialSwitchOutputTypes.insert(spatialSwitchOutputTypes.end(),
                                  scale.gatewayCount, *bits128);
  auto spatialRoutes = spatial->addSwitch(
      spatialSwitchInputs,
      SwitchSpec::spatial(spatialSwitchInputTypes, spatialSwitchOutputTypes,
                          fullConnectivity(spatialSwitchInputs.size(),
                                           spatialSwitchOutputTypes.size())));
  if (!spatialRoutes)
    return spatialRoutes.takeError();

  std::vector<SpatialValue> temporalSwitchInputs;
  std::vector<PortType> temporalSwitchInputTypes;
  appendValues(temporalSwitchInputs, temporalSwitchInputTypes, s2tFeedback);
  appendValues(temporalSwitchInputs, temporalSwitchInputTypes,
               temporalPeFeedback);
  appendValues(temporalSwitchInputs, temporalSwitchInputTypes,
               temporalMemoryFeedback);
  std::vector<PortType> temporalSwitchOutputTypes;
  temporalSwitchOutputTypes.insert(temporalSwitchOutputTypes.end(),
                                   scale.temporalPeCount * 5, *tagged128);
  for (std::uint32_t memory = 0; memory != scale.temporalMemoryCount; ++memory)
    temporalSwitchOutputTypes.insert(temporalSwitchOutputTypes.end(),
                                     temporalMemory->inputTypes().size() - 1,
                                     *tagged128);
  temporalSwitchOutputTypes.insert(temporalSwitchOutputTypes.end(),
                                   scale.gatewayCount, *tagged128);
  auto temporalRoutes = spatial->addSwitch(
      temporalSwitchInputs,
      SwitchSpec::temporal(
          temporalSwitchInputTypes, temporalSwitchOutputTypes,
          fullConnectivity(temporalSwitchInputs.size(),
                           temporalSwitchOutputTypes.size()),
          scale.temporalResidentContexts, [&]() {
            std::vector<std::uint32_t> cycle(temporalSwitchInputs.size());
            std::iota(cycle.begin(), cycle.end(), 0);
            return ::fabric::TemporalSwitchGrantPolicy(
                ::fabric::TemporalSwitchRoundRobin{std::move(cycle), 0});
          }()));
  if (!temporalRoutes)
    return temporalRoutes.takeError();

  std::uint32_t nextLoopOrdinal = 0;
  const FuDistribution spatialDistribution =
      makeFuDistribution(scale.spatialPeCount, nextLoopOrdinal);
  const FuDistribution temporalDistribution =
      makeFuDistribution(scale.temporalPeCount, nextLoopOrdinal);
  std::size_t spatialCursor = 0;
  std::size_t spatialFeedbackCursor = 0;
  const std::vector<PortType> spatialPeInputs(5, *bits128);
  const std::vector<PortType> spatialPeOutputs(4, *bits128);
  for (std::uint32_t site = 0; site != scale.spatialPeCount; ++site) {
    std::vector<SpatialValue> inputs;
    for (std::size_t ordinal = 0; ordinal != 5; ++ordinal) {
      auto value = indexed<SpatialValue>(*spatialRoutes, spatialCursor,
                                         "spatial switch");
      if (!value)
        return value.takeError();
      inputs.push_back(*value);
    }
    auto pe = spatial->addPe(
        inputs, PeSpec::spatial(spatialPeInputs, spatialPeOutputs));
    if (!pe)
      return pe.takeError();
    if (llvm::Error error = addFuCatalog(*pe, site, spatialDistribution))
      return std::move(error);
    for (std::size_t output = 0; output != 4; ++output) {
      auto value = pe->output(output);
      if (!value)
        return value.takeError();
      auto fifo = spatial->addFifo(
          *value, FifoSpec{*bits128, scale.temporalResidentContexts, true});
      if (!fifo)
        return fifo.takeError();
      if (llvm::Error error = spatial->resolveBackedge(
              std::move(spatialPeFeedback[spatialFeedbackCursor++].edge),
              *fifo))
        return std::move(error);
    }
  }
  std::size_t spatialMemoryFeedbackCursor = 0;
  for (std::uint32_t memory = 0; memory != scale.spatialMemoryCount; ++memory) {
    std::vector<SpatialValue> inputs;
    auto manager = spatial->input(scale.gatewayCount);
    if (!manager)
      return manager.takeError();
    inputs.push_back(*manager);
    for (std::size_t ordinal = 1; ordinal != spatialMemory->inputTypes().size();
         ++ordinal) {
      auto value = indexed<SpatialValue>(*spatialRoutes, spatialCursor,
                                         "spatial switch");
      if (!value)
        return value.takeError();
      inputs.push_back(*value);
    }
    auto outputs = spatial->addMemory(inputs, *spatialMemory);
    if (!outputs)
      return outputs.takeError();
    for (SpatialValue output : *outputs) {
      auto fifo = spatial->addFifo(
          output, FifoSpec{*bits128, scale.temporalResidentContexts, true});
      if (!fifo)
        return fifo.takeError();
      if (llvm::Error error = spatial->resolveBackedge(
              std::move(
                  spatialMemoryFeedback[spatialMemoryFeedbackCursor++].edge),
              *fifo))
        return std::move(error);
    }
  }

  std::size_t temporalCursor = 0;
  std::size_t temporalFeedbackCursor = 0;
  const std::vector<PortType> temporalPeInputs(5, *bits128);
  const std::vector<PortType> temporalPeOutputs(4, *tagged128);
  const TemporalPeParameters temporalParameters{
      scale.temporalResidentContexts, FuConfigurationMode::PerInstruction,
      ::fabric::OperandBufferMode::PerInstruction,
      scale.temporalResidentContexts,
      TemporalRegisterFifoParameters{scale.temporalResidentContexts,
                                     scale.temporalResidentContexts, 2}};
  for (std::uint32_t site = 0; site != scale.temporalPeCount; ++site) {
    std::vector<SpatialValue> inputs;
    for (std::size_t ordinal = 0; ordinal != 5; ++ordinal) {
      auto value = indexed<SpatialValue>(*temporalRoutes, temporalCursor,
                                         "temporal switch");
      if (!value)
        return value.takeError();
      inputs.push_back(*value);
    }
    auto pe = spatial->addPe(inputs, PeSpec::temporal(temporalPeInputs,
                                                      temporalPeOutputs,
                                                      temporalParameters));
    if (!pe)
      return pe.takeError();
    if (llvm::Error error = addFuCatalog(*pe, site, temporalDistribution))
      return std::move(error);
    for (std::size_t output = 0; output != 4; ++output) {
      auto value = pe->output(output);
      if (!value)
        return value.takeError();
      auto fifo = spatial->addFifo(
          *value, FifoSpec{*tagged128, scale.temporalResidentContexts, true});
      if (!fifo)
        return fifo.takeError();
      if (llvm::Error error = spatial->resolveBackedge(
              std::move(temporalPeFeedback[temporalFeedbackCursor++].edge),
              *fifo))
        return std::move(error);
    }
  }
  std::size_t temporalMemoryFeedbackCursor = 0;
  for (std::uint32_t memory = 0; memory != scale.temporalMemoryCount;
       ++memory) {
    std::vector<SpatialValue> inputs;
    auto manager = spatial->input(scale.gatewayCount);
    if (!manager)
      return manager.takeError();
    inputs.push_back(*manager);
    for (std::size_t ordinal = 1;
         ordinal != temporalMemory->inputTypes().size(); ++ordinal) {
      auto value = indexed<SpatialValue>(*temporalRoutes, temporalCursor,
                                         "temporal switch");
      if (!value)
        return value.takeError();
      inputs.push_back(*value);
    }
    auto outputs = spatial->addMemory(inputs, *temporalMemory);
    if (!outputs)
      return outputs.takeError();
    for (SpatialValue output : *outputs) {
      auto fifo = spatial->addFifo(
          output, FifoSpec{*tagged128, scale.temporalResidentContexts, true});
      if (!fifo)
        return fifo.takeError();
      if (llvm::Error error = spatial->resolveBackedge(
              std::move(
                  temporalMemoryFeedback[temporalMemoryFeedbackCursor++].edge),
              *fifo))
        return std::move(error);
    }
  }

  std::size_t s2tFeedbackCursor = 0;
  for (std::uint32_t gateway = 0; gateway != scale.gatewayCount; ++gateway) {
    auto data =
        indexed<SpatialValue>(*spatialRoutes, spatialCursor, "spatial switch");
    if (!data)
      return data.takeError();
    auto tag =
        indexed<SpatialValue>(*spatialRoutes, spatialCursor, "spatial switch");
    if (!tag)
      return tag.takeError();
    auto outputs = spatial->addBoundary(
        {*data, *tag}, BoundarySpec::s2t(*bits128, *tagBits, *tagged128));
    if (!outputs)
      return outputs.takeError();
    auto fifo = spatial->addFifo(
        outputs->front(),
        FifoSpec{*tagged128, scale.temporalResidentContexts, true});
    if (!fifo)
      return fifo.takeError();
    if (llvm::Error error = spatial->resolveBackedge(
            std::move(s2tFeedback[s2tFeedbackCursor++].edge), *fifo))
      return std::move(error);
  }
  std::size_t t2sFeedbackCursor = 0;
  for (std::uint32_t gateway = 0; gateway != scale.gatewayCount; ++gateway) {
    auto tagged = indexed<SpatialValue>(*temporalRoutes, temporalCursor,
                                        "temporal switch");
    if (!tagged)
      return tagged.takeError();
    auto outputs = spatial->addBoundary(
        {*tagged}, BoundarySpec::t2s(*tagged128, {*bits128, *tagBits}));
    if (!outputs)
      return outputs.takeError();
    for (SpatialValue output : *outputs) {
      auto fifo = spatial->addFifo(
          output, FifoSpec{*bits128, scale.temporalResidentContexts, true});
      if (!fifo)
        return fifo.takeError();
      if (llvm::Error error = spatial->resolveBackedge(
              std::move(t2sFeedback[t2sFeedbackCursor++].edge), *fifo))
        return std::move(error);
    }
  }

  std::vector<SpatialValue> moduleOutputs;
  for (std::uint32_t gateway = 0; gateway != scale.gatewayCount; ++gateway) {
    auto output =
        indexed<SpatialValue>(*spatialRoutes, spatialCursor, "spatial switch");
    if (!output)
      return output.takeError();
    moduleOutputs.push_back(*output);
  }
  if (spatialCursor != spatialRoutes->size() ||
      temporalCursor != temporalRoutes->size())
    return invalid("builtin switch output partition is incomplete");
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

  auto transportContract = exclusiveResourceContract();
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
      auto transportInput = transport->input(0);
      if (!transportInput)
        return transportInput.takeError();
      if (llvm::Error error = system->connect(*sourceEndpoint, *transportInput))
        return std::move(error);
      auto transportOutput = transport->output(0);
      if (!transportOutput)
        return transportOutput.takeError();
      auto destinationEndpoint =
          cores[destination].spatialTransportInput(gateway);
      if (!destinationEndpoint)
        return destinationEndpoint.takeError();
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
  auto systemMemory = makeGeneral64SystemMemory(
      {0, *systemMemoryCapacity, MemoryAccessDomainParameters{128, 128, 16},
       128},
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
  clockMembers.push_back(memoryService->domainMember());
  clockMembers.push_back(memoryEndpoint->domainMember());
  if (llvm::Error error = clock->close(clockMembers, *clockContract))
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
