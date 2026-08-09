#include "Hardware/RTL/Providers/FixedVectorParallelizeSerialize.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Fabric/IR/ImplementationFamily.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

using Group = ::dataflow::semantics::ActorResultProductionGroup;
using Schema = ::dataflow::OperationSchemaId;

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  unsigned elementWidth = 0;
  unsigned laneCount = 0;
};

struct PhysicalPorts final {
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> outputs;
};

struct ParallelizeGroups final {
  Group payload;
  Group terminal;
};

struct SerializeGroups final {
  Group item;
  Group close;
};

struct ParallelizeModeLogic final {
  mlir::Value hasBuffered;
  mlir::Value fullAfterAppend;
  mlir::Value appendedValue;
  mlir::Value appendedMask;
};

struct SerializeModeLogic final {
  mlir::Value hasActive;
  mlir::Value hasRemaining;
  mlir::Value selectedData;
  mlir::Value clearedMask;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_fixed_vector_cardinality_invalid: " +
                                     message);
}

llvm::Expected<FabricOperationProviderOutput>
unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

mlir::Value constant(mlir::OpBuilder &builder, mlir::Location location,
                     const llvm::APInt &value) {
  return circt::hw::ConstantOp::create(builder, location, value);
}

mlir::Value bitConstant(mlir::OpBuilder &builder, mlir::Location location,
                        bool value) {
  return constant(builder, location, llvm::APInt(1, value));
}

mlir::Value andAll(mlir::OpBuilder &builder, mlir::Location location,
                   llvm::ArrayRef<mlir::Value> values) {
  if (values.empty())
    return bitConstant(builder, location, true);
  mlir::Value result = values.front();
  for (mlir::Value value : values.drop_front())
    result = circt::comb::AndOp::create(builder, location, result, value);
  return result;
}

mlir::Value orAll(mlir::OpBuilder &builder, mlir::Location location,
                  llvm::ArrayRef<mlir::Value> values) {
  if (values.empty())
    return bitConstant(builder, location, false);
  mlir::Value result = values.front();
  for (mlir::Value value : values.drop_front())
    result = circt::comb::OrOp::create(builder, location, result, value);
  return result;
}

mlir::Value isEqual(mlir::OpBuilder &builder, mlir::Location location,
                    mlir::Value lhs, mlir::Value rhs) {
  return circt::comb::ICmpOp::create(
      builder, location, circt::comb::ICmpPredicate::eq, lhs, rhs, true);
}

mlir::Value isZero(mlir::OpBuilder &builder, mlir::Location location,
                   mlir::Value value) {
  const unsigned width =
      mlir::cast<mlir::IntegerType>(value.getType()).getWidth();
  return isEqual(builder, location, value,
                 constant(builder, location, llvm::APInt(width, 0)));
}

mlir::Value invert(mlir::OpBuilder &builder, mlir::Location location,
                   mlir::Value value) {
  return circt::comb::createOrFoldNot(builder, location, value);
}

bool contains(llvm::ArrayRef<std::uint32_t> ordinals, std::uint32_t ordinal) {
  return llvm::is_contained(ordinals, ordinal);
}

bool isOnce(const Group &group, llvm::ArrayRef<std::uint32_t> results) {
  return std::holds_alternative<
             ::dataflow::semantics::ActorResultProductionOnce>(group.repeat) &&
         llvm::ArrayRef<std::uint32_t>(group.activeResults).equals(results);
}

llvm::Expected<ParallelizeGroups> deriveParallelizeGroups() {
  auto cases = ::dataflow::semantics::projectActorHandshakeCases(
      Schema::DataflowParallelize, 2, 3);
  if (!cases)
    return cases.takeError();
  constexpr std::array<std::uint32_t, 3> payloadResults{0, 1, 2};
  constexpr std::array<std::uint32_t, 1> terminalResults{2};
  if (cases->size() != 4 || !(*cases)[0].productionGroups.empty() ||
      (*cases)[1].productionGroups.size() != 1 ||
      (*cases)[2].productionGroups.size() != 1 ||
      (*cases)[3].productionGroups.size() != 2 ||
      !isOnce((*cases)[1].productionGroups[0], payloadResults) ||
      !isOnce((*cases)[2].productionGroups[0], terminalResults) ||
      !isOnce((*cases)[3].productionGroups[0], payloadResults) ||
      !isOnce((*cases)[3].productionGroups[1], terminalResults))
    return invalid("parallelize schema lost its ordered 0/1/1/2 groups");
  return ParallelizeGroups{(*cases)[1].productionGroups[0],
                           (*cases)[2].productionGroups[0]};
}

llvm::Expected<SerializeGroups> deriveSerializeGroups() {
  using Repeat =
      ::dataflow::semantics::ActorResultProductionForEachDefinedOneLane;
  auto cases = ::dataflow::semantics::projectActorHandshakeCases(
      Schema::DataflowSerialize, 3, 2);
  if (!cases)
    return cases.takeError();
  constexpr std::array<std::uint32_t, 2> itemResults{0, 1};
  constexpr std::array<std::uint32_t, 1> closeResults{1};
  if (cases->size() != 2 || (*cases)[0].productionGroups.size() != 1 ||
      (*cases)[1].productionGroups.size() != 1)
    return invalid("serialize schema lost its active/close groups");
  const Group &item = (*cases)[0].productionGroups[0];
  const auto *repeat = std::get_if<Repeat>(&item.repeat);
  if (!repeat ||
      repeat->maskInputOrdinal !=
          static_cast<std::uint32_t>(
              ::dataflow::semantics::SerializeInput::Mask) ||
      !llvm::ArrayRef<std::uint32_t>(item.activeResults).equals(itemResults) ||
      !isOnce((*cases)[1].productionGroups[0], closeResults))
    return invalid("serialize schema lost defined-one lane order");
  return SerializeGroups{item, (*cases)[1].productionGroups[0]};
}

llvm::Expected<PhysicalPorts>
derivePhysicalPorts(const fabric::ResolvedFabricOpCapabilityView &capability,
                    std::size_t inputCount, std::size_t outputCount) {
  PhysicalPorts result;
  for (const auto &port : capability.physicalPorts) {
    if (port.reference.direction == fabric::FabricPortDirection::Input)
      result.inputs.push_back(&port);
    else if (port.reference.direction == fabric::FabricPortDirection::Output)
      result.outputs.push_back(&port);
    else
      return invalid("physical port has an unknown direction");
  }
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(result.inputs, byOrdinal);
  llvm::sort(result.outputs, byOrdinal);
  const auto dense = [](const auto &ports) {
    return llvm::all_of(llvm::enumerate(ports), [](const auto &entry) {
      return entry.value()->reference.ordinal == entry.index() &&
             entry.value()->payloadWidthBits != 0;
    });
  };
  if (result.inputs.size() != inputCount ||
      result.outputs.size() != outputCount || !dense(result.inputs) ||
      !dense(result.outputs))
    return invalid("physical port inventory is not exact and dense");
  return result;
}

llvm::Expected<LoweredMode> lowerMode(const Mode &mode, Schema schema) {
  if (mode.actor.schema != schema ||
      !std::holds_alternative<::dataflow::NoPayload>(mode.actor.payload))
    return invalid("behavior witness changed the registered actor");
  mlir::VectorType vector;
  mlir::VectorType mask;
  mlir::Type scalar;
  if (schema == Schema::DataflowParallelize) {
    if (mode.actor.type.getNumInputs() != 2 ||
        mode.actor.type.getNumResults() != 3)
      return invalid("parallelize behavior has wrong arity");
    scalar = mode.actor.type.getInput(0);
    vector = mlir::dyn_cast<mlir::VectorType>(mode.actor.type.getResult(0));
    mask = mlir::dyn_cast<mlir::VectorType>(mode.actor.type.getResult(1));
    if (!mode.actor.type.getInput(1).isInteger(1) ||
        !mode.actor.type.getResult(2).isInteger(1))
      return invalid("parallelize behavior has malformed phase ports");
  } else {
    if (mode.actor.type.getNumInputs() != 3 ||
        mode.actor.type.getNumResults() != 2)
      return invalid("serialize behavior has wrong arity");
    vector = mlir::dyn_cast<mlir::VectorType>(mode.actor.type.getInput(0));
    mask = mlir::dyn_cast<mlir::VectorType>(mode.actor.type.getInput(1));
    scalar = mode.actor.type.getResult(0);
    if (!mode.actor.type.getInput(2).isInteger(1) ||
        !mode.actor.type.getResult(1).isInteger(1))
      return invalid("serialize behavior has malformed phase ports");
  }
  if (!vector || vector.getRank() != 1 || vector.getDimSize(0) <= 0 ||
      vector.getElementType() != scalar || !mask ||
      mask.getShape() != vector.getShape() ||
      !mask.getElementType().isInteger(1) ||
      !llvm::isa<mlir::IntegerType, mlir::FloatType>(scalar))
    return invalid("adapter behavior is not scalar/rank-one-vector typed");
  const std::int64_t lanes = vector.getDimSize(0);
  if (lanes > std::numeric_limits<unsigned>::max())
    return invalid("adapter lane count exceeds the RTL domain");
  return LoweredMode{scalar.getIntOrFloatBitWidth(),
                     static_cast<unsigned>(lanes)};
}

llvm::Expected<std::vector<Mode>>
deriveModes(FabricOperationProviderRequest &request,
            const FiniteCodebookEncoding *&codebook,
            std::size_t &inactiveMode) {
  auto relation = request.capability.resolveSemanticFieldRelation(
      *request.leaf.getContext());
  if (!relation)
    return relation.takeError();
  const auto domain = relation->finiteBehaviorDomain();
  if (domain.empty())
    return invalid("Fabric returned an empty adapter behavior domain");

  std::vector<Mode> modes;
  modes.reserve(domain.size());
  inactiveMode = 0;
  codebook = nullptr;
  if (request.capability.configurationFieldSchema.empty()) {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None ||
        domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid("configuration-free adapter has a non-singleton domain");
    modes.push_back({domain.front().representativeActor, nullptr});
    return modes;
  }

  if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite ||
      request.capability.configurationFieldSchema.size() != 1)
    return invalid("configured adapter does not have one finite field");
  const auto &semanticField =
      request.capability.configurationFieldSchema.front();
  const ConfigurationFieldEncoding *field =
      request.configurationAbi.findOperationField(request.occurrence,
                                                  semanticField.ordinal);
  if (!field)
    return invalid("configured field is absent from the ABI");
  codebook = std::get_if<FiniteCodebookEncoding>(&field->semanticEncoding);
  if (!codebook || codebook->entries.size() != domain.size())
    return invalid("finite codebook does not exactly cover the domain");
  for (const auto &point : domain) {
    if (!point.semanticConfiguration)
      return invalid("configured behavior has no semantic value");
    const FiniteCodebookEntry *entry = detail::findFiniteCodebookEntry(
        *codebook, point.semanticConfiguration->bytes());
    if (!entry)
      return invalid("codebook omits an admitted adapter behavior");
    modes.push_back({point.representativeActor, entry});
  }
  const auto inactive = llvm::find_if(modes, [&](const Mode &mode) {
    return llvm::ArrayRef<std::uint8_t>(mode.codebookEntry->semanticValue)
        .equals(field->inactiveValue);
  });
  if (inactive == modes.end())
    return invalid("ABI inactive value is outside the behavior domain");
  inactiveMode = static_cast<std::size_t>(inactive - modes.begin());
  return modes;
}

std::vector<mlir::Value>
materializeModeSelection(mlir::OpBuilder &builder, mlir::Location location,
                         circt::hw::HWModulePortAccessor &accessor,
                         const FabricOperationProviderRequest &request,
                         llvm::ArrayRef<Mode> modes,
                         const FiniteCodebookEncoding *codebook) {
  std::vector<mlir::Value> selected(modes.size());
  if (!codebook) {
    selected.front() = bitConstant(builder, location, true);
    return selected;
  }
  mlir::Value configuration = accessor.getInput(
      "config_" +
      std::to_string(
          request.capability.configurationFieldSchema.front().ordinal));
  for (std::size_t index = 0; index < modes.size(); ++index) {
    mlir::Value code = constant(
        builder, location,
        detail::decodePhysicalCode(modes[index].codebookEntry->physicalCode,
                                   codebook->encodedBitCount));
    selected[index] = isEqual(builder, location, configuration, code);
  }
  return selected;
}

mlir::Value selectModeValue(mlir::OpBuilder &builder, mlir::Location location,
                            llvm::ArrayRef<mlir::Value> values,
                            std::size_t inactiveMode,
                            llvm::ArrayRef<mlir::Value> selected) {
  mlir::Value result = values[inactiveMode];
  for (std::size_t index = 0; index < values.size(); ++index)
    if (index != inactiveMode)
      result = circt::comb::MuxOp::create(builder, location, selected[index],
                                          values[index], result, true);
  return result;
}

mlir::Value extractField(mlir::OpBuilder &builder, mlir::Location location,
                         mlir::Value state,
                         const FabricOperationLeafStateFieldLayout &field) {
  return circt::comb::ExtractOp::create(builder, location, state,
                                        field.bitOffset, field.bitCount);
}

mlir::Value encodeState(mlir::OpBuilder &builder, mlir::Location location,
                        mlir::Value value, mlir::Value mask) {
  return circt::comb::ConcatOp::create(builder, location,
                                       mlir::ValueRange{mask, value});
}

mlir::Value groupCapacity(mlir::OpBuilder &builder, mlir::Location location,
                          const Group &group,
                          llvm::ArrayRef<mlir::Value> outputReady) {
  llvm::SmallVector<mlir::Value, 4> terms;
  for (std::uint32_t ordinal : group.activeResults)
    terms.push_back(outputReady[ordinal]);
  return andAll(builder, location, terms);
}

mlir::Value groupValid(mlir::OpBuilder &builder, mlir::Location location,
                       mlir::Value selected, const Group &group,
                       std::uint32_t ordinal,
                       llvm::ArrayRef<mlir::Value> outputReady) {
  if (!contains(group.activeResults, ordinal))
    return bitConstant(builder, location, false);
  llvm::SmallVector<mlir::Value, 4> terms{selected};
  for (std::uint32_t other : group.activeResults)
    if (other != ordinal)
      terms.push_back(outputReady[other]);
  return andAll(builder, location, terms);
}

llvm::APInt lowMask(unsigned width, unsigned count) {
  llvm::APInt result(width, 0);
  result.setLowBits(std::min(width, count));
  return result;
}

ParallelizeModeLogic
materializeParallelizeMode(mlir::OpBuilder &builder, mlir::Location location,
                           const LoweredMode &mode, mlir::Value inputData,
                           mlir::Value stateValue, mlir::Value stateMask,
                           unsigned valueWidth, unsigned maskWidth) {
  const llvm::APInt pending = lowMask(maskWidth, mode.laneCount - 1);
  mlir::Value fullAfterAppend = isEqual(builder, location, stateMask,
                                        constant(builder, location, pending));
  mlir::Value hasBuffered =
      invert(builder, location, isZero(builder, location, stateMask));

  mlir::Value appendedValue = stateValue;
  mlir::Value appendedMask = stateMask;
  mlir::Value element = circt::comb::ExtractOp::create(
      builder, location, inputData, 0, mode.elementWidth);
  element = detail::resizeUnsigned(builder, location, element, valueWidth);
  for (unsigned lane = 0; lane < mode.laneCount; ++lane) {
    mlir::Value atLane =
        isEqual(builder, location, stateMask,
                constant(builder, location, lowMask(maskWidth, lane)));
    mlir::Value shift = constant(
        builder, location, llvm::APInt(valueWidth, lane * mode.elementWidth));
    mlir::Value placed =
        circt::comb::ShlOp::create(builder, location, element, shift, true);
    appendedValue = circt::comb::MuxOp::create(
        builder, location, atLane,
        circt::comb::OrOp::create(builder, location, stateValue, placed, true),
        appendedValue, true);
    llvm::APInt laneBit(maskWidth, 0);
    laneBit.setBit(lane);
    appendedMask = circt::comb::MuxOp::create(
        builder, location, atLane,
        circt::comb::OrOp::create(builder, location, stateMask,
                                  constant(builder, location, laneBit), true),
        appendedMask, true);
  }
  return {hasBuffered, fullAfterAppend, appendedValue, appendedMask};
}

SerializeModeLogic
materializeSerializeMode(mlir::OpBuilder &builder, mlir::Location location,
                         const LoweredMode &mode, mlir::Value sourceValue,
                         mlir::Value sourceMask, unsigned maskWidth,
                         unsigned scalarOutputWidth) {
  mlir::Value logicalMask = circt::comb::AndOp::create(
      builder, location, sourceMask,
      constant(builder, location, lowMask(maskWidth, mode.laneCount)), true);
  mlir::Value hasActive =
      invert(builder, location, isZero(builder, location, logicalMask));
  mlir::Value one = constant(builder, location, llvm::APInt(maskWidth, 1));
  mlir::Value decremented =
      circt::comb::SubOp::create(builder, location, logicalMask, one, true);
  mlir::Value clearedMask = circt::comb::AndOp::create(
      builder, location, logicalMask, decremented, true);
  mlir::Value hasRemaining =
      invert(builder, location, isZero(builder, location, clearedMask));

  mlir::Value selectedData =
      constant(builder, location, llvm::APInt(scalarOutputWidth, 0));
  mlir::Value preceding = bitConstant(builder, location, false);
  for (unsigned lane = 0; lane < mode.laneCount; ++lane) {
    mlir::Value active =
        circt::comb::ExtractOp::create(builder, location, logicalMask, lane, 1);
    mlir::Value lowest = andAll(builder, location,
                                {active, invert(builder, location, preceding)});
    mlir::Value laneValue = circt::comb::ExtractOp::create(
        builder, location, sourceValue, lane * mode.elementWidth,
        mode.elementWidth);
    laneValue =
        detail::resizeUnsigned(builder, location, laneValue, scalarOutputWidth);
    selectedData = circt::comb::MuxOp::create(builder, location, lowest,
                                              laneValue, selectedData, true);
    preceding =
        circt::comb::OrOp::create(builder, location, preceding, active, true);
  }
  return {hasActive, hasRemaining, selectedData, clearedMask};
}

llvm::Expected<FabricOperationProviderOutput> materializeParallelize(
    FabricOperationProviderRequest request, const PhysicalPorts &ports,
    llvm::ArrayRef<Mode> modes, llvm::ArrayRef<LoweredMode> loweredModes,
    const FiniteCodebookEncoding *codebook, std::size_t inactiveMode,
    const FabricOperationLeafStateLayout &layout,
    const ParallelizeGroups &groups) {
  const auto *valueField =
      layout.find(FabricOperationLeafStateFieldKind::BufferedValue);
  const auto *maskField =
      layout.find(FabricOperationLeafStateFieldKind::BufferedMask);
  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        const std::array<mlir::Value, 2> inputValid{
            accessor.getInput("valid_input_0"),
            accessor.getInput("valid_input_1")};
        const std::array<mlir::Value, 3> outputReady{
            accessor.getInput("ready_output_0"),
            accessor.getInput("ready_output_1"),
            accessor.getInput("ready_output_2")};
        mlir::Value state = accessor.getInput("state_current");
        mlir::Value stateValue =
            extractField(bodyBuilder, location, state, *valueField);
        mlir::Value stateMask =
            extractField(bodyBuilder, location, state, *maskField);
        mlir::Value continuation = accessor.getInput("continuation_current");
        mlir::Value notContinuation =
            invert(bodyBuilder, location, continuation);
        mlir::Value phase = circt::comb::ExtractOp::create(
            bodyBuilder, location, accessor.getInput("data_input_1"), 0, 1);
        mlir::Value phaseTrue = andAll(bodyBuilder, location,
                                       {inputValid[1], phase, notContinuation});
        mlir::Value phaseClose =
            andAll(bodyBuilder, location,
                   {inputValid[1], invert(bodyBuilder, location, phase),
                    notContinuation});

        std::vector<ParallelizeModeLogic> materialized;
        materialized.reserve(loweredModes.size());
        for (const LoweredMode &mode : loweredModes)
          materialized.push_back(materializeParallelizeMode(
              bodyBuilder, location, mode, accessor.getInput("data_input_0"),
              stateValue, stateMask, valueField->bitCount,
              maskField->bitCount));
        std::vector<mlir::Value> selected = materializeModeSelection(
            bodyBuilder, location, accessor, request, modes, codebook);
        const auto select = [&](auto member) {
          std::vector<mlir::Value> values;
          values.reserve(materialized.size());
          for (const auto &mode : materialized)
            values.push_back(mode.*member);
          return selectModeValue(bodyBuilder, location, values, inactiveMode,
                                 selected);
        };
        mlir::Value hasBuffered = select(&ParallelizeModeLogic::hasBuffered);
        mlir::Value fullAfter = select(&ParallelizeModeLogic::fullAfterAppend);
        mlir::Value appendedValue =
            select(&ParallelizeModeLogic::appendedValue);
        mlir::Value appendedMask = select(&ParallelizeModeLogic::appendedMask);
        mlir::Value fullSelected = andAll(
            bodyBuilder, location, {phaseTrue, inputValid[0], fullAfter});
        mlir::Value accumulateSelected =
            andAll(bodyBuilder, location,
                   {phaseTrue, inputValid[0],
                    invert(bodyBuilder, location, fullAfter)});
        mlir::Value tailSelected =
            andAll(bodyBuilder, location, {phaseClose, hasBuffered});
        mlir::Value emptyCloseSelected =
            andAll(bodyBuilder, location,
                   {phaseClose, invert(bodyBuilder, location, hasBuffered)});
        mlir::Value terminalSelected = continuation;
        mlir::Value payloadCapacity =
            groupCapacity(bodyBuilder, location, groups.payload, outputReady);
        mlir::Value terminalCapacity =
            groupCapacity(bodyBuilder, location, groups.terminal, outputReady);
        mlir::Value fullFire =
            andAll(bodyBuilder, location, {fullSelected, payloadCapacity});
        mlir::Value tailFire =
            andAll(bodyBuilder, location, {tailSelected, payloadCapacity});
        mlir::Value emptyCloseFire = andAll(
            bodyBuilder, location, {emptyCloseSelected, terminalCapacity});
        mlir::Value terminalFire =
            andAll(bodyBuilder, location, {terminalSelected, terminalCapacity});

        mlir::Value trueInputCapacity = circt::comb::MuxOp::create(
            bodyBuilder, location, fullAfter, payloadCapacity,
            bitConstant(bodyBuilder, location, true), true);
        accessor.setOutput(
            "ready_input_0",
            andAll(bodyBuilder, location,
                   {notContinuation, inputValid[1], phase, trueInputCapacity}));
        mlir::Value closeCapacity =
            circt::comb::MuxOp::create(bodyBuilder, location, hasBuffered,
                                       payloadCapacity, terminalCapacity, true);
        mlir::Value readyPhaseTrue =
            andAll(bodyBuilder, location,
                   {notContinuation, inputValid[0], phase, trueInputCapacity});
        mlir::Value readyPhaseClose =
            andAll(bodyBuilder, location,
                   {notContinuation, invert(bodyBuilder, location, phase),
                    closeCapacity});
        accessor.setOutput(
            "ready_input_1",
            orAll(bodyBuilder, location, {readyPhaseTrue, readyPhaseClose}));

        for (std::uint32_t ordinal = 0; ordinal != ports.outputs.size();
             ++ordinal) {
          mlir::Value payloadValid = groupValid(
              bodyBuilder, location,
              orAll(bodyBuilder, location, {fullSelected, tailSelected}),
              groups.payload, ordinal, outputReady);
          mlir::Value terminalValid =
              groupValid(bodyBuilder, location,
                         orAll(bodyBuilder, location,
                               {emptyCloseSelected, terminalSelected}),
                         groups.terminal, ordinal, outputReady);
          accessor.setOutput(
              "valid_output_" + std::to_string(ordinal),
              orAll(bodyBuilder, location, {payloadValid, terminalValid}));
        }
        mlir::Value vectorOutput =
            circt::comb::MuxOp::create(bodyBuilder, location, fullSelected,
                                       appendedValue, stateValue, true);
        mlir::Value maskOutput = circt::comb::MuxOp::create(
            bodyBuilder, location, fullSelected, appendedMask, stateMask, true);
        accessor.setOutput(
            "data_output_0",
            detail::resizeUnsigned(bodyBuilder, location, vectorOutput,
                                   ports.outputs[0]->payloadWidthBits));
        accessor.setOutput(
            "data_output_1",
            detail::resizeUnsigned(bodyBuilder, location, maskOutput,
                                   ports.outputs[1]->payloadWidthBits));
        mlir::Value groupPhase =
            invert(bodyBuilder, location,
                   orAll(bodyBuilder, location,
                         {emptyCloseSelected, terminalSelected}));
        accessor.setOutput(
            "data_output_2",
            detail::resizeUnsigned(bodyBuilder, location, groupPhase,
                                   ports.outputs[2]->payloadWidthBits));
        accessor.setOutput(
            "final_production",
            orAll(bodyBuilder, location,
                  {fullSelected, emptyCloseSelected, terminalSelected}));

        mlir::Value zeroState = constant(
            bodyBuilder, location, llvm::APInt(layout.encodedBitCount(), 0));
        mlir::Value accumulatedState =
            encodeState(bodyBuilder, location, appendedValue, appendedMask);
        mlir::Value stateNext = circt::comb::MuxOp::create(
            bodyBuilder, location, accumulateSelected, accumulatedState, state,
            true);
        mlir::Value clearsState =
            orAll(bodyBuilder, location,
                  {fullFire, tailFire, emptyCloseFire, terminalFire});
        stateNext = circt::comb::MuxOp::create(
            bodyBuilder, location, clearsState, zeroState, stateNext, true);
        accessor.setOutput("state_next", stateNext);
        accessor.setOutput(
            "state_write",
            orAll(bodyBuilder, location, {accumulateSelected, clearsState}));
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

llvm::Expected<FabricOperationProviderOutput> materializeSerialize(
    FabricOperationProviderRequest request, const PhysicalPorts &ports,
    llvm::ArrayRef<Mode> modes, llvm::ArrayRef<LoweredMode> loweredModes,
    const FiniteCodebookEncoding *codebook, std::size_t inactiveMode,
    const FabricOperationLeafStateLayout &layout,
    const SerializeGroups &groups) {
  const auto *valueField =
      layout.find(FabricOperationLeafStateFieldKind::BufferedValue);
  const auto *maskField =
      layout.find(FabricOperationLeafStateFieldKind::BufferedMask);
  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        const std::array<mlir::Value, 3> inputValid{
            accessor.getInput("valid_input_0"),
            accessor.getInput("valid_input_1"),
            accessor.getInput("valid_input_2")};
        const std::array<mlir::Value, 2> outputReady{
            accessor.getInput("ready_output_0"),
            accessor.getInput("ready_output_1")};
        mlir::Value state = accessor.getInput("state_current");
        mlir::Value stateValue =
            extractField(bodyBuilder, location, state, *valueField);
        mlir::Value stateMask =
            extractField(bodyBuilder, location, state, *maskField);
        mlir::Value continuation = accessor.getInput("continuation_current");
        mlir::Value notContinuation =
            invert(bodyBuilder, location, continuation);
        mlir::Value phase = circt::comb::ExtractOp::create(
            bodyBuilder, location, accessor.getInput("data_input_2"), 0, 1);
        mlir::Value inputValue = detail::resizeUnsigned(
            bodyBuilder, location, accessor.getInput("data_input_0"),
            valueField->bitCount);
        mlir::Value inputMask = detail::resizeUnsigned(
            bodyBuilder, location, accessor.getInput("data_input_1"),
            maskField->bitCount);
        mlir::Value sourceValue = circt::comb::MuxOp::create(
            bodyBuilder, location, continuation, stateValue, inputValue, true);
        mlir::Value sourceMask = circt::comb::MuxOp::create(
            bodyBuilder, location, continuation, stateMask, inputMask, true);

        std::vector<SerializeModeLogic> materialized;
        materialized.reserve(loweredModes.size());
        for (const LoweredMode &mode : loweredModes)
          materialized.push_back(materializeSerializeMode(
              bodyBuilder, location, mode, sourceValue, sourceMask,
              maskField->bitCount, ports.outputs[0]->payloadWidthBits));
        std::vector<mlir::Value> selected = materializeModeSelection(
            bodyBuilder, location, accessor, request, modes, codebook);
        const auto select = [&](auto member) {
          std::vector<mlir::Value> values;
          values.reserve(materialized.size());
          for (const auto &mode : materialized)
            values.push_back(mode.*member);
          return selectModeValue(bodyBuilder, location, values, inactiveMode,
                                 selected);
        };
        mlir::Value hasActive = select(&SerializeModeLogic::hasActive);
        mlir::Value hasRemaining = select(&SerializeModeLogic::hasRemaining);
        mlir::Value selectedData = select(&SerializeModeLogic::selectedData);
        mlir::Value clearedMask = select(&SerializeModeLogic::clearedMask);
        mlir::Value allInputsValid =
            andAll(bodyBuilder, location,
                   {inputValid[0], inputValid[1], inputValid[2]});
        mlir::Value newTrue = andAll(bodyBuilder, location,
                                     {notContinuation, allInputsValid, phase});
        mlir::Value newActive =
            andAll(bodyBuilder, location, {newTrue, hasActive});
        mlir::Value zeroCommit =
            andAll(bodyBuilder, location,
                   {newTrue, invert(bodyBuilder, location, hasActive)});
        mlir::Value continuedActive =
            andAll(bodyBuilder, location, {continuation, hasActive});
        mlir::Value itemSelected =
            orAll(bodyBuilder, location, {newActive, continuedActive});
        mlir::Value closeSelected =
            andAll(bodyBuilder, location,
                   {notContinuation, inputValid[2],
                    invert(bodyBuilder, location, phase)});
        mlir::Value itemCapacity =
            groupCapacity(bodyBuilder, location, groups.item, outputReady);
        mlir::Value closeCapacity =
            groupCapacity(bodyBuilder, location, groups.close, outputReady);
        mlir::Value itemFire =
            andAll(bodyBuilder, location, {itemSelected, itemCapacity});
        mlir::Value closeFire =
            andAll(bodyBuilder, location, {closeSelected, closeCapacity});
        mlir::Value activeCapacity = circt::comb::MuxOp::create(
            bodyBuilder, location, hasActive, itemCapacity,
            bitConstant(bodyBuilder, location, true), true);

        accessor.setOutput("ready_input_0",
                           andAll(bodyBuilder, location,
                                  {notContinuation, inputValid[1],
                                   inputValid[2], phase, activeCapacity}));
        accessor.setOutput("ready_input_1",
                           andAll(bodyBuilder, location,
                                  {notContinuation, inputValid[0],
                                   inputValid[2], phase, activeCapacity}));
        mlir::Value readyTrue = andAll(bodyBuilder, location,
                                       {notContinuation, inputValid[0],
                                        inputValid[1], phase, activeCapacity});
        mlir::Value readyClose =
            andAll(bodyBuilder, location,
                   {notContinuation, invert(bodyBuilder, location, phase),
                    closeCapacity});
        accessor.setOutput("ready_input_2", orAll(bodyBuilder, location,
                                                  {readyTrue, readyClose}));

        for (std::uint32_t ordinal = 0; ordinal != ports.outputs.size();
             ++ordinal) {
          mlir::Value itemValid =
              groupValid(bodyBuilder, location, itemSelected, groups.item,
                         ordinal, outputReady);
          mlir::Value closeValid =
              groupValid(bodyBuilder, location, closeSelected, groups.close,
                         ordinal, outputReady);
          accessor.setOutput(
              "valid_output_" + std::to_string(ordinal),
              orAll(bodyBuilder, location, {itemValid, closeValid}));
        }
        accessor.setOutput("data_output_0", selectedData);
        accessor.setOutput(
            "data_output_1",
            detail::resizeUnsigned(bodyBuilder, location, itemSelected,
                                   ports.outputs[1]->payloadWidthBits));
        mlir::Value finalItem =
            andAll(bodyBuilder, location,
                   {itemSelected, invert(bodyBuilder, location, hasRemaining)});
        accessor.setOutput("final_production",
                           orAll(bodyBuilder, location,
                                 {finalItem, closeSelected, zeroCommit}));

        mlir::Value zeroState = constant(
            bodyBuilder, location, llvm::APInt(layout.encodedBitCount(), 0));
        mlir::Value remainingState =
            encodeState(bodyBuilder, location, sourceValue, clearedMask);
        mlir::Value itemState =
            circt::comb::MuxOp::create(bodyBuilder, location, hasRemaining,
                                       remainingState, zeroState, true);
        mlir::Value stateNext = circt::comb::MuxOp::create(
            bodyBuilder, location, itemFire, itemState, state, true);
        mlir::Value clearsState =
            orAll(bodyBuilder, location, {closeFire, zeroCommit});
        stateNext = circt::comb::MuxOp::create(
            bodyBuilder, location, clearsState, zeroState, stateNext, true);
        accessor.setOutput("state_next", stateNext);
        accessor.setOutput("state_write", orAll(bodyBuilder, location,
                                                {itemFire, clearsState}));
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

llvm::Expected<FabricOperationProviderOutput>
materializeAdapter(FabricOperationProviderRequest request,
                   ::fabric::ImplementationFamilyId expectedFamily,
                   Schema schema, std::size_t inputCount,
                   std::size_t outputCount) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily != expectedFamily)
    return invalid("provider received a different implementation family");
  const auto *parameters = std::get_if<::fabric::FixedVectorAdapterParams>(
      &request.capability.parameterizedCapability);
  if (!parameters)
    return invalid("capability has the wrong parameter schema");
  if (request.capability.enabledOperationSchemas != std::vector<Schema>{schema})
    return invalid("capability does not contain its exact adapter schema");
  auto maximumLanes = ::fabric::maximumFixedVectorAdapterLaneCount(*parameters);
  if (!maximumLanes)
    return maximumLanes.takeError();
  auto exactContract = ::fabric::isOrderedCardinalityOperationResourceContract(
      request.capability.resourceStateAndTimingContract, schema, *maximumLanes);
  if (!exactContract)
    return exactContract.takeError();
  if (!*exactContract)
    return unsupported(request);
  auto ports = derivePhysicalPorts(request.capability, inputCount, outputCount);
  if (!ports)
    return unsupported(request);
  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return std::move(error);

  auto stateLayout = deriveFabricOperationLeafStateLayout(request.capability);
  if (!stateLayout)
    return stateLayout.takeError();
  if (!*stateLayout)
    return invalid("ordered adapter has no state layout");
  const auto *valueField =
      (*stateLayout)->find(FabricOperationLeafStateFieldKind::BufferedValue);
  const auto *maskField =
      (*stateLayout)->find(FabricOperationLeafStateFieldKind::BufferedMask);
  if (!valueField || !maskField || (*stateLayout)->fields.size() != 2 ||
      valueField->bitOffset != 0 || valueField->bitCount == 0 ||
      maskField->bitOffset != valueField->bitCount ||
      maskField->bitCount != *maximumLanes ||
      (*stateLayout)->encodedBitCount() !=
          valueField->bitCount + maskField->bitCount)
    return invalid("adapter state is not packed value then mask");

  const FiniteCodebookEncoding *codebook = nullptr;
  std::size_t inactiveMode = 0;
  auto modes = deriveModes(request, codebook, inactiveMode);
  if (!modes)
    return modes.takeError();
  std::vector<LoweredMode> loweredModes;
  loweredModes.reserve(modes->size());
  for (const Mode &mode : *modes) {
    auto lowered = lowerMode(mode, schema);
    if (!lowered)
      return lowered.takeError();
    const std::uint64_t payloadWidth =
        static_cast<std::uint64_t>(lowered->elementWidth) * lowered->laneCount;
    if (lowered->elementWidth == 0 || lowered->laneCount == 0 ||
        lowered->laneCount > maskField->bitCount ||
        payloadWidth > valueField->bitCount)
      return invalid("adapter behavior exceeds its derived state carrier");
    if (schema == Schema::DataflowParallelize) {
      if (lowered->elementWidth > ports->inputs[0]->payloadWidthBits ||
          payloadWidth > ports->outputs[0]->payloadWidthBits ||
          lowered->laneCount > ports->outputs[1]->payloadWidthBits ||
          ports->inputs[1]->payloadWidthBits < 1 ||
          ports->outputs[2]->payloadWidthBits < 1)
        return unsupported(request);
    } else if (payloadWidth > ports->inputs[0]->payloadWidthBits ||
               lowered->laneCount > ports->inputs[1]->payloadWidthBits ||
               ports->inputs[2]->payloadWidthBits < 1 ||
               lowered->elementWidth > ports->outputs[0]->payloadWidthBits ||
               ports->outputs[1]->payloadWidthBits < 1) {
      return unsupported(request);
    }
    loweredModes.push_back(*lowered);
  }

  if (schema == Schema::DataflowParallelize) {
    auto groups = deriveParallelizeGroups();
    if (!groups)
      return groups.takeError();
    return materializeParallelize(std::move(request), *ports, *modes,
                                  loweredModes, codebook, inactiveMode,
                                  **stateLayout, *groups);
  }
  auto groups = deriveSerializeGroups();
  if (!groups)
    return groups.takeError();
  return materializeSerialize(std::move(request), *ports, *modes, loweredModes,
                              codebook, inactiveMode, **stateLayout, *groups);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFixedVectorParallelize(
    FabricOperationProviderRequest request) {
  return materializeAdapter(
      std::move(request),
      ::fabric::ImplementationFamilyId::FixedVectorParallelize,
      Schema::DataflowParallelize, 2, 3);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFixedVectorSerialize(
    FabricOperationProviderRequest request) {
  return materializeAdapter(
      std::move(request),
      ::fabric::ImplementationFamilyId::FixedVectorSerialize,
      Schema::DataflowSerialize, 3, 2);
}

} // namespace

llvm::Error registerPortableFixedVectorParallelizeSerializeProviders(
    FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry candidate = registry;
  const std::array registrations = {
      FabricOperationProviderRegistration{
          ::fabric::ImplementationFamilyId::FixedVectorParallelize,
          BackendRecipeKey::PortableSystemVerilog,
          {},
          materializePortableFixedVectorParallelize},
      FabricOperationProviderRegistration{
          ::fabric::ImplementationFamilyId::FixedVectorSerialize,
          BackendRecipeKey::PortableSystemVerilog,
          {},
          materializePortableFixedVectorSerialize}};
  for (const FabricOperationProviderRegistration &registration : registrations)
    if (llvm::Error error = candidate.add(registration))
      return error;
  registry = std::move(candidate);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl
