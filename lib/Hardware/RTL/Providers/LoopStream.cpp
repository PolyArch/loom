#include "Hardware/RTL/Providers/LoopStream.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

using ActorHandshakeCase = ::dataflow::semantics::ActorHandshakeCase;
using StreamCase = ::dataflow::semantics::StreamCase;
using StreamCaseDescriptor = ::dataflow::semantics::StreamCaseDescriptor;
using StreamMode = ::dataflow::semantics::StreamMode;
using Predicate = mlir::arith::CmpIPredicate;

struct CaseShape final {
  StreamCase transition;
  StreamCaseDescriptor descriptor;
  ActorHandshakeCase handshake;
};

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  unsigned width = 0;
  circt::comb::ICmpPredicate predicate = circt::comb::ICmpPredicate::eq;
};

struct MaterializedMode final {
  mlir::Value continues;
  mlir::Value currentOutput;
  mlir::Value stepped;
  mlir::Value activationCurrent;
  mlir::Value activationLimit;
  mlir::Value activationStep;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_loop_stream_invalid: " + message);
}

llvm::Expected<FabricOperationProviderOutput>
unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

mlir::Value bitConstant(mlir::OpBuilder &builder, mlir::Location location,
                        bool value) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(1, value));
}

mlir::Value streamModeConstant(mlir::OpBuilder &builder,
                               mlir::Location location, StreamMode mode) {
  return bitConstant(builder, location, mode == StreamMode::Running);
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

bool contains(llvm::ArrayRef<std::uint32_t> ordinals, std::uint32_t ordinal) {
  return llvm::is_contained(ordinals, ordinal);
}

llvm::Expected<std::vector<CaseShape>> deriveCaseShapes() {
  auto projected = ::dataflow::semantics::projectActorHandshakeCases(
      ::dataflow::OperationSchemaId::DataflowStream, 3, 2);
  if (!projected)
    return projected.takeError();
  constexpr std::array transitions{
      StreamCase::StartTrue, StreamCase::StartClose, StreamCase::ContinueTrue,
      StreamCase::ContinueClose};
  if (projected->size() != transitions.size())
    return invalid("dataflow.stream does not project four transition cases");

  std::vector<CaseShape> result;
  result.reserve(transitions.size());
  for (StreamCase transition : transitions) {
    const std::uint32_t ordinal = static_cast<std::uint32_t>(transition);
    const auto projectedCase =
        llvm::find_if(*projected, [&](const ActorHandshakeCase &candidate) {
          return candidate.ordinal == ordinal;
        });
    if (projectedCase == projected->end())
      return invalid("projected stream cases omit a schema transition");
    result.push_back({transition,
                      ::dataflow::semantics::streamCaseDescriptor(transition),
                      *projectedCase});
  }

  const CaseShape &startTrue = result[0];
  const CaseShape &startClose = result[1];
  const CaseShape &continueTrue = result[2];
  const CaseShape &continueClose = result[3];
  const auto samePublication = [](const CaseShape &lhs, const CaseShape &rhs) {
    return lhs.handshake.activeResults == rhs.handshake.activeResults &&
           lhs.descriptor.ivSource == rhs.descriptor.ivSource &&
           lhs.descriptor.emitPhase == rhs.descriptor.emitPhase &&
           lhs.descriptor.phase == rhs.descriptor.phase &&
           lhs.descriptor.nextMode == rhs.descriptor.nextMode;
  };
  if (startTrue.handshake.consumedInputs !=
          startClose.handshake.consumedInputs ||
      startTrue.descriptor.requiredMode != startClose.descriptor.requiredMode ||
      continueTrue.descriptor.requiredMode !=
          continueClose.descriptor.requiredMode ||
      !samePublication(startTrue, continueTrue) ||
      !samePublication(startClose, continueClose))
    return invalid("registered stream timing cannot share its staged cases");
  return result;
}

const CaseShape &caseShape(llvm::ArrayRef<CaseShape> cases,
                           StreamCase transition) {
  const auto found = llvm::find_if(cases, [&](const CaseShape &candidate) {
    return candidate.transition == transition;
  });
  assert(found != cases.end());
  return *found;
}

llvm::Expected<circt::comb::ICmpPredicate> lowerPredicate(Predicate predicate) {
  using CombPredicate = circt::comb::ICmpPredicate;
  switch (predicate) {
  case Predicate::eq:
    return CombPredicate::eq;
  case Predicate::ne:
    return CombPredicate::ne;
  case Predicate::slt:
    return CombPredicate::slt;
  case Predicate::sle:
    return CombPredicate::sle;
  case Predicate::sgt:
    return CombPredicate::sgt;
  case Predicate::sge:
    return CombPredicate::sge;
  case Predicate::ult:
    return CombPredicate::ult;
  case Predicate::ule:
    return CombPredicate::ule;
  case Predicate::ugt:
    return CombPredicate::ugt;
  case Predicate::uge:
    return CombPredicate::uge;
  }
  return invalid("continuation predicate is outside the closed domain");
}

llvm::Expected<LoweredMode>
lowerMode(const Mode &mode, ::dataflow::StreamStepKind fixedStepKind) {
  if (mode.actor.schema != ::dataflow::OperationSchemaId::DataflowStream ||
      mode.actor.type.getNumInputs() != 3 ||
      mode.actor.type.getNumResults() != 2)
    return invalid("behavior witness is not a dataflow.stream actor");
  auto recurrence =
      llvm::dyn_cast<mlir::IntegerType>(mode.actor.type.getInput(0));
  if (!recurrence || recurrence.getWidth() == 0 ||
      mode.actor.type.getInput(1) != recurrence ||
      mode.actor.type.getInput(2) != recurrence ||
      mode.actor.type.getResult(0) != recurrence ||
      !mode.actor.type.getResult(1).isInteger(1))
    return invalid("stream behavior is not uniformly recurrence typed");
  const auto *payload =
      std::get_if<::dataflow::StreamRecurrencePayload>(&mode.actor.payload);
  if (!payload || payload->stepKind != fixedStepKind)
    return invalid("behavior witness changed the fixed step kind");
  auto predicate = lowerPredicate(payload->predicate);
  if (!predicate)
    return predicate.takeError();
  return LoweredMode{recurrence.getWidth(), *predicate};
}

mlir::Value materializeStep(mlir::OpBuilder &builder, mlir::Location location,
                            ::dataflow::StreamStepKind kind,
                            mlir::Value current, mlir::Value step) {
  switch (kind) {
  case ::dataflow::StreamStepKind::Add:
    return circt::comb::AddOp::create(builder, location,
                                      mlir::ValueRange{current, step}, true);
  case ::dataflow::StreamStepKind::Sub:
    return circt::comb::SubOp::create(builder, location, current, step, true);
  case ::dataflow::StreamStepKind::Mul:
    return circt::comb::MulOp::create(builder, location,
                                      mlir::ValueRange{current, step}, true);
  case ::dataflow::StreamStepKind::SDiv:
    return circt::comb::DivSOp::create(builder, location, current, step, true);
  case ::dataflow::StreamStepKind::UDiv:
    return circt::comb::DivUOp::create(builder, location, current, step, true);
  case ::dataflow::StreamStepKind::ShL:
    return circt::comb::ShlOp::create(builder, location, current, step, true);
  case ::dataflow::StreamStepKind::AShr:
    return circt::comb::ShrSOp::create(builder, location, current, step, true);
  case ::dataflow::StreamStepKind::LShr:
    return circt::comb::ShrUOp::create(builder, location, current, step, true);
  }
  llvm_unreachable("unknown dataflow.stream step kind");
}

mlir::Value extractField(mlir::OpBuilder &builder, mlir::Location location,
                         mlir::Value state,
                         const FabricOperationLeafStateFieldLayout &field) {
  return circt::comb::ExtractOp::create(builder, location, state,
                                        field.bitOffset, field.bitCount);
}

mlir::Value encodeState(mlir::OpBuilder &builder, mlir::Location location,
                        mlir::Value mode, mlir::Value current,
                        mlir::Value limit, mlir::Value step) {
  return circt::comb::ConcatOp::create(
      builder, location, mlir::ValueRange{step, limit, current, mode});
}

mlir::Value selectModeValue(mlir::OpBuilder &builder, mlir::Location location,
                            llvm::ArrayRef<mlir::Value> values,
                            std::size_t inactiveMode,
                            llvm::ArrayRef<mlir::Value> selected) {
  mlir::Value result = values[inactiveMode];
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index == inactiveMode)
      continue;
    result = circt::comb::MuxOp::create(builder, location, selected[index],
                                        values[index], result, true);
  }
  return result;
}

MaterializedMode materializeMode(
    mlir::OpBuilder &builder, mlir::Location location,
    circt::hw::HWModulePortAccessor &accessor, const LoweredMode &mode,
    ::dataflow::StreamStepKind fixedStepKind, mlir::Value stateCurrent,
    mlir::Value stateLimit, mlir::Value stateStep, unsigned stateValueWidth,
    unsigned outputWidth) {
  mlir::Value current =
      detail::resizeUnsigned(builder, location, stateCurrent, mode.width);
  mlir::Value limit =
      detail::resizeUnsigned(builder, location, stateLimit, mode.width);
  mlir::Value step =
      detail::resizeUnsigned(builder, location, stateStep, mode.width);
  mlir::Value continues = circt::comb::ICmpOp::create(
      builder, location, mode.predicate, current, limit, true);
  mlir::Value stepped =
      materializeStep(builder, location, fixedStepKind, current, step);
  const auto activation = [&](unsigned ordinal) {
    return detail::resizeUnsigned(
        builder, location,
        detail::resizeUnsigned(
            builder, location,
            accessor.getInput("data_input_" + std::to_string(ordinal)),
            mode.width),
        stateValueWidth);
  };
  return {continues,
          detail::resizeUnsigned(builder, location, current, outputWidth),
          detail::resizeUnsigned(builder, location, stepped, stateValueWidth),
          activation(0),
          activation(1),
          activation(2)};
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableLoopStream(FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::LoopStream)
    return invalid("provider received a different implementation family");
  const auto *parameters = std::get_if<::fabric::LoopStreamParams>(
      &request.capability.parameterizedCapability);
  if (!parameters || !parameters->integerWidths.valid() ||
      parameters->integerWidths.empty() ||
      !parameters->continuationPredicates.valid() ||
      parameters->continuationPredicates.empty())
    return invalid("capability has malformed stream parameters");
  if (request.capability.enabledOperationSchemas !=
      std::vector<::dataflow::OperationSchemaId>{
          ::dataflow::OperationSchemaId::DataflowStream})
    return invalid("capability does not contain exactly dataflow.stream");

  auto actualContract = ::fabric::encodeResourceContractRecord(
      request.capability.resourceStateAndTimingContract);
  if (!actualContract)
    return actualContract.takeError();
  auto supportedContract = ::fabric::encodeResourceContractRecord(
      ::fabric::loopStreamOperationResourceContract());
  if (!supportedContract)
    return supportedContract.takeError();
  if (*actualContract != *supportedContract)
    return unsupported(request);

  auto cases = deriveCaseShapes();
  if (!cases)
    return cases.takeError();
  const CaseShape &startTrue = caseShape(*cases, StreamCase::StartTrue);
  const CaseShape &continueTrue = caseShape(*cases, StreamCase::ContinueTrue);
  const CaseShape &continueClose = caseShape(*cases, StreamCase::ContinueClose);

  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> outputs;
  for (const fabric::ResolvedFabricOpPhysicalPortView &port :
       request.capability.physicalPorts) {
    if (port.reference.direction == fabric::FabricPortDirection::Input)
      inputs.push_back(&port);
    else if (port.reference.direction == fabric::FabricPortDirection::Output)
      outputs.push_back(&port);
    else
      return invalid("capability has a physical port with unknown direction");
  }
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(inputs, byOrdinal);
  llvm::sort(outputs, byOrdinal);
  if (inputs.size() != 3 || outputs.size() != 2 ||
      inputs[0]->reference.ordinal != 0 || inputs[1]->reference.ordinal != 1 ||
      inputs[2]->reference.ordinal != 2 || outputs[0]->reference.ordinal != 0 ||
      outputs[1]->reference.ordinal != 1)
    return unsupported(request);
  if (llvm::any_of(
          inputs,
          [](const auto *port) { return port->payloadWidthBits == 0; }) ||
      outputs[0]->payloadWidthBits == 0 || outputs[1]->payloadWidthBits == 0)
    return invalid("capability has a zero-width stream port");

  auto layout = deriveFabricOperationLeafStateLayout(request.capability);
  if (!layout)
    return layout.takeError();
  if (!*layout)
    return invalid("capability has no stream state layout");
  const FabricOperationLeafStateLayout &stateLayout = **layout;
  const auto *modeField =
      stateLayout.find(FabricOperationLeafStateFieldKind::Mode);
  const auto *currentField =
      stateLayout.find(FabricOperationLeafStateFieldKind::Current);
  const auto *limitField =
      stateLayout.find(FabricOperationLeafStateFieldKind::Limit);
  const auto *stepField =
      stateLayout.find(FabricOperationLeafStateFieldKind::Step);
  if (!modeField || !currentField || !limitField || !stepField ||
      stateLayout.fields.size() != 4 || modeField->bitOffset != 0 ||
      modeField->bitCount != 1 || currentField->bitOffset != 1 ||
      currentField->bitCount == 0 ||
      limitField->bitOffset != 1 + currentField->bitCount ||
      limitField->bitCount != currentField->bitCount ||
      stepField->bitOffset != 1 + 2 * currentField->bitCount ||
      stepField->bitCount != currentField->bitCount ||
      stateLayout.encodedBitCount() != 1 + 3 * currentField->bitCount)
    return invalid("stream state is not packed mode-current-limit-step");
  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return std::move(error);

  auto relation = request.capability.resolveSemanticFieldRelation(
      *request.leaf.getContext());
  if (!relation)
    return relation.takeError();
  const auto domain = relation->finiteBehaviorDomain();
  if (domain.empty())
    return invalid("Fabric returned an empty stream behavior domain");

  const ConfigurationEncodingRelation *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  std::vector<Mode> modes;
  modes.reserve(domain.size());
  if (request.capability.configurationFieldSchema.empty()) {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None ||
        domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid(
          "configuration-free stream has a non-singleton behavior domain");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid("configured stream relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured stream capability requires one field");
    field = request.configurationAbi.findOperationEncodingRelation(
        request.occurrence,
        request.capability.configurationFieldSchema.front().ordinal);
    if (!field)
      return invalid("configured field is absent from the ABI");
    codebook = std::get_if<FiniteCodebookEncoding>(&field->semanticEncoding);
    if (!codebook)
      return invalid("configured field is not a finite codebook");
    if (codebook->entries.size() != domain.size())
      return invalid("codebook does not exactly cover the behavior domain");
    for (const auto &point : domain) {
      if (!point.semanticConfiguration)
        return invalid("configured behavior has no semantic value");
      const FiniteCodebookEntry *entry = detail::findFiniteCodebookEntry(
          *codebook, point.semanticConfiguration->bytes());
      if (!entry)
        return invalid("codebook has no admitted semantic value");
      modes.push_back({point.representativeActor, entry});
    }
  }

  std::size_t inactiveMode = 0;
  if (field) {
    const auto inactive = llvm::find_if(modes, [&](const Mode &mode) {
      return llvm::ArrayRef<std::uint8_t>(mode.codebookEntry->semanticValue)
          .equals(field->inactiveValue);
    });
    if (inactive == modes.end())
      return invalid("ABI inactive value is outside the behavior domain");
    inactiveMode = static_cast<std::size_t>(inactive - modes.begin());
  }

  std::vector<LoweredMode> loweredModes;
  loweredModes.reserve(modes.size());
  for (const Mode &mode : modes) {
    auto lowered = lowerMode(mode, parameters->fixedStepKind);
    if (!lowered)
      return lowered.takeError();
    if (lowered->width > currentField->bitCount ||
        llvm::any_of(inputs,
                     [&](const auto *port) {
                       return lowered->width > port->payloadWidthBits;
                     }) ||
        lowered->width > outputs[0]->payloadWidthBits)
      return invalid("stream behavior exceeds the physical datapath");
    loweredModes.push_back(*lowered);
  }

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
        mlir::Value currentState = accessor.getInput("state_current");
        mlir::Value currentMode =
            extractField(bodyBuilder, location, currentState, *modeField);
        const auto selectMode = [&](StreamMode mode) {
          mlir::Value encoded = streamModeConstant(bodyBuilder, location, mode);
          return mlir::Value(circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::eq,
              currentMode, encoded, true));
        };
        mlir::Value idle = selectMode(startTrue.descriptor.requiredMode);
        mlir::Value running = selectMode(continueTrue.descriptor.requiredMode);
        mlir::Value current =
            extractField(bodyBuilder, location, currentState, *currentField);
        mlir::Value limit =
            extractField(bodyBuilder, location, currentState, *limitField);
        mlir::Value step =
            extractField(bodyBuilder, location, currentState, *stepField);

        std::vector<MaterializedMode> materialized;
        materialized.reserve(loweredModes.size());
        for (const LoweredMode &mode : loweredModes)
          materialized.push_back(materializeMode(
              bodyBuilder, location, accessor, mode, parameters->fixedStepKind,
              current, limit, step, currentField->bitCount,
              outputs[0]->payloadWidthBits));

        std::vector<mlir::Value> selected(modes.size());
        if (field) {
          mlir::Value configuration = accessor.getInput(
              "config_" +
              std::to_string(
                  request.capability.configurationFieldSchema.front().ordinal));
          for (std::size_t index = 0; index < modes.size(); ++index) {
            mlir::Value code = circt::hw::ConstantOp::create(
                bodyBuilder, location,
                detail::decodePhysicalCode(
                    modes[index].codebookEntry->physicalCode,
                    codebook->encodedBitCount));
            selected[index] = circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::eq,
                configuration, code, true);
          }
        } else {
          selected.front() = bitConstant(bodyBuilder, location, true);
        }

        const auto select = [&](auto member) {
          std::vector<mlir::Value> values;
          values.reserve(materialized.size());
          for (const MaterializedMode &mode : materialized)
            values.push_back(mode.*member);
          return selectModeValue(bodyBuilder, location, values, inactiveMode,
                                 selected);
        };
        mlir::Value continues = select(&MaterializedMode::continues);
        mlir::Value currentOutput = select(&MaterializedMode::currentOutput);
        mlir::Value stepped = select(&MaterializedMode::stepped);
        mlir::Value activationCurrent =
            select(&MaterializedMode::activationCurrent);
        mlir::Value activationLimit =
            select(&MaterializedMode::activationLimit);
        mlir::Value activationStep = select(&MaterializedMode::activationStep);

        llvm::SmallVector<mlir::Value, 3> activationValids;
        for (std::uint32_t ordinal : startTrue.handshake.consumedInputs)
          activationValids.push_back(inputValid[ordinal]);
        mlir::Value allInputsValid =
            andAll(bodyBuilder, location, activationValids);
        mlir::Value accept =
            andAll(bodyBuilder, location, {idle, allInputsValid});
        for (unsigned ordinal = 0; ordinal < inputValid.size(); ++ordinal) {
          if (!contains(startTrue.handshake.consumedInputs, ordinal)) {
            accessor.setOutput("ready_input_" + std::to_string(ordinal),
                               bitConstant(bodyBuilder, location, false));
            continue;
          }
          llvm::SmallVector<mlir::Value, 3> readyTerms{idle};
          for (std::uint32_t other : startTrue.handshake.consumedInputs)
            if (other != ordinal)
              readyTerms.push_back(inputValid[other]);
          accessor.setOutput("ready_input_" + std::to_string(ordinal),
                             andAll(bodyBuilder, location, readyTerms));
        }

        mlir::Value trueSelected = circt::comb::AndOp::create(
            bodyBuilder, location, running, continues);
        mlir::Value closeSelected = circt::comb::AndOp::create(
            bodyBuilder, location, running,
            circt::comb::createOrFoldNot(bodyBuilder, location, continues));
        const auto outputCapacity = [&](const CaseShape &shape) {
          llvm::SmallVector<mlir::Value, 2> ready;
          for (std::uint32_t ordinal : shape.handshake.activeResults)
            ready.push_back(outputReady[ordinal]);
          return andAll(bodyBuilder, location, ready);
        };
        mlir::Value trueFire = circt::comb::AndOp::create(
            bodyBuilder, location, trueSelected, outputCapacity(continueTrue));
        mlir::Value closeFire =
            circt::comb::AndOp::create(bodyBuilder, location, closeSelected,
                                       outputCapacity(continueClose));

        for (unsigned ordinal = 0; ordinal != outputs.size(); ++ordinal) {
          llvm::SmallVector<mlir::Value, 2> validCases;
          const auto addValidCase = [&](mlir::Value selectedCase,
                                        const CaseShape &shape) {
            if (!contains(shape.handshake.activeResults, ordinal))
              return;
            llvm::SmallVector<mlir::Value, 3> terms{selectedCase};
            for (std::uint32_t other : shape.handshake.activeResults)
              if (other != ordinal)
                terms.push_back(outputReady[other]);
            validCases.push_back(andAll(bodyBuilder, location, terms));
          };
          addValidCase(trueSelected, continueTrue);
          addValidCase(closeSelected, continueClose);
          accessor.setOutput("valid_output_" + std::to_string(ordinal),
                             orAll(bodyBuilder, location, validCases));
        }
        accessor.setOutput("data_output_0", currentOutput);
        mlir::Value phase = circt::comb::MuxOp::create(
            bodyBuilder, location, trueSelected,
            bitConstant(bodyBuilder, location, continueTrue.descriptor.phase),
            bitConstant(bodyBuilder, location, continueClose.descriptor.phase),
            true);
        accessor.setOutput("data_output_1", detail::resizeUnsigned(
                                                bodyBuilder, location, phase,
                                                outputs[1]->payloadWidthBits));

        mlir::Value zeroValue = circt::hw::ConstantOp::create(
            bodyBuilder, location, llvm::APInt(currentField->bitCount, 0));
        mlir::Value acceptedState = encodeState(
            bodyBuilder, location,
            streamModeConstant(bodyBuilder, location,
                               continueTrue.descriptor.requiredMode),
            activationCurrent, activationLimit, activationStep);
        mlir::Value advancedState =
            encodeState(bodyBuilder, location,
                        streamModeConstant(bodyBuilder, location,
                                           continueTrue.descriptor.nextMode),
                        stepped, limit, step);
        mlir::Value closedState =
            encodeState(bodyBuilder, location,
                        streamModeConstant(bodyBuilder, location,
                                           continueClose.descriptor.nextMode),
                        zeroValue, zeroValue, zeroValue);
        mlir::Value nextState = circt::comb::MuxOp::create(
            bodyBuilder, location, accept, acceptedState, currentState, true);
        nextState = circt::comb::MuxOp::create(bodyBuilder, location, trueFire,
                                               advancedState, nextState, true);
        nextState = circt::comb::MuxOp::create(bodyBuilder, location, closeFire,
                                               closedState, nextState, true);
        accessor.setOutput("state_next", nextState);
        accessor.setOutput("state_write", orAll(bodyBuilder, location,
                                                {accept, trueFire, closeFire}));
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

} // namespace

llvm::Error
registerPortableLoopStreamProvider(FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::LoopStream,
                       BackendRecipeKey::PortableSystemVerilog,
                       {},
                       materializePortableLoopStream});
}

} // namespace loom::hardware::rtl
