#include "Hardware/RTL/Providers/LoopInvariant.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

using InvariantCase = ::dataflow::semantics::InvariantCase;
using InvariantCaseDescriptor = ::dataflow::semantics::InvariantCaseDescriptor;
using InvariantInput = ::dataflow::semantics::InvariantInput;
using InvariantOutputSource = ::dataflow::semantics::InvariantOutputSource;
using InvariantSemanticState = ::dataflow::semantics::InvariantSemanticState;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_loop_invariant_invalid: " + message);
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

struct MaterializedCase final {
  InvariantCaseDescriptor descriptor;
  mlir::Value selected;
  mlir::Value allInputsValid;
  mlir::Value fire;
};

llvm::Expected<FabricOperationProviderOutput>
materializePortableLoopInvariant(FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::LoopInvariant)
    return invalid("provider received a different implementation family");
  if (!std::holds_alternative<::fabric::TokenPlaneParams>(
          request.capability.parameterizedCapability))
    return invalid("capability has the wrong parameter schema");
  if (request.capability.enabledOperationSchemas !=
      std::vector<::dataflow::OperationSchemaId>{
          ::dataflow::OperationSchemaId::DataflowInvariant})
    return invalid("capability does not contain exactly dataflow.invariant");
  if (!request.capability.configurationFieldSchema.empty())
    return invalid("loop invariant capability has semantic configuration "
                   "fields");

  auto actualContract = ::fabric::encodeResourceContractRecord(
      request.capability.resourceStateAndTimingContract);
  if (!actualContract)
    return actualContract.takeError();
  auto supportedContract = ::fabric::encodeResourceContractRecord(
      ::fabric::loopInvariantOperationResourceContract());
  if (!supportedContract)
    return supportedContract.takeError();
  if (*actualContract != *supportedContract)
    return unsupported(request);

  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> outputs;
  for (const fabric::ResolvedFabricOpPhysicalPortView &port :
       request.capability.physicalPorts)
    (port.reference.direction == fabric::FabricPortDirection::Input ? inputs
                                                                    : outputs)
        .push_back(&port);
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(inputs, byOrdinal);
  llvm::sort(outputs, byOrdinal);
  if (inputs.size() != 2 || outputs.size() != 1 ||
      inputs[0]->reference.ordinal != 0 || inputs[1]->reference.ordinal != 1 ||
      outputs[0]->reference.ordinal != 0 || inputs[0]->payloadWidthBits == 0)
    return unsupported(request);

  auto layout =
      deriveTransparentLoopOperationLeafStateLayout(request.capability);
  if (!layout)
    return layout.takeError();
  if (!*layout)
    return invalid("capability has no transparent loop state layout");
  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return std::move(error);

  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  const TransparentLoopOperationLeafStateLayout stateLayout = **layout;
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        const std::array<mlir::Value, 2> inputValid{
            accessor.getInput("valid_input_0"),
            accessor.getInput("valid_input_1")};
        mlir::Value outputReady = accessor.getInput("ready_output_0");
        mlir::Value currentState = accessor.getInput("state_current");
        mlir::Value currentMode =
            stateLayout.encodedBitCount() == 1
                ? currentState
                : mlir::Value(circt::comb::ExtractOp::create(
                      bodyBuilder, location, currentState,
                      TransparentLoopOperationLeafStateLayout::modeBit, 1));
        mlir::Value phase = accessor.getInput("data_input_0");
        if (inputs[0]->payloadWidthBits != 1)
          phase = circt::comb::ExtractOp::create(bodyBuilder, location, phase,
                                                 0, 1);
        mlir::Value notPhase =
            circt::comb::createOrFoldNot(bodyBuilder, location, phase);

        mlir::Value currentPayload;
        if (stateLayout.payloadWidthBits != 0)
          currentPayload = circt::comb::ExtractOp::create(
              bodyBuilder, location, currentState,
              TransparentLoopOperationLeafStateLayout::invariantPayloadOffset,
              stateLayout.payloadWidthBits);

        const std::array<InvariantCase, 3> transitions{
            ::dataflow::semantics::selectInvariantCase(
                InvariantSemanticState::Initial, false),
            ::dataflow::semantics::selectInvariantCase(
                InvariantSemanticState::Running, true),
            ::dataflow::semantics::selectInvariantCase(
                InvariantSemanticState::Running, false)};
        std::vector<MaterializedCase> cases;
        cases.reserve(transitions.size());
        for (InvariantCase transition : transitions) {
          const InvariantCaseDescriptor descriptor =
              ::dataflow::semantics::invariantCaseDescriptor(transition);
          mlir::Value requiredMode = bitConstant(
              bodyBuilder, location,
              descriptor.requiredState == InvariantSemanticState::Running);
          mlir::Value selected = circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::eq,
              currentMode, requiredMode, true);
          if (descriptor.requiredPhase)
            selected = circt::comb::AndOp::create(
                bodyBuilder, location, selected,
                *descriptor.requiredPhase ? phase : notPhase);

          llvm::SmallVector<mlir::Value, 2> requiredValids;
          for (unsigned ordinal = 0; ordinal < inputValid.size(); ++ordinal)
            if (::dataflow::semantics::selectsSemanticInput(
                    descriptor.consumedInputs,
                    static_cast<InvariantInput>(ordinal)))
              requiredValids.push_back(inputValid[ordinal]);
          mlir::Value allInputsValid =
              andAll(bodyBuilder, location, requiredValids);
          llvm::SmallVector<mlir::Value, 3> fireTerms{selected, allInputsValid};
          if (descriptor.output != InvariantOutputSource::None)
            fireTerms.push_back(outputReady);
          cases.push_back(
              MaterializedCase{descriptor, selected, allInputsValid,
                               andAll(bodyBuilder, location, fireTerms)});
        }

        for (unsigned ordinal = 0; ordinal < inputValid.size(); ++ordinal) {
          llvm::SmallVector<mlir::Value, 3> readyCases;
          for (const MaterializedCase &candidate : cases) {
            const auto input = static_cast<InvariantInput>(ordinal);
            if (!::dataflow::semantics::selectsSemanticInput(
                    candidate.descriptor.consumedInputs, input))
              continue;
            llvm::SmallVector<mlir::Value, 4> terms{candidate.selected};
            for (unsigned other = 0; other < inputValid.size(); ++other) {
              const auto otherInput = static_cast<InvariantInput>(other);
              if (other != ordinal &&
                  ::dataflow::semantics::selectsSemanticInput(
                      candidate.descriptor.consumedInputs, otherInput))
                terms.push_back(inputValid[other]);
            }
            if (candidate.descriptor.output != InvariantOutputSource::None)
              terms.push_back(outputReady);
            readyCases.push_back(andAll(bodyBuilder, location, terms));
          }
          accessor.setOutput("ready_input_" + std::to_string(ordinal),
                             orAll(bodyBuilder, location, readyCases));
        }

        llvm::SmallVector<mlir::Value, 2> outputValidCases;
        for (const MaterializedCase &candidate : cases)
          if (candidate.descriptor.output != InvariantOutputSource::None)
            outputValidCases.push_back(circt::comb::AndOp::create(
                bodyBuilder, location, candidate.selected,
                candidate.allInputsValid));
        accessor.setOutput("valid_output_0",
                           orAll(bodyBuilder, location, outputValidCases));

        const auto zeroPayload = [&](unsigned width) -> mlir::Value {
          return circt::hw::ConstantOp::create(bodyBuilder, location,
                                               llvm::APInt(width, 0));
        };
        const auto inputPayload = [&](InvariantInput source,
                                      unsigned width) -> mlir::Value {
          const unsigned ordinal = static_cast<unsigned>(source);
          if (inputs[ordinal]->payloadWidthBits == 0)
            return zeroPayload(width);
          return detail::resizeUnsigned(
              bodyBuilder, location,
              accessor.getInput("data_input_" + std::to_string(ordinal)),
              width);
        };

        if (outputs[0]->payloadWidthBits != 0) {
          mlir::Value result;
          for (const MaterializedCase &candidate : cases) {
            mlir::Value value;
            switch (candidate.descriptor.output) {
            case InvariantOutputSource::None:
              continue;
            case InvariantOutputSource::InitInput:
              value = inputPayload(InvariantInput::Init,
                                   outputs[0]->payloadWidthBits);
              break;
            case InvariantOutputSource::Latched:
              value = stateLayout.payloadWidthBits == 0
                          ? zeroPayload(outputs[0]->payloadWidthBits)
                          : detail::resizeUnsigned(
                                bodyBuilder, location, currentPayload,
                                outputs[0]->payloadWidthBits);
              break;
            }
            result = result ? circt::comb::MuxOp::create(bodyBuilder, location,
                                                         candidate.selected,
                                                         value, result, true)
                            : value;
          }
          accessor.setOutput("data_output_0", result);
        }

        llvm::SmallVector<mlir::Value, 3> stateWrites;
        mlir::Value nextState = currentState;
        for (const MaterializedCase &candidate : cases) {
          stateWrites.push_back(candidate.fire);
          mlir::Value nextMode =
              bitConstant(bodyBuilder, location,
                          candidate.descriptor.nextState ==
                              InvariantSemanticState::Running);
          mlir::Value encodedState = nextMode;
          if (stateLayout.payloadWidthBits != 0) {
            mlir::Value nextPayload = currentPayload;
            if (candidate.descriptor.latchInput)
              nextPayload = inputPayload(*candidate.descriptor.latchInput,
                                         stateLayout.payloadWidthBits);
            if (candidate.descriptor.clearLatch)
              nextPayload = zeroPayload(stateLayout.payloadWidthBits);
            encodedState = circt::comb::ConcatOp::create(
                bodyBuilder, location, mlir::ValueRange{nextPayload, nextMode});
          }
          nextState =
              circt::comb::MuxOp::create(bodyBuilder, location, candidate.fire,
                                         encodedState, nextState, true);
        }
        accessor.setOutput("state_next", nextState);
        accessor.setOutput("state_write",
                           orAll(bodyBuilder, location, stateWrites));
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

} // namespace

llvm::Error registerPortableLoopInvariantProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::LoopInvariant,
                       BackendRecipeKey::PortableSystemVerilog,
                       {},
                       materializePortableLoopInvariant});
}

} // namespace loom::hardware::rtl
