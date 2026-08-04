#include "Hardware/RTL/Providers/LoopGate.h"

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
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

using GateCaseDescriptor = ::dataflow::semantics::GateCaseDescriptor;
using GateInput = ::dataflow::semantics::GateInput;
using GateSemanticState = ::dataflow::semantics::GateSemanticState;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_loop_gate_invalid: " + message);
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
  GateCaseDescriptor descriptor;
  mlir::Value selected;
  mlir::Value allInputsValid;
  mlir::Value outputCapacity;
  mlir::Value fire;
};

llvm::Expected<FabricOperationProviderOutput>
materializePortableLoopGate(FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::LoopGate)
    return invalid("provider received a different implementation family");
  if (!std::holds_alternative<::fabric::TokenPlaneParams>(
          request.capability.parameterizedCapability))
    return invalid("capability has the wrong parameter schema");
  if (request.capability.enabledOperationSchemas !=
      std::vector<::dataflow::OperationSchemaId>{
          ::dataflow::OperationSchemaId::DataflowGate})
    return invalid("capability does not contain exactly dataflow.gate");
  if (!request.capability.configurationFieldSchema.empty())
    return invalid("loop gate capability has semantic configuration fields");

  auto actualContract = ::fabric::encodeResourceContractRecord(
      request.capability.resourceStateAndTimingContract);
  if (!actualContract)
    return actualContract.takeError();
  auto supportedContract = ::fabric::encodeResourceContractRecord(
      ::fabric::loopGateOperationResourceContract());
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
  if (inputs.size() != 2 || outputs.size() != 2 ||
      inputs[0]->reference.ordinal != 0 || inputs[1]->reference.ordinal != 1 ||
      outputs[0]->reference.ordinal != 0 || outputs[1]->reference.ordinal != 1)
    return invalid("capability does not have the gate port roles");
  if (inputs[0]->payloadWidthBits == 0)
    return invalid("phase input has no low semantic bit");
  if (outputs[0]->payloadWidthBits == 0)
    return invalid("phase output has no low semantic bit");
  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.capability, request.configurationAbi))
    return std::move(error);

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
        const std::array<mlir::Value, 2> outputReady{
            accessor.getInput("ready_output_0"),
            accessor.getInput("ready_output_1")};
        mlir::Value currentState = accessor.getInput("state_current");
        mlir::Value phase = detail::resizeUnsigned(
            bodyBuilder, location, accessor.getInput("data_input_0"), 1);
        mlir::Value notPhase =
            circt::comb::createOrFoldNot(bodyBuilder, location, phase);

        const std::array<GateSemanticState, 2> states{GateSemanticState::Closed,
                                                      GateSemanticState::Open};
        const std::array<bool, 2> phases{false, true};
        std::vector<MaterializedCase> cases;
        cases.reserve(states.size() * phases.size());
        for (GateSemanticState state : states) {
          for (bool phaseValue : phases) {
            const GateCaseDescriptor descriptor =
                ::dataflow::semantics::gateCaseDescriptor(
                    ::dataflow::semantics::selectGateCase(state, phaseValue));
            mlir::Value requiredState = bitConstant(
                bodyBuilder, location,
                descriptor.requiredState == GateSemanticState::Open);
            mlir::Value selected = circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::eq,
                currentState, requiredState, true);
            selected = circt::comb::AndOp::create(
                bodyBuilder, location, selected,
                descriptor.requiredPhase ? phase : notPhase);

            llvm::SmallVector<mlir::Value, 2> requiredValids;
            for (unsigned ordinal = 0; ordinal < inputValid.size(); ++ordinal)
              if (::dataflow::semantics::selectsSemanticInput(
                      descriptor.consumedInputs,
                      static_cast<GateInput>(ordinal)))
                requiredValids.push_back(inputValid[ordinal]);
            mlir::Value allInputsValid =
                andAll(bodyBuilder, location, requiredValids);

            llvm::SmallVector<mlir::Value, 2> activeOutputReady;
            if (descriptor.emitPhase)
              activeOutputReady.push_back(outputReady[0]);
            if (descriptor.forwardedInput)
              activeOutputReady.push_back(outputReady[1]);
            mlir::Value outputCapacity =
                andAll(bodyBuilder, location, activeOutputReady);
            cases.push_back(MaterializedCase{
                descriptor, selected, allInputsValid, outputCapacity,
                andAll(bodyBuilder, location,
                       {selected, allInputsValid, outputCapacity})});
          }
        }

        for (unsigned ordinal = 0; ordinal < inputValid.size(); ++ordinal) {
          llvm::SmallVector<mlir::Value, 4> readyCases;
          for (const MaterializedCase &candidate : cases) {
            const auto input = static_cast<GateInput>(ordinal);
            if (!::dataflow::semantics::selectsSemanticInput(
                    candidate.descriptor.consumedInputs, input))
              continue;
            llvm::SmallVector<mlir::Value, 4> terms{candidate.selected,
                                                    candidate.outputCapacity};
            for (unsigned other = 0; other < inputValid.size(); ++other) {
              const auto otherInput = static_cast<GateInput>(other);
              if (other != ordinal &&
                  ::dataflow::semantics::selectsSemanticInput(
                      candidate.descriptor.consumedInputs, otherInput))
                terms.push_back(inputValid[other]);
            }
            readyCases.push_back(andAll(bodyBuilder, location, terms));
          }
          accessor.setOutput("ready_input_" + std::to_string(ordinal),
                             orAll(bodyBuilder, location, readyCases));
        }

        llvm::SmallVector<mlir::Value, 2> phaseValidCases;
        llvm::SmallVector<mlir::Value, 2> valueValidCases;
        for (const MaterializedCase &candidate : cases) {
          mlir::Value valid = circt::comb::AndOp::create(
              bodyBuilder, location, candidate.selected,
              candidate.allInputsValid);
          if (candidate.descriptor.emitPhase)
            phaseValidCases.push_back(valid);
          if (candidate.descriptor.forwardedInput)
            valueValidCases.push_back(valid);
        }
        accessor.setOutput("valid_output_0",
                           orAll(bodyBuilder, location, phaseValidCases));
        accessor.setOutput("valid_output_1",
                           orAll(bodyBuilder, location, valueValidCases));

        mlir::Value phaseResult = circt::hw::ConstantOp::create(
            bodyBuilder, location,
            llvm::APInt(outputs[0]->payloadWidthBits, 0));
        for (const MaterializedCase &candidate : cases) {
          if (!candidate.descriptor.emitPhase)
            continue;
          mlir::Value value = detail::resizeUnsigned(
              bodyBuilder, location,
              bitConstant(bodyBuilder, location, candidate.descriptor.phase),
              outputs[0]->payloadWidthBits);
          phaseResult = circt::comb::MuxOp::create(bodyBuilder, location,
                                                   candidate.selected, value,
                                                   phaseResult, true);
        }
        accessor.setOutput("data_output_0", phaseResult);

        if (outputs[1]->payloadWidthBits != 0) {
          mlir::Value value =
              inputs[1]->payloadWidthBits == 0
                  ? mlir::Value(circt::hw::ConstantOp::create(
                        bodyBuilder, location,
                        llvm::APInt(outputs[1]->payloadWidthBits, 0)))
                  : detail::resizeUnsigned(bodyBuilder, location,
                                           accessor.getInput("data_input_1"),
                                           outputs[1]->payloadWidthBits);
          accessor.setOutput("data_output_1", value);
        }

        llvm::SmallVector<mlir::Value, 4> stateWrites;
        mlir::Value nextState = currentState;
        for (const MaterializedCase &candidate : cases) {
          stateWrites.push_back(candidate.fire);
          mlir::Value value = bitConstant(bodyBuilder, location,
                                          candidate.descriptor.nextState ==
                                              GateSemanticState::Open);
          nextState = circt::comb::MuxOp::create(
              bodyBuilder, location, candidate.fire, value, nextState, true);
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

llvm::Error
registerPortableLoopGateProvider(FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::LoopGate,
                       BackendRecipeKey::PortableSystemVerilog,
                       {},
                       materializePortableLoopGate});
}

} // namespace loom::hardware::rtl
