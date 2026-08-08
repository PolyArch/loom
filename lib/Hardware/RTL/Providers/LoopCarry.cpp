#include "Hardware/RTL/Providers/LoopCarry.h"

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

using CarryCase = ::dataflow::semantics::CarryCase;
using CarryCaseDescriptor = ::dataflow::semantics::CarryCaseDescriptor;
using CarryInput = ::dataflow::semantics::CarryInput;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_loop_carry_invalid: " + message);
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
  CarryCaseDescriptor descriptor;
  mlir::Value selected;
  mlir::Value allInputsValid;
  mlir::Value fire;
};

llvm::Expected<FabricOperationProviderOutput>
materializePortableLoopCarry(FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::LoopCarry)
    return invalid("provider received a different implementation family");
  if (!std::holds_alternative<::fabric::TokenPlaneParams>(
          request.capability.parameterizedCapability))
    return invalid("capability has the wrong parameter schema");
  if (request.capability.enabledOperationSchemas !=
      std::vector<::dataflow::OperationSchemaId>{
          ::dataflow::OperationSchemaId::DataflowCarry})
    return invalid("capability does not contain exactly dataflow.carry");
  if (!request.capability.configurationFieldSchema.empty())
    return invalid("loop carry capability has semantic configuration fields");

  auto actualContract = ::fabric::encodeResourceContractRecord(
      request.capability.resourceStateAndTimingContract);
  if (!actualContract)
    return actualContract.takeError();
  auto supportedContract = ::fabric::encodeResourceContractRecord(
      ::fabric::loopCarryOperationResourceContract());
  if (!supportedContract)
    return supportedContract.takeError();
  if (*actualContract != *supportedContract)
    return llvm::make_error<FabricOperationProviderUnsupportedError>(
        request.capability.implementationFamily, request.recipe);

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
  if (inputs.size() != 3 || outputs.size() != 1 ||
      inputs[0]->reference.ordinal != 0 || inputs[1]->reference.ordinal != 1 ||
      inputs[2]->reference.ordinal != 2 || outputs[0]->reference.ordinal != 0)
    return invalid("capability does not have the carry port roles");
  if (inputs[0]->payloadWidthBits == 0)
    return invalid("phase input has no low semantic bit");
  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return std::move(error);

  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        std::array<mlir::Value, 3> inputValid{
            accessor.getInput("valid_input_0"),
            accessor.getInput("valid_input_1"),
            accessor.getInput("valid_input_2")};
        mlir::Value outputReady = accessor.getInput("ready_output_0");
        mlir::Value currentState = accessor.getInput("state_current");
        mlir::Value phase = accessor.getInput("data_input_0");
        if (inputs[0]->payloadWidthBits != 1)
          phase = circt::comb::ExtractOp::create(bodyBuilder, location, phase,
                                                 0, 1);
        mlir::Value notPhase =
            circt::comb::createOrFoldNot(bodyBuilder, location, phase);

        const std::array<CarryCase, 3> transitions{
            CarryCase::Init, CarryCase::Next, CarryCase::Close};
        std::vector<MaterializedCase> cases;
        cases.reserve(transitions.size());
        for (CarryCase transition : transitions) {
          const CarryCaseDescriptor descriptor =
              ::dataflow::semantics::carryCaseDescriptor(transition);
          mlir::Value requiredState = circt::hw::ConstantOp::create(
              bodyBuilder, location,
              encodeLoopCarryOperationLeafState(descriptor.requiredState));
          mlir::Value selected = circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::eq,
              currentState, requiredState, true);
          if (descriptor.requiredPhase)
            selected = circt::comb::AndOp::create(
                bodyBuilder, location, selected,
                *descriptor.requiredPhase ? phase : notPhase);

          llvm::SmallVector<mlir::Value, 3> requiredValids;
          for (unsigned ordinal = 0; ordinal < inputValid.size(); ++ordinal)
            if (::dataflow::semantics::selectsSemanticInput(
                    descriptor.consumedInputs,
                    static_cast<CarryInput>(ordinal)))
              requiredValids.push_back(inputValid[ordinal]);
          mlir::Value allInputsValid =
              andAll(bodyBuilder, location, requiredValids);
          llvm::SmallVector<mlir::Value, 3> fireTerms{selected, allInputsValid};
          if (descriptor.forwardedInput)
            fireTerms.push_back(outputReady);
          cases.push_back(
              MaterializedCase{descriptor, selected, allInputsValid,
                               andAll(bodyBuilder, location, fireTerms)});
        }

        for (unsigned ordinal = 0; ordinal < inputValid.size(); ++ordinal) {
          llvm::SmallVector<mlir::Value, 3> readyCases;
          for (const MaterializedCase &candidate : cases) {
            const auto input = static_cast<CarryInput>(ordinal);
            if (!::dataflow::semantics::selectsSemanticInput(
                    candidate.descriptor.consumedInputs, input))
              continue;
            llvm::SmallVector<mlir::Value, 4> terms{candidate.selected};
            for (unsigned other = 0; other < inputValid.size(); ++other) {
              const auto otherInput = static_cast<CarryInput>(other);
              if (other != ordinal &&
                  ::dataflow::semantics::selectsSemanticInput(
                      candidate.descriptor.consumedInputs, otherInput))
                terms.push_back(inputValid[other]);
            }
            if (candidate.descriptor.forwardedInput)
              terms.push_back(outputReady);
            readyCases.push_back(andAll(bodyBuilder, location, terms));
          }
          accessor.setOutput("ready_input_" + std::to_string(ordinal),
                             orAll(bodyBuilder, location, readyCases));
        }

        llvm::SmallVector<mlir::Value, 2> outputValidCases;
        for (const MaterializedCase &candidate : cases)
          if (candidate.descriptor.forwardedInput)
            outputValidCases.push_back(circt::comb::AndOp::create(
                bodyBuilder, location, candidate.selected,
                candidate.allInputsValid));
        accessor.setOutput("valid_output_0",
                           orAll(bodyBuilder, location, outputValidCases));

        if (outputs[0]->payloadWidthBits != 0) {
          const auto payload = [&](CarryInput source) {
            const unsigned ordinal = static_cast<unsigned>(source);
            if (inputs[ordinal]->payloadWidthBits == 0)
              return mlir::Value(circt::hw::ConstantOp::create(
                  bodyBuilder, location,
                  llvm::APInt(outputs[0]->payloadWidthBits, 0)));
            return detail::resizeUnsigned(
                bodyBuilder, location,
                accessor.getInput("data_input_" + std::to_string(ordinal)),
                outputs[0]->payloadWidthBits);
          };
          mlir::Value result;
          for (const MaterializedCase &candidate : cases) {
            if (!candidate.descriptor.forwardedInput)
              continue;
            mlir::Value value = payload(*candidate.descriptor.forwardedInput);
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
          mlir::Value value = circt::hw::ConstantOp::create(
              bodyBuilder, location,
              encodeLoopCarryOperationLeafState(
                  candidate.descriptor.nextState));
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
registerPortableLoopCarryProvider(FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::LoopCarry,
                       BackendRecipeKey::PortableSystemVerilog,
                       {},
                       materializePortableLoopCarry});
}

} // namespace loom::hardware::rtl
