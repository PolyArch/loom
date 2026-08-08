#include "Hardware/RTL/Providers/ScalarValueSelect.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_scalar_value_select_invalid: " +
                                     message);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarValueSelect(FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::ScalarValueSelect)
    return invalid("provider received a different implementation family");
  if (!std::holds_alternative<::fabric::ScalarValueSelectParams>(
          request.capability.parameterizedCapability))
    return invalid("capability has the wrong parameter schema");
  if (!request.capability.configurationFieldSchema.empty())
    return invalid("select capability has semantic configuration fields");

  auto actualContract = ::fabric::encodeResourceContractRecord(
      request.capability.resourceStateAndTimingContract);
  if (!actualContract)
    return actualContract.takeError();
  auto supportedContract = ::fabric::encodeResourceContractRecord(
      ::fabric::oneCycleElasticOperationResourceContract());
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
      inputs[2]->reference.ordinal != 2 || outputs[0]->reference.ordinal != 0 ||
      inputs[0]->payloadWidthBits == 0 || inputs[1]->payloadWidthBits == 0 ||
      inputs[2]->payloadWidthBits == 0 || outputs[0]->payloadWidthBits == 0)
    return llvm::make_error<FabricOperationProviderUnsupportedError>(
        request.capability.implementationFamily, request.recipe);

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
        mlir::Value condition = detail::resizeUnsigned(
            bodyBuilder, location, accessor.getInput("data_input_0"), 1);
        const unsigned valueWidth =
            std::max({inputs[1]->payloadWidthBits, inputs[2]->payloadWidthBits,
                      outputs[0]->payloadWidthBits});
        mlir::Value trueValue = detail::resizeUnsigned(
            bodyBuilder, location, accessor.getInput("data_input_1"),
            valueWidth);
        mlir::Value falseValue = detail::resizeUnsigned(
            bodyBuilder, location, accessor.getInput("data_input_2"),
            valueWidth);
        mlir::Value selected = circt::comb::MuxOp::create(
            bodyBuilder, location, condition, trueValue, falseValue, true);
        accessor.setOutput("data_output_0", detail::resizeUnsigned(
                                                bodyBuilder, location, selected,
                                                outputs[0]->payloadWidthBits));
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

} // namespace

llvm::Error registerPortableScalarValueSelectProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::ScalarValueSelect,
                       BackendRecipeKey::PortableSystemVerilog,
                       {},
                       materializePortableScalarValueSelect});
}

} // namespace loom::hardware::rtl
