#include "Hardware/RTL/Providers/ScalarIntegerMultiply.h"

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
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_scalar_integer_multiply_invalid: " +
                                     message);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarIntegerMultiply(
    FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::ScalarIntegerMultiply)
    return invalid("provider received a different implementation family");
  if (!std::holds_alternative<::fabric::ScalarIntegerParams>(
          request.capability.parameterizedCapability))
    return invalid("capability has the wrong parameter schema");
  if (!request.capability.configurationFieldSchema.empty())
    return invalid("multiply capability has semantic configuration fields");

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
  if (inputs.size() != 2 || outputs.size() != 1 ||
      inputs[0]->reference.ordinal != 0 || inputs[1]->reference.ordinal != 1 ||
      outputs[0]->reference.ordinal != 0 || inputs[0]->payloadWidthBits == 0 ||
      inputs[1]->payloadWidthBits == 0 || outputs[0]->payloadWidthBits == 0)
    return invalid("capability does not have the binary integer port shape");

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
        const unsigned arithmeticWidth =
            std::max({inputs[0]->payloadWidthBits, inputs[1]->payloadWidthBits,
                      outputs[0]->payloadWidthBits});
        mlir::Value lhs = detail::resizeUnsigned(
            bodyBuilder, location, accessor.getInput("data_input_0"),
            arithmeticWidth);
        mlir::Value rhs = detail::resizeUnsigned(
            bodyBuilder, location, accessor.getInput("data_input_1"),
            arithmeticWidth);
        mlir::Value product = circt::comb::MulOp::create(
            bodyBuilder, location, mlir::ValueRange{lhs, rhs}, true);
        accessor.setOutput("data_output_0", detail::resizeUnsigned(
                                                bodyBuilder, location, product,
                                                outputs[0]->payloadWidthBits));
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

} // namespace

llvm::Error registerPortableScalarIntegerMultiplyProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
                       BackendRecipeKey::PortableSystemVerilog,
                       {},
                       materializePortableScalarIntegerMultiply});
}

} // namespace loom::hardware::rtl
