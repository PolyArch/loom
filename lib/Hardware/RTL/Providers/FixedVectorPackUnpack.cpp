#include "Hardware/RTL/Providers/FixedVectorPackUnpack.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/Twine.h"

#include <variant>

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_fixed_vector_pack_unpack_invalid: " +
                                     message);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFixedVectorAdapter(
    FabricOperationProviderRequest request,
    ::fabric::ImplementationFamilyId expectedFamily) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily != expectedFamily)
    return invalid("provider received a different implementation family");
  if (!std::holds_alternative<::fabric::FixedVectorAdapterParams>(
          request.capability.parameterizedCapability))
    return invalid("capability has the wrong parameter schema");
  if (!request.capability.configurationFieldSchema.empty())
    return invalid("pack/unpack capability has semantic configuration fields");

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

  const fabric::ResolvedFabricOpPhysicalPortView *input = nullptr;
  const fabric::ResolvedFabricOpPhysicalPortView *output = nullptr;
  for (const fabric::ResolvedFabricOpPhysicalPortView &port :
       request.capability.physicalPorts) {
    if (port.reference.direction == fabric::FabricPortDirection::Input) {
      if (input)
        return llvm::make_error<FabricOperationProviderUnsupportedError>(
            request.capability.implementationFamily, request.recipe);
      input = &port;
    } else {
      if (output)
        return llvm::make_error<FabricOperationProviderUnsupportedError>(
            request.capability.implementationFamily, request.recipe);
      output = &port;
    }
  }
  if (!input || !output || input->reference.ordinal != 0 ||
      output->reference.ordinal != 0 || input->payloadWidthBits == 0 ||
      output->payloadWidthBits == 0)
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
        accessor.setOutput(
            "data_output_0",
            detail::resizeUnsigned(bodyBuilder, location,
                                   accessor.getInput("data_input_0"),
                                   output->payloadWidthBits));
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFixedVectorPack(FabricOperationProviderRequest request) {
  return materializePortableFixedVectorAdapter(
      request, ::fabric::ImplementationFamilyId::FixedVectorPack);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFixedVectorUnpack(FabricOperationProviderRequest request) {
  return materializePortableFixedVectorAdapter(
      request, ::fabric::ImplementationFamilyId::FixedVectorUnpack);
}

} // namespace

llvm::Error registerPortableFixedVectorPackProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::FixedVectorPack,
                       BackendRecipeKey::PortableSystemVerilog,
                       {},
                       materializePortableFixedVectorPack});
}

llvm::Error registerPortableFixedVectorUnpackProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::FixedVectorUnpack,
                       BackendRecipeKey::PortableSystemVerilog,
                       {},
                       materializePortableFixedVectorUnpack});
}

} // namespace loom::hardware::rtl
