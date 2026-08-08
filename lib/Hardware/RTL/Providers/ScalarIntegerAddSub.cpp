#include "Hardware/RTL/Providers/ScalarIntegerAddSub.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_scalar_integer_add_sub_invalid: " +
                                     message);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarIntegerAddSub(FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::ScalarIntegerAddSub)
    return invalid("provider received a different implementation family");
  const auto *parameters = std::get_if<::fabric::ScalarIntegerParams>(
      &request.capability.parameterizedCapability);
  if (!parameters)
    return invalid("capability has the wrong parameter schema");
  if (!parameters->pointerFormats.empty())
    return llvm::make_error<FabricOperationProviderUnsupportedError>(
        request.capability.implementationFamily, request.recipe);

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

  bool hasAdd = false;
  bool hasSubtract = false;
  for (::dataflow::OperationSchemaId schema :
       request.capability.enabledOperationSchemas) {
    if (schema == ::dataflow::OperationSchemaId::ArithAddI)
      hasAdd = true;
    else if (schema == ::dataflow::OperationSchemaId::ArithSubI)
      hasSubtract = true;
    else
      return invalid("capability contains a non-add/sub operation schema");
  }
  if (!hasAdd && !hasSubtract)
    return invalid("capability has no add or subtract schema");

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

  const ConfigurationFieldEncoding *field = nullptr;
  const FiniteCodebookEntry *nonInactiveEntry = nullptr;
  bool inactiveSubtract = hasSubtract && !hasAdd;
  if (hasAdd && hasSubtract) {
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured add/sub capability requires one field");
    field = request.configurationAbi.findField(
        request.capability.configurationFieldSchema.front());
    if (!field)
      return invalid("configured add/sub field is absent from the ABI");
    const auto *codebook =
        std::get_if<FiniteCodebookEncoding>(&field->semanticEncoding);
    if (!codebook)
      return invalid("configured add/sub field is not a finite codebook");
    if (codebook->entries.size() !=
        request.capability.enabledOperationSchemas.size())
      return invalid(
          "codebook does not exactly cover the operation-selection domain");
    auto addValue = request.capability.encodeOperationSelection(
        field->field, ::dataflow::OperationSchemaId::ArithAddI);
    if (!addValue)
      return addValue.takeError();
    auto subtractValue = request.capability.encodeOperationSelection(
        field->field, ::dataflow::OperationSchemaId::ArithSubI);
    if (!subtractValue)
      return subtractValue.takeError();
    const FiniteCodebookEntry *addEntry =
        detail::findFiniteCodebookEntry(*codebook, addValue->bytes());
    if (!addEntry)
      return invalid("codebook has no add semantic value");
    const FiniteCodebookEntry *subtractEntry =
        detail::findFiniteCodebookEntry(*codebook, subtractValue->bytes());
    if (!subtractEntry)
      return invalid("codebook has no subtract semantic value");
    if (llvm::ArrayRef<std::uint8_t>(field->inactiveValue)
            .equals(addValue->bytes())) {
      nonInactiveEntry = subtractEntry;
      inactiveSubtract = false;
    } else if (llvm::ArrayRef<std::uint8_t>(field->inactiveValue)
                   .equals(subtractValue->bytes())) {
      nonInactiveEntry = addEntry;
      inactiveSubtract = true;
    } else {
      return invalid("ABI inactive value is outside the add/sub domain");
    }
  } else if (!request.capability.configurationFieldSchema.empty()) {
    return invalid("singleton add/sub capability has a selector field");
  }

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
        mlir::Value subtract = circt::hw::ConstantOp::create(
            bodyBuilder, location, llvm::APInt(1, inactiveSubtract));
        if (hasAdd && hasSubtract) {
          const auto &codebook =
              std::get<FiniteCodebookEncoding>(field->semanticEncoding);
          mlir::Value code = circt::hw::ConstantOp::create(
              bodyBuilder, location,
              detail::decodePhysicalCode(nonInactiveEntry->physicalCode,
                                         codebook.encodedBitCount));
          mlir::Value selected = circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::eq,
              accessor.getInput("config_" +
                                std::to_string(field->field.ordinal)),
              code, true);
          mlir::Value selectedSubtract = circt::hw::ConstantOp::create(
              bodyBuilder, location, llvm::APInt(1, !inactiveSubtract));
          subtract =
              circt::comb::MuxOp::create(bodyBuilder, location, selected,
                                         selectedSubtract, subtract, true);
        }
        mlir::Value result =
            detail::addOrSubtract(bodyBuilder, location, lhs, rhs, subtract);
        accessor.setOutput("data_output_0", detail::resizeUnsigned(
                                                bodyBuilder, location, result,
                                                outputs[0]->payloadWidthBits));
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

} // namespace

llvm::Error registerPortableScalarIntegerAddSubProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                       BackendRecipeKey::PortableSystemVerilog,
                       {},
                       materializePortableScalarIntegerAddSub});
}

} // namespace loom::hardware::rtl
