#include "Hardware/RTL/Providers/ScalarUnsignedIntegerDivRem.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "portable_scalar_unsigned_integer_div_rem_invalid: " + message);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarUnsignedIntegerDivRem(
    FabricOperationProviderRequest request) {
  constexpr auto family =
      ::fabric::ImplementationFamilyId::ScalarUnsignedIntegerDivRem;
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily != family)
    return invalid("provider received a different implementation family");

  const ::fabric::ImplementationFamilyDescriptor &descriptor =
      ::fabric::implementationFamily(family);
  if (descriptor.familyId != family || descriptor.admittedSchemas.size() != 2)
    return invalid("generated family descriptor has an incompatible shape");
  if (::fabric::capabilityParamsSchema(
          request.capability.parameterizedCapability) !=
      descriptor.capabilityParamsSchema)
    return invalid("capability parameter schema does not match the generated "
                   "family descriptor");
  const auto *parameters = std::get_if<::fabric::ScalarIntegerParams>(
      &request.capability.parameterizedCapability);
  if (!parameters)
    return invalid("capability has the wrong parameter schema");
  if (!parameters->integerWidths.valid() || parameters->integerWidths.empty() ||
      !parameters->pointerFormats.valid())
    return invalid("capability has malformed scalar integer parameters");
  if (parameters->integerWidths.size() != 1)
    return llvm::make_error<FabricOperationProviderUnsupportedError>(
        family, request.recipe);
  const auto semanticWidthEntry = llvm::find_if(
      ::fabric::integerWidthDomain, [&](::fabric::IntegerWidth candidate) {
        return parameters->integerWidths.contains(candidate);
      });
  if (semanticWidthEntry == ::fabric::integerWidthDomain.end())
    return invalid("capability has no canonical scalar integer width");
  const unsigned semanticWidth = ::fabric::getBitWidth(*semanticWidthEntry);

  std::vector<::dataflow::OperationSchemaId> enabled;
  enabled.reserve(request.capability.enabledOperationSchemas.size());
  for (::dataflow::OperationSchemaId schema :
       request.capability.enabledOperationSchemas) {
    if (!llvm::is_contained(descriptor.admittedSchemas, schema))
      return invalid("capability escapes its generated family descriptor");
    if (llvm::is_contained(enabled, schema))
      return invalid("capability repeats an enabled operation schema");
    enabled.push_back(schema);
  }
  if (enabled.empty())
    return invalid("capability has no enabled operation schema");
  const ::dataflow::OperationSchemaId quotientSchema =
      descriptor.admittedSchemas[0];
  const ::dataflow::OperationSchemaId remainderSchema =
      descriptor.admittedSchemas[1];
  const bool hasQuotient = llvm::is_contained(enabled, quotientSchema);
  const bool hasRemainder = llvm::is_contained(enabled, remainderSchema);
  if (!hasQuotient && !hasRemainder)
    return invalid("capability has no generated div/rem member");

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
        family, request.recipe);

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
  if (inputs.size() != 2 || outputs.size() != 1 ||
      inputs[0]->reference.ordinal != 0 || inputs[1]->reference.ordinal != 1 ||
      outputs[0]->reference.ordinal != 0)
    return llvm::make_error<FabricOperationProviderUnsupportedError>(
        family, request.recipe);
  if (inputs[0]->payloadWidthBits == 0 || inputs[1]->payloadWidthBits == 0 ||
      outputs[0]->payloadWidthBits == 0)
    return invalid("capability has a zero-width physical data port");
  if (semanticWidth > inputs[0]->payloadWidthBits ||
      semanticWidth > inputs[1]->payloadWidthBits ||
      semanticWidth > outputs[0]->payloadWidthBits)
    return llvm::make_error<FabricOperationProviderUnsupportedError>(
        family, request.recipe);

  const ConfigurationFieldEncoding *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  const FiniteCodebookEntry *quotientEntry = nullptr;
  const FiniteCodebookEntry *remainderEntry = nullptr;
  bool inactiveQuotient = hasQuotient && !hasRemainder;
  if (hasQuotient && hasRemainder) {
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured div/rem capability requires one field");
    field = request.configurationAbi.findField(
        request.capability.configurationFieldSchema.front());
    if (!field)
      return invalid("configured div/rem field is absent from the ABI");
    codebook = std::get_if<FiniteCodebookEncoding>(&field->semanticEncoding);
    if (!codebook)
      return invalid("configured div/rem field is not a finite codebook");
    if (codebook->entries.size() != enabled.size())
      return invalid(
          "codebook does not exactly cover the operation-selection domain");

    auto quotientValue = request.capability.encodeOperationSelection(
        field->field, quotientSchema, *request.leaf.getContext());
    if (!quotientValue)
      return quotientValue.takeError();
    auto remainderValue = request.capability.encodeOperationSelection(
        field->field, remainderSchema, *request.leaf.getContext());
    if (!remainderValue)
      return remainderValue.takeError();
    quotientEntry =
        detail::findFiniteCodebookEntry(*codebook, quotientValue->bytes());
    if (!quotientEntry)
      return invalid("codebook has no quotient semantic value");
    remainderEntry =
        detail::findFiniteCodebookEntry(*codebook, remainderValue->bytes());
    if (!remainderEntry)
      return invalid("codebook has no remainder semantic value");
    if (llvm::ArrayRef<std::uint8_t>(field->inactiveValue)
            .equals(quotientValue->bytes())) {
      inactiveQuotient = true;
    } else if (llvm::ArrayRef<std::uint8_t>(field->inactiveValue)
                   .equals(remainderValue->bytes())) {
      inactiveQuotient = false;
    } else {
      return invalid("ABI inactive value is outside the div/rem domain");
    }
  } else if (!request.capability.configurationFieldSchema.empty()) {
    return invalid("singleton div/rem capability has a selector field");
  }

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
        mlir::Value lhs = detail::resizeUnsigned(
            bodyBuilder, location, accessor.getInput("data_input_0"),
            semanticWidth);
        mlir::Value rhs = detail::resizeUnsigned(
            bodyBuilder, location, accessor.getInput("data_input_1"),
            semanticWidth);
        mlir::Value zero = circt::hw::ConstantOp::create(
            bodyBuilder, location, llvm::APInt(semanticWidth, 0));
        mlir::Value one = circt::hw::ConstantOp::create(
            bodyBuilder, location, llvm::APInt(semanticWidth, 1));
        mlir::Value divisorIsZero = circt::comb::ICmpOp::create(
            bodyBuilder, location, circt::comb::ICmpPredicate::eq, rhs, zero,
            true);
        mlir::Value safeDivisor = circt::comb::MuxOp::create(
            bodyBuilder, location, divisorIsZero, one, rhs, true);
        mlir::Value quotient = circt::comb::DivUOp::create(
            bodyBuilder, location, lhs, safeDivisor, true);
        mlir::Value remainder;
        if (hasRemainder) {
          mlir::Value product = circt::comb::MulOp::create(
              bodyBuilder, location, mlir::ValueRange{quotient, safeDivisor},
              true);
          remainder = circt::comb::SubOp::create(bodyBuilder, location, lhs,
                                                 product, true);
        }

        mlir::Value result = hasQuotient ? quotient : remainder;
        if (hasQuotient && hasRemainder) {
          const FiniteCodebookEntry *nonInactiveEntry =
              inactiveQuotient ? remainderEntry : quotientEntry;
          mlir::Value nonInactiveResult =
              inactiveQuotient ? remainder : quotient;
          mlir::Value code = circt::hw::ConstantOp::create(
              bodyBuilder, location,
              detail::decodePhysicalCode(nonInactiveEntry->physicalCode,
                                         codebook->encodedBitCount));
          mlir::Value selected = circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::eq,
              accessor.getInput("config_" +
                                std::to_string(field->field.ordinal)),
              code, true);
          mlir::Value inactiveResult = inactiveQuotient ? quotient : remainder;
          result = circt::comb::MuxOp::create(bodyBuilder, location, selected,
                                              nonInactiveResult, inactiveResult,
                                              true);
        }
        result = circt::comb::MuxOp::create(bodyBuilder, location,
                                            divisorIsZero, zero, result, true);
        accessor.setOutput("data_output_0", detail::resizeUnsigned(
                                                bodyBuilder, location, result,
                                                outputs[0]->payloadWidthBits));
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

} // namespace

llvm::Error registerPortableScalarUnsignedIntegerDivRemProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add(
      {::fabric::ImplementationFamilyId::ScalarUnsignedIntegerDivRem,
       BackendRecipeKey::PortableSystemVerilog,
       {},
       materializePortableScalarUnsignedIntegerDivRem});
}

} // namespace loom::hardware::rtl
