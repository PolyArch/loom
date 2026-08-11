#include "Hardware/RTL/Providers/ScalarUnsignedIntegerDivRem.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

using Schema = ::dataflow::OperationSchemaId;

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  bool quotient = false;
  unsigned width = 0;
};

struct WidthResults final {
  unsigned width = 0;
  mlir::Value quotient;
  mlir::Value remainder;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "portable_scalar_unsigned_integer_div_rem_invalid: " + message);
}

llvm::Expected<LoweredMode> lowerMode(const Mode &mode) {
  bool quotient;
  switch (mode.actor.schema) {
  case Schema::ArithDivUI:
    quotient = true;
    break;
  case Schema::ArithRemUI:
    quotient = false;
    break;
  default:
    return invalid("behavior relation contains a foreign operation schema");
  }
  if (mode.actor.type.getNumInputs() != 2 ||
      mode.actor.type.getNumResults() != 1)
    return invalid("behavior relation contains a non-binary actor");
  const auto lhsType =
      mlir::dyn_cast<mlir::IntegerType>(mode.actor.type.getInput(0));
  const auto rhsType =
      mlir::dyn_cast<mlir::IntegerType>(mode.actor.type.getInput(1));
  const auto resultType =
      mlir::dyn_cast<mlir::IntegerType>(mode.actor.type.getResult(0));
  if (!lhsType || !rhsType || !resultType || !lhsType.isSignless() ||
      lhsType != rhsType || lhsType != resultType || lhsType.getWidth() == 0)
    return invalid("behavior relation contains an incompatible integer type");
  return LoweredMode{quotient, lhsType.getWidth()};
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

  auto relation = request.capability.resolveSemanticFieldRelation(
      *request.leaf.getContext());
  if (!relation)
    return relation.takeError();
  const auto domain = relation->finiteBehaviorDomain();
  const ConfigurationFieldEncoding *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  std::vector<Mode> modes;
  if (request.capability.configurationFieldSchema.empty()) {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None)
      return invalid("configuration-free relation is not fieldless");
    if (domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid("configuration-free relation is not a singleton");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() !=
        ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid("configured relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured capability requires exactly one field");
    field = request.configurationAbi.findOperationField(
        request.occurrence,
        request.capability.configurationFieldSchema.front().ordinal);
    if (!field)
      return invalid("configured field is absent from the ABI");
    codebook = std::get_if<FiniteCodebookEncoding>(&field->semanticEncoding);
    if (!codebook)
      return invalid("configured field is not a finite codebook");
    if (codebook->entries.size() != domain.size())
      return invalid("codebook does not exactly cover the behavior domain");
    modes.reserve(domain.size());
    for (const auto &point : domain) {
      if (!point.semanticConfiguration)
        return invalid("configured behavior has no semantic value");
      const FiniteCodebookEntry *entry = detail::findFiniteCodebookEntry(
          *codebook, point.semanticConfiguration->bytes());
      if (!entry)
        return invalid("codebook has no entry for an admitted semantic value");
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
  std::vector<unsigned> widths;
  loweredModes.reserve(modes.size());
  widths.reserve(modes.size());
  for (const Mode &mode : modes) {
    auto lowered = lowerMode(mode);
    if (!lowered)
      return lowered.takeError();
    if (lowered->width > inputs[0]->payloadWidthBits ||
        lowered->width > inputs[1]->payloadWidthBits ||
        lowered->width > outputs[0]->payloadWidthBits)
      return llvm::make_error<FabricOperationProviderUnsupportedError>(
          family, request.recipe);
    if (!llvm::is_contained(widths, lowered->width))
      widths.push_back(lowered->width);
    loweredModes.push_back(*lowered);
  }
  if (modes.empty())
    return invalid("behavior relation is empty");

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
        std::vector<WidthResults> widthResults;
        widthResults.reserve(widths.size());
        for (unsigned width : widths) {
          mlir::Value lhs = detail::resizeUnsigned(
              bodyBuilder, location, accessor.getInput("data_input_0"), width);
          mlir::Value rhs = detail::resizeUnsigned(
              bodyBuilder, location, accessor.getInput("data_input_1"), width);
          mlir::Value zero = circt::hw::ConstantOp::create(
              bodyBuilder, location, llvm::APInt(width, 0));
          mlir::Value one = circt::hw::ConstantOp::create(
              bodyBuilder, location, llvm::APInt(width, 1));
          mlir::Value divisorIsZero = circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::eq, rhs, zero,
              true);
          mlir::Value safeDivisor = circt::comb::MuxOp::create(
              bodyBuilder, location, divisorIsZero, one, rhs, true);
          mlir::Value rawQuotient = circt::comb::DivUOp::create(
              bodyBuilder, location, lhs, safeDivisor, true);
          mlir::Value product = circt::comb::MulOp::create(
              bodyBuilder, location,
              mlir::ValueRange{rawQuotient, safeDivisor}, true);
          mlir::Value rawRemainder = circt::comb::SubOp::create(
              bodyBuilder, location, lhs, product, true);
          mlir::Value quotient = circt::comb::MuxOp::create(
              bodyBuilder, location, divisorIsZero, zero, rawQuotient, true);
          mlir::Value remainder = circt::comb::MuxOp::create(
              bodyBuilder, location, divisorIsZero, zero, rawRemainder, true);
          widthResults.push_back({width, quotient, remainder});
        }

        std::vector<mlir::Value> results;
        results.reserve(loweredModes.size());
        for (const LoweredMode &mode : loweredModes) {
          const auto found =
              llvm::find_if(widthResults, [&](const auto &entry) {
                return entry.width == mode.width;
              });
          mlir::Value result =
              mode.quotient ? found->quotient : found->remainder;
          results.push_back(detail::resizeUnsigned(
              bodyBuilder, location, result, outputs[0]->payloadWidthBits));
        }

        mlir::Value result = results[inactiveMode];
        if (field) {
          mlir::Value configuration = accessor.getInput(
              "config_" +
              std::to_string(
                  request.capability.configurationFieldSchema.front().ordinal));
          for (std::size_t index = 0; index < modes.size(); ++index) {
            if (index == inactiveMode)
              continue;
            mlir::Value code = circt::hw::ConstantOp::create(
                bodyBuilder, location,
                detail::decodePhysicalCode(
                    modes[index].codebookEntry->physicalCode,
                    codebook->encodedBitCount));
            mlir::Value selected = circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::eq,
                configuration, code, true);
            result = circt::comb::MuxOp::create(bodyBuilder, location, selected,
                                                results[index], result, true);
          }
        }
        accessor.setOutput("data_output_0", result);
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
