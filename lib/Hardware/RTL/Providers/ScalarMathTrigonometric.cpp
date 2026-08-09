#include "Hardware/RTL/Providers/ScalarMathTrigonometric.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Common/SpecialMathAccuracy.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

enum class TrigFamily { Sin, Cos, Tan };
using Format = detail::PortableFloatFormat;

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  Format format;
  std::string functionName;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "portable_scalar_math_trigonometric_invalid: " + message);
}

llvm::Error unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

::fabric::ImplementationFamilyId familyId(TrigFamily family) {
  switch (family) {
  case TrigFamily::Sin:
    return ::fabric::ImplementationFamilyId::ScalarMathSin;
  case TrigFamily::Cos:
    return ::fabric::ImplementationFamilyId::ScalarMathCos;
  case TrigFamily::Tan:
    return ::fabric::ImplementationFamilyId::ScalarMathTan;
  }
  llvm_unreachable("unknown trigonometric family");
}

::dataflow::OperationSchemaId schemaId(TrigFamily family) {
  switch (family) {
  case TrigFamily::Sin:
    return ::dataflow::OperationSchemaId::MathSin;
  case TrigFamily::Cos:
    return ::dataflow::OperationSchemaId::MathCos;
  case TrigFamily::Tan:
    return ::dataflow::OperationSchemaId::MathTan;
  }
  llvm_unreachable("unknown trigonometric family");
}

llvm::StringRef shortName(TrigFamily family) {
  switch (family) {
  case TrigFamily::Sin:
    return "sin";
  case TrigFamily::Cos:
    return "cos";
  case TrigFamily::Tan:
    return "tan";
  }
  llvm_unreachable("unknown trigonometric family");
}

std::string functionName(TrigFamily family, const Format &format) {
  return "loom_trig_" + shortName(family).str() + "_e" +
         std::to_string(format.exponentBits) + "_f" +
         std::to_string(format.fractionBits);
}

bool isSupportedFormat(const Format &format) {
  return (format.exponentBits == 5 && format.fractionBits == 10) ||
         (format.exponentBits == 8 && format.fractionBits == 7);
}

llvm::Expected<LoweredMode>
lowerMode(TrigFamily family,
          const ::dataflow::CanonicalActorSchemaProjection &actor) {
  if (actor.schema != schemaId(family) || actor.type.getNumInputs() != 1 ||
      actor.type.getNumResults() != 1)
    return invalid("behavior is not the selected unary special-math operation");
  if (actor.type.getResult(0) != actor.type.getInput(0) ||
      llvm::isa<mlir::VectorType>(actor.type.getInput(0)))
    return invalid("behavior does not have one uniform scalar floating type");
  const auto *payload =
      std::get_if<::dataflow::SpecialMathPayload>(&actor.payload);
  if (!payload)
    return invalid("behavior has no special-math payload");
  if (!llvm::is_contained(::loom::specialMathAccuracyTiers(),
                          payload->accuracy))
    return invalid("behavior has an unknown accuracy tier");
  auto format = detail::resolvePortableFloatFormat(actor.type.getInput(0));
  if (!format)
    return invalid("behavior uses an unknown floating format");
  return LoweredMode{*format, functionName(family, *format)};
}

std::string signedConstant(std::int64_t value) {
  if (value >= 0)
    return "66'sd" + std::to_string(value);
  const std::uint64_t magnitude = static_cast<std::uint64_t>(-(value + 1)) + 1;
  return "-66'sd" + std::to_string(magnitude);
}

std::string buildRoundShiftFunction(const std::string &name) {
  std::string text;
  llvm::raw_string_ostream output(text);
  output
      << "function automatic [95:0] " << name
      << "(input [95:0] value, input integer distance);\n"
      << "  reg guard;\n"
      << "  reg sticky;\n"
      << "  integer index;\n"
      << "  begin\n"
      << "    " << name << " = 96'd0;\n"
      << "    guard = 1'b0;\n"
      << "    sticky = 1'b0;\n"
      << "    if (distance <= 0) begin\n"
      << "      " << name << " = value << (-distance);\n"
      << "    end else if (distance <= 96) begin\n"
      << "      " << name << " = value >> distance;\n"
      << "      guard = value[distance - 1];\n"
      << "      for (index = 0; index < 96; index = index + 1) begin\n"
      << "        if (index < distance - 1) sticky = sticky | value[index];\n"
      << "      end\n"
      << "      if (guard && (sticky || " << name << "[0]))\n"
      << "        " << name << " = " << name << " + 1'b1;\n"
      << "    end\n"
      << "  end\n"
      << "endfunction\n\n";
  return output.str();
}

std::string buildMultiplyFunction(const std::string &name) {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic signed [65:0] " << name
         << "(input signed [65:0] lhs, input signed [65:0] rhs);\n"
         << "  reg negative;\n"
         << "  reg [65:0] lhs_magnitude;\n"
         << "  reg [65:0] rhs_magnitude;\n"
         << "  reg [131:0] product;\n"
         << "  reg [131:0] shifted;\n"
         << "  begin\n"
         << "    negative = lhs[65] ^ rhs[65];\n"
         << "    lhs_magnitude = lhs[65] ? -lhs : lhs;\n"
         << "    rhs_magnitude = rhs[65] ? -rhs : rhs;\n"
         << "    product = lhs_magnitude * rhs_magnitude;\n"
         << "    shifted = product >> 64;\n"
         << "    " << name
         << " = negative ? -$signed(shifted[65:0])"
            " : $signed(shifted[65:0]);\n"
         << "  end\n"
         << "endfunction\n\n";
  return output.str();
}

std::string buildPackFunction(const Format &format, const std::string &base) {
  const std::string name = base + "_pack";
  const std::string roundName = base + "_round_shift";
  const unsigned width = format.width();
  const unsigned exponentBits = format.exponentBits;
  const unsigned fractionBits = format.fractionBits;
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponentBits) - 1;

  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic [" << width - 1 << ":0] " << name
         << "(input signed [95:0] fixed_value);\n"
         << "  reg sign_result;\n"
         << "  reg [95:0] magnitude;\n"
         << "  reg [95:0] rounded;\n"
         << "  reg found;\n"
         << "  integer leading_index;\n"
         << "  integer exponent_value;\n"
         << "  integer encoded_exponent;\n"
         << "  integer shift_amount;\n"
         << "  integer index;\n"
         << "  begin\n"
         << "    sign_result = fixed_value[95];\n"
         << "    magnitude = sign_result ? -fixed_value : fixed_value;\n"
         << "    " << name << " = {sign_result, " << width - 1 << "'d0};\n"
         << "    if (magnitude != 0) begin\n"
         << "      leading_index = 0;\n"
         << "      found = 1'b0;\n"
         << "      for (index = 95; index >= 0; index = index - 1) begin\n"
         << "        if (!found && magnitude[index]) begin\n"
         << "          leading_index = index;\n"
         << "          found = 1'b1;\n"
         << "        end\n"
         << "      end\n"
         << "      exponent_value = leading_index - 64;\n"
         << "      if (exponent_value < " << format.minimumExponent()
         << ") begin\n"
         << "        shift_amount = 64 + " << format.minimumExponent() << " - "
         << fractionBits << ";\n"
         << "        rounded = " << roundName << "(magnitude, shift_amount);\n"
         << "        if (rounded >= (96'd1 << " << fractionBits << "))\n"
         << "          " << name << " = {sign_result, " << exponentBits
         << "'d1, " << fractionBits << "'d0};\n"
         << "        else\n"
         << "          " << name << " = {sign_result, " << exponentBits
         << "'d0, rounded[" << fractionBits - 1 << ":0]};\n"
         << "      end else begin\n"
         << "        shift_amount = leading_index - " << fractionBits << ";\n"
         << "        rounded = " << roundName << "(magnitude, shift_amount);\n"
         << "        if (rounded[" << fractionBits + 1 << "]) begin\n"
         << "          rounded = rounded >> 1;\n"
         << "          exponent_value = exponent_value + 1;\n"
         << "        end\n"
         << "        if (exponent_value > " << format.maximumExponent()
         << ") begin\n"
         << "          " << name << " = {sign_result, " << exponentBits << "'d"
         << exponentMask << ", " << fractionBits << "'d0};\n"
         << "        end else begin\n"
         << "          encoded_exponent = exponent_value + " << format.bias()
         << ";\n"
         << "          " << name << " = {sign_result, encoded_exponent["
         << exponentBits - 1 << ":0], rounded[" << fractionBits - 1 << ":0]};\n"
         << "        end\n"
         << "      end\n"
         << "    end\n"
         << "  end\n"
         << "endfunction\n\n";
  return output.str();
}

std::string buildCoreFunction(TrigFamily family, const Format &format) {
  const std::string name = functionName(family, format);
  const std::string multiplyName = name + "_mul_q";
  const std::string roundName = name + "_round_shift";
  const std::string packName = name + "_pack";
  const unsigned width = format.width();
  const unsigned exponentBits = format.exponentBits;
  const unsigned fractionBits = format.fractionBits;
  const unsigned precision = format.precision();
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponentBits) - 1;
  const std::uint64_t quietBit = std::uint64_t{1} << (fractionBits - 1);
  const std::uint64_t quietNaN = (exponentMask << fractionBits) | quietBit;
  const std::uint64_t one = static_cast<std::uint64_t>(format.bias())
                            << fractionBits;
  const int tinyLimit = family == TrigFamily::Sin ? -6 : -7;
  const unsigned rangeWidth = std::max(
      256 + precision, static_cast<unsigned>(255 - tinyLimit + fractionBits));

  constexpr std::int64_t sinCoefficients[] = {
      -3074457345618258603LL, 153722867280912930LL, -3660068268593165LL,
      50834281508238LL};
  constexpr std::int64_t cosCoefficients[] = {
      -9223372036854775807LL - 1, 768614336404564651LL, -25620477880152155LL,
      457508533574146LL};

  std::string text;
  llvm::raw_string_ostream output(text);
  output
      << buildRoundShiftFunction(roundName)
      << buildMultiplyFunction(multiplyName) << buildPackFunction(format, name)
      << "function automatic [" << width - 1 << ":0] " << name << "(input ["
      << width - 1 << ":0] value);\n"
      << "  reg sign_input;\n"
      << "  reg [" << exponentBits - 1 << ":0] exponent_field;\n"
      << "  reg [" << fractionBits - 1 << ":0] fraction_field;\n"
      << "  reg [" << precision - 1 << ":0] significand;\n"
      << "  reg [" << rangeWidth - 1 << ":0] range_product;\n"
      << "  reg [" << rangeWidth - 1 << ":0] integer_part;\n"
      << "  reg [63:0] fraction_bits_q;\n"
      << "  reg round_up;\n"
      << "  reg [1:0] quadrant;\n"
      << "  reg signed [65:0] fraction_q;\n"
      << "  reg signed [65:0] remainder_q;\n"
      << "  reg signed [65:0] square_q;\n"
      << "  reg signed [65:0] sine_accumulator;\n"
      << "  reg signed [65:0] cosine_accumulator;\n"
      << "  reg signed [65:0] sine_q;\n"
      << "  reg signed [65:0] cosine_q;\n"
      << "  reg signed [95:0] sine_wide;\n"
      << "  reg signed [95:0] cosine_wide;\n"
      << "  reg signed [128:0] quotient_numerator;\n"
      << "  reg signed [128:0] quotient_denominator;\n"
      << "  reg signed [128:0] quotient_full;\n"
      << "  reg signed [95:0] selected_q;\n"
      << "  integer exponent_value;\n"
      << "  integer binary_exponent;\n"
      << "  integer reduction_shift;\n"
      << "  begin\n"
      << "    sign_input = value[" << width - 1 << "];\n"
      << "    exponent_field = value[" << width - 2 << ':' << fractionBits
      << "];\n"
      << "    fraction_field = value[" << fractionBits - 1 << ":0];\n"
      << "    " << name << " = " << width << "'h" << llvm::utohexstr(quietNaN)
      << ";\n"
      << "    if (exponent_field == " << exponentBits << "'d" << exponentMask
      << " && fraction_field != 0) begin\n"
      << "      " << name << " = value | " << width << "'h"
      << llvm::utohexstr(quietBit) << ";\n"
      << "    end else if (exponent_field == " << exponentBits << "'d"
      << exponentMask << ") begin\n"
      << "      " << name << " = " << width << "'h" << llvm::utohexstr(quietNaN)
      << ";\n"
      << "    end else if (exponent_field == 0 && fraction_field == 0) begin\n"
      << "      " << name << " = ";
  if (family == TrigFamily::Cos)
    output << width << "'h" << llvm::utohexstr(one) << ";\n";
  else
    output << "value;\n";
  output << "    end else begin\n"
         << "      exponent_value = integer'(exponent_field) - "
         << format.bias() << ";\n"
         << "      if (exponent_field == 0 || exponent_value <= " << tinyLimit
         << ") begin\n"
         << "        " << name << " = ";
  if (family == TrigFamily::Cos)
    output << width << "'h" << llvm::utohexstr(one) << ";\n";
  else
    output << "value;\n";
  output
      << "      end else begin\n"
      << "        significand = exponent_field == 0"
         " ? {1'b0, fraction_field} : {1'b1, fraction_field};\n"
      << "        binary_exponent = exponent_value - " << fractionBits << ";\n"
      << "        range_product = significand * "
         "256'"
         "ha2f9836e4e441529fc2757d1f534ddc0db6295993c439041fe5163abdebbc561;\n"
      << "        reduction_shift = 256 - binary_exponent;\n"
      << "        integer_part = range_product >> reduction_shift;\n"
      << "        round_up = range_product[reduction_shift - 1];\n"
      << "        quadrant = integer_part[1:0] + round_up;\n"
      << "        integer_part ="
         " range_product >> (reduction_shift - 64);\n"
      << "        fraction_bits_q = integer_part[63:0];\n"
      << "        fraction_q = 66'sd0;\n"
      << "        fraction_q[64:0] = {1'b0, fraction_bits_q};\n"
      << "        if (round_up)"
         " fraction_q = fraction_q - 66'sd18446744073709551616;\n"
      << "        remainder_q = " << multiplyName
      << "(fraction_q, 66'sd28976077832308491370);\n"
      << "        square_q = " << multiplyName
      << "(remainder_q, remainder_q);\n"
      << "        sine_accumulator = " << signedConstant(sinCoefficients[3])
      << ";\n";
  for (int index = 2; index >= 0; --index)
    output << "        sine_accumulator = "
           << signedConstant(sinCoefficients[index]) << " + " << multiplyName
           << "(square_q, sine_accumulator);\n";
  output << "        sine_q = " << multiplyName
         << "(remainder_q, 66'sd18446744073709551616 + " << multiplyName
         << "(square_q, sine_accumulator));\n"
         << "        cosine_accumulator = "
         << signedConstant(cosCoefficients[3]) << ";\n";
  for (int index = 2; index >= 0; --index)
    output << "        cosine_accumulator = "
           << signedConstant(cosCoefficients[index]) << " + " << multiplyName
           << "(square_q, cosine_accumulator);\n";
  output << "        cosine_q = 66'sd18446744073709551616 + " << multiplyName
         << "(square_q, cosine_accumulator);\n"
         << "        sine_wide = {{30{sine_q[65]}}, sine_q};\n"
         << "        cosine_wide = {{30{cosine_q[65]}}, cosine_q};\n";
  switch (family) {
  case TrigFamily::Sin:
    output << "        case (quadrant)\n"
           << "          2'd0: selected_q = sine_wide;\n"
           << "          2'd1: selected_q = cosine_wide;\n"
           << "          2'd2: selected_q = -sine_wide;\n"
           << "          default: selected_q = -cosine_wide;\n"
           << "        endcase\n"
           << "        if (sign_input) selected_q = -selected_q;\n";
    break;
  case TrigFamily::Cos:
    output << "        case (quadrant)\n"
           << "          2'd0: selected_q = cosine_wide;\n"
           << "          2'd1: selected_q = -sine_wide;\n"
           << "          2'd2: selected_q = -cosine_wide;\n"
           << "          default: selected_q = sine_wide;\n"
           << "        endcase\n";
    break;
  case TrigFamily::Tan:
    output << "        if (quadrant[0]) begin\n"
           << "          quotient_numerator = 129'sd0;\n"
           << "          quotient_numerator[65:0] = cosine_q;\n"
           << "          quotient_numerator ="
              " -(quotient_numerator <<< 64);\n"
           << "          quotient_denominator ="
              " {{63{sine_q[65]}}, sine_q};\n"
           << "        end else begin\n"
           << "          quotient_numerator = 129'sd0;\n"
           << "          quotient_numerator[65:0] = sine_q;\n"
           << "          quotient_numerator = quotient_numerator <<< 64;\n"
           << "          quotient_denominator ="
              " {{63{cosine_q[65]}}, cosine_q};\n"
           << "        end\n"
           << "        quotient_full ="
              " quotient_numerator / quotient_denominator;\n"
           << "        selected_q = quotient_full[95:0];\n"
           << "        if (sign_input) selected_q = -selected_q;\n";
    break;
  }
  output << "        " << name << " = " << packName << "(selected_q);\n"
         << "      end\n"
         << "    end\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

mlir::Value callMode(mlir::OpBuilder &builder, mlir::Location location,
                     const LoweredMode &mode, mlir::Value operand) {
  return circt::sv::VerbatimExprOp::create(
      builder, location, builder.getIntegerType(mode.format.width()),
      mode.functionName + "({{0}})", llvm::SmallVector<mlir::Value>{operand});
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarMathTrigonometric(
    FabricOperationProviderRequest request, TrigFamily family) {
  const auto expectedFamily = familyId(family);
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily != expectedFamily)
    return invalid("provider received a different implementation family");
  const auto &descriptor = ::fabric::implementationFamily(expectedFamily);
  if (descriptor.familyId != expectedFamily ||
      ::fabric::capabilityParamsSchema(
          request.capability.parameterizedCapability) !=
          descriptor.capabilityParamsSchema)
    return invalid("capability does not match its generated family descriptor");
  const auto *parameters = std::get_if<::fabric::ScalarSpecialMathParams>(
      &request.capability.parameterizedCapability);
  if (!parameters)
    return invalid("capability has the wrong special-math parameter schema");
  if (request.capability.enabledOperationSchemas !=
          std::vector<::dataflow::OperationSchemaId>{schemaId(family)} ||
      descriptor.admittedSchemas.size() != 1 ||
      descriptor.admittedSchemas.front() != schemaId(family))
    return invalid("capability does not contain its exact registered schema");

  auto actualContract = ::fabric::encodeResourceContractRecord(
      request.capability.resourceStateAndTimingContract);
  if (!actualContract)
    return actualContract.takeError();
  auto supportedContract = ::fabric::encodeResourceContractRecord(
      ::fabric::oneCycleElasticOperationResourceContract());
  if (!supportedContract)
    return supportedContract.takeError();
  if (*actualContract != *supportedContract)
    return unsupported(request);

  const auto &behavior = parameters->behavior;
  if (!behavior.roundingModes.valid() || behavior.roundingModes.size() != 1 ||
      !behavior.roundingModes.contains(
          mlir::arith::RoundingMode::to_nearest_even) ||
      !behavior.nanBehaviors.valid() || behavior.nanBehaviors.size() != 1 ||
      !behavior.nanBehaviors.contains(::fabric::FloatNaNBehavior::IEEE) ||
      !behavior.subnormalBehaviors.valid() ||
      behavior.subnormalBehaviors.size() != 1 ||
      !behavior.subnormalBehaviors.contains(
          ::fabric::FloatSubnormalBehavior::Preserve) ||
      !behavior.signedZeroBehaviors.valid() ||
      behavior.signedZeroBehaviors.size() != 1 ||
      !behavior.signedZeroBehaviors.contains(
          ::fabric::FloatSignedZeroBehavior::Preserve))
    return unsupported(request);
  if (!llvm::is_contained(::loom::specialMathAccuracyTiers(),
                          parameters->accuracyGuarantee))
    return invalid("capability has an unknown accuracy guarantee");

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
  if (inputs.size() != 1 || outputs.size() != 1 ||
      inputs[0]->reference.ordinal != 0 || outputs[0]->reference.ordinal != 0)
    return unsupported(request);
  if (inputs[0]->payloadWidthBits == 0 || outputs[0]->payloadWidthBits == 0)
    return invalid("capability has a zero-width physical data port");
  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return std::move(error);

  auto relation = request.capability.resolveSemanticFieldRelation(
      *request.leaf.getContext());
  if (!relation)
    return relation.takeError();
  const auto &domain = relation->finiteBehaviorDomain();
  const ConfigurationFieldEncoding *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  std::vector<Mode> modes;
  if (request.capability.configurationFieldSchema.empty()) {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None)
      return invalid(
          "configuration-free trigonometric relation is not fieldless");
    if (domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid(
          "configuration-free trigonometric relation is not a singleton");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid("configured trigonometric relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured trigonometric capability requires one field");
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
  if (modes.empty())
    return invalid("sealed trigonometric behavior relation is empty");

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
  std::set<std::string> functionNames;
  loweredModes.reserve(modes.size());
  for (const Mode &mode : modes) {
    auto lowered = lowerMode(family, mode.actor);
    if (!lowered)
      return lowered.takeError();
    if (!isSupportedFormat(lowered->format))
      return unsupported(request);
    if (!functionNames.insert(lowered->functionName).second)
      return invalid("sealed relation contains a duplicate trigonometric mode");
    const unsigned width = lowered->format.width();
    if (width > inputs[0]->payloadWidthBits ||
        width > outputs[0]->payloadWidthBits)
      return unsupported(request);
    loweredModes.push_back(std::move(*lowered));
  }

  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        std::string declarations;
        llvm::raw_string_ostream declarationStream(declarations);
        for (const LoweredMode &mode : loweredModes)
          declarationStream << buildCoreFunction(family, mode.format) << '\n';
        circt::sv::VerbatimOp::create(
            bodyBuilder, location,
            bodyBuilder.getStringAttr(declarationStream.str()));

        mlir::Value operand = detail::resizeUnsigned(
            bodyBuilder, location, accessor.getInput("data_input_0"), 16);
        std::vector<mlir::Value> results;
        results.reserve(loweredModes.size());
        for (const LoweredMode &mode : loweredModes) {
          mlir::Value result = callMode(bodyBuilder, location, mode, operand);
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

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarMathSin(FabricOperationProviderRequest request) {
  return materializePortableScalarMathTrigonometric(std::move(request),
                                                    TrigFamily::Sin);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarMathCos(FabricOperationProviderRequest request) {
  return materializePortableScalarMathTrigonometric(std::move(request),
                                                    TrigFamily::Cos);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarMathTan(FabricOperationProviderRequest request) {
  return materializePortableScalarMathTrigonometric(std::move(request),
                                                    TrigFamily::Tan);
}

} // namespace

llvm::Error registerPortableScalarMathTrigonometricProviders(
    FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry staged = registry;
  if (llvm::Error error =
          staged.add({::fabric::ImplementationFamilyId::ScalarMathSin,
                      BackendRecipeKey::PortableSystemVerilog,
                      {},
                      materializePortableScalarMathSin}))
    return error;
  if (llvm::Error error =
          staged.add({::fabric::ImplementationFamilyId::ScalarMathCos,
                      BackendRecipeKey::PortableSystemVerilog,
                      {},
                      materializePortableScalarMathCos}))
    return error;
  if (llvm::Error error =
          staged.add({::fabric::ImplementationFamilyId::ScalarMathTan,
                      BackendRecipeKey::PortableSystemVerilog,
                      {},
                      materializePortableScalarMathTan}))
    return error;
  registry = std::move(staged);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl
