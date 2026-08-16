#include "Hardware/RTL/Providers/MathExponential.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Common/SpecialMathAccuracy.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

enum class MathFamily { Exp, Exp2, ExpM1 };
using Format = detail::PortableFloatFormat;

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  Format format;
  std::string functionName;
};

constexpr unsigned fixedFractionBits = 24;
constexpr unsigned fixedWidth = 40;
constexpr unsigned productWidth = 2 * fixedWidth;
constexpr int expm1DirectPackingRangeIndex =
    static_cast<int>(fixedWidth - fixedFractionBits - 1);
constexpr int expm1NegativeOneRangeIndex = -static_cast<int>(fixedWidth);
constexpr std::int64_t fixedOne = INT64_C(16777216);
constexpr std::int64_t fixedHalf = INT64_C(8388608);
constexpr std::int64_t fixedLn2 = INT64_C(11629080);
constexpr std::int64_t fixedLog2E = INT64_C(24204406);
constexpr std::array<std::int64_t, 7> exponentialCoefficients = {
    INT64_C(16777216), INT64_C(16777216), INT64_C(8388608), INT64_C(2796203),
    INT64_C(699051),   INT64_C(139810),   INT64_C(23302)};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_math_exponential_invalid: " +
                                     message);
}

llvm::Error unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

::fabric::ImplementationFamilyId familyId(MathFamily family) {
  switch (family) {
  case MathFamily::Exp:
    return ::fabric::ImplementationFamilyId::ScalarMathExp;
  case MathFamily::Exp2:
    return ::fabric::ImplementationFamilyId::ScalarMathExp2;
  case MathFamily::ExpM1:
    return ::fabric::ImplementationFamilyId::ScalarMathExpM1;
  }
  llvm_unreachable("unknown exponential family");
}

::dataflow::OperationSchemaId schemaId(MathFamily family) {
  switch (family) {
  case MathFamily::Exp:
    return ::dataflow::OperationSchemaId::MathExp;
  case MathFamily::Exp2:
    return ::dataflow::OperationSchemaId::MathExp2;
  case MathFamily::ExpM1:
    return ::dataflow::OperationSchemaId::MathExpM1;
  }
  llvm_unreachable("unknown exponential family");
}

llvm::StringRef familySuffix(MathFamily family) {
  switch (family) {
  case MathFamily::Exp:
    return "exp";
  case MathFamily::Exp2:
    return "exp2";
  case MathFamily::ExpM1:
    return "expm1";
  }
  llvm_unreachable("unknown exponential family");
}

bool isSupportedFormat(const Format &format) {
  return (format.exponentBits == 5 && format.fractionBits == 10) ||
         (format.exponentBits == 8 && format.fractionBits == 7);
}

bool hasExactBehavior(const ::fabric::FloatBehaviorProfile &behavior) {
  return behavior.roundingModes.valid() && behavior.roundingModes.size() == 1 &&
         behavior.roundingModes.contains(
             mlir::arith::RoundingMode::to_nearest_even) &&
         behavior.nanBehaviors.valid() && behavior.nanBehaviors.size() == 1 &&
         behavior.nanBehaviors.contains(::fabric::FloatNaNBehavior::IEEE) &&
         behavior.subnormalBehaviors.valid() &&
         behavior.subnormalBehaviors.size() == 1 &&
         behavior.subnormalBehaviors.contains(
             ::fabric::FloatSubnormalBehavior::Preserve) &&
         behavior.signedZeroBehaviors.valid() &&
         behavior.signedZeroBehaviors.size() == 1 &&
         behavior.signedZeroBehaviors.contains(
             ::fabric::FloatSignedZeroBehavior::Preserve) &&
         behavior.requiredFastMath == mlir::arith::FastMathFlags::afn;
}

bool hasSupportedParameters(const ::fabric::ScalarSpecialMathParams &params) {
  if (!params.formats.valid() || params.formats.empty() ||
      params.accuracyGuarantee != SpecialMathAccuracyTier::Max4Ulp ||
      !hasExactBehavior(params.behavior))
    return false;
  for (::fabric::FloatFormat format : ::fabric::floatFormatDomain)
    if (params.formats.contains(format) &&
        format != ::fabric::FloatFormat::F16 &&
        format != ::fabric::FloatFormat::BF16)
      return false;
  return true;
}

std::string modeName(MathFamily family, const Format &format) {
  return "loom_math_" + familySuffix(family).str() + "_e" +
         std::to_string(format.exponentBits) + "_f" +
         std::to_string(format.fractionBits) + "_max4";
}

llvm::Expected<LoweredMode>
lowerMode(MathFamily family,
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
  if (payload->accuracy != SpecialMathAccuracyTier::Max4Ulp ||
      payload->flags != mlir::arith::FastMathFlags::afn)
    return invalid("behavior does not match the sealed accuracy contract");
  if (llvm::Error error =
          validateSpecialMathAccuracyContract(payload->accuracy, true))
    return std::move(error);
  auto format = detail::resolvePortableFloatFormat(actor.type.getInput(0));
  if (!format || !isSupportedFormat(*format))
    return invalid("behavior uses an unsupported floating format");
  return LoweredMode{*format, modeName(family, *format)};
}

std::string hexLiteral(unsigned width, std::uint64_t value) {
  std::ostringstream stream;
  stream << width << "'h" << std::hex << value;
  return stream.str();
}

std::string signedLiteral(unsigned width, std::int64_t value) {
  return std::to_string(width) + "'sd" + std::to_string(value);
}

std::string helperName(MathFamily family, const Format &format,
                       llvm::StringRef helper) {
  return "loom_math_" + familySuffix(family).str() + "_e" +
         std::to_string(format.exponentBits) + "_f" +
         std::to_string(format.fractionBits) + "_" + helper.str();
}

std::string buildRoundProductFunction(MathFamily family, const Format &format) {
  const std::string name = helperName(family, format, "round_product_q24");
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic signed [" << fixedWidth - 1 << ":0] " << name
         << "(input signed [" << productWidth - 1 << ":0] value);\n"
         << "  reg [" << productWidth - 1 << ":0] magnitude;\n"
         << "  reg [" << productWidth - fixedFractionBits - 1
         << ":0] rounded_magnitude;\n"
         << "  begin\n"
         << "    magnitude = value < 0 ? -value : value;\n"
         << "    rounded_magnitude = magnitude[" << productWidth - 1 << ':'
         << fixedFractionBits << "];\n"
         << "    if (magnitude[" << fixedFractionBits - 1 << ":0] > "
         << hexLiteral(fixedFractionBits,
                       std::uint64_t{1} << (fixedFractionBits - 1))
         << " ||\n"
         << "        (magnitude[" << fixedFractionBits - 1 << ":0] == "
         << hexLiteral(fixedFractionBits,
                       std::uint64_t{1} << (fixedFractionBits - 1))
         << " && rounded_magnitude[0]))\n"
         << "      rounded_magnitude = rounded_magnitude + 1'b1;\n"
         << "    " << name << " = value < 0\n"
         << "        ? -$signed(rounded_magnitude[" << fixedWidth - 1
         << ":0])\n"
         << "        : $signed(rounded_magnitude[" << fixedWidth - 1
         << ":0]);\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

std::string buildMultiplyFunction(MathFamily family, const Format &format) {
  const std::string name = helperName(family, format, "multiply_q24");
  const std::string round = helperName(family, format, "round_product_q24");
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic signed [" << fixedWidth - 1 << ":0] " << name
         << "(input signed [" << fixedWidth - 1 << ":0] lhs, input signed ["
         << fixedWidth - 1 << ":0] rhs);\n"
         << "  reg signed [" << productWidth - 1 << ":0] product;\n"
         << "  begin\n"
         << "    product = lhs * rhs;\n"
         << "    " << name << " = " << round << "(product);\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

std::string buildRoundMagnitudeFunction(MathFamily family,
                                        const Format &format) {
  const std::string name = helperName(family, format, "round_magnitude");
  std::string text;
  llvm::raw_string_ostream output(text);
  output
      << "function automatic [" << fixedWidth << ":0] " << name << "(input ["
      << fixedWidth - 1 << ":0] value, input integer distance);\n"
      << "  reg guard;\n"
      << "  reg sticky;\n"
      << "  integer index;\n"
      << "  begin\n"
      << "    " << name << " = " << fixedWidth + 1 << "'d0;\n"
      << "    guard = 1'b0;\n"
      << "    sticky = 1'b0;\n"
      << "    if (distance <= 0) begin\n"
      << "      " << name << " = {1'b0, value} << (-distance);\n"
      << "    end else begin\n"
      << "      " << name << " = {1'b0, value >> distance};\n"
      << "      for (index = 0; index < " << fixedWidth
      << "; index = index + 1) begin\n"
      << "        if (index == distance - 1) guard = value[index];\n"
      << "        if (index < distance - 1) sticky = sticky | value[index];\n"
      << "      end\n"
      << "      if (guard && (sticky || " << name << "[0]))\n"
      << "        " << name << " = " << name << " + 1'b1;\n"
      << "    end\n"
      << "  end\n"
      << "endfunction\n";
  return output.str();
}

std::string buildPackFunctions(MathFamily family, const Format &format) {
  const unsigned width = format.width();
  const unsigned exponentBits = format.exponentBits;
  const unsigned fractionBits = format.fractionBits;
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponentBits) - 1;
  const std::string round = helperName(family, format, "round_magnitude");
  const std::string scaled = helperName(family, format, "pack_scaled");
  const std::string fixed = helperName(family, format, "pack_fixed");
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic [" << width - 1 << ":0] " << scaled
         << "(input [" << fixedWidth - 1
         << ":0] magnitude, input integer scale, input sign_bit);\n"
         << "  reg [" << fixedWidth << ":0] rounded;\n"
         << "  reg found;\n"
         << "  reg [" << exponentBits - 1 << ":0] encoded_exponent;\n"
         << "  integer leading_index;\n"
         << "  integer result_exponent;\n"
         << "  integer shift_amount;\n"
         << "  integer index;\n"
         << "  begin\n"
         << "    " << scaled << " = {sign_bit, " << exponentBits << "'d0, "
         << fractionBits << "'d0};\n"
         << "    if (magnitude != 0) begin\n"
         << "      leading_index = 0;\n"
         << "      found = 1'b0;\n"
         << "      for (index = " << fixedWidth - 1
         << "; index >= 0; index = index - 1) begin\n"
         << "        if (!found && magnitude[index]) begin\n"
         << "          leading_index = index;\n"
         << "          found = 1'b1;\n"
         << "        end\n"
         << "      end\n"
         << "      result_exponent = leading_index + scale - "
         << fixedFractionBits << ";\n"
         << "      if (result_exponent > " << format.maximumExponent()
         << ") begin\n"
         << "        " << scaled << " = {sign_bit, "
         << hexLiteral(exponentBits, exponentMask) << ", " << fractionBits
         << "'d0};\n"
         << "      end else if (result_exponent >= " << format.minimumExponent()
         << ") begin\n"
         << "        shift_amount = leading_index - " << fractionBits << ";\n"
         << "        rounded = " << round << "(magnitude, shift_amount);\n"
         << "        if (rounded[" << fractionBits + 1 << "]) begin\n"
         << "          rounded = rounded >> 1;\n"
         << "          result_exponent = result_exponent + 1;\n"
         << "        end\n"
         << "        if (result_exponent > " << format.maximumExponent()
         << ") begin\n"
         << "          " << scaled << " = {sign_bit, "
         << hexLiteral(exponentBits, exponentMask) << ", " << fractionBits
         << "'d0};\n"
         << "        end else begin\n"
         << "          encoded_exponent = result_exponent + " << format.bias()
         << ";\n"
         << "          " << scaled << " = {sign_bit, encoded_exponent, rounded["
         << fractionBits - 1 << ":0]};\n"
         << "        end\n"
         << "      end else begin\n"
         << "        shift_amount = " << fixedFractionBits << " + "
         << format.minimumExponent() << " - " << fractionBits << " - scale;\n"
         << "        rounded = " << round << "(magnitude, shift_amount);\n"
         << "        if (rounded[" << fractionBits << "])\n"
         << "          " << scaled << " = {sign_bit, {{" << exponentBits - 1
         << "{1'b0}}, 1'b1}, " << fractionBits << "'d0};\n"
         << "        else\n"
         << "          " << scaled << " = {sign_bit, " << exponentBits
         << "'d0, rounded[" << fractionBits - 1 << ":0]};\n"
         << "      end\n"
         << "    end\n"
         << "  end\n"
         << "endfunction\n";
  if (family == MathFamily::ExpM1) {
    output << "\nfunction automatic [" << width - 1 << ":0] " << fixed
           << "(input signed [" << fixedWidth - 1 << ":0] value);\n"
           << "  reg [" << fixedWidth - 1 << ":0] magnitude;\n"
           << "  reg sign_bit;\n"
           << "  begin\n"
           << "    sign_bit = value < 0;\n"
           << "    magnitude = sign_bit ? -value : value;\n"
           << "    " << fixed << " = " << scaled
           << "(magnitude, 0, sign_bit);\n"
           << "  end\n"
           << "endfunction\n";
  }
  return output.str();
}

std::string buildCoreFunction(MathFamily family, const Format &format,
                              llvm::StringRef functionName) {
  const unsigned width = format.width();
  const unsigned exponentBits = format.exponentBits;
  const unsigned fractionBits = format.fractionBits;
  const unsigned precision = format.precision();
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponentBits) - 1;
  const std::uint64_t infinity = exponentMask << fractionBits;
  const std::uint64_t quietBit = std::uint64_t{1} << (fractionBits - 1);
  const std::uint64_t negativeOne =
      (std::uint64_t{1} << (width - 1)) |
      (static_cast<std::uint64_t>(format.bias()) << fractionBits);
  const std::uint64_t nearZeroLimit =
      static_cast<std::uint64_t>(format.bias() - static_cast<int>(fractionBits))
      << fractionBits;
  const std::string multiply = helperName(family, format, "multiply_q24");
  const std::string round = helperName(family, format, "round_magnitude");
  const std::string pack = helperName(family, format, "pack_scaled");
  const std::string packFixed = helperName(family, format, "pack_fixed");
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic [" << width - 1 << ":0] " << functionName
         << "(input [" << width - 1 << ":0] operand);\n"
         << "  reg sign_input;\n"
         << "  reg [" << exponentBits - 1 << ":0] exponent_input;\n"
         << "  reg [" << fractionBits - 1 << ":0] fraction_input;\n"
         << "  reg [" << precision - 1 << ":0] significand;\n"
         << "  reg [" << fixedWidth - 1 << ":0] magnitude_fixed;\n"
         << "  reg [" << fixedWidth - 1 << ":0] scaled_exponential;\n"
         << "  reg [" << fixedWidth << ":0] rounded_exponential;\n"
         << "  reg signed [" << fixedWidth - 1 << ":0] x_fixed;\n"
         << "  reg signed [" << fixedWidth - 1
         << ":0] logarithmic_coordinate;\n"
         << "  reg signed [" << fixedWidth - 1 << ":0] range_reduced;\n"
         << "  reg signed [" << fixedWidth - 1 << ":0] polynomial;\n"
         << "  reg signed [" << fixedWidth - 1 << ":0] exponential_fixed;\n"
         << "  reg signed [" << fixedWidth - 1 << ":0] expm1_fixed;\n"
         << "  integer exponent_value;\n"
         << "  integer shift_amount;\n"
         << "  integer range_index;\n"
         << (family == MathFamily::ExpM1 ? "  reg near_zero;\n" : "")
         << "  begin\n"
         << "    " << functionName << " = "
         << hexLiteral(width, infinity | quietBit) << ";\n"
         << "    sign_input = operand[" << width - 1 << "];\n"
         << "    exponent_input = operand[" << width - 2 << ':' << fractionBits
         << "];\n"
         << "    fraction_input = operand[" << fractionBits - 1 << ":0];\n";
  if (family == MathFamily::ExpM1)
    output << "    near_zero = operand[" << width - 2
           << ":0] <= " << hexLiteral(width - 1, nearZeroLimit) << ";\n";
  output << "    if (exponent_input == "
         << hexLiteral(exponentBits, exponentMask)
         << " && fraction_input != 0) begin\n"
         << "      " << functionName << " = operand | "
         << hexLiteral(width, quietBit) << ";\n"
         << "    end else if (exponent_input == "
         << hexLiteral(exponentBits, exponentMask)
         << " && fraction_input == 0) begin\n";
  if (family == MathFamily::ExpM1)
    output << "      " << functionName << " = sign_input ? "
           << hexLiteral(width, negativeOne) << " : "
           << hexLiteral(width, infinity) << ";\n";
  else
    output << "      " << functionName << " = sign_input ? "
           << hexLiteral(width, 0) << " : " << hexLiteral(width, infinity)
           << ";\n";
  output << "    end else if (exponent_input == 0 && fraction_input == 0) "
            "begin\n";
  if (family == MathFamily::ExpM1)
    output << "      " << functionName << " = operand;\n";
  else
    output << "      " << functionName << " = "
           << hexLiteral(width, static_cast<std::uint64_t>(format.bias())
                                    << fractionBits)
           << ";\n";
  output << "    end else if (exponent_input != 0 && "
         << "integer'(exponent_input) - " << format.bias() << " >= 8) begin\n";
  if (family == MathFamily::ExpM1)
    output << "      " << functionName << " = sign_input ? "
           << hexLiteral(width, negativeOne) << " : "
           << hexLiteral(width, infinity) << ";\n";
  else
    output << "      " << functionName << " = sign_input ? "
           << hexLiteral(width, 0) << " : " << hexLiteral(width, infinity)
           << ";\n";
  if (family == MathFamily::ExpM1)
    output << "    end else if (near_zero) begin\n"
           << "      " << functionName << " = operand;\n";
  output << "    end else begin\n"
         << "      significand = exponent_input == 0 ? {1'b0, "
            "fraction_input} : {1'b1, fraction_input};\n"
         << "      exponent_value = integer'(exponent_input);\n"
         << "      exponent_value = exponent_input == 0 ? "
         << format.minimumExponent() << " : exponent_value - " << format.bias()
         << ";\n"
         << "      shift_amount = exponent_value - " << fractionBits << " + "
         << fixedFractionBits << ";\n"
         << "      magnitude_fixed = " << fixedWidth << "'d0;\n"
         << "      magnitude_fixed[" << precision - 1 << ":0] = significand;\n"
         << "      if (shift_amount >= 0)\n"
         << "        magnitude_fixed = magnitude_fixed << shift_amount;\n"
         << "      else\n"
         << "        magnitude_fixed = magnitude_fixed >> (-shift_amount);\n"
         << "      x_fixed = sign_input ? -$signed(magnitude_fixed) : "
            "$signed(magnitude_fixed);\n";
  if (family == MathFamily::Exp2) {
    output << "      range_index = x_fixed >= 0 ? "
              "$signed((x_fixed + "
           << signedLiteral(fixedWidth, fixedHalf) << ") >>> "
           << fixedFractionBits << ") : -$signed(((-x_fixed) + "
           << signedLiteral(fixedWidth, fixedHalf) << ") >>> "
           << fixedFractionBits << ");\n"
           << "      logarithmic_coordinate = x_fixed - range_index * "
           << signedLiteral(fixedWidth, fixedOne) << ";\n"
           << "      range_reduced = " << multiply
           << "(logarithmic_coordinate, " << signedLiteral(fixedWidth, fixedLn2)
           << ");\n";
  } else {
    output << "      logarithmic_coordinate = " << multiply << "(x_fixed, "
           << signedLiteral(fixedWidth, fixedLog2E) << ");\n"
           << "      range_index = logarithmic_coordinate >= 0 ? "
              "$signed((logarithmic_coordinate + "
           << signedLiteral(fixedWidth, fixedHalf) << ") >>> "
           << fixedFractionBits << ") : -$signed(((-logarithmic_coordinate) + "
           << signedLiteral(fixedWidth, fixedHalf) << ") >>> "
           << fixedFractionBits << ");\n"
           << "      range_reduced = x_fixed - range_index * "
           << signedLiteral(fixedWidth, fixedLn2) << ";\n";
  }
  output << "      polynomial = "
         << signedLiteral(fixedWidth, exponentialCoefficients.back()) << ";\n";
  for (std::size_t index = exponentialCoefficients.size() - 1; index != 0;
       --index)
    output << "      polynomial = " << multiply
           << "(polynomial, range_reduced) + "
           << signedLiteral(fixedWidth, exponentialCoefficients[index - 1])
           << ";\n";
  output << "      exponential_fixed = polynomial;\n";
  if (family == MathFamily::ExpM1) {
    output << "      if (range_index >= " << expm1DirectPackingRangeIndex
           << ") begin\n"
           << "        " << functionName << " = " << pack
           << "(exponential_fixed, range_index, 1'b0);\n"
           << "      end else if (range_index < " << expm1NegativeOneRangeIndex
           << ") begin\n"
           << "        " << functionName << " = "
           << hexLiteral(width, negativeOne) << ";\n"
           << "      end else begin\n"
           << "        if (range_index >= 0) begin\n"
           << "          scaled_exponential = exponential_fixed << "
              "range_index;\n"
           << "        end else begin\n"
           << "          rounded_exponential = " << round
           << "(exponential_fixed, -range_index);\n"
           << "          scaled_exponential = rounded_exponential["
           << fixedWidth - 1 << ":0];\n"
           << "        end\n"
           << "        expm1_fixed = $signed(scaled_exponential) - "
           << signedLiteral(fixedWidth, fixedOne) << ";\n"
           << "        " << functionName << " = " << packFixed
           << "(expm1_fixed);\n"
           << "      end\n";
  } else {
    output << "      " << functionName << " = " << pack
           << "(exponential_fixed, range_index, 1'b0);\n";
  }
  output << "    end\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

std::string buildModeDeclarations(MathFamily family, const LoweredMode &mode) {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << buildRoundProductFunction(family, mode.format) << '\n'
         << buildMultiplyFunction(family, mode.format) << '\n'
         << buildRoundMagnitudeFunction(family, mode.format) << '\n'
         << buildPackFunctions(family, mode.format) << '\n'
         << buildCoreFunction(family, mode.format, mode.functionName) << '\n';
  return output.str();
}

mlir::Value materializeMode(mlir::OpBuilder &builder, mlir::Location location,
                            circt::hw::HWModulePortAccessor &accessor,
                            const LoweredMode &mode, unsigned outputWidth) {
  const unsigned width = mode.format.width();
  mlir::Value operand = detail::resizeUnsigned(
      builder, location, accessor.getInput("data_input_0"), width);
  mlir::Value result = circt::sv::VerbatimExprOp::create(
      builder, location, builder.getIntegerType(width),
      mode.functionName + "({{0}})",
      llvm::SmallVector<mlir::Value, 1>{operand});
  return detail::resizeUnsigned(builder, location, result, outputWidth);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableMathExponential(FabricOperationProviderRequest request,
                                   MathFamily family) {
  const auto expectedFamily = familyId(family);
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return unsupported(request);
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
  if (!hasSupportedParameters(*parameters))
    return unsupported(request);

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
  const ConfigurationEncodingRelation *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  std::vector<Mode> modes;
  if (request.capability.configurationFieldSchema.empty()) {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None)
      return invalid(
          "configuration-free special-math relation is not fieldless");
    if (domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid(
          "configuration-free special-math relation is not a singleton");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid("configured special-math relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured special-math capability requires one field");
    field = request.configurationAbi.findOperationEncodingRelation(
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
    return invalid("sealed special-math behavior relation is empty");

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
    if (!functionNames.insert(lowered->functionName).second)
      return invalid("sealed relation contains a duplicate special-math mode");
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
          declarationStream << buildModeDeclarations(family, mode);
        circt::sv::VerbatimOp::create(
            bodyBuilder, location,
            bodyBuilder.getStringAttr(declarationStream.str()));

        std::vector<mlir::Value> results;
        results.reserve(loweredModes.size());
        for (const LoweredMode &mode : loweredModes)
          results.push_back(materializeMode(bodyBuilder, location, accessor,
                                            mode,
                                            outputs[0]->payloadWidthBits));

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
materializePortableScalarMathExp(FabricOperationProviderRequest request) {
  return materializePortableMathExponential(std::move(request),
                                            MathFamily::Exp);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarMathExp2(FabricOperationProviderRequest request) {
  return materializePortableMathExponential(std::move(request),
                                            MathFamily::Exp2);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarMathExpM1(FabricOperationProviderRequest request) {
  return materializePortableMathExponential(std::move(request),
                                            MathFamily::ExpM1);
}

} // namespace

llvm::Error registerPortableMathExponentialProviders(
    FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry staged = registry;
  if (llvm::Error error =
          staged.add({::fabric::ImplementationFamilyId::ScalarMathExp,
                      BackendRecipeKey::PortableSystemVerilog,
                      {},
                      materializePortableScalarMathExp}))
    return error;
  if (llvm::Error error =
          staged.add({::fabric::ImplementationFamilyId::ScalarMathExp2,
                      BackendRecipeKey::PortableSystemVerilog,
                      {},
                      materializePortableScalarMathExp2}))
    return error;
  if (llvm::Error error =
          staged.add({::fabric::ImplementationFamilyId::ScalarMathExpM1,
                      BackendRecipeKey::PortableSystemVerilog,
                      {},
                      materializePortableScalarMathExpM1}))
    return error;
  registry = std::move(staged);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl
