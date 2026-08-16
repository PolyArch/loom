#include "Hardware/RTL/Providers/MathPower.h"

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

using Format = detail::PortableFloatFormat;

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  Format format;
  std::string functionName;
};

constexpr std::int64_t fixedOne = INT64_C(1099511627776);
constexpr std::int64_t fixedHalf = INT64_C(549755813888);
constexpr std::int64_t fixedLn2 = INT64_C(762123384786);
constexpr std::int64_t fixedLog2E = INT64_C(1586259972792);
constexpr std::int64_t fixedSqrt2 = INT64_C(1554944255988);
constexpr std::int64_t fixedLimit = INT64_C(562949953421312);
constexpr std::int64_t fixedExponentLimit = INT64_C(1152921504606846976);
constexpr std::array<std::int64_t, 11> exponentialCoefficients = {
    INT64_C(1099511627776), INT64_C(1099511627776), INT64_C(549755813888),
    INT64_C(183251937963),  INT64_C(45812984491),   INT64_C(9162596898),
    INT64_C(1527099483),    INT64_C(218157069),     INT64_C(27269634),
    INT64_C(3029959),       INT64_C(302996)};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_math_power_invalid: " + message);
}

llvm::Error unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
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

std::string modeName(const Format &format) {
  return "loom_math_pow_e" + std::to_string(format.exponentBits) + "_f" +
         std::to_string(format.fractionBits) + "_max4";
}

llvm::Expected<LoweredMode>
lowerMode(const ::dataflow::CanonicalActorSchemaProjection &actor) {
  if (actor.schema != ::dataflow::OperationSchemaId::MathPowF ||
      actor.type.getNumInputs() != 2 || actor.type.getNumResults() != 1)
    return invalid("behavior is not binary floating power");
  if (actor.type.getInput(1) != actor.type.getInput(0) ||
      actor.type.getResult(0) != actor.type.getInput(0) ||
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
  return LoweredMode{*format, modeName(*format)};
}

std::string hexLiteral(unsigned width, std::uint64_t value) {
  std::ostringstream stream;
  stream << width << "'h" << std::hex << value;
  return stream.str();
}

std::string signedLiteral(unsigned width, std::int64_t value) {
  return std::to_string(width) + "'sd" + std::to_string(value);
}

std::string helperName(const Format &format, llvm::StringRef helper) {
  return modeName(format) + "_" + helper.str();
}

std::string buildArithmeticFunctions(const Format &format) {
  const std::string round = helperName(format, "round_shift_q40");
  const std::string multiply = helperName(format, "multiply_q40");
  const std::string saturating = helperName(format, "multiply_sat_q40");
  const std::string divide = helperName(format, "divide_q40");
  std::string text;
  llvm::raw_string_ostream output(text);
  output
      << "function automatic signed [63:0] " << round
      << "(input signed [127:0] value);\n"
      << "  reg signed [127:0] magnitude;\n"
      << "  reg signed [127:0] shifted;\n"
      << "  begin\n"
      << "    if (value < 0) begin\n"
      << "      magnitude = -value;\n"
      << "      shifted = (magnitude + " << signedLiteral(128, fixedHalf)
      << ") >>> 40;\n"
      << "      " << round << " = -$signed(shifted[63:0]);\n"
      << "    end else begin\n"
      << "      shifted = (value + " << signedLiteral(128, fixedHalf)
      << ") >>> 40;\n"
      << "      " << round << " = $signed(shifted[63:0]);\n"
      << "    end\n"
      << "  end\n"
      << "endfunction\n\n"
      << "function automatic signed [63:0] " << multiply
      << "(input signed [63:0] lhs, input signed [63:0] rhs);\n"
      << "  reg signed [127:0] product;\n"
      << "  begin\n"
      << "    product = lhs * rhs;\n"
      << "    " << multiply << " = " << round << "(product);\n"
      << "  end\n"
      << "endfunction\n\n"
      << "function automatic signed [63:0] " << saturating
      << "(input signed [63:0] lhs, input signed [63:0] rhs);\n"
      << "  reg signed [127:0] product;\n"
      << "  reg signed [127:0] scaled;\n"
      << "  begin\n"
      << "    product = lhs * rhs;\n"
      << "    scaled = product >>> 40;\n"
      << "    if (scaled > " << signedLiteral(128, fixedLimit) << ")\n"
      << "      " << saturating << " = " << signedLiteral(64, fixedLimit)
      << ";\n"
      << "    else if (scaled < -" << signedLiteral(128, fixedLimit) << ")\n"
      << "      " << saturating << " = -" << signedLiteral(64, fixedLimit)
      << ";\n"
      << "    else\n"
      << "      " << saturating << " = " << round << "(product);\n"
      << "  end\n"
      << "endfunction\n\n"
      << "function automatic signed [63:0] " << divide
      << "(input signed [63:0] numerator, input signed [63:0] denominator);\n"
      << "  reg signed [127:0] scaled;\n"
      << "  reg signed [127:0] denominator_extended;\n"
      << "  reg signed [127:0] quotient;\n"
      << "  begin\n"
      << "    scaled = {{64{numerator[63]}}, numerator};\n"
      << "    scaled = scaled <<< 40;\n"
      << "    denominator_extended = {{64{denominator[63]}}, denominator};\n"
      << "    quotient = scaled / denominator_extended;\n"
      << "    " << divide << " = quotient[63:0];\n"
      << "  end\n"
      << "endfunction\n";
  return output.str();
}

std::string buildPackFunctions(const Format &format) {
  const unsigned width = format.width();
  const unsigned exponentBits = format.exponentBits;
  const unsigned fractionBits = format.fractionBits;
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponentBits) - 1;
  const std::string round = helperName(format, "round_magnitude");
  const std::string pack = helperName(format, "pack_scaled");
  std::string text;
  llvm::raw_string_ostream output(text);
  output
      << "function automatic [64:0] " << round
      << "(input [63:0] value, input integer distance);\n"
      << "  reg guard;\n"
      << "  reg sticky;\n"
      << "  integer index;\n"
      << "  begin\n"
      << "    " << round << " = 65'd0;\n"
      << "    guard = 1'b0;\n"
      << "    sticky = 1'b0;\n"
      << "    if (distance <= 0) begin\n"
      << "      " << round << " = {1'b0, value} << (-distance);\n"
      << "    end else begin\n"
      << "      " << round << " = {1'b0, value >> distance};\n"
      << "      for (index = 0; index < 64; index = index + 1) begin\n"
      << "        if (index == distance - 1) guard = value[index];\n"
      << "        if (index < distance - 1) sticky = sticky | value[index];\n"
      << "      end\n"
      << "      if (guard && (sticky || " << round << "[0]))\n"
      << "        " << round << " = " << round << " + 1'b1;\n"
      << "    end\n"
      << "  end\n"
      << "endfunction\n\n"
      << "function automatic [" << width - 1 << ":0] " << pack
      << "(input [63:0] magnitude, input integer scale, input sign_bit);\n"
      << "  reg [64:0] rounded;\n"
      << "  reg found;\n"
      << "  reg [" << exponentBits - 1 << ":0] encoded_exponent;\n"
      << "  integer encoded_exponent_value;\n"
      << "  integer leading_index;\n"
      << "  integer result_exponent;\n"
      << "  integer shift_amount;\n"
      << "  integer index;\n"
      << "  begin\n"
      << "    " << pack << " = {sign_bit, " << exponentBits << "'d0, "
      << fractionBits << "'d0};\n"
      << "    if (magnitude != 0) begin\n"
      << "      leading_index = 0;\n"
      << "      found = 1'b0;\n"
      << "      for (index = 63; index >= 0; index = index - 1) begin\n"
      << "        if (!found && magnitude[index]) begin\n"
      << "          leading_index = index;\n"
      << "          found = 1'b1;\n"
      << "        end\n"
      << "      end\n"
      << "      result_exponent = leading_index + scale - 40;\n"
      << "      if (result_exponent > " << format.maximumExponent()
      << ") begin\n"
      << "        " << pack << " = {sign_bit, "
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
      << "          " << pack << " = {sign_bit, "
      << hexLiteral(exponentBits, exponentMask) << ", " << fractionBits
      << "'d0};\n"
      << "        end else begin\n"
      << "          encoded_exponent_value = result_exponent + "
      << format.bias() << ";\n"
      << "          encoded_exponent = encoded_exponent_value["
      << exponentBits - 1 << ":0];\n"
      << "          " << pack << " = {sign_bit, encoded_exponent, rounded["
      << fractionBits - 1 << ":0]};\n"
      << "        end\n"
      << "      end else begin\n"
      << "        shift_amount = 40 + " << format.minimumExponent() << " - "
      << fractionBits << " - scale;\n"
      << "        rounded = " << round << "(magnitude, shift_amount);\n"
      << "        if (rounded[" << fractionBits << "])\n"
      << "          " << pack << " = {sign_bit, {{" << exponentBits - 1
      << "{1'b0}}, 1'b1}, " << fractionBits << "'d0};\n"
      << "        else\n"
      << "          " << pack << " = {sign_bit, " << exponentBits
      << "'d0, rounded[" << fractionBits - 1 << ":0]};\n"
      << "      end\n"
      << "    end\n"
      << "  end\n"
      << "endfunction\n";
  return output.str();
}

std::string buildLogarithmFunction(const Format &format) {
  const std::string name = helperName(format, "log2_q40");
  const std::string multiply = helperName(format, "multiply_q40");
  const std::string divide = helperName(format, "divide_q40");
  std::string text;
  llvm::raw_string_ostream output(text);
  output
      << "function automatic signed [63:0] " << name
      << "(input signed [63:0] input_mantissa, input integer input_exponent);\n"
      << "  reg signed [63:0] mantissa;\n"
      << "  reg signed [63:0] z;\n"
      << "  reg signed [63:0] z_squared;\n"
      << "  reg signed [63:0] term;\n"
      << "  reg signed [63:0] sum;\n"
      << "  reg signed [63:0] exponent_term;\n"
      << "  integer exponent_value;\n"
      << "  begin\n"
      << "    mantissa = input_mantissa;\n"
      << "    exponent_value = input_exponent;\n"
      << "    if (mantissa > " << signedLiteral(64, fixedSqrt2) << ") begin\n"
      << "      mantissa = mantissa >>> 1;\n"
      << "      exponent_value = exponent_value + 1;\n"
      << "    end\n"
      << "    z = " << divide << "(mantissa - " << signedLiteral(64, fixedOne)
      << ", mantissa + " << signedLiteral(64, fixedOne) << ");\n"
      << "    z_squared = " << multiply << "(z, z);\n"
      << "    term = z;\n"
      << "    sum = z;\n";
  for (unsigned denominator : {3U, 5U, 7U, 9U, 11U, 13U, 15U, 17U})
    output << "    term = " << multiply << "(term, z_squared);\n"
           << "    sum = sum + term / 64'sd" << denominator << ";\n";
  output << "    exponent_term = {{32{exponent_value[31]}}, "
            "exponent_value};\n"
         << "    exponent_term = exponent_term * "
         << signedLiteral(64, fixedOne) << ";\n"
         << "    " << name << " = " << multiply << "(sum <<< 1, "
         << signedLiteral(64, fixedLog2E) << ") + exponent_term;\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

std::string buildCoreFunction(const LoweredMode &mode) {
  const Format &format = mode.format;
  const unsigned width = format.width();
  const unsigned exponentBits = format.exponentBits;
  const unsigned fractionBits = format.fractionBits;
  const unsigned precision = format.precision();
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponentBits) - 1;
  const std::uint64_t infinity = exponentMask << fractionBits;
  const std::uint64_t quietBit = std::uint64_t{1} << (fractionBits - 1);
  const std::uint64_t quietNaN = infinity | quietBit;
  const std::uint64_t one = static_cast<std::uint64_t>(format.bias())
                            << fractionBits;
  const std::string multiply = helperName(format, "multiply_q40");
  const std::string saturating = helperName(format, "multiply_sat_q40");
  const std::string logarithm = helperName(format, "log2_q40");
  const std::string pack = helperName(format, "pack_scaled");
  const std::string exponentAllOnes = hexLiteral(exponentBits, exponentMask);
  std::string text;
  llvm::raw_string_ostream output(text);
  output
      << "function automatic [" << width - 1 << ":0] " << mode.functionName
      << "(input [" << width - 1 << ":0] base, input [" << width - 1
      << ":0] exponent);\n"
      << "  reg sign_base;\n"
      << "  reg sign_exponent;\n"
      << "  reg sign_result;\n"
      << "  reg [" << exponentBits - 1 << ":0] exponent_base;\n"
      << "  reg [" << exponentBits - 1 << ":0] exponent_exponent;\n"
      << "  reg [" << fractionBits - 1 << ":0] fraction_base;\n"
      << "  reg [" << fractionBits - 1 << ":0] fraction_exponent;\n"
      << "  reg [" << precision - 1 << ":0] significand_base;\n"
      << "  reg [" << precision - 1 << ":0] significand_exponent;\n"
      << "  reg [63:0] magnitude_exponent;\n"
      << "  reg signed [63:0] mantissa_q;\n"
      << "  reg signed [63:0] logarithm_q;\n"
      << "  reg signed [63:0] exponent_q;\n"
      << "  reg signed [63:0] power_q;\n"
      << "  reg signed [63:0] range_reduced;\n"
      << "  reg signed [63:0] polynomial;\n"
      << "  reg exponent_is_integer;\n"
      << "  reg exponent_is_odd;\n"
      << "  reg discarded_bit;\n"
      << "  integer base_exponent_value;\n"
      << "  integer exponent_exponent_value;\n"
      << "  integer shift_amount;\n"
      << "  integer range_index;\n"
      << "  integer index;\n"
      << "  begin\n"
      << "    " << mode.functionName << " = " << hexLiteral(width, quietNaN)
      << ";\n"
      << "    sign_base = base[" << width - 1 << "];\n"
      << "    sign_exponent = exponent[" << width - 1 << "];\n"
      << "    exponent_base = base[" << width - 2 << ':' << fractionBits
      << "];\n"
      << "    exponent_exponent = exponent[" << width - 2 << ':' << fractionBits
      << "];\n"
      << "    fraction_base = base[" << fractionBits - 1 << ":0];\n"
      << "    fraction_exponent = exponent[" << fractionBits - 1 << ":0];\n"
      << "    significand_exponent = exponent_exponent == 0"
         " ? {1'b0, fraction_exponent} : {1'b1, fraction_exponent};\n"
      << "    exponent_exponent_value = integer'(exponent_exponent) - "
      << format.bias() << ";\n"
      << "    exponent_is_integer = 1'b0;\n"
      << "    exponent_is_odd = 1'b0;\n"
      << "    discarded_bit = 1'b0;\n"
      << "    if (exponent_exponent == 0) begin\n"
      << "      exponent_is_integer = fraction_exponent == 0;\n"
      << "    end else if (exponent_exponent_value >= " << fractionBits
      << ") begin\n"
      << "      exponent_is_integer = 1'b1;\n"
      << "      if (exponent_exponent_value == " << fractionBits << ")\n"
      << "        exponent_is_odd = significand_exponent[0];\n"
      << "    end else if (exponent_exponent_value >= 0) begin\n"
      << "      shift_amount = " << fractionBits
      << " - exponent_exponent_value;\n"
      << "      for (index = 0; index < " << fractionBits
      << "; index = index + 1)\n"
      << "        if (index < shift_amount) "
         "discarded_bit = discarded_bit | significand_exponent[index];\n"
      << "      if (!discarded_bit) begin\n"
      << "        exponent_is_integer = 1'b1;\n"
      << "        exponent_is_odd = significand_exponent[shift_amount];\n"
      << "      end\n"
      << "    end\n"
      << "    sign_result = sign_base && exponent_is_odd;\n"
      << "    if (exponent[" << width - 2 << ":0] == 0) begin\n"
      << "      " << mode.functionName << " = " << hexLiteral(width, one)
      << ";\n"
      << "    end else if (base == " << hexLiteral(width, one) << ") begin\n"
      << "      " << mode.functionName << " = " << hexLiteral(width, one)
      << ";\n"
      << "    end else if (exponent_base == " << exponentAllOnes
      << " && fraction_base != 0) begin\n"
      << "      " << mode.functionName << " = base | "
      << hexLiteral(width, quietBit) << ";\n"
      << "    end else if (exponent_exponent == " << exponentAllOnes
      << " && fraction_exponent != 0) begin\n"
      << "      " << mode.functionName << " = exponent | "
      << hexLiteral(width, quietBit) << ";\n"
      << "    end else if (exponent_exponent == " << exponentAllOnes
      << ") begin\n"
      << "      if (base[" << width - 2
      << ":0] == " << hexLiteral(width - 1, one) << ")\n"
      << "        " << mode.functionName << " = " << hexLiteral(width, one)
      << ";\n"
      << "      else if (base[" << width - 2 << ":0] > "
      << hexLiteral(width - 1, one) << ")\n"
      << "        " << mode.functionName << " = sign_exponent ? "
      << hexLiteral(width, 0) << " : " << hexLiteral(width, infinity) << ";\n"
      << "      else\n"
      << "        " << mode.functionName << " = sign_exponent ? "
      << hexLiteral(width, infinity) << " : " << hexLiteral(width, 0) << ";\n"
      << "    end else if (exponent_base == " << exponentAllOnes << ") begin\n"
      << "      " << mode.functionName << " = {sign_result, sign_exponent ? "
      << width - 1 << "'d0 : " << hexLiteral(width - 1, infinity) << "};\n"
      << "    end else if (exponent_base == 0 && fraction_base == 0) begin\n"
      << "      " << mode.functionName << " = {sign_result, sign_exponent ? "
      << hexLiteral(width - 1, infinity) << " : " << width - 1 << "'d0};\n"
      << "    end else if (sign_base && !exponent_is_integer) begin\n"
      << "      " << mode.functionName << " = " << hexLiteral(width, quietNaN)
      << ";\n"
      << "    end else if (exponent == " << hexLiteral(width, one)
      << ") begin\n"
      << "      " << mode.functionName << " = base;\n"
      << "    end else begin\n"
      << "      significand_base = exponent_base == 0"
         " ? {1'b0, fraction_base} : {1'b1, fraction_base};\n"
      << "      base_exponent_value = integer'(exponent_base);\n"
      << "      base_exponent_value = exponent_base == 0 ? "
      << format.minimumExponent() << " : base_exponent_value - "
      << format.bias() << ";\n"
      << "      for (index = 0; index < " << precision
      << "; index = index + 1) begin\n"
      << "        if (!significand_base[" << precision - 1 << "]) begin\n"
      << "          significand_base = significand_base << 1;\n"
      << "          base_exponent_value = base_exponent_value - 1;\n"
      << "        end\n"
      << "      end\n"
      << "      mantissa_q = {{" << 64 - precision
      << "{1'b0}}, significand_base};\n"
      << "      mantissa_q = mantissa_q <<< " << 40 - fractionBits << ";\n"
      << "      logarithm_q = " << logarithm
      << "(mantissa_q, base_exponent_value);\n"
      << "      exponent_exponent_value = exponent_exponent == 0 ? "
      << format.minimumExponent() << " : integer'(exponent_exponent) - "
      << format.bias() << ";\n"
      << "      magnitude_exponent = 64'd0;\n"
      << "      if (exponent_exponent_value >= 20) begin\n"
      << "        magnitude_exponent = 64'd"
      << static_cast<std::uint64_t>(fixedExponentLimit) << ";\n"
      << "      end else begin\n"
      << "        shift_amount = exponent_exponent_value - " << fractionBits
      << " + 40;\n"
      << "        magnitude_exponent[" << precision - 1
      << ":0] = significand_exponent;\n"
      << "        if (shift_amount >= 0)\n"
      << "          magnitude_exponent = magnitude_exponent << shift_amount;\n"
      << "        else if (shift_amount > -64)\n"
      << "          magnitude_exponent = magnitude_exponent >> "
         "(-shift_amount);\n"
      << "        else\n"
      << "          magnitude_exponent = 64'd0;\n"
      << "      end\n"
      << "      exponent_q = sign_exponent ? -$signed(magnitude_exponent)"
         " : $signed(magnitude_exponent);\n"
      << "      power_q = " << saturating << "(logarithm_q, exponent_q);\n"
      << "      range_index = integer'(power_q >= 0 ? "
         "$signed((power_q + "
      << signedLiteral(64, fixedHalf) << ") >>> 40) : -$signed(((-power_q) + "
      << signedLiteral(64, fixedHalf) << ") >>> 40));\n"
      << "      range_reduced = power_q - range_index * "
      << signedLiteral(64, fixedOne) << ";\n"
      << "      range_reduced = " << multiply << "(range_reduced, "
      << signedLiteral(64, fixedLn2) << ");\n"
      << "      polynomial = "
      << signedLiteral(64, exponentialCoefficients.back()) << ";\n";
  for (std::size_t index = exponentialCoefficients.size() - 1; index != 0;
       --index)
    output << "      polynomial = " << multiply
           << "(polynomial, range_reduced) + "
           << signedLiteral(64, exponentialCoefficients[index - 1]) << ";\n";
  output << "      " << mode.functionName << " = " << pack
         << "(polynomial, range_index, sign_result);\n"
         << "    end\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

std::string buildModeDeclarations(const LoweredMode &mode) {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << buildArithmeticFunctions(mode.format) << '\n'
         << buildPackFunctions(mode.format) << '\n'
         << buildLogarithmFunction(mode.format) << '\n'
         << buildCoreFunction(mode) << '\n';
  return output.str();
}

mlir::Value materializeMode(mlir::OpBuilder &builder, mlir::Location location,
                            circt::hw::HWModulePortAccessor &accessor,
                            const LoweredMode &mode, unsigned outputWidth) {
  const unsigned width = mode.format.width();
  mlir::Value base = detail::resizeUnsigned(
      builder, location, accessor.getInput("data_input_0"), width);
  mlir::Value exponent = detail::resizeUnsigned(
      builder, location, accessor.getInput("data_input_1"), width);
  mlir::Value result = circt::sv::VerbatimExprOp::create(
      builder, location, builder.getIntegerType(width),
      mode.functionName + "({{0}}, {{1}})",
      llvm::SmallVector<mlir::Value, 2>{base, exponent});
  return detail::resizeUnsigned(builder, location, result, outputWidth);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableMathPower(FabricOperationProviderRequest request) {
  constexpr auto family = ::fabric::ImplementationFamilyId::ScalarMathPow;
  constexpr auto schema = ::dataflow::OperationSchemaId::MathPowF;
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return unsupported(request);
  if (request.capability.implementationFamily != family)
    return invalid("provider received a different implementation family");
  const auto &descriptor = ::fabric::implementationFamily(family);
  if (descriptor.familyId != family ||
      ::fabric::capabilityParamsSchema(
          request.capability.parameterizedCapability) !=
          descriptor.capabilityParamsSchema)
    return invalid("capability does not match its generated family descriptor");
  const auto *parameters = std::get_if<::fabric::ScalarSpecialMathParams>(
      &request.capability.parameterizedCapability);
  if (!parameters)
    return invalid("capability has the wrong special-math parameter schema");
  if (request.capability.enabledOperationSchemas !=
          std::vector<::dataflow::OperationSchemaId>{schema} ||
      descriptor.admittedSchemas.size() != 1 ||
      descriptor.admittedSchemas.front() != schema)
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
  if (inputs.size() != 2 || outputs.size() != 1 ||
      inputs[0]->reference.ordinal != 0 || inputs[1]->reference.ordinal != 1 ||
      outputs[0]->reference.ordinal != 0)
    return unsupported(request);
  if (inputs[0]->payloadWidthBits == 0 || inputs[1]->payloadWidthBits == 0 ||
      outputs[0]->payloadWidthBits == 0)
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
      return invalid("configuration-free math power relation is not fieldless");
    if (domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid("configuration-free math power relation is not singleton");
    if (domain.front().operandPorts != std::vector<std::uint64_t>({0, 1}) ||
        domain.front().resultPorts != std::vector<std::uint64_t>({0}))
      return invalid("sealed relation has incorrect physical correspondence");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid("configured math power relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured math power capability requires one field");
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
      if (point.operandPorts != std::vector<std::uint64_t>({0, 1}) ||
          point.resultPorts != std::vector<std::uint64_t>({0}))
        return invalid("sealed relation has incorrect physical correspondence");
      const FiniteCodebookEntry *entry = detail::findFiniteCodebookEntry(
          *codebook, point.semanticConfiguration->bytes());
      if (!entry)
        return invalid("codebook omits an admitted semantic value");
      modes.push_back({point.representativeActor, entry});
    }
  }
  if (modes.empty())
    return invalid("sealed math power behavior relation is empty");

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
    auto lowered = lowerMode(mode.actor);
    if (!lowered)
      return lowered.takeError();
    if (!functionNames.insert(lowered->functionName).second)
      return invalid("sealed relation contains a duplicate math power mode");
    const unsigned width = lowered->format.width();
    if (width > inputs[0]->payloadWidthBits ||
        width > inputs[1]->payloadWidthBits ||
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
          declarationStream << buildModeDeclarations(mode);
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

} // namespace

llvm::Error
registerPortableMathPowerProvider(FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry staged = registry;
  if (llvm::Error error =
          staged.add({::fabric::ImplementationFamilyId::ScalarMathPow,
                      BackendRecipeKey::PortableSystemVerilog,
                      {},
                      materializePortableMathPower}))
    return error;
  registry = std::move(staged);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl
