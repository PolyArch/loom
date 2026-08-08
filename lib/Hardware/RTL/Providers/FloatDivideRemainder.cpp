#include "Hardware/RTL/Providers/FloatDivideRemainder.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

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

enum class FloatFamily { Divide, Remainder };
using Format = detail::PortableFloatFormat;

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  Format format;
  mlir::arith::RoundingMode rounding;
  std::string functionName;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_float_divide_remainder_invalid: " +
                                     message);
}

llvm::Error unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

::fabric::ImplementationFamilyId familyId(FloatFamily family) {
  return family == FloatFamily::Divide
             ? ::fabric::ImplementationFamilyId::ScalarFloatDivide
             : ::fabric::ImplementationFamilyId::ScalarFloatRemainder;
}

::dataflow::OperationSchemaId schemaId(FloatFamily family) {
  return family == FloatFamily::Divide
             ? ::dataflow::OperationSchemaId::ArithDivF
             : ::dataflow::OperationSchemaId::ArithRemF;
}

llvm::StringRef roundingSuffix(mlir::arith::RoundingMode rounding) {
  using RoundingMode = mlir::arith::RoundingMode;
  switch (rounding) {
  case RoundingMode::to_nearest_even:
    return "rne";
  case RoundingMode::downward:
    return "rdn";
  case RoundingMode::upward:
    return "rup";
  case RoundingMode::toward_zero:
    return "rtz";
  case RoundingMode::to_nearest_away:
    return "rna";
  }
  llvm_unreachable("unknown floating rounding mode");
}

unsigned roundingCode(mlir::arith::RoundingMode rounding) {
  return static_cast<unsigned>(rounding);
}

std::string coreName(FloatFamily family, const Format &format) {
  return std::string("loom_float_") +
         (family == FloatFamily::Divide ? "divide" : "remainder") + "_e" +
         std::to_string(format.exponentBits) + "_f" +
         std::to_string(format.fractionBits) + "_core";
}

std::string modeName(FloatFamily family, const Format &format,
                     mlir::arith::RoundingMode rounding) {
  std::string name = std::string("loom_float_") +
                     (family == FloatFamily::Divide ? "divide" : "remainder") +
                     "_e" + std::to_string(format.exponentBits) + "_f" +
                     std::to_string(format.fractionBits);
  if (family == FloatFamily::Divide)
    name += "_" + roundingSuffix(rounding).str();
  return name;
}

llvm::Expected<LoweredMode>
lowerMode(FloatFamily family,
          const ::dataflow::CanonicalActorSchemaProjection &actor) {
  if (actor.schema != schemaId(family) || actor.type.getNumInputs() != 2 ||
      actor.type.getNumResults() != 1)
    return invalid("behavior is not the selected binary floating operation");
  if (actor.type.getInput(1) != actor.type.getInput(0) ||
      actor.type.getResult(0) != actor.type.getInput(0) ||
      llvm::isa<mlir::VectorType>(actor.type.getInput(0)))
    return invalid("behavior does not have one uniform scalar floating type");
  const auto *payload =
      std::get_if<::dataflow::FloatingPointPayload>(&actor.payload);
  if (!payload)
    return invalid("behavior has no floating-point payload");
  auto format = detail::resolvePortableFloatFormat(actor.type.getInput(0));
  if (!format)
    return invalid("behavior uses an unsupported floating format");
  if (family == FloatFamily::Remainder && payload->roundingMode)
    return invalid("remainder behavior unexpectedly selects rounding");
  const mlir::arith::RoundingMode rounding = payload->roundingMode.value_or(
      mlir::arith::RoundingMode::to_nearest_even);
  return LoweredMode{*format, rounding, modeName(family, *format, rounding)};
}

std::string hexLiteral(unsigned width, std::uint64_t value) {
  std::ostringstream stream;
  stream << width << "'h" << std::hex << value;
  return stream.str();
}

std::string buildDivideCoreFunction(const Format &format) {
  const unsigned width = format.width();
  const unsigned exponentBits = format.exponentBits;
  const unsigned fractionBits = format.fractionBits;
  const unsigned precision = format.precision();
  const unsigned dividendWidth = precision + fractionBits + 3;
  const unsigned extendedWidth = precision + 3;
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponentBits) - 1;
  const std::uint64_t fractionMask = (std::uint64_t{1} << fractionBits) - 1;
  const std::uint64_t infinity = exponentMask << fractionBits;
  const std::uint64_t quietBit = std::uint64_t{1} << (fractionBits - 1);
  const std::uint64_t quietNaN = infinity | quietBit;
  const std::string name = coreName(FloatFamily::Divide, format);
  const std::string exponentAllOnes = hexLiteral(exponentBits, exponentMask);
  const std::string exponentMaxFinite =
      hexLiteral(exponentBits, exponentMask - 1);
  const std::string fractionAllOnes = hexLiteral(fractionBits, fractionMask);

  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic [" << width - 1 << ":0] " << name << "(input ["
         << width - 1 << ":0] lhs, input [" << width - 1
         << ":0] rhs, input [2:0] rounding);\n"
         << "  reg sign_result;\n"
         << "  reg [" << exponentBits - 1 << ":0] exponent_lhs;\n"
         << "  reg [" << exponentBits - 1 << ":0] exponent_rhs;\n"
         << "  reg [" << exponentBits - 1 << ":0] exponent_result;\n"
         << "  reg [" << fractionBits - 1 << ":0] fraction_lhs;\n"
         << "  reg [" << fractionBits - 1 << ":0] fraction_rhs;\n"
         << "  reg [" << fractionBits - 1 << ":0] fraction_result;\n"
         << "  reg [" << precision - 1 << ":0] significand_lhs;\n"
         << "  reg [" << precision - 1 << ":0] significand_rhs;\n"
         << "  reg [" << precision - 1 << ":0] normalized_lhs;\n"
         << "  reg [" << precision - 1 << ":0] normalized_rhs;\n"
         << "  reg [" << dividendWidth - 1 << ":0] normalized_rhs_wide;\n"
         << "  reg [" << dividendWidth - 1 << ":0] dividend;\n"
         << "  reg [" << dividendWidth - 1 << ":0] quotient_full;\n"
         << "  reg [" << dividendWidth - 1 << ":0] remainder_full;\n"
         << "  reg [" << precision - 1 << ":0] division_remainder;\n"
         << "  reg [" << extendedWidth - 1 << ":0] extended_quotient;\n"
         << "  reg [" << extendedWidth - 1 << ":0] shifted_quotient;\n"
         << "  reg [" << precision << ":0] rounded;\n"
         << "  reg found_lhs;\n"
         << "  reg found_rhs;\n"
         << "  reg guard;\n"
         << "  reg sticky;\n"
         << "  reg increment;\n"
         << "  reg overflow_to_infinity;\n"
         << "  integer exponent_lhs_value;\n"
         << "  integer exponent_rhs_value;\n"
         << "  integer normalized_exponent_lhs;\n"
         << "  integer normalized_exponent_rhs;\n"
         << "  integer result_exponent_value;\n"
         << "  integer encoded_exponent;\n"
         << "  integer leading_lhs;\n"
         << "  integer leading_rhs;\n"
         << "  integer shift_amount;\n"
         << "  integer index;\n"
         << "  begin\n"
         << "    " << name << " = " << hexLiteral(width, quietNaN) << ";\n"
         << "    sign_result = lhs[" << width - 1 << "] ^ rhs[" << width - 1
         << "];\n"
         << "    exponent_lhs = lhs[" << width - 2 << ':' << fractionBits
         << "];\n"
         << "    exponent_rhs = rhs[" << width - 2 << ':' << fractionBits
         << "];\n"
         << "    fraction_lhs = lhs[" << fractionBits - 1 << ":0];\n"
         << "    fraction_rhs = rhs[" << fractionBits - 1 << ":0];\n"
         << "    if (exponent_lhs == " << exponentAllOnes
         << " && fraction_lhs != 0) begin\n"
         << "      " << name << " = lhs | " << hexLiteral(width, quietBit)
         << ";\n"
         << "    end else if (exponent_rhs == " << exponentAllOnes
         << " && fraction_rhs != 0) begin\n"
         << "      " << name << " = rhs | " << hexLiteral(width, quietBit)
         << ";\n"
         << "    end else if (((exponent_lhs == 0 && fraction_lhs == 0) &&\n"
         << "                  (exponent_rhs == 0 && fraction_rhs == 0)) ||\n"
         << "                 ((exponent_lhs == " << exponentAllOnes
         << " && fraction_lhs == 0) &&\n"
         << "                  (exponent_rhs == " << exponentAllOnes
         << " && fraction_rhs == 0))) begin\n"
         << "      " << name << " = " << hexLiteral(width, quietNaN) << ";\n"
         << "    end else if (exponent_lhs == " << exponentAllOnes
         << " && fraction_lhs == 0) begin\n"
         << "      " << name << " = {sign_result, " << exponentAllOnes << ", "
         << fractionBits << "'d0};\n"
         << "    end else if (exponent_rhs == " << exponentAllOnes
         << " && fraction_rhs == 0) begin\n"
         << "      " << name << " = {sign_result, " << exponentBits << "'d0, "
         << fractionBits << "'d0};\n"
         << "    end else if (exponent_rhs == 0 && fraction_rhs == 0) begin\n"
         << "      " << name << " = {sign_result, " << exponentAllOnes << ", "
         << fractionBits << "'d0};\n"
         << "    end else if (exponent_lhs == 0 && fraction_lhs == 0) begin\n"
         << "      " << name << " = {sign_result, " << exponentBits << "'d0, "
         << fractionBits << "'d0};\n"
         << "    end else begin\n"
         << "      significand_lhs = exponent_lhs == 0"
            " ? {1'b0, fraction_lhs} : {1'b1, fraction_lhs};\n"
         << "      significand_rhs = exponent_rhs == 0"
            " ? {1'b0, fraction_rhs} : {1'b1, fraction_rhs};\n"
         << "      exponent_lhs_value = integer'(exponent_lhs);\n"
         << "      exponent_rhs_value = integer'(exponent_rhs);\n"
         << "      exponent_lhs_value = exponent_lhs == 0 ? "
         << format.minimumExponent() << " : exponent_lhs_value - "
         << format.bias() << ";\n"
         << "      exponent_rhs_value = exponent_rhs == 0 ? "
         << format.minimumExponent() << " : exponent_rhs_value - "
         << format.bias() << ";\n"
         << "      leading_lhs = 0;\n"
         << "      leading_rhs = 0;\n"
         << "      found_lhs = 1'b0;\n"
         << "      found_rhs = 1'b0;\n"
         << "      for (index = " << precision - 1
         << "; index >= 0; index = index - 1) begin\n"
         << "        if (!found_lhs && significand_lhs[index]) begin\n"
         << "          leading_lhs = index;\n"
         << "          found_lhs = 1'b1;\n"
         << "        end\n"
         << "        if (!found_rhs && significand_rhs[index]) begin\n"
         << "          leading_rhs = index;\n"
         << "          found_rhs = 1'b1;\n"
         << "        end\n"
         << "      end\n"
         << "      normalized_lhs = significand_lhs << (" << fractionBits
         << " - leading_lhs);\n"
         << "      normalized_rhs = significand_rhs << (" << fractionBits
         << " - leading_rhs);\n"
         << "      normalized_exponent_lhs = exponent_lhs_value - "
         << fractionBits << " + leading_lhs;\n"
         << "      normalized_exponent_rhs = exponent_rhs_value - "
         << fractionBits << " + leading_rhs;\n"
         << "      dividend = normalized_lhs << " << fractionBits + 3 << ";\n"
         << "      normalized_rhs_wide = {{" << dividendWidth - precision
         << "{1'b0}}, normalized_rhs};\n"
         << "      quotient_full = dividend / normalized_rhs_wide;\n"
         << "      remainder_full = dividend % normalized_rhs_wide;\n"
         << "      division_remainder = remainder_full[" << precision - 1
         << ":0];\n"
         << "      extended_quotient = quotient_full[" << extendedWidth - 1
         << ":0];\n"
         << "      result_exponent_value = normalized_exponent_lhs - "
            "normalized_exponent_rhs;\n"
         << "      if (normalized_lhs < normalized_rhs) begin\n"
         << "        extended_quotient = extended_quotient << 1;\n"
         << "        result_exponent_value = result_exponent_value - 1;\n"
         << "      end\n"
         << "      shift_amount = 3;\n"
         << "      if (result_exponent_value < " << format.minimumExponent()
         << ")\n"
         << "        shift_amount = shift_amount + " << format.minimumExponent()
         << " - result_exponent_value;\n"
         << "      shifted_quotient = extended_quotient >> shift_amount;\n"
         << "      rounded = {1'b0, shifted_quotient[" << precision - 1
         << ":0]};\n"
         << "      guard = 1'b0;\n"
         << "      sticky = division_remainder != 0;\n"
         << "      for (index = 0; index < " << extendedWidth
         << "; index = index + 1) begin\n"
         << "        if (index == shift_amount - 1)"
            " guard = extended_quotient[index];\n"
         << "        if (index < shift_amount - 1)"
            " sticky = sticky | extended_quotient[index];\n"
         << "      end\n"
         << "      if (shift_amount > " << extendedWidth
         << ") sticky = sticky | (|extended_quotient);\n"
         << "      increment = 1'b0;\n"
         << "      case (rounding)\n"
         << "        3'd0: increment = guard && (sticky || rounded[0]);\n"
         << "        3'd1: increment = sign_result && (guard || sticky);\n"
         << "        3'd2: increment = !sign_result && (guard || sticky);\n"
         << "        3'd3: increment = 1'b0;\n"
         << "        3'd4: increment = guard;\n"
         << "        default: increment = guard && (sticky || rounded[0]);\n"
         << "      endcase\n"
         << "      rounded = rounded + increment;\n"
         << "      if (result_exponent_value >= " << format.minimumExponent()
         << ") begin\n"
         << "        if (rounded[" << precision << "]) begin\n"
         << "          rounded = rounded >> 1;\n"
         << "          result_exponent_value = result_exponent_value + 1;\n"
         << "        end\n"
         << "        if (result_exponent_value > " << format.maximumExponent()
         << ") begin\n"
         << "          overflow_to_infinity ="
            " rounding == 3'd0 || rounding == 3'd4 ||\n"
         << "              (rounding == 3'd2 && !sign_result) ||\n"
         << "              (rounding == 3'd1 && sign_result);\n"
         << "          if (overflow_to_infinity)\n"
         << "            " << name << " = {sign_result, " << exponentAllOnes
         << ", " << fractionBits << "'d0};\n"
         << "          else\n"
         << "            " << name << " = {sign_result, " << exponentMaxFinite
         << ", " << fractionAllOnes << "};\n"
         << "        end else begin\n"
         << "          encoded_exponent = result_exponent_value + "
         << format.bias() << ";\n"
         << "          exponent_result = encoded_exponent[" << exponentBits - 1
         << ":0];\n"
         << "          fraction_result = rounded[" << fractionBits - 1
         << ":0];\n"
         << "          " << name
         << " = {sign_result, exponent_result, fraction_result};\n"
         << "        end\n"
         << "      end else if (rounded[" << fractionBits << "]) begin\n"
         << "        " << name << " = {sign_result, " << exponentBits << "'d1, "
         << fractionBits << "'d0};\n"
         << "      end else begin\n"
         << "        fraction_result = rounded[" << fractionBits - 1 << ":0];\n"
         << "        " << name << " = {sign_result, " << exponentBits
         << "'d0, fraction_result};\n"
         << "      end\n"
         << "    end\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

std::string buildRemainderCoreFunction(const Format &format) {
  const unsigned width = format.width();
  const unsigned exponentBits = format.exponentBits;
  const unsigned fractionBits = format.fractionBits;
  const unsigned precision = format.precision();
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponentBits) - 1;
  const std::uint64_t infinity = exponentMask << fractionBits;
  const std::uint64_t quietBit = std::uint64_t{1} << (fractionBits - 1);
  const std::uint64_t quietNaN = infinity | quietBit;
  const std::string name = coreName(FloatFamily::Remainder, format);
  const std::string exponentAllOnes = hexLiteral(exponentBits, exponentMask);

  std::string text;
  llvm::raw_string_ostream output(text);
  output
      << "function automatic [" << width - 1 << ":0] " << name << "(input ["
      << width - 1 << ":0] lhs, input [" << width - 1 << ":0] rhs);\n"
      << "  reg sign_result;\n"
      << "  reg [" << exponentBits - 1 << ":0] exponent_lhs;\n"
      << "  reg [" << exponentBits - 1 << ":0] exponent_rhs;\n"
      << "  reg [" << exponentBits - 1 << ":0] exponent_result;\n"
      << "  reg [" << fractionBits - 1 << ":0] fraction_lhs;\n"
      << "  reg [" << fractionBits - 1 << ":0] fraction_rhs;\n"
      << "  reg [" << fractionBits - 1 << ":0] fraction_result;\n"
      << "  reg [" << precision - 1 << ":0] significand_lhs;\n"
      << "  reg [" << precision - 1 << ":0] significand_rhs;\n"
      << "  reg [" << precision - 1 << ":0] remainder_value;\n"
      << "  reg [" << precision - 1 << ":0] power_value;\n"
      << "  reg [" << precision - 1 << ":0] remainder_accumulator;\n"
      << "  reg [" << precision - 1 << ":0] power_accumulator;\n"
      << "  reg [" << precision << ":0] shifted_remainder;\n"
      << "  reg [" << precision + 1 << ":0] modular_candidate;\n"
      << "  reg [" << precision << ":0] doubled_multiplicand;\n"
      << "  reg [" << exponentBits - 1 << ":0] exponent_delta_bits;\n"
      << "  reg found;\n"
      << "  reg magnitude_is_smaller;\n"
      << "  integer exponent_lhs_value;\n"
      << "  integer exponent_rhs_value;\n"
      << "  integer exponent_delta;\n"
      << "  integer result_exponent_value;\n"
      << "  integer encoded_exponent;\n"
      << "  integer leading_index;\n"
      << "  integer index;\n"
      << "  integer multiply_index;\n"
      << "  begin\n"
      << "    " << name << " = " << hexLiteral(width, quietNaN) << ";\n"
      << "    sign_result = lhs[" << width - 1 << "];\n"
      << "    exponent_lhs = lhs[" << width - 2 << ':' << fractionBits << "];\n"
      << "    exponent_rhs = rhs[" << width - 2 << ':' << fractionBits << "];\n"
      << "    fraction_lhs = lhs[" << fractionBits - 1 << ":0];\n"
      << "    fraction_rhs = rhs[" << fractionBits - 1 << ":0];\n"
      << "    if (exponent_lhs == " << exponentAllOnes
      << " && fraction_lhs != 0) begin\n"
      << "      " << name << " = lhs | " << hexLiteral(width, quietBit) << ";\n"
      << "    end else if (exponent_rhs == " << exponentAllOnes
      << " && fraction_rhs != 0) begin\n"
      << "      " << name << " = rhs | " << hexLiteral(width, quietBit) << ";\n"
      << "    end else if ((exponent_lhs == " << exponentAllOnes
      << " && fraction_lhs == 0) ||\n"
      << "                 (exponent_rhs == 0 && fraction_rhs == 0)) begin\n"
      << "      " << name << " = " << hexLiteral(width, quietNaN) << ";\n"
      << "    end else if ((exponent_lhs == 0 && fraction_lhs == 0) ||\n"
      << "                 (exponent_rhs == " << exponentAllOnes
      << " && fraction_rhs == 0)) begin\n"
      << "      " << name << " = lhs;\n"
      << "    end else begin\n"
      << "      significand_lhs = exponent_lhs == 0"
         " ? {1'b0, fraction_lhs} : {1'b1, fraction_lhs};\n"
      << "      significand_rhs = exponent_rhs == 0"
         " ? {1'b0, fraction_rhs} : {1'b1, fraction_rhs};\n"
      << "      exponent_lhs_value = integer'(exponent_lhs);\n"
      << "      exponent_rhs_value = integer'(exponent_rhs);\n"
      << "      exponent_lhs_value = exponent_lhs == 0 ? "
      << format.minimumExponent() << " : exponent_lhs_value - " << format.bias()
      << ";\n"
      << "      exponent_rhs_value = exponent_rhs == 0 ? "
      << format.minimumExponent() << " : exponent_rhs_value - " << format.bias()
      << ";\n"
      << "      magnitude_is_smaller = exponent_lhs_value < "
         "exponent_rhs_value ||\n"
      << "          (exponent_lhs_value == exponent_rhs_value && "
         "significand_lhs < significand_rhs);\n"
      << "      if (magnitude_is_smaller) begin\n"
      << "        " << name << " = lhs;\n"
      << "      end else begin\n"
      << "        exponent_delta = exponent_lhs_value - exponent_rhs_value;\n"
      << "        exponent_delta_bits = exponent_delta[" << exponentBits - 1
      << ":0];\n"
      << "        power_value = significand_rhs == " << precision << "'d1 ? "
      << precision << "'d0 : " << precision << "'d1;\n"
      << "        for (index = 0; index < " << exponentBits
      << "; index = index + 1) begin\n"
      << "          power_accumulator = " << precision << "'d0;\n"
      << "          for (multiply_index = " << precision - 1
      << "; multiply_index >= 0; multiply_index = multiply_index - 1) begin\n"
      << "            modular_candidate = "
         "{1'b0, power_accumulator, 1'b0};\n"
      << "            if (power_value[multiply_index])\n"
      << "              modular_candidate = modular_candidate + {{2{1'b0}}, "
         "power_value};\n"
      << "            if (modular_candidate >= "
         "{1'b0, significand_rhs, 1'b0})\n"
      << "              modular_candidate = modular_candidate - "
         "{1'b0, significand_rhs, 1'b0};\n"
      << "            else if (modular_candidate >= "
         "{{2{1'b0}}, significand_rhs})\n"
      << "              modular_candidate = modular_candidate - "
         "{{2{1'b0}}, significand_rhs};\n"
      << "            power_accumulator = modular_candidate[" << precision - 1
      << ":0];\n"
      << "          end\n"
      << "          power_value = power_accumulator;\n"
      << "          if (exponent_delta_bits[" << exponentBits - 1
      << " - index]) begin\n"
      << "            doubled_multiplicand = {power_value, 1'b0};\n"
      << "            if (doubled_multiplicand >= "
         "{1'b0, significand_rhs})\n"
      << "              doubled_multiplicand = doubled_multiplicand - "
         "{1'b0, significand_rhs};\n"
      << "            power_value = doubled_multiplicand[" << precision - 1
      << ":0];\n"
      << "          end\n"
      << "        end\n"
      << "        remainder_accumulator = " << precision << "'d0;\n"
      << "        for (multiply_index = " << precision - 1
      << "; multiply_index >= 0; multiply_index = multiply_index - 1) begin\n"
      << "          modular_candidate = "
         "{1'b0, remainder_accumulator, 1'b0};\n"
      << "          if (significand_lhs[multiply_index])\n"
      << "            modular_candidate = modular_candidate + "
         "{{2{1'b0}}, power_value};\n"
      << "          if (modular_candidate >= "
         "{1'b0, significand_rhs, 1'b0})\n"
      << "            modular_candidate = modular_candidate - "
         "{1'b0, significand_rhs, 1'b0};\n"
      << "          else if (modular_candidate >= "
         "{{2{1'b0}}, significand_rhs})\n"
      << "            modular_candidate = modular_candidate - "
         "{{2{1'b0}}, significand_rhs};\n"
      << "          remainder_accumulator = modular_candidate[" << precision - 1
      << ":0];\n"
      << "        end\n"
      << "        remainder_value = remainder_accumulator;\n"
      << "        if (remainder_value == 0) begin\n"
      << "          " << name << " = {sign_result, " << exponentBits << "'d0, "
      << fractionBits << "'d0};\n"
      << "        end else begin\n"
      << "          leading_index = 0;\n"
      << "          found = 1'b0;\n"
      << "          for (index = " << precision - 1
      << "; index >= 0; index = index - 1) begin\n"
      << "            if (!found && remainder_value[index]) begin\n"
      << "              leading_index = index;\n"
      << "              found = 1'b1;\n"
      << "            end\n"
      << "          end\n"
      << "          result_exponent_value = exponent_rhs_value - "
      << fractionBits << " + leading_index;\n"
      << "          if (result_exponent_value >= " << format.minimumExponent()
      << ") begin\n"
      << "            shifted_remainder = {1'b0, remainder_value} << ("
      << fractionBits << " - leading_index);\n"
      << "            encoded_exponent = result_exponent_value + "
      << format.bias() << ";\n"
      << "            exponent_result = encoded_exponent[" << exponentBits - 1
      << ":0];\n"
      << "            fraction_result = shifted_remainder[" << fractionBits - 1
      << ":0];\n"
      << "            " << name
      << " = {sign_result, exponent_result, fraction_result};\n"
      << "          end else begin\n"
      << "            shifted_remainder = {1'b0, remainder_value} << "
         "(exponent_rhs_value - "
      << format.minimumExponent() << ");\n"
      << "            fraction_result = shifted_remainder[" << fractionBits - 1
      << ":0];\n"
      << "            " << name << " = {sign_result, " << exponentBits
      << "'d0, fraction_result};\n"
      << "          end\n"
      << "        end\n"
      << "      end\n"
      << "    end\n"
      << "  end\n"
      << "endfunction\n";
  return output.str();
}

std::string buildModeFunction(FloatFamily family, const LoweredMode &mode) {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic [" << mode.format.width() - 1 << ":0] "
         << mode.functionName << "(input [" << mode.format.width() - 1
         << ":0] lhs, input [" << mode.format.width() - 1 << ":0] rhs);\n"
         << "  begin\n"
         << "    " << mode.functionName << " = "
         << coreName(family, mode.format) << "(lhs, rhs";
  if (family == FloatFamily::Divide)
    output << ", 3'd" << roundingCode(mode.rounding);
  output << ");\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

mlir::Value callMode(mlir::OpBuilder &builder, mlir::Location location,
                     llvm::StringRef functionName, mlir::Value lhs,
                     mlir::Value rhs, unsigned width) {
  return circt::sv::VerbatimExprOp::create(
      builder, location, builder.getIntegerType(width),
      functionName.str() + "({{0}}, {{1}})",
      llvm::SmallVector<mlir::Value, 2>{lhs, rhs});
}

mlir::Value materializeMode(mlir::OpBuilder &builder, mlir::Location location,
                            circt::hw::HWModulePortAccessor &accessor,
                            const LoweredMode &mode, unsigned outputWidth) {
  const unsigned width = mode.format.width();
  mlir::Value lhs = detail::resizeUnsigned(
      builder, location, accessor.getInput("data_input_0"), width);
  mlir::Value rhs = detail::resizeUnsigned(
      builder, location, accessor.getInput("data_input_1"), width);
  mlir::Value result =
      callMode(builder, location, mode.functionName, lhs, rhs, width);
  return detail::resizeUnsigned(builder, location, result, outputWidth);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFloatDivideRemainder(FabricOperationProviderRequest request,
                                        FloatFamily family) {
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
  if (!std::holds_alternative<::fabric::ScalarFloatParams>(
          request.capability.parameterizedCapability))
    return invalid("capability has the wrong scalar floating parameter schema");
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
  const ConfigurationFieldEncoding *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  std::vector<Mode> modes;
  if (request.capability.configurationFieldSchema.empty()) {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None)
      return invalid("configuration-free floating relation is not fieldless");
    if (domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid("configuration-free floating relation is not a singleton");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid(
          "configured floating semantic field relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured floating capability requires one field");
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
    return invalid("sealed floating behavior relation is empty");

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
      return invalid("sealed relation contains a duplicate floating mode");
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
        std::vector<Format> emittedFormats;
        for (const LoweredMode &mode : loweredModes) {
          if (llvm::is_contained(emittedFormats, mode.format))
            continue;
          emittedFormats.push_back(mode.format);
          declarationStream << (family == FloatFamily::Divide
                                    ? buildDivideCoreFunction(mode.format)
                                    : buildRemainderCoreFunction(mode.format))
                            << '\n';
        }
        for (const LoweredMode &mode : loweredModes)
          declarationStream << buildModeFunction(family, mode) << '\n';
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
materializePortableScalarFloatDivide(FabricOperationProviderRequest request) {
  return materializePortableFloatDivideRemainder(std::move(request),
                                                 FloatFamily::Divide);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarFloatRemainder(
    FabricOperationProviderRequest request) {
  return materializePortableFloatDivideRemainder(std::move(request),
                                                 FloatFamily::Remainder);
}

} // namespace

llvm::Error registerPortableFloatDivideRemainderProviders(
    FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry staged = registry;
  if (llvm::Error error =
          staged.add({::fabric::ImplementationFamilyId::ScalarFloatDivide,
                      BackendRecipeKey::PortableSystemVerilog,
                      {},
                      materializePortableScalarFloatDivide}))
    return error;
  if (llvm::Error error =
          staged.add({::fabric::ImplementationFamilyId::ScalarFloatRemainder,
                      BackendRecipeKey::PortableSystemVerilog,
                      {},
                      materializePortableScalarFloatRemainder}))
    return error;
  registry = std::move(staged);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl
