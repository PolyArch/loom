#include "ProviderSupport.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <iomanip>
#include <sstream>

namespace loom::hardware::rtl::detail {

std::optional<PortableFloatFormat> resolvePortableFloatFormat(mlir::Type type) {
  if (mlir::isa<mlir::Float16Type>(type))
    return PortableFloatFormat{5, 10};
  if (mlir::isa<mlir::BFloat16Type>(type))
    return PortableFloatFormat{8, 7};
  if (mlir::isa<mlir::Float32Type>(type))
    return PortableFloatFormat{8, 23};
  if (mlir::isa<mlir::Float64Type>(type))
    return PortableFloatFormat{11, 52};
  return std::nullopt;
}

llvm::APInt decodePhysicalCode(llvm::ArrayRef<std::uint8_t> bytes,
                               std::uint64_t bitCount) {
  llvm::APInt result(static_cast<unsigned>(bitCount), 0);
  for (std::uint64_t bit = 0; bit < bitCount; ++bit)
    if (((bytes[static_cast<std::size_t>(bit / 8)] >> (bit % 8)) & 1U) != 0)
      result.setBit(static_cast<unsigned>(bit));
  return result;
}

const FiniteCodebookEntry *
findFiniteCodebookEntry(const FiniteCodebookEncoding &codebook,
                        llvm::ArrayRef<std::uint8_t> semanticValue) {
  const auto found =
      llvm::find_if(codebook.entries, [&](const FiniteCodebookEntry &entry) {
        return llvm::ArrayRef<std::uint8_t>(entry.semanticValue)
            .equals(semanticValue);
      });
  return found == codebook.entries.end() ? nullptr : &*found;
}

mlir::Value resizeUnsigned(mlir::OpBuilder &builder, mlir::Location location,
                           mlir::Value value, unsigned width) {
  const unsigned current =
      mlir::cast<mlir::IntegerType>(value.getType()).getWidth();
  if (current == width)
    return value;
  if (current > width)
    return circt::comb::ExtractOp::create(builder, location, value, 0, width);
  mlir::Value highZeros = circt::hw::ConstantOp::create(
      builder, location, llvm::APInt(width - current, 0));
  return circt::comb::ConcatOp::create(builder, location,
                                       mlir::ValueRange{highZeros, value});
}

mlir::Value addOrSubtract(mlir::OpBuilder &builder, mlir::Location location,
                          mlir::Value lhs, mlir::Value rhs,
                          mlir::Value subtract) {
  const unsigned width =
      mlir::cast<mlir::IntegerType>(lhs.getType()).getWidth();
  mlir::Value subtractMask =
      width == 1 ? subtract
                 : circt::comb::ReplicateOp::create(builder, location, subtract,
                                                    width);
  mlir::Value adjustedRhs =
      circt::comb::XorOp::create(builder, location, rhs, subtractMask);
  mlir::Value carryIn = resizeUnsigned(builder, location, subtract, width);
  return circt::comb::AddOp::create(
      builder, location, mlir::ValueRange{lhs, adjustedRhs, carryIn}, true);
}

namespace {

std::string hexLiteral(unsigned width, std::uint64_t value) {
  std::ostringstream stream;
  stream << width << "'h" << std::hex << value;
  return stream.str();
}

} // namespace

std::string buildPortableFloatFmaFunction(const PortableFloatFormat &format,
                                          const std::string &functionName) {
  const unsigned width = format.width();
  const unsigned exponentBits = format.exponentBits;
  const unsigned fractionBits = format.fractionBits;
  const unsigned precision = format.precision();
  const unsigned productWidth = 2 * precision;
  const unsigned accumulatorWidth = 2 * precision + 4;
  const unsigned alignmentTop = 2 * precision + 2;
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponentBits) - 1;
  const std::uint64_t infinity = exponentMask << fractionBits;
  const std::uint64_t quietBit = std::uint64_t{1} << (fractionBits - 1);
  const std::uint64_t quietNaN = infinity | quietBit;
  const std::string exponentAllOnes = hexLiteral(exponentBits, exponentMask);
  const std::string shiftName = functionName + "_shr_jam";

  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic [" << accumulatorWidth - 1 << ":0] "
         << shiftName << "(input [" << accumulatorWidth
         << "-1:0] value, input integer distance);\n"
         << "  integer index;\n"
         << "  reg sticky;\n"
         << "  begin\n"
         << "    " << shiftName << " = " << accumulatorWidth << "'d0;\n"
         << "    sticky = 1'b0;\n"
         << "    if (distance <= 0) begin\n"
         << "      " << shiftName << " = value;\n"
         << "    end else if (distance >= " << accumulatorWidth << ") begin\n"
         << "      " << shiftName << "[0] = |value;\n"
         << "    end else begin\n"
         << "      " << shiftName << " = value >> distance;\n"
         << "      for (index = 0; index < " << accumulatorWidth
         << "; index = index + 1) begin\n"
         << "        if (index < distance) sticky = sticky | value[index];\n"
         << "      end\n"
         << "      " << shiftName << "[0] = " << shiftName << "[0] | sticky;\n"
         << "    end\n"
         << "  end\n"
         << "endfunction\n\n";

  output
      << "function automatic [" << width - 1 << ":0] " << functionName
      << "(input [" << width - 1 << ":0] lhs, input [" << width - 1
      << ":0] rhs, input [" << width - 1 << ":0] addend);\n"
      << "  reg sign_lhs;\n"
      << "  reg sign_rhs;\n"
      << "  reg sign_addend;\n"
      << "  reg sign_product;\n"
      << "  reg sign_result;\n"
      << "  reg [" << exponentBits - 1 << ":0] exponent_lhs;\n"
      << "  reg [" << exponentBits - 1 << ":0] exponent_rhs;\n"
      << "  reg [" << exponentBits - 1 << ":0] exponent_addend;\n"
      << "  reg [" << exponentBits - 1 << ":0] exponent_result;\n"
      << "  reg [" << fractionBits - 1 << ":0] fraction_lhs;\n"
      << "  reg [" << fractionBits - 1 << ":0] fraction_rhs;\n"
      << "  reg [" << fractionBits - 1 << ":0] fraction_addend;\n"
      << "  reg [" << fractionBits - 1 << ":0] fraction_result;\n"
      << "  reg [" << precision - 1 << ":0] significand_lhs;\n"
      << "  reg [" << precision - 1 << ":0] significand_rhs;\n"
      << "  reg [" << precision - 1 << ":0] significand_addend;\n"
      << "  reg [" << productWidth - 1 << ":0] product;\n"
      << "  reg [" << accumulatorWidth - 1 << ":0] product_value;\n"
      << "  reg [" << accumulatorWidth - 1 << ":0] addend_value;\n"
      << "  reg [" << accumulatorWidth - 1 << ":0] aligned_product;\n"
      << "  reg [" << accumulatorWidth - 1 << ":0] aligned_addend;\n"
      << "  reg [" << accumulatorWidth - 1 << ":0] magnitude;\n"
      << "  reg [" << accumulatorWidth - 1 << ":0] shifted_magnitude;\n"
      << "  reg [" << precision << ":0] rounded;\n"
      << "  reg found;\n"
      << "  reg guard;\n"
      << "  reg sticky;\n"
      << "  reg increment;\n"
      << "  integer exponent_lhs_value;\n"
      << "  integer exponent_rhs_value;\n"
      << "  integer exponent_addend_value;\n"
      << "  integer exponent_product_value;\n"
      << "  integer common_exponent;\n"
      << "  integer result_exponent_value;\n"
      << "  integer encoded_exponent;\n"
      << "  integer shift_amount;\n"
      << "  integer leading_index;\n"
      << "  integer index;\n"
      << "  begin\n"
      << "    " << functionName << " = " << hexLiteral(width, quietNaN) << ";\n"
      << "    sign_lhs = lhs[" << width - 1 << "];\n"
      << "    sign_rhs = rhs[" << width - 1 << "];\n"
      << "    sign_addend = addend[" << width - 1 << "];\n"
      << "    sign_product = sign_lhs ^ sign_rhs;\n"
      << "    exponent_lhs = lhs[" << width - 2 << ':' << fractionBits << "];\n"
      << "    exponent_rhs = rhs[" << width - 2 << ':' << fractionBits << "];\n"
      << "    exponent_addend = addend[" << width - 2 << ':' << fractionBits
      << "];\n"
      << "    fraction_lhs = lhs[" << fractionBits - 1 << ":0];\n"
      << "    fraction_rhs = rhs[" << fractionBits - 1 << ":0];\n"
      << "    fraction_addend = addend[" << fractionBits - 1 << ":0];\n"
      << "    if (exponent_lhs == " << exponentAllOnes
      << " && fraction_lhs != 0) begin\n"
      << "      " << functionName << " = lhs | " << hexLiteral(width, quietBit)
      << ";\n"
      << "    end else if (exponent_rhs == " << exponentAllOnes
      << " && fraction_rhs != 0) begin\n"
      << "      " << functionName << " = rhs | " << hexLiteral(width, quietBit)
      << ";\n"
      << "    end else if (((exponent_lhs == " << exponentAllOnes
      << " && fraction_lhs == 0) &&\n"
      << "                 (exponent_rhs == 0 && fraction_rhs == 0)) ||\n"
      << "                ((exponent_rhs == " << exponentAllOnes
      << " && fraction_rhs == 0) &&\n"
      << "                 (exponent_lhs == 0 && fraction_lhs == 0))) begin\n"
      << "      " << functionName << " = " << hexLiteral(width, quietNaN)
      << ";\n"
      << "    end else if (exponent_addend == " << exponentAllOnes
      << " && fraction_addend != 0) begin\n"
      << "      " << functionName << " = addend | "
      << hexLiteral(width, quietBit) << ";\n"
      << "    end else if ((exponent_lhs == " << exponentAllOnes
      << " && fraction_lhs == 0) ||\n"
      << "                 (exponent_rhs == " << exponentAllOnes
      << " && fraction_rhs == 0)) begin\n"
      << "      if (exponent_addend == " << exponentAllOnes
      << " && fraction_addend == 0 && sign_addend != sign_product)\n"
      << "        " << functionName << " = " << hexLiteral(width, quietNaN)
      << ";\n"
      << "      else\n"
      << "        " << functionName << " = {sign_product, " << exponentAllOnes
      << ", " << fractionBits << "'d0};\n"
      << "    end else if (exponent_addend == " << exponentAllOnes
      << " && fraction_addend == 0) begin\n"
      << "      " << functionName << " = addend;\n"
      << "    end else begin\n"
      << "      significand_lhs = exponent_lhs == 0"
      << " ? {1'b0, fraction_lhs} : {1'b1, fraction_lhs};\n"
      << "      significand_rhs = exponent_rhs == 0"
      << " ? {1'b0, fraction_rhs} : {1'b1, fraction_rhs};\n"
      << "      significand_addend = exponent_addend == 0"
      << " ? {1'b0, fraction_addend} : {1'b1, fraction_addend};\n"
      << "      exponent_lhs_value = integer'(exponent_lhs);\n"
      << "      exponent_rhs_value = integer'(exponent_rhs);\n"
      << "      exponent_addend_value = integer'(exponent_addend);\n"
      << "      exponent_lhs_value = exponent_lhs == 0 ? "
      << format.minimumExponent() << " : exponent_lhs_value - " << format.bias()
      << ";\n"
      << "      exponent_rhs_value = exponent_rhs == 0 ? "
      << format.minimumExponent() << " : exponent_rhs_value - " << format.bias()
      << ";\n"
      << "      exponent_addend_value = exponent_addend == 0 ? "
      << format.minimumExponent() << " : exponent_addend_value - "
      << format.bias() << ";\n"
      << "      for (index = 0; index < " << precision
      << "; index = index + 1) begin\n"
      << "        if (significand_lhs != 0 && !significand_lhs["
      << precision - 1
      << "]) begin significand_lhs = significand_lhs << 1; "
         "exponent_lhs_value = exponent_lhs_value - 1; end\n"
      << "        if (significand_rhs != 0 && !significand_rhs["
      << precision - 1
      << "]) begin significand_rhs = significand_rhs << 1; "
         "exponent_rhs_value = exponent_rhs_value - 1; end\n"
      << "        if (significand_addend != 0 && !significand_addend["
      << precision - 1
      << "]) begin significand_addend = significand_addend << 1; "
         "exponent_addend_value = exponent_addend_value - 1; end\n"
      << "      end\n"
      << "      if (significand_lhs == 0 || significand_rhs == 0) begin\n"
      << "        if (significand_addend != 0)\n"
      << "          " << functionName << " = addend;\n"
      << "        else begin\n"
      << "          sign_result = sign_product == sign_addend"
         " ? sign_product : 1'b0;\n"
      << "          " << functionName << " = {sign_result, " << exponentBits
      << "'d0, " << fractionBits << "'d0};\n"
      << "        end\n"
      << "      end else begin\n"
      << "        product = significand_lhs * significand_rhs;\n"
      << "        if (product[" << productWidth - 1 << "]) begin\n"
      << "          exponent_product_value = exponent_lhs_value + "
         "exponent_rhs_value + 1;\n"
      << "        end else begin\n"
      << "          product = product << 1;\n"
      << "          exponent_product_value = exponent_lhs_value + "
         "exponent_rhs_value;\n"
      << "        end\n"
      << "        product_value = " << accumulatorWidth << "'d0;\n"
      << "        product_value[" << productWidth - 1 << ":0] = product;\n"
      << "        product_value = product_value << 3;\n"
      << "        addend_value = " << accumulatorWidth << "'d0;\n"
      << "        addend_value[" << precision - 1
      << ":0] = significand_addend;\n"
      << "        addend_value = addend_value << " << precision + 3 << ";\n"
      << "        if (significand_addend == 0 || "
         "exponent_product_value >= exponent_addend_value) begin\n"
      << "          common_exponent = exponent_product_value;\n"
      << "          aligned_product = product_value;\n"
      << "          aligned_addend = " << shiftName
      << "(addend_value, exponent_product_value - "
         "exponent_addend_value);\n"
      << "        end else begin\n"
      << "          common_exponent = exponent_addend_value;\n"
      << "          aligned_product = " << shiftName
      << "(product_value, exponent_addend_value - "
         "exponent_product_value);\n"
      << "          aligned_addend = addend_value;\n"
      << "        end\n"
      << "        if (sign_product == sign_addend) begin\n"
      << "          magnitude = aligned_product + aligned_addend;\n"
      << "          sign_result = sign_product;\n"
      << "        end else if (aligned_product > aligned_addend) begin\n"
      << "          magnitude = aligned_product - aligned_addend;\n"
      << "          sign_result = sign_product;\n"
      << "        end else if (aligned_addend > aligned_product) begin\n"
      << "          magnitude = aligned_addend - aligned_product;\n"
      << "          sign_result = sign_addend;\n"
      << "        end else begin\n"
      << "          magnitude = " << accumulatorWidth << "'d0;\n"
      << "          sign_result = 1'b0;\n"
      << "        end\n"
      << "        if (magnitude == 0) begin\n"
      << "          " << functionName << " = {sign_result, " << exponentBits
      << "'d0, " << fractionBits << "'d0};\n"
      << "        end else begin\n"
      << "          leading_index = 0;\n"
      << "          found = 1'b0;\n"
      << "          for (index = " << accumulatorWidth - 1
      << "; index >= 0; index = index - 1) begin\n"
      << "            if (!found && magnitude[index]) begin\n"
      << "              leading_index = index;\n"
      << "              found = 1'b1;\n"
      << "            end\n"
      << "          end\n"
      << "          result_exponent_value = common_exponent - " << alignmentTop
      << " + leading_index;\n"
      << "          if (result_exponent_value >= " << format.minimumExponent()
      << ")\n"
      << "            shift_amount = leading_index - " << fractionBits << ";\n"
      << "          else\n"
      << "            shift_amount = " << alignmentTop
      << " - common_exponent + " << format.minimumExponent() << " - "
      << fractionBits << ";\n"
      << "          rounded = " << precision + 1 << "'d0;\n"
      << "          guard = 1'b0;\n"
      << "          sticky = 1'b0;\n"
      << "          increment = 1'b0;\n"
      << "          shifted_magnitude = " << accumulatorWidth << "'d0;\n"
      << "          if (shift_amount <= 0) begin\n"
      << "            shifted_magnitude = magnitude << (-shift_amount);\n"
      << "            rounded = shifted_magnitude[" << precision << ":0];\n"
      << "          end else begin\n"
      << "            shifted_magnitude = magnitude >> shift_amount;\n"
      << "            rounded = shifted_magnitude[" << precision << ":0];\n"
      << "            for (index = 0; index < " << accumulatorWidth
      << "; index = index + 1) begin\n"
      << "              if (index == shift_amount - 1) "
         "guard = magnitude[index];\n"
      << "              if (index < shift_amount - 1) "
         "sticky = sticky | magnitude[index];\n"
      << "            end\n"
      << "            if (shift_amount > " << accumulatorWidth
      << ") sticky = |magnitude;\n"
      << "            increment = guard && (sticky || rounded[0]);\n"
      << "            rounded = rounded + increment;\n"
      << "          end\n"
      << "          if (result_exponent_value >= " << format.minimumExponent()
      << ") begin\n"
      << "            if (rounded[" << precision << "]) begin\n"
      << "              rounded = rounded >> 1;\n"
      << "              result_exponent_value = "
         "result_exponent_value + 1;\n"
      << "            end\n"
      << "            if (result_exponent_value > " << format.maximumExponent()
      << ") begin\n"
      << "              " << functionName << " = {sign_result, "
      << exponentAllOnes << ", " << fractionBits << "'d0};\n"
      << "            end else begin\n"
      << "              encoded_exponent = result_exponent_value + "
      << format.bias() << ";\n"
      << "              exponent_result = encoded_exponent[" << exponentBits - 1
      << ":0];\n"
      << "              fraction_result = rounded[" << fractionBits - 1
      << ":0];\n"
      << "              " << functionName
      << " = {sign_result, exponent_result, fraction_result};\n"
      << "            end\n"
      << "          end else if (rounded[" << fractionBits << "]) begin\n"
      << "            " << functionName << " = {sign_result, " << exponentBits
      << "'d1, " << fractionBits << "'d0};\n"
      << "          end else begin\n"
      << "            fraction_result = rounded[" << fractionBits - 1
      << ":0];\n"
      << "            " << functionName << " = {sign_result, " << exponentBits
      << "'d0, fraction_result};\n"
      << "          end\n"
      << "        end\n"
      << "      end\n"
      << "    end\n"
      << "  end\n"
      << "endfunction\n";
  return output.str();
}

} // namespace loom::hardware::rtl::detail
