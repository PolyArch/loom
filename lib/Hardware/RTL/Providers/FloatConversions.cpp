#include "Hardware/RTL/Providers/FloatConversions.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

using Family = ::fabric::ImplementationFamilyId;
using Format = detail::PortableFloatFormat;
using Schema = ::dataflow::OperationSchemaId;

enum class ConversionKind { FloatWidthCast, IntegerToFloat, FloatToInteger };

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  ConversionKind kind;
  Format sourceFormat{5, 10};
  Format destinationFormat{5, 10};
  unsigned integerWidth = 0;
  bool signedInteger = false;
  mlir::arith::RoundingMode rounding =
      mlir::arith::RoundingMode::to_nearest_even;
  std::string functionName;

  unsigned inputWidth() const {
    return kind == ConversionKind::IntegerToFloat ? integerWidth
                                                  : sourceFormat.width();
  }
  unsigned outputWidth() const {
    return kind == ConversionKind::FloatToInteger ? integerWidth
                                                  : destinationFormat.width();
  }
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_float_conversions_invalid: " +
                                     message);
}

llvm::Error unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

bool isConversionFamily(Family family) {
  return family == Family::ScalarFloatWidthCast ||
         family == Family::ScalarIntegerToFloat ||
         family == Family::ScalarFloatToInteger;
}

llvm::StringRef roundingSuffix(mlir::arith::RoundingMode rounding) {
  using Rounding = mlir::arith::RoundingMode;
  switch (rounding) {
  case Rounding::to_nearest_even:
    return "rne";
  case Rounding::downward:
    return "rdn";
  case Rounding::upward:
    return "rup";
  case Rounding::toward_zero:
    return "rtz";
  case Rounding::to_nearest_away:
    return "rna";
  }
  llvm_unreachable("unknown floating rounding mode");
}

std::string formatSuffix(const Format &format) {
  return "e" + std::to_string(format.exponentBits) + "_f" +
         std::to_string(format.fractionBits);
}

std::string widthCastName(const Format &source, const Format &destination,
                          mlir::arith::RoundingMode rounding) {
  return "loom_float_width_cast_" + formatSuffix(source) + "_to_" +
         formatSuffix(destination) + "_" + roundingSuffix(rounding).str();
}

std::string integerToFloatName(unsigned width, const Format &destination,
                               bool isSigned) {
  return "loom_" + std::string(isSigned ? "signed" : "unsigned") + "_i" +
         std::to_string(width) + "_to_float_" + formatSuffix(destination);
}

std::string floatToIntegerName(const Format &source, unsigned width,
                               bool isSigned) {
  return "loom_float_" + formatSuffix(source) + "_to_" +
         (isSigned ? "signed" : "unsigned") + "_i" + std::to_string(width);
}

llvm::Expected<LoweredMode>
lowerMode(Family family,
          const ::dataflow::CanonicalActorSchemaProjection &actor) {
  if (actor.type.getNumInputs() != 1 || actor.type.getNumResults() != 1)
    return invalid("behavior is not a unary conversion");
  if (mlir::isa<mlir::VectorType>(actor.type.getInput(0)) ||
      mlir::isa<mlir::VectorType>(actor.type.getResult(0)))
    return invalid("behavior is not scalar");

  if (family == Family::ScalarFloatWidthCast) {
    if (actor.schema != Schema::ArithExtF &&
        actor.schema != Schema::ArithTruncF)
      return invalid("width-cast behavior has a foreign schema");
    auto source = detail::resolvePortableFloatFormat(actor.type.getInput(0));
    auto destination =
        detail::resolvePortableFloatFormat(actor.type.getResult(0));
    if (!source || !destination)
      return invalid("width cast uses an unsupported floating format");
    const bool extension = actor.schema == Schema::ArithExtF;
    if ((extension && source->width() >= destination->width()) ||
        (!extension && source->width() <= destination->width()))
      return invalid("width cast has an invalid endpoint direction");
    const auto *payload =
        std::get_if<::dataflow::FloatingPointPayload>(&actor.payload);
    if (!payload)
      return invalid("width cast has no floating-point payload");
    const mlir::arith::RoundingMode rounding =
        extension ? mlir::arith::RoundingMode::to_nearest_even
                  : payload->roundingMode.value_or(
                        mlir::arith::RoundingMode::to_nearest_even);
    return LoweredMode{ConversionKind::FloatWidthCast,
                       *source,
                       *destination,
                       0,
                       false,
                       rounding,
                       widthCastName(*source, *destination, rounding)};
  }

  if (family == Family::ScalarIntegerToFloat) {
    if (actor.schema != Schema::ArithSIToFP &&
        actor.schema != Schema::ArithUIToFP)
      return invalid("integer-to-float behavior has a foreign schema");
    auto integer = mlir::dyn_cast<mlir::IntegerType>(actor.type.getInput(0));
    auto destination =
        detail::resolvePortableFloatFormat(actor.type.getResult(0));
    if (!integer || !destination)
      return invalid("integer-to-float behavior has unsupported endpoints");
    if (integer.getWidth() > 64)
      return invalid("integer-to-float behavior exceeds the portable emitter's "
                     "64-bit datapath");
    const bool isSigned = actor.schema == Schema::ArithSIToFP;
    if ((isSigned &&
         !std::holds_alternative<::dataflow::NoPayload>(actor.payload)) ||
        (!isSigned && !std::holds_alternative<::dataflow::NonNegativePayload>(
                          actor.payload)))
      return invalid("integer-to-float behavior has the wrong payload");
    return LoweredMode{
        ConversionKind::IntegerToFloat,
        Format{5, 10},
        *destination,
        integer.getWidth(),
        isSigned,
        mlir::arith::RoundingMode::to_nearest_even,
        integerToFloatName(integer.getWidth(), *destination, isSigned)};
  }

  if (family != Family::ScalarFloatToInteger)
    return invalid("provider received an unknown conversion family");
  const bool isSigned = actor.schema == Schema::ArithFPToSI ||
                        actor.schema == Schema::LLVMFPToSISat;
  const bool isUnsigned = actor.schema == Schema::ArithFPToUI ||
                          actor.schema == Schema::LLVMFPToUISat;
  if (!isSigned && !isUnsigned)
    return invalid("float-to-integer behavior has a foreign schema");
  auto source = detail::resolvePortableFloatFormat(actor.type.getInput(0));
  auto integer = mlir::dyn_cast<mlir::IntegerType>(actor.type.getResult(0));
  if (!source || !integer)
    return invalid("float-to-integer behavior has unsupported endpoints");
  if (integer.getWidth() > 64)
    return invalid("float-to-integer behavior exceeds the portable emitter's "
                   "64-bit datapath");
  if (!std::holds_alternative<::dataflow::NoPayload>(actor.payload))
    return invalid("float-to-integer behavior has the wrong payload");
  return LoweredMode{ConversionKind::FloatToInteger,
                     *source,
                     Format{5, 10},
                     integer.getWidth(),
                     isSigned,
                     mlir::arith::RoundingMode::toward_zero,
                     floatToIntegerName(*source, integer.getWidth(), isSigned)};
}

std::string unsignedLiteral(const llvm::APInt &value) {
  llvm::SmallString<64> digits;
  value.toStringUnsigned(digits, 16);
  return std::to_string(value.getBitWidth()) + "'h" + digits.str().str();
}

std::string zeroExtend(llvm::StringRef value, unsigned sourceWidth,
                       unsigned destinationWidth) {
  std::string text;
  llvm::raw_string_ostream output(text);
  if (sourceWidth == destinationWidth)
    output << value;
  else
    output << "{{" << destinationWidth - sourceWidth << "{1'b0}}, " << value
           << "}";
  return output.str();
}

llvm::StringRef overflowToInfinity(mlir::arith::RoundingMode rounding) {
  using Rounding = mlir::arith::RoundingMode;
  switch (rounding) {
  case Rounding::to_nearest_even:
  case Rounding::to_nearest_away:
    return "1'b1";
  case Rounding::downward:
    return "sign_value";
  case Rounding::upward:
    return "!sign_value";
  case Rounding::toward_zero:
    return "1'b0";
  }
  llvm_unreachable("unknown floating rounding mode");
}

std::string roundingExpression(mlir::arith::RoundingMode rounding) {
  using Rounding = mlir::arith::RoundingMode;
  switch (rounding) {
  case Rounding::to_nearest_even:
    return "guard_bit && (sticky_bit || retained[0])";
  case Rounding::to_nearest_away:
    return "guard_bit";
  case Rounding::downward:
    return "sign_value && discarded_bits";
  case Rounding::upward:
    return "!sign_value && discarded_bits";
  case Rounding::toward_zero:
    return "1'b0";
  }
  llvm_unreachable("unknown floating rounding mode");
}

void emitDiscardedBitAnalysis(llvm::raw_ostream &output,
                              llvm::StringRef indent) {
  output << indent << "retained = 64'd0;\n"
         << indent << "discarded_bits = 1'b0;\n"
         << indent << "guard_bit = 1'b0;\n"
         << indent << "sticky_bit = 1'b0;\n"
         << indent << "if (shift_amount < 64)\n"
         << indent << "  retained = significand >> shift_amount;\n"
         << indent << "for (bit_index = 0; bit_index < 64; "
         << "bit_index = bit_index + 1) begin\n"
         << indent << "  if (bit_index < shift_amount && "
         << "significand[bit_index]) discarded_bits = 1'b1;\n"
         << indent << "  if (bit_index + 1 < shift_amount && "
         << "significand[bit_index]) sticky_bit = 1'b1;\n"
         << indent << "end\n"
         << indent << "if (shift_amount > 0 && shift_amount <= 64)\n"
         << indent << "  guard_bit = significand[shift_amount - 1];\n";
}

std::string buildFloatWidthCastFunction(const LoweredMode &mode) {
  const Format &source = mode.sourceFormat;
  const Format &destination = mode.destinationFormat;
  const unsigned sourceWidth = source.width();
  const unsigned destinationWidth = destination.width();
  const std::uint64_t sourceExponentMask =
      (std::uint64_t{1} << source.exponentBits) - 1;
  const std::uint64_t destinationExponentMask =
      (std::uint64_t{1} << destination.exponentBits) - 1;

  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic [" << destinationWidth - 1 << ":0] "
         << mode.functionName << "(input [" << sourceWidth - 1
         << ":0] value);\n"
         << "  reg sign_value;\n"
         << "  reg [" << source.exponentBits - 1 << ":0] source_exponent;\n"
         << "  reg [" << source.fractionBits - 1 << ":0] source_fraction;\n"
         << "  reg [" << destination.exponentBits - 1
         << ":0] destination_exponent;\n"
         << "  reg [" << destination.fractionBits - 1
         << ":0] destination_fraction;\n"
         << "  reg [63:0] significand;\n"
         << "  reg [63:0] retained;\n"
         << "  reg [31:0] biased_exponent;\n"
         << "  reg discarded_bits;\n"
         << "  reg guard_bit;\n"
         << "  reg sticky_bit;\n"
         << "  reg increment;\n"
         << "  integer source_msb;\n"
         << "  integer unbiased_exponent;\n"
         << "  integer shift_amount;\n"
         << "  integer bit_index;\n"
         << "  begin\n"
         << "    sign_value = value[" << sourceWidth - 1 << "];\n"
         << "    source_exponent = value[" << sourceWidth - 2 << ":"
         << source.fractionBits << "];\n"
         << "    source_fraction = value[" << source.fractionBits - 1
         << ":0];\n"
         << "    destination_exponent = " << destination.exponentBits
         << "'d0;\n"
         << "    destination_fraction = " << destination.fractionBits
         << "'d0;\n"
         << "    significand = 64'd0;\n"
         << "    retained = 64'd0;\n"
         << "    biased_exponent = 32'd0;\n"
         << "    discarded_bits = 1'b0;\n"
         << "    guard_bit = 1'b0;\n"
         << "    sticky_bit = 1'b0;\n"
         << "    increment = 1'b0;\n"
         << "    source_msb = 0;\n"
         << "    unbiased_exponent = 0;\n"
         << "    shift_amount = 0;\n"
         << "    if (source_exponent == " << source.exponentBits << "'h"
         << llvm::utohexstr(sourceExponentMask) << ") begin\n"
         << "      destination_exponent = {" << destination.exponentBits
         << "{1'b1}};\n"
         << "      if (source_fraction != " << source.fractionBits
         << "'d0) begin\n"
         << "        significand = "
         << zeroExtend("source_fraction", source.fractionBits, 64) << ";\n";
  if (destination.fractionBits >= source.fractionBits)
    output << "        significand = significand << "
           << destination.fractionBits - source.fractionBits << ";\n";
  else
    output << "        significand = significand >> "
           << source.fractionBits - destination.fractionBits << ";\n";
  output << "        destination_fraction = significand["
         << destination.fractionBits - 1 << ":0];\n"
         << "        if (!source_fraction[" << source.fractionBits - 1
         << "]) destination_fraction[" << destination.fractionBits - 1
         << "] = 1'b1;\n"
         << "      end\n"
         << "    end else if (source_exponent == " << source.exponentBits
         << "'d0 && source_fraction == " << source.fractionBits
         << "'d0) begin\n"
         << "      destination_exponent = " << destination.exponentBits
         << "'d0;\n"
         << "      destination_fraction = " << destination.fractionBits
         << "'d0;\n"
         << "    end else begin\n"
         << "      if (source_exponent == " << source.exponentBits
         << "'d0) begin\n"
         << "        significand = "
         << zeroExtend("source_fraction", source.fractionBits, 64) << ";\n"
         << "        source_msb = 0;\n"
         << "        for (bit_index = 0; bit_index < " << source.fractionBits
         << "; bit_index = bit_index + 1)\n"
         << "          if (significand[bit_index]) source_msb = bit_index;\n"
         << "        unbiased_exponent = " << source.minimumExponent() << " - "
         << source.fractionBits << " + source_msb;\n"
         << "      end else begin\n"
         << "        significand = "
         << zeroExtend("source_fraction", source.fractionBits, 64) << ";\n"
         << "        significand[" << source.fractionBits << "] = 1'b1;\n"
         << "        source_msb = " << source.fractionBits << ";\n"
         << "        unbiased_exponent = "
         << zeroExtend("source_exponent", source.exponentBits, 32) << " - "
         << source.bias() << ";\n"
         << "      end\n"
         << "      if (unbiased_exponent >= " << destination.minimumExponent()
         << ") begin\n"
         << "        shift_amount = source_msb - " << destination.fractionBits
         << ";\n"
         << "        if (shift_amount > 0) begin\n";
  emitDiscardedBitAnalysis(output, "          ");
  output << "          increment = " << roundingExpression(mode.rounding)
         << ";\n"
         << "        end else begin\n"
         << "          retained = significand << (-shift_amount);\n"
         << "          increment = 1'b0;\n"
         << "        end\n"
         << "        if (increment) retained = retained + 1'b1;\n"
         << "        if (retained[" << destination.fractionBits + 1
         << "]) begin\n"
         << "          retained = retained >> 1;\n"
         << "          unbiased_exponent = unbiased_exponent + 1;\n"
         << "        end\n"
         << "        if (unbiased_exponent > " << destination.maximumExponent()
         << ") begin\n"
         << "          if (" << overflowToInfinity(mode.rounding) << ") begin\n"
         << "            destination_exponent = {" << destination.exponentBits
         << "{1'b1}};\n"
         << "            destination_fraction = " << destination.fractionBits
         << "'d0;\n"
         << "          end else begin\n"
         << "            destination_exponent = " << destination.exponentBits
         << "'h" << llvm::utohexstr(destinationExponentMask - 1) << ";\n"
         << "            destination_fraction = {" << destination.fractionBits
         << "{1'b1}};\n"
         << "          end\n"
         << "        end else begin\n"
         << "          biased_exponent = unbiased_exponent + "
         << destination.bias() << ";\n"
         << "          destination_exponent = biased_exponent["
         << destination.exponentBits - 1 << ":0];\n"
         << "          destination_fraction = retained["
         << destination.fractionBits - 1 << ":0];\n"
         << "        end\n"
         << "      end else begin\n"
         << "        shift_amount = source_msb + "
         << destination.minimumExponent() << " - " << destination.fractionBits
         << " - unbiased_exponent;\n"
         << "        if (shift_amount > 0) begin\n";
  emitDiscardedBitAnalysis(output, "          ");
  output << "          increment = " << roundingExpression(mode.rounding)
         << ";\n"
         << "        end else begin\n"
         << "          retained = significand << (-shift_amount);\n"
         << "          increment = 1'b0;\n"
         << "        end\n"
         << "        if (increment) retained = retained + 1'b1;\n"
         << "        if (retained[" << destination.fractionBits << "]) begin\n"
         << "          destination_exponent = " << destination.exponentBits
         << "'d1;\n"
         << "          destination_fraction = " << destination.fractionBits
         << "'d0;\n"
         << "        end else begin\n"
         << "          destination_exponent = " << destination.exponentBits
         << "'d0;\n"
         << "          destination_fraction = retained["
         << destination.fractionBits - 1 << ":0];\n"
         << "        end\n"
         << "      end\n"
         << "    end\n"
         << "    " << mode.functionName
         << " = {sign_value, destination_exponent, destination_fraction};\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

std::string buildIntegerToFloatFunction(const LoweredMode &mode) {
  const unsigned sourceWidth = mode.integerWidth;
  const Format &destination = mode.destinationFormat;
  const unsigned destinationWidth = destination.width();

  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic [" << destinationWidth - 1 << ":0] "
         << mode.functionName << "(input [" << sourceWidth - 1
         << ":0] value);\n"
         << "  reg sign_value;\n"
         << "  reg [" << sourceWidth - 1 << ":0] source_magnitude;\n"
         << "  reg [63:0] magnitude;\n"
         << "  reg [63:0] significand;\n"
         << "  reg [63:0] retained;\n"
         << "  reg [31:0] biased_exponent;\n"
         << "  reg discarded_bits;\n"
         << "  reg guard_bit;\n"
         << "  reg sticky_bit;\n"
         << "  reg increment;\n"
         << "  reg [" << destination.exponentBits - 1
         << ":0] destination_exponent;\n"
         << "  reg [" << destination.fractionBits - 1
         << ":0] destination_fraction;\n"
         << "  integer highest_bit;\n"
         << "  integer unbiased_exponent;\n"
         << "  integer shift_amount;\n"
         << "  integer bit_index;\n"
         << "  begin\n"
         << "    sign_value = 1'b0;\n"
         << "    source_magnitude = value;\n";
  if (mode.signedInteger)
    output << "    if (value[" << sourceWidth - 1 << "]) begin\n"
           << "      sign_value = 1'b1;\n"
           << "      source_magnitude = (~value) + " << sourceWidth << "'d1;\n"
           << "    end\n";
  output << "    magnitude = "
         << zeroExtend("source_magnitude", sourceWidth, 64) << ";\n"
         << "    retained = 64'd0;\n"
         << "    significand = 64'd0;\n"
         << "    biased_exponent = 32'd0;\n"
         << "    discarded_bits = 1'b0;\n"
         << "    guard_bit = 1'b0;\n"
         << "    sticky_bit = 1'b0;\n"
         << "    increment = 1'b0;\n"
         << "    destination_exponent = " << destination.exponentBits
         << "'d0;\n"
         << "    destination_fraction = " << destination.fractionBits
         << "'d0;\n"
         << "    highest_bit = 0;\n"
         << "    unbiased_exponent = 0;\n"
         << "    shift_amount = 0;\n"
         << "    if (magnitude != 64'd0) begin\n"
         << "      for (bit_index = 0; bit_index < " << sourceWidth
         << "; bit_index = bit_index + 1)\n"
         << "        if (magnitude[bit_index]) highest_bit = bit_index;\n"
         << "      unbiased_exponent = highest_bit;\n"
         << "      shift_amount = highest_bit - " << destination.fractionBits
         << ";\n"
         << "      if (shift_amount > 0) begin\n"
         << "        significand = magnitude;\n";
  emitDiscardedBitAnalysis(output, "        ");
  output << "        increment = guard_bit && (sticky_bit || retained[0]);\n"
         << "      end else begin\n"
         << "        retained = magnitude << (-shift_amount);\n"
         << "      end\n"
         << "      if (increment) retained = retained + 1'b1;\n"
         << "      if (retained[" << destination.fractionBits + 1
         << "]) begin\n"
         << "        retained = retained >> 1;\n"
         << "        unbiased_exponent = unbiased_exponent + 1;\n"
         << "      end\n"
         << "      if (unbiased_exponent > " << destination.maximumExponent()
         << ") begin\n"
         << "        destination_exponent = {" << destination.exponentBits
         << "{1'b1}};\n"
         << "        destination_fraction = " << destination.fractionBits
         << "'d0;\n"
         << "      end else begin\n"
         << "        biased_exponent = unbiased_exponent + "
         << destination.bias() << ";\n"
         << "        destination_exponent = biased_exponent["
         << destination.exponentBits - 1 << ":0];\n"
         << "        destination_fraction = retained["
         << destination.fractionBits - 1 << ":0];\n"
         << "      end\n"
         << "    end\n"
         << "    " << mode.functionName
         << " = {sign_value, destination_exponent, destination_fraction};\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

std::string buildFloatToIntegerFunction(const LoweredMode &mode) {
  const Format &source = mode.sourceFormat;
  const unsigned sourceWidth = source.width();
  const unsigned destinationWidth = mode.integerWidth;
  const llvm::APInt zero(destinationWidth, 0);
  const llvm::APInt maximum =
      mode.signedInteger ? llvm::APInt::getSignedMaxValue(destinationWidth)
                         : llvm::APInt::getAllOnes(destinationWidth);
  const llvm::APInt minimum =
      mode.signedInteger ? llvm::APInt::getSignedMinValue(destinationWidth)
                         : zero;
  const unsigned positiveLimitExponent =
      destinationWidth - (mode.signedInteger ? 1U : 0U);

  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic [" << destinationWidth - 1 << ":0] "
         << mode.functionName << "(input [" << sourceWidth - 1
         << ":0] value);\n"
         << "  reg sign_value;\n"
         << "  reg [" << source.exponentBits - 1 << ":0] exponent_value;\n"
         << "  reg [" << source.fractionBits - 1 << ":0] fraction_value;\n"
         << "  reg [63:0] significand;\n"
         << "  reg [63:0] magnitude;\n"
         << "  integer unbiased_exponent;\n"
         << "  integer shift_amount;\n"
         << "  begin\n"
         << "    sign_value = value[" << sourceWidth - 1 << "];\n"
         << "    exponent_value = value[" << sourceWidth - 2 << ":"
         << source.fractionBits << "];\n"
         << "    fraction_value = value[" << source.fractionBits - 1 << ":0];\n"
         << "    significand = 64'd0;\n"
         << "    magnitude = 64'd0;\n"
         << "    unbiased_exponent = 0;\n"
         << "    shift_amount = 0;\n"
         << "    " << mode.functionName << " = " << unsignedLiteral(zero)
         << ";\n"
         << "    if (exponent_value == {" << source.exponentBits
         << "{1'b1}}) begin\n"
         << "      if (fraction_value != " << source.fractionBits << "'d0)\n"
         << "        " << mode.functionName << " = " << unsignedLiteral(zero)
         << ";\n"
         << "      else if (sign_value)\n"
         << "        " << mode.functionName << " = " << unsignedLiteral(minimum)
         << ";\n"
         << "      else\n"
         << "        " << mode.functionName << " = " << unsignedLiteral(maximum)
         << ";\n"
         << "    end else if (exponent_value != " << source.exponentBits
         << "'d0) begin\n"
         << "      unbiased_exponent = "
         << zeroExtend("exponent_value", source.exponentBits, 32) << " - "
         << source.bias() << ";\n"
         << "      if (unbiased_exponent >= 0) begin\n";
  if (mode.signedInteger) {
    output << "        if (sign_value && unbiased_exponent >= "
           << destinationWidth - 1 << ")\n"
           << "          " << mode.functionName << " = "
           << unsignedLiteral(minimum) << ";\n"
           << "        else if (!sign_value && unbiased_exponent >= "
           << destinationWidth - 1 << ")\n"
           << "          " << mode.functionName << " = "
           << unsignedLiteral(maximum) << ";\n"
           << "        else begin\n";
  } else {
    output << "        if (sign_value)\n"
           << "          " << mode.functionName << " = "
           << unsignedLiteral(zero) << ";\n"
           << "        else if (unbiased_exponent >= " << positiveLimitExponent
           << ")\n"
           << "          " << mode.functionName << " = "
           << unsignedLiteral(maximum) << ";\n"
           << "        else begin\n";
  }
  output << "          significand = "
         << zeroExtend("fraction_value", source.fractionBits, 64) << ";\n"
         << "          significand[" << source.fractionBits << "] = 1'b1;\n"
         << "          shift_amount = unbiased_exponent - "
         << source.fractionBits << ";\n"
         << "          if (shift_amount >= 0)\n"
         << "            magnitude = significand << shift_amount;\n"
         << "          else\n"
         << "            magnitude = significand >> (-shift_amount);\n";
  if (mode.signedInteger)
    output << "          if (sign_value)\n"
           << "            " << mode.functionName << " = (~magnitude["
           << destinationWidth - 1 << ":0]) + " << destinationWidth << "'d1;\n"
           << "          else\n"
           << "            " << mode.functionName << " = magnitude["
           << destinationWidth - 1 << ":0];\n";
  else
    output << "          " << mode.functionName << " = magnitude["
           << destinationWidth - 1 << ":0];\n";
  output << "        end\n"
         << "      end\n"
         << "    end\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

std::string buildFunction(const LoweredMode &mode) {
  switch (mode.kind) {
  case ConversionKind::FloatWidthCast:
    return buildFloatWidthCastFunction(mode);
  case ConversionKind::IntegerToFloat:
    return buildIntegerToFloatFunction(mode);
  case ConversionKind::FloatToInteger:
    return buildFloatToIntegerFunction(mode);
  }
  llvm_unreachable("unknown conversion kind");
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFloatConversion(FabricOperationProviderRequest request) {
  const Family family = request.capability.implementationFamily;
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (!isConversionFamily(family))
    return invalid("provider received a different implementation family");
  if (family == Family::ScalarFloatWidthCast) {
    if (!std::holds_alternative<::fabric::ScalarFloatWidthCastParams>(
            request.capability.parameterizedCapability))
      return invalid("capability has the wrong width-cast parameter schema");
  } else if (!std::holds_alternative<
                 ::fabric::ScalarIntegerFloatConversionParams>(
                 request.capability.parameterizedCapability)) {
    return invalid("capability has the wrong conversion parameter schema");
  }
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
       request.capability.physicalPorts)
    (port.reference.direction == fabric::FabricPortDirection::Input ? inputs
                                                                    : outputs)
        .push_back(&port);
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(inputs, byOrdinal);
  llvm::sort(outputs, byOrdinal);
  if (inputs.size() != 1 || outputs.size() != 1 ||
      inputs[0]->reference.ordinal != 0 || outputs[0]->reference.ordinal != 0 ||
      inputs[0]->payloadWidthBits == 0 || outputs[0]->payloadWidthBits == 0)
    return invalid("capability does not have the unary conversion port shape");
  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return error;

  auto relation = request.capability.resolveSemanticFieldRelation(
      *request.leaf.getContext());
  if (!relation)
    return relation.takeError();
  const auto &domain = relation->finiteBehaviorDomain();
  if (domain.empty())
    return invalid("sealed semantic relation has no behavior points");

  const ConfigurationFieldEncoding *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  std::vector<Mode> modes;
  if (request.capability.configurationFieldSchema.empty()) {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None)
      return invalid("configuration-free relation is not fieldless");
    if (domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid("configuration-free capability is not a singleton");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid("configured conversion relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured conversion capability requires one field");
    field = request.configurationAbi.findOperationField(
        request.occurrence,
        request.capability.configurationFieldSchema.front().ordinal);
    if (!field)
      return invalid("configured field is absent from the occurrence ABI");
    codebook = std::get_if<FiniteCodebookEncoding>(&field->semanticEncoding);
    if (!codebook)
      return invalid("configured field is not a finite codebook");
    if (codebook->entries.size() != domain.size())
      return invalid("codebook does not exactly cover the sealed domain");
    modes.reserve(domain.size());
    for (const auto &point : domain) {
      if (!point.semanticConfiguration)
        return invalid("configured behavior has no semantic value");
      const FiniteCodebookEntry *entry = detail::findFiniteCodebookEntry(
          *codebook, point.semanticConfiguration->bytes());
      if (!entry)
        return invalid("codebook has no admitted semantic value");
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
      return invalid("ABI inactive value is outside the sealed domain");
    inactiveMode = static_cast<std::size_t>(inactive - modes.begin());
  }

  std::vector<LoweredMode> loweredModes;
  loweredModes.reserve(modes.size());
  for (const Mode &mode : modes) {
    auto lowered = lowerMode(family, mode.actor);
    if (!lowered)
      return lowered.takeError();
    if (lowered->inputWidth() > inputs[0]->payloadWidthBits ||
        lowered->outputWidth() > outputs[0]->payloadWidthBits)
      return invalid("behavior payload exceeds the physical datapath");
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
        std::vector<std::string> declaredFunctions;
        for (const LoweredMode &mode : loweredModes) {
          if (llvm::is_contained(declaredFunctions, mode.functionName))
            continue;
          declaredFunctions.push_back(mode.functionName);
          declarationStream << buildFunction(mode) << '\n';
        }
        circt::sv::VerbatimOp::create(
            bodyBuilder, location,
            bodyBuilder.getStringAttr(declarationStream.str()));

        std::vector<mlir::Value> selectedModes(modes.size());
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
            selectedModes[index] = circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::eq,
                configuration, code, true);
          }
        }

        mlir::Value physicalInput = accessor.getInput("data_input_0");
        std::vector<mlir::Value> results;
        results.reserve(loweredModes.size());
        for (const LoweredMode &mode : loweredModes) {
          mlir::Value semanticInput = detail::resizeUnsigned(
              bodyBuilder, location, physicalInput, mode.inputWidth());
          mlir::Value converted = circt::sv::VerbatimExprOp::create(
              bodyBuilder, location,
              bodyBuilder.getIntegerType(mode.outputWidth()),
              mode.functionName + "({{0}})", mlir::ValueRange{semanticInput});
          results.push_back(detail::resizeUnsigned(
              bodyBuilder, location, converted, outputs[0]->payloadWidthBits));
        }

        mlir::Value result = results[inactiveMode];
        if (field) {
          for (std::size_t index = 0; index < modes.size(); ++index) {
            if (index == inactiveMode)
              continue;
            result = circt::comb::MuxOp::create(bodyBuilder, location,
                                                selectedModes[index],
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

llvm::Error registerPortableFloatConversionProviders(
    FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry candidate = registry;
  for (Family family :
       {Family::ScalarFloatWidthCast, Family::ScalarIntegerToFloat,
        Family::ScalarFloatToInteger}) {
    if (llvm::Error error =
            candidate.add({family,
                           BackendRecipeKey::PortableSystemVerilog,
                           {},
                           materializePortableFloatConversion}))
      return error;
  }
  registry = std::move(candidate);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl
