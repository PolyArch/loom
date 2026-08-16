#include "Hardware/RTL/Providers/MathLogarithm.h"

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

enum class MathFamily { Log, Log2, Log10, Log1p };
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
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_math_logarithm_invalid: " + message);
}

llvm::Error unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

::fabric::ImplementationFamilyId familyId(MathFamily family) {
  using Id = ::fabric::ImplementationFamilyId;
  switch (family) {
  case MathFamily::Log:
    return Id::ScalarMathLog;
  case MathFamily::Log2:
    return Id::ScalarMathLog2;
  case MathFamily::Log10:
    return Id::ScalarMathLog10;
  case MathFamily::Log1p:
    return Id::ScalarMathLog1p;
  }
  llvm_unreachable("unknown logarithm family");
}

::dataflow::OperationSchemaId schemaId(MathFamily family) {
  using Id = ::dataflow::OperationSchemaId;
  switch (family) {
  case MathFamily::Log:
    return Id::MathLog;
  case MathFamily::Log2:
    return Id::MathLog2;
  case MathFamily::Log10:
    return Id::MathLog10;
  case MathFamily::Log1p:
    return Id::MathLog1p;
  }
  llvm_unreachable("unknown logarithm family");
}

llvm::StringRef familyName(MathFamily family) {
  switch (family) {
  case MathFamily::Log:
    return "log";
  case MathFamily::Log2:
    return "log2";
  case MathFamily::Log10:
    return "log10";
  case MathFamily::Log1p:
    return "log1p";
  }
  llvm_unreachable("unknown logarithm family");
}

std::string functionName(MathFamily family, const Format &format) {
  return "loom_math_" + familyName(family).str() + "_e" +
         std::to_string(format.exponentBits) + "_f" +
         std::to_string(format.fractionBits);
}

llvm::Expected<LoweredMode>
lowerMode(MathFamily family,
          const ::dataflow::CanonicalActorSchemaProjection &actor) {
  if (actor.schema != schemaId(family) || actor.type.getNumInputs() != 1 ||
      actor.type.getNumResults() != 1)
    return invalid("behavior is not the selected unary logarithm operation");
  if (actor.type.getResult(0) != actor.type.getInput(0) ||
      llvm::isa<mlir::VectorType>(actor.type.getInput(0)))
    return invalid("behavior does not have one uniform scalar floating type");
  const auto *payload =
      std::get_if<::dataflow::SpecialMathPayload>(&actor.payload);
  if (!payload)
    return invalid("behavior has no special-math accuracy projection");
  if (payload->accuracy != ::loom::SpecialMathAccuracyTier::Max4Ulp)
    return invalid("sealed behavior differs from the resource accuracy");
  auto format = detail::resolvePortableFloatFormat(actor.type.getInput(0));
  if (!format)
    return invalid("behavior uses an unsupported floating format");
  return LoweredMode{*format, functionName(family, *format)};
}

std::string hexLiteral(unsigned width, std::uint64_t value) {
  std::ostringstream stream;
  stream << width << "'h" << std::hex << value;
  return stream.str();
}

std::string buildFixedPointFunctions(const Format &format,
                                     llvm::StringRef prefix) {
  constexpr unsigned fractionalBits = 52;
  constexpr std::uint64_t one = std::uint64_t{1} << fractionalBits;
  constexpr std::uint64_t sqrtTwo = 6369051672525773ULL;
  constexpr std::uint64_t naturalLogTwo = 3121657384082680ULL;
  const unsigned width = format.width();
  const unsigned precision = format.precision();
  const unsigned exponentBits = format.exponentBits;
  const unsigned fractionBits = format.fractionBits;
  const std::string multiply = prefix.str() + "_mul_q";
  const std::string divide = prefix.str() + "_div_q";
  const std::string logarithm = prefix.str() + "_ln_q";
  const std::string pack = prefix.str() + "_pack_q";
  const std::string exponentAllOnes =
      hexLiteral(exponentBits, (std::uint64_t{1} << exponentBits) - 1);

  // Q11.52 range reduction keeps the divider numerator below its denominator.
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic signed [63:0] " << multiply
         << "(input signed [63:0] lhs, input signed [63:0] rhs);\n"
         << "  reg signed [127:0] product;\n"
         << "  begin\n"
         << "    product = lhs * rhs;\n"
         << "    " << multiply << " = product[" << fractionalBits + 63 << ':'
         << fractionalBits << "];\n"
         << "  end\n"
         << "endfunction\n\n"
         << "function automatic signed [63:0] " << divide
         << "(input signed [63:0] numerator, "
            "input signed [63:0] denominator);\n"
         << "  reg negative_result;\n"
         << "  reg [63:0] numerator_magnitude;\n"
         << "  reg [63:0] denominator_magnitude;\n"
         << "  reg [64:0] remainder;\n"
         << "  reg [63:0] quotient;\n"
         << "  integer index;\n"
         << "  begin\n"
         << "    negative_result = numerator[63] ^ denominator[63];\n"
         << "    numerator_magnitude = numerator[63] ? -numerator : "
            "numerator;\n"
         << "    denominator_magnitude = denominator[63] ? -denominator : "
            "denominator;\n"
         << "    remainder = {1'b0, numerator_magnitude};\n"
         << "    quotient = 64'd0;\n"
         << "    for (index = 0; index < " << fractionalBits
         << "; index = index + 1) begin\n"
         << "      remainder = remainder << 1;\n"
         << "      quotient = quotient << 1;\n"
         << "      if (remainder >= {1'b0, denominator_magnitude}) begin\n"
         << "        remainder = remainder - "
            "{1'b0, denominator_magnitude};\n"
         << "        quotient[0] = 1'b1;\n"
         << "      end\n"
         << "    end\n"
         << "    " << divide
         << " = negative_result ? -$signed(quotient) : $signed(quotient);\n"
         << "  end\n"
         << "endfunction\n\n"
         << "function automatic signed [63:0] " << logarithm
         << "(input signed [63:0] input_mantissa, "
            "input integer input_exponent);\n"
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
         << "    if (mantissa > 64'sd" << sqrtTwo << ") begin\n"
         << "      mantissa = mantissa >>> 1;\n"
         << "      exponent_value = exponent_value + 1;\n"
         << "    end\n"
         << "    z = " << divide << "(mantissa - 64'sd" << one
         << ", mantissa + 64'sd" << one << ");\n"
         << "    z_squared = " << multiply << "(z, z);\n"
         << "    term = z;\n"
         << "    sum = z;\n";
  for (std::uint64_t reciprocal :
       {1501199875790165ULL, 900719925474099ULL, 643371375338642ULL,
        500399958596722ULL, 409418147942772ULL, 346430740566962ULL,
        300239975158033ULL})
    output << "    term = " << multiply << "(term, z_squared);\n"
           << "    sum = sum + " << multiply << "(term, 64'sd" << reciprocal
           << ");\n";
  output << "    exponent_term = "
            "{{32{exponent_value[31]}}, exponent_value};\n"
         << "    exponent_term = exponent_term * 64'sd" << naturalLogTwo
         << ";\n"
         << "    " << logarithm << " = (sum <<< 1) + exponent_term;\n"
         << "  end\n"
         << "endfunction\n\n"
         << "function automatic [" << width - 1 << ":0] " << pack
         << "(input signed [63:0] value);\n"
         << "  reg sign_result;\n"
         << "  reg [63:0] magnitude;\n"
         << "  reg [63:0] significand;\n"
         << "  reg [" << precision << ":0] rounded;\n"
         << "  reg [" << exponentBits - 1 << ":0] exponent_result;\n"
         << "  reg [" << fractionBits - 1 << ":0] fraction_result;\n"
         << "  reg found;\n"
         << "  reg guard;\n"
         << "  reg sticky;\n"
         << "  reg increment;\n"
         << "  integer leading_index;\n"
         << "  integer exponent_value;\n"
         << "  integer encoded_exponent;\n"
         << "  integer shift_amount;\n"
         << "  integer index;\n"
         << "  begin\n"
         << "    " << pack << " = " << width << "'d0;\n"
         << "    if (value != 0) begin\n"
         << "      sign_result = value < 0;\n"
         << "      magnitude = sign_result ? -value : value;\n"
         << "      leading_index = 0;\n"
         << "      found = 1'b0;\n"
         << "      for (index = 63; index >= 0; index = index - 1) begin\n"
         << "        if (!found && magnitude[index]) begin\n"
         << "          leading_index = index;\n"
         << "          found = 1'b1;\n"
         << "        end\n"
         << "      end\n"
         << "      exponent_value = leading_index - " << fractionalBits << ";\n"
         << "      shift_amount = leading_index - " << fractionBits << ";\n"
         << "      guard = 1'b0;\n"
         << "      sticky = 1'b0;\n"
         << "      if (shift_amount > 0) begin\n"
         << "        significand = magnitude >> shift_amount;\n"
         << "        guard = magnitude[shift_amount - 1];\n"
         << "        for (index = 0; index < 64; index = index + 1)\n"
         << "          if (index < shift_amount - 1) "
            "sticky = sticky | magnitude[index];\n"
         << "      end else begin\n"
         << "        significand = magnitude << (-shift_amount);\n"
         << "      end\n"
         << "      increment = guard && (sticky || significand[0]);\n"
         << "      rounded = {1'b0, significand[" << precision - 1
         << ":0]} + increment;\n"
         << "      if (rounded[" << precision << "]) begin\n"
         << "        rounded = rounded >> 1;\n"
         << "        exponent_value = exponent_value + 1;\n"
         << "      end\n"
         << "      encoded_exponent = exponent_value + " << format.bias()
         << ";\n"
         << "      if (encoded_exponent >= "
         << (std::uint64_t{1} << exponentBits) - 1 << ") begin\n"
         << "        " << pack << " = {sign_result, " << exponentAllOnes << ", "
         << fractionBits << "'d0};\n"
         << "      end else if (encoded_exponent > 0) begin\n"
         << "        exponent_result = encoded_exponent[" << exponentBits - 1
         << ":0];\n"
         << "        fraction_result = rounded[" << fractionBits - 1 << ":0];\n"
         << "        " << pack
         << " = {sign_result, exponent_result, fraction_result};\n"
         << "      end else begin\n"
         << "        " << pack << " = {sign_result, " << width - 1 << "'d0};\n"
         << "      end\n"
         << "    end\n"
         << "  end\n"
         << "endfunction\n";
  return text;
}

std::string buildModeFunction(MathFamily family, const LoweredMode &mode) {
  constexpr unsigned fixedFractionBits = 52;
  constexpr std::uint64_t oneQ = std::uint64_t{1} << fixedFractionBits;
  constexpr std::uint64_t log2OfE = 6497320848556798ULL;
  constexpr std::uint64_t log10OfE = 1955888466868548ULL;
  const Format &format = mode.format;
  const unsigned width = format.width();
  const unsigned exponentBits = format.exponentBits;
  const unsigned fractionBits = format.fractionBits;
  const unsigned precision = format.precision();
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponentBits) - 1;
  const std::uint64_t infinity = exponentMask << fractionBits;
  const std::uint64_t quietBit = std::uint64_t{1} << (fractionBits - 1);
  const std::uint64_t quietNaN = infinity | quietBit;
  const std::uint64_t one =
      std::uint64_t((std::uint64_t{1} << (exponentBits - 1)) - 1)
      << fractionBits;
  const std::string prefix = mode.functionName;
  const std::string multiply = prefix + "_mul_q";
  const std::string logarithm = prefix + "_ln_q";
  const std::string pack = prefix + "_pack_q";
  const std::string exponentAllOnes = hexLiteral(exponentBits, exponentMask);

  std::string text = buildFixedPointFunctions(format, prefix);
  llvm::raw_string_ostream output(text);
  output << "\nfunction automatic [" << width - 1 << ":0] " << mode.functionName
         << "(input [" << width - 1 << ":0] operand);\n"
         << "  reg sign_operand;\n"
         << "  reg [" << exponentBits - 1 << ":0] exponent_operand;\n"
         << "  reg [" << fractionBits - 1 << ":0] fraction_operand;\n"
         << "  reg [" << precision - 1 << ":0] significand;\n"
         << "  reg signed [63:0] mantissa_q;\n"
         << "  reg signed [63:0] normalized_q;\n"
         << "  reg signed [63:0] value_q;\n"
         << "  integer exponent_value;\n"
         << "  integer normalized_exponent;\n"
         << "  integer index;\n"
         << "  begin\n"
         << "    " << mode.functionName << " = " << hexLiteral(width, quietNaN)
         << ";\n"
         << "    sign_operand = operand[" << width - 1 << "];\n"
         << "    exponent_operand = operand[" << width - 2 << ':'
         << fractionBits << "];\n"
         << "    fraction_operand = operand[" << fractionBits - 1 << ":0];\n"
         << "    if (exponent_operand == " << exponentAllOnes
         << " && fraction_operand != 0) begin\n"
         << "      " << mode.functionName << " = operand | "
         << hexLiteral(width, quietBit) << ";\n"
         << "    end else if (exponent_operand == 0 && "
            "fraction_operand == 0) begin\n";
  if (family == MathFamily::Log1p)
    output << "      " << mode.functionName << " = operand;\n";
  else
    output << "      " << mode.functionName << " = {1'b1, " << exponentAllOnes
           << ", " << fractionBits << "'d0};\n";
  output << "    end";
  if (family == MathFamily::Log1p) {
    output << " else if (exponent_operand == " << exponentAllOnes << ") begin\n"
           << "      " << mode.functionName << " = sign_operand ? "
           << hexLiteral(width, quietNaN) << " : operand;\n"
           << "    end else if (sign_operand && operand[" << width - 2
           << ":0] == " << hexLiteral(width - 1, one) << ") begin\n"
           << "      " << mode.functionName << " = {1'b1, " << exponentAllOnes
           << ", " << fractionBits << "'d0};\n"
           << "    end else if (sign_operand && operand[" << width - 2
           << ":0] > " << hexLiteral(width - 1, one) << ") begin\n"
           << "      " << mode.functionName << " = "
           << hexLiteral(width, quietNaN) << ";\n"
           << "    end else begin\n"
           << "      significand = exponent_operand == 0"
              " ? {1'b0, fraction_operand} : {1'b1, fraction_operand};\n"
           << "      exponent_value = integer'(exponent_operand);\n"
           << "      exponent_value = exponent_operand == 0 ? "
           << format.minimumExponent() << " : exponent_value - "
           << format.bias() << ";\n"
           << "      for (index = 0; index < " << precision
           << "; index = index + 1) begin\n"
           << "        if (significand != 0 && !significand[" << precision - 1
           << "]) begin significand = significand << 1; "
              "exponent_value = exponent_value - 1; end\n"
           << "      end\n"
           << "      if (exponent_value <= -" << fractionBits + 3 << ") begin\n"
           << "        " << mode.functionName << " = operand;\n"
           << "      end else begin\n"
           << "        mantissa_q = {{" << 64 - precision
           << "{1'b0}}, significand};\n"
           << "        mantissa_q = mantissa_q <<< "
           << fixedFractionBits - fractionBits << ";\n"
           << "        if (sign_operand) begin\n"
           << "          normalized_q = 64'sd" << oneQ
           << " - (mantissa_q >>> (-exponent_value));\n"
           << "          normalized_exponent = 0;\n"
           << "          for (index = 0; index < " << precision + 1
           << "; index = index + 1) begin\n"
           << "            if (normalized_q < 64'sd" << oneQ
           << ") begin normalized_q = normalized_q <<< 1; "
              "normalized_exponent = normalized_exponent - 1; end\n"
           << "          end\n"
           << "        end else if (exponent_value >= 0) begin\n"
           << "          normalized_q = mantissa_q + "
              "(64'sd"
           << oneQ << " >>> exponent_value);\n"
           << "          normalized_exponent = exponent_value;\n"
           << "          if (normalized_q >= 64'sd" << 2 * oneQ
           << ") begin normalized_q = normalized_q >>> 1; "
              "normalized_exponent = normalized_exponent + 1; end\n"
           << "        end else begin\n"
           << "          normalized_q = 64'sd" << oneQ
           << " + (mantissa_q >>> (-exponent_value));\n"
           << "          normalized_exponent = 0;\n"
           << "        end\n"
           << "        value_q = " << logarithm
           << "(normalized_q, normalized_exponent);\n"
           << "        " << mode.functionName << " = " << pack << "(value_q);\n"
           << "      end\n"
           << "    end\n";
  } else {
    output << " else if (sign_operand) begin\n"
           << "      " << mode.functionName << " = "
           << hexLiteral(width, quietNaN) << ";\n"
           << "    end else if (exponent_operand == " << exponentAllOnes
           << ") begin\n"
           << "      " << mode.functionName << " = operand;\n"
           << "    end else begin\n"
           << "      significand = exponent_operand == 0"
              " ? {1'b0, fraction_operand} : {1'b1, fraction_operand};\n"
           << "      exponent_value = integer'(exponent_operand);\n"
           << "      exponent_value = exponent_operand == 0 ? "
           << format.minimumExponent() << " : exponent_value - "
           << format.bias() << ";\n"
           << "      for (index = 0; index < " << precision
           << "; index = index + 1) begin\n"
           << "        if (significand != 0 && !significand[" << precision - 1
           << "]) begin significand = significand << 1; "
              "exponent_value = exponent_value - 1; end\n"
           << "      end\n"
           << "      mantissa_q = {{" << 64 - precision
           << "{1'b0}}, significand};\n"
           << "      mantissa_q = mantissa_q <<< "
           << fixedFractionBits - fractionBits << ";\n"
           << "      value_q = " << logarithm
           << "(mantissa_q, exponent_value);\n";
    if (family == MathFamily::Log2) {
      output << "      if (significand == "
             << hexLiteral(precision, std::uint64_t{1} << fractionBits)
             << ") begin\n"
             << "        value_q = "
                "{{32{exponent_value[31]}}, exponent_value};\n"
             << "        value_q = value_q <<< " << fixedFractionBits << ";\n"
             << "      end else begin\n"
             << "        value_q = " << multiply << "(value_q, 64'sd" << log2OfE
             << ");\n"
             << "      end\n";
    } else if (family == MathFamily::Log10) {
      output << "      value_q = " << multiply << "(value_q, 64'sd" << log10OfE
             << ");\n";
    }
    output << "      " << mode.functionName << " = " << pack << "(value_q);\n"
           << "    end\n";
  }
  output << "  end\n"
         << "endfunction\n";
  return text;
}

mlir::Value callMode(mlir::OpBuilder &builder, mlir::Location location,
                     llvm::StringRef function, mlir::Value operand,
                     unsigned width) {
  return circt::sv::VerbatimExprOp::create(
      builder, location, builder.getIntegerType(width),
      function.str() + "({{0}})", llvm::SmallVector<mlir::Value, 1>{operand});
}

mlir::Value materializeMode(mlir::OpBuilder &builder, mlir::Location location,
                            circt::hw::HWModulePortAccessor &accessor,
                            const LoweredMode &mode, unsigned outputWidth) {
  const unsigned width = mode.format.width();
  mlir::Value operand = detail::resizeUnsigned(
      builder, location, accessor.getInput("data_input_0"), width);
  mlir::Value result =
      callMode(builder, location, mode.functionName, operand, width);
  return detail::resizeUnsigned(builder, location, result, outputWidth);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableMathLogarithm(FabricOperationProviderRequest request,
                                 MathFamily family) {
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
  if (*actualContract != *supportedContract ||
      parameters->accuracyGuarantee != ::loom::SpecialMathAccuracyTier::Max4Ulp)
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
      return invalid("configuration-free logarithm relation is not fieldless");
    if (domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid("configuration-free logarithm relation is not singleton");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid("configured logarithm relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured logarithm capability requires one field");
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
        return invalid("codebook omits an admitted semantic value");
      modes.push_back({point.representativeActor, entry});
    }
  }
  if (modes.empty())
    return invalid("sealed logarithm behavior relation is empty");

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
    if (lowered->format.width() == 64)
      return unsupported(request);
    if (!functionNames.insert(lowered->functionName).second)
      return invalid("sealed relation contains a duplicate logarithm mode");
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
materializePortableScalarMathLog(FabricOperationProviderRequest request) {
  return materializePortableMathLogarithm(std::move(request), MathFamily::Log);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarMathLog2(FabricOperationProviderRequest request) {
  return materializePortableMathLogarithm(std::move(request), MathFamily::Log2);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarMathLog10(FabricOperationProviderRequest request) {
  return materializePortableMathLogarithm(std::move(request),
                                          MathFamily::Log10);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarMathLog1p(FabricOperationProviderRequest request) {
  return materializePortableMathLogarithm(std::move(request),
                                          MathFamily::Log1p);
}

} // namespace

llvm::Error registerPortableMathLogarithmProviders(
    FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry staged = registry;
  if (llvm::Error error =
          staged.add({::fabric::ImplementationFamilyId::ScalarMathLog,
                      BackendRecipeKey::PortableSystemVerilog,
                      {},
                      materializePortableScalarMathLog}))
    return error;
  if (llvm::Error error =
          staged.add({::fabric::ImplementationFamilyId::ScalarMathLog2,
                      BackendRecipeKey::PortableSystemVerilog,
                      {},
                      materializePortableScalarMathLog2}))
    return error;
  if (llvm::Error error =
          staged.add({::fabric::ImplementationFamilyId::ScalarMathLog10,
                      BackendRecipeKey::PortableSystemVerilog,
                      {},
                      materializePortableScalarMathLog10}))
    return error;
  if (llvm::Error error =
          staged.add({::fabric::ImplementationFamilyId::ScalarMathLog1p,
                      BackendRecipeKey::PortableSystemVerilog,
                      {},
                      materializePortableScalarMathLog1p}))
    return error;
  registry = std::move(staged);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl
