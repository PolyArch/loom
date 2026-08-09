#include "Hardware/RTL/Providers/MathRoot.h"

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
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

using Format = detail::PortableFloatFormat;

enum class RootFamily { Sqrt, Rsqrt };

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
                                 "portable_math_root_invalid: " + message);
}

llvm::Error unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

::fabric::ImplementationFamilyId familyId(RootFamily family) {
  return family == RootFamily::Sqrt
             ? ::fabric::ImplementationFamilyId::ScalarMathSqrt
             : ::fabric::ImplementationFamilyId::ScalarMathRsqrt;
}

::dataflow::OperationSchemaId schemaId(RootFamily family) {
  return family == RootFamily::Sqrt ? ::dataflow::OperationSchemaId::MathSqrt
                                    : ::dataflow::OperationSchemaId::MathRsqrt;
}

llvm::StringRef familyName(RootFamily family) {
  return family == RootFamily::Sqrt ? "sqrt" : "rsqrt";
}

std::string functionName(RootFamily family, const Format &format) {
  return "loom_math_" + familyName(family).str() + "_e" +
         std::to_string(format.exponentBits) + "_f" +
         std::to_string(format.fractionBits);
}

bool hasFastMathFlag(mlir::arith::FastMathFlags flags,
                     mlir::arith::FastMathFlags flag) {
  using Bits = std::underlying_type_t<mlir::arith::FastMathFlags>;
  return (static_cast<Bits>(flags) & static_cast<Bits>(flag)) != 0;
}

llvm::Expected<LoweredMode> lowerMode(RootFamily family, const Mode &mode) {
  if (mode.actor.schema != schemaId(family) ||
      mode.actor.type.getNumInputs() != 1 ||
      mode.actor.type.getNumResults() != 1 ||
      mode.actor.type.getInput(0) != mode.actor.type.getResult(0))
    return invalid("behavior is not the selected scalar math root family");
  const auto *payload =
      std::get_if<::dataflow::SpecialMathPayload>(&mode.actor.payload);
  if (!payload)
    return invalid("behavior has no special-math accuracy projection");
  if (llvm::Error error = loom::validateSpecialMathAccuracyContract(
          payload->accuracy,
          hasFastMathFlag(payload->flags, mlir::arith::FastMathFlags::afn)))
    return std::move(error);
  auto format = detail::resolvePortableFloatFormat(mode.actor.type.getInput(0));
  if (!format)
    return invalid("behavior uses an unsupported floating format");
  return LoweredMode{*format, functionName(family, *format)};
}

unsigned evenRadicandWidth(const Format &format) {
  const unsigned required = 3 * format.fractionBits + 4;
  return required + (required & 1U);
}

std::string buildDeclarations(RootFamily family, const Format &format) {
  const unsigned fractionBits = format.fractionBits;
  const unsigned radicandWidth = evenRadicandWidth(format);
  const unsigned rootBits = fractionBits + 2;
  const unsigned quotientWidth = 2 * rootBits;
  const unsigned roundingWidth = fractionBits + 3;
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "  reg sign_input;\n"
         << "  reg [" << format.exponentBits - 1 << ":0] exponent_input;\n"
         << "  reg [" << fractionBits - 1 << ":0] fraction_input;\n"
         << "  reg [" << fractionBits << ":0] mantissa;\n"
         << "  reg [" << rootBits + 1 << ":0] remainder;\n"
         << "  reg [" << rootBits - 1 << ":0] root;\n"
         << "  reg [" << rootBits + 1 << ":0] trial;\n"
         << "  reg [" << rootBits << ":0] rounded;\n";
  if (family == RootFamily::Sqrt)
    output << "  reg [" << radicandWidth - 1 << ":0] radicand;\n";
  else
    output << "  reg [" << fractionBits << ":0] normalized_mantissa;\n"
           << "  reg [" << quotientWidth - 1 << ":0] quotient;\n"
           << "  reg [" << fractionBits + 1 << ":0] divide_remainder;\n"
           << "  reg [" << roundingWidth - 1 << ":0] divide_remainder_four;\n";
  output << "  reg [" << format.exponentBits - 1 << ":0] exponent_result;\n"
         << "  reg [" << fractionBits - 1 << ":0] fraction_result;\n"
         << "  integer exponent_value;\n"
         << "  integer leading_index;\n"
         << "  integer normalized_exponent;\n"
         << "  integer parity;\n"
         << "  integer result_exponent_value;\n";
  if (family == RootFamily::Sqrt)
    output << "  integer shift_amount;\n";
  output << "  integer index;\n"
         << "  reg found;\n";
  return output.str();
}

std::string buildCommonDecode(const Format &format) {
  const unsigned fractionBits = format.fractionBits;
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "      mantissa = exponent_input == 0"
         << " ? {1'b0, fraction_input} : {1'b1, fraction_input};\n"
         << "      exponent_value = exponent_input == 0 ? "
         << format.minimumExponent() - static_cast<int>(fractionBits)
         << " : integer'(exponent_input) - " << format.bias() << " - "
         << fractionBits << ";\n"
         << "      leading_index = 0;\n"
         << "      found = 1'b0;\n"
         << "      for (index = " << fractionBits
         << "; index >= 0; index = index - 1) begin\n"
         << "        if (!found && mantissa[index]) begin\n"
         << "          leading_index = index;\n"
         << "          found = 1'b1;\n"
         << "        end\n"
         << "      end\n"
         << "      normalized_exponent = exponent_value + leading_index;\n"
         << "      parity = normalized_exponent & 1;\n";
  return output.str();
}

std::string buildIntegerSquareRoot(const Format &format,
                                   llvm::StringRef source) {
  const unsigned rootBits = format.fractionBits + 2;
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "      remainder = " << rootBits + 2 << "'d0;\n"
         << "      root = " << rootBits << "'d0;\n"
         << "      for (index = " << rootBits - 1
         << "; index >= 0; index = index - 1) begin\n"
         << "        remainder = {remainder[" << rootBits - 1 << ":0], "
         << source << "[(2 * index) +: 2]};\n"
         << "        trial = {root, 2'b01};\n"
         << "        if (remainder >= trial) begin\n"
         << "          remainder = remainder - trial;\n"
         << "          root = {root[" << rootBits - 2 << ":0], 1'b1};\n"
         << "        end else begin\n"
         << "          root = {root[" << rootBits - 2 << ":0], 1'b0};\n"
         << "        end\n"
         << "      end\n";
  return output.str();
}

std::string buildPack(const Format &format) {
  const unsigned fractionBits = format.fractionBits;
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "      if (rounded[" << fractionBits + 1 << "]) begin\n"
         << "        rounded = rounded >> 1;\n"
         << "        result_exponent_value = result_exponent_value + 1;\n"
         << "      end\n"
         << "      result_exponent_value = result_exponent_value + "
         << format.bias() << ";\n"
         << "      exponent_result = result_exponent_value["
         << format.exponentBits - 1 << ":0];\n"
         << "      fraction_result = rounded[" << fractionBits - 1 << ":0];\n";
  return output.str();
}

// Scaling x = mantissa * 2^exponent makes the correctly rounded significand
// the nearest integer to one finite-width integer square root.
std::string buildSqrtFunction(const Format &format, llvm::StringRef name) {
  const unsigned width = format.width();
  const unsigned fractionBits = format.fractionBits;
  const unsigned radicandWidth = evenRadicandWidth(format);
  const llvm::APInt quietBit =
      llvm::APInt::getOneBitSet(width, fractionBits - 1);
  const llvm::APInt quietNaN =
      llvm::APInt(width, (std::uint64_t{1} << format.exponentBits) - 1)
          .shl(fractionBits) |
      quietBit | llvm::APInt::getOneBitSet(width, width - 1);
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic [" << width - 1 << ":0] " << name << "(input ["
         << width - 1 << ":0] value);\n"
         << buildDeclarations(RootFamily::Sqrt, format) << "  begin\n"
         << "    sign_input = value[" << width - 1 << "];\n"
         << "    exponent_input = value[" << width - 2 << ":" << fractionBits
         << "];\n"
         << "    fraction_input = value[" << fractionBits - 1 << ":0];\n"
         << "    if (exponent_input == " << format.exponentBits << "'d"
         << (std::uint64_t{1} << format.exponentBits) - 1
         << " && fraction_input != 0) begin\n"
         << "      " << name << " = value | " << width << "'h";
  llvm::SmallString<32> quietDigits;
  quietBit.toStringUnsigned(quietDigits, 16);
  output << quietDigits << ";\n"
         << "    end else if (sign_input && !(exponent_input == 0 && "
            "fraction_input == 0)) begin\n"
         << "      " << name << " = " << width << "'h";
  llvm::SmallString<32> nanDigits;
  quietNaN.toStringUnsigned(nanDigits, 16);
  output << nanDigits << ";\n"
         << "    end else if (exponent_input == " << format.exponentBits << "'d"
         << (std::uint64_t{1} << format.exponentBits) - 1 << ") begin\n"
         << "      " << name << " = value;\n"
         << "    end else if (exponent_input == 0 && fraction_input == 0) "
            "begin\n"
         << "      " << name << " = value;\n"
         << "    end else begin\n"
         << buildCommonDecode(format)
         << "      result_exponent_value = (normalized_exponent - parity) "
            ">>> 1;\n"
         << "      shift_amount = " << 2 * fractionBits
         << " - leading_index + parity;\n"
         << "      radicand = " << radicandWidth << "'d0;\n"
         << "      radicand = {{" << radicandWidth - fractionBits - 1
         << "{1'b0}}, mantissa} << shift_amount;\n"
         << buildIntegerSquareRoot(format, "radicand")
         << "      rounded = {1'b0, root};\n"
         << "      if (remainder > {2'b00, root})\n"
         << "        rounded = rounded + " << fractionBits + 3 << "'d1;\n"
         << buildPack(format) << "      " << name
         << " = {1'b0, exponent_result, "
            "fraction_result};\n"
         << "    end\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

// For N = D*q + u and q = r*r + s, midpoint rounding compares s with r.
// Only s == r needs the residual comparison 4*u against D because 0 <= u < D.
std::string buildRsqrtFunction(const Format &format, llvm::StringRef name) {
  const unsigned width = format.width();
  const unsigned fractionBits = format.fractionBits;
  const unsigned rootBits = fractionBits + 2;
  const unsigned quotientWidth = 2 * rootBits;
  const unsigned divisionSteps = 2 * fractionBits + 3;
  const llvm::APInt quietBit =
      llvm::APInt::getOneBitSet(width, fractionBits - 1);
  const llvm::APInt exponentOnes =
      llvm::APInt(width, (std::uint64_t{1} << format.exponentBits) - 1)
          .shl(fractionBits);
  const llvm::APInt quietNaN =
      exponentOnes | quietBit | llvm::APInt::getOneBitSet(width, width - 1);
  std::string text;
  llvm::raw_string_ostream output(text);
  llvm::SmallString<32> quietDigits;
  llvm::SmallString<32> nanDigits;
  quietBit.toStringUnsigned(quietDigits, 16);
  quietNaN.toStringUnsigned(nanDigits, 16);
  output << "function automatic [" << width - 1 << ":0] " << name << "(input ["
         << width - 1 << ":0] value);\n"
         << buildDeclarations(RootFamily::Rsqrt, format) << "  begin\n"
         << "    sign_input = value[" << width - 1 << "];\n"
         << "    exponent_input = value[" << width - 2 << ":" << fractionBits
         << "];\n"
         << "    fraction_input = value[" << fractionBits - 1 << ":0];\n"
         << "    if (exponent_input == " << format.exponentBits << "'d"
         << (std::uint64_t{1} << format.exponentBits) - 1
         << " && fraction_input != 0) begin\n"
         << "      " << name << " = value | " << width << "'h" << quietDigits
         << ";\n"
         << "    end else if (sign_input && !(exponent_input == 0 && "
            "fraction_input == 0)) begin\n"
         << "      " << name << " = " << width << "'h" << nanDigits << ";\n"
         << "    end else if (exponent_input == 0 && fraction_input == 0) "
            "begin\n"
         << "      " << name << " = {1'b0, {" << format.exponentBits
         << "{1'b1}}, {" << fractionBits << "{1'b0}}};\n"
         << "    end else if (exponent_input == " << format.exponentBits << "'d"
         << (std::uint64_t{1} << format.exponentBits) - 1 << ") begin\n"
         << "      " << name << " = " << width << "'d0;\n"
         << "    end else begin\n"
         << buildCommonDecode(format)
         << "      result_exponent_value = -((normalized_exponent - parity) "
            ">>> 1) - 1;\n"
         << "      normalized_mantissa = mantissa << (" << fractionBits
         << " - leading_index);\n"
         << "      quotient = " << quotientWidth << "'d0;\n"
         << "      divide_remainder = " << fractionBits + 2
         << "'d1 << (parity != 0 ? " << fractionBits - 2 << " : "
         << fractionBits - 1 << ");\n"
         << "      for (index = " << divisionSteps - 1
         << "; index >= 0; index = index - 1) begin\n"
         << "        divide_remainder = {divide_remainder[" << fractionBits
         << ":0], 1'b0};\n"
         << "        if (divide_remainder >= {1'b0, "
            "normalized_mantissa}) begin\n"
         << "          divide_remainder = divide_remainder - "
            "normalized_mantissa;\n"
         << "          quotient = {quotient[" << quotientWidth - 2
         << ":0], 1'b1};\n"
         << "        end else begin\n"
         << "          quotient = {quotient[" << quotientWidth - 2
         << ":0], 1'b0};\n"
         << "        end\n"
         << "      end\n"
         << buildIntegerSquareRoot(format, "quotient")
         << "      rounded = {1'b0, root};\n"
         << "      if (remainder > {{2{1'b0}}, root}) begin\n"
         << "        rounded = rounded + " << fractionBits + 3 << "'d1;\n"
         << "      end else if (remainder == {{2{1'b0}}, root}) begin\n"
         << "        divide_remainder_four = {1'b0, divide_remainder} << 2;\n"
         << "        if (divide_remainder_four > {{2{1'b0}}, "
            "normalized_mantissa} ||\n"
         << "            (divide_remainder_four == {{2{1'b0}}, "
            "normalized_mantissa} && root[0]))\n"
         << "          rounded = rounded + " << fractionBits + 3 << "'d1;\n"
         << "      end\n"
         << buildPack(format) << "      " << name
         << " = {1'b0, exponent_result, "
            "fraction_result};\n"
         << "    end\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

std::string buildFunction(RootFamily family, const LoweredMode &mode) {
  return family == RootFamily::Sqrt
             ? buildSqrtFunction(mode.format, mode.functionName)
             : buildRsqrtFunction(mode.format, mode.functionName);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableMathRoot(FabricOperationProviderRequest request,
                            RootFamily family) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily != familyId(family))
    return invalid("provider received a different implementation family");
  if (!std::holds_alternative<::fabric::ScalarSpecialMathParams>(
          request.capability.parameterizedCapability))
    return invalid("capability has the wrong parameter schema");
  if (request.capability.enabledOperationSchemas !=
      std::vector<::dataflow::OperationSchemaId>{schemaId(family)})
    return invalid("capability does not contain exactly its math root schema");

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
    return unsupported(request);
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
      return invalid("configuration-free math root relation is not fieldless");
    if (domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid("configuration-free capability is not singleton");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid("configured math root relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured math root capability requires one field");
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
  std::set<std::string> names;
  loweredModes.reserve(modes.size());
  for (const Mode &mode : modes) {
    auto lowered = lowerMode(family, mode);
    if (!lowered)
      return lowered.takeError();
    if (!names.insert(lowered->functionName).second)
      return invalid("sealed relation contains a duplicate math root mode");
    const unsigned width = lowered->format.width();
    if (inputs[0]->payloadWidthBits < width ||
        outputs[0]->payloadWidthBits < width)
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
          declarationStream << buildFunction(family, mode) << '\n';
        circt::sv::VerbatimOp::create(
            bodyBuilder, location,
            bodyBuilder.getStringAttr(declarationStream.str()));

        std::vector<mlir::Value> results;
        results.reserve(loweredModes.size());
        for (const LoweredMode &mode : loweredModes) {
          const unsigned width = mode.format.width();
          mlir::Value input = detail::resizeUnsigned(
              bodyBuilder, location, accessor.getInput("data_input_0"), width);
          mlir::Value value = circt::sv::VerbatimExprOp::create(
              bodyBuilder, location, bodyBuilder.getIntegerType(width),
              mode.functionName + "({{0}})",
              llvm::SmallVector<mlir::Value, 1>{input});
          results.push_back(detail::resizeUnsigned(
              bodyBuilder, location, value, outputs[0]->payloadWidthBits));
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
materializePortableSqrt(FabricOperationProviderRequest request) {
  return materializePortableMathRoot(std::move(request), RootFamily::Sqrt);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableRsqrt(FabricOperationProviderRequest request) {
  return materializePortableMathRoot(std::move(request), RootFamily::Rsqrt);
}

} // namespace

llvm::Error
registerPortableMathRootProviders(FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry candidate = registry;
  if (llvm::Error error =
          candidate.add({::fabric::ImplementationFamilyId::ScalarMathSqrt,
                         BackendRecipeKey::PortableSystemVerilog,
                         {},
                         materializePortableSqrt}))
    return error;
  if (llvm::Error error =
          candidate.add({::fabric::ImplementationFamilyId::ScalarMathRsqrt,
                         BackendRecipeKey::PortableSystemVerilog,
                         {},
                         materializePortableRsqrt}))
    return error;
  registry = std::move(candidate);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl
