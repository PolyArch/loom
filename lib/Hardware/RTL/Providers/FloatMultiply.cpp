#include "Hardware/RTL/Providers/FloatMultiply.h"

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
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

enum class MultiplyFamily { Scalar, FixedVector };
using Format = detail::PortableFloatFormat;

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  Format format;
  mlir::arith::RoundingMode rounding;
  unsigned laneCount;
  std::string functionName;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_float_multiply_invalid: " + message);
}

llvm::Error unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
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

std::string coreName(const Format &format) {
  return "loom_float_multiply_e" + std::to_string(format.exponentBits) + "_f" +
         std::to_string(format.fractionBits) + "_core";
}

std::string modeName(const Format &format, mlir::arith::RoundingMode rounding) {
  return "loom_float_multiply_e" + std::to_string(format.exponentBits) + "_f" +
         std::to_string(format.fractionBits) + "_" +
         roundingSuffix(rounding).str();
}

llvm::Expected<LoweredMode>
lowerMode(MultiplyFamily family,
          const ::dataflow::CanonicalActorSchemaProjection &actor) {
  if (actor.schema != ::dataflow::OperationSchemaId::ArithMulF ||
      actor.type.getNumInputs() != 2 || actor.type.getNumResults() != 1)
    return invalid("behavior is not a binary floating multiply");
  if (actor.type.getInput(1) != actor.type.getInput(0) ||
      actor.type.getResult(0) != actor.type.getInput(0))
    return invalid("behavior does not have a uniform floating type");
  const auto *payload =
      std::get_if<::dataflow::FloatingPointPayload>(&actor.payload);
  if (!payload)
    return invalid("behavior has no floating-point payload");

  mlir::Type elementType = actor.type.getInput(0);
  unsigned laneCount = 1;
  if (family == MultiplyFamily::FixedVector) {
    auto vector = llvm::dyn_cast<mlir::VectorType>(elementType);
    if (!vector || vector.isScalable() || vector.getNumElements() == 0 ||
        vector.getNumElements() > std::numeric_limits<unsigned>::max())
      return invalid("behavior has an invalid fixed-vector shape");
    laneCount = static_cast<unsigned>(vector.getNumElements());
    elementType = vector.getElementType();
  } else if (llvm::isa<mlir::VectorType>(elementType)) {
    return invalid("scalar behavior uses a vector type");
  }

  auto format = detail::resolvePortableFloatFormat(elementType);
  if (!format)
    return invalid("behavior uses an unsupported floating format");
  const mlir::arith::RoundingMode rounding = payload->roundingMode.value_or(
      mlir::arith::RoundingMode::to_nearest_even);
  return LoweredMode{*format, rounding, laneCount, modeName(*format, rounding)};
}

std::string hexLiteral(unsigned width, std::uint64_t value) {
  std::ostringstream stream;
  stream << width << "'h" << std::hex << value;
  return stream.str();
}

std::string buildCoreFunction(const Format &format) {
  const unsigned width = format.width();
  const unsigned exponentBits = format.exponentBits;
  const unsigned fractionBits = format.fractionBits;
  const unsigned precision = format.precision();
  const unsigned productWidth = 2 * precision;
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponentBits) - 1;
  const std::uint64_t fractionMask = (std::uint64_t{1} << fractionBits) - 1;
  const std::uint64_t infinity = exponentMask << fractionBits;
  const std::uint64_t quietBit = std::uint64_t{1} << (fractionBits - 1);
  const std::uint64_t quietNaN = infinity | quietBit;
  const std::string name = coreName(format);
  const std::string exponentAllOnes = hexLiteral(exponentBits, exponentMask);
  const std::string exponentMaxFinite =
      hexLiteral(exponentBits, exponentMask - 1);
  const std::string fractionAllOnes = hexLiteral(fractionBits, fractionMask);

  std::string text;
  llvm::raw_string_ostream output(text);
  output
      << "function automatic [" << width - 1 << ":0] " << name << "(input ["
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
      << "  reg [" << productWidth - 1 << ":0] product;\n"
      << "  reg [" << productWidth - 1 << ":0] shifted_product;\n"
      << "  reg [" << precision << ":0] rounded;\n"
      << "  reg found;\n"
      << "  reg guard;\n"
      << "  reg sticky;\n"
      << "  reg increment;\n"
      << "  reg overflow_to_infinity;\n"
      << "  integer exponent_lhs_value;\n"
      << "  integer exponent_rhs_value;\n"
      << "  integer result_exponent_value;\n"
      << "  integer encoded_exponent;\n"
      << "  integer leading_index;\n"
      << "  integer shift_amount;\n"
      << "  integer index;\n"
      << "  begin\n"
      << "    " << name << " = " << hexLiteral(width, quietNaN) << ";\n"
      << "    sign_result = lhs[" << width - 1 << "] ^ rhs[" << width - 1
      << "];\n"
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
      << "    end else if (((exponent_lhs == " << exponentAllOnes
      << " && fraction_lhs == 0) &&\n"
      << "                  (exponent_rhs == 0 && fraction_rhs == 0)) ||\n"
      << "                 ((exponent_rhs == " << exponentAllOnes
      << " && fraction_rhs == 0) &&\n"
      << "                  (exponent_lhs == 0 && fraction_lhs == 0))) begin\n"
      << "      " << name << " = " << hexLiteral(width, quietNaN) << ";\n"
      << "    end else if ((exponent_lhs == " << exponentAllOnes
      << " && fraction_lhs == 0) ||\n"
      << "                 (exponent_rhs == " << exponentAllOnes
      << " && fraction_rhs == 0)) begin\n"
      << "      " << name << " = {sign_result, " << exponentAllOnes << ", "
      << fractionBits << "'d0};\n"
      << "    end else if ((exponent_lhs == 0 && fraction_lhs == 0) ||\n"
      << "                 (exponent_rhs == 0 && fraction_rhs == 0)) begin\n"
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
      << format.minimumExponent() << " : exponent_lhs_value - " << format.bias()
      << ";\n"
      << "      exponent_rhs_value = exponent_rhs == 0 ? "
      << format.minimumExponent() << " : exponent_rhs_value - " << format.bias()
      << ";\n"
      << "      product = significand_lhs * significand_rhs;\n"
      << "      leading_index = 0;\n"
      << "      found = 1'b0;\n"
      << "      for (index = " << productWidth - 1
      << "; index >= 0; index = index - 1) begin\n"
      << "        if (!found && product[index]) begin\n"
      << "          leading_index = index;\n"
      << "          found = 1'b1;\n"
      << "        end\n"
      << "      end\n"
      << "      result_exponent_value = exponent_lhs_value + "
         "exponent_rhs_value - "
      << 2 * fractionBits << " + leading_index;\n"
      << "      if (result_exponent_value >= " << format.minimumExponent()
      << ")\n"
      << "        shift_amount = leading_index - " << fractionBits << ";\n"
      << "      else\n"
      << "        shift_amount = " << format.minimumExponent() << " + "
      << fractionBits << " - exponent_lhs_value - exponent_rhs_value;\n"
      << "      shifted_product = " << productWidth << "'d0;\n"
      << "      rounded = " << precision + 1 << "'d0;\n"
      << "      guard = 1'b0;\n"
      << "      sticky = 1'b0;\n"
      << "      increment = 1'b0;\n"
      << "      if (shift_amount <= 0) begin\n"
      << "        shifted_product = product << (-shift_amount);\n"
      << "        rounded = shifted_product[" << precision << ":0];\n"
      << "      end else begin\n"
      << "        shifted_product = product >> shift_amount;\n"
      << "        rounded = shifted_product[" << precision << ":0];\n"
      << "        for (index = 0; index < " << productWidth
      << "; index = index + 1) begin\n"
      << "          if (index == shift_amount - 1) guard = product[index];\n"
      << "          if (index < shift_amount - 1)"
         " sticky = sticky | product[index];\n"
      << "        end\n"
      << "        if (shift_amount > " << productWidth
      << ") sticky = |product;\n"
      << "        case (rounding)\n"
      << "          3'd0: increment = guard && (sticky || rounded[0]);\n"
      << "          3'd1: increment = sign_result && (guard || sticky);\n"
      << "          3'd2: increment = !sign_result && (guard || sticky);\n"
      << "          3'd3: increment = 1'b0;\n"
      << "          3'd4: increment = guard;\n"
      << "          default: increment = guard && (sticky || rounded[0]);\n"
      << "        endcase\n"
      << "        rounded = rounded + increment;\n"
      << "      end\n"
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
      << "          fraction_result = rounded[" << fractionBits - 1 << ":0];\n"
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

std::string buildModeFunction(const LoweredMode &mode) {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic [" << mode.format.width() - 1 << ":0] "
         << mode.functionName << "(input [" << mode.format.width() - 1
         << ":0] lhs, input [" << mode.format.width() - 1 << ":0] rhs);\n"
         << "  begin\n"
         << "    " << mode.functionName << " = " << coreName(mode.format)
         << "(lhs, rhs, 3'd" << roundingCode(mode.rounding) << ");\n"
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
                            MultiplyFamily family, const LoweredMode &mode,
                            unsigned outputWidth) {
  const unsigned elementWidth = mode.format.width();
  if (family == MultiplyFamily::Scalar) {
    mlir::Value lhs = detail::resizeUnsigned(
        builder, location, accessor.getInput("data_input_0"), elementWidth);
    mlir::Value rhs = detail::resizeUnsigned(
        builder, location, accessor.getInput("data_input_1"), elementWidth);
    mlir::Value result =
        callMode(builder, location, mode.functionName, lhs, rhs, elementWidth);
    return detail::resizeUnsigned(builder, location, result, outputWidth);
  }

  std::vector<mlir::Value> laneResults;
  laneResults.reserve(mode.laneCount);
  for (unsigned lane = 0; lane < mode.laneCount; ++lane) {
    const unsigned lowBit = lane * elementWidth;
    mlir::Value lhs = circt::comb::ExtractOp::create(
        builder, location, accessor.getInput("data_input_0"), lowBit,
        elementWidth);
    mlir::Value rhs = circt::comb::ExtractOp::create(
        builder, location, accessor.getInput("data_input_1"), lowBit,
        elementWidth);
    laneResults.push_back(
        callMode(builder, location, mode.functionName, lhs, rhs, elementWidth));
  }
  mlir::Value packed = laneResults.front();
  if (laneResults.size() > 1) {
    std::vector<mlir::Value> highToLow(laneResults.rbegin(),
                                       laneResults.rend());
    packed = circt::comb::ConcatOp::create(builder, location, highToLow);
  }
  return detail::resizeUnsigned(builder, location, packed, outputWidth);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFloatMultiply(FabricOperationProviderRequest request,
                                 MultiplyFamily family) {
  const ::fabric::ImplementationFamilyId expectedFamily =
      family == MultiplyFamily::Scalar
          ? ::fabric::ImplementationFamilyId::ScalarFloatMultiply
          : ::fabric::ImplementationFamilyId::FixedVectorFloatMultiply;
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily != expectedFamily)
    return invalid("provider received a different implementation family");
  if (family == MultiplyFamily::Scalar) {
    if (!std::holds_alternative<::fabric::ScalarFloatParams>(
            request.capability.parameterizedCapability))
      return invalid("capability has the wrong scalar parameter schema");
  } else if (!std::holds_alternative<::fabric::FixedVectorFloatParams>(
                 request.capability.parameterizedCapability)) {
    return invalid("capability has the wrong fixed-vector parameter schema");
  }
  if (request.capability.enabledOperationSchemas !=
      std::vector<::dataflow::OperationSchemaId>{
          ::dataflow::OperationSchemaId::ArithMulF})
    return invalid("capability does not contain exactly arith.mulf");

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
  if (inputs.size() != 2 || outputs.size() != 1 ||
      inputs[0]->reference.ordinal != 0 || inputs[1]->reference.ordinal != 1 ||
      outputs[0]->reference.ordinal != 0 || inputs[0]->payloadWidthBits == 0 ||
      inputs[1]->payloadWidthBits == 0 || outputs[0]->payloadWidthBits == 0)
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
  const ConfigurationEncodingRelation *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  std::vector<Mode> modes;
  if (request.capability.configurationFieldSchema.empty()) {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None)
      return invalid("configuration-free multiply relation is not fieldless");
    if (domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid(
          "configuration-free capability has a non-singleton behavior domain");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid(
          "configured multiply semantic field relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured multiply capability requires one field");
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
      return invalid("sealed relation contains a duplicate multiply mode");
    const std::uint64_t activeWidth =
        static_cast<std::uint64_t>(lowered->format.width()) *
        lowered->laneCount;
    if (activeWidth > inputs[0]->payloadWidthBits ||
        activeWidth > inputs[1]->payloadWidthBits ||
        activeWidth > outputs[0]->payloadWidthBits)
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
          declarationStream << buildCoreFunction(mode.format) << '\n';
        }
        for (const LoweredMode &mode : loweredModes)
          declarationStream << buildModeFunction(mode) << '\n';
        circt::sv::VerbatimOp::create(
            bodyBuilder, location,
            bodyBuilder.getStringAttr(declarationStream.str()));

        std::vector<mlir::Value> results;
        results.reserve(loweredModes.size());
        for (const LoweredMode &mode : loweredModes)
          results.push_back(materializeMode(bodyBuilder, location, accessor,
                                            family, mode,
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
materializePortableScalarFloatMultiply(FabricOperationProviderRequest request) {
  return materializePortableFloatMultiply(std::move(request),
                                          MultiplyFamily::Scalar);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFixedVectorFloatMultiply(
    FabricOperationProviderRequest request) {
  return materializePortableFloatMultiply(std::move(request),
                                          MultiplyFamily::FixedVector);
}

} // namespace

llvm::Error registerPortableFloatMultiplyProviders(
    FabricOperationProviderRegistry &registry) {
  if (llvm::Error error =
          registry.add({::fabric::ImplementationFamilyId::ScalarFloatMultiply,
                        BackendRecipeKey::PortableSystemVerilog,
                        {},
                        materializePortableScalarFloatMultiply}))
    return error;
  return registry.add(
      {::fabric::ImplementationFamilyId::FixedVectorFloatMultiply,
       BackendRecipeKey::PortableSystemVerilog,
       {},
       materializePortableFixedVectorFloatMultiply});
}

} // namespace loom::hardware::rtl
