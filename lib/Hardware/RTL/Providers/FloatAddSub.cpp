#include "Hardware/RTL/Providers/FloatAddSub.h"

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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

using Family = ::fabric::ImplementationFamilyId;
using Schema = ::dataflow::OperationSchemaId;

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct Format final {
  unsigned exponentBits;
  unsigned fractionBits;

  unsigned width() const { return 1 + exponentBits + fractionBits; }
  unsigned precision() const { return fractionBits + 1; }
  unsigned accumulatorWidth() const { return precision() + 4; }
  int bias() const { return (1 << (exponentBits - 1)) - 1; }
  int minimumExponent() const { return 1 - bias(); }
  int maximumExponent() const { return bias(); }
};

struct LoweredMode final {
  Format format;
  mlir::arith::RoundingMode rounding;
  bool subtract;
  unsigned laneCount;
  std::string functionName;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_float_add_sub_invalid: " + message);
}

llvm::Expected<Format> lowerFormat(mlir::Type type) {
  if (mlir::isa<mlir::Float16Type>(type))
    return Format{5, 10};
  if (mlir::isa<mlir::BFloat16Type>(type))
    return Format{8, 7};
  if (mlir::isa<mlir::Float32Type>(type))
    return Format{8, 23};
  if (mlir::isa<mlir::Float64Type>(type))
    return Format{11, 52};
  return invalid("behavior uses an unsupported floating format");
}

llvm::Expected<LoweredMode> lowerMode(const Mode &mode, bool vectorFamily) {
  if (mode.actor.schema != Schema::ArithAddF &&
      mode.actor.schema != Schema::ArithSubF)
    return invalid("behavior has a non-add/sub schema");
  if (mode.actor.type.getNumInputs() != 2 ||
      mode.actor.type.getNumResults() != 1)
    return invalid("behavior has wrong arity");
  mlir::Type type = mode.actor.type.getInput(0);
  if (mode.actor.type.getInput(1) != type ||
      mode.actor.type.getResult(0) != type)
    return invalid("behavior does not have a uniform floating type");
  const auto *payload =
      std::get_if<::dataflow::FloatingPointPayload>(&mode.actor.payload);
  if (!payload)
    return invalid("behavior has no floating payload");
  const mlir::arith::RoundingMode rounding = payload->roundingMode.value_or(
      mlir::arith::RoundingMode::to_nearest_even);
  if (static_cast<std::uint32_t>(rounding) >
      mlir::arith::getMaxEnumValForRoundingMode())
    return invalid("behavior has an unknown rounding mode");

  unsigned laneCount = 1;
  mlir::Type element = type;
  if (vectorFamily) {
    auto vector = mlir::dyn_cast<mlir::VectorType>(type);
    if (!vector)
      return invalid("vector behavior does not have a vector type");
    const std::uint64_t lanes = vector.getNumElements();
    if (lanes == 0 || lanes > std::numeric_limits<unsigned>::max())
      return invalid("behavior lane count is outside the RTL domain");
    laneCount = static_cast<unsigned>(lanes);
    element = vector.getElementType();
  } else if (mlir::isa<mlir::VectorType>(type)) {
    return invalid("scalar behavior has a vector type");
  }
  auto format = lowerFormat(element);
  if (!format)
    return format.takeError();
  return LoweredMode{
      *format, rounding, mode.actor.schema == Schema::ArithSubF, laneCount,
      "loom_float_add_sub_e" + std::to_string(format->exponentBits) + "_f" +
          std::to_string(format->fractionBits)};
}

std::string hexLiteral(unsigned width, std::uint64_t value) {
  return std::to_string(width) + "'h" + llvm::utohexstr(value);
}

std::string buildFunction(const Format &format, llvm::StringRef functionName) {
  const unsigned width = format.width();
  const unsigned exponentBits = format.exponentBits;
  const unsigned fractionBits = format.fractionBits;
  const unsigned precision = format.precision();
  const unsigned accumulatorWidth = format.accumulatorWidth();
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponentBits) - 1;
  const std::uint64_t infinity = exponentMask << fractionBits;
  const std::uint64_t quietBit = std::uint64_t{1} << (fractionBits - 1);
  const std::uint64_t quietNaN = infinity | quietBit;
  const std::uint64_t maximumFinite = infinity - 1;
  const std::string shiftName = functionName.str() + "_shr_jam";

  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic [" << accumulatorWidth - 1 << ":0] "
         << shiftName << "(input [" << accumulatorWidth - 1
         << ":0] value, input integer distance);\n"
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

  output << "function automatic [" << width - 1 << ":0] " << functionName
         << "(input [" << width - 1 << ":0] lhs, input [" << width - 1
         << ":0] rhs, input subtract, input [2:0] rounding);\n"
         << "  reg sign_lhs;\n"
         << "  reg sign_rhs;\n"
         << "  reg sign_rhs_effective;\n"
         << "  reg sign_result;\n"
         << "  reg [" << exponentBits - 1 << ":0] exponent_lhs;\n"
         << "  reg [" << exponentBits - 1 << ":0] exponent_rhs;\n"
         << "  reg [" << exponentBits - 1 << ":0] exponent_result;\n"
         << "  reg [" << fractionBits - 1 << ":0] fraction_lhs;\n"
         << "  reg [" << fractionBits - 1 << ":0] fraction_rhs;\n"
         << "  reg [" << fractionBits - 1 << ":0] fraction_result;\n"
         << "  reg [" << precision - 1 << ":0] significand_lhs;\n"
         << "  reg [" << precision - 1 << ":0] significand_rhs;\n"
         << "  reg [" << accumulatorWidth - 1 << ":0] lhs_value;\n"
         << "  reg [" << accumulatorWidth - 1 << ":0] rhs_value;\n"
         << "  reg [" << accumulatorWidth - 1 << ":0] aligned_lhs;\n"
         << "  reg [" << accumulatorWidth - 1 << ":0] aligned_rhs;\n"
         << "  reg [" << accumulatorWidth - 1 << ":0] magnitude;\n"
         << "  reg [" << accumulatorWidth - 1 << ":0] shifted_magnitude;\n"
         << "  reg [" << precision << ":0] rounded;\n"
         << "  reg found;\n"
         << "  reg guard;\n"
         << "  reg sticky;\n"
         << "  reg discarded;\n"
         << "  reg increment;\n"
         << "  reg overflow_to_infinity;\n"
         << "  integer exponent_lhs_value;\n"
         << "  integer exponent_rhs_value;\n"
         << "  integer common_exponent;\n"
         << "  integer result_exponent_value;\n"
         << "  integer encoded_exponent;\n"
         << "  integer shift_amount;\n"
         << "  integer leading_index;\n"
         << "  integer index;\n"
         << "  begin\n"
         << "    " << functionName << " = " << hexLiteral(width, quietNaN)
         << ";\n"
         << "    sign_lhs = lhs[" << width - 1 << "];\n"
         << "    sign_rhs = rhs[" << width - 1 << "];\n"
         << "    sign_rhs_effective = sign_rhs ^ subtract;\n"
         << "    exponent_lhs = lhs[" << width - 2 << ':' << fractionBits
         << "];\n"
         << "    exponent_rhs = rhs[" << width - 2 << ':' << fractionBits
         << "];\n"
         << "    fraction_lhs = lhs[" << fractionBits - 1 << ":0];\n"
         << "    fraction_rhs = rhs[" << fractionBits - 1 << ":0];\n"
         << "    if (exponent_lhs == " << hexLiteral(exponentBits, exponentMask)
         << " && fraction_lhs != 0) begin\n"
         << "      " << functionName << " = lhs | "
         << hexLiteral(width, quietBit) << ";\n"
         << "    end else if (exponent_rhs == "
         << hexLiteral(exponentBits, exponentMask)
         << " && fraction_rhs != 0) begin\n"
         << "      " << functionName << " = rhs | "
         << hexLiteral(width, quietBit) << ";\n"
         << "    end else if (exponent_lhs == "
         << hexLiteral(exponentBits, exponentMask)
         << " && fraction_lhs == 0 && exponent_rhs == "
         << hexLiteral(exponentBits, exponentMask)
         << " && fraction_rhs == 0 && sign_lhs != sign_rhs_effective) begin\n"
         << "      " << functionName << " = " << hexLiteral(width, quietNaN)
         << ";\n"
         << "    end else if (exponent_lhs == "
         << hexLiteral(exponentBits, exponentMask)
         << " && fraction_lhs == 0) begin\n"
         << "      " << functionName << " = {sign_lhs, "
         << hexLiteral(exponentBits, exponentMask) << ", " << fractionBits
         << "'d0};\n"
         << "    end else if (exponent_rhs == "
         << hexLiteral(exponentBits, exponentMask)
         << " && fraction_rhs == 0) begin\n"
         << "      " << functionName << " = {sign_rhs_effective, "
         << hexLiteral(exponentBits, exponentMask) << ", " << fractionBits
         << "'d0};\n"
         << "    end else begin\n"
         << "      significand_lhs = exponent_lhs == 0"
            " ? {1'b0, fraction_lhs} : {1'b1, fraction_lhs};\n"
         << "      significand_rhs = exponent_rhs == 0"
            " ? {1'b0, fraction_rhs} : {1'b1, fraction_rhs};\n"
         << "      exponent_lhs_value = exponent_lhs == 0 ? "
         << format.minimumExponent() << " : integer'(exponent_lhs) - "
         << format.bias() << ";\n"
         << "      exponent_rhs_value = exponent_rhs == 0 ? "
         << format.minimumExponent() << " : integer'(exponent_rhs) - "
         << format.bias() << ";\n"
         << "      lhs_value = " << accumulatorWidth << "'d0;\n"
         << "      rhs_value = " << accumulatorWidth << "'d0;\n"
         << "      lhs_value[" << precision - 1 << ":0] = significand_lhs;\n"
         << "      rhs_value[" << precision - 1 << ":0] = significand_rhs;\n"
         << "      lhs_value = lhs_value << 3;\n"
         << "      rhs_value = rhs_value << 3;\n"
         << "      if (exponent_lhs_value >= exponent_rhs_value) begin\n"
         << "        common_exponent = exponent_lhs_value;\n"
         << "        aligned_lhs = lhs_value;\n"
         << "        aligned_rhs = " << shiftName
         << "(rhs_value, exponent_lhs_value - exponent_rhs_value);\n"
         << "      end else begin\n"
         << "        common_exponent = exponent_rhs_value;\n"
         << "        aligned_lhs = " << shiftName
         << "(lhs_value, exponent_rhs_value - exponent_lhs_value);\n"
         << "        aligned_rhs = rhs_value;\n"
         << "      end\n"
         << "      if (sign_lhs == sign_rhs_effective) begin\n"
         << "        magnitude = aligned_lhs + aligned_rhs;\n"
         << "        sign_result = sign_lhs;\n"
         << "      end else if (aligned_lhs > aligned_rhs) begin\n"
         << "        magnitude = aligned_lhs - aligned_rhs;\n"
         << "        sign_result = sign_lhs;\n"
         << "      end else if (aligned_rhs > aligned_lhs) begin\n"
         << "        magnitude = aligned_rhs - aligned_lhs;\n"
         << "        sign_result = sign_rhs_effective;\n"
         << "      end else begin\n"
         << "        magnitude = " << accumulatorWidth << "'d0;\n"
         << "        sign_result = rounding == 3'd1;\n"
         << "      end\n"
         << "      if (magnitude == 0) begin\n"
         << "        " << functionName << " = {sign_result, " << exponentBits
         << "'d0, " << fractionBits << "'d0};\n"
         << "      end else begin\n"
         << "        leading_index = 0;\n"
         << "        found = 1'b0;\n"
         << "        for (index = " << accumulatorWidth - 1
         << "; index >= 0; index = index - 1) begin\n"
         << "          if (!found && magnitude[index]) begin\n"
         << "            leading_index = index;\n"
         << "            found = 1'b1;\n"
         << "          end\n"
         << "        end\n"
         << "        result_exponent_value = common_exponent + leading_index - "
         << fractionBits + 3 << ";\n"
         << "        if (result_exponent_value >= " << format.minimumExponent()
         << ")\n"
         << "          shift_amount = leading_index - " << fractionBits << ";\n"
         << "        else\n"
         << "          shift_amount = " << format.minimumExponent()
         << " + 3 - common_exponent;\n"
         << "        rounded = " << precision + 1 << "'d0;\n"
         << "        shifted_magnitude = " << accumulatorWidth << "'d0;\n"
         << "        guard = 1'b0;\n"
         << "        sticky = 1'b0;\n"
         << "        if (shift_amount <= 0) begin\n"
         << "          shifted_magnitude = magnitude << (-shift_amount);\n"
         << "        end else begin\n"
         << "          shifted_magnitude = magnitude >> shift_amount;\n"
         << "          for (index = 0; index < " << accumulatorWidth
         << "; index = index + 1) begin\n"
         << "            if (index == shift_amount - 1)"
            " guard = magnitude[index];\n"
         << "            if (index < shift_amount - 1)"
            " sticky = sticky | magnitude[index];\n"
         << "          end\n"
         << "          if (shift_amount > " << accumulatorWidth
         << ") sticky = |magnitude;\n"
         << "        end\n"
         << "        rounded = shifted_magnitude[" << precision << ":0];\n"
         << "        discarded = guard | sticky;\n"
         << "        case (rounding)\n"
         << "          3'd0: increment = guard && (sticky || rounded[0]);\n"
         << "          3'd1: increment = sign_result && discarded;\n"
         << "          3'd2: increment = !sign_result && discarded;\n"
         << "          3'd3: increment = 1'b0;\n"
         << "          3'd4: increment = guard;\n"
         << "          default: increment = guard && (sticky || rounded[0]);\n"
         << "        endcase\n"
         << "        rounded = rounded + increment;\n"
         << "        if (result_exponent_value >= " << format.minimumExponent()
         << ") begin\n"
         << "          if (rounded[" << precision << "]) begin\n"
         << "            rounded = rounded >> 1;\n"
         << "            result_exponent_value = result_exponent_value + 1;\n"
         << "          end\n"
         << "          if (result_exponent_value > " << format.maximumExponent()
         << ") begin\n"
         << "            overflow_to_infinity ="
            " rounding == 3'd0 || rounding == 3'd4 ||\n"
         << "                (rounding == 3'd1 && sign_result) ||\n"
         << "                (rounding == 3'd2 && !sign_result);\n"
         << "            if (overflow_to_infinity)\n"
         << "              " << functionName << " = {sign_result, "
         << hexLiteral(exponentBits, exponentMask) << ", " << fractionBits
         << "'d0};\n"
         << "            else\n"
         << "              " << functionName << " = {sign_result, "
         << hexLiteral(width - 1, maximumFinite) << "};\n"
         << "          end else begin\n"
         << "            encoded_exponent = result_exponent_value + "
         << format.bias() << ";\n"
         << "            exponent_result = encoded_exponent["
         << exponentBits - 1 << ":0];\n"
         << "            fraction_result = rounded[" << fractionBits - 1
         << ":0];\n"
         << "            " << functionName
         << " = {sign_result, exponent_result, fraction_result};\n"
         << "          end\n"
         << "        end else if (rounded[" << fractionBits << "]) begin\n"
         << "          " << functionName << " = {sign_result, " << exponentBits
         << "'d1, " << fractionBits << "'d0};\n"
         << "        end else begin\n"
         << "          fraction_result = rounded[" << fractionBits - 1
         << ":0];\n"
         << "          " << functionName << " = {sign_result, " << exponentBits
         << "'d0, fraction_result};\n"
         << "        end\n"
         << "      end\n"
         << "    end\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

bool isSupportedSchemaSet(llvm::ArrayRef<Schema> schemas) {
  if (schemas.empty() || schemas.size() > 2)
    return false;
  bool sawAdd = false;
  bool sawSubtract = false;
  for (Schema schema : schemas) {
    if (schema == Schema::ArithAddF) {
      if (sawAdd)
        return false;
      sawAdd = true;
    } else if (schema == Schema::ArithSubF) {
      if (sawSubtract)
        return false;
      sawSubtract = true;
    } else {
      return false;
    }
  }
  return true;
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFloatAddSub(FabricOperationProviderRequest request,
                               Family expectedFamily, bool vectorFamily) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily != expectedFamily)
    return invalid("provider received a different implementation family");
  if (vectorFamily) {
    if (!std::holds_alternative<::fabric::FixedVectorFloatParams>(
            request.capability.parameterizedCapability))
      return invalid("capability has the wrong vector parameter schema");
  } else if (!std::holds_alternative<::fabric::ScalarFloatParams>(
                 request.capability.parameterizedCapability)) {
    return invalid("capability has the wrong scalar parameter schema");
  }
  if (!isSupportedSchemaSet(request.capability.enabledOperationSchemas))
    return invalid("capability has an invalid add/sub schema set");

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
        request.capability.implementationFamily, request.recipe);

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
    return invalid("capability does not have the binary floating port shape");
  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return std::move(error);

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
      return invalid(
          "configuration-free capability has a non-singleton domain");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid("configured semantic relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured float add/sub capability requires one field");
    field = request.configurationAbi.findOperationField(
        request.occurrence,
        request.capability.configurationFieldSchema.front().ordinal);
    if (!field)
      return invalid("configured field is absent from the ABI");
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
      return invalid("ABI inactive value is outside the sealed domain");
    inactiveMode = static_cast<std::size_t>(inactive - modes.begin());
  }

  std::vector<LoweredMode> loweredModes;
  loweredModes.reserve(modes.size());
  for (const Mode &mode : modes) {
    auto lowered = lowerMode(mode, vectorFamily);
    if (!lowered)
      return lowered.takeError();
    const std::uint64_t payloadWidth =
        static_cast<std::uint64_t>(lowered->format.width()) *
        lowered->laneCount;
    if (payloadWidth > inputs[0]->payloadWidthBits ||
        payloadWidth > inputs[1]->payloadWidthBits ||
        payloadWidth > outputs[0]->payloadWidthBits)
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
          declarationStream << buildFunction(mode.format, mode.functionName)
                            << '\n';
        }
        circt::sv::VerbatimOp::create(
            bodyBuilder, location,
            bodyBuilder.getStringAttr(declarationStream.str()));

        std::vector<mlir::Value> selectedModes(modes.size());
        mlir::Value subtract = circt::hw::ConstantOp::create(
            bodyBuilder, location,
            llvm::APInt(1, loweredModes[inactiveMode].subtract));
        mlir::Value rounding = circt::hw::ConstantOp::create(
            bodyBuilder, location,
            llvm::APInt(3, static_cast<std::uint32_t>(
                               loweredModes[inactiveMode].rounding)));
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
            mlir::Value modeSubtract = circt::hw::ConstantOp::create(
                bodyBuilder, location,
                llvm::APInt(1, loweredModes[index].subtract));
            subtract = circt::comb::MuxOp::create(bodyBuilder, location,
                                                  selectedModes[index],
                                                  modeSubtract, subtract, true);
            mlir::Value modeRounding = circt::hw::ConstantOp::create(
                bodyBuilder, location,
                llvm::APInt(3, static_cast<std::uint32_t>(
                                   loweredModes[index].rounding)));
            rounding = circt::comb::MuxOp::create(bodyBuilder, location,
                                                  selectedModes[index],
                                                  modeRounding, rounding, true);
          }
        }

        struct MaterializedGeometry final {
          std::string functionName;
          unsigned laneCount;
          mlir::Value result;
        };
        std::vector<MaterializedGeometry> geometries;
        std::vector<mlir::Value> results;
        results.reserve(loweredModes.size());
        for (const LoweredMode &mode : loweredModes) {
          const auto existing = llvm::find_if(
              geometries, [&](const MaterializedGeometry &geometry) {
                return geometry.functionName == mode.functionName &&
                       geometry.laneCount == mode.laneCount;
              });
          if (existing != geometries.end()) {
            results.push_back(existing->result);
            continue;
          }
          const unsigned width = mode.format.width();
          std::vector<mlir::Value> lanes;
          lanes.reserve(mode.laneCount);
          for (unsigned lane = 0; lane < mode.laneCount; ++lane) {
            const unsigned lowBit = lane * width;
            llvm::SmallVector<mlir::Value, 4> arguments;
            for (unsigned input = 0; input < 2; ++input) {
              mlir::Value value =
                  accessor.getInput("data_input_" + std::to_string(input));
              if (vectorFamily)
                value = circt::comb::ExtractOp::create(bodyBuilder, location,
                                                       value, lowBit, width);
              else
                value =
                    detail::resizeUnsigned(bodyBuilder, location, value, width);
              arguments.push_back(value);
            }
            arguments.push_back(subtract);
            arguments.push_back(rounding);
            lanes.push_back(circt::sv::VerbatimExprOp::create(
                bodyBuilder, location, bodyBuilder.getIntegerType(width),
                mode.functionName + "({{0}}, {{1}}, {{2}}, {{3}})", arguments));
          }
          mlir::Value packed = lanes.front();
          if (lanes.size() > 1) {
            std::vector<mlir::Value> highToLow(lanes.rbegin(), lanes.rend());
            packed =
                circt::comb::ConcatOp::create(bodyBuilder, location, highToLow);
          }
          mlir::Value result = detail::resizeUnsigned(
              bodyBuilder, location, packed, outputs[0]->payloadWidthBits);
          geometries.push_back({mode.functionName, mode.laneCount, result});
          results.push_back(result);
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

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarFloatAddSub(FabricOperationProviderRequest request) {
  return materializePortableFloatAddSub(request, Family::ScalarFloatAddSub,
                                        false);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFixedVectorFloatAddSub(
    FabricOperationProviderRequest request) {
  return materializePortableFloatAddSub(request, Family::FixedVectorFloatAddSub,
                                        true);
}

} // namespace

llvm::Error registerPortableFloatAddSubProviders(
    FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry candidate = registry;
  if (llvm::Error error =
          candidate.add({Family::ScalarFloatAddSub,
                         BackendRecipeKey::PortableSystemVerilog,
                         {},
                         materializePortableScalarFloatAddSub}))
    return error;
  if (llvm::Error error =
          candidate.add({Family::FixedVectorFloatAddSub,
                         BackendRecipeKey::PortableSystemVerilog,
                         {},
                         materializePortableFixedVectorFloatAddSub}))
    return error;
  registry = std::move(candidate);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl
