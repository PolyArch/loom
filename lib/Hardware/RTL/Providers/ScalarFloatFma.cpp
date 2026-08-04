#include "Hardware/RTL/Providers/ScalarFloatFma.h"

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
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct Format final {
  unsigned exponentBits;
  unsigned fractionBits;

  unsigned width() const { return 1 + exponentBits + fractionBits; }
  unsigned precision() const { return fractionBits + 1; }
  unsigned accumulatorWidth() const { return 2 * precision() + 4; }
  unsigned alignmentTop() const { return 2 * precision() + 2; }
  int bias() const { return (1 << (exponentBits - 1)) - 1; }
  int minimumExponent() const { return 1 - bias(); }
  int maximumExponent() const { return bias(); }
};

struct LoweredMode final {
  Format format;
  std::string functionName;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_scalar_float_fma_invalid: " +
                                     message);
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

llvm::Expected<LoweredMode> lowerMode(const Mode &mode) {
  if (mode.actor.schema != ::dataflow::OperationSchemaId::MathFma ||
      mode.actor.type.getNumInputs() != 3 ||
      mode.actor.type.getNumResults() != 1)
    return invalid("behavior is not a scalar floating FMA");
  mlir::Type type = mode.actor.type.getInput(0);
  if (mode.actor.type.getInput(1) != type ||
      mode.actor.type.getInput(2) != type ||
      mode.actor.type.getResult(0) != type)
    return invalid("behavior does not have a uniform floating type");
  const auto *payload =
      std::get_if<::dataflow::FloatingPointPayload>(&mode.actor.payload);
  if (!payload || payload->flags != mlir::arith::FastMathFlags::none ||
      (payload->roundingMode &&
       *payload->roundingMode != mlir::arith::RoundingMode::to_nearest_even))
    return invalid("behavior is outside the strict RNE floating profile");
  auto format = lowerFormat(type);
  if (!format)
    return format.takeError();
  return LoweredMode{*format, "loom_fma_e" +
                                  std::to_string(format->exponentBits) + "_f" +
                                  std::to_string(format->fractionBits)};
}

std::string hexLiteral(unsigned width, std::uint64_t value) {
  std::ostringstream stream;
  stream << width << "'h" << std::hex << value;
  return stream.str();
}

std::string buildFunction(const LoweredMode &mode) {
  const Format &format = mode.format;
  const unsigned width = format.width();
  const unsigned exponentBits = format.exponentBits;
  const unsigned fractionBits = format.fractionBits;
  const unsigned precision = format.precision();
  const unsigned productWidth = 2 * precision;
  const unsigned accumulatorWidth = format.accumulatorWidth();
  const unsigned alignmentTop = format.alignmentTop();
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponentBits) - 1;
  const std::uint64_t infinity = exponentMask << fractionBits;
  const std::uint64_t quietNaN =
      infinity | (std::uint64_t{1} << (fractionBits - 1));
  const std::string exponentAllOnes = hexLiteral(exponentBits, exponentMask);
  const std::string shiftName = mode.functionName + "_shr_jam";

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
      << "function automatic [" << width - 1 << ":0] " << mode.functionName
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
      << "    " << mode.functionName << " = " << hexLiteral(width, quietNaN)
      << ";\n"
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
      << "    if ((exponent_lhs == " << exponentAllOnes
      << " && fraction_lhs != 0) ||\n"
      << "        (exponent_rhs == " << exponentAllOnes
      << " && fraction_rhs != 0) ||\n"
      << "        (exponent_addend == " << exponentAllOnes
      << " && fraction_addend != 0)) begin\n"
      << "      " << mode.functionName << " = " << hexLiteral(width, quietNaN)
      << ";\n"
      << "    end else if (((exponent_lhs == " << exponentAllOnes
      << " && fraction_lhs == 0) &&\n"
      << "                 (exponent_rhs == 0 && fraction_rhs == 0)) ||\n"
      << "                ((exponent_rhs == " << exponentAllOnes
      << " && fraction_rhs == 0) &&\n"
      << "                 (exponent_lhs == 0 && fraction_lhs == 0))) begin\n"
      << "      " << mode.functionName << " = " << hexLiteral(width, quietNaN)
      << ";\n"
      << "    end else if ((exponent_lhs == " << exponentAllOnes
      << " && fraction_lhs == 0) ||\n"
      << "                 (exponent_rhs == " << exponentAllOnes
      << " && fraction_rhs == 0)) begin\n"
      << "      if (exponent_addend == " << exponentAllOnes
      << " && fraction_addend == 0 && sign_addend != sign_product)\n"
      << "        " << mode.functionName << " = " << hexLiteral(width, quietNaN)
      << ";\n"
      << "      else\n"
      << "        " << mode.functionName << " = {sign_product, "
      << exponentAllOnes << ", " << fractionBits << "'d0};\n"
      << "    end else if (exponent_addend == " << exponentAllOnes
      << " && fraction_addend == 0) begin\n"
      << "      " << mode.functionName << " = addend;\n"
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
      << "          " << mode.functionName << " = addend;\n"
      << "        else begin\n"
      << "          sign_result = sign_product == sign_addend"
         " ? sign_product : 1'b0;\n"
      << "          " << mode.functionName << " = {sign_result, "
      << exponentBits << "'d0, " << fractionBits << "'d0};\n"
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
      << "          " << mode.functionName << " = {sign_result, "
      << exponentBits << "'d0, " << fractionBits << "'d0};\n"
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
      << "              " << mode.functionName << " = {sign_result, "
      << exponentAllOnes << ", " << fractionBits << "'d0};\n"
      << "            end else begin\n"
      << "              encoded_exponent = result_exponent_value + "
      << format.bias() << ";\n"
      << "              exponent_result = encoded_exponent[" << exponentBits - 1
      << ":0];\n"
      << "              fraction_result = rounded[" << fractionBits - 1
      << ":0];\n"
      << "              " << mode.functionName
      << " = {sign_result, exponent_result, fraction_result};\n"
      << "            end\n"
      << "          end else if (rounded[" << fractionBits << "]) begin\n"
      << "            " << mode.functionName << " = {sign_result, "
      << exponentBits << "'d1, " << fractionBits << "'d0};\n"
      << "          end else begin\n"
      << "            fraction_result = rounded[" << fractionBits - 1
      << ":0];\n"
      << "            " << mode.functionName << " = {sign_result, "
      << exponentBits << "'d0, fraction_result};\n"
      << "          end\n"
      << "        end\n"
      << "      end\n"
      << "    end\n"
      << "  end\n"
      << "endfunction\n";
  return output.str();
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarFloatFma(FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::ScalarFloatFma)
    return invalid("provider received a different implementation family");
  if (!std::holds_alternative<::fabric::ScalarFloatParams>(
          request.capability.parameterizedCapability))
    return invalid("capability has the wrong parameter schema");
  if (request.capability.enabledOperationSchemas !=
      std::vector<::dataflow::OperationSchemaId>{
          ::dataflow::OperationSchemaId::MathFma})
    return invalid("capability does not contain exactly math.fma");

  auto actualContract = ::fabric::encodeResourceContractRecord(
      request.capability.resourceStateAndTimingContract);
  if (!actualContract)
    return actualContract.takeError();
  auto supportedContract = ::fabric::encodeResourceContractRecord(
      ::fabric::oneCycleElasticOperationResourceContract());
  if (!supportedContract)
    return supportedContract.takeError();
  if (*actualContract != *supportedContract)
    return invalid("resource contract is not the supported one-cycle elastic "
                   "contract");

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
  if (inputs.size() != 3 || outputs.size() != 1 ||
      inputs[0]->reference.ordinal != 0 || inputs[1]->reference.ordinal != 1 ||
      inputs[2]->reference.ordinal != 2 || outputs[0]->reference.ordinal != 0 ||
      inputs[0]->payloadWidthBits == 0 || inputs[1]->payloadWidthBits == 0 ||
      inputs[2]->payloadWidthBits == 0 || outputs[0]->payloadWidthBits == 0)
    return invalid("capability does not have the ternary floating port shape");
  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.capability, request.configurationAbi))
    return std::move(error);

  auto domain = request.capability.resolveFiniteBehaviorDomain(
      *request.leaf.getContext());
  if (!domain)
    return domain.takeError();
  const ConfigurationFieldEncoding *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  std::vector<Mode> modes;
  if (request.capability.configurationFieldSchema.empty()) {
    if (domain->size() != 1 || domain->front().semanticConfiguration)
      return invalid(
          "configuration-free capability has a non-singleton behavior domain");
    modes.push_back({std::move(domain->front().representativeActor), nullptr});
  } else {
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured scalar FMA capability requires one field");
    field = request.configurationAbi.findField(
        request.capability.configurationFieldSchema.front());
    if (!field)
      return invalid("configured field is absent from the ABI");
    codebook = std::get_if<FiniteCodebookEncoding>(&field->semanticEncoding);
    if (!codebook)
      return invalid("configured field is not a finite codebook");
    if (codebook->entries.size() != domain->size())
      return invalid(
          "codebook does not exactly cover the configuration domain");
    modes.reserve(domain->size());
    for (auto &point : *domain) {
      if (!point.semanticConfiguration)
        return invalid("configured behavior has no semantic value");
      const FiniteCodebookEntry *entry = detail::findFiniteCodebookEntry(
          *codebook, point.semanticConfiguration->bytes());
      if (!entry)
        return invalid("codebook has no entry for an admitted semantic value");
      modes.push_back({std::move(point.representativeActor), entry});
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
  loweredModes.reserve(modes.size());
  for (const Mode &mode : modes) {
    auto lowered = lowerMode(mode);
    if (!lowered)
      return lowered.takeError();
    const unsigned width = lowered->format.width();
    if (llvm::any_of(
            inputs,
            [=](const auto *port) { return port->payloadWidthBits < width; }) ||
        outputs[0]->payloadWidthBits < width)
      return invalid("behavior exceeds the physical datapath");
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
          declarationStream << buildFunction(mode) << '\n';
        circt::sv::VerbatimOp::create(
            bodyBuilder, location,
            bodyBuilder.getStringAttr(declarationStream.str()));

        std::vector<mlir::Value> results;
        results.reserve(loweredModes.size());
        for (const LoweredMode &mode : loweredModes) {
          const unsigned width = mode.format.width();
          llvm::SmallVector<mlir::Value, 3> arguments;
          for (unsigned index = 0; index < 3; ++index)
            arguments.push_back(detail::resizeUnsigned(
                bodyBuilder, location,
                accessor.getInput("data_input_" + std::to_string(index)),
                width));
          mlir::Value value = circt::sv::VerbatimExprOp::create(
              bodyBuilder, location, bodyBuilder.getIntegerType(width),
              mode.functionName + "({{0}}, {{1}}, {{2}})", arguments);
          results.push_back(detail::resizeUnsigned(
              bodyBuilder, location, value, outputs[0]->payloadWidthBits));
        }

        mlir::Value result = results[inactiveMode];
        if (field) {
          mlir::Value configuration = accessor.getInput(
              "config_" + std::to_string(field->field.ordinal));
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

llvm::Error registerPortableScalarFloatFmaProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::ScalarFloatFma,
                       BackendRecipeKey::PortableSystemVerilog,
                       {},
                       materializePortableScalarFloatFma});
}

} // namespace loom::hardware::rtl
