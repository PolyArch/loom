#include "Hardware/RTL/Providers/MathErf.h"

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

#include <array>
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

using Format = detail::PortableFloatFormat;

constexpr std::array<std::uint32_t, 129> kErfQ30 = {
    UINT32_C(0x00000000), UINT32_C(0x02418ac9), UINT32_C(0x0481f520),
    UINT32_C(0x06c02045), UINT32_C(0x08faf0d2), UINT32_C(0x0b31505f),
    UINT32_C(0x0d622f1c), UINT32_C(0x0f8c8554), UINT32_C(0x11af54e2),
    UINT32_C(0x13c9aa8c), UINT32_C(0x15da9f41), UINT32_C(0x17e15945),
    UINT32_C(0x19dd0d2b), UINT32_C(0x1bccfec2), UINT32_C(0x1db081ce),
    UINT32_C(0x1f86faa9), UINT32_C(0x214fdeb8), UINT32_C(0x230ab4c0),
    UINT32_C(0x24b71511), UINT32_C(0x2654a997), UINT32_C(0x27e32dba),
    UINT32_C(0x29626e27), UINT32_C(0x2ad2487a), UINT32_C(0x2c32aac1),
    UINT32_C(0x2d8392eb), UINT32_C(0x2ec50e1f), UINT32_C(0x2ff737f6),
    UINT32_C(0x311a39a9), UINT32_C(0x322e492a), UINT32_C(0x3333a832),
    UINT32_C(0x342aa343), UINT32_C(0x351390a1), UINT32_C(0x35eecf4f),
    UINT32_C(0x36bcc5fa), UINT32_C(0x377de1f8), UINT32_C(0x3832963b),
    UINT32_C(0x38db5a50), UINT32_C(0x3978a969), UINT32_C(0x3a0b0166),
    UINT32_C(0x3a92e1f4), UINT32_C(0x3b10cbb3), UINT32_C(0x3b853f6c),
    UINT32_C(0x3bf0bd52), UINT32_C(0x3c53c455), UINT32_C(0x3caed187),
    UINT32_C(0x3d025f8d), UINT32_C(0x3d4ee61e), UINT32_C(0x3d94d99b),
    UINT32_C(0x3dd4aaae), UINT32_C(0x3e0ec5fc), UINT32_C(0x3e4393e2),
    UINT32_C(0x3e737848), UINT32_C(0x3e9ed277), UINT32_C(0x3ec5fd00),
    UINT32_C(0x3ee94db3), UINT32_C(0x3f091597), UINT32_C(0x3f25a0f0),
    UINT32_C(0x3f3f3752), UINT32_C(0x3f561bb1), UINT32_C(0x3f6a8c83),
    UINT32_C(0x3f7cc3de), UINT32_C(0x3f8cf79e), UINT32_C(0x3f9b5994),
    UINT32_C(0x3fa817ae), UINT32_C(0x3fb35c28), UINT32_C(0x3fbd4dc1),
    UINT32_C(0x3fc60fe5), UINT32_C(0x3fcdc2e8), UINT32_C(0x3fd48432),
    UINT32_C(0x3fda6e71), UINT32_C(0x3fdf99cd), UINT32_C(0x3fe41c15),
    UINT32_C(0x3fe808ec), UINT32_C(0x3feb71f6), UINT32_C(0x3fee6703),
    UINT32_C(0x3ff0f632), UINT32_C(0x3ff32c1e), UINT32_C(0x3ff513fd),
    UINT32_C(0x3ff6b7bf), UINT32_C(0x3ff82033), UINT32_C(0x3ff9551f),
    UINT32_C(0x3ffa5d5c), UINT32_C(0x3ffb3ef2), UINT32_C(0x3ffbff26),
    UINT32_C(0x3ffca297), UINT32_C(0x3ffd2d4f), UINT32_C(0x3ffda2cf),
    UINT32_C(0x3ffe0625), UINT32_C(0x3ffe59f6), UINT32_C(0x3ffea08b),
    UINT32_C(0x3ffedbdd), UINT32_C(0x3fff0da0), UINT32_C(0x3fff3748),
    UINT32_C(0x3fff5a17), UINT32_C(0x3fff771e), UINT32_C(0x3fff8f47),
    UINT32_C(0x3fffa359), UINT32_C(0x3fffb3fc), UINT32_C(0x3fffc1c0),
    UINT32_C(0x3fffcd1f), UINT32_C(0x3fffd67d), UINT32_C(0x3fffde33),
    UINT32_C(0x3fffe487), UINT32_C(0x3fffe9b6), UINT32_C(0x3fffedf4),
    UINT32_C(0x3ffff16a), UINT32_C(0x3ffff43c), UINT32_C(0x3ffff687),
    UINT32_C(0x3ffff863), UINT32_C(0x3ffff9e4), UINT32_C(0x3ffffb1c),
    UINT32_C(0x3ffffc18), UINT32_C(0x3ffffce2), UINT32_C(0x3ffffd85),
    UINT32_C(0x3ffffe07), UINT32_C(0x3ffffe70), UINT32_C(0x3ffffec3),
    UINT32_C(0x3fffff06), UINT32_C(0x3fffff3b), UINT32_C(0x3fffff65),
    UINT32_C(0x3fffff86), UINT32_C(0x3fffffa0), UINT32_C(0x3fffffb5),
    UINT32_C(0x3fffffc5), UINT32_C(0x3fffffd2), UINT32_C(0x3fffffdc),
    UINT32_C(0x3fffffe4), UINT32_C(0x3fffffeb), UINT32_C(0x3fffffef),
};

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
                                 "portable_math_erf_invalid: " + message);
}

llvm::Expected<FabricOperationProviderOutput>
unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

bool supportsBehavior(const ::fabric::FloatBehaviorProfile &behavior) {
  return behavior.roundingModes.size() == 1 &&
         behavior.roundingModes.contains(
             mlir::arith::RoundingMode::to_nearest_even) &&
         behavior.nanBehaviors.size() == 1 &&
         behavior.nanBehaviors.contains(::fabric::FloatNaNBehavior::IEEE) &&
         behavior.subnormalBehaviors.size() == 1 &&
         behavior.subnormalBehaviors.contains(
             ::fabric::FloatSubnormalBehavior::Preserve) &&
         behavior.signedZeroBehaviors.size() == 1 &&
         behavior.signedZeroBehaviors.contains(
             ::fabric::FloatSignedZeroBehavior::Preserve) &&
         behavior.requiredFastMath == mlir::arith::FastMathFlags::afn;
}

bool supportsAccuracy(::loom::SpecialMathAccuracyTier accuracy) {
  using Tier = ::loom::SpecialMathAccuracyTier;
  return accuracy == Tier::Max1Ulp || accuracy == Tier::Max2Ulp ||
         accuracy == Tier::Max4Ulp;
}

llvm::Expected<std::optional<LoweredMode>> lowerMode(const Mode &mode) {
  if (mode.actor.schema != ::dataflow::OperationSchemaId::MathErf ||
      mode.actor.type.getNumInputs() != 1 ||
      mode.actor.type.getNumResults() != 1)
    return invalid("behavior is not unary scalar math.erf");
  mlir::Type type = mode.actor.type.getInput(0);
  if (mode.actor.type.getResult(0) != type || mlir::isa<mlir::VectorType>(type))
    return invalid("behavior does not have a uniform scalar type");
  const auto *payload =
      std::get_if<::dataflow::SpecialMathPayload>(&mode.actor.payload);
  if (!payload || !supportsAccuracy(payload->accuracy) ||
      !mlir::arith::bitEnumContainsAll(payload->flags,
                                       mlir::arith::FastMathFlags::afn))
    return invalid("behavior does not carry the approximate erf contract");
  auto format = detail::resolvePortableFloatFormat(type);
  if (!format)
    return invalid("behavior has an unknown floating format");
  if (!(*format == Format{5, 10}) && !(*format == Format{8, 7}))
    return std::optional<LoweredMode>{};
  return std::optional<LoweredMode>{
      LoweredMode{*format, "loom_erf_e" + std::to_string(format->exponentBits) +
                               "_f" + std::to_string(format->fractionBits)}};
}

std::string hexLiteral(unsigned width, std::uint64_t value) {
  std::ostringstream stream;
  stream << width << "'h" << std::hex << value;
  return stream.str();
}

std::string buildLookupFunction(const std::string &functionName) {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic [29:0] " << functionName
         << "(input [7:0] segment);\n"
         << "  begin\n"
         << "    case (segment)\n";
  for (std::size_t index = 0; index < kErfQ30.size(); ++index)
    output << "      8'd" << index << ": " << functionName << " = "
           << hexLiteral(30, kErfQ30[index]) << ";\n";
  output << "      default: " << functionName << " = "
         << hexLiteral(30, kErfQ30.back()) << ";\n"
         << "    endcase\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

std::string buildErfFunction(const Format &format,
                             const std::string &functionName) {
  const unsigned width = format.width();
  const unsigned exponentBits = format.exponentBits;
  const unsigned fractionBits = format.fractionBits;
  const unsigned precision = format.precision();
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponentBits) - 1;
  const std::uint64_t one =
      std::uint64_t((std::uint64_t{1} << (exponentBits - 1)) - 1)
      << fractionBits;
  const std::uint64_t four = one + (std::uint64_t{2} << fractionBits);
  const std::uint64_t threshold = std::uint64_t(format.bias() - 5)
                                  << fractionBits;
  const std::uint64_t quietBit = std::uint64_t{1} << (fractionBits - 1);
  const std::uint64_t scale =
      format == Format{5, 10} ? UINT64_C(0x3c83) : UINT64_C(0x3f90);
  const std::string lookupName = functionName + "_q30";
  const std::string scaleName = functionName + "_scale";
  const std::string exponentAllOnes = hexLiteral(exponentBits, exponentMask);

  std::string text;
  llvm::raw_string_ostream output(text);
  output << detail::buildPortableFloatFmaFunction(format, scaleName) << '\n'
         << buildLookupFunction(lookupName) << '\n'
         << "function automatic [" << width - 1 << ":0] " << functionName
         << "(input [" << width - 1 << ":0] value);\n"
         << "  reg sign;\n"
         << "  reg [" << exponentBits - 1 << ":0] exponent;\n"
         << "  reg [" << fractionBits - 1 << ":0] fraction;\n"
         << "  reg [14:0] magnitude_bits;\n"
         << "  reg [" << precision - 1 << ":0] significand;\n"
         << "  reg [31:0] x_q24;\n"
         << "  reg [7:0] segment;\n"
         << "  reg [18:0] segment_fraction;\n"
         << "  reg [29:0] lower_value;\n"
         << "  reg [29:0] upper_value;\n"
         << "  reg [29:0] delta;\n"
         << "  reg [63:0] product;\n"
         << "  reg [30:0] interpolation;\n"
         << "  reg [30:0] y_q30;\n"
         << "  reg [" << precision << ":0] rounded;\n"
         << "  reg [" << fractionBits - 1 << ":0] result_fraction;\n"
         << "  reg [" << exponentBits - 1 << ":0] result_exponent;\n"
         << "  reg found;\n"
         << "  reg guard;\n"
         << "  reg sticky;\n"
         << "  integer exponent_value;\n"
         << "  integer shift_amount;\n"
         << "  integer leading_index;\n"
         << "  integer encoded_exponent;\n"
         << "  integer index;\n"
         << "  begin\n"
         << "    sign = value[" << width - 1 << "];\n"
         << "    exponent = value[" << width - 2 << ':' << fractionBits
         << "];\n"
         << "    fraction = value[" << fractionBits - 1 << ":0];\n"
         << "    magnitude_bits = value[14:0];\n"
         << "    if (exponent == " << exponentAllOnes
         << " && fraction != 0) begin\n"
         << "      " << functionName << " = value | "
         << hexLiteral(width, quietBit) << ";\n"
         << "    end else if (exponent == " << exponentAllOnes << ") begin\n"
         << "      " << functionName << " = {sign, "
         << hexLiteral(width - 1, one) << "};\n"
         << "    end else if (magnitude_bits == 0) begin\n"
         << "      " << functionName << " = value;\n"
         << "    end else if (magnitude_bits >= " << hexLiteral(width - 1, four)
         << ") begin\n"
         << "      " << functionName << " = {sign, "
         << hexLiteral(width - 1, one) << "};\n"
         << "    end else if (magnitude_bits < "
         << hexLiteral(width - 1, threshold) << ") begin\n"
         << "      " << functionName << " = " << scaleName << "(value, "
         << hexLiteral(width, scale) << ", " << width << "'d0);\n"
         << "    end else begin\n"
         << "      significand = exponent == 0 ? {1'b0, fraction} "
            ": {1'b1, fraction};\n"
         << "      exponent_value = integer'(exponent) - " << format.bias()
         << ";\n"
         << "      shift_amount = exponent_value + 24 - " << fractionBits
         << ";\n"
         << "      x_q24 = 32'd0;\n"
         << "      if (shift_amount >= 0)\n"
         << "        x_q24 = significand << shift_amount;\n"
         << "      else\n"
         << "        x_q24 = significand >> (-shift_amount);\n"
         << "      segment = x_q24 >> 19;\n"
         << "      segment_fraction = x_q24[18:0];\n"
         << "      lower_value = " << lookupName << "(segment);\n"
         << "      upper_value = " << lookupName << "(segment + 1'b1);\n"
         << "      delta = upper_value - lower_value;\n"
         << "      product = delta * segment_fraction;\n"
         << "      interpolation = product >> 19;\n"
         << "      if (product[18:0] > 19'h40000 ||\n"
         << "          (product[18:0] == 19'h40000 && interpolation[0]))\n"
         << "        interpolation = interpolation + 1'b1;\n"
         << "      y_q30 = {1'b0, lower_value} + interpolation;\n"
         << "      leading_index = 0;\n"
         << "      found = 1'b0;\n"
         << "      for (index = 30; index >= 0; index = index - 1) begin\n"
         << "        if (!found && y_q30[index]) begin\n"
         << "          leading_index = index;\n"
         << "          found = 1'b1;\n"
         << "        end\n"
         << "      end\n"
         << "      encoded_exponent = leading_index - 30 + " << format.bias()
         << ";\n"
         << "      shift_amount = leading_index - " << fractionBits << ";\n"
         << "      rounded = " << precision + 1 << "'d0;\n"
         << "      guard = 1'b0;\n"
         << "      sticky = 1'b0;\n"
         << "      if (shift_amount <= 0) begin\n"
         << "        rounded = y_q30 << (-shift_amount);\n"
         << "      end else begin\n"
         << "        rounded = y_q30 >> shift_amount;\n"
         << "        for (index = 0; index < 31; index = index + 1) begin\n"
         << "          if (index == shift_amount - 1) guard = y_q30[index];\n"
         << "          if (index < shift_amount - 1) "
            "sticky = sticky | y_q30[index];\n"
         << "        end\n"
         << "        if (guard && (sticky || rounded[0]))\n"
         << "          rounded = rounded + 1'b1;\n"
         << "      end\n"
         << "      if (rounded[" << precision << "]) begin\n"
         << "        rounded = rounded >> 1;\n"
         << "        encoded_exponent = encoded_exponent + 1;\n"
         << "      end\n"
         << "      result_exponent = encoded_exponent[" << exponentBits - 1
         << ":0];\n"
         << "      result_fraction = rounded[" << fractionBits - 1 << ":0];\n"
         << "      " << functionName
         << " = {sign, result_exponent, result_fraction};\n"
         << "    end\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

mlir::Value materializeMode(mlir::OpBuilder &builder, mlir::Location location,
                            circt::hw::HWModulePortAccessor &accessor,
                            const LoweredMode &mode, unsigned outputWidth) {
  const unsigned width = mode.format.width();
  mlir::Value input = detail::resizeUnsigned(
      builder, location, accessor.getInput("data_input_0"), width);
  mlir::Value value = circt::sv::VerbatimExprOp::create(
      builder, location, builder.getIntegerType(width),
      mode.functionName + "({{0}})", llvm::SmallVector<mlir::Value>{input});
  return detail::resizeUnsigned(builder, location, value, outputWidth);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableMathErf(FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return unsupported(request);
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::ScalarMathErf)
    return invalid("provider received a different implementation family");
  const auto *parameters = std::get_if<::fabric::ScalarSpecialMathParams>(
      &request.capability.parameterizedCapability);
  if (!parameters)
    return invalid("capability has the wrong parameter schema");
  if (request.capability.enabledOperationSchemas !=
      std::vector<::dataflow::OperationSchemaId>{
          ::dataflow::OperationSchemaId::MathErf})
    return invalid("capability does not contain exactly math.erf");
  if (!supportsBehavior(parameters->behavior) ||
      !supportsAccuracy(parameters->accuracyGuarantee))
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
      inputs[0]->reference.ordinal != 0 || outputs[0]->reference.ordinal != 0 ||
      inputs[0]->payloadWidthBits == 0 || outputs[0]->payloadWidthBits == 0)
    return invalid("capability does not have the unary scalar erf port shape");
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
    return invalid("Fabric returned an empty scalar erf behavior domain");

  const ConfigurationFieldEncoding *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  std::vector<Mode> modes;
  modes.reserve(domain.size());
  if (request.capability.configurationFieldSchema.empty()) {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None ||
        domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid(
          "configuration-free scalar erf has a non-singleton relation");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid("configured scalar erf relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured scalar erf capability requires one field");
    field = request.configurationAbi.findOperationField(
        request.occurrence,
        request.capability.configurationFieldSchema.front().ordinal);
    if (!field)
      return invalid("configured scalar erf field is absent from the ABI");
    codebook = std::get_if<FiniteCodebookEncoding>(&field->semanticEncoding);
    if (!codebook)
      return invalid("configured scalar erf field is not a finite codebook");
    if (codebook->entries.size() != domain.size())
      return invalid("scalar erf codebook does not exactly cover the domain");
    for (const auto &point : domain) {
      if (!point.semanticConfiguration)
        return invalid("configured scalar erf behavior has no semantic value");
      const FiniteCodebookEntry *entry = detail::findFiniteCodebookEntry(
          *codebook, point.semanticConfiguration->bytes());
      if (!entry)
        return invalid("scalar erf codebook omits an admitted semantic value");
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
      return invalid("scalar erf ABI inactive value is outside the domain");
    inactiveMode = static_cast<std::size_t>(inactive - modes.begin());
  }

  std::vector<LoweredMode> loweredModes;
  loweredModes.reserve(modes.size());
  for (const Mode &mode : modes) {
    auto lowered = lowerMode(mode);
    if (!lowered)
      return lowered.takeError();
    if (!*lowered)
      return unsupported(request);
    const unsigned width = (*lowered)->format.width();
    if (width > inputs[0]->payloadWidthBits ||
        width > outputs[0]->payloadWidthBits)
      return invalid("scalar erf behavior exceeds the physical datapath");
    loweredModes.push_back(std::move(**lowered));
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
          declarationStream << buildErfFunction(mode.format, mode.functionName)
                            << '\n';
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
registerPortableMathErfProvider(FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::ScalarMathErf,
                       BackendRecipeKey::PortableSystemVerilog,
                       {},
                       materializePortableMathErf});
}

} // namespace loom::hardware::rtl
