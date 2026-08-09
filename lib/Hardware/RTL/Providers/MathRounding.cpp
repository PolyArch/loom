#include "Hardware/RTL/Providers/MathRounding.h"

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

enum class RoundingOperation { Floor, Ceil, Round, Trunc, RoundEven };
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
                                 "portable_math_rounding_invalid: " + message);
}

llvm::Expected<FabricOperationProviderOutput>
unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

llvm::Expected<RoundingOperation>
operationForFamily(::fabric::ImplementationFamilyId family) {
  using Family = ::fabric::ImplementationFamilyId;
  switch (family) {
  case Family::ScalarMathFloor:
    return RoundingOperation::Floor;
  case Family::ScalarMathCeil:
    return RoundingOperation::Ceil;
  case Family::ScalarMathRound:
    return RoundingOperation::Round;
  case Family::ScalarMathTrunc:
    return RoundingOperation::Trunc;
  case Family::ScalarMathRoundEven:
    return RoundingOperation::RoundEven;
  default:
    return invalid("provider received a different implementation family");
  }
}

::dataflow::OperationSchemaId schemaForOperation(RoundingOperation operation) {
  using Schema = ::dataflow::OperationSchemaId;
  switch (operation) {
  case RoundingOperation::Floor:
    return Schema::MathFloor;
  case RoundingOperation::Ceil:
    return Schema::MathCeil;
  case RoundingOperation::Round:
    return Schema::MathRound;
  case RoundingOperation::Trunc:
    return Schema::MathTrunc;
  case RoundingOperation::RoundEven:
    return Schema::MathRoundEven;
  }
  llvm_unreachable("unknown math-rounding operation");
}

bool hasFastMathFlag(mlir::arith::FastMathFlags flags,
                     mlir::arith::FastMathFlags flag) {
  return mlir::arith::bitEnumContainsAll(flags, flag);
}

std::string functionName(const Format &format) {
  return "loom_math_rounding_e" + std::to_string(format.exponentBits) + "_f" +
         std::to_string(format.fractionBits);
}

llvm::Expected<LoweredMode>
lowerMode(const ::dataflow::CanonicalActorSchemaProjection &actor,
          RoundingOperation operation) {
  if (actor.schema != schemaForOperation(operation) ||
      actor.type.getNumInputs() != 1 || actor.type.getNumResults() != 1 ||
      actor.type.getInput(0) != actor.type.getResult(0))
    return invalid("behavior is not the exact unary rounding operation");
  const auto *payload =
      std::get_if<::dataflow::SpecialMathPayload>(&actor.payload);
  if (!payload)
    return invalid("behavior has no typed special-math payload");
  if (llvm::Error error = ::loom::validateSpecialMathAccuracyContract(
          payload->accuracy,
          hasFastMathFlag(payload->flags, mlir::arith::FastMathFlags::afn)))
    return error;
  auto format = detail::resolvePortableFloatFormat(actor.type.getInput(0));
  if (!format)
    return invalid("behavior uses an unsupported floating format");
  return LoweredMode{*format, functionName(*format)};
}

std::string hexLiteral(unsigned width, std::uint64_t value) {
  std::ostringstream stream;
  stream << width << "'h" << std::hex << value;
  return stream.str();
}

std::string buildRoundingFunction(const LoweredMode &mode,
                                  RoundingOperation operation) {
  const unsigned width = mode.format.width();
  const unsigned exponentBits = mode.format.exponentBits;
  const unsigned fractionBits = mode.format.fractionBits;
  const std::uint64_t bias = mode.format.bias();
  const std::uint64_t oneMagnitude = bias << fractionBits;
  const std::uint64_t halfMagnitude = (bias - 1) << fractionBits;
  const std::uint64_t quietMask = std::uint64_t{1} << (fractionBits - 1);
  const bool needsMagnitude = operation == RoundingOperation::Round ||
                              operation == RoundingOperation::RoundEven;
  const bool needsDiscarded = operation == RoundingOperation::Floor ||
                              operation == RoundingOperation::Ceil;
  const bool needsGuard = operation == RoundingOperation::Round ||
                          operation == RoundingOperation::RoundEven;
  const bool needsSticky = operation == RoundingOperation::RoundEven;
  const bool needsRetainedOdd = operation == RoundingOperation::RoundEven;

  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic [" << width - 1 << ":0] " << mode.functionName
         << "(input [" << width - 1 << ":0] value);\n"
         << "  reg sign;\n"
         << "  reg [" << exponentBits - 1 << ":0] exponent;\n"
         << "  reg [" << fractionBits - 1 << ":0] fraction;\n";
  if (needsMagnitude)
    output << "  reg [" << width - 2 << ":0] magnitude;\n";
  output << "  reg [" << width - 1 << ":0] truncated_value;\n";
  if (needsDiscarded)
    output << "  reg discarded;\n";
  if (needsGuard)
    output << "  reg guard;\n";
  if (needsSticky)
    output << "  reg sticky;\n";
  if (needsRetainedOdd)
    output << "  reg retained_odd;\n";
  output << "  reg increment;\n"
         << "  integer unbiased_exponent;\n"
         << "  integer clear_count;\n"
         << "  integer index;\n"
         << "  begin\n"
         << "    sign = value[" << width - 1 << "];\n"
         << "    exponent = value[" << width - 2 << ":" << fractionBits
         << "];\n"
         << "    fraction = value[" << fractionBits - 1 << ":0];\n";
  if (needsMagnitude)
    output << "    magnitude = value[" << width - 2 << ":0];\n";
  output << "    truncated_value = value;\n";
  if (needsDiscarded)
    output << "    discarded = 1'b0;\n";
  if (needsGuard)
    output << "    guard = 1'b0;\n";
  if (needsSticky)
    output << "    sticky = 1'b0;\n";
  if (needsRetainedOdd)
    output << "    retained_odd = 1'b0;\n";
  output << "    increment = 1'b0;\n"
         << "    " << mode.functionName << " = value;\n"
         << "    if (&exponent) begin\n"
         << "      if ((|fraction) && !fraction[" << fractionBits - 1 << "])\n"
         << "        " << mode.functionName << " = value | "
         << hexLiteral(width, quietMask) << ";\n"
         << "    end else if ((exponent == 0) && (fraction == 0)) begin\n"
         << "      " << mode.functionName << " = value;\n"
         << "    end else begin\n"
         << "      if (exponent == 0)\n"
         << "        unbiased_exponent = " << 1 - mode.format.bias() << ";\n"
         << "      else begin\n"
         << "        unbiased_exponent = {{" << 32 - exponentBits
         << "{1'b0}}, exponent};\n"
         << "        unbiased_exponent = unbiased_exponent - " << bias << ";\n"
         << "      end\n"
         << "      if (unbiased_exponent >= " << fractionBits << ") begin\n"
         << "        " << mode.functionName << " = value;\n"
         << "      end else if (unbiased_exponent < 0) begin\n";

  switch (operation) {
  case RoundingOperation::Floor:
    output << "        if (sign)\n"
           << "          " << mode.functionName << " = {sign, "
           << hexLiteral(width - 1, oneMagnitude) << "};\n"
           << "        else\n"
           << "          " << mode.functionName << " = {sign, " << width - 1
           << "'d0};\n";
    break;
  case RoundingOperation::Ceil:
    output << "        if (!sign)\n"
           << "          " << mode.functionName << " = {sign, "
           << hexLiteral(width - 1, oneMagnitude) << "};\n"
           << "        else\n"
           << "          " << mode.functionName << " = {sign, " << width - 1
           << "'d0};\n";
    break;
  case RoundingOperation::Round:
    output << "        if (magnitude >= "
           << hexLiteral(width - 1, halfMagnitude) << ")\n"
           << "          " << mode.functionName << " = {sign, "
           << hexLiteral(width - 1, oneMagnitude) << "};\n"
           << "        else\n"
           << "          " << mode.functionName << " = {sign, " << width - 1
           << "'d0};\n";
    break;
  case RoundingOperation::Trunc:
    output << "        " << mode.functionName << " = {sign, " << width - 1
           << "'d0};\n";
    break;
  case RoundingOperation::RoundEven:
    output << "        if (magnitude > " << hexLiteral(width - 1, halfMagnitude)
           << ")\n"
           << "          " << mode.functionName << " = {sign, "
           << hexLiteral(width - 1, oneMagnitude) << "};\n"
           << "        else\n"
           << "          " << mode.functionName << " = {sign, " << width - 1
           << "'d0};\n";
    break;
  }

  output << "      end else begin\n"
         << "        clear_count = " << fractionBits
         << " - unbiased_exponent;\n";
  if (needsRetainedOdd)
    output << "        retained_odd = (unbiased_exponent == 0);\n";
  output << "        for (index = 0; index < " << fractionBits
         << "; index = index + 1) begin\n"
         << "          if (index < clear_count) begin\n"
         << "            truncated_value[index] = 1'b0;\n";
  if (needsDiscarded)
    output << "            discarded = discarded | value[index];\n";
  output << "          end\n";
  if (needsGuard)
    output << "          if (index == clear_count - 1)\n"
           << "            guard = value[index];\n";
  if (needsSticky)
    output << "          if (index < clear_count - 1)\n"
           << "            sticky = sticky | value[index];\n";
  if (needsRetainedOdd)
    output << "          if (index == clear_count)\n"
           << "            retained_odd = value[index];\n";
  output << "        end\n";
  switch (operation) {
  case RoundingOperation::Floor:
    output << "        increment = sign && discarded;\n";
    break;
  case RoundingOperation::Ceil:
    output << "        increment = !sign && discarded;\n";
    break;
  case RoundingOperation::Round:
    output << "        increment = guard;\n";
    break;
  case RoundingOperation::Trunc:
    output << "        increment = 1'b0;\n";
    break;
  case RoundingOperation::RoundEven:
    output << "        increment = guard && (sticky || retained_odd);\n";
    break;
  }
  output << "        if (increment)\n"
         << "          " << mode.functionName << " = truncated_value + ("
         << width << "'d1 << clear_count);\n"
         << "        else\n"
         << "          " << mode.functionName << " = truncated_value;\n"
         << "      end\n"
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
  mlir::Value result = circt::sv::VerbatimExprOp::create(
      builder, location, builder.getIntegerType(width),
      mode.functionName + "({{0}})", llvm::SmallVector<mlir::Value>{input});
  return detail::resizeUnsigned(builder, location, result, outputWidth);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableMathRounding(FabricOperationProviderRequest request,
                                ::fabric::ImplementationFamilyId family) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily != family)
    return invalid("provider received a different implementation family");
  auto operation = operationForFamily(family);
  if (!operation)
    return operation.takeError();
  if (!std::holds_alternative<::fabric::ScalarSpecialMathParams>(
          request.capability.parameterizedCapability))
    return invalid("capability has the wrong parameter schema");
  if (request.capability.enabledOperationSchemas !=
      std::vector<::dataflow::OperationSchemaId>{
          schemaForOperation(*operation)})
    return invalid("capability does not contain its exact generated schema");

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
    return unsupported(request);
  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return error;

  auto relation = request.capability.resolveSemanticFieldRelation(
      *request.leaf.getContext());
  if (!relation)
    return relation.takeError();
  const auto &domain = relation->finiteBehaviorDomain();
  const ConfigurationFieldEncoding *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  std::vector<Mode> modes;
  if (request.capability.configurationFieldSchema.empty()) {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None ||
        domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid(
          "configuration-free capability has a non-singleton behavior domain");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid("configured math-rounding relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured math-rounding capability requires one field");
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
  std::set<std::string> functionNames;
  loweredModes.reserve(modes.size());
  for (const Mode &mode : modes) {
    auto lowered = lowerMode(mode.actor, *operation);
    if (!lowered)
      return lowered.takeError();
    if (!functionNames.insert(lowered->functionName).second)
      return invalid("sealed relation contains a duplicate rounding format");
    if (lowered->format.width() > inputs[0]->payloadWidthBits ||
        lowered->format.width() > outputs[0]->payloadWidthBits)
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
          declarationStream << buildRoundingFunction(mode, *operation) << '\n';
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
materializeFloor(FabricOperationProviderRequest request) {
  return materializePortableMathRounding(
      std::move(request), ::fabric::ImplementationFamilyId::ScalarMathFloor);
}

llvm::Expected<FabricOperationProviderOutput>
materializeCeil(FabricOperationProviderRequest request) {
  return materializePortableMathRounding(
      std::move(request), ::fabric::ImplementationFamilyId::ScalarMathCeil);
}

llvm::Expected<FabricOperationProviderOutput>
materializeRound(FabricOperationProviderRequest request) {
  return materializePortableMathRounding(
      std::move(request), ::fabric::ImplementationFamilyId::ScalarMathRound);
}

llvm::Expected<FabricOperationProviderOutput>
materializeTrunc(FabricOperationProviderRequest request) {
  return materializePortableMathRounding(
      std::move(request), ::fabric::ImplementationFamilyId::ScalarMathTrunc);
}

llvm::Expected<FabricOperationProviderOutput>
materializeRoundEven(FabricOperationProviderRequest request) {
  return materializePortableMathRounding(
      std::move(request),
      ::fabric::ImplementationFamilyId::ScalarMathRoundEven);
}

} // namespace

llvm::Error registerPortableMathRoundingProviders(
    FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry candidate = registry;
  const std::array registrations = {
      FabricOperationProviderRegistration{
          ::fabric::ImplementationFamilyId::ScalarMathFloor,
          BackendRecipeKey::PortableSystemVerilog,
          {},
          materializeFloor},
      FabricOperationProviderRegistration{
          ::fabric::ImplementationFamilyId::ScalarMathCeil,
          BackendRecipeKey::PortableSystemVerilog,
          {},
          materializeCeil},
      FabricOperationProviderRegistration{
          ::fabric::ImplementationFamilyId::ScalarMathRound,
          BackendRecipeKey::PortableSystemVerilog,
          {},
          materializeRound},
      FabricOperationProviderRegistration{
          ::fabric::ImplementationFamilyId::ScalarMathTrunc,
          BackendRecipeKey::PortableSystemVerilog,
          {},
          materializeTrunc},
      FabricOperationProviderRegistration{
          ::fabric::ImplementationFamilyId::ScalarMathRoundEven,
          BackendRecipeKey::PortableSystemVerilog,
          {},
          materializeRoundEven}};
  for (const FabricOperationProviderRegistration &registration : registrations)
    if (llvm::Error error = candidate.add(registration))
      return error;
  registry = std::move(candidate);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl
