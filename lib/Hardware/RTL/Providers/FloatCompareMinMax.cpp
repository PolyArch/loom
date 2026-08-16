#include "Hardware/RTL/Providers/FloatCompareMinMax.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
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

enum class FamilyKind { Scalar, FixedVector };
enum class Role { Compare, Minimum, Maximum, MinNumber, MaxNumber };

using Format = detail::PortableFloatFormat;
using Schema = ::dataflow::OperationSchemaId;

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  Format format;
  Role role;
  mlir::arith::CmpFPredicate predicate;
  unsigned laneCount;
  bool assumeNoNaNs;

  bool isCompare() const { return role == Role::Compare; }

  unsigned inputPayloadWidth() const { return format.width() * laneCount; }

  unsigned resultPayloadWidth() const {
    return (isCompare() ? 1 : format.width()) * laneCount;
  }
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_float_compare_min_max_invalid: " +
                                     message);
}

llvm::Expected<FabricOperationProviderOutput>
unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

::fabric::ImplementationFamilyId familyId(FamilyKind family) {
  return family == FamilyKind::Scalar
             ? ::fabric::ImplementationFamilyId::ScalarFloatCompareMinMax
             : ::fabric::ImplementationFamilyId::FixedVectorFloatCompareMinMax;
}

bool hasExpectedParameterSchema(
    const fabric::ResolvedFabricOpCapabilityView &capability,
    FamilyKind family) {
  if (family == FamilyKind::Scalar)
    return std::holds_alternative<::fabric::ScalarFloatCompareMinMaxParams>(
        capability.parameterizedCapability);
  return std::holds_alternative<::fabric::FixedVectorFloatCompareMinMaxParams>(
      capability.parameterizedCapability);
}

llvm::Expected<Role> roleFor(Schema schema) {
  switch (schema) {
  case Schema::ArithCmpF:
    return Role::Compare;
  case Schema::ArithMinimumF:
    return Role::Minimum;
  case Schema::ArithMaximumF:
    return Role::Maximum;
  case Schema::ArithMinNumF:
    return Role::MinNumber;
  case Schema::ArithMaxNumF:
    return Role::MaxNumber;
  default:
    return invalid("behavior has a non-compare/min-max schema");
  }
}

llvm::Expected<LoweredMode>
lowerMode(FamilyKind family,
          const ::dataflow::CanonicalActorSchemaProjection &actor) {
  auto role = roleFor(actor.schema);
  if (!role)
    return role.takeError();
  if (actor.type.getNumInputs() != 2 || actor.type.getNumResults() != 1)
    return invalid("behavior has wrong arity");
  if (actor.type.getInput(0) != actor.type.getInput(1))
    return invalid("behavior inputs have different floating types");

  mlir::Type input = actor.type.getInput(0);
  mlir::Type element = input;
  unsigned laneCount = 1;
  mlir::VectorType inputVector;
  if (family == FamilyKind::FixedVector) {
    inputVector = mlir::dyn_cast<mlir::VectorType>(input);
    if (!inputVector || inputVector.isScalable() ||
        inputVector.getRank() == 0 ||
        llvm::any_of(inputVector.getShape(),
                     [](std::int64_t extent) { return extent <= 0; }))
      return invalid("behavior does not have a fixed positive vector type");
    const std::uint64_t lanes = inputVector.getNumElements();
    if (lanes == 0 || lanes > std::numeric_limits<unsigned>::max())
      return invalid("behavior lane count is outside the RTL domain");
    laneCount = static_cast<unsigned>(lanes);
    element = inputVector.getElementType();
  } else if (mlir::isa<mlir::VectorType>(input)) {
    return invalid("scalar behavior has a vector type");
  }

  auto format = detail::resolvePortableFloatFormat(element);
  if (!format)
    return invalid("behavior uses an unsupported floating format");
  if (laneCount > std::numeric_limits<unsigned>::max() / format->width())
    return invalid("behavior payload width exceeds the RTL domain");

  mlir::arith::CmpFPredicate predicate =
      mlir::arith::CmpFPredicate::AlwaysFalse;
  mlir::arith::FastMathFlags flags = mlir::arith::FastMathFlags::none;
  mlir::Type result = actor.type.getResult(0);
  if (*role == Role::Compare) {
    const auto *payload =
        std::get_if<::dataflow::FloatComparePayload>(&actor.payload);
    if (!payload || static_cast<std::uint64_t>(payload->predicate) >
                        mlir::arith::getMaxEnumValForCmpFPredicate())
      return invalid("comparison behavior has no valid typed predicate");
    predicate = payload->predicate;
    flags = payload->flags;
    if (family == FamilyKind::Scalar) {
      if (!result.isInteger(1))
        return invalid("scalar comparison result is not one bit");
    } else {
      auto resultVector = mlir::dyn_cast<mlir::VectorType>(result);
      if (!resultVector || resultVector.isScalable() ||
          resultVector.getShape() != inputVector.getShape() ||
          !resultVector.getElementType().isInteger(1))
        return invalid("vector comparison result has the wrong shape");
    }
  } else {
    const auto *payload =
        std::get_if<::dataflow::FloatingPointPayload>(&actor.payload);
    if (!payload || payload->roundingMode)
      return invalid("min/max behavior has a noncanonical payload");
    flags = payload->flags;
    if (result != input)
      return invalid("min/max behavior result differs from its inputs");
  }

  return LoweredMode{
      *format, *role, predicate, laneCount,
      mlir::arith::bitEnumContainsAll(flags, mlir::arith::FastMathFlags::nnan)};
}

std::string coreName(const Format &format) {
  return "loom_float_compare_min_max_e" + std::to_string(format.exponentBits) +
         "_f" + std::to_string(format.fractionBits);
}

std::string hexLiteral(unsigned width, std::uint64_t value) {
  return std::to_string(width) + "'h" + llvm::utohexstr(value);
}

std::string buildCoreFunction(const Format &format) {
  const unsigned width = format.width();
  const unsigned exponentBits = format.exponentBits;
  const unsigned fractionBits = format.fractionBits;
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponentBits) - 1;
  const std::uint64_t quietBit = std::uint64_t{1} << (fractionBits - 1);
  const std::string name = coreName(format);

  std::string text;
  llvm::raw_string_ostream output(text);
  output << "function automatic [" << width - 1 << ":0] " << name << "(input ["
         << width - 1 << ":0] lhs, input [" << width - 1
         << ":0] rhs, input [2:0] role, input [3:0] predicate, "
            "input assume_no_nan);\n"
         << "  reg sign_lhs;\n"
         << "  reg sign_rhs;\n"
         << "  reg [" << exponentBits - 1 << ":0] exponent_lhs;\n"
         << "  reg [" << exponentBits - 1 << ":0] exponent_rhs;\n"
         << "  reg [" << fractionBits - 1 << ":0] fraction_lhs;\n"
         << "  reg [" << fractionBits - 1 << ":0] fraction_rhs;\n"
         << "  reg lhs_nan;\n"
         << "  reg rhs_nan;\n"
         << "  reg lhs_signaling;\n"
         << "  reg rhs_signaling;\n"
         << "  reg lhs_zero;\n"
         << "  reg rhs_zero;\n"
         << "  reg unordered;\n"
         << "  reg equal;\n"
         << "  reg less;\n"
         << "  reg greater;\n"
         << "  reg comparison;\n"
         << "  begin\n"
         << "    sign_lhs = lhs[" << width - 1 << "];\n"
         << "    sign_rhs = rhs[" << width - 1 << "];\n"
         << "    exponent_lhs = lhs[" << width - 2 << ':' << fractionBits
         << "];\n"
         << "    exponent_rhs = rhs[" << width - 2 << ':' << fractionBits
         << "];\n"
         << "    fraction_lhs = lhs[" << fractionBits - 1 << ":0];\n"
         << "    fraction_rhs = rhs[" << fractionBits - 1 << ":0];\n"
         << "    lhs_nan = !assume_no_nan && exponent_lhs == "
         << hexLiteral(exponentBits, exponentMask) << " && fraction_lhs != 0;\n"
         << "    rhs_nan = !assume_no_nan && exponent_rhs == "
         << hexLiteral(exponentBits, exponentMask) << " && fraction_rhs != 0;\n"
         << "    lhs_signaling = lhs_nan && !fraction_lhs[" << fractionBits - 1
         << "];\n"
         << "    rhs_signaling = rhs_nan && !fraction_rhs[" << fractionBits - 1
         << "];\n"
         << "    lhs_zero = exponent_lhs == 0 && fraction_lhs == 0;\n"
         << "    rhs_zero = exponent_rhs == 0 && fraction_rhs == 0;\n"
         << "    unordered = lhs_nan || rhs_nan;\n"
         << "    equal = !unordered && ((lhs == rhs) || "
            "(lhs_zero && rhs_zero));\n"
         << "    less = 1'b0;\n"
         << "    if (!unordered && !equal) begin\n"
         << "      if (sign_lhs != sign_rhs)\n"
         << "        less = sign_lhs;\n"
         << "      else if (!sign_lhs)\n"
         << "        less = lhs[" << width - 2 << ":0] < rhs[" << width - 2
         << ":0];\n"
         << "      else\n"
         << "        less = lhs[" << width - 2 << ":0] > rhs[" << width - 2
         << ":0];\n"
         << "    end\n"
         << "    greater = !unordered && !equal && !less;\n"
         << "    comparison = 1'b0;\n"
         << "    case (predicate)\n"
         << "      4'd0: comparison = 1'b0;\n"
         << "      4'd1: comparison = equal;\n"
         << "      4'd2: comparison = greater;\n"
         << "      4'd3: comparison = greater || equal;\n"
         << "      4'd4: comparison = less;\n"
         << "      4'd5: comparison = less || equal;\n"
         << "      4'd6: comparison = !unordered && !equal;\n"
         << "      4'd7: comparison = !unordered;\n"
         << "      4'd8: comparison = unordered || equal;\n"
         << "      4'd9: comparison = unordered || greater;\n"
         << "      4'd10: comparison = unordered || greater || equal;\n"
         << "      4'd11: comparison = unordered || less;\n"
         << "      4'd12: comparison = unordered || less || equal;\n"
         << "      4'd13: comparison = unordered || !equal;\n"
         << "      4'd14: comparison = unordered;\n"
         << "      4'd15: comparison = 1'b1;\n"
         << "      default: comparison = 1'b0;\n"
         << "    endcase\n"
         << "    " << name << " = " << width << "'d0;\n"
         << "    case (role)\n"
         << "      3'd0: " << name << "[0] = comparison;\n"
         << "      3'd1: begin\n"
         << "        if (lhs_nan) " << name << " = lhs | "
         << hexLiteral(width, quietBit) << ";\n"
         << "        else if (rhs_nan) " << name << " = rhs | "
         << hexLiteral(width, quietBit) << ";\n"
         << "        else if (lhs_zero && rhs_zero && sign_lhs != sign_rhs)\n"
         << "          " << name << " = sign_lhs ? lhs : rhs;\n"
         << "        else " << name << " = (less || equal) ? lhs : rhs;\n"
         << "      end\n"
         << "      3'd2: begin\n"
         << "        if (lhs_nan) " << name << " = lhs | "
         << hexLiteral(width, quietBit) << ";\n"
         << "        else if (rhs_nan) " << name << " = rhs | "
         << hexLiteral(width, quietBit) << ";\n"
         << "        else if (lhs_zero && rhs_zero && sign_lhs != sign_rhs)\n"
         << "          " << name << " = sign_lhs ? rhs : lhs;\n"
         << "        else " << name << " = (greater || equal) ? lhs : rhs;\n"
         << "      end\n"
         << "      3'd3: begin\n"
         << "        if (lhs_signaling) " << name << " = lhs | "
         << hexLiteral(width, quietBit) << ";\n"
         << "        else if (rhs_signaling) " << name << " = rhs | "
         << hexLiteral(width, quietBit) << ";\n"
         << "        else if (lhs_nan) " << name << " = rhs;\n"
         << "        else if (rhs_nan) " << name << " = lhs;\n"
         << "        else if (lhs_zero && rhs_zero && sign_lhs != sign_rhs)\n"
         << "          " << name << " = sign_lhs ? lhs : rhs;\n"
         << "        else " << name << " = (less || equal) ? lhs : rhs;\n"
         << "      end\n"
         << "      3'd4: begin\n"
         << "        if (lhs_signaling) " << name << " = lhs | "
         << hexLiteral(width, quietBit) << ";\n"
         << "        else if (rhs_signaling) " << name << " = rhs | "
         << hexLiteral(width, quietBit) << ";\n"
         << "        else if (lhs_nan) " << name << " = rhs;\n"
         << "        else if (rhs_nan) " << name << " = lhs;\n"
         << "        else if (lhs_zero && rhs_zero && sign_lhs != sign_rhs)\n"
         << "          " << name << " = sign_lhs ? rhs : lhs;\n"
         << "        else " << name << " = (greater || equal) ? lhs : rhs;\n"
         << "      end\n"
         << "      default: " << name << " = " << width << "'d0;\n"
         << "    endcase\n"
         << "  end\n"
         << "endfunction\n";
  return output.str();
}

mlir::Value callCore(mlir::OpBuilder &builder, mlir::Location location,
                     const LoweredMode &mode, mlir::Value lhs,
                     mlir::Value rhs) {
  const std::string expression =
      coreName(mode.format) + "({{0}}, {{1}}, 3'd" +
      std::to_string(static_cast<unsigned>(mode.role)) + ", 4'd" +
      std::to_string(static_cast<unsigned>(mode.predicate)) + ", 1'b" +
      (mode.assumeNoNaNs ? "1" : "0") + ")";
  return circt::sv::VerbatimExprOp::create(
      builder, location, builder.getIntegerType(mode.format.width()),
      expression, llvm::SmallVector<mlir::Value, 2>{lhs, rhs});
}

mlir::Value materializeMode(mlir::OpBuilder &builder, mlir::Location location,
                            circt::hw::HWModulePortAccessor &accessor,
                            FamilyKind family, const LoweredMode &mode,
                            unsigned outputWidth) {
  const unsigned elementWidth = mode.format.width();
  if (family == FamilyKind::Scalar) {
    mlir::Value lhs = detail::resizeUnsigned(
        builder, location, accessor.getInput("data_input_0"), elementWidth);
    mlir::Value rhs = detail::resizeUnsigned(
        builder, location, accessor.getInput("data_input_1"), elementWidth);
    mlir::Value result = callCore(builder, location, mode, lhs, rhs);
    if (mode.isCompare())
      result = circt::comb::ExtractOp::create(builder, location, result, 0, 1);
    return detail::resizeUnsigned(builder, location, result, outputWidth);
  }

  std::vector<mlir::Value> laneResults;
  laneResults.reserve(mode.laneCount);
  for (unsigned lane = 0; lane != mode.laneCount; ++lane) {
    const unsigned lowBit = lane * elementWidth;
    mlir::Value lhs = circt::comb::ExtractOp::create(
        builder, location, accessor.getInput("data_input_0"), lowBit,
        elementWidth);
    mlir::Value rhs = circt::comb::ExtractOp::create(
        builder, location, accessor.getInput("data_input_1"), lowBit,
        elementWidth);
    mlir::Value result = callCore(builder, location, mode, lhs, rhs);
    if (mode.isCompare())
      result = circt::comb::ExtractOp::create(builder, location, result, 0, 1);
    laneResults.push_back(result);
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
materializePortableFloatCompareMinMax(FabricOperationProviderRequest request,
                                      FamilyKind family) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily != familyId(family))
    return invalid("provider received a different implementation family");
  if (!hasExpectedParameterSchema(request.capability, family))
    return invalid("capability has the wrong parameter schema");

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
  if (domain.empty())
    return invalid("Fabric returned an empty behavior domain");

  const ConfigurationEncodingRelation *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  std::vector<Mode> modes;
  modes.reserve(domain.size());
  if (request.capability.configurationFieldSchema.empty()) {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None ||
        domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid(
          "configuration-free capability has a non-singleton behavior domain");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid("configured compare/min/max relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid(
          "configured compare/min/max capability requires one field");
    field = request.configurationAbi.findOperationEncodingRelation(
        request.occurrence,
        request.capability.configurationFieldSchema.front().ordinal);
    if (!field)
      return invalid("configured field is absent from the ABI");
    codebook = std::get_if<FiniteCodebookEncoding>(&field->semanticEncoding);
    if (!codebook)
      return invalid("configured field is not a finite codebook");
    if (codebook->entries.size() != domain.size())
      return invalid(
          "codebook does not exactly cover the configuration domain");
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
  loweredModes.reserve(modes.size());
  for (const Mode &mode : modes) {
    auto lowered = lowerMode(family, mode.actor);
    if (!lowered)
      return lowered.takeError();
    if (lowered->inputPayloadWidth() > inputs[0]->payloadWidthBits ||
        lowered->inputPayloadWidth() > inputs[1]->payloadWidthBits ||
        lowered->resultPayloadWidth() > outputs[0]->payloadWidthBits)
      return unsupported(request);
    loweredModes.push_back(*lowered);
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
materializePortableScalarFloatCompareMinMax(
    FabricOperationProviderRequest request) {
  return materializePortableFloatCompareMinMax(std::move(request),
                                               FamilyKind::Scalar);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFixedVectorFloatCompareMinMax(
    FabricOperationProviderRequest request) {
  return materializePortableFloatCompareMinMax(std::move(request),
                                               FamilyKind::FixedVector);
}

} // namespace

llvm::Error registerPortableFloatCompareMinMaxProviders(
    FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry candidate = registry;
  if (llvm::Error error = candidate.add(
          {::fabric::ImplementationFamilyId::ScalarFloatCompareMinMax,
           BackendRecipeKey::PortableSystemVerilog,
           {},
           materializePortableScalarFloatCompareMinMax}))
    return error;
  if (llvm::Error error = candidate.add(
          {::fabric::ImplementationFamilyId::FixedVectorFloatCompareMinMax,
           BackendRecipeKey::PortableSystemVerilog,
           {},
           materializePortableFixedVectorFloatCompareMinMax}))
    return error;
  registry = std::move(candidate);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl
