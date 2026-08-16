#include "Hardware/RTL/Providers/IntegerCountZeros.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

enum class Direction { Leading, Trailing };

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  Direction direction;
  unsigned elementWidth = 0;
  unsigned laneCount = 0;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_integer_count_zeros_invalid: " +
                                     message);
}

llvm::Expected<FabricOperationProviderOutput>
unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

llvm::Expected<Direction> lowerDirection(::dataflow::OperationSchemaId schema) {
  using Schema = ::dataflow::OperationSchemaId;
  switch (schema) {
  case Schema::MathCountLeadingZeros:
  case Schema::LLVMCountLeadingZeros:
    return Direction::Leading;
  case Schema::MathCountTrailingZeros:
  case Schema::LLVMCountTrailingZeros:
    return Direction::Trailing;
  default:
    return invalid("Fabric returned a non-count-zero behavior witness");
  }
}

llvm::Expected<LoweredMode> lowerMode(const Mode &mode,
                                      ::fabric::ImplementationFamilyId family) {
  auto direction = lowerDirection(mode.actor.schema);
  if (!direction)
    return direction.takeError();
  if (mode.actor.type.getNumInputs() != 1 ||
      mode.actor.type.getNumResults() != 1)
    return invalid("behavior has wrong arity");

  if (family == ::fabric::ImplementationFamilyId::ScalarIntegerCountZeros) {
    auto input = llvm::dyn_cast<mlir::IntegerType>(mode.actor.type.getInput(0));
    if (!input || !input.isSignless() || mode.actor.type.getResult(0) != input)
      return invalid("scalar behavior is not a uniform signless integer");
    return LoweredMode{*direction, input.getWidth(), 1};
  }

  auto input = llvm::dyn_cast<mlir::VectorType>(mode.actor.type.getInput(0));
  if (!input || input.isScalable() || mode.actor.type.getResult(0) != input)
    return invalid("vector behavior is not a uniform fixed vector");
  auto element = llvm::dyn_cast<mlir::IntegerType>(input.getElementType());
  if (!element || !element.isSignless())
    return invalid("vector behavior element is not a signless integer");
  const std::uint64_t lanes = input.getNumElements();
  if (lanes == 0 || lanes > std::numeric_limits<unsigned>::max())
    return invalid("behavior lane count is outside the RTL domain");
  return LoweredMode{*direction, element.getWidth(),
                     static_cast<unsigned>(lanes)};
}

mlir::Value materializeLane(mlir::OpBuilder &builder, mlir::Location location,
                            mlir::Value input, Direction direction,
                            unsigned width) {
  mlir::Value result = circt::hw::ConstantOp::create(builder, location,
                                                     llvm::APInt(width, width));
  if (direction == Direction::Leading) {
    for (unsigned bit = 0; bit < width; ++bit) {
      mlir::Value selected =
          circt::comb::ExtractOp::create(builder, location, input, bit, 1);
      mlir::Value count = circt::hw::ConstantOp::create(
          builder, location, llvm::APInt(width, width - bit - 1));
      result = circt::comb::MuxOp::create(builder, location, selected, count,
                                          result, true);
    }
    return result;
  }

  for (unsigned bit = width; bit > 0; --bit) {
    const unsigned index = bit - 1;
    mlir::Value selected =
        circt::comb::ExtractOp::create(builder, location, input, index, 1);
    mlir::Value count = circt::hw::ConstantOp::create(
        builder, location, llvm::APInt(width, index));
    result = circt::comb::MuxOp::create(builder, location, selected, count,
                                        result, true);
  }
  return result;
}

mlir::Value materializeMode(mlir::OpBuilder &builder, mlir::Location location,
                            circt::hw::HWModulePortAccessor &accessor,
                            const LoweredMode &mode, unsigned outputWidth) {
  std::vector<mlir::Value> laneResults;
  laneResults.reserve(mode.laneCount);
  for (unsigned lane = 0; lane < mode.laneCount; ++lane) {
    const unsigned lowBit = lane * mode.elementWidth;
    mlir::Value input = circt::comb::ExtractOp::create(
        builder, location, accessor.getInput("data_input_0"), lowBit,
        mode.elementWidth);
    laneResults.push_back(materializeLane(builder, location, input,
                                          mode.direction, mode.elementWidth));
  }

  mlir::Value packed = laneResults.front();
  if (laneResults.size() > 1) {
    std::vector<mlir::Value> highToLow(laneResults.rbegin(),
                                       laneResults.rend());
    packed = circt::comb::ConcatOp::create(builder, location, highToLow);
  }
  return detail::resizeUnsigned(builder, location, packed, outputWidth);
}

bool hasExpectedParameterSchema(
    const fabric::ResolvedFabricOpCapabilityView &capability,
    ::fabric::ImplementationFamilyId family) {
  if (family == ::fabric::ImplementationFamilyId::ScalarIntegerCountZeros)
    return std::holds_alternative<::fabric::ScalarIntegerParams>(
        capability.parameterizedCapability);
  return std::holds_alternative<::fabric::FixedVectorIntegerParams>(
      capability.parameterizedCapability);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableIntegerCountZeros(
    FabricOperationProviderRequest request,
    ::fabric::ImplementationFamilyId expectedFamily) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily != expectedFamily)
    return invalid("provider received a different implementation family");
  if (!hasExpectedParameterSchema(request.capability, expectedFamily))
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
      return invalid("configured count-zero relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured count-zero capability requires one field");
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
    auto lowered = lowerMode(mode, expectedFamily);
    if (!lowered)
      return lowered.takeError();
    const std::uint64_t payloadWidth =
        static_cast<std::uint64_t>(lowered->elementWidth) * lowered->laneCount;
    if (payloadWidth > inputs[0]->payloadWidthBits ||
        payloadWidth > outputs[0]->payloadWidthBits)
      return invalid("behavior payload exceeds the physical datapath");
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
materializePortableScalarIntegerCountZeros(
    FabricOperationProviderRequest request) {
  return materializePortableIntegerCountZeros(
      request, ::fabric::ImplementationFamilyId::ScalarIntegerCountZeros);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFixedVectorIntegerCountZeros(
    FabricOperationProviderRequest request) {
  return materializePortableIntegerCountZeros(
      request, ::fabric::ImplementationFamilyId::FixedVectorIntegerCountZeros);
}

} // namespace

llvm::Error registerPortableIntegerCountZerosProviders(
    FabricOperationProviderRegistry &registry) {
  if (llvm::Error error = registry.add(
          {::fabric::ImplementationFamilyId::ScalarIntegerCountZeros,
           BackendRecipeKey::PortableSystemVerilog,
           {},
           materializePortableScalarIntegerCountZeros}))
    return error;
  return registry.add(
      {::fabric::ImplementationFamilyId::FixedVectorIntegerCountZeros,
       BackendRecipeKey::PortableSystemVerilog,
       {},
       materializePortableFixedVectorIntegerCountZeros});
}

} // namespace loom::hardware::rtl
