#include "Hardware/RTL/Providers/IntegerSaturatingAddSub.h"

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

using Family = ::fabric::ImplementationFamilyId;
using Schema = ::dataflow::OperationSchemaId;

enum class ArithmeticKind {
  SignedAdd,
  UnsignedAdd,
  SignedSubtract,
  UnsignedSubtract,
};

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  ArithmeticKind arithmetic;
  unsigned elementWidth;
  unsigned laneCount;
};

struct PhysicalPorts final {
  const fabric::ResolvedFabricOpPhysicalPortView *lhs;
  const fabric::ResolvedFabricOpPhysicalPortView *rhs;
  const fabric::ResolvedFabricOpPhysicalPortView *result;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "portable_integer_saturating_add_sub_invalid: " + message);
}

bool isSaturatingFamily(Family family) {
  return family == Family::ScalarIntegerSaturatingAddSub ||
         family == Family::FixedVectorIntegerSaturatingAddSub;
}

bool isSigned(ArithmeticKind arithmetic) {
  return arithmetic == ArithmeticKind::SignedAdd ||
         arithmetic == ArithmeticKind::SignedSubtract;
}

bool isSubtract(ArithmeticKind arithmetic) {
  return arithmetic == ArithmeticKind::SignedSubtract ||
         arithmetic == ArithmeticKind::UnsignedSubtract;
}

llvm::Expected<ArithmeticKind> arithmeticFor(Schema schema) {
  switch (schema) {
  case Schema::LLVMSAddSat:
    return ArithmeticKind::SignedAdd;
  case Schema::LLVMUAddSat:
    return ArithmeticKind::UnsignedAdd;
  case Schema::LLVMSSubSat:
    return ArithmeticKind::SignedSubtract;
  case Schema::LLVMUSubSat:
    return ArithmeticKind::UnsignedSubtract;
  default:
    return invalid("behavior has a non-saturating operation schema");
  }
}

llvm::Expected<LoweredMode> lowerMode(const Mode &mode, Family family) {
  auto arithmetic = arithmeticFor(mode.actor.schema);
  if (!arithmetic)
    return arithmetic.takeError();
  if (mode.actor.type.getNumInputs() != 2 ||
      mode.actor.type.getNumResults() != 1)
    return invalid("behavior has wrong arity");

  mlir::Type valueType = mode.actor.type.getInput(0);
  if (mode.actor.type.getInput(1) != valueType ||
      mode.actor.type.getResult(0) != valueType)
    return invalid("behavior does not have a uniform value type");

  if (family == Family::ScalarIntegerSaturatingAddSub) {
    auto integer = llvm::dyn_cast<mlir::IntegerType>(valueType);
    if (!integer || integer.getWidth() == 0)
      return invalid("scalar behavior does not have an integer type");
    return LoweredMode{*arithmetic, integer.getWidth(), 1};
  }

  auto vector = llvm::dyn_cast<mlir::VectorType>(valueType);
  if (!vector || vector.isScalable() || vector.getRank() == 0)
    return invalid("vector behavior does not have a fixed vector type");
  auto element = llvm::dyn_cast<mlir::IntegerType>(vector.getElementType());
  if (!element || element.getWidth() == 0)
    return invalid("vector behavior does not have integer elements");
  const std::uint64_t laneCount = vector.getNumElements();
  if (laneCount == 0 || laneCount > std::numeric_limits<unsigned>::max())
    return invalid("vector lane count is outside the RTL domain");
  return LoweredMode{*arithmetic, element.getWidth(),
                     static_cast<unsigned>(laneCount)};
}

llvm::Expected<PhysicalPorts>
resolvePhysicalPorts(const FabricOperationProviderRequest &request) {
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
    return invalid("capability does not have the binary integer port shape");
  return PhysicalPorts{inputs[0], inputs[1], outputs[0]};
}

mlir::Value constant(mlir::OpBuilder &builder, mlir::Location location,
                     const llvm::APInt &value) {
  return circt::hw::ConstantOp::create(builder, location, value);
}

mlir::Value widen(mlir::OpBuilder &builder, mlir::Location location,
                  mlir::Value value, unsigned width, bool signedValue) {
  mlir::Value high = signedValue
                         ? circt::comb::ExtractOp::create(builder, location,
                                                          value, width - 1, 1)
                         : constant(builder, location, llvm::APInt(1, 0));
  return circt::comb::ConcatOp::create(builder, location,
                                       mlir::ValueRange{high, value});
}

mlir::Value materializeSignedLane(mlir::OpBuilder &builder,
                                  mlir::Location location, mlir::Value lhs,
                                  mlir::Value rhs, unsigned width,
                                  bool subtract) {
  mlir::Value wideLhs = widen(builder, location, lhs, width, true);
  mlir::Value wideRhs = widen(builder, location, rhs, width, true);
  mlir::Value subtractValue =
      constant(builder, location, llvm::APInt(1, subtract));
  mlir::Value exact =
      detail::addOrSubtract(builder, location, wideLhs, wideRhs, subtractValue);

  const llvm::APInt minimum = llvm::APInt::getSignedMinValue(width);
  const llvm::APInt maximum = llvm::APInt::getSignedMaxValue(width);
  mlir::Value belowMinimum = circt::comb::ICmpOp::create(
      builder, location, circt::comb::ICmpPredicate::slt, exact,
      constant(builder, location, minimum.sext(width + 1)), true);
  mlir::Value aboveMaximum = circt::comb::ICmpOp::create(
      builder, location, circt::comb::ICmpPredicate::sgt, exact,
      constant(builder, location, maximum.sext(width + 1)), true);
  mlir::Value truncated =
      circt::comb::ExtractOp::create(builder, location, exact, 0, width);
  mlir::Value lowerBounded = circt::comb::MuxOp::create(
      builder, location, belowMinimum, constant(builder, location, minimum),
      truncated, true);
  return circt::comb::MuxOp::create(builder, location, aboveMaximum,
                                    constant(builder, location, maximum),
                                    lowerBounded, true);
}

mlir::Value materializeUnsignedLane(mlir::OpBuilder &builder,
                                    mlir::Location location, mlir::Value lhs,
                                    mlir::Value rhs, unsigned width,
                                    bool subtract) {
  mlir::Value subtractValue =
      constant(builder, location, llvm::APInt(1, subtract));
  if (subtract) {
    mlir::Value underflow = circt::comb::ICmpOp::create(
        builder, location, circt::comb::ICmpPredicate::ult, lhs, rhs, true);
    mlir::Value difference =
        detail::addOrSubtract(builder, location, lhs, rhs, subtractValue);
    return circt::comb::MuxOp::create(
        builder, location, underflow,
        constant(builder, location, llvm::APInt(width, 0)), difference, true);
  }

  mlir::Value exact = detail::addOrSubtract(
      builder, location, widen(builder, location, lhs, width, false),
      widen(builder, location, rhs, width, false), subtractValue);
  mlir::Value overflow =
      circt::comb::ExtractOp::create(builder, location, exact, width, 1);
  mlir::Value truncated =
      circt::comb::ExtractOp::create(builder, location, exact, 0, width);
  return circt::comb::MuxOp::create(
      builder, location, overflow,
      constant(builder, location, llvm::APInt::getMaxValue(width)), truncated,
      true);
}

mlir::Value materializeMode(mlir::OpBuilder &builder, mlir::Location location,
                            circt::hw::HWModulePortAccessor &accessor,
                            const LoweredMode &mode, unsigned outputWidth) {
  std::vector<mlir::Value> laneResults;
  laneResults.reserve(mode.laneCount);
  for (unsigned lane = 0; lane < mode.laneCount; ++lane) {
    const unsigned lowBit = lane * mode.elementWidth;
    mlir::Value lhs = circt::comb::ExtractOp::create(
        builder, location, accessor.getInput("data_input_0"), lowBit,
        mode.elementWidth);
    mlir::Value rhs = circt::comb::ExtractOp::create(
        builder, location, accessor.getInput("data_input_1"), lowBit,
        mode.elementWidth);
    laneResults.push_back(
        isSigned(mode.arithmetic)
            ? materializeSignedLane(builder, location, lhs, rhs,
                                    mode.elementWidth,
                                    isSubtract(mode.arithmetic))
            : materializeUnsignedLane(builder, location, lhs, rhs,
                                      mode.elementWidth,
                                      isSubtract(mode.arithmetic)));
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
materializePortableIntegerSaturatingAddSub(
    FabricOperationProviderRequest request, Family expectedFamily) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily != expectedFamily ||
      !isSaturatingFamily(expectedFamily))
    return invalid("provider received a different implementation family");
  if (expectedFamily == Family::ScalarIntegerSaturatingAddSub) {
    if (!std::holds_alternative<::fabric::ScalarIntegerParams>(
            request.capability.parameterizedCapability))
      return invalid("capability has the wrong scalar parameter schema");
  } else if (!std::holds_alternative<::fabric::FixedVectorIntegerParams>(
                 request.capability.parameterizedCapability)) {
    return invalid("capability has the wrong vector parameter schema");
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
    return llvm::make_error<FabricOperationProviderUnsupportedError>(
        request.capability.implementationFamily, request.recipe);

  auto ports = resolvePhysicalPorts(request);
  if (!ports)
    return ports.takeError();
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
      return invalid("configuration-free capability has a field relation");
    if (domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid(
          "configuration-free capability has a non-singleton behavior domain");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid("configured capability field relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured capability requires one field");
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
  loweredModes.reserve(modes.size());
  for (const Mode &mode : modes) {
    auto lowered = lowerMode(mode, expectedFamily);
    if (!lowered)
      return lowered.takeError();
    const std::uint64_t payloadWidth =
        static_cast<std::uint64_t>(lowered->elementWidth) * lowered->laneCount;
    if (payloadWidth > ports->lhs->payloadWidthBits ||
        payloadWidth > ports->rhs->payloadWidthBits ||
        payloadWidth > ports->result->payloadWidthBits)
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
        std::vector<mlir::Value> selectedModes(modes.size());
        if (field) {
          mlir::Value configuration = accessor.getInput(
              "config_" +
              std::to_string(
                  request.capability.configurationFieldSchema.front().ordinal));
          for (std::size_t index = 0; index < modes.size(); ++index) {
            if (index == inactiveMode)
              continue;
            mlir::Value code =
                constant(bodyBuilder, location,
                         detail::decodePhysicalCode(
                             modes[index].codebookEntry->physicalCode,
                             codebook->encodedBitCount));
            selectedModes[index] = circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::eq,
                configuration, code, true);
          }
        }

        std::vector<mlir::Value> results;
        results.reserve(loweredModes.size());
        for (const LoweredMode &mode : loweredModes)
          results.push_back(materializeMode(bodyBuilder, location, accessor,
                                            mode,
                                            ports->result->payloadWidthBits));

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
materializePortableScalarIntegerSaturatingAddSub(
    FabricOperationProviderRequest request) {
  return materializePortableIntegerSaturatingAddSub(
      request, Family::ScalarIntegerSaturatingAddSub);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFixedVectorIntegerSaturatingAddSub(
    FabricOperationProviderRequest request) {
  return materializePortableIntegerSaturatingAddSub(
      request, Family::FixedVectorIntegerSaturatingAddSub);
}

} // namespace

llvm::Error registerPortableIntegerSaturatingAddSubProviders(
    FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry candidate = registry;
  if (llvm::Error error =
          candidate.add({Family::ScalarIntegerSaturatingAddSub,
                         BackendRecipeKey::PortableSystemVerilog,
                         {},
                         materializePortableScalarIntegerSaturatingAddSub}))
    return error;
  if (llvm::Error error = candidate.add(
          {Family::FixedVectorIntegerSaturatingAddSub,
           BackendRecipeKey::PortableSystemVerilog,
           {},
           materializePortableFixedVectorIntegerSaturatingAddSub}))
    return error;
  registry = std::move(candidate);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl
