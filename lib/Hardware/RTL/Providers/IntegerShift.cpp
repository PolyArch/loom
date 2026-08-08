#include "Hardware/RTL/Providers/IntegerShift.h"

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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

using Schema = ::dataflow::OperationSchemaId;

enum class ShiftOperation { Left, LogicalRight, ArithmeticRight };

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  ShiftOperation operation;
  unsigned elementWidth = 0;
  unsigned laneCount = 0;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_integer_shift_invalid: " + message);
}

llvm::Expected<FabricOperationProviderOutput>
unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

llvm::Expected<ShiftOperation> lowerOperation(Schema schema) {
  switch (schema) {
  case Schema::ArithShLI:
    return ShiftOperation::Left;
  case Schema::ArithShRUI:
    return ShiftOperation::LogicalRight;
  case Schema::ArithShRSI:
    return ShiftOperation::ArithmeticRight;
  default:
    return invalid("Fabric returned a non-shift behavior witness");
  }
}

llvm::Expected<LoweredMode> lowerScalarMode(const Mode &mode) {
  if (mode.actor.type.getNumInputs() != 2 ||
      mode.actor.type.getNumResults() != 1)
    return invalid("scalar shift behavior has the wrong arity");
  auto value = llvm::dyn_cast<mlir::IntegerType>(mode.actor.type.getInput(0));
  if (!value || mode.actor.type.getInput(1) != value ||
      mode.actor.type.getResult(0) != value)
    return invalid("scalar shift behavior is not uniformly integer typed");
  auto operation = lowerOperation(mode.actor.schema);
  if (!operation)
    return operation.takeError();
  return LoweredMode{*operation, value.getWidth(), 1};
}

llvm::Expected<LoweredMode> lowerVectorMode(const Mode &mode) {
  if (mode.actor.type.getNumInputs() != 2 ||
      mode.actor.type.getNumResults() != 1)
    return invalid("vector shift behavior has the wrong arity");
  auto vector = llvm::dyn_cast<mlir::VectorType>(mode.actor.type.getInput(0));
  if (!vector || vector.isScalable() || mode.actor.type.getInput(1) != vector ||
      mode.actor.type.getResult(0) != vector)
    return invalid("vector shift behavior is not uniformly vector typed");
  auto element = llvm::dyn_cast<mlir::IntegerType>(vector.getElementType());
  if (!element)
    return invalid("vector shift behavior has a non-integer element");
  const std::uint64_t lanes = vector.getNumElements();
  if (lanes == 0 || lanes > std::numeric_limits<unsigned>::max())
    return invalid("vector shift lane count is outside the RTL domain");
  auto operation = lowerOperation(mode.actor.schema);
  if (!operation)
    return operation.takeError();
  return LoweredMode{*operation, element.getWidth(),
                     static_cast<unsigned>(lanes)};
}

mlir::Value materializeShift(mlir::OpBuilder &builder, mlir::Location location,
                             ShiftOperation operation, mlir::Value value,
                             mlir::Value amount) {
  switch (operation) {
  case ShiftOperation::Left:
    return circt::comb::ShlOp::create(builder, location, value, amount, true);
  case ShiftOperation::LogicalRight:
    return circt::comb::ShrUOp::create(builder, location, value, amount, true);
  case ShiftOperation::ArithmeticRight:
    return circt::comb::ShrSOp::create(builder, location, value, amount, true);
  }
  llvm_unreachable("unknown integer shift operation");
}

mlir::Value materializeScalarMode(mlir::OpBuilder &builder,
                                  mlir::Location location,
                                  circt::hw::HWModulePortAccessor &accessor,
                                  const LoweredMode &mode,
                                  unsigned arithmeticWidth,
                                  unsigned outputWidth) {
  const unsigned operationWidth =
      mode.operation == ShiftOperation::ArithmeticRight ? mode.elementWidth
                                                        : arithmeticWidth;
  mlir::Value value = detail::resizeUnsigned(
      builder, location, accessor.getInput("data_input_0"), operationWidth);
  mlir::Value amount = detail::resizeUnsigned(
      builder, location, accessor.getInput("data_input_1"), operationWidth);
  return detail::resizeUnsigned(
      builder, location,
      materializeShift(builder, location, mode.operation, value, amount),
      outputWidth);
}

mlir::Value materializeVectorMode(mlir::OpBuilder &builder,
                                  mlir::Location location,
                                  circt::hw::HWModulePortAccessor &accessor,
                                  const LoweredMode &mode,
                                  unsigned outputWidth) {
  std::vector<mlir::Value> laneResults;
  laneResults.reserve(mode.laneCount);
  for (unsigned lane = 0; lane < mode.laneCount; ++lane) {
    const unsigned lowBit = lane * mode.elementWidth;
    mlir::Value value = circt::comb::ExtractOp::create(
        builder, location, accessor.getInput("data_input_0"), lowBit,
        mode.elementWidth);
    mlir::Value amount = circt::comb::ExtractOp::create(
        builder, location, accessor.getInput("data_input_1"), lowBit,
        mode.elementWidth);
    laneResults.push_back(
        materializeShift(builder, location, mode.operation, value, amount));
  }
  mlir::Value packed = laneResults.front();
  if (laneResults.size() > 1) {
    std::vector<mlir::Value> highToLow(laneResults.rbegin(),
                                       laneResults.rend());
    packed = circt::comb::ConcatOp::create(builder, location, highToLow);
  }
  return detail::resizeUnsigned(builder, location, packed, outputWidth);
}

llvm::Expected<FabricOperationProviderOutput> materializePortableIntegerShift(
    FabricOperationProviderRequest request,
    ::fabric::ImplementationFamilyId expectedFamily) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily != expectedFamily)
    return invalid("provider received a different implementation family");
  const auto &descriptor = ::fabric::implementationFamily(expectedFamily);
  if (::fabric::capabilityParamsSchema(
          request.capability.parameterizedCapability) !=
      descriptor.capabilityParamsSchema)
    return invalid("capability parameter schema does not match its family");
  const bool isScalar =
      expectedFamily == ::fabric::ImplementationFamilyId::ScalarIntegerShift;
  if (isScalar) {
    const auto *parameters = std::get_if<::fabric::ScalarIntegerParams>(
        &request.capability.parameterizedCapability);
    if (!parameters || !parameters->integerWidths.valid() ||
        parameters->integerWidths.empty() ||
        !parameters->pointerFormats.valid())
      return invalid("capability has malformed scalar integer parameters");
  } else {
    const auto *parameters = std::get_if<::fabric::FixedVectorIntegerParams>(
        &request.capability.parameterizedCapability);
    if (!parameters || !parameters->elementWidths.valid() ||
        parameters->elementWidths.empty() || parameters->maxPayloadBits == 0)
      return invalid("capability has malformed vector integer parameters");
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
      outputs[0]->reference.ordinal != 0)
    return unsupported(request);
  if (inputs[0]->payloadWidthBits == 0 || inputs[1]->payloadWidthBits == 0 ||
      outputs[0]->payloadWidthBits == 0)
    return invalid("capability has a zero-width physical data port");
  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return std::move(error);

  auto relation = request.capability.resolveSemanticFieldRelation(
      *request.leaf.getContext());
  if (!relation)
    return relation.takeError();
  const auto domain = relation->finiteBehaviorDomain();
  if (domain.empty())
    return invalid("Fabric returned an empty shift behavior domain");

  const ConfigurationFieldEncoding *field = nullptr;
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
      return invalid("configured shift relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured shift capability requires one field");
    field = request.configurationAbi.findOperationField(
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

  const unsigned arithmeticWidth =
      std::max({inputs[0]->payloadWidthBits, inputs[1]->payloadWidthBits,
                outputs[0]->payloadWidthBits});
  std::vector<LoweredMode> loweredModes;
  loweredModes.reserve(modes.size());
  for (const Mode &mode : modes) {
    auto lowered = isScalar ? lowerScalarMode(mode) : lowerVectorMode(mode);
    if (!lowered)
      return lowered.takeError();
    const std::uint64_t payloadWidth =
        static_cast<std::uint64_t>(lowered->elementWidth) * lowered->laneCount;
    if (payloadWidth > inputs[0]->payloadWidthBits ||
        payloadWidth > inputs[1]->payloadWidthBits ||
        payloadWidth > outputs[0]->payloadWidthBits)
      return invalid("shift behavior exceeds the physical datapath");
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
        for (const LoweredMode &mode : loweredModes) {
          results.push_back(
              isScalar
                  ? materializeScalarMode(bodyBuilder, location, accessor, mode,
                                          arithmeticWidth,
                                          outputs[0]->payloadWidthBits)
                  : materializeVectorMode(bodyBuilder, location, accessor, mode,
                                          outputs[0]->payloadWidthBits));
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
materializePortableScalarIntegerShift(FabricOperationProviderRequest request) {
  return materializePortableIntegerShift(
      request, ::fabric::ImplementationFamilyId::ScalarIntegerShift);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFixedVectorIntegerShift(
    FabricOperationProviderRequest request) {
  return materializePortableIntegerShift(
      request, ::fabric::ImplementationFamilyId::FixedVectorIntegerShift);
}

} // namespace

llvm::Error registerPortableIntegerShiftProviders(
    FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry candidate = registry;
  if (llvm::Error error =
          candidate.add({::fabric::ImplementationFamilyId::ScalarIntegerShift,
                         BackendRecipeKey::PortableSystemVerilog,
                         {},
                         materializePortableScalarIntegerShift}))
    return error;
  if (llvm::Error error = candidate.add(
          {::fabric::ImplementationFamilyId::FixedVectorIntegerShift,
           BackendRecipeKey::PortableSystemVerilog,
           {},
           materializePortableFixedVectorIntegerShift}))
    return error;
  registry = std::move(candidate);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl
