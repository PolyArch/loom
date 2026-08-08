#include "Hardware/RTL/Providers/FloatSign.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"
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

enum class SignOperation { Negate, Absolute };

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  SignOperation operation;
  unsigned elementWidth = 0;
  unsigned laneCount = 0;

  unsigned payloadWidth() const { return elementWidth * laneCount; }
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_float_sign_invalid: " + message);
}

llvm::Expected<FabricOperationProviderOutput>
unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

bool hasExpectedParameterSchema(
    const fabric::ResolvedFabricOpCapabilityView &capability,
    ::fabric::ImplementationFamilyId family) {
  if (family == ::fabric::ImplementationFamilyId::ScalarFloatSign)
    return std::holds_alternative<::fabric::ScalarFloatParams>(
        capability.parameterizedCapability);
  return std::holds_alternative<::fabric::FixedVectorFloatParams>(
      capability.parameterizedCapability);
}

llvm::Expected<SignOperation>
signOperation(::dataflow::OperationSchemaId schema) {
  switch (schema) {
  case ::dataflow::OperationSchemaId::ArithNegF:
    return SignOperation::Negate;
  case ::dataflow::OperationSchemaId::MathAbsF:
    return SignOperation::Absolute;
  default:
    return invalid("Fabric returned a non-sign behavior witness");
  }
}

llvm::Expected<LoweredMode> lowerMode(const Mode &mode,
                                      ::fabric::ImplementationFamilyId family) {
  auto operation = signOperation(mode.actor.schema);
  if (!operation)
    return operation.takeError();
  if (mode.actor.type.getNumInputs() != 1 ||
      mode.actor.type.getNumResults() != 1)
    return invalid("behavior has wrong arity");
  if (!std::holds_alternative<::dataflow::FloatingPointPayload>(
          mode.actor.payload))
    return invalid("behavior has no typed floating payload");

  mlir::Type input = mode.actor.type.getInput(0);
  if (mode.actor.type.getResult(0) != input)
    return invalid("behavior result type differs from its input");

  mlir::Type element = input;
  std::uint64_t laneCount = 1;
  if (family == ::fabric::ImplementationFamilyId::FixedVectorFloatSign) {
    auto vector = llvm::dyn_cast<mlir::VectorType>(input);
    if (!vector || vector.isScalable() || vector.getRank() == 0 ||
        llvm::any_of(vector.getShape(),
                     [](std::int64_t extent) { return extent <= 0; }))
      return invalid("behavior does not have a fixed positive vector type");
    element = vector.getElementType();
    laneCount = vector.getNumElements();
  } else if (llvm::isa<mlir::VectorType>(input)) {
    return invalid("scalar behavior has a vector type");
  }

  auto floating = llvm::dyn_cast<mlir::FloatType>(element);
  if (!floating || floating.getWidth() == 0)
    return invalid("behavior element is not a supported floating type");
  if (laneCount == 0 || laneCount > std::numeric_limits<unsigned>::max() ||
      laneCount > std::numeric_limits<unsigned>::max() / floating.getWidth())
    return invalid("behavior payload width exceeds the RTL domain");
  return LoweredMode{*operation, floating.getWidth(),
                     static_cast<unsigned>(laneCount)};
}

mlir::Value materializeMode(mlir::OpBuilder &builder, mlir::Location location,
                            mlir::Value input, const LoweredMode &mode,
                            unsigned outputWidth) {
  const unsigned payloadWidth = mode.payloadWidth();
  input = detail::resizeUnsigned(builder, location, input, payloadWidth);
  llvm::APInt signMask(payloadWidth, 0);
  for (unsigned lane = 0; lane != mode.laneCount; ++lane)
    signMask.setBit(lane * mode.elementWidth + mode.elementWidth - 1);
  mlir::Value result;
  if (mode.operation == SignOperation::Negate)
    result = circt::comb::XorOp::create(
        builder, location, input,
        circt::hw::ConstantOp::create(builder, location, signMask));
  else
    result = circt::comb::AndOp::create(
        builder, location, input,
        circt::hw::ConstantOp::create(builder, location, ~signMask));
  return detail::resizeUnsigned(builder, location, result, outputWidth);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFloatSign(FabricOperationProviderRequest request,
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
    return invalid("capability does not have the unary sign port shape");

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
      return invalid("configured floating sign relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured floating sign capability requires one field");
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

  std::vector<LoweredMode> loweredModes;
  loweredModes.reserve(modes.size());
  for (const Mode &mode : modes) {
    auto lowered = lowerMode(mode, expectedFamily);
    if (!lowered)
      return lowered.takeError();
    if (lowered->payloadWidth() > inputs[0]->payloadWidthBits ||
        lowered->payloadWidth() > outputs[0]->payloadWidthBits)
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
          results.push_back(materializeMode(
              bodyBuilder, location, accessor.getInput("data_input_0"), mode,
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
materializePortableScalarFloatSign(FabricOperationProviderRequest request) {
  return materializePortableFloatSign(
      request, ::fabric::ImplementationFamilyId::ScalarFloatSign);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFixedVectorFloatSign(
    FabricOperationProviderRequest request) {
  return materializePortableFloatSign(
      request, ::fabric::ImplementationFamilyId::FixedVectorFloatSign);
}

} // namespace

llvm::Error
registerPortableFloatSignProviders(FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry candidate = registry;
  if (llvm::Error error =
          candidate.add({::fabric::ImplementationFamilyId::ScalarFloatSign,
                         BackendRecipeKey::PortableSystemVerilog,
                         {},
                         materializePortableScalarFloatSign}))
    return error;
  if (llvm::Error error =
          candidate.add({::fabric::ImplementationFamilyId::FixedVectorFloatSign,
                         BackendRecipeKey::PortableSystemVerilog,
                         {},
                         materializePortableFixedVectorFloatSign}))
    return error;
  registry = std::move(candidate);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl
