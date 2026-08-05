#include "Hardware/RTL/Providers/ScalarIntegerCast.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Dataflow/IR/OperationSchemaCodec.h"
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
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

using Schema = ::dataflow::OperationSchemaId;

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  std::optional<::fabric::ResolvedIndexWidth> resolvedIndexWidth;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  unsigned sourceWidth = 0;
  unsigned destinationWidth = 0;
  bool signExtend = false;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_scalar_integer_cast_invalid: " +
                                     message);
}

llvm::Error unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

llvm::Expected<unsigned>
endpointWidth(mlir::Type type,
              std::optional<::fabric::ResolvedIndexWidth> resolvedIndexWidth) {
  if (auto integer = llvm::dyn_cast<mlir::IntegerType>(type)) {
    if (!integer.isSignless() || integer.getWidth() == 0)
      return invalid("cast endpoint is not a nonzero signless integer");
    return integer.getWidth();
  }
  if (!llvm::isa<mlir::IndexType>(type))
    return invalid("cast endpoint is neither integer nor index");
  if (!resolvedIndexWidth)
    return invalid("index cast behavior has no resolved index width");
  return ::fabric::getResolvedIndexBitWidth(*resolvedIndexWidth);
}

llvm::Expected<LoweredMode> lowerMode(const Mode &mode) {
  auto canonical = ::dataflow::encodeCanonicalActorSchemaProjection(mode.actor);
  if (!canonical)
    return canonical.takeError();
  if (mode.actor.type.getNumInputs() != 1 ||
      mode.actor.type.getNumResults() != 1)
    return invalid("cast behavior has wrong arity");

  const mlir::Type sourceType = mode.actor.type.getInput(0);
  const mlir::Type destinationType = mode.actor.type.getResult(0);
  const bool sourceIsIndex = llvm::isa<mlir::IndexType>(sourceType);
  const bool destinationIsIndex = llvm::isa<mlir::IndexType>(destinationType);
  const bool indexSchema = mode.actor.schema == Schema::ArithIndexCast ||
                           mode.actor.schema == Schema::ArithIndexCastUI;
  if (indexSchema) {
    if (sourceIsIndex == destinationIsIndex || !mode.resolvedIndexWidth)
      return invalid(
          "index cast behavior requires one index endpoint and its width");
  } else {
    if (sourceIsIndex || destinationIsIndex || mode.resolvedIndexWidth)
      return invalid("ordinary cast behavior has an index width witness");
    if (mode.actor.schema != Schema::ArithExtSI &&
        mode.actor.schema != Schema::ArithExtUI &&
        mode.actor.schema != Schema::ArithTruncI)
      return invalid("behavior has a non-cast schema");
  }

  auto sourceWidth = endpointWidth(sourceType, mode.resolvedIndexWidth);
  if (!sourceWidth)
    return sourceWidth.takeError();
  auto destinationWidth =
      endpointWidth(destinationType, mode.resolvedIndexWidth);
  if (!destinationWidth)
    return destinationWidth.takeError();

  if (mode.actor.schema == Schema::ArithTruncI &&
      *sourceWidth <= *destinationWidth)
    return invalid("integer truncation behavior does not narrow");
  if ((mode.actor.schema == Schema::ArithExtSI ||
       mode.actor.schema == Schema::ArithExtUI) &&
      *sourceWidth >= *destinationWidth)
    return invalid("integer extension behavior does not widen");

  return LoweredMode{*sourceWidth, *destinationWidth,
                     *destinationWidth > *sourceWidth &&
                         (mode.actor.schema == Schema::ArithExtSI ||
                          mode.actor.schema == Schema::ArithIndexCast)};
}

mlir::Value materializeMode(mlir::OpBuilder &builder, mlir::Location location,
                            mlir::Value input, const LoweredMode &mode,
                            unsigned outputWidth) {
  mlir::Value source =
      detail::resizeUnsigned(builder, location, input, mode.sourceWidth);
  mlir::Value castValue;
  if (!mode.signExtend) {
    castValue = detail::resizeUnsigned(builder, location, source,
                                       mode.destinationWidth);
  } else {
    mlir::Value sign = circt::comb::ExtractOp::create(builder, location, source,
                                                      mode.sourceWidth - 1, 1);
    const unsigned extensionWidth = mode.destinationWidth - mode.sourceWidth;
    mlir::Value high = extensionWidth == 1
                           ? sign
                           : mlir::Value(circt::comb::ReplicateOp::create(
                                 builder, location, sign, extensionWidth));
    castValue = circt::comb::ConcatOp::create(builder, location,
                                              mlir::ValueRange{high, source});
  }
  return detail::resizeUnsigned(builder, location, castValue, outputWidth);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarIntegerCast(FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::ScalarIntegerCast)
    return invalid("provider received a different implementation family");
  if (!std::holds_alternative<::fabric::ScalarIntegerCastParams>(
          request.capability.parameterizedCapability))
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
       request.capability.physicalPorts)
    (port.reference.direction == fabric::FabricPortDirection::Input ? inputs
                                                                    : outputs)
        .push_back(&port);
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(inputs, byOrdinal);
  llvm::sort(outputs, byOrdinal);
  if (inputs.size() != 1 || outputs.size() != 1 ||
      inputs.front()->reference.ordinal != 0 ||
      outputs.front()->reference.ordinal != 0 ||
      inputs.front()->payloadWidthBits == 0 ||
      outputs.front()->payloadWidthBits == 0)
    return unsupported(request);
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
    modes.push_back({std::move(domain->front().representativeActor),
                     domain->front().resolvedIndexWidth, nullptr});
  } else {
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured cast capability requires one field");
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
      modes.push_back({std::move(point.representativeActor),
                       point.resolvedIndexWidth, entry});
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
    if (lowered->sourceWidth > inputs.front()->payloadWidthBits ||
        lowered->destinationWidth > outputs.front()->payloadWidthBits)
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
        std::vector<mlir::Value> results;
        results.reserve(loweredModes.size());
        mlir::Value input = accessor.getInput("data_input_0");
        for (const LoweredMode &mode : loweredModes)
          results.push_back(materializeMode(bodyBuilder, location, input, mode,
                                            outputs.front()->payloadWidthBits));

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

llvm::Error registerPortableScalarIntegerCastProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::ScalarIntegerCast,
                       BackendRecipeKey::PortableSystemVerilog,
                       {},
                       materializePortableScalarIntegerCast});
}

} // namespace loom::hardware::rtl
