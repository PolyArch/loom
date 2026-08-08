#include "Hardware/RTL/Providers/FixedVectorFloatFma.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

using Format = detail::PortableFloatFormat;

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  Format format;
  mlir::arith::RoundingMode rounding;
  unsigned laneCount;
  std::string functionName;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_fixed_vector_float_fma_invalid: " +
                                     message);
}

llvm::Error unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

llvm::Expected<LoweredMode> lowerMode(const Mode &mode) {
  if (mode.actor.schema != ::dataflow::OperationSchemaId::MathFma ||
      mode.actor.type.getNumInputs() != 3 ||
      mode.actor.type.getNumResults() != 1)
    return invalid("behavior is not a fixed-vector floating FMA");
  mlir::Type type = mode.actor.type.getInput(0);
  if (mode.actor.type.getInput(1) != type ||
      mode.actor.type.getInput(2) != type ||
      mode.actor.type.getResult(0) != type)
    return invalid("behavior does not have a uniform floating type");
  const auto *payload =
      std::get_if<::dataflow::FloatingPointPayload>(&mode.actor.payload);
  if (!payload)
    return invalid("behavior has no floating-point payload");
  auto vector = llvm::dyn_cast<mlir::VectorType>(type);
  if (!vector || vector.isScalable() || vector.getRank() == 0 ||
      vector.getNumElements() == 0 ||
      vector.getNumElements() > std::numeric_limits<unsigned>::max())
    return invalid("behavior has an invalid fixed-vector shape");
  auto format = detail::resolvePortableFloatFormat(vector.getElementType());
  if (!format)
    return invalid("behavior uses an unsupported floating format");
  return LoweredMode{*format,
                     payload->roundingMode.value_or(
                         mlir::arith::RoundingMode::to_nearest_even),
                     static_cast<unsigned>(vector.getNumElements()),
                     "loom_fixed_vector_fma_e" +
                         std::to_string(format->exponentBits) + "_f" +
                         std::to_string(format->fractionBits)};
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFixedVectorFloatFma(FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::FixedVectorFloatFma)
    return invalid("provider received a different implementation family");
  if (!std::holds_alternative<::fabric::FixedVectorFloatParams>(
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
  if (inputs.size() != 3 || outputs.size() != 1 ||
      inputs[0]->reference.ordinal != 0 || inputs[1]->reference.ordinal != 1 ||
      inputs[2]->reference.ordinal != 2 || outputs[0]->reference.ordinal != 0 ||
      inputs[0]->payloadWidthBits == 0 || inputs[1]->payloadWidthBits == 0 ||
      inputs[2]->payloadWidthBits == 0 || outputs[0]->payloadWidthBits == 0)
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
  const ConfigurationFieldEncoding *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  std::vector<Mode> modes;
  if (request.capability.configurationFieldSchema.empty()) {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None)
      return invalid("configuration-free FMA relation is not fieldless");
    if (domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid(
          "configuration-free capability has a non-singleton behavior domain");
    modes.push_back({domain.front().representativeActor, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid("configured FMA semantic field relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid(
          "configured fixed-vector FMA capability requires one field");
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
    auto lowered = lowerMode(mode);
    if (!lowered)
      return lowered.takeError();
    if (lowered->rounding != mlir::arith::RoundingMode::to_nearest_even)
      return unsupported(request);
    if (!functionNames.insert(lowered->functionName).second)
      return invalid("sealed relation contains a duplicate FMA mode");
    const std::uint64_t activeWidth =
        static_cast<std::uint64_t>(lowered->format.width()) *
        lowered->laneCount;
    if (llvm::any_of(inputs,
                     [=](const auto *port) {
                       return port->payloadWidthBits < activeWidth;
                     }) ||
        outputs[0]->payloadWidthBits < activeWidth)
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
          declarationStream << detail::buildPortableFloatFmaFunction(
                                   mode.format, mode.functionName)
                            << '\n';
        circt::sv::VerbatimOp::create(
            bodyBuilder, location,
            bodyBuilder.getStringAttr(declarationStream.str()));

        std::vector<mlir::Value> results;
        results.reserve(loweredModes.size());
        for (const LoweredMode &mode : loweredModes) {
          const unsigned elementWidth = mode.format.width();
          std::vector<mlir::Value> laneResults;
          laneResults.reserve(mode.laneCount);
          for (unsigned lane = 0; lane < mode.laneCount; ++lane) {
            llvm::SmallVector<mlir::Value, 3> arguments;
            const unsigned lowBit = lane * elementWidth;
            for (unsigned index = 0; index < 3; ++index)
              arguments.push_back(circt::comb::ExtractOp::create(
                  bodyBuilder, location,
                  accessor.getInput("data_input_" + std::to_string(index)),
                  lowBit, elementWidth));
            laneResults.push_back(circt::sv::VerbatimExprOp::create(
                bodyBuilder, location, bodyBuilder.getIntegerType(elementWidth),
                mode.functionName + "({{0}}, {{1}}, {{2}})", arguments));
          }
          mlir::Value value = laneResults.front();
          if (laneResults.size() > 1) {
            std::vector<mlir::Value> highToLow(laneResults.rbegin(),
                                               laneResults.rend());
            value =
                circt::comb::ConcatOp::create(bodyBuilder, location, highToLow);
          }
          results.push_back(detail::resizeUnsigned(
              bodyBuilder, location, value, outputs[0]->payloadWidthBits));
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

} // namespace

llvm::Error registerPortableFixedVectorFloatFmaProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::FixedVectorFloatFma,
                       BackendRecipeKey::PortableSystemVerilog,
                       {},
                       materializePortableFixedVectorFloatFma});
}

} // namespace loom::hardware::rtl
