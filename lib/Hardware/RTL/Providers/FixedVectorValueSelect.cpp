#include "Hardware/RTL/Providers/FixedVectorValueSelect.h"

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

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  unsigned elementWidth = 0;
  unsigned laneCount = 0;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "portable_fixed_vector_value_select_invalid: " + message);
}

llvm::Expected<LoweredMode> lowerMode(const Mode &mode) {
  if (mode.actor.schema != Schema::ArithSelect)
    return invalid("behavior has a non-select schema");
  if (mode.actor.type.getNumInputs() != 3 ||
      mode.actor.type.getNumResults() != 1)
    return invalid("behavior has wrong arity");
  auto values = llvm::dyn_cast<mlir::VectorType>(mode.actor.type.getInput(1));
  if (!values || mode.actor.type.getInput(2) != values ||
      mode.actor.type.getResult(0) != values)
    return invalid("behavior does not have a uniform value vector type");
  auto condition =
      llvm::dyn_cast<mlir::VectorType>(mode.actor.type.getInput(0));
  if (!condition || condition.getShape() != values.getShape() ||
      !condition.getElementType().isInteger(1))
    return invalid("behavior condition is not matching vector<i1>");
  if (!llvm::isa<mlir::IntegerType, mlir::FloatType>(values.getElementType()))
    return invalid("behavior value vector has an unsupported element type");
  const std::uint64_t lanes = values.getNumElements();
  if (lanes == 0 || lanes > std::numeric_limits<unsigned>::max())
    return invalid("behavior lane count is outside the RTL domain");
  const unsigned elementWidth = values.getElementTypeBitWidth();
  if (elementWidth == 0)
    return invalid("behavior element width is zero");
  return LoweredMode{elementWidth, static_cast<unsigned>(lanes)};
}

mlir::Value materializeMode(mlir::OpBuilder &builder, mlir::Location location,
                            circt::hw::HWModulePortAccessor &accessor,
                            const LoweredMode &mode, unsigned outputWidth) {
  std::vector<mlir::Value> laneResults;
  laneResults.reserve(mode.laneCount);
  for (unsigned lane = 0; lane < mode.laneCount; ++lane) {
    const unsigned lowBit = lane * mode.elementWidth;
    mlir::Value condition = circt::comb::ExtractOp::create(
        builder, location, accessor.getInput("data_input_0"), lane, 1);
    mlir::Value trueValue = circt::comb::ExtractOp::create(
        builder, location, accessor.getInput("data_input_1"), lowBit,
        mode.elementWidth);
    mlir::Value falseValue = circt::comb::ExtractOp::create(
        builder, location, accessor.getInput("data_input_2"), lowBit,
        mode.elementWidth);
    laneResults.push_back(circt::comb::MuxOp::create(
        builder, location, condition, trueValue, falseValue, true));
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
materializePortableFixedVectorValueSelect(
    FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::FixedVectorValueSelect)
    return invalid("provider received a different implementation family");
  if (!std::holds_alternative<::fabric::FixedVectorValueSelectParams>(
          request.capability.parameterizedCapability))
    return invalid("capability has the wrong parameter schema");
  if (request.capability.enabledOperationSchemas !=
      std::vector<Schema>{Schema::ArithSelect})
    return invalid("capability does not contain exactly arith.select");

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
  if (inputs.size() != 3 || outputs.size() != 1 ||
      inputs[0]->reference.ordinal != 0 || inputs[1]->reference.ordinal != 1 ||
      inputs[2]->reference.ordinal != 2 || outputs[0]->reference.ordinal != 0 ||
      inputs[0]->payloadWidthBits == 0 || inputs[1]->payloadWidthBits == 0 ||
      inputs[2]->payloadWidthBits == 0 || outputs[0]->payloadWidthBits == 0)
    return llvm::make_error<FabricOperationProviderUnsupportedError>(
        request.capability.implementationFamily, request.recipe);

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
      return invalid("configured vector select capability requires one field");
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
    const std::uint64_t payloadWidth =
        static_cast<std::uint64_t>(lowered->elementWidth) * lowered->laneCount;
    if (lowered->laneCount > inputs[0]->payloadWidthBits ||
        payloadWidth > inputs[1]->payloadWidthBits ||
        payloadWidth > inputs[2]->payloadWidthBits ||
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

llvm::Error registerPortableFixedVectorValueSelectProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::FixedVectorValueSelect,
                       BackendRecipeKey::PortableSystemVerilog,
                       {},
                       materializePortableFixedVectorValueSelect});
}

} // namespace loom::hardware::rtl
