#include "Hardware/RTL/Providers/ScalarIntegerCompareMinMax.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
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
using Predicate = ::mlir::arith::CmpIPredicate;

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct LoweredMode final {
  circt::comb::ICmpPredicate predicate;
  std::optional<unsigned> signedWidth;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "portable_scalar_integer_compare_min_max_invalid: " + message);
}

bool isSignedPredicate(Predicate predicate) {
  return predicate == Predicate::slt || predicate == Predicate::sle ||
         predicate == Predicate::sgt || predicate == Predicate::sge;
}

llvm::Expected<circt::comb::ICmpPredicate> combPredicate(Predicate predicate) {
  using CombPredicate = circt::comb::ICmpPredicate;
  switch (predicate) {
  case Predicate::eq:
    return CombPredicate::eq;
  case Predicate::ne:
    return CombPredicate::ne;
  case Predicate::slt:
    return CombPredicate::slt;
  case Predicate::sle:
    return CombPredicate::sle;
  case Predicate::sgt:
    return CombPredicate::sgt;
  case Predicate::sge:
    return CombPredicate::sge;
  case Predicate::ult:
    return CombPredicate::ult;
  case Predicate::ule:
    return CombPredicate::ule;
  case Predicate::ugt:
    return CombPredicate::ugt;
  case Predicate::uge:
    return CombPredicate::uge;
  }
  return invalid("comparison predicate is outside the closed domain");
}

llvm::Expected<Predicate> modePredicate(const Mode &mode) {
  switch (mode.actor.schema) {
  case Schema::ArithCmpI: {
    const auto *payload =
        std::get_if<::dataflow::IntegerComparePayload>(&mode.actor.payload);
    if (!payload)
      return invalid("comparison mode has no typed predicate");
    return payload->predicate;
  }
  case Schema::ArithMinSI:
    return Predicate::slt;
  case Schema::ArithMaxSI:
    return Predicate::sgt;
  case Schema::ArithMinUI:
    return Predicate::ult;
  case Schema::ArithMaxUI:
    return Predicate::ugt;
  default:
    return invalid("mode has a non-compare/min-max schema");
  }
}

llvm::Expected<std::optional<unsigned>> signedOperandWidth(const Mode &mode) {
  auto predicate = modePredicate(mode);
  if (!predicate)
    return predicate.takeError();
  if (!isSignedPredicate(*predicate))
    return std::optional<unsigned>{};
  return std::optional<unsigned>(
      llvm::cast<mlir::IntegerType>(mode.actor.type.getInput(0)).getWidth());
}

mlir::Value signExtendLow(mlir::OpBuilder &builder, mlir::Location location,
                          mlir::Value value, unsigned semanticWidth,
                          unsigned physicalWidth) {
  if (semanticWidth == physicalWidth)
    return value;
  mlir::Value low = circt::comb::ExtractOp::create(builder, location, value, 0,
                                                   semanticWidth);
  mlir::Value sign = circt::comb::ExtractOp::create(builder, location, low,
                                                    semanticWidth - 1, 1);
  const unsigned extensionWidth = physicalWidth - semanticWidth;
  mlir::Value high = extensionWidth == 1
                         ? sign
                         : circt::comb::ReplicateOp::create(
                               builder, location, sign, extensionWidth);
  return circt::comb::ConcatOp::create(builder, location,
                                       mlir::ValueRange{high, low});
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarIntegerCompareMinMax(
    FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::ScalarIntegerCompareMinMax)
    return invalid("provider received a different implementation family");

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
      return invalid(
          "configured compare/min/max capability requires one field");
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

  const unsigned arithmeticWidth =
      std::max({inputs[0]->payloadWidthBits, inputs[1]->payloadWidthBits,
                outputs[0]->payloadWidthBits});
  std::vector<LoweredMode> loweredModes;
  loweredModes.reserve(modes.size());
  for (const Mode &mode : modes) {
    auto predicate = modePredicate(mode);
    if (!predicate)
      return predicate.takeError();
    auto loweredPredicate = combPredicate(*predicate);
    if (!loweredPredicate)
      return loweredPredicate.takeError();
    auto width = signedOperandWidth(mode);
    if (!width)
      return width.takeError();
    if (*width && **width > arithmeticWidth)
      return invalid("signed operand width exceeds the physical datapath");
    loweredModes.push_back({*loweredPredicate, *width});
  }

  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        mlir::Value lhs = detail::resizeUnsigned(
            bodyBuilder, location, accessor.getInput("data_input_0"),
            arithmeticWidth);
        mlir::Value rhs = detail::resizeUnsigned(
            bodyBuilder, location, accessor.getInput("data_input_1"),
            arithmeticWidth);
        std::vector<mlir::Value> results;
        results.reserve(modes.size());
        for (auto [index, mode] : llvm::enumerate(modes)) {
          mlir::Value compareLhs = lhs;
          mlir::Value compareRhs = rhs;
          if (std::optional<unsigned> width = loweredModes[index].signedWidth) {
            compareLhs = signExtendLow(bodyBuilder, location, lhs, *width,
                                       arithmeticWidth);
            compareRhs = signExtendLow(bodyBuilder, location, rhs, *width,
                                       arithmeticWidth);
          }
          mlir::Value condition = circt::comb::ICmpOp::create(
              bodyBuilder, location, loweredModes[index].predicate, compareLhs,
              compareRhs, true);
          mlir::Value result = condition;
          if (mode.actor.schema != Schema::ArithCmpI)
            result = circt::comb::MuxOp::create(bodyBuilder, location,
                                                condition, lhs, rhs, true);
          results.push_back(detail::resizeUnsigned(
              bodyBuilder, location, result, outputs[0]->payloadWidthBits));
        }

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

llvm::Error registerPortableScalarIntegerCompareMinMaxProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add(
      {::fabric::ImplementationFamilyId::ScalarIntegerCompareMinMax,
       BackendRecipeKey::PortableSystemVerilog,
       {},
       materializePortableScalarIntegerCompareMinMax});
}

} // namespace loom::hardware::rtl
