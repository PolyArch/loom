#include "Hardware/RTL/Providers/FixedVectorShuffle.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/ImplementationFamily.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

using Schema = ::dataflow::OperationSchemaId;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_fixed_vector_shuffle_invalid: " +
                                     message);
}

llvm::Error unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

mlir::Value constant(mlir::OpBuilder &builder, mlir::Location location,
                     unsigned width, std::uint64_t value) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(width, value));
}

mlir::Value extractConfiguration(mlir::OpBuilder &builder,
                                 mlir::Location location,
                                 mlir::Value configuration,
                                 std::uint32_t offset, std::uint32_t bitCount,
                                 unsigned arithmeticWidth) {
  mlir::Value field = circt::comb::ExtractOp::create(
      builder, location, configuration, offset, bitCount);
  return detail::resizeUnsigned(builder, location, field, arithmeticWidth);
}

void materializeShuffleNetwork(
    mlir::OpBuilder &builder, mlir::Location location,
    circt::hw::HWModulePortAccessor &accessor,
    const ConfigurationFieldEncoding &field,
    const ::fabric::FixedVectorShuffleConfigurationLayout &layout,
    unsigned outputWidth, unsigned arithmeticWidth) {
  mlir::Value configuration =
      accessor.getInput("config_" + std::to_string(field.field.ordinal));
  mlir::Value zero = constant(builder, location, arithmeticWidth, 0);
  mlir::Value one = constant(builder, location, arithmeticWidth, 1);
  mlir::Value blockWidth = circt::comb::AddOp::create(
      builder, location,
      extractConfiguration(builder, location, configuration,
                           layout.blockWidthBitOffset,
                           layout.blockWidthBitCount, arithmeticWidth),
      one, true);
  mlir::Value leftBlockCount = circt::comb::AddOp::create(
      builder, location,
      extractConfiguration(builder, location, configuration,
                           layout.leftBlockCountBitOffset,
                           layout.blockCountBitCount, arithmeticWidth),
      one, true);
  mlir::Value resultBlockCount = circt::comb::AddOp::create(
      builder, location,
      extractConfiguration(builder, location, configuration,
                           layout.resultBlockCountBitOffset,
                           layout.resultBlockCountBitCount, arithmeticWidth),
      one, true);
  mlir::Value blockMask = circt::comb::SubOp::create(
      builder, location,
      circt::comb::ShlOp::create(builder, location, one, blockWidth), one);
  mlir::Value left = detail::resizeUnsigned(
      builder, location, accessor.getInput("data_input_0"), arithmeticWidth);
  mlir::Value right = detail::resizeUnsigned(
      builder, location, accessor.getInput("data_input_1"), arithmeticWidth);

  mlir::Value result = zero;
  for (std::uint32_t ordinal = 0; ordinal != layout.selectorCount; ++ordinal) {
    mlir::Value selector = extractConfiguration(
        builder, location, configuration,
        layout.selectorBitOffset + ordinal * layout.selectorBitCount,
        layout.selectorBitCount, arithmeticWidth);
    mlir::Value fromLeft = circt::comb::ICmpOp::create(
        builder, location, circt::comb::ICmpPredicate::ult, selector,
        leftBlockCount, true);
    mlir::Value active = circt::comb::ICmpOp::create(
        builder, location, circt::comb::ICmpPredicate::ult,
        constant(builder, location, arithmeticWidth, ordinal), resultBlockCount,
        true);
    mlir::Value leftOffset = circt::comb::MulOp::create(
        builder, location, selector, blockWidth, true);
    mlir::Value rightOrdinal =
        circt::comb::SubOp::create(builder, location, selector, leftBlockCount);
    mlir::Value rightOffset = circt::comb::MulOp::create(
        builder, location, rightOrdinal, blockWidth, true);
    mlir::Value leftBlock =
        circt::comb::ShrUOp::create(builder, location, left, leftOffset);
    mlir::Value rightBlock =
        circt::comb::ShrUOp::create(builder, location, right, rightOffset);
    mlir::Value selected = circt::comb::MuxOp::create(
        builder, location, fromLeft, leftBlock, rightBlock, true);
    selected = circt::comb::AndOp::create(builder, location, selected,
                                          blockMask, true);
    selected = circt::comb::MuxOp::create(builder, location, active, selected,
                                          zero, true);
    mlir::Value destinationOffset = circt::comb::MulOp::create(
        builder, location, blockWidth,
        constant(builder, location, arithmeticWidth, ordinal), true);
    mlir::Value placed = circt::comb::ShlOp::create(builder, location, selected,
                                                    destinationOffset);
    result = circt::comb::OrOp::create(builder, location, result, placed, true);
  }
  accessor.setOutput(
      "data_output_0",
      detail::resizeUnsigned(builder, location, result, outputWidth));
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFixedVectorShuffle(FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::FixedVectorShuffle)
    return invalid("provider received a different implementation family");
  const auto *parameters = std::get_if<::fabric::FixedVectorShuffleParams>(
      &request.capability.parameterizedCapability);
  if (!parameters)
    return invalid("capability has the wrong parameter schema");
  if (request.capability.enabledOperationSchemas !=
      std::vector<Schema>{Schema::VectorShuffle})
    return invalid("capability does not contain exactly vector.shuffle");

  auto layout =
      ::fabric::resolveFixedVectorShuffleConfigurationLayout(*parameters);
  if (!layout)
    return layout.takeError();

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
  if (parameters->maxOperandPayloadBits > inputs[0]->payloadWidthBits ||
      parameters->maxOperandPayloadBits > inputs[1]->payloadWidthBits ||
      parameters->maxResultPayloadBits > outputs[0]->payloadWidthBits)
    return unsupported(request);

  if (request.capability.configurationFieldSchema.size() != 1)
    return invalid("shuffle capability does not contain exactly one field");
  const ConfigurationFieldEncoding *field = request.configurationAbi.findField(
      request.capability.configurationFieldSchema.front());
  if (!field)
    return invalid("configured field is absent from the ABI");
  const auto *direct =
      std::get_if<DirectBitsEncoding>(&field->semanticEncoding);
  if (!direct)
    return invalid("shuffle configuration field is not DirectBits");
  if (direct->encodedBitCount != layout->encodedBitCount)
    return invalid("shuffle DirectBits field has the wrong bit count");

  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.capability, request.configurationAbi))
    return std::move(error);

  const unsigned arithmeticWidth =
      std::max({inputs[0]->payloadWidthBits, inputs[1]->payloadWidthBits,
                outputs[0]->payloadWidthBits, layout->blockWidthBitCount,
                layout->blockCountBitCount, layout->selectorBitCount});
  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        materializeShuffleNetwork(bodyBuilder, location, accessor, *field,
                                  *layout, outputs[0]->payloadWidthBits,
                                  arithmeticWidth);
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

} // namespace

llvm::Error registerPortableFixedVectorShuffleProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::FixedVectorShuffle,
                       BackendRecipeKey::PortableSystemVerilog,
                       {},
                       materializePortableFixedVectorShuffle});
}

} // namespace loom::hardware::rtl
