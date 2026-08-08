#include "Hardware/RTL/Providers/FixedVectorSliceAlignMerge.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

using Schema = ::dataflow::OperationSchemaId;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "portable_fixed_vector_slice_align_merge_invalid: " + message);
}

llvm::Expected<FabricOperationProviderOutput>
unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

bool hasSchema(llvm::ArrayRef<Schema> schemas, Schema expected) {
  return llvm::is_contained(schemas, expected);
}

mlir::Value extractField(mlir::OpBuilder &builder, mlir::Location location,
                         mlir::Value configuration, std::uint32_t offset,
                         std::uint32_t width) {
  return circt::comb::ExtractOp::create(builder, location, configuration,
                                        offset, width);
}

mlir::Value constant(mlir::OpBuilder &builder, mlir::Location location,
                     unsigned width, std::uint64_t value) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(width, value));
}

mlir::Value materializeEffectiveOffset(
    mlir::OpBuilder &builder, mlir::Location location,
    circt::hw::HWModulePortAccessor &accessor, mlir::Value configuration,
    const ::fabric::FixedVectorSliceAlignMergeConfigurationLayout &layout,
    unsigned arithmeticWidth) {
  mlir::Value offset = detail::resizeUnsigned(
      builder, location,
      extractField(builder, location, configuration,
                   layout.staticOffsetBitOffset, layout.offsetBitCount),
      arithmeticWidth);
  for (std::uint32_t ordinal = 0; ordinal != layout.dynamicStrideCount;
       ++ordinal) {
    mlir::Value stride = detail::resizeUnsigned(
        builder, location,
        extractField(builder, location, configuration,
                     layout.dynamicStrideBitOffset +
                         ordinal * layout.dynamicStrideBitCount,
                     layout.dynamicStrideBitCount),
        arithmeticWidth);
    mlir::Value index = detail::resizeUnsigned(
        builder, location,
        accessor.getInput("data_input_" + std::to_string(ordinal + 2)),
        arithmeticWidth);
    mlir::Value scaled = circt::comb::MulOp::create(
        builder, location, mlir::ValueRange{index, stride}, true);
    offset = circt::comb::AddOp::create(builder, location,
                                        mlir::ValueRange{offset, scaled}, true);
  }
  return offset;
}

mlir::Value materializeSliceWidth(
    mlir::OpBuilder &builder, mlir::Location location,
    mlir::Value configuration,
    const ::fabric::FixedVectorSliceAlignMergeConfigurationLayout &layout,
    unsigned arithmeticWidth) {
  mlir::Value encoded = detail::resizeUnsigned(
      builder, location,
      extractField(builder, location, configuration, layout.sliceWidthBitOffset,
                   layout.sliceWidthBitCount),
      arithmeticWidth);
  return circt::comb::AddOp::create(
      builder, location,
      mlir::ValueRange{encoded,
                       constant(builder, location, arithmeticWidth, 1)},
      true);
}

mlir::Value materializeLowMask(mlir::OpBuilder &builder,
                               mlir::Location location, mlir::Value sliceWidth,
                               unsigned workWidth) {
  mlir::Value one = constant(builder, location, workWidth, 1);
  mlir::Value shifted = circt::comb::ShlOp::create(
      builder, location, one,
      detail::resizeUnsigned(builder, location, sliceWidth, workWidth), true);
  return circt::comb::SubOp::create(builder, location, shifted, one, true);
}

mlir::Value materializeExtract(mlir::OpBuilder &builder,
                               mlir::Location location, mlir::Value source,
                               mlir::Value offset, mlir::Value lowMask,
                               unsigned workWidth) {
  mlir::Value aligned = circt::comb::ShrUOp::create(
      builder, location,
      detail::resizeUnsigned(builder, location, source, workWidth),
      detail::resizeUnsigned(builder, location, offset, workWidth), true);
  return circt::comb::AndOp::create(builder, location, aligned, lowMask, true);
}

mlir::Value materializeInsert(mlir::OpBuilder &builder, mlir::Location location,
                              mlir::Value inserted, mlir::Value destination,
                              mlir::Value offset, mlir::Value lowMask,
                              unsigned workWidth) {
  mlir::Value shift =
      detail::resizeUnsigned(builder, location, offset, workWidth);
  mlir::Value positionedMask =
      circt::comb::ShlOp::create(builder, location, lowMask, shift, true);
  mlir::Value insertedLow = circt::comb::AndOp::create(
      builder, location,
      detail::resizeUnsigned(builder, location, inserted, workWidth), lowMask,
      true);
  mlir::Value positionedValue =
      circt::comb::ShlOp::create(builder, location, insertedLow, shift, true);
  mlir::Value allOnes = circt::hw::ConstantOp::create(
      builder, location, llvm::APInt::getAllOnes(workWidth));
  mlir::Value inverseMask = circt::comb::XorOp::create(
      builder, location, positionedMask, allOnes, true);
  mlir::Value preserved = circt::comb::AndOp::create(
      builder, location,
      detail::resizeUnsigned(builder, location, destination, workWidth),
      inverseMask, true);
  return circt::comb::OrOp::create(builder, location, preserved,
                                   positionedValue, true);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFixedVectorSliceAlignMerge(
    FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::FixedVectorSliceAlignMerge)
    return invalid("provider received a different implementation family");
  const auto *parameters =
      std::get_if<::fabric::FixedVectorSliceAlignMergeParams>(
          &request.capability.parameterizedCapability);
  if (!parameters)
    return invalid("capability has the wrong parameter schema");

  auto relation = request.capability.resolveSemanticFieldRelation(
      *request.leaf.getContext());
  if (!relation)
    return relation.takeError();
  const auto *layout = relation->fixedVectorSliceAlignMergeLayout();
  if (!layout)
    return invalid("slice semantic field relation has no exact layout");
  const bool hasConfigurationField = relation->hasConfigurationField();
  if ((hasConfigurationField &&
       relation->kind() !=
           ::fabric::FabricOpSemanticFieldRelationKind::Direct) ||
      (!hasConfigurationField &&
       relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None))
    return invalid("slice semantic field relation has the wrong kind");
  if (hasConfigurationField &&
      (!relation->directEncodedBitCount() ||
       *relation->directEncodedBitCount() != layout->encodedBitCount))
    return invalid("slice semantic field relation has the wrong bit count");
  if (request.capability.configurationFieldSchema.size() !=
      static_cast<std::size_t>(hasConfigurationField))
    return invalid("capability semantic field does not match its layout");

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
  const std::size_t expectedInputs =
      2 + static_cast<std::size_t>(parameters->maxDynamicPositionRank);
  if (inputs.size() != expectedInputs || outputs.size() != 1 ||
      outputs.front()->reference.ordinal != 0 ||
      outputs.front()->payloadWidthBits == 0)
    return unsupported(request);
  for (std::size_t ordinal = 0; ordinal != inputs.size(); ++ordinal)
    if (inputs[ordinal]->reference.ordinal != ordinal ||
        inputs[ordinal]->payloadWidthBits == 0)
      return unsupported(request);

  unsigned requiredIndexWidth = 0;
  if (parameters->resolvedIndexWidths.contains(
          ::fabric::ResolvedIndexWidth::I32))
    requiredIndexWidth = 32;
  if (parameters->resolvedIndexWidths.contains(
          ::fabric::ResolvedIndexWidth::I64))
    requiredIndexWidth = 64;
  for (const auto *position : llvm::ArrayRef(inputs).drop_front(2))
    if (position->payloadWidthBits < requiredIndexWidth)
      return unsupported(request);

  const bool extract = hasSchema(request.capability.enabledOperationSchemas,
                                 Schema::VectorExtract);
  const bool insert = hasSchema(request.capability.enabledOperationSchemas,
                                Schema::VectorInsert);
  if ((extract &&
       (inputs[0]->payloadWidthBits < parameters->maxContainerPayloadBits ||
        outputs[0]->payloadWidthBits < parameters->maxSlicePayloadBits)) ||
      (insert &&
       (inputs[0]->payloadWidthBits < parameters->maxSlicePayloadBits ||
        inputs[1]->payloadWidthBits < parameters->maxContainerPayloadBits ||
        outputs[0]->payloadWidthBits < parameters->maxContainerPayloadBits)))
    return unsupported(request);

  const ConfigurationFieldEncoding *field = nullptr;
  if (hasConfigurationField) {
    field = request.configurationAbi.findOperationField(
        request.occurrence,
        request.capability.configurationFieldSchema.front().ordinal);
    if (!field)
      return invalid("configured field is absent from the ABI");
    const auto *direct =
        std::get_if<DirectBitsEncoding>(&field->semanticEncoding);
    if (!direct)
      return invalid("configured field is not DirectBits");
    if (direct->encodedBitCount != *relation->directEncodedBitCount())
      return invalid("DirectBits width does not match the resolved layout");
  }

  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return std::move(error);

  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        const unsigned arithmeticWidth =
            std::max({layout->offsetBitCount, layout->sliceWidthBitCount + 1,
                      layout->dynamicStrideBitCount});
        mlir::Value configuration;
        mlir::Value offset;
        mlir::Value sliceWidth;
        if (field) {
          configuration = accessor.getInput(
              "config_" +
              std::to_string(
                  request.capability.configurationFieldSchema.front().ordinal));
          offset = materializeEffectiveOffset(bodyBuilder, location, accessor,
                                              configuration, *layout,
                                              arithmeticWidth);
          sliceWidth = materializeSliceWidth(
              bodyBuilder, location, configuration, *layout, arithmeticWidth);
        } else {
          offset = constant(bodyBuilder, location, arithmeticWidth, 0);
          sliceWidth = constant(bodyBuilder, location, arithmeticWidth, 1);
        }
        const unsigned workWidth =
            std::max({inputs[0]->payloadWidthBits, inputs[1]->payloadWidthBits,
                      outputs[0]->payloadWidthBits});
        mlir::Value lowMask =
            materializeLowMask(bodyBuilder, location, sliceWidth, workWidth);
        mlir::Value result;
        if (layout->encodesMode) {
          mlir::Value extracted = materializeExtract(
              bodyBuilder, location, accessor.getInput("data_input_0"), offset,
              lowMask, workWidth);
          mlir::Value inserted = materializeInsert(
              bodyBuilder, location, accessor.getInput("data_input_0"),
              accessor.getInput("data_input_1"), offset, lowMask, workWidth);
          mlir::Value mode = extractField(bodyBuilder, location, configuration,
                                          layout->modeBitOffset, 1);
          result = circt::comb::MuxOp::create(bodyBuilder, location, mode,
                                              inserted, extracted, true);
        } else if (insert) {
          result = materializeInsert(
              bodyBuilder, location, accessor.getInput("data_input_0"),
              accessor.getInput("data_input_1"), offset, lowMask, workWidth);
        } else {
          result = materializeExtract(bodyBuilder, location,
                                      accessor.getInput("data_input_0"), offset,
                                      lowMask, workWidth);
        }
        accessor.setOutput("data_output_0", detail::resizeUnsigned(
                                                bodyBuilder, location, result,
                                                outputs[0]->payloadWidthBits));
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

} // namespace

llvm::Error registerPortableFixedVectorSliceAlignMergeProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add(
      {::fabric::ImplementationFamilyId::FixedVectorSliceAlignMerge,
       BackendRecipeKey::PortableSystemVerilog,
       {},
       materializePortableFixedVectorSliceAlignMerge});
}

} // namespace loom::hardware::rtl
