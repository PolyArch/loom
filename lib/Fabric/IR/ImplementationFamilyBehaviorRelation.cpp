//===- ImplementationFamilyBehaviorRelation.cpp -------------------------===//
//
// Derives and validates the semantic-field relation of one concrete Fabric
// operation capability.
//
//===----------------------------------------------------------------------===//

#include "Fabric/IR/ImplementationFamily.h"
#include "ImplementationFamilyBehaviorInternal.h"
#include "ImplementationFamilyFixedBehavior.h"
#include "ImplementationFamilyScalarFloatBehavior.h"
#include "ImplementationFamilyScalarIntegerBehavior.h"
#include "ImplementationFamilySpecialMath.h"
#include "ImplementationFamilyVectorIntegerBehavior.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace fabric;

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

enum class FiniteBehaviorRelationOwner : std::uint8_t {
  Generic,
  Fixed,
  ScalarFloat,
  ScalarInteger,
  FixedVectorInteger,
  Control,
  SpecialMath,
};

FiniteBehaviorRelationOwner
finiteBehaviorRelationOwner(ImplementationFamilyId family) {
  if (detail::ownsFixedBehaviorRelation(family))
    return FiniteBehaviorRelationOwner::Fixed;
  if (detail::ownsScalarFloatBehaviorRelation(family))
    return FiniteBehaviorRelationOwner::ScalarFloat;
  if (detail::ownsScalarIntegerBehaviorRelation(family))
    return FiniteBehaviorRelationOwner::ScalarInteger;
  if (detail::ownsFixedVectorIntegerBehaviorRelation(family))
    return FiniteBehaviorRelationOwner::FixedVectorInteger;
  if (detail::ownsControlBehaviorRelation(family))
    return FiniteBehaviorRelationOwner::Control;
  if (detail::ownsScalarSpecialMathBehaviorRelation(family))
    return FiniteBehaviorRelationOwner::SpecialMath;
  return FiniteBehaviorRelationOwner::Generic;
}

llvm::Error validatePackedShape(llvm::ArrayRef<std::uint8_t> value,
                                std::uint32_t bitCount) {
  const std::uint64_t byteCount =
      (static_cast<std::uint64_t>(bitCount) + 7) / 8;
  if (value.size() != byteCount)
    return reject("semantic value has the wrong byte count");
  const unsigned usedFinalBits = bitCount % 8;
  if (usedFinalBits != 0 && (value.back() >> usedFinalBits) != 0)
    return reject("semantic value has nonzero high padding bits");
  return llvm::Error::success();
}

std::uint64_t readPackedField(llvm::ArrayRef<std::uint8_t> value,
                              std::uint32_t offset, std::uint32_t width) {
  std::uint64_t result = 0;
  for (std::uint32_t bit = 0; bit != width; ++bit)
    if (((value[(offset + bit) / 8] >> ((offset + bit) % 8)) & 1U) != 0)
      result |= std::uint64_t{1} << bit;
  return result;
}

void writePackedBit(std::vector<std::uint8_t> &value, std::uint64_t offset,
                    bool bit) {
  if (bit)
    value[offset / 8] |= static_cast<std::uint8_t>(1U << (offset % 8));
}

bool admitsElementWidth(const IntegerWidthSet &integers,
                        const FloatFormatSet &floats, std::uint64_t width) {
  for (IntegerWidth integer : integerWidthDomain)
    if (integers.contains(integer) && width % getBitWidth(integer) == 0)
      return true;
  for (FloatFormat format : floatFormatDomain)
    if (floats.contains(format) && width % getBitWidth(format) == 0)
      return true;
  return false;
}

llvm::Error validatePhysicalCapacity(
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<std::uint64_t> operandPorts,
    llvm::ArrayRef<std::uint64_t> resultPorts,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    std::optional<ResolvedIndexWidth> resolvedIndexWidth) {
  if (operandPorts.size() != actor.type.getNumInputs() ||
      resultPorts.size() != actor.type.getNumResults())
    return reject("actor and physical port correspondence arity disagree");

  ::dataflow::CanonicalActorSchemaProjection represented = actor;
  if (resolvedIndexWidth) {
    auto projected = projectResolvedIndexTypes(
        actor, getResolvedIndexBitWidth(*resolvedIndexWidth));
    if (!projected)
      return projected.takeError();
    represented = std::move(*projected);
  }

  const auto verify = [](llvm::ArrayRef<::mlir::Type> types,
                         llvm::ArrayRef<std::uint64_t> ports,
                         llvm::ArrayRef<std::uint32_t> widths,
                         llvm::StringRef direction) -> llvm::Error {
    for (auto [type, port] : llvm::zip(types, ports)) {
      if (port >= widths.size())
        return reject("physical " + direction + " port is missing");
      std::string message;
      auto semanticWidth = getSemanticPayloadWidth(type, message);
      if (::mlir::failed(semanticWidth))
        return reject(message);
      if (*semanticWidth > widths[port])
        return reject("physical " + direction +
                      " port is narrower than the represented actor");
    }
    return llvm::Error::success();
  };
  if (llvm::Error error = verify(represented.type.getInputs(), operandPorts,
                                 physicalInputWidths, "input"))
    return error;
  return verify(represented.type.getResults(), resultPorts,
                physicalResultWidths, "result");
}

llvm::Error
validateSlice(llvm::ArrayRef<std::uint8_t> value,
              const FixedVectorSliceAlignMergeParams &params,
              llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
              llvm::ArrayRef<std::uint32_t> physicalInputWidths,
              llvm::ArrayRef<std::uint32_t> physicalResultWidths,
              const FixedVectorSliceAlignMergeConfigurationLayout &layout) {
  const std::uint64_t mode =
      layout.encodesMode ? readPackedField(value, layout.modeBitOffset, 1)
                         : (enabledSchemas.front() ==
                                    ::dataflow::OperationSchemaId::VectorInsert
                                ? 1
                                : 0);
  if (mode > 1 ||
      (mode == 0 &&
       !llvm::is_contained(enabledSchemas,
                           ::dataflow::OperationSchemaId::VectorExtract)) ||
      (mode == 1 &&
       !llvm::is_contained(enabledSchemas,
                           ::dataflow::OperationSchemaId::VectorInsert)))
    return reject("vector slice mode is not enabled");

  const std::uint64_t offset = readPackedField(
      value, layout.staticOffsetBitOffset, layout.offsetBitCount);
  const std::uint64_t sliceWidth =
      readPackedField(value, layout.sliceWidthBitOffset,
                      layout.sliceWidthBitCount) +
      1;
  if (sliceWidth > params.maxSlicePayloadBits ||
      !admitsElementWidth(params.integerElementWidths,
                          params.floatElementFormats, sliceWidth))
    return reject("vector slice width is outside the admitted element domain");
  if (offset % sliceWidth != 0)
    return reject("vector slice offset is not aligned to its width");

  std::vector<std::uint64_t> strides;
  bool reachedPadding = false;
  for (std::uint32_t ordinal = 0; ordinal != layout.dynamicStrideCount;
       ++ordinal) {
    const std::uint64_t stride = readPackedField(
        value,
        layout.dynamicStrideBitOffset + ordinal * layout.dynamicStrideBitCount,
        layout.dynamicStrideBitCount);
    if (stride == 0) {
      reachedPadding = true;
      continue;
    }
    if (reachedPadding)
      return reject("vector slice strides are not a nonzero prefix");
    if (stride > params.maxContainerPayloadBits)
      return reject("vector slice stride exceeds container capacity");
    strides.push_back(stride);
  }
  for (std::size_t ordinal = 1; ordinal != strides.size(); ++ordinal)
    if (strides[ordinal - 1] % strides[ordinal] != 0)
      return reject("vector slice strides are not row-major divisible");
  if (!strides.empty() && strides.back() % sliceWidth != 0)
    return reject("vector slice innermost stride is not slice-aligned");

  const std::uint64_t granularity =
      strides.empty() ? sliceWidth : strides.front();
  auto minimumContainer =
      llvm::checkedMulUnsigned(offset / granularity + 1, granularity);
  if (!minimumContainer)
    return reject("vector slice minimum container width overflows");
  if (physicalInputWidths.size() < 2 + params.maxDynamicPositionRank ||
      physicalResultWidths.empty())
    return reject("vector slice physical role inventory is incomplete");

  const std::uint32_t dynamicCount = static_cast<std::uint32_t>(strides.size());
  if (dynamicCount != 0) {
    bool commonIndexWidth = false;
    for (ResolvedIndexWidth indexWidth : resolvedIndexWidthDomain) {
      const unsigned bits = getResolvedIndexBitWidth(indexWidth);
      if (!params.resolvedIndexWidths.contains(indexWidth))
        continue;
      const bool fits = llvm::all_of(
          physicalInputWidths.slice(2, dynamicCount),
          [&](std::uint32_t physicalWidth) { return bits <= physicalWidth; });
      commonIndexWidth |= fits;
    }
    if (!commonIndexWidth)
      return reject("vector slice dynamic indices have no common width");
  }

  if (mode == 0) {
    if (*minimumContainer > params.maxContainerPayloadBits ||
        *minimumContainer > physicalInputWidths[0])
      return reject("vector slice container is not physically reachable");
    if (sliceWidth > physicalResultWidths[0])
      return reject("vector slice result is not physically reachable");
  } else {
    if (sliceWidth > physicalInputWidths[0])
      return reject("vector slice input is not physically reachable");
    if (*minimumContainer > params.maxContainerPayloadBits ||
        *minimumContainer > physicalInputWidths[1] ||
        *minimumContainer > physicalResultWidths[0])
      return reject("vector slice container is not physically reachable");
  }
  return llvm::Error::success();
}

llvm::Error
validateShuffle(llvm::ArrayRef<std::uint8_t> value,
                const FixedVectorShuffleParams &params,
                llvm::ArrayRef<std::uint32_t> physicalInputWidths,
                llvm::ArrayRef<std::uint32_t> physicalResultWidths,
                const FixedVectorShuffleConfigurationLayout &layout) {
  if (physicalInputWidths.size() < 2 || physicalResultWidths.empty())
    return reject("vector shuffle physical role inventory is incomplete");
  const std::uint64_t blockWidth =
      readPackedField(value, layout.blockWidthBitOffset,
                      layout.blockWidthBitCount) +
      1;
  const std::uint64_t leftBlocks =
      readPackedField(value, layout.leftBlockCountBitOffset,
                      layout.blockCountBitCount) +
      1;
  const std::uint64_t resultBlocks =
      readPackedField(value, layout.resultBlockCountBitOffset,
                      layout.resultBlockCountBitCount) +
      1;
  if (blockWidth > params.maxBlockPayloadBits ||
      !admitsElementWidth(params.integerElementWidths,
                          params.floatElementFormats, blockWidth))
    return reject("vector shuffle block width is outside its domain");
  if (leftBlocks >= params.maxSourceBlocks)
    return reject("vector shuffle left block count exhausts the source");
  if (resultBlocks > params.maxResultBlocks)
    return reject("vector shuffle result block count exceeds capacity");

  std::uint64_t maximumSelector = 0;
  for (std::uint32_t ordinal = 0; ordinal != layout.selectorCount; ++ordinal) {
    const std::uint64_t selector = readPackedField(
        value, layout.selectorBitOffset + ordinal * layout.selectorBitCount,
        layout.selectorBitCount);
    if (ordinal >= resultBlocks && selector != 0)
      return reject("vector shuffle padding selector is nonzero");
    if (ordinal < resultBlocks && selector >= params.maxSourceBlocks)
      return reject("vector shuffle selector is outside the source domain");
    if (ordinal < resultBlocks)
      maximumSelector = std::max(maximumSelector, selector);
  }
  const std::uint64_t sourceBlocks =
      std::max(leftBlocks + 1, maximumSelector + 1);
  if (sourceBlocks > params.maxSourceBlocks)
    return reject("vector shuffle source geometry exceeds capacity");
  const std::uint64_t rightBlocks = sourceBlocks - leftBlocks;

  const auto leftWidth = llvm::checkedMulUnsigned(leftBlocks, blockWidth);
  const auto rightWidth = llvm::checkedMulUnsigned(rightBlocks, blockWidth);
  const auto resultWidth = llvm::checkedMulUnsigned(resultBlocks, blockWidth);
  if (!leftWidth || !rightWidth || !resultWidth)
    return reject("vector shuffle geometry overflows");
  if (*leftWidth > params.maxOperandPayloadBits ||
      *leftWidth > physicalInputWidths[0] ||
      *rightWidth > params.maxOperandPayloadBits ||
      *rightWidth > physicalInputWidths[1])
    return reject("vector shuffle operand geometry is not reachable");
  if (*resultWidth > params.maxResultPayloadBits ||
      *resultWidth > physicalResultWidths[0])
    return reject("vector shuffle result geometry is not reachable");
  return llvm::Error::success();
}

llvm::Expected<::loom::CanonicalSemanticBytes>
projectConstant(const ::dataflow::CanonicalActorSchemaProjection &actor,
                std::uint32_t bitCount) {
  const auto *constant =
      std::get_if<::dataflow::ConstantValuePayload>(&actor.payload);
  if (!constant || !constant->value)
    return reject("constant projection contains no typed value");
  std::vector<std::uint8_t> bytes((bitCount + 7) / 8, 0);
  std::uint64_t cursor = 0;
  const auto append = [&](const llvm::APInt &bits) -> llvm::Error {
    if (cursor + bits.getBitWidth() > bitCount)
      return reject("constant value exceeds its direct carrier");
    for (unsigned bit = 0; bit != bits.getBitWidth(); ++bit)
      writePackedBit(bytes, cursor + bit, bits[bit]);
    cursor += bits.getBitWidth();
    return llvm::Error::success();
  };

  if (auto integer = llvm::dyn_cast<::mlir::IntegerAttr>(constant->value)) {
    if (llvm::Error error = append(integer.getValue()))
      return std::move(error);
  } else if (auto floating =
                 llvm::dyn_cast<::mlir::FloatAttr>(constant->value)) {
    if (llvm::Error error = append(floating.getValue().bitcastToAPInt()))
      return std::move(error);
  } else if (auto integers = llvm::dyn_cast<::mlir::DenseIntElementsAttr>(
                 constant->value)) {
    for (const llvm::APInt &integer : integers)
      if (llvm::Error error = append(integer))
        return std::move(error);
  } else if (auto floats =
                 llvm::dyn_cast<::mlir::DenseFPElementsAttr>(constant->value)) {
    for (const llvm::APFloat &floating : floats)
      if (llvm::Error error = append(floating.bitcastToAPInt()))
        return std::move(error);
  } else {
    return reject("constant value is outside the closed direct projector");
  }
  return ::loom::CanonicalSemanticBytes(std::move(bytes));
}

} // namespace

mlir::arith::FastMathFlags fabric::detail::minimalFloatingActorPermissions(
    const FloatBehaviorProfile &behavior) {
  using Bits = std::underlying_type_t<mlir::arith::FastMathFlags>;
  Bits flags = static_cast<Bits>(behavior.requiredFastMath);
  if (!behavior.nanBehaviors.contains(FloatNaNBehavior::IEEE))
    flags |= static_cast<Bits>(mlir::arith::FastMathFlags::nnan);
  if (!behavior.signedZeroBehaviors.contains(FloatSignedZeroBehavior::Preserve))
    flags |= static_cast<Bits>(mlir::arith::FastMathFlags::nsz);
  return static_cast<mlir::arith::FastMathFlags>(flags);
}

llvm::Error fabric::detail::validateImplementationFamilyBehaviorPoint(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<std::uint64_t> operandPorts,
    llvm::ArrayRef<std::uint64_t> resultPorts,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    std::optional<ResolvedIndexWidth> resolvedIndexWidth) {
  if (llvm::Error error = verifyImplementationFamilyPortCorrespondence(
          family, actor, operandPorts, resultPorts))
    return error;
  const TypedAdmissionProviderId provider =
      implementationFamily(family).typedAdmissionProvider;
  const bool routedSelector =
      provider == TypedAdmissionProviderId::MuxTokenAdmission ||
      provider == TypedAdmissionProviderId::DemuxTokenAdmission;
  if (resolvedIndexWidth && !routedSelector) {
    if (llvm::Error error = verifyImplementationFamilyAdmission(
            family, &params, actor,
            getResolvedIndexBitWidth(*resolvedIndexWidth)))
      return error;
  } else if (llvm::Error error =
                 verifyImplementationFamilyAdmission(family, &params, actor)) {
    return error;
  }
  return validatePhysicalCapacity(actor, operandPorts, resultPorts,
                                  physicalInputWidths, physicalResultWidths,
                                  resolvedIndexWidth);
}

llvm::Error fabric::FabricOpSemanticFieldRelation::validateSemanticValue(
    llvm::ArrayRef<std::uint8_t> value) const {
  if (kind_ == FabricOpSemanticFieldRelationKind::None)
    return reject("capability has no semantic value domain");
  if (kind_ == FabricOpSemanticFieldRelationKind::Finite) {
    if (llvm::any_of(finiteBehaviorDomain_, [&](const auto &point) {
          return point.semanticConfiguration &&
                 point.semanticConfiguration->bytes().equals(value);
        }))
      return llvm::Error::success();
    return reject("semantic value is outside the finite behavior domain");
  }

  if (llvm::Error error = validatePackedShape(value, directEncodedBitCount_))
    return error;
  if (family_ == ImplementationFamilyId::TokenConstant)
    return llvm::Error::success();
  if (family_ == ImplementationFamilyId::FixedVectorSliceAlignMerge)
    return validateSlice(value,
                         std::get<FixedVectorSliceAlignMergeParams>(params_),
                         enabledSchemas_, physicalInputWidths_,
                         physicalResultWidths_, *sliceLayout_);
  if (family_ == ImplementationFamilyId::FixedVectorShuffle)
    return validateShuffle(value, std::get<FixedVectorShuffleParams>(params_),
                           physicalInputWidths_, physicalResultWidths_,
                           *shuffleLayout_);
  return reject("direct behavior domain is not registered");
}

llvm::Expected<::loom::CanonicalSemanticBytes>
fabric::FabricOpSemanticFieldRelation::projectSemanticValue(
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<std::uint64_t> operandPorts,
    llvm::ArrayRef<std::uint64_t> resultPorts,
    std::optional<ResolvedIndexWidth> resolvedIndexWidth) const {
  if (kind_ == FabricOpSemanticFieldRelationKind::None)
    return reject("capability has no semantic field");
  if (!llvm::is_contained(enabledSchemas_, actor.schema))
    return reject("actor schema is not enabled by the concrete capability");
  auto canonicalActor = ::dataflow::encodeCanonicalActorSchemaProjection(actor);
  if (!canonicalActor)
    return canonicalActor.takeError();
  if (llvm::Error error = detail::validateImplementationFamilyBehaviorPoint(
          family_, params_, actor, operandPorts, resultPorts,
          physicalInputWidths_, physicalResultWidths_, resolvedIndexWidth))
    return std::move(error);

  auto projected = [&]() -> llvm::Expected<::loom::CanonicalSemanticBytes> {
    if (family_ == ImplementationFamilyId::TokenConstant)
      return projectConstant(actor, directEncodedBitCount_);
    switch (finiteBehaviorRelationOwner(family_)) {
    case FiniteBehaviorRelationOwner::Fixed:
      return reject("fixed behavior relation has no semantic field");
    case FiniteBehaviorRelationOwner::ScalarFloat:
      return detail::projectScalarFloatBehavior(family_, actor,
                                                finiteBehaviorDomain_);
    case FiniteBehaviorRelationOwner::ScalarInteger:
      return detail::projectScalarIntegerBehavior(
          family_, actor, resolvedIndexWidth, finiteBehaviorDomain_);
    case FiniteBehaviorRelationOwner::FixedVectorInteger:
      return detail::projectFixedVectorIntegerBehavior(family_, actor,
                                                       finiteBehaviorDomain_);
    case FiniteBehaviorRelationOwner::Control:
      return detail::projectControlBehaviorKey(
          family_, finiteBehaviorDomain_, actor, operandPorts, resultPorts);
    case FiniteBehaviorRelationOwner::SpecialMath:
      return detail::projectScalarSpecialMathBehavior(family_, actor,
                                                      finiteBehaviorDomain_);
    case FiniteBehaviorRelationOwner::Generic:
      return encodeImplementationFamilySemanticConfiguration(
          family_, params_, enabledSchemas_, physicalInputWidths_.size(),
          physicalResultWidths_.size(), actor, operandPorts, resultPorts,
          resolvedIndexWidth);
    }
    llvm_unreachable("unhandled finite behavior relation owner");
  }();
  if (!projected)
    return projected.takeError();
  if (llvm::Error error = validateSemanticValue(projected->bytes()))
    return std::move(error);
  return std::move(*projected);
}

llvm::Expected<fabric::FabricOpSemanticFieldRelation>
fabric::resolveFabricOpSemanticFieldRelation(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    ::mlir::MLIRContext &context) {
  if (llvm::is_contained(enabledSchemas,
                         ::dataflow::OperationSchemaId::LLVMGetElementPtr))
    return reject("GEP behavior relation requires canonical integer address "
                  "normalization");

  const FiniteBehaviorRelationOwner owner = finiteBehaviorRelationOwner(family);
  bool genericNeedsField = true;
  if (owner == FiniteBehaviorRelationOwner::Generic) {
    auto needsField = requiresSemanticConfigurationField(
        family, params, enabledSchemas, physicalInputWidths.size(),
        physicalResultWidths.size());
    if (!needsField)
      return needsField.takeError();
    genericNeedsField = *needsField;
  }

  std::optional<FixedVectorSliceAlignMergeConfigurationLayout> sliceLayout;
  std::optional<FixedVectorShuffleConfigurationLayout> shuffleLayout;
  std::uint32_t directBitCount = 0;
  bool direct = false;
  if (family == ImplementationFamilyId::TokenConstant) {
    const auto *constant = std::get_if<PayloadCapacityParams>(&params);
    if (!constant || physicalResultWidths.empty())
      return reject("constant direct carrier has no physical result");
    directBitCount =
        std::min(constant->maxPayloadBits, physicalResultWidths.front());
    if (directBitCount == 0)
      return reject("constant direct carrier has zero width");
    direct = true;
  } else if (family == ImplementationFamilyId::FixedVectorSliceAlignMerge) {
    auto resolvedLayout = resolveFixedVectorSliceAlignMergeConfigurationLayout(
        std::get<FixedVectorSliceAlignMergeParams>(params), enabledSchemas);
    if (!resolvedLayout)
      return resolvedLayout.takeError();
    sliceLayout = std::move(*resolvedLayout);
    directBitCount = sliceLayout->encodedBitCount;
    direct = true;
  } else if (family == ImplementationFamilyId::FixedVectorShuffle) {
    auto resolvedLayout = resolveFixedVectorShuffleConfigurationLayout(
        std::get<FixedVectorShuffleParams>(params));
    if (!resolvedLayout)
      return resolvedLayout.takeError();
    shuffleLayout = std::move(*resolvedLayout);
    directBitCount = shuffleLayout->encodedBitCount;
    direct = true;
  }

  if (direct)
    return FabricOpSemanticFieldRelation(
        FabricOpSemanticFieldRelationKind::Direct, family, params,
        std::vector<::dataflow::OperationSchemaId>(enabledSchemas.begin(),
                                                   enabledSchemas.end()),
        std::vector<std::uint32_t>(physicalInputWidths.begin(),
                                   physicalInputWidths.end()),
        std::vector<std::uint32_t>(physicalResultWidths.begin(),
                                   physicalResultWidths.end()),
        {}, directBitCount, std::move(sliceLayout), std::move(shuffleLayout));

  if (!genericNeedsField)
    return FabricOpSemanticFieldRelation(
        FabricOpSemanticFieldRelationKind::None, family, params,
        std::vector<::dataflow::OperationSchemaId>(enabledSchemas.begin(),
                                                   enabledSchemas.end()),
        std::vector<std::uint32_t>(physicalInputWidths.begin(),
                                   physicalInputWidths.end()),
        std::vector<std::uint32_t>(physicalResultWidths.begin(),
                                   physicalResultWidths.end()),
        {}, 0, std::nullopt, std::nullopt);

  auto domain = [&]()
      -> llvm::Expected<std::vector<FiniteImplementationFamilyBehaviorPoint>> {
    switch (owner) {
    case FiniteBehaviorRelationOwner::Fixed:
      return detail::resolveFixedBehaviorDomain(family, params, enabledSchemas,
                                                physicalInputWidths,
                                                physicalResultWidths, context);
    case FiniteBehaviorRelationOwner::ScalarFloat:
      return detail::resolveScalarFloatBehaviorDomain(
          family, params, enabledSchemas, physicalInputWidths,
          physicalResultWidths, context);
    case FiniteBehaviorRelationOwner::ScalarInteger:
      return detail::resolveScalarIntegerBehaviorDomain(
          family, params, enabledSchemas, physicalInputWidths,
          physicalResultWidths, context);
    case FiniteBehaviorRelationOwner::FixedVectorInteger:
      return detail::resolveFixedVectorIntegerBehaviorDomain(
          family, params, enabledSchemas, physicalInputWidths,
          physicalResultWidths, context);
    case FiniteBehaviorRelationOwner::Control:
      return detail::resolveControlBehaviorDomain(
          family, params, enabledSchemas, physicalInputWidths,
          physicalResultWidths, context);
    case FiniteBehaviorRelationOwner::SpecialMath:
      return detail::resolveScalarSpecialMathBehaviorDomain(
          family, params, enabledSchemas, physicalInputWidths,
          physicalResultWidths, context);
    case FiniteBehaviorRelationOwner::Generic:
      return resolveFiniteImplementationFamilyBehaviorDomain(
          family, params, enabledSchemas, physicalInputWidths.size(),
          physicalResultWidths.size(), context,
          [](const ::dataflow::CanonicalActorSchemaProjection &,
             std::optional<ResolvedIndexWidth>) {
            return llvm::Error::success();
          });
    }
    llvm_unreachable("unhandled finite behavior relation owner");
  }();
  if (!domain)
    return domain.takeError();

  const bool domainIsPhysicallyFiltered =
      owner != FiniteBehaviorRelationOwner::Generic;
  std::vector<FiniteImplementationFamilyBehaviorPoint> reachable;
  std::string firstPhysicalRejection;
  for (auto &point : *domain) {
    if (!domainIsPhysicallyFiltered) {
      if (llvm::Error error = validatePhysicalCapacity(
              point.representativeActor, point.operandPorts, point.resultPorts,
              physicalInputWidths, physicalResultWidths,
              point.resolvedIndexWidth)) {
        if (firstPhysicalRejection.empty())
          firstPhysicalRejection = llvm::toString(std::move(error));
        else
          llvm::consumeError(std::move(error));
        continue;
      }
    }
    reachable.push_back(std::move(point));
  }
  if (reachable.empty())
    return reject(firstPhysicalRejection.empty()
                      ? "concrete capability has no reachable behavior"
                      : firstPhysicalRejection);
  llvm::sort(reachable, [](const auto &lhs, const auto &rhs) {
    if (!lhs.semanticConfiguration)
      return rhs.semanticConfiguration.has_value();
    if (!rhs.semanticConfiguration)
      return false;
    return std::lexicographical_compare(
        lhs.semanticConfiguration->bytes().begin(),
        lhs.semanticConfiguration->bytes().end(),
        rhs.semanticConfiguration->bytes().begin(),
        rhs.semanticConfiguration->bytes().end());
  });
  reachable.erase(
      std::unique(reachable.begin(), reachable.end(),
                  [](const auto &lhs, const auto &rhs) {
                    return lhs.semanticConfiguration &&
                           rhs.semanticConfiguration &&
                           lhs.semanticConfiguration->bytes().equals(
                               rhs.semanticConfiguration->bytes());
                  }),
      reachable.end());
  FabricOpSemanticFieldRelationKind kind =
      FabricOpSemanticFieldRelationKind::Finite;
  if (reachable.size() == 1) {
    kind = FabricOpSemanticFieldRelationKind::None;
    reachable.front().semanticConfiguration = std::nullopt;
  }
  return FabricOpSemanticFieldRelation(
      kind, family, params,
      std::vector<::dataflow::OperationSchemaId>(enabledSchemas.begin(),
                                                 enabledSchemas.end()),
      std::vector<std::uint32_t>(physicalInputWidths.begin(),
                                 physicalInputWidths.end()),
      std::vector<std::uint32_t>(physicalResultWidths.begin(),
                                 physicalResultWidths.end()),
      std::move(reachable), 0, std::nullopt, std::nullopt);
}
