#include "Frontend/Compilation/OwnershipCandidateGenerator.h"

#include "Fabric/IR/ImplementationFamily.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::frontend {
namespace {

constexpr llvm::StringLiteral decisionSchema =
    "loom.spatial_ownership.decision.1.1";

enum class AddressProjectionTag : std::uint8_t {
  None = 0,
  RootRelative = 1,
  PointerAddressed = 2,
};

enum class ForallShapeTag : std::uint8_t {
  None = 0,
  GraphParallel = 1,
  LogicalThreadDomain = 2,
};

enum class CallSpecializationTag : std::uint8_t {
  None = 0,
  UniformExactConstants = 1,
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "spatial_ownership_decision_invalid: " +
                                     message);
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t readU32(llvm::ArrayRef<std::uint8_t> bytes) {
  std::uint32_t value = 0;
  for (std::uint8_t byte : bytes)
    value = (value << 8) | byte;
  return value;
}

bool isSupportedIndexWidth(std::uint32_t width) {
  return llvm::any_of(::fabric::resolvedIndexWidthDomain,
                      [&](::fabric::ResolvedIndexWidth candidate) {
                        return ::fabric::getResolvedIndexBitWidth(candidate) ==
                               width;
                      });
}

AddressProjectionTag
addressTag(const std::optional<SpatialAddressProjection> &projection) {
  if (!projection)
    return AddressProjectionTag::None;
  if (std::holds_alternative<RootRelativeAddressProjection>(*projection))
    return AddressProjectionTag::RootRelative;
  return AddressProjectionTag::PointerAddressed;
}

llvm::Expected<ForallShapeTag>
forallTag(const std::optional<ForallOwnershipShape> &shape) {
  if (!shape)
    return ForallShapeTag::None;
  switch (*shape) {
  case ForallOwnershipShape::GraphParallel:
    return ForallShapeTag::GraphParallel;
  case ForallOwnershipShape::LogicalThreadDomain:
    return ForallShapeTag::LogicalThreadDomain;
  }
  return invalid("decision has an unknown forall ownership shape");
}

llvm::Expected<CallSpecializationTag>
callTag(const std::optional<DirectCallSpecializationShape> &specialization) {
  if (!specialization)
    return CallSpecializationTag::None;
  switch (*specialization) {
  case DirectCallSpecializationShape::UniformExactConstants:
    return CallSpecializationTag::UniformExactConstants;
  }
  return invalid("decision has an unknown direct-call specialization shape");
}

} // namespace

llvm::ArrayRef<std::uint8_t> spatialOwnershipDecisionSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(decisionSchema.data()),
          decisionSchema.size()};
}

llvm::Expected<std::vector<std::uint8_t>>
encodeSpatialOwnershipDecision(const SpatialOwnershipDecision &decision) {
  if (decision.scope.selection.kind != StructuredEntityKind::Operation)
    return invalid("ownership scope does not reference an operation");
  std::vector<std::uint8_t> bytes =
      encodeStructuredEntityRef(decision.scope.selection);
  const AddressProjectionTag projectionTag =
      addressTag(decision.point.addressProjection);
  bytes.push_back(static_cast<std::uint8_t>(projectionTag));
  std::uint32_t indexWidth = 0;
  if (projectionTag == AddressProjectionTag::RootRelative)
    indexWidth = std::get<RootRelativeAddressProjection>(
                     *decision.point.addressProjection)
                     .canonicalIndexWidth;
  if (projectionTag == AddressProjectionTag::RootRelative &&
      !isSupportedIndexWidth(indexWidth))
    return invalid("root-relative projection has an unknown index width");
  appendU32(bytes, indexWidth);
  auto encodedForall = forallTag(decision.point.forallOwnershipShape);
  if (!encodedForall)
    return encodedForall.takeError();
  bytes.push_back(static_cast<std::uint8_t>(*encodedForall));
  auto encodedCall = callTag(decision.point.directCallSpecializationShape);
  if (!encodedCall)
    return encodedCall.takeError();
  bytes.push_back(static_cast<std::uint8_t>(*encodedCall));
  bytes.push_back(decision.point.directCallInlining ? 1 : 0);
  if (decision.point.directCallInlining) {
    const StructuredEntityRef &callSite =
        decision.point.directCallInlining->callSite;
    if (callSite.kind != StructuredEntityKind::Operation)
      return invalid("direct-call inline site is not an operation reference");
    if (callSite.parent != decision.scope.selection.parent)
      return invalid("direct-call inline site has a foreign parent");
    std::vector<std::uint8_t> encodedCallSite =
        encodeStructuredEntityRef(callSite);
    bytes.insert(bytes.end(), encodedCallSite.begin(), encodedCallSite.end());
  } else {
    bytes.insert(bytes.end(), structuredEntityRefWireSize, 0);
  }
  return bytes;
}

llvm::Expected<SpatialOwnershipDecision>
adoptSpatialOwnershipDecision(llvm::ArrayRef<std::uint8_t> canonicalBytes) {
  constexpr std::size_t suffixSize = 8 + structuredEntityRefWireSize;
  if (canonicalBytes.size() != structuredEntityRefWireSize + suffixSize)
    return invalid("decision payload has the wrong size");
  auto selection = decodeStructuredEntityRef(
      canonicalBytes.take_front(structuredEntityRefWireSize));
  if (!selection)
    return selection.takeError();
  if (selection->kind != StructuredEntityKind::Operation)
    return invalid("ownership scope does not reference an operation");

  llvm::ArrayRef<std::uint8_t> suffix =
      canonicalBytes.drop_front(structuredEntityRefWireSize);
  const std::uint8_t projectionTag = suffix[0];
  const std::uint32_t indexWidth = readU32(suffix.slice(1, 4));
  std::optional<SpatialAddressProjection> projection;
  switch (projectionTag) {
  case static_cast<std::uint8_t>(AddressProjectionTag::None):
    if (indexWidth != 0)
      return invalid("absent address projection has an index width");
    break;
  case static_cast<std::uint8_t>(AddressProjectionTag::RootRelative):
    if (!isSupportedIndexWidth(indexWidth))
      return invalid("root-relative projection has an unknown index width");
    projection = RootRelativeAddressProjection{indexWidth};
    break;
  case static_cast<std::uint8_t>(AddressProjectionTag::PointerAddressed):
    if (indexWidth != 0)
      return invalid("pointer-addressed projection has an index width");
    projection = PointerAddressedAddressProjection{};
    break;
  default:
    return invalid("decision payload has an unknown address projection");
  }

  std::optional<ForallOwnershipShape> forallShape;
  switch (suffix[5]) {
  case static_cast<std::uint8_t>(ForallShapeTag::None):
    break;
  case static_cast<std::uint8_t>(ForallShapeTag::GraphParallel):
    forallShape = ForallOwnershipShape::GraphParallel;
    break;
  case static_cast<std::uint8_t>(ForallShapeTag::LogicalThreadDomain):
    forallShape = ForallOwnershipShape::LogicalThreadDomain;
    break;
  default:
    return invalid("decision payload has an unknown forall shape");
  }

  std::optional<DirectCallSpecializationShape> callSpecialization;
  switch (suffix[6]) {
  case static_cast<std::uint8_t>(CallSpecializationTag::None):
    break;
  case static_cast<std::uint8_t>(CallSpecializationTag::UniformExactConstants):
    callSpecialization = DirectCallSpecializationShape::UniformExactConstants;
    break;
  default:
    return invalid("decision payload has an unknown call specialization");
  }

  std::optional<DirectCallInliningDecision> callInlining;
  llvm::ArrayRef<std::uint8_t> callSiteBytes =
      suffix.slice(8, structuredEntityRefWireSize);
  switch (suffix[7]) {
  case 0:
    if (llvm::any_of(callSiteBytes,
                     [](std::uint8_t byte) { return byte != 0; }))
      return invalid("absent direct-call inline site has nonzero bytes");
    break;
  case 1: {
    auto callSite = decodeStructuredEntityRef(callSiteBytes);
    if (!callSite)
      return callSite.takeError();
    if (callSite->kind != StructuredEntityKind::Operation)
      return invalid("direct-call inline site is not an operation reference");
    if (callSite->parent != selection->parent)
      return invalid("direct-call inline site has a foreign parent");
    callInlining = DirectCallInliningDecision{*callSite};
    break;
  }
  default:
    return invalid("decision payload has an unknown direct-call inline tag");
  }

  SpatialOwnershipDecision decision{
      SpatialOwnershipScope{*selection},
      SpatialOwnershipDecisionPoint{projection, forallShape, callSpecialization,
                                    callInlining}};
  auto reencoded = encodeSpatialOwnershipDecision(decision);
  if (!reencoded)
    return reencoded.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*reencoded) != canonicalBytes)
    return invalid("decision payload does not re-encode exactly");
  return decision;
}

} // namespace loom::frontend
