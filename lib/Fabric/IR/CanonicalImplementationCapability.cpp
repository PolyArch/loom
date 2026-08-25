//===- CanonicalImplementationCapability.cpp - Capability inverse --------===//
//
// Derives the least typed implementation-family capability that admits an
// exact set of canonical Dataflow actors. Candidate families remain owned by
// the generated registry; this file owns only the inverse policies.
//
//===----------------------------------------------------------------------===//

#include "Fabric/IR/ImplementationFamily.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

llvm::Error capabilityDerivationFailure(
    fabric::CanonicalCapabilityDerivationFailure failure,
    const llvm::Twine &message) {
  return llvm::make_error<fabric::CanonicalCapabilityDerivationError>(
      failure, message.str());
}

bool requiresRepresentationContext(mlir::Type type) {
  mlir::Type element = type;
  if (auto vector = llvm::dyn_cast<mlir::VectorType>(type))
    element = vector.getElementType();
  return llvm::isa<mlir::IndexType, mlir::LLVM::LLVMPointerType>(element);
}

bool actorRequiresRepresentationContext(
    const dataflow::CanonicalActorSchemaProjection &actor) {
  if (actor.schema == dataflow::OperationSchemaId::LLVMGetElementPtr)
    return true;
  return llvm::any_of(actor.type.getInputs(), requiresRepresentationContext) ||
         llvm::any_of(actor.type.getResults(), requiresRepresentationContext);
}

llvm::Expected<std::vector<const dataflow::CanonicalActorSchemaProjection *>>
canonicalActorOrder(
    llvm::ArrayRef<dataflow::CanonicalActorSchemaProjection> actors) {
  using Entry = std::pair<const dataflow::CanonicalActorSchemaProjection *,
                          loom::CanonicalSemanticBytes>;
  std::vector<Entry> entries;
  entries.reserve(actors.size());
  bool malformed = false;
  for (const dataflow::CanonicalActorSchemaProjection &actor : actors) {
    auto key = dataflow::encodeCanonicalActorSchemaProjection(actor);
    if (!key) {
      llvm::consumeError(key.takeError());
      malformed = true;
      continue;
    }
    entries.emplace_back(&actor, std::move(*key));
  }
  if (malformed)
    return capabilityDerivationFailure(
        fabric::CanonicalCapabilityDerivationFailure::InvalidActorProjection,
        "actor set contains a noncanonical schema projection");
  llvm::sort(entries, [](const Entry &left, const Entry &right) {
    return std::lexicographical_compare(
        left.second.bytes().begin(), left.second.bytes().end(),
        right.second.bytes().begin(), right.second.bytes().end());
  });
  std::vector<const dataflow::CanonicalActorSchemaProjection *> ordered;
  ordered.reserve(entries.size());
  for (const Entry &entry : entries)
    ordered.push_back(entry.first);
  return ordered;
}

llvm::Expected<fabric::IntegerWidth> deriveIntegerWidth(mlir::Type type) {
  auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
  if (!integer || !integer.isSignless())
    return capabilityDerivationFailure(
        fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
        "scalar integer capability requires a signless integer type");
  for (fabric::IntegerWidth width : fabric::integerWidthDomain)
    if (fabric::getBitWidth(width) == integer.getWidth()) {
      if (width == fabric::IntegerWidth::I1)
        return capabilityDerivationFailure(
            fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
            "ordinary scalar integer capability does not admit i1");
      return width;
    }
  return capabilityDerivationFailure(
      fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
      "scalar integer width is outside the registered capability domain");
}

llvm::Expected<fabric::FamilyCapabilityParams> deriveScalarIntegerEnvelope(
    fabric::ImplementationFamilyId family,
    llvm::ArrayRef<dataflow::CanonicalActorSchemaProjection> actors) {
  auto ordered = canonicalActorOrder(actors);
  if (!ordered)
    return ordered.takeError();
  for (const dataflow::CanonicalActorSchemaProjection *actor : *ordered)
    if (llvm::Error error =
            fabric::verifyImplementationFamilyActorShape(family, *actor))
      return capabilityDerivationFailure(
          fabric::CanonicalCapabilityDerivationFailure::InvalidActorProjection,
          llvm::toString(std::move(error)));
  if (llvm::any_of(actors, actorRequiresRepresentationContext))
    return capabilityDerivationFailure(
        fabric::CanonicalCapabilityDerivationFailure::
            UnsupportedAdmissionProvider,
        "canonical scalar capability derivation requires explicit index and "
        "pointer representation context");
  fabric::IntegerWidthSet widths;
  for (const dataflow::CanonicalActorSchemaProjection *actor : *ordered) {
    for (mlir::Type type : actor->type.getInputs()) {
      auto width = deriveIntegerWidth(type);
      if (!width)
        return width.takeError();
      widths.insert(*width);
    }
    for (mlir::Type type : actor->type.getResults()) {
      auto width = deriveIntegerWidth(type);
      if (!width)
        return width.takeError();
      widths.insert(*width);
    }
  }

  fabric::FamilyCapabilityParams parameters =
      fabric::ScalarIntegerParams{widths};
  for (const dataflow::CanonicalActorSchemaProjection *actor : *ordered) {
    if (llvm::Error error = fabric::verifyImplementationFamilyAdmission(
            family, &parameters, *actor))
      return capabilityDerivationFailure(
          fabric::CanonicalCapabilityDerivationFailure::InvalidActorProjection,
          llvm::toString(std::move(error)));
  }
  return parameters;
}

llvm::Expected<unsigned> derivePayloadWidth(mlir::Type type) {
  std::string message;
  mlir::FailureOr<unsigned> width =
      fabric::getSemanticPayloadWidth(type, message);
  if (mlir::failed(width))
    return capabilityDerivationFailure(
        fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
        message);
  return *width;
}

llvm::Expected<fabric::FamilyCapabilityParams> deriveTokenSyncEnvelope(
    fabric::ImplementationFamilyId family,
    llvm::ArrayRef<dataflow::CanonicalActorSchemaProjection> actors) {
  auto ordered = canonicalActorOrder(actors);
  if (!ordered)
    return ordered.takeError();
  for (const dataflow::CanonicalActorSchemaProjection *actor : *ordered)
    if (llvm::Error error =
            fabric::verifyImplementationFamilyActorShape(family, *actor))
      return capabilityDerivationFailure(
          fabric::CanonicalCapabilityDerivationFailure::InvalidActorProjection,
          llvm::toString(std::move(error)));
  if (llvm::any_of(actors, actorRequiresRepresentationContext))
    return capabilityDerivationFailure(
        fabric::CanonicalCapabilityDerivationFailure::
            UnsupportedAdmissionProvider,
        "canonical token capability derivation requires explicit index and "
        "pointer representation context");
  std::uint64_t maximumPayloadBits =
      fabric::RoutedTokenParams::minimumPayloadCapacityBits;
  std::uint64_t maximumFan = fabric::RoutedTokenParams::minimumFanCapacity;
  for (const dataflow::CanonicalActorSchemaProjection *actor : *ordered) {
    maximumFan =
        std::max<std::uint64_t>(maximumFan, actor->type.getNumInputs());
    for (mlir::Type type : actor->type.getInputs()) {
      auto width = derivePayloadWidth(type);
      if (!width)
        return width.takeError();
      maximumPayloadBits = std::max<std::uint64_t>(maximumPayloadBits, *width);
    }
    for (mlir::Type type : actor->type.getResults()) {
      auto width = derivePayloadWidth(type);
      if (!width)
        return width.takeError();
      maximumPayloadBits = std::max<std::uint64_t>(maximumPayloadBits, *width);
    }
  }
  if (maximumPayloadBits > std::numeric_limits<std::uint32_t>::max() ||
      maximumFan > std::numeric_limits<std::uint32_t>::max())
    return capabilityDerivationFailure(
        fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
        "canonical token-sync envelope exceeds representable payload or fan "
        "capacity");

  fabric::FamilyCapabilityParams parameters =
      fabric::RoutedTokenParams{static_cast<std::uint32_t>(maximumPayloadBits),
                                static_cast<std::uint32_t>(maximumFan)};
  for (const dataflow::CanonicalActorSchemaProjection *actor : *ordered) {
    if (llvm::Error error = fabric::verifyImplementationFamilyAdmission(
            family, &parameters, *actor))
      return capabilityDerivationFailure(
          fabric::CanonicalCapabilityDerivationFailure::InvalidActorProjection,
          llvm::toString(std::move(error)));
  }
  return parameters;
}

llvm::Expected<fabric::FamilyCapabilityParams> deriveCapabilityEnvelope(
    fabric::ImplementationFamilyId family,
    llvm::ArrayRef<dataflow::CanonicalActorSchemaProjection> actors) {
  const fabric::ImplementationFamilyDescriptor &descriptor =
      fabric::implementationFamily(family);
  switch (descriptor.typedAdmissionProvider) {
  case fabric::TypedAdmissionProviderId::ScalarOrdinaryIntegerAdmission:
    return deriveScalarIntegerEnvelope(family, actors);
  case fabric::TypedAdmissionProviderId::SyncTokenAdmission:
    return deriveTokenSyncEnvelope(family, actors);
  default:
    return capabilityDerivationFailure(
        fabric::CanonicalCapabilityDerivationFailure::
            UnsupportedAdmissionProvider,
        "canonical capability derivation is unavailable for admission "
        "provider '" +
            fabric::typedAdmissionProviderKeyword(
                descriptor.typedAdmissionProvider) +
            "'");
  }
}

} // namespace

char fabric::CanonicalCapabilityDerivationError::ID = 0;

void fabric::CanonicalCapabilityDerivationError::log(
    llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code
fabric::CanonicalCapabilityDerivationError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<fabric::CanonicalImplementationCapability>
fabric::deriveCanonicalImplementationCapability(
    ImplementationFamilyId family,
    llvm::ArrayRef<::dataflow::CanonicalActorSchemaProjection> actors) {
  if (actors.empty())
    return capabilityDerivationFailure(
        CanonicalCapabilityDerivationFailure::EmptyActorSet,
        "canonical capability derivation requires a non-empty actor set");

  if (static_cast<std::uint32_t>(family) >= implementationFamilyCount())
    return capabilityDerivationFailure(
        CanonicalCapabilityDerivationFailure::InvalidFamily,
        "canonical capability derivation requires a registered family");
  if (llvm::any_of(actors, [&](const auto &actor) {
        return !admitsOperationSchema(family, actor.schema);
      }))
    return capabilityDerivationFailure(
        CanonicalCapabilityDerivationFailure::FamilyDoesNotOwnSchema,
        "implementation family does not own every actor schema");

  auto parameters = deriveCapabilityEnvelope(family, actors);
  if (!parameters)
    return parameters.takeError();
  std::vector<::dataflow::OperationSchemaId> enabledSchemas;
  enabledSchemas.reserve(actors.size());
  for (const auto &actor : actors)
    enabledSchemas.push_back(actor.schema);
  llvm::sort(enabledSchemas, [](auto left, auto right) {
    return static_cast<std::uint32_t>(left) < static_cast<std::uint32_t>(right);
  });
  enabledSchemas.erase(
      std::unique(enabledSchemas.begin(), enabledSchemas.end()),
      enabledSchemas.end());
  return CanonicalImplementationCapability{family, std::move(*parameters),
                                           std::move(enabledSchemas)};
}
