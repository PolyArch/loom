//===- CanonicalImplementationCapability.cpp - Capability inverse --------===//
//
// Derives the least typed implementation-family capability that admits an
// exact set of canonical Dataflow actors. Candidate families remain owned by
// the generated registry; this file owns only the inverse policies.
//
//===----------------------------------------------------------------------===//

#include "Fabric/IR/ImplementationFamily.h"
#include "ImplementationFamilyActorShape.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>
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

/// Every inverse policy is the same frame around one envelope: the exact actor
/// set in canonical order, the provider's capability-independent shape, the
/// refusal of an endpoint whose resolved width belongs to the program rather
/// than to the family, the least envelope the policy derives, and the proof
/// that the forward admission accepts every actor under it. Only the envelope
/// differs, so it is the only thing a policy states.
llvm::Expected<fabric::FamilyCapabilityParams> deriveEnvelope(
    fabric::ImplementationFamilyId family,
    llvm::ArrayRef<dataflow::CanonicalActorSchemaProjection> actors,
    llvm::StringRef relation,
    llvm::function_ref<llvm::Expected<fabric::FamilyCapabilityParams>(
        llvm::ArrayRef<const dataflow::CanonicalActorSchemaProjection *>)>
        leastEnvelope) {
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
        "canonical " + relation +
            " derivation requires explicit index and pointer representation "
            "context");
  auto parameters = leastEnvelope(*ordered);
  if (!parameters)
    return parameters.takeError();
  for (const dataflow::CanonicalActorSchemaProjection *actor : *ordered)
    if (llvm::Error error = fabric::verifyImplementationFamilyAdmission(
            family, &*parameters, *actor))
      return capabilityDerivationFailure(
          fabric::CanonicalCapabilityDerivationFailure::InvalidActorProjection,
          llvm::toString(std::move(error)));
  return parameters;
}

llvm::Expected<fabric::IntegerWidth>
deriveIntegerWidth(mlir::Type type, llvm::StringRef relation) {
  auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
  if (!integer || !integer.isSignless())
    return capabilityDerivationFailure(
        fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
        relation + " capability requires a signless integer type");
  for (fabric::IntegerWidth width : fabric::integerWidthDomain)
    if (fabric::getBitWidth(width) == integer.getWidth()) {
      if (width == fabric::IntegerWidth::I1)
        return capabilityDerivationFailure(
            fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
            relation + " capability does not admit i1");
      return width;
    }
  return capabilityDerivationFailure(
      fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
      relation + " width is outside the registered capability domain");
}

/// The registered format of one floating endpoint. A format the closed domain
/// does not name has no capability to derive, which is a refusal rather than a
/// nearest admissible guess.
llvm::Expected<fabric::FloatFormat>
deriveFloatFormat(mlir::Type type, llvm::StringRef relation) {
  if (std::optional<fabric::FloatFormat> format =
          fabric::symbolizeFloatFormat(type))
    return *format;
  return capabilityDerivationFailure(
      fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
      relation + " format is outside the registered capability domain");
}

/// The least observable floating behavior that admits an exact actor set. A
/// strict implementation refines every relaxed actor, so the derived profile
/// requires no fast-math permission and preserves subnormals and signed zeros;
/// only the rounding domain is observed, and a set whose schemas never round
/// keeps the canonical default.
fabric::FloatBehaviorProfile deriveFloatBehavior(
    llvm::ArrayRef<const dataflow::CanonicalActorSchemaProjection *> actors,
    fabric::detail::FloatDatapath datapath) {
  fabric::FloatBehaviorProfile behavior =
      fabric::FloatBehaviorProfile::strictIEEE();
  fabric::RoundingModeSet roundingModes;
  for (const dataflow::CanonicalActorSchemaProjection *actor : actors) {
    if (!fabric::detail::uniformFloatShape(datapath, actor->schema).rounds)
      continue;
    if (std::optional<mlir::arith::RoundingMode> rounding =
            fabric::detail::arithmeticRounding(*actor))
      roundingModes.insert(*rounding);
  }
  if (!roundingModes.empty())
    behavior.roundingModes = roundingModes;
  return behavior;
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

/// Width of one cast endpoint. Unlike the ordinary integer datapath, a cast
/// relation admits i1: widening a predicate is an ordinary lowered operation.
llvm::Expected<fabric::IntegerWidth> deriveCastIntegerWidth(mlir::Type type) {
  auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
  if (!integer || !integer.isSignless())
    return capabilityDerivationFailure(
        fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
        "integer cast capability requires a signless integer endpoint");
  for (fabric::IntegerWidth width : fabric::integerWidthDomain)
    if (fabric::getBitWidth(width) == integer.getWidth())
      return width;
  return capabilityDerivationFailure(
      fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
      "integer cast width is outside the registered capability domain");
}

/// The exact ordinary integer widths an actor set uses on every endpoint.
llvm::Expected<fabric::FamilyCapabilityParams> deriveScalarIntegerEnvelope(
    fabric::ImplementationFamilyId family,
    llvm::ArrayRef<dataflow::CanonicalActorSchemaProjection> actors) {
  return deriveEnvelope(
      family, actors, "scalar integer",
      [](llvm::ArrayRef<const dataflow::CanonicalActorSchemaProjection *>
             ordered) -> llvm::Expected<fabric::FamilyCapabilityParams> {
        fabric::IntegerWidthSet widths;
        for (const dataflow::CanonicalActorSchemaProjection *actor : ordered)
          for (mlir::Type type : llvm::concat<const mlir::Type>(
                   actor->type.getInputs(), actor->type.getResults())) {
            auto width = deriveIntegerWidth(type, "ordinary scalar integer");
            if (!width)
              return width.takeError();
            widths.insert(*width);
          }
        return fabric::FamilyCapabilityParams{
            fabric::ScalarIntegerParams{widths}};
      });
}

/// The least cast relation that admits an exact actor set: the exact endpoint
/// width pairs its actors use. An index endpoint needs the program's resolved
/// index width, which this canonical context does not carry, so the frame
/// leaves it a typed unavailable rather than a guess.
llvm::Expected<fabric::FamilyCapabilityParams> deriveScalarIntegerCastEnvelope(
    fabric::ImplementationFamilyId family,
    llvm::ArrayRef<dataflow::CanonicalActorSchemaProjection> actors) {
  return deriveEnvelope(
      family, actors, "integer cast",
      [](llvm::ArrayRef<const dataflow::CanonicalActorSchemaProjection *>
             ordered) -> llvm::Expected<fabric::FamilyCapabilityParams> {
        fabric::IntegerCastRelation relation;
        for (const dataflow::CanonicalActorSchemaProjection *actor : ordered) {
          auto source = deriveCastIntegerWidth(actor->type.getInput(0));
          if (!source)
            return source.takeError();
          auto destination = deriveCastIntegerWidth(actor->type.getResult(0));
          if (!destination)
            return destination.takeError();
          relation.widthPairs.insert(*source, *destination);
        }
        if (!relation.widthPairs.valid() || relation.widthPairs.empty())
          return capabilityDerivationFailure(
              fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
              "integer cast actor set yields no admissible width relation");
        return fabric::FamilyCapabilityParams{
            fabric::ScalarIntegerCastParams{relation}};
      });
}

/// The least uniform scalar floating datapath: the exact formats its actors
/// compute in, and the least behavior profile that admits all of them.
llvm::Expected<fabric::FamilyCapabilityParams> deriveScalarFloatEnvelope(
    fabric::ImplementationFamilyId family,
    llvm::ArrayRef<dataflow::CanonicalActorSchemaProjection> actors) {
  return deriveEnvelope(
      family, actors, "scalar floating",
      [](llvm::ArrayRef<const dataflow::CanonicalActorSchemaProjection *>
             ordered) -> llvm::Expected<fabric::FamilyCapabilityParams> {
        fabric::FloatFormatSet formats;
        for (const dataflow::CanonicalActorSchemaProjection *actor : ordered) {
          auto format =
              deriveFloatFormat(actor->type.getInput(0), "scalar floating");
          if (!format)
            return format.takeError();
          formats.insert(*format);
        }
        return fabric::FamilyCapabilityParams{fabric::ScalarFloatParams{
            formats, deriveFloatBehavior(
                         ordered, fabric::detail::FloatDatapath::Scalar)}};
      });
}

/// The least uniform fixed-vector floating datapath: the exact element formats
/// its actors compute in, one payload capacity covering the widest vector they
/// present, and the least behavior profile that admits all of them. A lane
/// count is not a capability field, so the same envelope admits a narrower
/// vector of the same element format.
llvm::Expected<fabric::FamilyCapabilityParams> deriveFixedVectorFloatEnvelope(
    fabric::ImplementationFamilyId family,
    llvm::ArrayRef<dataflow::CanonicalActorSchemaProjection> actors) {
  return deriveEnvelope(
      family, actors, "fixed-vector floating",
      [](llvm::ArrayRef<const dataflow::CanonicalActorSchemaProjection *>
             ordered) -> llvm::Expected<fabric::FamilyCapabilityParams> {
        fabric::FloatFormatSet formats;
        std::uint64_t maximumPayloadBits = 0;
        for (const dataflow::CanonicalActorSchemaProjection *actor : ordered) {
          const auto vector =
              llvm::cast<mlir::VectorType>(actor->type.getInput(0));
          auto format = deriveFloatFormat(vector.getElementType(),
                                          "fixed-vector floating element");
          if (!format)
            return format.takeError();
          formats.insert(*format);
          auto width = derivePayloadWidth(vector);
          if (!width)
            return width.takeError();
          maximumPayloadBits =
              std::max<std::uint64_t>(maximumPayloadBits, *width);
        }
        if (maximumPayloadBits == 0 ||
            maximumPayloadBits > std::numeric_limits<std::uint32_t>::max())
          return capabilityDerivationFailure(
              fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
              "fixed-vector floating envelope exceeds the representable "
              "payload capacity");
        return fabric::FamilyCapabilityParams{fabric::FixedVectorFloatParams{
            formats,
            deriveFloatBehavior(ordered,
                                fabric::detail::FloatDatapath::FixedVector),
            static_cast<std::uint32_t>(maximumPayloadBits)}};
      });
}

/// The least loop-stream resource: the exact recurrence widths and
/// continuation predicates its actors use. The step kind is one fixed
/// implementation parameter rather than a domain, so an actor set that steps
/// two different ways has no single capability to derive and is refused by
/// name; two such streams are two resources.
llvm::Expected<fabric::FamilyCapabilityParams> deriveLoopStreamEnvelope(
    fabric::ImplementationFamilyId family,
    llvm::ArrayRef<dataflow::CanonicalActorSchemaProjection> actors) {
  return deriveEnvelope(
      family, actors, "loop stream",
      [](llvm::ArrayRef<const dataflow::CanonicalActorSchemaProjection *>
             ordered) -> llvm::Expected<fabric::FamilyCapabilityParams> {
        fabric::IntegerWidthSet widths;
        fabric::IntegerPredicateSet predicates;
        std::optional<dataflow::StreamStepKind> stepKind;
        for (const dataflow::CanonicalActorSchemaProjection *actor : ordered) {
          auto width =
              deriveIntegerWidth(actor->type.getInput(0), "stream recurrence");
          if (!width)
            return width.takeError();
          widths.insert(*width);
          const auto &payload =
              std::get<dataflow::StreamRecurrencePayload>(actor->payload);
          if (stepKind && *stepKind != payload.stepKind)
            return capabilityDerivationFailure(
                fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
                "stream actor set steps two ways, which one fixed step kind "
                "cannot carry");
          stepKind = payload.stepKind;
          predicates.insert(payload.predicate);
        }
        if (!predicates.valid() || predicates.empty())
          return capabilityDerivationFailure(
              fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
              "stream actor set yields no admissible continuation predicate");
        return fabric::FamilyCapabilityParams{
            fabric::LoopStreamParams{widths, *stepKind, predicates}};
      });
}

/// The token-plane capability record carries no field: a carry, invariant, or
/// gate resource is fully described by its shape and its payload lane. The
/// inverse policy therefore only has to prove that every actor is admissible,
/// which is exactly what the frame does.
llvm::Expected<fabric::FamilyCapabilityParams> deriveTokenPlaneEnvelope(
    fabric::ImplementationFamilyId family,
    llvm::ArrayRef<dataflow::CanonicalActorSchemaProjection> actors) {
  return deriveEnvelope(
      family, actors, "token-plane",
      [](llvm::ArrayRef<const dataflow::CanonicalActorSchemaProjection *>)
          -> llvm::Expected<fabric::FamilyCapabilityParams> {
        return fabric::FamilyCapabilityParams{fabric::TokenPlaneParams{}};
      });
}

/// The least constant resource: one payload capacity covering the widest value
/// its actors emit. The exact constant values stay with the software graphs, so
/// they are no part of the envelope. The record's capacity domain is positive,
/// so a set that emits only control takes the least representable capacity.
llvm::Expected<fabric::FamilyCapabilityParams> deriveConstantTokenEnvelope(
    fabric::ImplementationFamilyId family,
    llvm::ArrayRef<dataflow::CanonicalActorSchemaProjection> actors) {
  return deriveEnvelope(
      family, actors, "constant token",
      [](llvm::ArrayRef<const dataflow::CanonicalActorSchemaProjection *>
             ordered) -> llvm::Expected<fabric::FamilyCapabilityParams> {
        std::uint64_t maximumPayloadBits = 1;
        for (const dataflow::CanonicalActorSchemaProjection *actor : ordered) {
          auto width = derivePayloadWidth(actor->type.getResult(0));
          if (!width)
            return width.takeError();
          maximumPayloadBits =
              std::max<std::uint64_t>(maximumPayloadBits, *width);
        }
        if (maximumPayloadBits > std::numeric_limits<std::uint32_t>::max())
          return capabilityDerivationFailure(
              fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
              "constant token envelope exceeds the representable payload "
              "capacity");
        return fabric::FamilyCapabilityParams{fabric::PayloadCapacityParams{
            static_cast<std::uint32_t>(maximumPayloadBits)}};
      });
}

/// The least routed-token resource, shared by the sync, mux, and demux
/// families: one fan capacity covering the most lanes any actor routes and one
/// payload capacity covering its widest lane. The lane-count owner says how
/// many lanes each provider routes, and a selector is never wider than the
/// least payload capacity, so the envelope prices exactly the ordered ports.
llvm::Expected<fabric::FamilyCapabilityParams> deriveRoutedTokenEnvelope(
    fabric::ImplementationFamilyId family,
    llvm::ArrayRef<dataflow::CanonicalActorSchemaProjection> actors) {
  return deriveEnvelope(
      family, actors, "routed token",
      [family](llvm::ArrayRef<const dataflow::CanonicalActorSchemaProjection *>
                   ordered) -> llvm::Expected<fabric::FamilyCapabilityParams> {
        std::uint64_t maximumPayloadBits =
            fabric::RoutedTokenParams::minimumPayloadCapacityBits;
        std::uint64_t maximumFan =
            fabric::RoutedTokenParams::minimumFanCapacity;
        for (const dataflow::CanonicalActorSchemaProjection *actor : ordered) {
          std::optional<std::uint32_t> lanes =
              fabric::routedTokenLaneCount(family, *actor);
          if (!lanes)
            return capabilityDerivationFailure(
                fabric::CanonicalCapabilityDerivationFailure::
                    InvalidActorProjection,
                "routed-token actor routes no lane inventory");
          maximumFan = std::max<std::uint64_t>(maximumFan, *lanes);
          for (mlir::Type type : llvm::concat<const mlir::Type>(
                   actor->type.getInputs(), actor->type.getResults())) {
            auto width = derivePayloadWidth(type);
            if (!width)
              return width.takeError();
            maximumPayloadBits =
                std::max<std::uint64_t>(maximumPayloadBits, *width);
          }
        }
        if (maximumPayloadBits > std::numeric_limits<std::uint32_t>::max() ||
            maximumFan > std::numeric_limits<std::uint32_t>::max())
          return capabilityDerivationFailure(
              fabric::CanonicalCapabilityDerivationFailure::NoAdmittingFamily,
              "routed-token envelope exceeds the representable payload or fan "
              "capacity");
        return fabric::FamilyCapabilityParams{fabric::RoutedTokenParams{
            static_cast<std::uint32_t>(maximumPayloadBits),
            static_cast<std::uint32_t>(maximumFan)}};
      });
}

llvm::Expected<fabric::FamilyCapabilityParams> deriveCapabilityEnvelope(
    fabric::ImplementationFamilyId family,
    llvm::ArrayRef<dataflow::CanonicalActorSchemaProjection> actors) {
  const fabric::ImplementationFamilyDescriptor &descriptor =
      fabric::implementationFamily(family);
  switch (descriptor.typedAdmissionProvider) {
  case fabric::TypedAdmissionProviderId::ScalarOrdinaryIntegerAdmission:
    return deriveScalarIntegerEnvelope(family, actors);
  case fabric::TypedAdmissionProviderId::ScalarIntegerCastAdmission:
    return deriveScalarIntegerCastEnvelope(family, actors);
  case fabric::TypedAdmissionProviderId::ScalarUniformFloatAdmission:
    return deriveScalarFloatEnvelope(family, actors);
  case fabric::TypedAdmissionProviderId::FixedVectorUniformFloatAdmission:
    return deriveFixedVectorFloatEnvelope(family, actors);
  case fabric::TypedAdmissionProviderId::StreamAdmission:
    return deriveLoopStreamEnvelope(family, actors);
  case fabric::TypedAdmissionProviderId::TokenPlaneAdmission:
    return deriveTokenPlaneEnvelope(family, actors);
  case fabric::TypedAdmissionProviderId::ConstantTokenAdmission:
    return deriveConstantTokenEnvelope(family, actors);
  case fabric::TypedAdmissionProviderId::SyncTokenAdmission:
  case fabric::TypedAdmissionProviderId::MuxTokenAdmission:
  case fabric::TypedAdmissionProviderId::DemuxTokenAdmission:
    return deriveRoutedTokenEnvelope(family, actors);
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
