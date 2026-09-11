#include "StructuredScheduleInternal.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::frontend {
namespace detail {

llvm::Error invalidStructuredSchedule(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_schedule_invalid: " + message);
}

llvm::Error validateStructuredVectorScheduleCoordinate(
    const StructuredVectorScheduleCoordinate &coordinate) {
  if (coordinate.shape.size() != 1 || coordinate.shape.front() <= 1 ||
      coordinate.shape.front() > maximumCanonicalStructuredScheduleFactor)
    return invalidStructuredSchedule(
        "vector coordinate has no supported rank-one shape");
  if (coordinate.requiredAlignmentBytes == 0)
    return invalidStructuredSchedule(
        "vector coordinate has no required alignment");
  if (coordinate.tailPolicy > StructuredVectorTailPolicy::ReductionMask ||
      coordinate.aliasPolicy !=
          StructuredVectorAliasPolicy::ProviderProvenNoAlias ||
      coordinate.reductionSchedule >
          StructuredReductionSchedule::FloatingReassociated)
    return invalidStructuredSchedule(
        "vector coordinate has an unknown typed policy");
  if (coordinate.tailPolicy == StructuredVectorTailPolicy::ReductionMask &&
      coordinate.reductionSchedule == StructuredReductionSchedule::None)
    return invalidStructuredSchedule(
        "non-reduction vector coordinate selects a reduction mask");
  return llvm::Error::success();
}

} // namespace detail

namespace {

constexpr llvm::StringLiteral decisionSchema =
    "loom.structured_schedule.decision.6.0";

/// Replication decisions carry a canonical factor; interchange, parallel, and
/// vector decisions are factorless; a polyhedral decision is factorless for
/// the provider schedule itself and carries the tile factor of a tiled one.
bool admitsDecisionFactor(StructuredScheduleDecisionKind kind,
                          std::uint64_t factor) {
  const bool canonicalFactor =
      factor >= 2 && factor <= maximumCanonicalStructuredScheduleFactor;
  switch (kind) {
  case StructuredScheduleDecisionKind::Unroll:
  case StructuredScheduleDecisionKind::UnrollAndJam:
    return canonicalFactor;
  case StructuredScheduleDecisionKind::Tile:
    // A strip-mine tile size is canonical by its size or by its tile count,
    // exactly like a polyhedral tile factor.
    return factor >= 2;
  case StructuredScheduleDecisionKind::Interchange:
  case StructuredScheduleDecisionKind::Parallelize:
  case StructuredScheduleDecisionKind::ParallelizeNest:
  case StructuredScheduleDecisionKind::Vectorize:
    return factor == 0;
  case StructuredScheduleDecisionKind::PolyhedralSchedule:
    // A tile factor is canonical by its size or by its tile count; the
    // exact domain is re-enumerated from the trip count on replay.
    return factor == 0 || factor >= 2;
  }
  return false;
}

} // namespace

llvm::ArrayRef<std::uint8_t> structuredScheduleDecisionSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(decisionSchema.data()),
          decisionSchema.size()};
}

llvm::StringRef
structuredScheduleDecisionKindSpelling(StructuredScheduleDecisionKind kind) {
  switch (kind) {
  case StructuredScheduleDecisionKind::Tile:
    return "tile";
  case StructuredScheduleDecisionKind::Unroll:
    return "unroll";
  case StructuredScheduleDecisionKind::Interchange:
    return "interchange";
  case StructuredScheduleDecisionKind::UnrollAndJam:
    return "unroll_and_jam";
  case StructuredScheduleDecisionKind::Parallelize:
    return "parallelize";
  case StructuredScheduleDecisionKind::ParallelizeNest:
    return "parallelize_nest";
  case StructuredScheduleDecisionKind::Vectorize:
    return "vectorize";
  case StructuredScheduleDecisionKind::PolyhedralSchedule:
    return "polyhedral_schedule";
  }
  llvm_unreachable("unknown structured schedule decision kind");
}

llvm::Expected<std::vector<std::uint8_t>>
encodeStructuredScheduleDecision(const StructuredScheduleDecision &decision) {
  if (decision.loop.kind != StructuredEntityKind::Operation)
    return detail::invalidStructuredSchedule(
        "decision does not reference an operation");
  if (static_cast<std::uint32_t>(decision.kind) >
      static_cast<std::uint32_t>(
          StructuredScheduleDecisionKind::PolyhedralSchedule))
    return detail::invalidStructuredSchedule("decision has an unknown kind");
  if (!admitsDecisionFactor(decision.kind, decision.factor))
    return detail::invalidStructuredSchedule("decision has an invalid factor");
  if (decision.kind == StructuredScheduleDecisionKind::Vectorize) {
    if (!decision.vector)
      return detail::invalidStructuredSchedule(
          "vector decision has no vector coordinate");
    if (llvm::Error error = detail::validateStructuredVectorScheduleCoordinate(
            *decision.vector))
      return std::move(error);
  } else if (decision.vector) {
    return detail::invalidStructuredSchedule(
        "non-vector decision carries a vector coordinate");
  }
  std::vector<std::uint8_t> bytes = encodeStructuredEntityRef(decision.loop);
  const auto appendU32 = [&](std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  };
  const auto appendU64 = [&](std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  };
  appendU32(static_cast<std::uint32_t>(decision.kind));
  appendU64(decision.factor);
  if (decision.vector) {
    appendU32(static_cast<std::uint32_t>(decision.vector->shape.size()));
    for (std::uint64_t dimension : decision.vector->shape)
      appendU64(dimension);
    appendU32(static_cast<std::uint32_t>(decision.vector->tailPolicy));
    appendU64(decision.vector->requiredAlignmentBytes);
    appendU32(static_cast<std::uint32_t>(decision.vector->aliasPolicy));
    appendU32(static_cast<std::uint32_t>(decision.vector->reductionSchedule));
  }
  return bytes;
}

llvm::Expected<StructuredScheduleDecision>
adoptStructuredScheduleDecision(llvm::ArrayRef<std::uint8_t> canonicalBytes) {
  constexpr std::size_t scalarWireSize = structuredEntityRefWireSize + 12;
  if (canonicalBytes.size() < scalarWireSize)
    return detail::invalidStructuredSchedule("decision payload is truncated");
  auto loop = decodeStructuredEntityRef(
      canonicalBytes.take_front(structuredEntityRefWireSize));
  if (!loop)
    return loop.takeError();
  if (loop->kind != StructuredEntityKind::Operation)
    return detail::invalidStructuredSchedule(
        "decision does not reference an operation");
  llvm::ArrayRef<std::uint8_t> suffix =
      canonicalBytes.drop_front(structuredEntityRefWireSize);
  std::size_t offset = 0;
  const auto readU32 = [&]() -> llvm::Expected<std::uint32_t> {
    if (suffix.size() - offset < 4)
      return detail::invalidStructuredSchedule(
          "decision payload has a truncated u32");
    std::uint32_t value = 0;
    for (std::uint8_t byte : suffix.slice(offset, 4))
      value = (value << 8) | byte;
    offset += 4;
    return value;
  };
  const auto readU64 = [&]() -> llvm::Expected<std::uint64_t> {
    if (suffix.size() - offset < 8)
      return detail::invalidStructuredSchedule(
          "decision payload has a truncated u64");
    std::uint64_t value = 0;
    for (std::uint8_t byte : suffix.slice(offset, 8))
      value = (value << 8) | byte;
    offset += 8;
    return value;
  };
  auto kind = readU32();
  if (!kind)
    return kind.takeError();
  if (*kind > static_cast<std::uint32_t>(
                  StructuredScheduleDecisionKind::PolyhedralSchedule))
    return detail::invalidStructuredSchedule(
        "decision payload has an unknown kind");
  auto factor = readU64();
  if (!factor)
    return factor.takeError();
  const auto typedKind = static_cast<StructuredScheduleDecisionKind>(*kind);
  if (!admitsDecisionFactor(typedKind, *factor))
    return detail::invalidStructuredSchedule(
        "decision payload has an invalid factor");
  std::optional<StructuredVectorScheduleCoordinate> vector;
  if (typedKind == StructuredScheduleDecisionKind::Vectorize) {
    auto rank = readU32();
    if (!rank)
      return rank.takeError();
    if (*rank != 1)
      return detail::invalidStructuredSchedule(
          "vector decision payload has an unsupported rank");
    std::vector<std::uint64_t> shape;
    shape.reserve(*rank);
    for (std::uint32_t dimension = 0; dimension != *rank; ++dimension) {
      auto size = readU64();
      if (!size)
        return size.takeError();
      shape.push_back(*size);
    }
    auto tail = readU32();
    if (!tail)
      return tail.takeError();
    auto alignment = readU64();
    if (!alignment)
      return alignment.takeError();
    auto alias = readU32();
    if (!alias)
      return alias.takeError();
    auto reduction = readU32();
    if (!reduction)
      return reduction.takeError();
    vector.emplace(StructuredVectorScheduleCoordinate{
        std::move(shape), static_cast<StructuredVectorTailPolicy>(*tail),
        *alignment, static_cast<StructuredVectorAliasPolicy>(*alias),
        static_cast<StructuredReductionSchedule>(*reduction)});
    if (llvm::Error error =
            detail::validateStructuredVectorScheduleCoordinate(*vector))
      return std::move(error);
  }
  if (offset != suffix.size())
    return detail::invalidStructuredSchedule(
        "decision payload has trailing bytes");
  StructuredScheduleDecision decision{*loop, typedKind, *factor,
                                      std::move(vector)};
  auto reencoded = encodeStructuredScheduleDecision(decision);
  if (!reencoded)
    return reencoded.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*reencoded) != canonicalBytes)
    return detail::invalidStructuredSchedule(
        "decision payload does not re-encode exactly");
  return decision;
}

} // namespace loom::frontend
