#include "PnR/FrozenConstraintIndex.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::mapping;
using namespace loom::pnr;

using SpatialConstraintProjection = ::mapping::SpatialConstraintProjection;

char SpatialPnrFreezeFailure::ID;

void SpatialPnrFreezeFailure::log(llvm::raw_ostream &stream) const {
  stream << (kind_ == SpatialPnrFreezeFailureKind::Invalid
                 ? "spatial_pnr_freeze_invalid: "
                 : "spatial_pnr_proven_infeasible: ")
         << message_;
}

std::error_code SpatialPnrFreezeFailure::convertToErrorCode() const {
  return std::make_error_code(std::errc::invalid_argument);
}

namespace {

constexpr llvm::StringLiteral frozenArtifact = "FrozenConstraintIndex";
constexpr PnrCapacityContext subjectCountContext{
    frozenArtifact, "subjects", "subjects", PnrCapacityMeasure::Count};
constexpr PnrCapacityContext subjectIndexContext{
    frozenArtifact, "subjects", "subjects", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext domainOffsetContext{frozenArtifact, "restrictions",
                                                 "domain_values",
                                                 PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext domainCountContext{frozenArtifact, "domain_values",
                                                "domain_values",
                                                PnrCapacityMeasure::Count};
constexpr PnrCapacityContext relationOffsetContext{frozenArtifact, "relations",
                                                   "relation_members",
                                                   PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext relationCountContext{
    frozenArtifact, "relation_members", "relation_members",
    PnrCapacityMeasure::Count};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::Invalid, message.str());
}

llvm::Error infeasible(SpatialConstraintProjection projection,
                       const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::ProvenInfeasible, message.str(), projection);
}

std::size_t projectionOrdinal(SpatialConstraintProjection projection) {
  return static_cast<std::size_t>(projection);
}

bool allowsEmptyDomain(SpatialConstraintProjection projection) {
  switch (projection) {
  case SpatialConstraintProjection::NetAssignedTagValues:
  case SpatialConstraintProjection::NetSelectedPhysicalTraversals:
  case SpatialConstraintProjection::NetTraversalResourceStates:
  case SpatialConstraintProjection::MemoryAddressRegion:
    return true;
  case SpatialConstraintProjection::ComputePlacement:
  case SpatialConstraintProjection::ComputeParentPe:
  case SpatialConstraintProjection::ComputeInstructionContext:
  case SpatialConstraintProjection::ComputeFuContext:
  case SpatialConstraintProjection::MemoryPlacement:
  case SpatialConstraintProjection::SpatialTransferAttachment:
  case SpatialConstraintProjection::MemoryOperationPort:
  case SpatialConstraintProjection::MemoryBoundServices:
    return false;
  }
  llvm_unreachable("unknown Spatial constraint projection");
}

class DisjointSet final {
public:
  explicit DisjointSet(std::size_t size) : parent_(size) {
    std::iota(parent_.begin(), parent_.end(), std::size_t{0});
  }

  std::size_t find(std::size_t value) {
    if (parent_[value] == value)
      return value;
    parent_[value] = find(parent_[value]);
    return parent_[value];
  }

  void unite(std::size_t lhs, std::size_t rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return;
    const std::size_t representative = std::min(lhs, rhs);
    parent_[lhs == representative ? rhs : lhs] = representative;
  }

private:
  std::vector<std::size_t> parent_;
};

llvm::Expected<PnrIndex> checkedIndex(PnrCapacityContext context,
                                      std::size_t value) {
  return checkedPnrIndex(context, static_cast<std::uint64_t>(value));
}

llvm::Error preflightAppend(PnrCapacityContext context, std::size_t current,
                            std::size_t added) {
  auto end = checkedPnrIndexAdd(context, static_cast<std::uint64_t>(current),
                                static_cast<std::uint64_t>(added));
  if (!end)
    return end.takeError();
  return llvm::Error::success();
}

PnrIndex findSubject(const FrozenConstraintShard &shard,
                     const SpatialConstraintSubject &subject) {
  const auto found = llvm::find(shard.subjects(), subject);
  assert(found != shard.subjects().end());
  return static_cast<PnrIndex>(found - shard.subjects().begin());
}

} // namespace

class loom::pnr::FrozenConstraintIndexBuilder final {
public:
  static llvm::Expected<FrozenConstraintIndex>
  build(const SpatialMappingConstraintSetView &constraints) {
    FrozenConstraintIndex result;

    const auto remember =
        [&](SpatialConstraintProjection projection,
            const SpatialConstraintSubject &subject) -> llvm::Error {
      FrozenConstraintShard &shard =
          result.shards_[projectionOrdinal(projection)];
      if (!llvm::is_contained(shard.subjects_, subject)) {
        if (llvm::Error error =
                preflightAppend(subjectCountContext, shard.subjects_.size(), 1))
          return error;
        shard.subjects_.push_back(subject);
      }
      return llvm::Error::success();
    };

    for (const SpatialConstraintClauseView &clause : constraints.clauses()) {
      if (const auto *restriction =
              std::get_if<SpatialDomainRestrictionView>(&clause)) {
        if (llvm::Error error =
                remember(restriction->projection, restriction->subject))
          return std::move(error);
        continue;
      }
      if (const auto *equal = std::get_if<SpatialEqualView>(&clause)) {
        for (const SpatialConstraintSubject &subject : equal->subjects)
          if (llvm::Error error = remember(equal->projection, subject))
            return std::move(error);
        continue;
      }
      const auto &disjoint = std::get<SpatialDisjointView>(clause);
      for (const SpatialConstraintSubject &subject : disjoint.subjects)
        if (llvm::Error error = remember(disjoint.projection, subject))
          return std::move(error);
    }

    std::vector<DisjointSet> equality;
    equality.reserve(result.shards_.size());
    for (const FrozenConstraintShard &shard : result.shards_)
      equality.emplace_back(shard.subjects_.size());

    for (const SpatialConstraintClauseView &clause : constraints.clauses()) {
      const auto *equal = std::get_if<SpatialEqualView>(&clause);
      if (!equal)
        continue;
      FrozenConstraintShard &shard =
          result.shards_[projectionOrdinal(equal->projection)];
      const PnrIndex first = findSubject(shard, equal->subjects.front());
      for (const SpatialConstraintSubject &subject :
           llvm::drop_begin(equal->subjects))
        equality[projectionOrdinal(equal->projection)].unite(
            first, findSubject(shard, subject));
    }

    for (std::size_t ordinal = 0; ordinal < result.shards_.size(); ++ordinal) {
      FrozenConstraintShard &shard = result.shards_[ordinal];
      shard.subjectRepresentatives_.reserve(shard.subjects_.size());
      for (std::size_t subject = 0; subject < shard.subjects_.size();
           ++subject) {
        auto representative =
            checkedIndex(subjectIndexContext, equality[ordinal].find(subject));
        if (!representative)
          return representative.takeError();
        shard.subjectRepresentatives_.push_back(*representative);
      }
    }

    const auto appendRelation =
        [&](FrozenConstraintShard &shard,
            llvm::ArrayRef<SpatialConstraintSubject> subjects,
            std::vector<FrozenConstraintRelation> &rows) -> llvm::Error {
      auto offset =
          checkedIndex(relationOffsetContext, shard.relationMembers_.size());
      if (!offset)
        return offset.takeError();
      if (llvm::Error error =
              preflightAppend(relationCountContext,
                              shard.relationMembers_.size(), subjects.size()))
        return error;
      for (const SpatialConstraintSubject &subject : subjects)
        shard.relationMembers_.push_back(findSubject(shard, subject));
      auto count = checkedIndex(relationCountContext, subjects.size());
      if (!count)
        return count.takeError();
      rows.push_back({*offset, *count});
      return llvm::Error::success();
    };

    for (const SpatialConstraintClauseView &clause : constraints.clauses()) {
      if (const auto *restriction =
              std::get_if<SpatialDomainRestrictionView>(&clause)) {
        FrozenConstraintShard &shard =
            result.shards_[projectionOrdinal(restriction->projection)];
        const PnrIndex subject = findSubject(shard, restriction->subject);
        const PnrIndex representative = shard.subjectRepresentatives_[subject];
        if (restriction->admissibleDomain.empty() &&
            !allowsEmptyDomain(restriction->projection))
          return infeasible(restriction->projection,
                            "an explicit empty domain contradicts the "
                            "projection cardinality");
        if (llvm::any_of(shard.restrictions_, [&](const auto &row) {
              return row.subject == representative;
            }))
          return invalid(
              "canonical MappingConstraintSet repeats a restriction");
        auto offset =
            checkedIndex(domainOffsetContext, shard.domainValues_.size());
        if (!offset)
          return offset.takeError();
        if (llvm::Error error =
                preflightAppend(domainCountContext, shard.domainValues_.size(),
                                restriction->admissibleDomain.size()))
          return error;
        shard.domainValues_.insert(shard.domainValues_.end(),
                                   restriction->admissibleDomain.begin(),
                                   restriction->admissibleDomain.end());
        auto count = checkedIndex(domainCountContext,
                                  restriction->admissibleDomain.size());
        if (!count)
          return count.takeError();
        shard.restrictions_.push_back({representative, *offset, *count});
        continue;
      }
      if (const auto *equal = std::get_if<SpatialEqualView>(&clause)) {
        FrozenConstraintShard &shard =
            result.shards_[projectionOrdinal(equal->projection)];
        if (llvm::Error error =
                appendRelation(shard, equal->subjects, shard.equalityClasses_))
          return std::move(error);
        continue;
      }
      const auto &disjoint = std::get<SpatialDisjointView>(clause);
      FrozenConstraintShard &shard =
          result.shards_[projectionOrdinal(disjoint.projection)];
      if (llvm::Error error =
              appendRelation(shard, disjoint.subjects, shard.disjointGroups_))
        return std::move(error);
    }
    return result;
  }
};

FrozenConstraintIndex::FrozenConstraintIndex() {
  shards_.reserve(projectionCount);
  for (std::uint32_t ordinal = 0; ordinal < projectionCount; ++ordinal) {
    const auto projection =
        ::mapping::symbolizeSpatialConstraintProjection(ordinal);
    if (!projection)
      llvm_unreachable("Spatial constraint projection catalog has a gap");
    shards_.push_back(FrozenConstraintShard(*projection));
  }
}

const FrozenConstraintShard &
FrozenConstraintIndex::shard(SpatialConstraintProjection projection) const {
  const std::size_t ordinal = projectionOrdinal(projection);
  assert(ordinal < shards_.size());
  return shards_[ordinal];
}

bool FrozenConstraintIndex::empty() const {
  return llvm::all_of(shards_, [](const FrozenConstraintShard &shard) {
    return shard.empty();
  });
}

std::optional<llvm::ArrayRef<SpatialConstraintDomainValue>>
FrozenConstraintShard::restrictedDomain(
    const SpatialConstraintSubject &subject) const {
  const auto found = llvm::find(subjects_, subject);
  if (found == subjects_.end())
    return std::nullopt;
  const PnrIndex subjectIndex =
      static_cast<PnrIndex>(found - subjects_.begin());
  const PnrIndex representative = subjectRepresentatives_[subjectIndex];
  const auto restriction =
      llvm::find_if(restrictions_, [&](const FrozenConstraintRestriction &row) {
        return row.subject == representative;
      });
  if (restriction == restrictions_.end())
    return std::nullopt;
  return llvm::ArrayRef<SpatialConstraintDomainValue>(domainValues_)
      .slice(restriction->domainOffset, restriction->domainCount);
}

bool FrozenConstraintShard::empty() const {
  return restrictions_.empty() && equalityClasses_.empty() &&
         disjointGroups_.empty();
}

llvm::Expected<FrozenConstraintIndex>
loom::pnr::detail::buildFrozenConstraintIndex(
    const SpatialMappingConstraintSetView &constraints) {
  return FrozenConstraintIndexBuilder::build(constraints);
}
