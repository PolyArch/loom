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
using SystemConstraintProjection = ::mapping::SystemConstraintProjection;

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

llvm::Error spatialInvalid(const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::Invalid, message.str());
}

llvm::Error spatialInfeasible(SpatialConstraintProjection projection,
                              const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::ProvenInfeasible, message.str(), projection);
}

llvm::Error systemInvalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_constraint_freeze_invalid: " +
                                     message);
}

template <typename Projection>
std::size_t projectionOrdinal(Projection projection) {
  return static_cast<std::size_t>(projection);
}

bool spatialAllowsEmptyDomain(SpatialConstraintProjection projection) {
  switch (projection) {
  case SpatialConstraintProjection::NetAssignedTagValues:
  case SpatialConstraintProjection::NetSelectedPhysicalTraversals:
  case SpatialConstraintProjection::NetTraversalResourceStates:
  case SpatialConstraintProjection::MemoryBoundServices:
  case SpatialConstraintProjection::MemoryAddressRegion:
    return true;
  case SpatialConstraintProjection::ComputePlacement:
  case SpatialConstraintProjection::ComputeParentPe:
  case SpatialConstraintProjection::ComputeInstructionContext:
  case SpatialConstraintProjection::ComputeFuContext:
  case SpatialConstraintProjection::MemoryPlacement:
  case SpatialConstraintProjection::SpatialTransferAttachment:
  case SpatialConstraintProjection::MemoryOperationPort:
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

template <typename Shard, typename Subject>
PnrIndex findSubject(const Shard &shard, const Subject &subject) {
  const auto found = llvm::find(shard.subjects(), subject);
  assert(found != shard.subjects().end());
  return static_cast<PnrIndex>(found - shard.subjects().begin());
}

struct SpatialConstraintIndexTraits final {
  using Index = FrozenConstraintIndex;
  using Shard = FrozenConstraintShard;
  using View = SpatialMappingConstraintSetView;
  using Clause = SpatialConstraintClauseView;
  using Restriction = SpatialDomainRestrictionView;
  using Equal = SpatialEqualView;
  using Disjoint = SpatialDisjointView;
  using Projection = SpatialConstraintProjection;
  using Subject = SpatialConstraintSubject;

  static bool allowsEmptyDomain(Projection projection) {
    return spatialAllowsEmptyDomain(projection);
  }
  static llvm::Error invalid(const llvm::Twine &message) {
    return spatialInvalid(message);
  }
  static llvm::Error infeasible(Projection projection,
                                const llvm::Twine &message) {
    return spatialInfeasible(projection, message);
  }
};

struct SystemConstraintIndexTraits final {
  using Index = SystemFrozenConstraintIndex;
  using Shard = SystemFrozenConstraintShard;
  using View = SystemMappingConstraintSetView;
  using Clause = SystemConstraintClauseView;
  using Restriction = SystemDomainRestrictionView;
  using Equal = SystemEqualView;
  using Disjoint = SystemDisjointView;
  using Projection = SystemConstraintProjection;
  using Subject = SystemConstraintSubject;

  static bool allowsEmptyDomain(Projection) { return true; }
  static llvm::Error invalid(const llvm::Twine &message) {
    return systemInvalid(message);
  }
  static llvm::Error infeasible(Projection, const llvm::Twine &message) {
    return systemInvalid(message);
  }
};

} // namespace

template <typename Traits> class loom::pnr::FrozenConstraintIndexBuilder final {
public:
  static llvm::Expected<typename Traits::Index>
  build(const typename Traits::View &constraints) {
    typename Traits::Index result;

    const auto remember =
        [&](typename Traits::Projection projection,
            const typename Traits::Subject &subject) -> llvm::Error {
      typename Traits::Shard &shard =
          result.shards_[projectionOrdinal(projection)];
      if (!llvm::is_contained(shard.subjects_, subject)) {
        if (llvm::Error error =
                preflightAppend(subjectCountContext, shard.subjects_.size(), 1))
          return error;
        shard.subjects_.push_back(subject);
      }
      return llvm::Error::success();
    };

    for (const typename Traits::Clause &clause : constraints.clauses()) {
      if (const auto *restriction =
              std::get_if<typename Traits::Restriction>(&clause)) {
        if (llvm::Error error =
                remember(restriction->projection, restriction->subject))
          return std::move(error);
        continue;
      }
      if (const auto *equal = std::get_if<typename Traits::Equal>(&clause)) {
        for (const typename Traits::Subject &subject : equal->subjects)
          if (llvm::Error error = remember(equal->projection, subject))
            return std::move(error);
        continue;
      }
      // A Spatial no-good carries no projection and no subjects, so it
      // interns nothing here; it is indexed separately after this build.
      const auto *disjoint = std::get_if<typename Traits::Disjoint>(&clause);
      if (!disjoint)
        continue;
      for (const typename Traits::Subject &subject : disjoint->subjects)
        if (llvm::Error error = remember(disjoint->projection, subject))
          return std::move(error);
    }

    std::vector<DisjointSet> equality;
    equality.reserve(result.shards_.size());
    for (const typename Traits::Shard &shard : result.shards_)
      equality.emplace_back(shard.subjects_.size());

    for (const typename Traits::Clause &clause : constraints.clauses()) {
      const auto *equal = std::get_if<typename Traits::Equal>(&clause);
      if (!equal)
        continue;
      typename Traits::Shard &shard =
          result.shards_[projectionOrdinal(equal->projection)];
      const PnrIndex first = findSubject(shard, equal->subjects.front());
      for (const typename Traits::Subject &subject :
           llvm::drop_begin(equal->subjects))
        equality[projectionOrdinal(equal->projection)].unite(
            first, findSubject(shard, subject));
    }

    for (std::size_t ordinal = 0; ordinal < result.shards_.size(); ++ordinal) {
      typename Traits::Shard &shard = result.shards_[ordinal];
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
        [&](typename Traits::Shard &shard,
            llvm::ArrayRef<typename Traits::Subject> subjects,
            std::vector<FrozenConstraintRelation> &rows) -> llvm::Error {
      auto offset =
          checkedIndex(relationOffsetContext, shard.relationMembers_.size());
      if (!offset)
        return offset.takeError();
      if (llvm::Error error =
              preflightAppend(relationCountContext,
                              shard.relationMembers_.size(), subjects.size()))
        return error;
      for (const typename Traits::Subject &subject : subjects)
        shard.relationMembers_.push_back(findSubject(shard, subject));
      auto count = checkedIndex(relationCountContext, subjects.size());
      if (!count)
        return count.takeError();
      rows.push_back({*offset, *count});
      return llvm::Error::success();
    };

    for (const typename Traits::Clause &clause : constraints.clauses()) {
      if (const auto *restriction =
              std::get_if<typename Traits::Restriction>(&clause)) {
        typename Traits::Shard &shard =
            result.shards_[projectionOrdinal(restriction->projection)];
        const PnrIndex subject = findSubject(shard, restriction->subject);
        const PnrIndex representative = shard.subjectRepresentatives_[subject];
        if (restriction->admissibleDomain.empty() &&
            !Traits::allowsEmptyDomain(restriction->projection))
          return Traits::infeasible(
              restriction->projection,
              "an explicit empty domain contradicts the projection "
              "cardinality");
        if (llvm::any_of(shard.restrictions_, [&](const auto &row) {
              return row.subject == representative;
            }))
          return Traits::invalid(
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
      if (const auto *equal = std::get_if<typename Traits::Equal>(&clause)) {
        typename Traits::Shard &shard =
            result.shards_[projectionOrdinal(equal->projection)];
        if (llvm::Error error =
                appendRelation(shard, equal->subjects, shard.equalityClasses_))
          return std::move(error);
        continue;
      }
      const auto *disjoint = std::get_if<typename Traits::Disjoint>(&clause);
      if (!disjoint)
        continue;
      typename Traits::Shard &shard =
          result.shards_[projectionOrdinal(disjoint->projection)];
      if (llvm::Error error =
              appendRelation(shard, disjoint->subjects, shard.disjointGroups_))
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
  // A set that carries only no-goods still constrains the search, so it is not
  // empty. Consumers that early-out on empty() must not skip them.
  return noGoods_.empty() &&
         llvm::all_of(shards_, [](const FrozenConstraintShard &shard) {
           return shard.empty();
         });
}

SystemFrozenConstraintIndex::SystemFrozenConstraintIndex() {
  shards_.reserve(projectionCount);
  for (std::uint32_t ordinal = 0; ordinal < projectionCount; ++ordinal) {
    const auto projection =
        ::mapping::symbolizeSystemConstraintProjection(ordinal);
    if (!projection)
      llvm_unreachable("System constraint projection catalog has a gap");
    shards_.push_back(SystemFrozenConstraintShard(*projection));
  }
}

const SystemFrozenConstraintShard &SystemFrozenConstraintIndex::shard(
    SystemConstraintProjection projection) const {
  const std::size_t ordinal = projectionOrdinal(projection);
  assert(ordinal < shards_.size());
  return shards_[ordinal];
}

bool SystemFrozenConstraintIndex::empty() const {
  return llvm::all_of(shards_, [](const SystemFrozenConstraintShard &shard) {
    return shard.empty();
  });
}

namespace {

template <typename DomainValue, typename Shard, typename Subject>
std::optional<llvm::ArrayRef<DomainValue>>
lookupRestrictedDomain(const Shard &shard, const Subject &subject) {
  const auto found = llvm::find(shard.subjects(), subject);
  if (found == shard.subjects().end())
    return std::nullopt;
  const PnrIndex subjectIndex =
      static_cast<PnrIndex>(found - shard.subjects().begin());
  const PnrIndex representative = shard.subjectRepresentatives()[subjectIndex];
  const auto restriction = llvm::find_if(
      shard.restrictions(), [&](const FrozenConstraintRestriction &row) {
        return row.subject == representative;
      });
  if (restriction == shard.restrictions().end())
    return std::nullopt;
  return llvm::ArrayRef<DomainValue>(shard.domainValues())
      .slice(restriction->domainOffset, restriction->domainCount);
}

template <typename Shard> bool shardIsEmpty(const Shard &shard) {
  return shard.restrictions().empty() && shard.equalityClasses().empty() &&
         shard.disjointGroups().empty();
}

} // namespace

std::optional<llvm::ArrayRef<SpatialConstraintDomainValue>>
FrozenConstraintShard::restrictedDomain(
    const SpatialConstraintSubject &subject) const {
  return lookupRestrictedDomain<SpatialConstraintDomainValue>(*this, subject);
}

bool FrozenConstraintShard::empty() const { return shardIsEmpty(*this); }

std::optional<llvm::ArrayRef<SystemConstraintDomainValue>>
SystemFrozenConstraintShard::restrictedDomain(
    const SystemConstraintSubject &subject) const {
  return lookupRestrictedDomain<SystemConstraintDomainValue>(*this, subject);
}

bool SystemFrozenConstraintShard::empty() const { return shardIsEmpty(*this); }

llvm::Expected<FrozenConstraintIndex>
loom::pnr::detail::buildFrozenConstraintIndex(
    const SpatialMappingConstraintSetView &constraints) {
  auto index =
      FrozenConstraintIndexBuilder<SpatialConstraintIndexTraits>::build(
          constraints);
  if (!index)
    return index;
  // No-goods span projections, so they are indexed here rather than in any one
  // shard. The importer already established that each clause is non-empty and
  // canonically ordered, so this is a copy, not a second canonicalization.
  for (const SpatialConstraintClauseView &clause : constraints.clauses()) {
    const auto *noGood =
        std::get_if<SpatialRuntimeCounterexampleNoGoodView>(&clause);
    if (!noGood)
      continue;
    if (noGood->literals.empty())
      return SpatialConstraintIndexTraits::invalid(
          "canonical MappingConstraintSet holds an empty no-good clause");
    index->noGoods_.push_back(FrozenConstraintNoGood{noGood->literals});
  }
  return index;
}

llvm::Expected<SystemFrozenConstraintIndex>
loom::pnr::detail::buildFrozenConstraintIndex(
    const SystemMappingConstraintSetView &constraints) {
  return FrozenConstraintIndexBuilder<SystemConstraintIndexTraits>::build(
      constraints);
}
