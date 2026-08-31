#ifndef LOOM_PNR_FROZENCONSTRAINTINDEX_H
#define LOOM_PNR_FROZENCONSTRAINTINDEX_H

#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::pnr {

class FrozenConstraintIndex;
class SystemFrozenConstraintIndex;
class FrozenSpatialTransferIndex;
class FrozenEndpointRoutingTopology;

namespace detail {
llvm::Expected<FrozenConstraintIndex> buildFrozenConstraintIndex(
    const ::loom::mapping::SpatialMappingConstraintSetView &constraints);
llvm::Expected<SystemFrozenConstraintIndex> buildFrozenConstraintIndex(
    const ::loom::mapping::SystemMappingConstraintSetView &constraints);
/// Resolves every no-good literal against the frozen transfer and routing
/// indexes. A literal that names a producer, sink, traversal, or endpoint the
/// frozen domain does not own is a freeze failure, never a silently dropped
/// literal: dropping one would weaken the clause into a different no-good.
/// The clause already binds the exact Dataflow owner, so endpoint references
/// compare directly against the frozen domain.
llvm::Error
resolveFrozenConstraintNoGoods(FrozenConstraintIndex &constraints,
                               const FrozenSpatialTransferIndex &transfers,
                               const FrozenEndpointRoutingTopology &routing);
} // namespace detail

enum class SpatialPnrFreezeFailureKind : std::uint32_t {
  Invalid,
  ProvenInfeasible,
};

class SpatialPnrFreezeFailure final
    : public llvm::ErrorInfo<SpatialPnrFreezeFailure> {
public:
  static char ID;

  SpatialPnrFreezeFailure(SpatialPnrFreezeFailureKind kind, std::string message,
                          std::optional<::mapping::SpatialConstraintProjection>
                              projection = std::nullopt)
      : kind_(kind), message_(std::move(message)), projection_(projection) {}

  SpatialPnrFreezeFailureKind kind() const { return kind_; }
  const std::optional<::mapping::SpatialConstraintProjection> &
  projection() const {
    return projection_;
  }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  SpatialPnrFreezeFailureKind kind_;
  std::string message_;
  std::optional<::mapping::SpatialConstraintProjection> projection_;
};

struct FrozenConstraintRestriction final {
  PnrIndex subject = 0;
  PnrIndex domainOffset = 0;
  PnrIndex domainCount = 0;
};

struct FrozenConstraintRelation final {
  PnrIndex memberOffset = 0;
  PnrIndex memberCount = 0;
};

template <typename Traits> class FrozenConstraintIndexBuilder;

class FrozenConstraintShard final {
public:
  ::mapping::SpatialConstraintProjection projection() const {
    return projection_;
  }
  llvm::ArrayRef<::loom::mapping::SpatialConstraintSubject> subjects() const {
    return subjects_;
  }
  llvm::ArrayRef<PnrIndex> subjectRepresentatives() const {
    return subjectRepresentatives_;
  }
  llvm::ArrayRef<::loom::mapping::SpatialConstraintDomainValue>
  domainValues() const {
    return domainValues_;
  }
  llvm::ArrayRef<FrozenConstraintRestriction> restrictions() const {
    return restrictions_;
  }
  llvm::ArrayRef<FrozenConstraintRelation> equalityClasses() const {
    return equalityClasses_;
  }
  llvm::ArrayRef<FrozenConstraintRelation> disjointGroups() const {
    return disjointGroups_;
  }
  llvm::ArrayRef<PnrIndex> relationMembers() const { return relationMembers_; }

  std::optional<llvm::ArrayRef<::loom::mapping::SpatialConstraintDomainValue>>
  restrictedDomain(
      const ::loom::mapping::SpatialConstraintSubject &subject) const;
  bool empty() const;

private:
  explicit FrozenConstraintShard(
      ::mapping::SpatialConstraintProjection projection)
      : projection_(projection) {}

  ::mapping::SpatialConstraintProjection projection_;
  std::vector<::loom::mapping::SpatialConstraintSubject> subjects_;
  std::vector<PnrIndex> subjectRepresentatives_;
  std::vector<::loom::mapping::SpatialConstraintDomainValue> domainValues_;
  std::vector<FrozenConstraintRestriction> restrictions_;
  std::vector<FrozenConstraintRelation> equalityClasses_;
  std::vector<FrozenConstraintRelation> disjointGroups_;
  std::vector<PnrIndex> relationMembers_;

  friend class FrozenConstraintIndex;
  template <typename Traits> friend class FrozenConstraintIndexBuilder;
};

/// One frozen runtime-counterexample no-good: the listed exact choices may not
/// all hold at once. Literals are canonical and duplicate-free, and the clause
/// is never empty.
struct FrozenConstraintNoGood final {
  std::vector<::loom::mapping::SpatialNoGoodLiteral> literals;
};

/// One no-good literal resolved against the frozen search domain, so evaluating
/// a clause is an exact integer test with no reference decoding. `sink` is a
/// net-local sink ordinal and is engaged for branch-scoped traversal literals
/// and sink attachment literals.
struct FrozenNoGoodResolvedLiteral final {
  enum class Kind : std::uint8_t {
    NetUsesTraversal,
    TransferAttachmentEquals,
    NetTagEquals,
  };

  Kind kind = Kind::NetUsesTraversal;
  PnrIndex logicalNet = 0;
  std::optional<PnrIndex> sink;
  /// Traversal ordinal for NetUsesTraversal, endpoint ordinal for
  /// TransferAttachmentEquals.
  PnrIndex target = 0;
  /// Exact expected tag bits for NetTagEquals; absent for the other kinds.
  std::optional<llvm::APInt> tagValue;
};

/// One no-good clause resolved against the frozen search domain. Offsets index
/// the index-owned resolved-literal array.
struct FrozenNoGoodResolvedClause final {
  PnrIndex literalOffset = 0;
  PnrIndex literalCount = 0;
};

class FrozenConstraintIndex final {
public:
  static constexpr std::size_t projectionCount =
      ::mapping::getMaxEnumValForSpatialConstraintProjection() + 1;

  const FrozenConstraintShard &
  shard(::mapping::SpatialConstraintProjection projection) const;
  /// No-goods are cross-projection, so they are held by the index rather than
  /// by any one projection shard.
  llvm::ArrayRef<FrozenConstraintNoGood> noGoods() const { return noGoods_; }
  /// The same no-goods resolved against the frozen search domain. Empty until
  /// resolveFrozenConstraintNoGoods runs, which requires the frozen routing
  /// and transfer indexes that are built after the constraint index itself.
  llvm::ArrayRef<FrozenNoGoodResolvedClause> resolvedNoGoods() const {
    return resolvedNoGoods_;
  }
  llvm::ArrayRef<FrozenNoGoodResolvedLiteral> resolvedNoGoodLiterals() const {
    return resolvedNoGoodLiterals_;
  }
  /// Canonical clause ordinals affected by one logical net. This CSR index is
  /// derived from resolved literals and is never an artifact identity owner.
  llvm::ArrayRef<PnrIndex>
  resolvedNoGoodClausesForNet(PnrIndex logicalNet) const {
    assert(!resolvedNoGoodNetClauseOffsets_.empty() &&
           logicalNet < resolvedNoGoodNetClauseOffsets_.size() - 1);
    const PnrIndex begin = resolvedNoGoodNetClauseOffsets_[logicalNet];
    const PnrIndex end = resolvedNoGoodNetClauseOffsets_[logicalNet + 1];
    return llvm::ArrayRef<PnrIndex>(resolvedNoGoodNetClauses_)
        .slice(begin, end - begin);
  }
  bool empty() const;

private:
  FrozenConstraintIndex();

  std::vector<FrozenConstraintShard> shards_;
  std::vector<FrozenConstraintNoGood> noGoods_;
  std::vector<FrozenNoGoodResolvedClause> resolvedNoGoods_;
  std::vector<FrozenNoGoodResolvedLiteral> resolvedNoGoodLiterals_;
  std::vector<PnrIndex> resolvedNoGoodNetClauseOffsets_;
  std::vector<PnrIndex> resolvedNoGoodNetClauses_;

  template <typename Traits> friend class FrozenConstraintIndexBuilder;
  friend llvm::Expected<FrozenConstraintIndex>
  detail::buildFrozenConstraintIndex(
      const ::loom::mapping::SpatialMappingConstraintSetView &constraints);
  friend llvm::Error detail::resolveFrozenConstraintNoGoods(
      FrozenConstraintIndex &constraints,
      const FrozenSpatialTransferIndex &transfers,
      const FrozenEndpointRoutingTopology &routing);
};

class SystemFrozenConstraintShard final {
public:
  ::mapping::SystemConstraintProjection projection() const {
    return projection_;
  }
  llvm::ArrayRef<::loom::mapping::SystemConstraintSubject> subjects() const {
    return subjects_;
  }
  llvm::ArrayRef<PnrIndex> subjectRepresentatives() const {
    return subjectRepresentatives_;
  }
  llvm::ArrayRef<::loom::mapping::SystemConstraintDomainValue>
  domainValues() const {
    return domainValues_;
  }
  llvm::ArrayRef<FrozenConstraintRestriction> restrictions() const {
    return restrictions_;
  }
  llvm::ArrayRef<FrozenConstraintRelation> equalityClasses() const {
    return equalityClasses_;
  }
  llvm::ArrayRef<FrozenConstraintRelation> disjointGroups() const {
    return disjointGroups_;
  }
  llvm::ArrayRef<PnrIndex> relationMembers() const { return relationMembers_; }

  std::optional<llvm::ArrayRef<::loom::mapping::SystemConstraintDomainValue>>
  restrictedDomain(
      const ::loom::mapping::SystemConstraintSubject &subject) const;
  bool empty() const;

private:
  explicit SystemFrozenConstraintShard(
      ::mapping::SystemConstraintProjection projection)
      : projection_(projection) {}

  ::mapping::SystemConstraintProjection projection_;
  std::vector<::loom::mapping::SystemConstraintSubject> subjects_;
  std::vector<PnrIndex> subjectRepresentatives_;
  std::vector<::loom::mapping::SystemConstraintDomainValue> domainValues_;
  std::vector<FrozenConstraintRestriction> restrictions_;
  std::vector<FrozenConstraintRelation> equalityClasses_;
  std::vector<FrozenConstraintRelation> disjointGroups_;
  std::vector<PnrIndex> relationMembers_;

  friend class SystemFrozenConstraintIndex;
  template <typename Traits> friend class FrozenConstraintIndexBuilder;
};

class SystemFrozenConstraintIndex final {
public:
  static constexpr std::size_t projectionCount =
      ::mapping::getMaxEnumValForSystemConstraintProjection() + 1;

  const SystemFrozenConstraintShard &
  shard(::mapping::SystemConstraintProjection projection) const;
  bool empty() const;

private:
  SystemFrozenConstraintIndex();

  std::vector<SystemFrozenConstraintShard> shards_;

  template <typename Traits> friend class FrozenConstraintIndexBuilder;
};

namespace detail {
llvm::Expected<FrozenConstraintIndex> buildFrozenConstraintIndex(
    const ::loom::mapping::SpatialMappingConstraintSetView &constraints);
llvm::Expected<SystemFrozenConstraintIndex> buildFrozenConstraintIndex(
    const ::loom::mapping::SystemMappingConstraintSetView &constraints);
} // namespace detail

} // namespace loom::pnr

#endif // LOOM_PNR_FROZENCONSTRAINTINDEX_H
