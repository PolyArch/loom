#ifndef LOOM_PNR_FROZENCONSTRAINTINDEX_H
#define LOOM_PNR_FROZENCONSTRAINTINDEX_H

#include "Mapping/Artifact/MappingConstraintSet.h"
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
  friend class FrozenConstraintIndexBuilder;
};

class FrozenConstraintIndex final {
public:
  static constexpr std::size_t projectionCount =
      ::mapping::getMaxEnumValForSpatialConstraintProjection() + 1;

  const FrozenConstraintShard &
  shard(::mapping::SpatialConstraintProjection projection) const;
  bool empty() const;

private:
  FrozenConstraintIndex();

  std::vector<FrozenConstraintShard> shards_;

  friend class FrozenConstraintIndexBuilder;
};

namespace detail {
llvm::Expected<FrozenConstraintIndex> buildFrozenConstraintIndex(
    const ::loom::mapping::SpatialMappingConstraintSetView &constraints);
} // namespace detail

} // namespace loom::pnr

#endif // LOOM_PNR_FROZENCONSTRAINTINDEX_H
