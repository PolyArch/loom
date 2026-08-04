#ifndef LOOM_LIB_PNR_SPATIALMEMORYCONSTRAINTMODEL_H
#define LOOM_LIB_PNR_SPATIALMEMORYCONSTRAINTMODEL_H

#include "PnR/FrozenConstraintIndex.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace loom::pnr {

class FrozenSpatialMemoryIndex;
struct SpatialLogicalMemoryBindingSelection;

namespace detail {

enum class SpatialMemoryConstraintProjection : std::uint8_t {
  BoundServices,
  AddressRegion,
};

enum class SpatialMemoryConstraintRelationKind : std::uint8_t {
  Equal,
  Disjoint,
};

struct SpatialMemoryAddressInterval final {
  PnrIndex service = 0;
  std::uint64_t lower = 0;
  std::uint64_t upper = 0;

  friend bool operator==(const SpatialMemoryAddressInterval &lhs,
                         const SpatialMemoryAddressInterval &rhs) {
    return lhs.service == rhs.service && lhs.lower == rhs.lower &&
           lhs.upper == rhs.upper;
  }
};

struct SpatialMemoryConstraintRelation final {
  SpatialMemoryConstraintProjection projection =
      SpatialMemoryConstraintProjection::BoundServices;
  SpatialMemoryConstraintRelationKind kind =
      SpatialMemoryConstraintRelationKind::Equal;
  PnrIndex memberOffset = 0;
  PnrIndex memberCount = 0;
};

/// Dense removable projection of logical-memory service and address clauses.
/// MemoryBinding remains the sole selected-state authority.
class SpatialMemoryConstraintModel final {
public:
  static llvm::Expected<std::shared_ptr<const SpatialMemoryConstraintModel>>
  create(const FrozenSpatialMemoryIndex &memory,
         const FrozenConstraintIndex &constraints);

  llvm::Expected<PnrIndex> logicalBindingChoiceCapacity(PnrIndex binding) const;

  llvm::Expected<PnrIndex> collectLogicalBindingChoices(
      PnrIndex binding,
      llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> current,
      llvm::MutableArrayRef<SpatialLogicalMemoryBindingSelection> output) const;

  llvm::Error
  verify(llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections) const;

  bool hasConstraints() const { return hasConstraints_; }

private:
  struct RootDomain final {
    PnrIndex serviceOffset = 0;
    PnrIndex serviceCount = 0;
    PnrIndex addressOffset = 0;
    PnrIndex addressCount = 0;
    bool servicesRestricted = false;
    bool addressesRestricted = false;
  };

  struct TargetProjection final {
    PnrIndex service = getInvalidPnrIndex();
    std::uint64_t addressBaseBytes = 0;
  };

  llvm::Error
  projectRoot(PnrIndex root,
              llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections,
              std::vector<PnrIndex> &services,
              std::vector<SpatialMemoryAddressInterval> &addresses) const;
  llvm::Error verifyRootDomain(
      PnrIndex root,
      llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections) const;
  llvm::Error verifyRelation(
      const SpatialMemoryConstraintRelation &relation,
      llvm::ArrayRef<SpatialLogicalMemoryBindingSelection> selections) const;

  std::vector<std::optional<std::uint64_t>> bindingExtents_;
  std::vector<std::uint64_t> targetSizes_;
  std::vector<PnrIndex> bindingRoots_;
  std::vector<PnrIndex> rootBindingOffsets_;
  std::vector<PnrIndex> rootBindings_;
  std::vector<TargetProjection> targets_;
  std::vector<RootDomain> rootDomains_;
  std::vector<PnrIndex> serviceDomainValues_;
  std::vector<SpatialMemoryAddressInterval> addressDomainValues_;
  std::vector<SpatialMemoryConstraintRelation> relations_;
  std::vector<PnrIndex> relationMembers_;
  bool hasAddressDomainRestrictions_ = false;
  bool hasConstraints_ = false;
};

} // namespace detail
} // namespace loom::pnr

#endif // LOOM_LIB_PNR_SPATIALMEMORYCONSTRAINTMODEL_H
