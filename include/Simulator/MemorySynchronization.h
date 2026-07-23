#ifndef LOOM_SIMULATOR_MEMORYSYNCHRONIZATION_H
#define LOOM_SIMULATOR_MEMORYSYNCHRONIZATION_H

#include "Simulator/MemoryAtomicOrder.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <system_error>

namespace loom {
namespace sim {

/// Execution-local identity of one memory effect that participates in software
/// ordering. Only MemorySynchronization::declareEffect allocates one. It is a
/// handle, not the MemoryAction projection: actor occurrence, contract
/// reference, operands, and lanes stay with the caller.
class SyncEffectId {
public:
  explicit constexpr SyncEffectId(std::uint64_t value) : value_(value) {}

  constexpr std::uint64_t value() const { return value_; }

  friend constexpr bool operator==(SyncEffectId lhs, SyncEffectId rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(SyncEffectId lhs, SyncEffectId rhs) {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(SyncEffectId lhs, SyncEffectId rhs) {
    return lhs.value_ < rhs.value_;
  }

private:
  std::uint64_t value_;
};

/// Caller-resolved execution-local identity of one resolved synchronization
/// scope. The caller owns the resolution from a SyncScopeRef to a participant
/// domain; this engine only compares identities. It is a closed numeric
/// identity, never a string, and there is no scope lattice: two domains are
/// either the same or unrelated.
class SyncDomainId {
public:
  explicit constexpr SyncDomainId(std::uint64_t value) : value_(value) {}

  constexpr std::uint64_t value() const { return value_; }

  friend constexpr bool operator==(SyncDomainId lhs, SyncDomainId rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(SyncDomainId lhs, SyncDomainId rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint64_t value_;
};

/// The directional halves an effect declares. A seq_cst actor uses AcqRel here;
/// this engine owns no sequentially consistent total order.
enum class SyncRoleKind {
  Release,
  Acquire,
  AcqRel,
};

class MemorySynchronizationError final
    : public llvm::ErrorInfo<MemorySynchronizationError> {
public:
  /// Listed in rejection precedence order, so an update that breaks several
  /// rules always names the same one.
  enum class Kind {
    UnknownEffect,
    ForeignRelation,
    InitialVersionPublication,
    MismatchedCarry,
    DuplicateAssociation,
    DuplicateRole,
    RoleShapeConflict,
    DuplicateEdge,
    CyclicOrder,
    UnknownRole,
  };

  static char ID;

  MemorySynchronizationError(Kind kind, std::string message);

  Kind kind() const { return kind_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  Kind kind_;
  std::string message_;
};

/// Nonpersistent execution-local owner of the software relations that follow
/// reads-from: release visibility summaries, acquire-imported visibility,
/// synchronizes-with, happens-before, and the fence-through-reads-from
/// relation.
///
/// It builds on a bound MemoryAtomicOrder and never duplicates it: modification
/// order and reads-from stay there and are resolved through its record
/// accessors. This engine adds only the effect attribution that order
/// deliberately omits, plus the caller's sequenced-before facts.
///
/// Sequenced-before is explicit caller input from the finalized program. The
/// engine never infers it from MLIR, container layout, or host scheduling. It
/// owns no values, no lifecycle state, no timing, no scope interpretation, and
/// no sequentially consistent order.
///
/// Every relation view is a pure function of the accepted facts, so a caller
/// that records the same facts in a different valid order observes the same
/// relations. Views are returned as immutable snapshots, never as references
/// into engine state. There is no cache to invalidate.
///
/// Every rejected update is atomic: it consumes no effect id and leaves every
/// fact, association, and derived relation untouched.
class MemorySynchronization {
public:
  explicit MemorySynchronization(const MemoryAtomicOrder &order)
      : order_(&order) {}

  /// The sole source of effect identities.
  SyncEffectId declareEffect();

  /// Records one sequenced-before fact of the finalized program.
  llvm::Error sequencedBefore(SyncEffectId earlier, SyncEffectId later);

  /// Binds one effect to the version it appended, in exactly one resolved
  /// domain. `readsFrom` is present exactly for a read-modify-write or a
  /// successful compare-exchange, which makes the effect both a read and a
  /// write carrier and lets a release sequence carry through it.
  llvm::Error
  registerWrite(SyncEffectId effect, SyncDomainId domain,
                AtomicVersionId version,
                std::optional<AtomicReadId> readsFrom = std::nullopt);

  /// Binds one effect to the reads-from relation it selected, in exactly one
  /// resolved domain.
  llvm::Error registerRead(SyncEffectId effect, SyncDomainId domain,
                           AtomicReadId read);

  /// Declares an operation-shaped role, which publishes or imports through the
  /// effect's own carrier and never hooks another effect's carrier. Release
  /// requires a write carrier, Acquire requires a read carrier or a carried
  /// write, and AcqRel requires a carried write. The role has no domain of its
  /// own: it uses the carrier's.
  llvm::Error declareOperationRole(SyncEffectId effect, SyncRoleKind kind);

  /// Declares a fence-shaped role, which publishes through a sequenced-after
  /// write carrier or imports through a sequenced-before read carrier in the
  /// same domain. A fence has no addressed access, so it must have no carrier;
  /// that rule holds in either declaration order, so a carrier registration on
  /// an effect that already has a fence role is the same conflict.
  llvm::Error declareFenceRole(SyncEffectId effect, SyncRoleKind kind,
                               SyncDomainId domain);

  /// True when `origin` publishes a release that `target` imports through a
  /// recorded reads-from relation, with one domain identity across the origin,
  /// the publishing carrier, every release-sequence hop, the reading carrier,
  /// and the target. A domain mismatch or an absent reads-from is legal and
  /// simply yields no relation.
  bool synchronizesWith(SyncEffectId origin, SyncEffectId target) const;

  /// The transitive closure of sequenced-before and synchronizes-with.
  bool happensBefore(SyncEffectId earlier, SyncEffectId later) const;

  /// The release origins published through one version, which is origin and
  /// domain metadata rather than a visibility summary.
  llvm::Expected<llvm::SmallVector<SyncEffectId>>
  publishedOrigins(AtomicVersionId version) const;

  /// The exact effect set a release origin publishes, which is its
  /// happens-before predecessors and therefore already contains everything the
  /// origin's strand imported earlier.
  llvm::Expected<llvm::SmallVector<SyncEffectId>>
  visibilitySummary(SyncEffectId origin) const;

  /// The exact effect set an acquire-side role imports: every compatible
  /// origin's summary plus the origin. Applicability to effects sequenced after
  /// the acquire is happensBefore, not a second propagated-import view, so only
  /// an acquire-side role may be inspected here.
  llvm::Expected<llvm::SmallVector<SyncEffectId>>
  importedVisibility(SyncEffectId target) const;

private:
  /// One addressed access: exactly one resolved domain plus the atomic
  /// relations it owns. A carried write holds both.
  struct Carrier {
    SyncDomainId domain;
    std::optional<AtomicVersionId> version;
    std::optional<AtomicReadId> read;
  };

  /// A present `fenceDomain` is the fence shape and its resolved domain; an
  /// absent one is the operation shape, whose domain is the carrier's.
  struct Role {
    SyncRoleKind kind;
    std::optional<SyncDomainId> fenceDomain;
  };

  /// The whole mutable state. Every update validates a candidate copy and
  /// installs it only after the complete proposed relation graph is accepted,
  /// which is what makes rejection atomic by construction.
  struct Facts {
    std::uint64_t effects = 0;
    std::map<std::uint64_t, llvm::SmallVector<SyncEffectId, 2>> sequenced;
    std::map<std::uint64_t, Carrier> carriers;
    std::map<std::uint64_t, Role> roles;
    std::map<std::uint64_t, SyncEffectId> versionOwner;
    std::map<std::uint64_t, SyncEffectId> readOwner;
  };

  using Graph = std::map<std::uint64_t, llvm::SmallVector<SyncEffectId, 2>>;

  llvm::Error requireKnown(SyncEffectId effect) const;
  /// A fence has no addressed access, so it can never become a carrier. This is
  /// the mirror of the carrier check in declareFenceRole and keeps the shape
  /// rule independent of which fact the caller records first.
  llvm::Error requireNoFenceRole(SyncEffectId effect) const;
  llvm::Error commit(Facts candidate);

  const Carrier *carrierOf(const Facts &facts, SyncEffectId effect) const;
  const Role *roleOf(const Facts &facts, SyncEffectId effect) const;
  std::optional<SyncDomainId> domainOf(const Facts &facts,
                                       SyncEffectId effect) const;
  bool sequencedReaches(const Facts &facts, SyncEffectId from,
                        SyncEffectId to) const;

  llvm::SmallVector<SyncEffectId> collectOrigins(const Facts &facts,
                                                 AtomicVersionId version) const;
  void forEachSynchronization(
      const Facts &facts,
      llvm::function_ref<void(SyncEffectId, SyncEffectId)> action) const;
  Graph buildGraph(const Facts &facts, bool reversed) const;

  const MemoryAtomicOrder *order_;
  Facts facts_;
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_MEMORYSYNCHRONIZATION_H
