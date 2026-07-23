#ifndef LOOM_SIMULATOR_MEMORYATOMICORDER_H
#define LOOM_SIMULATOR_MEMORYATOMICORDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace loom {
namespace sim {

/// Exact and sole identity of one dynamic atomic object. A whole-payload
/// access derives one key and a per-lane access derives one key per active
/// lane, so repeated active addresses rebuild the same key and share one
/// modification order. The lane ordinal has no representation here because it
/// is never software ordering. The root id is execution local: it identifies a
/// logical memory root inside one run and is not a persistent artifact id.
struct AtomicObjectKey {
  std::uint64_t logicalRootId = 0;
  std::uint64_t canonicalByteOffset = 0;
  std::uint64_t accessByteSize = 0;

  friend constexpr bool operator==(const AtomicObjectKey &lhs,
                                   const AtomicObjectKey &rhs) {
    return lhs.logicalRootId == rhs.logicalRootId &&
           lhs.canonicalByteOffset == rhs.canonicalByteOffset &&
           lhs.accessByteSize == rhs.accessByteSize;
  }
  friend constexpr bool operator!=(const AtomicObjectKey &lhs,
                                   const AtomicObjectKey &rhs) {
    return !(lhs == rhs);
  }
  /// Total order over the exact key fields, used only to index per-object
  /// state. Distinct atomic objects have no memory-model relation, so this
  /// order carries no semantics.
  friend constexpr bool operator<(const AtomicObjectKey &lhs,
                                  const AtomicObjectKey &rhs) {
    if (lhs.logicalRootId != rhs.logicalRootId)
      return lhs.logicalRootId < rhs.logicalRootId;
    if (lhs.canonicalByteOffset != rhs.canonicalByteOffset)
      return lhs.canonicalByteOffset < rhs.canonicalByteOffset;
    return lhs.accessByteSize < rhs.accessByteSize;
  }
};

/// Identity of one modification-order version. Values are unique across every
/// object of one MemoryAtomicOrder, so a version selected from another object
/// is detectable instead of silently aliased.
class AtomicVersionId {
public:
  explicit constexpr AtomicVersionId(std::uint64_t value) : value_(value) {}

  constexpr std::uint64_t value() const { return value_; }

  friend constexpr bool operator==(AtomicVersionId lhs, AtomicVersionId rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(AtomicVersionId lhs, AtomicVersionId rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint64_t value_;
};

/// Identity of one reads-from record, unique across one MemoryAtomicOrder.
class AtomicReadId {
public:
  explicit constexpr AtomicReadId(std::uint64_t value) : value_(value) {}

  constexpr std::uint64_t value() const { return value_; }

  friend constexpr bool operator==(AtomicReadId lhs, AtomicReadId rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(AtomicReadId lhs, AtomicReadId rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint64_t value_;
};

/// One version of one atomic object's modification order. The engine is its
/// only author; callers observe copies. Values live in logical memory and
/// provider state, never here.
class AtomicVersionRecord {
public:
  AtomicVersionId id() const { return id_; }
  const AtomicObjectKey &key() const { return key_; }
  /// Absent only for the explicit initial version. Every appended version
  /// names the tail it was appended behind.
  const std::optional<AtomicVersionId> &predecessor() const {
    return predecessor_;
  }

private:
  friend class MemoryAtomicOrder;

  AtomicVersionRecord(AtomicVersionId id, const AtomicObjectKey &key,
                      std::optional<AtomicVersionId> predecessor)
      : id_(id), key_(key), predecessor_(predecessor) {}

  AtomicVersionId id_;
  AtomicObjectKey key_;
  std::optional<AtomicVersionId> predecessor_;
};

/// One reads-from relation: the version an action read from its object. The
/// engine is its only author; callers observe copies.
class AtomicReadRecord {
public:
  AtomicReadId id() const { return id_; }
  const AtomicObjectKey &key() const { return key_; }
  AtomicVersionId version() const { return version_; }

private:
  friend class MemoryAtomicOrder;

  AtomicReadRecord(AtomicReadId id, const AtomicObjectKey &key,
                   AtomicVersionId version)
      : id_(id), key_(key), version_(version) {}

  AtomicReadId id_;
  AtomicObjectKey key_;
  AtomicVersionId version_;
};

/// The reads-from relation and the appended version of one atomic
/// read-modify-write, committed together.
struct AtomicRmwResult {
  AtomicReadId read;
  AtomicVersionId version;
};

/// The provider's exact compare-exchange outcome. The engine never compares
/// values and never invents a failure. SpuriousFailure is the explicit choice
/// of a weak compare-exchange provider, so this engine holds no randomness and
/// no seed; supplying it only for a weak actor contract is the caller's duty.
enum class AtomicCompareExchangeDecision {
  Success,
  ComparisonFailure,
  SpuriousFailure,
};

/// One compare-exchange action. The appended version is present only for a
/// successful decision; both failure decisions record the read alone.
struct AtomicCompareExchangeResult {
  AtomicReadId read;
  std::optional<AtomicVersionId> version;
};

class MemoryAtomicOrderError final
    : public llvm::ErrorInfo<MemoryAtomicOrderError> {
public:
  enum class Kind {
    InvalidKey,
    DuplicateInitialization,
    UnknownObject,
    UnknownVersion,
    ForeignVersion,
    StaleVersion,
  };

  static char ID;

  MemoryAtomicOrderError(Kind kind, std::string message);

  Kind kind() const { return kind_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  Kind kind_;
  std::string message_;
};

/// Nonpersistent object-local relation state for dynamic atomic objects: one
/// modification order per AtomicObjectKey plus the reads-from relations that
/// select its versions.
///
/// The engine records the relations a provider selects; it decides nothing. It
/// owns no values, no scheduling, no reads-from choice, no spurious failure, no
/// synchronization scope, no release or acquire visibility, no happens-before,
/// no fence, and no sequentially consistent order.
///
/// Every rejected action is atomic: it appends no version, records no read,
/// consumes no id, and leaves every object untouched.
class MemoryAtomicOrder {
public:
  /// Creates the object's explicit initial version, which has no predecessor.
  llvm::Expected<AtomicVersionId> initializeObject(const AtomicObjectKey &key);

  /// Appends exactly one version behind the current tail.
  llvm::Expected<AtomicVersionId> atomicStore(const AtomicObjectKey &key);

  /// Records a reads-from relation to a provider-selected existing version and
  /// never writes. Narrowing that selection is visibility work this engine does
  /// not own, so any existing version of the object is accepted.
  llvm::Expected<AtomicReadId> atomicLoad(const AtomicObjectKey &key,
                                          AtomicVersionId source);

  /// Reads the immediately preceding version and appends its write in one
  /// commit. A predecessor that is no longer the tail is stale and rejected.
  llvm::Expected<AtomicRmwResult> atomicRmw(const AtomicObjectKey &key,
                                            AtomicVersionId predecessor);

  /// Applies the provider's decision. A successful compare-exchange is a
  /// read-modify-write, so its selection must be the current tail and it
  /// records the read plus one appended version in one commit. Either failure
  /// decision accepts any existing version, records exactly one reads-from
  /// relation, and appends nothing.
  llvm::Expected<AtomicCompareExchangeResult>
  compareExchange(const AtomicObjectKey &key, AtomicVersionId source,
                  AtomicCompareExchangeDecision decision);

  /// The object's modification order, oldest first. Absent when the key has no
  /// explicit initial version. The view is valid until the next accepted
  /// action.
  std::optional<llvm::ArrayRef<AtomicVersionId>>
  modificationOrder(const AtomicObjectKey &key) const;

  std::optional<AtomicVersionRecord> versionRecord(AtomicVersionId id) const;
  std::optional<AtomicReadRecord> readRecord(AtomicReadId id) const;

private:
  using VersionOrder = llvm::SmallVector<AtomicVersionId, 4>;

  /// Validates the key and resolves the modification order it names, rejecting
  /// an object that has no explicit initial version.
  llvm::Expected<VersionOrder *> resolveObject(const AtomicObjectKey &key);

  /// Rejects a version that no engine action produced or that belongs to
  /// another atomic object.
  llvm::Error validateSelection(const AtomicObjectKey &key,
                                AtomicVersionId version) const;

  /// Commit primitives. They allocate the only ids this engine hands out and
  /// run after every validation, so no accepted action can fail partway.
  AtomicVersionId appendVersion(VersionOrder &order, const AtomicObjectKey &key,
                                std::optional<AtomicVersionId> predecessor);
  AtomicReadId recordRead(const AtomicObjectKey &key, AtomicVersionId version);

  std::map<AtomicObjectKey, VersionOrder> objects_;
  std::vector<AtomicVersionRecord> versions_;
  std::vector<AtomicReadRecord> reads_;
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_MEMORYATOMICORDER_H
