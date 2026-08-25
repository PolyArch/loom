#ifndef LOOM_DSE_EXTERNALTOOLWORKLEDGER_H
#define LOOM_DSE_EXTERNALTOOLWORKLEDGER_H

#include <cstddef>
#include <cstdint>

namespace loom::dse {

inline constexpr std::size_t externalToolWorkLedgerCounterCount = 14;

/// Additive operational accounting for exact external-tool work. Cache
/// lookup and publication are observations of reserved work, not independent
/// work units.
struct ExternalToolWorkLedger final {
  std::uint64_t planned = 0;
  std::uint64_t reserved = 0;
  std::uint64_t consumed = 0;
  std::uint64_t avoided = 0;
  std::uint64_t cacheDisabled = 0;
  std::uint64_t cacheAvailable = 0;
  std::uint64_t cacheUnavailable = 0;
  std::uint64_t cacheHits = 0;
  std::uint64_t cacheMisses = 0;
  std::uint64_t cacheLockWaits = 0;
  std::uint64_t cacheDiscards = 0;
  std::uint64_t cacheDiscardFailures = 0;
  std::uint64_t cachePublications = 0;
  std::uint64_t cachePublicationFailures = 0;

  friend bool operator==(const ExternalToolWorkLedger &lhs,
                         const ExternalToolWorkLedger &rhs) {
    return lhs.planned == rhs.planned && lhs.reserved == rhs.reserved &&
           lhs.consumed == rhs.consumed && lhs.avoided == rhs.avoided &&
           lhs.cacheDisabled == rhs.cacheDisabled &&
           lhs.cacheAvailable == rhs.cacheAvailable &&
           lhs.cacheUnavailable == rhs.cacheUnavailable &&
           lhs.cacheHits == rhs.cacheHits &&
           lhs.cacheMisses == rhs.cacheMisses &&
           lhs.cacheLockWaits == rhs.cacheLockWaits &&
           lhs.cacheDiscards == rhs.cacheDiscards &&
           lhs.cacheDiscardFailures == rhs.cacheDiscardFailures &&
           lhs.cachePublications == rhs.cachePublications &&
           lhs.cachePublicationFailures == rhs.cachePublicationFailures;
  }
  friend bool operator!=(const ExternalToolWorkLedger &lhs,
                         const ExternalToolWorkLedger &rhs) {
    return !(lhs == rhs);
  }
};

} // namespace loom::dse

#endif // LOOM_DSE_EXTERNALTOOLWORKLEDGER_H
