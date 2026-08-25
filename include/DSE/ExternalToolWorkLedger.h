#ifndef LOOM_DSE_EXTERNALTOOLWORKLEDGER_H
#define LOOM_DSE_EXTERNALTOOLWORKLEDGER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace loom::dse {

inline constexpr std::size_t externalToolWorkLedgerCounterCount = 14;
using ExternalToolWorkLedgerCounters =
    std::array<std::uint64_t, externalToolWorkLedgerCounterCount>;

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

llvm::Error
validateExternalToolWorkLedger(const ExternalToolWorkLedger &ledger);

ExternalToolWorkLedgerCounters
externalToolWorkLedgerCounters(const ExternalToolWorkLedger &ledger);

llvm::Expected<ExternalToolWorkLedger> externalToolWorkLedgerFromCounters(
    const ExternalToolWorkLedgerCounters &counters);

llvm::Error
accumulateExternalToolWorkLedger(ExternalToolWorkLedger &total,
                                 const ExternalToolWorkLedger &addition);

struct PlanNodeExternalToolWorkLedger final {
  std::uint64_t planNodeOrdinal = 0;
  ExternalToolWorkLedger work;

  friend bool operator==(const PlanNodeExternalToolWorkLedger &lhs,
                         const PlanNodeExternalToolWorkLedger &rhs) {
    return lhs.planNodeOrdinal == rhs.planNodeOrdinal && lhs.work == rhs.work;
  }
};

/// Canonical immutable projection of Journal external-tool work. The total is
/// derived from strictly increasing plan-node rows and cannot diverge from
/// them.
class InvocationExternalToolWorkLedger final {
public:
  static llvm::Expected<InvocationExternalToolWorkLedger>
  get(llvm::ArrayRef<PlanNodeExternalToolWorkLedger> planNodes);

  const ExternalToolWorkLedger &total() const { return total_; }
  llvm::ArrayRef<PlanNodeExternalToolWorkLedger> planNodes() const {
    return planNodes_;
  }

  friend bool operator==(const InvocationExternalToolWorkLedger &lhs,
                         const InvocationExternalToolWorkLedger &rhs) {
    return lhs.total_ == rhs.total_ && lhs.planNodes_ == rhs.planNodes_;
  }
  friend bool operator!=(const InvocationExternalToolWorkLedger &lhs,
                         const InvocationExternalToolWorkLedger &rhs) {
    return !(lhs == rhs);
  }

private:
  InvocationExternalToolWorkLedger(
      ExternalToolWorkLedger total,
      std::vector<PlanNodeExternalToolWorkLedger> planNodes)
      : total_(total), planNodes_(std::move(planNodes)) {}

  ExternalToolWorkLedger total_;
  std::vector<PlanNodeExternalToolWorkLedger> planNodes_;
};

} // namespace loom::dse

#endif // LOOM_DSE_EXTERNALTOOLWORKLEDGER_H
