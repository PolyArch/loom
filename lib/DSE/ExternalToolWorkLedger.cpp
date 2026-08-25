#include "DSE/ExternalToolWorkLedger.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <limits>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "external_tool_work_ledger_invalid: " +
                                     message);
}

llvm::Error addCounter(std::uint64_t &total, std::uint64_t addition) {
  if (addition > std::numeric_limits<std::uint64_t>::max() - total)
    return invalid("counter overflows uint64");
  total += addition;
  return llvm::Error::success();
}

} // namespace

llvm::Error
validateExternalToolWorkLedger(const ExternalToolWorkLedger &ledger) {
  if (ledger.consumed > ledger.reserved ||
      ledger.avoided > ledger.reserved - ledger.consumed ||
      ledger.reserved > ledger.planned)
    return invalid("work exceeds its planned reservation");
  if (ledger.avoided != ledger.cacheHits)
    return invalid("avoided work differs from cache hits");
  if (ledger.cacheHits > ledger.cacheAvailable ||
      ledger.cacheMisses > ledger.cacheAvailable - ledger.cacheHits)
    return invalid("cache lookups exceed availability");
  if (ledger.cacheDiscards > ledger.cacheMisses ||
      ledger.cacheDiscardFailures > ledger.cacheMisses - ledger.cacheDiscards)
    return invalid("cache discards exceed misses");
  if (ledger.cachePublications > ledger.cacheMisses ||
      ledger.cachePublicationFailures >
          ledger.cacheMisses - ledger.cachePublications)
    return invalid("cache publications exceed misses");
  if (ledger.cacheDisabled > ledger.reserved ||
      ledger.cacheAvailable > ledger.reserved - ledger.cacheDisabled ||
      ledger.cacheUnavailable >
          ledger.reserved - ledger.cacheDisabled - ledger.cacheAvailable)
    return invalid("cache observations exceed reservations");
  if (ledger.cacheLockWaits > ledger.reserved)
    return invalid("cache waits exceed reservations");
  return llvm::Error::success();
}

ExternalToolWorkLedgerCounters
externalToolWorkLedgerCounters(const ExternalToolWorkLedger &ledger) {
  return {ledger.planned,           ledger.reserved,
          ledger.consumed,          ledger.avoided,
          ledger.cacheDisabled,     ledger.cacheAvailable,
          ledger.cacheUnavailable,  ledger.cacheHits,
          ledger.cacheMisses,       ledger.cacheLockWaits,
          ledger.cacheDiscards,     ledger.cacheDiscardFailures,
          ledger.cachePublications, ledger.cachePublicationFailures};
}

llvm::Expected<ExternalToolWorkLedger> externalToolWorkLedgerFromCounters(
    const ExternalToolWorkLedgerCounters &counters) {
  ExternalToolWorkLedger ledger{
      counters[0],  counters[1],  counters[2],  counters[3], counters[4],
      counters[5],  counters[6],  counters[7],  counters[8], counters[9],
      counters[10], counters[11], counters[12], counters[13]};
  if (llvm::Error error = validateExternalToolWorkLedger(ledger))
    return std::move(error);
  return ledger;
}

llvm::Error
accumulateExternalToolWorkLedger(ExternalToolWorkLedger &total,
                                 const ExternalToolWorkLedger &addition) {
  if (llvm::Error error = validateExternalToolWorkLedger(total))
    return error;
  if (llvm::Error error = validateExternalToolWorkLedger(addition))
    return error;
  ExternalToolWorkLedgerCounters accumulated =
      externalToolWorkLedgerCounters(total);
  const ExternalToolWorkLedgerCounters added =
      externalToolWorkLedgerCounters(addition);
  for (std::size_t index = 0; index != accumulated.size(); ++index)
    if (llvm::Error error = addCounter(accumulated[index], added[index]))
      return error;
  auto decoded = externalToolWorkLedgerFromCounters(accumulated);
  if (!decoded)
    return decoded.takeError();
  total = *decoded;
  return llvm::Error::success();
}

llvm::Expected<InvocationExternalToolWorkLedger>
InvocationExternalToolWorkLedger::get(
    llvm::ArrayRef<PlanNodeExternalToolWorkLedger> planNodes) {
  std::vector<PlanNodeExternalToolWorkLedger> canonical(planNodes.begin(),
                                                        planNodes.end());
  ExternalToolWorkLedger total;
  for (std::size_t index = 0; index != canonical.size(); ++index) {
    if (canonical[index].work.planned == 0)
      return invalid("plan-node row has no planned work");
    if (index != 0 && canonical[index - 1].planNodeOrdinal >=
                          canonical[index].planNodeOrdinal)
      return invalid("plan-node ordinals are not strictly increasing");
    if (llvm::Error error =
            accumulateExternalToolWorkLedger(total, canonical[index].work))
      return std::move(error);
  }
  return InvocationExternalToolWorkLedger(total, std::move(canonical));
}

} // namespace loom::dse
