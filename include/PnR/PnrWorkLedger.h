#ifndef LOOM_PNR_PNRWORKLEDGER_H
#define LOOM_PNR_PNRWORKLEDGER_H

#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>

namespace loom::pnr {

enum class PnrWorkKind : std::uint8_t {
  SeedAttempt,
  InitializerAssignment,
  EndpointExpansion,
  NegotiationIteration,
  CalibrationProposal,
  AnnealingBaseProposal,
  AnnealingMovableProposal,
  ExactRepairRegionDecision,
  ExactRepairSolverCall,
  LocalTransferAdoptionProbe,
  FinalClosureAttempt,
  Count,
};

inline constexpr std::size_t pnrWorkKindCount =
    static_cast<std::size_t>(PnrWorkKind::Count);

struct PnrWorkCounterRef final {
  std::uint64_t *planned = nullptr;
  std::uint64_t *consumed = nullptr;
};

/// Non-owning access to an invocation's canonical work counters. A semantic
/// owner reserves a logical slot immediately before executing it and consumes
/// that slot only after the owner boundary has run. Unbound views preserve the
/// same algorithms for callers that do not publish provider accounting.
class PnrWorkLedgerView final {
public:
  constexpr PnrWorkLedgerView() = default;
  explicit constexpr PnrWorkLedgerView(
      std::array<PnrWorkCounterRef, pnrWorkKindCount> counters)
      : counters_(counters) {}

  llvm::Error plan(PnrWorkKind kind, std::uint64_t amount = 1) const {
    const PnrWorkCounterRef counter = counters_[ordinal(kind)];
    if (!counter.planned && !counter.consumed)
      return llvm::Error::success();
    if (!counter.planned || !counter.consumed)
      return ledgerError(std::errc::invalid_argument,
                         "work counter binding is incomplete");
    if (amount > std::numeric_limits<std::uint64_t>::max() - *counter.planned)
      return ledgerError(std::errc::value_too_large,
                         "planned work counter overflows u64");
    *counter.planned += amount;
    return llvm::Error::success();
  }

  llvm::Error consume(PnrWorkKind kind, std::uint64_t amount = 1) const {
    const PnrWorkCounterRef counter = counters_[ordinal(kind)];
    if (!counter.planned && !counter.consumed)
      return llvm::Error::success();
    if (!counter.planned || !counter.consumed)
      return ledgerError(std::errc::invalid_argument,
                         "work counter binding is incomplete");
    if (*counter.consumed > *counter.planned ||
        amount > *counter.planned - *counter.consumed)
      return ledgerError(std::errc::invalid_argument,
                         "consumed work exceeds planned work");
    *counter.consumed += amount;
    return llvm::Error::success();
  }

private:
  static constexpr std::size_t ordinal(PnrWorkKind kind) {
    return static_cast<std::size_t>(kind);
  }

  static llvm::Error ledgerError(std::errc code, const char *message) {
    return llvm::createStringError(std::make_error_code(code),
                                   "PnR work ledger: %s", message);
  }

  std::array<PnrWorkCounterRef, pnrWorkKindCount> counters_{};
};

} // namespace loom::pnr

#endif // LOOM_PNR_PNRWORKLEDGER_H
