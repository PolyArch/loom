#ifndef LOOM_PNR_SPATIALPNRWORKLEDGER_H
#define LOOM_PNR_SPATIALPNRWORKLEDGER_H

#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>

namespace loom::pnr {

enum class SpatialPnrWorkKind : std::uint8_t {
  SeedAttempt,
  InitializerAssignment,
  EndpointExpansion,
  NegotiationIteration,
  CalibrationProposal,
  AnnealingBaseProposal,
  AnnealingMovableProposal,
  ExactRepairRegionDecision,
  ExactRepairSolverCall,
  FinalClosureAttempt,
  Count,
};

inline constexpr std::size_t spatialPnrWorkKindCount =
    static_cast<std::size_t>(SpatialPnrWorkKind::Count);

struct SpatialPnrWorkCounterRef final {
  std::uint64_t *planned = nullptr;
  std::uint64_t *consumed = nullptr;
};

/// Non-owning access to the invocation's canonical work counters. A semantic
/// owner reserves a logical slot immediately before executing it and consumes
/// that slot only after the owner boundary has run. Unbound views preserve the
/// same algorithms for callers that do not publish provider accounting.
class SpatialPnrWorkLedgerView final {
public:
  constexpr SpatialPnrWorkLedgerView() = default;
  explicit constexpr SpatialPnrWorkLedgerView(
      std::array<SpatialPnrWorkCounterRef, spatialPnrWorkKindCount> counters)
      : counters_(counters) {}

  llvm::Error plan(SpatialPnrWorkKind kind, std::uint64_t amount = 1) const {
    const SpatialPnrWorkCounterRef counter = counters_[ordinal(kind)];
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

  llvm::Error consume(SpatialPnrWorkKind kind, std::uint64_t amount = 1) const {
    const SpatialPnrWorkCounterRef counter = counters_[ordinal(kind)];
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
  static constexpr std::size_t ordinal(SpatialPnrWorkKind kind) {
    return static_cast<std::size_t>(kind);
  }

  static llvm::Error ledgerError(std::errc code, const char *message) {
    return llvm::createStringError(std::make_error_code(code),
                                   "Spatial PnR work ledger: %s", message);
  }

  std::array<SpatialPnrWorkCounterRef, spatialPnrWorkKindCount> counters_{};
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALPNRWORKLEDGER_H
