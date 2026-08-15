#ifndef LOOM_LIB_PNR_SYSTEM_SYSTEMNEGOTIATEDROUTER_H
#define LOOM_LIB_PNR_SYSTEM_SYSTEMNEGOTIATEDROUTER_H

#include "SystemServiceRouter.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <system_error>

namespace loom::pnr::detail {

enum class SystemRoutingClosureFailureKind : std::uint8_t {
  FixedTerminalCapacityCut,
  NonClosure,
  NoProgress,
};

enum class SystemRoutingClosureRequirement : std::uint8_t {
  PolicyAdmittedTemporary,
  Strict,
};

/// Exact fixed-terminal routing evidence that can reopen occurrence-level
/// graph binding choices. Service legs remain the router-owned subject; a
/// caller maps them to its own mutable decision domain.
struct SystemRoutingReopenWitness final {
  PnrIndex capacityCell = getInvalidPnrIndex();
  std::vector<PnrIndex> serviceLegs;
};

class SystemRoutingClosureFailure final
    : public llvm::ErrorInfo<SystemRoutingClosureFailure> {
public:
  static char ID;

  SystemRoutingClosureFailure(
      SystemRoutingClosureFailureKind kind, std::string message,
      std::optional<SystemRoutingReopenWitness> reopenWitness = std::nullopt)
      : kind_(kind), message_(std::move(message)),
        reopenWitness_(std::move(reopenWitness)) {}

  SystemRoutingClosureFailureKind kind() const { return kind_; }
  const std::optional<SystemRoutingReopenWitness> &reopenWitness() const {
    return reopenWitness_;
  }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  SystemRoutingClosureFailureKind kind_;
  std::string message_;
  std::optional<SystemRoutingReopenWitness> reopenWitness_;
};

llvm::Expected<CanonicalSystemServiceRoutes> negotiateSystemServiceRoutes(
    const FrozenSystemPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> threadChoices,
    llvm::ArrayRef<PnrIndex> graphChoices,
    llvm::ArrayRef<SystemInstructionResourceUseSelection>
        instructionResourceUses,
    llvm::ArrayRef<SystemServiceResourceUseSelection> serviceResourceUses,
    std::uint64_t &endpointExpansions, std::uint64_t &negotiationIterations,
    llvm::ArrayRef<PnrIndex> reroutedLegs = {},
    std::optional<SystemServiceRoutesView> priorRoutes = std::nullopt,
    std::optional<SystemServiceRouteTraversalExclusion> exclusion =
        std::nullopt,
    std::optional<SystemServiceRouteRepairRegion> repairRegion = std::nullopt,
    SystemRoutingClosureRequirement closureRequirement =
        SystemRoutingClosureRequirement::PolicyAdmittedTemporary,
    std::optional<SystemRoutingReopenWitness> *reopenWitness = nullptr);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SYSTEM_SYSTEMNEGOTIATEDROUTER_H
