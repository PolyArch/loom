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
  NonClosure,
};

enum class SystemRoutingClosureRequirement : std::uint8_t {
  PolicyAdmittedTemporary,
  Strict,
};

class SystemRoutingClosureFailure final
    : public llvm::ErrorInfo<SystemRoutingClosureFailure> {
public:
  static char ID;

  SystemRoutingClosureFailure(SystemRoutingClosureFailureKind kind,
                              std::string message)
      : kind_(kind), message_(std::move(message)) {}

  SystemRoutingClosureFailureKind kind() const { return kind_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  SystemRoutingClosureFailureKind kind_;
  std::string message_;
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
        SystemRoutingClosureRequirement::PolicyAdmittedTemporary);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SYSTEM_SYSTEMNEGOTIATEDROUTER_H
