#ifndef LOOM_PNR_SYSTEM_SYSTEMACTION_H
#define LOOM_PNR_SYSTEM_SYSTEMACTION_H

#include "Common/ResolvedPnrPolicy.h"
#include "PnR/DeterministicSearchProtocol.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace loom::pnr {

struct SystemExecutionBindingAction final {
  PnrIndex decision = 0;
  PnrIndex choice = 0;
};

struct SystemWholeLegRoutingAction final {
  PnrIndex leg = 0;
};

struct SystemSingleSinkRoutingAction final {
  PnrIndex leg = 0;
  PnrIndex sinkObligation = 0;
};

struct SystemRootedSubtreeRoutingAction final {
  PnrIndex leg = 0;
  PnrIndex rootEndpoint = 0;
};

struct SystemWitnessRegionRoutingAction final {
  ResolvedPnrViolationKind witnessKind =
      ResolvedPnrViolationKind::UnroutedObligation;
  PnrIndex witnessOrdinal = 0;
};

struct SystemGlobalRoutingAction final {};

using SystemTransportRoutingAction =
    std::variant<SystemWholeLegRoutingAction, SystemSingleSinkRoutingAction,
                 SystemRootedSubtreeRoutingAction,
                 SystemWitnessRegionRoutingAction, SystemGlobalRoutingAction>;

struct SystemServiceTargetAction final {
  PnrIndex context = 0;
  PnrIndex choice = 0;
};

struct SystemInstructionUsePatternAction final {
  PnrIndex use = 0;
  PnrIndex choice = 0;
};

struct SystemServiceUsePatternAction final {
  PnrIndex use = 0;
  PnrIndex choice = 0;
};

using SystemResourceAllocationAction =
    std::variant<SystemServiceTargetAction, SystemInstructionUsePatternAction,
                 SystemServiceUsePatternAction>;

using SystemMappingAction =
    std::variant<SystemExecutionBindingAction, SystemTransportRoutingAction,
                 SystemResourceAllocationAction>;

struct SystemActionChoiceRange final {
  PnrIndex choiceOffset = 0;
  PnrIndex choiceCount = 0;
};

struct SystemActionProposalDomain final {
  llvm::ArrayRef<SystemActionChoiceRange> bindingAnchors;
  llvm::ArrayRef<SystemExecutionBindingAction> bindingChoices;
  llvm::ArrayRef<SystemActionChoiceRange> routingAnchors;
  llvm::ArrayRef<SystemTransportRoutingAction> routingChoices;
  llvm::ArrayRef<SystemActionChoiceRange> resourceAnchors;
  llvm::ArrayRef<SystemResourceAllocationAction> resourceChoices;
};

llvm::Expected<std::optional<SystemMappingAction>>
proposeSystemAction(const ResolvedPnrActionProposalPolicy &policy,
                    SystemActionProposalDomain domain,
                    DeterministicPnrRandomStream &stream);

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMACTION_H
