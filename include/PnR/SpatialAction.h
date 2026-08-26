#ifndef LOOM_PNR_SPATIALACTION_H
#define LOOM_PNR_SPATIALACTION_H

#include "PnR/DeterministicSearchProtocol.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <optional>
#include <variant>

namespace loom::pnr {

struct SpatialComputeBindingAction final {
  PnrIndex realization;
  PnrIndex placement;
  PnrIndex instructionContext;
};

struct SpatialMemoryBindingAction final {
  PnrIndex realization;
  PnrIndex placement;
};

using SpatialRealizationBindingAction =
    std::variant<SpatialComputeBindingAction, SpatialMemoryBindingAction>;

enum class SpatialWholeNetDispositionKind : std::uint8_t {
  Preferred,
  External,
  RegisterFifo,
};

struct SpatialWholeNetRoutingAction final {
  PnrIndex logicalNet;
  SpatialWholeNetDispositionKind disposition =
      SpatialWholeNetDispositionKind::Preferred;
  PnrIndex registerFifoTransfer = getInvalidPnrIndex();
};

struct SpatialSingleSinkRoutingAction final {
  PnrIndex logicalNet;
  PnrIndex sinkObligation;
};

struct SpatialRootedSubtreeRoutingAction final {
  PnrIndex logicalNet;
  PnrIndex rootEndpoint;
};

struct SpatialWitnessRegionRoutingAction final {
  ResolvedPnrViolationKind witnessKind;
  PnrIndex witnessOrdinal;
};

struct SpatialGlobalRoutingAction final {};

using SpatialTransportRoutingAction =
    std::variant<SpatialWholeNetRoutingAction, SpatialSingleSinkRoutingAction,
                 SpatialRootedSubtreeRoutingAction,
                 SpatialWitnessRegionRoutingAction, SpatialGlobalRoutingAction>;

struct SpatialPortAttachmentAction final {
  PnrIndex demand;
  PnrIndex attachmentOption;
};

struct SpatialGraphBoundaryAttachmentAction final {
  PnrIndex boundary;
  PnrIndex attachmentOption;
};

struct SpatialMemoryOperationPlanAction final {
  PnrIndex actor;
  PnrIndex plan;
};

struct SpatialLogicalMemoryBindingAction final {
  PnrIndex binding;
  PnrIndex target;
  std::uint64_t physicalOffsetBytes;
};

struct SpatialMemoryUseDispatchAction final {
  PnrIndex use;
  PnrIndex dispatchOption;
};

struct SpatialMemoryExposureAction final {
  PnrIndex exposure;
  PnrIndex exposureOption;
};

using SpatialResourceAllocationAction = std::variant<
    SpatialPortAttachmentAction, SpatialGraphBoundaryAttachmentAction,
    SpatialMemoryOperationPlanAction, SpatialLogicalMemoryBindingAction,
    SpatialMemoryUseDispatchAction, SpatialMemoryExposureAction>;

using SpatialMappingAction =
    std::variant<SpatialRealizationBindingAction, SpatialTransportRoutingAction,
                 SpatialResourceAllocationAction>;

/// Dense canonical key for one closed Spatial Action. It is an ephemeral
/// search-cache key, not an Artifact identity or serialization.
struct SpatialActionKey final {
  std::array<std::uint64_t, 6> fields{};

  friend bool operator==(const SpatialActionKey &lhs,
                         const SpatialActionKey &rhs) {
    return lhs.fields == rhs.fields;
  }
  friend bool operator<(const SpatialActionKey &lhs,
                        const SpatialActionKey &rhs) {
    return lhs.fields < rhs.fields;
  }
};

SpatialActionKey spatialActionKey(const SpatialMappingAction &action);

class SpatialCandidateState;

/// True when the action re-selects exactly the candidate's current decision.
/// Such an action is a proven semantic no-op: every decision application
/// returns before marking a net when the selection is unchanged, so nothing it
/// touches can differ from the current candidate. Routing actions are never an
/// identity because their result comes from the router, not the action.
bool isIdentitySpatialAction(const SpatialCandidateState &candidate,
                             const SpatialMappingAction &action);

/// Validates one ephemeral ActionBatch in canonical typed-anchor order. A
/// batch is nonempty and may change each selected-decision anchor at most once.
llvm::Error validateCanonicalSpatialActionBatch(
    llvm::ArrayRef<SpatialMappingAction> actions);

struct SpatialActionChoiceRange final {
  PnrIndex choiceOffset;
  PnrIndex choiceCount;
};

/// Removable view of one exact candidate's canonical dynamic Action domain.
/// Choice ranges are contiguous, nonempty, and ordered by typed anchor.
struct SpatialActionProposalDomain final {
  llvm::ArrayRef<SpatialActionChoiceRange> realizationAnchors;
  llvm::ArrayRef<SpatialRealizationBindingAction> realizationChoices;
  llvm::ArrayRef<SpatialActionChoiceRange> transportAnchors;
  llvm::ArrayRef<SpatialTransportRoutingAction> transportChoices;
  llvm::ArrayRef<SpatialActionChoiceRange> resourceAnchors;
  llvm::ArrayRef<SpatialResourceAllocationAction> resourceChoices;
};

/// Selects kind, anchor, and choice with exactly three canonical bounded draws.
/// An empty selectable domain returns no Action without consuming entropy.
llvm::Expected<std::optional<SpatialMappingAction>>
proposeSpatialAction(const ResolvedPnrActionProposalPolicy &policy,
                     SpatialActionProposalDomain domain,
                     DeterministicPnrRandomStream &proposalStream);

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALACTION_H
