#ifndef LOOM_LIB_SIMULATOR_CGRAPHYSICALACTIONRUNTIME_H
#define LOOM_LIB_SIMULATOR_CGRAPHYSICALACTIONRUNTIME_H

#include "CGRAResourceRuntime.h"

#include "Simulator/CGRA/EventQueue.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::sim::detail {

/// Exact owner-relative timing of one selected Mapping ResourceUse. Ranks are
/// measured in the selected owner's local reference cycles. Event ordinals
/// provide canonical within-coordinate order only; they do not arbitrate.
struct CgraPhysicalUseTiming final {
  std::uint64_t selectedUseOrdinal = 0;
  std::uint32_t acquireRank = 0;
  std::optional<std::uint32_t> commitRank;
  std::uint32_t releaseRank = 0;
  std::uint32_t acquireEventOrdinal = 0;
  std::uint32_t releaseEventOrdinal = 0;
  std::optional<std::uint32_t> commitEventOrdinal;
  bool requiresCausalRelease = false;
};

enum class CgraPhysicalLifecycleKind : std::uint8_t {
  Requested,
  Granted,
  Committed,
  Retired,
};

struct CgraPhysicalLifecycleEvent final {
  CgraPhysicalLifecycleKind kind = CgraPhysicalLifecycleKind::Requested;
  std::uint64_t actionOrdinal = 0;
  std::uint64_t occurrenceOrdinal = 0;
  std::uint32_t ownerEventOrdinal = 0;
  SpatialEventCoordinate coordinate;
};

struct CgraPhysicalLifecycleFrame final {
  SpatialEventCoordinate coordinate;
  std::vector<CgraPhysicalLifecycleEvent> events;
};

struct CgraPhysicalActionRequest final {
  std::uint64_t actionOrdinal = 0;
  std::uint64_t occurrenceOrdinal = 0;
};

/// Execution-local lifecycle of selected physical ResourceUses. Resource
/// capacity and arbitration remain in CgraResourceRuntime; this layer only
/// applies owner-relative timing, retries blocked requests at reference-clock
/// boundaries, and closes the exact claim envelope on release.
class CgraPhysicalActionRuntime final {
public:
  enum class InternalKind : std::uint8_t {
    Commit,
    Release,
    CommitRelease,
    Acquire,
  };

  static llvm::Expected<CgraPhysicalActionRuntime>
  create(const CgraResourceRuntimePlan &resources,
         llvm::ArrayRef<CgraPhysicalUseTiming> uses);

  llvm::Expected<CgraPhysicalLifecycleEvent>
  request(std::uint64_t actionOrdinal, std::uint64_t occurrenceOrdinal,
          SpatialEventCoordinate coordinate);

  llvm::Expected<std::vector<CgraPhysicalLifecycleEvent>>
  requestBatch(llvm::ArrayRef<CgraPhysicalActionRequest> requests,
               SpatialEventCoordinate coordinate);

  llvm::Error satisfyCausalRelease(std::uint64_t actionOrdinal,
                                   std::uint64_t occurrenceOrdinal,
                                   SpatialEventCoordinate coordinate);

  /// Advances through one exact coordinate. A frame can contain no visible
  /// event when every acquisition attempt at that coordinate remains blocked.
  llvm::Expected<std::optional<CgraPhysicalLifecycleFrame>> advance();

  std::optional<SpatialEventCoordinate> nextCoordinate() const {
    return events_.nextCoordinate();
  }

  bool hasPendingActions() const { return activeActionCount_ != 0; }
  std::uint64_t pendingActionCount() const { return activeActionCount_; }

private:
  enum class ActionState : std::uint8_t { Requested, Granted, Retired };

  struct Action final {
    std::uint64_t actionOrdinal = 0;
    std::uint64_t occurrenceOrdinal = 0;
    ActionState state = ActionState::Requested;
    std::optional<CgraClaimEnvelope> envelope;
    bool intrinsicReleaseReached = false;
    bool causalReleaseReached = false;
  };

  CgraPhysicalActionRuntime(std::vector<CgraPhysicalUseTiming> uses,
                            CgraResourceRuntime resources)
      : uses_(std::move(uses)), resources_(std::move(resources)) {}

  llvm::Error schedule(std::uint64_t actionSlot, InternalKind kind,
                       SpatialEventCoordinate coordinate,
                       std::uint32_t ownerEventOrdinal);

  std::vector<CgraPhysicalUseTiming> uses_;
  CgraResourceRuntime resources_;
  CgraEventQueue events_{"CGRA physical action"};
  std::vector<Action> actions_;
  std::vector<std::uint64_t> freeActionSlots_;
  llvm::DenseMap<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t>
      activeActions_;
  std::uint64_t activeActionCount_ = 0;
  std::optional<SpatialEventCoordinate> lastCoordinate_;
};

} // namespace loom::sim::detail

#endif // LOOM_LIB_SIMULATOR_CGRAPHYSICALACTIONRUNTIME_H
