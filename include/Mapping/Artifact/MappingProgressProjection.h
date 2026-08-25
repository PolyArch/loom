#ifndef LOOM_MAPPING_ARTIFACT_MAPPINGPROGRESSPROJECTION_H
#define LOOM_MAPPING_ARTIFACT_MAPPINGPROGRESSPROJECTION_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"
#include "Mapping/Artifact/SystemPresburger.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom::mapping {

enum class MappingProgressClosureKind : std::uint8_t {
  ProvenNoClosedWaitSet,
  ProvenClosedWaitSet,
  ProofNotEstablished,
};

/// Exact reason selected by the shared progress kernel. This is derived
/// diagnostic state, not a persisted proof label or another legality owner.
enum class MappingProgressClosureReason : std::uint8_t {
  None,
  CyclicDataflowBasis,
  MissingDurableBoundary,
  ActivationCapacityExceeded,
  FixedPriorityStarvation,
  PossibleWaitCycle,
  FiniteBufferRecurrenceNotEstablished,
};

enum class MappingProgressWaitNodeKind : std::uint8_t {
  Active,
  Pending,
};

/// One node in a deterministic possible wait-cycle witness. Activation and
/// capacity-cell ordinals address the exact ephemeral projection consumed by
/// the shared kernel; they are diagnostics, not persistent artifact keys.
struct MappingProgressWaitCycleNode final {
  std::uint64_t activationGroupOrdinal = 0;
  MappingProgressWaitNodeKind kind = MappingProgressWaitNodeKind::Active;
  std::vector<std::uint64_t> activationOrdinals;
  std::vector<std::uint64_t> capacityCellOrdinals;
  std::vector<std::uint32_t> triggerEventOrdinals;
  std::vector<std::uint32_t> causalReleaseEventOrdinals;
};

struct MappingProgressClosure final {
  MappingProgressClosureKind kind =
      MappingProgressClosureKind::ProofNotEstablished;
  MappingProgressClosureReason reason =
      MappingProgressClosureReason::CyclicDataflowBasis;
  std::vector<MappingProgressWaitCycleNode> possibleWaitCycle;
};

enum class MappingDataflowProgressBasisKind : std::uint8_t {
  Acyclic,
  InitializedFeedback,
  Cyclic,
};

struct MappingDataflowProgressBasis final {
  MappingDataflowProgressBasisKind kind =
      MappingDataflowProgressBasisKind::Cyclic;
  std::uint64_t coveredActorCount = 0;
  std::uint64_t initializedFeedbackEdgeCount = 0;
  std::vector<::dataflow::ActorRef> residualCycle;
};

enum class MappingResourceGrantPolicyKind : std::uint8_t {
  None,
  FixedPriority,
  RoundRobin,
};

struct MappingResourceProgressUse final {
  std::string physicalOwnerKey;
  std::uint32_t requester = 0;
  MappingResourceGrantPolicyKind grantPolicy =
      MappingResourceGrantPolicyKind::None;
};

struct MappingProgressCapacityCellProjection final {
  std::uint64_t capacity = 0;
  std::uint64_t baselineOccupancy = 0;
};

struct MappingProgressCapacityClaimProjection final {
  std::uint64_t capacityCellOrdinal = 0;
  std::uint64_t amount = 0;
};

struct MappingProgressCausalReleaseProjection final {
  std::vector<::dataflow::EventFamilyKey> alternatives;
};

struct MappingProgressActivationProjection final {
  ExecutionContextKey context;
  ::dataflow::RootThreadLaunchRef relationRoot;
  std::vector<SystemPresburgerCell> relationDomain;
  std::vector<::dataflow::EventFamilyKey> triggerAlternatives;
  std::vector<MappingProgressCapacityClaimProjection> capacityClaims;
  std::vector<MappingProgressCausalReleaseProjection> causalRelease;
  MappingResourceProgressUse arbitration;
};

enum class MappingRouteProgressObligationKind : std::uint8_t {
  DurableBoundaryAfterDivergence,
  FiniteBufferRecurrence,
};

/// One physical progress obligation rebuilt from the selected route trees and
/// typed Fabric traversals or sink boundaries. An unestablished durable
/// boundary is a concrete closed-wait witness. An unestablished finite-buffer
/// recurrence remains incomplete because a finite replay cannot prove general
/// queue liveness. This value is not a persisted proof label.
struct MappingRouteProgressObligationProjection final {
  MappingRouteProgressObligationKind kind =
      MappingRouteProgressObligationKind::DurableBoundaryAfterDivergence;
  bool established = false;
};

/// The complete removable input to the shared Mapping progress kernel.
/// Spatial and System roots differ only in how they rebuild these fields.
struct MappingProgressProjection final {
  MappingDataflowProgressBasis basis;
  std::vector<MappingRouteProgressObligationProjection> routeObligations;
  std::vector<MappingProgressCapacityCellProjection> capacityCells;
  std::vector<MappingProgressActivationProjection> resourceActivations;
};

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_MAPPINGPROGRESSPROJECTION_H
