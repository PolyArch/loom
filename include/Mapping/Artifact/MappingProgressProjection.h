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
  std::vector<SystemPresburgerCell> relationDomain;
  std::vector<::dataflow::EventFamilyKey> triggerAlternatives;
  std::vector<MappingProgressCapacityClaimProjection> capacityClaims;
  std::vector<MappingProgressCausalReleaseProjection> causalRelease;
  MappingResourceProgressUse arbitration;
};

/// One selected route dependency whose dependent branch must become durable
/// after it diverges from its causal prerequisite branch. This value is
/// rebuilt from the selected route tree and typed Fabric traversals or sink
/// boundaries; it is not a persisted proof label.
struct MappingRouteProgressObligationProjection final {
  bool durableBoundaryAfterDivergence = false;
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
