#ifndef LOOM_SIMULATOR_SIMULATIONEXECUTION_H
#define LOOM_SIMULATOR_SIMULATIONEXECUTION_H

#include "Simulator/SimulationArtifacts.h"

#include "Common/Artifact.h"
#include "Evaluation/Case.h"
#include "Evaluation/Finding.h"
#include "Evaluation/NumericValue.h"
#include "Evaluation/OwnerValue.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::sim {

inline constexpr ArtifactSchemaDescriptor simulationExecutionSchema{
    "loom.simulation_execution", SchemaVersion{1, 0}};

struct RetiredExecution {};

struct HaltedExecution {
  evaluation::FindingKind findingKind;
  evaluation::OwnerValue witness;
};

struct StoppedByLimitExecution {};

using ExecutionTerminal =
    std::variant<RetiredExecution, HaltedExecution, StoppedByLimitExecution>;

struct SpatialEventCoordinate {
  evaluation::ExactRatio referenceCycle;
  std::uint64_t delta = 0;
};

/// Exact numeric and delta order: negative, zero, or positive. Equal
/// denominators compare their canonical numerators directly; other products
/// use unsigned 128-bit arithmetic, which exactly contains u64*u64.
int compareSpatialEventCoordinates(const SpatialEventCoordinate &lhs,
                                   const SpatialEventCoordinate &rhs);

struct SpatialProgressObservations {
  SpatialEventCoordinate launchAccepted;
  std::optional<SpatialEventCoordinate> graphRetirementVisible;
  SpatialEventCoordinate terminalObserved;
};

enum class ActivityWindow : std::uint32_t {
  LaunchToGraphRetirement = 0,
  LaunchToTerminal = 1,
};

enum class ActivityCoverage : std::uint32_t { Complete = 0, Partial = 1 };

struct ActorTransitionCounts {
  std::uint64_t committedFirings = 0;
  std::uint64_t retiredFirings = 0;
};

struct ActorTransitionEntry {
  dataflow::ActorRef actor;
  ActorTransitionCounts counts;
};

struct ActorTransitionsActivitySummary {
  ActivityWindow window = ActivityWindow::LaunchToTerminal;
  ActivityCoverage coverage = ActivityCoverage::Partial;
  std::vector<ActorTransitionEntry> transitions;
};

/// The currently owner-complete Spatial execution root. It carries no Spatial
/// discriminator: the exact Request's workload selects this form. System
/// authoring remains unavailable until the Deployment-owned workload root is
/// implemented.
struct SpatialSimulationExecution {
  ArtifactRootReference request;
  ExecutionTerminal terminal;
  SpatialFunctionalObservations functionalObservations;
  SpatialProgressObservations progressObservations;
  std::vector<ActorTransitionsActivitySummary> activitySummaries;
};

class CanonicalSimulationExecution {
public:
  CanonicalSimulationExecution(const CanonicalSimulationExecution &) = delete;
  CanonicalSimulationExecution(CanonicalSimulationExecution &&) = default;
  CanonicalSimulationExecution &
  operator=(const CanonicalSimulationExecution &) = delete;
  CanonicalSimulationExecution &
  operator=(CanonicalSimulationExecution &&) = default;

  const ArtifactIdentity &identity() const { return identity_; }
  const ArtifactRootReference &request() const { return model_.request; }
  const ExecutionTerminal &terminal() const { return model_.terminal; }
  const SpatialFunctionalObservations &functionalObservations() const {
    return model_.functionalObservations;
  }
  const SpatialProgressObservations &progressObservations() const {
    return model_.progressObservations;
  }
  llvm::ArrayRef<ActorTransitionsActivitySummary> activitySummaries() const {
    return model_.activitySummaries;
  }
  const CanonicalSemanticBytes &canonicalBytes() const { return bytes_; }

private:
  CanonicalSimulationExecution(ArtifactIdentity identity,
                               SpatialSimulationExecution model,
                               CanonicalSemanticBytes bytes)
      : identity_(identity), model_(std::move(model)),
        bytes_(std::move(bytes)) {}

  ArtifactIdentity identity_;
  SpatialSimulationExecution model_;
  CanonicalSemanticBytes bytes_;

  friend llvm::Expected<CanonicalSimulationExecution>
  finalizeSimulationExecution(const SpatialSimulationExecution &,
                              const evaluation::CaseArtifactResolution &,
                              const ArtifactStore &);
  friend llvm::Expected<CanonicalSimulationExecution>
  importSimulationExecution(const ArtifactRootReference &,
                            const evaluation::CaseArtifactResolution &,
                            const ArtifactStore &);
};

llvm::Expected<CanonicalSimulationExecution> finalizeSimulationExecution(
    const SpatialSimulationExecution &execution,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &store);

llvm::Expected<ArtifactRootReference>
publishSimulationExecution(const CanonicalSimulationExecution &execution,
                           const ArtifactStore &store);

llvm::Expected<CanonicalSimulationExecution>
importSimulationExecution(const ArtifactRootReference &reference,
                          const evaluation::CaseArtifactResolution &resolution,
                          const ArtifactStore &store);

/// Resolves the exact Request root carried by one stored execution. This is
/// the owner-level dependency projection used to assemble an import closure;
/// full execution validation still occurs through importSimulationExecution.
llvm::Expected<ArtifactRootReference>
simulationExecutionRequestReference(const ArtifactRootReference &reference,
                                    const ArtifactStore &store);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SIMULATIONEXECUTION_H
