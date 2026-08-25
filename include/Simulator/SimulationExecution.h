#ifndef LOOM_SIMULATOR_SIMULATIONEXECUTION_H
#define LOOM_SIMULATOR_SIMULATIONEXECUTION_H

#include "Simulator/SimulationArtifacts.h"

#include "Common/Artifact.h"
#include "Evaluation/Case.h"
#include "Evaluation/Finding.h"
#include "Evaluation/ModelDescriptor.h"
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
class BlobStore;
} // namespace loom

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

struct TerminalWitnessRef {
  evaluation::ModelOutputSlotRef executionOutputSlot;
  std::uint64_t executionOutputOrdinal = 0;
};

/// Simulation Artifacts owns the Evidence occurrence codec for terminal
/// witnesses. Finding owners select this codec but never duplicate it.
const evaluation::FindingOccurrenceCodec &terminalWitnessRefOccurrenceCodec();

struct SpatialEventCoordinate {
  evaluation::ExactRatio referenceCycle;
  std::uint64_t delta = 0;
};

/// Exact numeric and delta order: negative, zero, or positive. Equal
/// denominators compare their canonical numerators directly; other products
/// use unsigned 128-bit arithmetic, which exactly contains u64*u64.
int compareSpatialEventCoordinates(const SpatialEventCoordinate &lhs,
                                   const SpatialEventCoordinate &rhs);

/// Returns the exact integral reference-cycle distance when `to` does not
/// precede `from`; nonintegral, reversed, or overflowing intervals have no
/// integral projection.
std::optional<std::uint64_t>
integralSpatialReferenceCycleDistance(const SpatialEventCoordinate &from,
                                      const SpatialEventCoordinate &to);

struct SpatialProgressObservations {
  SpatialEventCoordinate launchAccepted;
  std::optional<SpatialEventCoordinate> graphRetirementVisible;
  SpatialEventCoordinate terminalObserved;
};

struct SystemFunctionalObservations {
  std::vector<ValueResultObservation> valueResults;
  std::vector<ValueResultObservation> externalValueOutputs;
  std::vector<CanonicalStreamSequence> externalStreamOutputs;
  std::vector<MemoryObservationPayload> memories;
};

struct SystemEventCoordinate {
  std::uint64_t gem5Tick = 0;
  std::uint64_t delta = 0;
};

int compareSystemEventCoordinates(const SystemEventCoordinate &lhs,
                                  const SystemEventCoordinate &rhs);

struct SystemProgressObservations {
  SystemEventCoordinate programEntryAccepted;
  std::optional<SystemEventCoordinate> programExitVisible;
  SystemEventCoordinate terminalObserved;
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

/// The Spatial observation form selected by a Spatial workload. It carries no
/// root discriminator: the exact Request's workload selects this form.
struct SpatialSimulationExecution {
  ArtifactRootReference request;
  ExecutionTerminal terminal;
  SpatialFunctionalObservations functionalObservations;
  SpatialProgressObservations progressObservations;
  std::vector<ActorTransitionsActivitySummary> activitySummaries;
};

/// Invocation-local result at a Spatial engine boundary. The exact workload
/// and runtime input supply its observation shape; this value has no Artifact
/// identity and carries neither a Request nor an engine selector. It is the
/// shared transport used by standalone adapters and the gem5 bridge.
struct SpatialEngineBoundaryResult {
  ExecutionTerminal terminal;
  SpatialFunctionalObservations functionalObservations;
  SpatialProgressObservations progressObservations;
  std::vector<ActorTransitionsActivitySummary> activitySummaries;
};

/// The Deployment-owned observation form selected by a System workload. The
/// root wire remains untagged; request -> workload selects this form.
struct SystemSimulationExecution {
  ArtifactRootReference request;
  ExecutionTerminal terminal;
  SystemFunctionalObservations functionalObservations;
  SystemProgressObservations progressObservations;
  // Schema 1.0 activity payloads use Spatial reference-cycle windows, so this
  // collection is required to be empty for System executions.
  std::vector<ActorTransitionsActivitySummary> activitySummaries;
};

using SimulationExecutionModel =
    std::variant<SpatialSimulationExecution, SystemSimulationExecution>;

class CanonicalSimulationExecution {
public:
  CanonicalSimulationExecution(const CanonicalSimulationExecution &) = delete;
  CanonicalSimulationExecution(CanonicalSimulationExecution &&) = default;
  CanonicalSimulationExecution &
  operator=(const CanonicalSimulationExecution &) = delete;
  CanonicalSimulationExecution &
  operator=(CanonicalSimulationExecution &&) = default;

  const ArtifactIdentity &identity() const { return identity_; }
  const ArtifactRootReference &request() const {
    return std::visit(
        [](const auto &root) -> const ArtifactRootReference & {
          return root.request;
        },
        model_);
  }
  const ExecutionTerminal &terminal() const {
    return std::visit(
        [](const auto &root) -> const ExecutionTerminal & {
          return root.terminal;
        },
        model_);
  }
  const SimulationExecutionModel &root() const { return model_; }
  const SpatialSimulationExecution *spatial() const {
    return std::get_if<SpatialSimulationExecution>(&model_);
  }
  const SystemSimulationExecution *system() const {
    return std::get_if<SystemSimulationExecution>(&model_);
  }
  const SpatialFunctionalObservations &spatialFunctionalObservations() const {
    return std::get<SpatialSimulationExecution>(model_).functionalObservations;
  }
  const SpatialProgressObservations &spatialProgressObservations() const {
    return std::get<SpatialSimulationExecution>(model_).progressObservations;
  }
  llvm::ArrayRef<ActorTransitionsActivitySummary>
  spatialActivitySummaries() const {
    return std::get<SpatialSimulationExecution>(model_).activitySummaries;
  }
  const CanonicalSemanticBytes &canonicalBytes() const { return bytes_; }

private:
  CanonicalSimulationExecution(ArtifactIdentity identity,
                               SpatialSimulationExecution model,
                               CanonicalSemanticBytes bytes)
      : identity_(identity), model_(std::move(model)),
        bytes_(std::move(bytes)) {}
  CanonicalSimulationExecution(ArtifactIdentity identity,
                               SystemSimulationExecution model,
                               CanonicalSemanticBytes bytes)
      : identity_(identity), model_(std::move(model)),
        bytes_(std::move(bytes)) {}

  ArtifactIdentity identity_;
  SimulationExecutionModel model_;
  CanonicalSemanticBytes bytes_;

  friend llvm::Expected<CanonicalSimulationExecution>
  finalizeSimulationExecution(const SpatialSimulationExecution &,
                              const evaluation::CaseArtifactResolution &,
                              const ArtifactStore &, const BlobStore &);
  friend llvm::Expected<CanonicalSimulationExecution>
  finalizeSimulationExecution(const SystemSimulationExecution &,
                              const evaluation::CaseArtifactResolution &,
                              const ArtifactStore &, const BlobStore &);
  friend llvm::Expected<CanonicalSimulationExecution>
  importSimulationExecution(const ArtifactRootReference &,
                            const evaluation::CaseArtifactResolution &,
                            const ArtifactStore &, const BlobStore &);
};

llvm::Expected<CanonicalSimulationExecution> finalizeSimulationExecution(
    const SpatialSimulationExecution &execution,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &store, const BlobStore &blobs);

llvm::Expected<CanonicalSimulationExecution> finalizeSimulationExecution(
    const SystemSimulationExecution &execution,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &store, const BlobStore &blobs);

llvm::Expected<ArtifactRootReference>
publishSimulationExecution(const CanonicalSimulationExecution &execution,
                           const ArtifactStore &store);

/// Imports the workload-selected Spatial or System execution form. Both the
/// Request and Deployment closure are validated through their exact stores.
llvm::Expected<CanonicalSimulationExecution>
importSimulationExecution(const ArtifactRootReference &reference,
                          const evaluation::CaseArtifactResolution &resolution,
                          const ArtifactStore &store, const BlobStore &blobs);

/// Resolves the exact Request root carried by one stored execution. This is
/// the owner-level dependency projection used to assemble an import closure;
/// full execution validation still occurs through importSimulationExecution.
llvm::Expected<ArtifactRootReference>
simulationExecutionRequestReference(const ArtifactRootReference &reference,
                                    const ArtifactStore &store);

llvm::Expected<std::vector<std::uint8_t>>
encodeSpatialEngineBoundaryResult(const SpatialEngineBoundaryResult &result,
                                  const ArtifactRootReference &workload,
                                  const ArtifactRootReference &runtimeInput,
                                  const ArtifactStore &store);

llvm::Expected<std::vector<std::uint8_t>> encodeSpatialEngineBoundaryResult(
    const SpatialEngineBoundaryResult &result,
    const ImportedSpatialSimulationInputs &inputs);

llvm::Expected<std::vector<std::uint8_t>> encodeSpatialEngineBoundaryResult(
    const SpatialEngineBoundaryResult &result,
    const ImportedSpatialSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput);

llvm::Expected<SpatialEngineBoundaryResult> decodeSpatialEngineBoundaryResult(
    llvm::ArrayRef<std::uint8_t> bytes, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ArtifactStore &store);

llvm::Expected<SpatialEngineBoundaryResult> decodeSpatialEngineBoundaryResult(
    llvm::ArrayRef<std::uint8_t> bytes,
    const ImportedSpatialSimulationInputs &inputs);

llvm::Expected<SpatialEngineBoundaryResult> decodeSpatialEngineBoundaryResult(
    llvm::ArrayRef<std::uint8_t> bytes,
    const ImportedSpatialSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SIMULATIONEXECUTION_H
