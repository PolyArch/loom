#ifndef LOOM_APPLICATION_RESOURCETIMEEXECUTION_H
#define LOOM_APPLICATION_RESOURCETIMEEXECUTION_H

#include "Common/Artifact.h"
#include "Runtime/DeploymentLoader.h"
#include "Runtime/ResourceTimeTransitionSelection.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::application {

class FinalizedApplicationRuntimeManifest;

inline constexpr ArtifactSchemaDescriptor
    applicationResourceTimeExecutionTraceSchema{
        "loom.application.resource_time_execution_trace", SchemaVersion{1, 0}};

enum class ApplicationResourceTimeEventOutcome : std::uint8_t {
  RootStarted,
  SelectedChild,
  NoLegalTransition,
};

llvm::StringRef applicationResourceTimeEventOutcomeSpelling(
    ApplicationResourceTimeEventOutcome outcome);

enum class ApplicationResourceTimeExecutionErrorReason : std::uint8_t {
  UnknownLifecycleEvent,
  InvalidOccurrence,
  NonMonotonicCoordinate,
  OccurrenceMismatch,
  AmbiguousLegalTransition,
  TransitionGraphUnavailable,
  TraceNotJoined,
  ManifestMismatch,
  ForeignSchema,
  MalformedEncoding,
  NonCanonicalEncoding,
  ReplayMismatch,
};

class ApplicationResourceTimeExecutionError final
    : public llvm::ErrorInfo<ApplicationResourceTimeExecutionError> {
public:
  static char ID;

  ApplicationResourceTimeExecutionError(
      ApplicationResourceTimeExecutionErrorReason reason, std::string message)
      : reason_(reason), message_(std::move(message)) {}

  ApplicationResourceTimeExecutionErrorReason reason() const { return reason_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  ApplicationResourceTimeExecutionErrorReason reason_;
  std::string message_;
};

/// One synchronously committed root event. A selected transition is the exact
/// compiler-owned edge resolved from the prepared graph. Its active
/// allocations, live-state sets, Mapping and Deployment endpoints, deltas,
/// and costs remain owned by ResourceTimeTransition and are not reauthored by
/// this trace. NoLegalTransition is a successful typed stay decision.
struct ApplicationResourceTimeExecutionEvent final {
  sim::SystemRootLifecycleObservation observation;
  dataflow::RootThreadLaunchRef root;
  ApplicationResourceTimeEventOutcome outcome =
      ApplicationResourceTimeEventOutcome::RootStarted;
  pnr::ResourceTimeTransitionEndpointReference parent;
  pnr::ResourceTimeTransitionEndpointReference current;
  std::vector<dataflow::RootThreadLaunchRef> activeRoots;
  std::vector<dataflow::RootThreadLaunchRef> completedRoots;
  std::optional<pnr::ResourceTimeTransition> transition;
};

/// Application-owned synchronous adapter over the one Runtime selector. It
/// applies a unique legal child at the exact completion callback, records a
/// typed stay when none exists, and refuses ambiguous policy choices. It does
/// not discover Mapping states, run PnR, or preempt live work.
class ApplicationResourceTimeExecutionSession final {
public:
  static llvm::Expected<ApplicationResourceTimeExecutionSession>
  createPrepared(pnr::ResourceTimeTransitionGraph graph,
                 runtime::LoadedDeployment &loaded,
                 const ArtifactStore &artifacts, const BlobStore &blobs);

  ApplicationResourceTimeExecutionSession(
      ApplicationResourceTimeExecutionSession &&) noexcept = default;
  ApplicationResourceTimeExecutionSession &
  operator=(ApplicationResourceTimeExecutionSession &&) noexcept = default;
  ApplicationResourceTimeExecutionSession(
      const ApplicationResourceTimeExecutionSession &) = delete;
  ApplicationResourceTimeExecutionSession &
  operator=(const ApplicationResourceTimeExecutionSession &) = delete;

  llvm::Expected<ApplicationResourceTimeExecutionEvent>
  apply(const sim::SystemRootLifecycleObservation &observation,
        runtime::LoadedDeployment &loaded);

  const runtime::ResourceTimeTransitionSelectionSession &selection() const {
    return selection_;
  }
  llvm::ArrayRef<ApplicationResourceTimeExecutionEvent> events() const {
    return events_;
  }
  bool joined() const { return selection_.mappedRootsJoined(); }

private:
  explicit ApplicationResourceTimeExecutionSession(
      runtime::ResourceTimeTransitionSelectionSession selection)
      : selection_(std::move(selection)) {}

  runtime::ResourceTimeTransitionSelectionSession selection_;
  std::vector<ApplicationResourceTimeExecutionEvent> events_;
};

/// Strictly replayable persistent projection of one joined session. The
/// runtime manifest remains the graph owner. Selected transition payloads and
/// QoR references are mechanically resolved from that manifest during import.
class FinalizedApplicationResourceTimeExecutionTrace final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const ArtifactRootReference &runtimeManifest() const {
    return runtimeManifest_;
  }
  llvm::ArrayRef<ArtifactRootReference> qorEvidence() const {
    return qorEvidence_;
  }
  llvm::ArrayRef<ApplicationResourceTimeExecutionEvent> events() const {
    return events_;
  }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }

private:
  FinalizedApplicationResourceTimeExecutionTrace(
      ArtifactRootReference reference, ArtifactRootReference runtimeManifest,
      std::vector<ArtifactRootReference> qorEvidence,
      std::vector<ApplicationResourceTimeExecutionEvent> events,
      CanonicalSemanticBytes canonicalBytes)
      : reference_(std::move(reference)),
        runtimeManifest_(std::move(runtimeManifest)),
        qorEvidence_(std::move(qorEvidence)), events_(std::move(events)),
        canonicalBytes_(std::move(canonicalBytes)) {}

  ArtifactRootReference reference_;
  ArtifactRootReference runtimeManifest_;
  std::vector<ArtifactRootReference> qorEvidence_;
  std::vector<ApplicationResourceTimeExecutionEvent> events_;
  CanonicalSemanticBytes canonicalBytes_;

  friend llvm::Expected<FinalizedApplicationResourceTimeExecutionTrace>
  publishApplicationResourceTimeExecutionTrace(
      const FinalizedApplicationRuntimeManifest &,
      const ApplicationResourceTimeExecutionSession &, const ArtifactStore &,
      const BlobStore &);
  friend llvm::Expected<FinalizedApplicationResourceTimeExecutionTrace>
  importApplicationResourceTimeExecutionTrace(const ArtifactRootReference &,
                                              const ArtifactStore &,
                                              const BlobStore &);
};

llvm::Expected<FinalizedApplicationResourceTimeExecutionTrace>
publishApplicationResourceTimeExecutionTrace(
    const FinalizedApplicationRuntimeManifest &manifest,
    const ApplicationResourceTimeExecutionSession &session,
    const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<FinalizedApplicationResourceTimeExecutionTrace>
importApplicationResourceTimeExecutionTrace(
    const ArtifactRootReference &reference, const ArtifactStore &artifacts,
    const BlobStore &blobs);

} // namespace loom::application

#endif // LOOM_APPLICATION_RESOURCETIMEEXECUTION_H
