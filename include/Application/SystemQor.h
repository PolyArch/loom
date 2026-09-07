#ifndef LOOM_APPLICATION_SYSTEMQOR_H
#define LOOM_APPLICATION_SYSTEMQOR_H

#include "Application/RuntimeManifest.h"
#include "Evaluation/Evidence.h"
#include "Simulator/SimulationExecution.h"
#include "Simulator/SystemActivity.h"

namespace llvm::json { class OStream; }
namespace loom { struct ResolvedConfig; }
namespace loom::application {

inline constexpr llvm::StringLiteral applicationSystemQorProjectionSchema =
    "loom.application.system_qor_projection";
inline constexpr llvm::StringLiteral applicationSystemQorProjectionVersion = "2.0";

/// Qualification is a strict complete-program speedup joined with a saturated
/// accelerator resource. Both resource branches are measured over the candidate's
/// accelerated window: the interval from its first root Start lifecycle event
/// through its last root Completion lifecycle event, host gaps between launches
/// included. Speedup remains the complete-program comparison, so a run that
/// spends most of its program on the host cannot hide that cost in the window.
/// A run with no launches has no window and no accelerated resource claim.
inline constexpr std::uint64_t applicationMinimumResourceUtilizationNumerator = 9;
inline constexpr std::uint64_t applicationMinimumResourceUtilizationDenominator = 10;

/// A candidate whose accelerated window covers less than this fraction of its
/// complete program is host bound: neither accelerator resource can explain its
/// runtime, because the accelerator is barely running.
inline constexpr std::uint64_t applicationHostBoundWindowNumerator = 1;
inline constexpr std::uint64_t applicationHostBoundWindowDenominator = 10;

enum class ApplicationSystemQorStatus : std::uint8_t { Qualified, NotQualified };

/// Typed classification of the measured candidate window, derived in this
/// order: a saturated memory service, then a saturated compute array, then a
/// window too small to matter, otherwise unsaturated latency. It explains the
/// same measurements the status uses and is not a second gate. DSE consumes it.
enum class ApplicationSystemBottleneck : std::uint8_t {
  MemoryBandwidthBound,
  ComputeBound,
  HostBound,
  LatencyBound,
};

llvm::StringRef
applicationSystemBottleneckSpelling(ApplicationSystemBottleneck bottleneck);

struct ApplicationSystemRunEvidence final {
  ArtifactRootReference execution;
  ArtifactRootReference evidence;
  std::optional<ArtifactRootReference> productOracleEvidence;
};

/// Derived measurements remain backed by their exact immutable owner roots.
struct ApplicationSystemRunMeasurement final {
  ApplicationSystemRunEvidence roots;
  ArtifactRootReference request;
  std::optional<ArtifactRootReference> productOracleRequest;
  std::uint64_t elapsedTicks;
  sim::SystemMemoryActivity memoryActivity;
  evaluation::ExactRatio memoryUtilization;
};

/// Compute-occupancy inputs the System driver aggregates for the candidate.
/// `retiredComputeFirings` sums the retired firings of Compute-kind actors over
/// every candidate Spatial invocation replayed standalone on the CGRA engine
/// across its LaunchToTerminal window. `mappedComputeUnits` counts the distinct
/// physical PE occurrences, spatial and temporal, that carry at least one
/// compute realization or binding in the selected SpatialMapping.
/// `launchedAccCores` counts the distinct AccCores that received at least one
/// invocation. `referenceCycleTicks` is the SpatialCore clock-domain period in
/// gem5 ticks, taken from the Fabric clock contract rather than assumed.
/// The owner keeps every formula; the driver supplies only these facts.
struct ApplicationSystemComputeInputs final {
  std::uint64_t retiredComputeFirings = 0;
  std::uint64_t mappedComputeUnits = 0;
  std::uint64_t launchedAccCores = 0;
  std::uint64_t referenceCycleTicks = 0;
};

/// One retired compute firing occupies its bound compute unit for exactly one
/// reference cycle. Occupancy is therefore the retired compute firings divided
/// by the compute-cycle capacity the launched accelerators offered across the
/// window: mapped compute units * launched AccCores * window reference cycles,
/// where window reference cycles are the window's gem5 ticks divided by the
/// SpatialCore clock period.
struct ApplicationSystemComputeMeasurement final {
  ApplicationSystemComputeInputs inputs;
  evaluation::ExactRatio occupancy;
};

/// The candidate's accelerated window and both resource occupancies measured
/// over it. Host-only members structurally have none: they launch nothing.
struct ApplicationSystemWindowMeasurement final {
  sim::SystemAcceleratedWindow window;
  evaluation::ExactRatio memoryUtilization;
  ApplicationSystemComputeMeasurement compute;
};

/// A validated post-execution relation, not another Artifact family or mutable
/// completion flag. Only qualification can construct it; JSON is a projection.
class ApplicationSystemQor final {
public:
  const ArtifactRootReference &runtimeManifest() const { return manifest_; }
  const ArtifactRootReference &gem5Binding() const { return binding_; }
  const ApplicationSystemRunMeasurement &hostOnly() const { return host_; }
  const ApplicationSystemRunMeasurement &candidate() const { return candidate_; }
  const ApplicationSystemWindowMeasurement &candidateWindow() const { return window_; }
  evaluation::ExactRatio speedup() const { return speedup_; }
  ApplicationSystemBottleneck bottleneck() const;
  ApplicationSystemQorStatus status() const;

private:
  ApplicationSystemQor(ArtifactRootReference manifest, ArtifactRootReference binding,
                       ApplicationSystemRunMeasurement host,
                       ApplicationSystemRunMeasurement candidate,
                       ApplicationSystemWindowMeasurement window,
                       evaluation::ExactRatio speedup)
      : manifest_(std::move(manifest)), binding_(std::move(binding)),
        host_(std::move(host)), candidate_(std::move(candidate)),
        window_(std::move(window)), speedup_(speedup) {}
  ArtifactRootReference manifest_;
  ArtifactRootReference binding_;
  ApplicationSystemRunMeasurement host_;
  ApplicationSystemRunMeasurement candidate_;
  ApplicationSystemWindowMeasurement window_;
  evaluation::ExactRatio speedup_;

  friend llvm::Expected<ApplicationSystemQor> qualifyApplicationSystemQor(
      const FinalizedApplicationRuntimeManifest &, const ApplicationSystemRunEvidence &,
      const evaluation::CaseArtifactResolution &, const ApplicationSystemRunEvidence &,
      const evaluation::CaseArtifactResolution &, const ApplicationSystemComputeInputs &,
      const ResolvedConfig &, const ArtifactStore &, const BlobStore &);
};

llvm::Expected<ApplicationSystemQor> qualifyApplicationSystemQor(
    const FinalizedApplicationRuntimeManifest &manifest,
    const ApplicationSystemRunEvidence &hostOnly,
    const evaluation::CaseArtifactResolution &hostResolution,
    const ApplicationSystemRunEvidence &candidate,
    const evaluation::CaseArtifactResolution &candidateResolution,
    const ApplicationSystemComputeInputs &candidateCompute,
    const ResolvedConfig &config, const ArtifactStore &artifacts, const BlobStore &blobs);

void writeApplicationSystemQorJsonFields(llvm::json::OStream &json,
                                       const ApplicationSystemQor &qor);

} // namespace loom::application
#endif
