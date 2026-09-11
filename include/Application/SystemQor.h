#ifndef LOOM_APPLICATION_SYSTEMQOR_H
#define LOOM_APPLICATION_SYSTEMQOR_H

#include "Application/RuntimeManifest.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Evaluation/Evidence.h"
#include "Simulator/SimulationExecution.h"
#include "Simulator/SystemActivity.h"

#include <optional>
#include <vector>

namespace llvm::json { class OStream; }
namespace loom { struct ResolvedConfig; }
namespace loom::application {

inline constexpr llvm::StringLiteral applicationSystemQorProjectionSchema =
    "loom.application.system_qor_projection";
inline constexpr llvm::StringLiteral applicationSystemQorProjectionVersion =
    "5.0";

/// Qualification compares the same source-declared computation on both images,
/// from prepared shared-memory inputs through visible output completion. All
/// intervening host work and dispatch remain charged. Full-program Runtime is
/// retained separately and never substitutes for a missing computation
/// interval.
inline constexpr std::uint64_t applicationMinimumResourceUtilizationNumerator = 9;
inline constexpr std::uint64_t applicationMinimumResourceUtilizationDenominator = 10;

/// An accelerator active for less than this fraction of useful computation
/// cannot explain that computation's runtime; its host residual dominates.
inline constexpr std::uint64_t applicationHostBoundWindowNumerator = 1;
inline constexpr std::uint64_t applicationHostBoundWindowDenominator = 10;

/// Configuration residency and invocation are per-AccCore phases aggregated
/// independently, so on an array whose cores are dispatched one at a time the
/// residency phase also covers the dispatch of the later cores. The invocation
/// phase must remain the majority of the accelerated window: above this the
/// window measures how long the array took to configure rather than the
/// computation that residency serves, and the invocation-phase saturation no
/// longer explains the measured speedup.
inline constexpr std::uint64_t applicationMaximumLaunchOverheadNumerator = 1;
inline constexpr std::uint64_t applicationMaximumLaunchOverheadDenominator = 2;

enum class ApplicationSystemQorStatus : std::uint8_t {
  Qualified,
  NotQualified,
  Unmeasured
};

/// Typed classification of the measured candidate window, derived in this
/// order: an accelerated window dominated by configuration residency, then a
/// saturated memory service, then a saturated compute array, then a window too
/// small to matter, otherwise unsaturated latency. It explains the same
/// measurements the status uses and is not a second gate. DSE consumes it.
enum class ApplicationSystemBottleneck : std::uint8_t {
  LaunchBound,
  MemoryBandwidthBound,
  ComputeBound,
  HostBound,
  LatencyBound,
  Unmeasured,
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
  std::optional<sim::SystemComputationInterval> computationInterval;
};

/// One operation class the candidate's Compute-kind actors realize: an
/// operation schema at one element width. It is the unit of the compute
/// speed-of-light bound because a Fabric's ability to issue a class is a fact
/// of its FU inventory, not of the part of it one Mapping happened to use.
/// The System driver supplies only these facts; the owner keeps every formula.
struct ApplicationSystemComputeClassInputs final {
  ::dataflow::OperationSchemaId schema{};
  std::uint32_t elementBits = 0;
  /// Retired firings of this class's actors over every measured candidate
  /// Spatial invocation replayed standalone on the CGRA engine across its
  /// LaunchToTerminal window, each firing weighted by the actor's lane count.
  std::uint64_t retiredElementFirings = 0;
  /// Element lanes one Fabric can issue for this class in one reference
  /// cycle: the sum over every FU operation node admitting the class of that
  /// node's result lanes. A Temporal PE's FU issues once per cycle however
  /// many resident instruction contexts share it.
  std::uint64_t peakIssueLanesPerCycle = 0;
  /// Realization slots one Fabric offers this class: one per admitting FU
  /// operation node on a Spatial PE and one per resident instruction context
  /// of a Temporal PE.
  std::uint64_t placementSlots = 0;
  /// Compute realizations of this class bound by the selected SpatialMappings.
  std::uint64_t boundRealizations = 0;
};

/// `classes` is sorted by (schema, elementBits) with no repeated class.
/// `launchedAccCores` counts the distinct AccCores that received at least one
/// invocation; every launched AccCore carries one instance of the same Fabric.
/// `referenceCycleTicks` is the SpatialCore clock-domain period in gem5 ticks,
/// taken from the Fabric clock contract rather than assumed.
struct ApplicationSystemComputeInputs final {
  std::vector<ApplicationSystemComputeClassInputs> classes;
  std::uint64_t launchedAccCores = 0;
  std::uint64_t referenceCycleTicks = 0;
};

/// Occupancy is the class's retired element firings divided by the element
/// lanes the launched Fabrics could have issued for it across the invocation
/// phase: peak issue lanes per cycle * launched AccCores * phase reference
/// cycles, where phase reference cycles are the phase's gem5 ticks divided by
/// the SpatialCore clock period. Placement utilization is the class's bound
/// realizations divided by its placement slots across the launched Fabrics.
struct ApplicationSystemComputeClassMeasurement final {
  ApplicationSystemComputeClassInputs inputs;
  evaluation::ExactRatio occupancy;
  evaluation::ExactRatio placementUtilization;
};

/// The class nearest its speed of light bounds compute: `occupancy` is the
/// largest class occupancy and `bindingClass` names it. Placement utilization
/// is reported and never gated; a full Temporal instruction table serializes
/// its work and is not a goal.
struct ApplicationSystemComputeMeasurement final {
  ApplicationSystemComputeInputs inputs;
  std::vector<ApplicationSystemComputeClassMeasurement> classes;
  evaluation::ExactRatio occupancy;
  std::optional<std::size_t> bindingClass;
  evaluation::ExactRatio placementUtilization;
};

/// The candidate's source-declared computation interval and the accelerated
/// window inside it. Saturation is measured over the invocation phase alone:
/// configuration residency moves the binary configuration image, which the
/// service observer never counts as application data, and charging its ticks
/// to the saturation denominator would credit a fat configuration image as
/// memory appetite. `launchOverhead` is the residency phase over the whole
/// accelerated window, and the accelerated window over the computation
/// interval explains the host-residual classification.
struct ApplicationSystemWindowMeasurement final {
  sim::SystemComputationInterval window;
  /// Absent when the computation completed no accelerator invocation.
  std::optional<sim::SystemAcceleratedPhases> phases;
  evaluation::ExactRatio launchOverhead;
  evaluation::ExactRatio memoryUtilization;
  ApplicationSystemComputeMeasurement compute;

  std::uint64_t acceleratedTicks() const {
    return phases ? phases->elapsedTicks() : 0;
  }
  std::uint64_t invocationTicks() const {
    return phases ? phases->invocation.elapsedTicks() : 0;
  }
};

/// A validated post-execution relation, not another Artifact family or mutable
/// completion flag. Only qualification can construct it; JSON is a projection.
class ApplicationSystemQor final {
public:
  const ArtifactRootReference &runtimeManifest() const { return manifest_; }
  const ArtifactRootReference &gem5Binding() const { return binding_; }
  const ApplicationSystemRunMeasurement &hostOnly() const { return host_; }
  const ApplicationSystemRunMeasurement &candidate() const { return candidate_; }
  const std::optional<ApplicationSystemWindowMeasurement> &
  candidateWindow() const {
    return window_;
  }
  std::optional<evaluation::ExactRatio> speedup() const { return speedup_; }
  ApplicationSystemBottleneck bottleneck() const;
  ApplicationSystemQorStatus status() const;

private:
  ApplicationSystemQor(ArtifactRootReference manifest,
                       ArtifactRootReference binding,
                       ApplicationSystemRunMeasurement host,
                       ApplicationSystemRunMeasurement candidate,
                       std::optional<ApplicationSystemWindowMeasurement> window,
                       std::optional<evaluation::ExactRatio> speedup)
      : manifest_(std::move(manifest)), binding_(std::move(binding)),
        host_(std::move(host)), candidate_(std::move(candidate)),
        window_(std::move(window)), speedup_(speedup) {}
  ArtifactRootReference manifest_;
  ArtifactRootReference binding_;
  ApplicationSystemRunMeasurement host_;
  ApplicationSystemRunMeasurement candidate_;
  std::optional<ApplicationSystemWindowMeasurement> window_;
  std::optional<evaluation::ExactRatio> speedup_;

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
