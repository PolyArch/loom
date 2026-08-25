#ifndef LOOM_DSE_RESOURCETIMESPECTRUM_H
#define LOOM_DSE_RESOURCETIMESPECTRUM_H

#include "Common/ExecutionControl.h"
#include "DSE/PreMappingFrontier.h"
#include "DSE/ResourceTimeFrontier.h"
#include "PnR/System/SystemMappingMigration.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::dse {

/// Exact correspondence between a schedule-region identity and the
/// Canonical Dataflow root whose SystemMapping allocation must prove it. The
/// allocation bounds come from the provider-owned resource-speedup curve;
/// only Exact bounds can establish a spectrum endpoint.
struct ResourceTimeRegionMapping final {
  ::dataflow::RootThreadLaunchRef root;
  /// Absent means the provider did not establish an exact lower feasibility
  /// boundary. Unknown lower bounds remain valid for intermediate evidence,
  /// but cannot establish a spectrum endpoint.
  std::optional<std::uint64_t> minimumFeasibleAccCoreCount;
  std::uint64_t maximumUsefulAccCoreCount = 0;
  ResourceTimeEstimateSupport minimumBoundSupport =
      ResourceTimeEstimateSupport::Unsupported;
  ResourceTimeEstimateSupport maximumBoundSupport =
      ResourceTimeEstimateSupport::Unsupported;
  /// Logical epochs are a Dataflow fact, not an endpoint label. A temporal
  /// endpoint still requires an explicit event-relative active-set schedule;
  /// this field is retained as correspondence evidence and is never used as
  /// a stand-alone proof of temporal reuse.
  std::uint64_t logicalEpochCount = 0;
};

struct VerifiedResourceTimeSpectrumScenario final {
  std::uint64_t scenarioOrdinal = 0;
  PreMappingSpectrumClass spectrumClass = PreMappingSpectrumClass::Intermediate;
  std::uint64_t peakConcurrentRegions = 0;
  std::uint64_t makespanPicoseconds = 0;
  std::vector<ArtifactRootReference> systemMappings;
  /// Replayable event-relative evidence retained after independent Mapping
  /// verification. Transition endpoints retain exact Deployment lineage when
  /// the scenario changes Mapping state.
  std::vector<::loom::pnr::ResourceTimeScheduleState> states;
  ::loom::pnr::ResourceTimeTransitionSequence transitions;
  /// Present when the scenario changes Mapping and every finite state has an
  /// exact Deployment endpoint. This is the runtime-selectable catalog; the
  /// sequence above is the chosen path through it.
  std::optional<::loom::pnr::ResourceTimeTransitionGraph> transitionGraph;
};

/// Every named Mapping has passed the ordinary independent SystemMapping
/// importer and every state allocation equals that Mapping's selected AccCore
/// set for the corresponding Dataflow root. Endpoint labels use only the
/// structural global bounds: the event-relative minimum concurrent active set
/// for MaxTemporal and the maximum concurrent active set for MaxSpatial.
/// Accelerated coverage, active-set membership, and per-region allocation are
/// checked independently. Schedules that do not attain a bound remain
/// verified intermediate points rather than inferred endpoints.
struct VerifiedResourceTimeSpectrum final {
  ArtifactRootReference dataflow;
  ArtifactRootReference fabric;
  std::vector<VerifiedResourceTimeSpectrumScenario> scenarios;
};

enum class ResourceTimeSpectrumIncompleteReason : std::uint8_t {
  Unsupported,
  ProofNotEstablished,
  CancelledOrTimeout,
};

llvm::StringRef resourceTimeSpectrumIncompleteReasonSpelling(
    ResourceTimeSpectrumIncompleteReason reason);

struct IncompleteResourceTimeSpectrum final {
  ResourceTimeSpectrumIncompleteReason reason =
      ResourceTimeSpectrumIncompleteReason::ProofNotEstablished;
  std::string diagnostic;
  std::uint64_t independentlyImportedMappingCount = 0;
};

using ResourceTimeSpectrumVerification =
    std::variant<VerifiedResourceTimeSpectrum, IncompleteResourceTimeSpectrum>;

struct ResourceTimeSpectrumFunnelAccounting final {
  std::uint64_t hintCandidates = 0;
  std::uint64_t matchingMappingChecks = 0;
  std::uint64_t independentlyImportedMappings = 0;
  std::uint64_t materializedScenarios = 0;
  std::uint64_t unmatchedHints = 0;
  std::uint64_t transitionUnsupportedHints = 0;
  std::uint64_t transitionProofFailures = 0;
  std::uint64_t verifiedScenarios = 0;
  std::uint64_t mappingImportRequests = 0;
  std::uint64_t mappingImportCacheHits = 0;
  std::uint64_t mappingImportCacheMisses = 0;
  std::uint64_t mappingImportRetainedBytes = 0;
  std::uint64_t mappingProgressQualified = 0;
  std::uint64_t mappingProgressProofNotEstablished = 0;
  std::uint64_t elapsedNanoseconds = 0;
};

struct ResourceTimeSpectrumFunnelResult final {
  ResourceTimeSpectrumVerification verification;
  ResourceTimeSpectrumFunnelAccounting accounting;
};

/// One state in an ordered finite Mapping/Deployment path. The path is
/// invocation-local compiler output; it is never inferred from allocation
/// counts or discovered by scanning the ArtifactStore. Path advancement is
/// allowed only at a compiler-known completion action.
struct ResourceTimeMappingDeploymentEndpoint final {
  ArtifactRootReference mapping;
  ArtifactRootReference deployment;
};

/// Independently imports every SystemMapping used by the schedule and proves
/// its event-relative allocation correspondence. It creates no Mapping
/// legality, cache, or candidate identity of its own.
llvm::Expected<ResourceTimeSpectrumVerification> verifyResourceTimeSpectrum(
    const ::loom::pnr::ResourceTimeScheduleWitness &witness,
    llvm::ArrayRef<ResourceTimeRegionMapping> regions,
    const ArtifactStore &store, ExecutionControlView executionControl = {},
    const BlobStore *blobs = nullptr);

/// Materializes only schedule hints whose per-state allocations are realized
/// by one already verified SystemMapping, then invokes the independent
/// spectrum verifier. A missing allocation is typed proof-not-established and
/// never triggers additional Mapping work here.
llvm::Expected<ResourceTimeSpectrumFunnelResult>
verifyResourceTimeMappingFinalists(
    llvm::ArrayRef<ResourceTimeScheduleHint> hints,
    llvm::ArrayRef<ResourceTimeRegionFeature> regions,
    llvm::ArrayRef<ResourceTimeRegionResourceBound> bounds,
    llvm::ArrayRef<ArtifactRootReference> systemMappings,
    const ArtifactStore &store, ExecutionControlView executionControl = {},
    std::optional<ResourceTimeConcurrencyBounds> concurrencyBounds =
        std::nullopt,
    const BlobStore *blobs = nullptr,
    llvm::ArrayRef<ResourceTimeMappingDeploymentEndpoint> mappingPath = {});

} // namespace loom::dse

#endif // LOOM_DSE_RESOURCETIMESPECTRUM_H
