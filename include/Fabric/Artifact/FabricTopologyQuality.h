#ifndef LOOM_FABRIC_ARTIFACT_FABRICTOPOLOGYQUALITY_H
#define LOOM_FABRIC_ARTIFACT_FABRICTOPOLOGYQUALITY_H

#include "Fabric/Identity/FabricRefImport.h"

#include "Dataflow/IR/OperationSchema.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::fabric {

/// Closed set of terminal owners whose external connectivity is measured.
/// Module roots measure PE and memory terminals. System roots measure attached
/// SpatialCore and service terminals.
enum class FabricTopologyTerminalKind : std::uint8_t {
  ProcessingElement,
  Memory,
  SpatialCore,
  ServiceEndpoint,
};

llvm::StringRef
fabricTopologyTerminalKindSpelling(FabricTopologyTerminalKind kind);

/// Exact first resources reached outward from one physical terminal. Point
/// connections, FIFO resources, and boundary resources are transparent. A
/// Module signature is recorded separately because it is correspondence, not
/// a routable or directly bound resource.
struct FabricTopologyPortQuality final {
  FabricTransportEndpointRef endpoint;
  std::vector<FabricTransportEndpointOwnerRef> routingResources;
  std::vector<FabricTransportEndpointOwnerRef> directResources;
  bool reachesModuleBoundary = false;

  bool unreachable() const {
    return routingResources.empty() && directResources.empty() &&
           !reachesModuleBoundary;
  }
};

/// Connectivity ratios are represented exactly by an owner's distinct
/// resource count over `ports.size()`. No floating-point value or threshold is
/// persisted. For a Module, routing resources are `fabric.switch`
/// occurrences. For a System, they are System transport resources.
struct FabricTopologyOwnerQuality final {
  FabricTopologyTerminalKind kind;
  FabricTransportEndpointOwnerRef owner;
  std::vector<FabricTopologyPortQuality> ports;
  std::vector<FabricTransportEndpointOwnerRef> distinctRoutingResources;
  std::vector<FabricTransportEndpointOwnerRef> distinctDirectResources;
  std::uint64_t boundaryPortCount = 0;
  std::uint64_t unreachablePortCount = 0;

  std::uint64_t portCount() const { return ports.size(); }
  std::uint64_t routingResourceCount() const {
    return distinctRoutingResources.size();
  }
  std::uint64_t directResourceCount() const {
    return distinctDirectResources.size();
  }
};

/// Exact hop extrema over PE subjects. A hop is one canonical physical
/// traversal after direction is erased solely for locality measurement. The
/// routability owner retains the directed graph; this projection does not
/// admit a route.
struct FabricTopologyHopExtreme final {
  std::uint64_t hops = 0;
  std::vector<FabricPeOccurrenceRef> subjects;
};

/// Threshold-free nearest-target distribution over one declared PE subject
/// set. Unreachable subjects remain explicit instead of receiving a sentinel
/// distance. The total is over reachable subjects only.
struct FabricTopologyHopDistribution final {
  std::uint64_t subjectCount = 0;
  std::uint64_t reachableSubjectCount = 0;
  std::uint64_t totalReachableHops = 0;
  std::vector<FabricPeOccurrenceRef> unreachableSubjects;
  std::optional<FabricTopologyHopExtreme> minimum;
  std::optional<FabricTopologyHopExtreme> maximum;
};

/// Schedule supply and physical locality for one Module scheduling domain.
/// Same-schedule PE distance excludes the subject itself. Matching-memory
/// distance considers only memories with an operation engine in this domain.
struct FabricTopologyScheduleQuality final {
  ::fabric::Schedule schedule;
  std::uint64_t peCount = 0;
  std::uint64_t memoryCount = 0;
  std::uint64_t switchCount = 0;
  FabricTopologyHopDistribution nearestSameSchedulePe;
  FabricTopologyHopDistribution nearestOtherSchedulePe;
  FabricTopologyHopDistribution nearestMatchingMemory;
  FabricTopologyHopDistribution nearestOtherScheduleMemory;
};

/// Physical distribution of one registered operation schema. `coverage`
/// measures every PE to its nearest supporting PE, while `supportingPeer`
/// measures each supporter to another supporter. Together with the exact
/// supporter inventory these distinguish broad coverage from a tight cluster.
struct FabricTopologyCapabilityQuality final {
  ::dataflow::OperationSchemaId schema;
  std::vector<FabricPeOccurrenceRef> supportingPes;
  std::uint64_t spatialPeCount = 0;
  std::uint64_t temporalPeCount = 0;
  FabricTopologyHopDistribution coverage;
  FabricTopologyHopDistribution supportingPeer;
};

/// Exact additive projection suitable for hardware-candidate ranking. Every
/// component retains a distinct physical meaning; callers choose ordering and
/// weights in the resolved objective policy rather than collapsing them here.
struct FabricTopologyDseQuality final {
  std::uint64_t unscheduledMemoryCount = 0;
  std::uint64_t scheduleSupplyGap = 0;
  std::uint64_t matchingMemoryUnreachablePeCount = 0;
  std::uint64_t matchingMemoryTotalReachableHops = 0;
  std::uint64_t capabilityCoverageUnreachablePeCount = 0;
  std::uint64_t capabilityCoverageTotalReachableHops = 0;
  std::uint64_t isolatedCapabilitySupportingPeCount = 0;
};

/// Rebuildable connectivity report for one exact canonical Fabric root. It is
/// diagnostic and DSE input, never another artifact or topology authority.
struct FabricTopologyQualityReport final {
  ArtifactIdentity artifact;
  FabricRootKind rootKind;
  std::vector<FabricTopologyOwnerQuality> owners;
  std::uint64_t unscheduledMemoryCount = 0;
  std::vector<FabricTopologyScheduleQuality> schedules;
  std::vector<FabricTopologyCapabilityQuality> capabilities;
};

struct FabricTopologyCountExtreme final {
  std::uint64_t value = 0;
  std::vector<FabricTransportEndpointOwnerRef> owners;
};

/// Canonical reduced ratio shared by every tied owner in one extreme. The
/// unreduced per-owner components remain available in the quality report.
struct FabricTopologyRatioExtreme final {
  std::uint64_t numerator = 0;
  std::uint64_t denominator = 1;
  std::vector<FabricTransportEndpointOwnerRef> owners;
};

/// Exact extrema for one terminal kind. Full owner records remain the
/// distribution; this projection makes sparse and tied outliers explicit.
/// Zero-port owners participate in port-count extrema but not ratio extrema.
struct FabricTopologyKindDistribution final {
  FabricTopologyTerminalKind kind;
  std::uint64_t ownerCount = 0;
  std::uint64_t zeroPortOwnerCount = 0;
  FabricTopologyCountExtreme minimumPortCount;
  FabricTopologyCountExtreme maximumPortCount;
  std::optional<FabricTopologyRatioExtreme> minimumRoutingRatio;
  std::optional<FabricTopologyRatioExtreme> maximumRoutingRatio;
  std::optional<FabricTopologyRatioExtreme> minimumDirectRatio;
  std::optional<FabricTopologyRatioExtreme> maximumDirectRatio;
};

llvm::Expected<FabricTopologyQualityReport>
analyzeFabricTopologyQuality(const FabricArtifactView &fabric);

/// Analyzes a root and every exact imported Module visible from it. Module
/// roots therefore return one report; System roots return the System report
/// followed by their canonical dependency reports.
llvm::Expected<std::vector<FabricTopologyQualityReport>>
analyzeFabricTopologyQualityClosure(const FabricArtifactView &fabric);

std::vector<FabricTopologyKindDistribution>
summarizeFabricTopologyQuality(const FabricTopologyQualityReport &report);

FabricTopologyDseQuality
projectFabricTopologyDseQuality(const FabricTopologyQualityReport &report);

/// Deterministic JSON projection of the derived report closure. Reference
/// spellings and integer ratio components remain exact; consumers never parse
/// presentation-only floating-point values.
llvm::Error writeFabricTopologyQualityJson(
    llvm::ArrayRef<FabricTopologyQualityReport> reports,
    llvm::raw_ostream &output);

} // namespace loom::fabric

#endif // LOOM_FABRIC_ARTIFACT_FABRICTOPOLOGYQUALITY_H
