#ifndef LOOM_EVALUATION_MODELS_SYSTEMRUNTIMEANALYTIC_H
#define LOOM_EVALUATION_MODELS_SYSTEMRUNTIMEANALYTIC_H

#include "Dataflow/IR/DataflowStructuralRefs.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::evaluation::models {

/// Platform facts of one exact System that bound every low-confidence System
/// runtime estimate. Each value is derived from the Fabric System root or the
/// builtin gem5 platform policy; the model owns no free timing constant.
struct SystemPlatformModel final {
  /// Period of the System clock that paces InstructionCores and SpatialCores.
  std::uint64_t clockPeriodPicoseconds = 0;
  /// Cycles one HostCore executable leaf occupies, from its realization kind.
  std::uint64_t hostCyclesPerInstructionLeaf = 0;
  std::uint64_t accCoreCount = 0;
  /// Bounded completion of one shared-memory request seen by a SpatialCore.
  std::uint64_t memoryLatencyPicoseconds = 0;
  /// Acceptance service cost of one byte at the shared memory service.
  std::uint64_t memoryServicePicosecondsPerByte = 0;
  /// Acceptance service cost of one operation at the shared memory service:
  /// its rate window divided by the operations it admits per window.
  std::uint64_t memoryServicePicosecondsPerOperation = 0;
  /// Line fills one SpatialCore may keep outstanding at the shared memory:
  /// the miss-status entries of its access cache, capped by the endpoint.
  std::uint64_t accCoreOutstandingRequests = 0;
  /// Bytes one outstanding SpatialCore request transfers: one line of its
  /// access cache.
  std::uint64_t accCoreRequestBytes = 0;
  /// Host-side cost of submitting one Spatial activation through Thread
  /// Dispatch; activations of one root serialize on the HostCore.
  std::uint64_t launchDispatchPicoseconds = 0;
  /// Accelerator-side fixed cost of one activation: InstructionCore entry,
  /// bridge programming, and completion signalling, excluding the wire fetch.
  std::uint64_t launchFixedPicoseconds = 0;
  /// Binary configuration image one SpatialCore loads before its first launch,
  /// read from its sole owner `packedConfigurationImageBytesPerAccCore`. The
  /// gem5 Spatial Bridge charges its configuration transport the same size.
  std::uint64_t configurationBytesPerCore = 0;

  friend bool operator==(const SystemPlatformModel &lhs,
                         const SystemPlatformModel &rhs) {
    return lhs.clockPeriodPicoseconds == rhs.clockPeriodPicoseconds &&
           lhs.hostCyclesPerInstructionLeaf ==
               rhs.hostCyclesPerInstructionLeaf &&
           lhs.accCoreCount == rhs.accCoreCount &&
           lhs.memoryLatencyPicoseconds == rhs.memoryLatencyPicoseconds &&
           lhs.memoryServicePicosecondsPerByte ==
               rhs.memoryServicePicosecondsPerByte &&
           lhs.memoryServicePicosecondsPerOperation ==
               rhs.memoryServicePicosecondsPerOperation &&
           lhs.accCoreOutstandingRequests == rhs.accCoreOutstandingRequests &&
           lhs.accCoreRequestBytes == rhs.accCoreRequestBytes &&
           lhs.launchDispatchPicoseconds == rhs.launchDispatchPicoseconds &&
           lhs.launchFixedPicoseconds == rhs.launchFixedPicoseconds &&
           lhs.configurationBytesPerCore == rhs.configurationBytesPerCore;
  }
};

/// Derives the platform model of one complete System root. The configuration
/// image size comes from the Hardware Configuration owner, which derives it
/// once per Fabric identity and memoizes it for the process.
llvm::Expected<SystemPlatformModel>
projectSystemPlatformModel(const fabric::FinalizedFabricRoot &fabricRoot);

/// Per-activation work of one static graph launch site, independent of the
/// AccCore allocation. Activations count dynamic graph firings of the site
/// over the complete program; every other field describes one firing.
struct AnalyticLaunchEstimate final {
  ::dataflow::StaticGraphLaunchRef launch;
  std::uint64_t activations = 0;
  /// Spatial reference cycles one activation occupies its SpatialCore.
  std::uint64_t computeCyclesPerActivation = 0;
  /// Bytes one activation moves through the shared memory service.
  std::uint64_t externalMemoryBytesPerActivation = 0;
  /// Memory actor firings one activation submits to the shared memory
  /// service; each is one request that occupies an outstanding slot.
  std::uint64_t memoryTransactionsPerActivation = 0;
  /// Distinct memory actors of the graph. A memory actor holds one request
  /// in flight until its response returns, so the actors, not only the
  /// service's outstanding slots, bound the requests one activation overlaps.
  std::uint64_t memoryActors = 0;
  /// Bytes one activation's invocation wire carries across the bridge.
  std::uint64_t boundaryPayloadBytesPerActivation = 0;

  friend bool operator==(const AnalyticLaunchEstimate &lhs,
                         const AnalyticLaunchEstimate &rhs) {
    return lhs.launch == rhs.launch && lhs.activations == rhs.activations &&
           lhs.computeCyclesPerActivation == rhs.computeCyclesPerActivation &&
           lhs.externalMemoryBytesPerActivation ==
               rhs.externalMemoryBytesPerActivation &&
           lhs.memoryTransactionsPerActivation ==
               rhs.memoryTransactionsPerActivation &&
           lhs.memoryActors == rhs.memoryActors &&
           lhs.boundaryPayloadBytesPerActivation ==
               rhs.boundaryPayloadBytesPerActivation;
  }
  friend bool operator!=(const AnalyticLaunchEstimate &lhs,
                         const AnalyticLaunchEstimate &rhs) {
    return !(lhs == rhs);
  }
};

/// The term that bounds one launch site's duration under one allocation.
enum class AnalyticLaunchBottleneck : std::uint8_t {
  Launch,
  Compute,
  MemoryBandwidth,
  MemoryLatency,
};

llvm::StringRef toString(AnalyticLaunchBottleneck bottleneck);

struct AnalyticLaunchDuration final {
  std::uint64_t picoseconds = 0;
  AnalyticLaunchBottleneck bottleneck = AnalyticLaunchBottleneck::Launch;
  /// The competing per-activation terms behind the bound, for diagnostics.
  std::uint64_t computePicoseconds = 0;
  std::uint64_t bandwidthPicoseconds = 0;
  std::uint64_t latencyChainPicoseconds = 0;
  std::uint64_t fixedPicoseconds = 0;
  std::uint64_t dispatchPicoseconds = 0;
};

/// Roofline duration of every activation of one launch site spread over
/// `accCores` SpatialCores: activations serialize their host dispatch, each
/// AccCore runs its share of activations back to back, and one activation
/// takes its fixed launch cost plus the largest of its compute time, its
/// share of the memory service bandwidth, and its latency-bound request
/// chain. The bottleneck names the largest term.
llvm::Expected<AnalyticLaunchDuration>
estimateLaunchDuration(const SystemPlatformModel &platform,
                       const AnalyticLaunchEstimate &launch,
                       std::uint64_t accCores);

/// The configuration residency phase of the accelerated window: the time
/// `accCores` SpatialCores take to stream the binary configuration image
/// concurrently through the shared memory service before their first launch.
/// The invocation phase is the launch durations above.
llvm::Expected<std::uint64_t>
estimateConfigurationResidencyPicoseconds(const SystemPlatformModel &platform,
                                          std::uint64_t accCores);

/// Serialized-host residual: executable leaves outside Spatial ownership.
llvm::Expected<std::uint64_t>
estimateHostResidualPicoseconds(const SystemPlatformModel &platform,
                                std::uint64_t instructionLeaves);

/// Appends the canonical bytes of the launch estimates to a cache key.
void appendAnalyticLaunchEstimates(
    std::vector<std::uint8_t> &bytes,
    llvm::ArrayRef<AnalyticLaunchEstimate> launches);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_SYSTEMRUNTIMEANALYTIC_H
