#ifndef LOOM_RUNTIME_GEM5_LOOM_SPATIAL_BRIDGE_HH
#define LOOM_RUNTIME_GEM5_LOOM_SPATIAL_BRIDGE_HH

#include "Runtime/Gem5BridgeWire.h"

#include "base/stats/group.hh"
#include "base/stats/units.hh"
#include "dev/dma_device.hh"
#include "params/LoomSpatialBridge.hh"
#include "runtime/gem5/loom_memory_service_probe.hh"
#include "runtime/gem5/loom_thread_dispatch.hh"
#include "sim/eventq.hh"

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gem5 {

class LoomSpatialEngineSession;

class LoomSpatialBridge final : public DmaDevice {
public:
  using Params = LoomSpatialBridgeParams;

  explicit LoomSpatialBridge(const Params &params);

  AddrRangeList getAddrRanges() const override;
  Tick read(PacketPtr packet) override;
  Tick write(PacketPtr packet) override;

  /// This Bridge's contribution to the accelerated window, flattened for the
  /// configuration script: configuration residency then invocation, each as
  /// begin tick, begin service occupancy, end tick, end service occupancy. An
  /// empty vector means this Bridge launched nothing inside the measured
  /// computation, so it contributes no phase.
  std::vector<std::uint64_t> acceleratedPhases() const;

private:
  friend class LoomSpatialEngineSession;

  enum class State : std::uint32_t {
    Idle = 0,
    Running = 1,
    WaitingForChannelCommit = 5,
    WaitingForCompletion = 6,
    Complete = 3,
    Failed = 4,
  };

  enum class ResultPublication {
    Published,
    TooLarge,
    OpenFailed,
    WriteFailed,
  };

  struct PerformanceStatistics final : public statistics::Group {
    explicit PerformanceStatistics(statistics::Group *parent);

    statistics::Scalar callbackCpuNanoseconds;
    statistics::Scalar engineWaitNanoseconds;
    statistics::Scalar messageCount;
    statistics::Scalar invocationCount;
    statistics::Scalar clockFailureCount;
    statistics::Scalar staticLaunchFetchCount;
  } performanceStatistics;

  /// One phase of this Bridge's accelerated window: the ticks it spans and the
  /// shared-memory service integral sampled at each end, so differencing the
  /// samples measures the service the phase consumed.
  struct PhaseObservation final {
    bool observed = false;
    std::uint64_t beginTick = 0;
    std::uint64_t beginOccupiedTicks = 0;
    std::uint64_t endTick = 0;
    std::uint64_t endOccupiedTicks = 0;

    void openAt(std::uint64_t tick, std::uint64_t occupied);
    void closeAt(std::uint64_t tick, std::uint64_t occupied);
  };

  struct CallbackAccounting final {
    std::uint64_t started = 0;
    bool valid = false;
  };

  /// One Bridge memory transaction the engine has handed to this Bridge. Each
  /// transaction owns its ready tick and its DMA completion, so the memory
  /// system observes the modeled concurrency of the Spatial memory service.
  struct MemoryTransaction final {
    MemoryTransaction(LoomSpatialBridge &bridge,
                      loom::runtime::Gem5BridgeMemoryRequest request);

    loom::runtime::Gem5BridgeMemoryRequest request;
    std::vector<std::uint8_t> buffer;
    EventFunctionWrapper issueEvent;
    EventFunctionWrapper completionEvent;
  };


  const Addr pioAddress;
  const Addr pioSize;
  const Tick pioDelay;
  const std::uint64_t bridgeSessionOrdinal;
  LoomSpatialEngineSession *engineSession;
  LoomThreadDispatch *const threadDispatch;
  LoomMemoryServiceProbe *const memoryService;
  /// Bytes of the Fabric-derived binary configuration image this SpatialCore
  /// loads. The launch image the guest stages is the functional transport of
  /// the immutable plane; this is what the modeled memory system carries.
  const std::uint64_t configurationImageBytes;
  const std::string resultPath;
  const std::uint64_t maximumMessageBytes;
  const bool collectPerformance;

  PhaseObservation configurationResidencyPhase;
  PhaseObservation invocationPhase;

  State state = State::Idle;
  std::uint32_t errorCode = 0;
  std::uint64_t nextSequence = 0;
  std::uint64_t staticLaunchAddress = 0;
  std::uint32_t staticLaunchSize = 0;
  std::uint64_t invocationAddress = 0;
  std::uint32_t invocationSize = 0;
  std::uint64_t activeStaticLaunchAddress = 0;
  std::uint32_t activeStaticLaunchSize = 0;
  std::uint64_t activeInvocationAddress = 0;
  std::uint32_t activeInvocationSize = 0;
  std::uint64_t lastCompletionTick = 0;
  /// Scratch of the timed configuration-image transfer. Its bytes are never
  /// read: the modeled memory system carries exactly this many bytes while the
  /// immutable plane itself arrives through the staged launch image.
  std::vector<std::uint8_t> configurationTransportBuffer;
  std::vector<std::uint8_t> staticLaunchPayload;
  /// The descriptor of the immutable plane currently resident in
  /// staticLaunchPayload. A Start naming the same descriptor reuses it.
  std::uint64_t residentStaticLaunchAddress = 0;
  std::uint32_t residentStaticLaunchSize = 0;
  std::vector<std::uint8_t> invocationPayload;
  std::vector<std::uint8_t> memorySnapshotPayload;
  std::map<std::uint64_t, std::unique_ptr<MemoryTransaction>>
      memoryTransactions;
  /// Answered transactions whose DMA completion event the queue is still
  /// servicing. They are reclaimed at the next boundary or completion.
  std::vector<std::unique_ptr<MemoryTransaction>> retiredMemoryTransactions;
  loom::runtime::Gem5BridgeCompletion pendingCompletion;
  std::uint64_t publishedResultBytes = 0;
  std::optional<std::chrono::steady_clock::time_point> engineWaitStarted;

  EventFunctionWrapper launchEvent;
  EventFunctionWrapper staticLaunchCompletionEvent;
  EventFunctionWrapper invocationCompletionEvent;
  EventFunctionWrapper completionEvent;
  EventFunctionWrapper channelCommitEvent;

  CallbackAccounting beginCallbackAccounting();
  void finishCallbackAccounting(CallbackAccounting accounting);
  /// Runs one Bridge callback under host CPU accounting.
  template <typename Action> void runAccounted(Action &&action) {
    const CallbackAccounting accounting = beginCallbackAccounting();
    action();
    finishCallbackAccounting(accounting);
  }
  void startEngineWait();
  void finishEngineWait();
  /// Whether the source-declared computation interval the Thread Dispatch
  /// device owns is open, which is when phase transitions are observed.
  bool measuring() const;
  /// The shared-memory service integral sampled now.
  std::uint64_t serviceOccupancy() const;
  ResultPublication publishResults();
  void fetchStaticLaunch();
  void completeConfigurationTransport();
  void fetchInvocation();
  void startLaunch();
  void acceptBoundary(const loom::runtime::Gem5BridgeMessage &message,
                      Tick causalTick);
  void completeChannelCommit();
  void acceptMemoryRequest(const loom::runtime::Gem5BridgeMessage &message,
                           Tick causalTick);
  void issueMemoryRequest(MemoryTransaction &transaction);
  void completeMemoryRequest(MemoryTransaction &transaction);
  void reclaimRetiredMemory();
  void completeInvocation();
  void fail(std::uint32_t code, const std::string &message);
  void resetBridge();
  std::uint32_t status() const;
};

} // namespace gem5

#endif // LOOM_RUNTIME_GEM5_LOOM_SPATIAL_BRIDGE_HH
