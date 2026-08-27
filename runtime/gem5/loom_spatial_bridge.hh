#ifndef LOOM_RUNTIME_GEM5_LOOM_SPATIAL_BRIDGE_HH
#define LOOM_RUNTIME_GEM5_LOOM_SPATIAL_BRIDGE_HH

#include "Runtime/Gem5BridgeWire.h"

#include "base/pollevent.hh"
#include "base/stats/group.hh"
#include "base/stats/units.hh"
#include "dev/dma_device.hh"
#include "params/LoomSpatialBridge.hh"
#include "sim/eventq.hh"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gem5 {

class LoomSpatialBridge final : public DmaDevice {
public:
  using Params = LoomSpatialBridgeParams;

  explicit LoomSpatialBridge(const Params &params);
  ~LoomSpatialBridge() override;

  AddrRangeList getAddrRanges() const override;
  Tick read(PacketPtr packet) override;
  Tick write(PacketPtr packet) override;

private:
  class EngineResponseEvent final : public PollEvent {
  public:
    EngineResponseEvent(LoomSpatialBridge &bridge, int descriptor);
    void process(int revents) override;

  private:
    LoomSpatialBridge &bridge;
  };

  enum class State : std::uint32_t {
    Idle = 0,
    Running = 1,
    WaitingForMemory = 2,
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
  } performanceStatistics;

  struct CallbackAccounting final {
    std::uint64_t started = 0;
    bool valid = false;
  };

  static constexpr std::uint32_t statusBusy = 1u << 0;
  static constexpr std::uint32_t statusDone = 1u << 1;
  static constexpr std::uint32_t statusError = 1u << 2;

  const Addr pioAddress;
  const Addr pioSize;
  const Tick pioDelay;
  const std::uint64_t bridgeSessionOrdinal;
  const std::string engineSocketPath;
  const std::string resultPath;
  const std::uint64_t maximumMessageBytes;
  const std::uint64_t maximumInvocations;
  const bool collectPerformance;

  int engineSocket = -1;
  std::unique_ptr<EngineResponseEvent> engineResponseEvent;
  bool engineCompletionReceived = false;
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
  std::vector<std::uint8_t> staticLaunchPayload;
  std::vector<std::uint8_t> invocationPayload;
  std::vector<std::uint8_t> memoryBuffer;
  loom::runtime::Gem5BridgeMemoryRequest pendingMemory;
  loom::runtime::Gem5BridgeCompletion pendingCompletion;
  loom::runtime::Gem5BridgeResultCollection completedResults;
  std::uint64_t publishedResultBytes = 0;
  std::optional<std::chrono::steady_clock::time_point> engineWaitStarted;

  EventFunctionWrapper launchEvent;
  EventFunctionWrapper staticLaunchCompletionEvent;
  EventFunctionWrapper invocationCompletionEvent;
  EventFunctionWrapper dmaCompletionEvent;
  EventFunctionWrapper completionEvent;
  EventFunctionWrapper engineDisconnectEvent;

  bool connectEngine();
  void scheduleEngineDisconnect();
  void disconnectEngine();
  bool sendMessage(const loom::runtime::Gem5BridgeMessage &message);
  bool receiveMessage(loom::runtime::Gem5BridgeMessage &message);
  CallbackAccounting beginCallbackAccounting();
  void finishCallbackAccounting(CallbackAccounting accounting);
  void runAccounted(void (LoomSpatialBridge::*action)());
  void startEngineWait();
  void finishEngineWait();
  ResultPublication publishResults();
  void fetchStaticLaunch();
  void fetchInvocation();
  void startLaunch();
  void consumeEngineMessage();
  void completeMemoryRequest();
  void completeInvocation();
  void fail(std::uint32_t code, const std::string &message);
  void resetBridge();
  std::uint32_t status() const;
};

} // namespace gem5

#endif // LOOM_RUNTIME_GEM5_LOOM_SPATIAL_BRIDGE_HH
