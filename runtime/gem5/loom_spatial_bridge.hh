#ifndef LOOM_RUNTIME_GEM5_LOOM_SPATIAL_BRIDGE_HH
#define LOOM_RUNTIME_GEM5_LOOM_SPATIAL_BRIDGE_HH

#include "Runtime/Gem5BridgeWire.h"

#include "dev/dma_device.hh"
#include "params/LoomSpatialBridge.hh"
#include "sim/eventq.hh"

#include <cstdint>
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
  enum class State : std::uint32_t {
    Idle = 0,
    Running = 1,
    WaitingForMemory = 2,
    Complete = 3,
    Failed = 4,
  };

  static constexpr std::uint32_t statusBusy = 1u << 0;
  static constexpr std::uint32_t statusDone = 1u << 1;
  static constexpr std::uint32_t statusError = 1u << 2;

  const Addr pioAddress;
  const Addr pioSize;
  const Tick pioDelay;
  const std::string engineSocketPath;
  const std::string resultPath;
  const std::uint64_t maximumMessageBytes;

  int engineSocket = -1;
  State state = State::Idle;
  std::uint32_t errorCode = 0;
  std::uint64_t nextSequence = 0;
  std::vector<std::uint8_t> launchPayload;
  std::vector<std::uint8_t> memoryBuffer;
  loom::runtime::Gem5BridgeMemoryRequest pendingMemory;
  loom::runtime::Gem5BridgeCompletion pendingCompletion;

  EventFunctionWrapper launchEvent;
  EventFunctionWrapper dmaCompletionEvent;
  EventFunctionWrapper completionEvent;

  bool connectEngine();
  bool sendMessage(const loom::runtime::Gem5BridgeMessage &message);
  bool receiveMessage(loom::runtime::Gem5BridgeMessage &message);
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
