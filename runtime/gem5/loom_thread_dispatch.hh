#ifndef LOOM_RUNTIME_GEM5_LOOM_THREAD_DISPATCH_HH
#define LOOM_RUNTIME_GEM5_LOOM_THREAD_DISPATCH_HH

#include "Runtime/Gem5DispatchABI.h"

#include "dev/io_device.hh"
#include "params/LoomThreadDispatch.hh"
#include "sim/eventq.hh"

#include <cstdint>
#include <fstream>
#include <vector>

namespace gem5 {

class LoomRiscvDeploymentWorkload;

class LoomThreadDispatch final : public BasicPioDevice {
public:
  using Params = LoomThreadDispatchParams;

  explicit LoomThreadDispatch(const Params &params);

  Tick read(PacketPtr packet) override;
  Tick write(PacketPtr packet) override;

private:
  enum class State : std::uint32_t {
    Idle = 0,
    Queued = 1,
    Running = 2,
    Finishing = 3,
    Complete = 4,
    Failed = 5
  };

  struct DispatchRecord final {
    State state = State::Idle;
    std::uint64_t occurrence = 0;
    std::uint64_t invocationAddress = 0;
    std::uint64_t invocationSize = 0;
    std::uint32_t errorCode = 0;
    bool workerFailed = false;
  };

  LoomRiscvDeploymentWorkload *const workload;
  std::uint64_t selectedTarget = 0;
  std::uint64_t invocationAddress = 0;
  std::uint64_t invocationSize = 0;
  std::uint64_t rootEventEntity = 0;
  std::uint64_t rootEventOccurrence = 0;
  std::uint64_t nextOccurrence = 1;
  std::uint64_t nextRootEventOccurrence = 1;
  std::uint32_t commandError = 0;
  std::vector<DispatchRecord> records;
  std::ofstream rootEventTrace;
  Tick lastRootEventTick = 0;
  std::uint64_t lastRootEventDelta = 0;
  bool hasRootEvent = false;
  EventFunctionWrapper serviceEvent;

  void service();
  void scheduleService();
  void failSelected(std::uint32_t code);
  bool recordRootEvent(loom::runtime::Gem5RootLifecycleAction action);
  DispatchRecord *selectedRecord();
  const DispatchRecord *selectedRecord() const;
  std::uint32_t status(const DispatchRecord &record) const;
  std::uint32_t status() const;
};

} // namespace gem5

#endif // LOOM_RUNTIME_GEM5_LOOM_THREAD_DISPATCH_HH
