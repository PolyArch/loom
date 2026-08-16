#ifndef LOOM_RUNTIME_GEM5_LOOM_THREAD_DISPATCH_HH
#define LOOM_RUNTIME_GEM5_LOOM_THREAD_DISPATCH_HH

#include "dev/io_device.hh"
#include "params/LoomThreadDispatch.hh"
#include "sim/eventq.hh"

#include <cstdint>

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
    Running = 1,
    Complete = 2,
    Failed = 3
  };

  LoomRiscvDeploymentWorkload *const workload;
  State state = State::Idle;
  std::uint64_t selectedTarget = 0;
  std::uint64_t invocationAddress = 0;
  std::uint64_t invocationSize = 0;
  std::uint64_t activeInvocationAddress = 0;
  std::uint64_t activeInvocationSize = 0;
  std::uint32_t errorCode = 0;
  EventFunctionWrapper dispatchEvent;
  EventFunctionWrapper completionEvent;

  void beginDispatch();
  void finishDispatch();
  void fail(std::uint32_t code);
  std::uint32_t status() const;
};

} // namespace gem5

#endif // LOOM_RUNTIME_GEM5_LOOM_THREAD_DISPATCH_HH
