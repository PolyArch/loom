#ifndef LOOM_RUNTIME_GEM5_LOOM_RISCV_DEPLOYMENT_WORKLOAD_HH
#define LOOM_RUNTIME_GEM5_LOOM_RISCV_DEPLOYMENT_WORKLOAD_HH

#include "arch/riscv/bare_metal/fs_workload.hh"
#include "params/LoomRiscvDeploymentWorkload.hh"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gem5 {

class ThreadContext;

class LoomRiscvDeploymentWorkload final : public RiscvISA::BareMetal {
public:
  using Params = LoomRiscvDeploymentWorkloadParams;

  enum class CompletionState : std::uint32_t {
    Pending = 0,
    Complete = 1,
    Invalid = 2,
  };

  explicit LoomRiscvDeploymentWorkload(const Params &params);
  ~LoomRiscvDeploymentWorkload() override;

  void initState() override;
  const loader::SymbolTable &symtab(ThreadContext *context) override;
  void writeMemoryObservations();

  bool dispatch(std::uint64_t targetOrdinal, Addr completionAddress);
  CompletionState complete(std::uint64_t targetOrdinal);
  std::size_t targetCount() const { return targets.size(); }

private:
  struct Target final {
    std::uint64_t cpuId = 0;
    std::uint64_t imageOrdinal = 0;
    std::string entrySymbol;
    Addr bridgeAddress = 0;
    Addr launchAddress = 0;
    std::uint64_t launchSize = 0;
  };

  const std::uint64_t hostCpuId;
  const std::string hostEntrySymbol;
  const Addr hostDispatchAddress;
  const Addr hostMemoryTableAddress;
  const std::uint64_t hostMemoryTableEntries;
  const Addr stackBase;
  const Addr stackStride;
  std::vector<std::unique_ptr<loader::ObjectFile>> instructionImages;
  std::vector<std::string> runtimeImagePaths;
  std::vector<Addr> runtimeImageAddresses;
  const std::string memoryObservationPath;
  std::vector<Addr> memoryObservationAddresses;
  std::vector<std::uint64_t> memoryObservationSizes;
  std::vector<Target> targets;
  std::vector<ThreadContext *> contexts;
  std::vector<std::uint64_t> activeTargets;

  ThreadContext *contextForCpu(std::uint64_t cpuId) const;
  Addr symbolAddress(const loader::SymbolTable &symbols,
                     const std::string &name) const;
  Addr stackPointer(std::uint64_t cpuId) const;
};

} // namespace gem5

#endif // LOOM_RUNTIME_GEM5_LOOM_RISCV_DEPLOYMENT_WORKLOAD_HH
