#ifndef LOOM_MULTICORESIM_MULTICORESIMSESSION_H
#define LOOM_MULTICORESIM_MULTICORESIMSESSION_H

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace mcsim {

// Describes one kernel assigned to a core.
struct KernelDescriptor {
  std::string name;
  unsigned coreId = 0;
  uint64_t estimatedCycles = 0;

  // Cycle offset from kernel start at which output data becomes available
  // for NoC injection. Zero means data is ready only after the kernel
  // finishes (no overlap).
  uint64_t outputReadyCycleOffset = 0;

  // Bytes produced by this kernel that need to be sent over the NoC.
  uint64_t outputBytes = 0;
};

// Describes a NoC transfer between cores.
struct NocTransferDescriptor {
  unsigned srcCoreId = 0;
  unsigned dstCoreId = 0;
  uint64_t bytes = 0;

  // Index of the source kernel in the per-core kernel list that produces
  // the data for this transfer.
  unsigned srcKernelIndex = 0;
};

// Per-kernel simulation result.
struct KernelResult {
  std::string name;
  unsigned coreId = 0;
  uint64_t startCycle = 0;
  uint64_t endCycle = 0;
  uint64_t cycles = 0;
};

// Per-transfer simulation result.
struct NocTransferResult {
  unsigned srcCoreId = 0;
  unsigned dstCoreId = 0;
  uint64_t injectionStartCycle = 0;
  uint64_t injectionEndCycle = 0;
  uint64_t bytes = 0;
};

// Aggregate result of a multi-core simulation.
struct MultiCoreSimResult {
  bool success = false;
  std::string errorMessage;
  uint64_t totalCycles = 0;
  std::vector<KernelResult> kernelResults;
  std::vector<NocTransferResult> nocTransferResults;
};

// Configuration for the multi-core simulator.
struct MultiCoreSimConfig {
  // NoC bandwidth in bytes per cycle.
  double nocBandwidthBytesPerCycle = 8.0;

  // NoC injection startup latency in cycles.
  uint64_t nocStartupLatencyCycles = 4;

  // Maximum number of cores.
  unsigned maxCores = 64;
};

// Simulates execution of multiple kernels across cores with interleaved
// NoC data transfers. Supports multiple kernels per core (executed
// sequentially) and overlapped NoC injection (transfer starts while the
// producing kernel is still running, once output data is ready).
class MultiCoreSimSession {
public:
  explicit MultiCoreSimSession(const MultiCoreSimConfig &config = {});

  // Add a kernel to be executed on a specific core.
  void addKernel(const KernelDescriptor &kernel);

  // Add a NoC transfer dependency.
  void addNocTransfer(const NocTransferDescriptor &transfer);

  // Run the simulation and return the result.
  MultiCoreSimResult run();

private:
  // Determine the cycle at which a core can start its next kernel, given
  // dependencies from NoC transfers.
  uint64_t computeKernelStartCycle(unsigned coreId,
                                   unsigned kernelIndex) const;

  MultiCoreSimConfig config_;

  // Per-core list of kernels in execution order.
  // coreKernels_[coreId] = list of kernels for that core.
  std::vector<std::vector<KernelDescriptor>> coreKernels_;

  std::vector<NocTransferDescriptor> nocTransfers_;

  // Populated during run().
  std::vector<KernelResult> kernelResults_;
  std::vector<NocTransferResult> nocTransferResults_;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_MULTICORESIMSESSION_H
