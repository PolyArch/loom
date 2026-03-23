#include "loom/MultiCoreSim/MultiCoreSimSession.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace loom {
namespace mcsim {

MultiCoreSimSession::MultiCoreSimSession(const MultiCoreSimConfig &config)
    : config_(config) {}

void MultiCoreSimSession::addKernel(const KernelDescriptor &kernel) {
  unsigned coreId = kernel.coreId;
  if (coreId >= coreKernels_.size())
    coreKernels_.resize(coreId + 1);
  coreKernels_[coreId].push_back(kernel);
}

void MultiCoreSimSession::addNocTransfer(const NocTransferDescriptor &transfer) {
  nocTransfers_.push_back(transfer);
}

uint64_t
MultiCoreSimSession::computeKernelStartCycle(unsigned coreId,
                                             unsigned kernelIndex) const {
  uint64_t earliest = 0;

  // A kernel must wait for all NoC transfers targeted at this core whose
  // destination is this kernel (or any earlier kernel, since kernels on a
  // core execute sequentially -- the core is not free until the prior
  // transfer is consumed).
  for (const auto &tr : nocTransferResults_) {
    if (tr.dstCoreId == coreId) {
      // The transfer must complete before this kernel can start.
      earliest = std::max(earliest, tr.injectionEndCycle);
    }
  }

  return earliest;
}

MultiCoreSimResult MultiCoreSimSession::run() {
  MultiCoreSimResult result;
  kernelResults_.clear();
  nocTransferResults_.clear();

  // Validate core IDs.
  for (size_t ci = 0; ci < coreKernels_.size(); ++ci) {
    if (ci >= config_.maxCores) {
      result.success = false;
      result.errorMessage = "core ID exceeds maxCores";
      return result;
    }
  }

  // Track the current cycle frontier for each core (when it becomes free).
  std::vector<uint64_t> coreFrontier(coreKernels_.size(), 0);

  // Process kernels per core in order. For each kernel, compute when it
  // can start (considering both the core frontier and incoming NoC
  // transfers), then simulate NoC injection for any transfers sourced
  // from this kernel with interleaved overlap.
  //
  // We iterate in a global ordering: process cores in round-robin over
  // their kernel indices to allow interleaving.

  // Determine the maximum number of kernels on any single core.
  unsigned maxKernelsPerCore = 0;
  for (const auto &kl : coreKernels_)
    maxKernelsPerCore = std::max(maxKernelsPerCore,
                                 static_cast<unsigned>(kl.size()));

  for (unsigned ki = 0; ki < maxKernelsPerCore; ++ki) {
    for (unsigned ci = 0; ci < coreKernels_.size(); ++ci) {
      if (ki >= coreKernels_[ci].size())
        continue;

      const KernelDescriptor &kd = coreKernels_[ci][ki];

      // Start cycle: max of core frontier and all incoming transfer
      // completions for this core.
      uint64_t startCycle = coreFrontier[ci];
      startCycle = std::max(startCycle, computeKernelStartCycle(ci, ki));

      uint64_t endCycle = startCycle + kd.estimatedCycles;

      KernelResult kr;
      kr.name = kd.name;
      kr.coreId = ci;
      kr.startCycle = startCycle;
      kr.endCycle = endCycle;
      kr.cycles = kd.estimatedCycles;
      kernelResults_.push_back(kr);

      // Update core frontier to reflect kernel completion.
      coreFrontier[ci] = endCycle;

      // Process NoC transfers sourced from this kernel. The injection
      // can start as soon as the output data is ready, which is
      // startCycle + outputReadyCycleOffset (interleaved with
      // execution). If outputReadyCycleOffset is zero, injection starts
      // at endCycle (no overlap).
      for (const auto &td : nocTransfers_) {
        if (td.srcCoreId != ci || td.srcKernelIndex != ki)
          continue;

        uint64_t injectionStart;
        if (kd.outputReadyCycleOffset > 0 &&
            kd.outputReadyCycleOffset < kd.estimatedCycles) {
          // Interleaved: output ready mid-execution.
          injectionStart = startCycle + kd.outputReadyCycleOffset;
        } else {
          // Non-interleaved: data ready only after kernel finishes.
          injectionStart = endCycle;
        }

        injectionStart += config_.nocStartupLatencyCycles;

        // Transfer duration based on bytes and NoC bandwidth.
        uint64_t transferCycles = 0;
        if (config_.nocBandwidthBytesPerCycle > 0.0) {
          transferCycles = static_cast<uint64_t>(std::ceil(
              static_cast<double>(td.bytes) /
              config_.nocBandwidthBytesPerCycle));
        }

        uint64_t injectionEnd = injectionStart + transferCycles;

        NocTransferResult tr;
        tr.srcCoreId = td.srcCoreId;
        tr.dstCoreId = td.dstCoreId;
        tr.injectionStartCycle = injectionStart;
        tr.injectionEndCycle = injectionEnd;
        tr.bytes = td.bytes;
        nocTransferResults_.push_back(tr);
      }
    }
  }

  // Total cycles = max of all core frontiers and all transfer completions.
  uint64_t totalCycles = 0;
  for (uint64_t f : coreFrontier)
    totalCycles = std::max(totalCycles, f);
  for (const auto &tr : nocTransferResults_)
    totalCycles = std::max(totalCycles, tr.injectionEndCycle);

  result.success = true;
  result.totalCycles = totalCycles;
  result.kernelResults = kernelResults_;
  result.nocTransferResults = nocTransferResults_;
  return result;
}

} // namespace mcsim
} // namespace loom
