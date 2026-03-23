#include "loom/SystemCompiler/DMAScheduler.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>

namespace loom {

namespace {

/// Construct a contract edge name from producer and consumer kernel names.
std::string makeEdgeName(const ContractSpec &contract) {
  return contract.producerKernel + " -> " + contract.consumerKernel;
}

/// Find the buffer allocation for a given contract edge name.
const BufferAllocation *
findAllocation(const BufferAllocationPlan &plan,
               const std::string &edgeName) {
  for (const auto &alloc : plan.allocations) {
    if (alloc.contractEdgeName == edgeName)
      return &alloc;
  }
  return nullptr;
}

/// Find the NoC route for a given contract edge name.
const NoCRoute *findRoute(const NoCSchedule &schedule,
                          const std::string &edgeName) {
  for (const auto &route : schedule.routes) {
    if (route.contractEdgeName == edgeName)
      return &route;
  }
  return nullptr;
}

} // namespace

//===----------------------------------------------------------------------===//
// DMAScheduler::schedule
//===----------------------------------------------------------------------===//

DMASchedule
DMAScheduler::schedule(const BufferAllocationPlan &bufferPlan,
                       const NoCSchedule &nocSchedule,
                       const std::vector<ContractSpec> &contracts,
                       const AssignmentResult &assignment,
                       const SystemArchitecture &arch,
                       const DMASchedulerOptions &opts) {
  DMASchedule result;
  result.tileComputeCycles = opts.estimatedComputeCycles;

  unsigned currentCycle = 0;
  unsigned totalDMACycles = 0;
  unsigned overlappedCycles = 0;

  if (opts.verbose) {
    llvm::errs() << "DMAScheduler: planning transfers for "
                 << contracts.size() << " contracts\n";
  }

  // Build DMA transfers for each cross-core contract.
  unsigned slotToggle = 0;

  for (const auto &contract : contracts) {
    auto prodIt = assignment.kernelToCore.find(contract.producerKernel);
    auto consIt = assignment.kernelToCore.find(contract.consumerKernel);
    if (prodIt == assignment.kernelToCore.end() ||
        consIt == assignment.kernelToCore.end())
      continue;

    unsigned prodCoreIdx = prodIt->second;
    unsigned consCoreIdx = consIt->second;

    // Skip intra-core transfers.
    if (prodCoreIdx == consCoreIdx)
      continue;

    std::string edgeName = makeEdgeName(contract);

    // Look up the buffer allocation.
    const BufferAllocation *alloc =
        findAllocation(bufferPlan, edgeName);
    if (!alloc)
      continue;

    // Look up the NoC route for timing.
    const NoCRoute *route = findRoute(nocSchedule, edgeName);

    // Compute transfer duration.
    unsigned durationCycles = 0;
    if (route) {
      durationCycles = route->transferLatencyCycles +
                       route->transferDurationCycles;
    } else {
      // Fallback: estimate from buffer size and NoC bandwidth.
      unsigned linkBW = arch.nocSpec.linkBandwidth;
      unsigned flitWidth = arch.nocSpec.flitWidth;
      uint64_t flits =
          (alloc->sizeBytes + flitWidth - 1) / flitWidth;
      durationCycles =
          linkBW > 0 ? static_cast<unsigned>((flits + linkBW - 1) / linkBW)
                     : static_cast<unsigned>(flits);
    }

    // Construct source and destination buffer descriptors.
    // Source is a producer-side buffer; destination is the allocated buffer.
    BufferAllocation srcBuf;
    srcBuf.contractEdgeName = edgeName;
    srcBuf.location = BufferAllocation::SPM_PRODUCER;
    srcBuf.offsetBytes = 0;
    srcBuf.sizeBytes = alloc->sizeBytes;
    srcBuf.elementCount = alloc->elementCount;
    srcBuf.coreInstanceIdx = prodCoreIdx;

    BufferAllocation dstBuf = *alloc;

    // Determine timing and double-buffering overlap.
    bool isDoubleBuffered = alloc->doubleBuffered;
    bool canOverlap = isDoubleBuffered && currentCycle > 0;

    DMATransfer xfer;
    xfer.contractEdgeName = edgeName;
    xfer.srcCore = contract.producerKernel;
    xfer.dstCore = contract.consumerKernel;
    xfer.srcCoreIdx = prodCoreIdx;
    xfer.dstCoreIdx = consCoreIdx;
    xfer.srcBuffer = srcBuf;
    xfer.dstBuffer = dstBuf;
    xfer.transferSizeBytes = alloc->sizeBytes;
    xfer.startCycle = currentCycle;
    xfer.durationCycles = durationCycles;
    xfer.endCycle = currentCycle + durationCycles;
    xfer.bufferSlot = isDoubleBuffered ? (slotToggle % 2) : 0;
    xfer.overlapWithCompute = canOverlap;

    result.transfers.push_back(xfer);

    if (canOverlap) {
      // Overlapped transfers run concurrently with compute.
      overlappedCycles += durationCycles;
    } else {
      // Non-overlapped transfers extend the tile time.
      currentCycle += durationCycles;
    }

    totalDMACycles += durationCycles;
    slotToggle++;

    if (opts.verbose) {
      llvm::errs() << "  DMA: " << edgeName << " | " << alloc->sizeBytes
                   << " bytes | " << durationCycles << " cycles"
                   << (canOverlap ? " (overlapped)" : "") << "\n";
    }
  }

  // Compute effective tile timing.
  unsigned nonOverlappedDMA = totalDMACycles - overlappedCycles;
  result.tileTransferCycles = nonOverlappedDMA;
  result.tileTotalCycles =
      opts.estimatedComputeCycles + nonOverlappedDMA;

  result.computeOverlapRatio =
      totalDMACycles > 0
          ? static_cast<double>(overlappedCycles) / totalDMACycles
          : 0.0;

  if (opts.verbose) {
    llvm::errs() << "DMAScheduler: " << result.transfers.size()
                 << " transfers, compute=" << opts.estimatedComputeCycles
                 << " dma_non_overlapped=" << nonOverlappedDMA
                 << " total_tile=" << result.tileTotalCycles
                 << " overlap_ratio=" << result.computeOverlapRatio
                 << "\n";
  }

  return result;
}

} // namespace loom
