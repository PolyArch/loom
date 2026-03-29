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

  unsigned totalDMACycles = 0;
  unsigned overlappedCycles = 0;

  if (opts.verbose) {
    llvm::errs() << "DMAScheduler: planning transfers for "
                 << contracts.size() << " contracts\n";
  }

  // Collect cross-core contract indices.
  struct XferEntry {
    size_t contractIdx;
    std::string edgeName;
  };
  std::vector<XferEntry> crossCoreEntries;

  for (size_t ci = 0; ci < contracts.size(); ++ci) {
    const auto &contract = contracts[ci];
    auto prodIt = assignment.kernelToCore.find(contract.producerKernel);
    auto consIt = assignment.kernelToCore.find(contract.consumerKernel);
    if (prodIt == assignment.kernelToCore.end() ||
        consIt == assignment.kernelToCore.end())
      continue;
    if (prodIt->second == consIt->second)
      continue;
    crossCoreEntries.push_back({ci, makeEdgeName(contract)});
  }

  // Build dependency graph among cross-core transfers.
  // If contract ci's consumer is contract cj's producer, then cj depends on ci.
  std::map<size_t, std::vector<size_t>> depGraph;
  std::map<size_t, unsigned> inDegree;
  for (size_t i = 0; i < crossCoreEntries.size(); ++i)
    inDegree[i] = 0;

  for (size_t i = 0; i < crossCoreEntries.size(); ++i) {
    const auto &ci = contracts[crossCoreEntries[i].contractIdx];
    for (size_t j = 0; j < crossCoreEntries.size(); ++j) {
      if (i == j)
        continue;
      const auto &cj = contracts[crossCoreEntries[j].contractIdx];
      if (ci.consumerKernel == cj.producerKernel) {
        depGraph[i].push_back(j);
        inDegree[j]++;
      }
    }
  }

  // Topological sort via Kahn's algorithm.
  std::vector<size_t> sortedOrder;
  std::vector<size_t> queue;
  for (const auto &entry : inDegree) {
    if (entry.second == 0)
      queue.push_back(entry.first);
  }
  while (!queue.empty()) {
    size_t cur = queue.back();
    queue.pop_back();
    sortedOrder.push_back(cur);
    for (size_t dep : depGraph[cur]) {
      inDegree[dep]--;
      if (inDegree[dep] == 0)
        queue.push_back(dep);
    }
  }

  // Append any remaining entries (cycles or missing).
  for (size_t i = 0; i < crossCoreEntries.size(); ++i) {
    bool found = false;
    for (size_t s : sortedOrder) {
      if (s == i) {
        found = true;
        break;
      }
    }
    if (!found)
      sortedOrder.push_back(i);
  }

  // Track when each kernel's data becomes available at the consumer.
  std::map<std::string, unsigned> kernelDataReady;

  unsigned slotToggle = 0;

  for (size_t si : sortedOrder) {
    const auto &entry = crossCoreEntries[si];
    const auto &contract = contracts[entry.contractIdx];

    unsigned prodCoreIdx =
        assignment.kernelToCore.find(contract.producerKernel)->second;
    unsigned consCoreIdx =
        assignment.kernelToCore.find(contract.consumerKernel)->second;

    const BufferAllocation *alloc =
        findAllocation(bufferPlan, entry.edgeName);
    if (!alloc)
      continue;

    const NoCRoute *route = findRoute(nocSchedule, entry.edgeName);

    unsigned durationCycles = 0;
    if (route) {
      durationCycles = route->transferLatencyCycles +
                       route->transferDurationCycles;
    } else {
      unsigned linkBW = arch.nocSpec.linkBandwidth;
      unsigned flitWidth = arch.nocSpec.flitWidth;
      uint64_t flits =
          (alloc->sizeBytes + flitWidth - 1) / flitWidth;
      durationCycles =
          linkBW > 0 ? static_cast<unsigned>((flits + linkBW - 1) / linkBW)
                     : static_cast<unsigned>(flits);
    }

    // Compute the earliest start time for this DMA transfer.
    // The producer must have completed its computation.
    unsigned producerCompletion =
        opts.kernelComputeCycles.count(contract.producerKernel)
            ? opts.kernelComputeCycles.at(contract.producerKernel)
            : opts.estimatedComputeCycles;

    // If the producer depends on earlier DMA transfers, account for arrival.
    unsigned producerDataReady =
        kernelDataReady.count(contract.producerKernel)
            ? kernelDataReady[contract.producerKernel]
            : 0;
    unsigned earliestStart = producerDataReady + producerCompletion;

    BufferAllocation srcBuf;
    srcBuf.contractEdgeName = entry.edgeName;
    srcBuf.location = BufferAllocation::SPM_PRODUCER;
    srcBuf.offsetBytes = 0;
    srcBuf.sizeBytes = alloc->sizeBytes;
    srcBuf.elementCount = alloc->elementCount;
    srcBuf.coreInstanceIdx = prodCoreIdx;

    BufferAllocation dstBuf = *alloc;

    bool isDoubleBuffered = alloc->doubleBuffered;
    bool canOverlap = isDoubleBuffered && earliestStart > 0;

    DMATransfer xfer;
    xfer.contractEdgeName = entry.edgeName;
    xfer.srcCore = contract.producerKernel;
    xfer.dstCore = contract.consumerKernel;
    xfer.srcCoreIdx = prodCoreIdx;
    xfer.dstCoreIdx = consCoreIdx;
    xfer.srcBuffer = srcBuf;
    xfer.dstBuffer = dstBuf;
    xfer.transferSizeBytes = alloc->sizeBytes;
    xfer.startCycle = earliestStart;
    xfer.durationCycles = durationCycles;
    xfer.endCycle = earliestStart + durationCycles;
    xfer.bufferSlot = isDoubleBuffered ? (slotToggle % 2) : 0;
    xfer.overlapWithCompute = canOverlap;

    result.transfers.push_back(xfer);

    // Update when the consumer's data becomes available.
    unsigned xferEnd = earliestStart + durationCycles;
    if (kernelDataReady.count(contract.consumerKernel) == 0 ||
        kernelDataReady[contract.consumerKernel] < xferEnd) {
      kernelDataReady[contract.consumerKernel] = xferEnd;
    }

    if (canOverlap) {
      overlappedCycles += durationCycles;
    }

    totalDMACycles += durationCycles;
    slotToggle++;

    if (opts.verbose) {
      llvm::errs() << "  DMA: " << entry.edgeName << " | "
                   << alloc->sizeBytes << " bytes | " << durationCycles
                   << " cycles | start=" << earliestStart
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
