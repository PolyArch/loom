#include "loom/SystemCompiler/BufferAllocator.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <numeric>

namespace loom {

namespace {

/// Construct a contract edge name from producer and consumer kernel names.
std::string makeEdgeName(const ContractSpec &contract) {
  return contract.producerKernel + " -> " + contract.consumerKernel;
}

/// Simple bump allocator state for a single core's SPM.
struct SPMBumpAllocator {
  uint64_t capacity = 0;
  uint64_t used = 0;

  bool canAllocate(uint64_t bytes) const { return used + bytes <= capacity; }

  uint64_t allocate(uint64_t bytes) {
    uint64_t offset = used;
    used += bytes;
    return offset;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// BufferAllocator::allocate
//===----------------------------------------------------------------------===//

BufferAllocationPlan
BufferAllocator::allocate(const AssignmentResult &assignment,
                          const std::vector<ContractSpec> &contracts,
                          const NoCSchedule &nocSchedule,
                          const SystemArchitecture &arch,
                          const BufferAllocatorOptions &opts) {
  BufferAllocationPlan plan;
  plan.l2TotalBytes = arch.sharedMemSpec.l2SizeBytes;

  unsigned totalCores = arch.totalCoreInstances();

  // Initialize per-core SPM bump allocators.
  std::map<unsigned, SPMBumpAllocator> spmAllocators;
  for (unsigned ci = 0; ci < totalCores; ++ci) {
    const auto &coreType = arch.typeForInstance(ci);
    uint64_t available = static_cast<uint64_t>(
        static_cast<double>(coreType.spmBytes) *
        (1.0 - opts.spmReserveFraction));
    spmAllocators[ci].capacity = available;
  }

  // L2 bump allocator state.
  uint64_t l2Capacity = arch.sharedMemSpec.l2SizeBytes;
  uint64_t l2Used = 0;
  unsigned nextL2Bank = 0;

  // DRAM virtual address allocator.
  uint64_t dramNextAddr = 0x80000000ULL; // arbitrary DRAM base

  // Collect cross-core contract indices and sort by data volume (largest first).
  struct ContractEntry {
    size_t idx;
    uint64_t volumeBytes;
  };
  std::vector<ContractEntry> crossCoreContracts;

  for (size_t ci = 0; ci < contracts.size(); ++ci) {
    const auto &contract = contracts[ci];
    auto prodIt = assignment.kernelToCore.find(contract.producerKernel);
    auto consIt = assignment.kernelToCore.find(contract.consumerKernel);
    if (prodIt == assignment.kernelToCore.end() ||
        consIt == assignment.kernelToCore.end())
      continue;

    // Only allocate buffers for cross-core edges.
    if (prodIt->second == consIt->second)
      continue;

    unsigned elemSize = estimateElementSize(contract.dataTypeName);
    int64_t bufElems = contract.minBufferElements;
    if (bufElems <= 0)
      bufElems = 1;

    uint64_t volume = static_cast<uint64_t>(bufElems) * elemSize;
    crossCoreContracts.push_back({ci, volume});
  }

  std::sort(crossCoreContracts.begin(), crossCoreContracts.end(),
            [](const ContractEntry &a, const ContractEntry &b) {
              return a.volumeBytes > b.volumeBytes;
            });

  if (opts.verbose) {
    llvm::errs() << "BufferAllocator: " << crossCoreContracts.size()
                 << " cross-core contracts to allocate\n";
  }

  // Allocate buffers using SPM-first policy.
  for (const auto &entry : crossCoreContracts) {
    const auto &contract = contracts[entry.idx];
    unsigned producerIdx =
        assignment.kernelToCore.find(contract.producerKernel)->second;
    unsigned consumerIdx =
        assignment.kernelToCore.find(contract.consumerKernel)->second;

    unsigned elemSize = estimateElementSize(contract.dataTypeName);
    int64_t bufElems = contract.minBufferElements;
    if (bufElems <= 0)
      bufElems = 1;

    bool useDoubleBuffer =
        opts.preferDoubleBuffering && contract.doubleBuffering;
    unsigned elementCount = static_cast<unsigned>(bufElems);
    if (useDoubleBuffer)
      elementCount *= 2;

    uint64_t bufferBytes = static_cast<uint64_t>(elementCount) * elemSize;

    std::string edgeName = makeEdgeName(contract);
    bool allocated = false;

    // Attempt 1: consumer SPM (preferred for streaming patterns).
    if (contract.visibility == Visibility::LOCAL_SPM ||
        contract.visibility == Visibility::SHARED_L2) {
      auto &consAlloc = spmAllocators[consumerIdx];
      if (consAlloc.canAllocate(bufferBytes)) {
        uint64_t offset = consAlloc.allocate(bufferBytes);

        BufferAllocation alloc;
        alloc.contractEdgeName = edgeName;
        alloc.location = BufferAllocation::SPM_CONSUMER;
        alloc.offsetBytes = offset;
        alloc.sizeBytes = bufferBytes;
        alloc.elementCount = elementCount;
        alloc.doubleBuffered = useDoubleBuffer;
        alloc.coreInstanceIdx = consumerIdx;
        plan.allocations.push_back(alloc);
        allocated = true;

        if (opts.verbose) {
          llvm::errs() << "  " << edgeName << ": SPM_CONSUMER (core "
                       << consumerIdx << ") " << bufferBytes << " bytes\n";
        }
      }
    }

    // Attempt 2: producer SPM (if consumer SPM is full).
    if (!allocated && (contract.visibility == Visibility::LOCAL_SPM ||
                       contract.visibility == Visibility::SHARED_L2)) {
      auto &prodAlloc = spmAllocators[producerIdx];
      if (prodAlloc.canAllocate(bufferBytes)) {
        uint64_t offset = prodAlloc.allocate(bufferBytes);

        BufferAllocation alloc;
        alloc.contractEdgeName = edgeName;
        alloc.location = BufferAllocation::SPM_PRODUCER;
        alloc.offsetBytes = offset;
        alloc.sizeBytes = bufferBytes;
        alloc.elementCount = elementCount;
        alloc.doubleBuffered = useDoubleBuffer;
        alloc.coreInstanceIdx = producerIdx;
        plan.allocations.push_back(alloc);
        allocated = true;

        if (opts.verbose) {
          llvm::errs() << "  " << edgeName << ": SPM_PRODUCER (core "
                       << producerIdx << ") " << bufferBytes << " bytes\n";
        }
      }
    }

    // Attempt 3: shared L2.
    if (!allocated && contract.visibility != Placement::EXTERNAL) {
      if (l2Used + bufferBytes <= l2Capacity) {
        unsigned bankIdx = nextL2Bank;
        nextL2Bank =
            (nextL2Bank + 1) % arch.sharedMemSpec.numBanks;

        BufferAllocation alloc;
        alloc.contractEdgeName = edgeName;
        alloc.location = BufferAllocation::SHARED_L2;
        alloc.offsetBytes = l2Used;
        alloc.sizeBytes = bufferBytes;
        alloc.elementCount = elementCount;
        alloc.doubleBuffered = useDoubleBuffer;
        alloc.l2BankIdx = bankIdx;
        plan.allocations.push_back(alloc);
        l2Used += bufferBytes;
        allocated = true;

        if (opts.verbose) {
          llvm::errs() << "  " << edgeName << ": SHARED_L2 (bank "
                       << bankIdx << ") " << bufferBytes << " bytes\n";
        }
      }
    }

    // Attempt 4: external DRAM fallback (always available).
    if (!allocated) {
      BufferAllocation alloc;
      alloc.contractEdgeName = edgeName;
      alloc.location = BufferAllocation::EXTERNAL_DRAM;
      alloc.offsetBytes = 0;
      alloc.sizeBytes = bufferBytes;
      alloc.elementCount = elementCount;
      alloc.doubleBuffered = useDoubleBuffer;
      alloc.dramBaseAddr = dramNextAddr;
      plan.allocations.push_back(alloc);
      dramNextAddr += bufferBytes;

      if (opts.verbose) {
        llvm::errs() << "  " << edgeName << ": EXTERNAL_DRAM "
                     << bufferBytes << " bytes\n";
      }
    }
  }

  // Build per-core SPM usage summary.
  for (unsigned ci = 0; ci < totalCores; ++ci) {
    const auto &allocator = spmAllocators[ci];
    const auto &coreType = arch.typeForInstance(ci);

    CoreSPMUsage usage;
    usage.coreName = coreType.typeName + "_" + std::to_string(ci);
    usage.coreInstanceIdx = ci;
    usage.usedBytes = allocator.used;
    usage.totalBytes = coreType.spmBytes;
    usage.utilization =
        coreType.spmBytes > 0
            ? static_cast<double>(allocator.used) / coreType.spmBytes
            : 0.0;
    plan.coreSPMUsage.push_back(usage);
  }

  plan.l2UsedBytes = l2Used;
  plan.feasible = true; // DRAM fallback ensures feasibility.

  if (opts.verbose) {
    llvm::errs() << "BufferAllocator: " << plan.allocations.size()
                 << " allocations, L2 used=" << l2Used << "/"
                 << l2Capacity << " bytes\n";
  }

  return plan;
}

} // namespace loom
