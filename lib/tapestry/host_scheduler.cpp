//===-- host_scheduler.cpp - Multi-kernel host code generation -----*- C++ -*-===//
//
// Generates host-side scheduling code for multi-kernel TDG execution.
// The scheduler performs a topological sort of the kernel DAG, plans DMA
// transfers at host/CGRA boundaries, inserts synchronization barriers,
// and generates a complete C host driver file.
//
//===----------------------------------------------------------------------===//

#include "tapestry/host_scheduler.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

using namespace tapestry;
using namespace loom::tapestry;

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Estimate data type size in bytes from a type name string.
static unsigned estimateDataTypeSize(const std::string &typeName) {
  if (typeName == "f64" || typeName == "i64" || typeName == "double")
    return 8;
  if (typeName == "f32" || typeName == "i32" || typeName == "float" ||
      typeName == "int")
    return 4;
  if (typeName == "f16" || typeName == "i16" || typeName == "short")
    return 2;
  if (typeName == "i8" || typeName == "char" || typeName == "byte")
    return 1;
  return 4; // Default: 4 bytes
}

//===----------------------------------------------------------------------===//
// HostScheduler Construction
//===----------------------------------------------------------------------===//

HostScheduler::HostScheduler(const HostScheduleOptions &options)
    : options_(options) {}

//===----------------------------------------------------------------------===//
// Topological Sort
//===----------------------------------------------------------------------===//

std::vector<std::string> HostScheduler::topologicalSort(
    const std::vector<KernelDesc> &kernels,
    const std::vector<ContractSpec> &contracts) {

  // Build adjacency list and in-degree map.
  std::unordered_map<std::string, std::vector<std::string>> adj;
  std::unordered_map<std::string, int> inDegree;

  for (const auto &k : kernels) {
    adj[k.name]; // Ensure all kernels appear.
    if (inDegree.find(k.name) == inDegree.end())
      inDegree[k.name] = 0;
  }

  for (const auto &c : contracts) {
    adj[c.producerKernel].push_back(c.consumerKernel);
    inDegree[c.consumerKernel]++;
    // Ensure producer also has an entry.
    if (inDegree.find(c.producerKernel) == inDegree.end())
      inDegree[c.producerKernel] = 0;
  }

  // Kahn's algorithm.
  std::queue<std::string> zeroIn;
  for (const auto &[name, deg] : inDegree) {
    if (deg == 0)
      zeroIn.push(name);
  }

  std::vector<std::string> sorted;
  while (!zeroIn.empty()) {
    std::string node = zeroIn.front();
    zeroIn.pop();
    sorted.push_back(node);

    for (const auto &neighbor : adj[node]) {
      inDegree[neighbor]--;
      if (inDegree[neighbor] == 0)
        zeroIn.push(neighbor);
    }
  }

  // If sorted.size() != kernels.size(), there is a cycle. Return what we have.
  return sorted;
}

//===----------------------------------------------------------------------===//
// DMA Transfer Planning
//===----------------------------------------------------------------------===//

std::vector<HostDMATransfer> HostScheduler::planDMATransfers(
    const std::vector<ScheduledKernel> &orderedKernels,
    const std::vector<ContractSpec> &contracts) {

  std::vector<HostDMATransfer> transfers;

  // Build a lookup from kernel name to its scheduled info.
  std::unordered_map<std::string, const ScheduledKernel *> kernelMap;
  for (const auto &sk : orderedKernels)
    kernelMap[sk.name] = &sk;

  for (const auto &contract : contracts) {
    auto producerIt = kernelMap.find(contract.producerKernel);
    auto consumerIt = kernelMap.find(contract.consumerKernel);

    if (producerIt == kernelMap.end() || consumerIt == kernelMap.end())
      continue;

    const auto *producer = producerIt->second;
    const auto *consumer = consumerIt->second;

    // Determine if a DMA transfer is needed.
    HostDMATransfer transfer;
    transfer.producerKernel = contract.producerKernel;
    transfer.consumerKernel = contract.consumerKernel;
    transfer.transferSizeBytes = contract.elementCount *
                                  estimateDataTypeSize(contract.dataType);

    bool needsDMA = false;

    if (producer->target == ExecutionTarget::CGRA &&
        consumer->target == ExecutionTarget::HOST) {
      transfer.direction = HostDMATransfer::CGRA_TO_HOST;
      transfer.srcCoreIdx = producer->coreInstanceIdx;
      transfer.srcAddrSymbol =
          options_.runtimePrefix + "spm_addr(" +
          std::to_string(producer->coreInstanceIdx) + ", \"" +
          contract.producerKernel + "_out\")";
      transfer.dstAddrSymbol =
          "host_buf_" + contract.consumerKernel + "_in";
      needsDMA = true;
    } else if (producer->target == ExecutionTarget::HOST &&
               consumer->target == ExecutionTarget::CGRA) {
      transfer.direction = HostDMATransfer::HOST_TO_CGRA;
      transfer.dstCoreIdx = consumer->coreInstanceIdx;
      transfer.srcAddrSymbol =
          "host_buf_" + contract.producerKernel + "_out";
      transfer.dstAddrSymbol =
          options_.runtimePrefix + "spm_addr(" +
          std::to_string(consumer->coreInstanceIdx) + ", \"" +
          contract.consumerKernel + "_in\")";
      needsDMA = true;
    } else if (producer->target == ExecutionTarget::CGRA &&
               consumer->target == ExecutionTarget::CGRA &&
               producer->coreInstanceIdx != consumer->coreInstanceIdx) {
      // CGRA-to-CGRA on different cores: handled by NoC, but may need
      // explicit DMA for SPM-to-SPM if cores are not directly connected.
      transfer.direction = HostDMATransfer::CGRA_TO_CGRA;
      transfer.srcCoreIdx = producer->coreInstanceIdx;
      transfer.dstCoreIdx = consumer->coreInstanceIdx;
      transfer.srcAddrSymbol =
          options_.runtimePrefix + "spm_addr(" +
          std::to_string(producer->coreInstanceIdx) + ", \"" +
          contract.producerKernel + "_out\")";
      transfer.dstAddrSymbol =
          options_.runtimePrefix + "spm_addr(" +
          std::to_string(consumer->coreInstanceIdx) + ", \"" +
          contract.consumerKernel + "_in\")";
      needsDMA = true;
    }

    if (needsDMA)
      transfers.push_back(transfer);
  }

  return transfers;
}

//===----------------------------------------------------------------------===//
// Barrier Planning
//===----------------------------------------------------------------------===//

std::vector<SyncBarrier> HostScheduler::planBarriers(
    const std::vector<ScheduledKernel> &orderedKernels) {

  std::vector<SyncBarrier> barriers;

  // Insert a barrier after each contiguous group of CGRA kernels,
  // before a HOST kernel or at the end of the schedule.
  std::unordered_set<unsigned> activeCores;
  unsigned barrierIdx = 0;

  for (size_t i = 0; i < orderedKernels.size(); ++i) {
    const auto &sk = orderedKernels[i];

    if (sk.target == ExecutionTarget::CGRA && sk.coreInstanceIdx >= 0) {
      activeCores.insert(static_cast<unsigned>(sk.coreInstanceIdx));
    }

    // Check if we need a barrier: this is a HOST kernel after CGRA,
    // or the last kernel.
    bool needsBarrier = false;
    if (!activeCores.empty()) {
      if (sk.target == ExecutionTarget::HOST) {
        needsBarrier = true;
      } else if (i == orderedKernels.size() - 1) {
        needsBarrier = true;
      } else if (i + 1 < orderedKernels.size() &&
                 orderedKernels[i + 1].target == ExecutionTarget::HOST) {
        needsBarrier = true;
      }
    }

    if (needsBarrier) {
      SyncBarrier barrier;
      barrier.label = "barrier_" + std::to_string(barrierIdx++);
      barrier.waitCoreIds.assign(activeCores.begin(), activeCores.end());
      std::sort(barrier.waitCoreIds.begin(), barrier.waitCoreIds.end());
      barriers.push_back(barrier);
      activeCores.clear();
    }
  }

  return barriers;
}

//===----------------------------------------------------------------------===//
// Build Schedule
//===----------------------------------------------------------------------===//

HostSchedule HostScheduler::buildSchedule(
    const std::vector<KernelDesc> &kernels,
    const std::vector<ContractSpec> &contracts,
    const BendersResult &assignments,
    const SystemArchitecture &arch) {

  HostSchedule schedule;
  schedule.pipelineName = arch.name;

  // Build a lookup from kernel name to its BendersResult assignment.
  std::unordered_map<std::string, const L2Assignment *> assignMap;
  for (const auto &a : assignments.assignments)
    assignMap[a.kernelName] = &a;

  // Topological sort.
  auto sortedNames = topologicalSort(kernels, contracts);

  // Build scheduled kernel entries.
  for (const auto &name : sortedNames) {
    ScheduledKernel sk;
    sk.name = name;

    auto assignIt = assignMap.find(name);
    if (assignIt != assignMap.end() && assignIt->second->mappingSuccess) {
      sk.target = ExecutionTarget::CGRA;
      sk.coreInstanceIdx = assignIt->second->coreInstanceIndex;
      if (assignIt->second->coreTypeIndex >= 0 &&
          static_cast<size_t>(assignIt->second->coreTypeIndex) <
              arch.coreTypes.size()) {
        sk.coreTypeName =
            arch.coreTypes[static_cast<size_t>(
                               assignIt->second->coreTypeIndex)]
                .name;
      }
    } else {
      // Kernel failed mapping or was not assigned: run on host.
      sk.target = ExecutionTarget::HOST;
      sk.hostFunctionName = name;
    }

    schedule.executionOrder.push_back(sk);
  }

  // Plan DMA transfers and barriers.
  schedule.dmaTransfers = planDMATransfers(schedule.executionOrder, contracts);
  schedule.barriers = planBarriers(schedule.executionOrder);

  return schedule;
}

//===----------------------------------------------------------------------===//
// Code Generation Helpers
//===----------------------------------------------------------------------===//

void HostScheduler::emitKernelAction(std::string &code,
                                     const ScheduledKernel &kernel) {
  const std::string &prefix = options_.runtimePrefix;

  if (kernel.target == ExecutionTarget::CGRA) {
    code += "    /* CGRA kernel: " + kernel.name;
    if (!kernel.coreTypeName.empty())
      code += " on " + kernel.coreTypeName;
    code += " (core " + std::to_string(kernel.coreInstanceIdx) + ") */\n";

    // Load configuration.
    code += "    " + prefix + "load_config(" +
            std::to_string(kernel.coreInstanceIdx) + ", config_" +
            kernel.name + ", sizeof(config_" + kernel.name + "));\n";

    // Launch.
    code += "    " + prefix + "launch(" +
            std::to_string(kernel.coreInstanceIdx) + ");\n";

    if (options_.emitTimingComments && kernel.estimatedCycles > 0) {
      code += "    /* estimated cycles: " +
              std::to_string(kernel.estimatedCycles) + " */\n";
    }
  } else {
    // Host kernel: direct function call.
    code += "    /* Host kernel: " + kernel.name + " */\n";
    code += "    " + kernel.hostFunctionName + "();\n";
  }
}

void HostScheduler::emitDMATransfer(std::string &code,
                                    const HostDMATransfer &transfer) {
  const std::string &prefix = options_.runtimePrefix;

  code += "    /* DMA: " + transfer.producerKernel + " -> " +
          transfer.consumerKernel + " (" +
          std::to_string(transfer.transferSizeBytes) + " bytes) */\n";

  code += "    " + prefix + "dma_transfer(\n";
  code += "        " + transfer.srcAddrSymbol + ",\n";
  code += "        " + transfer.dstAddrSymbol + ",\n";
  code += "        " + std::to_string(transfer.transferSizeBytes) + ");\n";

  if (options_.emitErrorChecks) {
    code += "    if (" + prefix + "dma_status() != 0) {\n";
    code += "        " + prefix + "log_error(\"DMA transfer failed: " +
            transfer.producerKernel + " -> " +
            transfer.consumerKernel + "\");\n";
    code += "        return -1;\n";
    code += "    }\n";
  }
}

void HostScheduler::emitBarrier(std::string &code,
                                const SyncBarrier &barrier) {
  const std::string &prefix = options_.runtimePrefix;

  code += "    /* Synchronization: " + barrier.label + " */\n";
  for (unsigned coreId : barrier.waitCoreIds) {
    code += "    " + prefix + "wait_completion(" +
            std::to_string(coreId) + ");\n";
  }
}

//===----------------------------------------------------------------------===//
// Host Code Generation
//===----------------------------------------------------------------------===//

bool HostScheduler::generateHostCode(const HostSchedule &schedule,
                                     const std::string &outputPath) {
  std::string code;
  const std::string &prefix = options_.runtimePrefix;
  std::string guardName = schedule.pipelineName;
  // Sanitize pipeline name for C identifier.
  for (char &ch : guardName) {
    if (!std::isalnum(static_cast<unsigned char>(ch)))
      ch = '_';
  }

  // File header.
  code += "/*\n";
  code += " * Auto-generated host driver for pipeline: " +
          schedule.pipelineName + "\n";
  code += " *\n";
  code += " * Kernel execution order determined by topological sort of the\n";
  code += " * task dataflow graph. DMA transfers inserted at host/CGRA\n";
  code += " * boundaries.\n";
  code += " */\n\n";

  code += "#include \"" + guardName + "_host_driver.h\"\n";
  code += "#include <stdio.h>\n";
  code += "#include <stdlib.h>\n";
  code += "#include <stdint.h>\n\n";

  // Forward declarations for host kernels.
  bool hasHostKernels = false;
  for (const auto &sk : schedule.executionOrder) {
    if (sk.target == ExecutionTarget::HOST) {
      code += "extern void " + sk.hostFunctionName + "(void);\n";
      hasHostKernels = true;
    }
  }
  if (hasHostKernels)
    code += "\n";

  // Configuration blob externs.
  for (const auto &sk : schedule.executionOrder) {
    if (sk.target == ExecutionTarget::CGRA) {
      code += "extern const uint8_t config_" + sk.name + "[];\n";
      code += "extern const size_t sizeof_config_" + sk.name + ";\n";
    }
  }
  code += "\n";

  // Main driver function.
  code += "int " + guardName + "_run(void) {\n";

  // Track which barrier index we are at.
  size_t barrierIdx = 0;
  size_t dmaIdx = 0;

  for (const auto &sk : schedule.executionOrder) {
    code += "\n";
    emitKernelAction(code, sk);

    // Check if a barrier should follow this kernel.
    if (barrierIdx < schedule.barriers.size()) {
      // Simple heuristic: emit barriers between kernel groups.
      // Check if the next kernel is HOST or we reached the barrier point.
      bool emitNow = false;

      // Check if any barrier's cores include this kernel's core.
      const auto &barrier = schedule.barriers[barrierIdx];
      if (sk.target == ExecutionTarget::CGRA) {
        for (unsigned cid : barrier.waitCoreIds) {
          if (cid == static_cast<unsigned>(sk.coreInstanceIdx)) {
            emitNow = true;
            break;
          }
        }
      }

      if (emitNow) {
        // Check if next kernel needs this barrier.
        code += "\n";
        emitBarrier(code, barrier);
        barrierIdx++;

        // Emit any DMA transfers associated with this barrier point.
        while (dmaIdx < schedule.dmaTransfers.size()) {
          // Check if this DMA's producer matches a recently completed kernel.
          const auto &dma = schedule.dmaTransfers[dmaIdx];
          code += "\n";
          emitDMATransfer(code, dma);
          dmaIdx++;

          // Simple heuristic: emit one DMA per barrier for now.
          // In a production system, this would match DMA to the
          // specific contract edge crossing the barrier.
          break;
        }
      }
    }
  }

  // Emit any remaining barriers and DMAs.
  while (barrierIdx < schedule.barriers.size()) {
    code += "\n";
    emitBarrier(code, schedule.barriers[barrierIdx]);
    barrierIdx++;
  }
  while (dmaIdx < schedule.dmaTransfers.size()) {
    code += "\n";
    emitDMATransfer(code, schedule.dmaTransfers[dmaIdx]);
    dmaIdx++;
  }

  code += "\n    return 0;\n";
  code += "}\n";

  // Write to file.
  std::error_code ec;
  llvm::raw_fd_ostream os(outputPath, ec);
  if (ec) {
    llvm::errs() << "HostScheduler: cannot write to " << outputPath << ": "
                  << ec.message() << "\n";
    return false;
  }
  os << code;
  return true;
}

//===----------------------------------------------------------------------===//
// Host Header Generation
//===----------------------------------------------------------------------===//

bool HostScheduler::generateHostHeader(const HostSchedule &schedule,
                                       const std::string &outputPath) {
  std::string code;

  std::string guardName = schedule.pipelineName;
  for (char &ch : guardName) {
    if (!std::isalnum(static_cast<unsigned char>(ch)))
      ch = '_';
  }
  std::string guardMacro = guardName;
  std::transform(guardMacro.begin(), guardMacro.end(), guardMacro.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  guardMacro += "_HOST_DRIVER_H";

  code += "/*\n";
  code += " * Auto-generated host driver header for pipeline: " +
          schedule.pipelineName + "\n";
  code += " */\n\n";

  code += "#ifndef " + guardMacro + "\n";
  code += "#define " + guardMacro + "\n\n";

  code += "#include <stdint.h>\n";
  code += "#include <stddef.h>\n\n";

  code += "#ifdef __cplusplus\n";
  code += "extern \"C\" {\n";
  code += "#endif\n\n";

  // Runtime API declarations.
  const std::string &prefix = options_.runtimePrefix;
  code += "/* Runtime API */\n";
  code += "int " + prefix + "load_config(unsigned core_id, "
          "const uint8_t *config, size_t size);\n";
  code += "int " + prefix + "launch(unsigned core_id);\n";
  code += "int " + prefix + "wait_completion(unsigned core_id);\n";
  code += "int " + prefix + "dma_transfer(void *src, void *dst, "
          "size_t size);\n";
  code += "int " + prefix + "dma_status(void);\n";
  code += "void *" + prefix + "spm_addr(unsigned core_id, "
          "const char *symbol);\n";
  code += "void " + prefix + "log_error(const char *msg);\n\n";

  // Driver entry point.
  code += "/* Pipeline driver entry point */\n";
  code += "int " + guardName + "_run(void);\n\n";

  // Kernel count and core count.
  unsigned cgraCount = 0;
  unsigned hostCount = 0;
  for (const auto &sk : schedule.executionOrder) {
    if (sk.target == ExecutionTarget::CGRA)
      cgraCount++;
    else
      hostCount++;
  }

  code += "/* Pipeline statistics */\n";
  code += "#define " + guardMacro + "_CGRA_KERNELS " +
          std::to_string(cgraCount) + "\n";
  code += "#define " + guardMacro + "_HOST_KERNELS " +
          std::to_string(hostCount) + "\n";
  code += "#define " + guardMacro + "_DMA_TRANSFERS " +
          std::to_string(schedule.dmaTransfers.size()) + "\n\n";

  code += "#ifdef __cplusplus\n";
  code += "}\n";
  code += "#endif\n\n";

  code += "#endif /* " + guardMacro + " */\n";

  // Write to file.
  std::error_code ec;
  llvm::raw_fd_ostream os(outputPath, ec);
  if (ec) {
    llvm::errs() << "HostScheduler: cannot write header to " << outputPath
                  << ": " << ec.message() << "\n";
    return false;
  }
  os << code;
  return true;
}
