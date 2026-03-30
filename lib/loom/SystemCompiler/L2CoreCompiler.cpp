#include "loom/SystemCompiler/L2CoreCompiler.h"
#include "loom/Mapper/ADGFlattener.h"
#include "loom/Mapper/ConfigGen.h"
#include "loom/Mapper/DFGBuilder.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/MapperOptions.h"
#include "loom/Mapper/MapperTiming.h"
#include "loom/Mapper/MappingState.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <numeric>
#include <set>

namespace loom {

/// Compute the set of ADG node IDs that fall outside a given partition region.
/// Uses the ADGFlattener's grid position mapping to determine which nodes
/// are outside the partition's row/col bounds.
static std::set<IdIndex>
computeExcludedNodesForPartition(const PartitionSpec &partition,
                                 const Graph &adg,
                                 const ADGFlattener &flattener) {
  std::set<IdIndex> excluded;
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(adg.nodes.size());
       ++nodeId) {
    if (!adg.getNode(nodeId))
      continue;
    auto gridPos = flattener.getNodeGridPos(nodeId);
    // Nodes without a grid position (e.g., module I/O) are not excluded,
    // since they may be needed for routing regardless of partition.
    if (gridPos.first < 0 || gridPos.second < 0)
      continue;
    unsigned row = static_cast<unsigned>(gridPos.first);
    unsigned col = static_cast<unsigned>(gridPos.second);
    if (row < partition.rowStart || row >= partition.rowEnd ||
        col < partition.colStart || col >= partition.colEnd) {
      excluded.insert(nodeId);
    }
  }
  return excluded;
}

L2Result L2CoreCompiler::compile(const L2Assignment &assignment,
                                 const MapperOptions &baseMapperOpts,
                                 mlir::MLIRContext *ctx) {
  L2Result l2Result;
  l2Result.costSummary.coreInstanceName = assignment.coreInstanceName;
  l2Result.costSummary.coreType = assignment.coreType;

  // Flatten the core ADG.
  ADGFlattener adgFlattener;
  if (!adgFlattener.flatten(assignment.coreADG, ctx)) {
    llvm::errs() << "L2CoreCompiler: failed to flatten ADG for core '"
                 << assignment.coreInstanceName << "'\n";
    l2Result.costSummary.success = false;
    return l2Result;
  }
  const Graph &adg = adgFlattener.getADG();

  ResourceTracker tracker;
  bool allMapped = true;

  bool hasPartitions =
      !assignment.kernelPartitions.empty() &&
      assignment.kernelPartitions.size() == assignment.kernels.size();

  unsigned kernelIdx = 0;
  for (const auto &kernel : assignment.kernels) {
    L2KernelResult kernelResult;
    kernelResult.kernelName = kernel.kernelName;

    // Build DFG from the kernel's handshake.func module.
    DFGBuilder dfgBuilder;
    if (!dfgBuilder.build(kernel.kernelDFG, ctx)) {
      llvm::errs() << "L2CoreCompiler: failed to build DFG for kernel '"
                   << kernel.kernelName << "'\n";
      kernelResult.success = false;
      InfeasibilityCut cut;
      cut.kernelName = kernel.kernelName;
      cut.coreType = assignment.coreType;
      cut.reason = CutReason::TYPE_MISMATCH;
      cut.evidence = IIInfo{0, kernel.targetII.value_or(1)};
      kernelResult.cut = cut;
      l2Result.kernelResults.push_back(std::move(kernelResult));
      allMapped = false;
      continue;
    }
    const Graph &dfg = dfgBuilder.getDFG();

    // Configure mapper options for this kernel.
    MapperOptions opts = baseMapperOpts;
    if (kernel.targetII) {
      // The mapper does not have a direct targetII field; the timing options
      // influence II through recurrence analysis. We reduce the budget to
      // encourage faster convergence when a target is specified.
    }

    // Apply resource exclusions from prior kernel mappings.
    opts.excludedNodes = tracker.getUsedNodes();

    // Apply partition constraints for SPATIAL_SHARING mode.
    // Exclude all ADG nodes outside this kernel's assigned partition.
    if (hasPartitions) {
      std::set<IdIndex> partitionExcluded =
          computeExcludedNodesForPartition(
              assignment.kernelPartitions[kernelIdx], adg, adgFlattener);
      opts.excludedNodes.insert(partitionExcluded.begin(),
                                partitionExcluded.end());
    }

    // Run the mapper.
    Mapper mapper;
    Mapper::Result mapResult =
        mapper.run(dfg, adg, adgFlattener, assignment.coreADG, opts);

    if (mapResult.success) {
      kernelResult.success = true;
      kernelResult.mapperResult = std::move(mapResult);

      // Extract metrics.
      KernelMetrics metrics = extractMetrics(
          *kernelResult.mapperResult, dfg, adg, adgFlattener, kernel.kernelName);
      l2Result.costSummary.kernelMetrics.push_back(metrics);

      // Track resources used by this mapping for subsequent kernels.
      tracker.addMapping(kernelResult.mapperResult->state, adg);

      // Generate configuration blob.
      ConfigGen configGen;
      llvm::SmallString<128> tempBasePath;
      std::error_code tempEc = llvm::sys::fs::createTemporaryFile(
          "loom_l2_config", "tmp", tempBasePath);
      if (tempEc) {
        llvm::errs() << "L2CoreCompiler: failed to create temp config base: "
                     << tempEc.message() << "\n";
        kernelResult.success = false;
        allMapped = false;
        l2Result.kernelResults.push_back(std::move(kernelResult));
        break;
      }
      std::string basePath = std::string(tempBasePath);
      llvm::sys::fs::remove(tempBasePath);
      bool configOk = configGen.generate(
          kernelResult.mapperResult->state, dfg, adg, adgFlattener,
          kernelResult.mapperResult->edgeKinds,
          kernelResult.mapperResult->fuConfigs, basePath,
          baseMapperOpts.seed,
          &kernelResult.mapperResult->techMapPlan,
          &kernelResult.mapperResult->techMapMetrics,
          &kernelResult.mapperResult->timingSummary,
          &kernelResult.mapperResult->searchSummary,
          kernelResult.mapperResult->techMapDiagnostics);
      for (llvm::StringRef suffix :
           {".config.bin", ".config.json", ".config.h", ".map.json",
            ".map.txt"}) {
        llvm::sys::fs::remove(basePath + suffix.str());
      }
      if (configOk) {
        kernelResult.configBlob = configGen.getConfigBlob();
      }
    } else {
      kernelResult.success = false;
      kernelResult.cut = analyzeFailure(mapResult, dfg, adg, adgFlattener,
                                        kernel.kernelName,
                                        assignment.coreType, kernel.targetII);
      allMapped = false;
    }

    l2Result.kernelResults.push_back(std::move(kernelResult));

    // If a kernel failed, stop mapping subsequent kernels since the core
    // assignment is infeasible.
    if (!allMapped)
      break;

    kernelIdx++;
  }

  l2Result.allKernelsMapped = allMapped;
  l2Result.costSummary.success = allMapped;

  // Compute aggregate metrics.
  if (allMapped && !l2Result.costSummary.kernelMetrics.empty()) {
    double totalPE = 0.0;
    double totalSPM = 0.0;
    double maxRouting = 0.0;
    for (const auto &km : l2Result.costSummary.kernelMetrics) {
      totalPE += km.peUtilization;
      totalSPM += static_cast<double>(km.spmBytesUsed);
      maxRouting = std::max(maxRouting, km.switchUtilization);
    }
    // Clamp PE utilization to [0, 1] since sequential mappings accumulate.
    l2Result.costSummary.totalPEUtilization = std::min(totalPE, 1.0);
    l2Result.costSummary.totalSPMUtilization = totalSPM;
    l2Result.costSummary.routingPressure = maxRouting;

    // Build aggregate config: merge via ADGPartitioner for SPATIAL_SHARING
    // partitioned cores, otherwise concatenate per-kernel blobs.
    if (hasPartitions) {
      // Collect per-partition config blobs aligned with kernel order.
      std::vector<std::vector<uint8_t>> partConfigs;
      for (const auto &kr : l2Result.kernelResults) {
        partConfigs.push_back(kr.configBlob.value_or(std::vector<uint8_t>{}));
      }

      // Reconstruct PartitionPlan from the per-kernel partition specs.
      PartitionPlan plan;
      plan.partitions.assign(assignment.kernelPartitions.begin(),
                             assignment.kernelPartitions.end());
      plan.totalRows = 0;
      plan.totalCols = 0;
      for (const auto &ps : plan.partitions) {
        if (ps.rowEnd > plan.totalRows) plan.totalRows = ps.rowEnd;
        if (ps.colEnd > plan.totalCols) plan.totalCols = ps.colEnd;
      }
      plan.totalPEs = plan.totalRows * plan.totalCols;

      // Compute full config size as the sum of partition config sizes,
      // which equals totalPEs * bytesPerPE when partitions tile perfectly.
      size_t fullConfigSize = 0;
      for (const auto &pc : partConfigs)
        fullConfigSize += pc.size();

      l2Result.aggregateConfig =
          ADGPartitioner::mergeConfigurations(partConfigs, plan, fullConfigSize);
    } else {
      for (const auto &kr : l2Result.kernelResults) {
        if (kr.configBlob) {
          l2Result.aggregateConfig.insert(l2Result.aggregateConfig.end(),
                                          kr.configBlob->begin(),
                                          kr.configBlob->end());
        }
      }
    }
  } else if (!allMapped) {
    // Set the first failure cut on the cost summary.
    for (const auto &kr : l2Result.kernelResults) {
      if (kr.cut) {
        l2Result.costSummary.cut = *kr.cut;
        break;
      }
    }
  }

  return l2Result;
}

} // namespace loom
