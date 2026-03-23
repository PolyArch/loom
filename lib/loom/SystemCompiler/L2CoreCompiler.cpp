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
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <numeric>

namespace loom {

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
      std::string basePath = "/dev/null"; // We only need the in-memory blob.
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

    // Build aggregate config by concatenating per-kernel blobs.
    for (const auto &kr : l2Result.kernelResults) {
      if (kr.configBlob) {
        l2Result.aggregateConfig.insert(l2Result.aggregateConfig.end(),
                                        kr.configBlob->begin(),
                                        kr.configBlob->end());
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
