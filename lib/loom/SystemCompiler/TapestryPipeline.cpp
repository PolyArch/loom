#include "loom/SystemCompiler/TapestryPipeline.h"
#include "loom/ContractInference/ContractInference.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/SourceMgr.h"

#include <sstream>

namespace loom {
namespace syscomp {

TapestryPipeline::TapestryPipeline(const BendersDriverOptions &options)
    : options_(options) {}

void TapestryPipeline::addTask(const BendersTask &task) {
  tasks_.push_back(task);
}

void TapestryPipeline::addEdge(const BendersEdge &edge) {
  edges_.push_back(edge);
}

std::string TapestryPipeline::run() {
  legalityPassed_ = false;

  // Stage 1: Benders decomposition to partition tasks across cores.
  BendersDriver driver(options_);
  for (const auto &t : tasks_)
    driver.addTask(t);
  for (const auto &e : edges_)
    driver.addEdge(e);

  bendersResult_ = driver.solve();
  if (!bendersResult_.feasible)
    return "Benders partitioning failed: " + bendersResult_.statusMessage;

  // Stage 2: Contract legality checking.
  // Build contracts from the partition result and edge set.
  tdg::ResourceBudget budget;
  budget.nocBandwidthBytesPerCycle = options_.nocBandwidthBytesPerCycle;
  budget.spmBudgetBytes = options_.spmBudgetBytes;

  tdg::ContractLegalityChecker checker(budget);
  std::vector<tdg::Contract> contracts;

  for (const auto &e : edges_) {
    unsigned srcCore = bendersResult_.taskAssignment[e.srcTaskIndex];
    unsigned dstCore = bendersResult_.taskAssignment[e.dstTaskIndex];
    if (srcCore == dstCore)
      continue; // Intra-core edges need no NoC contract.

    tdg::Contract c;
    c.producerCoreId = srcCore;
    c.consumerCoreId = dstCore;
    c.dataBytes = e.dataBytes;
    c.producerCycles = tasks_[e.srcTaskIndex].estimatedCycles;
    // Assume the system compiler allocates sufficient buffers.
    c.minBufferElements = 1;
    c.allocatedBufferElements = 4;
    c.spmBytesRequested = tasks_[e.srcTaskIndex].spmBytes;
    contracts.push_back(c);
  }

  tdg::LegalityResult lr = checker.checkAll(contracts);
  if (!lr.legal)
    return "contract legality check failed: " + lr.message;

  legalityPassed_ = true;

  // Stage 3: Multi-core simulation.
  mcsim::MultiCoreSimConfig simConfig;
  simConfig.nocBandwidthBytesPerCycle = options_.nocBandwidthBytesPerCycle;
  simConfig.maxCores = options_.numCores;

  mcsim::MultiCoreSimSession sim(simConfig);

  // Build per-core kernel lists from the partition.
  // Group tasks by assigned core, preserving task index order.
  std::vector<std::vector<unsigned>> coreTaskIndices(options_.numCores);
  for (unsigned ti = 0; ti < tasks_.size(); ++ti)
    coreTaskIndices[bendersResult_.taskAssignment[ti]].push_back(ti);

  // Track kernel index within each core for NoC transfer descriptors.
  std::vector<unsigned> taskToKernelIndex(tasks_.size(), 0);

  for (unsigned ci = 0; ci < options_.numCores; ++ci) {
    for (unsigned ki = 0; ki < coreTaskIndices[ci].size(); ++ki) {
      unsigned ti = coreTaskIndices[ci][ki];
      taskToKernelIndex[ti] = ki;

      mcsim::KernelDescriptor kd;
      kd.name = tasks_[ti].name;
      kd.coreId = ci;
      kd.estimatedCycles = tasks_[ti].estimatedCycles;
      kd.outputBytes = tasks_[ti].outputBytes;
      // Allow interleaved NoC injection at 75% of kernel execution.
      if (kd.estimatedCycles > 0)
        kd.outputReadyCycleOffset = (kd.estimatedCycles * 3) / 4;
      sim.addKernel(kd);
    }
  }

  // Add NoC transfers for cross-core edges.
  for (const auto &e : edges_) {
    unsigned srcCore = bendersResult_.taskAssignment[e.srcTaskIndex];
    unsigned dstCore = bendersResult_.taskAssignment[e.dstTaskIndex];
    if (srcCore == dstCore)
      continue;

    mcsim::NocTransferDescriptor td;
    td.srcCoreId = srcCore;
    td.dstCoreId = dstCore;
    td.bytes = e.dataBytes;
    td.srcKernelIndex = taskToKernelIndex[e.srcTaskIndex];
    sim.addNocTransfer(td);
  }

  simResult_ = sim.run();
  if (!simResult_.success)
    return "multi-core simulation failed: " + simResult_.errorMessage;

  return {};
}

} // namespace syscomp

// -----------------------------------------------------------------------
// Config-driven full pipeline implementation (used by tapestry CLI tools)
// -----------------------------------------------------------------------

TapestryPipelineResult TapestryPipeline::run(const TapestryPipelineConfig &config,
                                             mlir::MLIRContext &context) {
  TapestryPipelineResult result;

  // Stub: the config-driven pipeline stages are not yet wired.
  // For now, report success with empty results so the tools can link and
  // exercise the command-line parsing and dialect registration paths.

  for (auto stage : config.stages) {
    switch (stage) {
    case PipelineStage::COMPILE: {
      // Run ContractInferencePass on the TDG module before BendersDriver.
      // This fills in missing contract fields (rates, tile shape, buffer
      // sizes, visibility) and warns on unsupported backpressure modes.
      if (!config.tdgPath.empty()) {
        auto tdgModuleRef = mlir::parseSourceFile<mlir::ModuleOp>(
            config.tdgPath, &context);
        if (tdgModuleRef) {
          if (config.verbose)
            llvm::errs() << "Running ContractInferencePass...\n";

          ContractInferencePass::Options ciOpts;
          ciOpts.defaultSPMCapacityBytes = config.ciSPMCapacityBytes;
          ciOpts.sharedL2CapacityBytes = config.ciL2CapacityBytes;
          ciOpts.spmThresholdFraction = config.ciSPMThresholdFraction;
          ciOpts.l2ThresholdFraction = config.ciL2ThresholdFraction;
          ciOpts.defaultProducerLatencyCycles = config.ciProducerLatencyCycles;

          ContractInferencePass ciPass;
          (void)ciPass.run(*tdgModuleRef, ciOpts);

          if (config.verbose)
            llvm::errs() << "ContractInferencePass completed.\n";
        }
      }

      PipelineCompilationResult compResult;
      compResult.metrics.numBendersIterations = 0;
      compResult.metrics.compilationTimeSec = 0.0;
      result.compilationResult = compResult;
      break;
    }
    case PipelineStage::SIMULATE: {
      PipelineSimResult simResult;
      simResult.totalGlobalCycles = 0;
      simResult.nocStats.totalFlitsTransferred = 0;
      result.simResult = simResult;
      break;
    }
    case PipelineStage::RTLGEN: {
      PipelineRTLResult rtlResult;
      result.rtlResult = rtlResult;
      break;
    }
    }
  }

  result.success = true;
  result.reportPath = config.outputDir + "/report.json";
  return result;
}

} // namespace loom
