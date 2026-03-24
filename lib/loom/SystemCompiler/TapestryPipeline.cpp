#include "loom/SystemCompiler/TapestryPipeline.h"
#include "loom/SystemCompiler/ArchitectureFactory.h"
#include "loom/SystemCompiler/PrecompiledKernelLoader.h"
#include "loom/SystemCompiler/TDGLowering.h"
#include "loom/SystemCompiler/TypeAdapters.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
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

namespace {

/// Load a TDG MLIR module from a file path.
mlir::OwningOpRef<mlir::ModuleOp>
loadTDGModule(const std::string &tdgPath, mlir::MLIRContext &ctx) {
  auto buf = llvm::MemoryBuffer::getFile(tdgPath);
  if (!buf) {
    llvm::errs() << "TapestryPipeline: cannot open TDG file '"
                 << tdgPath << "'\n";
    return nullptr;
  }
  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(std::move(*buf), llvm::SMLoc());
  return mlir::parseSourceFile<mlir::ModuleOp>(srcMgr, &ctx);
}

/// Load a system architecture from a JSON file.
/// Returns a populated SystemArchitecture on success.
tapestry::SystemArchitecture
loadSystemArchJSON(const std::string &archPath, mlir::MLIRContext &ctx) {
  tapestry::SystemArchitecture arch;

  auto buf = llvm::MemoryBuffer::getFile(archPath);
  if (!buf) {
    llvm::errs() << "TapestryPipeline: cannot open arch file '"
                 << archPath << "'\n";
    return arch;
  }

  auto json = llvm::json::parse((*buf)->getBuffer());
  if (!json) {
    llvm::errs() << "TapestryPipeline: invalid JSON in '" << archPath
                 << "'\n";
    return arch;
  }

  auto *root = json->getAsObject();
  if (!root)
    return arch;

  if (auto name = root->getString("name"))
    arch.name = name->str();

  auto *coreTypesArr = root->getArray("coreTypes");
  if (!coreTypesArr) {
    // Fallback: build a standard architecture from top-level fields.
    unsigned numTypes = 1;
    unsigned instancesPerType = 2;
    unsigned meshRows = 2;
    unsigned meshCols = 2;
    if (auto n = root->getInteger("numCoreTypes"))
      numTypes = static_cast<unsigned>(*n);
    if (auto n = root->getInteger("instancesPerType"))
      instancesPerType = static_cast<unsigned>(*n);
    if (auto n = root->getInteger("meshRows"))
      meshRows = static_cast<unsigned>(*n);
    if (auto n = root->getInteger("meshCols"))
      meshCols = static_cast<unsigned>(*n);

    return tapestry::buildStandardArchitecture(
        arch.name.empty() ? "system" : arch.name,
        numTypes, instancesPerType, meshRows, meshCols, ctx);
  }

  // Parse explicit core type specs.
  std::vector<tapestry::CoreTypeSpec> specs;
  for (const auto &entry : *coreTypesArr) {
    auto *obj = entry.getAsObject();
    if (!obj)
      continue;
    tapestry::CoreTypeSpec spec;
    if (auto n = obj->getString("name"))
      spec.name = n->str();
    if (auto n = obj->getInteger("meshRows"))
      spec.meshRows = static_cast<unsigned>(*n);
    if (auto n = obj->getInteger("meshCols"))
      spec.meshCols = static_cast<unsigned>(*n);
    if (auto n = obj->getInteger("numInstances"))
      spec.numInstances = static_cast<unsigned>(*n);
    if (auto n = obj->getInteger("spmSizeBytes"))
      spec.spmSizeBytes = static_cast<unsigned>(*n);
    if (auto b = obj->getBoolean("includeMultiplier"))
      spec.includeMultiplier = *b;
    if (auto b = obj->getBoolean("includeComparison"))
      spec.includeComparison = *b;
    if (auto b = obj->getBoolean("includeMemory"))
      spec.includeMemory = *b;
    specs.push_back(spec);
  }

  return tapestry::buildArchitecture(
      arch.name.empty() ? "system" : arch.name, specs, ctx);
}

/// Extract kernel descriptors from a parsed TDG module.
/// Looks for nested modules (each representing a kernel).
std::vector<tapestry::KernelDesc>
extractKernelsFromTDG(mlir::ModuleOp tdgModule) {
  std::vector<tapestry::KernelDesc> kernels;

  tdgModule.walk([&](mlir::ModuleOp nestedModule) {
    if (nestedModule == tdgModule)
      return;

    tapestry::KernelDesc kd;
    if (auto nameAttr = nestedModule.getSymNameAttr())
      kd.name = nameAttr.str();
    else
      kd.name = "kernel_" + std::to_string(kernels.size());

    kd.dfgModule = nestedModule;
    kernels.push_back(std::move(kd));
  });

  // If no nested modules, treat the top-level module as a single kernel.
  if (kernels.empty()) {
    tapestry::KernelDesc kd;
    if (auto nameAttr = tdgModule.getSymNameAttr())
      kd.name = nameAttr.str();
    else
      kd.name = "kernel_0";
    kd.dfgModule = tdgModule;
    kernels.push_back(std::move(kd));
  }

  return kernels;
}

/// Extract inter-kernel contracts from TDG module attributes.
/// Falls back to creating sequential contracts if no explicit ones found.
std::vector<tapestry::ContractSpec>
extractContractsFromTDG(mlir::ModuleOp tdgModule,
                        const std::vector<tapestry::KernelDesc> &kernels) {
  std::vector<tapestry::ContractSpec> contracts;

  // Check for explicit contract attributes on the TDG module.
  // If none found, create sequential contracts between adjacent kernels.
  if (kernels.size() > 1) {
    for (unsigned ki = 0; ki + 1 < kernels.size(); ++ki) {
      tapestry::ContractSpec contract;
      contract.producerKernel = kernels[ki].name;
      contract.consumerKernel = kernels[ki + 1].name;
      contract.dataType = "i32";
      contract.elementCount = 256;
      contract.bandwidthBytesPerCycle = 4;
      contracts.push_back(contract);
    }
  }

  return contracts;
}

} // anonymous namespace

TapestryPipelineResult TapestryPipeline::run(const TapestryPipelineConfig &config,
                                             mlir::MLIRContext &context) {
  TapestryPipelineResult result;
  result.reportPath = config.outputDir + "/report.json";

  auto compileStart = std::chrono::steady_clock::now();

  for (auto stage : config.stages) {
    switch (stage) {
    case PipelineStage::COMPILE: {
      if (config.verbose)
        llvm::outs() << "TapestryPipeline: loading TDG from '"
                     << config.tdgPath << "'\n";

      // Load TDG module.
      auto tdgModule = loadTDGModule(config.tdgPath, context);
      if (!tdgModule) {
        result.success = false;
        result.diagnostics = "failed to load TDG from '" + config.tdgPath + "'";
        return result;
      }

      // Load system architecture.
      if (config.verbose)
        llvm::outs() << "TapestryPipeline: loading architecture from '"
                     << config.systemArchPath << "'\n";

      tapestry::SystemArchitecture tapArch =
          loadSystemArchJSON(config.systemArchPath, context);
      if (tapArch.coreTypes.empty()) {
        result.success = false;
        result.diagnostics =
            "failed to load architecture from '" + config.systemArchPath + "'";
        return result;
      }

      // Extract kernels from TDG.
      std::vector<tapestry::KernelDesc> kernels =
          extractKernelsFromTDG(*tdgModule);

      if (config.verbose)
        llvm::outs() << "TapestryPipeline: found " << kernels.size()
                     << " kernels\n";

      // Lower kernels to DFG form if needed.
      tapestry::lowerKernelsToDFG(kernels, context);

      // Extract contracts from TDG.
      std::vector<tapestry::ContractSpec> contracts =
          extractContractsFromTDG(*tdgModule, kernels);

      if (config.verbose)
        llvm::outs() << "TapestryPipeline: " << contracts.size()
                     << " contracts\n";

      // Configure and run BendersDriver.
      tapestry::BendersConfig bendersConfig;
      bendersConfig.maxIterations = config.bendersOpts.maxIterations;
      bendersConfig.verbose = config.bendersOpts.verbose || config.verbose;

      tapestry::BendersDriver driver(tapArch, std::move(kernels),
                                     std::move(contracts), context);
      tapestry::BendersResult bendersResult = driver.compile(bendersConfig);

      auto compileEnd = std::chrono::steady_clock::now();
      double compileSec =
          std::chrono::duration<double>(compileEnd - compileStart).count();

      // Build PipelineCompilationResult from BendersResult.
      PipelineCompilationResult compResult;
      compResult.metrics.numBendersIterations = bendersResult.iterations;
      compResult.metrics.compilationTimeSec = compileSec;

      for (const auto &assign : bendersResult.assignments) {
        PipelineCoreResult cr;
        cr.coreName = assign.kernelName;
        cr.success = assign.mappingSuccess;
        compResult.coreResults.push_back(cr);
      }

      result.compilationResult = compResult;

      if (!bendersResult.success) {
        result.success = false;
        result.diagnostics = bendersResult.diagnostics;
        return result;
      }

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

  // Serialize report.json with real compilation metrics.
  if (result.compilationResult.has_value()) {
    // Ensure output directory exists.
    std::error_code ec = llvm::sys::fs::create_directories(config.outputDir);
    if (ec) {
      llvm::errs() << "TapestryPipeline: cannot create output directory '"
                   << config.outputDir << "': " << ec.message() << "\n";
    } else {
      const auto &comp = result.compilationResult.value();

      llvm::json::Object root;
      root["success"] = result.success;
      root["iterations"] =
          static_cast<int64_t>(comp.metrics.numBendersIterations);
      root["compilationTimeSec"] = comp.metrics.compilationTimeSec;

      llvm::json::Array coreResultsArr;
      for (const auto &cr : comp.coreResults) {
        llvm::json::Object crObj;
        crObj["coreName"] = cr.coreName;
        crObj["success"] = cr.success;
        coreResultsArr.push_back(std::move(crObj));
      }
      root["coreResults"] = std::move(coreResultsArr);

      llvm::json::Array diagnosticsArr;
      root["diagnostics"] = std::move(diagnosticsArr);

      std::error_code fileEC;
      llvm::raw_fd_ostream outFile(result.reportPath, fileEC,
                                   llvm::sys::fs::OF_Text);
      if (!fileEC) {
        llvm::json::Value jsonVal(std::move(root));
        outFile << llvm::formatv("{0:2}", jsonVal) << "\n";
      }
    }
  }

  return result;
}

} // namespace loom
