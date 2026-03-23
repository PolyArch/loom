#include "loom/SystemCompiler/TapestryPipeline.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <fstream>
#include <sstream>

using namespace loom;
using namespace mlir;

//===----------------------------------------------------------------------===//
// TapestryPipelineConfig
//===----------------------------------------------------------------------===//

bool TapestryPipelineConfig::shouldRunStage(PipelineStage stage) const {
  if (stages.empty())
    return true; // empty means run all
  for (auto s : stages) {
    if (s == stage)
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// TapestryPipeline -- Main entry point
//===----------------------------------------------------------------------===//

TapestryPipelineResult TapestryPipeline::run(
    const TapestryPipelineConfig &config, MLIRContext &ctx) {
  TapestryPipelineResult result;
  result.success = false;

  // Ensure output directory exists
  std::error_code ec = llvm::sys::fs::create_directories(config.outputDir);
  if (ec) {
    result.diagnostics = "Failed to create output directory: " + ec.message();
    return result;
  }

  // Compilation stage
  if (config.shouldRunStage(PipelineStage::COMPILE)) {
    if (config.verbose)
      llvm::outs() << "[tapestry] Running compilation stage...\n";

    TapestryCompilationResult compResult = runCompile(config, ctx);
    if (!compResult.success) {
      result.diagnostics = "Compilation failed: " + compResult.diagnostics;
      result.compilationResult = std::move(compResult);
      return result;
    }

    // Write per-core config binaries
    writeConfigBinaries(compResult, config.outputDir);
    writeSystemOutputs(compResult, config.outputDir);

    result.compilationResult = std::move(compResult);

    if (config.verbose)
      llvm::outs() << "[tapestry] Compilation completed successfully.\n";
  }

  // Simulation stage
  if (config.shouldRunStage(PipelineStage::SIMULATE)) {
    if (!result.compilationResult.has_value()) {
      result.diagnostics = "Simulation requires compilation result.";
      return result;
    }
    if (config.verbose)
      llvm::outs() << "[tapestry] Running simulation stage...\n";

    auto simResult =
        runSimulate(result.compilationResult.value(), config);

    if (!simResult.success) {
      result.diagnostics = "Simulation failed: " + simResult.errorMessage;
      result.simResult = std::move(simResult);
      return result;
    }

    writeSimResults(simResult, config.outputDir);
    result.simResult = std::move(simResult);

    if (config.verbose)
      llvm::outs() << "[tapestry] Simulation completed successfully.\n";
  }

  // RTL generation stage
  if (config.shouldRunStage(PipelineStage::RTLGEN)) {
    if (!result.compilationResult.has_value()) {
      result.diagnostics = "RTL generation requires compilation result.";
      return result;
    }
    if (config.verbose)
      llvm::outs() << "[tapestry] Running RTL generation stage...\n";

    auto rtlResult =
        runRtlGen(result.compilationResult.value(), config, ctx);

    if (!rtlResult.success) {
      result.diagnostics = "RTL generation failed.";
      result.rtlResult = std::move(rtlResult);
      return result;
    }

    result.rtlResult = std::move(rtlResult);

    if (config.verbose)
      llvm::outs() << "[tapestry] RTL generation completed successfully.\n";
  }

  // Generate JSON report
  result.reportPath = generateReport(result, config.outputDir);

  result.success = true;
  return result;
}

//===----------------------------------------------------------------------===//
// Compile Stage
//===----------------------------------------------------------------------===//

TapestryCompilationResult TapestryPipeline::runCompile(
    const TapestryPipelineConfig &config, MLIRContext &ctx) {
  TapestryCompilationResult failResult;
  failResult.success = false;

  // Load TDG module
  if (config.tdgPath.empty()) {
    failResult.diagnostics = "No TDG input path specified.";
    return failResult;
  }

  auto bufOrErr = llvm::MemoryBuffer::getFile(config.tdgPath);
  if (!bufOrErr) {
    failResult.diagnostics = "Cannot open TDG file: " + config.tdgPath;
    return failResult;
  }

  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(std::move(*bufOrErr), llvm::SMLoc());
  OwningOpRef<ModuleOp> tdgModule = parseSourceFile<ModuleOp>(srcMgr, &ctx);
  if (!tdgModule) {
    failResult.diagnostics = "Failed to parse TDG MLIR: " + config.tdgPath;
    return failResult;
  }

  // Load system architecture
  SystemArchitecture arch = loadSystemArch(config.systemArchPath);

  // Build compilation input
  TapestryCompilationInput input;
  input.tdgModule = tdgModule.get();
  input.architecture = arch;
  input.baseMapperOpts = config.baseMapperOpts;
  input.ctx = &ctx;

  // Run Benders driver
  BendersDriver driver;
  auto start = std::chrono::steady_clock::now();
  TapestryCompilationResult compResult = driver.compile(input, config.bendersOpts);
  auto end = std::chrono::steady_clock::now();

  double elapsed =
      std::chrono::duration<double>(end - start).count();
  compResult.metrics.compilationTimeSec = elapsed;

  return compResult;
}

//===----------------------------------------------------------------------===//
// Simulate Stage
//===----------------------------------------------------------------------===//

mcsim::MultiCoreSimResult TapestryPipeline::runSimulate(
    const TapestryCompilationResult &compilation,
    const TapestryPipelineConfig &config) {
  mcsim::MultiCoreSimResult failResult;
  failResult.success = false;

  mcsim::MultiCoreSimSession session(config.simConfig);

  // Add each core to the simulation session
  for (const auto &coreResult : compilation.coreResults) {
    mcsim::CoreSpec spec;
    spec.name = coreResult.coreInstanceName;
    spec.coreType = coreResult.coreType;
    spec.configBlob = coreResult.aggregateConfigBlob;
    // Note: model is left default-constructed; the simulation session
    // will build the cycle model from the config blob.

    std::string err = session.addCore(spec);
    if (!err.empty()) {
      failResult.errorMessage =
          "Failed to add core " + spec.name + ": " + err;
      return failResult;
    }
  }

  // Run simulation
  return session.run();
}

//===----------------------------------------------------------------------===//
// RTL Generation Stage
//===----------------------------------------------------------------------===//

svgen::MultiCoreSVGenResult TapestryPipeline::runRtlGen(
    const TapestryCompilationResult &compilation,
    const TapestryPipelineConfig &config, MLIRContext &ctx) {
  // Build compilation descriptor
  svgen::MultiCoreCompilationDesc desc = buildCompilationDesc(compilation);

  // Configure SVGen options
  svgen::MultiCoreSVGenOptions opts = config.svgenOpts;
  if (opts.outputDir.empty())
    opts.outputDir = config.outputDir + "/rtl";
  if (opts.rtlSourceDir.empty())
    opts.rtlSourceDir = config.rtlSourceDir;

  // Set mesh dimensions from architecture if not explicitly configured
  if (opts.meshRows == 1 && opts.meshCols == 1) {
    unsigned totalCores = static_cast<unsigned>(compilation.coreResults.size());
    // Default to a roughly square mesh
    opts.meshCols = 1;
    while (opts.meshCols * opts.meshCols < totalCores)
      opts.meshCols++;
    opts.meshRows = (totalCores + opts.meshCols - 1) / opts.meshCols;
  }

  // Generate multi-core config image
  svgen::MultiCoreConfigImage configImg =
      svgen::generateMultiCoreConfig(desc);

  std::string configBinPath = config.outputDir + "/system_config.bin";
  std::string configJsonPath = config.outputDir + "/system_config.json";
  configImg.writeBinary(configBinPath);
  configImg.writeJSON(configJsonPath);

  // Generate SystemVerilog
  return svgen::generateMultiCoreSV(desc, opts, &ctx);
}

//===----------------------------------------------------------------------===//
// System Architecture Loader
//===----------------------------------------------------------------------===//

SystemArchitecture TapestryPipeline::loadSystemArch(
    const std::string &jsonPath) {
  SystemArchitecture arch;

  if (jsonPath.empty())
    return arch;

  auto bufOrErr = llvm::MemoryBuffer::getFile(jsonPath);
  if (!bufOrErr) {
    llvm::errs() << "[tapestry] Cannot open system arch file: "
                 << jsonPath << "\n";
    return arch;
  }

  auto jsonOrErr = llvm::json::parse((*bufOrErr)->getBuffer());
  if (!jsonOrErr) {
    llvm::errs() << "[tapestry] Failed to parse JSON: " << jsonPath << "\n";
    return arch;
  }

  auto *root = jsonOrErr->getAsObject();
  if (!root)
    return arch;

  // Parse core types
  if (auto *coreTypes = root->getArray("core_types")) {
    for (const auto &ct : *coreTypes) {
      auto *ctObj = ct.getAsObject();
      if (!ctObj)
        continue;

      CoreTypeSpec spec;
      if (auto name = ctObj->getString("name"))
        spec.typeName = name->str();
      if (auto count = ctObj->getInteger("instance_count"))
        spec.instanceCount = static_cast<unsigned>(*count);
      if (auto spm = ctObj->getInteger("spm_bytes"))
        spec.spmBytes = static_cast<uint64_t>(*spm);

      arch.coreTypes.push_back(std::move(spec));
    }
  }

  // Parse NoC spec
  if (auto *noc = root->getObject("noc")) {
    if (auto topo = noc->getString("topology")) {
      if (*topo == "mesh")
        arch.nocSpec.topology = L1NoCSpec::MESH;
      else if (*topo == "ring")
        arch.nocSpec.topology = L1NoCSpec::RING;
      else if (*topo == "hierarchical")
        arch.nocSpec.topology = L1NoCSpec::HIERARCHICAL;
    }
    if (auto fw = noc->getInteger("flit_width"))
      arch.nocSpec.flitWidth = static_cast<unsigned>(*fw);
    if (auto vc = noc->getInteger("virtual_channels"))
      arch.nocSpec.virtualChannels = static_cast<unsigned>(*vc);
    if (auto bw = noc->getInteger("link_bandwidth"))
      arch.nocSpec.linkBandwidth = static_cast<unsigned>(*bw);
    if (auto rps = noc->getInteger("router_pipeline_stages"))
      arch.nocSpec.routerPipelineStages = static_cast<unsigned>(*rps);
  }

  // Parse shared memory spec
  if (auto *shm = root->getObject("shared_memory")) {
    if (auto l2 = shm->getInteger("l2_size_bytes"))
      arch.sharedMemSpec.l2SizeBytes = static_cast<uint64_t>(*l2);
    if (auto banks = shm->getInteger("num_banks"))
      arch.sharedMemSpec.numBanks = static_cast<unsigned>(*banks);
    if (auto bw = shm->getInteger("bank_width_bytes"))
      arch.sharedMemSpec.bankWidthBytes = static_cast<unsigned>(*bw);
  }

  return arch;
}

//===----------------------------------------------------------------------===//
// Output Helpers
//===----------------------------------------------------------------------===//

svgen::MultiCoreCompilationDesc TapestryPipeline::buildCompilationDesc(
    const TapestryCompilationResult &compilation) {
  svgen::MultiCoreCompilationDesc desc;
  desc.success = compilation.success;

  for (const auto &cr : compilation.coreResults) {
    svgen::MultiCoreCoreDesc coreDesc;
    coreDesc.coreInstanceName = cr.coreInstanceName;
    coreDesc.coreType = cr.coreType;
    coreDesc.adgModule = cr.adgModule;
    coreDesc.aggregateConfigBlob = cr.aggregateConfigBlob;
    coreDesc.configSlices = cr.configSlices;
    desc.coreDescs.push_back(std::move(coreDesc));
  }

  return desc;
}

void TapestryPipeline::writeConfigBinaries(
    const TapestryCompilationResult &compilation,
    const std::string &outputDir) {
  for (const auto &cr : compilation.coreResults) {
    std::string coreDir = outputDir + "/" + cr.coreInstanceName;
    llvm::sys::fs::create_directories(coreDir);

    std::string configPath = coreDir + "/config.bin";
    std::error_code ec;
    llvm::raw_fd_ostream os(configPath, ec, llvm::sys::fs::OF_None);
    if (!ec) {
      os.write(reinterpret_cast<const char *>(cr.aggregateConfigBlob.data()),
               cr.aggregateConfigBlob.size());
    }
  }
}

void TapestryPipeline::writeSystemOutputs(
    const TapestryCompilationResult &compilation,
    const std::string &outputDir) {
  // Write system metrics
  {
    std::string path = outputDir + "/system_metrics.json";
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
    if (!ec)
      os << metricsToJson(compilation.metrics);
  }

  // Write iteration history
  {
    std::string path = outputDir + "/benders_history.json";
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
    if (!ec)
      os << iterationHistoryToJson(compilation.iterationHistory);
  }
}

void TapestryPipeline::writeSimResults(
    const mcsim::MultiCoreSimResult &simResult,
    const std::string &outputDir) {
  std::string path = outputDir + "/sim_results.json";
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return;

  llvm::json::Object root;
  root["success"] = simResult.success;
  root["totalGlobalCycles"] = static_cast<int64_t>(simResult.totalGlobalCycles);

  // Per-core results
  llvm::json::Array coreArr;
  for (const auto &cr : simResult.coreResults) {
    llvm::json::Object coreObj;
    coreObj["coreName"] = cr.coreName;
    coreObj["coreType"] = cr.coreType;
    coreObj["activeCycles"] = static_cast<int64_t>(cr.activeCycles);
    coreObj["stallCycles"] = static_cast<int64_t>(cr.stallCycles);
    coreObj["idleCycles"] = static_cast<int64_t>(cr.idleCycles);
    coreObj["utilization"] = cr.utilization;
    coreArr.push_back(std::move(coreObj));
  }
  root["coreResults"] = std::move(coreArr);

  // NoC stats
  llvm::json::Object nocObj;
  nocObj["totalFlitsTransferred"] =
      static_cast<int64_t>(simResult.nocStats.totalFlitsTransferred);
  nocObj["totalTransferCycles"] =
      static_cast<int64_t>(simResult.nocStats.totalTransferCycles);
  nocObj["avgLinkUtilization"] = simResult.nocStats.avgLinkUtilization;
  nocObj["maxLinkUtilization"] = simResult.nocStats.maxLinkUtilization;
  nocObj["contentionStallCycles"] =
      static_cast<int64_t>(simResult.nocStats.contentionStallCycles);
  root["nocStats"] = std::move(nocObj);

  // Memory stats
  llvm::json::Object memObj;
  memObj["spmReads"] = static_cast<int64_t>(simResult.memStats.spmReads);
  memObj["spmWrites"] = static_cast<int64_t>(simResult.memStats.spmWrites);
  memObj["l2Reads"] = static_cast<int64_t>(simResult.memStats.l2Reads);
  memObj["l2Writes"] = static_cast<int64_t>(simResult.memStats.l2Writes);
  memObj["dramReads"] = static_cast<int64_t>(simResult.memStats.dramReads);
  memObj["dramWrites"] = static_cast<int64_t>(simResult.memStats.dramWrites);
  memObj["dmaTotalBytes"] = static_cast<int64_t>(simResult.memStats.dmaTotalBytes);
  memObj["dmaTotalCycles"] = static_cast<int64_t>(simResult.memStats.dmaTotalCycles);
  root["memStats"] = std::move(memObj);

  os << llvm::json::Value(std::move(root)) << "\n";
}

//===----------------------------------------------------------------------===//
// JSON Serialization
//===----------------------------------------------------------------------===//

std::string TapestryPipeline::metricsToJson(const SystemMetrics &metrics) {
  llvm::json::Object obj;
  obj["throughput"] = metrics.throughput;
  obj["criticalPathLatency"] = metrics.criticalPathLatency;
  obj["totalNoCTransferCycles"] = metrics.totalNoCTransferCycles;
  obj["avgCoreUtilization"] = metrics.avgCoreUtilization;
  obj["maxCoreUtilization"] = metrics.maxCoreUtilization;
  obj["numBendersIterations"] = static_cast<int64_t>(metrics.numBendersIterations);
  obj["compilationTimeSec"] = metrics.compilationTimeSec;

  std::string result;
  llvm::raw_string_ostream sstr(result);
  sstr << llvm::json::Value(std::move(obj));
  return result;
}

std::string TapestryPipeline::iterationHistoryToJson(
    const std::vector<IterationRecord> &history) {
  llvm::json::Array arr;
  for (const auto &rec : history) {
    llvm::json::Object obj;
    obj["iteration"] = static_cast<int64_t>(rec.iteration);
    obj["converged"] = rec.converged;
    obj["numCuts"] = static_cast<int64_t>(rec.cuts.size());
    obj["numCostSummaries"] = static_cast<int64_t>(rec.costSummaries.size());
    obj["assignmentFeasible"] = rec.assignment.feasible;
    obj["objectiveValue"] = rec.assignment.objectiveValue;
    arr.push_back(std::move(obj));
  }

  std::string result;
  llvm::raw_string_ostream sstr(result);
  sstr << llvm::json::Value(std::move(arr));
  return result;
}

//===----------------------------------------------------------------------===//
// Report Generation
//===----------------------------------------------------------------------===//

std::string TapestryPipeline::generateReport(
    const TapestryPipelineResult &result,
    const std::string &outputDir) {
  std::string reportPath = outputDir + "/pipeline_report.json";

  llvm::json::Object root;
  root["success"] = result.success;
  root["diagnostics"] = result.diagnostics;

  // Compilation summary
  if (result.compilationResult.has_value()) {
    const auto &comp = result.compilationResult.value();
    llvm::json::Object compObj;
    compObj["success"] = comp.success;
    compObj["numCores"] = static_cast<int64_t>(comp.coreResults.size());
    compObj["numIterations"] =
        static_cast<int64_t>(comp.metrics.numBendersIterations);
    compObj["compilationTimeSec"] = comp.metrics.compilationTimeSec;

    // Per-core summaries
    llvm::json::Array coreArr;
    for (const auto &cr : comp.coreResults) {
      llvm::json::Object coreObj;
      coreObj["coreInstanceName"] = cr.coreInstanceName;
      coreObj["coreType"] = cr.coreType;
      coreObj["numAssignedKernels"] =
          static_cast<int64_t>(cr.assignedKernels.size());
      coreObj["configBlobSize"] =
          static_cast<int64_t>(cr.aggregateConfigBlob.size());

      llvm::json::Array kernelNames;
      for (const auto &k : cr.assignedKernels)
        kernelNames.push_back(k);
      coreObj["assignedKernels"] = std::move(kernelNames);

      coreArr.push_back(std::move(coreObj));
    }
    compObj["coreResults"] = std::move(coreArr);

    // System metrics
    llvm::json::Object metricsObj;
    metricsObj["throughput"] = comp.metrics.throughput;
    metricsObj["criticalPathLatency"] = comp.metrics.criticalPathLatency;
    metricsObj["totalNoCTransferCycles"] = comp.metrics.totalNoCTransferCycles;
    metricsObj["avgCoreUtilization"] = comp.metrics.avgCoreUtilization;
    metricsObj["maxCoreUtilization"] = comp.metrics.maxCoreUtilization;
    compObj["systemMetrics"] = std::move(metricsObj);

    root["compilation"] = std::move(compObj);
  }

  // Simulation summary
  if (result.simResult.has_value()) {
    const auto &sim = result.simResult.value();
    llvm::json::Object simObj;
    simObj["success"] = sim.success;
    simObj["totalGlobalCycles"] = static_cast<int64_t>(sim.totalGlobalCycles);
    simObj["numCores"] = static_cast<int64_t>(sim.coreResults.size());
    root["simulation"] = std::move(simObj);
  }

  // RTL summary
  if (result.rtlResult.has_value()) {
    const auto &rtl = result.rtlResult.value();
    llvm::json::Object rtlObj;
    rtlObj["success"] = rtl.success;
    rtlObj["systemTopFile"] = rtl.systemTopFile;
    rtlObj["numGeneratedFiles"] =
        static_cast<int64_t>(rtl.allGeneratedFiles.size());
    root["rtlGeneration"] = std::move(rtlObj);
  }

  // Write report
  std::error_code ec;
  llvm::raw_fd_ostream os(reportPath, ec, llvm::sys::fs::OF_Text);
  if (!ec)
    os << llvm::json::Value(std::move(root)) << "\n";

  return reportPath;
}
