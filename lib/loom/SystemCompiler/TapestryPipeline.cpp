#include "loom/SystemCompiler/TapestryPipeline.h"
#include "loom/ContractInference/ContractInference.h"
#include "loom/Dialect/TDG/TDGOps.h"
#include "loom/SystemCompiler/ArchitectureFactory.h"
#include "loom/SystemCompiler/ExecutionModel.h"
#include "loom/SystemCompiler/PrecompiledKernelLoader.h"
#include "loom/SystemCompiler/SystemTypes.h"
#include "loom/SystemCompiler/TDGLowering.h"
#include "loom/SystemCompiler/TypeAdapters.h"
#include "loom/SVGen/MultiCoreSVGen.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <sstream>
#include <utility>

namespace loom {
namespace syscomp {

TapestryPipeline::TapestryPipeline(
    const HierarchicalCompilerOptions &options)
    : options_(options) {}

void TapestryPipeline::addTask(const CompilerTask &task) {
  tasks_.push_back(task);
}

void TapestryPipeline::addEdge(const TaskEdge &edge) {
  edges_.push_back(edge);
}

std::string TapestryPipeline::run() {
  legalityPassed_ = false;

  // Stage 1: Hierarchical decomposition to partition tasks across cores.
  HierarchicalCompiler compiler(options_);
  for (const auto &t : tasks_)
    compiler.addTask(t);
  for (const auto &e : edges_)
    compiler.addEdge(e);

  compilerResult_ = compiler.solve();
  if (!compilerResult_.feasible)
    return "Hierarchical partitioning failed: " +
           compilerResult_.statusMessage;

  // Stage 2: Contract legality checking.
  // Build contracts from the partition result and edge set.
  tdg::ResourceBudget budget;
  budget.nocBandwidthBytesPerCycle = options_.nocBandwidthBytesPerCycle;
  budget.spmBudgetBytes = options_.spmBudgetBytes;

  tdg::ContractLegalityChecker checker(budget);
  std::vector<tdg::Contract> contracts;

  for (const auto &e : edges_) {
    unsigned srcCore = compilerResult_.taskAssignment[e.srcTaskIndex];
    unsigned dstCore = compilerResult_.taskAssignment[e.dstTaskIndex];
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
    coreTaskIndices[compilerResult_.taskAssignment[ti]].push_back(ti);

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
    unsigned srcCore = compilerResult_.taskAssignment[e.srcTaskIndex];
    unsigned dstCore = compilerResult_.taskAssignment[e.dstTaskIndex];
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
    else if (auto n = obj->getString("typeName"))
      spec.name = n->str();
    if (auto n = obj->getInteger("meshRows"))
      spec.meshRows = static_cast<unsigned>(*n);
    if (auto n = obj->getInteger("meshCols"))
      spec.meshCols = static_cast<unsigned>(*n);
    if (auto n = obj->getInteger("numInstances"))
      spec.numInstances = static_cast<unsigned>(*n);
    else if (auto n = obj->getInteger("instanceCount"))
      spec.numInstances = static_cast<unsigned>(*n);
    if (auto n = obj->getInteger("spmSizeBytes"))
      spec.spmSizeBytes = static_cast<unsigned>(*n);
    else if (auto n = obj->getInteger("spmBytes"))
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
/// Looks for nested modules first (each representing a kernel), then falls
/// back to top-level func.func ops if the TDG is a plain module of kernels.
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

  if (kernels.empty()) {
    for (auto funcOp : tdgModule.getOps<mlir::func::FuncOp>()) {
      tapestry::KernelDesc kd;
      if (auto nameAttr = funcOp.getSymNameAttr())
        kd.name = nameAttr.str();
      else
        kd.name = "kernel_" + std::to_string(kernels.size());

      auto kernelModule = mlir::ModuleOp::create(tdgModule.getLoc());
      auto clonedFunc = funcOp.clone();
      kernelModule.getBody()->push_back(clonedFunc.getOperation());
      kd.dfgModule = kernelModule;
      kernels.push_back(std::move(kd));
    }
  }

  // If no nested modules and no top-level func.func ops, treat the top-level
  // module as a single kernel.
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

/// Return the byte width of an MLIR type (0 if unknown).
static uint64_t elementSizeBytes(mlir::Type ty) {
  if (ty.isF64() || ty.isInteger(64))
    return 8;
  if (ty.isF32() || ty.isInteger(32))
    return 4;
  if (ty.isF16() || ty.isBF16() || ty.isInteger(16))
    return 2;
  if (ty.isInteger(8))
    return 1;
  return 0;
}

/// Stringify an MLIR Type to a human-readable name (e.g. "i32", "f16").
static std::string stringifyMLIRType(mlir::Type ty) {
  std::string result;
  llvm::raw_string_ostream os(result);
  ty.print(os);
  return result;
}

/// Compute element count from a tile_shape attribute string.
/// The tile_shape is a comma-separated list of dimension sizes inside
/// brackets, e.g. "[128, 256]". Returns 0 when tile_shape is absent or
/// contains symbolic (non-numeric) dimensions.
static uint64_t computeElementCountFromTileShape(
    std::optional<llvm::StringRef> tileShapeStr) {
  if (!tileShapeStr || tileShapeStr->empty())
    return 0;

  llvm::StringRef shape = *tileShapeStr;
  // Strip leading/trailing brackets if present.
  shape = shape.trim();
  if (shape.starts_with("["))
    shape = shape.drop_front(1);
  if (shape.ends_with("]"))
    shape = shape.drop_back(1);

  uint64_t product = 1;
  llvm::SmallVector<llvm::StringRef> dims;
  shape.split(dims, ',');
  for (auto dimRef : dims) {
    llvm::StringRef dimStr = dimRef.trim();
    if (dimStr.empty())
      continue;
    uint64_t val;
    if (dimStr.getAsInteger(10, val) || val == 0)
      return 0; // Symbolic or zero dimension; cannot compute statically.
    product *= val;
  }
  return product;
}

/// Extract inter-kernel contracts from tdg.contract ops in the TDG module.
/// Walks tdg.graph -> tdg.contract ops to read real contract data, including
/// all 4 TDC edge dimensions: ordering, throughput, placement, tile_shape.
std::vector<tapestry::ContractSpec>
extractContractsFromTDG(mlir::ModuleOp tdgModule,
                        const std::vector<tapestry::KernelDesc> &kernels) {
  std::vector<tapestry::ContractSpec> contracts;

  // Walk all tdg.graph ops (typically one per module).
  tdgModule.walk([&](loom::tdg::GraphOp graphOp) {
    graphOp.walk([&](loom::tdg::ContractOp contractOp) {
      tapestry::ContractSpec contract;
      contract.producerKernel = contractOp.getProducer().str();
      contract.consumerKernel = contractOp.getConsumer().str();

      // Stringify the MLIR data_type attribute (e.g. i32, f32).
      contract.dataType = stringifyMLIRType(contractOp.getDataType());

      // Compute element count from tile_shape when available.
      std::optional<llvm::StringRef> tileShapeStr;
      if (auto tsAttr = contractOp.getTileShapeAttr())
        tileShapeStr = tsAttr.getValue();
      contract.elementCount = computeElementCountFromTileShape(tileShapeStr);

      // Derive bandwidth from element size (one element per cycle baseline).
      uint64_t elemBytes = elementSizeBytes(contractOp.getDataType());
      contract.bandwidthBytesPerCycle = elemBytes > 0 ? elemBytes : 1;

      // Extract all 4 TDC edge dimensions when present on the op.
      if (auto ordAttr = contractOp.getOrdering())
        contract.ordering = ordAttr->str();
      if (auto thrAttr = contractOp.getThroughput())
        contract.throughput = thrAttr->str();
      if (auto plcAttr = contractOp.getPlacement())
        contract.placement = plcAttr->str();
      if (tileShapeStr)
        contract.tileShape = tileShapeStr->str();

      contracts.push_back(contract);
    });
  });

  return contracts;
}

/// Extract path contracts from tdg.path_contract ops in the TDG module.
/// Walks tdg.graph -> tdg.path_contract ops and converts them to TDCPathSpec.
/// The start_edge / end_edge references are symbolic names that encode the
/// producer->consumer pair in the format "producer_to_consumer".
std::vector<loom::TDCPathSpec>
extractPathContractsFromTDG(mlir::ModuleOp tdgModule) {
  std::vector<loom::TDCPathSpec> paths;

  tdgModule.walk([&](loom::tdg::GraphOp graphOp) {
    // Build a map from edge symbolic names to (producer, consumer) pairs.
    // Edge names follow the convention "producer_to_consumer" unless a
    // contract defines a different symbolic name.
    std::map<std::string, std::pair<std::string, std::string>> edgeNameMap;
    graphOp.walk([&](loom::tdg::ContractOp contractOp) {
      std::string producer = contractOp.getProducer().str();
      std::string consumer = contractOp.getConsumer().str();
      std::string edgeName = producer + "_to_" + consumer;
      edgeNameMap[edgeName] = {producer, consumer};
      // Also register by just producer name as a fallback.
      edgeNameMap[producer] = {producer, consumer};
    });

    graphOp.walk([&](loom::tdg::PathContractOp pathOp) {
      loom::TDCPathSpec spec;
      std::string startEdgeName = pathOp.getStartEdge().str();
      std::string endEdgeName = pathOp.getEndEdge().str();

      auto startIt = edgeNameMap.find(startEdgeName);
      if (startIt != edgeNameMap.end()) {
        spec.startProducer = startIt->second.first;
        spec.startConsumer = startIt->second.second;
      } else {
        // Fallback: use the edge name directly as producer with empty consumer.
        spec.startProducer = startEdgeName;
      }

      auto endIt = edgeNameMap.find(endEdgeName);
      if (endIt != edgeNameMap.end()) {
        spec.endProducer = endIt->second.first;
        spec.endConsumer = endIt->second.second;
      } else {
        spec.endProducer = endEdgeName;
      }

      spec.latency = pathOp.getLatency().str();
      paths.push_back(std::move(spec));
    });
  });

  return paths;
}

static loom::svgen::MultiCoreCompilationDesc
buildMultiCoreCompilationDesc(const tapestry::CompilationResult &compResult,
                              const tapestry::SystemArchitecture &tapArch) {
  std::map<std::pair<int, int>, loom::svgen::MultiCoreCoreDesc> coreMap;

  for (const auto &assign : compResult.assignments) {
    if (!assign.mappingSuccess)
      continue;
    if (assign.coreTypeIndex < 0)
      continue;

    const unsigned typeIdx = static_cast<unsigned>(assign.coreTypeIndex);
    if (typeIdx >= tapArch.coreTypes.size())
      continue;

    const auto key = std::make_pair(assign.coreTypeIndex,
                                    assign.coreInstanceIndex);
    auto &core = coreMap[key];
    const auto &tapCore = tapArch.coreTypes[typeIdx];

    if (core.coreType.empty())
      core.coreType = tapCore.name;
    if (core.coreInstanceName.empty()) {
      core.coreInstanceName =
          tapCore.name + "_" + std::to_string(assign.coreInstanceIndex);
    }

    if (assign.coreADG)
      core.adgModule = assign.coreADG;
    else if (!core.adgModule)
      core.adgModule = tapCore.adgModule;
  }

  loom::svgen::MultiCoreCompilationDesc desc;
  for (auto &entry : coreMap) {
    if (!entry.second.adgModule)
      continue;
    desc.coreDescs.push_back(std::move(entry.second));
  }
  return desc;
}

} // anonymous namespace

TapestryPipelineResult TapestryPipeline::run(const TapestryPipelineConfig &config,
                                             mlir::MLIRContext &context) {
  TapestryPipelineResult result;
  result.reportPath = config.outputDir + "/report.json";

  auto compileStart = std::chrono::steady_clock::now();

  // Verification inputs populated during COMPILE, reused in SIMULATE for
  // dynamic verification when simulation results are available.
  std::vector<loom::TDCEdgeSpec> savedVerifyEdgeSpecs;
  std::vector<loom::TDCEdgeSpecOrigin> savedVerifyOrigins;
  std::vector<loom::TDCPathSpec> savedVerifyPaths;
  loom::BufferAllocationPlan savedVerifyBufPlan;
  std::vector<loom::EdgeTileDimensions> savedVerifyTileDims;
  std::vector<loom::EdgeSchedulingSlot> savedVerifySchedSlots;
  std::map<std::string, int64_t> savedVerifyParams;
  std::optional<tapestry::CompilationResult> compiledResult;
  std::optional<tapestry::SystemArchitecture> compiledArch;

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

      // Run ContractInferencePass on the TDG module before compilation.
      {
        if (config.verbose)
          llvm::errs() << "Running ContractInferencePass...\n";

        ContractInferencePass::Options ciOpts;
        ciOpts.defaultSPMCapacityBytes = config.ciSPMCapacityBytes;
        ciOpts.sharedL2CapacityBytes = config.ciL2CapacityBytes;
        ciOpts.spmThresholdFraction = config.ciSPMThresholdFraction;
        ciOpts.l2ThresholdFraction = config.ciL2ThresholdFraction;
        ciOpts.defaultProducerLatencyCycles = config.ciProducerLatencyCycles;

        ContractInferencePass ciPass;
        (void)ciPass.run(*tdgModule, ciOpts);

        if (config.verbose)
          llvm::errs() << "ContractInferencePass completed.\n";
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
      compiledArch = tapArch;

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

      // Save contract metadata before moving into the compiler, since
      // we need them later for TDC verification evidence.
      std::vector<tapestry::ContractSpec> savedContracts = contracts;

      // Extract path contracts from tdg.path_contract ops (multi-edge
      // latency constraints) so they participate in L1 assignment pruning.
      std::vector<loom::TDCPathSpec> tdgPathSpecs =
          extractPathContractsFromTDG(*tdgModule);

      if (config.verbose)
        llvm::outs() << "TapestryPipeline: " << tdgPathSpecs.size()
                     << " path contracts\n";

      // Configure and run HierarchicalCompiler.
      tapestry::CompilerConfig compilerConfig;
      compilerConfig.maxIterations = config.bendersOpts.maxIterations;
      compilerConfig.verbose = config.bendersOpts.verbose || config.verbose;
      compilerConfig.executionModel = config.executionModel;

      tapestry::HierarchicalCompiler compiler(tapArch, std::move(kernels),
                                              std::move(contracts), context);
      compiler.setPathSpecs(tdgPathSpecs);
      tapestry::CompilationResult compResult =
          compiler.compile(compilerConfig);
      compiledResult = compResult;

      auto compileEnd = std::chrono::steady_clock::now();
      double compileSec =
          std::chrono::duration<double>(compileEnd - compileStart).count();

      // Build PipelineCompilationResult from CompilationResult.
      PipelineCompilationResult pipeCompResult;
      pipeCompResult.metrics.numBendersIterations = compResult.iterations;
      pipeCompResult.metrics.compilationTimeSec = compileSec;

      for (const auto &assign : compResult.assignments) {
        PipelineCoreResult cr;
        cr.coreName = assign.kernelName;
        cr.success = assign.mappingSuccess;
        pipeCompResult.coreResults.push_back(cr);
      }

      result.compilationResult = pipeCompResult;

      // Use the temporal schedule from compilation result if available,
      // otherwise build a minimal schedule from the execution model config.
      if (compResult.temporalSchedule.has_value()) {
        result.temporalSchedule = compResult.temporalSchedule.value();
      } else {
        TemporalSchedule schedule;
        schedule.mode = config.executionModel.mode;
        schedule.systemLatencyCycles = 0;
        schedule.maxCoreCycles = 0;
        schedule.nocOverheadCycles = 0;
        result.temporalSchedule = schedule;
      }

      if (config.verbose) {
        llvm::outs() << "TapestryPipeline: temporal scheduling ("
                     << executionModeToString(config.executionModel.mode)
                     << ", reconfigCycles="
                     << config.executionModel.reconfigCycles << ")\n";
      }

      if (!compResult.success) {
        result.success = false;
        result.diagnostics = compResult.diagnostics;
        return result;
      }

      // Run TDC contract verification on the compilation results.
      {
        // Build TDCEdgeSpecs from the saved contracts, populating all
        // 4 TDC edge dimensions (ordering, throughput, placement, shape)
        // when they were present on the original tdg.contract ops.
        std::vector<loom::TDCEdgeSpec> verifyEdges;
        for (const auto &c : savedContracts) {
          loom::TDCEdgeSpec es;
          es.producerKernel = c.producerKernel;
          es.consumerKernel = c.consumerKernel;
          es.dataTypeName = c.dataType;

          if (c.ordering)
            es.ordering = loom::orderingFromString(*c.ordering);
          if (c.throughput)
            es.throughput = *c.throughput;
          if (c.placement)
            es.placement = loom::placementFromString(*c.placement);
          if (c.tileShape)
            es.shape = *c.tileShape;

          verifyEdges.push_back(std::move(es));
        }

        // Infer missing dimensions so we have origin tracking.
        loom::InferenceResult inferred = loom::inferEdgeContracts(verifyEdges);

        // Assemble available compile-time outputs for static verification.
        loom::BufferAllocationPlan verifyBufPlan;
        if (compResult.bufferPlan.has_value())
          verifyBufPlan = compResult.bufferPlan.value();

        // Extract real tile dimensions from L2 compilation results.
        // Each contract's elementCount represents the tile size produced by
        // the compilation; expose it as a 1-D tile dimension vector.
        std::vector<loom::EdgeTileDimensions> verifyTileDims;
        for (const auto &c : savedContracts) {
          loom::EdgeTileDimensions td;
          td.producerKernel = c.producerKernel;
          td.consumerKernel = c.consumerKernel;
          td.dimensions.push_back(static_cast<int64_t>(c.elementCount));
          verifyTileDims.push_back(std::move(td));
        }

        // Extract real scheduling slots from the temporal schedule.
        // For each contract edge, find producer and consumer kernel timings
        // and build completion/start time vectors.
        std::vector<loom::EdgeSchedulingSlot> verifySchedSlots;
        if (compResult.temporalSchedule.has_value()) {
          const auto &tempSched = compResult.temporalSchedule.value();

          // Build kernel-name -> timing lookup from the temporal schedule.
          std::map<std::string, loom::KernelTiming> timingMap;
          for (const auto &cs : tempSched.coreSchedules) {
            for (size_t ki = 0; ki < cs.kernelOrder.size(); ++ki) {
              if (ki < cs.kernelTimings.size())
                timingMap[cs.kernelOrder[ki]] = cs.kernelTimings[ki];
            }
          }

          for (const auto &c : savedContracts) {
            auto prodIt = timingMap.find(c.producerKernel);
            auto consIt = timingMap.find(c.consumerKernel);
            if (prodIt != timingMap.end() && consIt != timingMap.end()) {
              loom::EdgeSchedulingSlot slot;
              slot.producerKernel = c.producerKernel;
              slot.consumerKernel = c.consumerKernel;

              // Build per-tile timing: producer completes at
              // startTime + executionCycles; consumer begins at startTime.
              const auto &prodTiming = prodIt->second;
              const auto &consTiming = consIt->second;
              uint64_t prodCompletion =
                  prodTiming.startTime + prodTiming.executionCycles;
              uint64_t consStart = consTiming.startTime;

              slot.producerCompletionTimes.push_back(prodCompletion);
              slot.consumerStartTimes.push_back(consStart);
              verifySchedSlots.push_back(std::move(slot));
            }
          }
        }

        // Extract path contracts from tdg.path_contract ops if present.
        std::vector<loom::TDCPathSpec> verifyPaths =
            extractPathContractsFromTDG(*tdgModule);
        std::map<std::string, int64_t> verifyParams;

        // Persist verification inputs for dynamic re-verification in
        // SIMULATE stage.
        savedVerifyEdgeSpecs = inferred.edgeSpecs;
        savedVerifyOrigins = inferred.origins;
        savedVerifyPaths = verifyPaths;
        savedVerifyBufPlan = verifyBufPlan;
        savedVerifyTileDims = verifyTileDims;
        savedVerifySchedSlots = verifySchedSlots;
        savedVerifyParams = verifyParams;

        loom::TDCVerificationReport verifyReport =
            loom::verifyContracts(
                inferred.edgeSpecs, inferred.origins, verifyPaths,
                verifyBufPlan, verifyTileDims, verifySchedSlots,
                std::nullopt, std::nullopt, verifyParams);

        if (!verifyReport.allSatisfied) {
          std::string verifyDiag = "TDC verification failures:";
          for (const auto &d : verifyReport.diagnostics)
            verifyDiag += " " + d + ";";
          for (const auto &er : verifyReport.edgeResults) {
            for (const auto &d : er.diagnostics)
              verifyDiag += " [" + er.producerKernel + "->"
                  + er.consumerKernel + "] " + d + ";";
          }
          if (result.diagnostics.empty())
            result.diagnostics = verifyDiag;
          else
            result.diagnostics += "; " + verifyDiag;
        }

        // Store the full verification report in the pipeline result.
        result.tdcVerificationReport = verifyReport;

        if (config.verbose) {
          llvm::outs() << "TapestryPipeline: TDC verification "
                       << (verifyReport.allSatisfied ? "PASSED" : "FAILED")
                       << " (" << verifyReport.edgeResults.size()
                       << " edge checks, "
                       << verifyReport.pathResults.size()
                       << " path checks)\n";
          for (const auto &d : verifyReport.diagnostics)
            llvm::outs() << "  TDC: " << d << "\n";
        }
      }

      break;
    }
    case PipelineStage::SIMULATE: {
      // The SIMULATE stage delegates to external simulation tools.
      // If a simulation results file exists at the expected path,
      // parse it and populate dynamic metrics for TDC verification.
      std::string simResultsPath =
          config.outputDir + "/sim_results.json";

      auto simBuf = llvm::MemoryBuffer::getFile(simResultsPath);
      if (!simBuf) {
        // No simulation results file found; simulation was not run.
        result.success = false;
        result.diagnostics =
            "SIMULATE stage: no simulation results found at '" +
            simResultsPath + "'; run MultiCoreSimSession or the "
            "external simulator to produce this file";
        return result;
      }

      // Parse the simulation results JSON.
      auto simJson = llvm::json::parse((*simBuf)->getBuffer());
      if (!simJson || !simJson->getAsObject()) {
        result.success = false;
        result.diagnostics =
            "SIMULATE stage: invalid JSON in '" + simResultsPath + "'";
        return result;
      }

      auto *simRoot = simJson->getAsObject();

      // Populate PipelineSimResult from the JSON.
      PipelineSimResult simRes;
      if (auto gc = simRoot->getInteger("totalGlobalCycles"))
        simRes.totalGlobalCycles = static_cast<uint64_t>(*gc);
      if (auto *nocObj = simRoot->getObject("nocStats")) {
        if (auto ft = nocObj->getInteger("totalFlitsTransferred"))
          simRes.nocStats.totalFlitsTransferred = static_cast<uint64_t>(*ft);
      }
      if (auto *coresArr = simRoot->getArray("coreResults")) {
        for (const auto &entry : *coresArr) {
          auto *obj = entry.getAsObject();
          if (!obj)
            continue;
          PipelineCoreSimResult csr;
          if (auto cid = obj->getInteger("coreId"))
            csr.coreId = static_cast<unsigned>(*cid);
          if (auto cyc = obj->getInteger("cycles"))
            csr.cycles = static_cast<uint64_t>(*cyc);
          if (auto comp = obj->getBoolean("completed"))
            csr.completed = *comp;
          simRes.coreResults.push_back(csr);
        }
      }
      result.simResult = simRes;

      // Parse dynamic edge metrics for TDC verification.
      std::optional<std::vector<loom::DynamicEdgeMetrics>> dynEdgeMetrics;
      std::optional<std::vector<loom::DynamicPathMetrics>> dynPathMetrics;

      if (auto *edgeArr = simRoot->getArray("edgeMetrics")) {
        std::vector<loom::DynamicEdgeMetrics> edgeVec;
        for (const auto &entry : *edgeArr) {
          auto *obj = entry.getAsObject();
          if (!obj)
            continue;
          loom::DynamicEdgeMetrics dem;
          if (auto p = obj->getString("producerKernel"))
            dem.producerKernel = p->str();
          if (auto c = obj->getString("consumerKernel"))
            dem.consumerKernel = c->str();
          if (auto t = obj->getNumber("sustainedThroughput"))
            dem.sustainedThroughput = *t;
          if (auto v = obj->getInteger("orderingViolationCount"))
            dem.orderingViolationCount = *v;
          edgeVec.push_back(std::move(dem));
        }
        dynEdgeMetrics = std::move(edgeVec);
      }

      if (auto *pathArr = simRoot->getArray("pathMetrics")) {
        std::vector<loom::DynamicPathMetrics> pathVec;
        for (const auto &entry : *pathArr) {
          auto *obj = entry.getAsObject();
          if (!obj)
            continue;
          loom::DynamicPathMetrics dpm;
          if (auto p = obj->getString("startProducer"))
            dpm.startProducer = p->str();
          if (auto c = obj->getString("startConsumer"))
            dpm.startConsumer = c->str();
          if (auto p = obj->getString("endProducer"))
            dpm.endProducer = p->str();
          if (auto c = obj->getString("endConsumer"))
            dpm.endConsumer = c->str();
          if (auto l = obj->getInteger("observedLatency"))
            dpm.observedLatency = *l;
          pathVec.push_back(std::move(dpm));
        }
        dynPathMetrics = std::move(pathVec);
      }

      // Re-run TDC verification with dynamic metrics if compile-stage
      // verification inputs were populated and dynamic data is available.
      if (!savedVerifyEdgeSpecs.empty() &&
          (dynEdgeMetrics.has_value() || dynPathMetrics.has_value())) {
        loom::TDCVerificationReport dynReport =
            loom::verifyContracts(
                savedVerifyEdgeSpecs, savedVerifyOrigins,
                savedVerifyPaths, savedVerifyBufPlan,
                savedVerifyTileDims, savedVerifySchedSlots,
                dynEdgeMetrics, dynPathMetrics, savedVerifyParams);

        // Replace the static-only report with the combined report.
        result.tdcVerificationReport = dynReport;

        if (!dynReport.allSatisfied) {
          std::string dynDiag = "TDC dynamic verification failures:";
          for (const auto &d : dynReport.diagnostics)
            dynDiag += " " + d + ";";
          for (const auto &er : dynReport.edgeResults) {
            for (const auto &d : er.diagnostics)
              dynDiag += " [" + er.producerKernel + "->"
                  + er.consumerKernel + "] " + d + ";";
          }
          for (const auto &pr : dynReport.pathResults) {
            for (const auto &d : pr.diagnostics)
              dynDiag += " [path " + pr.startProducer + "->"
                  + pr.endConsumer + "] " + d + ";";
          }
          if (result.diagnostics.empty())
            result.diagnostics = dynDiag;
          else
            result.diagnostics += "; " + dynDiag;
        }

        if (config.verbose) {
          llvm::outs() << "TapestryPipeline: dynamic TDC verification "
                       << (dynReport.allSatisfied ? "PASSED" : "FAILED")
                       << " (" << dynReport.edgeResults.size()
                       << " edge checks, "
                       << dynReport.pathResults.size()
                       << " path checks)\n";
        }
      }

      break;
    }
    case PipelineStage::RTLGEN: {
      if (!compiledResult.has_value() || !compiledArch.has_value()) {
        result.success = false;
        result.diagnostics =
            "RTLGEN stage requires a successful COMPILE stage first";
        return result;
      }

      loom::svgen::MultiCoreCompilationDesc rtlCompilation =
          buildMultiCoreCompilationDesc(*compiledResult, *compiledArch);
      if (rtlCompilation.coreDescs.empty()) {
        result.success = false;
        result.diagnostics =
            "RTLGEN stage has no mapped cores to emit";
        return result;
      }

      loom::svgen::MultiCoreSVGenOptions svgenOpts;
      svgenOpts.outputDir = config.outputDir;
      if (config.rtlSourceDir.empty()) {
#ifdef LOOM_SOURCE_DIR
        svgenOpts.rtlSourceDir = std::string(LOOM_SOURCE_DIR) + "/src/rtl";
#else
        svgenOpts.rtlSourceDir = "src/rtl";
#endif
      } else {
        svgenOpts.rtlSourceDir = config.rtlSourceDir;
      }
      svgenOpts.fpIpProfile = config.svgenOpts.fpIpProfile;
      svgenOpts.meshRows = config.svgenOpts.meshRows;
      svgenOpts.meshCols = config.svgenOpts.meshCols;
      if (svgenOpts.meshRows == 0 || svgenOpts.meshCols == 0) {
        const unsigned numCores =
            static_cast<unsigned>(rtlCompilation.coreDescs.size());
        const unsigned side = std::max(
            1u, static_cast<unsigned>(
                    std::ceil(std::sqrt(static_cast<double>(numCores)))));
        if (svgenOpts.meshRows == 0)
          svgenOpts.meshRows = side;
        if (svgenOpts.meshCols == 0)
          svgenOpts.meshCols = (numCores + side - 1) / side;
      }

      loom::svgen::MultiCoreSVGenResult rtlGenResult =
          loom::svgen::generateMultiCoreSV(rtlCompilation, svgenOpts,
                                           &context);
      if (!rtlGenResult.success) {
        result.success = false;
        result.diagnostics =
            "RTL generation failed for compiled multicore system";
        return result;
      }

      PipelineRTLResult rtlResult;
      rtlResult.systemTopFile = rtlGenResult.systemTopFile;
      rtlResult.allGeneratedFiles = rtlGenResult.allGeneratedFiles;
      result.rtlResult = std::move(rtlResult);
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
