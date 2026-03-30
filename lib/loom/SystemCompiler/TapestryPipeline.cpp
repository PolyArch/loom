#include "loom/SystemCompiler/TapestryPipeline.h"
#include "loom/ContractInference/ContractInference.h"
#include "loom/Dialect/TDG/TDGOps.h"
#include "loom/SVGen/MultiCoreSVGen.h"
#include "loom/SystemCompiler/ArchitectureFactory.h"
#include "loom/SystemCompiler/ExecutionModel.h"
#include "loom/SystemCompiler/PrecompiledKernelLoader.h"
#include "loom/SystemCompiler/SystemTypes.h"
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
#include <map>
#include <sstream>

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
/// Walks tdg.graph -> tdg.contract ops to read real contract data.
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

      contracts.push_back(contract);
    });
  });

  return contracts;
}

} // anonymous namespace

TapestryPipelineResult TapestryPipeline::run(const TapestryPipelineConfig &config,
                                             mlir::MLIRContext &context) {
  TapestryPipelineResult result;
  result.reportPath = config.outputDir + "/report.json";

  auto compileStart = std::chrono::steady_clock::now();

  // State that persists across pipeline stages so that SIMULATE and RTLGEN
  // can consume the outputs produced by COMPILE.
  std::optional<tapestry::CompilationResult> internalCompResult;
  std::vector<tapestry::ContractSpec> savedContracts;

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
      // we need them later for TDC verification evidence and simulation.
      savedContracts = contracts;

      // Configure and run HierarchicalCompiler.
      tapestry::CompilerConfig compilerConfig;
      compilerConfig.maxIterations = config.bendersOpts.maxIterations;
      compilerConfig.verbose = config.bendersOpts.verbose || config.verbose;
      compilerConfig.executionModel = config.executionModel;

      tapestry::HierarchicalCompiler compiler(tapArch, std::move(kernels),
                                              std::move(contracts), context);
      tapestry::CompilationResult compResult =
          compiler.compile(compilerConfig);

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

      // Preserve the full internal compilation result for downstream stages.
      internalCompResult = std::move(compResult);

      // Run TDC contract verification on the compilation results.
      {
        const auto &compRef = internalCompResult.value();

        // Build TDCEdgeSpecs from the saved contracts.
        std::vector<loom::TDCEdgeSpec> verifyEdges;
        for (const auto &c : savedContracts) {
          loom::TDCEdgeSpec es;
          es.producerKernel = c.producerKernel;
          es.consumerKernel = c.consumerKernel;
          es.dataTypeName = c.dataType;
          verifyEdges.push_back(std::move(es));
        }

        // Infer missing dimensions so we have origin tracking.
        loom::InferenceResult inferred = loom::inferEdgeContracts(verifyEdges);

        // Assemble available compile-time outputs for static verification.
        loom::BufferAllocationPlan verifyBufPlan;
        if (compRef.bufferPlan.has_value())
          verifyBufPlan = compRef.bufferPlan.value();

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
        if (compRef.temporalSchedule.has_value()) {
          const auto &tempSched = compRef.temporalSchedule.value();

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

        std::vector<loom::TDCPathSpec> verifyPaths;
        std::map<std::string, int64_t> verifyParams;

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
      // The SIMULATE stage uses MultiCoreSimSession to estimate system-level
      // latency from the compilation result's temporal schedule and contracts.
      if (!internalCompResult.has_value() ||
          !result.temporalSchedule.has_value()) {
        result.success = false;
        result.diagnostics =
            "SIMULATE stage requires a preceding COMPILE stage";
        return result;
      }

      const auto &tempSched = result.temporalSchedule.value();
      const auto &compRef = internalCompResult.value();

      if (config.verbose)
        llvm::outs() << "TapestryPipeline: running multi-core simulation\n";

      // Build kernel-name -> (coreInstanceIndex, timing) mapping.
      std::map<std::string, unsigned> kernelToCoreInstance;
      std::map<std::string, KernelTiming> kernelTimingMap;
      for (const auto &cs : tempSched.coreSchedules) {
        for (size_t ki = 0; ki < cs.kernelOrder.size(); ++ki) {
          // Derive core instance index from the assignment list.
          for (const auto &assign : compRef.assignments) {
            if (assign.kernelName == cs.kernelOrder[ki] &&
                assign.coreInstanceIndex >= 0) {
              kernelToCoreInstance[cs.kernelOrder[ki]] =
                  static_cast<unsigned>(assign.coreInstanceIndex);
              break;
            }
          }
          if (ki < cs.kernelTimings.size())
            kernelTimingMap[cs.kernelOrder[ki]] = cs.kernelTimings[ki];
        }
      }

      // Determine total core count from assignments.
      unsigned maxCoreId = 0;
      for (const auto &assign : compRef.assignments) {
        if (assign.coreInstanceIndex >= 0)
          maxCoreId = std::max(maxCoreId,
                               static_cast<unsigned>(assign.coreInstanceIndex));
      }

      mcsim::MultiCoreSimConfig simCfg;
      simCfg.maxCores = maxCoreId + 1;

      mcsim::MultiCoreSimSession sim(simCfg);

      // Add kernels to the simulator from the temporal schedule.
      for (const auto &cs : tempSched.coreSchedules) {
        for (size_t ki = 0; ki < cs.kernelOrder.size(); ++ki) {
          const std::string &kernelName = cs.kernelOrder[ki];

          mcsim::KernelDescriptor kd;
          kd.name = kernelName;

          auto coreIt = kernelToCoreInstance.find(kernelName);
          kd.coreId = (coreIt != kernelToCoreInstance.end())
                          ? coreIt->second : 0;

          auto timingIt = kernelTimingMap.find(kernelName);
          if (timingIt != kernelTimingMap.end())
            kd.estimatedCycles = timingIt->second.executionCycles;

          // Estimate output bytes from contracts where this kernel is producer.
          for (const auto &c : savedContracts) {
            if (c.producerKernel == kernelName)
              kd.outputBytes += c.elementCount * c.bandwidthBytesPerCycle;
          }

          // Allow interleaved NoC injection at 75% of kernel execution.
          if (kd.estimatedCycles > 0)
            kd.outputReadyCycleOffset = (kd.estimatedCycles * 3) / 4;

          sim.addKernel(kd);
        }
      }

      // Track per-core kernel index for NoC transfer descriptors.
      std::map<std::string, unsigned> kernelIndexOnCore;
      {
        std::map<unsigned, unsigned> coreKernelCount;
        for (const auto &cs : tempSched.coreSchedules) {
          for (const auto &kernelName : cs.kernelOrder) {
            auto coreIt = kernelToCoreInstance.find(kernelName);
            unsigned coreId = (coreIt != kernelToCoreInstance.end())
                                  ? coreIt->second : 0;
            kernelIndexOnCore[kernelName] = coreKernelCount[coreId]++;
          }
        }
      }

      // Add NoC transfers for cross-core contract edges.
      for (const auto &c : savedContracts) {
        auto srcIt = kernelToCoreInstance.find(c.producerKernel);
        auto dstIt = kernelToCoreInstance.find(c.consumerKernel);
        if (srcIt == kernelToCoreInstance.end() ||
            dstIt == kernelToCoreInstance.end())
          continue;
        if (srcIt->second == dstIt->second)
          continue; // Intra-core: no NoC transfer needed.

        mcsim::NocTransferDescriptor td;
        td.srcCoreId = srcIt->second;
        td.dstCoreId = dstIt->second;
        td.bytes = c.elementCount * c.bandwidthBytesPerCycle;
        td.srcKernelIndex = kernelIndexOnCore[c.producerKernel];
        sim.addNocTransfer(td);
      }

      mcsim::MultiCoreSimResult simRes = sim.run();

      // Translate into the pipeline result structure.
      PipelineSimResult pipeSimResult;
      pipeSimResult.totalGlobalCycles = simRes.totalCycles;

      uint64_t totalFlits = 0;
      for (const auto &tr : simRes.nocTransferResults)
        totalFlits += tr.bytes;
      pipeSimResult.nocStats.totalFlitsTransferred = totalFlits;

      for (const auto &kr : simRes.kernelResults) {
        PipelineCoreSimResult pcsr;
        pcsr.coreId = kr.coreId;
        pcsr.cycles = kr.cycles;
        pcsr.completed = true;
        pipeSimResult.coreResults.push_back(pcsr);
      }

      result.simResult = pipeSimResult;

      if (config.verbose) {
        llvm::outs() << "TapestryPipeline: simulation complete, "
                     << simRes.totalCycles << " total cycles, "
                     << simRes.kernelResults.size() << " kernels, "
                     << totalFlits << " NoC bytes transferred\n";
      }

      if (!simRes.success) {
        result.success = false;
        result.diagnostics = "multi-core simulation failed: " +
                             simRes.errorMessage;
        return result;
      }

      break;
    }
    case PipelineStage::RTLGEN: {
      // The RTLGEN stage uses MultiCoreSVGen to produce system-level
      // SystemVerilog from the compilation result.
      if (!internalCompResult.has_value()) {
        result.success = false;
        result.diagnostics =
            "RTLGEN stage requires a preceding COMPILE stage";
        return result;
      }

      const auto &compRef = internalCompResult.value();

      if (config.verbose)
        llvm::outs() << "TapestryPipeline: generating multi-core RTL\n";

      // Build the MultiCoreCompilationDesc from the internal compilation
      // result's assignments and ADG modules.
      svgen::MultiCoreCompilationDesc svCompilation;
      svCompilation.success = compRef.success;

      for (const auto &assign : compRef.assignments) {
        svgen::MultiCoreCoreDesc coreDesc;
        coreDesc.coreInstanceName = assign.kernelName;
        coreDesc.coreType = assign.kernelName; // Use kernel name as type
        coreDesc.adgModule = assign.coreADG;
        // Config blobs are not available from the L2Assignment (tapestry
        // namespace); leave empty -- SVGen will generate without them.
        svCompilation.coreDescs.push_back(std::move(coreDesc));
      }

      svgen::MultiCoreSVGenOptions svOpts;
      svOpts.outputDir = config.outputDir + "/rtl";
      svOpts.rtlSourceDir = config.rtlSourceDir;
      svOpts.fpIpProfile = config.svgenOpts.fpIpProfile;
      if (config.svgenOpts.meshRows > 0)
        svOpts.meshRows = config.svgenOpts.meshRows;
      if (config.svgenOpts.meshCols > 0)
        svOpts.meshCols = config.svgenOpts.meshCols;

      svgen::MultiCoreSVGenResult svResult =
          svgen::generateMultiCoreSV(svCompilation, svOpts, &context);

      // Translate into the pipeline result structure.
      PipelineRTLResult pipeRtlResult;
      pipeRtlResult.systemTopFile = svResult.systemTopFile;
      pipeRtlResult.allGeneratedFiles = svResult.allGeneratedFiles;
      result.rtlResult = pipeRtlResult;

      if (config.verbose) {
        llvm::outs() << "TapestryPipeline: RTL generation "
                     << (svResult.success ? "succeeded" : "failed")
                     << ", " << svResult.allGeneratedFiles.size()
                     << " files generated\n";
      }

      if (!svResult.success) {
        result.success = false;
        result.diagnostics = "RTL generation failed";
        return result;
      }

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
