//===-- HWOuterOptimizer.cpp - System-level hardware optimizer ---*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/HWOuterOptimizer.h"
#include "loom/ADG/DomainADGGenerator.h"
#include "loom/ADG/KHGGenerator.h"
#include "loom/ADG/SystemADGBuilder.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace loom {

// ---------------------------------------------------------------------------
// CoreRole string conversions
// ---------------------------------------------------------------------------

const char *coreRoleToString(CoreRole role) {
  switch (role) {
  case CoreRole::FP_HEAVY:
    return "fp_heavy";
  case CoreRole::CONTROL_HEAVY:
    return "control_heavy";
  case CoreRole::MEMORY_HEAVY:
    return "memory_heavy";
  case CoreRole::BALANCED:
    return "balanced";
  }
  return "balanced";
}

CoreRole coreRoleFromString(const std::string &s) {
  if (s == "fp_heavy")
    return CoreRole::FP_HEAVY;
  if (s == "control_heavy")
    return CoreRole::CONTROL_HEAVY;
  if (s == "memory_heavy")
    return CoreRole::MEMORY_HEAVY;
  return CoreRole::BALANCED;
}

// ---------------------------------------------------------------------------
// HWOuterOptimizer
// ---------------------------------------------------------------------------

HWOuterOptimizer::HWOuterOptimizer(const HWOuterOptimizerOptions &options)
    : options_(options) {}

HWOuterOptimizerResult
HWOuterOptimizer::optimize(const std::vector<ContractSpec> &contracts,
                           const std::vector<KernelProfile> &kernelProfiles) {
  HWOuterOptimizerResult result;
  auto wallStart = std::chrono::steady_clock::now();

  // Create a temporary working directory
  std::string workDir = options_.workDir;
  bool createdTmpDir = false;
  if (workDir.empty()) {
    // Use a temporary directory
    std::error_code ec;
    auto tmpPath = std::filesystem::temp_directory_path(ec);
    if (ec) {
      result.diagnostics = "Failed to get temp directory: " + ec.message();
      return result;
    }
    workDir = (tmpPath / "outer_hw_XXXXXX").string();
    std::filesystem::create_directories(workDir, ec);
    if (ec) {
      result.diagnostics = "Failed to create work directory: " + ec.message();
      return result;
    }
    createdTmpDir = true;
  } else {
    std::error_code ec;
    std::filesystem::create_directories(workDir, ec);
  }

  // Write workload profile JSON for the Python optimizer
  std::string workloadPath =
      writeWorkloadJSON(contracts, kernelProfiles, workDir);
  if (workloadPath.empty()) {
    result.diagnostics = "Failed to write workload JSON";
    return result;
  }

  // Output path for the topology spec
  std::string outputPath = workDir + "/topology_spec.json";

  // Build the Python command
  std::string pythonBin = options_.pythonBin;
  std::string scriptPath = options_.pythonScriptPath;
  if (scriptPath.empty()) {
    result.diagnostics = "Python script path not set";
    return result;
  }

  std::ostringstream cmd;
  cmd << pythonBin << " -c \""
      << "import sys; sys.path.insert(0, '" << scriptPath << "/../..'); "
      << "from scripts.dse.hw_outer_optimizer import run_outer_hw; "
      << "from scripts.dse.proxy_model import WorkloadProfile, "
      << "KernelProfile, ContractEdge; "
      << "import json; "
      << "wl_data = json.load(open('" << workloadPath << "')); "
      << "kernels = [KernelProfile("
      << "name=k['name'], "
      << "op_histogram=k.get('op_histogram', {}), "
      << "memory_footprint_bytes=k.get('memory_footprint_bytes', 0), "
      << "loads_per_iter=k.get('loads_per_iter', 0), "
      << "stores_per_iter=k.get('stores_per_iter', 0), "
      << "dfg_node_count=k.get('dfg_node_count', 0), "
      << "assigned_core_type_idx=k.get('assigned_core_type_idx', 0)"
      << ") for k in wl_data.get('kernels', [])]; "
      << "contracts = [ContractEdge("
      << "producer=c['producer'], "
      << "consumer=c['consumer'], "
      << "production_rate=c.get('production_rate', 1.0), "
      << "element_size_bytes=c.get('element_size_bytes', 4)"
      << ") for c in wl_data.get('contracts', [])]; "
      << "wl = WorkloadProfile(kernels=kernels, contracts=contracts); "
      << "r = run_outer_hw(wl, max_iterations="
      << options_.maxIterations
      << ", seed=" << options_.seed
      << ", output_path='" << outputPath << "')\"";

  // Execute the Python optimizer
  if (options_.verbose) {
    llvm::errs() << "OUTER-HW: running Python optimizer\n";
  }

  int exitCode = std::system(cmd.str().c_str());

  if (exitCode != 0) {
    result.diagnostics =
        "Python optimizer exited with code " + std::to_string(exitCode);
    return result;
  }

  // Parse the output topology spec
  if (!parseTopologyJSON(outputPath, result.topology)) {
    result.diagnostics = "Failed to parse topology spec from " + outputPath;
    return result;
  }

  result.success = true;

  auto wallEnd = std::chrono::steady_clock::now();
  result.wallTimeSec =
      std::chrono::duration<double>(wallEnd - wallStart).count();

  // Clean up temp directory if we created it
  if (createdTmpDir) {
    std::error_code ec;
    std::filesystem::remove_all(workDir, ec);
  }

  return result;
}

std::string HWOuterOptimizer::generateSystemMLIR(
    const HWOuterOptimizerResult &result, mlir::MLIRContext *ctx) {
  if (!result.success || !ctx)
    return "";

  const auto &topo = result.topology;
  adg::SystemADGBuilder builder("system");

  // Configure NoC
  adg::NoCSpec nocSpec;
  if (topo.nocTopology == "mesh")
    nocSpec.topology = adg::NoCSpec::MESH;
  else if (topo.nocTopology == "ring")
    nocSpec.topology = adg::NoCSpec::RING;
  else
    nocSpec.topology = adg::NoCSpec::HIERARCHICAL;
  nocSpec.linkBandwidth = topo.nocBandwidth;
  builder.setNoCSpec(nocSpec);

  // Configure shared memory
  adg::SharedMemorySpec memSpec;
  memSpec.l2SizeBytes = topo.l2TotalSizeKB * 1024;
  memSpec.numBanks = topo.l2BankCount;
  builder.setSharedMemorySpec(memSpec);

  // Register core types and instantiate cores using KHGGenerator for real ADGs
  for (const auto &entry : topo.coreLibrary.entries) {
    std::string typeName = "core_type_" + std::to_string(entry.typeIndex);

    // Try to generate a real ADG from the type ID. First check KHG types
    // (combinatorial), then domain-specific types (D1-D6).
    std::string mlirText;
    if (adg::isValidKHGTypeId(entry.typeId)) {
      adg::KHGTypeParams khgParams = adg::paramsFromTypeId(entry.typeId);
      mlirText = adg::generateKHGADG(khgParams);
    } else if (adg::isValidDomainTypeId(entry.typeId)) {
      adg::DomainTypeParams domainParams =
          adg::domainParamsFromTypeId(entry.typeId);
      mlirText = adg::generateDomainADG(domainParams);
    }

    // Fall back to a minimal ADG only if the type ID is unrecognized
    if (mlirText.empty())
      mlirText = "module @" + typeName + " {}\n";

    // Parse the MLIR text into a ModuleOp
    llvm::SourceMgr srcMgr;
    srcMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBuffer(mlirText), llvm::SMLoc());
    auto parsedModule = mlir::parseSourceFile<mlir::ModuleOp>(srcMgr, ctx);
    if (!parsedModule)
      continue;

    auto handle = builder.registerCoreType(typeName, *parsedModule);

    // Instantiate the requested number of cores
    for (unsigned inst = 0; inst < entry.instanceCount; ++inst) {
      // Find the placement for this core
      int row = 0, col = 0;
      for (const auto &p : topo.corePlacements) {
        if (p.typeIndex == entry.typeIndex && p.instanceId == inst) {
          row = p.row;
          col = p.col;
          break;
        }
      }

      std::string instName =
          typeName + "_inst" + std::to_string(inst);
      builder.instantiateCore(handle, instName, row, col);
    }
  }

  mlir::ModuleOp builtModule = builder.build();
  if (!builtModule)
    return "";

  // Print the built module to a string
  std::string mlirText;
  llvm::raw_string_ostream os(mlirText);
  builtModule->print(os);
  return mlirText;
}

// ---------------------------------------------------------------------------
// Workload JSON serialization
// ---------------------------------------------------------------------------

std::string HWOuterOptimizer::writeWorkloadJSON(
    const std::vector<ContractSpec> &contracts,
    const std::vector<KernelProfile> &kernelProfiles,
    const std::string &outputDir) {
  llvm::json::Object root;

  // Serialize kernel profiles
  llvm::json::Array kernelsArray;
  for (const auto &kp : kernelProfiles) {
    llvm::json::Object kObj;
    kObj["name"] = kp.name;
    kObj["estimatedSPMBytes"] = static_cast<int64_t>(kp.estimatedSPMBytes);
    kObj["estimatedMinII"] = static_cast<int64_t>(kp.estimatedMinII);
    kObj["dfg_node_count"] = static_cast<int64_t>(kp.totalOpCount());
    kObj["memory_footprint_bytes"] =
        static_cast<int64_t>(kp.estimatedSPMBytes);

    // Build op histogram
    llvm::json::Object opHist;
    for (const auto &[opType, count] : kp.requiredOps) {
      opHist[opType] = static_cast<int64_t>(count);
    }
    kObj["op_histogram"] = std::move(opHist);

    // Approximate loads/stores from op histogram
    int64_t loads = 0;
    int64_t stores = 0;
    for (const auto &[opType, count] : kp.requiredOps) {
      std::string opLower = opType;
      std::transform(opLower.begin(), opLower.end(), opLower.begin(),
                     ::tolower);
      if (opLower.find("load") != std::string::npos)
        loads += count;
      else if (opLower.find("store") != std::string::npos)
        stores += count;
    }
    kObj["loads_per_iter"] = loads;
    kObj["stores_per_iter"] = stores;

    kernelsArray.push_back(std::move(kObj));
  }
  root["kernels"] = std::move(kernelsArray);

  // Serialize contracts
  llvm::json::Array contractsArray;
  for (const auto &cs : contracts) {
    llvm::json::Object cObj;
    cObj["producer"] = cs.producerKernel;
    cObj["consumer"] = cs.consumerKernel;

    if (cs.productionRate.has_value())
      cObj["production_rate"] =
          static_cast<double>(cs.productionRate.value());
    else
      cObj["production_rate"] = 1.0;

    cObj["element_size_bytes"] =
        static_cast<int64_t>(estimateElementSize(cs.dataTypeName));

    contractsArray.push_back(std::move(cObj));
  }
  root["contracts"] = std::move(contractsArray);

  // Serialize 30-type library metadata for the Python optimizer.
  // Build the canonical type list: D1-D6 (domain-specific) followed by
  // 24 KHG combinatorial types, matching Python core_type_library ordering.
  root["num_library_types"] = static_cast<int64_t>(NUM_LIBRARY_TYPES);

  llvm::json::Array typeMetadataArray;
  unsigned typeIdx = 0;

  // Helper lambda to serialize a CoreTypeDesignParams-like record.
  auto serializeTypeEntry = [&](unsigned idx, const std::string &typeId,
                                unsigned arrayRows, unsigned arrayCols,
                                unsigned fuAlu, unsigned fuMul, unsigned fuFp,
                                unsigned fuMem, unsigned spmKB,
                                bool isTemporal, unsigned instrSlots,
                                unsigned numRegs, unsigned dataWidth,
                                bool hasFP, bool hasInt,
                                const std::string &computeMix, bool hasSPM,
                                const std::string &arraySize) {
    llvm::json::Object tObj;
    tObj["type_index"] = static_cast<int64_t>(idx);
    tObj["type_id"] = typeId;
    tObj["compute_mix"] = computeMix;
    tObj["has_spm"] = hasSPM;
    tObj["array_size"] = arraySize;
    tObj["array_rows"] = static_cast<int64_t>(arrayRows);
    tObj["array_cols"] = static_cast<int64_t>(arrayCols);
    tObj["fu_alu_count"] = static_cast<int64_t>(fuAlu);
    tObj["fu_mul_count"] = static_cast<int64_t>(fuMul);
    tObj["fu_fp_count"] = static_cast<int64_t>(fuFp);
    tObj["fu_mem_count"] = static_cast<int64_t>(fuMem);
    tObj["spm_size_kb"] = static_cast<int64_t>(spmKB);
    tObj["is_temporal"] = isTemporal;
    tObj["instruction_slots"] = static_cast<int64_t>(instrSlots);
    tObj["num_registers"] = static_cast<int64_t>(numRegs);
    tObj["data_width"] = static_cast<int64_t>(dataWidth);
    tObj["has_fp"] = hasFP;
    tObj["has_int"] = hasInt;
    typeMetadataArray.push_back(std::move(tObj));
  };

  // Domain-specific types D1-D6
  for (const auto &dp : adg::allDomainTypes()) {
    // Domain types use a mixed compute classification based on FU ratios
    std::string computeMix = "int_mix";
    if (dp.fuFpCount >= dp.fuAluCount)
      computeMix = "fp_mix";
    else if (dp.fuMemCount > dp.fuAluCount && dp.fuMemCount > dp.fuFpCount)
      computeMix = "mem_mix";

    std::string arraySz = (dp.arrayRows >= 6) ? "large" : "small";

    serializeTypeEntry(typeIdx++, dp.typeId, dp.arrayRows, dp.arrayCols,
                       dp.fuAluCount, dp.fuMulCount, dp.fuFpCount,
                       dp.fuMemCount, dp.spmSizeKB, dp.isTemporal,
                       dp.instructionSlots, dp.numRegisters, dp.dataWidth,
                       dp.hasFP(), /*hasInt=*/true, computeMix,
                       dp.hasSPM(), arraySz);
  }

  // KHG combinatorial types (24 types)
  for (const auto &kp : adg::allKHGTypes()) {
    std::string computeMix = "int_mix";
    if (kp.computeMix == adg::KHGComputeMix::FP_HEAVY)
      computeMix = "fp_mix";
    else if (kp.computeMix == adg::KHGComputeMix::MIXED)
      computeMix = "mem_mix";

    std::string arraySz =
        (kp.arraySize == adg::KHGArraySize::SIZE_12) ? "large" : "small";
    bool hasSPM = (kp.spmPresence == adg::KHGSPMPresence::WITH_SPM);
    bool isTemporal = (kp.peKind == adg::KHGPEKind::TEMPORAL);
    bool hasFP = (kp.fuFpCount > 0);

    serializeTypeEntry(typeIdx++, kp.typeId, kp.arrayRows, kp.arrayCols,
                       kp.fuAluCount, kp.fuMulCount, kp.fuFpCount,
                       /*fuMem=*/2, kp.spmSizeKB, isTemporal,
                       kp.instructionSlots, kp.numRegisters, kp.dataWidth,
                       hasFP, /*hasInt=*/true, computeMix, hasSPM, arraySz);
  }

  root["type_metadata"] = std::move(typeMetadataArray);

  // Write to file
  std::string path = outputDir + "/workload.json";
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec) {
    return "";
  }
  os << llvm::json::Value(std::move(root));
  os.flush();

  return path;
}

// ---------------------------------------------------------------------------
// Topology JSON parsing
// ---------------------------------------------------------------------------

bool HWOuterOptimizer::parseTopologyJSON(const std::string &jsonPath,
                                         SystemTopologySpec &outSpec) {
  auto bufOrErr = llvm::MemoryBuffer::getFile(jsonPath);
  if (!bufOrErr)
    return false;

  auto parsed = llvm::json::parse((*bufOrErr)->getBuffer());
  if (!parsed)
    return false;

  auto *root = parsed->getAsObject();
  if (!root)
    return false;

  // Parse NoC
  if (auto *noc = root->getObject("noc")) {
    if (auto topo = noc->getString("topology"))
      outSpec.nocTopology = topo->str();
    if (auto bw = noc->getInteger("bandwidth"))
      outSpec.nocBandwidth = static_cast<unsigned>(*bw);
    if (auto rows = noc->getInteger("mesh_rows"))
      outSpec.meshRows = static_cast<unsigned>(*rows);
    if (auto cols = noc->getInteger("mesh_cols"))
      outSpec.meshCols = static_cast<unsigned>(*cols);
  }

  // Parse shared memory
  if (auto *mem = root->getObject("shared_memory")) {
    if (auto sz = mem->getInteger("l2_total_size_kb"))
      outSpec.l2TotalSizeKB = static_cast<uint64_t>(*sz);
    if (auto banks = mem->getInteger("l2_bank_count"))
      outSpec.l2BankCount = static_cast<unsigned>(*banks);
  }

  // Parse core library: accept both the object form ({"core_types": [...]})
  // and the flat array form that Python system_graph_generator outputs.
  if (auto *lib = root->getObject("core_library")) {
    parseCoreLibraryFromObject(lib, outSpec.coreLibrary);
  } else if (auto *libArr = root->getArray("core_library")) {
    parseCoreLibraryFromArray(libArr, outSpec.coreLibrary);
  }

  // Parse core placements
  if (auto *placements = root->getArray("core_placement")) {
    for (const auto &pVal : *placements) {
      auto *pObj = pVal.getAsObject();
      if (!pObj)
        continue;

      CorePlacement cp;
      if (auto t = pObj->getInteger("type"))
        cp.typeIndex = static_cast<unsigned>(*t);
      if (auto i = pObj->getInteger("instance"))
        cp.instanceId = static_cast<unsigned>(*i);
      if (auto r = pObj->getInteger("row"))
        cp.row = static_cast<int>(*r);
      if (auto c = pObj->getInteger("col"))
        cp.col = static_cast<int>(*c);

      outSpec.corePlacements.push_back(cp);
    }
  }

  return true;
}

/// Parse a CoreTypeDesignParams from a JSON object containing per-type
/// design metadata (FU counts, array geometry, SPM config, etc.).
static void parseDesignParams(const llvm::json::Object *tObj,
                              CoreTypeDesignParams &dp) {
  if (!tObj)
    return;

  // Check for nested "design_params" object first, then fall back to
  // top-level fields (flat format from writeWorkloadJSON).
  const llvm::json::Object *src = tObj->getObject("design_params");
  if (!src)
    src = tObj;

  if (auto v = src->getInteger("array_rows"))
    dp.arrayRows = static_cast<unsigned>(*v);
  if (auto v = src->getInteger("array_cols"))
    dp.arrayCols = static_cast<unsigned>(*v);
  if (auto v = src->getInteger("fu_alu_count"))
    dp.fuAluCount = static_cast<unsigned>(*v);
  if (auto v = src->getInteger("fu_mul_count"))
    dp.fuMulCount = static_cast<unsigned>(*v);
  if (auto v = src->getInteger("fu_fp_count"))
    dp.fuFpCount = static_cast<unsigned>(*v);
  if (auto v = src->getInteger("fu_mem_count"))
    dp.fuMemCount = static_cast<unsigned>(*v);
  if (auto v = src->getInteger("spm_size_kb"))
    dp.spmSizeKB = static_cast<unsigned>(*v);
  if (auto v = src->getBoolean("is_temporal"))
    dp.isTemporal = *v;
  if (auto v = src->getInteger("instruction_slots"))
    dp.instructionSlots = static_cast<unsigned>(*v);
  if (auto v = src->getInteger("num_registers"))
    dp.numRegisters = static_cast<unsigned>(*v);
  if (auto v = src->getInteger("data_width"))
    dp.dataWidth = static_cast<unsigned>(*v);
  if (auto v = src->getBoolean("has_fp"))
    dp.hasFP = *v;
  if (auto v = src->getBoolean("has_int"))
    dp.hasInt = *v;
}

bool HWOuterOptimizer::parseCoreLibraryFromObject(
    const llvm::json::Object *root, CoreTypeLibrary &outLibrary) {
  if (!root)
    return false;

  auto *types = root->getArray("core_types");
  if (!types)
    return false;

  outLibrary.entries.clear();
  for (const auto &tVal : *types) {
    auto *tObj = tVal.getAsObject();
    if (!tObj)
      continue;

    CoreTypeLibraryEntry entry;
    if (auto idx = tObj->getInteger("type_index"))
      entry.typeIndex = static_cast<unsigned>(*idx);
    if (auto id = tObj->getString("type_id"))
      entry.typeId = id->str();
    if (auto role = tObj->getString("role"))
      entry.role = coreRoleFromString(role->str());
    if (auto count = tObj->getInteger("instance_count"))
      entry.instanceCount = static_cast<unsigned>(*count);
    if (auto pes = tObj->getInteger("min_pes"))
      entry.minPEs = static_cast<unsigned>(*pes);
    if (auto spm = tObj->getInteger("min_spm_kb"))
      entry.minSPMKB = static_cast<unsigned>(*spm);

    // Parse combinatorial dimensions
    if (auto cm = tObj->getString("compute_mix")) {
      std::string cmStr = cm->str();
      if (cmStr == "fp_mix")
        entry.computeMix = ComputeMix::FP_MIX;
      else if (cmStr == "mem_mix")
        entry.computeMix = ComputeMix::MEM_MIX;
      else
        entry.computeMix = ComputeMix::INT_MIX;
    }
    if (auto spm = tObj->getBoolean("has_spm"))
      entry.hasSPM = *spm;
    if (auto sz = tObj->getString("array_size")) {
      entry.arraySize =
          (sz->str() == "large") ? ArraySize::LARGE : ArraySize::SMALL;
    }

    if (auto *fuTypes = tObj->getArray("required_fu_types")) {
      for (const auto &fuVal : *fuTypes) {
        if (auto fuStr = fuVal.getAsString())
          entry.requiredFUTypes.push_back(fuStr->str());
      }
    }

    if (auto *kernels = tObj->getArray("assigned_kernels")) {
      for (const auto &kVal : *kernels) {
        if (auto kStr = kVal.getAsString())
          entry.assignedKernels.push_back(kStr->str());
      }
    }

    parseDesignParams(tObj, entry.designParams);

    outLibrary.entries.push_back(std::move(entry));
  }

  return true;
}

bool HWOuterOptimizer::parseCoreLibraryFromArray(
    const llvm::json::Array *libArr, CoreTypeLibrary &outLibrary) {
  if (!libArr)
    return false;

  outLibrary.entries.clear();
  for (const auto &tVal : *libArr) {
    auto *tObj = tVal.getAsObject();
    if (!tObj)
      continue;

    CoreTypeLibraryEntry entry;
    if (auto idx = tObj->getInteger("type_index"))
      entry.typeIndex = static_cast<unsigned>(*idx);
    if (auto id = tObj->getString("type_id"))
      entry.typeId = id->str();
    if (auto role = tObj->getString("role"))
      entry.role = coreRoleFromString(role->str());
    if (auto count = tObj->getInteger("instance_count"))
      entry.instanceCount = static_cast<unsigned>(*count);
    if (auto pes = tObj->getInteger("min_pes"))
      entry.minPEs = static_cast<unsigned>(*pes);
    if (auto spm = tObj->getInteger("min_spm_kb"))
      entry.minSPMKB = static_cast<unsigned>(*spm);

    // Parse combinatorial dimensions
    if (auto cm = tObj->getString("compute_mix")) {
      std::string cmStr = cm->str();
      if (cmStr == "fp_mix")
        entry.computeMix = ComputeMix::FP_MIX;
      else if (cmStr == "mem_mix")
        entry.computeMix = ComputeMix::MEM_MIX;
      else
        entry.computeMix = ComputeMix::INT_MIX;
    }
    if (auto spm = tObj->getBoolean("has_spm"))
      entry.hasSPM = *spm;
    if (auto sz = tObj->getString("array_size")) {
      entry.arraySize =
          (sz->str() == "large") ? ArraySize::LARGE : ArraySize::SMALL;
    }

    if (auto *fuTypes = tObj->getArray("required_fu_types")) {
      for (const auto &fuVal : *fuTypes) {
        if (auto fuStr = fuVal.getAsString())
          entry.requiredFUTypes.push_back(fuStr->str());
      }
    }

    if (auto *kernels = tObj->getArray("assigned_kernels")) {
      for (const auto &kVal : *kernels) {
        if (auto kStr = kVal.getAsString())
          entry.assignedKernels.push_back(kStr->str());
      }
    }

    parseDesignParams(tObj, entry.designParams);

    outLibrary.entries.push_back(std::move(entry));
  }

  return !outLibrary.entries.empty();
}

} // namespace loom
