//===-- TypeAdapters.cpp - Bridge between type namespaces ----------*- C++ -*-===//
//
// Adapter functions that convert between the three type ecosystems used in
// the Tapestry multi-core compiler.
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/TypeAdapters.h"
#include "loom/SystemCompiler/InfeasibilityCut.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>

namespace loom {

//===----------------------------------------------------------------------===//
// tapestry::SystemArchitecture -> loom::SystemArchitecture
//===----------------------------------------------------------------------===//

/// Extract FU type counts from an ADG MLIR module by walking its operations.
static std::map<std::string, unsigned>
extractFUTypeCounts(mlir::ModuleOp adgModule) {
  std::map<std::string, unsigned> counts;
  if (!adgModule)
    return counts;

  adgModule.walk([&](mlir::Operation *op) {
    llvm::StringRef opName = op->getName().getStringRef();
    // Look for PE body operations that represent FU capabilities.
    // The ADG uses fabric.pe_body containing arith/math operations.
    if (opName.starts_with("arith.") || opName.starts_with("math.")) {
      counts[opName.str()]++;
    }
  });

  return counts;
}

/// Count total PEs in an ADG module.
static unsigned countPEs(mlir::ModuleOp adgModule) {
  if (!adgModule)
    return 0;

  unsigned count = 0;
  adgModule.walk([&](mlir::Operation *op) {
    llvm::StringRef opName = op->getName().getStringRef();
    if (opName == "fabric.pe" || opName == "fabric.spatial_pe")
      ++count;
  });
  return count;
}

/// Count total FU nodes in an ADG module.
static unsigned countFUs(mlir::ModuleOp adgModule) {
  if (!adgModule)
    return 0;

  unsigned count = 0;
  adgModule.walk([&](mlir::Operation *op) {
    llvm::StringRef opName = op->getName().getStringRef();
    if (opName == "fabric.fu" || opName.starts_with("arith.") ||
        opName.starts_with("math."))
      ++count;
  });
  return count;
}

SystemArchitecture
toL1Architecture(const tapestry::SystemArchitecture &tapArch,
                 mlir::MLIRContext *ctx) {
  (void)ctx;
  SystemArchitecture arch;

  for (const auto &tapCore : tapArch.coreTypes) {
    CoreTypeSpec coreSpec;
    coreSpec.typeName = tapCore.name;
    coreSpec.instanceCount = tapCore.numInstances;
    coreSpec.spmBytes = tapCore.spmSizeBytes;
    coreSpec.numPEs = tapCore.totalPEs > 0 ? tapCore.totalPEs
                                           : countPEs(tapCore.adgModule);
    coreSpec.numFUs = tapCore.totalFUs > 0 ? tapCore.totalFUs
                                           : countFUs(tapCore.adgModule);
    coreSpec.fuTypeCounts = extractFUTypeCounts(tapCore.adgModule);

    // If the ADG does not expose explicit FU type counts, create synthetic
    // entries based on the total FU count so the L1 solver has capacity data.
    if (coreSpec.fuTypeCounts.empty() && coreSpec.numFUs > 0) {
      coreSpec.fuTypeCounts["arith.addi"] = coreSpec.numFUs;
    }

    arch.coreTypes.push_back(std::move(coreSpec));
  }

  // Default NoC spec: mesh with dimensions inferred from total core count.
  unsigned totalCores = arch.totalCoreInstances();
  if (totalCores > 0) {
    unsigned side = 1;
    while (side * side < totalCores)
      ++side;
    arch.nocSpec.meshRows = side;
    arch.nocSpec.meshCols = (totalCores + side - 1) / side;
  }

  return arch;
}

//===----------------------------------------------------------------------===//
// tapestry::KernelDesc -> loom::KernelProfile
//===----------------------------------------------------------------------===//

KernelProfile toKernelProfile(const tapestry::KernelDesc &kernelDesc,
                              mlir::MLIRContext *ctx) {
  KernelProfiler profiler;
  if (kernelDesc.dfgModule) {
    KernelProfile profile = profiler.profile(kernelDesc.dfgModule, ctx);
    if (profile.name.empty())
      profile.name = kernelDesc.name;
    return profile;
  }

  // Fallback: create a minimal profile from the descriptor fields.
  KernelProfile profile;
  profile.name = kernelDesc.name;
  profile.estimatedSPMBytes = kernelDesc.requiredMemoryBytes;
  // Use requiredFUs as a rough op-count proxy.
  if (kernelDesc.requiredFUs > 0)
    profile.requiredOps["arith.addi"] = kernelDesc.requiredFUs;
  return profile;
}

std::vector<KernelProfile>
toKernelProfiles(const std::vector<tapestry::KernelDesc> &kernels,
                 mlir::MLIRContext *ctx) {
  std::vector<KernelProfile> profiles;
  profiles.reserve(kernels.size());
  for (const auto &k : kernels)
    profiles.push_back(toKernelProfile(k, ctx));
  return profiles;
}

//===----------------------------------------------------------------------===//
// tapestry::ContractSpec -> loom::ContractSpec
//===----------------------------------------------------------------------===//

ContractSpec toL1Contract(const tapestry::ContractSpec &tapContract) {
  ContractSpec contract;
  contract.producerKernel = tapContract.producerKernel;
  contract.consumerKernel = tapContract.consumerKernel;
  contract.dataTypeName = tapContract.dataType;

  if (tapContract.elementCount > 0) {
    contract.productionRate = static_cast<int64_t>(tapContract.elementCount);
    contract.consumptionRate = static_cast<int64_t>(tapContract.elementCount);
  }

  if (tapContract.bandwidthBytesPerCycle > 0) {
    // Bandwidth hint for NoC cost estimation.
  }

  return contract;
}

std::vector<ContractSpec>
toL1Contracts(const std::vector<tapestry::ContractSpec> &tapContracts) {
  std::vector<ContractSpec> contracts;
  contracts.reserve(tapContracts.size());
  for (const auto &tc : tapContracts)
    contracts.push_back(toL1Contract(tc));
  return contracts;
}

//===----------------------------------------------------------------------===//
// Kernel DFG Map
//===----------------------------------------------------------------------===//

std::map<std::string, mlir::ModuleOp>
buildKernelDFGMap(const std::vector<tapestry::KernelDesc> &kernels) {
  std::map<std::string, mlir::ModuleOp> dfgMap;
  for (const auto &k : kernels) {
    if (k.dfgModule)
      dfgMap[k.name] = k.dfgModule;
  }
  return dfgMap;
}

//===----------------------------------------------------------------------===//
// ADG Module Lookup
//===----------------------------------------------------------------------===//

mlir::ModuleOp findCoreADG(const tapestry::SystemArchitecture &tapArch,
                           const std::string &coreTypeName) {
  for (const auto &ct : tapArch.coreTypes) {
    if (ct.name == coreTypeName)
      return ct.adgModule;
  }
  return mlir::ModuleOp();
}

void populateL2ADGs(std::vector<L2Assignment> &l2Assignments,
                    const tapestry::SystemArchitecture &tapArch) {
  for (auto &l2 : l2Assignments) {
    if (!l2.coreADG)
      l2.coreADG = findCoreADG(tapArch, l2.coreType);
  }
}

//===----------------------------------------------------------------------===//
// TapestryCompilationResult -> tapestry::BendersResult
//===----------------------------------------------------------------------===//

tapestry::BendersResult
toBendersResult(const TapestryCompilationResult &compResult,
                const std::vector<tapestry::KernelDesc> &kernels,
                const tapestry::SystemArchitecture &arch,
                unsigned iterations) {
  tapestry::BendersResult result;
  result.success = compResult.success;
  result.iterations = iterations;

  double totalCost = 0.0;
  for (const auto &cr : compResult.coreResults) {
    for (const auto &kernelName : cr.assignedKernels) {
      tapestry::L2Assignment assign;
      assign.kernelName = kernelName;

      // Find the core type index.
      for (unsigned typeIdx = 0; typeIdx < arch.coreTypes.size(); ++typeIdx) {
        if (arch.coreTypes[typeIdx].name == cr.coreType) {
          assign.coreTypeIndex = static_cast<int>(typeIdx);
          assign.coreADG = arch.coreTypes[typeIdx].adgModule;
          break;
        }
      }

      // Check if this kernel was successfully mapped.
      assign.mappingSuccess = cr.l2Result.allKernelsMapped;
      for (const auto &kr : cr.l2Result.kernelResults) {
        if (kr.kernelName == kernelName) {
          assign.mappingSuccess = kr.success;
          if (kr.success && kr.mapperResult) {
            assign.mappingCost = 1.0;
          }
          break;
        }
      }

      totalCost += assign.mappingCost;
      result.assignments.push_back(std::move(assign));
    }
  }

  result.totalCost = totalCost;
  return result;
}

//===----------------------------------------------------------------------===//
// JSON Serialization
//===----------------------------------------------------------------------===//

bool serializeResultJSON(const TapestryCompilationResult &result,
                         unsigned iterations,
                         double compilationTimeSec,
                         const std::string &outputPath) {
  // Ensure parent directory exists.
  llvm::StringRef dir = llvm::sys::path::parent_path(outputPath);
  if (!dir.empty()) {
    std::error_code ec = llvm::sys::fs::create_directories(dir);
    if (ec) {
      llvm::errs() << "serializeResultJSON: cannot create directory '"
                   << dir << "': " << ec.message() << "\n";
      return false;
    }
  }

  // Build JSON using LLVM's JSON support.
  llvm::json::Object root;
  root["success"] = result.success;
  root["iterations"] = static_cast<int64_t>(iterations);
  root["compilationTimeSec"] = compilationTimeSec;

  // Per-core results.
  llvm::json::Array coreResultsArr;
  for (const auto &cr : result.coreResults) {
    llvm::json::Object coreObj;
    coreObj["coreInstanceName"] = cr.coreInstanceName;
    coreObj["coreType"] = cr.coreType;
    coreObj["allKernelsMapped"] = cr.l2Result.allKernelsMapped;

    llvm::json::Array assignedKernelsArr;
    for (const auto &kn : cr.assignedKernels)
      assignedKernelsArr.push_back(kn);
    coreObj["assignedKernels"] = std::move(assignedKernelsArr);

    llvm::json::Array kernelResultsArr;
    for (const auto &kr : cr.l2Result.kernelResults) {
      llvm::json::Object krObj;
      krObj["kernelName"] = kr.kernelName;
      krObj["success"] = kr.success;
      if (kr.cut) {
        llvm::json::Object cutObj;
        cutObj["kernelName"] = kr.cut->kernelName;
        cutObj["coreType"] = kr.cut->coreType;
        cutObj["reason"] = cutReasonToString(kr.cut->reason);
        krObj["cut"] = std::move(cutObj);
      }
      kernelResultsArr.push_back(std::move(krObj));
    }
    coreObj["kernelResults"] = std::move(kernelResultsArr);

    // Cost summary metrics.
    const auto &cs = cr.l2Result.costSummary;
    llvm::json::Object metricsObj;
    metricsObj["totalPEUtilization"] = cs.totalPEUtilization;
    metricsObj["totalSPMUtilization"] = cs.totalSPMUtilization;
    metricsObj["routingPressure"] = cs.routingPressure;
    coreObj["metrics"] = std::move(metricsObj);

    coreResultsArr.push_back(std::move(coreObj));
  }
  root["coreResults"] = std::move(coreResultsArr);

  // NoC schedule summary.
  llvm::json::Object nocObj;
  nocObj["totalTransferCycles"] =
      static_cast<int64_t>(result.finalNoCSchedule.totalTransferCycles);
  nocObj["maxLinkUtilization"] = result.finalNoCSchedule.maxLinkUtilization;
  nocObj["hasContention"] = result.finalNoCSchedule.hasContention;
  root["nocSchedule"] = std::move(nocObj);

  // Diagnostics.
  llvm::json::Array diagnosticsArr;
  root["diagnostics"] = std::move(diagnosticsArr);

  // Write to file.
  std::error_code ec;
  llvm::raw_fd_ostream outFile(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "serializeResultJSON: cannot open '" << outputPath
                 << "': " << ec.message() << "\n";
    return false;
  }

  llvm::json::Value jsonVal(std::move(root));
  outFile << llvm::formatv("{0:2}", jsonVal) << "\n";
  return true;
}

} // namespace loom
