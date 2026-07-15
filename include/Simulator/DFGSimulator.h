#ifndef LOOM_SIMULATOR_DFG_SIMULATOR_H
#define LOOM_SIMULATOR_DFG_SIMULATOR_H

#include "Simulator/OperationCostModel.h"
#include "Simulator/OperationSemantics.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <string>

namespace loom {
namespace sim {

struct DFGRuntimeArg {
  unsigned index = 0;
  std::string value;
};

struct DFGMemoryArg {
  unsigned index = 0;
  std::int64_t byteOffset = 0;
  std::string values;
};

struct DFGGlobalMemoryArg {
  std::string symbol;
  std::int64_t byteOffset = 0;
  std::string values;
};

struct DFGSimulationOptions {
  std::string graphName;
  std::string workloadName;
  llvm::SmallVector<DFGRuntimeArg> args;
  llvm::SmallVector<DFGMemoryArg> memories;
  llvm::SmallVector<DFGGlobalMemoryArg> globalMemories;
  std::uint64_t maxEventSteps = 100000;
};

struct DFGSimulationReport {
  std::string schemaVersion = "2.1";
  std::string kind = "dfg_sim_report";
  std::string workload;
  std::string graph;
  std::string status;
  std::string metricDefinition =
      "weighted_operations_plus_library_work_diversity_and_address.v1";
  std::string operationSemanticsSource = kOperationSemanticsSource;
  std::string operationCostModelSource = kOperationCostModelSource;
  std::uint64_t operationCostScore = 0;
  std::uint64_t weightedOperationScore = 0;
  std::uint64_t modeledLibraryScore = 0;
  std::uint64_t operationDiversityScore = 0;
  std::uint64_t memoryAddressScore = 0;
  std::uint64_t wavefrontSteps = 0;
  std::uint64_t eventCount = 0;
  std::uint64_t dynamicWorkItems = 0;
  std::map<std::string, std::uint64_t> operationFireCounts;
  std::map<std::string, std::uint64_t> modeledLibraryCalls;
  llvm::SmallVector<std::string> finalOutputs;
  std::map<std::string, llvm::SmallVector<std::string>> finalMemoryState;
  llvm::SmallVector<std::string> diagnostics;
};

llvm::Expected<DFGSimulationReport>
simulateDataflowGraph(::mlir::ModuleOp module,
                      const DFGSimulationOptions &options);

llvm::Error writeDFGSimulationReportJson(llvm::StringRef outputPath,
                                         const DFGSimulationReport &report);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_DFG_SIMULATOR_H
