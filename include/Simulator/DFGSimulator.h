#ifndef LOOM_SIMULATOR_DFG_SIMULATOR_H
#define LOOM_SIMULATOR_DFG_SIMULATOR_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace loom {
namespace sim {

struct DFGRuntimeArg {
  unsigned index = 0;
  std::string value;
};

struct DFGSimulationOptions {
  std::string graphName;
  std::string workloadName;
  llvm::SmallVector<DFGRuntimeArg> args;
  std::uint64_t maxEventSteps = 100000;
};

struct DFGSimulationReport {
  int schemaVersion = 1;
  std::string kind = "dfg_sim_report";
  std::string workload;
  std::string graph;
  std::string status;
  std::string metricDefinition = "optimistic_event_steps";
  std::uint64_t optimisticCycles = 0;
  std::uint64_t eventCount = 0;
  llvm::SmallVector<std::string> finalOutputs;
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
