#ifndef LOOM_SIMULATOR_DFG_SIMULATOR_H
#define LOOM_SIMULATOR_DFG_SIMULATOR_H

#include "Simulator/OperationSemantics.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>

namespace dataflow {
class CanonicalDataflowArtifact;
struct RootedGraphLaunchRef;
} // namespace dataflow

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

struct DFGSimulationOptions {
  std::string graphName;
  std::string workloadName;
  llvm::SmallVector<DFGRuntimeArg> args;
  llvm::SmallVector<DFGMemoryArg> memories;
  std::uint64_t invocations = 1;
  std::uint64_t maxEventSteps = 100000;
  std::optional<std::chrono::steady_clock::time_point> executionDeadline;
};

struct DFGSimulationReport {
  std::string schemaVersion = "3.0";
  std::string kind = "dfg_sim_report";
  std::string workload;
  std::string graph;
  std::string status;
  std::string operationSemanticsSource = kOperationSemanticsSource;
  std::uint64_t wavefrontSteps = 0;
  std::uint64_t eventCount = 0;
  std::uint64_t dynamicWorkItems = 0;
  std::map<dataflow::OperationSchemaId, std::uint64_t> operationFireCounts;
  std::map<std::string, std::uint64_t> modeledLibraryCalls;
  llvm::SmallVector<std::string> finalOutputs;
  llvm::SmallVector<llvm::SmallVector<std::string>> finalStreamOutputs;
  std::map<std::string, llvm::SmallVector<std::string>> finalMemoryState;
  std::map<std::string, std::string> finalMemoryRoots;
  llvm::SmallVector<std::string> diagnostics;
};

/// A run accepted by the typed DFG provider but unable to satisfy its graph
/// retirement contract. The report remains transient execution evidence. A
/// source-backed semantic gate may classify this exact outcome as a candidate
/// mismatch without relabeling execution limits or provider failures.
class NonRetiredDFGExecutionError final
    : public llvm::ErrorInfo<NonRetiredDFGExecutionError> {
public:
  static char ID;

  explicit NonRetiredDFGExecutionError(DFGSimulationReport report);

  const DFGSimulationReport &report() const { return report_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  DFGSimulationReport report_;
};

/// One successfully retired DFG execution. Non-retired and rejected runs are
/// available through simulateDfgWorkload; this exact API returns observations
/// only when the graph has satisfied its retirement contract.
struct RetiredDFGSimulation {
  DFGSimulationReport report;
  SpatialFunctionalObservations observations;
};

/// Disposable execution cache for repeated activations of one exact rooted
/// graph. The plan owns a strict import of the Canonical Dataflow artifact and
/// derives every provider, channel, and actor entry from that owner. It is not
/// persistent, does not affect artifact identity, and carries no dynamic run
/// state between activations.
class PreparedDfgExecution {
public:
  PreparedDfgExecution(PreparedDfgExecution &&) noexcept;
  PreparedDfgExecution &operator=(PreparedDfgExecution &&) noexcept;
  ~PreparedDfgExecution();

  PreparedDfgExecution(const PreparedDfgExecution &) = delete;
  PreparedDfgExecution &operator=(const PreparedDfgExecution &) = delete;

private:
  struct Impl;
  explicit PreparedDfgExecution(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;

  friend llvm::Expected<PreparedDfgExecution>
  prepareDfgExecution(const dataflow::CanonicalDataflowArtifact &,
                      const dataflow::RootedGraphLaunchRef &);
  friend llvm::Expected<RetiredDFGSimulation> simulateRetiredDfgWorkload(
      const PreparedDfgExecution &, const CanonicalSimulationWorkload &,
      const CanonicalSimulationRuntimeInput &, std::uint64_t,
      std::optional<std::chrono::steady_clock::time_point>);
};

llvm::Expected<PreparedDfgExecution>
prepareDfgExecution(const dataflow::CanonicalDataflowArtifact &program,
                    const dataflow::RootedGraphLaunchRef &launch);

llvm::Expected<DFGSimulationReport>
simulateDataflowGraph(::mlir::ModuleOp module,
                      const DFGSimulationOptions &options);

/// Executes one exact spatial workload against its finalized Canonical
/// Dataflow owner. The workload and runtime input are admitted through their
/// shared typed wire; CLI strings are not an intermediate representation.
llvm::Expected<DFGSimulationReport>
simulateDfgWorkload(const dataflow::CanonicalDataflowArtifact &program,
                    const CanonicalSimulationWorkload &workload,
                    const CanonicalSimulationRuntimeInput &runtimeInput,
                    std::uint64_t maxEventSteps = 100000);

llvm::Expected<RetiredDFGSimulation> simulateRetiredDfgWorkload(
    const dataflow::CanonicalDataflowArtifact &program,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxEventSteps = 100000,
    std::optional<std::chrono::steady_clock::time_point> executionDeadline =
        std::nullopt);

llvm::Expected<RetiredDFGSimulation> simulateRetiredDfgWorkload(
    const PreparedDfgExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxEventSteps = 100000,
    std::optional<std::chrono::steady_clock::time_point> executionDeadline =
        std::nullopt);

llvm::Error writeDFGSimulationReportJson(llvm::StringRef outputPath,
                                         const DFGSimulationReport &report);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_DFG_SIMULATOR_H
