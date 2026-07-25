//===- DFGSimulatorReport.cpp - What one DFG run reports ------------------===//
//
// One owner for what a finished run projects outward: the report status a
// retained RunFailure projects to, the observations such a run may still
// export, the derived cost counters, and the legacy JSON encoding of all of
// them.
//
// The run driver keeps its own lifecycle classification, so a run that
// retained no failure is named there rather than here. This module only
// overrides that name when the run failed at runtime.
//
//===----------------------------------------------------------------------===//

#include "DFGSimulatorInternal.h"
#include "Simulator/DFGSimulator.h"

#include "Simulator/OperationCostModel.h"

#include "Dataflow/IR/DataflowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>

using namespace loom::sim;
using namespace loom::sim::detail;

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {

bool applyRunFailureTerminal(const SimulatorState &state,
                             DFGSimulationReport &report) {
  switch (state.failure) {
  case RunFailure::None:
    return false;
  case RunFailure::UnsupportedCapability:
    report.status = "unsupported";
    return true;
  case RunFailure::ProviderInvariant:
    report.status = "execution_failed";
    return true;
  }
  llvm_unreachable("the run failure kind is closed");
}

static std::shared_ptr<MemoryValue> memoryForValue(SimulatorState &state,
                                                   mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    auto memory = state.memories.find(value);
    if (memory != state.memories.end())
      return memory->second;
    if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast.getInputs().size() != 1)
        return {};
      value = cast.getInputs().front();
      continue;
    }
    return {};
  }
  return {};
}

static std::optional<std::uint64_t> memoryRootIdForValue(SimulatorState &state,
                                                         mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    auto root = state.memoryRootIds.find(value);
    if (root != state.memoryRootIds.end())
      return root->second;
    if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast.getInputs().size() != 1)
        return std::nullopt;
      value = cast.getInputs().front();
      continue;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

bool hasPendingVectorGroups(SimulatorState &state) {
  bool pending = false;
  for (auto &entry : state.parallelizeStates) {
    if (entry.second.semanticState.pendingItems == 0)
      continue;
    pending = true;
    state.diagnostics.push_back(
        "dataflow.parallelize ended with pending lanes; emit a false "
        "continuation token to flush the partial vector group");
  }
  return pending;
}

static llvm::Expected<llvm::SmallVector<std::string>>
serializeMemoryValue(const MemoryValue &memory, mlir::Operation *scope) {
  llvm::SmallVector<std::string> values;
  for (auto [index, token] : llvm::enumerate(memory.elements)) {
    if (!memory.initialized[index]) {
      values.push_back("uninitialized");
      continue;
    }
    auto value = tokenToString(token, memory.elementType, scope);
    if (!value)
      return value.takeError();
    values.push_back(std::move(*value));
  }
  return values;
}

llvm::Expected<std::string>
memoryFixtureFromSerializedValues(llvm::ArrayRef<std::string> values) {
  std::string fixture;
  llvm::raw_string_ostream os(fixture);
  for (auto [index, value] : llvm::enumerate(values)) {
    llvm::StringRef serialized(value);
    size_t separator = serialized.find(':');
    if (separator == llvm::StringRef::npos)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "cannot reuse serialized memory value '%s' for another invocation",
          value.c_str());
    if (index != 0)
      os << ',';
    os << serialized.drop_front(separator + 1);
  }
  return os.str();
}

llvm::Error captureFinalMemoryState(dataflow::GraphOp graph,
                                    SimulatorState &state,
                                    DFGSimulationReport &report) {
  mlir::Block &entry = graph.getBody().front();
  for (unsigned index = 0, end = graph.getFunctionType().getNumInputs();
       index < end; ++index) {
    mlir::BlockArgument arg = entry.getArgument(index + 1);
    std::shared_ptr<MemoryValue> memory = memoryForValue(state, arg);
    std::string port = llvm::formatv("arg{0}", index).str();
    if (memory) {
      auto values = serializeMemoryValue(*memory, graph);
      if (!values)
        return values.takeError();
      report.finalMemoryState[port] = std::move(*values);
    }
    if (auto rootId = memoryRootIdForValue(state, arg))
      report.finalMemoryRoots[port] =
          llvm::formatv("memory_root{0}", *rootId).str();
  }
  auto ret = mlir::cast<dataflow::GraphReturnOp>(entry.getTerminator());
  for (auto [index, memoryResult] : llvm::enumerate(ret.getMemories())) {
    std::shared_ptr<MemoryValue> memory = memoryForValue(state, memoryResult);
    std::string port = llvm::formatv("memory_result{0}", index).str();
    if (memory) {
      auto values = serializeMemoryValue(*memory, graph);
      if (!values)
        return values.takeError();
      report.finalMemoryState[port] = std::move(*values);
    }
    if (auto rootId = memoryRootIdForValue(state, memoryResult))
      report.finalMemoryRoots[port] =
          llvm::formatv("memory_root{0}", *rootId).str();
  }
  return llvm::Error::success();
}

std::uint64_t estimateWeightedOperationScore(
    const std::map<std::string, std::uint64_t> &operationFireCounts,
    llvm::SmallVectorImpl<std::string> &diagnostics) {
  std::uint64_t score = 0;
  for (const auto &[opName, fireCount] : operationFireCounts) {
    if (fireCount == 0)
      continue;
    auto costOrErr = estimateOperationCost(opName);
    if (!costOrErr) {
      diagnostics.push_back(llvm::toString(costOrErr.takeError()));
      continue;
    }
    score += costOrErr->baseScore;
    if (fireCount > 1)
      score += (fireCount - 1) * costOrErr->repeatScore;
  }
  return score;
}

static std::uint64_t dynamicWorkItems(const SimulatorState &state) {
  std::uint64_t maxStreamItems = 0;
  for (const auto &entry : state.streamTrueEmissionCounts)
    maxStreamItems = std::max(maxStreamItems, entry.second);
  std::uint64_t maxSeededItems = 0;
  for (const auto &entry : state.seededTokenCounts) {
    if (mlir::isa<mlir::NoneType>(entry.first.getType()))
      continue;
    maxSeededItems = std::max(maxSeededItems, entry.second);
  }
  const std::uint64_t workItems = std::max(maxStreamItems, maxSeededItems);
  if (workItems == 0 && state.eventCount > 0)
    return 1;
  return workItems;
}

void projectRunObservations(SimulatorState &state,
                            DFGSimulationReport &report) {
  report.dynamicWorkItems = dynamicWorkItems(state);
  report.eventCount = state.eventCount;
  report.operationFireCounts = state.operationFireCounts;
  report.modeledLibraryCalls = state.modeledLibraryCalls;
  report.weightedOperationScore = estimateWeightedOperationScore(
      state.operationFireCounts, state.diagnostics);
  report.operationCostScore = report.weightedOperationScore;
  report.modeledLibraryScore = state.modeledLibraryScore;
  report.operationCostScore += report.modeledLibraryScore;
  report.operationDiversityScore = report.operationFireCounts.size();
  report.operationCostScore += report.operationDiversityScore;
  report.memoryAddressScore = state.memoryAddressScore;
  report.operationCostScore += report.memoryAddressScore;
  // Execution records every rejected attempt, which is what classifies an
  // actor transition as failed. The report projects each distinct reason once;
  // re-polling an actor whose inputs did not change repeats no new reason.
  for (const std::string &reason : state.diagnostics)
    if (!llvm::is_contained(report.diagnostics, reason))
      report.diagnostics.push_back(reason);
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim

llvm::Error
loom::sim::writeDFGSimulationReportJson(llvm::StringRef outputPath,
                                        const DFGSimulationReport &report) {
  llvm::SmallString<256> parent(outputPath);
  llvm::sys::path::remove_filename(parent);
  if (!parent.empty()) {
    if (std::error_code ec = llvm::sys::fs::create_directories(parent))
      return llvm::createStringError(ec, "could not create %s", parent.c_str());
  }

  llvm::json::Object root;
  root["schema_version"] = report.schemaVersion;
  root["kind"] = report.kind;
  root["workload"] = report.workload;
  root["graph"] = report.graph;
  root["status"] = report.status;
  root["metric_definition"] = report.metricDefinition;
  root["operation_semantics_source"] = report.operationSemanticsSource;
  root["operation_cost_model_source"] = report.operationCostModelSource;
  if (report.status == "pass") {
    root["operation_cost_score"] = report.operationCostScore;
    root["weighted_operation_score"] =
        static_cast<int64_t>(report.weightedOperationScore);
    root["modeled_library_score"] = report.modeledLibraryScore;
    root["operation_diversity_score"] = report.operationDiversityScore;
    root["memory_address_score"] = report.memoryAddressScore;
    llvm::json::Array scoreBreakdown;
    scoreBreakdown.push_back(llvm::json::Object{
        {"category", "weighted_operations"},
        {"score", static_cast<int64_t>(report.weightedOperationScore)},
        {"evidence", "operation_fire_counts"},
        {"heuristic", true},
    });
    scoreBreakdown.push_back(llvm::json::Object{
        {"category", "modeled_library_work"},
        {"score", static_cast<int64_t>(report.modeledLibraryScore)},
        {"evidence", "modeled_library_calls and modeled workload dimensions"},
        {"heuristic", true},
    });
    scoreBreakdown.push_back(llvm::json::Object{
        {"category", "operation_diversity"},
        {"score", static_cast<int64_t>(report.operationDiversityScore)},
        {"evidence", "distinct operation_fire_counts keys"},
        {"heuristic", true},
    });
    scoreBreakdown.push_back(llvm::json::Object{
        {"category", "computed_memory_address"},
        {"score", static_cast<int64_t>(report.memoryAddressScore)},
        {"evidence", "computed dataflow.load/store address operands"},
        {"heuristic", true},
    });
    root["score_breakdown"] = std::move(scoreBreakdown);
  }
  root["wavefront_steps"] = report.wavefrontSteps;
  root["event_count"] = report.eventCount;
  root["dynamic_work_items"] = report.dynamicWorkItems;

  llvm::json::Object fireCounts;
  for (const auto &[opName, count] : report.operationFireCounts)
    fireCounts[opName] = count;
  root["operation_fire_counts"] = std::move(fireCounts);

  llvm::json::Object libraryCalls;
  for (const auto &[callee, count] : report.modeledLibraryCalls)
    libraryCalls[callee] = count;
  root["modeled_library_calls"] = std::move(libraryCalls);

  llvm::json::Array outputs;
  for (const std::string &value : report.finalOutputs)
    outputs.push_back(value);
  root["final_outputs"] = std::move(outputs);

  llvm::json::Array streamOutputs;
  for (const auto &stream : report.finalStreamOutputs) {
    llvm::json::Array streamValues;
    for (const std::string &value : stream)
      streamValues.push_back(value);
    streamOutputs.push_back(std::move(streamValues));
  }
  root["final_stream_outputs"] = std::move(streamOutputs);

  llvm::json::Object finalMemoryState;
  for (const auto &[argument, values] : report.finalMemoryState) {
    llvm::json::Array memoryValues;
    for (const std::string &value : values)
      memoryValues.push_back(value);
    finalMemoryState[argument] = std::move(memoryValues);
  }
  root["final_memory_state"] = std::move(finalMemoryState);

  llvm::json::Object finalMemoryRoots;
  for (const auto &[port, rootId] : report.finalMemoryRoots)
    finalMemoryRoots[port] = rootId;
  root["final_memory_roots"] = std::move(finalMemoryRoots);

  llvm::json::Array diagnostics;
  for (const std::string &diagnostic : report.diagnostics)
    diagnostics.push_back(diagnostic);
  root["diagnostics"] = std::move(diagnostics);

  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return llvm::createStringError(ec, "could not open %s",
                                   outputPath.str().c_str());
  out << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
  return llvm::Error::success();
}
