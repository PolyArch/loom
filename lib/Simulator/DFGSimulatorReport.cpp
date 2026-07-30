//===- DFGSimulatorReport.cpp - What one DFG run reports ------------------===//
//
// One owner for what a finished run projects outward: the report status a
// retained RunFailure projects to, the observations such a run may still
// export, and the legacy JSON encoding of those observations.
//
// The run driver keeps its own lifecycle classification, so a run that
// retained no failure is named there rather than here. This module only
// overrides that name when the run failed at runtime.
//
//===----------------------------------------------------------------------===//

#include "DFGSimulatorInternal.h"
#include "Simulator/DFGSimulator.h"

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

std::optional<MemoryView> resolveMemoryView(SimulatorState &state,
                                            mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    auto view = state.memoryViews.find(value);
    if (view != state.memoryViews.end())
      return view->second;
    auto memory = state.memories.find(value);
    if (memory != state.memories.end()) {
      mlir::Type elementType;
      if (auto type = mlir::dyn_cast<mlir::MemRefType>(value.getType()))
        elementType = type.getElementType();
      return MemoryView{memory->second, value, 0, elementType};
    }
    if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
      value = cast.getSource();
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
serializeMemoryValue(const MemoryView &view, SimulatorState &state,
                     mlir::Operation *scope) {
  if (!view.elementType)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "memory observation has no exact element type");
  auto layout = resolveMemoryElementLayout(view.elementType, scope);
  if (!layout)
    return layout.takeError();
  llvm::SmallVector<std::string> values;
  // finalMemoryRoots names the complete backing object. A graph argument may
  // be a view into that object, but this legacy report projection must not
  // silently discard bytes before the view.
  const std::size_t begin = 0;
  const std::size_t stride = layout->byteCount;
  for (std::size_t offset = begin; offset + stride <= view.memory->bytes.size();
       offset += stride) {
    bool initialized = true;
    for (std::size_t byte = offset; byte < offset + stride; ++byte)
      initialized &= view.memory->initialized[byte];
    if (!initialized) {
      values.push_back("uninitialized");
      continue;
    }
    auto token = readMemoryElementResolved(
        view, offset, view.elementType, *layout, state, "memory observation");
    if (!token)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "memory observation decode failed");
    auto value = tokenToString(*token, view.elementType, scope);
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
    std::optional<MemoryView> memory = resolveMemoryView(state, arg);
    std::string port = llvm::formatv("arg{0}", index).str();
    if (memory) {
      auto values = serializeMemoryValue(*memory, state, graph);
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
    std::optional<MemoryView> memory = resolveMemoryView(state, memoryResult);
    std::string port = llvm::formatv("memory_result{0}", index).str();
    if (memory) {
      auto values = serializeMemoryValue(*memory, state, graph);
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
  report.operationFireCounts.clear();
  for (auto [ordinal, count] : llvm::enumerate(state.operationFireCounts)) {
    if (count == 0)
      continue;
    report.operationFireCounts.emplace(
        static_cast<dataflow::OperationSchemaId>(ordinal), count);
  }
  report.modeledLibraryCalls = state.modeledLibraryCalls;
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
  root["operation_semantics_source"] = report.operationSemanticsSource;
  root["wavefront_steps"] = report.wavefrontSteps;
  root["event_count"] = report.eventCount;
  root["dynamic_work_items"] = report.dynamicWorkItems;

  std::map<std::string, std::uint64_t> serializedFireCounts;
  for (const auto &[schema, count] : report.operationFireCounts)
    serializedFireCounts[dataflow::operationSchemaSpelling(schema).str()] =
        count;
  llvm::json::Object fireCounts;
  for (const auto &[opName, count] : serializedFireCounts)
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
