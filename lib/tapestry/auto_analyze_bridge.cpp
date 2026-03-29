//===-- auto_analyze_bridge.cpp - AutoAnalyzeResult -> TaskGraph bridge -----===//
//
// Converts AutoAnalyzeResult to tapestry::TaskGraph, bridging automatic
// source analysis with the TDG MLIR emission pipeline.
//
//===----------------------------------------------------------------------===//

#include "tapestry/auto_analyze.h"
#include "tapestry/task_graph.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace tapestry {

//===----------------------------------------------------------------------===//
// sizeOfType -- data type name to byte size mapping
//===----------------------------------------------------------------------===//

unsigned sizeOfType(const std::string &typeName) {
  static const std::unordered_map<std::string, unsigned> typeMap = {
      {"f64", 8}, {"f32", 4}, {"f16", 2}, {"i64", 8}, {"u64", 8},
      {"i32", 4}, {"u32", 4}, {"i16", 2}, {"u16", 2}, {"i8", 1},
      {"u8", 1},
  };

  auto it = typeMap.find(typeName);
  if (it != typeMap.end())
    return it->second;
  return 4; // fallback for unknown types
}

//===----------------------------------------------------------------------===//
// mapKernelTarget -- KernelTarget -> ExecutionTarget conversion
//===----------------------------------------------------------------------===//

static ExecutionTarget mapKernelTarget(KernelTarget kt) {
  switch (kt) {
  case KernelTarget::CGRA:
    return ExecutionTarget::CGRA;
  case KernelTarget::HOST:
    return ExecutionTarget::HOST;
  case KernelTarget::AUTO:
    return ExecutionTarget::AUTO_DETECT;
  }
  return ExecutionTarget::AUTO_DETECT;
}

//===----------------------------------------------------------------------===//
// buildTaskGraphFromAnalysis
//===----------------------------------------------------------------------===//

TaskGraph buildTaskGraphFromAnalysis(const AutoAnalyzeResult &result) {
  TaskGraph tg(result.entryFunc + "_auto");

  // Create kernel nodes from call site bindings.
  std::vector<KernelHandle> kernelHandles;
  kernelHandles.reserve(result.callBindings.size());

  for (const auto &binding : result.callBindings) {
    auto kh = tg.kernel(binding.kernelName);
    kh.target(mapKernelTarget(binding.target));
    kernelHandles.push_back(kh);
  }

  // Create edges from inferred data dependencies.
  for (const auto &edge : result.edges) {
    if (edge.producerIndex >= kernelHandles.size() ||
        edge.consumerIndex >= kernelHandles.size())
      continue;

    auto eh = tg.connect(kernelHandles[edge.producerIndex],
                         kernelHandles[edge.consumerIndex]);

    // Set data type if available.
    if (!edge.dependency.dataType.empty())
      eh.data_type(edge.dependency.dataType);

    // Set ordering from dependency sequentiality.
    eh.ordering(edge.dependency.isSequential ? Ordering::FIFO
                                             : Ordering::UNORDERED);

    // Set data volume if element count is known.
    if (edge.dependency.elementCount.has_value()) {
      uint64_t volume = edge.dependency.elementCount.value() *
                        sizeOfType(edge.dependency.dataType);
      eh.data_volume(volume);
    }
  }

  return tg;
}

} // namespace tapestry
