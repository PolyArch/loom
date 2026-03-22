#include "loom/Simulator/RuntimeImageBuilder.h"
#include "loom/Simulator/StaticModelBuilder.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_map>

namespace loom {
namespace sim {

namespace {

bool isMemrefType(mlir::Type type) { return mlir::isa<mlir::MemRefType>(type); }

std::optional<unsigned> findInputBoundaryOrdinal(const StaticMappedModel &model,
                                                 IdIndex swNodeId) {
  for (const auto &binding : model.getInputBindings()) {
    if (binding.swNodeId == swNodeId)
      return binding.boundaryOrdinal;
  }
  return std::nullopt;
}

std::optional<unsigned>
findOutputBoundaryOrdinal(const StaticMappedModel &model, IdIndex swNodeId) {
  for (const auto &binding : model.getOutputBindings()) {
    if (binding.swNodeId == swNodeId)
      return binding.boundaryOrdinal;
  }
  return std::nullopt;
}

std::optional<RuntimeMemorySlotBinding>
resolveMemoryBindingForArg(const Graph &dfg, const StaticMappedModel &model,
                           IdIndex memrefNodeId, int64_t argIndex) {
  const Node *node = dfg.getNode(memrefNodeId);
  if (!node || node->outputPorts.empty())
    return std::nullopt;
  const Port *port = dfg.getPort(node->outputPorts.front());
  if (!port)
    return std::nullopt;

  for (IdIndex edgeId : port->connectedEdges) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!dstPort)
      continue;
    IdIndex dstNodeId = dstPort->parentNode;
    for (const auto &binding : model.getMemoryBindings()) {
      if (binding.swNodeId != dstNodeId)
        continue;
      RuntimeMemorySlotBinding resolved;
      resolved.memrefArgIndex = argIndex;
      resolved.regionId = binding.regionId;
      resolved.elemSizeLog2 = binding.elemSizeLog2;
      return resolved;
    }
  }
  return std::nullopt;
}

} // namespace

bool buildRuntimeImage(const Graph &dfg, const Graph &adg,
                       const MappingState &mapping,
                       llvm::ArrayRef<PEContainment> peContainment,
                       llvm::ArrayRef<loom::ConfigGen::ConfigSlice> configSlices,
                       llvm::ArrayRef<uint32_t> configWords,
                       RuntimeImage &image, std::string &error) {
  RuntimeImage built;
  if (!buildStaticMappedModel(dfg, adg, mapping, peContainment,
                              built.staticModel)) {
    error = "failed to build static mapped model";
    return false;
  }
  built.configImage.slices.clear();
  built.configImage.slices.reserve(configSlices.size());
  for (const auto &slice : configSlices) {
    StaticConfigSlice staticSlice;
    staticSlice.name = slice.name;
    staticSlice.kind = slice.kind;
    staticSlice.hwNode = slice.hwNode;
    staticSlice.wordOffset = slice.wordOffset;
    staticSlice.wordCount = slice.wordCount;
    staticSlice.complete = slice.complete;
    built.configImage.slices.push_back(std::move(staticSlice));
  }
  built.configImage.words.assign(configWords.begin(), configWords.end());

  struct ScalarInfo {
    int64_t argIndex = -1;
    unsigned portIdx = 0;
  };
  struct MemoryInfo {
    int64_t argIndex = -1;
    unsigned regionId = 0;
    unsigned elemSizeLog2 = 0;
  };
  struct OutputInfo {
    int64_t resultIndex = -1;
    unsigned portIdx = 0;
  };

  std::vector<ScalarInfo> scalarInfos;
  std::vector<MemoryInfo> memoryInfos;
  std::vector<OutputInfo> outputInfos;

  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++nodeId) {
    const Node *node = dfg.getNode(nodeId);
    if (!node)
      continue;
    if (node->kind == Node::ModuleInputNode && !node->outputPorts.empty()) {
      const Port *port = dfg.getPort(node->outputPorts.front());
      if (!port)
        continue;
      int64_t argIndex = getNodeAttrInt(node, "arg_index", -1);
      if (mlir::isa<mlir::NoneType>(port->type)) {
        std::optional<unsigned> boundaryOrdinal =
            findInputBoundaryOrdinal(built.staticModel, nodeId);
        if (boundaryOrdinal)
          built.controlImage.startTokenPort =
              static_cast<int32_t>(*boundaryOrdinal);
        continue;
      }
      if (isMemrefType(port->type)) {
        std::optional<RuntimeMemorySlotBinding> resolved =
            resolveMemoryBindingForArg(dfg, built.staticModel, nodeId, argIndex);
        if (resolved) {
          memoryInfos.push_back(
              {resolved->memrefArgIndex, resolved->regionId,
               resolved->elemSizeLog2});
        }
        continue;
      }
      std::optional<unsigned> boundaryOrdinal =
          findInputBoundaryOrdinal(built.staticModel, nodeId);
      if (boundaryOrdinal)
        scalarInfos.push_back({argIndex, *boundaryOrdinal});
      continue;
    }
    if (node->kind == Node::ModuleOutputNode && !node->inputPorts.empty()) {
      const Port *port = dfg.getPort(node->inputPorts.front());
      if (!port || mlir::isa<mlir::NoneType>(port->type))
        continue;
      int64_t resultIndex = getNodeAttrInt(node, "result_index", -1);
      std::optional<unsigned> boundaryOrdinal =
          findOutputBoundaryOrdinal(built.staticModel, nodeId);
      if (boundaryOrdinal)
        outputInfos.push_back({resultIndex, *boundaryOrdinal});
    }
  }

  std::sort(scalarInfos.begin(), scalarInfos.end(),
            [](const ScalarInfo &lhs, const ScalarInfo &rhs) {
              return lhs.argIndex < rhs.argIndex;
            });
  std::sort(memoryInfos.begin(), memoryInfos.end(),
            [](const MemoryInfo &lhs, const MemoryInfo &rhs) {
              return lhs.argIndex < rhs.argIndex;
            });
  std::sort(outputInfos.begin(), outputInfos.end(),
            [](const OutputInfo &lhs, const OutputInfo &rhs) {
              return lhs.resultIndex < rhs.resultIndex;
            });

  built.controlImage.scalarBindings.clear();
  built.controlImage.scalarBindings.reserve(scalarInfos.size());
  for (size_t idx = 0; idx < scalarInfos.size(); ++idx) {
    RuntimeScalarSlotBinding binding;
    binding.slot = static_cast<uint32_t>(idx);
    binding.argIndex = scalarInfos[idx].argIndex;
    binding.portIdx = scalarInfos[idx].portIdx;
    built.controlImage.scalarBindings.push_back(std::move(binding));
  }

  built.controlImage.memoryBindings.clear();
  built.controlImage.memoryBindings.reserve(memoryInfos.size());
  for (size_t idx = 0; idx < memoryInfos.size(); ++idx) {
    RuntimeMemorySlotBinding binding;
    binding.slot = static_cast<uint32_t>(idx);
    binding.memrefArgIndex = memoryInfos[idx].argIndex;
    binding.regionId = memoryInfos[idx].regionId;
    binding.elemSizeLog2 = memoryInfos[idx].elemSizeLog2;
    built.controlImage.memoryBindings.push_back(std::move(binding));
  }

  built.controlImage.outputBindings.clear();
  built.controlImage.outputBindings.reserve(outputInfos.size());
  for (size_t idx = 0; idx < outputInfos.size(); ++idx) {
    RuntimeOutputSlotBinding binding;
    binding.slot = static_cast<uint32_t>(idx);
    binding.resultIndex = outputInfos[idx].resultIndex;
    binding.portIdx = outputInfos[idx].portIdx;
    built.controlImage.outputBindings.push_back(std::move(binding));
  }

  image = std::move(built);
  return true;
}

} // namespace sim
} // namespace loom
