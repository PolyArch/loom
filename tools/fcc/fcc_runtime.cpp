#include "fcc_runtime.h"

#include "fcc_args.h"

#include "fcc/ADG/ADGVerifier.h"
#include "fcc/Mapper/ADGFlattener.h"
#include "fcc/Mapper/DFGBuilder.h"
#include "fcc/Simulator/SimArtifactWriter.h"
#include "fcc/Simulator/SimInputSynthesis.h"
#include "fcc/Simulator/SimSession.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace mlir;

namespace fcc {

namespace {

using llvm::json::Array;
using llvm::json::Object;
using llvm::json::Value;

struct RuntimeNodeBinding {
  IdIndex swNode = INVALID_ID;
  IdIndex hwNode = INVALID_ID;
};

struct RuntimeScalarBinding {
  unsigned slot = 0;
  int64_t argIndex = -1;
  unsigned portIdx = 0;
};

struct RuntimeMemoryBinding {
  unsigned slot = 0;
  int64_t memrefArgIndex = -1;
  unsigned regionId = 0;
  unsigned elemSizeLog2 = 0;
};

struct RuntimeOutputBinding {
  unsigned slot = 0;
  int64_t resultIndex = -1;
  unsigned portIdx = 0;
};

struct RuntimeManifestData {
  std::string caseName;
  std::string dfgMlirPath;
  std::string adgMlirPath;
  std::string configBinPath;
  unsigned startTokenPortIdx = std::numeric_limits<unsigned>::max();
  std::vector<RuntimeNodeBinding> nodeBindings;
  std::vector<RuntimeScalarBinding> scalarBindings;
  std::vector<RuntimeMemoryBinding> memoryBindings;
  std::vector<RuntimeOutputBinding> outputBindings;
};

struct RuntimeScalarArg {
  unsigned slot = 0;
  std::vector<uint64_t> data;
  std::vector<uint16_t> tags;
};

struct RuntimeMemoryImage {
  unsigned slot = 0;
  std::vector<uint8_t> bytes;
};

struct RuntimeRequestData {
  unsigned startTokenCount = 1;
  std::vector<uint32_t> configWords;
  std::vector<RuntimeScalarArg> scalarArgs;
  std::vector<RuntimeMemoryImage> memoryRegions;
};

static std::optional<int64_t> getInt64(const Value &value) {
  if (auto integer = value.getAsInteger())
    return *integer;
  return std::nullopt;
}

template <typename T>
bool parseIntegerVector(const Value *value, std::vector<T> &out,
                        llvm::StringRef fieldName, std::string &error) {
  if (!value) {
    error = ("missing field '" + fieldName + "'").str();
    return false;
  }
  const Array *array = value->getAsArray();
  if (!array) {
    error = ("field '" + fieldName + "' must be an array").str();
    return false;
  }
  out.clear();
  out.reserve(array->size());
  for (size_t i = 0; i < array->size(); ++i) {
    auto integer = getInt64((*array)[i]);
    if (!integer) {
      error = ("field '" + fieldName + "' must contain integers").str();
      return false;
    }
    if (*integer < 0 ||
        static_cast<uint64_t>(*integer) >
            static_cast<uint64_t>(std::numeric_limits<T>::max())) {
      error = ("field '" + fieldName + "' contains out-of-range value").str();
      return false;
    }
    out.push_back(static_cast<T>(*integer));
  }
  return true;
}

static std::string makeAbsolutePath(llvm::StringRef path) {
  llvm::SmallString<256> absolute(path);
  if (!absolute.empty())
    llvm::sys::fs::make_absolute(absolute);
  return std::string(absolute.str());
}

static OwningOpRef<ModuleOp> loadModule(const std::string &path,
                                        MLIRContext &context) {
  llvm::SourceMgr srcMgr;
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    llvm::errs() << "fcc: cannot open MLIR file: " << path << "\n";
    return {};
  }
  srcMgr.AddNewSourceBuffer(std::move(*buffer), llvm::SMLoc());
  return parseSourceFile<ModuleOp>(srcMgr, &context);
}

static bool isMemrefType(Type type) { return isa<MemRefType>(type); }

static std::vector<unsigned> buildBoundaryOrdinals(const Graph &graph,
                                                   Node::Kind kind) {
  std::vector<unsigned> ordinals(graph.nodes.size(),
                                 std::numeric_limits<unsigned>::max());
  unsigned nextOrdinal = 0;
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(graph.nodes.size());
       ++nodeId) {
    const Node *node = graph.getNode(nodeId);
    if (!node || node->kind != kind)
      continue;
    ordinals[nodeId] = nextOrdinal++;
  }
  return ordinals;
}

static std::optional<unsigned> resolveInputOrdinal(const Graph &dfg,
                                                   const Graph &adg,
                                                   const MappingState &mapping,
                                                   IdIndex swNodeId) {
  if (swNodeId < 0 ||
      swNodeId >= static_cast<IdIndex>(mapping.swNodeToHwNode.size()))
    return std::nullopt;
  IdIndex hwNodeId = mapping.swNodeToHwNode[swNodeId];
  if (hwNodeId == INVALID_ID)
    return std::nullopt;
  std::vector<unsigned> ordinals =
      buildBoundaryOrdinals(adg, Node::ModuleInputNode);
  if (hwNodeId >= static_cast<IdIndex>(ordinals.size()))
    return std::nullopt;
  if (ordinals[hwNodeId] == std::numeric_limits<unsigned>::max())
    return std::nullopt;
  return ordinals[hwNodeId];
}

static std::optional<unsigned>
resolveOutputOrdinal(const Graph &dfg, const Graph &adg,
                     const MappingState &mapping, IdIndex swNodeId) {
  if (swNodeId < 0 ||
      swNodeId >= static_cast<IdIndex>(mapping.swNodeToHwNode.size()))
    return std::nullopt;
  IdIndex hwNodeId = mapping.swNodeToHwNode[swNodeId];
  if (hwNodeId == INVALID_ID)
    return std::nullopt;
  std::vector<unsigned> ordinals =
      buildBoundaryOrdinals(adg, Node::ModuleOutputNode);
  if (hwNodeId >= static_cast<IdIndex>(ordinals.size()))
    return std::nullopt;
  if (ordinals[hwNodeId] == std::numeric_limits<unsigned>::max())
    return std::nullopt;
  return ordinals[hwNodeId];
}

static std::optional<IdIndex> findStartTokenInputNode(const Graph &dfg) {
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++nodeId) {
    const Node *node = dfg.getNode(nodeId);
    if (!node || node->kind != Node::ModuleInputNode || node->outputPorts.empty())
      continue;
    const Port *port = dfg.getPort(node->outputPorts.front());
    if (port && isa<NoneType>(port->type))
      return nodeId;
  }
  return std::nullopt;
}

static bool writeJsonToFile(const Value &value, const std::string &path) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "fcc: failed to open " << path << ": " << ec.message()
                 << "\n";
    return false;
  }
  out << llvm::formatv("{0:2}", value);
  out << "\n";
  return true;
}

static bool loadRuntimeManifest(const std::string &path,
                                RuntimeManifestData &manifest,
                                std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot open runtime manifest";
    return false;
  }
  auto parsed = llvm::json::parse(buffer.get()->getBuffer());
  if (!parsed) {
    error = "runtime manifest is not valid JSON";
    return false;
  }
  const Object *root = parsed->getAsObject();
  if (!root) {
    error = "runtime manifest root must be an object";
    return false;
  }

  manifest = RuntimeManifestData();
  manifest.caseName = root->getString("case_name").value_or("").str();
  manifest.dfgMlirPath = root->getString("dfg_mlir").value_or("").str();
  manifest.adgMlirPath = root->getString("adg_mlir").value_or("").str();
  manifest.configBinPath = root->getString("config_bin").value_or("").str();
  manifest.startTokenPortIdx =
      static_cast<unsigned>(root->getInteger("start_token_port").value_or(
          std::numeric_limits<int64_t>::max()));

  const Array *nodeBindings = root->getArray("node_bindings");
  if (!nodeBindings) {
    error = "runtime manifest missing node_bindings";
    return false;
  }
  for (const Value &value : *nodeBindings) {
    const Object *object = value.getAsObject();
    if (!object) {
      error = "node_bindings entries must be objects";
      return false;
    }
    auto swNode = object->getInteger("sw_node");
    auto hwNode = object->getInteger("hw_node");
    if (!swNode || !hwNode) {
      error = "node_bindings entries must contain sw_node and hw_node";
      return false;
    }
    manifest.nodeBindings.push_back(
        {static_cast<IdIndex>(*swNode), static_cast<IdIndex>(*hwNode)});
  }

  auto parseBindingArray = [&](llvm::StringRef fieldName, auto &output,
                               auto parseOne) -> bool {
    const Array *array = root->getArray(fieldName);
    if (!array)
      return true;
    for (const Value &value : *array) {
      const Object *object = value.getAsObject();
      if (!object) {
        error = ("field '" + fieldName + "' entries must be objects").str();
        return false;
      }
      if (!parseOne(*object))
        return false;
    }
    return true;
  };

  if (!parseBindingArray("scalar_args", manifest.scalarBindings,
                         [&](const Object &object) {
                           auto slot = object.getInteger("slot");
                           auto argIndex = object.getInteger("arg_index");
                           auto portIdx = object.getInteger("port_idx");
                           if (!slot || !argIndex || !portIdx) {
                             error = "scalar_args entries must contain slot, "
                                     "arg_index, and port_idx";
                             return false;
                           }
                           manifest.scalarBindings.push_back(
                               {static_cast<unsigned>(*slot), *argIndex,
                                static_cast<unsigned>(*portIdx)});
                           return true;
                         }))
    return false;

  if (!parseBindingArray("memory_regions", manifest.memoryBindings,
                         [&](const Object &object) {
                           auto slot = object.getInteger("slot");
                           auto memrefArgIndex =
                               object.getInteger("memref_arg_index");
                           auto regionId = object.getInteger("region_id");
                           auto elemSizeLog2 =
                               object.getInteger("elem_size_log2");
                           if (!slot || !memrefArgIndex || !regionId ||
                               !elemSizeLog2) {
                             error = "memory_regions entries must contain slot, "
                                     "memref_arg_index, region_id, and "
                                     "elem_size_log2";
                             return false;
                           }
                           manifest.memoryBindings.push_back(
                               {static_cast<unsigned>(*slot), *memrefArgIndex,
                                static_cast<unsigned>(*regionId),
                                static_cast<unsigned>(*elemSizeLog2)});
                           return true;
                         }))
    return false;

  if (!parseBindingArray("outputs", manifest.outputBindings,
                         [&](const Object &object) {
                           auto slot = object.getInteger("slot");
                           auto resultIndex = object.getInteger("result_index");
                           auto portIdx = object.getInteger("port_idx");
                           if (!slot || !resultIndex || !portIdx) {
                             error = "outputs entries must contain slot, "
                                     "result_index, and port_idx";
                             return false;
                           }
                           manifest.outputBindings.push_back(
                               {static_cast<unsigned>(*slot), *resultIndex,
                                static_cast<unsigned>(*portIdx)});
                           return true;
                         }))
    return false;

  if (manifest.dfgMlirPath.empty() || manifest.adgMlirPath.empty() ||
      manifest.configBinPath.empty()) {
    error = "runtime manifest missing dfg_mlir, adg_mlir, or config_bin";
    return false;
  }
  return true;
}

static bool parseScalarArg(const Object &object, RuntimeScalarArg &arg,
                           std::string &error) {
  auto slot = object.getInteger("slot");
  if (!slot) {
    error = "scalar_args entries must contain slot";
    return false;
  }
  arg.slot = static_cast<unsigned>(*slot);
  if (auto value = object.getInteger("value")) {
    arg.data = {static_cast<uint64_t>(*value)};
    arg.tags.clear();
    return true;
  }
  if (!parseIntegerVector<uint64_t>(object.get("data"), arg.data, "data",
                                    error))
    return false;
  if (const Value *tags = object.get("tags")) {
    if (!parseIntegerVector<uint16_t>(tags, arg.tags, "tags", error))
      return false;
    if (arg.tags.size() != arg.data.size()) {
      error = "scalar_args tags length must match data length";
      return false;
    }
  }
  return true;
}

static bool parseMemoryImage(const Object &object, RuntimeMemoryImage &image,
                             std::string &error) {
  auto slot = object.getInteger("slot");
  if (!slot) {
    error = "memory_regions entries must contain slot";
    return false;
  }
  image.slot = static_cast<unsigned>(*slot);
  return parseIntegerVector<uint8_t>(object.get("bytes"), image.bytes, "bytes",
                                     error);
}

static bool loadRuntimeRequest(const std::string &path, RuntimeRequestData &req,
                               std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot open runtime request";
    return false;
  }
  auto parsed = llvm::json::parse(buffer.get()->getBuffer());
  if (!parsed) {
    error = "runtime request is not valid JSON";
    return false;
  }
  const Object *root = parsed->getAsObject();
  if (!root) {
    error = "runtime request root must be an object";
    return false;
  }

  req = RuntimeRequestData();
  req.startTokenCount =
      static_cast<unsigned>(root->getInteger("start_token_count").value_or(1));
  if (const Value *configWords = root->get("config_words")) {
    if (!parseIntegerVector<uint32_t>(configWords, req.configWords,
                                      "config_words", error))
      return false;
  }

  if (const Array *scalarArgs = root->getArray("scalar_args")) {
    for (const Value &value : *scalarArgs) {
      const Object *object = value.getAsObject();
      if (!object) {
        error = "scalar_args entries must be objects";
        return false;
      }
      RuntimeScalarArg arg;
      if (!parseScalarArg(*object, arg, error))
        return false;
      req.scalarArgs.push_back(std::move(arg));
    }
  }

  if (const Array *memoryRegions = root->getArray("memory_regions")) {
    for (const Value &value : *memoryRegions) {
      const Object *object = value.getAsObject();
      if (!object) {
        error = "memory_regions entries must be objects";
        return false;
      }
      RuntimeMemoryImage image;
      if (!parseMemoryImage(*object, image, error))
        return false;
      req.memoryRegions.push_back(std::move(image));
    }
  }

  return true;
}

static bool readBinaryFile(const std::string &path, std::vector<uint8_t> &bytes,
                           std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(path, /*IsText=*/false);
  if (!buffer) {
    error = "cannot open binary file";
    return false;
  }
  llvm::StringRef data = buffer.get()->getBuffer();
  bytes.assign(data.bytes_begin(), data.bytes_end());
  return true;
}

static std::vector<uint8_t>
configWordsToBytes(const std::vector<uint32_t> &words) {
  std::vector<uint8_t> bytes(words.size() * sizeof(uint32_t), 0);
  for (size_t i = 0; i < words.size(); ++i) {
    uint32_t word = words[i];
    bytes[i * 4 + 0] = static_cast<uint8_t>(word & 0xffu);
    bytes[i * 4 + 1] = static_cast<uint8_t>((word >> 8) & 0xffu);
    bytes[i * 4 + 2] = static_cast<uint8_t>((word >> 16) & 0xffu);
    bytes[i * 4 + 3] = static_cast<uint8_t>((word >> 24) & 0xffu);
  }
  return bytes;
}

} // namespace

bool writeRuntimeManifest(const std::string &path, const std::string &caseName,
                          const std::string &dfgMlirPath,
                          const std::string &adgMlirPath,
                          const std::string &configBinPath,
                          const Graph &dfg, const Graph &adg,
                          const MappingState &mapping) {
  sim::SynthesizedSetup setup =
      sim::synthesizeSimulationSetup(dfg, adg, mapping, /*vectorLength=*/1);

  std::optional<IdIndex> startNodeId = findStartTokenInputNode(dfg);
  std::optional<unsigned> startPortIdx;
  if (startNodeId)
    startPortIdx = resolveInputOrdinal(dfg, adg, mapping, *startNodeId);

  struct ScalarInfo {
    int64_t argIndex = -1;
    unsigned portIdx = 0;
  };
  std::vector<ScalarInfo> scalarInfos;
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++nodeId) {
    const Node *node = dfg.getNode(nodeId);
    if (!node || node->kind != Node::ModuleInputNode || node->outputPorts.empty())
      continue;
    const Port *port = dfg.getPort(node->outputPorts.front());
    if (!port || isMemrefType(port->type) || isa<NoneType>(port->type))
      continue;
    std::optional<unsigned> portIdx =
        resolveInputOrdinal(dfg, adg, mapping, nodeId);
    if (!portIdx)
      continue;
    scalarInfos.push_back({getNodeAttrInt(node, "arg_index", -1), *portIdx});
  }
  std::sort(scalarInfos.begin(), scalarInfos.end(),
            [](const ScalarInfo &lhs, const ScalarInfo &rhs) {
              return lhs.argIndex < rhs.argIndex;
            });

  std::unordered_map<int64_t, unsigned> memrefSlotByArgIndex;
  std::vector<int64_t> memrefOrder;
  for (const auto &region : setup.memoryRegions) {
    if (region.memrefArgIndex < 0)
      continue;
    if (memrefSlotByArgIndex.count(region.memrefArgIndex))
      continue;
    memrefOrder.push_back(region.memrefArgIndex);
  }
  std::sort(memrefOrder.begin(), memrefOrder.end());
  for (size_t i = 0; i < memrefOrder.size(); ++i)
    memrefSlotByArgIndex[memrefOrder[i]] = static_cast<unsigned>(i);

  struct OutputInfo {
    int64_t resultIndex = -1;
    unsigned portIdx = 0;
  };
  std::vector<OutputInfo> outputInfos;
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++nodeId) {
    const Node *node = dfg.getNode(nodeId);
    if (!node || node->kind != Node::ModuleOutputNode || node->inputPorts.empty())
      continue;
    const Port *port = dfg.getPort(node->inputPorts.front());
    if (!port || isa<NoneType>(port->type))
      continue;
    std::optional<unsigned> portIdx =
        resolveOutputOrdinal(dfg, adg, mapping, nodeId);
    if (!portIdx)
      continue;
    outputInfos.push_back(
        {getNodeAttrInt(node, "result_index", -1), *portIdx});
  }
  std::sort(outputInfos.begin(), outputInfos.end(),
            [](const OutputInfo &lhs, const OutputInfo &rhs) {
              return lhs.resultIndex < rhs.resultIndex;
            });

  Array nodeBindingsJson;
  for (IdIndex swNode = 0;
       swNode < static_cast<IdIndex>(mapping.swNodeToHwNode.size()); ++swNode) {
    IdIndex hwNode = mapping.swNodeToHwNode[swNode];
    if (hwNode == INVALID_ID)
      continue;
    Object binding;
    binding["sw_node"] = static_cast<int64_t>(swNode);
    binding["hw_node"] = static_cast<int64_t>(hwNode);
    nodeBindingsJson.push_back(std::move(binding));
  }

  Array scalarJson;
  for (size_t slot = 0; slot < scalarInfos.size(); ++slot) {
    Object binding;
    binding["slot"] = static_cast<int64_t>(slot);
    binding["arg_index"] = scalarInfos[slot].argIndex;
    binding["port_idx"] = static_cast<int64_t>(scalarInfos[slot].portIdx);
    scalarJson.push_back(std::move(binding));
  }

  Array memoryJson;
  for (const auto &region : setup.memoryRegions) {
    if (region.memrefArgIndex < 0)
      continue;
    Object binding;
    binding["slot"] =
        static_cast<int64_t>(memrefSlotByArgIndex[region.memrefArgIndex]);
    binding["memref_arg_index"] = region.memrefArgIndex;
    binding["region_id"] = static_cast<int64_t>(region.regionId);
    binding["elem_size_log2"] = static_cast<int64_t>(region.elemSizeLog2);
    memoryJson.push_back(std::move(binding));
  }

  Array outputJson;
  for (size_t slot = 0; slot < outputInfos.size(); ++slot) {
    Object binding;
    binding["slot"] = static_cast<int64_t>(slot);
    binding["result_index"] = outputInfos[slot].resultIndex;
    binding["port_idx"] = static_cast<int64_t>(outputInfos[slot].portIdx);
    outputJson.push_back(std::move(binding));
  }

  Object root;
  root["case_name"] = caseName;
  root["dfg_mlir"] = makeAbsolutePath(dfgMlirPath);
  root["adg_mlir"] = makeAbsolutePath(adgMlirPath);
  root["config_bin"] = makeAbsolutePath(configBinPath);
  root["start_token_port"] = startPortIdx
                                 ? static_cast<int64_t>(*startPortIdx)
                                 : static_cast<int64_t>(-1);
  root["node_bindings"] = std::move(nodeBindingsJson);
  root["scalar_args"] = std::move(scalarJson);
  root["memory_regions"] = std::move(memoryJson);
  root["outputs"] = std::move(outputJson);
  return writeJsonToFile(Value(std::move(root)), path);
}

int runRuntimeReplay(const FccArgs &args, MLIRContext &context) {
  RuntimeManifestData manifest;
  RuntimeRequestData request;
  std::string error;

  if (!loadRuntimeManifest(args.runtimeManifestPath, manifest, error)) {
    llvm::errs() << "fcc: runtime replay failed to load manifest: " << error
                 << "\n";
    return 1;
  }
  if (!loadRuntimeRequest(args.runtimeRequestPath, request, error)) {
    llvm::errs() << "fcc: runtime replay failed to load request: " << error
                 << "\n";
    return 1;
  }

  auto dfgModule = loadModule(manifest.dfgMlirPath, context);
  if (!dfgModule)
    return 1;
  auto adgModule = loadModule(manifest.adgMlirPath, context);
  if (!adgModule)
    return 1;
  if (failed(verifyFabricModule(*adgModule))) {
    llvm::errs() << "fcc: runtime replay ADG verification failed\n";
    return 1;
  }

  ADGFlattener flattener;
  if (!flattener.flatten(*adgModule, &context)) {
    llvm::errs() << "fcc: runtime replay ADG flattening failed\n";
    return 1;
  }
  DFGBuilder dfgBuilder;
  if (!dfgBuilder.build(*dfgModule, &context)) {
    llvm::errs() << "fcc: runtime replay DFG build failed\n";
    return 1;
  }

  MappingState mapping;
  mapping.init(dfgBuilder.getDFG(), flattener.getADG(), &flattener);
  for (const RuntimeNodeBinding &binding : manifest.nodeBindings) {
    if (mapping.mapNode(binding.swNode, binding.hwNode, dfgBuilder.getDFG(),
                        flattener.getADG()) != ActionResult::Success) {
      llvm::errs() << "fcc: runtime replay failed to rebuild node mapping for "
                   << binding.swNode << " -> " << binding.hwNode << "\n";
      return 1;
    }
  }

  sim::SimConfig simConfig;
  simConfig.maxCycles = args.simMaxCycles;
  sim::SimSession session(nullptr, simConfig);
  sim::SimArtifactWriter artifactWriter;

  std::vector<uint8_t> configBlob;
  if (!request.configWords.empty()) {
    configBlob = configWordsToBytes(request.configWords);
  } else if (!readBinaryFile(manifest.configBinPath, configBlob, error)) {
    llvm::errs() << "fcc: runtime replay failed to load config: " << error
                 << "\n";
    return 1;
  }

  if (std::string err = session.connect(); !err.empty()) {
    llvm::errs() << "fcc: runtime replay connect failed: " << err << "\n";
    return 1;
  }
  if (std::string err = session.buildFromMappedState(
          dfgBuilder.getDFG(), flattener.getADG(), mapping);
      !err.empty()) {
    llvm::errs() << "fcc: runtime replay graph build failed: " << err << "\n";
    return 1;
  }
  if (std::string err = session.loadConfig(configBlob); !err.empty()) {
    llvm::errs() << "fcc: runtime replay config load failed: " << err << "\n";
    return 1;
  }

  if (manifest.startTokenPortIdx != std::numeric_limits<unsigned>::max() &&
      request.startTokenCount > 0) {
    std::vector<uint64_t> tokens(request.startTokenCount, 0);
    if (std::string err =
            session.setInput(manifest.startTokenPortIdx, tokens, {});
        !err.empty()) {
      llvm::errs() << "fcc: runtime replay start-token bind failed: " << err
                   << "\n";
      return 1;
    }
  }

  std::unordered_map<unsigned, RuntimeScalarArg> scalarArgsBySlot;
  for (const RuntimeScalarArg &arg : request.scalarArgs)
    scalarArgsBySlot[arg.slot] = arg;
  for (const RuntimeScalarBinding &binding : manifest.scalarBindings) {
    auto it = scalarArgsBySlot.find(binding.slot);
    if (it == scalarArgsBySlot.end())
      continue;
    if (std::string err =
            session.setInput(binding.portIdx, it->second.data, it->second.tags);
        !err.empty()) {
      llvm::errs() << "fcc: runtime replay scalar bind failed for slot "
                   << binding.slot << ": " << err << "\n";
      return 1;
    }
  }

  std::unordered_map<unsigned, RuntimeMemoryImage> memoryBySlot;
  for (const RuntimeMemoryImage &image : request.memoryRegions)
    memoryBySlot[image.slot] = image;

  std::vector<std::vector<uint8_t>> memoryStorage;
  memoryStorage.reserve(manifest.memoryBindings.size());
  for (const RuntimeMemoryBinding &binding : manifest.memoryBindings) {
    auto it = memoryBySlot.find(binding.slot);
    if (it == memoryBySlot.end()) {
      llvm::errs() << "fcc: runtime replay missing memory slot " << binding.slot
                   << "\n";
      return 1;
    }
    memoryStorage.push_back(it->second.bytes);
    std::vector<uint8_t> &bytes = memoryStorage.back();
    if (std::string err = session.setExtMemoryBacking(binding.regionId,
                                                      bytes.data(),
                                                      bytes.size());
        !err.empty()) {
      llvm::errs() << "fcc: runtime replay memory bind failed for slot "
                   << binding.slot << ": " << err << "\n";
      return 1;
    }
  }

  auto [simResult, invokeErr] = session.invoke();
  std::string tracePath = args.runtimeTracePath.empty()
                              ? args.outputDir + "/" + manifest.caseName +
                                    ".runtime.trace"
                              : args.runtimeTracePath;
  std::string statPath = args.runtimeStatPath.empty()
                             ? args.outputDir + "/" + manifest.caseName +
                                   ".runtime.stat"
                             : args.runtimeStatPath;
  artifactWriter.writeTrace(simResult, tracePath);
  artifactWriter.writeStat(simResult, statPath);

  Array outputsJson;
  for (const RuntimeOutputBinding &binding : manifest.outputBindings) {
    Object output;
    output["slot"] = static_cast<int64_t>(binding.slot);
    Array dataArray;
    for (uint64_t value : session.getOutput(binding.portIdx))
      dataArray.push_back(static_cast<int64_t>(value));
    Array tagArray;
    for (uint16_t tag : session.getOutputTags(binding.portIdx))
      tagArray.push_back(static_cast<int64_t>(tag));
    output["data"] = std::move(dataArray);
    output["tags"] = std::move(tagArray);
    outputsJson.push_back(std::move(output));
  }

  Array memoryJson;
  for (size_t i = 0; i < manifest.memoryBindings.size(); ++i) {
    const RuntimeMemoryBinding &binding = manifest.memoryBindings[i];
    Object region;
    region["slot"] = static_cast<int64_t>(binding.slot);
    region["region_id"] = static_cast<int64_t>(binding.regionId);
    Array bytesArray;
    for (uint8_t byte : memoryStorage[i])
      bytesArray.push_back(static_cast<int64_t>(byte));
    region["bytes"] = std::move(bytesArray);
    memoryJson.push_back(std::move(region));
  }

  Object root;
  bool success = invokeErr.empty() && simResult.success;
  root["success"] = success;
  root["error_message"] =
      invokeErr.empty() ? simResult.errorMessage : invokeErr;
  root["cycle_count"] = static_cast<int64_t>(simResult.totalCycles);
  root["trace_path"] = makeAbsolutePath(tracePath);
  root["stat_path"] = makeAbsolutePath(statPath);
  root["outputs"] = std::move(outputsJson);
  root["memory_regions"] = std::move(memoryJson);
  if (!writeJsonToFile(Value(std::move(root)), args.runtimeResultPath))
    return 1;

  if (!success) {
    llvm::errs() << "fcc: runtime replay failed";
    if (!invokeErr.empty())
      llvm::errs() << ": " << invokeErr;
    else if (!simResult.errorMessage.empty())
      llvm::errs() << ": " << simResult.errorMessage;
    llvm::errs() << "\n";
    return 1;
  }
  return 0;
}

} // namespace fcc
