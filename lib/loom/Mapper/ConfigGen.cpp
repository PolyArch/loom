//===-- ConfigGen.cpp - Configuration bitstream generation ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/ConfigGen.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"

namespace loom {

namespace {

llvm::StringRef getNodeName(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "sym_name") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

} // namespace

bool ConfigGen::generate(const MappingState &state, const Graph &dfg,
                         const Graph &adg, const std::string &basePath,
                         bool dumpMapping, const std::string &profile,
                         int seed) {
  nodeConfigs.clear();
  configBlob.clear();
  totalConfigWords = 0;

  // Generate per-node config fragments for each mapped ADG node.
  uint32_t currentOffset = 0;

  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode || hwNode->kind != Node::OperationNode)
      continue;

    // Check if any SW node is mapped to this HW node.
    if (hwId >= state.hwNodeToSwNodes.size() ||
        state.hwNodeToSwNodes[hwId].empty())
      continue;

    NodeConfig nc;
    nc.name = getNodeName(hwNode).str();
    if (nc.name.empty())
      nc.name = "node_" + std::to_string(hwId);
    nc.wordOffset = currentOffset;

    // Generate a minimal config word per mapped operation.
    for (IdIndex swId : state.hwNodeToSwNodes[hwId]) {
      // Config word encodes: operation index, temporal slot if applicable.
      uint32_t configWord = swId;

      if (swId < state.temporalPEAssignments.size()) {
        const auto &tpa = state.temporalPEAssignments[swId];
        if (tpa.slot != INVALID_ID) {
          configWord |= (tpa.slot << 16);
        }
        if (tpa.opcode != INVALID_ID) {
          configWord |= (tpa.opcode << 24);
        }
      }

      nc.words.push_back(configWord);
    }

    nc.wordCount = static_cast<uint32_t>(nc.words.size());
    currentOffset += nc.wordCount;
    nodeConfigs.push_back(nc);
  }

  totalConfigWords = currentOffset;

  // Assemble config blob (little-endian, 32-bit words).
  configBlob.resize(totalConfigWords * 4, 0);
  for (const auto &nc : nodeConfigs) {
    for (uint32_t i = 0; i < nc.wordCount; ++i) {
      uint32_t byteOffset = (nc.wordOffset + i) * 4;
      uint32_t word = nc.words[i];
      if (byteOffset + 3 < configBlob.size()) {
        configBlob[byteOffset + 0] = word & 0xFF;
        configBlob[byteOffset + 1] = (word >> 8) & 0xFF;
        configBlob[byteOffset + 2] = (word >> 16) & 0xFF;
        configBlob[byteOffset + 3] = (word >> 24) & 0xFF;
      }
    }
  }

  // Write output files.
  if (!writeBinary(basePath + ".config.bin"))
    return false;
  if (!writeAddrHeader(basePath + "_addr.h"))
    return false;
  if (dumpMapping) {
    if (!writeMappingJson(state, dfg, adg, basePath + ".mapping.json",
                          profile, seed))
      return false;
  }

  return true;
}

bool ConfigGen::writeBinary(const std::string &path) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_None);
  if (ec)
    return false;

  out.write(reinterpret_cast<const char *>(configBlob.data()),
            configBlob.size());
  return true;
}

bool ConfigGen::writeAddrHeader(const std::string &path) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return false;

  out << "#ifndef LOOM_CONFIG_ADDR_H\n";
  out << "#define LOOM_CONFIG_ADDR_H\n\n";
  out << "#define LOOM_CONFIG_DEPTH " << totalConfigWords << "\n";
  out << "#define LOOM_CONFIG_WIDTH " << wordWidthBits << "\n\n";

  for (const auto &nc : nodeConfigs) {
    std::string upper = nc.name;
    for (char &c : upper) {
      if (c >= 'a' && c <= 'z')
        c -= 32;
      else if (c == '.' || c == '-')
        c = '_';
    }
    out << "#define LOOM_ADDR_" << upper << " " << nc.wordOffset << "\n";
    out << "#define LOOM_SIZE_" << upper << " " << nc.wordCount << "\n";
  }

  out << "\n#endif // LOOM_CONFIG_ADDR_H\n";
  return true;
}

bool ConfigGen::writeMappingJson(const MappingState &state, const Graph &dfg,
                                 const Graph &adg, const std::string &path,
                                 const std::string &profile, int seed) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return false;

  llvm::json::OStream json(out, 2);
  json.objectBegin();

  json.attribute("version", 1);
  json.attribute("status", "success");
  json.attribute("profile", profile);
  json.attribute("seed", seed);

  // Placement.
  json.attributeBegin("placement");
  json.objectBegin();
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swNodeToHwNode.size());
       ++i) {
    IdIndex hwNode = state.swNodeToHwNode[i];
    if (hwNode == INVALID_ID)
      continue;
    const Node *sw = dfg.getNode(i);
    if (!sw || sw->kind != Node::OperationNode)
      continue;

    json.attributeBegin(std::to_string(i));
    json.objectBegin();
    json.attribute("hwNode", static_cast<int64_t>(hwNode));
    const Node *hw = adg.getNode(hwNode);
    if (hw) {
      json.attribute("hwNodeName", getNodeName(hw));
    }
    json.objectEnd();
    json.attributeEnd();
  }
  json.objectEnd();
  json.attributeEnd();

  // Port binding.
  json.attributeBegin("portBinding");
  json.objectBegin();
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swPortToHwPort.size());
       ++i) {
    IdIndex hwPort = state.swPortToHwPort[i];
    if (hwPort == INVALID_ID)
      continue;
    json.attribute(std::to_string(i), static_cast<int64_t>(hwPort));
  }
  json.objectEnd();
  json.attributeEnd();

  // Routes.
  json.attributeBegin("routes");
  json.objectBegin();
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++i) {
    const auto &pathVec = state.swEdgeToHwPaths[i];
    if (pathVec.empty())
      continue;

    json.attributeBegin(std::to_string(i));
    json.objectBegin();

    const Edge *edge = dfg.getEdge(i);
    if (edge) {
      json.attribute("srcSwPort", static_cast<int64_t>(edge->srcPort));
      json.attribute("dstSwPort", static_cast<int64_t>(edge->dstPort));
    }

    json.attributeBegin("hwPath");
    json.arrayBegin();
    for (IdIndex p : pathVec)
      json.value(static_cast<int64_t>(p));
    json.arrayEnd();
    json.attributeEnd();

    json.objectEnd();
    json.attributeEnd();
  }
  json.objectEnd();
  json.attributeEnd();

  // Temporal assignments.
  json.attributeBegin("temporal");
  json.objectBegin();
  for (IdIndex i = 0;
       i < static_cast<IdIndex>(state.temporalPEAssignments.size()); ++i) {
    const auto &tpa = state.temporalPEAssignments[i];
    if (tpa.slot == INVALID_ID)
      continue;
    json.attributeBegin(std::to_string(i));
    json.objectBegin();
    json.attribute("slot", static_cast<int64_t>(tpa.slot));
    json.attribute("tag", static_cast<int64_t>(tpa.tag));
    json.attribute("opcode", static_cast<int64_t>(tpa.opcode));
    json.objectEnd();
    json.attributeEnd();
  }
  json.objectEnd();
  json.attributeEnd();

  // Cost.
  json.attributeBegin("cost");
  json.objectBegin();
  json.attribute("total", state.totalCost);
  json.attribute("placementPressure", state.placementPressure);
  json.attribute("routingCost", state.routingCost);
  json.attribute("temporalCost", state.temporalCost);
  json.attribute("perfProxy", state.perfProxyCost);
  json.attribute("configFootprint", state.configFootprint);
  json.objectEnd();
  json.attributeEnd();

  json.objectEnd();
  return true;
}

} // namespace loom
