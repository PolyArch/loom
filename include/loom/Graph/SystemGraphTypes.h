#ifndef LOOM_GRAPH_SYSTEMGRAPHTYPES_H
#define LOOM_GRAPH_SYSTEMGRAPHTYPES_H

/// Concrete node/edge types for the system-level 2x2 graph model.
///
/// KernelNode + DataDependency form the Software System Graph (SSG).
/// CoreNode   + NoCLink       form the Hardware System Graph (SHG).
///
/// All types are MLIR-independent: they depend only on llvm/Support/JSON.h
/// and standard C++ headers.

#include "loom/Graph/SystemGraph.h"

#include "llvm/Support/JSON.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>

namespace loom {

//===----------------------------------------------------------------------===//
// KernelNode (SSG node)
//===----------------------------------------------------------------------===//

struct KernelNode {
  std::string kernelId;
  std::set<std::string> variantSet;

  /// Lightweight compute-profile summary (avoids MLIR KernelProfile dep).
  struct ComputeProfile {
    unsigned estimatedMinII = 1;
    uint64_t estimatedSPMBytes = 0;
    double estimatedComputeCycles = 0.0;
  };
  std::optional<ComputeProfile> computeProfile;

  llvm::json::Value toJSON() const {
    llvm::json::Object obj;
    obj["kernelId"] = kernelId;
    llvm::json::Array vars;
    for (const auto &v : variantSet)
      vars.push_back(v);
    obj["variantSet"] = std::move(vars);
    if (computeProfile) {
      llvm::json::Object cp;
      cp["estimatedMinII"] =
          static_cast<int64_t>(computeProfile->estimatedMinII);
      cp["estimatedSPMBytes"] =
          static_cast<int64_t>(computeProfile->estimatedSPMBytes);
      cp["estimatedComputeCycles"] = computeProfile->estimatedComputeCycles;
      obj["computeProfile"] = std::move(cp);
    }
    return llvm::json::Value(std::move(obj));
  }

  static KernelNode fromJSON(const llvm::json::Value &val) {
    KernelNode node;
    auto *obj = val.getAsObject();
    if (!obj)
      return node;
    if (auto id = obj->getString("kernelId"))
      node.kernelId = id->str();
    if (auto *arr = obj->getArray("variantSet")) {
      for (const auto &elem : *arr) {
        if (auto s = elem.getAsString())
          node.variantSet.insert(s->str());
      }
    }
    if (auto *cpObj = obj->getObject("computeProfile")) {
      KernelNode::ComputeProfile cp;
      if (auto v = cpObj->getInteger("estimatedMinII"))
        cp.estimatedMinII = static_cast<unsigned>(*v);
      if (auto v = cpObj->getInteger("estimatedSPMBytes"))
        cp.estimatedSPMBytes = static_cast<uint64_t>(*v);
      if (auto v = cpObj->getNumber("estimatedComputeCycles"))
        cp.estimatedComputeCycles = *v;
      node.computeProfile = cp;
    }
    return node;
  }

  std::string dotLabel() const {
    std::string label = kernelId;
    if (!variantSet.empty()) {
      label += "\\n[";
      bool first = true;
      for (const auto &v : variantSet) {
        if (!first)
          label += ",";
        label += v;
        first = false;
      }
      label += "]";
    }
    return label;
  }
};

//===----------------------------------------------------------------------===//
// DataDependency (SSG edge)
//===----------------------------------------------------------------------===//

struct DataDependency {
  std::string producerKernel;
  std::string consumerKernel;
  uint64_t dataVolume = 0;
  std::optional<std::string> edgeContractRef;

  llvm::json::Value toJSON() const {
    llvm::json::Object obj;
    obj["producerKernel"] = producerKernel;
    obj["consumerKernel"] = consumerKernel;
    obj["dataVolume"] = static_cast<int64_t>(dataVolume);
    if (edgeContractRef)
      obj["edgeContractRef"] = *edgeContractRef;
    return llvm::json::Value(std::move(obj));
  }

  static DataDependency fromJSON(const llvm::json::Value &val) {
    DataDependency dep;
    auto *obj = val.getAsObject();
    if (!obj)
      return dep;
    if (auto s = obj->getString("producerKernel"))
      dep.producerKernel = s->str();
    if (auto s = obj->getString("consumerKernel"))
      dep.consumerKernel = s->str();
    if (auto v = obj->getInteger("dataVolume"))
      dep.dataVolume = static_cast<uint64_t>(*v);
    if (auto s = obj->getString("edgeContractRef"))
      dep.edgeContractRef = s->str();
    return dep;
  }

  std::string dotLabel() const {
    std::string label = std::to_string(dataVolume) + "B";
    if (edgeContractRef)
      label += "\\n(" + *edgeContractRef + ")";
    return label;
  }
};

//===----------------------------------------------------------------------===//
// CoreNode (SHG node)
//===----------------------------------------------------------------------===//

struct CoreNode {
  std::string coreType;
  std::optional<std::string> adgRef;

  /// Resource capacity fields.
  unsigned peCount = 0;
  unsigned fuCount = 0;
  uint64_t spmBytes = 0;

  llvm::json::Value toJSON() const {
    llvm::json::Object obj;
    obj["coreType"] = coreType;
    if (adgRef)
      obj["adgRef"] = *adgRef;
    obj["peCount"] = static_cast<int64_t>(peCount);
    obj["fuCount"] = static_cast<int64_t>(fuCount);
    obj["spmBytes"] = static_cast<int64_t>(spmBytes);
    return llvm::json::Value(std::move(obj));
  }

  static CoreNode fromJSON(const llvm::json::Value &val) {
    CoreNode node;
    auto *obj = val.getAsObject();
    if (!obj)
      return node;
    if (auto s = obj->getString("coreType"))
      node.coreType = s->str();
    if (auto s = obj->getString("adgRef"))
      node.adgRef = s->str();
    if (auto v = obj->getInteger("peCount"))
      node.peCount = static_cast<unsigned>(*v);
    if (auto v = obj->getInteger("fuCount"))
      node.fuCount = static_cast<unsigned>(*v);
    if (auto v = obj->getInteger("spmBytes"))
      node.spmBytes = static_cast<uint64_t>(*v);
    return node;
  }

  std::string dotLabel() const {
    std::string label = coreType;
    label += "\\nPE:" + std::to_string(peCount);
    label += " FU:" + std::to_string(fuCount);
    label += "\\nSPM:" + std::to_string(spmBytes) + "B";
    return label;
  }
};

//===----------------------------------------------------------------------===//
// NoCLink (SHG edge)
//===----------------------------------------------------------------------===//

struct NoCLink {
  std::string srcCore;
  std::string dstCore;
  unsigned bandwidth = 0; // flits per cycle
  unsigned latency = 0;   // pipeline cycles

  llvm::json::Value toJSON() const {
    llvm::json::Object obj;
    obj["srcCore"] = srcCore;
    obj["dstCore"] = dstCore;
    obj["bandwidth"] = static_cast<int64_t>(bandwidth);
    obj["latency"] = static_cast<int64_t>(latency);
    return llvm::json::Value(std::move(obj));
  }

  static NoCLink fromJSON(const llvm::json::Value &val) {
    NoCLink link;
    auto *obj = val.getAsObject();
    if (!obj)
      return link;
    if (auto s = obj->getString("srcCore"))
      link.srcCore = s->str();
    if (auto s = obj->getString("dstCore"))
      link.dstCore = s->str();
    if (auto v = obj->getInteger("bandwidth"))
      link.bandwidth = static_cast<unsigned>(*v);
    if (auto v = obj->getInteger("latency"))
      link.latency = static_cast<unsigned>(*v);
    return link;
  }

  std::string dotLabel() const {
    return "BW:" + std::to_string(bandwidth) +
           " Lat:" + std::to_string(latency);
  }
};

//===----------------------------------------------------------------------===//
// Type aliases for concrete graph types
//===----------------------------------------------------------------------===//

/// Software System Graph: kernels connected by data dependencies.
using SSG = SystemGraph<KernelNode, DataDependency>;

/// Hardware System Graph: cores connected by NoC links.
using SHG = SystemGraph<CoreNode, NoCLink>;

} // namespace loom

#endif // LOOM_GRAPH_SYSTEMGRAPHTYPES_H
