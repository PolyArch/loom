//===-- VizHTMLHelpers.h - Shared helpers for HTML visualization ---*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Internal shared types and utility functions used by the VizHTMLExporter
// implementation files (VizHTMLExporter.cpp, VizGridLayout.cpp,
// VizJSONWriters.cpp).
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_VIZ_VIZHTMLHELPERS_H
#define LOOM_VIZ_VIZHTMLHELPERS_H

#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Simulator/SimTypes.h"

#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "loom/Dialect/Fabric/FabricTypeUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"

#include <string>
#include <vector>

namespace loom {

//===----------------------------------------------------------------------===//
// Shared types
//===----------------------------------------------------------------------===//

struct GridCoord {
  int col = -1;
  int row = -1;
  bool valid = false;
  bool inferred = false;
};

struct AreaInfo {
  int w = 1;
  int h = 1;
  double cost = 1.0;
};

struct DFGNodeStyle {
  const char *shape;
  const char *fillColor;
};

using BodyOpsMap = llvm::StringMap<llvm::SmallVector<std::string, 4>>;
using DFGOpMap = llvm::DenseMap<IdIndex, mlir::Operation *>;

//===----------------------------------------------------------------------===//
// Inline utility functions
//===----------------------------------------------------------------------===//

inline std::string hwId(IdIndex i) { return "hw_" + std::to_string(i); }
inline std::string swId(IdIndex i) { return "sw_" + std::to_string(i); }
inline std::string hwEdgeId(IdIndex i) { return "hwedge_" + std::to_string(i); }
inline std::string swEdgeId(IdIndex i) { return "swedge_" + std::to_string(i); }

inline llvm::StringRef getNodeStrAttr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes)
    if (attr.getName() == name)
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return s.getValue();
  return "";
}

inline int64_t getNodeIntAttr(const Node *node, llvm::StringRef name,
                              int64_t dflt = 0) {
  for (auto &attr : node->attributes)
    if (attr.getName() == name)
      if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        return i.getInt();
  return dflt;
}

inline std::string nodeName(const Node *node, IdIndex i) {
  llvm::StringRef sn = getNodeStrAttr(node, "sym_name");
  return sn.empty() ? ("node_" + std::to_string(i)) : sn.str();
}

inline IdIndex resolveContainer(
    IdIndex id, const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer) {
  auto it = fuToContainer.find(id);
  return it != fuToContainer.end() ? it->second : id;
}

inline llvm::SmallVector<std::string, 4> getNodeBodyOps(const Node *node) {
  llvm::SmallVector<std::string, 4> ops;
  for (auto &attr : node->attributes) {
    if (attr.getName() == "body_ops") {
      if (auto arr = mlir::dyn_cast<mlir::ArrayAttr>(attr.getValue())) {
        for (auto el : arr) {
          if (auto s = mlir::dyn_cast<mlir::StringAttr>(el))
            ops.push_back(s.getValue().str());
        }
      }
      break;
    }
  }
  return ops;
}

inline std::string nodeTypeStr(const Node *node) {
  if (node->kind == Node::ModuleInputNode)
    return "input";
  if (node->kind == Node::ModuleOutputNode)
    return "output";
  llvm::StringRef opName = getNodeStrAttr(node, "op_name");
  if (opName.starts_with("fabric."))
    return opName.str();
  llvm::StringRef resClass = getNodeStrAttr(node, "resource_class");
  if (resClass == "routing") return "fabric.switch";
  if (resClass == "memory") return "fabric.memory";
  if (resClass == "functional") return "fabric.pe";
  return "fabric.pe";
}

inline std::string printType(mlir::Type type) {
  std::string str;
  llvm::raw_string_ostream os(str);
  type.print(os);
  return str;
}

inline int bitWidthFromType(mlir::Type type) {
  if (!type) return 32;
  if (auto taggedType = mlir::dyn_cast<dataflow::TaggedType>(type)) {
    auto valWidth = fabric::getNativeBitWidth(taggedType.getValueType());
    int tagWidth = taggedType.getTagType().getWidth();
    return (valWidth ? static_cast<int>(*valWidth) : 32) + tagWidth;
  }
  auto width = fabric::getNativeBitWidth(type);
  if (width)
    return static_cast<int>(*width);
  if (mlir::isa<mlir::NoneType>(type)) return 1;
  return 32;
}

inline std::string edgeTypeStr(const Edge *edge, const Graph &g) {
  const Port *srcPort = g.getPort(edge->srcPort);
  if (!srcPort) return "native";
  mlir::Type type = srcPort->type;
  if (!type) return "native";
  if (mlir::isa<dataflow::TaggedType>(type)) return "tagged";
  if (mlir::isa<mlir::MemRefType>(type)) return "memref";
  if (mlir::isa<mlir::NoneType>(type)) return "control";
  return "native";
}

inline bool isRoutingNode(const Node *node) {
  llvm::StringRef opName = getNodeStrAttr(node, "op_name");
  if (opName == "fabric.switch" || opName == "fabric.temporal_sw" ||
      opName == "fabric.fifo")
    return true;
  llvm::StringRef resClass = getNodeStrAttr(node, "resource_class");
  return resClass == "routing";
}

//===----------------------------------------------------------------------===//
// Function declarations (implemented in separate .cpp files)
//===----------------------------------------------------------------------===//

// --- VizHTMLExporter.cpp ---

llvm::DenseMap<IdIndex, IdIndex> buildFUToContainerMap(const Graph &adg);

llvm::DenseMap<IdIndex, int>
buildFULocalIndexMap(const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer);

DFGOpMap buildDFGNodeToOpMap(mlir::ModuleOp dfgModule);

BodyOpsMap buildPEBodyOpsMap(mlir::Operation *fabricModule);

llvm::SmallVector<std::string, 4>
lookupBodyOps(const Node *node, llvm::StringRef symName,
              const BodyOpsMap &bodyOps);

void writeBodyOpsJSON(llvm::json::OStream &json, const Node *node,
                      llvm::StringRef symName, const BodyOpsMap &bodyOps);

AreaInfo computeArea(const Node *node, const Graph &adg,
                     const BodyOpsMap &bodyOps);

std::string buildDFGDot(const Graph &dfg, const DFGOpMap &dfgOpMap);

// --- VizGridLayout.cpp ---

GridCoord extractGridFromName(llvm::StringRef name, int meshBandSize = 10,
                              int temporalRowOffset = 0,
                              int planeBandSize = 0,
                              int meshBaseOffset = 0);

void inferMissingCoords(
    const Graph &adg,
    llvm::DenseMap<IdIndex, GridCoord> &nodeCoords,
    const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer);

void inferMissingCoordsPlaneAware(
    const Graph &adg,
    llvm::DenseMap<IdIndex, GridCoord> &nodeCoords,
    const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer,
    const llvm::DenseMap<IdIndex, int> &nodePlane,
    int crossPlaneIdx = -1);

llvm::DenseMap<IdIndex, int> detectAndSeparateWidthPlanes(
    const Graph &adg,
    llvm::DenseMap<IdIndex, GridCoord> &nodeCoords,
    const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer,
    llvm::DenseMap<int, std::string> &outPlaneLabels);

// --- VizJSONWriters.cpp ---

std::string htmlEscape(llvm::StringRef s);
std::string jsonEscape(llvm::StringRef s);
std::string scriptSafe(const std::string &s);

DFGNodeStyle dfgNodeStyle(llvm::StringRef opName, Node::Kind kind);

std::string getMLIRLabelSuffix(mlir::Operation *op);
void writeMLIRAttrs(llvm::json::OStream &json, mlir::Operation *op);

void writeADGGraphJSON(llvm::raw_ostream &os, const Graph &adg,
                       const MappingState &state,
                       const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer,
                       const llvm::DenseMap<IdIndex, int> &fuLocalIndex,
                       const BodyOpsMap &bodyOps);

void writeMappingDataJSON(llvm::raw_ostream &os, const Graph &adg,
                          const Graph &dfg, const MappingState &state,
                          const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer,
                          const llvm::DenseMap<IdIndex, int> &fuLocalIndex);

void writeSWMetadataJSON(llvm::raw_ostream &os, const Graph &dfg,
                         const MappingState &state,
                         const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer,
                         const DFGOpMap &dfgOpMap);

void writeHWMetadataJSON(llvm::raw_ostream &os, const Graph &adg,
                         const MappingState &state,
                         const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer,
                         const BodyOpsMap &bodyOps);

void writeTraceDataJSON(llvm::raw_ostream &os,
                        const std::vector<sim::TraceEvent> &events,
                        uint64_t totalCycles, uint64_t configCycles);

} // namespace loom

#endif // LOOM_VIZ_VIZHTMLHELPERS_H
