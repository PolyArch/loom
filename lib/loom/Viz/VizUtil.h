//===-- VizUtil.h - Shared visualization utilities ----------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Common constants, color palettes, and helper functions shared across all
// DOT exporters. Internal header, not part of the public API.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_VIZ_VIZUTIL_H
#define LOOM_VIZ_VIZUTIL_H

#include "loom/Mapper/Graph.h"
#include "loom/Mapper/Types.h"

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <sstream>
#include <string>

namespace loom {
namespace viz {

// Escape a string for safe use in DOT labels.
inline std::string dotEscape(llvm::StringRef s) {
  std::string result;
  result.reserve(s.size());
  for (char c : s) {
    switch (c) {
    case '"':
      result += "\\\"";
      break;
    case '\\':
      result += "\\\\";
      break;
    case '<':
      result += "\\<";
      break;
    case '>':
      result += "\\>";
      break;
    case '{':
      result += "\\{";
      break;
    case '}':
      result += "\\}";
      break;
    case '|':
      result += "\\|";
      break;
    case '\n':
      result += "\\n";
      break;
    default:
      result += c;
      break;
    }
  }
  return result;
}

// Route palette: 12 colors for distinguishing mapped edge routes.
inline const char *routeColor(size_t index) {
  static const char *palette[] = {
      "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
      "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324",
  };
  return palette[index % 12];
}

// Get the "op_name" string attribute from a node, or empty string.
inline std::string getNodeOpName(const Node *node) {
  for (const auto &attr : node->attributes) {
    if (attr.getName().str() == "op_name") {
      if (auto strAttr = llvm::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue().str();
    }
  }
  return "";
}

// Get a string attribute value from a node by key.
inline std::string getNodeStrAttr(const Node *node, llvm::StringRef key) {
  for (const auto &attr : node->attributes) {
    if (attr.getName().str() == key.str()) {
      if (auto strAttr = llvm::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue().str();
    }
  }
  return "";
}

// Get an integer attribute value from a node by key.
inline int64_t getNodeIntAttr(const Node *node, llvm::StringRef key,
                              int64_t defaultVal = 0) {
  for (const auto &attr : node->attributes) {
    if (attr.getName().str() == key.str()) {
      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        return intAttr.getInt();
    }
  }
  return defaultVal;
}

// Determine the dialect prefix of an operation name (e.g., "arith" from
// "arith.addi").
inline std::string getDialect(const std::string &opName) {
  auto dot = opName.find('.');
  if (dot != std::string::npos)
    return opName.substr(0, dot);
  return opName;
}

/// DFG node shape/color based on operation name per spec-viz-dfg.md.
struct DFGNodeStyle {
  const char *shape;
  const char *fillColor;
  const char *fontColor;
};

inline DFGNodeStyle getDFGNodeStyle(const std::string &opName,
                                    Node::Kind kind) {
  if (kind == Node::ModuleInputNode)
    return {"invhouse", "lightpink", "black"};
  if (kind == Node::ModuleOutputNode)
    return {"house", "lightcoral", "black"};

  // Check specific operations first.
  if (opName == "handshake.constant")
    return {"ellipse", "gold", "black"};
  if (opName == "handshake.cond_br")
    return {"diamond", "lightyellow", "black"};
  if (opName == "handshake.mux")
    return {"invtriangle", "lightyellow", "black"};
  if (opName == "handshake.join")
    return {"triangle", "lightyellow", "black"};
  if (opName == "handshake.load")
    return {"box", "skyblue", "black"};
  if (opName == "handshake.store")
    return {"box", "lightsalmon", "black"};
  if (opName == "handshake.memory")
    return {"cylinder", "skyblue", "black"};
  if (opName == "handshake.extmemory")
    return {"hexagon", "gold", "black"};
  if (opName == "handshake.sink")
    return {"point", "gray", "black"};
  if (opName == "dataflow.carry")
    return {"octagon", "lightgreen", "black"};
  if (opName == "dataflow.gate")
    return {"octagon", "palegreen", "black"};
  if (opName == "dataflow.invariant")
    return {"octagon", "mintcream", "black"};
  if (opName == "dataflow.stream")
    return {"doubleoctagon", "lightgreen", "black"};

  // Check by dialect prefix.
  std::string dialect = getDialect(opName);
  if (dialect == "arith")
    return {"box", "lightblue", "black"};
  if (dialect == "math")
    return {"box", "plum", "black"};
  if (dialect == "dataflow")
    return {"octagon", "lightgreen", "black"};
  if (dialect == "handshake")
    return {"box", "lightyellow", "black"};

  // Unknown/fallback.
  return {"star", "red", "white"};
}

/// ADG node shape/color based on operation name per spec-viz-adg.md.
struct ADGNodeStyle {
  const char *shape;
  const char *fillColor;
  const char *fontColor;
};

inline ADGNodeStyle getADGNodeStyle(const std::string &opName,
                                    Node::Kind kind) {
  if (kind == Node::ModuleInputNode)
    return {"invhouse", "lightpink", "black"};
  if (kind == Node::ModuleOutputNode)
    return {"house", "lightcoral", "black"};

  if (opName == "fabric.pe")
    return {"Msquare", "darkgreen", "white"};
  if (opName == "fabric.temporal_pe")
    return {"Msquare", "purple4", "white"};
  if (opName == "fabric.switch")
    return {"diamond", "lightgray", "black"};
  if (opName == "fabric.temporal_sw")
    return {"diamond", "slategray", "white"};
  if (opName == "fabric.memory")
    return {"cylinder", "skyblue", "black"};
  if (opName == "fabric.extmemory")
    return {"hexagon", "gold", "black"};
  if (opName == "fabric.add_tag")
    return {"trapezium", "lightcyan", "black"};
  if (opName == "fabric.map_tag")
    return {"trapezium", "orchid", "black"};
  if (opName == "fabric.del_tag")
    return {"invtrapezium", "lightcyan", "black"};
  if (opName == "fabric.instance")
    return {"box", "wheat", "black"};

  return {"star", "red", "white"};
}

/// Get the mapped dialect color for overlay/side-by-side coloring.
/// Returns the fill color based on the SW operation dialect.
inline const char *getMappedDialectColor(const std::string &opName) {
  if (opName.empty())
    return "white";

  std::string dialect = getDialect(opName);
  if (dialect == "arith")
    return "lightblue";
  if (dialect == "dataflow")
    return "lightgreen";
  if (dialect == "math")
    return "plum";

  // Specific handshake ops.
  if (opName == "handshake.cond_br" || opName == "handshake.mux" ||
      opName == "handshake.join")
    return "lightyellow";
  if (opName == "handshake.load" || opName == "handshake.store")
    return "lightsalmon";
  if (opName == "handshake.memory" || opName == "handshake.extmemory")
    return "skyblue";
  if (opName == "handshake.constant")
    return "gold";

  // Generic handshake fallback.
  if (dialect == "handshake")
    return "lightyellow";

  return "lightgray";
}

} // namespace viz
} // namespace loom

#endif // LOOM_VIZ_VIZUTIL_H
