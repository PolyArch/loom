//===-- VizGridLayout.cpp - Grid coordinate extraction and layout --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Grid coordinate extraction from node names, coordinate inference for
// unplaced nodes, and width-plane detection/separation for multi-width ADGs.
//
//===----------------------------------------------------------------------===//

#include "loom/Viz/VizHTMLHelpers.h"

#include <cmath>
#include <queue>
#include <regex>

namespace loom {

// ---- Grid coordinate extraction from node names ----

static int meshLetterOffset(char letter, int bandSize, int baseOffset) {
  return baseOffset + (letter - 'a') * bandSize;
}

GridCoord extractGridFromName(llvm::StringRef name, int meshBandSize,
                              int temporalRowOffset, int planeBandSize,
                              int meshBaseOffset) {
  GridCoord gc;
  std::string nameStr = name.str();

  // ---- Switch patterns (even positions) ----
  static const std::regex reSW_WD("sw_w(\\d+)_(\\d+)_(\\d+)_(\\d+)");
  static const std::regex reSW_W("sw_w(\\d+)_(\\d+)_(\\d+)");
  static const std::regex reL_SW("l(\\d+)_sw_(\\d+)_(\\d+)");
  static const std::regex reSW("^sw_(\\d+)_(\\d+)$");
  static const std::regex reTSW("^tsw_(\\d+)_(\\d+)$");

  // ---- PE/TPE patterns with mesh prefix (odd positions) ----
  static const std::regex rePE_MESH("^(?:pe|tpe)_([a-z])_(\\d+)_(\\d+)$");
  static const std::regex reTPE_RC("tpe_r(\\d+)_c(\\d+)");
  static const std::regex rePE_RC("_r(\\d+)_c(\\d+)$");
  static const std::regex rePE("^pe_(\\d+)_(\\d+)$");
  static const std::regex reTPE_LETTER("^tpe_([a-z])$");
  static const std::regex reTPE_NUM("^tpe_(\\d+)$");
  static const std::regex rePE_MESH_1D("^(?:pe|tpe)_([a-z])_(\\d+)$");

  std::smatch m;

  if (std::regex_search(nameStr, m, reSW_WD)) {
    int depth = std::stoi(m[2]);
    gc.col = depth * planeBandSize + std::stoi(m[4]) * 2;
    gc.row = std::stoi(m[3]) * 2;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, reSW_W)) {
    int width = std::stoi(m[1]);
    gc.col = width * planeBandSize + std::stoi(m[3]) * 2;
    gc.row = std::stoi(m[2]) * 2;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, reL_SW)) {
    int lattice = std::stoi(m[1]);
    gc.col = lattice * planeBandSize + std::stoi(m[3]) * 2;
    gc.row = std::stoi(m[2]) * 2;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, reSW)) {
    gc.col = std::stoi(m[2]) * 2;
    gc.row = std::stoi(m[1]) * 2;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, reTSW)) {
    gc.col = std::stoi(m[2]) * 2;
    gc.row = temporalRowOffset + std::stoi(m[1]) * 2;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, rePE_MESH)) {
    int offset = meshLetterOffset(m[1].str()[0], meshBandSize, meshBaseOffset);
    gc.col = offset + std::stoi(m[3]) * 2 + 1;
    gc.row = std::stoi(m[2]) * 2 + 1;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, reTPE_RC)) {
    gc.col = std::stoi(m[2]) * 2 + 1;
    gc.row = temporalRowOffset + std::stoi(m[1]) * 2 + 1;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, rePE_RC)) {
    gc.col = std::stoi(m[2]) * 2 + 1;
    gc.row = std::stoi(m[1]) * 2 + 1;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, rePE)) {
    gc.col = std::stoi(m[2]) * 2 + 1;
    gc.row = std::stoi(m[1]) * 2 + 1;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, reTPE_LETTER)) {
    int ordinal = m[1].str()[0] - 'a';
    gc.col = 1;
    gc.row = ordinal * 2 + 1;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, reTPE_NUM)) {
    int ordinal = std::stoi(m[1]);
    gc.col = 1;
    gc.row = ordinal * 2 + 1;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, rePE_MESH_1D)) {
    int offset = meshLetterOffset(m[1].str()[0], meshBandSize, meshBaseOffset);
    int idx = std::stoi(m[2]);
    gc.col = offset + idx * 2 + 1;
    gc.row = 1;
    gc.valid = true;
  }

  return gc;
}

// ---- Neighbor-based coordinate inference ----

void inferMissingCoords(
    const Graph &adg,
    llvm::DenseMap<IdIndex, GridCoord> &nodeCoords,
    const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
      const Node *node = adg.getNode(i);
      if (!node || fuToContainer.count(i))
        continue;
      if (nodeCoords.count(i) && nodeCoords[i].valid)
        continue;
      if (node->kind == Node::ModuleInputNode ||
          node->kind == Node::ModuleOutputNode)
        continue;
      int sumCol = 0, sumRow = 0, count = 0;
      auto collectNeighbor = [&](IdIndex parentNode) {
        IdIndex resolved = resolveContainer(parentNode, fuToContainer);
        if (nodeCoords.count(resolved) && nodeCoords[resolved].valid) {
          sumCol += nodeCoords[resolved].col;
          sumRow += nodeCoords[resolved].row;
          count++;
        }
      };

      for (IdIndex pid : node->inputPorts) {
        const Port *p = adg.getPort(pid);
        if (!p) continue;
        for (IdIndex eid : p->connectedEdges) {
          const Edge *e = adg.getEdge(eid);
          if (!e) continue;
          const Port *sp = adg.getPort(e->srcPort);
          if (sp) collectNeighbor(sp->parentNode);
        }
      }
      for (IdIndex pid : node->outputPorts) {
        const Port *p = adg.getPort(pid);
        if (!p) continue;
        for (IdIndex eid : p->connectedEdges) {
          const Edge *e = adg.getEdge(eid);
          if (!e) continue;
          const Port *dp = adg.getPort(e->dstPort);
          if (dp) collectNeighbor(dp->parentNode);
        }
      }

      if (count > 0) {
        GridCoord gc;
        gc.col = (sumCol + count / 2) / count;
        gc.row = (sumRow + count / 2) / count;
        gc.valid = true;
        gc.inferred = true;
        nodeCoords[i] = gc;
        changed = true;
      }
    }
  }
}

// ---- Width-plane helpers ----

static bool isCrossPlaneNode(const Node *node, const Graph &adg) {
  if (isRoutingNode(node)) return false;
  llvm::DenseSet<int> widths;
  for (IdIndex pid : node->inputPorts) {
    const Port *p = adg.getPort(pid);
    if (p && p->type && !mlir::isa<mlir::NoneType>(p->type))
      widths.insert(bitWidthFromType(p->type));
  }
  for (IdIndex pid : node->outputPorts) {
    const Port *p = adg.getPort(pid);
    if (p && p->type && !mlir::isa<mlir::NoneType>(p->type))
      widths.insert(bitWidthFromType(p->type));
  }
  return widths.size() > 1;
}

static int getRoutingBitWidth(const Node *node, const Graph &adg) {
  for (IdIndex pid : node->outputPorts) {
    const Port *p = adg.getPort(pid);
    if (p && p->type)
      return bitWidthFromType(p->type);
  }
  for (IdIndex pid : node->inputPorts) {
    const Port *p = adg.getPort(pid);
    if (p && p->type)
      return bitWidthFromType(p->type);
  }
  return 32;
}

static std::string getRoutingTypeKey(const Node *node, const Graph &adg) {
  for (IdIndex pid : node->outputPorts) {
    const Port *p = adg.getPort(pid);
    if (p && p->type) {
      if (mlir::isa<mlir::NoneType>(p->type))
        return "none";
      std::string str;
      llvm::raw_string_ostream os(str);
      p->type.print(os);
      return str;
    }
  }
  for (IdIndex pid : node->inputPorts) {
    const Port *p = adg.getPort(pid);
    if (p && p->type) {
      if (mlir::isa<mlir::NoneType>(p->type))
        return "none";
      std::string str;
      llvm::raw_string_ostream os(str);
      p->type.print(os);
      return str;
    }
  }
  return "unknown";
}

// ---- Plane-aware coordinate inference ----

void inferMissingCoordsPlaneAware(
    const Graph &adg,
    llvm::DenseMap<IdIndex, GridCoord> &nodeCoords,
    const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer,
    const llvm::DenseMap<IdIndex, int> &nodePlane,
    int crossPlaneIdx) {

  bool changed = true;
  while (changed) {
    changed = false;
    for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
      const Node *node = adg.getNode(i);
      if (!node || fuToContainer.count(i))
        continue;
      if (nodeCoords.count(i) && nodeCoords[i].valid)
        continue;
      if (node->kind == Node::ModuleInputNode ||
          node->kind == Node::ModuleOutputNode)
        continue;

      int myPlane = -1;
      auto myPlaneIt = nodePlane.find(i);
      if (myPlaneIt != nodePlane.end())
        myPlane = myPlaneIt->second;

      int sumCol = 0, sumRow = 0, count = 0;
      auto collectNeighbor = [&](IdIndex parentNode) {
        IdIndex resolved = resolveContainer(parentNode, fuToContainer);
        if (myPlane >= 0 && myPlane != crossPlaneIdx) {
          auto nbIt = nodePlane.find(resolved);
          if (nbIt == nodePlane.end() || nbIt->second != myPlane)
            return;
        }
        if (nodeCoords.count(resolved) && nodeCoords[resolved].valid) {
          sumCol += nodeCoords[resolved].col;
          sumRow += nodeCoords[resolved].row;
          count++;
        }
      };

      for (IdIndex pid : node->inputPorts) {
        const Port *p = adg.getPort(pid);
        if (!p) continue;
        for (IdIndex eid : p->connectedEdges) {
          const Edge *e = adg.getEdge(eid);
          if (!e) continue;
          const Port *sp = adg.getPort(e->srcPort);
          if (sp) collectNeighbor(sp->parentNode);
        }
      }
      for (IdIndex pid : node->outputPorts) {
        const Port *p = adg.getPort(pid);
        if (!p) continue;
        for (IdIndex eid : p->connectedEdges) {
          const Edge *e = adg.getEdge(eid);
          if (!e) continue;
          const Port *dp = adg.getPort(e->dstPort);
          if (dp) collectNeighbor(dp->parentNode);
        }
      }

      if (count > 0) {
        GridCoord gc;
        gc.col = (sumCol + count / 2) / count;
        gc.row = (sumRow + count / 2) / count;
        gc.valid = true;
        gc.inferred = true;
        nodeCoords[i] = gc;
        changed = true;
      }
    }
  }
}

// ---- Width-plane detection and column separation ----

llvm::DenseMap<IdIndex, int> detectAndSeparateWidthPlanes(
    const Graph &adg,
    llvm::DenseMap<IdIndex, GridCoord> &nodeCoords,
    const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer,
    llvm::DenseMap<int, std::string> &outPlaneLabels) {

  llvm::DenseMap<IdIndex, int> nodePlane;

  // Check if separation is needed: detect PE coordinate collisions.
  llvm::DenseMap<std::pair<int, int>, llvm::SmallVector<IdIndex, 2>> pePositions;
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node || fuToContainer.count(i)) continue;
    std::string type = nodeTypeStr(node);
    if (type != "fabric.pe" && type != "fabric.temporal_pe") continue;
    auto it = nodeCoords.find(i);
    if (it == nodeCoords.end() || !it->second.valid) continue;
    pePositions[{it->second.col, it->second.row}].push_back(i);
  }

  bool hasCollision = false;
  for (auto &kv : pePositions) {
    if (kv.second.size() > 1) { hasCollision = true; break; }
  }
  if (!hasCollision) return nodePlane;

  // Build connected components among routing nodes.
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> routingAdj;
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.edges.size()); ++i) {
    const Edge *e = adg.getEdge(i);
    if (!e) continue;
    const Port *sp = adg.getPort(e->srcPort);
    const Port *dp = adg.getPort(e->dstPort);
    if (!sp || !dp) continue;

    IdIndex srcId = sp->parentNode;
    IdIndex dstId = dp->parentNode;
    if (fuToContainer.count(srcId)) srcId = fuToContainer.lookup(srcId);
    if (fuToContainer.count(dstId)) dstId = fuToContainer.lookup(dstId);

    const Node *srcNode = adg.getNode(srcId);
    const Node *dstNode = adg.getNode(dstId);
    if (!srcNode || !dstNode) continue;
    if (!isRoutingNode(srcNode) || !isRoutingNode(dstNode)) continue;

    routingAdj[srcId].push_back(dstId);
    routingAdj[dstId].push_back(srcId);
  }

  // BFS to find connected components.
  llvm::DenseMap<IdIndex, int> routingComponent;
  int numComponents = 0;
  for (auto &kv : routingAdj) {
    IdIndex startId = kv.first;
    if (routingComponent.count(startId)) continue;
    int comp = numComponents++;
    std::queue<IdIndex> queue;
    queue.push(startId);
    routingComponent[startId] = comp;
    while (!queue.empty()) {
      IdIndex cur = queue.front();
      queue.pop();
      auto adjIt = routingAdj.find(cur);
      if (adjIt == routingAdj.end()) continue;
      for (IdIndex nb : adjIt->second) {
        if (!routingComponent.count(nb)) {
          routingComponent[nb] = comp;
          queue.push(nb);
        }
      }
    }
  }

  // Assign isolated routing nodes their own component.
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node || fuToContainer.count(i)) continue;
    if (!isRoutingNode(node)) continue;
    if (routingComponent.count(i)) continue;
    routingComponent[i] = numComponents++;
  }

  // Collect per-component metadata.
  llvm::DenseMap<int, int> componentWidth;
  std::map<int, std::string> componentTypeKey;
  for (auto &kv : routingComponent) {
    const Node *node = adg.getNode(kv.first);
    if (!node || !isRoutingNode(node)) continue;
    int comp = kv.second;
    if (!componentTypeKey.count(comp)) {
      componentWidth[comp] = getRoutingBitWidth(node, adg);
      componentTypeKey[comp] = getRoutingTypeKey(node, adg);
    }
  }

  // Sort components by (bitWidth, typeKey) for visual ordering.
  struct CompInfo {
    int compId;
    int bitWidth;
    std::string typeKey;
  };
  llvm::SmallVector<CompInfo, 8> compInfos;
  for (auto &kv : componentTypeKey) {
    auto wIt = componentWidth.find(kv.first);
    int bw = wIt != componentWidth.end() ? wIt->second : 0;
    compInfos.push_back({kv.first, bw, kv.second});
  }
  std::sort(compInfos.begin(), compInfos.end(),
            [](const CompInfo &a, const CompInfo &b) {
    if (a.bitWidth != b.bitWidth) return a.bitWidth < b.bitWidth;
    return a.typeKey < b.typeKey;
  });

  llvm::DenseMap<int, int> compToPlane;
  for (int idx = 0; idx < static_cast<int>(compInfos.size()); ++idx)
    compToPlane[compInfos[idx].compId] = idx;

  // Build plane labels.
  for (int idx = 0; idx < static_cast<int>(compInfos.size()); ++idx) {
    if (compInfos[idx].typeKey == "none")
      outPlaneLabels[idx] = "ctrl";
    else
      outPlaneLabels[idx] = std::to_string(compInfos[idx].bitWidth) + "b";
  }

  int numPlanes = static_cast<int>(compInfos.size());
  if (numPlanes <= 1) return nodePlane;

  // Assign non-routing nodes to planes based on most-connected component.
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node || fuToContainer.count(i)) continue;
    if (isRoutingNode(node)) continue;
    if (node->kind == Node::ModuleInputNode ||
        node->kind == Node::ModuleOutputNode)
      continue;

    llvm::DenseMap<int, int> compEdgeCount;
    auto countNeighbor = [&](IdIndex neighborId) {
      if (fuToContainer.count(neighborId))
        neighborId = fuToContainer.lookup(neighborId);
      auto compIt = routingComponent.find(neighborId);
      if (compIt != routingComponent.end())
        compEdgeCount[compIt->second]++;
    };

    for (IdIndex pid : node->inputPorts) {
      const Port *p = adg.getPort(pid);
      if (!p) continue;
      if (p->type && mlir::isa<mlir::NoneType>(p->type))
        continue;
      for (IdIndex eid : p->connectedEdges) {
        const Edge *e = adg.getEdge(eid);
        if (!e) continue;
        const Port *sp = adg.getPort(e->srcPort);
        if (sp) countNeighbor(sp->parentNode);
      }
    }
    for (IdIndex pid : node->outputPorts) {
      const Port *p = adg.getPort(pid);
      if (!p) continue;
      if (p->type && mlir::isa<mlir::NoneType>(p->type))
        continue;
      for (IdIndex eid : p->connectedEdges) {
        const Edge *e = adg.getEdge(eid);
        if (!e) continue;
        const Port *dp = adg.getPort(e->dstPort);
        if (dp) countNeighbor(dp->parentNode);
      }
    }

    // Fallback: count ALL ports including NoneType.
    if (compEdgeCount.empty()) {
      for (IdIndex pid : node->inputPorts) {
        const Port *p = adg.getPort(pid);
        if (!p) continue;
        for (IdIndex eid : p->connectedEdges) {
          const Edge *e = adg.getEdge(eid);
          if (!e) continue;
          const Port *sp = adg.getPort(e->srcPort);
          if (sp) countNeighbor(sp->parentNode);
        }
      }
      for (IdIndex pid : node->outputPorts) {
        const Port *p = adg.getPort(pid);
        if (!p) continue;
        for (IdIndex eid : p->connectedEdges) {
          const Edge *e = adg.getEdge(eid);
          if (!e) continue;
          const Port *dp = adg.getPort(e->dstPort);
          if (dp) countNeighbor(dp->parentNode);
        }
      }
    }

    int bestComp = -1;
    int bestCount = 0;
    for (auto &ce : compEdgeCount) {
      if (ce.second > bestCount) {
        bestCount = ce.second;
        bestComp = ce.first;
      }
    }
    if (bestComp >= 0)
      routingComponent[i] = bestComp;
  }

  // Record plane assignments. Cross-plane nodes get a special plane index.
  int crossPlaneIdx = numPlanes;
  for (auto &kv : routingComponent) {
    const Node *nd = adg.getNode(kv.first);
    if (nd && !isRoutingNode(nd) && isCrossPlaneNode(nd, adg)) {
      nodePlane[kv.first] = crossPlaneIdx;
      continue;
    }
    auto planeIt = compToPlane.find(kv.second);
    if (planeIt != compToPlane.end())
      nodePlane[kv.first] = planeIt->second;
  }

  bool hasCrossPlane = false;
  for (auto &kv : nodePlane) {
    if (kv.second == crossPlaneIdx) { hasCrossPlane = true; break; }
  }
  if (hasCrossPlane)
    outPlaneLabels[crossPlaneIdx] = "cross";

  // Compute local extents for the 2D plane matrix layout.
  int maxLocalCol = 0;
  int maxLocalRow = 0;
  for (auto &kv : nodeCoords) {
    if (kv.second.valid) {
      maxLocalCol = std::max(maxLocalCol, kv.second.col);
      maxLocalRow = std::max(maxLocalRow, kv.second.row);
    }
  }
  int bandGap = 4;
  int planeBandCol = maxLocalCol + bandGap + 1;
  int planeBandRow = maxLocalRow + bandGap + 1;

  int planeGridCols = static_cast<int>(
      std::ceil(std::sqrt(static_cast<double>(numPlanes))));

  auto planeColOffset = [&](int planeIdx) -> int {
    return (planeIdx % planeGridCols) * planeBandCol;
  };
  auto planeRowOffset = [&](int planeIdx) -> int {
    return (planeIdx / planeGridCols) * planeBandRow;
  };

  // Apply 2D offsets to nodes with known coordinates.
  for (auto &kv : nodePlane) {
    IdIndex nodeId = kv.first;
    int planeIdx = kv.second;

    const Node *node = adg.getNode(nodeId);
    if (node) {
      llvm::StringRef sn = getNodeStrAttr(node, "sym_name");
      if (sn.starts_with("sw_w"))
        continue;
      if (sn.size() > 2 && sn[0] == 'l' && std::isdigit(sn[1]) &&
          sn.contains("_sw_"))
        continue;
      if (getNodeIntAttr(node, "viz_col", -1) >= 0)
        continue;
    }

    auto coordIt = nodeCoords.find(nodeId);
    if (coordIt != nodeCoords.end() && coordIt->second.valid) {
      coordIt->second.col += planeColOffset(planeIdx);
      coordIt->second.row += planeRowOffset(planeIdx);
    }
  }

  // Seed grid positions for unplaced switches in every plane.
  for (int plane = 0; plane < numPlanes; ++plane) {
    llvm::SmallVector<IdIndex, 64> unplacedSwitches;
    for (auto &kv : nodePlane) {
      if (kv.second != plane) continue;
      const Node *nd = adg.getNode(kv.first);
      if (!nd || !isRoutingNode(nd)) continue;
      if (nodeCoords.count(kv.first) && nodeCoords[kv.first].valid)
        continue;
      unplacedSwitches.push_back(kv.first);
    }
    if (unplacedSwitches.empty()) continue;

    std::sort(unplacedSwitches.begin(), unplacedSwitches.end());

    int N = static_cast<int>(unplacedSwitches.size());
    int meshCols = static_cast<int>(
        std::ceil(std::sqrt(static_cast<double>(N))));
    int colOff = planeColOffset(plane);
    int rowOff = planeRowOffset(plane);

    for (int idx = 0; idx < N; ++idx) {
      int r = idx / meshCols;
      int c = idx % meshCols;
      GridCoord gc;
      gc.col = colOff + c * 2;
      gc.row = rowOff + r * 2;
      gc.valid = true;
      gc.inferred = true;
      nodeCoords[unplacedSwitches[idx]] = gc;
    }
  }

  // Run plane-aware coordinate inference for remaining unplaced nodes.
  inferMissingCoordsPlaneAware(adg, nodeCoords, fuToContainer, nodePlane,
                               crossPlaneIdx);

  return nodePlane;
}

} // namespace loom
