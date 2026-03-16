// fcc Visualization Renderer
// Dual-panel viewer: D3.js ADG grid + Graphviz DFG
// Based on loom's visualization architecture.
//
// Expects globals: adgGraph, DFG_DATA, mappingData, TRACE

(function() {
"use strict";

// --- Constants ---
var MIN_CELL = 40;
var CELL = MIN_CELL;  // Recalculated dynamically
var HOP_R = 4;
var LANE_W = 2;
var GRID_SP = 2;      // Tight spacing — PE and SW are adjacent hardware
var GRID_PAD = 2;
var FU_OP_NODE_W = 44; // Width of an op node inside FU
var FU_OP_NODE_H = 18; // Height of an op node inside FU
var FU_OP_ARROW_GAP = 6; // Gap between op nodes for arrow
var FU_GAP = 4;        // Gap between FU boxes
var FU_BOX_H = 30;     // Single-op FU box height
var PE_TITLE_H = 22;   // PE header height
var PE_PAD = 8;        // Padding inside PE
var FU_CHAR_W = 6.5;   // Char width for monospace FU labels
var MIN_FU_BOX_W = 60; // Min FU box width
var nodeSize = {};   // Per-node {w,h} computed dynamically
var ROUTE_PALETTE = [
  "#e6194b","#3cb44b","#4363d8","#f58231",
  "#911eb4","#42d4f4","#f032e6","#bfef45",
  "#fabed4","#469990","#dcbeff","#9A6324"
];

var NODE_COLORS = {
  "fabric.spatial_pe":  { fill: "#1a2f4a", border: "#2a5f8f", text: "#c8d6e5" },
  "fabric.spatial_sw":  { fill: "#2a3f5f", border: "#4a6380", text: "#8a9bb5" },
  "fabric.memory":      { fill: "#1a3f4a", border: "#2a7f8f", text: "#8ad6e5" },
  "fabric.extmemory":   { fill: "#3a2f1a", border: "#8f7f2a", text: "#e5d68a" },
  "fabric.fifo":        { fill: "#252535", border: "#4a4a6a", text: "#9a9ab5" },
  "input":              { fill: "#2a1f3a", border: "#6a4f8f", text: "#c8a5e5" },
  "output":             { fill: "#3a1f1f", border: "#8f4f4f", text: "#e5a5a5" }
};

var TYPE_SHORT = {
  "fabric.spatial_pe": "PE", "fabric.spatial_sw": "SW",
  "fabric.memory": "MEM", "fabric.extmemory": "EXTM",
  "fabric.fifo": "FIFO", "input": "IN", "output": "OUT"
};

// --- State ---
var currentMode = "sidebyside";
var adgZoom, dfgZoom;
var adgSvgG, dfgSvgG;
var savedNodeFills = {};

var hwNodeMap = {};
var adgAdj = {};

if (typeof adgGraph !== "undefined" && adgGraph) {
  adgGraph.nodes.forEach(function(n) {
    hwNodeMap[n.id] = n;
    adgAdj[n.id] = [];
  });
  adgGraph.edges.forEach(function(e) {
    if (adgAdj[e.srcNode]) adgAdj[e.srcNode].push({ edge: e.id, node: e.dstNode });
    if (adgAdj[e.dstNode]) adgAdj[e.dstNode].push({ edge: e.id, node: e.srcNode });
  });
}

// --- Utility ---
function nodeColorScheme(n) {
  var key = n.type || "fabric.spatial_pe";
  return NODE_COLORS[key] || NODE_COLORS["fabric.spatial_pe"];
}

function truncLabel(s, maxLen) {
  if (!s) return "";
  if (s.length <= maxLen) return s;
  return s.substring(0, maxLen - 2) + "..";
}

function ff(v) { return v.toFixed(1); }

// Strip dialect prefix from op names for cleaner display
function shortOpName(opName) {
  if (!opName) return "";
  if (opName.indexOf("static_mux") >= 0) return "mux(sel)";
  var dot = opName.lastIndexOf(".");
  if (dot >= 0) return opName.substring(dot + 1);
  return opName;
}

// Compute per-FU box height based on number of ops
function fuBoxHeight(fu) {
  var numOps = (fu.ops && fu.ops.length > 0) ? fu.ops.length : 1;
  if (numOps <= 1) return FU_BOX_H;
  // DAG layout: arrange ops in rows (wider, not tall)
  var dagLayout = computeFUDagLayout(fu.ops);
  return 14 + dagLayout.rows * (FU_OP_NODE_H + FU_OP_ARROW_GAP) + 8;
}

// Compute DAG layout for multi-op FU: row assignment and connections
function computeFUDagLayout(ops) {
  if (!ops || ops.length <= 1) return { rows: 1, cols: 1, nodes: [] };
  var n = ops.length;
  // Heuristic: mux is input (row 0), other compute ops in subsequent rows
  // Last non-mux op is output (last row)
  var muxIdx = -1;
  for (var i = 0; i < n; i++) {
    if (ops[i].indexOf("static_mux") >= 0) { muxIdx = i; break; }
  }
  // Assign rows: for a typical MAC (muli, addi, mux), arrange as:
  //   Row 0: muli (+ mux if present as selector)
  //   Row 1: addi (output)
  // For general case: spread ops across rows
  var nodes = [];
  var nonMux = [];
  for (var i = 0; i < n; i++) {
    if (i === muxIdx) continue;
    nonMux.push(i);
  }
  // Layer 0: first half of non-mux ops + mux
  // Layer 1: second half of non-mux ops
  var layer0Count = Math.ceil(nonMux.length / 2);
  var layer0 = nonMux.slice(0, layer0Count);
  var layer1 = nonMux.slice(layer0Count);

  // Add mux to layer 0 if present
  if (muxIdx >= 0) layer0.unshift(muxIdx);

  var maxCols = Math.max(layer0.length, layer1.length);
  var rows = layer1.length > 0 ? 2 : 1;

  for (var i = 0; i < layer0.length; i++) {
    nodes.push({ opIdx: layer0[i], row: 0, col: i, totalInRow: layer0.length });
  }
  for (var i = 0; i < layer1.length; i++) {
    nodes.push({ opIdx: layer1[i], row: 1, col: i, totalInRow: layer1.length });
  }

  // Connections: each layer-1 op connects from all layer-0 ops (fan-in)
  var edges = [];
  for (var i = 0; i < layer0.length; i++) {
    for (var j = 0; j < layer1.length; j++) {
      edges.push({ from: layer0[i], to: layer1[j] });
    }
  }
  // If only 1 layer with multiple ops, connect sequentially
  if (rows === 1 && layer0.length > 1) {
    for (var i = 0; i < layer0.length - 1; i++) {
      edges.push({ from: layer0[i], to: layer0[i + 1] });
    }
  }

  return { rows: rows, cols: maxCols, nodes: nodes, edges: edges };
}

// Build a lookup: for a given PE hw node id, return the mapped FU name (if any)
function getMappedFUForPE(hwNodeId) {
  if (!mappingData || !mappingData.placements) return null;
  for (var i = 0; i < mappingData.placements.length; i++) {
    var p = mappingData.placements[i];
    if (p.adgNode === hwNodeId) return p.hwName;
  }
  return null;
}

// =================================================================
// Grid Layout Engine
// =================================================================
function gridLayoutEngine(allNodes, edges) {
  var nodeById = {};
  var internal = [];
  var sentinels = [];
  allNodes.forEach(function(n) {
    nodeById[n.id] = n;
    if (n.class === "boundary") sentinels.push(n);
    else internal.push(n);
  });

  var N = internal.length;
  var idxOf = {};
  internal.forEach(function(n, i) { idxOf[n.id] = i; });

  // Build adjacency
  var adjList = new Array(N);
  for (var i = 0; i < N; i++) adjList[i] = [];
  edges.forEach(function(e) {
    var si = idxOf[e.srcNode], di = idxOf[e.dstNode];
    if (si === undefined || di === undefined || si === di) return;
    if (adjList[si].indexOf(di) === -1) adjList[si].push(di);
    if (adjList[di].indexOf(si) === -1) adjList[di].push(si);
  });

  var occ = {};
  var gpos = {};
  function ck(gc, gr) { return gc + "," + gr; }
  function occupy(nid, gc, gr) {
    gpos[nid] = { gc: gc, gr: gr, gw: 1, gh: 1 };
    occ[ck(gc, gr)] = nid;
  }

  // Place nodes with explicit coordinates
  var placed = {};
  internal.forEach(function(n) {
    if (n.gridCol != null && n.gridRow != null) {
      var gc = n.gridCol * GRID_SP + GRID_PAD;
      var gr = n.gridRow * GRID_SP + GRID_PAD;
      placed[n.id] = { gc: gc, gr: gr };
    }
  });

  // Propagate to unplaced nodes via neighbor centroids
  for (var pass = 0; pass < 10; pass++) {
    var progress = false;
    for (var i = 0; i < N; i++) {
      if (placed[internal[i].id]) continue;
      var sx = 0, sy = 0, cnt = 0;
      var nb = adjList[i];
      for (var k = 0; k < nb.length; k++) {
        var nbId = internal[nb[k]].id;
        if (placed[nbId]) {
          sx += placed[nbId].gc;
          sy += placed[nbId].gr;
          cnt++;
        }
      }
      if (cnt > 0) {
        placed[internal[i].id] = {
          gc: Math.round(sx / cnt),
          gr: Math.round(sy / cnt)
        };
        progress = true;
      }
    }
    if (!progress) break;
  }

  // Compute grid extent
  var maxC = 0, maxR = 0;
  var minC = Infinity, minR = Infinity;
  Object.keys(placed).forEach(function(nid) {
    var p = placed[nid];
    if (p.gc > maxC) maxC = p.gc;
    if (p.gr > maxR) maxR = p.gr;
    if (p.gc < minC) minC = p.gc;
    if (p.gr < minR) minR = p.gr;
  });
  if (minC === Infinity) { minC = GRID_PAD; maxC = GRID_PAD + 10; minR = GRID_PAD; maxR = GRID_PAD + 10; }

  var gridW = maxC + GRID_PAD + 2;
  var gridH = maxR + GRID_PAD + 2;

  function isFree(gc, gr) {
    return !occ[ck(gc, gr)];
  }
  function findFree(ic, ir) {
    ic = Math.round(ic); ir = Math.round(ir);
    if (isFree(ic, ir)) return { gc: ic, gr: ir };
    for (var d = 1; d <= 50; d++) {
      for (var dx = -d; dx <= d; dx++) {
        var ady = d - Math.abs(dx);
        var cands = ady === 0 ? [0] : [-ady, ady];
        for (var ci = 0; ci < cands.length; ci++) {
          if (isFree(ic + dx, ir + cands[ci]))
            return { gc: ic + dx, gr: ir + cands[ci] };
        }
      }
    }
    return { gc: gridW, gr: gridH };
  }

  // Occupy with collision avoidance, sorted by degree (most-connected first)
  var targets = [];
  internal.forEach(function(n) {
    var p = placed[n.id];
    if (!p) {
      p = { gc: Math.round((minC + maxC) / 2), gr: Math.round((minR + maxR) / 2) };
    }
    var idx = idxOf[n.id];
    targets.push({ node: n, gc: p.gc, gr: p.gr, degree: idx !== undefined ? adjList[idx].length : 0 });
  });
  targets.sort(function(a, b) { return b.degree - a.degree; });
  targets.forEach(function(t) {
    var spot = findFree(t.gc, t.gr);
    occupy(t.node.id, spot.gc, spot.gr);
  });

  // Place sentinels above and below grid
  var inSent = sentinels.filter(function(s) { return s.type !== "output"; });
  var outSent = sentinels.filter(function(s) { return s.type === "output"; });

  function sentinelIdealCol(s) {
    var sx = 0, cnt = 0;
    edges.forEach(function(e) {
      var peer = null;
      if (e.srcNode === s.id) peer = e.dstNode;
      else if (e.dstNode === s.id) peer = e.srcNode;
      if (peer && gpos[peer]) { sx += gpos[peer].gc; cnt++; }
    });
    return cnt > 0 ? Math.round(sx / cnt) : GRID_PAD;
  }

  inSent.forEach(function(s) {
    var gc = sentinelIdealCol(s);
    var spot = findFree(gc, -2);
    occupy(s.id, spot.gc, spot.gr);
  });
  outSent.forEach(function(s) {
    var gc = sentinelIdealCol(s);
    var spot = findFree(gc, gridH + 1);
    occupy(s.id, spot.gc, spot.gr);
  });

  return { gpos: gpos, occ: occ, gridW: gridW, gridH: gridH };
}

// =================================================================
// Port Position Computation
// =================================================================
function computeGridPorts(nodes, edges, gpos) {
  var portInfo = {};
  var nodeOut = {}, nodeIn = {};

  edges.forEach(function(e) {
    if (!nodeOut[e.srcNode]) nodeOut[e.srcNode] = [];
    if (!nodeIn[e.dstNode]) nodeIn[e.dstNode] = [];
    if (!portInfo[e.srcPort]) {
      nodeOut[e.srcNode].push(e.srcPort);
      portInfo[e.srcPort] = { nodeId: e.srcNode, isOutput: true, peers: [] };
    }
    portInfo[e.srcPort].peers.push(e.dstNode);
    if (!portInfo[e.dstPort]) {
      nodeIn[e.dstNode].push(e.dstPort);
      portInfo[e.dstPort] = { nodeId: e.dstNode, isOutput: false, peers: [] };
    }
    portInfo[e.dstPort].peers.push(e.srcNode);
  });

  // Compute per-node sizes based on type (module-level for use in rendering)
  nodeSize = {};
  nodes.forEach(function(n) {
    var w, h;
    if (n.type === "fabric.spatial_pe" && n.fuDetails && n.fuDetails.length > 0) {
      // Three-level sizing: PE > FU > ops (mini-DFG inside each FU)
      // Use square grid layout for FUs
      var numFU = n.fuDetails.length;
      var fuCols = Math.ceil(Math.sqrt(numFU));
      var fuRows = Math.ceil(numFU / fuCols);

      // Compute per-row max FU height (not global max)
      var rowMaxH = [];
      for (var ri = 0; ri < fuRows; ri++) rowMaxH.push(FU_BOX_H);
      n.fuDetails.forEach(function(fu, fi) {
        var row = Math.floor(fi / fuCols);
        rowMaxH[row] = Math.max(rowMaxH[row], fuBoxHeight(fu));
      });
      var totalFuH = 0;
      for (var ri = 0; ri < fuRows; ri++) totalFuH += rowMaxH[ri] + FU_GAP;

      var fuW = MIN_FU_BOX_W;
      w = fuW * fuCols + (fuCols - 1) * (FU_GAP + 2) + PE_PAD * 2;
      h = PE_TITLE_H + totalFuH + PE_PAD;
      w = Math.max(w, 140);
      h = Math.max(h, 100);
    } else if (n.type === "fabric.spatial_sw") {
      // Size proportional to port count, show internal connections
      var swInPorts = (n.ports && n.ports.in) ? n.ports.in : 4;
      var swOutPorts = (n.ports && n.ports.out) ? n.ports.out : 4;
      var swMaxPorts = Math.max(swInPorts, swOutPorts);
      var swSide = Math.max(60, swMaxPorts * 7 + 20);
      w = swSide; h = swSide;
    } else if (n.type === "fabric.extmemory") {
      w = 68; h = 34;
    } else if (n.type === "fabric.memory") {
      w = 68; h = 34;
    } else if (n.type === "input" || n.type === "output") {
      w = 52; h = 26;
    } else {
      w = 34; h = 34;
    }
    nodeSize[n.id] = { w: w, h: h };
  });

  var portPos = {};
  nodes.forEach(function(n) {
    var np = gpos[n.id];
    if (!np) return;
    var ns = nodeSize[n.id] || { w: CELL - 4, h: CELL - 4 };
    var cx = (np.gc + 0.5) * CELL;
    var cy = (np.gr + 0.5) * CELL;
    var halfW = ns.w / 2;
    var halfH = ns.h / 2;

    var allPorts = (nodeIn[n.id] || []).concat(nodeOut[n.id] || []);
    var sides = { top: [], right: [], bottom: [], left: [] };

    allPorts.forEach(function(pid) {
      var pi = portInfo[pid];
      if (!pi) return;
      var dx = 0, dy = 0, cnt = 0;
      pi.peers.forEach(function(peerId) {
        var pp = gpos[peerId];
        if (pp) {
          dx += (pp.gc + 0.5) * CELL - cx;
          dy += (pp.gr + 0.5) * CELL - cy;
          cnt++;
        }
      });
      if (cnt === 0) { sides.right.push(pid); return; }
      var angle = Math.atan2(dy / cnt, dx / cnt);
      if (angle >= -Math.PI / 4 && angle < Math.PI / 4) sides.right.push(pid);
      else if (angle >= Math.PI / 4 && angle < 3 * Math.PI / 4) sides.bottom.push(pid);
      else if (angle >= -3 * Math.PI / 4 && angle < -Math.PI / 4) sides.top.push(pid);
      else sides.left.push(pid);
    });

    function placeOnSide(ports, axis, fixedVal, start, end, side) {
      if (ports.length === 0) return;
      var step = (end - start) / (ports.length + 1);
      ports.forEach(function(pid, i) {
        var t = start + step * (i + 1);
        if (axis === "h") portPos[pid] = { x: cx + t, y: cy + fixedVal, side: side };
        else portPos[pid] = { x: cx + fixedVal, y: cy + t, side: side };
      });
    }
    placeOnSide(sides.top, "h", -halfH, -halfW, halfW, "top");
    placeOnSide(sides.right, "v", halfW, -halfH, halfH, "right");
    placeOnSide(sides.bottom, "h", halfH, -halfW, halfW, "bottom");
    placeOnSide(sides.left, "v", -halfW, -halfH, halfH, "left");
  });

  return { pos: portPos, info: portInfo, nodeIn: nodeIn, nodeOut: nodeOut };
}

// =================================================================
// Manhattan Router (BFS through free grid cells)
// =================================================================
function manhattanRouter(edges, layout) {
  var gpos = layout.gpos;
  var occ = layout.occ;
  var gridW = layout.gridW, gridH = layout.gridH;

  function ck(gc, gr) { return gc + "," + gr; }

  function adjFree(p) {
    var res = [];
    for (var dc = -1; dc <= 1; dc++) {
      for (var dr = -1; dr <= 1; dr++) {
        if (dc === 0 && dr === 0) continue;
        if (dc !== 0 && dr !== 0) continue;
        var gc = p.gc + dc, gr = p.gr + dr;
        if (!occ[ck(gc, gr)]) res.push({ gc: gc, gr: gr });
      }
    }
    return res;
  }

  function bfs(srcId, dstId) {
    var sp = gpos[srcId], dp = gpos[dstId];
    if (!sp || !dp) return null;
    var hGap = Math.abs(dp.gc - sp.gc);
    var vGap = Math.abs(dp.gr - sp.gr);
    if (hGap + vGap <= 2) return [];

    var starts = adjFree(sp);
    var endSet = {};
    adjFree(dp).forEach(function(c) { endSet[ck(c.gc, c.gr)] = true; });
    if (starts.length === 0 || Object.keys(endSet).length === 0) return null;

    var visited = {};
    var prev = {};
    var queue = [];
    starts.forEach(function(s) {
      var k = ck(s.gc, s.gr);
      visited[k] = true; prev[k] = null;
      queue.push(s);
    });

    var head = 0;
    var bound = Math.max(gridW, gridH) + 15;
    var dirs4 = [[1,0],[-1,0],[0,1],[0,-1]];
    while (head < queue.length) {
      var cur = queue[head++];
      var curK = ck(cur.gc, cur.gr);
      if (endSet[curK]) {
        var path = [];
        var k = curK;
        while (k) {
          var parts = k.split(",");
          path.unshift({ gc: parseInt(parts[0]), gr: parseInt(parts[1]) });
          k = prev[k];
        }
        return path;
      }
      for (var di = 0; di < 4; di++) {
        var nc = cur.gc + dirs4[di][0], nr = cur.gr + dirs4[di][1];
        var nk = ck(nc, nr);
        if (visited[nk] || occ[nk]) continue;
        if (nc < -bound || nr < -bound || nc > bound || nr > bound) continue;
        visited[nk] = true;
        prev[nk] = curK;
        queue.push({ gc: nc, gr: nr });
      }
    }
    return null;
  }

  // Route all edges and track segment usage for lane offsets
  var edgePaths = {};
  var segCount = {};
  var segIdx = {};

  function sk(gc1, gr1, gc2, gr2) {
    if (gc1 < gc2 || (gc1 === gc2 && gr1 < gr2))
      return gc1 + "," + gr1 + "-" + gc2 + "," + gr2;
    return gc2 + "," + gr2 + "-" + gc1 + "," + gr1;
  }

  // Cell direction tracking for crossing detection
  var cellEdges = {};

  function addCellDir(gc, gr, edgeId, dir) {
    var k = ck(gc, gr);
    if (!cellEdges[k]) cellEdges[k] = [];
    var arr = cellEdges[k];
    for (var i = 0; i < arr.length; i++)
      if (arr[i].edgeId === edgeId && arr[i].dir === dir) return;
    arr.push({ edgeId: edgeId, dir: dir });
  }

  var routeStart = performance.now();
  edges.forEach(function(e) {
    if (performance.now() - routeStart > 2000) {
      edgePaths[e.id] = null;
      return;
    }
    var path = bfs(e.srcNode, e.dstNode);
    edgePaths[e.id] = path;
    if (!path || path.length < 2) return;

    // Segment usage
    for (var i = 0; i < path.length - 1; i++) {
      var s = sk(path[i].gc, path[i].gr, path[i+1].gc, path[i+1].gr);
      segCount[s] = (segCount[s] || 0) + 1;
    }

    // Cell directions for crossing detection
    for (var i = 0; i < path.length; i++) {
      if (i < path.length - 1) {
        var dir = (path[i].gr === path[i+1].gr) ? "H" : "V";
        addCellDir(path[i].gc, path[i].gr, e.id, dir);
        addCellDir(path[i+1].gc, path[i+1].gr, e.id, dir);
      }
    }
  });

  // Assign lanes per segment
  var edgeLanes = {};
  edges.forEach(function(e) {
    var path = edgePaths[e.id];
    if (!path || path.length < 2) return;
    edgeLanes[e.id] = [];
    for (var i = 0; i < path.length - 1; i++) {
      var s = sk(path[i].gc, path[i].gr, path[i+1].gc, path[i+1].gr);
      if (!segIdx[s]) segIdx[s] = 0;
      var li = segIdx[s]++;
      var total = segCount[s] || 1;
      edgeLanes[e.id].push({
        offset: (li - (total - 1) / 2) * LANE_W
      });
    }
  });

  // Detect crossings: cells with both H and V edges from different sources
  var crossings = {};
  Object.keys(cellEdges).forEach(function(k) {
    var arr = cellEdges[k];
    var hSet = {}, vSet = {};
    arr.forEach(function(d) {
      if (d.dir === "H") hSet[d.edgeId] = true;
      else vSet[d.edgeId] = true;
    });
    var hIds = Object.keys(hSet), vIds = Object.keys(vSet);
    if (hIds.length === 0 || vIds.length === 0) return;
    var hasCross = false;
    for (var i = 0; i < hIds.length && !hasCross; i++)
      if (!vSet[hIds[i]]) hasCross = true;
    for (var i = 0; i < vIds.length && !hasCross; i++)
      if (!hSet[vIds[i]]) hasCross = true;
    if (hasCross) crossings[k] = true;
  });

  return { edgePaths: edgePaths, edgeLanes: edgeLanes, crossings: crossings };
}

// =================================================================
// Build SVG path for Manhattan-routed edge
// =================================================================
function buildManhattanSVG(srcP, dstP, path, lanes, crossings) {
  if (!srcP || !dstP) return "";
  if (!path || path.length === 0) {
    if (Math.abs(srcP.y - dstP.y) < 0.5 || Math.abs(srcP.x - dstP.x) < 0.5)
      return "M" + ff(srcP.x) + "," + ff(srcP.y) + "L" + ff(dstP.x) + "," + ff(dstP.y);
    return "M" + ff(srcP.x) + "," + ff(srcP.y) +
           "L" + ff(dstP.x) + "," + ff(srcP.y) +
           "L" + ff(dstP.x) + "," + ff(dstP.y);
  }

  // Build pixel waypoints with per-segment lane offsets
  var pts = [srcP];
  for (var i = 0; i < path.length; i++) {
    var bx = (path[i].gc + 0.5) * CELL;
    var by = (path[i].gr + 0.5) * CELL;
    var dx = 0, dy = 0;
    var si = (i < path.length - 1) ? i : (lanes ? lanes.length - 1 : -1);
    if (lanes && si >= 0 && si < lanes.length && si < path.length - 1) {
      if (path[si].gr === path[si + 1].gr) dy = lanes[si].offset;
      else dx = lanes[si].offset;
    } else if (lanes && i === path.length - 1 && lanes.length > 0) {
      var li = lanes.length - 1;
      if (li < path.length - 1 && path[li].gr === path[li + 1].gr) dy = lanes[li].offset;
      else if (li < path.length - 1) dx = lanes[li].offset;
    }
    pts.push({ x: bx + dx, y: by + dy });
  }
  pts.push(dstP);

  // Insert knee points for non-axis-aligned first/last segments
  var kneesBefore = 0;
  if (pts.length >= 2) {
    if (pts[0].x !== pts[1].x && pts[0].y !== pts[1].y) {
      var knee;
      if (srcP.side === "left" || srcP.side === "right")
        knee = { x: pts[1].x, y: pts[0].y };
      else
        knee = { x: pts[0].x, y: pts[1].y };
      pts.splice(1, 0, knee);
      kneesBefore++;
    }
    var last = pts.length - 1;
    if (pts[last].x !== pts[last - 1].x && pts[last].y !== pts[last - 1].y) {
      var knee2;
      if (dstP.side === "left" || dstP.side === "right")
        knee2 = { x: pts[last - 1].x, y: pts[last].y };
      else
        knee2 = { x: pts[last].x, y: pts[last - 1].y };
      pts.splice(last, 0, knee2);
    }
  }

  // Build SVG path with hop-over arcs at crossings
  var d = "M" + ff(pts[0].x) + "," + ff(pts[0].y);
  for (var i = 1; i < pts.length; i++) {
    var px = pts[i].x, py = pts[i].y;
    var hopHere = false;

    // Check hop: only for routing waypoints going straight-through horizontally
    if (crossings) {
      var pi = i - 1 - kneesBefore;
      if (pi >= 0 && pi < path.length) {
        var cellK = path[pi].gc + "," + path[pi].gr;
        if (crossings[cellK]) {
          var prevH = (i >= 2) && Math.abs(pts[i - 1].y - py) < 1;
          var nextH = (i < pts.length - 1) && Math.abs(pts[i + 1].y - py) < 1;
          if (prevH && nextH) hopHere = true;
        }
      }
    }

    if (hopHere) {
      var goRight = px > pts[i - 1].x;
      if (goRight) {
        d += "L" + ff(px - HOP_R) + "," + ff(py);
        d += "A" + HOP_R + "," + HOP_R + " 0 0 1 " + ff(px + HOP_R) + "," + ff(py);
      } else {
        d += "L" + ff(px + HOP_R) + "," + ff(py);
        d += "A" + HOP_R + "," + HOP_R + " 0 0 0 " + ff(px - HOP_R) + "," + ff(py);
      }
    } else {
      d += "L" + ff(px) + "," + ff(py);
    }
  }
  return d;
}

// =================================================================
// Dynamic PE sizing
// =================================================================

/// Compute the CELL size needed to fit PEs with their FU contents.
/// Returns the calculated cell size in pixels.
function computeDynamicCellSize(nodes) {
  var maxFuCount = 0;
  var maxFuLabelLen = 0;
  nodes.forEach(function(n) {
    if (n.type !== "fabric.spatial_pe" || !n.fuDetails || n.fuDetails.length === 0)
      return;
    maxFuCount = Math.max(maxFuCount, n.fuDetails.length);
    n.fuDetails.forEach(function(fu) {
      var label = (fu.ops && fu.ops.length > 0) ? fu.ops[0] : fu.name;
      maxFuLabelLen = Math.max(maxFuLabelLen, label.length);
    });
  });

  if (maxFuCount === 0) return MIN_CELL;

  // PE size = sum of FU sizes + margins (square grid layout).
  var fuCols = Math.ceil(Math.sqrt(maxFuCount));
  var fuRows = Math.ceil(maxFuCount / fuCols);

  // Compute per-row max FU height (using the largest PE as reference)
  var rowMaxH = [];
  for (var ri = 0; ri < fuRows; ri++) rowMaxH.push(FU_BOX_H);
  nodes.forEach(function(n) {
    if (n.type !== "fabric.spatial_pe" || !n.fuDetails) return;
    n.fuDetails.forEach(function(fu, fi) {
      var row = Math.floor(fi / fuCols);
      if (row < fuRows) rowMaxH[row] = Math.max(rowMaxH[row], fuBoxHeight(fu));
    });
  });
  var totalFuH = 0;
  for (var ri = 0; ri < fuRows; ri++) totalFuH += rowMaxH[ri] + FU_GAP;
  var neededH = PE_TITLE_H + totalFuH + PE_PAD * 2;

  var fuBoxW = MIN_FU_BOX_W;
  var neededW = fuBoxW * fuCols + (fuCols - 1) * FU_GAP + PE_PAD * 2;

  var peSize = Math.max(neededW, neededH);

  // Also account for SW node sizes (proportional to port count)
  var maxSwSize = 0;
  nodes.forEach(function(n) {
    if (n.type !== "fabric.spatial_sw") return;
    var swIn = (n.ports && n.ports.in) ? n.ports.in : 4;
    var swOut = (n.ports && n.ports.out) ? n.ports.out : 4;
    var swSide = Math.max(60, Math.max(swIn, swOut) * 7 + 20);
    maxSwSize = Math.max(maxSwSize, swSide);
  });

  var maxNodeSize = Math.max(peSize, maxSwSize);

  // CELL controls grid spacing for routing. Nodes render at their full size
  // (which may exceed CELL), but routing BFS uses CELL-sized grid cells.
  // Cap CELL so the grid is dense enough to fit on screen. Nodes overlap
  // adjacent empty cells visually but routing still works in the free cells.
  // Use half the max node size + margin so adjacent nodes don't fully overlap.
  var cellForRouting = Math.max(MIN_CELL, Math.round(maxNodeSize * 0.55));
  return cellForRouting;
}

// =================================================================
// ADG Rendering
// =================================================================
function renderADG() {
  if (typeof adgGraph === "undefined" || !adgGraph || !adgGraph.nodes) return;
  var svg = d3.select("#svg-adg");
  d3.select("#status-bar").text("Computing layout...");

  // Add arrow marker for FU internal DAG
  var defs = svg.append("defs");
  defs.append("marker")
    .attr("id", "fu-arrow").attr("viewBox", "0 0 6 6")
    .attr("refX", 5).attr("refY", 3)
    .attr("markerWidth", 5).attr("markerHeight", 5)
    .attr("orient", "auto")
    .append("path").attr("d", "M0,0 L6,3 L0,6 Z")
    .attr("fill", "rgba(255,255,255,0.3)");

  // Dynamically size cells based on PE content
  CELL = computeDynamicCellSize(adgGraph.nodes);

  // 1. Grid layout
  var layout = gridLayoutEngine(adgGraph.nodes, adgGraph.edges);
  var gpos = layout.gpos;

  // 2. Port positions
  var portData = computeGridPorts(adgGraph.nodes, adgGraph.edges, gpos);
  var portPos = portData.pos;
  var portInfo = portData.info;
  var nodeInPorts = portData.nodeIn;
  var nodeOutPorts = portData.nodeOut;

  // 3. Route edges
  d3.select("#status-bar").text("Routing edges...");
  var routing = manhattanRouter(adgGraph.edges, layout);

  // 4. SVG setup
  var g = svg.append("g");
  adgSvgG = g;
  adgZoom = d3.zoom()
    .scaleExtent([0.05, 5])
    .on("zoom", function(event) { g.attr("transform", event.transform); });
  svg.call(adgZoom);

  // 5. Compute layout extent
  var ext = { mnX: Infinity, mxX: -Infinity, mnY: Infinity, mxY: -Infinity };
  Object.keys(gpos).forEach(function(nid) {
    var p = gpos[nid];
    ext.mnX = Math.min(ext.mnX, p.gc * CELL);
    ext.mxX = Math.max(ext.mxX, (p.gc + 1) * CELL);
    ext.mnY = Math.min(ext.mnY, p.gr * CELL);
    ext.mxY = Math.max(ext.mxY, (p.gr + 1) * CELL);
  });

  // 6. Background grid
  var gridG = g.append("g").attr("class", "grid");
  var gx0 = Math.floor(ext.mnX / CELL) * CELL - CELL;
  var gx1 = Math.ceil(ext.mxX / CELL) * CELL + CELL;
  var gy0 = Math.floor(ext.mnY / CELL) * CELL - CELL;
  var gy1 = Math.ceil(ext.mxY / CELL) * CELL + CELL;
  for (var gy = gy0; gy <= gy1; gy += CELL) {
    gridG.append("line").attr("class", "grid-line")
      .attr("x1", gx0).attr("y1", gy).attr("x2", gx1).attr("y2", gy);
  }
  for (var gx = gx0; gx <= gx1; gx += CELL) {
    gridG.append("line").attr("class", "grid-line")
      .attr("x1", gx).attr("y1", gy0).attr("x2", gx).attr("y2", gy1);
  }

  // 7. Draw edges (Manhattan routing)
  var edgeG = g.append("g").attr("class", "edges");
  adgGraph.edges.forEach(function(e) {
    var srcP = portPos[e.srcPort];
    var dstP = portPos[e.dstPort];
    if (!srcP && gpos[e.srcNode]) {
      var sp = gpos[e.srcNode];
      srcP = { x: (sp.gc + 0.5) * CELL, y: (sp.gr + 0.5) * CELL };
    }
    if (!dstP && gpos[e.dstNode]) {
      var dp = gpos[e.dstNode];
      dstP = { x: (dp.gc + 0.5) * CELL, y: (dp.gr + 0.5) * CELL };
    }
    if (!srcP || !dstP) return;

    var path = routing.edgePaths[e.id];
    var lanes = routing.edgeLanes ? routing.edgeLanes[e.id] : null;
    var svgPath = buildManhattanSVG(srcP, dstP, path, lanes, routing.crossings);

    var color = "#555";
    var dasharray = null;
    var sw = 1.5;
    if (e.edgeType === "control") { color = "#999"; dasharray = "4,2"; sw = 1.0; }
    else if (e.edgeType === "memref") { color = "#2255cc"; dasharray = "2,3"; sw = 2.0; }

    var ep = edgeG.append("path")
      .attr("class", "adg-edge").attr("id", e.id)
      .attr("d", svgPath)
      .attr("stroke", color)
      .attr("stroke-width", sw)
      .attr("opacity", 0.5)
      .attr("fill", "none");
    if (dasharray) ep.attr("stroke-dasharray", dasharray);
  });

  // 8. Draw nodes
  var nodeG = g.append("g").attr("class", "nodes");
  adgGraph.nodes.forEach(function(n) {
    var p = gpos[n.id];
    if (!p) return;
    var colors = nodeColorScheme(n);
    var cx = (p.gc + 0.5) * CELL;
    var cy = (p.gr + 0.5) * CELL;

    // PE nodes are drawn larger to show nested FU details
    var isPE = (n.type === "fabric.spatial_pe");
    var hasFUs = isPE && n.fuDetails && n.fuDetails.length > 0;
    var ns = nodeSize[n.id] || { w: CELL - 4, h: CELL - 4 };
    var w = ns.w, h = ns.h;

    var ng = nodeG.append("g")
      .attr("class", "adg-node").attr("id", n.id)
      .attr("transform", "translate(" + cx + "," + cy + ")");

    // Node shape
    ng.append("rect")
      .attr("x", -w / 2).attr("y", -h / 2)
      .attr("width", w).attr("height", h)
      .attr("rx", 3).attr("ry", 3)
      .attr("fill", colors.fill)
      .attr("stroke", colors.border)
      .attr("stroke-width", 1.5);

    // Port indicators with labels
    // Build port label map from portDetails if available, else generate from counts
    var portLabelMap = {};
    if (n.portDetails && n.portDetails.length > 0) {
      n.portDetails.forEach(function(pd) {
        // portDetails entries have {id, dir, idx}
        var prefix = pd.dir === "out" ? "O" : "I";
        portLabelMap[pd.id] = prefix + pd.idx;
      });
    } else if (n.ports) {
      // Generate labels from port counts
      var inIdx = 0, outIdx = 0;
      var inPortIds = nodeInPorts[n.id] || [];
      var outPortIds = nodeOutPorts[n.id] || [];
      inPortIds.forEach(function(pid) { portLabelMap[pid] = "I" + inIdx++; });
      outPortIds.forEach(function(pid) { portLabelMap[pid] = "O" + outIdx++; });
    }

    var allPorts = (nodeInPorts[n.id] || []).concat(nodeOutPorts[n.id] || []);
    allPorts.forEach(function(pid) {
      var pp = portPos[pid];
      if (!pp) return;
      var pi = portInfo[pid];
      var isOut = pi && pi.isOutput;
      var portColor = isOut ? "#ff6b35" : "#4ecdc4";
      var px = pp.x - cx, py = pp.y - cy;
      ng.append("circle")
        .attr("cx", px).attr("cy", py)
        .attr("r", 3)
        .attr("fill", portColor)
        .attr("stroke", "#0c1220").attr("stroke-width", 0.5);

      // Port label text
      var pLabel = portLabelMap[pid];
      if (pLabel) {
        var labelDx = 0, labelDy = 0;
        var anchor = "middle";
        if (pp.side === "top") { labelDy = -5; }
        else if (pp.side === "bottom") { labelDy = 8; }
        else if (pp.side === "left") { labelDx = -6; anchor = "end"; }
        else if (pp.side === "right") { labelDx = 6; anchor = "start"; }
        ng.append("text")
          .attr("x", px + labelDx).attr("y", py + labelDy)
          .attr("text-anchor", anchor)
          .attr("fill", portColor)
          .attr("font-size", "6px")
          .attr("font-weight", "500")
          .text(pLabel);
      }
    });

    // Label
    var label = n.name || n.id;
    var shortType = TYPE_SHORT[n.type] || n.type || "";
    if (label.length > 12) label = label.substring(0, 10) + "..";

    if (hasFUs) {
      // === THREE-LEVEL PE VISUALIZATION with mini-DFG ===
      var fuCount = n.fuDetails.length;
      var fuCols = Math.ceil(Math.sqrt(fuCount));
      var mappedFUName = getMappedFUForPE(n.id);

      // Compute per-row max FU height
      var fuRows = Math.ceil(fuCount / fuCols);
      var rowMaxH = [];
      for (var ri = 0; ri < fuRows; ri++) rowMaxH.push(FU_BOX_H);
      n.fuDetails.forEach(function(fu, fi) {
        var row = Math.floor(fi / fuCols);
        rowMaxH[row] = Math.max(rowMaxH[row], fuBoxHeight(fu));
      });

      var fuBoxW = fuCols > 1
        ? (w - PE_PAD * 2 - FU_GAP) / fuCols
        : w - PE_PAD * 2;

      // PE title bar
      ng.append("rect")
        .attr("x", -w/2).attr("y", -h/2)
        .attr("width", w).attr("height", PE_TITLE_H)
        .attr("rx", 6).attr("ry", 6)
        .attr("fill", "#1e3a5f");
      ng.append("text")
        .attr("x", 0).attr("y", -h/2 + PE_TITLE_H/2 + 3)
        .attr("text-anchor", "middle")
        .attr("fill", "#4ecdc4").attr("font-size", "9px").attr("font-weight", "600")
        .text(shortType + " " + truncLabel(label, 14));

      // Track the mapped FU box center for internal routing lines
      var mappedFUCenter = null;

      // Precompute per-row Y offsets
      var rowY = [];
      var cumY = -h/2 + PE_TITLE_H + 4;
      for (var ri = 0; ri < fuRows; ri++) {
        rowY.push(cumY);
        cumY += rowMaxH[ri] + FU_GAP;
      }

      // Draw each FU box with mini-DFG inside
      for (var fi = 0; fi < fuCount; fi++) {
        var col = fi % fuCols;
        var row = Math.floor(fi / fuCols);
        var fx = -w/2 + PE_PAD + col * (fuBoxW + FU_GAP + 2);
        var fy = rowY[row];
        var fuDet = n.fuDetails[fi];
        var thisIsMultiOp = fuDet.ops && fuDet.ops.length > 1;
        var thisFuH = fuBoxHeight(fuDet);
        var isMapped = (mappedFUName && fuDet.name === mappedFUName);
        var fuOpacity = isMapped ? 1.0 : (mappedFUName ? 0.35 : 0.7);

        // FU container box
        var fuG = ng.append("g").attr("opacity", fuOpacity);

        fuG.append("rect")
          .attr("class", "fu-box" + (isMapped ? " fu-mapped" : ""))
          .attr("x", fx).attr("y", fy)
          .attr("width", fuBoxW).attr("height", thisFuH)
          .attr("rx", 3).attr("ry", 3)
          .attr("fill", isMapped ? "rgba(78,205,196,0.15)" : "rgba(255,255,255,0.04)")
          .attr("stroke", isMapped ? "#4ecdc4" : "rgba(255,255,255,0.12)")
          .attr("stroke-width", isMapped ? 1.5 : 0.6);

        // Glow effect for mapped FU
        if (isMapped) {
          fuG.append("rect")
            .attr("class", "fu-glow")
            .attr("x", fx - 1).attr("y", fy - 1)
            .attr("width", fuBoxW + 2).attr("height", thisFuH + 2)
            .attr("rx", 4).attr("ry", 4)
            .attr("fill", "none")
            .attr("stroke", "#4ecdc4")
            .attr("stroke-width", 0.5)
            .attr("opacity", 0.4);
        }

        // FU name label at top of box
        fuG.append("text")
          .attr("x", fx + fuBoxW/2).attr("y", fy + 9)
          .attr("text-anchor", "middle")
          .attr("fill", isMapped ? "#4ecdc4" : "rgba(255,255,255,0.5)")
          .attr("font-size", "6px").attr("font-weight", "500")
          .text(fuDet.name);

        // Draw mini-DFG inside FU: DAG layout for multi-op, single ellipse for single-op
        var ops = (fuDet.ops && fuDet.ops.length > 0) ? fuDet.ops : [fuDet.name];
        var numOps = ops.length;
        var opStartY = fy + 13;

        if (numOps <= 1) {
          // Single-op FU: one centered ellipse
          var opCenterX = fx + fuBoxW / 2;
          var opNodeW = Math.min(FU_OP_NODE_W, fuBoxW - 10);
          var sName = shortOpName(ops[0]);
          fuG.append("ellipse")
            .attr("cx", opCenterX).attr("cy", opStartY + FU_OP_NODE_H / 2)
            .attr("rx", opNodeW / 2).attr("ry", FU_OP_NODE_H / 2)
            .attr("fill", isMapped ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.06)")
            .attr("stroke", isMapped ? "rgba(255,255,255,0.4)" : "rgba(255,255,255,0.15)")
            .attr("stroke-width", 0.5);
          fuG.append("text")
            .attr("x", opCenterX).attr("y", opStartY + FU_OP_NODE_H / 2 + 3)
            .attr("text-anchor", "middle")
            .attr("fill", isMapped ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.6)")
            .attr("font-size", "7px")
            .text(sName);
        } else {
          // Multi-op FU: DAG layout with fan-in/fan-out
          var dagLayout = computeFUDagLayout(ops);
          var opNodeW = Math.min(FU_OP_NODE_W, (fuBoxW - 8) / Math.max(dagLayout.cols, 1) - 4);
          // Compute positions for each op
          var opPositions = {};
          dagLayout.nodes.forEach(function(dn) {
            var rowY = opStartY + dn.row * (FU_OP_NODE_H + FU_OP_ARROW_GAP);
            var totalW = dn.totalInRow * (opNodeW + 4) - 4;
            var startX = fx + fuBoxW / 2 - totalW / 2;
            var opX = startX + dn.col * (opNodeW + 4) + opNodeW / 2;
            opPositions[dn.opIdx] = { cx: opX, cy: rowY + FU_OP_NODE_H / 2 };

            var opName = ops[dn.opIdx];
            var sName = shortOpName(opName);
            var isMux = opName.indexOf("static_mux") >= 0;

            if (isMux) {
              // Trapezoid for mux
              var tx1 = opX - opNodeW / 2 + 3;
              var tx2 = opX + opNodeW / 2 - 3;
              var bx1 = opX - opNodeW / 2;
              var bx2 = opX + opNodeW / 2;
              var trapPath = "M" + ff(tx1) + "," + ff(rowY) +
                "L" + ff(tx2) + "," + ff(rowY) +
                "L" + ff(bx2) + "," + ff(rowY + FU_OP_NODE_H) +
                "L" + ff(bx1) + "," + ff(rowY + FU_OP_NODE_H) + "Z";
              fuG.append("path")
                .attr("d", trapPath)
                .attr("fill", isMapped ? "rgba(255,215,0,0.18)" : "rgba(255,215,0,0.08)")
                .attr("stroke", isMapped ? "rgba(255,215,0,0.6)" : "rgba(255,215,0,0.3)")
                .attr("stroke-width", 0.6);
            } else {
              fuG.append("ellipse")
                .attr("cx", opX).attr("cy", rowY + FU_OP_NODE_H / 2)
                .attr("rx", opNodeW / 2).attr("ry", FU_OP_NODE_H / 2)
                .attr("fill", isMapped ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.06)")
                .attr("stroke", isMapped ? "rgba(255,255,255,0.4)" : "rgba(255,255,255,0.15)")
                .attr("stroke-width", 0.5);
            }

            fuG.append("text")
              .attr("x", opX).attr("y", rowY + FU_OP_NODE_H / 2 + 3)
              .attr("text-anchor", "middle")
              .attr("fill", isMux ? "#ffd166" : (isMapped ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.6)"))
              .attr("font-size", "7px")
              .text(sName);
          });

          // Draw DAG edges between ops
          if (dagLayout.edges) {
            dagLayout.edges.forEach(function(de) {
              var fromP = opPositions[de.from];
              var toP = opPositions[de.to];
              if (!fromP || !toP) return;
              fuG.append("line")
                .attr("class", "fu-dag-arrow")
                .attr("x1", fromP.cx).attr("y1", fromP.cy + FU_OP_NODE_H / 2 + 1)
                .attr("x2", toP.cx).attr("y2", toP.cy - FU_OP_NODE_H / 2 - 1)
                .attr("stroke", isMapped ? "rgba(78,205,196,0.5)" : "rgba(255,255,255,0.2)")
                .attr("stroke-width", 1)
                .attr("marker-end", "url(#fu-arrow)");
            });
          }

          // FU boundary port squares (small rectangles on left and right edges)
          var fuPortSize = 3;
          var numFuIn = 2, numFuOut = 1;
          for (var pi = 0; pi < numFuIn; pi++) {
            var py = fy + thisFuH * (pi + 1) / (numFuIn + 1);
            fuG.append("rect")
              .attr("x", fx - fuPortSize/2).attr("y", py - fuPortSize/2)
              .attr("width", fuPortSize).attr("height", fuPortSize)
              .attr("fill", "#4ecdc4").attr("stroke", "#0c1220").attr("stroke-width", 0.3);
          }
          for (var pi = 0; pi < numFuOut; pi++) {
            var py = fy + thisFuH * (pi + 1) / (numFuOut + 1);
            fuG.append("rect")
              .attr("x", fx + fuBoxW - fuPortSize/2).attr("y", py - fuPortSize/2)
              .attr("width", fuPortSize).attr("height", fuPortSize)
              .attr("fill", "#ff6b35").attr("stroke", "#0c1220").attr("stroke-width", 0.3);
          }
        }

        // Track center of mapped FU for internal routing
        if (isMapped) {
          mappedFUCenter = { x: opCenterX, y: fy + thisFuH / 2 };
        }
      }

      // PE internal routing: lines from input ports to mapped FU top, from FU bottom to output ports
      if (mappedFUCenter) {
        // Compute FU top/bottom Y from the mapped FU box
        var mappedFUTop = mappedFUCenter.y - 10;  // approximate top of FU
        var mappedFUBot = mappedFUCenter.y + 10;  // approximate bottom of FU
        var inPorts = nodeInPorts[n.id] || [];
        var outPorts = nodeOutPorts[n.id] || [];
        inPorts.forEach(function(pid) {
          var pp = portPos[pid];
          if (!pp) return;
          ng.append("line")
            .attr("class", "pe-internal-route")
            .attr("x1", pp.x - cx).attr("y1", pp.y - cy)
            .attr("x2", mappedFUCenter.x).attr("y2", mappedFUTop)
            .attr("stroke", "rgba(78,205,196,0.35)")
            .attr("stroke-width", 0.7);
        });
        outPorts.forEach(function(pid) {
          var pp = portPos[pid];
          if (!pp) return;
          ng.append("line")
            .attr("class", "pe-internal-route")
            .attr("x1", mappedFUCenter.x).attr("y1", mappedFUBot)
            .attr("x2", pp.x - cx).attr("y2", pp.y - cy)
            .attr("stroke", "rgba(255,107,53,0.35)")
            .attr("stroke-width", 0.7);
        });
      }
    } else if (n.type === "fabric.spatial_sw") {
      // === SW VISUALIZATION with internal route connections ===
      var swW = ns.w, swH = ns.h;
      var swInCount = (n.ports && n.ports.in) ? n.ports.in : 0;
      var swOutCount = (n.ports && n.ports.out) ? n.ports.out : 0;

      // Label at top
      ng.append("text")
        .attr("x", 0).attr("y", -swH/2 + 10)
        .attr("text-anchor", "middle")
        .attr("fill", colors.text).attr("font-size", "7px").attr("font-weight", "600")
        .text(shortType + " " + truncLabel(label, 10));

      // Compute internal port positions around the SW boundary
      // Input ports on left side, output ports on right side
      var swInternalPorts = { in: [], out: [] };
      var inStep = swH / (swInCount + 1);
      for (var ii = 0; ii < swInCount; ii++) {
        var iy = -swH/2 + inStep * (ii + 1);
        swInternalPorts.in.push({ x: -swW/2 + 4, y: iy, idx: ii });
        // Draw port dot
        ng.append("circle")
          .attr("cx", -swW/2 + 4).attr("cy", iy)
          .attr("r", 2).attr("fill", "#4ecdc4").attr("stroke", "#0c1220").attr("stroke-width", 0.3);
        ng.append("text")
          .attr("x", -swW/2 + 10).attr("y", iy + 2)
          .attr("text-anchor", "start")
          .attr("fill", "#4ecdc4").attr("font-size", "5px")
          .text("I" + ii);
      }
      var outStep = swH / (swOutCount + 1);
      for (var oi = 0; oi < swOutCount; oi++) {
        var oy = -swH/2 + outStep * (oi + 1);
        swInternalPorts.out.push({ x: swW/2 - 4, y: oy, idx: oi });
        ng.append("circle")
          .attr("cx", swW/2 - 4).attr("cy", oy)
          .attr("r", 2).attr("fill", "#ff6b35").attr("stroke", "#0c1220").attr("stroke-width", 0.3);
        ng.append("text")
          .attr("x", swW/2 - 10).attr("y", oy + 2)
          .attr("text-anchor", "end")
          .attr("fill", "#ff6b35").attr("font-size", "5px")
          .text("O" + oi);
      }

      // Draw internal route connections if mappingData.swRoutes has data for this node
      if (mappingData && mappingData.swRoutes && mappingData.swRoutes[n.id]) {
        var routes = mappingData.swRoutes[n.id];
        routes.forEach(function(route, ri) {
          var inIdx = route[0], outIdx = route[1];
          if (inIdx < swInternalPorts.in.length && outIdx < swInternalPorts.out.length) {
            var sp = swInternalPorts.in[inIdx];
            var dp = swInternalPorts.out[outIdx];
            var routeColor = ROUTE_PALETTE[ri % ROUTE_PALETTE.length];
            ng.append("line")
              .attr("x1", sp.x + 3).attr("y1", sp.y)
              .attr("x2", dp.x - 3).attr("y2", dp.y)
              .attr("stroke", routeColor)
              .attr("stroke-width", 1.2)
              .attr("opacity", 0.7);
          }
        });
      }
    } else {
      // Standard label rendering for other node types
      ng.append("text")
        .attr("dy", "-0.1em").attr("text-anchor", "middle")
        .attr("fill", colors.text).attr("font-size", "7px")
        .text(shortType);

      ng.append("text")
        .attr("dy", "0.9em").attr("text-anchor", "middle")
        .attr("fill", colors.text).attr("font-size", "7px")
        .text(truncLabel(label, 10));
    }

    // Mapping overlay label
    if (mappingData && mappingData.hwToSw && mappingData.hwToSw[n.id]) {
      var swIds = mappingData.hwToSw[n.id];
      if (swIds.length > 0) {
        var swNode = null;
        if (typeof DFG_DATA !== "undefined" && DFG_DATA && DFG_DATA.nodes) {
          var swIdNum = parseInt(swIds[0].replace("sw_", ""));
          DFG_DATA.nodes.forEach(function(dn) {
            if (dn.id === swIdNum) swNode = dn;
          });
        }
        var mappedLabel = swNode ? swNode.opName : swIds[0];
        ng.append("text")
          .attr("dy", "2.0em").attr("text-anchor", "middle")
          .attr("fill", "#ffcc00").attr("font-size", "6px").attr("font-weight", "bold")
          .text(truncLabel(mappedLabel, 12));

        // Thicker border for mapped nodes
        ng.select("rect")
          .attr("stroke-width", 2.5)
          .attr("stroke", "#222");
      }
    }

    // Interaction
    ng.on("mouseenter", function() { highlightNode(n.id, "hw"); })
      .on("mouseleave", function() { clearHighlights(); })
      .on("click", function(event) {
        event.stopPropagation();
        highlightNode(n.id, "hw");
        showDetail(n.id, "hw");
      });
  });

  // Fit to view
  fitPanel("adg");
  d3.select("#status-bar").text("Ready (" + adgGraph.nodes.length + " nodes, " +
                                 adgGraph.edges.length + " edges)");

  svg.on("click", function() {
    closeDetail();
    if (currentMode === "overlay") clearRouteHighlight();
  });
  svg.on("dblclick.zoom", null);
  svg.on("dblclick", function() { fitPanel("adg"); });
}

// =================================================================
// DFG Rendering
// =================================================================
function renderDFG() {
  if (typeof DFG_DATA === "undefined" || !DFG_DATA || !DFG_DATA.dot) {
    d3.select("#svg-dfg").append("text")
      .attr("x", 20).attr("y", 40)
      .attr("fill", "#888").text("No DFG data available.");
    return;
  }

  var svg = d3.select("#svg-dfg");
  d3.select("#status-bar").text("Initializing Graphviz WASM...");

  // Try Viz.js / @viz-js/viz
  if (typeof Viz !== "undefined") {
    var vizPromise;
    try {
      // Viz.js v3+ API
      if (typeof Viz.instance === "function") {
        vizPromise = Viz.instance();
      } else {
        var vizInst = new Viz();
        vizPromise = vizInst.renderSVGElement(DFG_DATA.dot).then(function(el) {
          return { renderString: function() { return el.outerHTML; } };
        });
      }
    } catch(e) {
      vizPromise = null;
    }

    if (vizPromise) {
      vizPromise.then(function(viz) {
        d3.select("#status-bar").text("Rendering DFG...");
        var svgStr;
        if (typeof viz.renderString === "function") {
          svgStr = viz.renderString(DFG_DATA.dot, { engine: "dot", format: "svg" });
        } else {
          svgStr = viz;
        }
        insertDFGRendered(svgStr, svg);
      }).catch(function(err) {
        console.error("Viz.js error:", err);
        renderDFGFallback(svg);
      });
      return;
    }
  }

  // VizStandalone (loom's approach)
  if (typeof VizStandalone !== "undefined") {
    VizStandalone.instance().then(function(viz) {
      d3.select("#status-bar").text("Rendering DFG...");
      var svgStr = viz.renderString(DFG_DATA.dot, { engine: "dot", format: "svg" });
      insertDFGRendered(svgStr, svg);
    }).catch(function(err) {
      console.error("VizStandalone error:", err);
      renderDFGFallback(svg);
    });
    return;
  }

  renderDFGFallback(svg);
}

function insertDFGRendered(svgStr, svg) {
  var container = document.getElementById("panel-dfg");
  var wrapper = document.createElement("div");
  wrapper.innerHTML = svgStr;
  var rendered = wrapper.querySelector("svg");
  if (!rendered) {
    d3.select("#status-bar").text("DFG render failed");
    return;
  }
  rendered.setAttribute("id", "svg-dfg");
  rendered.removeAttribute("viewBox");
  rendered.removeAttribute("width");
  rendered.removeAttribute("height");
  rendered.style.width = "100%";
  rendered.style.height = "100%";
  var old = document.getElementById("svg-dfg");
  if (old) old.parentNode.replaceChild(rendered, old);

  var dfgSvgSel = d3.select("#svg-dfg");
  var innerG = dfgSvgSel.select("g");
  dfgSvgG = innerG;

  dfgZoom = d3.zoom()
    .scaleExtent([0.1, 5])
    .on("zoom", function(event) { innerG.attr("transform", event.transform); });
  dfgSvgSel.call(dfgZoom);

  // Make DFG nodes clickable with cross-highlight
  dfgSvgSel.selectAll(".node").each(function() {
    var el = d3.select(this);
    var titleEl = el.select("title");
    var nodeTitle = titleEl.empty() ? "" : titleEl.text();
    var nodeId = el.attr("id") || nodeTitle;
    if (nodeId) {
      el.on("mouseenter", function() { highlightNode(nodeId, "sw"); })
        .on("mouseleave", function() { clearHighlights(); })
        .on("click", function(event) {
          event.stopPropagation();
          highlightNode(nodeId, "sw");
          showDetail(nodeId, "sw");
        });
    }
  });

  dfgSvgSel.on("click", function() { closeDetail(); });
  dfgSvgSel.on("dblclick.zoom", null);
  dfgSvgSel.on("dblclick", function() { fitPanel("dfg"); });
  d3.select("#status-bar").text("Ready");
  fitPanel("dfg");
}

function renderDFGFallback(svg) {
  if (!DFG_DATA || !DFG_DATA.nodes) return;

  dfgSvgG = svg.append("g").attr("class", "dfg-root");
  dfgZoom = d3.zoom()
    .scaleExtent([0.1, 10])
    .on("zoom", function(event) { dfgSvgG.attr("transform", event.transform); });
  svg.call(dfgZoom);

  var y = 30;
  DFG_DATA.nodes.forEach(function(n) {
    var g = dfgSvgG.append("g")
      .attr("class", "node")
      .attr("id", "sw_" + n.id)
      .attr("transform", "translate(100," + y + ")")
      .style("cursor", "pointer");

    var label = n.opName || ("node_" + n.id);
    g.append("rect")
      .attr("x", -80).attr("y", -12)
      .attr("width", 160).attr("height", 24)
      .attr("rx", 4).attr("ry", 4)
      .attr("fill", "#ddd")
      .attr("stroke", "#999")
      .attr("stroke-width", 1);

    g.append("text")
      .attr("text-anchor", "middle").attr("dy", "0.35em")
      .attr("font-size", "10px").attr("fill", "#333")
      .text(truncLabel(label, 20));

    g.on("click", function(event) {
      event.stopPropagation();
      highlightNode("sw_" + n.id, "sw");
      showDetail("sw_" + n.id, "sw");
    });

    y += 32;
  });

  d3.select("#status-bar").text("DFG: fallback layout. Load Graphviz WASM for better layout.");
  fitPanel("dfg");
}

// =================================================================
// Fit Panel
// =================================================================
function fitPanel(panel) {
  var svg, g, zoom;
  if (panel === "adg") {
    svg = d3.select("#svg-adg"); g = adgSvgG; zoom = adgZoom;
  } else {
    svg = d3.select("#svg-dfg"); g = dfgSvgG; zoom = dfgZoom;
  }
  if (!g || !zoom) return;
  var gNode = g.node();
  if (!gNode) return;
  var bbox = gNode.getBBox();
  if (bbox.width === 0 || bbox.height === 0) return;
  var svgNode = svg.node();
  var svgW = svgNode.clientWidth || 800;
  var svgH = svgNode.clientHeight || 600;
  var padding = 40;
  var scale = Math.min(
    (svgW - padding * 2) / bbox.width,
    (svgH - padding * 2) / bbox.height, 2
  );
  var tx = svgW / 2 - (bbox.x + bbox.width / 2) * scale;
  var ty = svgH / 2 - (bbox.y + bbox.height / 2) * scale;
  svg.transition().duration(500)
    .call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}

// =================================================================
// Cross-Highlighting
// =================================================================
function highlightNode(nodeId, domain) {
  clearHighlights();
  var el = d3.select("#" + CSS.escape(nodeId));
  if (!el.empty()) {
    el.select("rect, polygon, ellipse").classed("node-highlight", true);
  }

  if (domain === "hw") {
    // Dim non-adjacent ADG nodes
    var adjacentNodes = {};
    var adjacentEdges = {};
    adjacentNodes[nodeId] = true;
    (adgAdj[nodeId] || []).forEach(function(a) {
      adjacentNodes[a.node] = true;
      adjacentEdges[a.edge] = true;
    });
    d3.selectAll(".adg-node").each(function() {
      var nEl = d3.select(this);
      if (!adjacentNodes[nEl.attr("id")]) nEl.classed("node-dimmed", true);
    });
    d3.selectAll(".adg-edge").each(function() {
      var eEl = d3.select(this);
      if (!adjacentEdges[eEl.attr("id")]) eEl.classed("edge-dimmed", true);
    });

    // Cross-highlight to DFG
    if (mappingData && mappingData.hwToSw) {
      var swIds = mappingData.hwToSw[nodeId];
      if (swIds) {
        swIds.forEach(function(swId) {
          var swEl = d3.select("#" + CSS.escape(swId));
          if (!swEl.empty()) {
            swEl.select("polygon, ellipse, rect, path").classed("cross-highlight", true);
          }
        });
      }
    }
  }

  if (domain === "sw") {
    // Cross-highlight to ADG
    if (mappingData && mappingData.swToHw) {
      var hwId = mappingData.swToHw[nodeId];
      if (hwId) {
        var hwEl = d3.select("#" + CSS.escape(hwId));
        if (!hwEl.empty()) {
          hwEl.select("rect").classed("cross-highlight", true);
        }
      }
    }
  }
}

function clearHighlights() {
  d3.selectAll(".node-highlight").classed("node-highlight", false);
  d3.selectAll(".cross-highlight").classed("cross-highlight", false);
  d3.selectAll(".node-dimmed").classed("node-dimmed", false);
  d3.selectAll(".edge-dimmed").classed("edge-dimmed", false);
}

function clearRouteHighlight() {
  d3.selectAll(".route-trace").classed("route-trace", false);
}

// =================================================================
// Detail Panel
// =================================================================
function showDetail(nodeId, domain) {
  var panel = document.getElementById("detail-panel");
  var content = document.getElementById("detail-content");
  var text = "";

  if (domain === "hw") {
    var n = hwNodeMap[nodeId];
    if (n) {
      text += "Name:      " + (n.name || nodeId) + "\n";
      text += "Type:      " + (n.type || "unknown") + "\n";
      if (n.gridCol != null) text += "Grid:      (" + n.gridRow + ", " + n.gridCol + ")\n";
      if (n.ports) text += "Ports:     " + n.ports.in + " in, " + n.ports.out + " out\n";
      if (n.fuDetails && n.fuDetails.length > 0) {
        text += "FUs (" + n.fuDetails.length + "):\n";
        n.fuDetails.forEach(function(fd) {
          var ops = fd.ops ? fd.ops.join(", ") : "";
          text += "  @" + fd.name + (ops ? " -> " + ops : "") + "\n";
        });
      } else if (n.fus && n.fus.length > 0) {
        text += "FU ops:    " + n.fus.join(", ") + "\n";
      }
      if (mappingData && mappingData.hwToSw && mappingData.hwToSw[nodeId]) {
        text += "Mapped SW: " + mappingData.hwToSw[nodeId].join(", ") + "\n";
      }
    } else {
      text = "HW Node " + nodeId + ": (unknown)";
    }
  } else if (domain === "sw") {
    var swIdNum = parseInt(nodeId.replace("sw_", ""));
    if (typeof DFG_DATA !== "undefined" && DFG_DATA && DFG_DATA.nodes) {
      var swNode = null;
      DFG_DATA.nodes.forEach(function(dn) {
        if (dn.id === swIdNum) swNode = dn;
      });
      if (swNode) {
        text += "Operation: " + (swNode.opName || "unknown") + "\n";
        text += "Kind:      " + (swNode.kind || "op") + "\n";
        if (mappingData && mappingData.swToHw && mappingData.swToHw[nodeId]) {
          text += "Mapped to: " + mappingData.swToHw[nodeId] + "\n";
        }
      } else {
        text = "SW Node " + nodeId;
      }
    } else {
      text = "SW Node " + nodeId;
    }
  }

  content.textContent = text;
  panel.classList.add("visible");
}

function closeDetail() {
  document.getElementById("detail-panel").classList.remove("visible");
}

// =================================================================
// Mode Toggle
// =================================================================
function setMode(mode) {
  var previousMode = currentMode;
  currentMode = mode;
  var adgPanel = document.getElementById("panel-adg");
  var dfgPanel = document.getElementById("panel-dfg");
  var divider = document.getElementById("panel-divider");
  var restoreBtn = document.getElementById("btn-restore");

  if (previousMode === "overlay" && mode !== "overlay") removeOverlay();

  adgPanel.classList.remove("panel-maximized", "panel-hidden");
  dfgPanel.classList.remove("panel-maximized", "panel-hidden");
  divider.classList.remove("divider-hidden");
  if (restoreBtn) restoreBtn.style.display = "none";

  document.querySelectorAll("#mode-buttons button").forEach(function(b) {
    b.classList.remove("active");
  });

  if (mode === "sidebyside") {
    var sbs = document.getElementById("btn-sidebyside");
    if (sbs) sbs.classList.add("active");
    adgPanel.style.flex = "0 0 55%";
    dfgPanel.style.flex = "0 0 45%";
  } else if (mode === "overlay") {
    var ovl = document.getElementById("btn-overlay");
    if (ovl) ovl.classList.add("active");
    adgPanel.classList.add("panel-maximized");
    dfgPanel.classList.add("panel-hidden");
    divider.classList.add("divider-hidden");
    renderOverlay();
  } else if (mode === "maximize-adg" || mode === "maximize-dfg") {
    var showPanel = mode === "maximize-adg" ? adgPanel : dfgPanel;
    var hidePanel = mode === "maximize-adg" ? dfgPanel : adgPanel;
    showPanel.classList.add("panel-maximized");
    hidePanel.classList.add("panel-hidden");
    divider.classList.add("divider-hidden");
    if (restoreBtn) restoreBtn.style.display = "inline-block";
  }
}

// =================================================================
// Overlay Mode
// =================================================================
function renderOverlay() {
  if (!mappingData || !adgSvgG) return;
  savedNodeFills = {};
  adgSvgG.selectAll(".overlay-group").remove();
  var overlayG = adgSvgG.append("g").attr("class", "overlay-group");

  adgGraph.nodes.forEach(function(n) {
    var nodeEl = d3.select("#" + CSS.escape(n.id));
    if (nodeEl.empty()) return;
    var shapeEl = nodeEl.select("rect");
    if (shapeEl.empty()) return;

    savedNodeFills[n.id] = {
      fill: shapeEl.attr("fill"),
      sw: shapeEl.attr("stroke-width")
    };

    var swIds = mappingData.hwToSw ? mappingData.hwToSw[n.id] : null;
    if (swIds && swIds.length > 0) {
      shapeEl.attr("fill", "#4CAF50");
      shapeEl.attr("stroke-width", "3");
      shapeEl.attr("stroke", "#222");
    } else if (n.class !== "boundary") {
      shapeEl.classed("unmapped-node", true);
    }
  });

  adgSvgG.selectAll(".adg-edge").classed("base-edge-overlay", true);

  // Draw route overlays
  if (mappingData.routes) {
    mappingData.routes.forEach(function(route, idx) {
      if (!route.path || route.path.length < 2) return;
      var color = ROUTE_PALETTE[idx % ROUTE_PALETTE.length];
      // Find the path through the ADG edges
      for (var i = 0; i < route.path.length - 1; i++) {
        var srcId = route.path[i];
        var dstId = route.path[i + 1];
        // Find edge between these nodes
        adgGraph.edges.forEach(function(e) {
          if (e.srcNode === srcId && e.dstNode === dstId) {
            var origEdge = d3.select("#" + CSS.escape(e.id));
            if (!origEdge.empty()) {
              var pathData = origEdge.attr("d");
              if (pathData) {
                overlayG.append("path")
                  .attr("class", "route-overlay")
                  .attr("d", pathData)
                  .attr("stroke", color)
                  .attr("stroke-width", 3);
              }
            }
          }
        });
      }
    });
  }
}

function removeOverlay() {
  if (!adgSvgG) return;
  adgSvgG.selectAll(".overlay-group").remove();

  adgGraph.nodes.forEach(function(n) {
    var nodeEl = d3.select("#" + CSS.escape(n.id));
    if (nodeEl.empty()) return;
    var shapeEl = nodeEl.select("rect");
    if (shapeEl.empty()) return;
    shapeEl.classed("unmapped-node", false);
    if (savedNodeFills[n.id]) {
      shapeEl.attr("fill", savedNodeFills[n.id].fill);
      shapeEl.attr("stroke-width", savedNodeFills[n.id].sw || "1.5");
      var colors = nodeColorScheme(hwNodeMap[n.id] || n);
      shapeEl.attr("stroke", colors.border);
    }
  });

  adgSvgG.selectAll(".adg-edge").classed("base-edge-overlay", false);
  clearRouteHighlight();
  savedNodeFills = {};
}

// =================================================================
// Panel Divider Drag
// =================================================================
function setupDivider() {
  var divider = document.getElementById("panel-divider");
  var graphArea = document.getElementById("graph-area");
  var adgPanel = document.getElementById("panel-adg");
  var dfgPanel = document.getElementById("panel-dfg");
  if (!divider || !adgPanel || !dfgPanel) return;
  var dragging = false;

  divider.addEventListener("mousedown", function(e) { dragging = true; e.preventDefault(); });
  document.addEventListener("mousemove", function(e) {
    if (!dragging || currentMode !== "sidebyside") return;
    var rect = graphArea.getBoundingClientRect();
    var pct = ((e.clientX - rect.left) / rect.width) * 100;
    pct = Math.max(20, Math.min(80, pct));
    adgPanel.style.flex = "0 0 " + pct + "%";
    dfgPanel.style.flex = "0 0 " + (100 - pct) + "%";
  });
  document.addEventListener("mouseup", function() { dragging = false; });
}

// =================================================================
// Keyboard Shortcuts
// =================================================================
function setupKeyboard() {
  document.addEventListener("keydown", function(e) {
    if (e.key === "Escape") { closeDetail(); return; }
    if (e.key === "1") { setMode("maximize-adg"); return; }
    if (e.key === "2") { setMode("maximize-dfg"); return; }
    if (e.key === "0") { setMode("sidebyside"); return; }
  });
}

// =================================================================
// Trace Playback
// =================================================================
function initTracePlayback() {
  if (typeof TRACE === "undefined" || !TRACE || TRACE === "null") return;

  var events = TRACE.events || [];
  var totalCycles = TRACE.totalCycles || 0;

  var slider = document.getElementById("trace-slider");
  var cycleLabel = document.getElementById("trace-cycle");
  var playBtn = document.getElementById("trace-play");
  var stepBack = document.getElementById("trace-step-back");
  var stepFwd = document.getElementById("trace-step-fwd");
  var speedSelect = document.getElementById("trace-speed");

  if (!slider || !playBtn) return;

  slider.max = totalCycles;
  var currentCycle = 0;
  var playing = false;
  var playTimer = null;
  var speed = 5;

  function clearTraceHighlights() {
    d3.selectAll(".trace-fire").classed("trace-fire", false);
    d3.selectAll(".trace-stall-in").classed("trace-stall-in", false);
    d3.selectAll(".trace-stall-out").classed("trace-stall-out", false);
  }

  function showCycle(cycle) {
    currentCycle = Math.max(0, Math.min(cycle, totalCycles));
    slider.value = currentCycle;
    if (cycleLabel) {
      cycleLabel.textContent = "Cycle: " + currentCycle + " / " + totalCycles;
    }
    clearTraceHighlights();
    events.forEach(function(ev) {
      if (ev.cycle !== currentCycle) return;
      var cls = "trace-fire";
      if (ev.kind === 1) cls = "trace-stall-in";
      else if (ev.kind === 2) cls = "trace-stall-out";
      var nodeEl = d3.select("#" + CSS.escape(ev.hwNode));
      if (!nodeEl.empty()) {
        nodeEl.select("rect").classed(cls, true);
      }
    });
  }

  slider.addEventListener("input", function() {
    showCycle(parseInt(slider.value, 10));
  });

  if (stepBack) {
    stepBack.addEventListener("click", function() { showCycle(currentCycle - 1); });
  }
  if (stepFwd) {
    stepFwd.addEventListener("click", function() { showCycle(currentCycle + 1); });
  }

  playBtn.addEventListener("click", function() {
    playing = !playing;
    playBtn.textContent = playing ? "Pause" : "Play";
    if (playing) {
      playTimer = setInterval(function() {
        currentCycle += speed;
        if (currentCycle > totalCycles) currentCycle = 0;
        showCycle(currentCycle);
      }, 50);
    } else if (playTimer) {
      clearInterval(playTimer);
      playTimer = null;
    }
  });

  if (speedSelect) {
    speedSelect.addEventListener("change", function() {
      speed = parseInt(this.value, 10);
    });
  }
}

// =================================================================
// Initialize
// =================================================================
function init() {
  var sbsBtn = document.getElementById("btn-sidebyside");
  var ovlBtn = document.getElementById("btn-overlay");
  var fitBtn = document.getElementById("btn-fit");
  var restoreBtn = document.getElementById("btn-restore");
  var detailClose = document.getElementById("detail-close");

  if (sbsBtn) sbsBtn.addEventListener("click", function() { setMode("sidebyside"); });
  if (ovlBtn) ovlBtn.addEventListener("click", function() { setMode("overlay"); });
  if (fitBtn) fitBtn.addEventListener("click", function() { fitPanel("adg"); fitPanel("dfg"); });
  if (restoreBtn) restoreBtn.addEventListener("click", function() { setMode("sidebyside"); });
  if (detailClose) detailClose.addEventListener("click", function() { closeDetail(); });

  document.querySelectorAll(".panel-header").forEach(function(header) {
    header.addEventListener("dblclick", function() {
      var panel = header.parentElement.id;
      if (currentMode === "sidebyside")
        setMode(panel === "panel-adg" ? "maximize-adg" : "maximize-dfg");
      else setMode("sidebyside");
    });
  });

  setupDivider();
  setupKeyboard();
  renderADG();
  renderDFG();
  initTracePlayback();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}

})();
