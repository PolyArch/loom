// Loom Visualization Renderer
// Dual-panel viewer: D3.js ADG grid + Graphviz DFG
//
// Expects globals: adgGraph, dfgDot, mappingData, swNodeMetadata, hwNodeMetadata

(function() {
"use strict";

// --- Constants ---
var CELL = 56;          // grid cell size in pixels
var HOP_R = 5;          // hop-over arc radius
var LANE_W = 3;         // lane spacing for parallel edges
var GRID_SP = 4;        // grid expansion factor (routing channels between nodes)
var GRID_PAD = 3;       // padding cells around layout
var ROUTE_PALETTE = [
  "#e6194b","#3cb44b","#4363d8","#f58231",
  "#911eb4","#42d4f4","#f032e6","#bfef45",
  "#fabed4","#469990","#dcbeff","#9A6324"
];

var NODE_COLORS = {
  "fabric.pe":          { fill: "#2d7d2d", border: "#1a5c1a", text: "white" },
  "fabric.temporal_pe": { fill: "#551a8b", border: "#3d1266", text: "white" },
  "fabric.switch":      { fill: "#d3d3d3", border: "#888888", text: "black" },
  "fabric.temporal_sw": { fill: "#708090", border: "#4a5568", text: "white" },
  "fabric.memory":      { fill: "#87ceeb", border: "#4a90a4", text: "black" },
  "fabric.extmemory":   { fill: "#ffd700", border: "#b8960f", text: "black" },
  "fabric.fifo":        { fill: "#e8e8e8", border: "#aaaaaa", text: "black" },
  "fabric.add_tag":     { fill: "#e0ffff", border: "#66aaaa", text: "black" },
  "fabric.map_tag":     { fill: "#da70d6", border: "#9b4d96", text: "black" },
  "fabric.del_tag":     { fill: "#e0ffff", border: "#66aaaa", text: "black" },
  "input":              { fill: "#ffb6c1", border: "#cc8899", text: "black" },
  "output":             { fill: "#f08080", border: "#cc6666", text: "black" }
};

var TYPE_SHORT_NAMES = {
  "fabric.pe": "pe", "fabric.temporal_pe": "tpe",
  "fabric.switch": "switch", "fabric.temporal_sw": "tsw",
  "fabric.memory": "mem", "fabric.extmemory": "extmem",
  "fabric.fifo": "fifo", "fabric.add_tag": "add_tag",
  "fabric.map_tag": "map_tag", "fabric.del_tag": "del_tag",
  "input": "in", "output": "out"
};

var DIALECT_COLORS = {
  "arith":     "#add8e6",
  "dataflow":  "#90ee90",
  "handshake.cond_br": "#ffffe0",
  "handshake.mux":     "#ffffe0",
  "handshake.join":    "#ffffe0",
  "handshake.load":    "#87ceeb",
  "handshake.store":   "#ffa07a",
  "handshake.memory":  "#87ceeb",
  "handshake.extmemory": "#87ceeb",
  "handshake.constant": "#ffd700",
  "math":      "#dda0dd"
};

// --- State ---
var currentMode = "sidebyside";
var selectedNode = null;
var adgZoom, dfgZoom;
var adgSvgG, dfgSvgG;
var savedNodeFills = {};
var savedGpos = null, savedPortPos = null, savedPortInfo = null;

var hwNodeMap = {};
var hwEdgeMap = {};
var adgAdj = {};
var dfgAdj = {};
var dfgEdgeIdCache = new Map();
if (typeof adgGraph !== "undefined") {
  adgGraph.nodes.forEach(function(n) {
    hwNodeMap[n.id] = n;
    adgAdj[n.id] = [];
  });
  adgGraph.edges.forEach(function(e) {
    hwEdgeMap[e.id] = e;
    if (adgAdj[e.srcNode]) adgAdj[e.srcNode].push({ edge: e.id, node: e.dstNode });
    if (adgAdj[e.dstNode]) adgAdj[e.dstNode].push({ edge: e.id, node: e.srcNode });
  });
}

// --- Utility ---
function strokeWidthForBits(bits, isTagged) {
  var w;
  if (bits <= 1) w = 1.0;
  else if (bits <= 16) w = 1.5;
  else if (bits <= 32) w = 2.0;
  else if (bits <= 64) w = 3.0;
  else w = 4.0;
  return isTagged ? w + 0.5 : w;
}
function getDialectColor(opName) {
  if (!opName) return null;
  if (DIALECT_COLORS[opName]) return DIALECT_COLORS[opName];
  var prefix = opName.split(".")[0];
  return DIALECT_COLORS[prefix] || null;
}
function nodeTypeKey(node) {
  if (node.type) return node.type;
  if (node.class === "boundary")
    return node.name && node.name.indexOf("out") >= 0 ? "output" : "input";
  return "fabric.pe";
}
function nodeColorScheme(node) {
  var key = nodeTypeKey(node);
  return NODE_COLORS[key] || NODE_COLORS["fabric.pe"];
}

// Seeded PRNG (mulberry32)
function mulberry32(seed) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function hashGraph(nodes, edges) {
  var h = 5381;
  nodes.forEach(function(n) {
    for (var i = 0; i < n.id.length; i++)
      h = ((h << 5) + h + n.id.charCodeAt(i)) | 0;
  });
  edges.forEach(function(e) {
    for (var i = 0; i < e.id.length; i++)
      h = ((h << 5) + h + e.id.charCodeAt(i)) | 0;
  });
  return h >>> 0;
}
function ff(v) { return v.toFixed(1); }

// =================================================================
// Grid Layout Engine
// =================================================================
// Two modes controlled by vizConfig.neato:
//   false (default): Direct grid placement using viz_row/viz_col coordinates.
//     Nodes with explicit gridCol/gridRow are placed at those positions with
//     GRID_SP spacing. Nodes without coordinates use centroid of neighbors.
//   true: SMACOF stress majorization (neato-style) that optimizes positions
//     to respect graph-theoretic distances, using viz_row/viz_col as initial
//     hints for fast convergence.
function gridLayoutEngine(allNodes, edges) {
  var useNeato = (typeof vizConfig !== "undefined" && vizConfig.neato);
  var rng = mulberry32(hashGraph(allNodes, edges));
  var layoutStart = performance.now();

  // Separate sentinels (boundary I/O nodes) from internal nodes
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

  // --- Build adjacency list (deduplicated) ---
  var adjList = new Array(N);
  for (var i = 0; i < N; i++) adjList[i] = [];
  edges.forEach(function(e) {
    var si = idxOf[e.srcNode], di = idxOf[e.dstNode];
    if (si === undefined || di === undefined || si === di) return;
    if (adjList[si].indexOf(di) === -1) adjList[si].push(di);
    if (adjList[di].indexOf(si) === -1) adjList[di].push(si);
  });

  // --- Shared helpers ---
  var occ = {};
  var gpos = {};
  function ck(gc, gr) { return gc + "," + gr; }
  function occupy(nid, gc, gr, gw, gh) {
    gw = gw || 1; gh = gh || 1;
    gpos[nid] = { gc: gc, gr: gr, gw: gw, gh: gh };
    for (var dc = 0; dc < gw; dc++)
      for (var dr = 0; dr < gh; dr++)
        occ[ck(gc + dc, gr + dr)] = nid;
  }

  var gridW, gridH;

  if (!useNeato) {
    // ====== Direct placement mode ======
    // Place each node at its gridCol/gridRow scaled by GRID_SP.
    // Nodes without coordinates get centroid of placed neighbors.
    var placed = new Uint8Array(N);
    var gcArr = new Int32Array(N);
    var grArr = new Int32Array(N);

    // Assign grid positions from explicit coordinates
    internal.forEach(function(n, i) {
      if (n.gridCol != null && n.gridRow != null) {
        gcArr[i] = n.gridCol * GRID_SP + GRID_PAD;
        grArr[i] = n.gridRow * GRID_SP + GRID_PAD;
        placed[i] = 1;
      }
    });

    // Propagate to unplaced nodes via neighbor centroids
    for (var pass = 0; pass < 10; pass++) {
      var progress = false;
      for (var i = 0; i < N; i++) {
        if (placed[i]) continue;
        var sx = 0, sy = 0, cnt = 0;
        var nb = adjList[i];
        for (var k = 0; k < nb.length; k++) {
          if (placed[nb[k]]) { sx += gcArr[nb[k]]; sy += grArr[nb[k]]; cnt++; }
        }
        if (cnt > 0) {
          gcArr[i] = Math.round(sx / cnt);
          grArr[i] = Math.round(sy / cnt);
          placed[i] = 1;
          progress = true;
        }
      }
      if (!progress) break;
    }

    // Still unplaced: put at bounding box center
    var bMinC = Infinity, bMaxC = -Infinity, bMinR = Infinity, bMaxR = -Infinity;
    for (var i = 0; i < N; i++) {
      if (!placed[i]) continue;
      if (gcArr[i] < bMinC) bMinC = gcArr[i]; if (gcArr[i] > bMaxC) bMaxC = gcArr[i];
      if (grArr[i] < bMinR) bMinR = grArr[i]; if (grArr[i] > bMaxR) bMaxR = grArr[i];
    }
    if (bMinC === Infinity) { bMinC = GRID_PAD; bMaxC = GRID_PAD + 10; bMinR = GRID_PAD; bMaxR = GRID_PAD + 10; }
    for (var i = 0; i < N; i++) {
      if (!placed[i]) {
        gcArr[i] = Math.round((bMinC + bMaxC) / 2);
        grArr[i] = Math.round((bMinR + bMaxR) / 2);
      }
    }

    // Compute grid size (accounting for multi-cell nodes)
    var maxC = 0, maxR = 0;
    for (var i = 0; i < N; i++) {
      var nw = (internal[i].areaW || 1) - 1;
      var nh = (internal[i].areaH || 1) - 1;
      if (gcArr[i] + nw > maxC) maxC = gcArr[i] + nw;
      if (grArr[i] + nh > maxR) maxR = grArr[i] + nh;
    }
    gridW = maxC + GRID_PAD + 2;
    gridH = maxR + GRID_PAD + 2;
    if (gridW < 2 * GRID_PAD + 2) gridW = 2 * GRID_PAD + 2;
    if (gridH < 2 * GRID_PAD + 2) gridH = 2 * GRID_PAD + 2;

    function isFree(gc, gr) {
      return gc >= 0 && gr >= -5 && gc < gridW + 5 && gr < gridH + 5 &&
             !occ[ck(gc, gr)];
    }
    function isRectFree(gc, gr, gw, gh) {
      for (var dc = 0; dc < gw; dc++)
        for (var dr = 0; dr < gh; dr++)
          if (!isFree(gc + dc, gr + dr)) return false;
      return true;
    }
    function findFree(ic, ir, gw, gh) {
      gw = gw || 1; gh = gh || 1;
      ic = Math.round(ic); ir = Math.round(ir);
      if (isRectFree(ic, ir, gw, gh)) return { gc: ic, gr: ir };
      var maxD = Math.max(gridW, gridH) + 10;
      for (var d = 1; d <= maxD; d++) {
        for (var dx = -d; dx <= d; dx++) {
          var ady = d - Math.abs(dx);
          var cands = ady === 0 ? [0] : [-ady, ady];
          for (var ci = 0; ci < cands.length; ci++) {
            var gc = ic + dx, gr = ir + cands[ci];
            if (isRectFree(gc, gr, gw, gh)) return { gc: gc, gr: gr };
          }
        }
      }
      return { gc: gridW, gr: gridH };
    }

    // Build targets sorted by degree (most-connected first)
    var gridTargets = [];
    for (var i = 0; i < N; i++) {
      gridTargets.push({ node: internal[i], gc: gcArr[i], gr: grArr[i], degree: adjList[i].length,
                          gw: internal[i].areaW || 1, gh: internal[i].areaH || 1 });
    }
    gridTargets.sort(function(a, b) { return b.degree - a.degree; });
    gridTargets.forEach(function(t) {
      var spot = findFree(t.gc, t.gr, t.gw, t.gh);
      occupy(t.node.id, spot.gc, spot.gr, t.gw, t.gh);
    });

  } else {
    // ====== Neato mode (stress majorization) ======
    // All-pairs shortest path via BFS
    var INF = N + 1;
    var dist = new Int16Array(N * N);
    for (var k = 0; k < N * N; k++) dist[k] = INF;
    for (var i = 0; i < N; i++) {
      dist[i * N + i] = 0;
      var queue = [i], head = 0;
      while (head < queue.length) {
        var u = queue[head++];
        var neighbors = adjList[u];
        for (var k = 0; k < neighbors.length; k++) {
          var v = neighbors[k];
          if (dist[i * N + v] === INF) {
            dist[i * N + v] = dist[i * N + u] + 1;
            queue.push(v);
          }
        }
      }
    }

    // Initialize positions from gridCol/gridRow
    var X = new Float64Array(N);
    var Y = new Float64Array(N);
    var hasInit = new Uint8Array(N);
    internal.forEach(function(n, i) {
      if (n.gridCol != null && n.gridRow != null) {
        X[i] = n.gridCol;
        Y[i] = n.gridRow;
        hasInit[i] = 1;
      }
    });

    // Propagate: uninitialized nodes get centroid of initialized neighbors
    for (var pass = 0; pass < 5; pass++) {
      for (var i = 0; i < N; i++) {
        if (hasInit[i]) continue;
        var sx = 0, sy = 0, cnt = 0;
        var nb = adjList[i];
        for (var k = 0; k < nb.length; k++) {
          if (hasInit[nb[k]]) { sx += X[nb[k]]; sy += Y[nb[k]]; cnt++; }
        }
        if (cnt > 0) {
          X[i] = sx / cnt + (rng() - 0.5) * 0.1;
          Y[i] = sy / cnt + (rng() - 0.5) * 0.1;
          hasInit[i] = 1;
        }
      }
    }

    // Still uninitialized: random within bounding box
    var bMinX = Infinity, bMaxX = -Infinity, bMinY = Infinity, bMaxY = -Infinity;
    for (var i = 0; i < N; i++) {
      if (!hasInit[i]) continue;
      if (X[i] < bMinX) bMinX = X[i]; if (X[i] > bMaxX) bMaxX = X[i];
      if (Y[i] < bMinY) bMinY = Y[i]; if (Y[i] > bMaxY) bMaxY = Y[i];
    }
    if (bMinX === Infinity) { bMinX = 0; bMaxX = 10; bMinY = 0; bMaxY = 10; }
    var bRange = Math.max(bMaxX - bMinX, bMaxY - bMinY, 5);
    for (var i = 0; i < N; i++) {
      if (!hasInit[i]) {
        X[i] = (bMinX + bMaxX) / 2 + (rng() - 0.5) * bRange;
        Y[i] = (bMinY + bMaxY) / 2 + (rng() - 0.5) * bRange;
      }
    }

    // SMACOF stress majorization
    var IDEAL_SCALE = 2.0;
    if (N > 1) {
      var MAX_ITER = 300;
      var CONVERGE_TOL = 1e-4;
      var wt = new Float64Array(N * N);
      var ideal = new Float64Array(N * N);
      for (var i = 0; i < N; i++) {
        for (var j = 0; j < N; j++) {
          if (i === j || dist[i * N + j] >= INF) continue;
          var dij = dist[i * N + j] * IDEAL_SCALE;
          ideal[i * N + j] = dij;
          wt[i * N + j] = 1.0 / (dij * dij);
        }
      }

      for (var iter = 0; iter < MAX_ITER; iter++) {
        if (performance.now() - layoutStart > 2000) break;
        var newX = new Float64Array(N);
        var newY = new Float64Array(N);
        var maxMove = 0;
        for (var i = 0; i < N; i++) {
          var sumWx = 0, sumWy = 0, sumW = 0;
          for (var j = 0; j < N; j++) {
            var w = wt[i * N + j];
            if (w === 0) continue;
            var dij = ideal[i * N + j];
            var dx = X[i] - X[j];
            var dy = Y[i] - Y[j];
            var eucDist = Math.sqrt(dx * dx + dy * dy);
            if (eucDist < 1e-8) {
              dx = (rng() - 0.5) * 1e-4;
              dy = (rng() - 0.5) * 1e-4;
              eucDist = Math.sqrt(dx * dx + dy * dy);
            }
            sumW += w;
            sumWx += w * (X[j] + dij * dx / eucDist);
            sumWy += w * (Y[j] + dij * dy / eucDist);
          }
          if (sumW > 1e-12) {
            newX[i] = sumWx / sumW;
            newY[i] = sumWy / sumW;
          } else {
            newX[i] = X[i];
            newY[i] = Y[i];
          }
          var mx = Math.abs(newX[i] - X[i]);
          var my = Math.abs(newY[i] - Y[i]);
          if (mx > maxMove) maxMove = mx;
          if (my > maxMove) maxMove = my;
        }
        X = newX;
        Y = newY;
        if (maxMove < CONVERGE_TOL) break;
      }
    }

    // Grid snapping
    var sMinX = Infinity, sMaxX = -Infinity, sMinY = Infinity, sMaxY = -Infinity;
    for (var i = 0; i < N; i++) {
      if (X[i] < sMinX) sMinX = X[i]; if (X[i] > sMaxX) sMaxX = X[i];
      if (Y[i] < sMinY) sMinY = Y[i]; if (Y[i] > sMaxY) sMaxY = Y[i];
    }
    if (sMinX === Infinity) { sMinX = 0; sMaxX = 5; sMinY = 0; sMaxY = 5; }

    var stressToGrid = GRID_SP / IDEAL_SCALE;
    gridW = Math.round((sMaxX - sMinX) * stressToGrid) + 2 * GRID_PAD + 2;
    gridH = Math.round((sMaxY - sMinY) * stressToGrid) + 2 * GRID_PAD + 2;
    if (gridW < 2 * GRID_PAD + 2) gridW = 2 * GRID_PAD + 2;
    if (gridH < 2 * GRID_PAD + 2) gridH = 2 * GRID_PAD + 2;

    function isFree(gc, gr) {
      return gc >= 0 && gr >= -5 && gc < gridW + 5 && gr < gridH + 5 &&
             !occ[ck(gc, gr)];
    }
    function isRectFree(gc, gr, gw, gh) {
      for (var dc = 0; dc < gw; dc++)
        for (var dr = 0; dr < gh; dr++)
          if (!isFree(gc + dc, gr + dr)) return false;
      return true;
    }
    function findFree(ic, ir, gw, gh) {
      gw = gw || 1; gh = gh || 1;
      ic = Math.round(ic); ir = Math.round(ir);
      if (isRectFree(ic, ir, gw, gh)) return { gc: ic, gr: ir };
      var maxD = Math.max(gridW, gridH) + 10;
      for (var d = 1; d <= maxD; d++) {
        for (var dx = -d; dx <= d; dx++) {
          var ady = d - Math.abs(dx);
          var cands = ady === 0 ? [0] : [-ady, ady];
          for (var ci = 0; ci < cands.length; ci++) {
            var gc = ic + dx, gr = ir + cands[ci];
            if (isRectFree(gc, gr, gw, gh)) return { gc: gc, gr: gr };
          }
        }
      }
      return { gc: gridW, gr: gridH };
    }

    var gridTargets = [];
    for (var i = 0; i < N; i++) {
      gridTargets.push({
        idx: i,
        node: internal[i],
        gc: Math.round((X[i] - sMinX) * stressToGrid) + GRID_PAD,
        gr: Math.round((Y[i] - sMinY) * stressToGrid) + GRID_PAD,
        degree: adjList[i].length,
        gw: internal[i].areaW || 1, gh: internal[i].areaH || 1
      });
    }
    gridTargets.sort(function(a, b) { return b.degree - a.degree; });
    gridTargets.forEach(function(t) {
      var spot = findFree(t.gc, t.gr, t.gw, t.gh);
      occupy(t.node.id, spot.gc, spot.gr, t.gw, t.gh);
    });
  }

  // --- Sentinel Placement (boundary I/O nodes above/below main grid) ---
  var inSent = sentinels.filter(function(s) { return nodeTypeKey(s) !== "output"; });
  var outSent = sentinels.filter(function(s) { return nodeTypeKey(s) === "output"; });

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

  function isFreeS(gc, gr) {
    return gc >= 0 && gr >= -5 && gc < gridW + 5 && gr < gridH + 5 &&
           !occ[ck(gc, gr)];
  }
  function findFreeS(ic, ir) {
    ic = Math.round(ic); ir = Math.round(ir);
    if (isFreeS(ic, ir)) return { gc: ic, gr: ir };
    var maxD = Math.max(gridW, gridH) + 10;
    for (var d = 1; d <= maxD; d++) {
      for (var dx = -d; dx <= d; dx++) {
        var ady = d - Math.abs(dx);
        var cands = ady === 0 ? [0] : [-ady, ady];
        for (var ci = 0; ci < cands.length; ci++) {
          var gc = ic + dx, gr = ir + cands[ci];
          if (isFreeS(gc, gr)) return { gc: gc, gr: gr };
        }
      }
    }
    return { gc: gridW, gr: gridH };
  }

  inSent.forEach(function(s) {
    var gc = sentinelIdealCol(s);
    var spot = findFreeS(gc, -2);
    occupy(s.id, spot.gc, spot.gr);
  });
  outSent.forEach(function(s) {
    var gc = sentinelIdealCol(s);
    var spot = findFreeS(gc, gridH + 1);
    occupy(s.id, spot.gc, spot.gr);
  });

  return { gpos: gpos, occ: occ, gridW: gridW, gridH: gridH };
}

// =================================================================
// Manhattan Router (BFS through free grid cells)
// =================================================================
function manhattanRouter(edges, layout) {
  var gpos = layout.gpos;
  var occ = layout.occ;
  var gridW = layout.gridW, gridH = layout.gridH;
  var dirs4 = [[1,0],[-1,0],[0,1],[0,-1]];

  function ck(gc, gr) { return gc + "," + gr; }

  // Get free cells adjacent to a node
  function adjFree(p) {
    var res = [];
    for (var dc = -1; dc <= p.gw; dc++) {
      for (var dr = -1; dr <= p.gh; dr++) {
        if (dc >= 0 && dc < p.gw && dr >= 0 && dr < p.gh) continue;
        var gc = p.gc + dc, gr = p.gr + dr;
        if (!occ[ck(gc, gr)]) res.push({ gc: gc, gr: gr });
      }
    }
    return res;
  }

  // Check if grid-level L-path between two nodes is clear of occupied cells
  function lPathClear(srcId, dstId, hFirst) {
    var sp = gpos[srcId], dp = gpos[dstId];
    var sc = sp.gc + Math.floor(sp.gw / 2), sr = sp.gr + Math.floor(sp.gh / 2);
    var dc = dp.gc + Math.floor(dp.gw / 2), dr = dp.gr + Math.floor(dp.gh / 2);
    if (hFirst) {
      for (var c = Math.min(sc, dc); c <= Math.max(sc, dc); c++) {
        var o = occ[ck(c, sr)];
        if (o && o !== srcId && o !== dstId) return false;
      }
      for (var r = Math.min(sr, dr); r <= Math.max(sr, dr); r++) {
        var o = occ[ck(dc, r)];
        if (o && o !== srcId && o !== dstId) return false;
      }
    } else {
      for (var r = Math.min(sr, dr); r <= Math.max(sr, dr); r++) {
        var o = occ[ck(sc, r)];
        if (o && o !== srcId && o !== dstId) return false;
      }
      for (var c = Math.min(sc, dc); c <= Math.max(sc, dc); c++) {
        var o = occ[ck(c, dr)];
        if (o && o !== srcId && o !== dstId) return false;
      }
    }
    return true;
  }

  // BFS shortest path through free cells
  function bfs(srcId, dstId) {
    var sp = gpos[srcId], dp = gpos[dstId];
    if (!sp || !dp) return null;
    // Adjacent nodes: use L-path shortcut if clear
    var hGap = Math.max(0, Math.max(dp.gc - (sp.gc + sp.gw), sp.gc - (dp.gc + dp.gw)));
    var vGap = Math.max(0, Math.max(dp.gr - (sp.gr + sp.gh), sp.gr - (dp.gr + dp.gh)));
    if (hGap + vGap <= 1) {
      if (lPathClear(srcId, dstId, true)) return [];
      if (lPathClear(srcId, dstId, false)) return [];
    }

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

  // Route all edges
  var edgePaths = {};
  var segCount = {};
  var segIdx = {};

  function sk(gc1, gr1, gc2, gr2) {
    if (gc1 < gc2 || (gc1 === gc2 && gr1 < gr2))
      return gc1 + "," + gr1 + "-" + gc2 + "," + gr2;
    return gc2 + "," + gr2 + "-" + gc1 + "," + gr1;
  }

  // Cell direction tracking for crossing detection
  var cellEdges = {};  // "gc,gr" -> Set of {edgeId, dir}

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
    // Time budget: stop routing after 2s, use fallback for remaining
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

  // Detect crossings: cells with H and V edges from different sources
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
    // Check at least one H edge differs from all V edges
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
    // Adjacent or no path: L-path (horizontal then vertical)
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
    // Use outgoing segment offset (or incoming for last waypoint)
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

  // Build SVG path string with hop-over arcs
  var d = "M" + ff(pts[0].x) + "," + ff(pts[0].y);
  for (var i = 1; i < pts.length; i++) {
    var px = pts[i].x, py = pts[i].y;
    var hopHere = false;

    // Check hop: only for routing waypoints going straight-through horizontally
    if (crossings) {
      var pi = i - 1 - kneesBefore; // offset by inserted knees
      if (pi >= 0 && pi < path.length) {
        var cellK = path[pi].gc + "," + path[pi].gr;
        if (crossings[cellK]) {
          // Straight-through horizontal: prev and next on same row
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
// Port Position Computation (grid-based, angle-optimised)
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

  var portPos = {};
  nodes.forEach(function(n) {
    var np = gpos[n.id];
    if (!np) return;
    var cx = (np.gc + np.gw / 2) * CELL;
    var cy = (np.gr + np.gh / 2) * CELL;
    var halfW = np.gw * CELL / 2 - 2;
    var halfH = np.gh * CELL / 2 - 2;

    var allPorts = (nodeIn[n.id] || []).concat(nodeOut[n.id] || []);
    var sides = { top: [], right: [], bottom: [], left: [] };

    allPorts.forEach(function(pid) {
      var pi = portInfo[pid];
      if (!pi) return;
      var dx = 0, dy = 0, cnt = 0;
      pi.peers.forEach(function(peerId) {
        var pp = gpos[peerId];
        if (pp) {
          dx += (pp.gc + pp.gw / 2) * CELL - cx;
          dy += (pp.gr + pp.gh / 2) * CELL - cy;
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
// ADG Rendering
// =================================================================
function renderADG() {
  if (typeof adgGraph === "undefined" || !adgGraph.nodes) return;
  var svg = d3.select("#svg-adg");
  d3.select("#status-bar").text("Computing layout...");

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

  // 5. Layout extent
  var ext = { mnX: Infinity, mxX: -Infinity, mnY: Infinity, mxY: -Infinity };
  Object.keys(gpos).forEach(function(nid) {
    var p = gpos[nid];
    ext.mnX = Math.min(ext.mnX, p.gc * CELL);
    ext.mxX = Math.max(ext.mxX, (p.gc + p.gw) * CELL);
    ext.mnY = Math.min(ext.mnY, p.gr * CELL);
    ext.mxY = Math.max(ext.mxY, (p.gr + p.gh) * CELL);
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

  // 6b. Width-plane overlay rectangles
  // Group nodes by widthPlane, compute bounding boxes, draw labeled backgrounds.
  var planeGroups = {};
  adgGraph.nodes.forEach(function(n) {
    if (n.widthPlane == null) return;
    var p = gpos[n.id];
    if (!p) return;
    var key = n.widthPlane;
    if (!planeGroups[key]) {
      planeGroups[key] = {
        label: n.widthPlaneLabel || (key + ""),
        mnX: Infinity, mxX: -Infinity, mnY: Infinity, mxY: -Infinity
      };
    }
    var pg = planeGroups[key];
    var x0 = p.gc * CELL;
    var x1 = (p.gc + p.gw) * CELL;
    var y0 = p.gr * CELL;
    var y1 = (p.gr + p.gh) * CELL;
    if (x0 < pg.mnX) pg.mnX = x0;
    if (x1 > pg.mxX) pg.mxX = x1;
    if (y0 < pg.mnY) pg.mnY = y0;
    if (y1 > pg.mxY) pg.mxY = y1;
  });

  var PLANE_COLORS = [
    "rgba(70,130,180,0.08)",
    "rgba(60,179,113,0.08)",
    "rgba(218,112,214,0.08)",
    "rgba(255,165,0,0.08)",
    "rgba(220,20,60,0.08)",
    "rgba(100,149,237,0.08)",
    "rgba(144,238,144,0.08)",
    "rgba(255,215,0,0.08)"
  ];
  var PLANE_BORDER_COLORS = [
    "rgba(70,130,180,0.4)",
    "rgba(60,179,113,0.4)",
    "rgba(218,112,214,0.4)",
    "rgba(255,165,0,0.4)",
    "rgba(220,20,60,0.4)",
    "rgba(100,149,237,0.4)",
    "rgba(144,238,144,0.4)",
    "rgba(255,215,0,0.4)"
  ];

  var planeKeys = Object.keys(planeGroups).sort(function(a, b) { return +a - +b; });
  if (planeKeys.length > 1) {
    var planeG = g.insert("g", ".edges").attr("class", "width-planes");
    planeKeys.forEach(function(key, idx) {
      var pg = planeGroups[key];
      if (pg.mnX === Infinity) return;
      var pad = CELL * 0.6;
      var rx = pg.mnX - pad;
      var ry = pg.mnY - pad;
      var rw = pg.mxX - pg.mnX + pad * 2;
      var rh = pg.mxY - pg.mnY + pad * 2;
      var isCross = pg.label === "cross";
      planeG.append("rect")
        .attr("x", rx).attr("y", ry)
        .attr("width", rw).attr("height", rh)
        .attr("rx", 8).attr("ry", 8)
        .attr("fill", isCross ? "rgba(128,128,128,0.06)" : PLANE_COLORS[idx % PLANE_COLORS.length])
        .attr("stroke", isCross ? "rgba(128,128,128,0.5)" : PLANE_BORDER_COLORS[idx % PLANE_BORDER_COLORS.length])
        .attr("stroke-width", isCross ? 2 : 1.5)
        .attr("stroke-dasharray", isCross ? "4,4" : "6,3");
      planeG.append("text")
        .attr("x", rx + 6).attr("y", ry + 14)
        .attr("font-size", "11px")
        .attr("font-weight", "bold")
        .attr("fill", isCross ? "rgba(128,128,128,0.7)" : PLANE_BORDER_COLORS[idx % PLANE_BORDER_COLORS.length])
        .text(isCross ? "Cross-plane" : "Plane " + pg.label);
    });
  }

  // 7. Draw edges (Manhattan routing with hop-over arcs)
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
    var lanes = routing.edgeLanes[e.id];
    var svgPath = buildManhattanSVG(srcP, dstP, path, lanes, routing.crossings);

    var sw = strokeWidthForBits(e.valueBitWidth || e.bitWidth || 32,
                                 e.edgeType === "tagged");
    var color = "#555";
    var dasharray = null;
    if (e.edgeType === "tagged") { color = "#7b2d8b"; dasharray = "6,3"; }
    else if (e.edgeType === "memref") { color = "#2255cc"; dasharray = "2,3"; }
    else if (e.edgeType === "control") { color = "#999"; dasharray = "4,2"; }

    var ep = edgeG.append("path")
      .attr("class", "adg-edge").attr("id", e.id)
      .attr("d", svgPath)
      .attr("stroke", color)
      .attr("stroke-width", Math.max(sw * 0.7, 0.7))
      .attr("opacity", 0.6);
    if (dasharray) ep.attr("stroke-dasharray", dasharray);
  });

  // 8. Draw nodes (ALL rectangles)
  var nodeG = g.append("g").attr("class", "nodes");
  adgGraph.nodes.forEach(function(n) {
    var p = gpos[n.id];
    if (!p) return;
    var colors = nodeColorScheme(n);
    var cx = (p.gc + p.gw / 2) * CELL;
    var cy = (p.gr + p.gh / 2) * CELL;
    var w = p.gw * CELL - 4;
    var h = p.gh * CELL - 4;

    var ng = nodeG.append("g")
      .attr("class", "adg-node").attr("id", n.id)
      .attr("transform", "translate(" + cx + "," + cy + ")");

    ng.append("rect")
      .attr("x", -w / 2).attr("y", -h / 2)
      .attr("width", w).attr("height", h)
      .attr("rx", 3).attr("ry", 3)
      .attr("fill", colors.fill)
      .attr("stroke", colors.border)
      .attr("stroke-width", 1.5);

    // Port indicators
    var allPorts = (nodeInPorts[n.id] || []).concat(nodeOutPorts[n.id] || []);
    allPorts.forEach(function(pid) {
      var pp = portPos[pid];
      if (!pp) return;
      var pi = portInfo[pid];
      ng.append("circle")
        .attr("cx", pp.x - cx).attr("cy", pp.y - cy)
        .attr("r", 2)
        .attr("fill", pi && pi.isOutput ? "#e94560" : "#4363d8")
        .attr("stroke", "white").attr("stroke-width", 0.5);
    });

    // Label
    var label = n.name || n.id;
    // Anonymous nodes: show "<type>\n(<id>)" two-line format
    var anonMatch = /^node_(\d+)$/.exec(label);
    if (anonMatch) {
      var tKey = nodeTypeKey(n);
      var shortName = TYPE_SHORT_NAMES[tKey] || tKey.replace("fabric.", "");
      var tEl = ng.append("text")
        .attr("text-anchor", "middle")
        .attr("fill", colors.text).attr("font-size", "7px");
      tEl.append("tspan").attr("x", 0).attr("dy", "-0.3em").text(shortName);
      tEl.append("tspan").attr("x", 0).attr("dy", "1.1em").text("(" + anonMatch[1] + ")");
    } else {
      if (label.length > 12) label = label.substring(0, 10) + "..";
      ng.append("text")
        .attr("dy", "0.35em").attr("text-anchor", "middle")
        .attr("fill", colors.text).attr("font-size", "8px")
        .text(label);
    }

    // Interaction
    ng.on("mouseenter", function() { highlightNode(n.id, "hw"); })
      .on("mouseleave", function() { clearHighlights(); })
      .on("click", function(event) { event.stopPropagation(); showDetail(n.id, "hw"); });
  });

  // Save layout state for overlay mode
  savedGpos = gpos;
  savedPortPos = portPos;
  savedPortInfo = portInfo;

  // Fit to view
  fitPanel("adg");
  d3.select("#status-bar").text("Ready");

  svg.on("click", function() {
    closeDetail();
    if (currentMode === "overlay") clearRouteHighlight();
  });
  svg.on("dblclick.zoom", null);
  svg.on("dblclick", function() { fitPanel("adg"); });
}

// --- DFG Rendering ---
function renderDFG() {
  if (typeof dfgDot === "undefined" || !dfgDot || typeof VizStandalone === "undefined") return;
  var svg = d3.select("#svg-dfg");
  d3.select("#status-bar").text("Initializing Graphviz WASM...");

  VizStandalone.instance().then(function(viz) {
    d3.select("#status-bar").text("Rendering DFG...");
    var svgStr = viz.renderString(dfgDot, { engine: "dot", format: "svg" });
    var container = document.getElementById("panel-dfg");
    var svgWrapper = document.createElement("div");
    svgWrapper.innerHTML = svgStr;
    var renderedSvg = svgWrapper.querySelector("svg");
    if (!renderedSvg) {
      d3.select("#status-bar").text("DFG render failed");
      return;
    }
    renderedSvg.setAttribute("id", "svg-dfg");
    renderedSvg.removeAttribute("viewBox");
    renderedSvg.removeAttribute("width");
    renderedSvg.removeAttribute("height");
    renderedSvg.style.width = "100%";
    renderedSvg.style.height = "100%";
    var oldSvg = document.getElementById("svg-dfg");
    if (oldSvg) oldSvg.parentNode.replaceChild(renderedSvg, oldSvg);

    var dfgSvgSel = d3.select("#svg-dfg");
    var innerG = dfgSvgSel.select("g");
    dfgSvgG = innerG;

    dfgZoom = d3.zoom()
      .scaleExtent([0.1, 5])
      .on("zoom", function(event) { innerG.attr("transform", event.transform); });
    dfgSvgSel.call(dfgZoom);

    dfgSvgSel.selectAll(".node").each(function() {
      var el = d3.select(this);
      var titleEl = el.select("title");
      var nodeTitle = titleEl.empty() ? "" : titleEl.text();
      var nodeId = el.attr("id");
      if (!nodeId && nodeTitle) nodeId = nodeTitle;
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

    dfgSvgSel.selectAll(".edge").each(function() {
      var el = d3.select(this);
      var titleEl = el.select("title");
      var edgeTitle = titleEl.empty() ? "" : titleEl.text();
      var edgeId = el.attr("id") || edgeTitle;
      if (edgeId && mappingData && mappingData.routes && mappingData.routes[edgeId]) {
        el.style("cursor", "pointer");
        el.on("mouseenter", function() { highlightRoute(edgeId); })
          .on("mouseleave", function() { clearRouteHighlight(); })
          .on("click", function(event) {
            event.stopPropagation();
            clearRouteHighlight();
            highlightRoute(edgeId);
          });
      }
    });

    dfgAdj = {};
    dfgEdgeIdCache = new Map();
    dfgSvgSel.selectAll(".node").each(function() {
      var nId = d3.select(this).attr("id");
      if (nId) dfgAdj[nId] = [];
    });
    dfgSvgSel.selectAll(".edge").each(function() {
      var el = d3.select(this);
      var titleEl = el.select("title");
      if (titleEl.empty()) return;
      var parts = titleEl.text().split("->");
      if (parts.length !== 2) return;
      var src = parts[0].trim(), dst = parts[1].trim();
      var eId = el.attr("id") || titleEl.text();
      dfgEdgeIdCache.set(this, eId);
      if (dfgAdj[src]) dfgAdj[src].push({ edge: eId, node: dst });
      if (dfgAdj[dst]) dfgAdj[dst].push({ edge: eId, node: src });
    });

    dfgSvgSel.on("click", function() { closeDetail(); });
    dfgSvgSel.on("dblclick.zoom", null);
    dfgSvgSel.on("dblclick", function() { fitPanel("dfg"); });
    d3.select("#status-bar").text("Ready");
    fitPanel("dfg");
  }).catch(function(err) {
    d3.select("#status-bar").text("Graphviz init failed: " + err.message);
    console.error("Viz.js error:", err);
  });
}

// --- Fit Panel ---
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

// --- Cross-Highlighting ---
function highlightNode(nodeId, domain) {
  clearHighlights();
  var el = d3.select("#" + CSS.escape(nodeId));
  if (!el.empty()) {
    el.select("rect, ellipse").classed("node-highlight", true);
  }

  if (domain === "hw") {
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
  }

  if (domain === "sw" && dfgAdj[nodeId]) {
    var adjSwN = {}, adjSwE = {};
    adjSwN[nodeId] = true;
    dfgAdj[nodeId].forEach(function(a) {
      adjSwN[a.node] = true; adjSwE[a.edge] = true;
    });
    d3.select("#svg-dfg").selectAll(".node").each(function() {
      var nEl = d3.select(this);
      if (!adjSwN[nEl.attr("id")]) nEl.classed("node-dimmed", true);
    });
    d3.select("#svg-dfg").selectAll(".edge").each(function() {
      var eId = dfgEdgeIdCache.get(this) || d3.select(this).attr("id") || "";
      if (!adjSwE[eId]) d3.select(this).classed("edge-dimmed", true);
    });
  }

  if (!mappingData) return;
  if (domain === "sw" && mappingData.swToHw) {
    var hwId = mappingData.swToHw[nodeId];
    var hwFocusEls = [];
    if (hwId) {
      var hwEl = d3.select("#" + CSS.escape(hwId));
      if (!hwEl.empty()) {
        hwEl.select("rect").classed("cross-highlight", true);
        hwFocusEls.push(hwEl);
      }
    }
    if (hwFocusEls.length > 0) focusOnRegion("adg", hwFocusEls);
  } else if (domain === "hw" && mappingData.hwToSw) {
    var swIds = mappingData.hwToSw[nodeId];
    var swFocusEls = [];
    if (swIds) {
      swIds.forEach(function(swId) {
        var swEl = d3.select("#" + CSS.escape(swId));
        if (!swEl.empty()) {
          swEl.select("polygon, ellipse, rect, path").classed("cross-highlight", true);
          swFocusEls.push(swEl);
        }
      });
    }
    if (swFocusEls.length > 0) focusOnRegion("dfg", swFocusEls);
  }
}

function clearHighlights() {
  d3.selectAll(".node-highlight").classed("node-highlight", false);
  d3.selectAll(".cross-highlight").classed("cross-highlight", false);
  d3.selectAll(".node-dimmed").classed("node-dimmed", false);
  d3.selectAll(".edge-dimmed").classed("edge-dimmed", false);
}

function highlightRoute(edgeId) {
  if (!mappingData || !mappingData.routes) return;
  var route = mappingData.routes[edgeId];
  if (!route || !route.hwPath) return;
  var routeEls = [];
  route.hwPath.forEach(function(hop) {
    if (hop.hwEdgeId) {
      var edgeEl = d3.select("#" + CSS.escape(hop.hwEdgeId));
      if (!edgeEl.empty()) {
        edgeEl.classed("route-trace", true);
        routeEls.push(edgeEl);
      }
    }
    if (hop.hwNodeId) {
      var nodeEl = d3.select("#" + CSS.escape(hop.hwNodeId));
      if (!nodeEl.empty()) routeEls.push(nodeEl);
    }
  });
  if (routeEls.length > 0) focusOnRegion("adg", routeEls);
}

function clearRouteHighlight() {
  d3.selectAll(".route-trace").classed("route-trace", false);
}

// Zoom the given panel to fit a list of D3 selections with 30% margin.
// Transforms each element's local bbox into the zoom group's coordinate space
// so that nested transform groups (e.g. ADG nodes with translate) are handled.
function focusOnRegion(panel, elements) {
  var g;
  if (panel === "adg") g = adgSvgG;
  else g = dfgSvgG;
  if (!g) return;
  var groupNode = g.node();
  if (!groupNode) return;
  var groupCTM = groupNode.getCTM();
  if (!groupCTM) return;
  var invGroupCTM = groupCTM.inverse();

  var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  elements.forEach(function(el) {
    var node = el.node();
    if (!node) return;
    var bbox = node.getBBox();
    var elCTM = node.getCTM();
    if (!elCTM) return;
    // Matrix that maps from element-local coords to zoom-group coords.
    var matrix = invGroupCTM.multiply(elCTM);
    // Transform all 4 bbox corners (handles rotation/skew if any).
    var corners = [
      new DOMPoint(bbox.x, bbox.y),
      new DOMPoint(bbox.x + bbox.width, bbox.y),
      new DOMPoint(bbox.x, bbox.y + bbox.height),
      new DOMPoint(bbox.x + bbox.width, bbox.y + bbox.height)
    ];
    corners.forEach(function(pt) {
      var tp = pt.matrixTransform(matrix);
      minX = Math.min(minX, tp.x);
      minY = Math.min(minY, tp.y);
      maxX = Math.max(maxX, tp.x);
      maxY = Math.max(maxY, tp.y);
    });
  });
  if (minX === Infinity) return;

  // Add 30% margin on each side.
  var w = maxX - minX, h = maxY - minY;
  var marginFrac = 0.3;
  minX -= w * marginFrac / 2;
  minY -= h * marginFrac / 2;
  maxX += w * marginFrac / 2;
  maxY += h * marginFrac / 2;
  w = maxX - minX;
  h = maxY - minY;

  var svg, zoom;
  if (panel === "adg") { svg = d3.select("#svg-adg"); zoom = adgZoom; }
  else { svg = d3.select("#svg-dfg"); zoom = dfgZoom; }
  if (!svg || !zoom) return;
  var svgNode = svg.node();
  var svgW = svgNode.clientWidth || 800;
  var svgH = svgNode.clientHeight || 600;

  var scale = Math.min(svgW / w, svgH / h, 2);
  var cx = (minX + maxX) / 2;
  var cy = (minY + maxY) / 2;
  var tx = svgW / 2 - cx * scale;
  var ty = svgH / 2 - cy * scale;

  svg.transition().duration(500)
    .call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}

// --- Detail Panel ---
function showDetail(nodeId, domain) {
  var panel = document.getElementById("detail-panel");
  var content = document.getElementById("detail-content");
  var text = "";

  if (domain === "sw" && typeof swNodeMetadata !== "undefined" && swNodeMetadata[nodeId]) {
    var meta = swNodeMetadata[nodeId];
    text += "Operation: " + (meta.op || "unknown") + "\n";
    if (meta.types) text += "Types:     " + meta.types + "\n";
    if (meta.loc) text += "Source:    " + meta.loc + "\n";
    if (meta.hwTarget) text += "Mapped to: " + meta.hwTarget + "\n";
    if (meta.attrs) {
      Object.keys(meta.attrs).forEach(function(k) { text += k + ": " + meta.attrs[k] + "\n"; });
    }
  } else if (domain === "hw" && typeof hwNodeMetadata !== "undefined" && hwNodeMetadata[nodeId]) {
    var hmeta = hwNodeMetadata[nodeId];
    text += "Name:      " + (hmeta.name || nodeId) + "\n";
    text += "Type:      " + (hmeta.type || "unknown") + "\n";
    if (hmeta.body_ops && hmeta.body_ops.length > 0)
      text += "Body:      " + hmeta.body_ops.join(", ") + "\n";
    if (hmeta.ports)
      text += "Ports:     " + hmeta.ports.in + " in, " + hmeta.ports.out + " out\n";
    if (hmeta.mappedSw && hmeta.mappedSw.length > 0)
      text += "Mapped SW: " + hmeta.mappedSw.join(", ") + "\n";
    if (mappingData && mappingData.temporal) {
      var tEntries = [];
      Object.keys(mappingData.temporal).forEach(function(swId) {
        var t = mappingData.temporal[swId];
        if (t.container === nodeId) tEntries.push({ swId: swId, info: t });
      });
      if (tEntries.length > 0) {
        text += "\nTemporal Assignments:\n";
        tEntries.forEach(function(te) {
          var swMeta = swNodeMetadata ? swNodeMetadata[te.swId] : null;
          var opName = swMeta ? swMeta.op : te.swId;
          text += "  " + (te.info.fuName || "FU") + ": " + opName;
          text += " [slot " + te.info.slot + ", tag " + te.info.tag + "]\n";
        });
      }
    }
  } else {
    text = "No metadata for " + nodeId;
  }

  content.textContent = text;
  panel.classList.add("visible");
  selectedNode = nodeId;
}

function closeDetail() {
  document.getElementById("detail-panel").classList.remove("visible");
  selectedNode = null;
}

// --- Mode Toggle ---
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
  restoreBtn.style.display = "none";

  document.querySelectorAll("#mode-buttons button").forEach(function(b) {
    b.classList.remove("active");
  });

  if (mode === "sidebyside") {
    document.getElementById("btn-sidebyside").classList.add("active");
    adgPanel.style.flex = "0 0 55%";
    dfgPanel.style.flex = "0 0 45%";
  } else if (mode === "overlay") {
    document.getElementById("btn-overlay").classList.add("active");
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
    restoreBtn.style.display = "inline-block";
  }
}

// --- Overlay Mode ---
function renderOverlay() {
  if (!mappingData || !adgSvgG) return;
  savedNodeFills = {};
  adgSvgG.selectAll(".overlay-group").remove();
  var overlayG = adgSvgG.append("g").attr("class", "overlay-group");

  // Build active routing nodes set
  var activeRoutingNodes = {};
  if (mappingData.routes) {
    Object.keys(mappingData.routes).forEach(function(edgeId) {
      var route = mappingData.routes[edgeId];
      if (!route.hwPath || route.hwPath.length < 2) return;
      for (var hi = 0; hi < route.hwPath.length - 1; hi++) {
        var hop = route.hwPath[hi];
        if (!hop.hwEdgeId) continue;
        var eInfo = hwEdgeMap[hop.hwEdgeId];
        if (eInfo) activeRoutingNodes[eInfo.dstNode] = true;
      }
    });
  }

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
      var firstSw = swNodeMetadata ? swNodeMetadata[swIds[0]] : null;
      var dialectColor = firstSw ? getDialectColor(firstSw.op) : null;
      shapeEl.attr("fill", dialectColor || "#4CAF50");
      shapeEl.attr("stroke-width", "3");
      shapeEl.attr("stroke", "#222");
    } else if (n.class !== "boundary") {
      if (activeRoutingNodes[n.id]) {
        nodeEl.classed("active-routing-node", true);
      } else {
        shapeEl.classed("unmapped-node", true);
      }
    }
  });

  adgSvgG.selectAll(".adg-edge").classed("base-edge-overlay", true);

  if (mappingData.routes) {
    var sortedEdges = Object.keys(mappingData.routes).sort(function(a, b) {
      var aNum = parseInt(a.replace("sw_", "").replace("swedge_", ""));
      var bNum = parseInt(b.replace("sw_", "").replace("swedge_", ""));
      return aNum - bNum;
    });
    sortedEdges.forEach(function(edgeId, idx) {
      var route = mappingData.routes[edgeId];
      if (!route.hwPath || route.hwPath.length === 0) return;
      var color = route.color || ROUTE_PALETTE[idx % ROUTE_PALETTE.length];
      route.hwPath.forEach(function(hop) {
        if (!hop.hwEdgeId) return;
        var origEdge = d3.select("#" + CSS.escape(hop.hwEdgeId));
        if (origEdge.empty()) return;
        var pathData = origEdge.attr("d");
        if (pathData) {
          var edgeInfo = hwEdgeMap[hop.hwEdgeId];
          var sw = 3;
          if (edgeInfo)
            sw = strokeWidthForBits(edgeInfo.valueBitWidth || edgeInfo.bitWidth || 32,
                                     edgeInfo.edgeType === "tagged");
          overlayG.append("path")
            .attr("class", "route-overlay").attr("d", pathData)
            .attr("stroke", color).attr("stroke-width", sw)
            .on("click", function(event) {
              event.stopPropagation();
              clearRouteHighlight();
              d3.selectAll(".route-overlay[stroke='" + color + "']")
                .classed("route-trace", true);
            });
        }
      });
    });
  }

  // Draw crossbar lines for switch/temporal_sw internal connections
  if (mappingData.routes && savedPortPos) {
    var crossbarG = overlayG.append("g").attr("class", "crossbar-lines");
    var crossbarSeen = {};
    var cbEdges = Object.keys(mappingData.routes).sort(function(a, b) {
      var aNum = parseInt(a.replace("sw_", "").replace("swedge_", ""));
      var bNum = parseInt(b.replace("sw_", "").replace("swedge_", ""));
      return aNum - bNum;
    });
    cbEdges.forEach(function(edgeId, idx) {
      var route = mappingData.routes[edgeId];
      if (!route.hwPath || route.hwPath.length < 2) return;
      var color = route.color || ROUTE_PALETTE[idx % ROUTE_PALETTE.length];
      for (var hi = 0; hi < route.hwPath.length - 1; hi++) {
        var hop = route.hwPath[hi];
        var nextHop = route.hwPath[hi + 1];
        if (!hop.hwEdgeId) continue;
        var eInfo = hwEdgeMap[hop.hwEdgeId];
        if (!eInfo) continue;
        var midNode = eInfo.dstNode;
        var midNodeInfo = hwNodeMap[midNode];
        if (!midNodeInfo) continue;
        var midType = midNodeInfo.type;
        if (midType !== "fabric.switch" && midType !== "fabric.temporal_sw") continue;
        var inPort = hop.dst;
        var outPort = nextHop.src;
        if (!inPort || !outPort) continue;
        if (savedPortInfo && savedPortInfo[inPort] && savedPortInfo[outPort]) {
          if (savedPortInfo[inPort].nodeId !== savedPortInfo[outPort].nodeId) continue;
        }
        var cbKey = midNode + ":" + inPort + ":" + outPort;
        if (crossbarSeen[cbKey]) continue;
        crossbarSeen[cbKey] = true;
        var inPos = savedPortPos[inPort];
        var outPos = savedPortPos[outPort];
        if (!inPos || !outPos) continue;
        crossbarG.append("line")
          .attr("class", "crossbar-line")
          .attr("x1", inPos.x).attr("y1", inPos.y)
          .attr("x2", outPos.x).attr("y2", outPos.y)
          .attr("stroke", color)
          .attr("stroke-width", 1.5);
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
    nodeEl.classed("active-routing-node", false);
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

// --- Panel Divider Drag ---
function setupDivider() {
  var divider = document.getElementById("panel-divider");
  var graphArea = document.getElementById("graph-area");
  var adgPanel = document.getElementById("panel-adg");
  var dfgPanel = document.getElementById("panel-dfg");
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

// --- Keyboard Shortcuts ---
function setupKeyboard() {
  document.addEventListener("keydown", function(e) {
    if (e.key === "Escape") { closeDetail(); return; }
    if (e.key === "1") { setMode("maximize-adg"); return; }
    if (e.key === "2") { setMode("maximize-dfg"); return; }
    if (e.key === "0") { setMode("sidebyside"); return; }
  });
}

// --- Initialize ---
function init() {
  document.getElementById("btn-sidebyside").addEventListener("click", function() {
    setMode("sidebyside");
  });
  document.getElementById("btn-overlay").addEventListener("click", function() {
    setMode("overlay");
  });
  document.getElementById("btn-fit").addEventListener("click", function() {
    fitPanel("adg"); fitPanel("dfg");
  });
  document.getElementById("btn-restore").addEventListener("click", function() {
    setMode("sidebyside");
  });
  document.getElementById("detail-close").addEventListener("click", function() {
    closeDetail();
  });

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

// --- Trace Playback Module ---

function initTracePlayback() {
  if (typeof traceData === "undefined" || !traceData || traceData === "null")
    return;

  var cycleEvents = traceData.cycleEvents || {};
  var totalCycles = traceData.totalCycles || 0;
  var configCycles = traceData.configCycles || 0;

  var slider = document.getElementById("trace-slider");
  var cycleLabel = document.getElementById("trace-cycle");
  var playBtn = document.getElementById("trace-play");
  var stepBack = document.getElementById("trace-step-back");
  var stepFwd = document.getElementById("trace-step-fwd");
  var speedSelect = document.getElementById("trace-speed");

  if (!slider || !playBtn) return;

  var currentCycle = 0;
  var playing = false;
  var playTimer = null;
  var speed = 5;

  var TRACE_CLASSES = ["trace-fire", "trace-stall-in", "trace-stall-out",
                       "trace-route", "trace-config"];
  var EDGE_TRACE_CLASS = "trace-route-edge";

  // Event kind to CSS class mapping.
  var KIND_CLASS = {
    0: "trace-fire",
    1: "trace-stall-in",
    2: "trace-stall-out",
    3: "trace-route",
    4: "trace-config"
  };

  // Build port-identity-based edge lookups for route highlighting.
  // portToEdge[dstPortId] = edge element ID (maps a destination port to its edge).
  // nodeInputPortIds[nodeId] = ordered array of input port IDs from the ADG node.
  var portToEdge = {};
  var nodeInputPortIds = {};
  if (typeof adgGraph !== "undefined" && adgGraph) {
    adgGraph.edges.forEach(function(e) {
      if (e.dstPort) portToEdge[e.dstPort] = e.id;
    });
    adgGraph.nodes.forEach(function(n) {
      if (n.ports && n.ports.inputIds) {
        nodeInputPortIds[n.id] = n.ports.inputIds;
      }
    });
  }

  // Heatmap state.
  var heatmapActive = false;
  var nodeUtilization = traceData.nodeUtilization || {};
  var heatmapBtn = document.getElementById("trace-heatmap");

  function utilToColor(u) {
    // Green (high util) -> Yellow (mid) -> Red (low util).
    var r, g, b;
    if (u >= 0.5) {
      var t = (u - 0.5) * 2;
      r = Math.round(255 * (1 - t));
      g = 200;
      b = Math.round(50 * (1 - t));
    } else {
      var t = u * 2;
      r = 255;
      g = Math.round(200 * t);
      b = 0;
    }
    return "rgb(" + r + "," + g + "," + b + ")";
  }

  function applyHeatmap() {
    if (!heatmapActive) return;
    var keys = Object.keys(nodeUtilization);
    for (var k = 0; k < keys.length; k++) {
      var nid = keys[k];
      var nodeGroup = document.getElementById("hw_" + nid);
      if (!nodeGroup) continue;
      var rect = nodeGroup.querySelector("rect");
      if (!rect) continue;
      if (!savedNodeFills["hm_" + nid])
        savedNodeFills["hm_" + nid] = rect.getAttribute("fill");
      rect.setAttribute("fill", utilToColor(nodeUtilization[nid]));
      rect.setAttribute("opacity", "0.85");
    }
  }

  function clearHeatmap() {
    var keys = Object.keys(nodeUtilization);
    for (var k = 0; k < keys.length; k++) {
      var nid = keys[k];
      var nodeGroup = document.getElementById("hw_" + nid);
      if (!nodeGroup) continue;
      var rect = nodeGroup.querySelector("rect");
      if (!rect) continue;
      var saved = savedNodeFills["hm_" + nid];
      if (saved) {
        rect.setAttribute("fill", saved);
        rect.removeAttribute("opacity");
      }
    }
  }

  if (heatmapBtn) {
    heatmapBtn.addEventListener("click", function() {
      heatmapActive = !heatmapActive;
      heatmapBtn.classList.toggle("active", heatmapActive);
      if (heatmapActive) applyHeatmap();
      else clearHeatmap();
    });
  }

  function clearHighlights() {
    TRACE_CLASSES.forEach(function(cls) {
      var elems = document.querySelectorAll("." + cls);
      for (var i = 0; i < elems.length; i++)
        elems[i].classList.remove(cls);
    });
    // Clear edge route highlights.
    var edgeHls = document.querySelectorAll("." + EDGE_TRACE_CLASS);
    for (var i = 0; i < edgeHls.length; i++)
      edgeHls[i].classList.remove(EDGE_TRACE_CLASS);
  }

  function showCycle(cycle) {
    currentCycle = Math.max(0, Math.min(cycle, totalCycles));
    slider.value = currentCycle;
    cycleLabel.textContent = "Cycle: " + currentCycle + " / " + totalCycles;

    clearHighlights();

    var events = cycleEvents[String(currentCycle)];
    if (!events) return;

    for (var i = 0; i < events.length; i++) {
      var hwNodeId = events[i][0];
      var kind = events[i][1];
      var cls = KIND_CLASS[kind];
      if (!cls) continue;

      // Find the ADG node element by hw_ id.
      var nodeId = "hw_" + hwNodeId;
      var nodeGroup = document.getElementById(nodeId);
      if (nodeGroup) {
        // Highlight the rect inside the node group.
        var rect = nodeGroup.querySelector("rect");
        if (rect) rect.classList.add(cls);
      }

      // For EV_ROUTE_USE, highlight the specific routed input edge.
      // arg0 = input port index on the switch node. Look up via port identity.
      if (kind === 3) {
        var arg0 = events[i][2] || 0;
        var portIds = nodeInputPortIds[nodeId];
        if (portIds && arg0 < portIds.length) {
          var edgeId = portToEdge[portIds[arg0]];
          if (edgeId) {
            var edgeEl = document.getElementById(edgeId);
            if (edgeEl) edgeEl.classList.add(EDGE_TRACE_CLASS);
          }
        }
      }
    }
  }

  slider.addEventListener("input", function() {
    showCycle(parseInt(slider.value, 10));
  });

  stepBack.addEventListener("click", function() {
    showCycle(currentCycle - 1);
  });

  stepFwd.addEventListener("click", function() {
    showCycle(currentCycle + 1);
  });

  speedSelect.addEventListener("change", function() {
    speed = parseInt(speedSelect.value, 10);
    if (playing) {
      clearInterval(playTimer);
      playTimer = setInterval(tick, Math.max(16, Math.floor(1000 / (60 * speed))));
    }
  });

  function tick() {
    if (currentCycle >= totalCycles) {
      stopPlay();
      return;
    }
    showCycle(currentCycle + speed);
  }

  function startPlay() {
    playing = true;
    playBtn.textContent = "Pause";
    playTimer = setInterval(tick, Math.max(16, Math.floor(1000 / (60 * speed))));
  }

  function stopPlay() {
    playing = false;
    playBtn.textContent = "Play";
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
  }

  playBtn.addEventListener("click", function() {
    if (playing) stopPlay();
    else startPlay();
  });

  // Keyboard shortcuts for trace.
  document.addEventListener("keydown", function(e) {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === " " && document.getElementById("trace-toolbar")) {
      e.preventDefault();
      if (playing) stopPlay();
      else startPlay();
    }
    if (e.key === "ArrowLeft" && document.getElementById("trace-toolbar")) {
      e.preventDefault();
      showCycle(currentCycle - 1);
    }
    if (e.key === "ArrowRight" && document.getElementById("trace-toolbar")) {
      e.preventDefault();
      showCycle(currentCycle + 1);
    }
  });

  showCycle(0);
}

if (document.readyState === "loading")
  document.addEventListener("DOMContentLoaded", init);
else init();

})();
