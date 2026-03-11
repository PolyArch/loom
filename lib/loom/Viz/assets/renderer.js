// Loom Visualization Renderer
// Dual-panel viewer: D3.js ADG grid + Graphviz DFG
//
// Expects globals: adgGraph, dfgDot, mappingData, swNodeMetadata, hwNodeMetadata

(function() {
"use strict";

// --- Constants ---
var CELL_SIZE = 120;
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

// Build lookup maps from adgGraph for fast access
var hwNodeMap = {};
var hwEdgeMap = {};
if (typeof adgGraph !== "undefined") {
  adgGraph.nodes.forEach(function(n) { hwNodeMap[n.id] = n; });
  adgGraph.edges.forEach(function(e) { hwEdgeMap[e.id] = e; });
}

// --- Utility ---
function strokeWidthForBits(bits, isTagged) {
  var w;
  if (bits <= 1)       w = 1.0;
  else if (bits <= 16) w = 1.5;
  else if (bits <= 32) w = 2.0;
  else if (bits <= 64) w = 3.0;
  else                 w = 4.0;
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
  if (node.class === "boundary") {
    return node.name && node.name.indexOf("out") >= 0 ? "output" : "input";
  }
  return "fabric.pe";
}

function nodeColorScheme(node) {
  var key = nodeTypeKey(node);
  return NODE_COLORS[key] || NODE_COLORS["fabric.pe"];
}

// --- Sugiyama Fallback Layout ---
function sugiyamaLayout(nodes, edges) {
  if (nodes.length === 0) return;

  // Build adjacency
  var adj = {}, radj = {};
  nodes.forEach(function(n) { adj[n.id] = []; radj[n.id] = []; });
  edges.forEach(function(e) {
    if (adj[e.srcNode] && adj[e.dstNode]) {
      adj[e.srcNode].push(e.dstNode);
      radj[e.dstNode].push(e.srcNode);
    }
  });

  // Phase 1: cycle removal via DFS back-edge reversal
  var visited = {}, inStack = {}, reversed = {};
  function dfs(u) {
    visited[u] = true;
    inStack[u] = true;
    (adj[u] || []).forEach(function(v) {
      if (inStack[v]) {
        reversed[u + "->" + v] = true;
      } else if (!visited[v]) {
        dfs(v);
      }
    });
    inStack[u] = false;
  }
  nodes.forEach(function(n) { if (!visited[n.id]) dfs(n.id); });

  // Build DAG adjacency (with reversed edges flipped)
  var dagAdj = {}, dagRadj = {};
  nodes.forEach(function(n) { dagAdj[n.id] = []; dagRadj[n.id] = []; });
  edges.forEach(function(e) {
    if (!dagAdj[e.srcNode] || !dagAdj[e.dstNode]) return;
    if (reversed[e.srcNode + "->" + e.dstNode]) {
      dagAdj[e.dstNode].push(e.srcNode);
      dagRadj[e.srcNode].push(e.dstNode);
    } else {
      dagAdj[e.srcNode].push(e.dstNode);
      dagRadj[e.dstNode].push(e.srcNode);
    }
  });

  // Phase 2: longest-path layer assignment
  var layer = {};
  var topo = [];
  var visited2 = {};
  function topoSort(u) {
    visited2[u] = true;
    (dagAdj[u] || []).forEach(function(v) {
      if (!visited2[v]) topoSort(v);
    });
    topo.push(u);
  }
  nodes.forEach(function(n) { if (!visited2[n.id]) topoSort(n.id); });
  topo.reverse();

  topo.forEach(function(u) {
    var preds = dagRadj[u] || [];
    if (preds.length === 0) {
      layer[u] = 0;
    } else {
      var maxL = 0;
      preds.forEach(function(p) { maxL = Math.max(maxL, (layer[p] || 0) + 1); });
      layer[u] = maxL;
    }
  });

  // Phase 3: barycenter crossing minimization
  var maxLayer = 0;
  nodes.forEach(function(n) { maxLayer = Math.max(maxLayer, layer[n.id] || 0); });

  var layers = [];
  for (var li = 0; li <= maxLayer; li++) layers.push([]);
  nodes.forEach(function(n) { layers[layer[n.id] || 0].push(n.id); });

  var pos = {};
  layers.forEach(function(l) {
    l.forEach(function(nid, idx) { pos[nid] = idx; });
  });

  // Alternating sweeps
  for (var sweep = 0; sweep < 8; sweep++) {
    if (sweep % 2 === 0) {
      // top-down
      for (var lj = 1; lj <= maxLayer; lj++) {
        layers[lj].forEach(function(nid) {
          var preds = dagRadj[nid] || [];
          if (preds.length > 0) {
            var sum = 0;
            preds.forEach(function(p) { sum += (pos[p] || 0); });
            pos[nid] = sum / preds.length;
          }
        });
        layers[lj].sort(function(a, b) { return (pos[a] || 0) - (pos[b] || 0); });
        layers[lj].forEach(function(nid, idx) { pos[nid] = idx; });
      }
    } else {
      // bottom-up
      for (var lk = maxLayer - 1; lk >= 0; lk--) {
        layers[lk].forEach(function(nid) {
          var succs = dagAdj[nid] || [];
          if (succs.length > 0) {
            var sum = 0;
            succs.forEach(function(s) { sum += (pos[s] || 0); });
            pos[nid] = sum / succs.length;
          }
        });
        layers[lk].sort(function(a, b) { return (pos[a] || 0) - (pos[b] || 0); });
        layers[lk].forEach(function(nid, idx) { pos[nid] = idx; });
      }
    }
  }

  // Phase 4: assign grid coordinates
  nodes.forEach(function(n) {
    if (n.gridRow === null || n.gridRow === undefined) {
      n.gridRow = (layer[n.id] || 0) * 2;
    }
    if (n.gridCol === null || n.gridCol === undefined) {
      n.gridCol = (pos[n.id] || 0) * 2;
    }
  });
}

// --- ADG Rendering ---
function renderADG() {
  if (typeof adgGraph === "undefined" || !adgGraph.nodes) return;

  var svg = d3.select("#svg-adg");
  var width = svg.node().clientWidth || 800;
  var height = svg.node().clientHeight || 600;

  // Check if fallback layout needed
  var nonSentinel = adgGraph.nodes.filter(function(n) { return n.class !== "boundary"; });
  var hasCoords = nonSentinel.filter(function(n) {
    return n.gridCol !== null && n.gridCol !== undefined &&
           n.gridRow !== null && n.gridRow !== undefined;
  });

  if (hasCoords.length < nonSentinel.length) {
    var threshold = nonSentinel.length * 0.5;
    if (hasCoords.length < threshold) {
      // Re-layout entire graph
      nonSentinel.forEach(function(n) { n.gridCol = null; n.gridRow = null; });
    }
    sugiyamaLayout(
      nonSentinel.filter(function(n) {
        return n.gridCol === null || n.gridCol === undefined;
      }),
      adgGraph.edges
    );
  }

  // Place I/O sentinels at boundaries
  var maxRow = 0, maxCol = 0;
  adgGraph.nodes.forEach(function(n) {
    if (n.gridRow !== null && n.gridRow !== undefined) maxRow = Math.max(maxRow, n.gridRow + (n.areaH || 1));
    if (n.gridCol !== null && n.gridCol !== undefined) maxCol = Math.max(maxCol, n.gridCol + (n.areaW || 1));
  });

  var ioIdx = 0;
  adgGraph.nodes.forEach(function(n) {
    if (n.class === "boundary" && (n.gridCol === null || n.gridCol === undefined)) {
      var isOutput = n.name && n.name.indexOf("out") >= 0;
      n.gridRow = isOutput ? maxRow + 1 : -1;
      n.gridCol = ioIdx;
      ioIdx++;
    }
  });

  // Create node position lookup
  var nodePos = {};
  adgGraph.nodes.forEach(function(n) {
    var w = (n.areaW || 1);
    var h = (n.areaH || 1);
    nodePos[n.id] = {
      x: ((n.gridCol || 0) + w / 2) * CELL_SIZE,
      y: ((n.gridRow || 0) + h / 2) * CELL_SIZE,
      w: w * CELL_SIZE,
      h: h * CELL_SIZE
    };
  });

  // Setup zoom
  var g = svg.append("g");
  adgSvgG = g;

  adgZoom = d3.zoom()
    .scaleExtent([0.05, 5])
    .on("zoom", function(event) { g.attr("transform", event.transform); });
  svg.call(adgZoom);

  // Grid lines
  var gridG = g.append("g").attr("class", "grid");
  for (var gr = -1; gr <= maxRow + 2; gr++) {
    gridG.append("line").attr("class", "grid-line")
      .attr("x1", -CELL_SIZE).attr("y1", gr * CELL_SIZE)
      .attr("x2", (maxCol + 2) * CELL_SIZE).attr("y2", gr * CELL_SIZE);
  }
  for (var gc = -1; gc <= maxCol + 2; gc++) {
    gridG.append("line").attr("class", "grid-line")
      .attr("x1", gc * CELL_SIZE).attr("y1", -CELL_SIZE)
      .attr("x2", gc * CELL_SIZE).attr("y2", (maxRow + 2) * CELL_SIZE);
  }

  // Draw edges
  var edgeG = g.append("g").attr("class", "edges");
  adgGraph.edges.forEach(function(e) {
    var srcPos = nodePos[e.srcNode];
    var dstPos = nodePos[e.dstNode];
    if (!srcPos || !dstPos) return;

    var sw = strokeWidthForBits(e.valueBitWidth || e.bitWidth || 32,
                                 e.edgeType === "tagged");
    var color = "#333333";
    var dasharray = null;
    if (e.edgeType === "tagged") { color = "#7b2d8b"; dasharray = "6,3"; }
    else if (e.edgeType === "memref") { color = "#2255cc"; dasharray = "2,3"; }
    else if (e.edgeType === "control") { color = "#999999"; dasharray = "4,2"; }

    // Manhattan routing for non-adjacent nodes
    var dx = Math.abs(srcPos.x - dstPos.x);
    var dy = Math.abs(srcPos.y - dstPos.y);
    var path;

    if (dx < CELL_SIZE * 1.5 && dy < CELL_SIZE * 1.5) {
      // Adjacent: straight line
      path = "M" + srcPos.x + "," + srcPos.y + "L" + dstPos.x + "," + dstPos.y;
    } else {
      // Manhattan routing: go vertical first, then horizontal
      var midY = (srcPos.y + dstPos.y) / 2;
      path = "M" + srcPos.x + "," + srcPos.y +
             "L" + srcPos.x + "," + midY +
             "L" + dstPos.x + "," + midY +
             "L" + dstPos.x + "," + dstPos.y;
    }

    var edgePath = edgeG.append("path")
      .attr("class", "adg-edge")
      .attr("id", e.id)
      .attr("d", path)
      .attr("stroke", color)
      .attr("stroke-width", sw);

    if (dasharray) edgePath.attr("stroke-dasharray", dasharray);
  });

  // Draw nodes
  var nodeG = g.append("g").attr("class", "nodes");
  adgGraph.nodes.forEach(function(n) {
    var p = nodePos[n.id];
    if (!p) return;

    var colors = nodeColorScheme(n);
    var ng = nodeG.append("g")
      .attr("class", "adg-node")
      .attr("id", n.id)
      .attr("transform", "translate(" + p.x + "," + p.y + ")");

    var halfW = p.w / 2 - 4;
    var halfH = p.h / 2 - 4;
    var key = nodeTypeKey(n);

    if (key === "fabric.switch" || key === "fabric.temporal_sw") {
      // Diamond shape
      var dSize = Math.min(halfW, halfH);
      ng.append("polygon")
        .attr("points", "0," + (-dSize) + " " + dSize + ",0 0," + dSize + " " + (-dSize) + ",0")
        .attr("fill", colors.fill)
        .attr("stroke", colors.border)
        .attr("stroke-width", 2);
    } else if (key === "fabric.fifo" || key === "fabric.add_tag" ||
               key === "fabric.map_tag" || key === "fabric.del_tag") {
      // Small circle
      ng.append("circle")
        .attr("r", Math.min(halfW, halfH))
        .attr("fill", colors.fill)
        .attr("stroke", colors.border)
        .attr("stroke-width", 1.5);
    } else {
      // Rounded rectangle (PE, temporal_pe, memory, etc.)
      ng.append("rect")
        .attr("x", -halfW).attr("y", -halfH)
        .attr("width", halfW * 2).attr("height", halfH * 2)
        .attr("rx", 6).attr("ry", 6)
        .attr("fill", colors.fill)
        .attr("stroke", colors.border)
        .attr("stroke-width", 2);
    }

    // Label
    var label = n.name || n.id;
    if (label.length > 16) label = label.substring(0, 14) + "..";
    ng.append("text")
      .attr("dy", "0.35em")
      .attr("text-anchor", "middle")
      .attr("fill", colors.text)
      .text(label);

    // Interaction
    ng.on("mouseenter", function() { highlightNode(n.id, "hw"); })
      .on("mouseleave", function() { clearHighlights(); })
      .on("click", function(event) { event.stopPropagation(); showDetail(n.id, "hw"); });
  });

  // Fit to view
  fitPanel("adg");

  // Background click to close detail
  svg.on("click", function() { closeDetail(); });
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

    // Insert rendered SVG into panel
    var container = document.getElementById("panel-dfg");
    var svgWrapper = document.createElement("div");
    svgWrapper.innerHTML = svgStr;
    var renderedSvg = svgWrapper.querySelector("svg");
    if (!renderedSvg) {
      d3.select("#status-bar").text("DFG render failed");
      return;
    }

    // Replace the placeholder SVG with the rendered one
    renderedSvg.setAttribute("id", "svg-dfg");
    renderedSvg.style.width = "100%";
    renderedSvg.style.height = "100%";
    var oldSvg = document.getElementById("svg-dfg");
    if (oldSvg) oldSvg.parentNode.replaceChild(renderedSvg, oldSvg);

    // Apply d3 zoom
    var dfgSvgSel = d3.select("#svg-dfg");
    var innerG = dfgSvgSel.select("g");
    dfgSvgG = innerG;

    dfgZoom = d3.zoom()
      .scaleExtent([0.1, 5])
      .on("zoom", function(event) {
        innerG.attr("transform", event.transform);
      });
    dfgSvgSel.call(dfgZoom);

    // Attach interaction handlers to DFG nodes
    dfgSvgSel.selectAll(".node").each(function() {
      var el = d3.select(this);
      var titleEl = el.select("title");
      var nodeTitle = titleEl.empty() ? "" : titleEl.text();

      // Try to find the id from our sw_ prefix convention
      var nodeId = el.attr("id");
      if (!nodeId && nodeTitle) {
        // Graphviz uses the node name as title; our DOT uses sw_<id> as node name
        nodeId = nodeTitle;
      }

      if (nodeId) {
        el.on("mouseenter", function() { highlightNode(nodeId, "sw"); })
          .on("mouseleave", function() { clearHighlights(); })
          .on("click", function(event) { event.stopPropagation(); showDetail(nodeId, "sw"); });
      }
    });

    // Edge interaction for route preview
    dfgSvgSel.selectAll(".edge").each(function() {
      var el = d3.select(this);
      var titleEl = el.select("title");
      var edgeTitle = titleEl.empty() ? "" : titleEl.text();
      var edgeId = el.attr("id") || edgeTitle;

      if (edgeId && mappingData && mappingData.routes && mappingData.routes[edgeId]) {
        el.style("cursor", "pointer");
        el.on("mouseenter", function() { highlightRoute(edgeId); })
          .on("mouseleave", function() { clearRouteHighlight(); });
      }
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
    svg = d3.select("#svg-adg");
    g = adgSvgG;
    zoom = adgZoom;
  } else {
    svg = d3.select("#svg-dfg");
    g = dfgSvgG;
    zoom = dfgZoom;
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
    (svgH - padding * 2) / bbox.height,
    2
  );
  var tx = svgW / 2 - (bbox.x + bbox.width / 2) * scale;
  var ty = svgH / 2 - (bbox.y + bbox.height / 2) * scale;

  svg.transition().duration(500)
    .call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}

// --- Cross-Highlighting ---
function highlightNode(nodeId, domain) {
  clearHighlights();

  // Highlight the hovered node
  var el = d3.select("#" + CSS.escape(nodeId));
  if (!el.empty()) {
    el.select("rect, polygon, circle, ellipse").classed("node-highlight", true);
  }

  if (!mappingData) return;

  // Cross-highlight
  if (domain === "sw" && mappingData.swToHw) {
    var hwId = mappingData.swToHw[nodeId];
    if (hwId) {
      var hwEl = d3.select("#" + CSS.escape(hwId));
      if (!hwEl.empty()) {
        hwEl.select("rect, polygon, circle").classed("cross-highlight", true);
        scrollToElement(hwEl, "adg");
      }
    }
  } else if (domain === "hw" && mappingData.hwToSw) {
    var swIds = mappingData.hwToSw[nodeId];
    if (swIds) {
      swIds.forEach(function(swId) {
        var swEl = d3.select("#" + CSS.escape(swId));
        if (!swEl.empty()) {
          swEl.select("polygon, ellipse, rect, path").classed("cross-highlight", true);
        }
      });
      if (swIds.length > 0) scrollToElement(d3.select("#" + CSS.escape(swIds[0])), "dfg");
    }
  }
}

function clearHighlights() {
  d3.selectAll(".node-highlight").classed("node-highlight", false);
  d3.selectAll(".cross-highlight").classed("cross-highlight", false);
}

function highlightRoute(edgeId) {
  if (!mappingData || !mappingData.routes) return;
  var route = mappingData.routes[edgeId];
  if (!route || !route.hwPath) return;

  route.hwPath.forEach(function(hop) {
    if (hop.hwEdgeId) {
      var edgeEl = d3.select("#" + CSS.escape(hop.hwEdgeId));
      if (!edgeEl.empty()) {
        edgeEl.classed("route-trace", true);
      }
    }
  });
}

function clearRouteHighlight() {
  d3.selectAll(".route-trace").classed("route-trace", false);
}

function scrollToElement(el, panel) {
  // Smooth scroll the panel to center the target element
  if (el.empty()) return;
  var node = el.node();
  if (!node) return;
  var bbox = node.getBBox();

  var svg, zoom;
  if (panel === "adg") {
    svg = d3.select("#svg-adg");
    zoom = adgZoom;
  } else {
    svg = d3.select("#svg-dfg");
    zoom = dfgZoom;
  }
  if (!svg || !zoom) return;

  // Get current transform and check if element is visible
  var svgNode = svg.node();
  var svgW = svgNode.clientWidth;
  var svgH = svgNode.clientHeight;
  var currentTransform = d3.zoomTransform(svgNode);

  var cx = bbox.x + bbox.width / 2;
  var cy = bbox.y + bbox.height / 2;
  var screenX = currentTransform.applyX(cx);
  var screenY = currentTransform.applyY(cy);

  // Only scroll if element is off-screen
  if (screenX < 0 || screenX > svgW || screenY < 0 || screenY > svgH) {
    var newTx = svgW / 2 - cx * currentTransform.k;
    var newTy = svgH / 2 - cy * currentTransform.k;
    svg.transition().duration(400)
      .call(zoom.transform,
            d3.zoomIdentity.translate(newTx, newTy).scale(currentTransform.k));
  }
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
      Object.keys(meta.attrs).forEach(function(k) {
        text += k + ": " + meta.attrs[k] + "\n";
      });
    }
  } else if (domain === "hw" && typeof hwNodeMetadata !== "undefined" && hwNodeMetadata[nodeId]) {
    var hmeta = hwNodeMetadata[nodeId];
    text += "Name:      " + (hmeta.name || nodeId) + "\n";
    text += "Type:      " + (hmeta.type || "unknown") + "\n";
    if (hmeta.body_ops && hmeta.body_ops.length > 0) {
      text += "Body:      " + hmeta.body_ops.join(", ") + "\n";
    }
    if (hmeta.ports) {
      text += "Ports:     " + hmeta.ports.in + " in, " + hmeta.ports.out + " out\n";
    }
    if (hmeta.mappedSw && hmeta.mappedSw.length > 0) {
      text += "Mapped SW: " + hmeta.mappedSw.join(", ") + "\n";
    }

    // Temporal PE detail
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
  currentMode = mode;
  var adgPanel = document.getElementById("panel-adg");
  var dfgPanel = document.getElementById("panel-dfg");
  var divider = document.getElementById("panel-divider");
  var restoreBtn = document.getElementById("btn-restore");

  // Reset classes
  adgPanel.classList.remove("panel-maximized", "panel-hidden");
  dfgPanel.classList.remove("panel-maximized", "panel-hidden");
  divider.classList.remove("divider-hidden");
  restoreBtn.style.display = "none";

  // Update button states
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
  } else if (mode === "maximize-adg") {
    adgPanel.classList.add("panel-maximized");
    dfgPanel.classList.add("panel-hidden");
    divider.classList.add("divider-hidden");
    restoreBtn.style.display = "inline-block";
  } else if (mode === "maximize-dfg") {
    dfgPanel.classList.add("panel-maximized");
    adgPanel.classList.add("panel-hidden");
    divider.classList.add("divider-hidden");
    restoreBtn.style.display = "inline-block";
  }
}

// --- Overlay Mode ---
function renderOverlay() {
  if (!mappingData || !adgSvgG) return;

  // Remove previous overlay
  adgSvgG.selectAll(".overlay-group").remove();
  var overlayG = adgSvgG.append("g").attr("class", "overlay-group");

  // Dim unmapped nodes, recolor mapped nodes
  adgGraph.nodes.forEach(function(n) {
    var nodeEl = d3.select("#" + CSS.escape(n.id));
    if (nodeEl.empty()) return;

    var swIds = mappingData.hwToSw ? mappingData.hwToSw[n.id] : null;
    if (swIds && swIds.length > 0) {
      // Get dialect color from first mapped SW op
      var firstSw = swNodeMetadata ? swNodeMetadata[swIds[0]] : null;
      var dialectColor = firstSw ? getDialectColor(firstSw.op) : null;
      if (dialectColor) {
        nodeEl.select("rect, polygon, circle").attr("fill", dialectColor);
      }
    } else if (n.class !== "boundary") {
      nodeEl.select("rect, polygon, circle").classed("unmapped-node", true);
    }
  });

  // Dim base edges
  adgSvgG.selectAll(".adg-edge").classed("base-edge-overlay", true);

  // Draw route overlays
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
          overlayG.append("path")
            .attr("class", "route-overlay")
            .attr("d", pathData)
            .attr("stroke", color)
            .attr("stroke-width", 3)
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
}

// --- Panel Divider Drag ---
function setupDivider() {
  var divider = document.getElementById("panel-divider");
  var graphArea = document.getElementById("graph-area");
  var adgPanel = document.getElementById("panel-adg");
  var dfgPanel = document.getElementById("panel-dfg");
  var dragging = false;

  divider.addEventListener("mousedown", function(e) {
    dragging = true;
    e.preventDefault();
  });

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
    if (e.key === "Escape") {
      closeDetail();
      return;
    }
    if (e.key === "1") { setMode("maximize-adg"); return; }
    if (e.key === "2") { setMode("maximize-dfg"); return; }
    if (e.key === "0") { setMode("sidebyside"); return; }
  });
}

// --- Initialize ---
function init() {
  // Mode buttons
  document.getElementById("btn-sidebyside").addEventListener("click", function() {
    setMode("sidebyside");
  });
  document.getElementById("btn-overlay").addEventListener("click", function() {
    setMode("overlay");
  });
  document.getElementById("btn-fit").addEventListener("click", function() {
    fitPanel("adg");
    fitPanel("dfg");
  });
  document.getElementById("btn-restore").addEventListener("click", function() {
    setMode("sidebyside");
  });
  document.getElementById("detail-close").addEventListener("click", function() {
    closeDetail();
  });

  // Panel header double-click to maximize
  document.querySelectorAll(".panel-header").forEach(function(header) {
    header.addEventListener("dblclick", function() {
      var panel = header.parentElement.id;
      if (currentMode === "sidebyside") {
        setMode(panel === "panel-adg" ? "maximize-adg" : "maximize-dfg");
      } else {
        setMode("sidebyside");
      }
    });
  });

  setupDivider();
  setupKeyboard();

  // Render
  renderADG();
  renderDFG();
}

// Start when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}

})();
