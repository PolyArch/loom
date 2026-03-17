// fcc Visualization Renderer - Clean rebuild
// Side-by-side: ADG (D3.js) + DFG (direct SVG)
// Expects globals: ADG_DATA, DFG_DATA, MAPPING_DATA (optional)
(function() {
"use strict";

// Ensure MAPPING_DATA exists
if (typeof MAPPING_DATA === "undefined") window.MAPPING_DATA = null;

// ============================================================
// Mapping state (cross-highlighting + overlay)
// ============================================================

var mappingIdx = null;       // Built from MAPPING_DATA
var adgCompBoxes = {};       // comp name -> {x,y,w,h}
var adgFUBoxes = {};         // "pe_name/fu_name" -> {x,y,w,h}
var fuPortPos = {};          // "pe_name/fu_name/in_0" -> {x, y}
var peIngressMap = {};       // pe name -> external viz key -> pe input viz key
var peEgressMap = {};        // pe name -> external viz key -> pe output viz key
var switchRouteIdx = null;   // switch route summaries from mapping data
var peRouteIdx = null;       // PE internal route summaries from mapping data
var adgRenderHints = null;   // complexity-aware ADG rendering hints
var currentMode = "sidebyside";
var mappingEnabled = false;  // Toggled by user; starts Off
var TYPE_LABEL_SINGLE_LINE_RATIO = 0.9;

function heapPush(heap, item) {
  heap.push(item);
  var idx = heap.length - 1;
  while (idx > 0) {
    var parent = Math.floor((idx - 1) / 2);
    if (heap[parent].f <= heap[idx].f) break;
    var tmp = heap[parent];
    heap[parent] = heap[idx];
    heap[idx] = tmp;
    idx = parent;
  }
}

function heapPop(heap) {
  if (heap.length === 0) return null;
  var top = heap[0];
  var tail = heap.pop();
  if (heap.length === 0) return top;
  heap[0] = tail;
  var idx = 0;
  while (true) {
    var left = idx * 2 + 1;
    var right = left + 1;
    var smallest = idx;
    if (left < heap.length && heap[left].f < heap[smallest].f) smallest = left;
    if (right < heap.length && heap[right].f < heap[smallest].f) smallest = right;
    if (smallest === idx) break;
    var tmp = heap[idx];
    heap[idx] = heap[smallest];
    heap[smallest] = tmp;
    idx = smallest;
  }
  return top;
}

function buildMappingIndex() {
  if (!MAPPING_DATA || !MAPPING_DATA.node_mappings) return null;
  var idx = {
    swToHw: {},
    hwToSw: {},
    fuToSw: {},
    swToMemArgs: {},
    memArgToSw: {},
    memArgToHw: {},
    hwToMemArgs: {}
  };
  MAPPING_DATA.node_mappings.forEach(function(m) {
    idx.swToHw[m.sw_node] = m;
    var compName = m.pe_name || m.hw_name;
    if (compName) {
      if (!idx.hwToSw[compName]) idx.hwToSw[compName] = [];
      idx.hwToSw[compName].push(m);
    }
    if (m.pe_name) {
      var fuKey = m.pe_name + "/" + m.hw_name;
      if (!idx.fuToSw[fuKey]) idx.fuToSw[fuKey] = [];
      idx.fuToSw[fuKey].push(m);
    }
  });

  (MAPPING_DATA.memory_regions || []).forEach(function(mem) {
    var hwName = mem.hw_name;
    if (!hwName) return;
    if (!idx.hwToMemArgs[hwName]) idx.hwToMemArgs[hwName] = [];
    (mem.regions || []).forEach(function(region) {
      var swNode = region.sw_node;
      var argIdx = region.memref_arg_index;
      if (swNode !== undefined) {
        if (!idx.swToMemArgs[swNode]) idx.swToMemArgs[swNode] = [];
        if (idx.swToMemArgs[swNode].indexOf(argIdx) < 0)
          idx.swToMemArgs[swNode].push(argIdx);
      }
      if (argIdx !== undefined) {
        if (!idx.memArgToSw[argIdx]) idx.memArgToSw[argIdx] = [];
        if (swNode !== undefined && idx.memArgToSw[argIdx].indexOf(swNode) < 0)
          idx.memArgToSw[argIdx].push(swNode);
        if (!idx.memArgToHw[argIdx]) idx.memArgToHw[argIdx] = [];
        if (idx.memArgToHw[argIdx].indexOf(hwName) < 0)
          idx.memArgToHw[argIdx].push(hwName);
        if (idx.hwToMemArgs[hwName].indexOf(argIdx) < 0)
          idx.hwToMemArgs[hwName].push(argIdx);
      }
    });
  });
  return idx;
}

function isSwitchPortKind(kind) {
  return kind === "sw" || kind === "temporal_sw";
}

function buildADGConnectionMaps() {
  peIngressMap = {};
  peEgressMap = {};
  if (!ADG_DATA || !ADG_DATA.connections) return;

  ADG_DATA.connections.forEach(function(conn) {
    if (conn.to && conn.to !== "module_out") {
      var fromKey = conn.from === "module_in"
        ? "module_in_" + conn.fromIdx
        : conn.from + "_out_" + conn.fromIdx;
      var peInKey = conn.to + "_in_" + conn.toIdx;
      if (!peIngressMap[conn.to]) peIngressMap[conn.to] = {};
      peIngressMap[conn.to][fromKey] = peInKey;
    }

    if (conn.from && conn.from !== "module_in") {
      var toKey = conn.to === "module_out"
        ? "module_out_" + conn.toIdx
        : conn.to + "_in_" + conn.toIdx;
      var peOutKey = conn.from + "_out_" + conn.fromIdx;
      if (!peEgressMap[conn.from]) peEgressMap[conn.from] = {};
      peEgressMap[conn.from][toKey] = peOutKey;
    }
  });
}

function getPEIngressPortKey(peName, externalVizKey) {
  return peIngressMap[peName] ? peIngressMap[peName][externalVizKey] : null;
}

function getPEEgressPortKey(peName, externalVizKey) {
  return peEgressMap[peName] ? peEgressMap[peName][externalVizKey] : null;
}

function deriveSwitchRoutesFromPaths() {
  if (!MAPPING_DATA || !MAPPING_DATA.edge_routings) return [];
  var routesByKey = {};

  MAPPING_DATA.edge_routings.forEach(function(er) {
    var path = er.path || [];
    for (var i = 1; i + 1 < path.length; i += 2) {
      var inInfo = portLookup[path[i]];
      var outInfo = portLookup[path[i + 1]];
      if (!inInfo || !outInfo) continue;
      if (!isSwitchPortKind(inInfo.kind) || !isSwitchPortKind(outInfo.kind)) continue;
      if (inInfo.component !== outInfo.component) continue;
      var key = inInfo.component + ":" + inInfo.index + ":" + outInfo.index;
      if (!routesByKey[key]) {
        routesByKey[key] = {
          component: inInfo.component,
          input_port_id: path[i],
          output_port_id: path[i + 1],
          input_port: inInfo.index,
          output_port: outInfo.index,
          sw_edges: []
        };
      }
      routesByKey[key].sw_edges.push(er.sw_edge);
    }
  });

  return Object.keys(routesByKey).sort().map(function(key) {
    return routesByKey[key];
  });
}

function buildSwitchRouteIndex() {
  var idx = { routes: [], byEdge: {}, byComponent: {} };
  if (!MAPPING_DATA) return idx;

  var routes = MAPPING_DATA.switch_routes || deriveSwitchRoutesFromPaths();
  routes.forEach(function(route) {
    var normalized = {
      component: route.component,
      input_port_id: route.input_port_id,
      output_port_id: route.output_port_id,
      input_port: route.input_port,
      output_port: route.output_port,
      sw_edges: (route.sw_edges || []).slice()
    };
    idx.routes.push(normalized);
    if (!idx.byComponent[normalized.component]) idx.byComponent[normalized.component] = [];
    idx.byComponent[normalized.component].push(normalized);
    normalized.sw_edges.forEach(function(swEdge) {
      if (!idx.byEdge[swEdge]) idx.byEdge[swEdge] = [];
      idx.byEdge[swEdge].push(normalized);
    });
  });
  return idx;
}

function buildPERouteIndex() {
  var idx = { routes: [], byEdge: {}, byPE: {} };
  if (!MAPPING_DATA || !MAPPING_DATA.edge_routings) return idx;

  var seen = {};
  MAPPING_DATA.edge_routings.forEach(function(er) {
    var path = er.path || [];
    for (var i = 0; i + 1 < path.length; i += 2) {
      var srcInfo = portLookup[path[i]];
      var dstInfo = portLookup[path[i + 1]];
      if (!srcInfo || !dstInfo) continue;

      if (dstInfo.kind === "fu" && srcInfo.kind !== "fu") {
        var peInKey = getPEIngressPortKey(dstInfo.pe, srcInfo.vizKey);
        if (!peInKey) continue;
        var inKey = er.sw_edge + ":in:" + peInKey + ":" + dstInfo.fuPortKey;
        if (seen[inKey]) continue;
        seen[inKey] = true;
        var inRoute = {
          swEdge: er.sw_edge,
          peName: dstInfo.pe,
          direction: "in",
          pePortKey: peInKey,
          fuPortKey: dstInfo.fuPortKey
        };
        idx.routes.push(inRoute);
        if (!idx.byEdge[er.sw_edge]) idx.byEdge[er.sw_edge] = [];
        idx.byEdge[er.sw_edge].push(inRoute);
        if (!idx.byPE[dstInfo.pe]) idx.byPE[dstInfo.pe] = [];
        idx.byPE[dstInfo.pe].push(inRoute);
      }

      if (srcInfo.kind === "fu" && dstInfo.kind !== "fu") {
        var peOutKey = getPEEgressPortKey(srcInfo.pe, dstInfo.vizKey);
        if (!peOutKey) continue;
        var outKey = er.sw_edge + ":out:" + peOutKey + ":" + srcInfo.fuPortKey;
        if (seen[outKey]) continue;
        seen[outKey] = true;
        var outRoute = {
          swEdge: er.sw_edge,
          peName: srcInfo.pe,
          direction: "out",
          pePortKey: peOutKey,
          fuPortKey: srcInfo.fuPortKey
        };
        idx.routes.push(outRoute);
        if (!idx.byEdge[er.sw_edge]) idx.byEdge[er.sw_edge] = [];
        idx.byEdge[er.sw_edge].push(outRoute);
        if (!idx.byPE[srcInfo.pe]) idx.byPE[srcInfo.pe] = [];
        idx.byPE[srcInfo.pe].push(outRoute);
      }
    }
  });

  return idx;
}

function splitDisplayLabel(text) {
  if (!text) return [""];
  if (text.indexOf("\n") >= 0) return text.split("\n");
  var dot = text.indexOf(".");
  if (dot >= 0) return [text.slice(0, dot), text.slice(dot + 1)];
  return [text];
}

function renderTypeLabel(g, x, y, width, text, opts) {
  opts = opts || {};
  var charPx = opts.charPx || 6;
  var fontSize = opts.fontSize || "9px";
  var fontWeight = opts.fontWeight || "600";
  var fill = opts.fill || "#fff";
  var lineGap = opts.lineGap || 10;
  var singleOffsetY = opts.singleOffsetY || 14;
  var multiOffsetY0 = opts.multiOffsetY0 || 12;
  var multiOffsetY1 = opts.multiOffsetY1 || (multiOffsetY0 + lineGap);
  var lines = splitDisplayLabel(text);
  var fitsSingle = (text.length * charPx) < (width * TYPE_LABEL_SINGLE_LINE_RATIO);

  if (fitsSingle || lines.length === 1) {
    g.append("text").attr("x", x).attr("y", y + singleOffsetY)
      .attr("fill", fill).attr("font-size", fontSize).attr("font-weight", fontWeight)
      .text(text);
    return;
  }

  g.append("text").attr("x", x).attr("y", y + multiOffsetY0)
    .attr("fill", fill).attr("font-size", fontSize).attr("font-weight", fontWeight)
    .text(lines[0]);
  g.append("text").attr("x", x).attr("y", y + multiOffsetY1)
    .attr("fill", fill).attr("font-size", fontSize).attr("font-weight", fontWeight)
    .text(lines.slice(1).join("."));
}

function shortenText(text, maxLen) {
  if (!text) return "";
  if (text.length <= maxLen) return text;
  return text.slice(0, Math.max(1, maxLen - 1)) + "\u2026";
}

function miniFUOpStyle(opName) {
  var style = {
    fill: "#1a3050",
    stroke: "#5dade2",
    text: "#c8d6e5",
    shape: "ellipse"
  };
  if (!opName) return style;
  if (opName.indexOf("static_mux") >= 0 || opName.indexOf("mux") >= 0) {
    style.fill = "#3a3520";
    style.stroke = "#ffd166";
    style.text = "#ffe29b";
    style.shape = "mux";
  } else if (opName.indexOf("arith.") === 0) {
    style.fill = "#1a3050";
    style.stroke = "#5dade2";
  } else if (opName.indexOf("dataflow.") === 0) {
    style.fill = "#203c2f";
    style.stroke = "#58d68d";
  }
  return style;
}

function analyzeFunctionUnit(fu) {
  var inputEdges = Array.isArray(fu && fu.inputEdges) ? fu.inputEdges : [];
  var dagEdges = Array.isArray(fu && fu.edges) ? fu.edges : [];
  var outputEdges = Array.isArray(fu && fu.outputEdges) ? fu.outputEdges : [];
  var opCount = Array.isArray(fu && fu.ops) ? fu.ops.length : 0;
  var indegree = [];
  var succ = [];
  var level = [];
  var rows = {};
  var maxLevel = 0;
  var processed = 0;
  var i;

  for (i = 0; i < opCount; i++) {
    indegree[i] = 0;
    succ[i] = [];
    level[i] = 0;
  }

  dagEdges.forEach(function(edge) {
    if (!edge || edge.length < 2) return;
    if (edge[0] < 0 || edge[0] >= opCount || edge[1] < 0 || edge[1] >= opCount) return;
    succ[edge[0]].push(edge[1]);
    indegree[edge[1]] += 1;
  });

  var queue = [];
  for (i = 0; i < opCount; i++) {
    if (indegree[i] === 0) queue.push(i);
  }
  queue.sort(function(a, b) { return a - b; });

  while (queue.length > 0) {
    var cur = queue.shift();
    processed += 1;
    succ[cur].forEach(function(dst) {
      level[dst] = Math.max(level[dst], level[cur] + 1);
      maxLevel = Math.max(maxLevel, level[dst]);
      indegree[dst] -= 1;
      if (indegree[dst] === 0) queue.push(dst);
    });
    queue.sort(function(a, b) { return a - b; });
  }

  for (i = 0; i < opCount; i++) {
    if (!rows[level[i]]) rows[level[i]] = [];
    rows[level[i]].push(i);
  }

  Object.keys(rows).forEach(function(key) {
    rows[key].sort(function(a, b) { return a - b; });
  });

  var maxRowCount = 0;
  Object.keys(rows).forEach(function(key) {
    maxRowCount = Math.max(maxRowCount, rows[key].length);
  });

  var bandLoads = [];
  function addBandSpan(srcLevel, dstLevel) {
    for (var band = srcLevel + 1; band <= dstLevel; band++)
      bandLoads[band] = (bandLoads[band] || 0) + 1;
  }
  inputEdges.forEach(function(edge) {
    if (!edge || edge.length < 2 || edge[1] < 0 || edge[1] >= opCount) return;
    addBandSpan(-1, level[edge[1]]);
  });
  dagEdges.forEach(function(edge) {
    if (!edge || edge.length < 2) return;
    if (edge[0] < 0 || edge[0] >= opCount || edge[1] < 0 || edge[1] >= opCount) return;
    addBandSpan(level[edge[0]], level[edge[1]]);
  });
  outputEdges.forEach(function(edge) {
    if (!edge || edge.length < 2) return;
    if (edge[0] >= 0 && edge[0] < opCount) {
      addBandSpan(level[edge[0]], maxLevel + 1);
    } else {
      addBandSpan(-1, maxLevel + 1);
    }
  });

  var maxBandLoad = 1;
  bandLoads.forEach(function(load) {
    maxBandLoad = Math.max(maxBandLoad, load || 0);
  });

  return {
    fu: fu,
    inputEdges: inputEdges,
    dagEdges: dagEdges,
    outputEdges: outputEdges,
    opCount: opCount,
    level: level,
    rows: rows,
    maxLevel: maxLevel,
    maxRowCount: Math.max(1, maxRowCount),
    maxBandLoad: maxBandLoad,
    bandLoads: bandLoads
  };
}

function measureFunctionUnitLayout(fu) {
  var analysis = analyzeFunctionUnit(fu);
  var rowCount = Math.max(1, analysis.maxLevel + 1);
  var opW = analysis.maxRowCount > 1 ? 62 : 56;
  var opH = 34;
  var channelGap = Math.max(26, Math.min(52, 20 + analysis.maxBandLoad * 6));
  var colGap = Math.max(26, Math.min(56, 18 + analysis.maxBandLoad * 5));
  var innerPadX = 14;
  var innerTop = 26;
  var innerBottom = 22;
  var rowWidth = analysis.maxRowCount * opW + Math.max(0, analysis.maxRowCount - 1) * colGap;
  var portSpanW = Math.max(
    opW,
    (Math.max(0, (fu.numIn || 0) - 1) * 26) + opW,
    (Math.max(0, (fu.numOut || 0) - 1) * 26) + opW
  );
  var trackReserve = Math.max(0, analysis.maxBandLoad - 1) * 14;
  var innerW = Math.max(104, rowWidth + trackReserve, portSpanW);
  var innerH = rowCount * opH + (rowCount + 1) * channelGap;

  return {
    analysis: analysis,
    opW: opW,
    opH: opH,
    channelGap: channelGap,
    colGap: colGap,
    innerPadX: innerPadX,
    innerTop: innerTop,
    innerBottom: innerBottom,
    innerW: innerW,
    innerH: innerH,
    boxW: innerW + innerPadX * 2,
    boxH: innerTop + innerH + innerBottom
  };
}

function borderPortX(boxX, boxW, index, count) {
  if (count <= 1) return boxX + boxW / 2;
  var edgeInset = Math.max(12, Math.min(22, boxW * 0.14));
  var usableW = Math.max(0, boxW - edgeInset * 2);
  return boxX + edgeInset + usableW * index / (count - 1);
}

function computeADGRenderHints(data, mappingEnabled) {
  var totalPEs = 0;
  var totalFUs = 0;
  var totalFUOps = 0;
  var totalConnections = (data && data.connections) ? data.connections.length : 0;

  ((data && data.components) || []).forEach(function(comp) {
    if (comp.kind !== "spatial_pe" && comp.kind !== "temporal_pe") return;
    totalPEs += 1;
    (comp.fus || []).forEach(function(fu) {
      totalFUs += 1;
      totalFUOps += (fu.ops || []).length;
    });
  });

  var denseScene =
    totalPEs >= 16 ||
    totalFUs >= 160 ||
    totalFUOps >= 320 ||
    totalConnections >= 120;

  return {
    denseScene: denseScene,
    wrapComponents: denseScene,
    wrapRowWidth: denseScene ? 3200 : 0,
    compactAllFUs: denseScene && !mappingEnabled,
    compactCollapsedBoxW: denseScene ? 68 : 104,
    compactCollapsedBoxH: denseScene ? 16 : 22,
    compactFuBoxMargin: denseScene ? 6 : 12,
    peInnerPadX: denseScene ? 14 : 20,
    peInnerPadY: denseScene ? 24 : 30,
    peMinW: denseScene ? 120 : 200,
    peMinH: denseScene ? 84 : 90
  };
}

function buildPERenderLayout(peDef, mappingEnabled, peUsedFUs, renderHints) {
  var fus = peDef.fus || [];
  renderHints = renderHints || {};
  var forceCompact = !!renderHints.compactAllFUs;
  var fuBoxMargin = forceCompact
    ? (renderHints.compactFuBoxMargin || 6)
    : 12;
  var collapsedBoxW = forceCompact
    ? (renderHints.compactCollapsedBoxW || 68)
    : 104;
  var collapsedBoxH = forceCompact
    ? (renderHints.compactCollapsedBoxH || 16)
    : 22;
  var hasMapping = Object.keys(peUsedFUs || {}).length > 0;
  var fuLayouts = [];
  var fuWidths = [];
  var fuHeights = [];

  fus.forEach(function(fu) {
    if ((mappingEnabled && hasMapping && !peUsedFUs[fu.name]) || forceCompact) {
      fuLayouts.push({
        collapsed: true,
        boxW: collapsedBoxW,
        boxH: collapsedBoxH
      });
      fuWidths.push(collapsedBoxW);
      fuHeights.push(collapsedBoxH);
      return;
    }
    var layout = measureFunctionUnitLayout(fu);
    fuLayouts.push(layout);
    fuWidths.push(layout.boxW);
    fuHeights.push(layout.boxH);
  });

  var peMappingMargin = (mappingEnabled && hasMapping) ? 40 : 0;
  var peInnerPadX = renderHints.peInnerPadX || 20;
  var peInnerPadY = renderHints.peInnerPadY || 30;
  var totalFuW = 0;
  fuWidths.forEach(function(w) { totalFuW += w; });
  var peW = Math.max(
    renderHints.peMinW || 200,
    totalFuW + Math.max(0, fus.length - 1) * fuBoxMargin + peInnerPadX * 2 + peMappingMargin
  );
  var maxFuH = 0;
  fuHeights.forEach(function(h) { maxFuH = Math.max(maxFuH, h); });
  var peH = Math.max(
    renderHints.peMinH || 90,
    peInnerPadY + maxFuH + peInnerPadY
  );

  return {
    fus: fus,
    fuLayouts: fuLayouts,
    fuWidths: fuWidths,
    fuHeights: fuHeights,
    fuBoxMargin: fuBoxMargin,
    peMappingMargin: peMappingMargin,
    peInnerPadX: peInnerPadX,
    peInnerPadY: peInnerPadY,
    peW: peW,
    peH: peH
  };
}
