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
  var idx = { swToHw: {}, hwToSw: {}, fuToSw: {} };
  MAPPING_DATA.node_mappings.forEach(function(m) {
    idx.swToHw[m.sw_node] = m;
    if (!idx.hwToSw[m.pe_name]) idx.hwToSw[m.pe_name] = [];
    idx.hwToSw[m.pe_name].push(m);
    var fuKey = m.pe_name + "/" + m.hw_name;
    if (!idx.fuToSw[fuKey]) idx.fuToSw[fuKey] = [];
    idx.fuToSw[fuKey].push(m);
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

function buildPERenderLayout(peDef, mappingEnabled, peUsedFUs) {
  var fus = peDef.fus || [];
  var fuBoxMargin = 12;
  var collapsedBoxW = 104;
  var collapsedBoxH = 22;
  var hasMapping = Object.keys(peUsedFUs || {}).length > 0;
  var fuLayouts = [];
  var fuWidths = [];
  var fuHeights = [];

  fus.forEach(function(fu) {
    if (mappingEnabled && hasMapping && !peUsedFUs[fu.name]) {
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
  var totalFuW = 0;
  fuWidths.forEach(function(w) { totalFuW += w; });
  var peW = Math.max(200, totalFuW + Math.max(0, fus.length - 1) * fuBoxMargin + 40 + peMappingMargin);
  var maxFuH = 0;
  fuHeights.forEach(function(h) { maxFuH = Math.max(maxFuH, h); });
  var peH = 30 + maxFuH + 30;

  return {
    fus: fus,
    fuLayouts: fuLayouts,
    fuWidths: fuWidths,
    fuHeights: fuHeights,
    fuBoxMargin: fuBoxMargin,
    peMappingMargin: peMappingMargin,
    peW: peW,
    peH: peH
  };
}

function renderMiniFUGraph(parentG, item) {
  var fu = item.fu;
  if (!fu || !fu.ops || fu.ops.length === 0) return;

  var layoutSpec = item.layoutSpec || measureFunctionUnitLayout(fu);
  var analysis = layoutSpec.analysis;
  var inputEdges = analysis.inputEdges;
  var dagEdges = analysis.dagEdges;
  var outputEdges = analysis.outputEdges;
  var opCount = analysis.opCount;
  var level = analysis.level.slice();
  var maxLevel = analysis.maxLevel;
  var incomingRefs = [];
  var outgoingRefs = [];
  var i;

  for (i = 0; i < opCount; i++) {
    incomingRefs[i] = [];
    outgoingRefs[i] = [];
  }
  inputEdges.forEach(function(edge) {
    if (!edge || edge.length < 2 || edge[1] < 0 || edge[1] >= opCount) return;
    incomingRefs[edge[1]].push({ kind: "fu_in", index: edge[0] });
  });
  dagEdges.forEach(function(edge) {
    if (!edge || edge.length < 2 || edge[0] < 0 || edge[0] >= opCount || edge[1] < 0 || edge[1] >= opCount) return;
    outgoingRefs[edge[0]].push({ kind: "op", index: edge[1] });
    incomingRefs[edge[1]].push({ kind: "op", index: edge[0] });
  });
  outputEdges.forEach(function(edge) {
    if (!edge || edge.length < 2 || edge[0] < 0 || edge[0] >= opCount) return;
    outgoingRefs[edge[0]].push({ kind: "fu_out", index: edge[1] });
  });

  var levels = analysis.rows;

  var clipId = "fu-clip-" + item.peLabel + "-" + fu.name;
  var defs = parentG.append("defs");
  defs.append("clipPath").attr("id", clipId)
    .append("rect")
    .attr("x", item.boxX + 1).attr("y", item.boxY + 1)
    .attr("width", Math.max(1, item.boxW - 2))
    .attr("height", Math.max(1, item.boxH - 2));

  var g = parentG.append("g").attr("clip-path", "url(#" + clipId + ")");
  var nodePos = {};
  var usableTop = item.y;
  var usableBottom = item.y + item.h;
  var channelGap = layoutSpec.channelGap;
  var opW = layoutSpec.opW;
  var opH = layoutSpec.opH;

  Object.keys(levels).forEach(function(keyStr) {
    var key = parseInt(keyStr, 10);
    var row = levels[key];
    var xStep = item.w / (row.length + 1);
    var rowStagger = row.length > 1
      ? Math.min(6, Math.max(2, channelGap * 0.2))
      : 0;
    row.forEach(function(opIdx, idx) {
      var yOffset = (idx - (row.length - 1) / 2) * rowStagger;
      nodePos[opIdx] = {
        x: item.x + xStep * (idx + 1),
        y: clamp(
          item.y + channelGap + opH / 2 + key * (opH + channelGap) + yOffset,
          usableTop + opH / 2,
          usableBottom - opH / 2
        )
      };
    });
  });

  function anchorXForRef(ref) {
    if (!ref) return null;
    if (ref.kind === "op" && nodePos[ref.index]) return nodePos[ref.index].x;
    if (ref.kind === "fu_in") {
      if (item.inputAnchors && item.inputAnchors[ref.index]) return item.inputAnchors[ref.index].x;
      return borderPortX(item.boxX, item.boxW, ref.index, Math.max(1, fu.numIn));
    }
    if (ref.kind === "fu_out") {
      if (item.outputAnchors && item.outputAnchors[ref.index]) return item.outputAnchors[ref.index].x;
      return borderPortX(item.boxX, item.boxW, ref.index, Math.max(1, fu.numOut));
    }
    return null;
  }
  function meanRefX(refs) {
    if (!refs || refs.length === 0) return null;
    var sum = 0;
    var count = 0;
    refs.forEach(function(ref) {
      var x = anchorXForRef(ref);
      if (x == null) return;
      sum += x;
      count += 1;
    });
    return count > 0 ? sum / count : null;
  }
  function desiredNodeX(opIdx) {
    var inX = meanRefX(incomingRefs[opIdx]);
    var outX = meanRefX(outgoingRefs[opIdx]);
    if (inX != null && outX != null) return (inX * 0.55) + (outX * 0.45);
    if (inX != null) return inX;
    if (outX != null) return outX;
    return nodePos[opIdx].x;
  }
  function placeRowByDesired(row) {
    if (!row || row.length === 0) return;
    var minX = item.x + opW / 2 + 2;
    var maxX = item.x + item.w - opW / 2 - 2;
    if (row.length === 1) {
      nodePos[row[0]].x = clamp(desiredNodeX(row[0]), minX, maxX);
      return;
    }

    var available = Math.max(0, maxX - minX);
    var spacing = Math.min(opW + 12, available / (row.length - 1));
    var ordered = row.slice().sort(function(a, b) {
      var da = desiredNodeX(a);
      var db = desiredNodeX(b);
      if (Math.abs(da - db) > 0.5) return da - db;
      return a - b;
    });
    var placed = [];
    ordered.forEach(function(opIdx, idx) {
      var lower = minX + spacing * idx;
      var upper = maxX - spacing * (ordered.length - 1 - idx);
      placed[idx] = clamp(desiredNodeX(opIdx), lower, upper);
    });
    for (var fi = 1; fi < placed.length; fi++)
      placed[fi] = Math.max(placed[fi], placed[fi - 1] + spacing);
    for (var bi = placed.length - 2; bi >= 0; bi--)
      placed[bi] = Math.min(placed[bi], placed[bi + 1] - spacing);
    ordered.forEach(function(opIdx, idx) {
      nodePos[opIdx].x = clamp(placed[idx], minX, maxX);
    });
  }

  for (var pass = 0; pass < 4; pass++) {
    for (var topLevel = 0; topLevel <= maxLevel; topLevel++)
      placeRowByDesired(levels[topLevel] || []);
    for (var bottomLevel = maxLevel; bottomLevel >= 0; bottomLevel--)
      placeRowByDesired(levels[bottomLevel] || []);
  }

  function endpointKey(desc) {
    return desc.kind + ":" + desc.index;
  }

  function baseEndpoint(desc) {
    if (desc.kind === "fu_in") {
      if (item.inputAnchors && item.inputAnchors[desc.index])
        return { x: item.inputAnchors[desc.index].x, y: item.inputAnchors[desc.index].y };
      return {
        x: borderPortX(item.boxX, item.boxW, desc.index, Math.max(1, fu.numIn)),
        y: item.boxY
      };
    }
    if (desc.kind === "fu_out") {
      if (item.outputAnchors && item.outputAnchors[desc.index])
        return { x: item.outputAnchors[desc.index].x, y: item.outputAnchors[desc.index].y };
      return {
        x: borderPortX(item.boxX, item.boxW, desc.index, Math.max(1, fu.numOut)),
        y: item.boxY + item.boxH
      };
    }
    if (desc.kind === "op_in")
      return { x: nodePos[desc.index].x, y: nodePos[desc.index].y - opH / 2 };
    return { x: nodePos[desc.index].x, y: nodePos[desc.index].y + opH / 2 };
  }

  var rawEdges = [];
  inputEdges.forEach(function(edge) {
    if (!edge || edge.length < 2 || !nodePos[edge[1]]) return;
    rawEdges.push({
      kind: "input",
      srcDesc: { kind: "fu_in", index: edge[0] },
      dstDesc: { kind: "op_in", index: edge[1] },
      color: "#4ecdc4",
      dash: null
    });
  });
  dagEdges.forEach(function(edge) {
    if (!edge || edge.length < 2 || !nodePos[edge[0]] || !nodePos[edge[1]]) return;
    rawEdges.push({
      kind: "dag",
      srcDesc: { kind: "op_out", index: edge[0] },
      dstDesc: { kind: "op_in", index: edge[1] },
      color: "#6a8faf",
      dash: null
    });
  });
  outputEdges.forEach(function(edge) {
    if (!edge || edge.length < 2) return;
    rawEdges.push({
      kind: "output",
      srcDesc: edge[0] >= 0
        ? { kind: "op_out", index: edge[0] }
        : { kind: "fu_in", index: -(edge[0] + 1) },
      dstDesc: { kind: "fu_out", index: edge[1] },
      color: edge[0] >= 0 ? "#ff6b35" : "#888888",
      dash: edge[0] >= 0 ? null : "2,2"
    });
  });

  function assignOrderedSlots(edges, selectKey, scoreFn, slotProp, totalProp) {
    var groups = {};
    edges.forEach(function(edge) {
      var key = selectKey(edge);
      if (!key) return;
      if (!groups[key]) groups[key] = [];
      groups[key].push(edge);
    });
    Object.keys(groups).forEach(function(key) {
      groups[key].sort(function(a, b) { return scoreFn(a) - scoreFn(b); });
      groups[key].forEach(function(edge, idx) {
        edge[slotProp] = idx;
        edge[totalProp] = groups[key].length;
      });
    });
  }

  assignOrderedSlots(
    rawEdges,
    function(edge) { return edge.dstDesc.kind === "op_in" ? endpointKey(edge.dstDesc) : null; },
    function(edge) { return baseEndpoint(edge.srcDesc).x; },
    "_dstSlot", "_dstSlotTotal");
  assignOrderedSlots(
    rawEdges,
    function(edge) { return edge.srcDesc.kind === "op_out" ? endpointKey(edge.srcDesc) : null; },
    function(edge) { return baseEndpoint(edge.dstDesc).x; },
    "_srcSlot", "_srcSlotTotal");
  assignOrderedSlots(
    rawEdges,
    function(edge) { return edge.srcDesc.kind === "fu_in" ? endpointKey(edge.srcDesc) : null; },
    function(edge) { return baseEndpoint(edge.dstDesc).x; },
    "_srcLane", "_srcLaneTotal");
  assignOrderedSlots(
    rawEdges,
    function(edge) { return edge.dstDesc.kind === "fu_out" ? endpointKey(edge.dstDesc) : null; },
    function(edge) { return baseEndpoint(edge.srcDesc).x; },
    "_dstLane", "_dstLaneTotal");

  function resolvePort(desc, slot, total) {
    var base = baseEndpoint(desc);
    if (desc.kind === "op_in" || desc.kind === "op_out") {
      var span = Math.max(16, opW - 16);
      var actualTotal = Math.max(1, total || 1);
      var actualSlot = slot == null ? 0 : slot;
      base = {
        x: nodePos[desc.index].x - span / 2 + span * (actualSlot + 1) / (actualTotal + 1),
        y: desc.kind === "op_in" ? nodePos[desc.index].y - opH / 2 : nodePos[desc.index].y + opH / 2
      };
    }
    return {
      x: base.x,
      y: base.y,
      nx: 0,
      ny: (desc.kind === "fu_in" || desc.kind === "op_out") ? 1 : -1,
      key: endpointKey(desc),
      allowLaneShift: desc.kind === "fu_in" || desc.kind === "fu_out",
      kind: desc.kind
    };
  }

  var GRID_STEP = 4;
  var PORT_STUB = Math.min(18, Math.max(13, opH * 0.72));
  var LANE_SPACING = 6;
  var NODE_CLEARANCE = Math.max(12, Math.round(opH * 0.55));
  var NODE_HIT_MARGIN = 2;
  var CURVE_RADIUS = 4.5;
  var TURN_PENALTY = 3.2;
  var REUSED_SEGMENT_PENALTY = 120;
  var routeLeft = item.x + 4;
  var routeTop = item.boxY + 8;
  var routeRight = item.x + item.w - 4;
  var routeBottom = item.boxY + item.boxH - 8;
  var gMinX = routeLeft;
  var gMinY = routeTop;
  var gCols = Math.max(1, Math.floor((routeRight - routeLeft) / GRID_STEP));
  var gRows = Math.max(1, Math.floor((routeBottom - routeTop) / GRID_STEP));
  var nodeBoxes = [];
  var usedMiniSegments = [];

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }
  function cleanPolyline(points) {
    if (!points || points.length === 0) return [];
    var deduped = [];
    points.forEach(function(pt) {
      if (!deduped.length) {
        deduped.push({ x: pt.x, y: pt.y });
        return;
      }
      var prev = deduped[deduped.length - 1];
      if (Math.abs(prev.x - pt.x) > 0.5 || Math.abs(prev.y - pt.y) > 0.5)
        deduped.push({ x: pt.x, y: pt.y });
    });
    if (deduped.length <= 2) return deduped;
    var simplified = [deduped[0]];
    for (var pi = 1; pi < deduped.length - 1; pi++) {
      var a = simplified[simplified.length - 1];
      var b = deduped[pi];
      var c = deduped[pi + 1];
      if ((Math.abs(a.x - b.x) < 0.5 && Math.abs(b.x - c.x) < 0.5) ||
          (Math.abs(a.y - b.y) < 0.5 && Math.abs(b.y - c.y) < 0.5))
        continue;
      simplified.push(b);
    }
    simplified.push(deduped[deduped.length - 1]);
    return simplified;
  }
  function expandedNodeBox(opIdx) {
    return {
      x0: nodePos[opIdx].x - opW / 2 - NODE_CLEARANCE,
      y0: nodePos[opIdx].y - opH / 2 - NODE_CLEARANCE,
      x1: nodePos[opIdx].x + opW / 2 + NODE_CLEARANCE,
      y1: nodePos[opIdx].y + opH / 2 + NODE_CLEARANCE
    };
  }
  function intervalOverlaps(a0, a1, b0, b1) {
    return Math.max(a0, b0) < Math.min(a1, b1) - 0.5;
  }
  function verticalHitsBox(x, y0, y1, box) {
    return x > box.x0 - NODE_HIT_MARGIN && x < box.x1 + NODE_HIT_MARGIN &&
      intervalOverlaps(Math.min(y0, y1), Math.max(y0, y1), box.y0, box.y1);
  }
  function horizontalHitsBox(y, x0, x1, box) {
    return y > box.y0 - NODE_HIT_MARGIN && y < box.y1 + NODE_HIT_MARGIN &&
      intervalOverlaps(Math.min(x0, x1), Math.max(x0, x1), box.x0, box.x1);
  }
  function edgeLaneShift(port, laneIndex, laneTotal) {
    if (!port.allowLaneShift) return 0;
    return ((laneIndex || 0) - (Math.max(1, laneTotal || 1) - 1) / 2) * LANE_SPACING;
  }
  function anchorForPort(port, laneIndex, laneTotal) {
    var shift = edgeLaneShift(port, laneIndex, laneTotal);
    return {
      x: clamp(port.x + shift, routeLeft, routeRight),
      y: clamp(port.y + port.ny * PORT_STUB, routeTop, routeBottom)
    };
  }
  function pathHitsNodeBoxes(points) {
    for (var pi = 0; pi < points.length - 1; pi++) {
      var a = points[pi];
      var b = points[pi + 1];
      for (var bi = 0; bi < nodeBoxes.length; bi++) {
        if (Math.abs(a.x - b.x) < 0.5) {
          if (verticalHitsBox(a.x, a.y, b.y, nodeBoxes[bi])) return true;
        } else if (Math.abs(a.y - b.y) < 0.5) {
          if (horizontalHitsBox(a.y, a.x, b.x, nodeBoxes[bi])) return true;
        }
      }
    }
    return false;
  }
  function pathOverlapsUsed(points) {
    for (var pi = 0; pi < points.length - 1; pi++) {
      var a = points[pi];
      var b = points[pi + 1];
      var isVertical = Math.abs(a.x - b.x) < 0.5;
      var lo = isVertical ? Math.min(a.y, b.y) : Math.min(a.x, b.x);
      var hi = isVertical ? Math.max(a.y, b.y) : Math.max(a.x, b.x);
      for (var ui = 0; ui < usedMiniSegments.length; ui++) {
        var seg = usedMiniSegments[ui];
        if (seg.isVertical !== isVertical) continue;
        if (Math.abs(seg.fixed - (isVertical ? a.x : a.y)) >= 0.5) continue;
        if (intervalOverlaps(lo, hi, seg.lo, seg.hi)) return true;
      }
    }
    return false;
  }
  function reservePathSegments(points) {
    for (var pi = 0; pi < points.length - 1; pi++) {
      var a = points[pi];
      var b = points[pi + 1];
      if (Math.abs(a.x - b.x) < 0.5) {
        usedMiniSegments.push({
          isVertical: true,
          fixed: a.x,
          lo: Math.min(a.y, b.y),
          hi: Math.max(a.y, b.y)
        });
      } else if (Math.abs(a.y - b.y) < 0.5) {
        usedMiniSegments.push({
          isVertical: false,
          fixed: a.y,
          lo: Math.min(a.x, b.x),
          hi: Math.max(a.x, b.x)
        });
      }
    }
  }
  function buildCandidateLaneXs(startAnchor, endAnchor, fromPort, toPort) {
    var y0 = Math.min(startAnchor.y, endAnchor.y);
    var y1 = Math.max(startAnchor.y, endAnchor.y);
    var preferredX = (startAnchor.x + endAnchor.x) / 2;
    var candidates = [
      startAnchor.x,
      endAnchor.x,
      preferredX,
      clamp(startAnchor.x - 18, routeLeft + 1, routeRight - 1),
      clamp(endAnchor.x + 18, routeLeft + 1, routeRight - 1),
      routeLeft + 2,
      routeRight - 2
    ];
    var intervals = [];
    nodeBoxes.forEach(function(box) {
      if (intervalOverlaps(y0, y1, box.y0, box.y1))
        intervals.push({ x0: clamp(box.x0, routeLeft, routeRight), x1: clamp(box.x1, routeLeft, routeRight) });
    });
    intervals.sort(function(a, b) { return a.x0 - b.x0; });
    var merged = [];
    var spanMinX = null;
    var spanMaxX = null;
    intervals.forEach(function(cur) {
      spanMinX = spanMinX == null ? cur.x0 : Math.min(spanMinX, cur.x0);
      spanMaxX = spanMaxX == null ? cur.x1 : Math.max(spanMaxX, cur.x1);
      if (!merged.length || cur.x0 > merged[merged.length - 1].x1 + 0.5) {
        merged.push({ x0: cur.x0, x1: cur.x1 });
      } else {
        merged[merged.length - 1].x1 = Math.max(merged[merged.length - 1].x1, cur.x1);
      }
    });
    var gapStart = routeLeft;
    merged.forEach(function(interval) {
      if (interval.x0 - gapStart > 4)
        candidates.push((gapStart + interval.x0) / 2);
      gapStart = Math.max(gapStart, interval.x1);
    });
    if (routeRight - gapStart > 4)
      candidates.push((gapStart + routeRight) / 2);
    if (spanMinX != null && spanMaxX != null) {
      candidates.push(clamp(spanMinX - 14, routeLeft + 1, routeRight - 1));
      candidates.push(clamp(spanMaxX + 14, routeLeft + 1, routeRight - 1));
    }
    if (toPort.kind === "fu_out") {
      var preferRight = toPort.x >= item.boxX + item.boxW / 2;
      var outerLane = preferRight ? routeRight - 3 : routeLeft + 3;
      var spanEscape = spanMinX != null && spanMaxX != null
        ? (preferRight ? spanMaxX + 20 : spanMinX - 20)
        : outerLane;
      candidates.push(clamp(spanEscape, routeLeft + 1, routeRight - 1));
      candidates.push(outerLane);
    }
    if (fromPort.kind === "fu_in" && toPort.kind === "op_in") {
      var inputRight = fromPort.x >= item.boxX + item.boxW / 2;
      candidates.push(inputRight ? routeRight - 10 : routeLeft + 10);
    }
    return candidates;
  }
  function buildMiniPathForLane(fromPort, toPort, startAnchor, endAnchor, laneX) {
    var pts = [
      { x: fromPort.x, y: fromPort.y },
      { x: startAnchor.x, y: startAnchor.y }
    ];
    if (Math.abs(laneX - startAnchor.x) > 0.5)
      pts.push({ x: laneX, y: startAnchor.y });
    if (Math.abs(endAnchor.y - startAnchor.y) > 0.5)
      pts.push({ x: laneX, y: endAnchor.y });
    if (Math.abs(laneX - endAnchor.x) > 0.5)
      pts.push({ x: endAnchor.x, y: endAnchor.y });
    pts.push({ x: toPort.x, y: toPort.y });
    return cleanPolyline(pts);
  }
  function buildMiniEdgePolyline(fromPort, toPort, srcLane, srcLaneTotal, dstLane, dstLaneTotal) {
    var startAnchor = anchorForPort(fromPort, srcLane, srcLaneTotal);
    var endAnchor = anchorForPort(toPort, dstLane, dstLaneTotal);
    var candidates = buildCandidateLaneXs(startAnchor, endAnchor, fromPort, toPort);
    var best = null;
    var seen = {};
    var centerX = item.boxX + item.boxW / 2;

    candidates.forEach(function(rawX) {
      var laneX = Math.round(clamp(rawX, routeLeft + 1, routeRight - 1) * 2) / 2;
      if (seen[laneX]) return;
      seen[laneX] = true;
      var pts = buildMiniPathForLane(fromPort, toPort, startAnchor, endAnchor, laneX);
      if (pathHitsNodeBoxes(pts)) return;
      var score = 0;
      for (var pi = 0; pi < pts.length - 1; pi++) {
        score += Math.abs(pts[pi + 1].x - pts[pi].x) + Math.abs(pts[pi + 1].y - pts[pi].y);
      }
      score += Math.abs(laneX - (startAnchor.x + endAnchor.x) / 2) * 0.6;
      score += (pts.length - 2) * 12;
      if (toPort.kind === "fu_out") {
        if (toPort.x >= centerX) score += Math.max(0, centerX - laneX) * 2.4;
        else score += Math.max(0, laneX - centerX) * 2.4;
      }
      if (fromPort.kind === "fu_in" && toPort.kind === "op_in") {
        if (fromPort.x >= centerX) score += Math.max(0, centerX - laneX) * 1.6;
        else score += Math.max(0, laneX - centerX) * 1.6;
      }
      if (pathOverlapsUsed(pts)) score += 500;
      if (!best || score < best.score) best = { pts: pts, score: score };
    });

    if (!best) {
      var fallback = buildMiniPathForLane(fromPort, toPort, startAnchor, endAnchor,
        clamp((startAnchor.x + endAnchor.x) / 2, routeLeft + 1, routeRight - 1));
      reservePathSegments(fallback);
      return fallback;
    }

    reservePathSegments(best.pts);
    return best.pts;
  }
  function buildRoundedPath(points) {
    if (!points || points.length === 0) return "";
    if (points.length === 1) return "M" + points[0].x + "," + points[0].y;
    var d = "M" + points[0].x + "," + points[0].y;
    for (var pi = 1; pi < points.length - 1; pi++) {
      var prev = points[pi - 1];
      var cur = points[pi];
      var next = points[pi + 1];
      var inDx = cur.x - prev.x;
      var inDy = cur.y - prev.y;
      var outDx = next.x - cur.x;
      var outDy = next.y - cur.y;
      var inLen = Math.sqrt(inDx * inDx + inDy * inDy);
      var outLen = Math.sqrt(outDx * outDx + outDy * outDy);
      if (inLen < 0.5 || outLen < 0.5) {
        d += "L" + cur.x + "," + cur.y;
        continue;
      }
      var radius = Math.min(CURVE_RADIUS, inLen / 2, outLen / 2);
      var entry = {
        x: cur.x - (inDx / inLen) * radius,
        y: cur.y - (inDy / inLen) * radius
      };
      var exit = {
        x: cur.x + (outDx / outLen) * radius,
        y: cur.y + (outDy / outLen) * radius
      };
      d += "L" + entry.x + "," + entry.y;
      d += "Q" + cur.x + "," + cur.y + " " + exit.x + "," + exit.y;
    }
    d += "L" + points[points.length - 1].x + "," + points[points.length - 1].y;
    return d;
  }

  for (i = 0; i < opCount; i++)
    nodeBoxes.push(expandedNodeBox(i));

  rawEdges.forEach(function(edge) {
    var srcBase = baseEndpoint(edge.srcDesc);
    var dstBase = baseEndpoint(edge.dstDesc);
    edge._span = Math.abs(srcBase.x - dstBase.x) + Math.abs(srcBase.y - dstBase.y);
  });
  rawEdges.sort(function(a, b) {
    var prio = { dag: 0, input: 1, output: 2 };
    if (prio[a.kind] !== prio[b.kind]) return prio[a.kind] - prio[b.kind];
    if (a._span !== b._span) return a._span - b._span;
    return a.color < b.color ? -1 : 1;
  });

  var plannedEdges = rawEdges.map(function(edge, idx) {
    return {
      order: idx,
      color: edge.color,
      dash: edge.dash,
      fromPort: resolvePort(edge.srcDesc, edge._srcSlot, edge._srcSlotTotal),
      toPort: resolvePort(edge.dstDesc, edge._dstSlot, edge._dstSlotTotal),
      srcLane: edge._srcLane || 0,
      srcLaneTotal: edge._srcLaneTotal || 1,
      dstLane: edge._dstLane || 0,
      dstLaneTotal: edge._dstLaneTotal || 1
    };
  });

  var routedEdges = [];
  plannedEdges.forEach(function(edge) {
    routedEdges.push({
      color: edge.color,
      dash: edge.dash,
      pts: buildMiniEdgePolyline(
        edge.fromPort, edge.toPort,
        edge.srcLane, edge.srcLaneTotal,
        edge.dstLane, edge.dstLaneTotal
      )
    });
  });

  routedEdges.forEach(function(edge) {
    g.append("path")
      .attr("d", buildRoundedPath(edge.pts))
      .attr("fill", "none")
      .attr("stroke", edge.color)
      .attr("stroke-width", 1.05)
      .attr("opacity", 0.84)
      .attr("stroke-linecap", "round")
      .attr("stroke-linejoin", "round")
      .attr("stroke-dasharray", edge.dash);
  });

  for (i = 0; i < opCount; i++) {
    var style = miniFUOpStyle(fu.ops[i]);
    var group = g.append("g");
    if (style.shape === "mux") {
      group.append("polygon")
        .attr("points", [
          (nodePos[i].x - opW / 2 + 6) + "," + (nodePos[i].y - opH / 2),
          (nodePos[i].x + opW / 2) + "," + (nodePos[i].y - opH / 2),
          (nodePos[i].x + opW / 2 - 6) + "," + (nodePos[i].y + opH / 2),
          (nodePos[i].x - opW / 2) + "," + (nodePos[i].y + opH / 2)
        ].join(" "))
        .attr("fill", style.fill)
        .attr("stroke", style.stroke)
        .attr("stroke-width", 1);
    } else {
      group.append("ellipse")
        .attr("cx", nodePos[i].x).attr("cy", nodePos[i].y)
        .attr("rx", opW / 2).attr("ry", opH / 2)
        .attr("fill", style.fill)
        .attr("stroke", style.stroke)
        .attr("stroke-width", 1);
    }
    splitDisplayLabel(fu.ops[i]).forEach(function(line, lineIdx) {
      group.append("text")
        .attr("x", nodePos[i].x)
        .attr("y", nodePos[i].y - (splitDisplayLabel(fu.ops[i]).length - 1) * 3 + lineIdx * 7 + 2)
        .attr("text-anchor", "middle")
        .attr("fill", style.text)
        .attr("font-size", "5.5px")
        .text(shortenText(line, 10));
    });
  }
}

// ============================================================
// ADG Renderer: draws spatial_pe with nested FUs
// ============================================================

function renderADG() {
  // Reset module-level state for re-render
  adgCompBoxes = {};
  adgFUBoxes = {};
  fuPortPos = {};

  var svg = d3.select("#svg-adg");
  svg.selectAll("*").remove(); // Clear for re-render

  if (!ADG_DATA || ADG_DATA === "null") {
    svg.append("text").attr("x",20).attr("y",40).attr("fill","#888")
      .text("No ADG data.");
    d3.select("#status-bar").text("No ADG");
    return;
  }

  var g = svg.append("g");
  var zoom = d3.zoom().scaleExtent([0.1, 5])
    .on("zoom", function(ev) { g.attr("transform", ev.transform); });
  svg.call(zoom);

  var data = ADG_DATA;
  var portRegistry = {};

  // Export refs for fitView and cross-highlighting
  renderADG._svg = svg; renderADG._g = g; renderADG._zoom = zoom;
  renderADG._portRegistry = portRegistry;

  // Pre-compute internal content dimensions to determine module margin
  var maxPeW = 0, maxPeH = 0;
  data.components.forEach(function(c) {
    if (c.kind === "spatial_pe") {
      var preLayout = buildPERenderLayout(c, mappingEnabled, {});
      maxPeW = Math.max(maxPeW, preLayout.peW);
      maxPeH = Math.max(maxPeH, preLayout.peH);
    }
  });

  // Also compute SW dimensions
  var maxSwW = 0, maxSwH = 0;
  data.components.forEach(function(c) {
    if (c.kind === "spatial_sw") {
      var maxP = Math.max(c.numInputs || 4, c.numOutputs || 4);
      var side = Math.max(80, maxP * 30 + 30);
      maxSwW = Math.max(maxSwW, side);
      maxSwH = Math.max(maxSwH, side);
    }
  });

  // Total content width = PE + SW + gap between them
  var totalContentW = maxPeW + (maxSwW > 0 ? maxSwW + 40 : 0);
  var totalContentH = Math.max(maxPeH, maxSwH);

  // Module margin: generous space for edge routing
  // Each side margin = sqrt(peArea / 4), clamped
  var MOD_MARGIN = Math.max(70, Math.min(200, Math.round(Math.sqrt(totalContentW * totalContentH / 4))));
  var EDGE_SPACING = 8;
  var EDGE_BORDER_GAP = 14;

  var modInPortColor = "#c39bd3";
  var modOutPortColor = "#f0b27a";
  var modBorderColor = "#9b59b6";

  var PAD = 20;
  var modX = PAD, modY = PAD;
  var modW = totalContentW + MOD_MARGIN * 2;
  var modH = totalContentH + MOD_MARGIN * 2 + 28;

  // Component starting position (after module title + margin)
  var compStartX = modX + MOD_MARGIN;
  var compStartY = modY + 28 + MOD_MARGIN;
  var yOff = compStartY;
  var xOff = compStartX;

  // Draw each component
  data.components.forEach(function(comp) {
    if (comp.kind === "spatial_pe" || (comp.kind === "instance" && comp.module)) {
      // Find the PE definition for instances
      var peDef = comp;
      if (comp.kind === "instance") {
        data.components.forEach(function(c) {
          if (c.kind === "spatial_pe" && c.name === comp.module) peDef = c;
        });
      }

      var peX = xOff;
      var peY = yOff;
      var peLabel = comp.kind === "instance" ? comp.name : peDef.name;

      // Determine which FUs are used by mapping (for collapsing)
      var peUsedFUs = {};
      if (mappingEnabled && MAPPING_DATA && MAPPING_DATA.node_mappings) {
        MAPPING_DATA.node_mappings.forEach(function(m) {
          if (m.pe_name === peLabel) peUsedFUs[m.hw_name] = true;
        });
      }
      var hasMapping = Object.keys(peUsedFUs).length > 0;
      var peLayout = buildPERenderLayout(peDef, mappingEnabled, peUsedFUs);
      var fus = peLayout.fus;
      var fuLayouts = peLayout.fuLayouts;
      var fuWidths = peLayout.fuWidths;
      var fuHeights = peLayout.fuHeights;
      var fuBoxMargin = peLayout.fuBoxMargin;
      var peMappingMargin = peLayout.peMappingMargin;
      var peW = peLayout.peW;
      var peH = peLayout.peH;

      // Store bounding box for edge routing avoidance
      comp._bbox = { name: peLabel, x: peX, y: peY, w: peW, h: peH };
      adgCompBoxes[peLabel] = comp._bbox;

      // PE border
      g.append("rect").attr("x", peX).attr("y", peY)
        .attr("width", peW).attr("height", peH)
        .attr("rx", 6).attr("fill", "#1a2f4a")
        .attr("stroke", "#2a5f8f").attr("stroke-width", 2)
        .attr("data-comp-name", peLabel).attr("data-comp-kind", "spatial_pe");

      // PE labels: type top-left (one or two lines based on width)
      var peTypeStr = "fabric.spatial_pe";
      var peTypePx = peTypeStr.length * 7; // approx char width
      if (peTypePx < peW * TYPE_LABEL_SINGLE_LINE_RATIO) {
        g.append("text").attr("x", peX + 6).attr("y", peY + 14)
          .attr("fill", "#4ecdc4").attr("font-size", "10px").attr("font-weight", "600")
          .text(peTypeStr);
      } else {
        g.append("text").attr("x", peX + 6).attr("y", peY + 12)
          .attr("fill", "#4ecdc4").attr("font-size", "10px").attr("font-weight", "600")
          .text("fabric");
        g.append("text").attr("x", peX + 6).attr("y", peY + 24)
          .attr("fill", "#4ecdc4").attr("font-size", "10px").attr("font-weight", "600")
          .text("spatial_pe");
      }
      g.append("text").attr("x", peX + peW - 6).attr("y", peY + peH - 6)
        .attr("text-anchor", "end")
        .attr("fill", "rgba(78,205,196,0.6)").attr("font-size", "9px")
        .text(peLabel);

      // PE input ports on LEFT border
      var peInCount = peDef.numInputs || 3;
      var pePorts = { in: [], out: [] };
      for (var pi = 0; pi < peInCount; pi++) {
        var py = peY + 30 + (peH - 60) * (pi + 1) / (peInCount + 1);
        pePorts.in.push({ x: peX, y: py });
        portRegistry[peLabel + "_in_" + pi] = {
          x: peX, y: py,
          side: "left", owner: peLabel, ownerKind: "component",
          nx: -1, ny: 0
        };
        g.append("rect").attr("x", peX - 5).attr("y", py - 4)
          .attr("width", 10).attr("height", 8)
          .attr("fill", "#4ecdc4").attr("stroke", "#0c1220").attr("stroke-width", 1);
        g.append("text").attr("x", peX + 8).attr("y", py + 3)
          .attr("fill", "rgba(78,205,196,0.7)").attr("font-size", "8px")
          .text("I" + pi);
      }

      // PE output ports on RIGHT border
      var peOutCount = peDef.numOutputs || 2;
      for (var pi = 0; pi < peOutCount; pi++) {
        var py = peY + 30 + (peH - 60) * (pi + 1) / (peOutCount + 1);
        pePorts.out.push({ x: peX + peW, y: py });
        portRegistry[peLabel + "_out_" + pi] = {
          x: peX + peW, y: py,
          side: "right", owner: peLabel, ownerKind: "component",
          nx: 1, ny: 0
        };
        g.append("rect").attr("x", peX + peW - 5).attr("y", py - 4)
          .attr("width", 10).attr("height", 8)
          .attr("fill", "#ff6b35").attr("stroke", "#0c1220").attr("stroke-width", 1);
        g.append("text").attr("x", peX + peW - 8).attr("y", py + 3)
          .attr("text-anchor", "end")
          .attr("fill", "rgba(255,107,53,0.7)").attr("font-size", "8px")
          .text("O" + pi);
      }

      // Draw each FU inside the PE
      var fuX = peX + 20 + (peMappingMargin / 2);
      var fuY = peY + 30;
      fus.forEach(function(fu, fi) {
        var fuLayoutSpec = fuLayouts[fi];
        var fuH = fuHeights[fi];
        var fuW = fuWidths[fi];
        var isCollapsedFU = !!fuLayoutSpec.collapsed;

        adgFUBoxes[peLabel + "/" + fu.name] = {x: fuX, y: fuY, w: fuW, h: fuH};

        if (isCollapsedFU) {
          // Collapsed FU: small rect with type + name only, no ports, no DOT
          g.append("rect").attr("x", fuX).attr("y", fuY)
            .attr("width", fuW).attr("height", fuH)
            .attr("rx", 3).attr("fill", "rgba(255,255,255,0.02)")
            .attr("stroke", "rgba(255,255,255,0.06)").attr("stroke-width", 0.5)
            .attr("stroke-dasharray", "3,2")
            .attr("data-fu-name", fu.name).attr("data-pe-name", peLabel);
          g.append("text")
            .attr("x", fuX + fuW/2).attr("y", fuY + fuH/2 + 3)
            .attr("text-anchor", "middle")
            .attr("fill", "rgba(255,255,255,0.2)").attr("font-size", "7px")
            .text(fu.name);
          fuX += fuW + fuBoxMargin;
          return; // skip detailed rendering
        }

        var ops = fu.ops || [];
        var edges = fu.edges || [];
        var numOps = ops.length;

        // FU box
        g.append("rect").attr("x", fuX).attr("y", fuY)
          .attr("width", fuW).attr("height", fuH)
          .attr("rx", 4).attr("fill", "rgba(255,255,255,0.04)")
          .attr("stroke", "rgba(255,255,255,0.15)").attr("stroke-width", 1)
          .attr("data-fu-name", fu.name).attr("data-pe-name", peLabel);

        // FU labels: type top-left (one or two lines based on width)
        var fuTypeStr = "fabric.function_unit";
        var fuTypePx = fuTypeStr.length * 4.5;
        if (fuTypePx < fuW * TYPE_LABEL_SINGLE_LINE_RATIO) {
          g.append("text").attr("x", fuX + 4).attr("y", fuY + 10)
            .attr("fill", "rgba(255,255,255,0.5)").attr("font-size", "6px")
            .text(fuTypeStr);
        } else {
          g.append("text").attr("x", fuX + 4).attr("y", fuY + 9)
            .attr("fill", "rgba(255,255,255,0.5)").attr("font-size", "6px")
            .text("fabric");
          g.append("text").attr("x", fuX + 4).attr("y", fuY + 17)
            .attr("fill", "rgba(255,255,255,0.5)").attr("font-size", "6px")
            .text("function_unit");
        }
        g.append("text").attr("x", fuX + fuW - 4).attr("y", fuY + fuH - 4)
          .attr("text-anchor", "end")
          .attr("fill", "rgba(255,255,255,0.4)").attr("font-size", "7px")
          .text(fu.name);

        // FU border ports: squares on TOP (input) and BOTTOM (output).
        // The mini DFG should connect directly to these border ports, so it is
        // rendered first and the port markers are drawn on top afterward.
        var fuBorderInPorts = [];
        for (var ip = 0; ip < fu.numIn; ip++) {
        var px = fuX + fuW * (ip + 1) / (fu.numIn + 1);
        px = borderPortX(fuX, fuW, ip, Math.max(1, fu.numIn));
        fuBorderInPorts.push({ x: px, y: fuY, idx: ip });
        fuPortPos[peLabel + "/" + fu.name + "/in_" + ip] = {x: px, y: fuY};
      }
      var fuBorderOutPorts = [];
      for (var op = 0; op < fu.numOut; op++) {
        var px = borderPortX(fuX, fuW, op, Math.max(1, fu.numOut));
        fuBorderOutPorts.push({ x: px, y: fuY + fuH, idx: op });
        fuPortPos[peLabel + "/" + fu.name + "/out_" + op] = {x: px, y: fuY + fuH};
      }

        // FU internals: render the mini DAG directly to avoid Graphviz startup.
        if (fu.ops && fu.ops.length > 0) {
          renderMiniFUGraph(g, {
            fu: fu,
            peLabel: peLabel,
            x: fuX + fuLayoutSpec.innerPadX,
            y: fuY + fuLayoutSpec.innerTop,
            w: Math.max(24, fuLayoutSpec.innerW),
            h: Math.max(18, fuLayoutSpec.innerH),
            boxX: fuX,
            boxY: fuY,
            boxW: fuW,
            boxH: fuH,
            layoutSpec: fuLayoutSpec,
            inputAnchors: fuBorderInPorts.map(function(p) { return {x: p.x, y: p.y}; }),
            outputAnchors: fuBorderOutPorts.map(function(p) { return {x: p.x, y: p.y}; })
          });

        } else {
          // No ops - just show FU name
          g.append("text").attr("x", fuX + fuW/2).attr("y", fuY + fuH/2 + 4)
            .attr("text-anchor", "middle")
            .attr("fill", "rgba(255,255,255,0.4)").attr("font-size", "10px")
            .text("(empty)");
        }

        fuBorderInPorts.forEach(function(port) {
          g.append("rect").attr("x", port.x - 4).attr("y", port.y - 4)
            .attr("width", 8).attr("height", 8)
            .attr("fill", "#2a5f5f").attr("stroke", "#4ecdc4").attr("stroke-width", 1);
          g.append("text").attr("x", port.x).attr("y", port.y - 7)
            .attr("text-anchor", "middle")
            .attr("fill", "rgba(78,205,196,0.7)").attr("font-size", "6px")
            .text("I" + port.idx);
        });
        fuBorderOutPorts.forEach(function(port) {
          g.append("rect").attr("x", port.x - 4).attr("y", port.y - 4)
            .attr("width", 8).attr("height", 8)
            .attr("fill", "#5f2a1a").attr("stroke", "#ff6b35").attr("stroke-width", 1);
          g.append("text").attr("x", port.x).attr("y", port.y + 12)
            .attr("text-anchor", "middle")
            .attr("fill", "rgba(255,107,53,0.7)").attr("font-size", "6px")
            .text("O" + port.idx);
        });

        fuX += fuW + fuBoxMargin;
      });

      // PE internal routing: mux/demux connections from PE ports to used FU ports
      if (mappingEnabled && hasMapping && MAPPING_DATA && MAPPING_DATA.edge_routings) {
        var peEdgeG = g.append("g").attr("class", "pe-internal-edges");
        routePEInternal(peEdgeG, peLabel, peX, peY, peW, peH, fus, fuHeights,
                        fuWidths, fuBoxMargin, peMappingMargin, portRegistry, data);
      }

      xOff = peX + peW + 40; // Move xOff past PE for next component
      yOff = Math.max(yOff, peY + peH + 20);
    }

    // spatial_sw rendering
    if (comp.kind === "spatial_sw") {
      var swLabel = comp.name;
      var swInCount = comp.numInputs || 4;
      var swOutCount = comp.numOutputs || 4;
      var maxP = Math.max(swInCount, swOutCount);
      var swW = Math.max(80, maxP * 30 + 30);
      var swH = swW; // Square
      var swX = xOff;
      var swY = compStartY + (totalContentH - swH) / 2; // Vertically centered

      comp._bbox = { name: swLabel, x: swX, y: swY, w: swW, h: swH };
      adgCompBoxes[swLabel] = comp._bbox;

      // SW border
      g.append("rect").attr("x", swX).attr("y", swY)
        .attr("width", swW).attr("height", swH)
        .attr("rx", 4).attr("fill", "#2a3f5f")
        .attr("stroke", "#4a6380").attr("stroke-width", 2)
        .attr("data-comp-name", swLabel).attr("data-comp-kind", "spatial_sw");

      // SW labels: type top-left (one or two lines based on width)
      var swTypeStr = "fabric.spatial_sw";
      var swTypePx = swTypeStr.length * 6.5;
      if (swTypePx < swW * TYPE_LABEL_SINGLE_LINE_RATIO) {
        g.append("text").attr("x", swX + 6).attr("y", swY + 14)
          .attr("fill", "#8a9bb5").attr("font-size", "9px").attr("font-weight", "600")
          .text(swTypeStr);
      } else {
        g.append("text").attr("x", swX + 6).attr("y", swY + 12)
          .attr("fill", "#8a9bb5").attr("font-size", "9px").attr("font-weight", "600")
          .text("fabric");
        g.append("text").attr("x", swX + 6).attr("y", swY + 22)
          .attr("fill", "#8a9bb5").attr("font-size", "9px").attr("font-weight", "600")
          .text("spatial_sw");
      }
      g.append("text").attr("x", swX + swW - 6).attr("y", swY + swH - 6)
        .attr("text-anchor", "end")
        .attr("fill", "rgba(138,155,181,0.6)").attr("font-size", "8px")
        .text(swLabel);

      // SW input ports on LEFT border
      for (var pi = 0; pi < swInCount; pi++) {
        var py = swY + 24 + (swH - 30) * (pi + 1) / (swInCount + 1);
        portRegistry[swLabel + "_in_" + pi] = {
          x: swX, y: py,
          side: "left", owner: swLabel, ownerKind: "component",
          nx: -1, ny: 0
        };
        g.append("rect").attr("x", swX - 5).attr("y", py - 4)
          .attr("width", 10).attr("height", 8)
          .attr("fill", "#4ecdc4").attr("stroke", "#0c1220").attr("stroke-width", 1);
        g.append("text").attr("x", swX + 8).attr("y", py + 3)
          .attr("fill", "rgba(78,205,196,0.7)").attr("font-size", "7px")
          .text("I" + pi);
      }

      // SW output ports on RIGHT border
      for (var pi = 0; pi < swOutCount; pi++) {
        var py = swY + 24 + (swH - 30) * (pi + 1) / (swOutCount + 1);
        portRegistry[swLabel + "_out_" + pi] = {
          x: swX + swW, y: py,
          side: "right", owner: swLabel, ownerKind: "component",
          nx: 1, ny: 0
        };
        g.append("rect").attr("x", swX + swW - 5).attr("y", py - 4)
          .attr("width", 10).attr("height", 8)
          .attr("fill", "#ff6b35").attr("stroke", "#0c1220").attr("stroke-width", 1);
        g.append("text").attr("x", swX + swW - 8).attr("y", py + 3)
          .attr("text-anchor", "end")
          .attr("fill", "rgba(255,107,53,0.7)").attr("font-size", "7px")
          .text("O" + pi);
      }

      xOff = swX + swW + 40;
      yOff = Math.max(yOff, swY + swH + 20);
    }

    if (comp.kind === "memory") {
      var memLabel = comp.name;
      var memInCount = comp.numInputs || 1;
      var memOutCount = comp.numOutputs || 1;
      var sidePortCount = Math.max(1, Math.max(memInCount - 1, memOutCount));
      var memW = 120;
      var memH = Math.max(96, 40 + sidePortCount * 26);
      var memX = xOff;
      var memY = compStartY + (totalContentH - memH) / 2;

      comp._bbox = { name: memLabel, x: memX, y: memY, w: memW, h: memH };
      adgCompBoxes[memLabel] = comp._bbox;

      g.append("rect").attr("x", memX).attr("y", memY)
        .attr("width", memW).attr("height", memH)
        .attr("rx", 5).attr("fill", "#2f3d27")
        .attr("stroke", "#7fb069").attr("stroke-width", 2)
        .attr("data-comp-name", memLabel).attr("data-comp-kind", "memory");

      g.append("text").attr("x", memX + 6).attr("y", memY + 14)
        .attr("fill", "#b8d8a8").attr("font-size", "9px").attr("font-weight", "600")
        .text(comp.memoryKind === "extmemory" ? "fabric.extmemory" : "fabric.memory");
      g.append("text").attr("x", memX + memW - 6).attr("y", memY + memH - 6)
        .attr("text-anchor", "end")
        .attr("fill", "rgba(184,216,168,0.65)").attr("font-size", "8px")
        .text(memLabel);

      if (memInCount > 0) {
        var topX = memX + memW / 2;
        portRegistry[memLabel + "_in_0"] = {
          x: topX, y: memY,
          side: "top", owner: memLabel, ownerKind: "component",
          nx: 0, ny: 1
        };
        g.append("rect").attr("x", topX - 5).attr("y", memY - 5)
          .attr("width", 10).attr("height", 10)
          .attr("fill", "#c39bd3").attr("stroke", "#0c1220").attr("stroke-width", 1);
        g.append("text").attr("x", topX).attr("y", memY - 9)
          .attr("text-anchor", "middle")
          .attr("fill", "rgba(195,155,211,0.8)").attr("font-size", "7px")
          .text("M");
      }

      for (var mi = 1; mi < memInCount; mi++) {
        var inY = memY + 20 + (memH - 30) * mi / memInCount;
        portRegistry[memLabel + "_in_" + mi] = {
          x: memX, y: inY,
          side: "left", owner: memLabel, ownerKind: "component",
          nx: -1, ny: 0
        };
        g.append("rect").attr("x", memX - 5).attr("y", inY - 4)
          .attr("width", 10).attr("height", 8)
          .attr("fill", "#4ecdc4").attr("stroke", "#0c1220").attr("stroke-width", 1);
        g.append("text").attr("x", memX + 8).attr("y", inY + 3)
          .attr("fill", "rgba(78,205,196,0.7)").attr("font-size", "7px")
          .text("I" + mi);
      }

      for (var mo = 0; mo < memOutCount; mo++) {
        var outY = memY + 20 + (memH - 30) * (mo + 1) / (memOutCount + 1);
        portRegistry[memLabel + "_out_" + mo] = {
          x: memX + memW, y: outY,
          side: "right", owner: memLabel, ownerKind: "component",
          nx: 1, ny: 0
        };
        g.append("rect").attr("x", memX + memW - 5).attr("y", outY - 4)
          .attr("width", 10).attr("height", 8)
          .attr("fill", "#ff6b35").attr("stroke", "#0c1220").attr("stroke-width", 1);
        g.append("text").attr("x", memX + memW - 8).attr("y", outY + 3)
          .attr("text-anchor", "end")
          .attr("fill", "rgba(255,107,53,0.7)").attr("font-size", "7px")
          .text("O" + mo);
      }

      xOff = memX + memW + 40;
      yOff = Math.max(yOff, memY + memH + 20);
    }
  });

  yOff += 20;

  // Recompute module size from actual component bounding boxes.
  // Module must encompass ALL components with equal margin on all sides.
  var actualMaxX = modX + MOD_MARGIN, actualMaxY = modY + 28 + MOD_MARGIN;
  var actualMinX = actualMaxX, actualMinY = actualMaxY;
  data.components.forEach(function(c) {
    if (c._bbox) {
      actualMinX = Math.min(actualMinX, c._bbox.x);
      actualMinY = Math.min(actualMinY, c._bbox.y);
      actualMaxX = Math.max(actualMaxX, c._bbox.x + c._bbox.w);
      actualMaxY = Math.max(actualMaxY, c._bbox.y + c._bbox.h);
    }
  });
  // Internal content area
  var contentW = actualMaxX - actualMinX;
  var contentH = actualMaxY - actualMinY;
  var contentArea = contentW * contentH;
  // Margin: total margin area = content area → each side = sqrt(contentArea / 4)
  MOD_MARGIN = Math.max(60, Math.round(Math.sqrt(contentArea / 4)));
  // Recompute module bounds to wrap all content + margin
  modX = actualMinX - MOD_MARGIN;
  modY = actualMinY - MOD_MARGIN - 28; // -28 for title
  modW = contentW + MOD_MARGIN * 2;
  modH = contentH + MOD_MARGIN * 2 + 28;

  // Module border (behind everything)
  g.insert("rect", ":first-child")
    .attr("x", modX).attr("y", modY)
    .attr("width", modW).attr("height", modH)
    .attr("rx", 8).attr("fill", "rgba(20,10,40,0.25)")
    .attr("stroke", modBorderColor).attr("stroke-width", 2)
    .attr("stroke-dasharray", "6,3");

  // Module labels: type top-left (one or two lines based on width)
  var modTypeStr = "fabric.module";
  var modTypePx = modTypeStr.length * 8;
  if (modTypePx < modW * TYPE_LABEL_SINGLE_LINE_RATIO) {
    g.append("text").attr("x", modX + 10).attr("y", modY + 16)
      .attr("fill", modBorderColor).attr("font-size", "11px").attr("font-weight", "600")
      .text(modTypeStr);
  } else {
    g.append("text").attr("x", modX + 10).attr("y", modY + 14)
      .attr("fill", modBorderColor).attr("font-size", "11px").attr("font-weight", "600")
      .text("fabric");
    g.append("text").attr("x", modX + 10).attr("y", modY + 27)
      .attr("fill", modBorderColor).attr("font-size", "11px").attr("font-weight", "600")
      .text("module");
  }
  g.append("text").attr("x", modX + modW - 10).attr("y", modY + modH - 8)
    .attr("text-anchor", "end")
    .attr("fill", "rgba(155,89,182,0.6)").attr("font-size", "10px")
    .text("@" + (data.name || "adg"));

  // Module INPUT ports ON top border
  for (var i = 0; i < data.numInputs; i++) {
    var px = modX + modW * (i + 1) / (data.numInputs + 1);
    portRegistry["module_in_" + i] = {
      x: px, y: modY,
      side: "top", owner: "module", ownerKind: "module",
      nx: 0, ny: 1
    };
    g.append("rect").attr("x", px - 5).attr("y", modY - 5)
      .attr("width", 10).attr("height", 10)
      .attr("fill", modInPortColor).attr("stroke", modBorderColor).attr("stroke-width", 1.5);
    g.append("text").attr("x", px).attr("y", modY - 9)
      .attr("text-anchor", "middle").attr("fill", modInPortColor).attr("font-size", "8px")
      .text("I" + i);
  }

  // Module OUTPUT ports ON bottom border
  for (var i = 0; i < data.numOutputs; i++) {
    var px = modX + modW * (i + 1) / (data.numOutputs + 1);
    portRegistry["module_out_" + i] = {
      x: px, y: modY + modH,
      side: "bottom", owner: "module", ownerKind: "module",
      nx: 0, ny: -1
    };
    g.append("rect").attr("x", px - 5).attr("y", modY + modH - 5)
      .attr("width", 10).attr("height", 10)
      .attr("fill", modOutPortColor).attr("stroke", modBorderColor).attr("stroke-width", 1.5);
    g.append("text").attr("x", px).attr("y", modY + modH + 16)
      .attr("text-anchor", "middle").attr("fill", modOutPortColor).attr("font-size", "8px")
      .text("O" + i);
  }

  // Draw module-level connections with a routed orthogonal channel graph.
  // Edges stay inside the module, avoid hardware bodies, never share a routed
  // segment, and use hop-over arcs only at true crossings.
  if (data.connections) {
    var compBoxes = [];
    data.components.forEach(function(c) {
      if (c._bbox) compBoxes.push(c._bbox);
    });

    var edgeG = g.insert("g", ":first-child").attr("class", "hw-edges");
    var edgeHitG = g.append("g").attr("class", "hw-edge-hits");
    var GRID_STEP = 10;
    var BORDER_CLEARANCE = EDGE_BORDER_GAP + 10;
    var COMPONENT_CLEARANCE = 16;
    var PORT_STUB = 18;
    var LANE_SPACING = 18;
    var TURN_PENALTY = 1.8;
    var REVERSE_PENALTY = 2.6;
    var NEAR_BLOCK_PENALTY = 1.1;
    var REUSED_SEGMENT_PENALTY = 50;
    var HOP_R = 5;
    var HOP_MARGIN = HOP_R + 2;

    var routeLeft = modX + BORDER_CLEARANCE;
    var routeTop = modY + BORDER_CLEARANCE;
    var routeRight = modX + modW - BORDER_CLEARANCE;
    var routeBottom = modY + modH - BORDER_CLEARANCE;
    var gMinX = routeLeft;
    var gMinY = routeTop;
    var gCols = Math.max(1, Math.floor((routeRight - routeLeft) / GRID_STEP));
    var gRows = Math.max(1, Math.floor((routeBottom - routeTop) / GRID_STEP));

    var hardBlocked = {};
    var blockedPenalty = {};
    var usedGridEdges = {};

    function clamp(v, lo, hi) {
      return Math.max(lo, Math.min(hi, v));
    }
    function gk(gc, gr) {
      return gc + "," + gr;
    }
    function sk(gc, gr, dir) {
      return gc + "," + gr + "," + dir;
    }
    function edgeKey(gc0, gr0, gc1, gr1) {
      if (gc0 < gc1 || (gc0 === gc1 && gr0 <= gr1))
        return gc0 + "," + gr0 + "|" + gc1 + "," + gr1;
      return gc1 + "," + gr1 + "|" + gc0 + "," + gr0;
    }
    function pixToGrid(px, py) {
      return {
        gc: clamp(Math.round((px - gMinX) / GRID_STEP), 0, gCols),
        gr: clamp(Math.round((py - gMinY) / GRID_STEP), 0, gRows)
      };
    }
    function gridToPix(gc, gr) {
      return { x: gMinX + gc * GRID_STEP, y: gMinY + gr * GRID_STEP };
    }
    function isBlocked(gc, gr) {
      return gc < 0 || gr < 0 || gc > gCols || gr > gRows || !!hardBlocked[gk(gc, gr)];
    }
    function simplifyGridPath(path) {
      if (!path || path.length === 0) return [];
      var out = [path[0]];
      for (var i = 1; i < path.length; i++) {
        var prev = out[out.length - 1];
        var cur = path[i];
        if (prev.gc !== cur.gc || prev.gr !== cur.gr) out.push(cur);
      }
      if (out.length <= 2) return out;
      var simplified = [out[0]];
      for (var i = 1; i < out.length - 1; i++) {
        var a = simplified[simplified.length - 1];
        var b = out[i];
        var c = out[i + 1];
        if ((a.gc === b.gc && b.gc === c.gc) || (a.gr === b.gr && b.gr === c.gr))
          continue;
        simplified.push(b);
      }
      simplified.push(out[out.length - 1]);
      return simplified;
    }
    function cleanPolyline(points) {
      if (!points || points.length === 0) return [];
      var deduped = [];
      points.forEach(function(pt) {
        if (!deduped.length) {
          deduped.push({ x: pt.x, y: pt.y });
          return;
        }
        var prev = deduped[deduped.length - 1];
        if (Math.abs(prev.x - pt.x) > 0.5 || Math.abs(prev.y - pt.y) > 0.5)
          deduped.push({ x: pt.x, y: pt.y });
      });
      if (deduped.length <= 2) return deduped;
      var simplified = [deduped[0]];
      for (var i = 1; i < deduped.length - 1; i++) {
        var a = simplified[simplified.length - 1];
        var b = deduped[i];
        var c = deduped[i + 1];
        if ((Math.abs(a.x - b.x) < 0.5 && Math.abs(b.x - c.x) < 0.5) ||
            (Math.abs(a.y - b.y) < 0.5 && Math.abs(b.y - c.y) < 0.5))
          continue;
        simplified.push(b);
      }
      simplified.push(deduped[deduped.length - 1]);
      return simplified;
    }
    function findGridNodeNear(px, py, normal) {
      var base = pixToGrid(px, py);
      for (var radius = 0; radius <= 14; radius++) {
        var best = null;
        for (var dc = -radius; dc <= radius; dc++) {
          for (var dr = -radius; dr <= radius; dr++) {
            if (Math.abs(dc) + Math.abs(dr) > radius) continue;
            var gc = base.gc + dc;
            var gr = base.gr + dr;
            if (isBlocked(gc, gr)) continue;
            var score = Math.abs(dc) + Math.abs(dr);
            if (normal)
              score -= (dc * normal.nx + dr * normal.ny) * 0.15;
            if (!best || score < best.score)
              best = { gc: gc, gr: gr, score: score };
          }
        }
        if (best) return { gc: best.gc, gr: best.gr };
      }
      return base;
    }
    function anchorForPort(port, lane) {
      var offset = PORT_STUB + lane * LANE_SPACING;
      var px = clamp(port.x + port.nx * offset, routeLeft, routeRight);
      var py = clamp(port.y + port.ny * offset, routeTop, routeBottom);
      var grid = findGridNodeNear(px, py, port);
      var snapped = gridToPix(grid.gc, grid.gr);
      return { gc: grid.gc, gr: grid.gr, x: snapped.x, y: snapped.y };
    }
    function reserveGridPath(path) {
      for (var i = 0; i < path.length - 1; i++) {
        usedGridEdges[edgeKey(path[i].gc, path[i].gr, path[i + 1].gc, path[i + 1].gr)] = true;
      }
    }
    function routeGrid(start, goal, allowReuse) {
      var dirs = [
        { dc: 1, dr: 0 },
        { dc: -1, dr: 0 },
        { dc: 0, dr: 1 },
        { dc: 0, dr: -1 }
      ];
      var open = [];
      var bestCost = {};
      var prev = {};
      var closed = {};
      var startKey = sk(start.gc, start.gr, -1);
      bestCost[startKey] = 0;
      heapPush(open, { gc: start.gc, gr: start.gr, dir: -1, f: 0 });

      var goalKey = null;
      while (open.length > 0) {
        var cur = heapPop(open);
        var curKey = sk(cur.gc, cur.gr, cur.dir);
        if (closed[curKey]) continue;
        closed[curKey] = true;

        if (cur.gc === goal.gc && cur.gr === goal.gr) {
          goalKey = curKey;
          break;
        }

        var curCost = bestCost[curKey];
        for (var di = 0; di < dirs.length; di++) {
          var nc = cur.gc + dirs[di].dc;
          var nr = cur.gr + dirs[di].dr;
          if (isBlocked(nc, nr)) continue;

          var segKey = edgeKey(cur.gc, cur.gr, nc, nr);
          if (!allowReuse && usedGridEdges[segKey]) continue;

          var stepCost = 1 + (blockedPenalty[gk(nc, nr)] || 0);
          if (cur.dir >= 0 && di !== cur.dir) stepCost += TURN_PENALTY;
          if (cur.dir >= 0 && di === (cur.dir ^ 1)) stepCost += REVERSE_PENALTY;
          if (allowReuse && usedGridEdges[segKey]) stepCost += REUSED_SEGMENT_PENALTY;

          var nextKey = sk(nc, nr, di);
          var newCost = curCost + stepCost;
          if (bestCost[nextKey] !== undefined && newCost >= bestCost[nextKey]) continue;

          bestCost[nextKey] = newCost;
          prev[nextKey] = curKey;
          var heuristic = Math.abs(nc - goal.gc) + Math.abs(nr - goal.gr);
          heapPush(open, { gc: nc, gr: nr, dir: di, f: newCost + heuristic });
        }
      }

      if (!goalKey) return null;

      var path = [];
      var key = goalKey;
      while (key) {
        var parts = key.split(",");
        path.unshift({ gc: parseInt(parts[0], 10), gr: parseInt(parts[1], 10) });
        key = prev[key];
      }
      return path;
    }
    function routeAnchors(start, goal) {
      return routeGrid(start, goal, false) || routeGrid(start, goal, true);
    }
    function buildFallbackPath(startAnchor, endAnchor) {
      var bendA = { x: startAnchor.x, y: endAnchor.y };
      var bendB = { x: endAnchor.x, y: startAnchor.y };
      var da = Math.abs(startAnchor.y - endAnchor.y);
      var db = Math.abs(startAnchor.x - endAnchor.x);
      return db < da ? [startAnchor, bendB, endAnchor] : [startAnchor, bendA, endAnchor];
    }
    function connectPortToAnchor(port, anchor) {
      var pts = [{ x: port.x, y: port.y }];
      if (port.nx !== 0) {
        if (Math.abs(anchor.x - port.x) > 0.5)
          pts.push({ x: anchor.x, y: port.y });
        if (Math.abs(anchor.y - port.y) > 0.5)
          pts.push({ x: anchor.x, y: anchor.y });
      } else {
        if (Math.abs(anchor.y - port.y) > 0.5)
          pts.push({ x: port.x, y: anchor.y });
        if (Math.abs(anchor.x - port.x) > 0.5)
          pts.push({ x: anchor.x, y: anchor.y });
      }
      return pts;
    }
    function connectAnchorToPort(anchor, port) {
      var pts = [{ x: anchor.x, y: anchor.y }];
      if (port.nx !== 0) {
        if (Math.abs(anchor.y - port.y) > 0.5)
          pts.push({ x: anchor.x, y: port.y });
        if (Math.abs(anchor.x - port.x) > 0.5)
          pts.push({ x: port.x, y: port.y });
      } else {
        if (Math.abs(anchor.x - port.x) > 0.5)
          pts.push({ x: port.x, y: anchor.y });
        if (Math.abs(anchor.y - port.y) > 0.5)
          pts.push({ x: port.x, y: port.y });
      }
      return pts;
    }
    function buildEdgePolyline(fromPort, toPort, startAnchor, endAnchor, gridPath) {
      var pts = connectPortToAnchor(fromPort, startAnchor);
      if (gridPath && gridPath.length > 0) {
        var routePts = simplifyGridPath(gridPath).map(function(p) { return gridToPix(p.gc, p.gr); });
        routePts.forEach(function(p, idx) {
          if (idx === 0 || idx === routePts.length - 1) return;
          pts.push(p);
        });
      } else {
        buildFallbackPath(startAnchor, endAnchor).forEach(function(p, idx) {
          if (idx === 0 || idx === 2) return;
          pts.push(p);
        });
      }
      pts.push({ x: endAnchor.x, y: endAnchor.y });
      connectAnchorToPort(endAnchor, toPort).forEach(function(p, idx) {
        if (idx === 0) return;
        pts.push(p);
      });
      return cleanPolyline(pts);
    }
    function findCrossings(edgeA, edgeB) {
      var crosses = [];
      for (var i = 0; i < edgeA.pts.length - 1; i++) {
        for (var j = 0; j < edgeB.pts.length - 1; j++) {
          var a1 = edgeA.pts[i], a2 = edgeA.pts[i + 1];
          var b1 = edgeB.pts[j], b2 = edgeB.pts[j + 1];
          var aH = Math.abs(a1.y - a2.y) < 0.5;
          var bH = Math.abs(b1.y - b2.y) < 0.5;
          if (aH === bH) continue;

          if (aH) {
            var y = a1.y;
            var x = b1.x;
            var aMinX = Math.min(a1.x, a2.x), aMaxX = Math.max(a1.x, a2.x);
            var bMinY = Math.min(b1.y, b2.y), bMaxY = Math.max(b1.y, b2.y);
            if (x > aMinX + HOP_MARGIN && x < aMaxX - HOP_MARGIN &&
                y > bMinY + HOP_MARGIN && y < bMaxY - HOP_MARGIN) {
              crosses.push({ segIdx: i, x: x, y: y, isHoriz: true });
            }
          } else {
            var x = a1.x;
            var y = b1.y;
            var aMinY = Math.min(a1.y, a2.y), aMaxY = Math.max(a1.y, a2.y);
            var bMinX = Math.min(b1.x, b2.x), bMaxX = Math.max(b1.x, b2.x);
            if (y > aMinY + HOP_MARGIN && y < aMaxY - HOP_MARGIN &&
                x > bMinX + HOP_MARGIN && x < bMaxX - HOP_MARGIN) {
              crosses.push({ segIdx: i, x: x, y: y, isHoriz: false });
            }
          }
        }
      }
      return crosses;
    }
    function buildEdgePathD(pts, crosses) {
      var d = "M" + pts[0].x + "," + pts[0].y;
      for (var si = 0; si < pts.length - 1; si++) {
        var p1 = pts[si];
        var p2 = pts[si + 1];
        var isH = Math.abs(p1.y - p2.y) < 0.5;
        var segCrosses = crosses.filter(function(c) { return c.segIdx === si; });
        if (segCrosses.length === 0) {
          d += "L" + p2.x + "," + p2.y;
          continue;
        }

        segCrosses.sort(function(a, b) {
          if (isH)
            return p1.x < p2.x ? a.x - b.x : b.x - a.x;
          return p1.y < p2.y ? a.y - b.y : b.y - a.y;
        });

        var cursor = { x: p1.x, y: p1.y };
        segCrosses.forEach(function(cx) {
          if (isH) {
            var dir = p2.x > p1.x ? 1 : -1;
            var beforeX = cx.x - dir * HOP_R;
            var afterX = cx.x + dir * HOP_R;
            if ((dir > 0 && beforeX <= cursor.x + 0.5) || (dir < 0 && beforeX >= cursor.x - 0.5))
              return;
            d += "L" + beforeX + "," + cx.y;
            d += "A" + HOP_R + "," + HOP_R + " 0 0 " + (dir > 0 ? "1" : "0") + " "
              + afterX + "," + cx.y;
            cursor = { x: afterX, y: cx.y };
          } else {
            var dir = p2.y > p1.y ? 1 : -1;
            var beforeY = cx.y - dir * HOP_R;
            var afterY = cx.y + dir * HOP_R;
            if ((dir > 0 && beforeY <= cursor.y + 0.5) || (dir < 0 && beforeY >= cursor.y - 0.5))
              return;
            d += "L" + cx.x + "," + beforeY;
            d += "A" + HOP_R + "," + HOP_R + " 0 0 " + (dir > 0 ? "0" : "1") + " "
              + cx.x + "," + afterY;
            cursor = { x: cx.x, y: afterY };
          }
        });

        if (Math.abs(cursor.x - p2.x) > 0.5 || Math.abs(cursor.y - p2.y) > 0.5)
          d += "L" + p2.x + "," + p2.y;
      }
      return d;
    }

    compBoxes.forEach(function(box) {
      var c0 = Math.floor((box.x - COMPONENT_CLEARANCE - gMinX) / GRID_STEP);
      var r0 = Math.floor((box.y - COMPONENT_CLEARANCE - gMinY) / GRID_STEP);
      var c1 = Math.ceil((box.x + box.w + COMPONENT_CLEARANCE - gMinX) / GRID_STEP);
      var r1 = Math.ceil((box.y + box.h + COMPONENT_CLEARANCE - gMinY) / GRID_STEP);
      for (var gc = c0; gc <= c1; gc++) {
        for (var gr = r0; gr <= r1; gr++) {
          if (gc < 0 || gr < 0 || gc > gCols || gr > gRows) continue;
          hardBlocked[gk(gc, gr)] = true;
        }
      }
    });

    for (var gc = 0; gc <= gCols; gc++) {
      for (var gr = 0; gr <= gRows; gr++) {
        if (hardBlocked[gk(gc, gr)]) continue;
        var penalty = 0;
        for (var dc = -2; dc <= 2; dc++) {
          for (var dr = -2; dr <= 2; dr++) {
            var dist = Math.abs(dc) + Math.abs(dr);
            if (dist === 0 || dist > 2) continue;
            if (hardBlocked[gk(gc + dc, gr + dr)])
              penalty = Math.max(penalty, dist === 1 ? NEAR_BLOCK_PENALTY : 0.35);
          }
        }
        if (penalty > 0) blockedPenalty[gk(gc, gr)] = penalty;
      }
    }

    var sourceUse = {};
    var destUse = {};
    var plannedEdges = [];
    data.connections.forEach(function(conn, idx) {
      var fromKey = conn.from === "module_in" ? "module_in_" + conn.fromIdx : conn.from + "_out_" + conn.fromIdx;
      var toKey = conn.to === "module_out" ? "module_out_" + conn.toIdx : conn.to + "_in_" + conn.toIdx;
      var fromPort = portRegistry[fromKey];
      var toPort = portRegistry[toKey];
      if (!fromPort || !toPort) return;

      var srcLane = sourceUse[fromKey] || 0;
      var dstLane = destUse[toKey] || 0;
      sourceUse[fromKey] = srcLane + 1;
      destUse[toKey] = dstLane + 1;

      plannedEdges.push({
        originalIndex: idx,
        fromKey: fromKey,
        toKey: toKey,
        fromPort: fromPort,
        toPort: toPort,
        srcLane: srcLane,
        dstLane: dstLane
      });
    });

    plannedEdges.sort(function(a, b) {
      var aBoundary = (a.fromPort.ownerKind === "module" ? 1 : 0) + (a.toPort.ownerKind === "module" ? 1 : 0);
      var bBoundary = (b.fromPort.ownerKind === "module" ? 1 : 0) + (b.toPort.ownerKind === "module" ? 1 : 0);
      if (aBoundary !== bBoundary) return aBoundary - bBoundary;
      var aSpan = Math.abs(a.fromPort.x - a.toPort.x) + Math.abs(a.fromPort.y - a.toPort.y);
      var bSpan = Math.abs(b.fromPort.x - b.toPort.x) + Math.abs(b.fromPort.y - b.toPort.y);
      if (aSpan !== bSpan) return aSpan - bSpan;
      return a.originalIndex - b.originalIndex;
    });

    var allEdgePts = [];
    plannedEdges.forEach(function(edgePlan) {
      var startAnchor = anchorForPort(edgePlan.fromPort, edgePlan.srcLane);
      var endAnchor = anchorForPort(edgePlan.toPort, edgePlan.dstLane);
      var gridPath = routeAnchors(startAnchor, endAnchor);
      if (gridPath) reserveGridPath(gridPath);

      allEdgePts.push({
        order: edgePlan.originalIndex,
        fromKey: edgePlan.fromKey,
        toKey: edgePlan.toKey,
        pts: buildEdgePolyline(edgePlan.fromPort, edgePlan.toPort, startAnchor, endAnchor, gridPath)
      });
    });

    var allCrossings = allEdgePts.map(function() { return []; });
    for (var ai = 0; ai < allEdgePts.length; ai++) {
      for (var bi = ai + 1; bi < allEdgePts.length; bi++) {
        findCrossings(allEdgePts[bi], allEdgePts[ai]).forEach(function(c) {
          allCrossings[bi].push(c);
        });
      }
    }

    allEdgePts.forEach(function(edge, ei) {
      var pathD = buildEdgePathD(edge.pts, allCrossings[ei]);
      edgeG.append("path")
        .attr("d", pathD).attr("fill", "none")
        .attr("stroke", "#8a9bb5").attr("stroke-width", 1.5)
        .attr("stroke-linecap", "round").attr("stroke-linejoin", "round")
        .attr("opacity", 0.6).attr("class", "hw-edge-vis")
        .attr("data-from", edge.fromKey).attr("data-to", edge.toKey);

      edgeHitG.append("path")
        .attr("d", pathD).attr("fill", "none")
        .attr("stroke", "rgba(0,0,0,0)").attr("stroke-width", 14)
        .style("cursor", "pointer").style("pointer-events", "all")
        .on("click", function(ev) {
          ev.stopPropagation();
          selectEdge(edge.fromKey, edge.toKey);
        });
    });

  }

  if (mappingEnabled && switchRouteIdx && switchRouteIdx.routes.length > 0) {
    var switchRouteG = g.append("g").attr("class", "sw-internal-routes");
    drawActiveSwitchRoutes(switchRouteG);
  }

  // Selection state
  var selectedFrom = null, selectedTo = null;

  function selectEdge(fk, tk) {
    clearSelection();
    selectedFrom = fk; selectedTo = tk;
    // Highlight the matching visible edge
    g.selectAll(".hw-edge-vis").each(function() {
      var el = d3.select(this);
      if (el.attr("data-from") === fk && el.attr("data-to") === tk) {
        el.attr("stroke", "#4ecdc4").attr("stroke-width", 3).attr("opacity", 1);
      }
    });
    // Highlight both ports with a glow ring
    drawPortGlow(fk);
    drawPortGlow(tk);
  }

  function drawPortGlow(key) {
    var p = portRegistry[key];
    if (!p) return;
    g.append("rect")
      .attr("x", p.x - 8).attr("y", p.y - 8)
      .attr("width", 16).attr("height", 16).attr("rx", 3)
      .attr("fill", "none").attr("stroke", "#4ecdc4")
      .attr("stroke-width", 2.5).attr("class", "port-glow")
      .style("filter", "drop-shadow(0 0 4px #4ecdc4)");
  }

  function clearSelection() {
    selectedFrom = null; selectedTo = null;
    g.selectAll(".hw-edge-vis")
      .attr("stroke", "#8a9bb5").attr("stroke-width", 1.5).attr("opacity", 0.6);
    g.selectAll(".port-glow").remove();
  }

  // Also allow clicking a port to highlight its edge
  function setupPortClick(key) {
    var p = portRegistry[key];
    if (!p) return;
    g.append("rect")
      .attr("x", p.x - 6).attr("y", p.y - 6)
      .attr("width", 12).attr("height", 12)
      .attr("fill", "rgba(0,0,0,0)").style("cursor", "pointer")
      .on("click", function(ev) {
        ev.stopPropagation();
        // Find edge connected to this port
        g.selectAll(".hw-edge-vis").each(function() {
          var el = d3.select(this);
          if (el.attr("data-from") === key || el.attr("data-to") === key) {
            selectEdge(el.attr("data-from"), el.attr("data-to"));
          }
        });
      });
  }
  Object.keys(portRegistry).forEach(function(k) { setupPortClick(k); });

  // ADG component click -> cross-highlight mapped DFG nodes
  if (MAPPING_DATA) {
    g.selectAll("rect[data-comp-name]").each(function() {
      var el = d3.select(this);
      var compName = el.attr("data-comp-name");
      // Add invisible hit area over the component
      var bbox = adgCompBoxes[compName];
      if (!bbox) return;
      // Shrink hit area to 90% so border port clicks aren't intercepted
      var shrink = 0.05;
      var hx = bbox.x + bbox.w * shrink, hy = bbox.y + bbox.h * shrink;
      var hw = bbox.w * (1 - 2*shrink), hh = bbox.h * (1 - 2*shrink);
      g.append("rect")
        .attr("x", hx).attr("y", hy)
        .attr("width", hw).attr("height", hh)
        .attr("fill", "rgba(0,0,0,0)").style("cursor", "pointer")
        .style("pointer-events", "all").attr("class", "comp-hit")
        .on("click", function(ev) {
          ev.stopPropagation();
          onADGCompClick(compName);
        });
    });
  }

  // Click background to clear
  svg.on("click", function() { clearSelection(); clearCrossHighlight(); });

  // Arrow marker for DAG edges
  svg.append("defs").append("marker")
    .attr("id", "arrow").attr("viewBox", "0 0 6 6")
    .attr("refX", 5).attr("refY", 3)
    .attr("markerWidth", 5).attr("markerHeight", 5)
    .attr("orient", "auto")
    .append("path").attr("d", "M0,0 L6,3 L0,6 Z")
    .attr("fill", "rgba(255,255,255,0.3)");

  d3.select("#status-bar").text("Ready");

  // Fit to view
  setTimeout(function() { fitView(svg, g, zoom); }, 100);
}

// ============================================================
// DFG Renderer: direct SVG with typed ports/edges
// ============================================================

renderDFG._svg = null;
renderDFG._g = null;
renderDFG._zoom = null;

function getDFGNodePalette(node) {
  if (node.kind === "input") {
    return { fill: "#214c73", stroke: "#7ec8ff", text: "#d9ecff" };
  }
  if (node.kind === "output") {
    return { fill: "#6a3023", stroke: "#ff9b7a", text: "#ffe1d8" };
  }
  var opName = node.op || node.label || "";
  if (opName.indexOf("arith.") === 0) {
    return { fill: "#1a3050", stroke: "#5dade2", text: "#d8ebff" };
  }
  if (opName.indexOf("handshake.") === 0) {
    return { fill: "#3a3520", stroke: "#ffd166", text: "#ffe29b" };
  }
  if (opName.indexOf("dataflow.") === 0) {
    return { fill: "#203c2f", stroke: "#58d68d", text: "#d9f7e6" };
  }
  return { fill: "#293548", stroke: "#8aa4c2", text: "#c8d6e5" };
}

function getDFGEdgeStyle(edgeType) {
  if (edgeType === "memref") {
    return { color: "#4f8cff", dash: "2,4" };
  }
  if (edgeType === "control") {
    return { color: "#9aa3b2", dash: "6,4" };
  }
  return { color: "#7ec8ff", dash: null };
}

function estimateDFGNodeSize(node) {
  var inputs = Array.isArray(node.inputs) ? node.inputs : [];
  var outputs = Array.isArray(node.outputs) ? node.outputs : [];
  if (node.kind === "input" || node.kind === "output") {
    return { w: 132, h: 62 };
  }

  var labelLines = splitDisplayLabel(node.display || node.label || "");
  var maxLabelLen = 0;
  labelLines.forEach(function(line) { maxLabelLen = Math.max(maxLabelLen, line.length); });
  var maxPortCount = Math.max(inputs.length, outputs.length, 1);
  var w = Math.max(128, maxLabelLen * 8 + 36, maxPortCount * 58);
  return { w: w, h: 82 };
}

function buildDFGLayout(data) {
  var nodes = (data.nodes || []).map(function(node) {
    var copy = {};
    Object.keys(node).forEach(function(key) { copy[key] = node[key]; });
    copy.inputs = Array.isArray(node.inputs) ? node.inputs.slice() : [];
    copy.outputs = Array.isArray(node.outputs) ? node.outputs.slice() : [];
    copy.size = estimateDFGNodeSize(copy);
    copy.order = copy.id;
    copy.level = copy.kind === "input" ? 0 : null;
    return copy;
  });
  var edges = (data.edges || []).map(function(edge) {
    var copy = {};
    Object.keys(edge).forEach(function(key) { copy[key] = edge[key]; });
    return copy;
  });

  var nodeById = {};
  nodes.forEach(function(node) { nodeById[node.id] = node; });

  var forwardEdges = [];
  var indegree = {};
  nodes.forEach(function(node) { indegree[node.id] = 0; });
  edges.forEach(function(edge) {
    if (edge.from < edge.to) {
      forwardEdges.push(edge);
      indegree[edge.to] += 1;
    }
  });

  var succ = {};
  nodes.forEach(function(node) { succ[node.id] = []; });
  forwardEdges.forEach(function(edge) { succ[edge.from].push(edge); });

  var queue = nodes.filter(function(node) { return indegree[node.id] === 0; })
    .sort(function(a, b) { return a.order - b.order; });

  while (queue.length > 0) {
    var cur = queue.shift();
    var curLevel = cur.level == null ? 0 : cur.level;
    succ[cur.id].forEach(function(edge) {
      var dst = nodeById[edge.to];
      if (!dst) return;
      var nextLevel = curLevel + 1;
      if (dst.level == null || dst.level < nextLevel) dst.level = nextLevel;
      indegree[dst.id] -= 1;
      if (indegree[dst.id] === 0) queue.push(dst);
    });
    queue.sort(function(a, b) { return a.order - b.order; });
  }

  nodes.forEach(function(node) {
    if (node.level == null) node.level = node.kind === "output" ? 2 : 1;
  });

  var maxPredLevel = 0;
  edges.forEach(function(edge) {
    var src = nodeById[edge.from];
    var dst = nodeById[edge.to];
    if (!src || !dst) return;
    if (dst.kind === "output") {
      dst.level = Math.max(dst.level, src.level + 1);
    }
    maxPredLevel = Math.max(maxPredLevel, src.level, dst.level);
  });

  var rows = {};
  nodes.forEach(function(node) {
    if (!rows[node.level]) rows[node.level] = [];
    rows[node.level].push(node);
  });

  Object.keys(rows).forEach(function(key) {
    rows[key].sort(function(a, b) { return a.order - b.order; });
  });

  var levelKeys = Object.keys(rows).map(function(v) { return parseInt(v, 10); })
    .sort(function(a, b) { return a - b; });
  var rowGap = 170;
  var topPad = 96;
  var baseWidth = 760;

  levelKeys.forEach(function(level) {
    var row = rows[level];
    var rowWidth = Math.max(baseWidth, row.length * 220);
    var step = rowWidth / (row.length + 1);
    row.forEach(function(node, idx) {
      node.x = 80 + step * (idx + 1);
      node.y = topPad + level * rowGap;
    });
  });

  return { nodes: nodes, edges: edges, nodeById: nodeById };
}

function getDFGPortAnchor(node, dir, portIndex) {
  if (node.kind === "input") {
    return { x: node.x, y: node.y + node.size.h / 2 - 2 };
  }
  if (node.kind === "output") {
    return { x: node.x, y: node.y - node.size.h / 2 + 2 };
  }

  var ports = dir === "in" ? node.inputs : node.outputs;
  var count = Math.max(ports.length, 1);
  var slot = (portIndex == null ? 0 : portIndex) + 1;
  var x = node.x - node.size.w / 2 + (node.size.w * slot) / (count + 1);
  var y = dir === "in" ? node.y - node.size.h / 2 + 2 : node.y + node.size.h / 2 - 2;
  return { x: x, y: y };
}

function buildDFGEdgePath(edge, nodeById) {
  var src = nodeById[edge.from];
  var dst = nodeById[edge.to];
  if (!src || !dst) return "";

  var srcPt = getDFGPortAnchor(src, "out", edge.from_port || 0);
  var dstPt = getDFGPortAnchor(dst, "in", edge.to_port || 0);
  var isForward = srcPt.y < dstPt.y - 8;
  var bend = 70 + (edge.id % 3) * 22;
  if (isForward) {
    var midY = (srcPt.y + dstPt.y) / 2;
    return "M" + srcPt.x + "," + srcPt.y +
      " C" + srcPt.x + "," + midY + " " + dstPt.x + "," + midY + " " + dstPt.x + "," + dstPt.y;
  }

  var bendX = Math.max(srcPt.x, dstPt.x) + bend;
  return "M" + srcPt.x + "," + srcPt.y +
    " C" + bendX + "," + srcPt.y + " " + bendX + "," + dstPt.y + " " + dstPt.x + "," + dstPt.y;
}

function buildDFGArrowPoints(edge, nodeById) {
  var src = nodeById[edge.from];
  var dst = nodeById[edge.to];
  if (!src || !dst) return "";
  var srcPt = getDFGPortAnchor(src, "out", edge.from_port || 0);
  var dstPt = getDFGPortAnchor(dst, "in", edge.to_port || 0);
  var isForward = srcPt.y < dstPt.y - 8;
  var dx = isForward ? 0 : -1;
  var dy = isForward ? (dstPt.y >= srcPt.y ? 1 : -1) : 0;
  var len = 8;
  var wing = 4;
  return [
    dstPt.x + "," + dstPt.y,
    (dstPt.x - dx * len + dy * wing) + "," + (dstPt.y - dy * len - dx * wing),
    (dstPt.x - dx * len - dy * wing) + "," + (dstPt.y - dy * len + dx * wing)
  ].join(" ");
}

function drawDFGNode(parentG, node) {
  var palette = getDFGNodePalette(node);
  var group = parentG.append("g")
    .attr("class", "node dfg-node")
    .attr("data-dfg-node", node.id)
    .style("cursor", "pointer");

  if (node.kind === "input") {
    group.append("polygon")
      .attr("class", "dfg-node-shape")
      .attr("points", [
        node.x + "," + (node.y + node.size.h / 2),
        (node.x - node.size.w / 2) + "," + (node.y - node.size.h / 2),
        (node.x + node.size.w / 2) + "," + (node.y - node.size.h / 2)
      ].join(" "))
      .attr("fill", palette.fill)
      .attr("stroke", palette.stroke)
      .attr("stroke-width", 1.6);
  } else if (node.kind === "output") {
    group.append("polygon")
      .attr("class", "dfg-node-shape")
      .attr("points", [
        (node.x - node.size.w / 2) + "," + (node.y + node.size.h / 2),
        (node.x + node.size.w / 2) + "," + (node.y + node.size.h / 2),
        node.x + "," + (node.y - node.size.h / 2)
      ].join(" "))
      .attr("fill", palette.fill)
      .attr("stroke", palette.stroke)
      .attr("stroke-width", 1.6);
  } else {
    group.append("ellipse")
      .attr("class", "dfg-node-shape")
      .attr("cx", node.x).attr("cy", node.y)
      .attr("rx", node.size.w / 2).attr("ry", node.size.h / 2)
      .attr("fill", palette.fill)
      .attr("stroke", palette.stroke)
      .attr("stroke-width", 1.6);
  }

  var lines;
  if (node.kind === "input") {
    lines = [node.label, shortenText((node.name ? node.name + " : " : "") + (node.type || ""), 28)];
  } else if (node.kind === "output") {
    lines = [node.label, shortenText((node.name ? node.name + " : " : "") + (node.type || ""), 28)];
  } else {
    lines = splitDisplayLabel(node.display || node.label || "");
  }

  lines.forEach(function(line, idx) {
    group.append("text")
      .attr("x", node.x)
      .attr("y", node.y - (lines.length - 1) * 8 + idx * 16 + 4)
      .attr("text-anchor", "middle")
      .attr("fill", palette.text)
      .attr("font-size", idx === lines.length - 1 && (node.kind === "input" || node.kind === "output") ? "10px" : "11px")
      .attr("font-weight", idx === 0 ? 600 : 400)
      .text(line);
  });

  if (node.kind === "op") {
    var portLabelSize = "8px";
    node.inputs.forEach(function(port, idx) {
      var anchor = getDFGPortAnchor(node, "in", idx);
      group.append("circle")
        .attr("cx", anchor.x).attr("cy", anchor.y)
        .attr("r", 2.2).attr("fill", getDFGEdgeStyle(port.type === "none" ? "control" : (port.type && port.type.indexOf("memref") === 0 ? "memref" : "data")).color);
      group.append("text")
        .attr("x", anchor.x).attr("y", anchor.y - 7)
        .attr("text-anchor", "middle")
        .attr("fill", "rgba(200,214,229,0.8)")
        .attr("font-size", portLabelSize)
        .text(shortenText(port.name || ("I" + idx), 10));
    });
    node.outputs.forEach(function(port, idx) {
      var anchor = getDFGPortAnchor(node, "out", idx);
      group.append("circle")
        .attr("cx", anchor.x).attr("cy", anchor.y)
        .attr("r", 2.2).attr("fill", getDFGEdgeStyle(port.type === "none" ? "control" : (port.type && port.type.indexOf("memref") === 0 ? "memref" : "data")).color);
      group.append("text")
        .attr("x", anchor.x).attr("y", anchor.y + 14)
        .attr("text-anchor", "middle")
        .attr("fill", "rgba(200,214,229,0.8)")
        .attr("font-size", portLabelSize)
        .text(shortenText(port.name || ("O" + idx), 10));
    });
  }

  var titleLines = [];
  if (node.kind === "input" || node.kind === "output") {
    titleLines.push(node.label);
    if (node.name) titleLines.push(node.name);
    if (node.type) titleLines.push(node.type);
  } else {
    titleLines.push(node.label || "");
    node.inputs.forEach(function(port, idx) {
      titleLines.push("in" + idx + ": " + (port.name || "") + " : " + (port.type || ""));
    });
    node.outputs.forEach(function(port, idx) {
      titleLines.push("out" + idx + ": " + (port.name || "") + " : " + (port.type || ""));
    });
  }
  group.append("title").text(titleLines.join("\n"));

  return group;
}

function renderDFG() {
  var svg = d3.select("#svg-dfg");
  svg.selectAll("*").remove();
  if (!DFG_DATA || DFG_DATA === "null" || !DFG_DATA.nodes || DFG_DATA.nodes.length === 0) {
    svg.append("text").attr("x", 20).attr("y", 40).attr("fill", "#888")
      .text("No DFG data.");
    return;
  }

  d3.select("#status-bar").text("Rendering DFG...");

  var model = buildDFGLayout(DFG_DATA);
  var g = svg.append("g").attr("class", "dfg-root");
  var edgeG = g.append("g").attr("class", "dfg-edges");
  var nodeG = g.append("g").attr("class", "dfg-nodes");

  var zoom = d3.zoom().scaleExtent([0.1, 5])
    .on("zoom", function(ev) { g.attr("transform", ev.transform); });
  svg.call(zoom);

  model.edges.forEach(function(edge) {
    var style = getDFGEdgeStyle(edge.edge_type);
    var path = buildDFGEdgePath(edge, model.nodeById);
    var group = edgeG.append("g")
      .attr("class", "edge dfg-edge")
      .attr("data-dfg-edge", edge.id);
    group.append("path")
      .attr("d", path)
      .attr("fill", "none")
      .attr("stroke", style.color)
      .attr("stroke-width", 1.7)
      .attr("stroke-linecap", "round")
      .attr("stroke-linejoin", "round")
      .attr("stroke-dasharray", style.dash);
    group.append("path")
      .attr("d", path)
      .attr("fill", "none")
      .attr("stroke", "rgba(0,0,0,0)")
      .attr("stroke-width", 12)
      .attr("class", "dfg-edge-hit")
      .style("cursor", "pointer")
      .on("click", function(ev) {
        ev.stopPropagation();
        onDFGEdgeDataClick(edge);
      });
    group.append("polygon")
      .attr("points", buildDFGArrowPoints(edge, model.nodeById))
      .attr("fill", style.color)
      .attr("stroke", style.color);
    group.append("title").text((edge.edge_type || "data") + " : " + (edge.value_type || ""));
  });

  model.nodes.forEach(function(node) {
    drawDFGNode(nodeG, node)
      .on("click", function(ev) {
        ev.stopPropagation();
        onDFGNodeClick(node.id);
      });
  });

  svg.on("click", function() { clearCrossHighlight(); });

  renderDFG._svg = svg;
  renderDFG._g = g;
  renderDFG._zoom = zoom;

  setTimeout(function() { fitView(svg, g, zoom); }, 100);
  d3.select("#status-bar").text("Ready");
}

// ============================================================
// Utilities
// ============================================================

function fitView(svg, g, zoom) {
  var gNode = g.node();
  if (!gNode) return;
  var bbox = gNode.getBBox();
  if (bbox.width === 0 || bbox.height === 0) return;
  var svgNode = svg.node();
  var svgW = svgNode.clientWidth || 800;
  var svgH = svgNode.clientHeight || 600;
  var pad = 40;
  var scale = Math.min((svgW - pad*2) / bbox.width, (svgH - pad*2) / bbox.height, 2);
  var tx = svgW/2 - (bbox.x + bbox.width/2) * scale;
  var ty = svgH/2 - (bbox.y + bbox.height/2) * scale;
  svg.transition().duration(400)
    .call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}

// ============================================================
// Cross-highlighting (side-by-side mode)
// ============================================================

function setupDFGInteraction() {
  return;
}

// ============================================================
// PE internal routing (mux/demux connections)
// Uses A* grid routing at PE scale, same approach as module level
// ============================================================

function routePEInternal(edgeG, peLabel, peX, peY, peW, peH,
                         fus, fuHeights, fuWidths, fuBoxMargin, mappingMargin,
                         portRegistry, adgData) {
  var peConnections = peRouteIdx && peRouteIdx.byPE[peLabel]
    ? peRouteIdx.byPE[peLabel].slice()
    : [];
  if (peConnections.length === 0) return;

  peConnections.sort(function(a, b) {
    if (a.direction !== b.direction) return a.direction < b.direction ? -1 : 1;
    if (a.pePortKey !== b.pePortKey) return a.pePortKey < b.pePortKey ? -1 : 1;
    if (a.fuPortKey !== b.fuPortKey) return a.fuPortKey < b.fuPortKey ? -1 : 1;
    return a.swEdge - b.swEdge;
  });

  // Build FU obstacle boxes inside the PE.
  var fuObstacles = [];
  var fuStartX = peX + 20 + (mappingMargin / 2);
  var fxCur = fuStartX;
  var fuYStart = peY + 30;
  fus.forEach(function(fu, fi) {
    fuObstacles.push({
      name: fu.name,
      x: fxCur,
      y: fuYStart,
      w: fuWidths[fi],
      h: fuHeights[fi]
    });
    fxCur += fuWidths[fi] + fuBoxMargin;
  });

  // A* grid routing within PE bounds
  var GS = 6; // Grid step (finer than module level)
  var CLEARANCE = 8;
  var rLeft = peX + CLEARANCE;
  var rTop = peY + CLEARANCE;
  var rRight = peX + peW - CLEARANCE;
  var rBottom = peY + peH - CLEARANCE;
  var gMinX = rLeft, gMinY = rTop;
  var gCols = Math.max(1, Math.floor((rRight - rLeft) / GS));
  var gRows = Math.max(1, Math.floor((rBottom - rTop) / GS));

  var blocked = {};
  var usedSegs = {};

  function gk(c, r) { return c + "," + r; }
  function ek(c0, r0, c1, r1) {
    if (c0 < c1 || (c0 === c1 && r0 <= r1)) return c0+","+r0+"|"+c1+","+r1;
    return c1+","+r1+"|"+c0+","+r0;
  }
  function toGrid(px, py) {
    return {
      gc: Math.max(0, Math.min(gCols, Math.round((px - gMinX) / GS))),
      gr: Math.max(0, Math.min(gRows, Math.round((py - gMinY) / GS)))
    };
  }
  function toPix(gc, gr) { return { x: gMinX + gc * GS, y: gMinY + gr * GS }; }

  // Block FU obstacle areas
  fuObstacles.forEach(function(ob) {
    var c0 = Math.floor((ob.x - 4 - gMinX) / GS);
    var r0 = Math.floor((ob.y - 4 - gMinY) / GS);
    var c1 = Math.ceil((ob.x + ob.w + 4 - gMinX) / GS);
    var r1 = Math.ceil((ob.y + ob.h + 4 - gMinY) / GS);
    for (var gc = c0; gc <= c1; gc++)
      for (var gr = r0; gr <= r1; gr++)
        if (gc >= 0 && gr >= 0 && gc <= gCols && gr <= gRows)
          blocked[gk(gc, gr)] = true;
  });

  // A* routing function (compact version of module-level router)
  function routeAStar(start, goal) {
    var dirs = [{dc:1,dr:0},{dc:-1,dr:0},{dc:0,dr:1},{dc:0,dr:-1}];
    var open = [], cost = {}, prev = {}, closed = {};
    var sk = start.gc+","+start.gr+",-1";
    cost[sk] = 0;
    heapPush(open, {gc:start.gc, gr:start.gr, dir:-1, f:0});
    var goalK = null;
    while (open.length > 0) {
      var cur = heapPop(open);
      var ck = cur.gc+","+cur.gr+","+cur.dir;
      if (closed[ck]) continue;
      closed[ck] = true;
      if (cur.gc === goal.gc && cur.gr === goal.gr) { goalK = ck; break; }
      var cc = cost[ck];
      for (var di = 0; di < 4; di++) {
        var nc = cur.gc+dirs[di].dc, nr = cur.gr+dirs[di].dr;
        if (nc<0||nr<0||nc>gCols||nr>gRows||blocked[gk(nc,nr)]) continue;
        var seg = ek(cur.gc,cur.gr,nc,nr);
        var sc = 1;
        if (cur.dir >= 0 && di !== cur.dir) sc += 1.5;
        if (cur.dir >= 0 && di === (cur.dir^1)) sc += 2;
        if (usedSegs[seg]) sc += 30;
        var nk = nc+","+nr+","+di;
        var newC = cc + sc;
        if (cost[nk] !== undefined && newC >= cost[nk]) continue;
        cost[nk] = newC;
        prev[nk] = ck;
        heapPush(open, {gc:nc, gr:nr, dir:di, f:newC + Math.abs(nc-goal.gc)+Math.abs(nr-goal.gr)});
      }
    }
    if (!goalK) return null;
    var p = [], k = goalK;
    while (k) {
      var pts = k.split(",");
      p.unshift({gc:parseInt(pts[0]),gr:parseInt(pts[1])});
      k = prev[k];
    }
    return p;
  }

  function reservePath(path) {
    for (var i = 0; i < path.length - 1; i++)
      usedSegs[ek(path[i].gc,path[i].gr,path[i+1].gc,path[i+1].gr)] = true;
  }

  function simplify(path) {
    if (path.length <= 2) return path;
    var out = [path[0]];
    for (var i = 1; i < path.length - 1; i++) {
      var a = out[out.length-1], b = path[i], c = path[i+1];
      if ((a.gc===b.gc && b.gc===c.gc) || (a.gr===b.gr && b.gr===c.gr)) continue;
      out.push(b);
    }
    out.push(path[path.length-1]);
    return out;
  }

  // Route each PE-internal connection
  peConnections.forEach(function(conn) {
    var pePort = portRegistry[conn.pePortKey];
    var fuPort = fuPortPos[conn.fuPortKey];
    if (!pePort || !fuPort) return;

    var isIngress = conn.direction === "in";
    var routeStart = isIngress ? pePort : fuPort;
    var routeEnd = isIngress ? fuPort : pePort;

    // Find nearest unblocked grid cells for start/end
    var startPx = isIngress
      ? {x: pePort.x + 12, y: pePort.y}
      : {x: fuPort.x, y: fuPort.y + 10};
    var endPx = isIngress
      ? {x: fuPort.x, y: fuPort.y - 10}
      : {x: pePort.x - 12, y: pePort.y};
    var start = toGrid(startPx.x, startPx.y);
    var end = toGrid(endPx.x, endPx.y);

    // Unblock start/end cells temporarily
    var startWasBlocked = blocked[gk(start.gc, start.gr)];
    var endWasBlocked = blocked[gk(end.gc, end.gr)];
    delete blocked[gk(start.gc, start.gr)];
    delete blocked[gk(end.gc, end.gr)];

    var gridPath = routeAStar(start, end);

    // Restore blocked state
    if (startWasBlocked) blocked[gk(start.gc, start.gr)] = true;
    if (endWasBlocked) blocked[gk(end.gc, end.gr)] = true;

    if (!gridPath) {
      // Fallback: straight line
      edgeG.append("line")
        .attr("x1", routeStart.x).attr("y1", routeStart.y)
        .attr("x2", routeEnd.x).attr("y2", routeEnd.y)
        .attr("class", "pe-route-base")
        .attr("data-sw-edge", conn.swEdge)
        .attr("stroke", "rgba(200,214,229,0.42)")
        .attr("stroke-width", 1.3)
        .attr("stroke-dasharray", "3,2")
        .attr("opacity", 0.75);
      return;
    }

    reservePath(gridPath);
    var simplified = simplify(gridPath);
    var pts = [];
    pts.push({x: routeStart.x, y: routeStart.y});
    simplified.forEach(function(gp) { pts.push(toPix(gp.gc, gp.gr)); });
    pts.push({x: routeEnd.x, y: routeEnd.y});

    // Build SVG path
    var d = "M" + pts[0].x + "," + pts[0].y;
    for (var pi = 1; pi < pts.length; pi++)
      d += "L" + pts[pi].x + "," + pts[pi].y;

    edgeG.append("path")
      .attr("d", d).attr("fill", "none")
      .attr("class", "pe-route-base")
      .attr("data-sw-edge", conn.swEdge)
      .attr("stroke", "rgba(200,214,229,0.4)")
      .attr("stroke-width", 1.4)
      .attr("stroke-linecap", "round").attr("stroke-linejoin", "round")
      .attr("opacity", 0.8);

    // Arrow at the signal destination.
    var last = pts[pts.length - 1];
    var prev = pts[pts.length - 2];
    var dx = last.x - prev.x, dy = last.y - prev.y;
    var len = Math.sqrt(dx*dx + dy*dy);
    if (len > 0) {
      dx /= len; dy /= len;
      edgeG.append("polygon")
        .attr("class", "pe-route-arrow")
        .attr("data-sw-edge", conn.swEdge)
        .attr("points", [
          last.x + "," + last.y,
          (last.x - dx*5 + dy*3) + "," + (last.y - dy*5 - dx*3),
          (last.x - dx*5 - dy*3) + "," + (last.y - dy*5 + dx*3)
        ].join(" "))
        .attr("fill", "rgba(200,214,229,0.5)").attr("opacity", 0.85);
    }
  });
}

function drawActiveSwitchRoutes(parentG) {
  if (!switchRouteIdx || !switchRouteIdx.routes || !renderADG._portRegistry) return;

  switchRouteIdx.routes.forEach(function(route) {
    if (!route.sw_edges || route.sw_edges.length === 0) return;
    var inKey = route.component + "_in_" + route.input_port;
    var outKey = route.component + "_out_" + route.output_port;
    var pIn = renderADG._portRegistry[inKey];
    var pOut = renderADG._portRegistry[outKey];
    if (!pIn || !pOut) return;

    parentG.append("line")
      .attr("class", "sw-route-base crossbar-line")
      .attr("data-sw-edge", route.sw_edges[0])
      .attr("x1", pIn.x).attr("y1", pIn.y)
      .attr("x2", pOut.x).attr("y2", pOut.y)
      .attr("stroke", "rgba(200,214,229,0.38)")
      .attr("stroke-width", 1.35)
      .attr("opacity", 0.7)
      .style("pointer-events", "none");
  });
}

// ============================================================
// Port table: map flat port IDs to viz port keys
// ============================================================

var portLookup = {}; // portId -> {kind, name, index, dir, vizKey}

function buildPortLookup() {
  portLookup = {};
  if (!MAPPING_DATA || !MAPPING_DATA.port_table) return;
  MAPPING_DATA.port_table.forEach(function(p) {
    var info = { kind: p.kind, index: p.index, dir: p.dir };
    if (p.kind === "module_in") {
      info.vizKey = "module_in_" + p.index;
      info.component = "module_in";
    } else if (p.kind === "module_out") {
      info.vizKey = "module_out_" + p.index;
      info.component = "module_out";
    } else if (isSwitchPortKind(p.kind) || p.kind === "fifo" || p.kind === "memory") {
      info.component = p.name;
      info.vizKey = p.name + "_" + p.dir + "_" + p.index;
    } else if (p.kind === "fu") {
      info.component = p.pe;
      info.fu = p.fu;
      info.pe = p.pe;
      info.fuPortKey = p.pe + "/" + p.fu + "/" + p.dir + "_" + p.index;
      info.vizKey = info.fuPortKey;
    }
    portLookup[p.id] = info;
  });
}

// Find which components are used by routing (PEs with mapped ops + SWs with routes through them)
function getUsedComponents() {
  var used = {};
  if (!MAPPING_DATA) return used;
  MAPPING_DATA.node_mappings.forEach(function(m) { used[m.pe_name] = true; });
  if (MAPPING_DATA.edge_routings && MAPPING_DATA.port_table) {
    MAPPING_DATA.edge_routings.forEach(function(er) {
      var path = er.path;
      for (var i = 0; i < path.length; i++) {
        var p = portLookup[path[i]];
        if (p && p.component && p.component !== "module_in" && p.component !== "module_out")
          used[p.component] = true;
      }
    });
  }
  return used;
}

function collectUsedVisibleEdges() {
  var usedByKey = {};
  if (!MAPPING_DATA || !MAPPING_DATA.edge_routings) return [];

  MAPPING_DATA.edge_routings.forEach(function(er) {
    var path = er.path || [];
    for (var i = 0; i + 1 < path.length; i += 2) {
      var srcInfo = portLookup[path[i]];
      var dstInfo = portLookup[path[i + 1]];
      if (!srcInfo || !dstInfo) continue;

      var fromKey = null;
      var toKey = null;
      if (dstInfo.kind === "fu" && srcInfo.kind !== "fu") {
        fromKey = srcInfo.vizKey;
        toKey = getPEIngressPortKey(dstInfo.pe, srcInfo.vizKey);
      } else if (srcInfo.kind === "fu" && dstInfo.kind !== "fu") {
        fromKey = getPEEgressPortKey(srcInfo.pe, dstInfo.vizKey);
        toKey = dstInfo.vizKey;
      } else if (srcInfo.kind !== "fu" && dstInfo.kind !== "fu") {
        fromKey = srcInfo.vizKey;
        toKey = dstInfo.vizKey;
      }

      if (!fromKey || !toKey) continue;
      var edgeKey = fromKey + "->" + toKey;
      if (!usedByKey[edgeKey]) {
        usedByKey[edgeKey] = {
          fromKey: fromKey,
          toKey: toKey,
          swEdges: []
        };
      }
      if (usedByKey[edgeKey].swEdges.indexOf(er.sw_edge) < 0)
        usedByKey[edgeKey].swEdges.push(er.sw_edge);
    }
  });

  return Object.keys(usedByKey).sort().map(function(key) {
    return usedByKey[key];
  });
}

// ============================================================
// Cross-highlighting handlers
// ============================================================

function onDFGNodeClick(nodeIdx) {
  if (!mappingEnabled) return;
  clearCrossHighlight();
  highlightDFGNode(nodeIdx);

  // Check if this is an input sentinel
  if (DFG_DATA && DFG_DATA.nodes) {
    var nodeInfo = DFG_DATA.nodes.find(function(n) { return n.id === nodeIdx; });
    if (nodeInfo) {
      if (nodeInfo.kind === "input" && nodeInfo.arg_index !== undefined) {
        // Highlight module input port and all edges from this node
        highlightModulePort("module_in_" + nodeInfo.arg_index);
        highlightEdgesFromNode(nodeIdx);
        return;
      }
      if (nodeInfo.kind === "output") {
        // Highlight all module output ports used by edges to this node
        highlightEdgesToNode(nodeIdx);
        return;
      }
    }
  }

  // Operation node - highlight mapped PE/FU
  var m = mappingIdx ? mappingIdx.swToHw[nodeIdx] : null;
  if (m) {
    highlightADGComp(m.pe_name);
    highlightADGFU(m.pe_name, m.hw_name);
    focusADGOn(m.pe_name);
  }
}

function onDFGEdgeDataClick(edgeData) {
  if (!mappingEnabled) return;
  clearCrossHighlight();
  highlightDFGNode(edgeData.from);
  highlightDFGNode(edgeData.to);
  highlightRoutingPath(edgeData.id);
}

function onDFGEdgeClick(fromIdx, toIdx) {
  if (!mappingEnabled) return;
  clearCrossHighlight();
  highlightDFGNode(fromIdx);
  highlightDFGNode(toIdx);

  // Find the DFG edge and its routing path
  if (DFG_DATA && DFG_DATA.edges && MAPPING_DATA && MAPPING_DATA.edge_routings) {
    var dfgEdge = DFG_DATA.edges.find(function(e) {
      return e.from === fromIdx && e.to === toIdx;
    });
    if (dfgEdge) {
      highlightRoutingPath(dfgEdge.id);
    }
  }
}

function onADGCompClick(compName) {
  if (!mappingEnabled) return;
  clearCrossHighlight();
  highlightADGComp(compName);
  if (mappingIdx && mappingIdx.hwToSw[compName]) {
    mappingIdx.hwToSw[compName].forEach(function(m) {
      highlightDFGNode(m.sw_node);
    });
    focusDFGOn(mappingIdx.hwToSw[compName].map(function(m) { return m.sw_node; }));
  }
}

// Highlight all routing paths for edges originating from a DFG node
function highlightEdgesFromNode(nodeIdx) {
  if (!DFG_DATA || !DFG_DATA.edges || !MAPPING_DATA) return;
  DFG_DATA.edges.forEach(function(e) {
    if (e.from === nodeIdx) highlightRoutingPath(e.id);
  });
}

// Highlight all routing paths for edges going to a DFG node (return)
function highlightEdgesToNode(nodeIdx) {
  if (!DFG_DATA || !DFG_DATA.edges || !MAPPING_DATA) return;
  DFG_DATA.edges.forEach(function(e) {
    if (e.to === nodeIdx) highlightRoutingPath(e.id);
  });
}

function highlightFUPort(fuPortKey, color) {
  var p = fuPortPos[fuPortKey];
  if (!p || !renderADG._g) return;
  var c = color || "#4ecdc4";
  renderADG._g.append("rect")
    .attr("x", p.x - 7).attr("y", p.y - 7)
    .attr("width", 14).attr("height", 14).attr("rx", 3)
    .attr("fill", "none").attr("stroke", c)
    .attr("stroke-width", 2.2).attr("class", "cross-highlight-rect")
    .style("filter", "drop-shadow(0 0 4px " + c + ")");
}

function resetInternalRouteStyles() {
  if (!renderADG._g) return;
  renderADG._g.selectAll(".sw-route-base")
    .attr("stroke", "rgba(200,214,229,0.38)")
    .attr("stroke-width", 1.35)
    .attr("opacity", 0.7)
    .style("filter", null);
  renderADG._g.selectAll(".pe-route-base")
    .attr("stroke", "rgba(200,214,229,0.4)")
    .attr("stroke-width", 1.4)
    .attr("opacity", 0.8)
    .attr("stroke-dasharray", null)
    .style("filter", null);
  renderADG._g.selectAll(".pe-route-arrow")
    .attr("fill", "rgba(200,214,229,0.5)")
    .attr("opacity", 0.85);
}

function highlightSwitchRoutesForEdge(swEdgeId, color) {
  if (!renderADG._g) return;
  var c = color || "#4ecdc4";
  renderADG._g.selectAll(".sw-route-base[data-sw-edge='" + swEdgeId + "']")
    .attr("stroke", c)
    .attr("stroke-width", 2.6)
    .attr("opacity", 1)
    .style("filter", "drop-shadow(0 0 3px " + c + ")");
}

function highlightPERoutesForEdge(swEdgeId, color) {
  if (!renderADG._g) return;
  var c = color || "#4ecdc4";
  renderADG._g.selectAll(".pe-route-base[data-sw-edge='" + swEdgeId + "']")
    .attr("stroke", c)
    .attr("stroke-width", 2.3)
    .attr("opacity", 1)
    .style("filter", "drop-shadow(0 0 3px " + c + ")");
  renderADG._g.selectAll(".pe-route-arrow[data-sw-edge='" + swEdgeId + "']")
    .attr("fill", c)
    .attr("opacity", 1);
}

// Highlight a full routing path on the ADG
function highlightRoutingPath(swEdgeId) {
  if (!MAPPING_DATA || !MAPPING_DATA.edge_routings || !renderADG._g) return;
  var routing = MAPPING_DATA.edge_routings.find(function(er) {
    return er.sw_edge === swEdgeId;
  });
  if (!routing || !routing.path || routing.path.length < 2) return;

  var path = routing.path;
  var colors = ["#4ecdc4", "#ff6b35", "#ffd166", "#5dade2", "#a29bfe", "#fd79a8"];
  var color = colors[swEdgeId % colors.length];

  highlightSwitchRoutesForEdge(swEdgeId, color);
  highlightPERoutesForEdge(swEdgeId, color);

  // Process path in pairs: [outPort, inPort, outPort, inPort, ...]
  for (var i = 0; i < path.length - 1; i += 2) {
    var srcPort = portLookup[path[i]];
    var dstPort = portLookup[path[i + 1]];
    if (!srcPort || !dstPort) continue;

    if (dstPort.kind === "fu" && srcPort.kind !== "fu") {
      highlightModulePort(srcPort.vizKey, color);
      highlightFUPort(dstPort.fuPortKey, color);
      var peIngressKey = getPEIngressPortKey(dstPort.pe, srcPort.vizKey);
      if (peIngressKey) {
        highlightModulePort(peIngressKey, color);
        highlightHWEdge(srcPort.vizKey, peIngressKey, color);
      }
      continue;
    }

    if (srcPort.kind === "fu" && dstPort.kind !== "fu") {
      highlightModulePort(dstPort.vizKey, color);
      highlightFUPort(srcPort.fuPortKey, color);
      var peEgressKey = getPEEgressPortKey(srcPort.pe, dstPort.vizKey);
      if (peEgressKey) {
        highlightModulePort(peEgressKey, color);
        highlightHWEdge(peEgressKey, dstPort.vizKey, color);
      }
      continue;
    }

    if (srcPort.vizKey) highlightModulePort(srcPort.vizKey, color);
    if (dstPort.vizKey) highlightModulePort(dstPort.vizKey, color);
    if (srcPort.vizKey && dstPort.vizKey)
      highlightHWEdge(srcPort.vizKey, dstPort.vizKey, color);
  }
}

function highlightModulePort(vizKey, color) {
  if (!renderADG._portRegistry) return;
  var p = renderADG._portRegistry[vizKey];
  if (!p) return;
  var c = color || "#4ecdc4";
  renderADG._g.append("rect")
    .attr("x", p.x - 8).attr("y", p.y - 8)
    .attr("width", 16).attr("height", 16).attr("rx", 3)
    .attr("fill", "none").attr("stroke", c)
    .attr("stroke-width", 2.5).attr("class", "cross-highlight-rect")
    .style("filter", "drop-shadow(0 0 4px " + c + ")");
}

function highlightHWEdge(fromKey, toKey, color) {
  if (!renderADG._g) return;
  var c = color || "#4ecdc4";
  // Find the hardware edge matching these ports
  renderADG._g.selectAll(".hw-edge-vis").each(function() {
    var el = d3.select(this);
    var ef = el.attr("data-from");
    var et = el.attr("data-to");
    // Match: the connection from a component output to another component input
    // Port keys might not match exactly (fu ports vs pe ports), so check component level
    if (portsMatch(ef, fromKey) && portsMatch(et, toKey)) {
      el.attr("stroke", c).attr("stroke-width", 3).attr("opacity", 1);
    }
  });
}

function portsMatch(edgePortKey, lookupKey) {
  if (edgePortKey === lookupKey) return true;
  // Handle fu port -> pe port mapping: "pe_0_in_0" matches "pe_0_in_0"
  // Also handle module_in_0 -> module_in_0
  return false;
}

function drawCrossbarHighlight(inPort, outPort, color) {
  if (!renderADG._g || !renderADG._portRegistry) return;
  var pIn = renderADG._portRegistry[inPort.vizKey];
  var pOut = renderADG._portRegistry[outPort.vizKey];
  if (!pIn || !pOut) return;
  renderADG._g.append("line")
    .attr("x1", pIn.x).attr("y1", pIn.y)
    .attr("x2", pOut.x).attr("y2", pOut.y)
    .attr("stroke", color).attr("stroke-width", 2)
    .attr("stroke-dasharray", "4,2")
    .attr("class", "cross-highlight-rect crossbar-line")
    .style("filter", "drop-shadow(0 0 3px " + color + ")");
}

// ============================================================
// Mapping toggle
// ============================================================

function setMappingEnabled(on) {
  mappingEnabled = on;

  if (!on && currentMode === "overlay") {
    setMode("sidebyside");
  }

  // Re-render ADG with new mapping mode (collapses FUs, adds PE routing)
  renderADG();
  // Re-setup DFG interaction handlers
  setupDFGInteraction();

  var btnOvl = document.getElementById("btn-overlay");
  if (btnOvl) {
    btnOvl.classList.toggle("disabled", !on);
    btnOvl.disabled = !on;
  }
  var btnMap = document.getElementById("btn-mapping");
  if (btnMap) {
    btnMap.textContent = on ? "Mapping: On" : "Mapping: Off";
    btnMap.classList.toggle("active", on);
  }
}

// ============================================================
// DFG node/edge highlighting helpers
// ============================================================

function highlightDFGNode(nodeIdx) {
  var dfgEl = document.getElementById("svg-dfg");
  if (!dfgEl) return;
  var nodeG = dfgEl.querySelector("g.node[data-dfg-node='" + nodeIdx + "']");
  if (!nodeG) {
    var titles = dfgEl.querySelectorAll("g.node title");
    titles.forEach(function(t) {
      if (t.textContent.trim() === "n" + nodeIdx)
        nodeG = t.parentElement;
    });
  }
  if (!nodeG) return;
  var shape = nodeG.querySelector(".dfg-node-shape") ||
    nodeG.querySelector("ellipse, polygon, path");
  if (shape) {
    shape.classList.add("cross-highlight-dfg");
    shape.setAttribute("data-orig-stroke", shape.getAttribute("stroke") || "");
    shape.setAttribute("data-orig-sw", shape.getAttribute("stroke-width") || "");
  }
}

function highlightADGComp(compName) {
  if (!renderADG._g) return;
  var bbox = adgCompBoxes[compName];
  if (!bbox) return;
  renderADG._g.append("rect")
    .attr("x", bbox.x - 3).attr("y", bbox.y - 3)
    .attr("width", bbox.w + 6).attr("height", bbox.h + 6)
    .attr("rx", 8).attr("class", "cross-highlight-rect");
}

function highlightADGFU(peName, fuName) {
  if (!renderADG._g) return;
  var fuKey = peName + "/" + fuName;
  var bbox = adgFUBoxes[fuKey];
  if (!bbox) return;
  renderADG._g.append("rect")
    .attr("x", bbox.x - 2).attr("y", bbox.y - 2)
    .attr("width", bbox.w + 4).attr("height", bbox.h + 4)
    .attr("rx", 5).attr("class", "cross-highlight-rect")
    .attr("stroke", "#ffd166");
}

function focusADGOn(compName) {
  if (!renderADG._svg || !renderADG._g || !renderADG._zoom) return;
  var bbox = adgCompBoxes[compName];
  if (!bbox) return;
  var svgNode = renderADG._svg.node();
  var svgW = svgNode.clientWidth || 800;
  var svgH = svgNode.clientHeight || 600;
  var pad = 80;
  var scale = Math.min((svgW - pad*2) / bbox.w, (svgH - pad*2) / bbox.h, 2);
  scale = Math.min(scale, 1.5);
  var tx = svgW/2 - (bbox.x + bbox.w/2) * scale;
  var ty = svgH/2 - (bbox.y + bbox.h/2) * scale;
  renderADG._svg.transition().duration(500)
    .call(renderADG._zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}

function focusDFGOn(nodeIndices) {
  if (!renderDFG._svg || !renderDFG._g || !renderDFG._zoom) return;
  var dfgEl = document.getElementById("svg-dfg");
  if (!dfgEl) return;
  var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  nodeIndices.forEach(function(idx) {
    var nodeG = dfgEl.querySelector("g.node[data-dfg-node='" + idx + "']");
    if (!nodeG) return;
    var r = nodeG.getBBox();
    minX = Math.min(minX, r.x); minY = Math.min(minY, r.y);
    maxX = Math.max(maxX, r.x + r.width); maxY = Math.max(maxY, r.y + r.height);
  });
  if (minX === Infinity) return;
  var bw = maxX - minX, bh = maxY - minY;
  var svgNode = renderDFG._svg.node();
  var svgW = svgNode.clientWidth || 800;
  var svgH = svgNode.clientHeight || 600;
  var pad = 60;
  var scale = Math.min((svgW - pad*2) / Math.max(bw, 1), (svgH - pad*2) / Math.max(bh, 1), 2);
  scale = Math.min(scale, 1.5);
  var tx = svgW/2 - (minX + bw/2) * scale;
  var ty = svgH/2 - (minY + bh/2) * scale;
  renderDFG._svg.transition().duration(500)
    .call(renderDFG._zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}

function clearCrossHighlight() {
  if (renderADG._g) {
    renderADG._g.selectAll(".cross-highlight-rect").remove();
    // Reset highlighted hw edges
    renderADG._g.selectAll(".hw-edge-vis")
      .attr("stroke", "#8a9bb5").attr("stroke-width", 1.5).attr("opacity", 0.6);
    resetInternalRouteStyles();
  }
  var dfgEl = document.getElementById("svg-dfg");
  if (dfgEl) {
    dfgEl.querySelectorAll(".cross-highlight-dfg").forEach(function(el) {
      el.classList.remove("cross-highlight-dfg");
      var origStroke = el.getAttribute("data-orig-stroke");
      var origSw = el.getAttribute("data-orig-sw");
      if (origStroke) el.setAttribute("stroke", origStroke);
      if (origSw) el.setAttribute("stroke-width", origSw);
    });
  }
}

// ============================================================
// Overlay mode
// ============================================================

function renderOverlay() {
  if (!mappingIdx || !renderADG._g) return;
  clearOverlay();
  var g = renderADG._g;
  var overlayG = g.append("g").attr("class", "overlay-group");

  // Dim unused components (not just unmapped PEs -- also unused SWs)
  var usedComps = getUsedComponents();
  Object.keys(adgCompBoxes).forEach(function(name) {
    if (usedComps[name]) return;
    var box = adgCompBoxes[name];
    overlayG.append("rect")
      .attr("x", box.x - 2).attr("y", box.y - 2)
      .attr("width", box.w + 4).attr("height", box.h + 4)
      .attr("rx", 4).attr("fill", "rgba(12,18,32,0.7)")
      .attr("class", "overlay-dim-cover").style("pointer-events", "none");
  });

  drawOverlayUsedEdges(overlayG);
  drawOverlaySwitchRoutes(overlayG);
  drawOverlayPERoutes(overlayG);
}

function drawOverlayUsedEdges(parentG) {
  if (!renderADG._g) return;
  var colors = ["#4ecdc4", "#ff6b35", "#ffd166", "#5dade2", "#a29bfe", "#fd79a8"];
  var usedEdges = collectUsedVisibleEdges();
  var edgeMap = {};
  usedEdges.forEach(function(edge) {
    edgeMap[edge.fromKey + "->" + edge.toKey] = edge;
  });

  renderADG._g.selectAll(".hw-edge-vis").each(function() {
    var base = d3.select(this);
    var fromKey = base.attr("data-from");
    var toKey = base.attr("data-to");
    var entry = edgeMap[fromKey + "->" + toKey];
    if (!entry || !entry.swEdges || entry.swEdges.length === 0) return;

    var swEdge = entry.swEdges[0];
    var color = colors[swEdge % colors.length];
    parentG.append("path")
      .attr("d", base.attr("d"))
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 2.6)
      .attr("stroke-linecap", "round")
      .attr("stroke-linejoin", "round")
      .attr("opacity", 0.96)
      .attr("class", "overlay-used-edge")
      .style("pointer-events", "none");
  });
}

function drawOverlaySwitchRoutes(parentG) {
  if (!switchRouteIdx || !switchRouteIdx.routes || !renderADG._portRegistry) return;
  var colors = ["#4ecdc4", "#ff6b35", "#ffd166", "#5dade2", "#a29bfe", "#fd79a8"];
  switchRouteIdx.routes.forEach(function(route) {
    if (!route.sw_edges || route.sw_edges.length === 0) return;
    var swEdge = route.sw_edges[0];
    var color = colors[swEdge % colors.length];
    var pIn = renderADG._portRegistry[route.component + "_in_" + route.input_port];
    var pOut = renderADG._portRegistry[route.component + "_out_" + route.output_port];
    if (!pIn || !pOut) return;

    parentG.append("line")
      .attr("x1", pIn.x).attr("y1", pIn.y)
      .attr("x2", pOut.x).attr("y2", pOut.y)
      .attr("stroke", color)
      .attr("stroke-width", 2)
      .attr("opacity", 0.92)
      .attr("class", "overlay-used-internal overlay-used-sw")
      .style("pointer-events", "none");
  });
}

function drawOverlayPERoutes(parentG) {
  if (!renderADG._g) return;
  var colors = ["#4ecdc4", "#ff6b35", "#ffd166", "#5dade2", "#a29bfe", "#fd79a8"];

  renderADG._g.selectAll(".pe-route-base").each(function() {
    var base = d3.select(this);
    var swEdge = parseInt(base.attr("data-sw-edge"), 10);
    if (isNaN(swEdge)) return;
    var color = colors[swEdge % colors.length];
    parentG.append("path")
      .attr("d", base.attr("d"))
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 2.2)
      .attr("stroke-linecap", "round")
      .attr("stroke-linejoin", "round")
      .attr("opacity", 0.94)
      .attr("class", "overlay-used-internal overlay-used-pe")
      .style("pointer-events", "none");
  });

  renderADG._g.selectAll(".pe-route-arrow").each(function() {
    var arrow = d3.select(this);
    var swEdge = parseInt(arrow.attr("data-sw-edge"), 10);
    if (isNaN(swEdge)) return;
    var color = colors[swEdge % colors.length];
    parentG.append("polygon")
      .attr("points", arrow.attr("points"))
      .attr("fill", color)
      .attr("opacity", 0.98)
      .attr("class", "overlay-used-arrow")
      .style("pointer-events", "none");
  });
}

function clearOverlay() {
  if (renderADG._g) {
    renderADG._g.selectAll(".overlay-group").remove();
  }
}

// ============================================================
// Mode toggle
// ============================================================

function setMode(mode) {
  // Block overlay when mapping is off
  if (mode === "overlay" && !mappingEnabled) return;
  currentMode = mode;
  clearCrossHighlight();

  var adgPanel = document.getElementById("panel-adg");
  var dfgPanel = document.getElementById("panel-dfg");
  var divider = document.getElementById("panel-divider");

  if (mode === "overlay") {
    adgPanel.classList.add("panel-full");
    dfgPanel.classList.add("panel-hidden");
    divider.classList.add("divider-hidden");
    renderOverlay();
    // Re-fit ADG view
    if (renderADG._svg && renderADG._g && renderADG._zoom)
      setTimeout(function() { fitView(renderADG._svg, renderADG._g, renderADG._zoom); }, 200);
  } else {
    adgPanel.classList.remove("panel-full");
    dfgPanel.classList.remove("panel-hidden");
    divider.classList.remove("divider-hidden");
    clearOverlay();
    // Auto-fit both panels after layout restores
    setTimeout(function() {
      if (renderADG._svg && renderADG._g && renderADG._zoom)
        fitView(renderADG._svg, renderADG._g, renderADG._zoom);
      if (renderDFG._svg && renderDFG._g && renderDFG._zoom)
        fitView(renderDFG._svg, renderDFG._g, renderDFG._zoom);
    }, 200);
  }

  // Update button states
  var btnSbs = document.getElementById("btn-sidebyside");
  var btnOvl = document.getElementById("btn-overlay");
  if (btnSbs) btnSbs.classList.toggle("active", mode === "sidebyside");
  if (btnOvl) btnOvl.classList.toggle("active", mode === "overlay");
}

// ============================================================
// Init
// ============================================================

function init() {
  // Build mapping index and port lookup
  mappingIdx = buildMappingIndex();
  buildPortLookup();
  buildADGConnectionMaps();
  switchRouteIdx = buildSwitchRouteIndex();
  peRouteIdx = buildPERouteIndex();

  // Add mapping toggle + mode buttons when mapping data is present
  var toolbar = document.getElementById("toolbar");
  if (toolbar && MAPPING_DATA) {
    var statusBar = document.getElementById("status-bar");

    // Mapping toggle
    var sep1 = document.createElement("span");
    sep1.className = "toolbar-sep";
    sep1.textContent = "|";
    var btnMap = document.createElement("button");
    btnMap.id = "btn-mapping";
    btnMap.className = "mapping-toggle";
    btnMap.textContent = "Mapping: Off";
    btnMap.addEventListener("click", function() {
      setMappingEnabled(!mappingEnabled);
    });
    toolbar.insertBefore(sep1, statusBar);
    toolbar.insertBefore(btnMap, statusBar);

    // Mode separator
    var sep2 = document.createElement("span");
    sep2.className = "toolbar-sep";
    sep2.textContent = "|";
    toolbar.insertBefore(sep2, statusBar);

    // Side-by-Side button (always active by default)
    var btnSbs = document.createElement("button");
    btnSbs.id = "btn-sidebyside";
    btnSbs.className = "mode-btn active";
    btnSbs.textContent = "Side-by-Side";
    btnSbs.addEventListener("click", function() { setMode("sidebyside"); });
    toolbar.insertBefore(btnSbs, statusBar);

    // Overlay button (disabled by default since mapping starts Off)
    var btnOvl = document.createElement("button");
    btnOvl.id = "btn-overlay";
    btnOvl.className = "mode-btn disabled";
    btnOvl.textContent = "Overlay";
    btnOvl.disabled = true;
    btnOvl.addEventListener("click", function() { setMode("overlay"); });
    toolbar.insertBefore(btnOvl, statusBar);
  }

  var fitBtn = document.getElementById("btn-fit");
  if (fitBtn) fitBtn.addEventListener("click", function() {
    if (renderADG._svg) fitView(renderADG._svg, renderADG._g, renderADG._zoom);
    // DFG: try to fit the current DFG svg
    if (renderDFG._svg && renderDFG._g && renderDFG._zoom) {
      fitView(renderDFG._svg, renderDFG._g, renderDFG._zoom);
    } else {
      var dfgSvg = d3.select("#svg-dfg");
      var dfgInner = dfgSvg.select("g");
      if (!dfgInner.empty()) {
        var dfgZoom = d3.zoom().scaleExtent([0.1, 5])
          .on("zoom", function(ev) { dfgInner.attr("transform", ev.transform); });
        dfgSvg.call(dfgZoom);
        fitView(dfgSvg, dfgInner, dfgZoom);
      }
    }
  });

  // Panel divider drag
  var divider = document.getElementById("panel-divider");
  var graphArea = document.getElementById("graph-area");
  var adgPanel = document.getElementById("panel-adg");
  var dfgPanel = document.getElementById("panel-dfg");
  if (divider) {
    var dragging = false;
    divider.addEventListener("mousedown", function(e) { dragging = true; e.preventDefault(); });
    document.addEventListener("mousemove", function(e) {
      if (!dragging) return;
      var rect = graphArea.getBoundingClientRect();
      var pct = ((e.clientX - rect.left) / rect.width) * 100;
      pct = Math.max(20, Math.min(80, pct));
      adgPanel.style.flex = "0 0 " + pct + "%";
      dfgPanel.style.flex = "0 0 " + (100 - pct) + "%";
    });
    document.addEventListener("mouseup", function() { dragging = false; });
  }

  renderADG();
  renderDFG();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
})();
