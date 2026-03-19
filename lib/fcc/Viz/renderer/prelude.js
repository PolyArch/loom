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
var fuConfigIdx = null;      // FU runtime config summaries from mapping data
var dfgStatusIdx = null;     // DFG mapped / routed status summaries
var adgCompBoxes = {};       // comp name -> {x,y,w,h}
var adgFUBoxes = {};         // "pe_name/fu_name" -> {x,y,w,h}
var fuPortPos = {};          // "pe_name/fu_name/in_0" -> {x, y}
var peIngressMap = {};       // pe name -> external viz key -> pe input viz key
var peEgressMap = {};        // pe name -> external viz key -> pe output viz key
var adgConnectionIdx = null; // visible ADG connections indexed by endpoint keys
var routingPrimitiveIdx = null; // unified visual routing primitives, built lazily
var switchRouteIdx = null;   // switch route summaries from mapping data
var peRouteIdx = null;       // PE internal route summaries from mapping data
var adgRenderHints = null;   // complexity-aware ADG rendering hints
var adgModuleRouteCache = {}; // scene signature -> routed module edge paths
var mappingEnabled = !!MAPPING_DATA;
var currentMode = "sidebyside";
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

function buildFUConfigIndex() {
  var idx = { byKey: {}, byHwNode: {} };
  if (!MAPPING_DATA || !MAPPING_DATA.fu_configs) return idx;
  MAPPING_DATA.fu_configs.forEach(function(entry) {
    if (entry.hw_node !== undefined) idx.byHwNode[entry.hw_node] = entry;
    if (entry.pe_name && entry.hw_name)
      idx.byKey[entry.pe_name + "/" + entry.hw_name] = entry;
  });
  return idx;
}

function buildDFGStatusIndex() {
  var idx = {
    mappedNodeIds: {},
    routedEdgeIds: {}
  };
  if (!MAPPING_DATA) return idx;

  (MAPPING_DATA.node_mappings || []).forEach(function(entry) {
    if (entry.sw_node !== undefined && entry.sw_node !== null)
      idx.mappedNodeIds[entry.sw_node] = true;
  });

  (MAPPING_DATA.edge_routings || []).forEach(function(entry) {
    if (entry.sw_edge === undefined || entry.sw_edge === null) return;
    if (entry.kind === "unrouted") return;
    idx.routedEdgeIds[entry.sw_edge] = true;
  });
  return idx;
}

function isSwitchPortKind(kind) {
  return kind === "sw" || kind === "temporal_sw";
}

function buildADGConnectionIndex() {
  var idx = { edges: [], byFrom: {}, byTo: {} };
  if (!ADG_DATA || !ADG_DATA.connections) return idx;

  ADG_DATA.connections.forEach(function(conn) {
    var fromKey = conn.from === "module_in"
      ? "module_in_" + conn.fromIdx
      : conn.from + "_out_" + conn.fromIdx;
    var toKey = conn.to === "module_out"
      ? "module_out_" + conn.toIdx
      : conn.to + "_in_" + conn.toIdx;
    var edge = {
      fromKey: fromKey,
      toKey: toKey,
      from: conn.from,
      to: conn.to,
      fromIdx: conn.fromIdx,
      toIdx: conn.toIdx
    };
    idx.edges.push(edge);
    if (!idx.byFrom[fromKey]) idx.byFrom[fromKey] = [];
    if (!idx.byTo[toKey]) idx.byTo[toKey] = [];
    idx.byFrom[fromKey].push(edge);
    idx.byTo[toKey].push(edge);
  });
  return idx;
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

function inferDirectPEIngressPortKey(portInfo) {
  if (!portInfo || portInfo.kind !== "fu" || !portInfo.pe) return null;
  return portInfo.pe + "_in_" + portInfo.index;
}

function inferDirectPEEgressPortKey(portInfo) {
  if (!portInfo || portInfo.kind !== "fu" || !portInfo.pe) return null;
  return portInfo.pe + "_out_" + portInfo.index;
}

function getNodeMappingByHwNode(hwNode) {
  if (!MAPPING_DATA || !MAPPING_DATA.node_mappings) return null;
  for (var i = 0; i < MAPPING_DATA.node_mappings.length; ++i) {
    var entry = MAPPING_DATA.node_mappings[i];
    if (entry.hw_node === hwNode) return entry;
  }
  return null;
}

function buildRoutingPrimitiveIndex() {
  var idx = {
    primitives: [],
    byEdge: {},
    moduleEdges: [],
    switchRoutes: [],
    componentRoutes: [],
    memoryBindings: [],
    functionUnitInternal: []
  };
  if (!MAPPING_DATA || !MAPPING_DATA.edge_routings) return idx;

  var moduleEdgeMap = {};
  var switchRouteMap = {};
  var componentRouteMap = {};
  var memoryBindingMap = {};
  var functionUnitInternalMap = {};

  function pushSwEdge(list, swEdge) {
    if (list.indexOf(swEdge) < 0) list.push(swEdge);
  }

  function addIndexedPrimitive(map, key, seed) {
    if (!map[key]) map[key] = seed;
    pushSwEdge(map[key].swEdges, seed.swEdges[0]);
    return map[key];
  }

  function addModuleEdge(swEdge, fromKey, toKey) {
    if (!fromKey || !toKey) return;
    addIndexedPrimitive(moduleEdgeMap, fromKey + "->" + toKey, {
      kind: "module_edge",
      fromKey: fromKey,
      toKey: toKey,
      swEdges: [swEdge]
    });
  }

  function addSwitchRoute(swEdge, component, inputPortId, outputPortId,
                          inputPort, outputPort) {
    if (!component) return;
    addIndexedPrimitive(switchRouteMap,
                        component + ":" + inputPort + ":" + outputPort, {
      kind: "switch_route",
      component: component,
      inputPortId: inputPortId,
      outputPortId: outputPortId,
      inputPort: inputPort,
      outputPort: outputPort,
      swEdges: [swEdge]
    });
  }

  function addComponentRoute(swEdge, component, direction,
                             componentPortKey, functionUnitPortKey) {
    if (!component || !componentPortKey || !functionUnitPortKey) return;
    addIndexedPrimitive(componentRouteMap,
                        component + ":" + direction + ":" + componentPortKey +
                          ":" + functionUnitPortKey, {
      kind: "component_route",
      component: component,
      direction: direction,
      componentPortKey: componentPortKey,
      functionUnitPortKey: functionUnitPortKey,
      swEdges: [swEdge]
    });
  }

  function addMemoryBinding(swEdge, component, portKey, direction) {
    if (!component || !portKey) return;
    var visibleEdges = direction === "out"
      ? (adgConnectionIdx && adgConnectionIdx.byFrom[portKey]
          ? adgConnectionIdx.byFrom[portKey] : [])
      : (adgConnectionIdx && adgConnectionIdx.byTo[portKey]
          ? adgConnectionIdx.byTo[portKey] : []);
    addIndexedPrimitive(memoryBindingMap, component + ":" + direction + ":" +
      portKey, {
      kind: "memory_binding",
      component: component,
      portKey: portKey,
      direction: direction,
      visibleEdges: visibleEdges.map(function(edge) {
        return { fromKey: edge.fromKey, toKey: edge.toKey };
      }),
      swEdges: [swEdge]
    });
  }

  function addFunctionUnitInternal(swEdge, hwNode) {
    var mapping = getNodeMappingByHwNode(hwNode);
    var component = mapping && mapping.pe_name ? mapping.pe_name : null;
    var functionUnit = mapping && mapping.hw_name ? mapping.hw_name : null;
    addIndexedPrimitive(functionUnitInternalMap,
                        String(hwNode) + ":" + (component || "") + ":" +
                          (functionUnit || ""), {
      kind: "function_unit_internal",
      hwNode: hwNode,
      component: component,
      functionUnit: functionUnit,
      swEdges: [swEdge]
    });
  }

  MAPPING_DATA.edge_routings.forEach(function(routing) {
    var swEdge = routing.sw_edge;
    if (routing.kind === "intra_fu") {
      addFunctionUnitInternal(swEdge, routing.hw_node);
      return;
    }

    var path = routing.path || [];
    if (path.length === 2 && path[0] === path[1]) {
      var directPort = portLookup[path[0]];
      if (!directPort) return;
      if (directPort.kind === "memory") {
        if (directPort.dir === "in" && directPort.index === 0)
          addMemoryBinding(swEdge, directPort.component,
                           directPort.component + "_in_0", "in");
        else if (directPort.dir === "out" && directPort.index === 0)
          addMemoryBinding(swEdge, directPort.component,
                           directPort.component + "_out_0", "out");
      }
      return;
    }

    for (var j = 0; j + 1 < path.length; ++j) {
      var hopSrc = portLookup[path[j]];
      var hopDst = portLookup[path[j + 1]];
      if (!hopSrc || !hopDst) continue;
      if (isSwitchPortKind(hopSrc.kind) && isSwitchPortKind(hopDst.kind) &&
          hopSrc.component === hopDst.component) {
        addSwitchRoute(swEdge, hopSrc.component, path[j], path[j + 1],
                       hopSrc.index, hopDst.index);
      }
    }

    for (var i = 0; i + 1 < path.length; i += 2) {
      var srcInfo = portLookup[path[i]];
      var dstInfo = portLookup[path[i + 1]];
      if (!srcInfo || !dstInfo) continue;

      if (isSwitchPortKind(srcInfo.kind) && isSwitchPortKind(dstInfo.kind) &&
          srcInfo.component === dstInfo.component) {
        addSwitchRoute(swEdge, srcInfo.component, path[i], path[i + 1],
                       srcInfo.index, dstInfo.index);
        continue;
      }

      if (dstInfo.kind === "fu" && srcInfo.kind !== "fu") {
        var ingressPortKey = getPEIngressPortKey(dstInfo.pe, srcInfo.vizKey);
        addComponentRoute(swEdge, dstInfo.pe, "in",
                          ingressPortKey,
                          dstInfo.fuPortKey);
        if (srcInfo.vizKey && ingressPortKey)
          addModuleEdge(swEdge, srcInfo.vizKey, ingressPortKey);
        continue;
      }

      if (srcInfo.kind === "fu" && dstInfo.kind !== "fu") {
        var egressPortKey = getPEEgressPortKey(srcInfo.pe, dstInfo.vizKey);
        addComponentRoute(swEdge, srcInfo.pe, "out",
                          egressPortKey,
                          srcInfo.fuPortKey);
        if (egressPortKey && dstInfo.vizKey)
          addModuleEdge(swEdge, egressPortKey, dstInfo.vizKey);
        continue;
      }

      if (srcInfo.kind === "fu" && dstInfo.kind === "fu" &&
          srcInfo.pe !== dstInfo.pe) {
        var directOutKey = inferDirectPEEgressPortKey(srcInfo);
        var directInKey = inferDirectPEIngressPortKey(dstInfo);
        addComponentRoute(swEdge, srcInfo.pe, "out",
                          directOutKey, srcInfo.fuPortKey);
        addComponentRoute(swEdge, dstInfo.pe, "in",
                          directInKey, dstInfo.fuPortKey);
        addModuleEdge(swEdge, directOutKey, directInKey);
        continue;
      }

      if (srcInfo.kind !== "fu" && dstInfo.kind !== "fu")
        addModuleEdge(swEdge, srcInfo.vizKey, dstInfo.vizKey);
    }
  });

  function appendPrimitivesFrom(map, target) {
    Object.keys(map).sort().forEach(function(key) {
      var primitive = map[key];
      target.push(primitive);
      primitive.swEdges.forEach(function(swEdge) {
        if (!idx.byEdge[swEdge]) idx.byEdge[swEdge] = [];
        idx.byEdge[swEdge].push(primitive);
      });
      idx.primitives.push(primitive);
    });
  }

  appendPrimitivesFrom(moduleEdgeMap, idx.moduleEdges);
  appendPrimitivesFrom(switchRouteMap, idx.switchRoutes);
  appendPrimitivesFrom(componentRouteMap, idx.componentRoutes);
  appendPrimitivesFrom(memoryBindingMap, idx.memoryBindings);
  appendPrimitivesFrom(functionUnitInternalMap, idx.functionUnitInternal);
  return idx;
}

function ensureRoutingPrimitiveIndex() {
  if (routingPrimitiveIdx) return routingPrimitiveIdx;
  routingPrimitiveIdx = buildRoutingPrimitiveIndex();
  return routingPrimitiveIdx;
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
  var routes = MAPPING_DATA ? (MAPPING_DATA.switch_routes || []) : [];
  if (routes.length === 0) {
    var primitiveIdx = ensureRoutingPrimitiveIndex();
    routes = primitiveIdx && primitiveIdx.switchRoutes &&
        primitiveIdx.switchRoutes.length > 0
      ? primitiveIdx.switchRoutes
      : (MAPPING_DATA ? deriveSwitchRoutesFromPaths() : []);
  }
  routes.forEach(function(route) {
    var normalized = {
      component: route.component,
      input_port_id: route.input_port_id !== undefined
        ? route.input_port_id : route.inputPortId,
      output_port_id: route.output_port_id !== undefined
        ? route.output_port_id : route.outputPortId,
      input_port: route.input_port !== undefined
        ? route.input_port : route.inputPort,
      output_port: route.output_port !== undefined
        ? route.output_port : route.outputPort,
      sw_edges: (route.sw_edges || route.swEdges || []).slice()
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
  var routes = MAPPING_DATA ? (MAPPING_DATA.pe_routes || []) : [];
  if (routes.length === 0) {
    var primitiveIdx = ensureRoutingPrimitiveIndex();
    routes = primitiveIdx && primitiveIdx.componentRoutes
      ? primitiveIdx.componentRoutes
      : [];
  }

  routes.forEach(function(route) {
    var normalized = {
      swEdge: route.sw_edge !== undefined ? route.sw_edge : route.swEdges[0],
      swEdges: (route.sw_edges || route.swEdges || []).slice(),
      peName: route.pe_name !== undefined ? route.pe_name : route.component,
      direction: route.direction,
      pePortKey: route.pe_port_key !== undefined
        ? route.pe_port_key : route.componentPortKey,
      fuPortKey: route.fu_port_key !== undefined
        ? route.fu_port_key : route.functionUnitPortKey
    };
    if (normalized.swEdges.length === 0 && normalized.swEdge !== undefined)
      normalized.swEdges = [normalized.swEdge];
    idx.routes.push(normalized);
    normalized.swEdges.forEach(function(swEdge) {
      if (!idx.byEdge[swEdge]) idx.byEdge[swEdge] = [];
      idx.byEdge[swEdge].push(normalized);
    });
    if (!idx.byPE[normalized.peName]) idx.byPE[normalized.peName] = [];
    idx.byPE[normalized.peName].push(normalized);
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
  if (opName.indexOf("mux") >= 0 || opName.indexOf("mux") >= 0) {
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
    compactAllFUs: false,
    compactCollapsedBoxW: 104,
    compactCollapsedBoxH: 22,
    compactFuBoxMargin: 12,
    peInnerPadX: 20,
    peInnerPadY: 30,
    peMinW: 200,
    peMinH: 90
  };
}

function buildPERenderLayout(peDef, mappingEnabled, peUsedFUs, renderHints) {
  var fus = peDef.fus || [];
  renderHints = renderHints || {};
  var forceCompact = !!renderHints.compactAllFUs;
  var fuBoxMargin = forceCompact
    ? (renderHints.compactFuBoxMargin || 6)
    : 12;
  function collapsedFUWidth(fuName) {
    var labelW = Math.max(24, fuName.length * 7 + 14);
    var baseW = forceCompact
      ? (renderHints.compactCollapsedBoxW || 68)
      : 0;
    return Math.max(baseW, labelW);
  }
  var collapsedBoxH = forceCompact
    ? (renderHints.compactCollapsedBoxH || 16)
    : 22;
  var hasMapping = Object.keys(peUsedFUs || {}).length > 0;
  var fuLayouts = [];
  var fuWidths = [];
  var fuHeights = [];

  fus.forEach(function(fu) {
    var shouldCollapse =
      forceCompact ||
      (mappingEnabled && (!hasMapping || !peUsedFUs[fu.name]));
    if (shouldCollapse) {
      var collapsedW = collapsedFUWidth(fu.name || "");
      fuLayouts.push({
        collapsed: true,
        boxW: collapsedW,
        boxH: collapsedBoxH
      });
      fuWidths.push(collapsedW);
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
  var rowGap = Math.max(14, Math.round(fuBoxMargin * 1.15));
  var peMinW = renderHints.peMinW || 200;
  var peMinH = renderHints.peMinH || 90;
  var fuBoxes = [];
  var peW = peMinW;
  var peH = peMinH;

  if (fus.length > 0) {
    var bestLayout = null;
    for (var cols = 1; cols <= fus.length; cols++) {
      var rowWidths = [];
      var rowHeights = [];
      for (var start = 0; start < fus.length; start += cols) {
        var end = Math.min(fus.length, start + cols);
        var rowWidth = 0;
        var rowHeight = 0;
        for (var i = start; i < end; i++) {
          rowWidth += fuWidths[i];
          if (i > start) rowWidth += fuBoxMargin;
          rowHeight = Math.max(rowHeight, fuHeights[i]);
        }
        rowWidths.push(rowWidth);
        rowHeights.push(rowHeight);
      }

      var contentW = 0;
      rowWidths.forEach(function(w) { contentW = Math.max(contentW, w); });
      var contentH = 0;
      rowHeights.forEach(function(h) { contentH += h; });
      if (rowHeights.length > 1)
        contentH += (rowHeights.length - 1) * rowGap;

      var candidateW = Math.max(peMinW, contentW + peInnerPadX * 2 + peMappingMargin);
      var candidateH = Math.max(peMinH, contentH + peInnerPadY * 2);
      var longSide = Math.max(candidateW, candidateH);
      var shortSide = Math.max(1, Math.min(candidateW, candidateH));
      var aspectPenalty = longSide / shortSide;
      var areaPenalty = candidateW * candidateH;
      var score = (aspectPenalty - 1.0) * 1000.0 + areaPenalty * 0.0001;

      if (!bestLayout || score < bestLayout.score) {
        bestLayout = {
          cols: cols,
          rowWidths: rowWidths,
          rowHeights: rowHeights,
          contentW: contentW,
          contentH: contentH,
          peW: candidateW,
          peH: candidateH,
          score: score
        };
      }
    }

    peW = bestLayout.peW;
    peH = bestLayout.peH;
    var localY = peInnerPadY;
    for (var row = 0, startIdx = 0; row < bestLayout.rowHeights.length; row++) {
      var rowWidth = bestLayout.rowWidths[row];
      var localX = peInnerPadX + (peMappingMargin / 2) +
        Math.max(0, (bestLayout.contentW - rowWidth) / 2);
      var endIdx = Math.min(fus.length, startIdx + bestLayout.cols);
      var rowHeight = bestLayout.rowHeights[row];
      for (var fi = startIdx; fi < endIdx; fi++) {
        fuBoxes[fi] = {
          x: localX,
          y: localY + Math.max(0, (rowHeight - fuHeights[fi]) / 2),
          w: fuWidths[fi],
          h: fuHeights[fi]
        };
        localX += fuWidths[fi] + fuBoxMargin;
      }
      localY += rowHeight + rowGap;
      startIdx = endIdx;
    }
  }

  return {
    fus: fus,
    fuLayouts: fuLayouts,
    fuWidths: fuWidths,
    fuHeights: fuHeights,
    fuBoxes: fuBoxes,
    fuBoxMargin: fuBoxMargin,
    rowGap: rowGap,
    peMappingMargin: peMappingMargin,
    peInnerPadX: peInnerPadX,
    peInnerPadY: peInnerPadY,
    peW: peW,
    peH: peH
  };
}
