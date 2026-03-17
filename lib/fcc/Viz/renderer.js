// fcc Visualization Renderer - Clean rebuild
// Side-by-side: ADG (D3.js) + DFG (Graphviz WASM)
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

  var pendingFUDots = []; // Queue for async Graphviz FU renders

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
      var fus = c.fus || [];
      var fuBoxW = 140, fuBoxMargin = 12;
      var pw = Math.max(200, fus.length * (fuBoxW + fuBoxMargin) + fuBoxMargin + 40);
      var fuHeights = fus.map(function(fu) {
        var n = fu.ops ? fu.ops.length : 0;
        return 30 + Math.max(n, 1) * 28 + 30;
      });
      var mh = 0; fuHeights.forEach(function(h) { mh = Math.max(mh, h); });
      var ph = 30 + mh + 30;
      maxPeW = Math.max(maxPeW, pw);
      maxPeH = Math.max(maxPeH, ph);
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
      var COLLAPSED_FU_H = 22;

      // Compute PE size based on FUs
      var fus = peDef.fus || [];
      var fuBoxW = 140;
      var fuBoxMargin = 12;

      // When mapping on: give extra margin for PE internal routing
      var peMappingMargin = (mappingEnabled && hasMapping) ? 40 : 0;
      var peW = Math.max(200, fus.length * (fuBoxW + fuBoxMargin) + fuBoxMargin + 40 + peMappingMargin);
      var peContentH = 0;

      // Pre-compute FU heights (collapsed for unused FUs)
      var fuHeights = [];
      fus.forEach(function(fu) {
        if (mappingEnabled && hasMapping && !peUsedFUs[fu.name]) {
          fuHeights.push(COLLAPSED_FU_H);
        } else {
          var numOps = fu.ops ? fu.ops.length : 0;
          var h = 30 + Math.max(numOps, 1) * 28 + 30;
          fuHeights.push(h);
        }
      });
      var maxFuH = 0;
      fuHeights.forEach(function(h) { maxFuH = Math.max(maxFuH, h); });
      var peH = 30 + maxFuH + 30; // title + FU area + bottom margin

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
      if (peTypePx < peW * 0.5) {
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
        var fuH = fuHeights[fi];
        var fuW = fuBoxW;
        var isCollapsedFU = mappingEnabled && hasMapping && !peUsedFUs[fu.name];

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
        if (fuTypePx < fuW * 0.5) {
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

        // FU border ports: squares on TOP (input) and BOTTOM (output)
        // These match the mini-DFG I/O nodes in style and order (left-to-right)
        var fuBorderInPorts = [];
        for (var ip = 0; ip < fu.numIn; ip++) {
          var px = fuX + fuW * (ip + 1) / (fu.numIn + 1);
          fuBorderInPorts.push({ x: px, y: fuY, idx: ip });
          fuPortPos[peLabel + "/" + fu.name + "/in_" + ip] = {x: px, y: fuY};
          g.append("rect").attr("x", px - 4).attr("y", fuY - 4)
            .attr("width", 8).attr("height", 8)
            .attr("fill", "#2a5f5f").attr("stroke", "#4ecdc4").attr("stroke-width", 1);
          g.append("text").attr("x", px).attr("y", fuY - 7)
            .attr("text-anchor", "middle")
            .attr("fill", "rgba(78,205,196,0.7)").attr("font-size", "6px")
            .text("I" + ip);
        }
        var fuBorderOutPorts = [];
        for (var op = 0; op < fu.numOut; op++) {
          var px = fuX + fuW * (op + 1) / (fu.numOut + 1);
          fuBorderOutPorts.push({ x: px, y: fuY + fuH, idx: op });
          fuPortPos[peLabel + "/" + fu.name + "/out_" + op] = {x: px, y: fuY + fuH};
          g.append("rect").attr("x", px - 4).attr("y", fuY + fuH - 4)
            .attr("width", 8).attr("height", 8)
            .attr("fill", "#5f2a1a").attr("stroke", "#ff6b35").attr("stroke-width", 1);
          g.append("text").attr("x", px).attr("y", fuY + fuH + 12)
            .attr("text-anchor", "middle")
            .attr("fill", "rgba(255,107,53,0.7)").attr("font-size", "6px")
            .text("O" + op);
        }

        // FU internals: Graphviz renders ops + internal I/O nodes + edges
        if (fu.dot) {
          var fuInnerPadX = 10;
          var fuInnerTop = 24;
          var fuInnerBottom = 18;
          pendingFUDots.push({
            dot: fu.dot,
            x: fuX + fuInnerPadX,
            y: fuY + fuInnerTop,
            w: Math.max(24, fuW - fuInnerPadX * 2),
            h: Math.max(18, fuH - fuInnerTop - fuInnerBottom)
          });

        } else {
          // No ops - just show FU name
          g.append("text").attr("x", fuX + fuW/2).attr("y", fuY + fuH/2 + 4)
            .attr("text-anchor", "middle")
            .attr("fill", "rgba(255,255,255,0.4)").attr("font-size", "10px")
            .text("(empty)");
        }

        fuX += fuW + fuBoxMargin;
      });

      // PE internal routing: mux/demux connections from PE ports to used FU ports
      if (mappingEnabled && hasMapping && MAPPING_DATA && MAPPING_DATA.edge_routings) {
        var peEdgeG = g.append("g").attr("class", "pe-internal-edges");
        routePEInternal(peEdgeG, peLabel, peX, peY, peW, peH, fus, fuHeights,
                        fuBoxW, fuBoxMargin, peMappingMargin, portRegistry, data);
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
      if (swTypePx < swW * 0.5) {
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
  if (modTypePx < modW * 0.5) {
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
      open.push({ gc: start.gc, gr: start.gr, dir: -1, f: 0 });

      var goalKey = null;
      while (open.length > 0) {
        open.sort(function(a, b) { return a.f - b.f; });
        var cur = open.shift();
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
          open.push({ gc: nc, gr: nr, dir: di, f: newCost + heuristic });
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

  // Render pending FU DOTs with Graphviz (async)
  if (pendingFUDots.length > 0 && typeof Viz !== "undefined" && typeof Viz.instance === "function") {
    d3.select("#status-bar").text("Rendering FU DAGs...");
    Viz.instance().then(function(viz) {
      pendingFUDots.forEach(function(item) {
        try {
          var svgStr = viz.renderString(item.dot, { engine: "dot", format: "svg" });
          var wrapper = document.createElement("div");
          wrapper.innerHTML = svgStr;
          var renderedSvg = wrapper.querySelector("svg");
          if (!renderedSvg) return;
          // Use nested <svg> with viewBox for clean containment
          var vb = renderedSvg.getAttribute("viewBox");
          if (!vb) {
            var rw = parseFloat(renderedSvg.getAttribute("width")) || 100;
            var rh = parseFloat(renderedSvg.getAttribute("height")) || 80;
            vb = "0 0 " + rw + " " + rh;
          }
          var nested = g.append("svg")
            .attr("x", item.x).attr("y", item.y)
            .attr("width", item.w).attr("height", item.h)
            .attr("viewBox", vb)
            .attr("preserveAspectRatio", "xMidYMid meet")
            .style("overflow", "hidden");
          var innerG = renderedSvg.querySelector("g");
          if (innerG) nested.node().appendChild(innerG.cloneNode(true));
        } catch(err) {
          console.error("FU DOT render error:", err);
        }
      });
      d3.select("#status-bar").text("Ready");
      setTimeout(function() { fitView(svg, g, zoom); }, 200);
    }).catch(function(err) {
      console.error("Viz.js FU error:", err);
      d3.select("#status-bar").text("Ready (FU render failed)");
    });
  } else {
    d3.select("#status-bar").text("Ready");
  }

  // Fit to view
  setTimeout(function() { fitView(svg, g, zoom); }, 100);
}

// ============================================================
// DFG Renderer: Graphviz WASM
// ============================================================

renderDFG._svg = null;
renderDFG._g = null;
renderDFG._zoom = null;

function renderDFG() {
  var svg = d3.select("#svg-dfg");
  if (!DFG_DATA || DFG_DATA === "null" || !DFG_DATA.dot) {
    svg.append("text").attr("x",20).attr("y",40).attr("fill","#888")
      .text("No DFG data.");
    return;
  }

  d3.select("#status-bar").text("Rendering DFG...");

  // Try Viz.js
  if (typeof Viz !== "undefined" && typeof Viz.instance === "function") {
    Viz.instance().then(function(viz) {
      var svgStr = viz.renderString(DFG_DATA.dot, { engine: "dot", format: "svg" });
      insertDFG(svgStr);
    }).catch(function(err) {
      console.error("Viz.js error:", err);
      svg.append("text").attr("x",20).attr("y",40).attr("fill","#888")
        .text("Graphviz failed: " + err);
    });
  } else {
    svg.append("text").attr("x",20).attr("y",40).attr("fill","#888")
      .text("Graphviz WASM not loaded.");
  }

  function insertDFG(svgStr) {
    var container = document.getElementById("panel-dfg");
    var wrapper = document.createElement("div");
    wrapper.innerHTML = svgStr;
    var rendered = wrapper.querySelector("svg");
    if (!rendered) return;
    applyDFGTheme(rendered);
    rendered.setAttribute("id", "svg-dfg");
    rendered.removeAttribute("viewBox");
    rendered.removeAttribute("width");
    rendered.removeAttribute("height");
    rendered.style.width = "100%";
    rendered.style.height = "100%";
    var old = document.getElementById("svg-dfg");
    if (old) old.parentNode.replaceChild(rendered, old);

    var dfgSvg = d3.select("#svg-dfg");
    var innerG = dfgSvg.select("g");
    var dfgZoom = d3.zoom().scaleExtent([0.1, 5])
      .on("zoom", function(ev) { innerG.attr("transform", ev.transform); });
    dfgSvg.call(dfgZoom);

    // Store DFG refs for cross-highlighting
    renderDFG._svg = dfgSvg;
    renderDFG._g = innerG;
    renderDFG._zoom = dfgZoom;

    // Setup DFG node click handlers for cross-highlighting
    setupDFGInteraction();

    // Click DFG background to clear
    dfgSvg.on("click", function() { clearCrossHighlight(); });

    setTimeout(function() { fitView(dfgSvg, innerG, dfgZoom); }, 100);
    d3.select("#status-bar").text("Ready");
  }
}

function applyDFGTheme(rendered) {
  rendered.style.background = "transparent";

  var graphBg = rendered.querySelector("g.graph > polygon");
  if (graphBg) {
    graphBg.setAttribute("fill", "transparent");
    graphBg.setAttribute("stroke", "none");
  }

  rendered.querySelectorAll("g.edge path").forEach(function(path) {
    path.setAttribute("stroke", "#7f95b2");
    path.setAttribute("stroke-width", "1.45");
    path.setAttribute("fill", "none");
  });
  rendered.querySelectorAll("g.edge polygon").forEach(function(poly) {
    poly.setAttribute("fill", "#7f95b2");
    poly.setAttribute("stroke", "#7f95b2");
  });

  rendered.querySelectorAll("g.node text").forEach(function(text) {
    text.setAttribute("fill", "#c8d6e5");
    text.setAttribute("font-family", "JetBrains Mono, monospace");
  });
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
  if (!mappingIdx) return;
  var dfgEl = document.getElementById("svg-dfg");
  if (!dfgEl) return;

  // Graphviz SVG nodes have <g class="node"><title>nX</title>...</g>
  var nodeGs = dfgEl.querySelectorAll("g.node");
  nodeGs.forEach(function(nodeG) {
    if (nodeG.getAttribute("data-fcc-bound") === "1") return;
    var title = nodeG.querySelector("title");
    if (!title) return;
    var nid = title.textContent.trim();
    var match = nid.match(/^n(\d+)$/);
    if (!match) return;
    var idx = parseInt(match[1], 10);
    nodeG.setAttribute("data-dfg-node", idx);
    nodeG.setAttribute("data-fcc-bound", "1");
    nodeG.style.cursor = "pointer";
    nodeG.addEventListener("click", function(ev) {
      ev.stopPropagation();
      onDFGNodeClick(idx);
    });
  });

  // DFG edges: <g class="edge"><title>nX->nY</title>...</g>
  var edgeGs = dfgEl.querySelectorAll("g.edge");
  edgeGs.forEach(function(edgeG) {
    if (edgeG.getAttribute("data-fcc-bound") === "1") return;
    var title = edgeG.querySelector("title");
    if (!title) return;
    edgeG.setAttribute("data-fcc-bound", "1");
    edgeG.style.cursor = "pointer";
    edgeG.addEventListener("click", function(ev) {
      ev.stopPropagation();
      // Parse edge title to find source/dest nodes
      var text = title.textContent.trim();
      var parts = text.match(/^n(\d+)->n(\d+)$/);
      if (parts) {
        var fromIdx = parseInt(parts[1], 10);
        var toIdx = parseInt(parts[2], 10);
        onDFGEdgeClick(fromIdx, toIdx);
      }
    });
  });
}

// ============================================================
// PE internal routing (mux/demux connections)
// Uses A* grid routing at PE scale, same approach as module level
// ============================================================

function routePEInternal(edgeG, peLabel, peX, peY, peW, peH,
                         fus, fuHeights, fuBoxW, fuBoxMargin, mappingMargin,
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
      w: fuBoxW,
      h: fuHeights[fi]
    });
    fxCur += fuBoxW + fuBoxMargin;
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
    open.push({gc:start.gc, gr:start.gr, dir:-1, f:0});
    var goalK = null;
    while (open.length > 0) {
      open.sort(function(a,b){return a.f - b.f;});
      var cur = open.shift();
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
        open.push({gc:nc, gr:nr, dir:di, f:newC + Math.abs(nc-goal.gc)+Math.abs(nr-goal.gr)});
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
        if (p && p.kind === "sw") used[p.component] = true;
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
      if (nodeInfo.kind === "input") {
        // Highlight module input port and all edges from this node
        highlightModulePort("module_in_" + nodeInfo.label.replace("arg", ""));
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
  var shape = nodeG.querySelector("ellipse, polygon, path");
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
