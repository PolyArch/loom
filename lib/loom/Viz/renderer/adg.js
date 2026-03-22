// ============================================================
// ADG Renderer: draws PE-like modules with nested FUs
// ============================================================

function computeADGModuleMargin(contentW, contentH, areaRatio) {
  if (!(contentW > 0) || !(contentH > 0)) return 24;
  var sum = contentW + contentH;
  var extraArea = Math.max(0, areaRatio) * contentW * contentH;
  var disc = sum * sum + 4 * extraArea;
  var margin = (Math.sqrt(Math.max(0, disc)) - sum) / 4;
  return Math.max(24, Math.round(margin));
}

function normalizeVizName(name) {
  if (typeof name !== "string") return name;
  if (name.indexOf("__add_tag_") === 0 ||
      name.indexOf("__del_tag_") === 0 ||
      name.indexOf("__map_tag_") === 0)
    return name.slice(2);
  return name;
}

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
  var moduleInputLabels = ((((typeof ADG_LAYOUT_DATA !== "undefined") &&
    ADG_LAYOUT_DATA) ? ADG_LAYOUT_DATA.moduleInputLabels : null) || []);
  var moduleOutputLabels = ((((typeof ADG_LAYOUT_DATA !== "undefined") &&
    ADG_LAYOUT_DATA) ? ADG_LAYOUT_DATA.moduleOutputLabels : null) || []);
  var explicitLayoutByName = {};
  var explicitRouteByKey = {};
  var SWITCH_PORT_SIZE = 8;
  var SWITCH_PORT_PITCH = 24;
  var SWITCH_MIN_SIDE = 84;
  var SWITCH_INPUT_SIDE_ORDER = ["top", "left"];
  var SWITCH_OUTPUT_SIDE_ORDER = ["right", "bottom"];
  var PE_PORT_SIZE = 8;
  var PE_INPUT_SIDE_ORDER = ["top", "left"];
  var PE_OUTPUT_SIDE_ORDER = ["right", "bottom"];
  function buildPortSideCounts(count, order) {
    var counts = {};
    order.forEach(function(side) { counts[side] = 0; });
    for (var i = 0; i < count; ++i)
      counts[order[i % order.length]] += 1;
    return counts;
  }
  function buildSwitchPortLayout(numInputs, numOutputs) {
    var inCounts = buildPortSideCounts(numInputs, SWITCH_INPUT_SIDE_ORDER);
    var outCounts = buildPortSideCounts(numOutputs, SWITCH_OUTPUT_SIDE_ORDER);
    var maxSideSlots = 0;
    SWITCH_INPUT_SIDE_ORDER.forEach(function(side) {
      maxSideSlots = Math.max(maxSideSlots, inCounts[side]);
    });
    SWITCH_OUTPUT_SIDE_ORDER.forEach(function(side) {
      maxSideSlots = Math.max(maxSideSlots, outCounts[side]);
    });
    var side = Math.max(SWITCH_MIN_SIDE,
      32 + (Math.max(1, maxSideSlots) + 1) * SWITCH_PORT_PITCH);
    return {
      side: side,
      inCounts: inCounts,
      outCounts: outCounts
    };
  }
  function getSwitchPortPoint(swX, swY, swW, swH, side, slotIndex, slotCount) {
    var ratio = (slotIndex + 1) / (slotCount + 1);
    if (side === "top")
      return { x: swX + swW * ratio, y: swY, nx: 0, ny: -1 };
    if (side === "bottom")
      return { x: swX + swW * ratio, y: swY + swH, nx: 0, ny: 1 };
    if (side === "left")
      return { x: swX, y: swY + swH * ratio, nx: -1, ny: 0 };
    return { x: swX + swW, y: swY + swH * ratio, nx: 1, ny: 0 };
  }
  function getPEPortPoint(peX, peY, peW, peH, side, slotIndex, slotCount) {
    var ratio = (slotIndex + 1) / (slotCount + 1);
    if (side === "top")
      return { x: peX + peW * ratio, y: peY, nx: 0, ny: -1 };
    if (side === "bottom")
      return { x: peX + peW * ratio, y: peY + peH, nx: 0, ny: 1 };
    if (side === "left")
      return { x: peX, y: peY + peH * ratio, nx: -1, ny: 0 };
    return { x: peX + peW, y: peY + peH * ratio, nx: 1, ny: 0 };
  }
  (((typeof ADG_LAYOUT_DATA !== "undefined") && ADG_LAYOUT_DATA &&
    ADG_LAYOUT_DATA.components) || []).forEach(function(entry) {
    if (!entry || !entry.name) return;
    explicitLayoutByName[normalizeVizName(entry.name)] = entry;
  });
  (((typeof ADG_LAYOUT_DATA !== "undefined") && ADG_LAYOUT_DATA &&
    ADG_LAYOUT_DATA.routes) || []).forEach(function(route) {
    if (!route || !route.from || route.from_port === undefined ||
        !route.to || route.to_port === undefined)
      return;
    var fromName = route.from === "module_in" ? route.from : normalizeVizName(route.from);
    var toName = route.to === "module_out" ? route.to : normalizeVizName(route.to);
    var fromKey = fromName === "module_in"
      ? "module_in_" + route.from_port
      : fromName + "_out_" + route.from_port;
    var toKey = toName === "module_out"
      ? "module_out_" + route.to_port
      : toName + "_in_" + route.to_port;
    var key = fromKey + "->" + toKey;
    explicitRouteByKey[key] = route;
  });
  var explicitRoutePoints = [];
  Object.keys(explicitRouteByKey).forEach(function(key) {
    var route = explicitRouteByKey[key];
    (route.points || []).forEach(function(pt) {
      if (!pt || typeof pt.x !== "number" || typeof pt.y !== "number") return;
      explicitRoutePoints.push({ x: pt.x, y: pt.y });
    });
  });
  adgRenderHints = computeADGRenderHints(data, mappingEnabled);
  var portRegistry = {};

  // Export refs for fitView and cross-highlighting
  renderADG._svg = svg; renderADG._g = g; renderADG._zoom = zoom;
  renderADG._portRegistry = portRegistry;

  // Pre-compute internal content dimensions to determine module margin
  var maxPeW = 0, maxPeH = 0;
  data.components.forEach(function(c) {
    if (c.kind === "spatial_pe" || c.kind === "temporal_pe") {
      var preLayout = buildPERenderLayout(c, mappingEnabled, {}, adgRenderHints);
      maxPeW = Math.max(maxPeW, preLayout.peW);
      maxPeH = Math.max(maxPeH, preLayout.peH);
    }
  });

  // Also compute SW dimensions
  var maxSwW = 0, maxSwH = 0;
  data.components.forEach(function(c) {
    if (c.kind === "spatial_sw" || c.kind === "temporal_sw") {
      var swLayout = buildSwitchPortLayout(c.numInputs || 4, c.numOutputs || 4);
      var side = swLayout.side;
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
  var wrapComponents = !!(adgRenderHints && adgRenderHints.wrapComponents);
  var wrapRowWidth = adgRenderHints && adgRenderHints.wrapRowWidth ? adgRenderHints.wrapRowWidth : 0;
  var skipDenseModuleEdges = false;
  var rowY = compStartY;
  var rowBottom = compStartY;

  function reserveComponentSlot(boxW, boxH, gapX) {
    if (!wrapComponents || !wrapRowWidth) return null;
    if (xOff > compStartX && (xOff - compStartX + boxW) > wrapRowWidth) {
      xOff = compStartX;
      rowY = rowBottom + 40;
    }
    var pos = { x: xOff, y: rowY };
    xOff = pos.x + boxW + gapX;
    rowBottom = Math.max(rowBottom, pos.y + boxH);
    yOff = rowBottom + 20;
    return pos;
  }

  function getExplicitComponentPos(name, boxW, boxH) {
    var entry = explicitLayoutByName[name];
    if (!entry) return null;
    if (typeof entry.center_x !== "number" || typeof entry.center_y !== "number")
      return null;
    return {
      x: entry.center_x - boxW / 2,
      y: entry.center_y - boxH / 2
    };
  }
  function getExplicitComponentSize(name, fallbackW, fallbackH) {
    var entry = explicitLayoutByName[name];
    if (!entry) return { width: fallbackW, height: fallbackH };
    var width = typeof entry.width === "number" ? entry.width : fallbackW;
    var height = typeof entry.height === "number" ? entry.height : fallbackH;
    return { width: width, height: height };
  }
  function computeMemoryBoxHeight(numInputs, numOutputs) {
    var sidePortCount = Math.max(1, Math.max(Math.max(0, numInputs - 1), numOutputs));
    return Math.max(96, 40 + sidePortCount * 26);
  }

  // Draw each component
  data.components.forEach(function(comp) {
    if (comp.kind === "spatial_pe" || comp.kind === "temporal_pe" ||
        (comp.kind === "instance" && comp.module)) {
      // Find the PE definition for instances
      var peDef = comp;
      if (comp.kind === "instance") {
        data.components.forEach(function(c) {
          if ((c.kind === "spatial_pe" || c.kind === "temporal_pe") &&
              c.name === comp.module) peDef = c;
        });
      }

      var peLabel = comp.kind === "instance" ? comp.name : peDef.name;

      // Determine which FUs are used by mapping (for collapsing)
      var peUsedFUs = {};
      if (mappingEnabled && MAPPING_DATA && MAPPING_DATA.node_mappings) {
        MAPPING_DATA.node_mappings.forEach(function(m) {
          if (m.pe_name === peLabel) peUsedFUs[m.hw_name] = true;
        });
      }
      var hasMapping = Object.keys(peUsedFUs).length > 0;
      var peLayout = buildPERenderLayout(peDef, mappingEnabled, peUsedFUs, adgRenderHints);
      var fus = peLayout.fus;
      var fuLayouts = peLayout.fuLayouts;
      var fuWidths = peLayout.fuWidths;
      var fuHeights = peLayout.fuHeights;
      var fuBoxes = peLayout.fuBoxes;
      var fuBoxMargin = peLayout.fuBoxMargin;
      var peMappingMargin = peLayout.peMappingMargin;
      var peInnerPadX = peLayout.peInnerPadX;
      var peInnerPadY = peLayout.peInnerPadY;
      var peW = peLayout.peW;
      var peH = peLayout.peH;
      var pePos = getExplicitComponentPos(peLabel, peW, peH);
      var peAutoPos = pePos ? null : reserveComponentSlot(peW, peH, 40);
      var peX = pePos ? pePos.x : (peAutoPos ? peAutoPos.x : xOff);
      var peY = pePos ? pePos.y : (peAutoPos ? peAutoPos.y : yOff);

      // Store bounding box for edge routing avoidance
      comp._bbox = { name: peLabel, x: peX, y: peY, w: peW, h: peH };
      adgCompBoxes[peLabel] = comp._bbox;

      // PE border
      g.append("rect").attr("x", peX).attr("y", peY)
        .attr("width", peW).attr("height", peH)
        .attr("rx", 6).attr("fill", "#1a2f4a")
        .attr("stroke", "#2a5f8f").attr("stroke-width", 2)
        .attr("data-comp-name", peLabel).attr("data-comp-kind", peDef.kind || comp.kind);

      // PE labels: type top-left (one or two lines based on width)
      var peIsTemporal = (peDef.kind === "temporal_pe" || comp.kind === "temporal_pe");
      var peTypeStr = peIsTemporal ? "fabric.temporal_pe" : "fabric.spatial_pe";
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
          .text(peIsTemporal ? "temporal_pe" : "spatial_pe");
      }
      g.append("text").attr("x", peX + peW - 6).attr("y", peY + peH - 6)
        .attr("text-anchor", "end")
        .attr("fill", "rgba(78,205,196,0.6)").attr("font-size", "9px")
        .text(peLabel);

      var peInCount = peDef.numInputs || 3;
      var peOutCount = peDef.numOutputs || 2;
      var peInputSideCounts = buildPortSideCounts(peInCount, PE_INPUT_SIDE_ORDER);
      var peOutputSideCounts = buildPortSideCounts(peOutCount, PE_OUTPUT_SIDE_ORDER);
      var peInputSideOffsets = { top: 0, left: 0 };
      var peOutputSideOffsets = { right: 0, bottom: 0 };

      for (var pi = 0; pi < peInCount; pi++) {
        var inputSide = PE_INPUT_SIDE_ORDER[pi % PE_INPUT_SIDE_ORDER.length];
        var inputSlot = peInputSideOffsets[inputSide]++;
        var inputPt = getPEPortPoint(peX, peY, peW, peH, inputSide,
                                     inputSlot, peInputSideCounts[inputSide]);
        portRegistry[peLabel + "_in_" + pi] = {
          x: inputPt.x, y: inputPt.y,
          side: inputSide, owner: peLabel, ownerKind: "component",
          nx: inputPt.nx, ny: inputPt.ny
        };
        g.append("rect").attr("x", inputPt.x - PE_PORT_SIZE / 2)
          .attr("y", inputPt.y - PE_PORT_SIZE / 2)
          .attr("width", PE_PORT_SIZE).attr("height", PE_PORT_SIZE)
          .attr("fill", "#4ecdc4").attr("stroke", "#0c1220").attr("stroke-width", 1);
        if (inputSide === "top") {
          g.append("text").attr("x", inputPt.x).attr("y", inputPt.y - 7)
            .attr("text-anchor", "middle")
            .attr("fill", "rgba(78,205,196,0.7)").attr("font-size", "8px")
            .text("I" + pi);
        } else {
          g.append("text").attr("x", inputPt.x - 7).attr("y", inputPt.y + 3)
            .attr("text-anchor", "end")
            .attr("fill", "rgba(78,205,196,0.7)").attr("font-size", "8px")
            .text("I" + pi);
        }
      }

      for (var po = 0; po < peOutCount; po++) {
        var outputSide = PE_OUTPUT_SIDE_ORDER[po % PE_OUTPUT_SIDE_ORDER.length];
        var outputSlot = peOutputSideOffsets[outputSide]++;
        var outputPt = getPEPortPoint(peX, peY, peW, peH, outputSide,
                                      outputSlot, peOutputSideCounts[outputSide]);
        portRegistry[peLabel + "_out_" + po] = {
          x: outputPt.x, y: outputPt.y,
          side: outputSide, owner: peLabel, ownerKind: "component",
          nx: outputPt.nx, ny: outputPt.ny
        };
        g.append("rect").attr("x", outputPt.x - PE_PORT_SIZE / 2)
          .attr("y", outputPt.y - PE_PORT_SIZE / 2)
          .attr("width", PE_PORT_SIZE).attr("height", PE_PORT_SIZE)
          .attr("fill", "#ff6b35").attr("stroke", "#0c1220").attr("stroke-width", 1);
        if (outputSide === "bottom") {
          g.append("text").attr("x", outputPt.x).attr("y", outputPt.y + 16)
            .attr("text-anchor", "middle")
            .attr("fill", "rgba(255,107,53,0.7)").attr("font-size", "8px")
            .text("O" + po);
        } else {
          g.append("text").attr("x", outputPt.x + 7).attr("y", outputPt.y + 3)
            .attr("text-anchor", "start")
            .attr("fill", "rgba(255,107,53,0.7)").attr("font-size", "8px")
            .text("O" + po);
        }
      }

      // Draw each FU inside the PE
      fus.forEach(function(fu, fi) {
        var fuLayoutSpec = fuLayouts[fi];
        var fuBox = fuBoxes[fi] || {
          x: peInnerPadX + (peMappingMargin / 2),
          y: peInnerPadY,
          w: fuWidths[fi],
          h: fuHeights[fi]
        };
        var fuX = peX + fuBox.x;
        var fuY = peY + fuBox.y;
        var fuH = fuBox.h;
        var fuW = fuBox.w;
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
          var fuConfig = null;
          if (mappingEnabled && fuConfigIdx && fuConfigIdx.byKey)
            fuConfig = fuConfigIdx.byKey[peLabel + "/" + fu.name] || null;
          renderMiniFUGraph(g, {
            fu: fu,
            fuConfig: fuConfig,
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

      });

      // PE internal routing: mux/demux connections from PE ports to used FU ports
      if (mappingEnabled && hasMapping && MAPPING_DATA && MAPPING_DATA.edge_routings) {
        var peEdgeG = g.append("g").attr("class", "pe-internal-edges");
        routePEInternal(peEdgeG, peLabel, peX, peY, peW, peH, fuBoxes,
                        portRegistry, data);
      }

      if (!pePos && !peAutoPos) {
        xOff = peX + peW + 40; // Move xOff past PE for next component
        yOff = Math.max(yOff, peY + peH + 20);
      }
    }

    // spatial_sw / temporal_sw rendering
    if (comp.kind === "spatial_sw" || comp.kind === "temporal_sw") {
      var swLabel = comp.name;
      var swInCount = comp.numInputs || 4;
      var swOutCount = comp.numOutputs || 4;
      var swLayout = buildSwitchPortLayout(swInCount, swOutCount);
      var swW = swLayout.side;
      var swH = swW; // Square
      var swPos = getExplicitComponentPos(swLabel, swW, swH);
      var swAutoPos = swPos ? null : reserveComponentSlot(swW, swH, 40);
      var swX = swPos ? swPos.x : (swAutoPos ? swAutoPos.x : xOff);
      var swY = swPos ? swPos.y
        : (swAutoPos ? swAutoPos.y : compStartY + (totalContentH - swH) / 2); // Vertically centered
      var isTemporalSW = comp.kind === "temporal_sw";
      var swFill = isTemporalSW ? "#3a334d" : "#2a3f5f";
      var swStroke = isTemporalSW ? "#a98fe0" : "#4a6380";
      var swText = isTemporalSW ? "#d5c6ff" : "#8a9bb5";
      var swTypeStr = isTemporalSW ? "fabric.temporal_sw" : "fabric.spatial_sw";

      comp._bbox = { name: swLabel, x: swX, y: swY, w: swW, h: swH };
      adgCompBoxes[swLabel] = comp._bbox;

      // SW border
      g.append("rect").attr("x", swX).attr("y", swY)
        .attr("width", swW).attr("height", swH)
        .attr("rx", 4).attr("fill", swFill)
        .attr("stroke", swStroke).attr("stroke-width", 2)
        .attr("data-comp-name", swLabel).attr("data-comp-kind", comp.kind);

      // SW labels: type top-left (one or two lines based on width)
      var swTypePx = swTypeStr.length * 6.5;
      if (swTypePx < swW * TYPE_LABEL_SINGLE_LINE_RATIO) {
        g.append("text").attr("x", swX + 6).attr("y", swY + 14)
          .attr("fill", swText).attr("font-size", "9px").attr("font-weight", "600")
          .text(swTypeStr);
      } else {
        g.append("text").attr("x", swX + 6).attr("y", swY + 12)
          .attr("fill", swText).attr("font-size", "9px").attr("font-weight", "600")
          .text("fabric");
        g.append("text").attr("x", swX + 6).attr("y", swY + 22)
          .attr("fill", swText).attr("font-size", "9px").attr("font-weight", "600")
          .text(isTemporalSW ? "temporal_sw" : "spatial_sw");
      }
      g.append("text").attr("x", swX + swW - 6).attr("y", swY + swH - 6)
        .attr("text-anchor", "end")
        .attr("fill", isTemporalSW ? "rgba(213,198,255,0.65)" : "rgba(138,155,181,0.6)")
        .attr("font-size", "8px")
        .text(swLabel);

      var inputSideOffsets = { top: 0, left: 0 };
      for (var pi = 0; pi < swInCount; pi++) {
        var inputSide = SWITCH_INPUT_SIDE_ORDER[pi % SWITCH_INPUT_SIDE_ORDER.length];
        var inputSideLocalIdx = inputSideOffsets[inputSide]++;
        var inputSlotIdx = inputSideLocalIdx;
        var inputSlotCount = swLayout.inCounts[inputSide];
        var inputPt = getSwitchPortPoint(swX, swY, swW, swH,
                                         inputSide, inputSlotIdx,
                                         inputSlotCount);
        portRegistry[swLabel + "_in_" + pi] = {
          x: inputPt.x, y: inputPt.y,
          side: inputSide, owner: swLabel, ownerKind: "component",
          nx: inputPt.nx, ny: inputPt.ny
        };
        g.append("rect").attr("x", inputPt.x - SWITCH_PORT_SIZE / 2)
          .attr("y", inputPt.y - SWITCH_PORT_SIZE / 2)
          .attr("width", SWITCH_PORT_SIZE).attr("height", SWITCH_PORT_SIZE)
          .attr("fill", "#4ecdc4").attr("stroke", "#0c1220").attr("stroke-width", 1);
        if (inputSide === "top") {
          g.append("text").attr("x", inputPt.x).attr("y", inputPt.y - 7)
            .attr("text-anchor", "middle")
            .attr("fill", "rgba(78,205,196,0.7)").attr("font-size", "7px")
            .text("I" + pi);
        } else if (inputSide === "bottom") {
          g.append("text").attr("x", inputPt.x).attr("y", inputPt.y + 14)
            .attr("text-anchor", "middle")
            .attr("fill", "rgba(78,205,196,0.7)").attr("font-size", "7px")
            .text("I" + pi);
        } else if (inputSide === "left") {
          g.append("text").attr("x", inputPt.x - 7).attr("y", inputPt.y + 3)
            .attr("text-anchor", "end")
            .attr("fill", "rgba(78,205,196,0.7)").attr("font-size", "7px")
            .text("I" + pi);
        } else {
          g.append("text").attr("x", inputPt.x + 7).attr("y", inputPt.y + 3)
            .attr("text-anchor", "start")
            .attr("fill", "rgba(78,205,196,0.7)").attr("font-size", "7px")
            .text("I" + pi);
        }
      }

      var outputSideOffsets = { right: 0, bottom: 0 };
      for (var po = 0; po < swOutCount; po++) {
        var outputSide = SWITCH_OUTPUT_SIDE_ORDER[po % SWITCH_OUTPUT_SIDE_ORDER.length];
        var outputSideLocalIdx = outputSideOffsets[outputSide]++;
        var outputSlotIdx = outputSideLocalIdx;
        var outputSlotCount = swLayout.outCounts[outputSide];
        var outputPt = getSwitchPortPoint(swX, swY, swW, swH,
                                          outputSide, outputSlotIdx,
                                          outputSlotCount);
        portRegistry[swLabel + "_out_" + po] = {
          x: outputPt.x, y: outputPt.y,
          side: outputSide, owner: swLabel, ownerKind: "component",
          nx: outputPt.nx, ny: outputPt.ny
        };
        g.append("rect").attr("x", outputPt.x - SWITCH_PORT_SIZE / 2)
          .attr("y", outputPt.y - SWITCH_PORT_SIZE / 2)
          .attr("width", SWITCH_PORT_SIZE).attr("height", SWITCH_PORT_SIZE)
          .attr("fill", "#ff6b35").attr("stroke", "#0c1220").attr("stroke-width", 1);
        if (outputSide === "top") {
          g.append("text").attr("x", outputPt.x).attr("y", outputPt.y - 7)
            .attr("text-anchor", "middle")
            .attr("fill", "rgba(255,107,53,0.7)").attr("font-size", "7px")
            .text("O" + po);
        } else if (outputSide === "bottom") {
          g.append("text").attr("x", outputPt.x).attr("y", outputPt.y + 14)
            .attr("text-anchor", "middle")
            .attr("fill", "rgba(255,107,53,0.7)").attr("font-size", "7px")
            .text("O" + po);
        } else if (outputSide === "left") {
          g.append("text").attr("x", outputPt.x - 7).attr("y", outputPt.y + 3)
            .attr("text-anchor", "end")
            .attr("fill", "rgba(255,107,53,0.7)").attr("font-size", "7px")
            .text("O" + po);
        } else {
          g.append("text").attr("x", outputPt.x + 7).attr("y", outputPt.y + 3)
            .attr("text-anchor", "start")
            .attr("fill", "rgba(255,107,53,0.7)").attr("font-size", "7px")
            .text("O" + po);
        }
      }

      if (!swPos && !swAutoPos) {
        xOff = swX + swW + 40;
        yOff = Math.max(yOff, swY + swH + 20);
      }
    }

    if (comp.kind === "add_tag" || comp.kind === "del_tag" ||
        comp.kind === "map_tag" || comp.kind === "fifo") {
      var nodeLabel = comp.name;
      var nodeFallbackW = comp.kind === "fifo" ? 100 : 92;
      var nodeFallbackH = comp.kind === "fifo" ? 56 : 52;
      var nodeSize = getExplicitComponentSize(nodeLabel, nodeFallbackW, nodeFallbackH);
      var nodeW = nodeSize.width;
      var nodeH = nodeSize.height;
      var nodePos = getExplicitComponentPos(nodeLabel, nodeW, nodeH);
      var nodeAutoPos = nodePos ? null : reserveComponentSlot(nodeW, nodeH, 26);
      var nodeX = nodePos ? nodePos.x : (nodeAutoPos ? nodeAutoPos.x : xOff);
      var nodeY = nodePos ? nodePos.y
        : (nodeAutoPos ? nodeAutoPos.y : compStartY + (totalContentH - nodeH) / 2);
      var nodeFill = comp.kind === "add_tag" ? "#304d46"
        : comp.kind === "del_tag" ? "#4d3b2f"
        : comp.kind === "map_tag" ? "#413b52"
        : "#2f4050";
      var nodeStroke = comp.kind === "add_tag" ? "#4ecdc4"
        : comp.kind === "del_tag" ? "#ffb27a"
        : comp.kind === "map_tag" ? "#c6b3ff"
        : "#7aa0c0";
      var typeStr = comp.kind === "fifo" ? "fabric.fifo" : "fabric." + comp.kind;

      comp._bbox = { name: nodeLabel, x: nodeX, y: nodeY, w: nodeW, h: nodeH };
      adgCompBoxes[nodeLabel] = comp._bbox;

      g.append("rect").attr("x", nodeX).attr("y", nodeY)
        .attr("width", nodeW).attr("height", nodeH)
        .attr("rx", 4).attr("fill", nodeFill)
        .attr("stroke", nodeStroke).attr("stroke-width", 1.8)
        .attr("data-comp-name", nodeLabel).attr("data-comp-kind", comp.kind);

      renderTypeLabel(g, nodeX + 5, nodeY, nodeW - 10, typeStr, {
        fill: nodeStroke,
        fontSize: "8px",
        fontWeight: "600",
        charPx: 5.5,
        lineGap: 9,
        singleOffsetY: 12,
        multiOffsetY0: 11,
        multiOffsetY1: 20
      });
      g.append("text").attr("x", nodeX + nodeW - 5).attr("y", nodeY + nodeH - 6)
        .attr("text-anchor", "end")
        .attr("fill", "rgba(255,255,255,0.55)").attr("font-size", "7px")
        .text(nodeLabel);

      var inY = nodeY + nodeH / 2;
      portRegistry[nodeLabel + "_in_0"] = {
        x: nodeX, y: inY,
        side: "left", owner: nodeLabel, ownerKind: "component",
        nx: -1, ny: 0
      };
      g.append("rect").attr("x", nodeX - 5).attr("y", inY - 4)
        .attr("width", 10).attr("height", 8)
        .attr("fill", "#4ecdc4").attr("stroke", "#0c1220").attr("stroke-width", 1);

      portRegistry[nodeLabel + "_out_0"] = {
        x: nodeX + nodeW, y: inY,
        side: "right", owner: nodeLabel, ownerKind: "component",
        nx: 1, ny: 0
      };
      g.append("rect").attr("x", nodeX + nodeW - 5).attr("y", inY - 4)
        .attr("width", 10).attr("height", 8)
        .attr("fill", "#ff6b35").attr("stroke", "#0c1220").attr("stroke-width", 1);

      if (!nodePos && !nodeAutoPos) {
        xOff = nodeX + nodeW + 26;
        yOff = Math.max(yOff, nodeY + nodeH + 20);
      }
    }

    if (comp.kind === "memory") {
      var memLabel = comp.name;
      var memInCount = comp.numInputs || 1;
      var memOutCount = comp.numOutputs || 1;
      var hasTopMemref = comp.memoryKind === "extmemory";
      var inputLabels = comp.inputLabels || [];
      var outputLabels = comp.outputLabels || [];
      var hasBottomMemref = comp.memoryKind === "memory" &&
        memOutCount > 0 &&
        outputLabels.length > 0 &&
        outputLabels[memOutCount - 1] === "memref";
      var memSize = getExplicitComponentSize(memLabel, 170, computeMemoryBoxHeight(memInCount, memOutCount));
      var memW = memSize.width;
      var memH = memSize.height;
      var memPos = getExplicitComponentPos(memLabel, memW, memH);
      var memAutoPos = memPos ? null : reserveComponentSlot(memW, memH, 40);
      var memX = memPos ? memPos.x : (memAutoPos ? memAutoPos.x : xOff);
      var memY = memPos ? memPos.y
        : (memAutoPos ? memAutoPos.y : compStartY + (totalContentH - memH) / 2);

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

      if (hasTopMemref && memInCount > 0) {
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
          .text(inputLabels[0] || "memref");
      }

      var firstSideInput = hasTopMemref ? 1 : 0;
      for (var mi = firstSideInput; mi < memInCount; mi++) {
        var sideIdx = hasTopMemref ? mi : (mi + 1);
        var inY = memY + 20 + (memH - 30) * sideIdx / memInCount;
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
          .text(inputLabels[mi] || ("I" + mi));
      }

      for (var mo = 0; mo < memOutCount; mo++) {
        var isBottomMemref = hasBottomMemref && mo === memOutCount - 1;
        var outX = isBottomMemref ? (memX + memW / 2) : (memX + memW);
        var outY = isBottomMemref
          ? (memY + memH)
          : (memY + 20 + (memH - 30) * (mo + 1) / (memOutCount + 1));
        portRegistry[memLabel + "_out_" + mo] = {
          x: outX, y: outY,
          side: isBottomMemref ? "bottom" : "right",
          owner: memLabel, ownerKind: "component",
          nx: isBottomMemref ? 0 : 1, ny: isBottomMemref ? 1 : 0
        };
        g.append("rect").attr("x", outX - 5).attr("y", outY - 4)
          .attr("width", 10).attr("height", 8)
          .attr("fill", "#ff6b35").attr("stroke", "#0c1220").attr("stroke-width", 1);
        if (isBottomMemref) {
          g.append("text").attr("x", outX).attr("y", outY + 14)
            .attr("text-anchor", "middle")
            .attr("fill", "rgba(255,107,53,0.7)").attr("font-size", "7px")
            .text(outputLabels[mo] || ("O" + mo));
        } else {
          g.append("text").attr("x", memX + memW - 8).attr("y", outY + 3)
            .attr("text-anchor", "end")
            .attr("fill", "rgba(255,107,53,0.7)").attr("font-size", "7px")
            .text(outputLabels[mo] || ("O" + mo));
        }
      }

      if (!memPos && !memAutoPos) {
        xOff = memX + memW + 40;
        yOff = Math.max(yOff, memY + memH + 20);
      }
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
  // Internal content area is defined by component boxes only.
  var contentW = actualMaxX - actualMinX;
  var contentH = actualMaxY - actualMinY;
  // Margin: total margin area should be about 20% of content area.
  MOD_MARGIN = computeADGModuleMargin(contentW, contentH, 0.20);
  // First compute module bounds from component content only.
  modX = actualMinX - MOD_MARGIN;
  modY = actualMinY - MOD_MARGIN - 28; // -28 for title
  modW = contentW + MOD_MARGIN * 2;
  modH = contentH + MOD_MARGIN * 2 + 28;
  // Then expand minimally to include explicit precomputed routes.
  var modRight = modX + modW;
  var modBottom = modY + modH;
  explicitRoutePoints.forEach(function(pt) {
    modX = Math.min(modX, pt.x);
    modY = Math.min(modY, pt.y);
    modRight = Math.max(modRight, pt.x);
    modBottom = Math.max(modBottom, pt.y);
  });
  modW = modRight - modX;
  modH = modBottom - modY;

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
      .text(moduleInputLabels[i] || ("input" + i));
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
      .text(moduleOutputLabels[i] || ("output" + i));
  }

  // Draw module-level connections with a routed orthogonal channel graph.
  // Edges stay inside the module, avoid hardware bodies, never share a routed
  // segment, and use hop-over arcs only at true crossings.
  if (data.connections && !skipDenseModuleEdges) {
    var compBoxes = [];
    data.components.forEach(function(c) {
      if (c._bbox) compBoxes.push(c._bbox);
    });

    var edgeG = g.insert("g", ":first-child").attr("class", "hw-edges");
    var edgeHitG = g.append("g").attr("class", "hw-edge-hits");
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
        dstLane: dstLane,
        explicitRoute: explicitRouteByKey[fromKey + "->" + toKey] || null
      });
    });
    var allPlannedEdgesExplicit =
      plannedEdges.length > 0 &&
      plannedEdges.every(function(edgePlan) { return !!edgePlan.explicitRoute; });
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

    function buildRouteSceneSignature() {
      var compSig = data.components.filter(function(c) { return !!c._bbox; })
        .map(function(c) {
          return [
            c._bbox.name,
            Math.round(c._bbox.x),
            Math.round(c._bbox.y),
            Math.round(c._bbox.w),
            Math.round(c._bbox.h)
          ];
        }).sort(function(a, b) {
          return a[0] < b[0] ? -1 : (a[0] > b[0] ? 1 : 0);
        });
      var connSig = data.connections.map(function(conn) {
        return [conn.from, conn.fromIdx, conn.to, conn.toIdx];
      });
      return JSON.stringify({
        name: data.name || "",
        mod: [Math.round(modX), Math.round(modY), Math.round(modW), Math.round(modH)],
        in: data.numInputs || 0,
        out: data.numOutputs || 0,
        comps: compSig,
        conns: connSig
      });
    }

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
    function isGridPointBlocked(gc, gr) {
      return gc < 0 || gr < 0 || gc > gCols || gr > gRows || isBlocked(gc, gr);
    }
    function segmentGridPoints(a, b) {
      var pts = [];
      if (a.gc === b.gc) {
        var stepR = a.gr <= b.gr ? 1 : -1;
        for (var gr = a.gr; gr !== b.gr + stepR; gr += stepR)
          pts.push({ gc: a.gc, gr: gr });
        return pts;
      }
      if (a.gr === b.gr) {
        var stepC = a.gc <= b.gc ? 1 : -1;
        for (var gc = a.gc; gc !== b.gc + stepC; gc += stepC)
          pts.push({ gc: gc, gr: a.gr });
        return pts;
        }
      return null;
    }
    function segmentGridClear(a, b, allowReuse) {
      var pts = segmentGridPoints(a, b);
      if (!pts || pts.length === 0) return false;
      for (var i = 0; i < pts.length; i++) {
        if (isGridPointBlocked(pts[i].gc, pts[i].gr)) return false;
      }
      for (var i = 0; i < pts.length - 1; i++) {
        var segKey = edgeKey(pts[i].gc, pts[i].gr, pts[i + 1].gc, pts[i + 1].gr);
        if (!allowReuse && usedGridEdges[segKey]) return false;
      }
      return true;
    }
    function buildCandidateGridPaths(start, goal) {
      function midpoint(lo, hi) {
        return clamp(Math.round((lo + hi) / 2), 0, gCols);
      }
      function midpointRow(lo, hi) {
        return clamp(Math.round((lo + hi) / 2), 0, gRows);
      }
      var cands = [];
      cands.push([start, goal]);
      cands.push([start, { gc: goal.gc, gr: start.gr }, goal]);
      cands.push([start, { gc: start.gc, gr: goal.gr }, goal]);
      var midGc = midpoint(start.gc, goal.gc);
      var midGr = midpointRow(start.gr, goal.gr);
      cands.push([
        start,
        { gc: midGc, gr: start.gr },
        { gc: midGc, gr: goal.gr },
        goal
      ]);
      cands.push([
        start,
        { gc: start.gc, gr: midGr },
        { gc: goal.gc, gr: midGr },
        goal
      ]);
      return cands.map(function(path) {
        var out = [path[0]];
        for (var i = 1; i < path.length; i++) {
          var prev = out[out.length - 1];
          var cur = path[i];
          if (prev.gc === cur.gc && prev.gr === cur.gr) continue;
          out.push(cur);
        }
        return out;
      });
    }
    function trySimpleGridRoute(start, goal) {
      var candidates = buildCandidateGridPaths(start, goal);
      for (var ci = 0; ci < candidates.length; ci++) {
        var cand = candidates[ci];
        var ok = true;
        for (var i = 0; i < cand.length - 1; i++) {
          if (!segmentGridClear(cand[i], cand[i + 1], false)) {
            ok = false;
            break;
          }
        }
        if (ok) {
          var pts = [];
          cand.forEach(function(p, idx) {
            if (idx === 0) {
              pts.push({ gc: p.gc, gr: p.gr });
              return;
            }
            var segPts = segmentGridPoints(cand[idx - 1], p);
            if (!segPts) return;
            for (var si = 1; si < segPts.length; si++)
              pts.push(segPts[si]);
          });
          return pts;
        }
      }
      return null;
    }
    function routeGrid(start, goal, allowReuse) {
      var simplePath = trySimpleGridRoute(start, goal);
      if (simplePath) return simplePath;
      function routeGridWithBounds(bounds) {
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
            if (nc < bounds.minGc || nr < bounds.minGr ||
                nc > bounds.maxGc || nr > bounds.maxGr)
              continue;
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

      var localPad = Math.max(
        10,
        Math.min(48,
          Math.ceil((Math.abs(start.gc - goal.gc) + Math.abs(start.gr - goal.gr)) * 0.35))
      );
      var localBounds = {
        minGc: Math.max(0, Math.min(start.gc, goal.gc) - localPad),
        minGr: Math.max(0, Math.min(start.gr, goal.gr) - localPad),
        maxGc: Math.min(gCols, Math.max(start.gc, goal.gc) + localPad),
        maxGr: Math.min(gRows, Math.max(start.gr, goal.gr) + localPad)
      };
      var fullBounds = { minGc: 0, minGr: 0, maxGc: gCols, maxGr: gRows };
      return routeGridWithBounds(localBounds) || routeGridWithBounds(fullBounds);
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
    function buildSimpleEdgePathD(pts) {
      if (!pts || pts.length === 0) return "";
      var d = "M" + pts[0].x + "," + pts[0].y;
      for (var i = 1; i < pts.length; i++)
        d += "L" + pts[i].x + "," + pts[i].y;
      return d;
    }

    if (!allPlannedEdgesExplicit) {
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
    }

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
    if (allPlannedEdgesExplicit) {
      plannedEdges.forEach(function(edgePlan) {
        allEdgePts.push({
          order: edgePlan.originalIndex,
          fromKey: edgePlan.fromKey,
          toKey: edgePlan.toKey,
          pts: cleanPolyline([{ x: edgePlan.fromPort.x, y: edgePlan.fromPort.y }]
              .concat((edgePlan.explicitRoute.points || []).map(function(pt) {
                return { x: pt.x, y: pt.y };
              }))
              .concat([{ x: edgePlan.toPort.x, y: edgePlan.toPort.y }]))
        });
      });
    } else {
      var sceneSignature = buildRouteSceneSignature();
      var cachedEdgePts = adgModuleRouteCache[sceneSignature];
      if (cachedEdgePts) {
        allEdgePts = cachedEdgePts.map(function(edge) {
          return {
            order: edge.order,
            fromKey: edge.fromKey,
            toKey: edge.toKey,
            pts: edge.pts.map(function(pt) { return { x: pt.x, y: pt.y }; })
          };
        });
      } else {
        plannedEdges.forEach(function(edgePlan) {
          var startAnchor = null;
          var endAnchor = null;
          var gridPath = null;
          if (!edgePlan.explicitRoute) {
            startAnchor = anchorForPort(edgePlan.fromPort, edgePlan.srcLane);
            endAnchor = anchorForPort(edgePlan.toPort, edgePlan.dstLane);
            gridPath = routeAnchors(startAnchor, endAnchor);
            if (gridPath) reserveGridPath(gridPath);
          }

          allEdgePts.push({
            order: edgePlan.originalIndex,
            fromKey: edgePlan.fromKey,
            toKey: edgePlan.toKey,
            pts: edgePlan.explicitRoute
              ? cleanPolyline([{ x: edgePlan.fromPort.x, y: edgePlan.fromPort.y }]
                  .concat((edgePlan.explicitRoute.points || []).map(function(pt) {
                    return { x: pt.x, y: pt.y };
                  }))
                  .concat([{ x: edgePlan.toPort.x, y: edgePlan.toPort.y }]))
              : buildEdgePolyline(edgePlan.fromPort, edgePlan.toPort,
                                  startAnchor, endAnchor, gridPath)
          });
        });
        adgModuleRouteCache[sceneSignature] = allEdgePts.map(function(edge) {
          return {
            order: edge.order,
            fromKey: edge.fromKey,
            toKey: edge.toKey,
            pts: edge.pts.map(function(pt) { return { x: pt.x, y: pt.y }; })
          };
        });
      }
    }

    var allCrossings = null;
    if (!allPlannedEdgesExplicit) {
      allCrossings = allEdgePts.map(function() { return []; });
      for (var ai = 0; ai < allEdgePts.length; ai++) {
        for (var bi = ai + 1; bi < allEdgePts.length; bi++) {
          findCrossings(allEdgePts[bi], allEdgePts[ai]).forEach(function(c) {
            allCrossings[bi].push(c);
          });
        }
      }
    }

    allEdgePts.forEach(function(edge, ei) {
      var pathD = allPlannedEdgesExplicit
        ? buildSimpleEdgePathD(edge.pts)
        : buildEdgePathD(edge.pts, allCrossings[ei]);
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
    if (mappingEnabled)
      clearCrossHighlight();
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
    if (mappingEnabled && typeof onADGEdgeClick === "function")
      onADGEdgeClick(fk, tk);
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
