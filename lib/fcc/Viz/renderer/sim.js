// ============================================================
// Simulation playback
// ============================================================

var simTraceIdx = null;
var simulationEnabled = false;
var simulationFrameIndex = 0;
var simulationAutoPlay = false;
var simulationTimer = null;
var simulationFrameDelayMs = 220;

function buildSimulationIndex() {
  if (!SIM_TRACE_DATA || !SIM_TRACE_DATA.events || !Array.isArray(SIM_TRACE_DATA.events))
    return null;
  if (SIM_TRACE_DATA.version !== 1)
    return null;

  var idx = {
    version: SIM_TRACE_DATA.version,
    traceKind: SIM_TRACE_DATA.trace_kind || "",
    moduleByHwNode: {},
    incidentSwEdgesByHwNode: {},
    frames: [],
    frameByCycle: {},
    maxCycle: 0
  };

  (SIM_TRACE_DATA.modules || []).forEach(function(mod) {
    idx.moduleByHwNode[mod.hw_node_id] = mod;
  });

  function pushUnique(list, value) {
    if (value == null || list.indexOf(value) >= 0) return;
    list.push(value);
  }

  if (mappingIdx && mappingIdx.hwNodeToSwNodes && DFG_DATA &&
      Array.isArray(DFG_DATA.edges)) {
    Object.keys(mappingIdx.hwNodeToSwNodes).forEach(function(hwNodeId) {
      var swNodes = mappingIdx.hwNodeToSwNodes[hwNodeId] || [];
      if (swNodes.length === 0) return;
      var swNodeSet = {};
      swNodes.forEach(function(swNodeId) {
        swNodeSet[String(swNodeId)] = true;
      });
      var swEdges = [];
      DFG_DATA.edges.forEach(function(edge) {
        if (!edge) return;
        if (swNodeSet[String(edge.from)] || swNodeSet[String(edge.to)])
          pushUnique(swEdges, edge.id);
      });
      idx.incidentSwEdgesByHwNode[hwNodeId] = swEdges;
    });
  }

  var byCycle = {};
  SIM_TRACE_DATA.events.forEach(function(ev) {
    var cycle = ev.cycle || 0;
    if (!byCycle[cycle]) byCycle[cycle] = [];
    byCycle[cycle].push(ev);
    if (cycle > idx.maxCycle) idx.maxCycle = cycle;
  });

  Object.keys(byCycle)
    .map(function(key) { return parseInt(key, 10); })
    .sort(function(a, b) { return a - b; })
    .forEach(function(cycle, frameIdx) {
      idx.frameByCycle[cycle] = frameIdx;
      idx.frames.push({
        cycle: cycle,
        events: byCycle[cycle]
      });
    });

  return idx;
}

function clearSimulationHighlights() {
  if (renderADG._g)
    renderADG._g.selectAll(".sim-highlight-group").remove();

  var dfgEl = document.getElementById("svg-dfg");
  if (dfgEl) {
    dfgEl.querySelectorAll(".sim-highlight-dfg").forEach(function(el) {
      el.classList.remove("sim-highlight-dfg");
    });
    dfgEl.querySelectorAll(".sim-highlight-dfg-edge").forEach(function(el) {
      el.classList.remove("sim-highlight-dfg-edge");
    });
    dfgEl.querySelectorAll(".sim-highlight-dfg-arrow").forEach(function(el) {
      el.classList.remove("sim-highlight-dfg-arrow");
    });
  }
}

function ensureSimulationLayer() {
  if (!renderADG._g) return null;
  var layer = renderADG._g.select(".sim-highlight-group");
  if (layer.empty())
    layer = renderADG._g.append("g").attr("class", "sim-highlight-group");
  return layer;
}

function drawSimComp(compName, color) {
  var layer = ensureSimulationLayer();
  if (!layer || !adgCompBoxes[compName]) return;
  var bbox = adgCompBoxes[compName];
  layer.append("rect")
    .attr("x", bbox.x - 4).attr("y", bbox.y - 4)
    .attr("width", bbox.w + 8).attr("height", bbox.h + 8)
    .attr("rx", 10)
    .attr("stroke", color)
    .attr("fill", "rgba(0,0,0,0)")
    .attr("class", "sim-highlight-box");
}

function drawSimFU(peName, fuName, color) {
  var layer = ensureSimulationLayer();
  if (!layer) return;
  var key = peName + "/" + fuName;
  var bbox = adgFUBoxes[key];
  if (!bbox) {
    drawSimComp(peName, color);
    return;
  }
  layer.append("rect")
    .attr("x", bbox.x - 3).attr("y", bbox.y - 3)
    .attr("width", bbox.w + 6).attr("height", bbox.h + 6)
    .attr("rx", 6)
    .attr("stroke", color)
    .attr("fill", "rgba(0,0,0,0)")
    .attr("class", "sim-highlight-box");
}

function drawSimPort(vizKey, color) {
  if (!renderADG._portRegistry || !renderADG._portRegistry[vizKey]) return;
  var layer = ensureSimulationLayer();
  if (!layer) return;
  var p = renderADG._portRegistry[vizKey];
  layer.append("rect")
    .attr("x", p.x - 7).attr("y", p.y - 7)
    .attr("width", 14).attr("height", 14)
    .attr("rx", 3)
    .attr("stroke", color)
    .attr("fill", "rgba(0,0,0,0)")
    .attr("class", "sim-highlight-port");
}

function drawSimHWEdge(fromKey, toKey, color) {
  if (!renderADG._g) return;
  var layer = ensureSimulationLayer();
  if (!layer) return;
  renderADG._g.selectAll(".hw-edge-vis").each(function() {
    var el = d3.select(this);
    if (el.attr("data-from") !== fromKey || el.attr("data-to") !== toKey) return;
    layer.append("path")
      .attr("d", el.attr("d"))
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("class", "sim-highlight-edge");
  });
}

function drawSimPERoutesForEdge(swEdgeId, color) {
  if (!renderADG._g) return;
  var layer = ensureSimulationLayer();
  if (!layer) return;
  renderADG._g.selectAll(".pe-route-base[data-sw-edge='" + swEdgeId + "']")
    .each(function() {
      var el = d3.select(this);
      layer.append("path")
        .attr("d", el.attr("d"))
        .attr("fill", "none")
        .attr("stroke", color)
        .attr("class", "sim-highlight-edge");
    });
  renderADG._g.selectAll(".pe-route-arrow[data-sw-edge='" + swEdgeId + "']")
    .each(function() {
      var el = d3.select(this);
      layer.append("path")
        .attr("d", el.attr("d"))
        .attr("fill", color)
        .attr("stroke", "none")
        .attr("opacity", 1)
        .attr("class", "sim-highlight-arrow");
    });
}

function drawSimRoutingPrimitive(primitive, swEdgeId, color) {
  if (!primitive) return;
  if (primitive.kind === "module_edge") {
    if (primitive.fromKey) drawSimPort(primitive.fromKey, color);
    if (primitive.toKey) drawSimPort(primitive.toKey, color);
    if (primitive.fromKey && primitive.toKey)
      drawSimHWEdge(primitive.fromKey, primitive.toKey, color);
    return;
  }

  if (primitive.kind === "switch_route") {
    drawSimSwitchRoute(primitive.component, primitive.inputPort,
                       primitive.outputPort, color);
    return;
  }

  if (primitive.kind === "component_route") {
    if (primitive.componentPortKey)
      drawSimPort(primitive.componentPortKey, color);
    if (primitive.functionUnitPortKey)
      drawSimPort(primitive.functionUnitPortKey, color);
    drawSimPERoutesForEdge(swEdgeId, color);
    return;
  }

  if (primitive.kind === "memory_binding") {
    (primitive.visibleEdges || []).forEach(function(edge) {
      drawSimHWEdge(edge.fromKey, edge.toKey, color);
      if (edge.fromKey) drawSimPort(edge.fromKey, color);
      if (edge.toKey) drawSimPort(edge.toKey, color);
    });
  }
}

function drawSimRoutingPath(swEdgeId, color) {
  if (swEdgeId == null) return;
  var primitiveIdx = ensureRoutingPrimitiveIndex();
  if (!primitiveIdx || !primitiveIdx.byEdge) return;
  var primitives = primitiveIdx.byEdge[swEdgeId] || [];
  primitives.forEach(function(primitive) {
    drawSimRoutingPrimitive(primitive, swEdgeId, color);
  });
}

function drawSimSwitchRoute(componentName, inputIdx, outputIdx, color) {
  if (!renderADG._portRegistry) return;
  var inKey = componentName + "_in_" + inputIdx;
  var outKey = componentName + "_out_" + outputIdx;
  var pIn = renderADG._portRegistry[inKey];
  var pOut = renderADG._portRegistry[outKey];
  if (!pIn || !pOut) {
    drawSimComp(componentName, color);
    return;
  }
  var layer = ensureSimulationLayer();
  if (!layer) return;
  layer.append("line")
    .attr("x1", pIn.x).attr("y1", pIn.y)
    .attr("x2", pOut.x).attr("y2", pOut.y)
    .attr("stroke", color)
    .attr("class", "sim-highlight-edge");
}

function drawSimMemoryBinding(componentName, color) {
  if (!MAPPING_DATA || !MAPPING_DATA.memory_regions) return;
  (MAPPING_DATA.memory_regions || []).forEach(function(binding) {
    if (binding.hw_name !== componentName) return;
    (binding.visible_edges || binding.visibleEdges || []).forEach(function(edge) {
      drawSimHWEdge(edge.fromKey, edge.toKey, color);
      if (edge.fromKey) drawSimPort(edge.fromKey, color);
      if (edge.toKey) drawSimPort(edge.toKey, color);
    });
  });
}

function highlightSimDFGNode(nodeIdx) {
  var dfgEl = document.getElementById("svg-dfg");
  if (!dfgEl) return;
  var nodeG = dfgEl.querySelector("g.node[data-dfg-node='" + nodeIdx + "']");
  if (!nodeG) return;
  var shape = nodeG.querySelector(".dfg-node-shape") ||
    nodeG.querySelector("ellipse, polygon, path");
  if (shape)
    shape.classList.add("sim-highlight-dfg");
}

function renderSimulationFrame() {
  clearSimulationHighlights();
  if (!simulationEnabled || !simTraceIdx || simTraceIdx.frames.length === 0)
    return;

  var frame = simTraceIdx.frames[simulationFrameIndex];
  if (!frame) return;

  frame.events.forEach(function(ev) {
    var meta = simTraceIdx.moduleByHwNode[ev.hw_node_id] || null;
    var color = (ev.event_kind === "node_fire") ? "#ffd166" :
                (ev.event_kind === "route_use") ? "#4ecdc4" :
                (ev.event_kind === "device_error") ? "#ff6b6b" : "#a29bfe";

    if (meta) {
      if (meta.kind === "function_unit") {
        drawSimFU(meta.component_name, meta.function_unit_name || meta.name, color);
        var swNodes = (mappingIdx && mappingIdx.hwNodeToSwNodes)
          ? (mappingIdx.hwNodeToSwNodes[ev.hw_node_id] || [])
          : [];
        swNodes.forEach(function(swNodeId) { highlightSimDFGNode(swNodeId); });
        var swEdges = simTraceIdx.incidentSwEdgesByHwNode[ev.hw_node_id] || [];
        swEdges.forEach(function(swEdgeId) {
          drawSimRoutingPath(swEdgeId, color);
        });
      } else if (meta.kind === "boundary_input") {
        drawSimPort("module_in_" + meta.boundary_ordinal, color);
      } else if (meta.kind === "boundary_output") {
        drawSimPort("module_out_" + meta.boundary_ordinal, color);
      } else {
        drawSimComp(meta.component_name || meta.name, color);
        if (meta.kind === "spatial_sw" || meta.kind === "temporal_sw")
          drawSimSwitchRoute(meta.component_name || meta.name, ev.arg0, ev.lane, color);
        if (meta.kind === "memory" || meta.kind === "extmemory")
          drawSimMemoryBinding(meta.component_name || meta.name, color);
      }
    }
  });

  updateSimulationStatus();
}

function stopSimulationPlayback() {
  simulationAutoPlay = false;
  if (simulationTimer) {
    window.clearTimeout(simulationTimer);
    simulationTimer = null;
  }
  updateSimulationControls();
}

function scheduleSimulationPlayback() {
  if (!simulationEnabled || !simulationAutoPlay || !simTraceIdx) return;
  if (simulationTimer)
    window.clearTimeout(simulationTimer);
  simulationTimer = window.setTimeout(function() {
    simulationTimer = null;
    if (!simulationAutoPlay || !simTraceIdx) return;
    if (simulationFrameIndex + 1 < simTraceIdx.frames.length) {
      simulationFrameIndex += 1;
      renderSimulationFrame();
      scheduleSimulationPlayback();
    } else {
      stopSimulationPlayback();
    }
  }, simulationFrameDelayMs);
}

function stepSimulation(delta) {
  if (!simTraceIdx || simTraceIdx.frames.length === 0) return;
  stopSimulationPlayback();
  simulationFrameIndex = Math.max(0, Math.min(simTraceIdx.frames.length - 1,
                                             simulationFrameIndex + delta));
  renderSimulationFrame();
}

function resetSimulationPlayback() {
  stopSimulationPlayback();
  simulationFrameIndex = 0;
  renderSimulationFrame();
}

function setSimulationEnabled(on) {
  simulationEnabled = !!on && !!simTraceIdx;
  if (!simulationEnabled)
    stopSimulationPlayback();
  renderSimulationFrame();
  updateSimulationControls();
}

function updateSimulationStatus() {
  var label = document.getElementById("sim-cycle-label");
  if (!label) return;
  if (!simulationEnabled || !simTraceIdx || simTraceIdx.frames.length === 0) {
    label.textContent = "Cycle: -";
    return;
  }
  var frame = simTraceIdx.frames[simulationFrameIndex];
  label.textContent = "Cycle: " + frame.cycle + " | Event Frame: " +
    (simulationFrameIndex + 1) + "/" + simTraceIdx.frames.length;
}

function updateSimulationControls() {
  var btnSim = document.getElementById("btn-sim-toggle");
  var btnReset = document.getElementById("btn-sim-reset");
  var btnStop = document.getElementById("btn-sim-stop");
  var btnStep = document.getElementById("btn-sim-step");
  var btnBack = document.getElementById("btn-sim-back");
  var btnAuto = document.getElementById("btn-sim-auto");
  if (btnSim) {
    btnSim.textContent = simulationEnabled ? "Simulation: On" : "Simulation: Off";
    btnSim.classList.toggle("active", simulationEnabled);
  }
  var disabled = !simTraceIdx || !simulationEnabled;
  if (btnReset) btnReset.disabled = disabled;
  if (btnStop) btnStop.disabled = disabled || !simulationAutoPlay;
  if (btnStep) btnStep.disabled = disabled;
  if (btnBack) btnBack.disabled = disabled;
  if (btnAuto) {
    btnAuto.disabled = disabled;
    btnAuto.classList.toggle("active", simulationAutoPlay);
    btnAuto.textContent = simulationAutoPlay ? "Auto Play: On" : "Auto Play";
  }
  updateSimulationStatus();
}

function initSimulationPlayback(toolbar, statusBar) {
  simTraceIdx = buildSimulationIndex();
  if (!toolbar || !simTraceIdx)
    return;

  var sep = document.createElement("span");
  sep.className = "toolbar-sep";
  sep.textContent = "|";
  toolbar.insertBefore(sep, statusBar);

  var btnSim = document.createElement("button");
  btnSim.id = "btn-sim-toggle";
  btnSim.className = "mapping-toggle";
  btnSim.addEventListener("click", function() {
    setSimulationEnabled(!simulationEnabled);
  });
  toolbar.insertBefore(btnSim, statusBar);

  var btnStop = document.createElement("button");
  btnStop.id = "btn-sim-stop";
  btnStop.className = "mode-btn";
  btnStop.textContent = "Stop";
  btnStop.addEventListener("click", stopSimulationPlayback);
  toolbar.insertBefore(btnStop, statusBar);

  var btnReset = document.createElement("button");
  btnReset.id = "btn-sim-reset";
  btnReset.className = "mode-btn";
  btnReset.textContent = "Reset";
  btnReset.addEventListener("click", resetSimulationPlayback);
  toolbar.insertBefore(btnReset, statusBar);

  var btnBack = document.createElement("button");
  btnBack.id = "btn-sim-back";
  btnBack.className = "mode-btn";
  btnBack.textContent = "Back";
  btnBack.addEventListener("click", function() { stepSimulation(-1); });
  toolbar.insertBefore(btnBack, statusBar);

  var btnStep = document.createElement("button");
  btnStep.id = "btn-sim-step";
  btnStep.className = "mode-btn";
  btnStep.textContent = "Step";
  btnStep.addEventListener("click", function() { stepSimulation(1); });
  toolbar.insertBefore(btnStep, statusBar);

  var btnAuto = document.createElement("button");
  btnAuto.id = "btn-sim-auto";
  btnAuto.className = "mode-btn";
  btnAuto.addEventListener("click", function() {
    if (!simulationEnabled) return;
    simulationAutoPlay = !simulationAutoPlay;
    updateSimulationControls();
    if (simulationAutoPlay)
      scheduleSimulationPlayback();
  });
  toolbar.insertBefore(btnAuto, statusBar);

  var cycleLabel = document.createElement("span");
  cycleLabel.id = "sim-cycle-label";
  cycleLabel.className = "sim-cycle-label";
  toolbar.insertBefore(cycleLabel, statusBar);

  updateSimulationControls();
}

function refreshSimulationPlayback() {
  if (!simTraceIdx) return;
  renderSimulationFrame();
  updateSimulationControls();
}
