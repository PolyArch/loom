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
  fuConfigIdx = buildFUConfigIndex();
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
