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

