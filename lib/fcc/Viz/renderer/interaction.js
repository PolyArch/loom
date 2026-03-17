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
    } else if (isSwitchPortKind(p.kind) || p.kind === "fifo" || p.kind === "memory" ||
               p.kind === "add_tag" || p.kind === "del_tag" || p.kind === "map_tag") {
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
        if (nodeInfo.type && nodeInfo.type.indexOf("memref") === 0) {
          var hwNames = mappingIdx && mappingIdx.memArgToHw[nodeInfo.arg_index]
            ? mappingIdx.memArgToHw[nodeInfo.arg_index]
            : [];
          var swNodes = mappingIdx && mappingIdx.memArgToSw[nodeInfo.arg_index]
            ? mappingIdx.memArgToSw[nodeInfo.arg_index]
            : [];
          hwNames.forEach(function(hwName) {
            highlightADGComp(hwName);
          });
          swNodes.forEach(function(swNode) {
            highlightDFGNode(swNode);
          });
          if (hwNames.length > 0)
            focusADGOn(hwNames[0]);
          if (swNodes.length > 0)
            focusDFGOn([nodeIdx].concat(swNodes));
        }
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
    var compName = m.pe_name || m.hw_name;
    if (compName) {
      highlightADGComp(compName);
      focusADGOn(compName);
    }
    if (m.pe_name)
      highlightADGFU(m.pe_name, m.hw_name);

    var memArgs = mappingIdx && mappingIdx.swToMemArgs[nodeIdx]
      ? mappingIdx.swToMemArgs[nodeIdx]
      : [];
    var focusNodes = [nodeIdx];
    memArgs.forEach(function(argIdx) {
      var argNodeId = getDFGInputNodeByArgIndex(argIdx);
      if (argNodeId != null) {
        highlightDFGNode(argNodeId);
        focusNodes.push(argNodeId);
      }
    });
    if (memArgs.length > 0) {
      highlightMemrefEdgesForSwNode(nodeIdx);
      focusDFGOn(focusNodes);
    }
  }
}

function onDFGEdgeDataClick(edgeData) {
  if (!mappingEnabled) return;
  clearCrossHighlight();
  highlightDFGNode(edgeData.from);
  highlightDFGNode(edgeData.to);
  if (edgeData.edge_type === "memref")
    highlightMemoryInterfaceForSwNode(edgeData.to);
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
    var focusNodes = [];
    mappingIdx.hwToSw[compName].forEach(function(m) {
      highlightDFGNode(m.sw_node);
      focusNodes.push(m.sw_node);
      highlightMemrefEdgesForSwNode(m.sw_node);
      var memArgs = mappingIdx.swToMemArgs[m.sw_node] || [];
      memArgs.forEach(function(argIdx) {
        var argNodeId = getDFGInputNodeByArgIndex(argIdx);
        if (argNodeId != null) {
          highlightDFGNode(argNodeId);
          focusNodes.push(argNodeId);
        }
      });
    });
    focusDFGOn(focusNodes);
  }
}

function getDFGInputNodeByArgIndex(argIdx) {
  if (!DFG_DATA || !DFG_DATA.nodes) return null;
  for (var i = 0; i < DFG_DATA.nodes.length; ++i) {
    var node = DFG_DATA.nodes[i];
    if (node.kind === "input" && node.arg_index === argIdx)
      return node.id;
  }
  return null;
}

function highlightMemoryInterfaceForSwNode(swNode) {
  if (!mappingIdx) return;
  var mapping = mappingIdx.swToHw[swNode];
  if (!mapping) return;
  var compName = mapping.pe_name || mapping.hw_name;
  if (!compName) return;
  highlightADGComp(compName);
  highlightMemoryInterfaceBinding(compName);
}

function highlightMemrefEdgesForSwNode(swNode) {
  if (!DFG_DATA || !DFG_DATA.edges) return;
  DFG_DATA.edges.forEach(function(edge) {
    if (edge.to === swNode && edge.edge_type === "memref")
      highlightRoutingPath(edge.id);
  });
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

function highlightMemoryInterfaceBinding(compName, color) {
  if (!renderADG._g || !compName) return false;
  var c = color || "#4ecdc4";
  var memPortKey = compName + "_in_0";
  var found = false;

  renderADG._g.selectAll(".hw-edge-vis").each(function() {
    var el = d3.select(this);
    var fromKey = el.attr("data-from");
    var toKey = el.attr("data-to");
    if (toKey !== memPortKey) return;

    el.attr("stroke", c)
      .attr("stroke-width", 3)
      .attr("opacity", 1);
    if (fromKey) highlightModulePort(fromKey, c);
    highlightModulePort(toKey, c);
    found = true;
  });

  return found;
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

  if (path.length === 2 && path[0] === path[1]) {
    var directPort = portLookup[path[0]];
    if (directPort) {
      if (directPort.component)
        highlightADGComp(directPort.component);
      if (directPort.kind === "memory" && directPort.index === 0 &&
          directPort.dir === "in" && directPort.component) {
        highlightMemoryInterfaceBinding(directPort.component, color);
      } else if (directPort.vizKey) {
        highlightModulePort(directPort.vizKey, color);
      }
    }
    return;
  }

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
