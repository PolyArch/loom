function renderMiniFUGraph(parentG, item) {
  var fu = item.fu;
  if (!fu || !fu.ops || fu.ops.length === 0) return;
  var fuConfig = item.fuConfig || null;
  var muxFieldByOpIndex = {};
  if (fuConfig && fuConfig.fields) {
    fuConfig.fields.forEach(function(field) {
      if (field && field.op_index !== undefined)
        muxFieldByOpIndex[field.op_index] = field;
    });
  }

  var layoutSpec = item.layoutSpec || measureFunctionUnitLayout(fu);
  var analysis = layoutSpec.analysis;
  var opCount = analysis.opCount;
  function normalizeInputEdges(rawEdges) {
    var nextOperand = [];
    for (var ni = 0; ni < opCount; ni++) nextOperand[ni] = 0;
    return (rawEdges || []).map(function(edge) {
      var srcArg = edge.srcArg !== undefined ? edge.srcArg : edge[0];
      var dstOp = edge.dstOp !== undefined ? edge.dstOp : edge[1];
      var dstOperand = edge.dstOperand !== undefined
        ? edge.dstOperand
        : (edge.length > 2 ? edge[2] : nextOperand[dstOp]++);
      return { srcArg: srcArg, dstOp: dstOp, dstOperand: dstOperand };
    });
  }
  function normalizeDagEdges(rawEdges) {
    var nextOperand = [];
    for (var ni = 0; ni < opCount; ni++) nextOperand[ni] = 0;
    return (rawEdges || []).map(function(edge) {
      var srcOp = edge.srcOp !== undefined ? edge.srcOp : edge[0];
      var dstOp = edge.dstOp !== undefined ? edge.dstOp : edge[1];
      var srcResult = edge.srcResult !== undefined
        ? edge.srcResult
        : (edge.length > 2 ? edge[2] : 0);
      var dstOperand = edge.dstOperand !== undefined
        ? edge.dstOperand
        : (edge.length > 3 ? edge[3] : nextOperand[dstOp]++);
      return {
        srcOp: srcOp,
        dstOp: dstOp,
        srcResult: srcResult,
        dstOperand: dstOperand
      };
    });
  }
  function normalizeOutputEdges(rawEdges) {
    var nextResult = {};
    return (rawEdges || []).map(function(edge) {
      var srcVal = edge.srcOwner !== undefined ? edge.srcOwner : edge[0];
      var dstPort = edge.yieldIdx !== undefined ? edge.yieldIdx : edge[1];
      var srcResult;
      if (edge.srcResult !== undefined) srcResult = edge.srcResult;
      else if (edge.length > 2) srcResult = edge[2];
      else {
        var key = String(srcVal);
        srcResult = nextResult[key] || 0;
        nextResult[key] = srcResult + 1;
      }
      return { srcVal: srcVal, dstPort: dstPort, srcResult: srcResult };
    });
  }

  var inputEdges = normalizeInputEdges(analysis.inputEdges);
  var dagEdges = normalizeDagEdges(analysis.dagEdges);
  var outputEdges = normalizeOutputEdges(analysis.outputEdges);
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
    if (!edge || edge.dstOp < 0 || edge.dstOp >= opCount) return;
    incomingRefs[edge.dstOp].push({
      kind: "fu_in",
      index: edge.srcArg,
      operand: edge.dstOperand
    });
  });
  dagEdges.forEach(function(edge) {
    if (!edge || edge.srcOp < 0 || edge.srcOp >= opCount ||
        edge.dstOp < 0 || edge.dstOp >= opCount) return;
    outgoingRefs[edge.srcOp].push({
      kind: "op",
      index: edge.dstOp,
      result: edge.srcResult,
      operand: edge.dstOperand
    });
    incomingRefs[edge.dstOp].push({
      kind: "op",
      index: edge.srcOp,
      result: edge.srcResult,
      operand: edge.dstOperand
    });
  });
  outputEdges.forEach(function(edge) {
    if (!edge || edge.srcVal < 0 || edge.srcVal >= opCount) return;
    outgoingRefs[edge.srcVal].push({
      kind: "fu_out",
      index: edge.dstPort,
      result: edge.srcResult
    });
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

  function isMuxOp(opIdx) {
    var opName = fu.ops && fu.ops[opIdx] ? fu.ops[opIdx] : "";
    return opName.indexOf("mux") >= 0;
  }
  function staticMuxMode(opIdx) {
    if (!isMuxOp(opIdx)) return null;
    var inCount = incomingRefs[opIdx] ? incomingRefs[opIdx].length : 0;
    var outCount = outgoingRefs[opIdx] ? outgoingRefs[opIdx].length : 0;
    if (inCount > 1 && outCount <= 1) return "fanin";
    if (inCount <= 1 && outCount > 1) return "fanout";
    if (inCount > outCount) return "fanin";
    if (outCount > inCount) return "fanout";
    return null;
  }
  function styleMuxEdge(edge) {
    function applyDisconnect(markerAt) {
      edge.dash = "3,3";
      edge.markerType = "x";
      edge.markerAt = markerAt;
    }

    if (edge.dstDesc.kind === "op_in" && isMuxOp(edge.dstDesc.index)) {
      var sinkField = muxFieldByOpIndex[edge.dstDesc.index] || null;
      if (sinkField && sinkField.disconnect) {
        applyDisconnect("end");
        return;
      }
      if (staticMuxMode(edge.dstDesc.index) === "fanin" && sinkField) {
        if (edge.dstOperand === sinkField.sel) {
          edge.dash = null;
          edge.markerType = sinkField.discard ? "circle" : "arrow";
          edge.markerAt = "end";
        } else {
          edge.dash = "3,3";
        }
      }
    }

    if (edge.srcDesc.kind === "op_out" && isMuxOp(edge.srcDesc.index)) {
      var srcField = muxFieldByOpIndex[edge.srcDesc.index] || null;
      if (srcField && srcField.disconnect) {
        applyDisconnect("start");
        return;
      }
      if (staticMuxMode(edge.srcDesc.index) === "fanout" && srcField) {
        if ((edge.srcResult || 0) === srcField.sel) {
          edge.dash = null;
          edge.markerType = srcField.discard ? "circle" : "arrow";
          edge.markerAt = "start";
        } else {
          edge.dash = "3,3";
        }
      }
    }
  }

  var rawEdges = [];
  inputEdges.forEach(function(edge) {
    if (!edge || !nodePos[edge.dstOp]) return;
    var inputEdge = {
      kind: "input",
      srcDesc: { kind: "fu_in", index: edge.srcArg },
      dstDesc: { kind: "op_in", index: edge.dstOp },
      color: "#4ecdc4",
      dash: null,
      dstOperand: edge.dstOperand,
      srcResult: 0,
      markerType: null,
      markerAt: null
    };
    styleMuxEdge(inputEdge);
    rawEdges.push(inputEdge);
  });
  dagEdges.forEach(function(edge) {
    if (!edge || !nodePos[edge.srcOp] || !nodePos[edge.dstOp]) return;
    var dagEdge = {
      kind: "dag",
      srcDesc: { kind: "op_out", index: edge.srcOp },
      dstDesc: { kind: "op_in", index: edge.dstOp },
      color: "#6a8faf",
      dash: null,
      srcResult: edge.srcResult,
      dstOperand: edge.dstOperand,
      markerType: null,
      markerAt: null
    };
    styleMuxEdge(dagEdge);
    rawEdges.push(dagEdge);
  });
  outputEdges.forEach(function(edge) {
    if (!edge) return;
    var outputEdge = {
      kind: "output",
      srcDesc: edge.srcVal >= 0
        ? { kind: "op_out", index: edge.srcVal }
        : { kind: "fu_in", index: -(edge.srcVal + 1) },
      dstDesc: { kind: "fu_out", index: edge.dstPort },
      color: edge.srcVal >= 0 ? "#ff6b35" : "#888888",
      dash: edge.srcVal >= 0 ? null : "2,2",
      srcResult: edge.srcResult,
      dstOperand: 0,
      markerType: null,
      markerAt: null
    };
    styleMuxEdge(outputEdge);
    rawEdges.push(outputEdge);
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
  function edgeMarkerPose(points, at) {
    if (!points || points.length < 2) return null;
    var a, b, pos;
    if (at === "start") {
      a = points[0];
      b = points[1];
    } else {
      a = points[points.length - 2];
      b = points[points.length - 1];
    }
    var dx = b.x - a.x;
    var dy = b.y - a.y;
    var len = Math.sqrt(dx * dx + dy * dy);
    if (len < 0.5) return null;
    var ux = dx / len;
    var uy = dy / len;
    if (at === "start")
      pos = { x: a.x + ux * 5, y: a.y + uy * 5, ux: ux, uy: uy };
    else
      pos = { x: b.x - ux * 5, y: b.y - uy * 5, ux: ux, uy: uy };
    return pos;
  }
  function drawEdgeMarker(edge) {
    if (!edge.markerType || !edge.markerAt) return;
    var pose = edgeMarkerPose(edge.pts, edge.markerAt);
    if (!pose) return;
    var px = -pose.uy;
    var py = pose.ux;
    if (edge.markerType === "arrow") {
      var tipX = pose.x + pose.ux * 4;
      var tipY = pose.y + pose.uy * 4;
      var baseX = pose.x - pose.ux * 2;
      var baseY = pose.y - pose.uy * 2;
      g.append("polygon")
        .attr("points", [
          tipX + "," + tipY,
          (baseX + px * 2.5) + "," + (baseY + py * 2.5),
          (baseX - px * 2.5) + "," + (baseY - py * 2.5)
        ].join(" "))
        .attr("fill", edge.color)
        .attr("stroke", edge.color)
        .attr("stroke-width", 0.6)
        .attr("opacity", 0.95);
      return;
    }
    if (edge.markerType === "circle") {
      g.append("circle")
        .attr("cx", pose.x)
        .attr("cy", pose.y)
        .attr("r", 2.6)
        .attr("fill", "none")
        .attr("stroke", edge.color)
        .attr("stroke-width", 1.1)
        .attr("opacity", 0.95);
      return;
    }
    if (edge.markerType === "x") {
      var r = 2.8;
      g.append("line")
        .attr("x1", pose.x - r).attr("y1", pose.y - r)
        .attr("x2", pose.x + r).attr("y2", pose.y + r)
        .attr("stroke", edge.color)
        .attr("stroke-width", 1.1)
        .attr("stroke-linecap", "round")
        .attr("opacity", 0.95);
      g.append("line")
        .attr("x1", pose.x - r).attr("y1", pose.y + r)
        .attr("x2", pose.x + r).attr("y2", pose.y - r)
        .attr("stroke", edge.color)
        .attr("stroke-width", 1.1)
        .attr("stroke-linecap", "round")
        .attr("opacity", 0.95);
    }
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
      markerType: edge.markerType,
      markerAt: edge.markerAt,
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
      markerType: edge.markerType,
      markerAt: edge.markerAt,
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
      .attr("stroke-width", edge.markerType && edge.markerType !== "x" ? 1.15 : 1.05)
      .attr("opacity", 0.84)
      .attr("stroke-linecap", "round")
      .attr("stroke-linejoin", "round")
      .attr("stroke-dasharray", edge.dash);
    drawEdgeMarker(edge);
  });

  for (i = 0; i < opCount; i++) {
    var style = miniFUOpStyle(fu.ops[i]);
    var muxField = muxFieldByOpIndex[i] || null;
    var muxMode = staticMuxMode(i);
    var group = g.append("g");
    if (style.shape === "mux") {
      var topInset = muxMode === "fanout" ? 7 : 0;
      var bottomInset = muxMode === "fanin" ? 7 : 0;
      group.append("polygon")
        .attr("points", [
          (nodePos[i].x - opW / 2 + topInset) + "," + (nodePos[i].y - opH / 2),
          (nodePos[i].x + opW / 2 - topInset) + "," + (nodePos[i].y - opH / 2),
          (nodePos[i].x + opW / 2 - bottomInset) + "," + (nodePos[i].y + opH / 2),
          (nodePos[i].x - opW / 2 + bottomInset) + "," + (nodePos[i].y + opH / 2)
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
    if (muxField) {
      group.append("text")
        .attr("x", nodePos[i].x)
        .attr("y", nodePos[i].y + opH / 2 + 7)
        .attr("text-anchor", "middle")
        .attr("fill", "#ffd166")
        .attr("font-size", "5px")
        .text("sel=" + muxField.sel);
    }
  }
}
