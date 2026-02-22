//===-- HTMLViewer.cpp - Self-contained HTML viewer generation -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Generates self-contained HTML files that embed DOT sources and an inline
// DOT-to-SVG renderer for client-side Graphviz rendering with interactive
// features (zoom, pan, hover, click detail panel), per docs/spec-viz-gui.md.
//
// The HTML file is fully self-contained with no external dependencies,
// allowing offline viewing without network access.
//
//===----------------------------------------------------------------------===//

#include "loom/Viz/DOTExporter.h"

#include <sstream>
#include <string>

namespace loom {
namespace viz {

namespace {

// Escape a string for embedding in a JavaScript string literal.
std::string jsStringEscape(const std::string &s) {
  std::string result;
  result.reserve(s.size() + 32);
  for (char c : s) {
    switch (c) {
    case '\\':
      result += "\\\\";
      break;
    case '"':
      result += "\\\"";
      break;
    case '\'':
      result += "\\'";
      break;
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
    case '<':
      // Prevent </script> injection.
      result += "\\x3c";
      break;
    default:
      result += c;
      break;
    }
  }
  return result;
}

// Generate inline CSS for the viewer.
std::string generateCSS() {
  return R"CSS(
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: Helvetica, Arial, sans-serif; background: #f5f5f5; }
    #toolbar {
      display: flex; align-items: center; gap: 8px;
      padding: 8px 16px; background: #333; color: #fff;
    }
    #toolbar button {
      padding: 4px 12px; border: 1px solid #666; border-radius: 4px;
      background: #555; color: #fff; cursor: pointer; font-size: 14px;
    }
    #toolbar button:hover { background: #777; }
    #toolbar h1 { font-size: 16px; margin-right: auto; }
    #graph-area {
      display: flex; flex: 1; overflow: hidden;
      height: calc(100vh - 48px - 200px);
    }
    #graph-left, #graph-right {
      flex: 1; overflow: auto; position: relative; background: #fff;
      border: 1px solid #ddd;
    }
    #graph-right { display: none; border-left: 2px solid #999; }
    #graph-left svg, #graph-right svg { width: 100%; height: auto; }
    #detail-panel {
      display: none; position: fixed; bottom: 0; left: 0; right: 0;
      max-height: 200px; overflow-y: auto;
      background: #fff; border-top: 2px solid #333; padding: 12px 16px;
      font-size: 13px; font-family: monospace;
    }
    #detail-panel.visible { display: block; }
    #detail-close {
      position: absolute; top: 4px; right: 8px;
      background: none; border: none; font-size: 18px; cursor: pointer;
    }
    .highlight polygon, .highlight ellipse, .highlight path,
    .highlight rect { stroke: #ff6600 !important; stroke-width: 3px !important; }
    .cross-highlight polygon, .cross-highlight ellipse,
    .cross-highlight path, .cross-highlight rect {
      stroke: #0066ff !important; stroke-width: 3px !important;
      stroke-dasharray: 5,3 !important;
    }
    .route-trace path {
      stroke: #ff3300 !important; stroke-width: 4px !important;
      stroke-dasharray: none !important;
    }
    .side-by-side #graph-right { display: block; }
    .side-by-side #graph-area #graph-left,
    .side-by-side #graph-area #graph-right { flex: 1; }
    .node { cursor: pointer; }
    .node:hover rect, .node:hover ellipse, .node:hover polygon {
      stroke: #ff6600; stroke-width: 2.5px;
    }
  )CSS";
}

// Generate the self-contained inline DOT-to-SVG renderer JavaScript.
// This replaces the viz.js CDN dependency with a minimal built-in renderer.
std::string inlineRendererJS() {
  return R"JS(
    var DotRenderer = (function() {
      function escXml(s) {
        return s.replace(/&/g,'&amp;').replace(/</g,'&lt;')
                .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
      }

      function parseDOT(dot) {
        var nodes = {}, edges = [], rankdir = 'TB', order = [];
        var lines = dot.split('\n');
        for (var i = 0; i < lines.length; i++) {
          var L = lines[i].trim();
          if (!L || L.match(/^\s*\}/) || L.match(/^digraph\b/) ||
              L.match(/^\s*\{/) || L.match(/^\s*node\s*\[/) ||
              L.match(/^\s*edge\s*\[/)) {
            var rm = L.match(/rankdir\s*=\s*(\w+)/);
            if (rm) rankdir = rm[1];
            continue;
          }
          var rm2 = L.match(/rankdir\s*=\s*(\w+)/);
          if (rm2) { rankdir = rm2[1]; continue; }
          var em = L.match(/^\s*(\S+)\s+->\s+(\S+)/);
          if (em) {
            var eattr = {};
            var eam = L.match(/\[([^\]]*)\]/);
            if (eam) {
              var cm = eam[1].match(/color="([^"]*)"/);
              if (cm) eattr.color = cm[1];
              var pm = eam[1].match(/penwidth=([\d.]+)/);
              if (pm) eattr.pw = parseFloat(pm[1]);
            }
            edges.push({s: em[1], d: em[2], attr: eattr});
            continue;
          }
          var nm = L.match(/^\s*(\S+)\s+\[(.+)\]/);
          if (nm) {
            var id = nm[1], a = nm[2];
            var lb = id, fc = '#ddd', sh = 'box', ni = '';
            var m;
            if (m = a.match(/label="([^"]*)"/)) lb = m[1];
            if (m = a.match(/fillcolor="([^"]*)"/)) fc = m[1];
            if (m = a.match(/shape=(\w+)/)) sh = m[1];
            if (m = a.match(/id="([^"]*)"/)) ni = m[1];
            nodes[id] = {lb: lb.replace(/\\n/g,'\n'), fc: fc,
                         sh: sh, ni: ni || id};
            order.push(id);
          }
        }
        return {nodes: nodes, edges: edges, rankdir: rankdir, order: order};
      }

      function layout(g) {
        var ids = g.order.length > 0 ? g.order : Object.keys(g.nodes);
        var indeg = {}, adj = {};
        ids.forEach(function(i) { indeg[i] = 0; adj[i] = []; });
        g.edges.forEach(function(e) {
          if (g.nodes[e.s] && g.nodes[e.d]) {
            indeg[e.d]++;
            adj[e.s].push(e.d);
          }
        });
        var layers = [], done = {}, q = [];
        ids.forEach(function(i) { if (indeg[i] === 0) q.push(i); });
        while (q.length) {
          layers.push(q.slice());
          var nq = [];
          q.forEach(function(i) {
            done[i] = true;
            adj[i].forEach(function(d) {
              indeg[d]--;
              if (indeg[d] === 0 && !done[d]) nq.push(d);
            });
          });
          q = nq;
        }
        ids.forEach(function(i) {
          if (!done[i]) {
            if (!layers.length) layers.push([]);
            layers[layers.length - 1].push(i);
          }
        });
        var isLR = (g.rankdir === 'LR');
        var nW = 140, nH = 52, gMaj = 170, gMin = 68;
        var pos = {};
        layers.forEach(function(layer, li) {
          var total = layer.length * (isLR ? nH : nW) +
                      (layer.length - 1) * gMin;
          var offset = -total / 2;
          layer.forEach(function(id, ni) {
            var maj = li * gMaj + 80;
            var mn = offset + ni * ((isLR ? nH : nW) + gMin) +
                     (isLR ? nH : nW) / 2 + 400;
            pos[id] = isLR ? {x: maj, y: mn} : {x: mn, y: maj};
          });
        });
        return pos;
      }

      function toSVG(g, pos) {
        var nW = 140, nH = 52;
        var minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
        Object.keys(pos).forEach(function(id) {
          var p = pos[id];
          minX = Math.min(minX, p.x - nW / 2);
          minY = Math.min(minY, p.y - nH / 2);
          maxX = Math.max(maxX, p.x + nW / 2);
          maxY = Math.max(maxY, p.y + nH / 2);
        });
        if (minX > maxX) { minX = 0; minY = 0; maxX = 200; maxY = 100; }
        var pad = 50;
        var vw = maxX - minX + pad * 2;
        var vh = maxY - minY + pad * 2;
        var ox = minX - pad, oy = minY - pad;
        var s = '<svg xmlns="http://www.w3.org/2000/svg" '
              + 'width="' + vw + '" height="' + vh + '" '
              + 'viewBox="' + ox + ' ' + oy + ' ' + vw + ' ' + vh + '">\n';
        s += '<defs><marker id="ah" markerWidth="10" markerHeight="7" '
           + 'refX="10" refY="3.5" orient="auto">'
           + '<polygon points="0 0,10 3.5,0 7" fill="#555"/>'
           + '</marker></defs>\n';

        g.edges.forEach(function(e) {
          if (!pos[e.s] || !pos[e.d]) return;
          var a = pos[e.s], b = pos[e.d];
          var ec = (e.attr && e.attr.color) ? e.attr.color : '#888';
          var pw = (e.attr && e.attr.pw) ? e.attr.pw : 1.5;
          s += '<line x1="' + a.x + '" y1="' + a.y
             + '" x2="' + b.x + '" y2="' + b.y
             + '" stroke="' + ec + '" stroke-width="' + pw
             + '" marker-end="url(#ah)"/>\n';
        });

        var nodeIds = g.order.length > 0 ? g.order : Object.keys(g.nodes);
        nodeIds.forEach(function(id) {
          if (!pos[id] || !g.nodes[id]) return;
          var n = g.nodes[id], p = pos[id];
          s += '<g class="node" id="' + escXml(n.ni) + '">\n';
          if (n.sh === 'diamond') {
            var pts = p.x + ',' + (p.y - nH/2) + ' '
                    + (p.x + nW/2) + ',' + p.y + ' '
                    + p.x + ',' + (p.y + nH/2) + ' '
                    + (p.x - nW/2) + ',' + p.y;
            s += '<polygon points="' + pts + '" fill="' + n.fc
               + '" stroke="#333" stroke-width="1.5"/>\n';
          } else if (n.sh === 'ellipse' || n.sh === 'point') {
            s += '<ellipse cx="' + p.x + '" cy="' + p.y
               + '" rx="' + (nW/2) + '" ry="' + (nH/2)
               + '" fill="' + n.fc
               + '" stroke="#333" stroke-width="1.5"/>\n';
          } else if (n.sh === 'cylinder') {
            s += '<rect x="' + (p.x - nW/2) + '" y="' + (p.y - nH/2)
               + '" width="' + nW + '" height="' + nH
               + '" rx="' + (nW/4) + '" fill="' + n.fc
               + '" stroke="#333" stroke-width="1.5"/>\n';
          } else {
            s += '<rect x="' + (p.x - nW/2) + '" y="' + (p.y - nH/2)
               + '" width="' + nW + '" height="' + nH
               + '" rx="4" fill="' + n.fc
               + '" stroke="#333" stroke-width="1.5"/>\n';
          }
          var lns = n.lb.split('\n'), fs = 11;
          lns.forEach(function(ln, li) {
            var ty = p.y + (li - (lns.length - 1) / 2) * (fs + 1) + 4;
            s += '<text x="' + p.x + '" y="' + ty
               + '" text-anchor="middle" font-size="' + fs
               + '" font-family="Helvetica,Arial,sans-serif">'
               + escXml(ln) + '</text>\n';
          });
          s += '</g>\n';
        });

        s += '</svg>';
        return s;
      }

      return {
        render: function(dot) {
          var g = parseDOT(dot);
          var p = layout(g);
          return toSVG(g, p);
        }
      };
    })();

    window.renderDot = function(d) { return DotRenderer.render(d); };
  )JS";
}

// Generate inline JavaScript for rendering and interaction.
std::string generateJS(const std::string &vizLevel) {
  std::ostringstream js;

  js << R"JS(
    var currentMode = 'primary';

    function renderGraph() {
      var leftEl = document.getElementById('graph-left');
      var rightEl = document.getElementById('graph-right');

      try {
        if (vizLevel === 'mapped' && currentMode === 'sidebyside') {
          leftEl.innerHTML = window.renderDot(dotSources.dfg);
          rightEl.innerHTML = window.renderDot(dotSources.adg);
          document.body.classList.add('side-by-side');
        } else if (vizLevel === 'mapped' && currentMode === 'overlay') {
          leftEl.innerHTML = window.renderDot(dotSources.overlay);
          document.body.classList.remove('side-by-side');
          rightEl.innerHTML = '';
        } else {
          leftEl.innerHTML = window.renderDot(dotSources.primary);
          document.body.classList.remove('side-by-side');
          rightEl.innerHTML = '';
        }
      } catch (e) {
        leftEl.innerHTML = '<pre style="padding:20px;color:red;">Render error: '
          + e.message + '</pre>';
      }

      attachInteraction();
    }

    function attachInteraction() {
      document.querySelectorAll('#graph-left .node, #graph-right .node')
        .forEach(function(node) {
          node.addEventListener('mouseenter', function() {
            node.classList.add('highlight');
            crossHighlight(node, true);
          });
          node.addEventListener('mouseleave', function() {
            node.classList.remove('highlight');
            crossHighlight(node, false);
          });
          node.addEventListener('click', function() {
            showDetail(node);
          });
        });
    }

    function crossHighlight(node, active) {
      if (vizLevel !== 'mapped' || currentMode !== 'sidebyside') return;

      var nodeId = node.id;
      if (!nodeId) return;

      document.querySelectorAll('.cross-highlight')
        .forEach(function(el) { el.classList.remove('cross-highlight'); });

      if (!active) return;

      if (nodeId.indexOf('sw_') === 0) {
        var hwId = mappingData.swToHw[nodeId];
        if (hwId) {
          var target = document.getElementById(hwId);
          if (target) {
            target.classList.add('cross-highlight');
            target.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        }
      } else if (nodeId.indexOf('hw_') === 0) {
        var swIds = mappingData.hwToSw[nodeId] || [];
        swIds.forEach(function(swId) {
          var target = document.getElementById(swId);
          if (target) {
            target.classList.add('cross-highlight');
          }
        });
        if (swIds.length > 0) {
          var first = document.getElementById(swIds[0]);
          if (first) first.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }
    }

    function showDetail(node) {
      var panel = document.getElementById('detail-panel');
      var content = document.getElementById('detail-content');
      var nodeId = node.id;
      if (!nodeId) return;

      var meta = nodeMetadata[nodeId] || {};
      var html = '<table>';
      for (var key in meta) {
        if (!meta.hasOwnProperty(key)) continue;
        var value = meta[key];
        if (typeof value === 'object') {
          html += '<tr><td><b>' + key + '</b></td><td><pre>'
            + JSON.stringify(value, null, 2) + '</pre></td></tr>';
        } else {
          html += '<tr><td><b>' + key + '</b></td><td>' + value + '</td></tr>';
        }
      }
      html += '</table>';
      if (Object.keys(meta).length === 0) {
        html = '<p>No metadata for ' + nodeId + '</p>';
      }
      content.innerHTML = html;
      panel.classList.add('visible');
    }

    // Toolbar handlers.
    document.getElementById('detail-close').addEventListener('click', function() {
      document.getElementById('detail-panel').classList.remove('visible');
    });

    document.getElementById('btn-zoom-in').addEventListener('click', function() {
      zoomGraph(1.2);
    });
    document.getElementById('btn-zoom-out').addEventListener('click', function() {
      zoomGraph(0.8);
    });
    document.getElementById('btn-fit').addEventListener('click', function() {
      fitGraph();
    });

    var zoomLevel = 1.0;
    function zoomGraph(factor) {
      zoomLevel *= factor;
      document.querySelectorAll('#graph-left svg, #graph-right svg')
        .forEach(function(svg) {
          svg.style.transform = 'scale(' + zoomLevel + ')';
          svg.style.transformOrigin = 'top left';
        });
    }
    function fitGraph() {
      zoomLevel = 1.0;
      document.querySelectorAll('#graph-left svg, #graph-right svg')
        .forEach(function(svg) {
          svg.style.transform = '';
        });
    }

    var modeToggle = document.getElementById('btn-mode-toggle');
    if (vizLevel === 'mapped') {
      modeToggle.style.display = 'inline-block';
      currentMode = 'overlay';
      modeToggle.textContent = 'Side-by-Side';
      modeToggle.addEventListener('click', function() {
        if (currentMode === 'overlay') {
          currentMode = 'sidebyside';
          modeToggle.textContent = 'Overlay';
        } else {
          currentMode = 'overlay';
          modeToggle.textContent = 'Side-by-Side';
        }
        renderGraph();
      });
    }

    // Background click to close detail.
    document.getElementById('graph-area').addEventListener('click', function(e) {
      if (e.target.closest('.node')) return;
      document.getElementById('detail-panel').classList.remove('visible');
    });

    renderGraph();
  )JS";

  return js.str();
}

} // namespace

std::string generateHTML(const std::string &dotString,
                         const std::string &title) {
  std::ostringstream html;

  html << "<!DOCTYPE html>\n<html>\n<head>\n";
  html << "  <meta charset=\"UTF-8\">\n";
  html << "  <title>" << title << "</title>\n";
  html << "  <style>" << generateCSS() << "</style>\n";
  html << "</head>\n<body>\n";

  // Toolbar.
  html << "<div id=\"toolbar\">\n";
  html << "  <h1>" << title << "</h1>\n";
  html << "  <button id=\"btn-zoom-in\">+</button>\n";
  html << "  <button id=\"btn-zoom-out\">-</button>\n";
  html << "  <button id=\"btn-fit\">Fit</button>\n";
  html << "  <button id=\"btn-mode-toggle\" style=\"display:none\">"
          "Overlay | Side-by-Side</button>\n";
  html << "</div>\n";

  // Graph area.
  html << "<div id=\"graph-area\">\n";
  html << "  <div id=\"graph-left\"></div>\n";
  html << "  <div id=\"graph-right\"></div>\n";
  html << "</div>\n";

  // Detail panel.
  html << "<div id=\"detail-panel\">\n";
  html << "  <button id=\"detail-close\">&times;</button>\n";
  html << "  <div id=\"detail-content\"></div>\n";
  html << "</div>\n";

  // Embedded data.
  html << "<script>\n";
  html << "  var vizLevel = \"dfg\";\n";
  html << "  var dotSources = { primary: \"" << jsStringEscape(dotString)
       << "\" };\n";
  html << "  var mappingData = { swToHw: {}, hwToSw: {}, routes: {} };\n";
  html << "  var nodeMetadata = {};\n";
  html << "</script>\n";

  // Self-contained inline DOT renderer (no external dependencies).
  html << "<script>\n";
  html << inlineRendererJS();
  html << "</script>\n";

  // Interaction and rendering.
  html << "<script>\n";
  html << generateJS("dfg");
  html << "</script>\n";

  html << "</body>\n</html>\n";
  return html.str();
}

std::string generateMappedHTML(const std::string &overlayDot,
                               const std::string &dfgDot,
                               const std::string &adgDot,
                               const std::string &mappingJson,
                               const std::string &metadataJson,
                               const std::string &title) {
  std::ostringstream html;

  html << "<!DOCTYPE html>\n<html>\n<head>\n";
  html << "  <meta charset=\"UTF-8\">\n";
  html << "  <title>" << title << "</title>\n";
  html << "  <style>" << generateCSS() << "</style>\n";
  html << "</head>\n<body>\n";

  // Toolbar.
  html << "<div id=\"toolbar\">\n";
  html << "  <h1>" << title << "</h1>\n";
  html << "  <button id=\"btn-zoom-in\">+</button>\n";
  html << "  <button id=\"btn-zoom-out\">-</button>\n";
  html << "  <button id=\"btn-fit\">Fit</button>\n";
  html << "  <button id=\"btn-mode-toggle\" style=\"display:none\">"
          "Overlay | Side-by-Side</button>\n";
  html << "</div>\n";

  // Graph area.
  html << "<div id=\"graph-area\">\n";
  html << "  <div id=\"graph-left\"></div>\n";
  html << "  <div id=\"graph-right\"></div>\n";
  html << "</div>\n";

  // Detail panel.
  html << "<div id=\"detail-panel\">\n";
  html << "  <button id=\"detail-close\">&times;</button>\n";
  html << "  <div id=\"detail-content\"></div>\n";
  html << "</div>\n";

  // Embedded data.
  html << "<script>\n";
  html << "  var vizLevel = \"mapped\";\n";
  html << "  var dotSources = {\n";
  html << "    primary: \"" << jsStringEscape(overlayDot) << "\",\n";
  html << "    overlay: \"" << jsStringEscape(overlayDot) << "\",\n";
  html << "    dfg: \"" << jsStringEscape(dfgDot) << "\",\n";
  html << "    adg: \"" << jsStringEscape(adgDot) << "\"\n";
  html << "  };\n";
  html << "  var mappingData = " << mappingJson << ";\n";
  html << "  var nodeMetadata = " << metadataJson << ";\n";
  html << "</script>\n";

  // Self-contained inline DOT renderer (no external dependencies).
  html << "<script>\n";
  html << inlineRendererJS();
  html << "</script>\n";

  // Interaction JS.
  html << "<script>\n";
  html << generateJS("mapped");
  html << "</script>\n";

  html << "</body>\n</html>\n";
  return html.str();
}

} // namespace viz
} // namespace loom
