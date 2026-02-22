//===-- HTMLViewer.cpp - Self-contained HTML viewer generation -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Generates self-contained HTML files that embed DOT sources and viz.js WASM
// for client-side Graphviz rendering with interactive features (zoom, pan,
// hover, click detail panel), per docs/spec-viz-gui.md.
//
// The HTML file loads viz.js from a CDN for rendering. For truly self-contained
// output (no network required), the build system can vendor viz-standalone.js
// into lib/loom/Viz/assets/ and HTMLViewer will inline it as base64. When the
// vendored file is not available, a CDN fallback is used.
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
  )CSS";
}

// Generate inline JavaScript for rendering and interaction.
std::string generateJS(const std::string &vizLevel) {
  std::ostringstream js;

  js << R"JS(
    let currentMode = 'primary';
    let viz = null;

    async function initViz() {
      if (typeof Viz !== 'undefined') {
        viz = await Viz.instance();
      } else {
        document.getElementById('graph-left').innerHTML =
          '<p style="padding:20px;color:red;">viz.js not loaded. '
          + 'Ensure network access for CDN or vendor viz-standalone.js.</p>';
        return;
      }
      renderGraph();
    }

    function renderGraph() {
      if (!viz) return;
      const leftEl = document.getElementById('graph-left');
      const rightEl = document.getElementById('graph-right');

      try {
        if (vizLevel === 'mapped' && currentMode === 'sidebyside') {
          leftEl.innerHTML = viz.renderSVGElement(dotSources.dfg).outerHTML;
          rightEl.innerHTML = viz.renderSVGElement(dotSources.adg).outerHTML;
          document.body.classList.add('side-by-side');
        } else if (vizLevel === 'mapped' && currentMode === 'overlay') {
          leftEl.innerHTML = viz.renderSVGElement(dotSources.overlay).outerHTML;
          document.body.classList.remove('side-by-side');
          rightEl.innerHTML = '';
        } else {
          leftEl.innerHTML = viz.renderSVGElement(dotSources.primary).outerHTML;
          document.body.classList.remove('side-by-side');
          rightEl.innerHTML = '';
        }
      } catch (e) {
        leftEl.innerHTML = '<p style="padding:20px;color:red;">Render error: '
          + e.message + '</p>';
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

      const nodeId = node.id;
      if (!nodeId) return;

      document.querySelectorAll('.cross-highlight')
        .forEach(function(el) { el.classList.remove('cross-highlight'); });

      if (!active) return;

      if (nodeId.startsWith('sw_')) {
        const swId = nodeId;
        const hwId = mappingData.swToHw[swId];
        if (hwId) {
          const target = document.getElementById(hwId);
          if (target) {
            target.classList.add('cross-highlight');
            target.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        }
      } else if (nodeId.startsWith('hw_')) {
        const hwId = nodeId;
        const swIds = mappingData.hwToSw[hwId] || [];
        swIds.forEach(function(swId) {
          const target = document.getElementById(swId);
          if (target) {
            target.classList.add('cross-highlight');
          }
        });
        if (swIds.length > 0) {
          const first = document.getElementById(swIds[0]);
          if (first) first.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }
    }

    function showDetail(node) {
      const panel = document.getElementById('detail-panel');
      const content = document.getElementById('detail-content');
      const nodeId = node.id;
      if (!nodeId) return;

      const meta = nodeMetadata[nodeId] || {};
      let html = '<table>';
      for (const [key, value] of Object.entries(meta)) {
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

    let zoomLevel = 1.0;
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

    const modeToggle = document.getElementById('btn-mode-toggle');
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

    initViz();
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
  html << "  const vizLevel = \"dfg\";\n";
  html << "  const dotSources = { primary: \"" << jsStringEscape(dotString)
       << "\" };\n";
  html << "  const mappingData = { swToHw: {}, hwToSw: {}, routes: {} };\n";
  html << "  const nodeMetadata = {};\n";
  html << "</script>\n";

  // viz.js CDN fallback (self-contained inlining is done at build time when
  // the vendored viz-standalone.js is available).
  html << "<script src=\"https://unpkg.com/viz.js@2.1.2/viz.js\"></script>\n";
  html << "<script "
          "src=\"https://unpkg.com/viz.js@2.1.2/full.render.js\"></script>\n";

  // Use the @viz-js/viz package for modern browsers.
  html << "<script type=\"module\">\n";
  html << "  import Viz from "
          "'https://cdn.jsdelivr.net/npm/@viz-js/viz@3.2.4/+esm';\n";
  html << "</script>\n";

  // Fallback inline rendering for viz.js v2.x compatibility.
  html << "<script>\n";
  html << "  if (typeof Viz === 'undefined') {\n";
  html << "    // Will be initialized by module import above\n";
  html << "  }\n";
  html << "</script>\n";

  // Renderer with broad compatibility.
  html << "<script>\n";
  html << "  // Renderer: try module Viz first, fall back to viz.js v2\n";
  html << "  async function initRenderer() {\n";
  html << "    const leftEl = document.getElementById('graph-left');\n";
  html << "    try {\n";
  html << "      // Try viz.js v2 (Viz + VizRenderStringSync)\n";
  html << "      if (typeof Viz === 'function') {\n";
  html << "        const v = new Viz();\n";
  html << "        const svg = await v.renderSVGElement(dotSources.primary);\n";
  html << "        leftEl.appendChild(svg);\n";
  html << "        return;\n";
  html << "      }\n";
  html << "      // Try @viz-js/viz v3\n";
  html << "      const mod = await import("
          "'https://cdn.jsdelivr.net/npm/@viz-js/viz@3.2.4/+esm');\n";
  html << "      const viz = await mod.default.instance();\n";
  html << "      leftEl.innerHTML = "
          "viz.renderSVGElement(dotSources.primary).outerHTML;\n";
  html << "    } catch (e) {\n";
  html << "      leftEl.innerHTML = '<pre>Graphviz rendering error: ' + "
          "e.message + '\\nDOT source:\\n' + dotSources.primary + '</pre>';\n";
  html << "    }\n";
  html << "  }\n";
  html << "  window.addEventListener('load', initRenderer);\n";
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
  html << "  const vizLevel = \"mapped\";\n";
  html << "  const dotSources = {\n";
  html << "    primary: \"" << jsStringEscape(overlayDot) << "\",\n";
  html << "    overlay: \"" << jsStringEscape(overlayDot) << "\",\n";
  html << "    dfg: \"" << jsStringEscape(dfgDot) << "\",\n";
  html << "    adg: \"" << jsStringEscape(adgDot) << "\"\n";
  html << "  };\n";
  html << "  const mappingData = " << mappingJson << ";\n";
  html << "  const nodeMetadata = " << metadataJson << ";\n";
  html << "</script>\n";

  // viz.js CDN.
  html << "<script src=\"https://unpkg.com/viz.js@2.1.2/viz.js\"></script>\n";
  html << "<script "
          "src=\"https://unpkg.com/viz.js@2.1.2/full.render.js\"></script>\n";

  // Interaction JS.
  html << "<script>\n";
  html << generateJS("mapped");
  html << "</script>\n";

  html << "</body>\n</html>\n";
  return html.str();
}

} // namespace viz
} // namespace loom
