#include "FabricVisualizationInternal.h"

#include "Fabric/Visualization/FabricVisualization.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <string>

namespace loom::fabric::visualization {
namespace {

std::string escapeHtml(llvm::StringRef text) {
  std::string escaped;
  escaped.reserve(text.size());
  for (char character : text) {
    switch (character) {
    case '&':
      escaped += "&amp;";
      break;
    case '<':
      escaped += "&lt;";
      break;
    case '>':
      escaped += "&gt;";
      break;
    case '"':
      escaped += "&quot;";
      break;
    case '\'':
      escaped += "&#39;";
      break;
    default:
      escaped += character;
      break;
    }
  }
  return escaped;
}

std::string edgePath(const Edge &edge) {
  std::string path;
  llvm::raw_string_ostream stream(path);
  for (std::size_t index = 0; index < edge.route.size(); ++index) {
    stream << (index == 0 ? "M " : " L ")
           << llvm::format("%.2f", edge.route[index].x) << ' '
           << llvm::format("%.2f", edge.route[index].y);
  }
  return path;
}

llvm::StringRef edgeClass(llvm::StringRef kind) {
  if (kind == "domain")
    return "edge-domain";
  if (kind == "attachment")
    return "edge-attachment";
  if (kind == "fu-route")
    return "edge-fu";
  return "edge-transport";
}

void writeIcon(llvm::raw_ostream &output, llvm::StringRef name) {
  output << "<svg aria-hidden=\"true\" viewBox=\"0 0 24 24\" "
            "fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" "
            "stroke-linecap=\"round\" stroke-linejoin=\"round\">";
  if (name == "zoom-in")
    output << "<circle cx=\"11\" cy=\"11\" r=\"8\"/><path d=\"m21 "
              "21-4.3-4.3M11 8v6M8 11h6\"/>";
  else if (name == "zoom-out")
    output << "<circle cx=\"11\" cy=\"11\" r=\"8\"/><path d=\"m21 "
              "21-4.3-4.3M8 11h6\"/>";
  else if (name == "fit")
    output << "<path d=\"M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 "
              "2v3M8 21H5a2 2 0 0 1-2-2v-3M16 21h3a2 2 0 0 0 2-2v-3\"/>";
  else if (name == "reset")
    output << "<path d=\"M3 12a9 9 0 1 0 3-6.7L3 8\"/><path d=\"M3 "
              "3v5h5\"/>";
  else if (name == "search")
    output << "<circle cx=\"11\" cy=\"11\" r=\"8\"/><path d=\"m21 "
              "21-4.3-4.3\"/>";
  output << "</svg>";
}

void writeGraph(const Graph &graph, bool active, llvm::raw_ostream &output) {
  output << "<svg class=\"graph-view" << (active ? " is-active" : "")
         << "\" data-view-id=\"" << escapeHtml(graph.id)
         << "\" data-view-kind=\"" << escapeHtml(graph.kind)
         << "\" data-layout-engine=\"loom-layered-v1\" viewBox=\"0 0 "
         << llvm::format("%.2f", graph.width) << ' '
         << llvm::format("%.2f", graph.height)
         << "\" role=\"img\" aria-label=\"" << escapeHtml(graph.title) << "\">";
  output
      << "<defs><marker id=\"arrow-" << escapeHtml(graph.id)
      << "\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" "
         "markerWidth=\"7\" markerHeight=\"7\" orient=\"auto-start-reverse\">"
         "<path d=\"M 0 0 L 10 5 L 0 10 z\"/></marker></defs>";
  output << "<g class=\"edges\">";
  for (const Edge &edge : graph.edges) {
    if (edge.route.empty())
      continue;
    output << "<path class=\"edge " << edgeClass(edge.kind) << "\" d=\""
           << edgePath(edge) << "\" marker-end=\"url(#arrow-"
           << escapeHtml(graph.id) << ")\"/>";
  }
  output << "</g><g class=\"nodes\">";
  for (const Node &node : graph.nodes) {
    const std::string kind = escapeHtml(node.kind);
    output << "<g class=\"node\" tabindex=\"0\" data-node-id=\""
           << escapeHtml(node.id) << "\" data-label=\""
           << escapeHtml(node.label) << "\" data-detail=\""
           << escapeHtml(node.detail) << "\" data-entity-kind=\"" << kind
           << "\" data-x=\"" << llvm::format("%.2f", node.x) << "\" data-y=\""
           << llvm::format("%.2f", node.y) << "\" transform=\"translate("
           << llvm::format("%.2f", node.x) << ' '
           << llvm::format("%.2f", node.y) << ")\">";
    output << "<rect width=\"" << llvm::format("%.2f", node.width)
           << "\" height=\"" << llvm::format("%.2f", node.height)
           << "\" rx=\"6\"/><rect class=\"node-accent\" width=\"5\" "
              "height=\""
           << llvm::format("%.2f", node.height) << "\" rx=\"2\"/>";
    output << "<text class=\"node-title\" x=\"18\" y=\"29\">"
           << escapeHtml(node.label)
           << "</text><text class=\"node-detail\" "
              "x=\"18\" y=\"51\">"
           << escapeHtml(node.detail) << "</text></g>";
  }
  output << "</g></svg>";
}

} // namespace

llvm::Error writeHtml(const Document &document, llvm::raw_ostream &output) {
  output
      << "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
         "<meta name=\"viewport\" "
         "content=\"width=device-width,initial-scale=1\">"
         "<title>"
      << escapeHtml(document.title)
      << "</title><style>"
         ":root{color-scheme:light;--ink:#17202a;--muted:#66717d;"
         "--line:#c9d1d9;--panel:#f7f9fa;--paper:#fff;--teal:#087f8c;"
         "--green:#2f855a;--amber:#b7791f;--red:#c2413b;--violet:#7257a8;"
         "--shadow:0 8px 24px rgba(23,32,42,.09)}"
         "*{box-sizing:border-box}html,body{height:100%;margin:0}"
         "body{font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe "
         "UI,sans-serif;"
         "color:var(--ink);background:var(--paper);overflow:hidden}"
         ".app{display:grid;grid-template-columns:260px minmax(0,1fr) 300px;"
         "grid-template-rows:58px minmax(0,1fr);height:100%}"
         ".topbar{grid-column:1/-1;display:flex;align-items:center;gap:16px;"
         "padding:0 18px;border-bottom:1px solid "
         "#dce2e7;background:#fff;z-index:4}"
         ".brand{font-weight:720;font-size:17px;white-space:nowrap}.root-id{"
         "font:12px "
         "ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--muted);"
         "overflow:hidden;"
         "text-overflow:ellipsis;white-space:nowrap;max-width:42vw}"
         ".tools{margin-left:auto;display:flex;gap:6px}.icon-button{width:36px;"
         "height:36px;"
         "display:grid;place-items:center;border:1px solid "
         "#d7dde2;background:#fff;border-radius:6px;"
         "color:#34404b;cursor:pointer}.icon-button:hover{background:#eef5f5;"
         "border-color:#9fc6ca}"
         ".icon-button svg{width:18px;height:18px}"
         ".sidebar,.inspector{min-width:0;background:var(--panel);overflow:"
         "auto}"
         ".sidebar{border-right:1px solid #dce2e7;padding:16px 12px}"
         ".inspector{border-left:1px solid #dce2e7;padding:20px}"
         ".section-label{font-size:11px;font-weight:760;text-transform:"
         "uppercase;color:#77828d;"
         "margin:2px 8px "
         "9px}.search{position:relative;margin-bottom:16px}.search "
         "svg{position:absolute;"
         "left:10px;top:10px;width:16px;height:16px;color:#74808b}.search "
         "input{width:100%;height:36px;"
         "padding:0 10px 0 34px;border:1px solid "
         "#ced6dc;border-radius:6px;background:#fff;color:var(--ink)}"
         ".view-list{display:flex;flex-direction:column;gap:4px}.view-button{"
         "text-align:left;border:0;"
         "border-radius:6px;padding:10px "
         "11px;background:transparent;color:#34404b;cursor:pointer}"
         ".view-button:hover{background:#e8eeef}.view-button.is-active{"
         "background:#dceff0;color:#075b64;"
         "font-weight:680}.view-button "
         "small{display:block;color:#77828d;margin-top:2px;font-weight:450}"
         ".canvas{position:relative;overflow:hidden;background-color:#fbfcfc;"
         "background-image:linear-gradient(#edf0f2 1px,transparent 1px),"
         "linear-gradient(90deg,#edf0f2 1px,transparent "
         "1px);background-size:24px 24px}"
         ".graph-view{display:none;width:100%;height:100%;touch-action:none;"
         "user-select:none}"
         ".graph-view.is-active{display:block}.edge{fill:none;stroke:#71808c;"
         "stroke-width:1.8;opacity:.72}.edge-domain{stroke:#9b7a31;stroke-"
         "dasharray:5 5}"
         ".edge-attachment{stroke:var(--violet);stroke-dasharray:8 "
         "4}.edge-fu{stroke:var(--teal)}"
         ".edge-transport{stroke:#526b78}.edge+*{pointer-events:none}marker "
         "path{fill:#526b78}"
         ".node{cursor:pointer;outline:none}.node "
         "rect:first-child{fill:#fff;stroke:#aeb9c1;stroke-width:1.3;"
         "filter:drop-shadow(0 3px 5px rgba(23,32,42,.08))}.node:hover "
         "rect:first-child,.node:focus rect:first-child,"
         ".node.is-selected "
         "rect:first-child{stroke:var(--teal);stroke-width:2.5}.node.is-match "
         "rect:first-child{"
         "stroke:var(--amber);stroke-width:3}.node-accent{fill:var(--teal);"
         "stroke:none!important;filter:none!important}"
         ".node[data-entity-kind*=\"memory\"] .node-accent{fill:var(--amber)}"
         ".node[data-entity-kind*=\"switch\"] .node-accent{fill:var(--green)}"
         ".node[data-entity-kind*=\"boundary\"] .node-accent{fill:var(--red)}"
         ".node[data-entity-kind*=\"acc_core\"] "
         ".node-accent{fill:var(--violet)}"
         ".node-title{font-size:13px;font-weight:700;fill:var(--ink);pointer-"
         "events:none}"
         ".node-detail{font-size:10.5px;fill:var(--muted);pointer-events:none}"
         ".inspector h2{font-size:17px;margin:0 0 4px}.inspector "
         ".subtitle{color:var(--muted);margin:0 0 20px}"
         ".fact{padding:12px 0;border-top:1px solid #dbe1e5}.fact "
         "dt{font-size:11px;text-transform:uppercase;"
         "font-weight:750;color:#77828d}.fact dd{margin:5px 0 "
         "0;overflow-wrap:anywhere}.mono{font-family:"
         "ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px}.hint{"
         "color:var(--muted);font-size:12px}"
         "@media(max-width:900px){.app{grid-template-columns:210px "
         "minmax(0,1fr);grid-template-rows:58px "
         "minmax(0,1fr) "
         "180px}.inspector{grid-column:1/-1;border-left:0;border-top:1px solid "
         "#dce2e7;"
         "display:grid;grid-template-columns:1fr 2fr;gap:14px;padding:14px "
         "18px}.inspector .facts{display:flex;"
         "gap:20px;overflow:auto}.fact{border-top:0;min-width:150px;padding:0}."
         "root-id{display:none}}"
         "@media(max-width:620px){.app{grid-template-columns:1fr;grid-template-"
         "rows:54px 128px minmax(0,1fr) 164px}"
         ".topbar{padding:0 "
         "10px}.brand{font-size:15px}.sidebar{border-right:0;border-bottom:1px "
         "solid #dce2e7;"
         "padding:10px}.search{margin-bottom:8px}.view-list{flex-direction:row;"
         "overflow:auto}.view-button{min-width:150px}"
         ".canvas{grid-row:3}.inspector{grid-row:4}.section-label{display:none}"
         "}"
         "</style></head><body><main class=\"app\">";
  output << "<header class=\"topbar\"><div class=\"brand\">"
         << escapeHtml(document.title) << "</div><div class=\"root-id\">"
         << escapeHtml(document.rootIdentity) << "</div><div class=\"tools\">";
  for (auto button :
       {std::pair<llvm::StringRef, llvm::StringRef>("zoom-out", "Zoom out"),
        {"zoom-in", "Zoom in"},
        {"fit", "Fit graph"},
        {"reset", "Reset view"}}) {
    output << "<button class=\"icon-button\" type=\"button\" data-action=\""
           << button.first << "\" title=\"" << button.second
           << "\" aria-label=\"" << button.second << "\">";
    writeIcon(output, button.first);
    output << "</button>";
  }
  output << "</div></header><aside class=\"sidebar\"><div class=\"search\">";
  writeIcon(output, "search");
  output << "<input id=\"node-search\" type=\"search\" placeholder=\"Find "
            "resource\" "
            "aria-label=\"Find resource\"></div><div "
            "class=\"section-label\">Views</div>"
            "<nav class=\"view-list\" aria-label=\"Fabric views\">";
  for (auto [index, graph] : llvm::enumerate(document.graphs))
    output << "<button type=\"button\" class=\"view-button"
           << (index == 0 ? " is-active" : "") << "\" data-view-target=\""
           << escapeHtml(graph.id) << "\">" << escapeHtml(graph.title)
           << "<small>" << graph.nodes.size() << " nodes / "
           << graph.edges.size() << " links</small></button>";
  output << "</nav></aside><section class=\"canvas\" id=\"canvas\">";
  for (auto [index, graph] : llvm::enumerate(document.graphs))
    writeGraph(graph, index == 0, output);
  output
      << "</section><aside class=\"inspector\"><div><h2 "
         "id=\"inspect-title\">Fabric overview</h2>"
         "<p class=\"subtitle\" id=\"inspect-subtitle\">Select a resource to "
         "inspect it.</p></div>"
         "<dl class=\"facts\"><div class=\"fact\"><dt>View</dt><dd "
         "id=\"inspect-view\">"
      << (document.graphs.empty() ? "None"
                                  : escapeHtml(document.graphs.front().title))
      << "</dd></div><div class=\"fact\"><dt>Kind</dt><dd "
         "id=\"inspect-kind\">Canonical Fabric</dd>"
         "</div><div class=\"fact\"><dt>Identity</dt><dd class=\"mono\" "
         "id=\"inspect-id\">"
      << escapeHtml(document.rootIdentity)
      << "</dd></div></dl></aside></main><script>"
         "(()=>{'use strict';const "
         "views=[...document.querySelectorAll('.graph-view')];"
         "const buttons=[...document.querySelectorAll('.view-button')];let "
         "active=views[0]||null;"
         "const states=new Map();for(const svg of views){const "
         "b=svg.viewBox.baseVal;states.set(svg,{"
         "origin:[b.x,b.y,b.width,b.height],box:[b.x,b.y,b.width,b.height]});}"
         "const apply=svg=>{const "
         "s=states.get(svg);svg.setAttribute('viewBox',s.box.join(' '));};"
         "const setView=id=>{for(const svg of "
         "views)svg.classList.toggle('is-active',svg.dataset.viewId===id);"
         "for(const b of "
         "buttons)b.classList.toggle('is-active',b.dataset.viewTarget===id);"
         "active=views.find(v=>v.dataset.viewId===id)||active;document."
         "getElementById('inspect-view').textContent="
         "buttons.find(b=>b.dataset.viewTarget===id)?.childNodes[0]?."
         "textContent||id;};"
         "for(const b of "
         "buttons)b.addEventListener('click',()=>setView(b.dataset.viewTarget))"
         ";"
         "const zoom=f=>{if(!active)return;const "
         "s=states.get(active),[x,y,w,h]=s.box,nw=w*f,nh=h*f;"
         "s.box=[x+(w-nw)/2,y+(h-nh)/2,nw,nh];apply(active);};"
         "document.querySelector('[data-action=zoom-in]').onclick=()=>zoom(.82)"
         ";"
         "document.querySelector('[data-action=zoom-out]').onclick=()=>zoom(1."
         "22);"
         "document.querySelector('[data-action=fit]').onclick=()=>{if(active){"
         "states.get(active).box=[...states.get(active).origin];apply(active);}"
         "};"
         "document.querySelector('[data-action=reset]').onclick=()=>{for(const "
         "svg of "
         "views){states.get(svg).box=[...states.get(svg).origin];apply(svg);}};"
         "for(const svg of views){let "
         "drag=null;svg.addEventListener('pointerdown',e=>{if(e.target.closest("
         "'.node'))return;"
         "svg.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,box:"
         "[...states.get(svg).box]};});"
         "svg.addEventListener('pointermove',e=>{if(!drag)return;const "
         "s=states.get(svg),rect=svg.getBoundingClientRect();"
         "s.box=[drag.box[0]-(e.clientX-drag.x)*drag.box[2]/"
         "rect.width,drag.box[1]-(e.clientY-drag.y)*drag.box[3]/"
         "rect.height,drag.box[2],drag.box[3]];apply(svg);});"
         "svg.addEventListener('pointerup',()=>drag=null);svg.addEventListener("
         "'wheel',e=>{e.preventDefault();"
         "const "
         "s=states.get(svg),rect=svg.getBoundingClientRect(),f=e.deltaY<0?.88:"
         "1.14,px=(e.clientX-rect.left)/rect.width,"
         "py=(e.clientY-rect.top)/"
         "rect.height,nw=s.box[2]*f,nh=s.box[3]*f;s.box=[s.box[0]+(s.box[2]-nw)"
         "*px,"
         "s.box[1]+(s.box[3]-nh)*py,nw,nh];apply(svg);},{passive:false});}"
         "const "
         "select=node=>{document.querySelectorAll('.node.is-selected').forEach("
         "n=>n.classList.remove('is-selected'));"
         "node.classList.add('is-selected');document.getElementById('inspect-"
         "title').textContent=node.dataset.label;"
         "document.getElementById('inspect-subtitle').textContent=node.dataset."
         "detail;"
         "document.getElementById('inspect-kind').textContent=node.dataset."
         "entityKind;"
         "document.getElementById('inspect-id').textContent=node.dataset."
         "nodeId;};"
         "document.querySelectorAll('.node').forEach(n=>{n.addEventListener('"
         "click',()=>select(n));"
         "n.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' "
         "')select(n);});});"
         "document.getElementById('node-search').addEventListener('input',e=>{"
         "const q=e.target.value.trim().toLowerCase();"
         "document.querySelectorAll('.node').forEach(n=>n.classList.toggle('is-"
         "match',q&&"
         "(n.dataset.label+' '+n.dataset.detail+' "
         "'+n.dataset.entityKind).toLowerCase().includes(q)));});"
         "})();</script></body></html>";
  return llvm::Error::success();
}

} // namespace loom::fabric::visualization

namespace loom::fabric {

llvm::Error writeFabricVisualizationHtml(const FinalizedFabricRoot &root,
                                         const ArtifactStore &store,
                                         llvm::raw_ostream &output) {
  auto document = visualization::buildDocument(root, store);
  if (!document)
    return document.takeError();
  return visualization::writeHtml(*document, output);
}

} // namespace loom::fabric
