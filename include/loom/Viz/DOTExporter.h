//===-- DOTExporter.h - DOT graph export for visualization --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Functions to export DFG, ADG, and mapped visualization data as DOT strings,
// and to generate self-contained HTML viewer files.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_VIZ_DOTEXPORTER_H
#define LOOM_VIZ_DOTEXPORTER_H

#include <string>

namespace mlir {
class Operation;
} // namespace mlir

namespace loom {

class Graph;
class MappingState;

namespace viz {

/// Display mode for DOT export.
enum class DOTMode {
  Structure, // Compact topology view
  Detailed,  // Full ports, parameters, and attributes
};

/// Options controlling DOT export behavior.
struct DOTOptions {
  DOTMode mode = DOTMode::Structure;
  bool includeAttributes = true;
};

/// Export a software dataflow graph (DFG) from a handshake.func to DOT.
/// The DFG is represented as a Graph built by DFGBuilder.
/// @param dfg  The software dataflow graph.
/// @param opts Export options.
/// @return     DOT string for the DFG.
std::string exportDFGDot(const Graph &dfg, const DOTOptions &opts = {});

/// Export a hardware architecture description graph (ADG) to DOT.
/// The ADG is represented as a Graph built by ADGFlattener.
/// @param adg  The hardware architecture graph.
/// @param opts Export options (Structure vs Detailed mode).
/// @return     DOT string for the ADG.
std::string exportADGDot(const Graph &adg, const DOTOptions &opts = {});

/// Export a mapped visualization (DFG placed on ADG) to DOT.
/// Produces overlay DOT showing ADG with mapped SW annotations.
/// @param dfg   The software dataflow graph.
/// @param adg   The hardware architecture graph.
/// @param state The mapping state with forward/reverse mappings.
/// @param opts  Export options.
/// @return      DOT string for the overlay view.
std::string exportMappedOverlayDot(const Graph &dfg, const Graph &adg,
                                   const MappingState &state,
                                   const DOTOptions &opts = {});

/// Export side-by-side DFG DOT for mapped visualization.
/// DFG nodes carry extra id attributes for cross-linking.
/// @param dfg   The software dataflow graph.
/// @param state The mapping state.
/// @param opts  Export options.
/// @return      DOT string for the DFG panel.
std::string exportMappedDFGDot(const Graph &dfg, const MappingState &state,
                               const DOTOptions &opts = {});

/// Export side-by-side ADG DOT for mapped visualization.
/// ADG nodes carry mapping annotations and dialect-based coloring.
/// @param dfg   The software dataflow graph (for dialect lookup).
/// @param adg   The hardware architecture graph.
/// @param state The mapping state.
/// @param opts  Export options.
/// @return      DOT string for the ADG panel.
std::string exportMappedADGDot(const Graph &dfg, const Graph &adg,
                               const MappingState &state,
                               const DOTOptions &opts = {});

/// Generate a self-contained HTML file from DOT string(s).
/// The HTML embeds viz.js WASM for client-side Graphviz rendering.
/// @param dotString Primary DOT string to render.
/// @param title     Page title.
/// @return          Complete HTML string.
std::string generateHTML(const std::string &dotString,
                         const std::string &title = "Loom Visualization");

/// Generate a self-contained HTML file for mapped visualization.
/// Embeds overlay DOT, side-by-side DFG/ADG DOTs, and mapping metadata.
/// @param overlayDot  Overlay mode DOT string.
/// @param dfgDot      Side-by-side DFG DOT string.
/// @param adgDot      Side-by-side ADG DOT string.
/// @param mappingJson JSON string with swToHw/hwToSw/routes data.
/// @param metadataJson JSON string with node metadata.
/// @param title       Page title.
/// @return            Complete HTML string.
std::string generateMappedHTML(const std::string &overlayDot,
                               const std::string &dfgDot,
                               const std::string &adgDot,
                               const std::string &mappingJson,
                               const std::string &metadataJson,
                               const std::string &title = "Loom Mapped View");

} // namespace viz
} // namespace loom

#endif // LOOM_VIZ_DOTEXPORTER_H
