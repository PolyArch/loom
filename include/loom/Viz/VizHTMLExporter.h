//===-- VizHTMLExporter.h - Self-contained HTML visualization ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Exports a self-contained .viz.html file from mapper results. The HTML embeds
// all data (ADG graph, DFG DOT, mapping state, metadata) and vendored JS/CSS
// assets so it opens in any modern browser with zero network access.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_VIZ_VIZHTMLEXPORTER_H
#define LOOM_VIZ_VIZHTMLEXPORTER_H

#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Simulator/SimTypes.h"

#include "mlir/IR/BuiltinOps.h"

#include <string>
#include <vector>

namespace loom {

class VizHTMLExporter {
public:
  /// Emit a self-contained .viz.html file at <basePath>.viz.html.
  /// Returns true on success.
  /// \p dfgModule is the Handshake MLIR module (for DFG attribute enrichment).
  /// \p fabricModule is the fabric MLIR module (for PE body ops extraction).
  /// \p traceEvents optional trace data for simulation playback controls.
  /// \p totalCycles / \p configCycles are used only when traceEvents is set.
  bool emitHTML(const Graph &adg, const Graph &dfg,
                const MappingState &state,
                mlir::ModuleOp dfgModule,
                mlir::Operation *fabricModule,
                const std::string &basePath,
                bool vizNeato = false,
                const std::vector<sim::TraceEvent> *traceEvents = nullptr,
                uint64_t totalCycles = 0, uint64_t configCycles = 0,
                const std::vector<sim::PerfSnapshot> *nodePerf = nullptr,
                sim::TraceMode traceMode = sim::TraceMode::Full);
};

} // namespace loom

#endif // LOOM_VIZ_VIZHTMLEXPORTER_H
