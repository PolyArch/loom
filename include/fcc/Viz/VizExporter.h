#ifndef FCC_VIZ_VIZEXPORTER_H
#define FCC_VIZ_VIZEXPORTER_H

#include "fcc/Mapper/ADGFlattener.h"
#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/MappingState.h"

#include "mlir/Support/LogicalResult.h"

#include <string>
#include <vector>

namespace fcc {

/// A single trace event for animation playback.
/// Placeholder for future simulator integration.
struct TraceEvent {
  uint64_t cycle = 0;
  IdIndex hwNodeId = INVALID_ID;
  enum Kind { Fire, StallIn, StallOut } kind = Fire;
};

/// Export an interactive HTML visualization of the ADG, DFG, mapping,
/// and (optionally) trace events.
///
/// The output file is fully self-contained and can be opened in any
/// modern browser.
mlir::LogicalResult exportVisualization(const std::string &outputPath,
                                        const Graph &adg, const Graph &dfg,
                                        const MappingState &mapping,
                                        const ADGFlattener &flattener,
                                        const std::vector<TraceEvent> &trace = {});

} // namespace fcc

#endif // FCC_VIZ_VIZEXPORTER_H
