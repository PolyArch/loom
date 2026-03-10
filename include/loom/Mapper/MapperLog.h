//===-- MapperLog.h - Verbose logging for mapper pipeline ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// MapperLog captures structured, per-stage diagnostic information during
// place-and-route. When verbose mode is enabled, the log is written to
// <base_path>.log alongside other mapping artifacts.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_MAPPERLOG_H
#define LOOM_MAPPER_MAPPERLOG_H

#include "loom/Mapper/Types.h"

#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <string>
#include <vector>

namespace loom {

class Graph;
class MappingState;

class MapperLog {
public:
  MapperLog() = default;

  /// Enable or disable logging.
  void setEnabled(bool v) { enabled = v; }
  bool isEnabled() const { return enabled; }

  /// Mark the start of a named stage (e.g. "Preprocessing", "Placement").
  void beginStage(llvm::StringRef name);

  /// Mark the end of the current stage. Records elapsed time.
  void endStage();

  /// Log a free-form informational message within the current stage.
  void info(const std::string &msg);

  /// Log a per-node placement decision.
  void logPlacement(IdIndex swNode, IdIndex hwNode,
                    llvm::StringRef swName, llvm::StringRef hwName,
                    double score);

  /// Log a routing attempt for an edge.
  void logRouteAttempt(IdIndex edgeId, IdIndex srcPort, IdIndex dstPort,
                       bool success, unsigned hops);

  /// Log an edge legality rejection during path search.
  void logEdgeRejection(IdIndex srcPort, IdIndex dstPort,
                        llvm::StringRef reason);

  /// Log refinement iteration result.
  void logRefinement(int iteration, unsigned failedEdges, bool improved);

  /// Log temporal assignment for a SW node.
  void logTemporalAssignment(IdIndex swNode, IdIndex tpeId,
                             IdIndex slot, IdIndex tag, IdIndex opcode);

  /// Log temporal SW assignment entry.
  void logTemporalSWEntry(IdIndex tswNode, IdIndex slot,
                          IdIndex tag, uint64_t routeMask);

  /// Log cost breakdown at end of pipeline.
  void logCost(double total, double placement, double routing,
               double temporal, double perfProxy, double configFp);

  /// Log a validation result.
  void logValidation(llvm::StringRef check, bool passed,
                     const std::string &detail);

  /// Log a summary of mapping state (counts of mapped/unmapped entities).
  void logStateSummary(const MappingState &state, const Graph &dfg,
                       const Graph &adg);

  /// Write accumulated log to a file. Returns true on success.
  bool writeToFile(const std::string &path) const;

private:
  bool enabled = false;

  struct Entry {
    enum Kind {
      STAGE_BEGIN,
      STAGE_END,
      INFO,
      PLACEMENT,
      ROUTE_ATTEMPT,
      EDGE_REJECTION,
      REFINEMENT,
      TEMPORAL_PE,
      TEMPORAL_SW,
      COST,
      VALIDATION,
      STATE_SUMMARY,
    };
    Kind kind;
    std::string text;
  };

  std::vector<Entry> entries;

  std::string currentStage;
  std::chrono::steady_clock::time_point stageStart;
};

} // namespace loom

#endif // LOOM_MAPPER_MAPPERLOG_H
