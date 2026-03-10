//===-- MapperLog.cpp - Verbose logging for mapper pipeline --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/MapperLog.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

namespace loom {

void MapperLog::beginStage(llvm::StringRef name) {
  if (!enabled)
    return;
  currentStage = name.str();
  stageStart = std::chrono::steady_clock::now();
  entries.push_back(
      {Entry::STAGE_BEGIN,
       "==== " + currentStage + " ===="});
}

void MapperLog::endStage() {
  if (!enabled)
    return;
  auto elapsed = std::chrono::steady_clock::now() - stageStart;
  double ms =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() /
      1000.0;

  std::string text;
  llvm::raw_string_ostream os(text);
  os << "---- " << currentStage
     << llvm::format(" completed (%.3f ms)", ms)
     << " ----";
  entries.push_back({Entry::STAGE_END, text});
  currentStage.clear();
}

void MapperLog::info(const std::string &msg) {
  if (!enabled)
    return;
  entries.push_back({Entry::INFO, "  [info] " + msg});
}

void MapperLog::logPlacement(IdIndex swNode, IdIndex hwNode,
                             llvm::StringRef swName, llvm::StringRef hwName,
                             double score) {
  if (!enabled)
    return;
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "  [place] SW N" << swNode << " (" << swName << ")"
     << " -> HW H" << hwNode << " (" << hwName << ")"
     << llvm::format("  score=%.4f", score);
  entries.push_back({Entry::PLACEMENT, text});
}

void MapperLog::logRouteAttempt(IdIndex edgeId, IdIndex srcPort,
                                IdIndex dstPort, bool success,
                                unsigned hops) {
  if (!enabled)
    return;
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "  [route] E" << edgeId
     << "  srcPort=" << srcPort << " -> dstPort=" << dstPort;
  if (success)
    os << "  OK  hops=" << hops;
  else
    os << "  FAIL (no path)";
  entries.push_back({Entry::ROUTE_ATTEMPT, text});
}

void MapperLog::logEdgeRejection(IdIndex srcPort, IdIndex dstPort,
                                 llvm::StringRef reason) {
  if (!enabled)
    return;
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "    [reject] port " << srcPort << " -> " << dstPort
     << "  reason: " << reason;
  entries.push_back({Entry::EDGE_REJECTION, text});
}

void MapperLog::logRefinement(int iteration, unsigned failedEdges,
                              bool improved) {
  if (!enabled)
    return;
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "  [refine] iter=" << iteration
     << "  failedEdges=" << failedEdges
     << (improved ? "  improved" : "  no improvement");
  entries.push_back({Entry::REFINEMENT, text});
}

void MapperLog::logTemporalAssignment(IdIndex swNode, IdIndex tpeId,
                                      IdIndex slot, IdIndex tag,
                                      IdIndex opcode) {
  if (!enabled)
    return;
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "  [temporal-pe] SW N" << swNode
     << " -> TPE H" << tpeId
     << "  slot=" << slot << "  tag=" << tag << "  opcode=" << opcode;
  entries.push_back({Entry::TEMPORAL_PE, text});
}

void MapperLog::logTemporalSWEntry(IdIndex tswNode, IdIndex slot,
                                   IdIndex tag, uint64_t routeMask) {
  if (!enabled)
    return;
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "  [temporal-sw] H" << tswNode
     << "  slot=" << slot << "  tag=" << tag
     << llvm::format("  routeMask=0x%llx", routeMask);
  entries.push_back({Entry::TEMPORAL_SW, text});
}

void MapperLog::logCost(double total, double placement, double routing,
                        double temporal, double perfProxy, double configFp) {
  if (!enabled)
    return;
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "  [cost]"
     << llvm::format("  total=%.4f", total)
     << llvm::format("  placement=%.4f", placement)
     << llvm::format("  routing=%.4f", routing)
     << llvm::format("  temporal=%.4f", temporal)
     << llvm::format("  perfProxy=%.4f", perfProxy)
     << llvm::format("  configFp=%.4f", configFp);
  entries.push_back({Entry::COST, text});
}

void MapperLog::logValidation(llvm::StringRef check, bool passed,
                              const std::string &detail) {
  if (!enabled)
    return;
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "  [validate] " << check << ": " << (passed ? "PASS" : "FAIL");
  if (!detail.empty())
    os << "  " << detail;
  entries.push_back({Entry::VALIDATION, text});
}

void MapperLog::logStateSummary(const MappingState &state, const Graph &dfg,
                                const Graph &adg) {
  if (!enabled)
    return;

  unsigned totalSwNodes = 0, mappedNodes = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    if (dfg.getNode(i)) {
      ++totalSwNodes;
      if (i < state.swNodeToHwNode.size() &&
          state.swNodeToHwNode[i] != INVALID_ID)
        ++mappedNodes;
    }
  }

  unsigned totalEdges = 0, routedEdges = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    if (dfg.getEdge(i)) {
      ++totalEdges;
      if (i < state.swEdgeToHwPaths.size() &&
          !state.swEdgeToHwPaths[i].empty())
        ++routedEdges;
    }
  }

  std::string text;
  llvm::raw_string_ostream os(text);
  os << "  [summary] nodes: " << mappedNodes << "/" << totalSwNodes
     << " mapped  edges: " << routedEdges << "/" << totalEdges << " routed";
  entries.push_back({Entry::STATE_SUMMARY, text});
}

bool MapperLog::writeToFile(const std::string &path) const {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return false;

  out << "=== Loom Mapper Verbose Log ===\n\n";

  for (const auto &entry : entries) {
    out << entry.text << "\n";
    // Add blank line after stage end for readability.
    if (entry.kind == Entry::STAGE_END)
      out << "\n";
  }

  return true;
}

} // namespace loom
