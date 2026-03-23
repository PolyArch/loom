#include "loom/SystemCompiler/L2CoreCompiler.h"
#include "loom/Mapper/ADGFlattener.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/MapperTiming.h"
#include "loom/Mapper/MappingState.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace loom {

namespace {

/// Check whether the DFG requires operation types that the ADG does not
/// support at all (TYPE_MISMATCH), or whether there are not enough FU
/// instances of a required type (INSUFFICIENT_FU).
///
/// Returns true if a cut was produced.
bool checkFUAvailability(const Graph &dfg, const Graph &adg,
                         const std::string &kernelName,
                         const std::string &coreType,
                         InfeasibilityCut &cut) {
  // Count operations required by the DFG, keyed by op_name.
  llvm::StringMap<unsigned> requiredOps;
  for (IdIndex swId = 0; swId < static_cast<IdIndex>(dfg.nodes.size());
       ++swId) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode || swNode->kind != Node::OperationNode)
      continue;
    llvm::StringRef opName = getNodeAttrStr(swNode, "op_name");
    if (opName.empty())
      continue;
    requiredOps[opName]++;
  }

  // Count FUs available in the ADG, keyed by supported op names.
  // A single FU may support multiple operations; each gets credit.
  llvm::StringMap<unsigned> availableOps;
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      continue;
    if (getNodeAttrStr(hwNode, "resource_class") != "functional")
      continue;
    llvm::StringRef opKind = getNodeAttrStr(hwNode, "op_kind");
    if (!opKind.empty())
      availableOps[opKind]++;
    llvm::StringRef opName = getNodeAttrStr(hwNode, "op_name");
    if (!opName.empty() && opName != opKind)
      availableOps[opName]++;
  }

  // First pass: check for complete type mismatches (zero availability).
  for (const auto &entry : requiredOps) {
    llvm::StringRef opName = entry.getKey();
    unsigned needed = entry.getValue();
    auto it = availableOps.find(opName);
    if (it == availableOps.end() || it->second == 0) {
      cut.kernelName = kernelName;
      cut.coreType = coreType;
      cut.reason = CutReason::TYPE_MISMATCH;
      cut.evidence = FUShortage{opName.str(), needed, 0};
      return true;
    }
  }

  // Second pass: check for FU shortages (needed > available).
  for (const auto &entry : requiredOps) {
    llvm::StringRef opName = entry.getKey();
    unsigned needed = entry.getValue();
    auto it = availableOps.find(opName);
    unsigned available = (it != availableOps.end()) ? it->second : 0;
    if (needed > available) {
      cut.kernelName = kernelName;
      cut.coreType = coreType;
      cut.reason = CutReason::INSUFFICIENT_FU;
      cut.evidence = FUShortage{opName.str(), needed, available};
      return true;
    }
  }

  return false;
}

/// Check the mapper diagnostics string for routing congestion indicators.
bool checkRoutingCongestion(const Mapper::Result &result,
                            const std::string &kernelName,
                            const std::string &coreType,
                            InfeasibilityCut &cut) {
  llvm::StringRef diag(result.diagnostics);
  bool hasRoutingIssue = diag.contains("routing") || diag.contains("unrouted") ||
                         diag.contains("congestion") || diag.contains("edge");

  if (!hasRoutingIssue)
    return false;

  // Compute utilization from the search summary as a proxy.
  double utilization = 0.0;
  if (result.searchSummary.routedLaneCount > 0) {
    utilization = 1.0; // If some routing was attempted but failed, report 100%.
  }

  cut.kernelName = kernelName;
  cut.coreType = coreType;
  cut.reason = CutReason::ROUTING_CONGESTION;
  cut.evidence = CongestionInfo{utilization};
  return true;
}

/// Check for SPM overflow: DFG requires more memory than available.
bool checkSPMOverflow(const Graph &dfg, const Graph &adg,
                      const std::string &kernelName,
                      const std::string &coreType,
                      InfeasibilityCut &cut) {
  // Count memory nodes required by the DFG.
  unsigned requiredMemNodes = 0;
  for (IdIndex swId = 0; swId < static_cast<IdIndex>(dfg.nodes.size());
       ++swId) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode || swNode->kind != Node::OperationNode)
      continue;
    llvm::StringRef opName = getNodeAttrStr(swNode, "op_name");
    if (opName == "handshake.memory" || opName == "handshake.extmemory")
      ++requiredMemNodes;
  }

  // Count memory nodes available in the ADG.
  unsigned availableMemNodes = 0;
  uint64_t totalSpmBytes = 0;
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      continue;
    if (getNodeAttrStr(hwNode, "resource_class") != "memory")
      continue;
    ++availableMemNodes;
    int64_t sz = getNodeAttrInt(hwNode, "mem_size_bytes", 0);
    if (sz > 0)
      totalSpmBytes += static_cast<uint64_t>(sz);
  }

  if (requiredMemNodes > availableMemNodes) {
    cut.kernelName = kernelName;
    cut.coreType = coreType;
    cut.reason = CutReason::SPM_OVERFLOW;
    cut.evidence = SPMInfo{static_cast<uint64_t>(requiredMemNodes),
                           static_cast<uint64_t>(availableMemNodes)};
    return true;
  }

  return false;
}

} // namespace

InfeasibilityCut analyzeFailure(const Mapper::Result &result, const Graph &dfg,
                                const Graph &adg,
                                const ADGFlattener &flattener,
                                const std::string &kernelName,
                                const std::string &coreType,
                                std::optional<unsigned> targetII) {
  InfeasibilityCut cut;

  // Priority 1: TYPE_MISMATCH or INSUFFICIENT_FU
  if (checkFUAvailability(dfg, adg, kernelName, coreType, cut))
    return cut;

  // Priority 2: SPM_OVERFLOW
  if (checkSPMOverflow(dfg, adg, kernelName, coreType, cut))
    return cut;

  // Priority 3: ROUTING_CONGESTION
  if (checkRoutingCongestion(result, kernelName, coreType, cut))
    return cut;

  // Default: II_UNACHIEVABLE
  cut.kernelName = kernelName;
  cut.coreType = coreType;
  cut.reason = CutReason::II_UNACHIEVABLE;
  unsigned minII = result.timingSummary.estimatedInitiationInterval;
  unsigned target = targetII.value_or(1);
  cut.evidence = IIInfo{minII, target};
  return cut;
}

} // namespace loom
