#include "loom/SystemCompiler/NoCScheduler.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <numeric>

namespace loom {

//===----------------------------------------------------------------------===//
// Routing Utilities
//===----------------------------------------------------------------------===//

std::pair<int, int> corePosition(unsigned coreInstanceIdx,
                                 unsigned meshCols) {
  int row = static_cast<int>(coreInstanceIdx / meshCols);
  int col = static_cast<int>(coreInstanceIdx % meshCols);
  return {row, col};
}

std::vector<std::pair<int, int>> computeXYRoute(std::pair<int, int> src,
                                                std::pair<int, int> dst) {
  std::vector<std::pair<int, int>> path;
  path.push_back(src);

  int curRow = src.first;
  int curCol = src.second;

  // X dimension first (column movement).
  while (curCol != dst.second) {
    curCol += (dst.second > curCol) ? 1 : -1;
    path.push_back({curRow, curCol});
  }

  // Then Y dimension (row movement).
  while (curRow != dst.first) {
    curRow += (dst.first > curRow) ? 1 : -1;
    path.push_back({curRow, curCol});
  }

  return path;
}

std::vector<std::pair<int, int>> computeYXRoute(std::pair<int, int> src,
                                                std::pair<int, int> dst) {
  std::vector<std::pair<int, int>> path;
  path.push_back(src);

  int curRow = src.first;
  int curCol = src.second;

  // Y dimension first (row movement).
  while (curRow != dst.first) {
    curRow += (dst.first > curRow) ? 1 : -1;
    path.push_back({curRow, curCol});
  }

  // Then X dimension (column movement).
  while (curCol != dst.second) {
    curCol += (dst.second > curCol) ? 1 : -1;
    path.push_back({curRow, curCol});
  }

  return path;
}

//===----------------------------------------------------------------------===//
// Link Key for Utilization Tracking
//===----------------------------------------------------------------------===//

namespace {

/// Directed link between two adjacent mesh nodes.
struct DirectedLink {
  std::pair<int, int> src;
  std::pair<int, int> dst;

  bool operator<(const DirectedLink &other) const {
    if (src != other.src)
      return src < other.src;
    return dst < other.dst;
  }
};

/// Accumulated bandwidth demand and contract names for a link.
struct LinkDemand {
  double bandwidthDemand = 0.0;
  std::vector<std::string> contractNames;
};

/// Construct a contract edge name from producer and consumer kernel names.
std::string makeEdgeName(const ContractSpec &contract) {
  return contract.producerKernel + " -> " + contract.consumerKernel;
}

} // namespace

//===----------------------------------------------------------------------===//
// NoCScheduler::schedule
//===----------------------------------------------------------------------===//

NoCSchedule NoCScheduler::schedule(const AssignmentResult &assignment,
                                   const std::vector<ContractSpec> &contracts,
                                   const SystemArchitecture &arch,
                                   const NoCSchedulerOptions &opts) {
  NoCSchedule result;
  std::map<DirectedLink, LinkDemand> linkUsage;

  unsigned meshCols = arch.nocSpec.meshCols;
  unsigned flitWidth = arch.nocSpec.flitWidth;
  unsigned routerStages = arch.nocSpec.routerPipelineStages;
  unsigned linkBW = arch.nocSpec.linkBandwidth;

  if (opts.verbose) {
    llvm::errs() << "NoCScheduler: scheduling " << contracts.size()
                 << " contracts on " << arch.nocSpec.meshRows << "x"
                 << meshCols << " mesh\n";
  }

  for (const auto &contract : contracts) {
    // Look up producer and consumer core assignments.
    auto prodIt = assignment.kernelToCore.find(contract.producerKernel);
    auto consIt = assignment.kernelToCore.find(contract.consumerKernel);
    if (prodIt == assignment.kernelToCore.end() ||
        consIt == assignment.kernelToCore.end()) {
      continue;
    }

    unsigned prodCoreIdx = prodIt->second;
    unsigned consCoreIdx = consIt->second;

    // Skip intra-core transfers.
    if (prodCoreIdx == consCoreIdx)
      continue;

    std::pair<int, int> srcPos = corePosition(prodCoreIdx, meshCols);
    std::pair<int, int> dstPos = corePosition(consCoreIdx, meshCols);

    // Compute route based on routing policy.
    std::vector<std::pair<int, int>> hops;
    if (opts.routing == NoCSchedulerOptions::XY_DOR)
      hops = computeXYRoute(srcPos, dstPos);
    else
      hops = computeYXRoute(srcPos, dstPos);

    // Number of links traversed = number of hops - 1.
    unsigned numLinks = hops.empty() ? 0 : static_cast<unsigned>(hops.size()) - 1;

    // Compute data volume and flit count.
    int64_t productionRate = 1;
    if (contract.productionRate.has_value())
      productionRate = contract.productionRate.value();

    unsigned elemSize = estimateElementSize(contract.dataTypeName);
    uint64_t dataVolumeBytes =
        static_cast<uint64_t>(productionRate) * elemSize;

    // Ceiling division for flits.
    uint64_t totalFlits =
        (dataVolumeBytes + flitWidth - 1) / flitWidth;

    // Transfer latency: pipeline fill time.
    unsigned pipelineLatency = numLinks * routerStages;

    // Transfer duration: serialization time.
    unsigned duration =
        linkBW > 0
            ? static_cast<unsigned>((totalFlits + linkBW - 1) / linkBW)
            : static_cast<unsigned>(totalFlits);

    // Bandwidth demand in flits per cycle for this route.
    // Use steady-state ratio if available, otherwise assume 1:1.
    double steadyStateRatio = 1.0;
    if (contract.steadyStateRatio.has_value()) {
      auto ratio = contract.steadyStateRatio.value();
      if (ratio.second != 0)
        steadyStateRatio =
            static_cast<double>(ratio.first) / ratio.second;
    }
    double bwDemand =
        steadyStateRatio > 0.0
            ? static_cast<double>(totalFlits) / steadyStateRatio
            : static_cast<double>(totalFlits);

    std::string edgeName = makeEdgeName(contract);

    NoCRoute route;
    route.contractEdgeName = edgeName;
    route.producerCore = contract.producerKernel;
    route.consumerCore = contract.consumerKernel;
    route.hops = hops;
    route.numHops = numLinks;
    route.bandwidthFlitsPerCycle =
        static_cast<unsigned>(std::ceil(bwDemand));
    route.transferLatencyCycles = pipelineLatency;
    route.totalFlits = totalFlits;
    route.transferDurationCycles = duration;
    route.producerCoreIdx = prodCoreIdx;
    route.consumerCoreIdx = consCoreIdx;

    result.routes.push_back(route);

    // Accumulate link usage for utilization tracking.
    for (size_t i = 0; i + 1 < hops.size(); ++i) {
      DirectedLink link{hops[i], hops[i + 1]};
      linkUsage[link].bandwidthDemand += bwDemand;
      linkUsage[link].contractNames.push_back(edgeName);
    }

    if (opts.verbose) {
      llvm::errs() << "  Route: " << edgeName << " | hops=" << numLinks
                   << " flits=" << totalFlits << " duration=" << duration
                   << " cycles\n";
    }
  }

  // Build link utilization records.
  double maxUtil = 0.0;
  double sumUtil = 0.0;

  for (const auto &entry : linkUsage) {
    double capacity = static_cast<double>(linkBW);
    double util = capacity > 0.0 ? entry.second.bandwidthDemand / capacity
                                 : 0.0;

    LinkUtilization lu;
    lu.srcNode = entry.first.src;
    lu.dstNode = entry.first.dst;
    lu.utilization = util;
    lu.contracts = entry.second.contractNames;
    result.linkUtilizations.push_back(lu);

    maxUtil = std::max(maxUtil, util);
    sumUtil += util;
  }

  result.maxLinkUtilization = maxUtil;
  result.avgLinkUtilization =
      linkUsage.empty() ? 0.0 : sumUtil / linkUsage.size();
  result.hasContention = maxUtil > 1.0;

  // Total transfer cycles: sum of all route durations.
  unsigned totalCycles = 0;
  for (const auto &route : result.routes)
    totalCycles += route.transferDurationCycles;
  result.totalTransferCycles = totalCycles;

  if (opts.verbose) {
    llvm::errs() << "NoCScheduler: " << result.routes.size() << " routes, "
                 << "max link util=" << maxUtil
                 << (result.hasContention ? " (CONTENTION)" : "")
                 << "\n";
  }

  return result;
}

} // namespace loom
