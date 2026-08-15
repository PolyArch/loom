#include "SpatialCandidateSelectionTestSupport.h"

#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::test {

llvm::Error selectReachableGraphBoundaries(
    pnr::SpatialCandidateState &candidate, pnr::SpatialMoveTransaction &move,
    llvm::ArrayRef<pnr::PnrIndex> selectedPortAttachments,
    bool requireDistinctEndpoints) {
  const auto &problem = candidate.problem();
  const auto reachable = [&](pnr::PnrIndex source, pnr::PnrIndex destination) {
    const auto &routing = problem.routing();
    std::vector<std::uint8_t> visited(routing.routingEndpoints().size(), 0);
    std::vector<pnr::PnrIndex> worklist{source};
    visited[source] = 1;
    for (std::size_t cursor = 0; cursor < worklist.size(); ++cursor) {
      const auto current = worklist[cursor];
      if (current == destination)
        return true;
      const auto offsets = routing.adjacencyOffsets();
      for (pnr::PnrIndex arc = offsets[current]; arc != offsets[current + 1];
           ++arc) {
        const auto next = routing.routingArcs()[arc].target;
        if (!visited[next]) {
          visited[next] = 1;
          worklist.push_back(next);
        }
      }
    }
    return false;
  };
  const auto selectedEndpoint =
      [&](pnr::FrozenSpatialTerminalBinding binding,
          std::optional<std::pair<pnr::PnrIndex, pnr::PnrIndex>> override) {
        pnr::PnrIndex option = 0;
        if (binding.kind == pnr::FrozenSpatialTerminalBindingKind::PortDemand) {
          option = selectedPortAttachments.empty()
                       ? candidate.portAttachment(binding.index)
                       : selectedPortAttachments[binding.index];
        } else if (override && override->first == binding.index) {
          option = override->second;
        } else {
          option = candidate.graphBoundaryAttachment(binding.index);
        }
        return problem.ports().attachmentOptions()[option].endpoint;
      };
  std::vector<pnr::PnrIndex> selectedBoundaryEndpoints;
  for (auto [boundaryOrdinal, boundary] :
       llvm::enumerate(problem.ports().graphBoundaries())) {
    const auto netOrdinal = boundary.logicalNet;
    const auto &net = problem.transfers().logicalNets()[netOrdinal];
    const bool ingressBoundary =
        std::holds_alternative<dataflow::GraphIngressTokenRef>(
            boundary.terminal);
    bool selected = false;
    for (pnr::PnrIndex option = boundary.attachmentOptionOffset;
         option !=
         boundary.attachmentOptionOffset + boundary.attachmentOptionCount;
         ++option) {
      const auto override =
          std::make_pair(static_cast<pnr::PnrIndex>(boundaryOrdinal), option);
      const pnr::PnrIndex endpoint =
          problem.ports().attachmentOptions()[option].endpoint;
      if (requireDistinctEndpoints && ingressBoundary &&
          llvm::is_contained(selectedBoundaryEndpoints, endpoint))
        continue;
      const auto source = selectedEndpoint(
          problem.transfers().logicalNetSourceBindings()[netOrdinal], override);
      bool connects = true;
      for (const auto sink : problem.transfers().logicalNetSinkBindings().slice(
               net.sinkOffset, net.sinkCount))
        connects &= reachable(source, selectedEndpoint(sink, override));
      if (!connects)
        continue;
      if (llvm::Error error = move.setGraphBoundaryAttachment(
              static_cast<pnr::PnrIndex>(boundaryOrdinal), option))
        return error;
      if (ingressBoundary)
        selectedBoundaryEndpoints.push_back(endpoint);
      selected = true;
      break;
    }
    if (!selected)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "graph boundary %zu has no reachable%s "
                                     "attachment",
                                     static_cast<std::size_t>(boundaryOrdinal),
                                     requireDistinctEndpoints ? " distinct"
                                                              : "");
  }
  return llvm::Error::success();
}

} // namespace loom::test
