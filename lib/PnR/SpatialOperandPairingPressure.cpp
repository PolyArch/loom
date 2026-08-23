#include "SpatialOperandPairingPressure.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <system_error>

namespace loom::pnr::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_operand_pairing_pressure_invalid: " + message);
}

llvm::Expected<std::uint64_t>
groupPressure(const FrozenSpatialPnrProblem &problem,
              llvm::ArrayRef<PnrIndex> portAttachments,
              llvm::ArrayRef<PnrIndex> registerFifoTransfers, PnrIndex group) {
  const FrozenSpatialPortIndex &ports = problem.ports();
  if (group >= ports.operandPairingGroups().size())
    return invalid("pairing group is out of range");
  llvm::SmallVector<PnrIndex, 4> ingresses;
  for (PnrIndex demand : ports.operandPairingGroupMembers(group)) {
    if (demand >= portAttachments.size() ||
        portAttachments[demand] >= ports.attachmentOptions().size())
      return invalid("pairing member attachment is out of range");
    const PnrIndex logicalNet = ports.portDemands()[demand].logicalNet;
    if (logicalNet >= registerFifoTransfers.size())
      return invalid("pairing member logical net is out of range");
    if (registerFifoTransfers[logicalNet] != getInvalidPnrIndex())
      continue;
    const FrozenSpatialAttachmentOption &option =
        ports.attachmentOptions()[portAttachments[demand]];
    if (option.progressBoundary !=
        ::loom::mapping::SpatialDurableProgressBoundaryKind::
            TemporalPeOperandQueue)
      continue;
    ingresses.push_back(option.endpoint);
  }
  if (ingresses.size() < 2)
    return std::uint64_t{0};
  llvm::sort(ingresses);
  const std::size_t distinct = std::distance(
      ingresses.begin(), std::unique(ingresses.begin(), ingresses.end()));
  return static_cast<std::uint64_t>(ingresses.size() - distinct);
}

} // namespace

llvm::Expected<std::uint64_t> measureSpatialOperandIngressPressure(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> registerFifoTransfers,
    llvm::ArrayRef<PnrIndex> pairingGroups) {
  if (portAttachments.size() != problem.ports().portDemands().size())
    return invalid("PortAttachment selection has the wrong width");
  if (registerFifoTransfers.size() != problem.transfers().logicalNets().size())
    return invalid("register-FIFO selection has the wrong width");
  if (!llvm::is_sorted(pairingGroups) ||
      std::adjacent_find(pairingGroups.begin(), pairingGroups.end()) !=
          pairingGroups.end())
    return invalid("pairing-group subset is not canonical");
  std::uint64_t total = 0;
  for (PnrIndex group : pairingGroups) {
    auto pressure =
        groupPressure(problem, portAttachments, registerFifoTransfers, group);
    if (!pressure)
      return pressure.takeError();
    if (*pressure > std::numeric_limits<std::uint64_t>::max() - total)
      return invalid("shared-ingress pressure exceeds u64");
    total += *pressure;
  }
  return total;
}

llvm::Expected<std::uint64_t> measureSpatialOperandIngressPressure(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> registerFifoTransfers) {
  if (portAttachments.size() != problem.ports().portDemands().size())
    return invalid("PortAttachment selection has the wrong width");
  if (registerFifoTransfers.size() != problem.transfers().logicalNets().size())
    return invalid("register-FIFO selection has the wrong width");
  std::uint64_t total = 0;
  for (auto indexed : llvm::enumerate(problem.ports().operandPairingGroups())) {
    if (indexed.index() > getPnrIndexMax())
      return invalid("pairing-group inventory exceeds PnrIndex");
    auto pressure =
        groupPressure(problem, portAttachments, registerFifoTransfers,
                      static_cast<PnrIndex>(indexed.index()));
    if (!pressure)
      return pressure.takeError();
    if (*pressure > std::numeric_limits<std::uint64_t>::max() - total)
      return invalid("shared-ingress pressure exceeds u64");
    total += *pressure;
  }
  return total;
}

} // namespace loom::pnr::detail
