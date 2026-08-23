#include "SpatialCandidateOperandPairingPreference.h"

#include <cstdint>
#include <limits>
#include <system_error>

namespace loom::pnr::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_operand_pairing_preference_invalid: " + message);
}

} // namespace

llvm::Expected<std::uint64_t> scoreSpatialOperandPairingAttachment(
    const FrozenSpatialPnrProblem &problem, PnrIndex demand,
    PnrIndex attachmentOption,
    llvm::ArrayRef<PnrIndex> selectedAttachmentOptions) {
  const FrozenSpatialPortIndex &ports = problem.ports();
  if (demand >= ports.portDemands().size() ||
      attachmentOption >= ports.attachmentOptions().size() ||
      selectedAttachmentOptions.size() != ports.portDemands().size())
    return invalid("attachment scoring domain is incomplete");
  const FrozenSpatialAttachmentOption &candidate =
      ports.attachmentOptions()[attachmentOption];
  if (candidate.progressBoundary !=
      ::loom::mapping::SpatialDurableProgressBoundaryKind::
          TemporalPeOperandQueue)
    return std::uint64_t{0};

  std::uint64_t pressure = 0;
  for (PnrIndex group : ports.operandPairingGroupsForDemand(demand)) {
    bool ingressAlreadySelected = false;
    for (PnrIndex peer : ports.operandPairingGroupMembers(group)) {
      if (peer == demand ||
          selectedAttachmentOptions[peer] == getInvalidPnrIndex())
        continue;
      const PnrIndex peerOption = selectedAttachmentOptions[peer];
      if (peerOption >= ports.attachmentOptions().size())
        return invalid("selected peer attachment is out of range");
      const FrozenSpatialAttachmentOption &selectedPeer =
          ports.attachmentOptions()[peerOption];
      ingressAlreadySelected |=
          selectedPeer.progressBoundary ==
              ::loom::mapping::SpatialDurableProgressBoundaryKind::
                  TemporalPeOperandQueue &&
          selectedPeer.endpoint == candidate.endpoint;
    }
    if (!ingressAlreadySelected)
      continue;
    if (pressure == std::numeric_limits<std::uint64_t>::max())
      return invalid("attachment pressure exceeds u64");
    ++pressure;
  }
  return pressure;
}

} // namespace loom::pnr::detail
