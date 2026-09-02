#ifndef LOOM_LIB_PNR_SPATIALCANDIDATESTATEINTERNAL_H
#define LOOM_LIB_PNR_SPATIALCANDIDATESTATEINTERNAL_H

#include "PnR/SpatialCandidateState.h"

#include "llvm/ADT/Twine.h"

#include <optional>
#include <system_error>

namespace loom::pnr::detail {

inline llvm::Error candidateError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial candidate state: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

inline bool rangeContains(PnrIndex offset, PnrIndex count, PnrIndex value) {
  return value >= offset && value - offset < count;
}

inline llvm::ArrayRef<PnrIndex>
computePlacementFragments(const FrozenSpatialHandshakeIndex &handshake,
                          PnrIndex placement) {
  const auto offsets = handshake.computePlacementFragmentOffsets();
  return handshake.computePlacementFragments().slice(
      offsets[placement], offsets[placement + 1] - offsets[placement]);
}

inline llvm::ArrayRef<PnrIndex>
memoryPlanFragments(const FrozenSpatialHandshakeIndex &handshake,
                    PnrIndex plan) {
  const auto record = handshake.memoryOperationPlans()[plan];
  return handshake.memoryPlanFragments().slice(record.fragmentOffset,
                                               record.fragmentCount);
}

inline llvm::ArrayRef<PnrIndex>
localTransferFragments(const FrozenSpatialHandshakeIndex &handshake,
                       PnrIndex option) {
  const auto offsets = handshake.localTransferFragmentOffsets();
  return handshake.localTransferFragments().slice(
      offsets[option], offsets[option + 1] - offsets[option]);
}

inline std::optional<PnrIndex>
attachmentTraversal(const FrozenSpatialPortIndex &ports, PnrIndex option) {
  return ports.attachmentOptions()[option].localTraversal;
}

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALCANDIDATESTATEINTERNAL_H
