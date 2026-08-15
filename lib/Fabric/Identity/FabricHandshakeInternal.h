#ifndef LOOM_FABRIC_IDENTITY_FABRICHANDSHAKEINTERNAL_H
#define LOOM_FABRIC_IDENTITY_FABRICHANDSHAKEINTERNAL_H

#include "Fabric/Identity/FabricHandshake.h"

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace loom::fabric::detail {

class HandshakeOwnerModelBuilder final {
public:
  explicit HandshakeOwnerModelBuilder(FabricHandshakeOwner owner);

  std::uint32_t boundarySignal(HandshakeSignalRef signal);
  std::uint32_t junction(llvm::ArrayRef<std::uint8_t> ownerLocalKey);
  void addFragment(HandshakeFragmentSelector selector,
                   std::vector<std::pair<std::uint32_t, std::uint32_t>> arcs);
  llvm::Expected<HandshakeOwnerModel> finish();

private:
  struct PendingFragment final {
    HandshakeFragmentSelector selector;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> arcs;
  };

  HandshakeOwnerModel model_;
  std::map<std::vector<std::uint8_t>, std::uint32_t> nodes_;
  std::vector<PendingFragment> pending_;
};

std::vector<std::uint8_t> handshakeSignalKey(const HandshakeSignalRef &signal);

llvm::Expected<HandshakeOwnerModel>
compileFuHandshakeModel(const FabricArtifactView &view,
                        FabricFuOccurrenceRef occurrence);

} // namespace loom::fabric::detail

#endif // LOOM_FABRIC_IDENTITY_FABRICHANDSHAKEINTERNAL_H
