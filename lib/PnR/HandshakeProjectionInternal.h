#ifndef LOOM_LIB_PNR_HANDSHAKEPROJECTIONINTERNAL_H
#define LOOM_LIB_PNR_HANDSHAKEPROJECTIONINTERNAL_H

#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom::pnr::detail {

struct HandshakeNodeIdentity final {
  std::optional<::loom::fabric::HandshakeSignalRef> boundarySignal;
  PnrIndex owner = 0;
  std::uint32_t localNode = 0;

  friend bool operator==(const HandshakeNodeIdentity &lhs,
                         const HandshakeNodeIdentity &rhs) {
    return lhs.boundarySignal == rhs.boundarySignal && lhs.owner == rhs.owner &&
           lhs.localNode == rhs.localNode;
  }
};

struct HandshakeArcIdentity final {
  HandshakeNodeIdentity source;
  HandshakeNodeIdentity destination;

  friend bool operator==(const HandshakeArcIdentity &lhs,
                         const HandshakeArcIdentity &rhs) {
    return lhs.source == rhs.source && lhs.destination == rhs.destination;
  }
};

struct RebuiltHandshakeSelection final {
  std::vector<PnrIndex> fragmentRefcounts;
  std::vector<PnrIndex> activeFragments;
  std::vector<PnrIndex> traversalRefcounts;
  std::vector<PnrIndex> allGroupSelectedWitnessCounts;
};

/// Writes the canonical identity key into a caller-owned buffer. Hot callers
/// resolve one node per handshake arc per accepted move, so they reuse a single
/// buffer instead of allocating a key per lookup.
void assignNodeKey(const HandshakeNodeIdentity &identity, std::string &key);

std::string nodeKey(const HandshakeNodeIdentity &identity);

llvm::Expected<HandshakeNodeIdentity>
nodeIdentity(PnrIndex owner, const ::loom::fabric::HandshakeOwnerModel &model,
             std::uint32_t localNode);

llvm::Expected<HandshakeArcIdentity>
arcIdentity(PnrIndex owner, const ::loom::fabric::HandshakeOwnerModel &model,
            const ::loom::fabric::HandshakeOwnerArc &arc);

llvm::Error
rebuildHandshakeSelectionInto(const FrozenSpatialHandshakeIndex &index,
                              llvm::ArrayRef<PnrIndex> selectedFragments,
                              llvm::ArrayRef<PnrIndex> traversalUses,
                              RebuiltHandshakeSelection &result);

llvm::Expected<RebuiltHandshakeSelection>
rebuildHandshakeSelection(const FrozenSpatialHandshakeIndex &index,
                          llvm::ArrayRef<PnrIndex> selectedFragments,
                          llvm::ArrayRef<PnrIndex> traversalUses);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_HANDSHAKEPROJECTIONINTERNAL_H
