#ifndef LOOM_HARDWARE_RTL_BLOCKGATENETLISTCOMPOSITION_H
#define LOOM_HARDWARE_RTL_BLOCKGATENETLISTCOMPOSITION_H

#include "Hardware/Implementation/RepresentationIndex.h"
#include "Hardware/RTL/BlockGateNetlist.h"

namespace loom::hardware::rtl {

struct BlockGateNetlistCompilationUnit final {
  ImplementationPayload payload;
  ArtifactRootReference contributor;
};

struct BlockGateNetlistChildBoundary final {
  std::string definition;
  std::uint64_t multiplicity;
  std::vector<RepresentationBoundaryPort> ports;
};

/// Transient composition projected from exact Source and accepted child
/// products. No caller-authored hierarchy or netlist text is admitted.
struct BlockGateNetlistComposition final {
  std::vector<BlockGateNetlistCompilationUnit> units;
  std::vector<RepresentationLocator> definitions;
  std::vector<BlockGateNetlistChildBoundary> children;
};

/// Verifies the complete direct-child Source relation, one coherent
/// platform/corner/library contract, and an unambiguous immutable payload
/// union. A shared blob is retained once with its canonical contributor.
llvm::Expected<BlockGateNetlistComposition> composeBlockGateNetlistChildren(
    const FinalizedRtlBlockSource &source,
    llvm::ArrayRef<FinalizedBlockGateNetlist> children,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_BLOCKGATENETLISTCOMPOSITION_H
