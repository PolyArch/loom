#ifndef LOOM_HARDWARE_RTL_BLOCKGATENETLIST_H
#define LOOM_HARDWARE_RTL_BLOCKGATENETLIST_H

#include "Hardware/RTL/RtlBlockSource.h"
#include "ImplementationPlatform/TechnologyCorner.h"

namespace loom::hardware::rtl {

inline constexpr ArtifactSchemaDescriptor blockGateNetlistSchema{
    "loom.block_gate_netlist", SchemaVersion{1, 0}};

/// One complete logic-synthesis result for a reusable block. Tool execution
/// provenance belongs to the producing invocation. This Artifact owns the
/// source, technology, mapped cell library and exact structural representation.
struct BlockGateNetlistDraft final {
  ArtifactRootReference source;
  ArtifactRootReference implementationPlatform;
  platform::TechnologyCornerRef corner;
  std::string standardCellContract;
  ExternalFileFingerprint standardCellLibrary;
  ImplementationRepresentationRoot representation;
};

class FinalizedBlockGateNetlist final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const BlockGateNetlistDraft &netlist() const { return netlist_; }

private:
  FinalizedBlockGateNetlist(ArtifactRootReference reference,
                            BlockGateNetlistDraft netlist)
      : reference_(std::move(reference)), netlist_(std::move(netlist)) {}
  ArtifactRootReference reference_;
  BlockGateNetlistDraft netlist_;

  friend llvm::Expected<FinalizedBlockGateNetlist>
  importBlockGateNetlist(const ArtifactRootReference &,
                         const ExternalImplementationContractCatalog &,
                         const ArtifactStore &, const BlobStore &);
};

llvm::Expected<FinalizedBlockGateNetlist>
finalizeBlockGateNetlist(BlockGateNetlistDraft draft,
                         const ExternalImplementationContractCatalog &contracts,
                         const ArtifactStore &artifacts,
                         const BlobStore &blobs);

llvm::Expected<FinalizedBlockGateNetlist>
importBlockGateNetlist(const ArtifactRootReference &reference,
                       const ExternalImplementationContractCatalog &contracts,
                       const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_BLOCKGATENETLIST_H
