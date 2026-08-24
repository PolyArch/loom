#ifndef LOOM_FABRIC_ARTIFACT_FABRICSYSTEMREFERENCEREMAPPER_H
#define LOOM_FABRIC_ARTIFACT_FABRICSYSTEMREFERENCEREMAPPER_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <map>
#include <utility>
#include <vector>

namespace loom::fabric {

/// Rewrites persistent Fabric references through one exact System
/// canonicalization lineage. It changes only entity and transfer-pattern
/// identities; structural roles, inventory ordinals, and traversal fields are
/// preserved by the references' own field declarations.
class FabricSystemReferenceRemapper final {
public:
  static llvm::Expected<FabricSystemReferenceRemapper>
  get(llvm::ArrayRef<FabricSystemEntityCorrespondence> entities,
      llvm::ArrayRef<FabricSystemTransferPatternCorrespondence>
          transferPatterns);

  template <FabricEntityKind Kind>
  llvm::Expected<FabricTypedEntityRef<Kind>>
  remap(const FabricTypedEntityRef<Kind> &reference) const {
    auto found = entities_.find({Kind, reference.id()});
    if (found == entities_.end())
      return missing("entity");
    return FabricTypedEntityRef<Kind>(found->second);
  }

  template <typename Ref>
  llvm::Expected<Ref> remap(const Ref &reference) const;

  template <FabricInventoryKind Inventory>
  llvm::Expected<FabricOwnerProjection<Inventory>>
  remap(const FabricOwnerProjection<Inventory> &reference) const {
    auto catalog = remap(reference.catalog());
    if (!catalog)
      return catalog.takeError();
    return FabricOwnerProjection<Inventory>(std::move(*catalog));
  }

  template <FabricRefinementKind Refinement, typename Underlying>
  llvm::Expected<FabricRefinedRef<Refinement, Underlying>>
  remap(const FabricRefinedRef<Refinement, Underlying> &reference) const {
    auto underlying = remap(reference.underlying());
    if (!underlying)
      return underlying.takeError();
    return FabricRefinedRef<Refinement, Underlying>(std::move(*underlying));
  }

  llvm::Expected<FabricTransportEndpointOwnerRef>
  remap(const FabricTransportEndpointOwnerRef &reference) const;
  llvm::Expected<FabricMemoryEndpointOwnerRef>
  remap(const FabricMemoryEndpointOwnerRef &reference) const;
  llvm::Expected<FabricInventoryOwnerRef>
  remap(const FabricInventoryOwnerRef &reference) const;
  llvm::Expected<FabricHardwareDomainMemberRef>
  remap(const FabricHardwareDomainMemberRef &reference) const;
  llvm::Expected<FabricMemoryServiceRef>
  remap(const FabricMemoryServiceRef &reference) const;
  llvm::Expected<FabricTransferPatternRef>
  remap(const FabricTransferPatternRef &reference) const;
  llvm::Expected<FabricPhysicalTraversalRef>
  remap(const FabricPhysicalTraversalRef &reference) const;

private:
  using EntityKey = std::pair<FabricEntityKind, FabricEntityId>;

  FabricSystemReferenceRemapper(
      std::map<EntityKey, FabricEntityId> entities,
      std::map<std::vector<std::uint8_t>, FabricTransferPatternRef>
          transferPatterns)
      : entities_(std::move(entities)),
        transferPatterns_(std::move(transferPatterns)) {}

  static llvm::Error missing(llvm::StringRef kind);

  std::map<EntityKey, FabricEntityId> entities_;
  std::map<std::vector<std::uint8_t>, FabricTransferPatternRef>
      transferPatterns_;
};

namespace detail {
struct FabricSystemReferenceRemapVisitor final {
  const FabricSystemReferenceRemapper &remapper;
  llvm::Error error = llvm::Error::success();

  template <typename Enum> void tag(Enum &) {}
  void ordinal(FabricOrdinal &) {}

  template <typename Ref> void ref(Ref &reference) {
    if (error)
      return;
    auto mapped = remapper.remap(reference);
    if (!mapped)
      error = mapped.takeError();
    else
      reference = std::move(*mapped);
  }
};
} // namespace detail

template <typename Ref>
llvm::Expected<Ref>
FabricSystemReferenceRemapper::remap(const Ref &reference) const {
  Ref result = reference;
  detail::FabricSystemReferenceRemapVisitor visitor{*this};
  Ref::visitFields(result, visitor);
  if (visitor.error)
    return std::move(visitor.error);
  return result;
}

} // namespace loom::fabric

#endif // LOOM_FABRIC_ARTIFACT_FABRICSYSTEMREFERENCEREMAPPER_H
