#ifndef LOOM_FABRIC_ARTIFACT_FABRICMODULEROOTVIEW_H
#define LOOM_FABRIC_ARTIFACT_FABRICMODULEROOTVIEW_H

#include "Fabric/Identity/FabricRefImport.h"

namespace loom::fabric {

/// Zero-copy typed refinement of one complete immutable Module root view.
/// Root-owned domain relations remain stored by the underlying artifact view.
class FabricModuleRootView final {
public:
  const FabricArtifactView &artifact() const { return artifact_; }

  llvm::ArrayRef<FabricModuleDomainSlotRef> domainSlots() const;
  llvm::ArrayRef<ModuleDomainAssignment> domainAssignments() const;

private:
  explicit FabricModuleRootView(FabricArtifactView artifact)
      : artifact_(std::move(artifact)) {}

  FabricArtifactView artifact_;

  friend llvm::Expected<FabricModuleRootView>
  requireModuleRoot(const FabricArtifactView &view);
};

llvm::Expected<FabricModuleRootView>
requireModuleRoot(const FabricArtifactView &view);

} // namespace loom::fabric

#endif // LOOM_FABRIC_ARTIFACT_FABRICMODULEROOTVIEW_H
