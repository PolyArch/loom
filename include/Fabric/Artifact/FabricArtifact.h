#ifndef LOOM_FABRIC_ARTIFACT_FABRICARTIFACT_H
#define LOOM_FABRIC_ARTIFACT_FABRICARTIFACT_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <utility>
#include <vector>

namespace fabric {
class ModuleOp;
class SystemOp;
} // namespace fabric

namespace loom::fabric {

/// The immutable result of publishing and independently importing one exact
/// Fabric root. This is an owner result over loom.fabric 1.0, not another
/// artifact family or a caller-constructible topology view.
class FinalizedFabricRoot final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  llvm::ArrayRef<FabricDirectDependency> directDependencies() const {
    return directDependencies_;
  }
  const FabricArtifactView &view() const { return view_; }

private:
  FinalizedFabricRoot(ArtifactRootReference reference,
                      CanonicalSemanticBytes canonicalBytes,
                      std::vector<FabricDirectDependency> directDependencies,
                      FabricArtifactView view)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)),
        directDependencies_(std::move(directDependencies)),
        view_(std::move(view)) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  std::vector<FabricDirectDependency> directDependencies_;
  FabricArtifactView view_;

  friend llvm::Expected<FinalizedFabricRoot>
  finalizeFabricRoot(::fabric::ModuleOp source, const ArtifactStore &store);
  friend llvm::Expected<FinalizedFabricRoot>
  finalizeFabricRoot(::fabric::SystemOp source,
                     llvm::ArrayRef<ArtifactRootReference> importedModules,
                     const ArtifactStore &store);
  friend llvm::Expected<FinalizedFabricRoot>
  importEntireFabricRoot(const ArtifactRootReference &reference,
                         const ArtifactStore &store);
};

/// Finalizes one complete Module authoring root and publishes its single
/// canonical loom.fabric object after strict independent reimport succeeds.
llvm::Expected<FinalizedFabricRoot>
finalizeFabricRoot(::fabric::ModuleOp source, const ArtifactStore &store);

/// Finalizes one complete System authoring root. Every supplied reference is
/// an ImportedModule dependency; fields inside the root own dependency use.
llvm::Expected<FinalizedFabricRoot>
finalizeFabricRoot(::fabric::SystemOp source,
                   llvm::ArrayRef<ArtifactRootReference> importedModules,
                   const ArtifactStore &store);

/// Resolves and strictly imports one exact published loom.fabric root.
llvm::Expected<FinalizedFabricRoot>
importEntireFabricRoot(const ArtifactRootReference &reference,
                       const ArtifactStore &store);

} // namespace loom::fabric

#endif // LOOM_FABRIC_ARTIFACT_FABRICARTIFACT_H
