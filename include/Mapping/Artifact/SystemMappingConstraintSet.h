#ifndef LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGCONSTRAINTSET_H
#define LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGCONSTRAINTSET_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/IR/MappingOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom::mapping {

llvm::Expected<CanonicalSemanticBytes>
writeCanonicalSystemConstraintAssembly(::mapping::ConstraintsSystemOp root);

class SystemMappingConstraintSetView final {
public:
  static llvm::Expected<SystemMappingConstraintSetView>
  import(const ArtifactIdentity &identity, ::mapping::ConstraintsSystemOp root,
         const ::dataflow::CanonicalDataflowProgramView &dataflow,
         const ::loom::fabric::FabricSystemRootView &fabric,
         const ArtifactStore &store);

  const ArtifactIdentity &identity() const { return identity_; }
  const ArtifactIdentity &dataflowIdentity() const { return dataflowIdentity_; }
  const ArtifactIdentity &fabricIdentity() const { return fabricIdentity_; }
  llvm::ArrayRef<::dataflow::RootThreadLaunchRef> rootThreadLaunches() const {
    return rootThreadLaunches_;
  }
  llvm::ArrayRef<ArtifactRootReference> spatialMappingReferences() const {
    return spatialMappingReferences_;
  }
  std::uint64_t clauseCount() const { return clauseCount_; }

private:
  SystemMappingConstraintSetView(
      ArtifactIdentity identity, ArtifactIdentity dataflowIdentity,
      ArtifactIdentity fabricIdentity,
      std::vector<::dataflow::RootThreadLaunchRef> rootThreadLaunches,
      std::vector<ArtifactRootReference> spatialMappingReferences,
      std::uint64_t clauseCount)
      : identity_(std::move(identity)),
        dataflowIdentity_(std::move(dataflowIdentity)),
        fabricIdentity_(std::move(fabricIdentity)),
        rootThreadLaunches_(std::move(rootThreadLaunches)),
        spatialMappingReferences_(std::move(spatialMappingReferences)),
        clauseCount_(clauseCount) {}

  ArtifactIdentity identity_;
  ArtifactIdentity dataflowIdentity_;
  ArtifactIdentity fabricIdentity_;
  std::vector<::dataflow::RootThreadLaunchRef> rootThreadLaunches_;
  std::vector<ArtifactRootReference> spatialMappingReferences_;
  std::uint64_t clauseCount_ = 0;
};

class FinalizedSystemMappingConstraintSet final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  const SystemMappingConstraintSetView &view() const { return view_; }

private:
  FinalizedSystemMappingConstraintSet(ArtifactRootReference reference,
                                      CanonicalSemanticBytes canonicalBytes,
                                      SystemMappingConstraintSetView view)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)), view_(std::move(view)) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  SystemMappingConstraintSetView view_;

  friend llvm::Expected<FinalizedSystemMappingConstraintSet>
  finalizeSystemMappingConstraintSet(::mapping::ConstraintsSystemOp source,
                                     const ArtifactStore &store);
  friend llvm::Expected<FinalizedSystemMappingConstraintSet>
  finalizeSystemMappingConstraintSet(
      ::mapping::ConstraintsSystemOp source,
      const ::dataflow::CanonicalDataflowProgramView &dataflow,
      const ::loom::fabric::FabricSystemRootView &fabric,
      const ArtifactStore &store);
  friend llvm::Expected<FinalizedSystemMappingConstraintSet>
  importSystemMappingConstraintSet(const ArtifactRootReference &reference,
                                   const ArtifactStore &store);
};

llvm::Expected<FinalizedSystemMappingConstraintSet>
finalizeSystemMappingConstraintSet(::mapping::ConstraintsSystemOp source,
                                   const ArtifactStore &store);

llvm::Expected<FinalizedSystemMappingConstraintSet>
finalizeSystemMappingConstraintSet(
    ::mapping::ConstraintsSystemOp source,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ArtifactStore &store);

/// Materializes and finalizes the unique empty clause sequence for one exact
/// D/F/root-launch closure. Absence of an Artifact is never interpreted as an
/// empty constraint set.
llvm::Expected<FinalizedSystemMappingConstraintSet>
finalizeEmptySystemMappingConstraintSet(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> rootThreadLaunches,
    const ArtifactStore &store);

llvm::Expected<FinalizedSystemMappingConstraintSet>
importSystemMappingConstraintSet(const ArtifactRootReference &reference,
                                 const ArtifactStore &store);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGCONSTRAINTSET_H
