#ifndef LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGARTIFACT_H
#define LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGARTIFACT_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/SystemPresburger.h"
#include "Mapping/IR/MappingOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <utility>
#include <vector>

namespace loom::mapping {

template <typename Target> struct SystemPresburgerClauseView final {
  std::vector<SystemPresburgerCell> cells;
  Target target;
};

struct SystemThreadExecutionBindingView final {
  ::dataflow::RootThreadLaunchRef key;
  std::vector<SystemPresburgerClauseView<::loom::fabric::AccCoreOccurrenceRef>>
      clauses;
  std::optional<::loom::fabric::AccCoreOccurrenceRef> defaultTarget;
};

struct SystemGraphExecutionBindingView final {
  ::dataflow::RootedGraphLaunchRef key;
  std::vector<SystemPresburgerClauseView<ArtifactRootReference>> clauses;
  std::optional<ArtifactRootReference> defaultTarget;
};

/// Strictly reconstructed execution portion of one mapping.system root.
/// Service and ResourceUse closure are intentionally not represented here.
class SystemExecutionBindingView final {
public:
  llvm::ArrayRef<::dataflow::RootThreadLaunchRef> rootThreadLaunches() const {
    return rootThreadLaunches_;
  }
  llvm::ArrayRef<ArtifactRootReference> spatialMappingImports() const {
    return spatialMappingImports_;
  }
  llvm::ArrayRef<SystemThreadExecutionBindingView> threadBindings() const {
    return threadBindings_;
  }
  llvm::ArrayRef<SystemGraphExecutionBindingView> graphBindings() const {
    return graphBindings_;
  }

private:
  SystemExecutionBindingView(
      std::vector<::dataflow::RootThreadLaunchRef> rootThreadLaunches,
      std::vector<ArtifactRootReference> spatialMappingImports,
      std::vector<SystemThreadExecutionBindingView> threadBindings,
      std::vector<SystemGraphExecutionBindingView> graphBindings)
      : rootThreadLaunches_(std::move(rootThreadLaunches)),
        spatialMappingImports_(std::move(spatialMappingImports)),
        threadBindings_(std::move(threadBindings)),
        graphBindings_(std::move(graphBindings)) {}

  std::vector<::dataflow::RootThreadLaunchRef> rootThreadLaunches_;
  std::vector<ArtifactRootReference> spatialMappingImports_;
  std::vector<SystemThreadExecutionBindingView> threadBindings_;
  std::vector<SystemGraphExecutionBindingView> graphBindings_;

  friend llvm::Expected<SystemExecutionBindingView>
  strictImportSystemExecutionBindings(
      const CanonicalSemanticBytes &,
      const ::dataflow::CanonicalDataflowProgramView &,
      const ::loom::fabric::FabricSystemRootView &, const ArtifactStore &);
};

llvm::Expected<CanonicalSemanticBytes>
writeCanonicalSystemMappingAssembly(::mapping::SystemOp root);

/// Strictly parses, semantically validates, canonically re-emits, and adopts
/// the execution records. It never publishes an incomplete SystemMapping.
llvm::Expected<SystemExecutionBindingView> strictImportSystemExecutionBindings(
    const CanonicalSemanticBytes &bytes,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ArtifactStore &store);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SYSTEMMAPPINGARTIFACT_H
