#ifndef LOOM_FABRIC_ARTIFACT_FABRICARTIFACT_H
#define LOOM_FABRIC_ARTIFACT_FABRICARTIFACT_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace fabric {
class ModuleOp;
class ModuleDomainAuthoringRelation;
class SystemOp;
} // namespace fabric

namespace loom::fabric {

namespace detail {
class FabricArtifactImportSessionState;
}

enum class FabricArtifactImportSessionMode : std::uint8_t {
  ReuseEnclosing,
  Isolated,
};

struct FabricArtifactImportSessionStatistics final {
  std::uint64_t importRequests = 0;
  std::uint64_t uniqueConstructions = 0;
  std::uint64_t cacheHits = 0;
  std::uint64_t cacheMisses = 0;
  std::uint64_t bytesRead = 0;
  std::uint64_t bytesCopied = 0;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t deterministicWork = 0;
  std::uint64_t retainedPayloadBytes = 0;
  std::uint64_t entryCount = 0;
};

/// Installs one bounded cache of strictly imported immutable Fabric roots for
/// a synchronous invocation. Isolated scopes never reuse enclosing results and
/// are used by independent replay verifiers.
class FabricArtifactImportSession final {
public:
  explicit FabricArtifactImportSession(
      FabricArtifactImportSessionMode mode =
          FabricArtifactImportSessionMode::ReuseEnclosing);
  ~FabricArtifactImportSession();

  FabricArtifactImportSession(const FabricArtifactImportSession &) = delete;
  FabricArtifactImportSession &
  operator=(const FabricArtifactImportSession &) = delete;

  FabricArtifactImportSessionStatistics statistics() const;

private:
  std::unique_ptr<detail::FabricArtifactImportSessionState> owned_;
  detail::FabricArtifactImportSessionState *active_ = nullptr;
  detail::FabricArtifactImportSessionState *previous_ = nullptr;
};

/// The immutable result of publishing and independently importing one exact
/// Fabric root. This is an owner result over loom.fabric 3.x, not another
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
  friend llvm::Expected<FinalizedFabricRoot> finalizeFabricRoot(
      ::fabric::ModuleOp source,
      const ::fabric::ModuleDomainAuthoringRelation &domainRelation,
      const ArtifactStore &store);
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

/// Finalizes one Module together with its sole pre-canonical domain authoring
/// relation. The relation is consumed only for canonical materialization.
llvm::Expected<FinalizedFabricRoot> finalizeFabricRoot(
    ::fabric::ModuleOp source,
    const ::fabric::ModuleDomainAuthoringRelation &domainRelation,
    const ArtifactStore &store);

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

/// Writes the canonical MLIR payload of one finalized Fabric root as a
/// human-readable textual projection. Artifact identity remains owned by the
/// canonical bytecode and envelope stored in `root`.
llvm::Error writeFabricMlir(const FinalizedFabricRoot &root,
                            llvm::raw_ostream &output);

} // namespace loom::fabric

#endif // LOOM_FABRIC_ARTIFACT_FABRICARTIFACT_H
