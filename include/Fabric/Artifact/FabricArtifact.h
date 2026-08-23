#ifndef LOOM_FABRIC_ARTIFACT_FABRICARTIFACT_H
#define LOOM_FABRIC_ARTIFACT_FABRICARTIFACT_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Common/InvocationDiagnosticLog.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
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

struct FinalizedFabricSystemProjection;
struct FinalizedFabricModuleProjection;

namespace detail {
class FabricArtifactImportSessionState;
llvm::Expected<FinalizedFabricSystemProjection> finalizeFabricSystem(
    ::fabric::SystemOp source,
    llvm::ArrayRef<ArtifactRootReference> importedModules,
    const ArtifactStore &store, bool captureCorrespondence);
llvm::Expected<FinalizedFabricModuleProjection> finalizeFabricModule(
    ::fabric::ModuleOp source,
    const ::fabric::ModuleDomainAuthoringRelation &domainRelation,
    const ArtifactStore &store, bool captureCorrespondence);
}

enum class FabricArtifactImportSessionMode : std::uint8_t {
  ReuseEnclosing,
  Isolated,
};

inline constexpr std::size_t defaultFabricArtifactImportSessionEntryLimit = 64;

enum class FabricArtifactImportVerificationDomain : std::uint8_t {
  SourceInvocation,
  IndependentReplay,
};

struct FabricArtifactImportSessionStatistics final {
  std::uint64_t importRequests = 0;
  std::uint64_t uniqueConstructions = 0;
  std::uint64_t uncachedConstructions = 0;
  std::uint64_t cacheHits = 0;
  std::uint64_t cacheMisses = 0;
  std::uint64_t coalescedWaits = 0;
  std::uint64_t revalidationCount = 0;
  std::uint64_t revalidatedBytes = 0;
  std::uint64_t bytesRead = 0;
  std::uint64_t bytesCopied = 0;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t constructionNanosecondsSaved = 0;
  std::uint64_t deterministicWork = 0;
  std::uint64_t retainedPayloadBytes = 0;
  std::uint64_t retainedPayloadBytesReused = 0;
  std::uint64_t entryCount = 0;
  std::uint64_t entryLimit = 0;
};

/// Installs one bounded cache of strictly imported immutable Fabric roots for
/// a synchronous invocation. Isolated scopes never reuse enclosing results and
/// are used by independent replay verifiers.
class FabricArtifactImportSession final {
public:
  class Attachment final {
  public:
    Attachment() = default;
    explicit operator bool() const { return static_cast<bool>(state_); }

  private:
    explicit Attachment(
        std::shared_ptr<detail::FabricArtifactImportSessionState> state)
        : state_(std::move(state)) {}

    std::shared_ptr<detail::FabricArtifactImportSessionState> state_;
    friend class FabricArtifactImportSession;
  };

  explicit FabricArtifactImportSession(
      FabricArtifactImportSessionMode mode =
          FabricArtifactImportSessionMode::ReuseEnclosing,
      std::size_t entryLimit = defaultFabricArtifactImportSessionEntryLimit);
  explicit FabricArtifactImportSession(const Attachment &attachment);
  ~FabricArtifactImportSession();

  FabricArtifactImportSession(const FabricArtifactImportSession &) = delete;
  FabricArtifactImportSession &
  operator=(const FabricArtifactImportSession &) = delete;

  static Attachment currentAttachment();
  Attachment attachment() const { return Attachment(active_); }
  FabricArtifactImportSessionStatistics statistics() const;

private:
  std::shared_ptr<detail::FabricArtifactImportSessionState> active_;
  std::shared_ptr<detail::FabricArtifactImportSessionState> previous_;
};

void emitFabricArtifactImportSessionStatistics(
    FabricArtifactImportVerificationDomain domain,
    InvocationDiagnosticStage stage,
    const FabricArtifactImportSessionStatistics &statistics);

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
  friend llvm::Expected<FinalizedFabricModuleProjection>
  finalizeFabricModuleWithCorrespondence(
      ::fabric::ModuleOp source,
      const ::fabric::ModuleDomainAuthoringRelation &domainRelation,
      const ArtifactStore &store);
  friend llvm::Expected<FinalizedFabricRoot>
  finalizeFabricRoot(::fabric::SystemOp source,
                     llvm::ArrayRef<ArtifactRootReference> importedModules,
                     const ArtifactStore &store);
  friend llvm::Expected<FinalizedFabricSystemProjection>
  finalizeFabricSystemWithCorrespondence(
      ::fabric::SystemOp source,
      llvm::ArrayRef<ArtifactRootReference> importedModules,
      const ArtifactStore &store);
  friend llvm::Expected<FinalizedFabricSystemProjection>
  detail::finalizeFabricSystem(
      ::fabric::SystemOp source,
      llvm::ArrayRef<ArtifactRootReference> importedModules,
      const ArtifactStore &store, bool captureCorrespondence);
  friend llvm::Expected<FinalizedFabricModuleProjection>
  detail::finalizeFabricModule(
      ::fabric::ModuleOp source,
      const ::fabric::ModuleDomainAuthoringRelation &domainRelation,
      const ArtifactStore &store, bool captureCorrespondence);
  friend llvm::Expected<FinalizedFabricRoot>
  importEntireFabricRoot(const ArtifactRootReference &reference,
                         const ArtifactStore &store);
};

struct FabricModuleEntityReference final {
  FabricEntityKind kind = FabricEntityKind::FabricPeOccurrence;
  FabricEntityId id = 0;
  std::uint64_t occurrenceOrdinal = 0;

  friend bool operator==(const FabricModuleEntityReference &lhs,
                         const FabricModuleEntityReference &rhs) {
    return lhs.kind == rhs.kind && lhs.id == rhs.id &&
           lhs.occurrenceOrdinal == rhs.occurrenceOrdinal;
  }
};

struct FabricModuleEntityCorrespondence final {
  FabricModuleEntityReference source;
  FabricModuleEntityReference target;

  friend bool operator==(const FabricModuleEntityCorrespondence &lhs,
                         const FabricModuleEntityCorrespondence &rhs) {
    return lhs.source == rhs.source && lhs.target == rhs.target;
  }
};

/// Finalizer-owned transient correspondence for every Module-local occurrence
/// entity in one derived root. Source references belong to the authored parent
/// namespace and targets belong to `root`. The relation is produced by the
/// same canonical-labeling transaction and is never serialized as Fabric.
struct FinalizedFabricModuleProjection final {
  FinalizedFabricRoot root;
  std::vector<FabricModuleEntityCorrespondence> entities;
};

struct FabricSystemEntityReference final {
  FabricEntityKind kind = FabricEntityKind::FabricModuleTemplate;
  FabricEntityId id = 0;

  friend bool operator==(const FabricSystemEntityReference &lhs,
                         const FabricSystemEntityReference &rhs) {
    return lhs.kind == rhs.kind && lhs.id == rhs.id;
  }
  friend bool operator!=(const FabricSystemEntityReference &lhs,
                         const FabricSystemEntityReference &rhs) {
    return !(lhs == rhs);
  }
};

struct FabricSystemEntityCorrespondence final {
  FabricSystemEntityReference source;
  FabricSystemEntityReference target;

  friend bool operator==(const FabricSystemEntityCorrespondence &lhs,
                         const FabricSystemEntityCorrespondence &rhs) {
    return lhs.source == rhs.source && lhs.target == rhs.target;
  }
};

struct FabricSystemTransferPatternCorrespondence final {
  FabricTransferPatternRef source;
  FabricTransferPatternRef target;

  friend bool operator==(
      const FabricSystemTransferPatternCorrespondence &lhs,
      const FabricSystemTransferPatternCorrespondence &rhs) {
    return lhs.source == rhs.source && lhs.target == rhs.target;
  }
};

/// Finalizer-owned ephemeral correspondence for every entity and transfer
/// pattern in one System authoring root. Source references belong to the
/// authoring root and targets belong to `root`. Fabric does not serialize this
/// transformation lineage.
struct FinalizedFabricSystemProjection final {
  FinalizedFabricRoot root;
  std::vector<FabricSystemEntityCorrespondence> entities;
  std::vector<FabricSystemTransferPatternCorrespondence> transferPatterns;
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

llvm::Expected<FinalizedFabricModuleProjection>
finalizeFabricModuleWithCorrespondence(
    ::fabric::ModuleOp source,
    const ::fabric::ModuleDomainAuthoringRelation &domainRelation,
    const ArtifactStore &store);

/// Finalizes one complete System authoring root. Every supplied reference is
/// an ImportedModule dependency; fields inside the root own dependency use.
llvm::Expected<FinalizedFabricRoot>
finalizeFabricRoot(::fabric::SystemOp source,
                   llvm::ArrayRef<ArtifactRootReference> importedModules,
                   const ArtifactStore &store);

/// Finalizes one System and publishes the complete transient correspondence
/// produced by the same canonical-labeling transaction.
llvm::Expected<FinalizedFabricSystemProjection>
finalizeFabricSystemWithCorrespondence(
    ::fabric::SystemOp source,
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
