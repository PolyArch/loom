#ifndef LOOM_PNR_PNRDERIVEDCONTEXT_H
#define LOOM_PNR_PNRDERIVEDCONTEXT_H

#include "Common/Artifact.h"
#include "Common/ComponentViewDigest.h"
#include "Common/MappingDebugLog.h"
#include "Fabric/Identity/FabricHandshake.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <cstddef>
#include <memory>

namespace loom::fabric {
struct FabricTopologyQualityReport;
}

namespace loom::pnr {

namespace detail {
struct FabricDerivedContextStorage;
class PnrDerivedContextSessionState;
}

enum class PnrDerivedContextSessionMode : std::uint8_t {
  ReuseEnclosing,
  Isolated,
};

inline constexpr std::size_t defaultPnrDerivedContextSessionEntryLimit = 64;

struct PnrDerivedContextSessionStatistics final {
  std::uint64_t requests = 0;
  std::uint64_t cacheHits = 0;
  std::uint64_t cacheMisses = 0;
  std::uint64_t coalescedWaits = 0;
  std::uint64_t revalidationCount = 0;
  std::uint64_t uniqueConstructions = 0;
  std::uint64_t uncachedConstructions = 0;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t constructionNanosecondsSaved = 0;
  std::uint64_t deterministicWork = 0;
  std::uint64_t retainedBytes = 0;
  std::uint64_t retainedBytesReused = 0;
  std::uint64_t entryCount = 0;
  std::uint64_t entryLimit = 0;
};

struct DerivedContextCacheAccess final {
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
};

/// Installs one bounded cache for immutable PnR derived contexts. Attachments
/// carry the same session into in-process workers without sharing candidate,
/// transaction, scratch, random, or budget state.
class PnrDerivedContextSession final {
public:
  class Attachment final {
  public:
    Attachment() = default;
    explicit operator bool() const { return static_cast<bool>(state_); }

  private:
    explicit Attachment(
        std::shared_ptr<detail::PnrDerivedContextSessionState> state)
        : state_(std::move(state)) {}

    std::shared_ptr<detail::PnrDerivedContextSessionState> state_;
    friend class PnrDerivedContextSession;
  };

  explicit PnrDerivedContextSession(
      PnrDerivedContextSessionMode mode =
          PnrDerivedContextSessionMode::ReuseEnclosing,
      std::size_t entryLimit = defaultPnrDerivedContextSessionEntryLimit);
  explicit PnrDerivedContextSession(const Attachment &attachment);
  ~PnrDerivedContextSession();

  PnrDerivedContextSession(const PnrDerivedContextSession &) = delete;
  PnrDerivedContextSession &
  operator=(const PnrDerivedContextSession &) = delete;

  static Attachment currentAttachment();
  PnrDerivedContextSessionStatistics statistics() const;

private:
  std::shared_ptr<detail::PnrDerivedContextSessionState> active_;
  std::shared_ptr<detail::PnrDerivedContextSessionState> previous_;
};

struct DerivedContextConstructionStatistics final {
  std::uint64_t constructionCount = 0;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t retainedBytes = 0;
  std::uint64_t deterministicWork = 0;
};

struct FabricDerivedContextStatistics final {
  DerivedContextConstructionStatistics staticContext;
  DerivedContextConstructionStatistics timingContext;
  std::uint64_t resourceOwnerCount = 0;
  std::uint64_t endpointCount = 0;
  std::uint64_t traversalCount = 0;
  std::uint64_t routingArcCount = 0;
  std::uint64_t handshakeOwnerCount = 0;
  std::uint64_t handshakeStructuralTemplateCount = 0;
  std::uint64_t handshakeBindingInstanceCount = 0;
  std::uint64_t handshakeStructuralNodeCount = 0;
  std::uint64_t handshakeStructuralArcCount = 0;
  std::uint64_t handshakeStructuralFragmentCount = 0;
  std::uint64_t handshakeUnconditionalArcCount = 0;
  std::uint64_t handshakeNodeCount = 0;
  std::uint64_t handshakeArcCount = 0;
  std::uint64_t handshakeFragmentCount = 0;
};

/// One bounded invocation-local owner for the immutable Fabric-only and
/// Fabric-plus-timing projections. It has no persistent codec and never owns
/// mutable search state.
class FabricDerivedContextBundle final {
public:
  FabricDerivedContextBundle(FabricDerivedContextBundle &&) noexcept = default;
  FabricDerivedContextBundle &
  operator=(FabricDerivedContextBundle &&) noexcept = default;
  FabricDerivedContextBundle(const FabricDerivedContextBundle &) = delete;
  FabricDerivedContextBundle &
  operator=(const FabricDerivedContextBundle &) = delete;
  ~FabricDerivedContextBundle() = default;

  const ArtifactIdentity &fabricIdentity() const;
  const ComponentViewDigest::Storage &physicalTimingDigestBytes() const;
  llvm::ArrayRef<std::uint8_t> staticContextKey() const;
  llvm::ArrayRef<std::uint8_t> timingContextKey() const;
  const FabricDerivedContextStatistics &statistics() const;
  const ::loom::fabric::FabricHandshakeContext &handshakeContext() const;
  const ::loom::fabric::FabricTopologyQualityReport *
  topologyQualityDiagnostic() const;

private:
  explicit FabricDerivedContextBundle(
      std::shared_ptr<const detail::FabricDerivedContextStorage> storage)
      : storage_(std::move(storage)) {}

  std::shared_ptr<const detail::FabricDerivedContextStorage> storage_;

  friend class FrozenSpatialPnrProblemBuilder;
  friend llvm::Expected<FabricDerivedContextBundle>
  buildFabricDerivedContextBundle(
      const ::loom::fabric::FabricArtifactView &,
      const ::loom::fabric::FabricPhysicalTimingProfileView &);
  friend llvm::Error revalidateFabricDerivedContextBundle(
      const FabricDerivedContextBundle &,
      const ::loom::fabric::FabricArtifactView &,
      const ::loom::fabric::FabricPhysicalTimingProfileView &);
};

llvm::Expected<FabricDerivedContextBundle> buildFabricDerivedContextBundle(
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricPhysicalTimingProfileView &physicalTiming,
    DerivedContextCacheAccess *staticAccess = nullptr,
    DerivedContextCacheAccess *timingAccess = nullptr);

llvm::Error revalidateFabricDerivedContextBundle(
    const FabricDerivedContextBundle &bundle,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricPhysicalTimingProfileView &physicalTiming);

void emitFabricDerivedContextStatistics(
    const FabricDerivedContextBundle &bundle, mapping_debug::Stage stage,
    std::uint64_t staticHits, std::uint64_t staticMisses,
    std::uint64_t timingHits, std::uint64_t timingMisses);

} // namespace loom::pnr

#endif // LOOM_PNR_PNRDERIVEDCONTEXT_H
