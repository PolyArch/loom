#ifndef LOOM_PNR_PNRDERIVEDCONTEXT_H
#define LOOM_PNR_PNRDERIVEDCONTEXT_H

#include "Common/Artifact.h"
#include "Common/ComponentViewDigest.h"
#include "Common/MappingDebugLog.h"
#include "Fabric/Identity/FabricHandshake.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>

namespace loom::pnr {

namespace detail {
struct FabricDerivedContextStorage;
}

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
  const FabricDerivedContextStatistics &statistics() const;
  const ::loom::fabric::FabricHandshakeContext &handshakeContext() const;

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
    const ::loom::fabric::FabricPhysicalTimingProfileView &physicalTiming);

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
