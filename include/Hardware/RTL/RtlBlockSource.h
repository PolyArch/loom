#ifndef LOOM_HARDWARE_RTL_RTLBLOCKSOURCE_H
#define LOOM_HARDWARE_RTL_RTLBLOCKSOURCE_H

#include "Hardware/RTL/RtlBlockClosure.h"
#include "Hardware/RTL/SpatialCoreImplementation.h"

namespace loom::hardware::rtl {

inline constexpr ArtifactSchemaDescriptor rtlBlockSourceSchema{
    "loom.rtl_block_source", SchemaVersion{1, 0}};

/// An occurrence-free complete RTL block, mechanically derived from one exact
/// portable HardwareImplementation. The parent occurrence is retained by the
/// deriving invocation, never folded into reusable source content identity.
class FinalizedRtlBlockSource final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const RtlBlockSourceProjection &projection() const { return projection_; }
  const RtlDomainPortNames &domainPorts() const { return domainPorts_; }
  const std::optional<fabric::ClockDomainContractRecord> &clock() const {
    return clock_;
  }
  const BlobDigest &closureIdentity() const { return closureIdentity_; }
  llvm::StringRef top() const {
    return projection_.graph.modules[projection_.graph.topModule].emittedName;
  }
  /// Derived from the System clock contract; no independently authored SDC.
  std::string generationConstraint() const;

private:
  FinalizedRtlBlockSource(
      ArtifactRootReference reference, RtlBlockSourceProjection projection,
      RtlDomainPortNames domainPorts,
      std::optional<fabric::ClockDomainContractRecord> clock,
      BlobDigest closureIdentity)
      : reference_(std::move(reference)), projection_(std::move(projection)),
        domainPorts_(std::move(domainPorts)), clock_(std::move(clock)),
        closureIdentity_(closureIdentity) {}

  ArtifactRootReference reference_;
  RtlBlockSourceProjection projection_;
  RtlDomainPortNames domainPorts_;
  std::optional<fabric::ClockDomainContractRecord> clock_;
  BlobDigest closureIdentity_;

  friend llvm::Expected<FinalizedRtlBlockSource>
  importRtlBlockSource(const ArtifactRootReference &, const ArtifactStore &,
                       const BlobStore &);
};

/// Selects an exact post-lowering definition ordinal from the canonical
/// projection of implementation. Replays that implementation before deriving
/// source; a supplied projection is never accepted as independent authority.
llvm::Expected<FinalizedRtlBlockSource> finalizePortableRtlBlockSource(
    const FinalizedConfigurationABI &configurationAbi,
    const FinalizedHardwareImplementation &implementation,
    std::size_t definition, const ArtifactStore &artifacts,
    const BlobStore &blobs);

/// Replays the parent association without publishing anything. A
/// self-consistent source Artifact alone never proves it is the selected block
/// of a parent.
llvm::Error verifyPortableRtlBlockSourceDerivation(
    const FinalizedRtlBlockSource &source,
    const FinalizedConfigurationABI &configurationAbi,
    const FinalizedHardwareImplementation &implementation,
    std::size_t definition, const BlobStore &blobs);

/// Checks stored framing, all definition source digests and exact dependency
/// references, canonical content names/order, root interface, and domain
/// geometry. This validates the reusable block itself; associating it with a
/// parent occurrence requires the parent's separate mechanical derivation.
llvm::Expected<FinalizedRtlBlockSource>
importRtlBlockSource(const ArtifactRootReference &reference,
                     const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_RTLBLOCKSOURCE_H
