#ifndef LOOM_DSE_SPATIALTOPOLOGYCANDIDATEGENERATOR_H
#define LOOM_DSE_SPATIALTOPOLOGYCANDIDATEGENERATOR_H

#include "DSE/CandidateGenerator.h"
#include "DSE/HardwareDecision.h"

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    spatialTopologyCandidateGeneratorKind(13);

class ResolvedSpatialTopologyRewriteConfigView final {
public:
  llvm::ArrayRef<SpatialTopologyDecision> decisions() const {
    return decisions_;
  }
  std::uint64_t maxChildrenPerParent() const { return maxChildrenPerParent_; }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedSpatialTopologyRewriteConfigView(
      std::vector<SpatialTopologyDecision> decisions,
      std::uint64_t maxChildrenPerParent,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : decisions_(std::move(decisions)),
        maxChildrenPerParent_(maxChildrenPerParent),
        canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  std::vector<SpatialTopologyDecision> decisions_;
  std::uint64_t maxChildrenPerParent_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedSpatialTopologyRewriteConfigView>
  resolveSpatialTopologyRewriteConfig(
      llvm::ArrayRef<SpatialTopologyDecisionDomain>, std::uint64_t);
  friend llvm::Expected<ResolvedSpatialTopologyRewriteConfigView>
  adoptResolvedSpatialTopologyRewriteConfigView(llvm::ArrayRef<std::uint8_t>,
                                                llvm::ArrayRef<std::uint8_t>,
                                                const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t> resolvedSpatialTopologyRewriteConfigSchemaBytes();
llvm::Expected<ResolvedSpatialTopologyRewriteConfigView>
resolveSpatialTopologyRewriteConfig(
    llvm::ArrayRef<SpatialTopologyDecisionDomain> domains,
    std::uint64_t maxChildrenPerParent);
llvm::Expected<ResolvedSpatialTopologyRewriteConfigView>
adoptResolvedSpatialTopologyRewriteConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const CandidateGeneratorDescriptor &
spatialTopologyCandidateGeneratorDescriptor();
llvm::Error registerSpatialTopologyCandidateGenerator();
llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindSpatialTopologyCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> parents);
llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveSpatialTopologyCandidateGeneratorBinding(
    const ResolvedSpatialTopologyRewriteConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_SPATIALTOPOLOGYCANDIDATEGENERATOR_H
