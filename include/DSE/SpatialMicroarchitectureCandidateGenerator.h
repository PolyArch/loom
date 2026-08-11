#ifndef LOOM_DSE_SPATIALMICROARCHITECTURECANDIDATEGENERATOR_H
#define LOOM_DSE_SPATIALMICROARCHITECTURECANDIDATEGENERATOR_H

#include "DSE/CandidateGenerator.h"
#include "DSE/HardwareDecision.h"

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    spatialMicroarchitectureCandidateGeneratorKind(14);

class ResolvedSpatialMicroarchitectureRewriteConfigView final {
public:
  llvm::ArrayRef<SpatialMicroarchitectureDecision> decisions() const {
    return decisions_;
  }
  std::uint64_t maxChildrenPerParent() const { return maxChildrenPerParent_; }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedSpatialMicroarchitectureRewriteConfigView(
      std::vector<SpatialMicroarchitectureDecision> decisions,
      std::uint64_t maxChildrenPerParent,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : decisions_(std::move(decisions)),
        maxChildrenPerParent_(maxChildrenPerParent),
        canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  std::vector<SpatialMicroarchitectureDecision> decisions_;
  std::uint64_t maxChildrenPerParent_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedSpatialMicroarchitectureRewriteConfigView>
  resolveSpatialMicroarchitectureRewriteConfig(
      llvm::ArrayRef<SpatialMicroarchitectureDecisionDomain>, std::uint64_t);
  friend llvm::Expected<ResolvedSpatialMicroarchitectureRewriteConfigView>
  adoptResolvedSpatialMicroarchitectureRewriteConfigView(
      llvm::ArrayRef<std::uint8_t>, llvm::ArrayRef<std::uint8_t>,
      const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t>
resolvedSpatialMicroarchitectureRewriteConfigSchemaBytes();
llvm::Expected<ResolvedSpatialMicroarchitectureRewriteConfigView>
resolveSpatialMicroarchitectureRewriteConfig(
    llvm::ArrayRef<SpatialMicroarchitectureDecisionDomain> domains,
    std::uint64_t maxChildrenPerParent);
llvm::Expected<ResolvedSpatialMicroarchitectureRewriteConfigView>
adoptResolvedSpatialMicroarchitectureRewriteConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const CandidateGeneratorDescriptor &
spatialMicroarchitectureCandidateGeneratorDescriptor();
llvm::Error registerSpatialMicroarchitectureCandidateGenerator();
llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindSpatialMicroarchitectureCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> parents);
llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveSpatialMicroarchitectureCandidateGeneratorBinding(
    const ResolvedSpatialMicroarchitectureRewriteConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_SPATIALMICROARCHITECTURECANDIDATEGENERATOR_H
