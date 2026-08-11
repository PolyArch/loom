#ifndef LOOM_DSE_SYSTEMCOMPOSITIONCANDIDATEGENERATOR_H
#define LOOM_DSE_SYSTEMCOMPOSITIONCANDIDATEGENERATOR_H

#include "DSE/CandidateGenerator.h"
#include "DSE/HardwareDecision.h"

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    systemCompositionCandidateGeneratorKind(15);

class ResolvedSystemCompositionRewriteConfigView final {
public:
  llvm::ArrayRef<SystemCompositionDecision> decisions() const {
    return decisions_;
  }
  std::uint64_t maxChildrenPerParent() const { return maxChildrenPerParent_; }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedSystemCompositionRewriteConfigView(
      std::vector<SystemCompositionDecision> decisions,
      std::uint64_t maxChildrenPerParent,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : decisions_(std::move(decisions)),
        maxChildrenPerParent_(maxChildrenPerParent),
        canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  std::vector<SystemCompositionDecision> decisions_;
  std::uint64_t maxChildrenPerParent_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedSystemCompositionRewriteConfigView>
  resolveSystemCompositionRewriteConfig(
      llvm::ArrayRef<SystemCompositionDecisionDomain>, std::uint64_t);
  friend llvm::Expected<ResolvedSystemCompositionRewriteConfigView>
  adoptResolvedSystemCompositionRewriteConfigView(llvm::ArrayRef<std::uint8_t>,
                                                  llvm::ArrayRef<std::uint8_t>,
                                                  const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t>
resolvedSystemCompositionRewriteConfigSchemaBytes();
llvm::Expected<ResolvedSystemCompositionRewriteConfigView>
resolveSystemCompositionRewriteConfig(
    llvm::ArrayRef<SystemCompositionDecisionDomain> domains,
    std::uint64_t maxChildrenPerParent);
llvm::Expected<ResolvedSystemCompositionRewriteConfigView>
adoptResolvedSystemCompositionRewriteConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const CandidateGeneratorDescriptor &
systemCompositionCandidateGeneratorDescriptor();
llvm::Error registerSystemCompositionCandidateGenerator();
llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindSystemCompositionCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> parents,
    llvm::ArrayRef<ArtifactRootReference> admissibleModules);
llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveSystemCompositionCandidateGeneratorBinding(
    const ResolvedSystemCompositionRewriteConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_SYSTEMCOMPOSITIONCANDIDATEGENERATOR_H
