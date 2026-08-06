#ifndef LOOM_DSE_DATAFLOWREWRITECANDIDATEGENERATOR_H
#define LOOM_DSE_DATAFLOWREWRITECANDIDATEGENERATOR_H

#include "Common/ComponentViewDigest.h"
#include "DSE/CandidateGenerator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom {
struct ResolvedConfig;
}

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    dataflowRewriteCandidateGeneratorKind(4);

class ResolvedDataflowRewriteGeneratorConfigView final {
public:
  std::uint64_t scopeExpansionLimit() const { return scopeExpansionLimit_; }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedDataflowRewriteGeneratorConfigView(
      std::uint64_t scopeExpansionLimit,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : scopeExpansionLimit_(scopeExpansionLimit),
        canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  std::uint64_t scopeExpansionLimit_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedDataflowRewriteGeneratorConfigView>
  projectResolvedDataflowRewriteGeneratorConfigView(const ResolvedConfig &);
  friend llvm::Expected<ResolvedDataflowRewriteGeneratorConfigView>
  adoptResolvedDataflowRewriteGeneratorConfigView(llvm::ArrayRef<std::uint8_t>,
                                                  llvm::ArrayRef<std::uint8_t>,
                                                  const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t>
resolvedDataflowRewriteGeneratorConfigSchemaBytes();

llvm::Expected<ResolvedDataflowRewriteGeneratorConfigView>
projectResolvedDataflowRewriteGeneratorConfigView(const ResolvedConfig &config);

llvm::Expected<ResolvedDataflowRewriteGeneratorConfigView>
adoptResolvedDataflowRewriteGeneratorConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const CandidateGeneratorDescriptor &
dataflowRewriteCandidateGeneratorDescriptor();
const CandidateGeneratorOwnerLineagePayloadContract &
dataflowRewriteCandidateLineagePayloadContract();
llvm::Error registerDataflowRewriteCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindDataflowRewriteCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> canonicalDataflowPrograms,
    const ArtifactRootReference &fabric);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveDataflowRewriteCandidateGeneratorBinding(
    const ResolvedDataflowRewriteGeneratorConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_DATAFLOWREWRITECANDIDATEGENERATOR_H
