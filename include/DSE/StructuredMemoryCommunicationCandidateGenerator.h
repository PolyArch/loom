#ifndef LOOM_DSE_STRUCTUREDMEMORYCOMMUNICATIONCANDIDATEGENERATOR_H
#define LOOM_DSE_STRUCTUREDMEMORYCOMMUNICATIONCANDIDATEGENERATOR_H

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
    structuredMemoryCommunicationCandidateGeneratorKind(5);

class ResolvedStructuredMemoryCommunicationGeneratorConfigView final {
public:
  std::uint32_t scopeExpansionLimit() const { return scopeExpansionLimit_; }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedStructuredMemoryCommunicationGeneratorConfigView(
      std::uint32_t scopeExpansionLimit,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : scopeExpansionLimit_(scopeExpansionLimit),
        canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  std::uint32_t scopeExpansionLimit_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<
      ResolvedStructuredMemoryCommunicationGeneratorConfigView>
  projectResolvedStructuredMemoryCommunicationGeneratorConfigView(
      const ResolvedConfig &);
  friend llvm::Expected<
      ResolvedStructuredMemoryCommunicationGeneratorConfigView>
  adoptResolvedStructuredMemoryCommunicationGeneratorConfigView(
      llvm::ArrayRef<std::uint8_t>, llvm::ArrayRef<std::uint8_t>,
      const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t>
resolvedStructuredMemoryCommunicationGeneratorConfigSchemaBytes();

llvm::Expected<ResolvedStructuredMemoryCommunicationGeneratorConfigView>
projectResolvedStructuredMemoryCommunicationGeneratorConfigView(
    const ResolvedConfig &config);

llvm::Expected<ResolvedStructuredMemoryCommunicationGeneratorConfigView>
adoptResolvedStructuredMemoryCommunicationGeneratorConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const CandidateGeneratorDescriptor &
structuredMemoryCommunicationCandidateGeneratorDescriptor();
llvm::Error registerStructuredMemoryCommunicationCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindStructuredMemoryCommunicationCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveStructuredMemoryCommunicationCandidateGeneratorBinding(
    const ResolvedStructuredMemoryCommunicationGeneratorConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_STRUCTUREDMEMORYCOMMUNICATIONCANDIDATEGENERATOR_H
