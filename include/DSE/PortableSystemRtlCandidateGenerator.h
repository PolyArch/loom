#ifndef LOOM_DSE_PORTABLESYSTEMRTLCANDIDATEGENERATOR_H
#define LOOM_DSE_PORTABLESYSTEMRTLCANDIDATEGENERATOR_H

#include "DSE/CandidateGenerator.h"

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    portableSystemRtlCandidateGeneratorKind(16);

class ResolvedPortableSystemRtlConfigView final {
public:
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedPortableSystemRtlConfigView(std::vector<std::uint8_t> canonicalBytes,
                                      ComponentViewDigest digest)
      : canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedPortableSystemRtlConfigView>
  resolvePortableSystemRtlConfig();
  friend llvm::Expected<ResolvedPortableSystemRtlConfigView>
  adoptResolvedPortableSystemRtlConfigView(llvm::ArrayRef<std::uint8_t>,
                                           llvm::ArrayRef<std::uint8_t>,
                                           const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t> resolvedPortableSystemRtlConfigSchemaBytes();
llvm::Expected<ResolvedPortableSystemRtlConfigView>
resolvePortableSystemRtlConfig();
llvm::Expected<ResolvedPortableSystemRtlConfigView>
adoptResolvedPortableSystemRtlConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const CandidateGeneratorDescriptor &
portableSystemRtlCandidateGeneratorDescriptor();
llvm::Error registerPortableSystemRtlCandidateGenerator();
llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindPortableSystemRtlCandidateGeneratorInputs(
    const ArtifactRootReference &system,
    const ArtifactRootReference &configurationAbi,
    llvm::ArrayRef<ArtifactRootReference> interconnectImplementations = {});
llvm::Expected<ResolvedCandidateGeneratorBinding>
resolvePortableSystemRtlCandidateGeneratorBinding(
    const ResolvedPortableSystemRtlConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_PORTABLESYSTEMRTLCANDIDATEGENERATOR_H
