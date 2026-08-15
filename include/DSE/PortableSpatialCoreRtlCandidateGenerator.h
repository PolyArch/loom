#ifndef LOOM_DSE_PORTABLESPATIALCORERTLCANDIDATEGENERATOR_H
#define LOOM_DSE_PORTABLESPATIALCORERTLCANDIDATEGENERATOR_H

#include "DSE/CandidateGenerator.h"

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    portableSpatialCoreRtlCandidateGeneratorKind(16);

class ResolvedPortableSpatialCoreRtlConfigView final {
public:
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedPortableSpatialCoreRtlConfigView(
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedPortableSpatialCoreRtlConfigView>
  resolvePortableSpatialCoreRtlConfig();
  friend llvm::Expected<ResolvedPortableSpatialCoreRtlConfigView>
  adoptResolvedPortableSpatialCoreRtlConfigView(
      llvm::ArrayRef<std::uint8_t>, llvm::ArrayRef<std::uint8_t>,
      const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t>
resolvedPortableSpatialCoreRtlConfigSchemaBytes();
llvm::Expected<ResolvedPortableSpatialCoreRtlConfigView>
resolvePortableSpatialCoreRtlConfig();
llvm::Expected<ResolvedPortableSpatialCoreRtlConfigView>
adoptResolvedPortableSpatialCoreRtlConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const CandidateGeneratorDescriptor &
portableSpatialCoreRtlCandidateGeneratorDescriptor();
llvm::Error registerPortableSpatialCoreRtlCandidateGenerator();
llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindPortableSpatialCoreRtlCandidateGeneratorInputs(
    const ArtifactRootReference &system,
    const ArtifactRootReference &configurationAbi);
llvm::Expected<ResolvedCandidateGeneratorBinding>
resolvePortableSpatialCoreRtlCandidateGeneratorBinding(
    const ResolvedPortableSpatialCoreRtlConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_PORTABLESPATIALCORERTLCANDIDATEGENERATOR_H
