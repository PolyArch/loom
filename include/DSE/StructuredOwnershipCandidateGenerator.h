#ifndef LOOM_DSE_STRUCTUREDOWNERSHIPCANDIDATEGENERATOR_H
#define LOOM_DSE_STRUCTUREDOWNERSHIPCANDIDATEGENERATOR_H

#include "Common/ComponentViewDigest.h"
#include "DSE/CandidateGenerator.h"
#include "DSE/StructuredOwnership.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom {
struct ResolvedConfig;
}

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    structuredOwnershipCandidateGeneratorKind(1);

class ResolvedStructuredOwnershipGeneratorConfigView final {
public:
  std::uint64_t scopeExpansionLimit() const { return scopeExpansionLimit_; }
  std::optional<std::uint64_t> maximumMaterializationAttempts() const {
    return maximumMaterializationAttempts_;
  }
  llvm::ArrayRef<frontend::StructuredEntityRef> protocolCallableRoots() const {
    return protocolCallableRoots_;
  }
  StructuredOwnershipGenerationIntent generationIntent() const {
    return generationIntent_;
  }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedStructuredOwnershipGeneratorConfigView(
      std::uint64_t scopeExpansionLimit,
      std::optional<std::uint64_t> maximumMaterializationAttempts,
      std::vector<frontend::StructuredEntityRef> protocolCallableRoots,
      StructuredOwnershipGenerationIntent generationIntent,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : scopeExpansionLimit_(scopeExpansionLimit),
        maximumMaterializationAttempts_(maximumMaterializationAttempts),
        protocolCallableRoots_(std::move(protocolCallableRoots)),
        generationIntent_(generationIntent),
        canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  std::uint64_t scopeExpansionLimit_;
  std::optional<std::uint64_t> maximumMaterializationAttempts_;
  std::vector<frontend::StructuredEntityRef> protocolCallableRoots_;
  StructuredOwnershipGenerationIntent generationIntent_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedStructuredOwnershipGeneratorConfigView>
  projectResolvedStructuredOwnershipGeneratorConfigView(
      const ResolvedConfig &, llvm::ArrayRef<frontend::StructuredEntityRef>,
      StructuredOwnershipGenerationIntent, std::optional<std::uint64_t>);
  friend llvm::Expected<ResolvedStructuredOwnershipGeneratorConfigView>
  adoptResolvedStructuredOwnershipGeneratorConfigView(
      llvm::ArrayRef<std::uint8_t>, llvm::ArrayRef<std::uint8_t>,
      const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t>
resolvedStructuredOwnershipGeneratorConfigSchemaBytes();

llvm::Expected<ResolvedStructuredOwnershipGeneratorConfigView>
projectResolvedStructuredOwnershipGeneratorConfigView(
    const ResolvedConfig &config,
    llvm::ArrayRef<frontend::StructuredEntityRef> protocolCallableRoots = {},
    StructuredOwnershipGenerationIntent generationIntent =
        StructuredOwnershipGenerationIntent::Balanced,
    std::optional<std::uint64_t> maximumMaterializationAttempts =
        std::nullopt);

llvm::Expected<ResolvedStructuredOwnershipGeneratorConfigView>
adoptResolvedStructuredOwnershipGeneratorConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const CandidateGeneratorDescriptor &
structuredOwnershipCandidateGeneratorDescriptor();
llvm::Error registerStructuredOwnershipCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindStructuredOwnershipCandidateGeneratorInputs(
    const ArtifactRootReference &structuredProgram,
    const ArtifactRootReference &fabric, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveStructuredOwnershipCandidateGeneratorBinding(
    const ResolvedStructuredOwnershipGeneratorConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_STRUCTUREDOWNERSHIPCANDIDATEGENERATOR_H
