#ifndef LOOM_DSE_STRUCTUREDSPECIALMATHACCURACYCANDIDATEGENERATOR_H
#define LOOM_DSE_STRUCTUREDSPECIALMATHACCURACYCANDIDATEGENERATOR_H

#include "Common/Artifact.h"
#include "Common/ComponentViewDigest.h"
#include "DSE/CandidateGenerator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    structuredSpecialMathAccuracyCandidateGeneratorKind(10);

class ResolvedStructuredSpecialMathAccuracyGeneratorConfigView final {
public:
  std::optional<std::uint64_t> maximumMaterializationAttempts() const {
    return maximumMaterializationAttempts_;
  }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalViewBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedStructuredSpecialMathAccuracyGeneratorConfigView(
      std::optional<std::uint64_t> maximumMaterializationAttempts,
      std::vector<std::uint8_t> canonicalViewBytes, ComponentViewDigest digest)
      : maximumMaterializationAttempts_(maximumMaterializationAttempts),
        canonicalViewBytes_(std::move(canonicalViewBytes)), digest_(digest) {}

  std::optional<std::uint64_t> maximumMaterializationAttempts_;
  std::vector<std::uint8_t> canonicalViewBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<
      ResolvedStructuredSpecialMathAccuracyGeneratorConfigView>
  projectResolvedStructuredSpecialMathAccuracyGeneratorConfigView(
      std::optional<std::uint64_t>);
  friend llvm::Expected<
      ResolvedStructuredSpecialMathAccuracyGeneratorConfigView>
  adoptResolvedStructuredSpecialMathAccuracyGeneratorConfigView(
      llvm::ArrayRef<std::uint8_t>, llvm::ArrayRef<std::uint8_t>,
      const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t>
resolvedStructuredSpecialMathAccuracyGeneratorConfigSchemaBytes();
llvm::Expected<ResolvedStructuredSpecialMathAccuracyGeneratorConfigView>
projectResolvedStructuredSpecialMathAccuracyGeneratorConfigView(
    std::optional<std::uint64_t> maximumMaterializationAttempts =
        std::nullopt);
llvm::Expected<ResolvedStructuredSpecialMathAccuracyGeneratorConfigView>
adoptResolvedStructuredSpecialMathAccuracyGeneratorConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const CandidateGeneratorDescriptor &
structuredSpecialMathAccuracyCandidateGeneratorDescriptor();
llvm::Error registerStructuredSpecialMathAccuracyCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindStructuredSpecialMathAccuracyCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms,
    const ArtifactRootReference &fabric);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveStructuredSpecialMathAccuracyCandidateGeneratorBinding(
    const ResolvedStructuredSpecialMathAccuracyGeneratorConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_STRUCTUREDSPECIALMATHACCURACYCANDIDATEGENERATOR_H
