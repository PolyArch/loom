#ifndef LOOM_DSE_STRUCTUREDSCHEDULECANDIDATEGENERATOR_H
#define LOOM_DSE_STRUCTUREDSCHEDULECANDIDATEGENERATOR_H

#include "Common/ComponentViewDigest.h"
#include "DSE/CandidateGenerator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom {
struct ResolvedConfig;
}

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    structuredScheduleCandidateGeneratorKind(2);

enum class StructuredScheduleGenerationIntent : std::uint8_t {
  Balanced = 0,
  RequireLogicalThreadDomain = 1,
  ForbidLogicalThreadDomain = 2,
};

class ResolvedStructuredScheduleGeneratorConfigView final {
public:
  std::uint64_t scopeExpansionLimit() const { return scopeExpansionLimit_; }
  std::optional<std::uint64_t> maximumMaterializationAttempts() const {
    return maximumMaterializationAttempts_;
  }
  StructuredScheduleGenerationIntent generationIntent() const {
    return generationIntent_;
  }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedStructuredScheduleGeneratorConfigView(
      std::uint64_t scopeExpansionLimit,
      std::optional<std::uint64_t> maximumMaterializationAttempts,
      StructuredScheduleGenerationIntent generationIntent,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : scopeExpansionLimit_(scopeExpansionLimit),
        maximumMaterializationAttempts_(maximumMaterializationAttempts),
        generationIntent_(generationIntent),
        canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  std::uint64_t scopeExpansionLimit_;
  std::optional<std::uint64_t> maximumMaterializationAttempts_;
  StructuredScheduleGenerationIntent generationIntent_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedStructuredScheduleGeneratorConfigView>
  projectResolvedStructuredScheduleGeneratorConfigView(
      const ResolvedConfig &, StructuredScheduleGenerationIntent,
      std::optional<std::uint64_t>);
  friend llvm::Expected<ResolvedStructuredScheduleGeneratorConfigView>
  adoptResolvedStructuredScheduleGeneratorConfigView(
      llvm::ArrayRef<std::uint8_t>, llvm::ArrayRef<std::uint8_t>,
      const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t>
resolvedStructuredScheduleGeneratorConfigSchemaBytes();

llvm::Expected<ResolvedStructuredScheduleGeneratorConfigView>
projectResolvedStructuredScheduleGeneratorConfigView(
    const ResolvedConfig &config,
    StructuredScheduleGenerationIntent intent =
        StructuredScheduleGenerationIntent::Balanced,
    std::optional<std::uint64_t> maximumMaterializationAttempts = std::nullopt);

llvm::Expected<ResolvedStructuredScheduleGeneratorConfigView>
adoptResolvedStructuredScheduleGeneratorConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const CandidateGeneratorDescriptor &
structuredScheduleCandidateGeneratorDescriptor();
llvm::Error registerStructuredScheduleCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindStructuredScheduleCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms,
    const ArtifactRootReference &fabric);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveStructuredScheduleCandidateGeneratorBinding(
    const ResolvedStructuredScheduleGeneratorConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_STRUCTUREDSCHEDULECANDIDATEGENERATOR_H
