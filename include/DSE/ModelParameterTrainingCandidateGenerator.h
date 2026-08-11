#ifndef LOOM_DSE_MODELPARAMETERTRAININGCANDIDATEGENERATOR_H
#define LOOM_DSE_MODELPARAMETERTRAININGCANDIDATEGENERATOR_H

#include "DSE/CandidateGenerator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    fpaGbdtTrainingCandidateGeneratorKind(17);
inline constexpr CandidateGeneratorKind
    systemRuntimeGbdtTrainingCandidateGeneratorKind(18);

struct DeterministicGbdtTrainingConfig final {
  std::uint64_t seed = 0;
  std::uint32_t treeCount = 0;
  std::uint32_t maximumDepth = 0;
  std::uint32_t minimumTrainingRowsPerLeaf = 0;
  std::uint32_t learningRateNumerator = 0;
  std::uint32_t learningRateDenominator = 0;
};

class ResolvedDeterministicGbdtTrainingConfigView final {
public:
  const DeterministicGbdtTrainingConfig &config() const { return config_; }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedDeterministicGbdtTrainingConfigView(
      DeterministicGbdtTrainingConfig config,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : config_(config), canonicalBytes_(std::move(canonicalBytes)),
        digest_(digest) {}

  DeterministicGbdtTrainingConfig config_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedDeterministicGbdtTrainingConfigView>
  resolveDeterministicGbdtTrainingConfig(
      const DeterministicGbdtTrainingConfig &);
  friend llvm::Expected<ResolvedDeterministicGbdtTrainingConfigView>
  adoptResolvedDeterministicGbdtTrainingConfigView(llvm::ArrayRef<std::uint8_t>,
                                                   llvm::ArrayRef<std::uint8_t>,
                                                   const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t>
resolvedDeterministicGbdtTrainingConfigSchemaBytes();

llvm::Expected<ResolvedDeterministicGbdtTrainingConfigView>
resolveDeterministicGbdtTrainingConfig(
    const DeterministicGbdtTrainingConfig &config);

llvm::Expected<ResolvedDeterministicGbdtTrainingConfigView>
adoptResolvedDeterministicGbdtTrainingConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const CandidateGeneratorDescriptor &
fpaGbdtTrainingCandidateGeneratorDescriptor();
llvm::Error registerFpaGbdtTrainingCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindFpaGbdtTrainingCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> training,
    llvm::ArrayRef<ArtifactRootReference> validation,
    llvm::ArrayRef<ArtifactRootReference> heldOut,
    const std::optional<ArtifactRootReference> &priorBundle = std::nullopt);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveFpaGbdtTrainingCandidateGeneratorBinding(
    const ResolvedDeterministicGbdtTrainingConfigView &config);

const CandidateGeneratorDescriptor &
systemRuntimeGbdtTrainingCandidateGeneratorDescriptor();

/// Registers the exact kind-18 descriptor without an execution provider.
/// Ground-truth import remains unavailable until the Gem5 Simulation Binding
/// owner supplies its production typed view and closure projector.
llvm::Error registerSystemRuntimeGbdtTrainingCandidateGeneratorDescriptor();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindSystemRuntimeGbdtTrainingCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> training,
    llvm::ArrayRef<ArtifactRootReference> validation,
    llvm::ArrayRef<ArtifactRootReference> heldOut,
    const std::optional<ArtifactRootReference> &priorBundle = std::nullopt);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveSystemRuntimeGbdtTrainingCandidateGeneratorBinding(
    const ResolvedDeterministicGbdtTrainingConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_MODELPARAMETERTRAININGCANDIDATEGENERATOR_H
