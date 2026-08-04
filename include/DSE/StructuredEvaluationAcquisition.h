#ifndef LOOM_DSE_STRUCTUREDEVALUATIONACQUISITION_H
#define LOOM_DSE_STRUCTUREDEVALUATIONACQUISITION_H

#include "DSE/PromotionAcquisition.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::dse {

inline constexpr PromotionAcquisitionKind
    structuredEvaluationPromotionAcquisitionKind(0);

class ResolvedStructuredEvaluationAcquisitionConfigView final {
public:
  llvm::ArrayRef<EvidenceObligationTemplateRef> evidenceObligations() const {
    return evidenceObligations_;
  }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedStructuredEvaluationAcquisitionConfigView(
      std::vector<EvidenceObligationTemplateRef> evidenceObligations,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : evidenceObligations_(std::move(evidenceObligations)),
        canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  std::vector<EvidenceObligationTemplateRef> evidenceObligations_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedStructuredEvaluationAcquisitionConfigView>
      projectResolvedStructuredEvaluationAcquisitionConfigView(
          llvm::ArrayRef<EvidenceObligationTemplateRef>);
  friend llvm::Expected<ResolvedStructuredEvaluationAcquisitionConfigView>
  adoptResolvedStructuredEvaluationAcquisitionConfigView(
      llvm::ArrayRef<std::uint8_t>, llvm::ArrayRef<std::uint8_t>,
      const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t>
resolvedStructuredEvaluationAcquisitionConfigSchemaBytes();

llvm::Expected<ResolvedStructuredEvaluationAcquisitionConfigView>
projectResolvedStructuredEvaluationAcquisitionConfigView(
    llvm::ArrayRef<EvidenceObligationTemplateRef> evidenceObligations);

llvm::Expected<ResolvedStructuredEvaluationAcquisitionConfigView>
adoptResolvedStructuredEvaluationAcquisitionConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const PromotionAcquisitionDescriptor &
structuredEvaluationPromotionAcquisitionDescriptor();
llvm::Error registerStructuredEvaluationPromotionAcquisition();

llvm::Expected<std::vector<PromotionAcquisitionInputBinding>>
bindStructuredEvaluationPromotionInputs(
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms,
    const ArtifactRootReference &fabric, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput);

llvm::Expected<ResolvedPromotionAcquisitionBinding>
resolveStructuredEvaluationPromotionAcquisitionBinding(
    const ResolvedStructuredEvaluationAcquisitionConfigView &config);

llvm::Expected<EvidenceObligationTemplate>
prepareStructuredFabricAnalyticEvidenceObligationTemplate(
    const ArtifactRootReference &prototypeCandidate,
    const ArtifactRootReference &fabric, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &store);

llvm::Expected<EvidenceObligationTemplate>
prepareStructuredProgramFunctionalEvidenceObligationTemplate(
    const ArtifactRootReference &prototypeCandidate,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &store);

} // namespace loom::dse

#endif // LOOM_DSE_STRUCTUREDEVALUATIONACQUISITION_H
