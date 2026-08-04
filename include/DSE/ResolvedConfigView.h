#ifndef LOOM_DSE_RESOLVEDCONFIGVIEW_H
#define LOOM_DSE_RESOLVEDCONFIGVIEW_H

#include "Common/ComponentViewDigest.h"
#include "DSE/EvidenceObligation.h"
#include "DSE/Plan.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom::dse {

struct ModelAuthorization final {
  evaluation::EvaluationModelDescriptorRef descriptor;
};

/// The sealed policy view consumed by the central DSE controller. Catalog
/// ordinals are local to these exact bytes; executable state and caches are
/// derived and remain outside the view.
class ResolvedDseConfigView final {
public:
  static llvm::Expected<ResolvedDseConfigView>
  get(std::vector<ModelAuthorization> modelAuthorizations,
      std::vector<EvidenceObligationTemplate> evidenceObligationTemplates,
      ResolvedObjectiveCatalogs objectiveCatalogs,
      std::vector<QualityGatePolicy> qualityGatePolicies,
      std::vector<DsePlanNodeDefinition> planNodes);

  llvm::ArrayRef<ModelAuthorization> modelAuthorizations() const {
    return modelAuthorizations_;
  }
  llvm::ArrayRef<EvidenceObligationTemplate>
  evidenceObligationTemplates() const {
    return evidenceObligationTemplates_;
  }
  const ResolvedObjectiveCatalogs &objectiveCatalogs() const {
    return objectiveCatalogs_;
  }
  llvm::ArrayRef<QualityGatePolicy> qualityGatePolicies() const {
    return plan_.qualityGatePolicies();
  }
  const ResolvedDsePlan &plan() const { return plan_; }
  llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes() const;
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedDseConfigView(
      std::vector<ModelAuthorization> modelAuthorizations,
      std::vector<EvidenceObligationTemplate> evidenceObligationTemplates,
      ResolvedObjectiveCatalogs objectiveCatalogs, ResolvedDsePlan plan,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : modelAuthorizations_(std::move(modelAuthorizations)),
        evidenceObligationTemplates_(std::move(evidenceObligationTemplates)),
        objectiveCatalogs_(std::move(objectiveCatalogs)),
        plan_(std::move(plan)), canonicalBytes_(std::move(canonicalBytes)),
        digest_(digest) {}

  std::vector<ModelAuthorization> modelAuthorizations_;
  std::vector<EvidenceObligationTemplate> evidenceObligationTemplates_;
  ResolvedObjectiveCatalogs objectiveCatalogs_;
  ResolvedDsePlan plan_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;
};

llvm::Expected<ResolvedDseConfigView>
adoptResolvedDseConfigView(llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
                           llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
                           const ComponentViewDigest &digest);

} // namespace loom::dse

#endif // LOOM_DSE_RESOLVEDCONFIGVIEW_H
