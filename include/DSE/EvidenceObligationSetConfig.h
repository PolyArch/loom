#ifndef LOOM_DSE_EVIDENCEOBLIGATIONSETCONFIG_H
#define LOOM_DSE_EVIDENCEOBLIGATIONSETCONFIG_H

#include "DSE/EvidenceObligation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom::dse {

/// Canonical finite set of Evidence obligations selected by one Promote
/// acquisition binding. The acquisition descriptor owns typed inputs and
/// provider behavior; this shared owner only encodes the selected ordinals.
class ResolvedEvidenceObligationSetConfigView final {
public:
  llvm::ArrayRef<EvidenceObligationTemplateRef> evidenceObligations() const {
    return evidenceObligations_;
  }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedEvidenceObligationSetConfigView(
      std::vector<EvidenceObligationTemplateRef> evidenceObligations,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : evidenceObligations_(std::move(evidenceObligations)),
        canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  std::vector<EvidenceObligationTemplateRef> evidenceObligations_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedEvidenceObligationSetConfigView>
      projectResolvedEvidenceObligationSetConfigView(
          llvm::ArrayRef<EvidenceObligationTemplateRef>);
  friend llvm::Expected<ResolvedEvidenceObligationSetConfigView>
  adoptResolvedEvidenceObligationSetConfigView(llvm::ArrayRef<std::uint8_t>,
                                               llvm::ArrayRef<std::uint8_t>,
                                               const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t> resolvedEvidenceObligationSetConfigSchemaBytes();

llvm::Expected<ResolvedEvidenceObligationSetConfigView>
projectResolvedEvidenceObligationSetConfigView(
    llvm::ArrayRef<EvidenceObligationTemplateRef> evidenceObligations);

llvm::Expected<ResolvedEvidenceObligationSetConfigView>
adoptResolvedEvidenceObligationSetConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

llvm::Error validateResolvedEvidenceObligationSetConfigView(
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

llvm::Expected<std::vector<EvidenceObligationTemplateRef>>
resolveEvidenceObligationSetConfig(
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes);

} // namespace loom::dse

#endif // LOOM_DSE_EVIDENCEOBLIGATIONSETCONFIG_H
