#ifndef LOOM_DSE_PROMOTIONACQUISITION_H
#define LOOM_DSE_PROMOTIONACQUISITION_H

#include "DSE/CandidateGenerator.h"
#include "DSE/EvidenceObligation.h"
#include "DSE/Promotion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse {

class PromotionAcquisitionKind final {
public:
  explicit constexpr PromotionAcquisitionKind(std::uint32_t ordinal)
      : ordinal_(ordinal) {}

  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(PromotionAcquisitionKind lhs,
                                   PromotionAcquisitionKind rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(PromotionAcquisitionKind lhs,
                                   PromotionAcquisitionKind rhs) {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(PromotionAcquisitionKind lhs,
                                  PromotionAcquisitionKind rhs) {
    return lhs.ordinal_ < rhs.ordinal_;
  }

private:
  std::uint32_t ordinal_;
};

class PromotionAcquisitionInputSlotRef final {
public:
  explicit constexpr PromotionAcquisitionInputSlotRef(std::uint32_t ordinal)
      : ordinal_(ordinal) {}

  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(PromotionAcquisitionInputSlotRef lhs,
                                   PromotionAcquisitionInputSlotRef rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }

private:
  std::uint32_t ordinal_;
};

struct PromotionAcquisitionInputSlotDescriptor final {
  PromotionAcquisitionInputSlotRef ref;
  llvm::StringLiteral spelling;
  PlanValueRole role;
  const ArtifactSchemaDescriptor *schema;
  PlanValueCardinality cardinality;
};

struct PromotionAcquisitionDescriptor;

class PromotionAcquisitionDescriptorRef final {
public:
  static llvm::Expected<PromotionAcquisitionDescriptorRef>
  get(ArtifactSchemaDescriptor descriptorSchema, PromotionAcquisitionKind kind);

  PromotionAcquisitionKind kind() const { return kind_; }
  const PromotionAcquisitionDescriptor *descriptor() const;

  friend bool operator==(const PromotionAcquisitionDescriptorRef &lhs,
                         const PromotionAcquisitionDescriptorRef &rhs) {
    return lhs.descriptorSchema_ == rhs.descriptorSchema_ &&
           lhs.kind_ == rhs.kind_;
  }
  friend bool operator!=(const PromotionAcquisitionDescriptorRef &lhs,
                         const PromotionAcquisitionDescriptorRef &rhs) {
    return !(lhs == rhs);
  }

private:
  PromotionAcquisitionDescriptorRef(ArtifactSchemaDescriptor descriptorSchema,
                                    PromotionAcquisitionKind kind)
      : descriptorSchema_(descriptorSchema), kind_(kind) {}

  ArtifactSchemaDescriptor descriptorSchema_;
  PromotionAcquisitionKind kind_;
};

struct PromotionAcquisitionDescriptor final {
  static constexpr ArtifactSchemaDescriptor schema{
      "loom.dse.promotion_acquisition_descriptor", SchemaVersion{1, 0}};

  PromotionAcquisitionKind kind;
  llvm::StringLiteral spelling;
  llvm::StringLiteral stableIdentity;
  llvm::ArrayRef<PromotionAcquisitionInputSlotDescriptor> inputSlots;
  PromotionAcquisitionInputSlotRef candidateInputSlot;
  evaluation::CaseSubjectRoleRef candidateRole;
  ResolvedDseConfigViewContract resolvedConfigView;
  llvm::Expected<std::vector<EvidenceObligationTemplateRef>> (
      *resolveEvidenceObligations)(
      llvm::ArrayRef<std::uint8_t> canonicalConfigBytes);

  PromotionAcquisitionDescriptorRef reference() const;
  const PromotionAcquisitionInputSlotDescriptor *
  findInputSlot(PromotionAcquisitionInputSlotRef ref) const;
};

llvm::Error registerPromotionAcquisitionDescriptor(
    const PromotionAcquisitionDescriptor &descriptor);
const PromotionAcquisitionDescriptor *
findPromotionAcquisitionDescriptor(PromotionAcquisitionKind kind);

struct PromotionAcquisitionInputBinding final {
  PromotionAcquisitionInputSlotRef slot;
  std::vector<ArtifactRootReference> artifacts;
};

class ResolvedPromotionAcquisitionBinding final {
public:
  static llvm::Expected<ResolvedPromotionAcquisitionBinding>
  get(PromotionAcquisitionDescriptorRef descriptor,
      llvm::ArrayRef<std::uint8_t> canonicalConfigBytes,
      const ComponentViewDigest &configDigest);

  PromotionAcquisitionDescriptorRef descriptorRef() const {
    return descriptor_;
  }
  llvm::ArrayRef<std::uint8_t> canonicalConfigBytes() const {
    return canonicalConfigBytes_;
  }
  const ComponentViewDigest &configDigest() const { return configDigest_; }
  llvm::ArrayRef<EvidenceObligationTemplateRef> evidenceObligations() const {
    return evidenceObligations_;
  }

private:
  ResolvedPromotionAcquisitionBinding(
      PromotionAcquisitionDescriptorRef descriptor,
      std::vector<std::uint8_t> canonicalConfigBytes,
      ComponentViewDigest configDigest,
      std::vector<EvidenceObligationTemplateRef> evidenceObligations)
      : descriptor_(descriptor),
        canonicalConfigBytes_(std::move(canonicalConfigBytes)),
        configDigest_(configDigest),
        evidenceObligations_(std::move(evidenceObligations)) {}

  PromotionAcquisitionDescriptorRef descriptor_;
  std::vector<std::uint8_t> canonicalConfigBytes_;
  ComponentViewDigest configDigest_;
  std::vector<EvidenceObligationTemplateRef> evidenceObligations_;
};

llvm::Error validatePromotionAcquisitionInputBindings(
    PromotionAcquisitionDescriptorRef descriptor,
    llvm::ArrayRef<PromotionAcquisitionInputBinding> inputBindings);

struct CompletedPromotionAcquisition final {
  std::vector<PromotionEvidence> evidence;
};

enum class PromotionAcquisitionIncompleteReason : std::uint8_t {
  ProviderUnavailable,
  SemanticWorkLimit,
  ObjectiveUnavailable,
  Unsupported,
};

struct IncompletePromotionAcquisition final {
  PromotionAcquisitionIncompleteReason reason;
  std::vector<PromotionEvidence> retainedEvidence;
};

using PromotionAcquisitionOutcome =
    std::variant<CompletedPromotionAcquisition, IncompletePromotionAcquisition>;

struct PromotionEvidenceAcquisitionTask final {
  EvidenceObligationTemplateRef obligationTemplate;
  const EvidenceObligationTemplate *obligation;
  ArtifactRootReference candidate;
  /// Invocation-local view shared by every candidate for this obligation.
  llvm::ArrayRef<EvidenceAcquisitionInputBinding> inputBindings;
};

struct ResolvedPromotionEvidenceAcquisitionTask final {
  std::uint64_t replicateIndex;
  /// Read-only invocation-local closure. Providers may share one exact
  /// resolution across every task whose case inputs are identical.
  std::shared_ptr<const evaluation::CaseArtifactResolution> resolution;
  /// Optional candidate-local subsets of the task's already bound inputs.
  /// The controller verifies exact slot coverage, canonical ordering, and
  /// subset membership before Request construction. Absence reuses the full
  /// task input bindings.
  std::optional<std::vector<EvidenceAcquisitionInputBinding>> selectedInputs;
};

struct CompletedPromotionAcquisitionResolution final {
  /// Positional results in the exact canonical task order supplied to the
  /// provider. Request and Evidence construction remain central-owned.
  std::vector<ResolvedPromotionEvidenceAcquisitionTask> tasks;
};

struct IncompletePromotionAcquisitionResolution final {
  PromotionAcquisitionIncompleteReason reason;
};

using PromotionAcquisitionResolutionOutcome =
    std::variant<CompletedPromotionAcquisitionResolution,
                 IncompletePromotionAcquisitionResolution>;

using PromotionAcquisitionProviderFunction =
    llvm::Expected<PromotionAcquisitionResolutionOutcome> (*)(
        const ResolvedPromotionAcquisitionBinding &binding,
        llvm::ArrayRef<PromotionAcquisitionInputBinding> inputBindings,
        llvm::ArrayRef<PromotionEvidenceAcquisitionTask> tasks,
        const ArtifactStore &store);

struct PromotionAcquisitionProvider final {
  PromotionAcquisitionDescriptorRef descriptor;
  PromotionAcquisitionProviderFunction resolve;
};

llvm::Error registerPromotionAcquisitionProvider(
    const PromotionAcquisitionProvider &provider);

struct PromotionAcquisitionTaskDomain final {
  llvm::ArrayRef<ArtifactRootReference> candidates;
  llvm::ArrayRef<EvidenceObligationTemplateRef> evidenceObligations;
};

llvm::Expected<PromotionAcquisitionOutcome> invokePromotionAcquisition(
    llvm::ArrayRef<PromotionAcquisitionInputBinding> inputBindings,
    const ResolvedPromotionAcquisitionBinding &binding,
    llvm::ArrayRef<EvidenceObligationTemplate> evidenceObligationTemplates,
    PromotionAcquisitionTaskDomain taskDomain, const ArtifactStore &store,
    const BlobStore &blobs);

} // namespace loom::dse

#endif // LOOM_DSE_PROMOTIONACQUISITION_H
