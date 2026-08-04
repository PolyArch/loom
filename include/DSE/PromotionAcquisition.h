#ifndef LOOM_DSE_PROMOTIONACQUISITION_H
#define LOOM_DSE_PROMOTIONACQUISITION_H

#include "DSE/CandidateGenerator.h"
#include "DSE/Promotion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
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
      std::vector<PromotionAcquisitionInputBinding> inputBindings,
      llvm::ArrayRef<std::uint8_t> canonicalConfigBytes,
      const ComponentViewDigest &configDigest);

  PromotionAcquisitionDescriptorRef descriptorRef() const {
    return descriptor_;
  }
  llvm::ArrayRef<PromotionAcquisitionInputBinding> inputBindings() const {
    return inputBindings_;
  }
  llvm::ArrayRef<std::uint8_t> canonicalConfigBytes() const {
    return canonicalConfigBytes_;
  }
  const ComponentViewDigest &configDigest() const { return configDigest_; }
  const PromotionAcquisitionInputBinding *
  findInputBinding(PromotionAcquisitionInputSlotRef slot) const;

private:
  ResolvedPromotionAcquisitionBinding(
      PromotionAcquisitionDescriptorRef descriptor,
      std::vector<PromotionAcquisitionInputBinding> inputBindings,
      std::vector<std::uint8_t> canonicalConfigBytes,
      ComponentViewDigest configDigest)
      : descriptor_(descriptor), inputBindings_(std::move(inputBindings)),
        canonicalConfigBytes_(std::move(canonicalConfigBytes)),
        configDigest_(configDigest) {}

  PromotionAcquisitionDescriptorRef descriptor_;
  std::vector<PromotionAcquisitionInputBinding> inputBindings_;
  std::vector<std::uint8_t> canonicalConfigBytes_;
  ComponentViewDigest configDigest_;
};

struct CompletedPromotionAcquisition final {
  std::vector<PromotionEvidence> evidence;
  std::vector<CandidateObjectiveVector> objectives;
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

using PromotionAcquisitionProviderFunction =
    llvm::Expected<PromotionAcquisitionOutcome> (*)(
        const ResolvedPromotionAcquisitionBinding &binding,
        const ObjectiveProgram *objectiveProgram, const ArtifactStore &store);

struct PromotionAcquisitionProvider final {
  PromotionAcquisitionDescriptorRef descriptor;
  PromotionAcquisitionProviderFunction acquire;
};

llvm::Error registerPromotionAcquisitionProvider(
    const PromotionAcquisitionProvider &provider);

llvm::Expected<PromotionAcquisitionOutcome>
invokePromotionAcquisition(const ResolvedPromotionAcquisitionBinding &binding,
                           const ObjectiveProgram *objectiveProgram,
                           const ArtifactStore &store);

} // namespace loom::dse

#endif // LOOM_DSE_PROMOTIONACQUISITION_H
