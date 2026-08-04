#ifndef LOOM_DSE_CANDIDATEGENERATOR_H
#define LOOM_DSE_CANDIDATEGENERATOR_H

#include "Common/Artifact.h"
#include "Common/ComponentViewDigest.h"
#include "DSE/PlanValue.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::dse {

inline constexpr ArtifactSchemaDescriptor candidateGeneratorDescriptorSchema{
    "loom.candidate_generator_descriptor", SchemaVersion{1, 0}};

class CandidateGeneratorKind final {
public:
  explicit constexpr CandidateGeneratorKind(std::uint32_t ordinal)
      : ordinal_(ordinal) {}

  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(CandidateGeneratorKind lhs,
                                   CandidateGeneratorKind rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(CandidateGeneratorKind lhs,
                                   CandidateGeneratorKind rhs) {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(CandidateGeneratorKind lhs,
                                  CandidateGeneratorKind rhs) {
    return lhs.ordinal_ < rhs.ordinal_;
  }

private:
  std::uint32_t ordinal_;
};

class CandidateGeneratorInputSlotRef final {
public:
  explicit constexpr CandidateGeneratorInputSlotRef(std::uint32_t ordinal)
      : ordinal_(ordinal) {}
  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(CandidateGeneratorInputSlotRef lhs,
                                   CandidateGeneratorInputSlotRef rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(CandidateGeneratorInputSlotRef lhs,
                                   CandidateGeneratorInputSlotRef rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint32_t ordinal_;
};

class CandidateGeneratorOutputSlotRef final {
public:
  explicit constexpr CandidateGeneratorOutputSlotRef(std::uint32_t ordinal)
      : ordinal_(ordinal) {}
  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(CandidateGeneratorOutputSlotRef lhs,
                                   CandidateGeneratorOutputSlotRef rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(CandidateGeneratorOutputSlotRef lhs,
                                   CandidateGeneratorOutputSlotRef rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint32_t ordinal_;
};

class CandidateGeneratorWorkUnitRef final {
public:
  explicit constexpr CandidateGeneratorWorkUnitRef(std::uint32_t ordinal)
      : ordinal_(ordinal) {}
  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(CandidateGeneratorWorkUnitRef lhs,
                                   CandidateGeneratorWorkUnitRef rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }

private:
  std::uint32_t ordinal_;
};

class CandidateGeneratorProjectionSlotRef final {
public:
  explicit constexpr CandidateGeneratorProjectionSlotRef(std::uint32_t ordinal)
      : ordinal_(ordinal) {}
  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(CandidateGeneratorProjectionSlotRef lhs,
                                   CandidateGeneratorProjectionSlotRef rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }

private:
  std::uint32_t ordinal_;
};

struct CandidateGeneratorInputBinding final {
  CandidateGeneratorInputSlotRef slot;
  std::vector<ArtifactRootReference> artifacts;
};

struct CandidateGeneratorInputSlotDescriptor final {
  CandidateGeneratorInputSlotRef slot;
  llvm::StringRef semanticRole;
  PlanValueRole role;
  const ArtifactSchemaDescriptor *schema;
  PlanValueCardinality cardinality;
};

struct CandidateGeneratorOutputSlotDescriptor final {
  CandidateGeneratorOutputSlotRef slot;
  llvm::StringRef semanticRole;
  PlanValueRole role;
  const ArtifactSchemaDescriptor *schema;
  PlanValueCardinality cardinality;
};

enum class CandidateGeneratorDeterminism : std::uint32_t {
  Deterministic,
  IndependentReplicates,
};

struct CandidateGeneratorWorkUnitDescriptor final {
  CandidateGeneratorWorkUnitRef unit;
  llvm::StringRef spelling;
};

enum class CandidateGeneratorProjectionKind : std::uint32_t {
  MappingViolation,
  MappingMeasure,
  EvaluationMetric,
  EvaluationFinding,
};

struct CandidateGeneratorProjectionSlotDescriptor final {
  CandidateGeneratorProjectionSlotRef slot;
  llvm::StringRef semanticRole;
  CandidateGeneratorProjectionKind kind;
};

struct CandidateGeneratorDescriptor;

class CandidateGeneratorDescriptorRef final {
public:
  static llvm::Expected<CandidateGeneratorDescriptorRef>
  get(const ArtifactSchemaDescriptor &descriptorSchema,
      CandidateGeneratorKind kind);

  const ArtifactSchemaDescriptor &descriptorSchema() const {
    return descriptorSchema_;
  }
  CandidateGeneratorKind kind() const { return kind_; }
  const CandidateGeneratorDescriptor *descriptor() const;

  friend bool operator==(const CandidateGeneratorDescriptorRef &lhs,
                         const CandidateGeneratorDescriptorRef &rhs) {
    return lhs.descriptorSchema_ == rhs.descriptorSchema_ &&
           lhs.kind_ == rhs.kind_;
  }
  friend bool operator!=(const CandidateGeneratorDescriptorRef &lhs,
                         const CandidateGeneratorDescriptorRef &rhs) {
    return !(lhs == rhs);
  }

private:
  CandidateGeneratorDescriptorRef(ArtifactSchemaDescriptor descriptorSchema,
                                  CandidateGeneratorKind kind)
      : descriptorSchema_(descriptorSchema), kind_(kind) {}

  ArtifactSchemaDescriptor descriptorSchema_;
  CandidateGeneratorKind kind_;
};

struct CandidateGeneratorDescriptor final {
  CandidateGeneratorKind kind;
  llvm::StringRef spelling;
  llvm::StringRef implementationSemanticIdentity;
  llvm::ArrayRef<CandidateGeneratorInputSlotDescriptor> inputSlots;
  llvm::ArrayRef<CandidateGeneratorOutputSlotDescriptor> outputSlots;
  ResolvedDseConfigViewContract resolvedConfigView;
  CandidateGeneratorDeterminism determinism;
  llvm::ArrayRef<CandidateGeneratorWorkUnitDescriptor> workUnits;
  llvm::ArrayRef<CandidateGeneratorProjectionSlotDescriptor> projectionSlots;

  CandidateGeneratorDescriptorRef reference() const;
  const CandidateGeneratorInputSlotDescriptor *
  findInputSlot(CandidateGeneratorInputSlotRef slot) const;
  const CandidateGeneratorOutputSlotDescriptor *
  findOutputSlot(CandidateGeneratorOutputSlotRef slot) const;
};

llvm::Error registerCandidateGeneratorDescriptor(
    const CandidateGeneratorDescriptor &descriptor);
const CandidateGeneratorDescriptor *
findCandidateGeneratorDescriptor(CandidateGeneratorKind kind);

class ResolvedCandidateGeneratorBinding final {
public:
  static llvm::Expected<ResolvedCandidateGeneratorBinding>
  get(CandidateGeneratorDescriptorRef descriptor,
      std::vector<CandidateGeneratorInputBinding> inputBindings,
      llvm::ArrayRef<std::uint8_t> canonicalConfigBytes,
      const ComponentViewDigest &configDigest);

  CandidateGeneratorDescriptorRef descriptorRef() const { return descriptor_; }
  llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings() const {
    return inputBindings_;
  }
  llvm::ArrayRef<std::uint8_t> canonicalConfigBytes() const {
    return canonicalConfigBytes_;
  }
  const ComponentViewDigest &configDigest() const { return configDigest_; }

  const CandidateGeneratorInputBinding *
  findInputBinding(CandidateGeneratorInputSlotRef slot) const;

private:
  ResolvedCandidateGeneratorBinding(
      CandidateGeneratorDescriptorRef descriptor,
      std::vector<CandidateGeneratorInputBinding> inputBindings,
      std::vector<std::uint8_t> canonicalConfigBytes,
      ComponentViewDigest configDigest)
      : descriptor_(descriptor), inputBindings_(std::move(inputBindings)),
        canonicalConfigBytes_(std::move(canonicalConfigBytes)),
        configDigest_(configDigest) {}

  CandidateGeneratorDescriptorRef descriptor_;
  std::vector<CandidateGeneratorInputBinding> inputBindings_;
  std::vector<std::uint8_t> canonicalConfigBytes_;
  ComponentViewDigest configDigest_;
};

struct CandidateGeneratorOutputBinding final {
  CandidateGeneratorOutputSlotRef slot;
  std::vector<ArtifactRootReference> artifacts;
};

struct CompletedCandidateGeneratorInvocation final {
  std::vector<CandidateGeneratorOutputBinding> outputBindings;
};

enum class CandidateGeneratorIncompleteReason : std::uint32_t {
  ProofNotEstablished = 0,
  SemanticLimitReached = 1,
  ProviderUnavailable = 2,
  Unsupported = 3,
};

struct IncompleteCandidateGeneratorInvocation final {
  CandidateGeneratorIncompleteReason reason;
  std::vector<CandidateGeneratorOutputBinding> retainedOutputBindings;
};

using CandidateGeneratorInvocationOutcome =
    std::variant<CompletedCandidateGeneratorInvocation,
                 IncompleteCandidateGeneratorInvocation>;

using CandidateGeneratorProviderFunction =
    llvm::Expected<CandidateGeneratorInvocationOutcome> (*)(
        const ResolvedCandidateGeneratorBinding &, const ArtifactStore &);

struct CandidateGeneratorProvider final {
  CandidateGeneratorDescriptorRef descriptor;
  CandidateGeneratorProviderFunction invoke;
};

llvm::Error
registerCandidateGeneratorProvider(const CandidateGeneratorProvider &provider);

/// Invokes the exact registered provider and canonicalizes every typed output
/// set. Missing implementation is a typed Incomplete outcome.
llvm::Expected<CandidateGeneratorInvocationOutcome>
invokeCandidateGenerator(const ResolvedCandidateGeneratorBinding &binding,
                         const ArtifactStore &store);

} // namespace loom::dse

#endif // LOOM_DSE_CANDIDATEGENERATOR_H
