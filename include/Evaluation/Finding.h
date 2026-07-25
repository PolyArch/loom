#ifndef LOOM_EVALUATION_FINDING_H
#define LOOM_EVALUATION_FINDING_H

#include "Evaluation/Case.h"
#include "Evaluation/OwnerValue.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::evaluation {

class ModelOutputSlotRef;
class FindingOccurrenceContext;

/// A stable registry ordinal naming one finding kind within the exact
/// Evaluation schema version.
class FindingKind {
public:
  explicit constexpr FindingKind(std::uint32_t ordinal) : ordinal_(ordinal) {}

  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(FindingKind lhs, FindingKind rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(FindingKind lhs, FindingKind rhs) {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(FindingKind lhs, FindingKind rhs) {
    return lhs.ordinal_ < rhs.ordinal_;
  }

private:
  std::uint32_t ordinal_;
};

enum class FindingResultForm : std::uint8_t { Absent, Present, NotApplicable };

constexpr std::uint8_t findingResultFormMask(FindingResultForm form) {
  return std::uint8_t{1} << static_cast<std::uint8_t>(form);
}

constexpr std::uint8_t allFindingResultFormsMask() {
  return findingResultFormMask(FindingResultForm::Absent) |
         findingResultFormMask(FindingResultForm::Present) |
         findingResultFormMask(FindingResultForm::NotApplicable);
}

struct FindingPayloadSchemaDescriptor {
  llvm::StringRef identity;
  SchemaVersion version;
};

/// The FindingKind owner supplies the complete occurrence codec. Evaluation
/// owns only outer framing and dispatch; the adopted value remains owner-typed.
struct FindingOccurrenceCodec {
  FindingPayloadSchemaDescriptor occurrenceSchema;
  llvm::Expected<std::vector<std::uint8_t>> (*encode)(
      const OwnerValue &occurrence);
  llvm::Expected<OwnerValue> (*decode)(
      llvm::ArrayRef<std::uint8_t> canonicalPayload);
  llvm::Error (*validate)(const OwnerValue &occurrence,
                          const FindingOccurrenceContext &context);
};

struct FindingDescriptor {
  FindingKind kind;
  llvm::StringRef spelling;
  llvm::StringRef semanticDefinition;
  llvm::ArrayRef<ScopeFormDescriptor> scopeForms;
  llvm::ArrayRef<ConditionApplicabilityPattern>
      permittedRequestConditionPatterns;
  FindingOccurrenceCodec occurrenceCodec;
  std::optional<FindingPayloadSchemaDescriptor> terminalWitnessSchema;
};

llvm::Error registerFindingDescriptor(const FindingDescriptor &descriptor);
const FindingDescriptor *findFindingDescriptor(FindingKind kind);
llvm::Error requireFindingOccurrenceOwner(
    const FindingDescriptor &descriptor);

llvm::Expected<FindingKind> parseFindingKind(llvm::StringRef spelling);
llvm::StringRef toString(FindingKind kind);

struct FindingQuery {
  FindingKind kind;
  EvaluationScope scope;

  friend bool operator==(const FindingQuery &lhs, const FindingQuery &rhs) {
    return lhs.kind == rhs.kind && lhs.scope == rhs.scope;
  }
  friend bool operator!=(const FindingQuery &lhs, const FindingQuery &rhs) {
    return !(lhs == rhs);
  }
};

llvm::Error validateFindingQuery(const FindingQuery &query);
llvm::Expected<std::vector<FindingQuery>>
canonicalizeFindingQueries(llvm::ArrayRef<FindingQuery> queries);
std::vector<std::uint8_t> canonicalFindingQueryKey(const FindingQuery &query);

llvm::Expected<std::string> serializeFindingQuery(const FindingQuery &query);
llvm::Expected<FindingQuery> parseFindingQuery(llvm::StringRef json);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_FINDING_H
