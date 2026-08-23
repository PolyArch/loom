#ifndef LOOM_DSE_OBJECTIVE_H
#define LOOM_DSE_OBJECTIVE_H

#include "Common/ResolvedPnrPolicy.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace loom::dse {

class ObjectiveUnavailableError final
    : public llvm::ErrorInfo<ObjectiveUnavailableError> {
public:
  static char ID;

  explicit ObjectiveUnavailableError(std::string detail)
      : detail_(std::move(detail)) {}

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  std::string detail_;
};

struct EvaluationMetricObjectiveValue final {
  std::uint32_t evidenceObligationTemplate;
  std::uint64_t metricRequestOrdinal;
  ResolvedObjectiveScalar value;
};

struct ObjectiveSourceValues final {
  llvm::ArrayRef<std::uint64_t> mappingViolations;
  llvm::ArrayRef<std::uint64_t> mappingMeasures;
  llvm::ArrayRef<EvaluationMetricObjectiveValue> evaluationMetrics;
};

/// One invocation-local candidate measure dimension. Candidate measures have
/// no persistent ResolvedConfig spelling or Mapping-measure meaning; their
/// ordinals are owned by the caller that constructs the program.
struct CandidateMeasureObjectiveDimension final {
  std::uint32_t measureOrdinal = 0;
  ResolvedObjectiveDirection direction =
      ResolvedObjectiveDirection::Minimize;
  std::uint64_t lowerIndex = 0;
  std::uint64_t upperIndex = 0;
};

struct CandidateMeasureObjectiveCatalogs final {
  std::vector<CandidateMeasureObjectiveDimension> dimensions;
  std::vector<ResolvedWeightedObjectiveLevel> weightedLevels;
  std::vector<ResolvedTotalOrdering> totalOrderings;
};

class ObjectiveVector final {
public:
  llvm::ArrayRef<std::uint64_t> codes() const { return codes_; }

private:
  explicit ObjectiveVector(std::size_t dimensionCount)
      : codes_(dimensionCount, 0) {}

  llvm::SmallVector<std::uint64_t, 16> codes_;

  friend class ObjectiveProgram;
};

struct ObjectiveWideValue final {
  std::uint64_t high = 0;
  std::uint64_t low = 0;

  friend bool operator==(ObjectiveWideValue lhs, ObjectiveWideValue rhs) {
    return lhs.high == rhs.high && lhs.low == rhs.low;
  }
  friend bool operator!=(ObjectiveWideValue lhs, ObjectiveWideValue rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(ObjectiveWideValue lhs, ObjectiveWideValue rhs) {
    return lhs.high < rhs.high || (lhs.high == rhs.high && lhs.low < rhs.low);
  }
};

enum class ObjectiveDifferenceSign : std::int8_t {
  Negative = -1,
  Zero = 0,
  Positive = 1,
};

struct ObjectiveSignedDifference final {
  ObjectiveDifferenceSign sign = ObjectiveDifferenceSign::Zero;
  ObjectiveWideValue magnitude;
};

enum class ParetoRelation : std::uint8_t {
  Equivalent,
  Dominates,
  Dominated,
  Incomparable,
};

/// Preflighted, removable projection of one exact resolved objective catalog.
/// Evaluation writes into caller-owned vectors without allocation; the
/// catalog remains the semantic authority.
class ObjectiveProgram final {
public:
  static llvm::Expected<ObjectiveProgram>
  get(const ResolvedObjectiveCatalogs &catalogs);

  /// Builds the same compiled objective machinery for transient candidate
  /// features without pretending those features are Mapping measures or
  /// persisted Evaluation requests.
  static llvm::Expected<ObjectiveProgram>
  getCandidateMeasures(const CandidateMeasureObjectiveCatalogs &catalogs);

  ObjectiveVector makeVector() const {
    return ObjectiveVector(dimensions_.size());
  }

  std::size_t dimensionCount() const { return dimensions_.size(); }
  std::size_t weightedLevelCount() const { return levels_.size(); }
  std::size_t totalOrderingCount() const { return orderings_.size(); }

  llvm::Error evaluate(ObjectiveSourceValues sources,
                       ObjectiveVector &result) const;

  llvm::Error evaluateCandidateMeasures(
      llvm::ArrayRef<std::uint64_t> measures, ObjectiveVector &result) const;

  llvm::Expected<ObjectiveWideValue>
  weightedLevelValue(const ObjectiveVector &vector,
                     std::uint32_t weightedLevel) const;

  /// Returns left minus right without narrowing to a signed host integer.
  llvm::Expected<ObjectiveSignedDifference>
  signedWeightedLevelDifference(const ObjectiveVector &left,
                                const ObjectiveVector &right,
                                std::uint32_t weightedLevel) const;

  /// Returns a negative value when left ranks before right. Equal objective
  /// values are resolved by the canonical candidate semantic key.
  llvm::Expected<int>
  compareTotalOrdering(const ObjectiveVector &left,
                       llvm::ArrayRef<std::uint8_t> leftCandidateKey,
                       const ObjectiveVector &right,
                       llvm::ArrayRef<std::uint8_t> rightCandidateKey,
                       std::uint32_t totalOrdering) const;

  llvm::Expected<ParetoRelation>
  comparePareto(const ObjectiveVector &left, const ObjectiveVector &right,
                llvm::ArrayRef<std::uint32_t> dimensions) const;

private:
  struct CompiledDimension final {
    ResolvedObjectiveScalarSource source;
    ResolvedObjectiveDirection direction;
    ResolvedObjectiveScalar origin;
    ResolvedObjectiveScalar quantum;
    std::uint64_t lowerIndex;
    std::uint64_t upperIndex;
  };

  struct CompiledTerm final {
    std::uint32_t dimension;
    std::uint64_t weight;
  };

  struct CompiledLevel final {
    std::uint32_t termOffset;
    std::uint32_t termCount;
  };

  struct CompiledOrdering final {
    std::uint32_t levelOffset;
    std::uint32_t levelCount;
  };

  std::vector<CompiledDimension> dimensions_;
  std::vector<CompiledTerm> terms_;
  std::vector<CompiledLevel> levels_;
  std::vector<std::uint32_t> orderingLevels_;
  std::vector<CompiledOrdering> orderings_;
  bool candidateMeasureProgram_ = false;
};

} // namespace loom::dse

#endif // LOOM_DSE_OBJECTIVE_H
