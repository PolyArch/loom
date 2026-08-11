#ifndef LOOM_DSE_PLANVALUE_H
#define LOOM_DSE_PLANVALUE_H

#include "Common/Artifact.h"
#include "Common/ComponentViewDigest.h"
#include "Evaluation/ModelParameterBundle.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <variant>
#include <vector>

namespace loom::dse {

struct ResolvedDseConfigViewContract final {
  llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes;
  llvm::Error (*validateCanonical)(llvm::ArrayRef<std::uint8_t> bytes,
                                   const ComponentViewDigest &digest);
};

enum class PlanValueRole : std::uint32_t {
  CandidateSet = 0,
  EvidenceSet = 1,
  SimulationExecutionSet = 2,
};

enum class PlanValueCardinality : std::uint32_t {
  ExactlyOne = 0,
  ZeroOrOne = 1,
  NonEmptySet = 2,
  FiniteSet = 3,
};

enum class CalibrationPartitionRole : std::uint32_t {
  Training = 0,
  Validation = 1,
  HeldOut = 2,
};

struct PlanCardinalityBounds final {
  std::uint64_t minimum = 0;
  std::uint64_t maximum = 0;
};

constexpr PlanCardinalityBounds
planCardinalityBounds(PlanValueCardinality cardinality) {
  switch (cardinality) {
  case PlanValueCardinality::ExactlyOne:
    return {1, 1};
  case PlanValueCardinality::ZeroOrOne:
    return {0, 1};
  case PlanValueCardinality::NonEmptySet:
    return {1, std::numeric_limits<std::uint64_t>::max()};
  case PlanValueCardinality::FiniteSet:
    return {0, std::numeric_limits<std::uint64_t>::max()};
  }
  return {1, 0};
}

constexpr bool planCardinalityContains(PlanValueCardinality cardinality,
                                       std::uint64_t count) {
  const PlanCardinalityBounds bounds = planCardinalityBounds(cardinality);
  return count >= bounds.minimum && count <= bounds.maximum;
}

constexpr bool planCardinalityCanFlow(PlanValueCardinality producer,
                                      PlanValueCardinality consumer) {
  const PlanCardinalityBounds produced = planCardinalityBounds(producer);
  const PlanCardinalityBounds accepted = planCardinalityBounds(consumer);
  return produced.minimum >= accepted.minimum &&
         produced.maximum <= accepted.maximum;
}

struct PlanOutputRef final {
  std::uint64_t producerNodeOrdinal = 0;
  std::uint32_t outputSlotOrdinal = 0;

  friend bool operator==(PlanOutputRef lhs, PlanOutputRef rhs) {
    return lhs.producerNodeOrdinal == rhs.producerNodeOrdinal &&
           lhs.outputSlotOrdinal == rhs.outputSlotOrdinal;
  }
  friend bool operator!=(PlanOutputRef lhs, PlanOutputRef rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(PlanOutputRef lhs, PlanOutputRef rhs) {
    if (lhs.producerNodeOrdinal != rhs.producerNodeOrdinal)
      return lhs.producerNodeOrdinal < rhs.producerNodeOrdinal;
    return lhs.outputSlotOrdinal < rhs.outputSlotOrdinal;
  }
};

struct ExactPlanArtifacts final {
  std::vector<ArtifactRootReference> artifacts;
};

/// Explicit finite union of prior plan outputs. The bound is semantic work
/// policy: runtime resolution canonicalizes and deduplicates the union before
/// retaining its first `maximumArtifacts` roots.
struct BoundedPlanOutputJoin final {
  std::vector<PlanOutputRef> outputs;
  std::uint64_t maximumArtifacts = 0;
};

using PlanInputBinding =
    std::variant<ExactPlanArtifacts, PlanOutputRef, BoundedPlanOutputJoin>;

struct PlanValueDescriptor final {
  PlanValueRole role;
  ArtifactSchemaDescriptor schema;
  PlanValueCardinality cardinality;
  std::optional<evaluation::ModelParameterContractRef> modelParameterContract =
      std::nullopt;
  std::optional<CalibrationPartitionRole> calibrationPartitionRole =
      std::nullopt;
};

} // namespace loom::dse

#endif // LOOM_DSE_PLANVALUE_H
