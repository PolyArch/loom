#ifndef LOOM_DSE_PLANVALUE_H
#define LOOM_DSE_PLANVALUE_H

#include "Common/Artifact.h"
#include "Common/ComponentViewDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
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
};

struct ExactPlanArtifacts final {
  std::vector<ArtifactRootReference> artifacts;
};

using PlanInputBinding = std::variant<ExactPlanArtifacts, PlanOutputRef>;

struct PlanValueDescriptor final {
  PlanValueRole role;
  ArtifactSchemaDescriptor schema;
  PlanValueCardinality cardinality;
};

} // namespace loom::dse

#endif // LOOM_DSE_PLANVALUE_H
