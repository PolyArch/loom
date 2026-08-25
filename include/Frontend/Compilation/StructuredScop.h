#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDSCOP_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDSCOP_H

#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::frontend {

enum class StructuredScopRefusalKind : std::uint32_t {
  NotAffineLoop = 0,
  NestedAffineRoot = 1,
  NonCanonicalIterationDomain = 2,
  DomainProofNotEstablished = 3,
  NestedControl = 4,
  UnsupportedEffect = 5,
  UnsupportedOperation = 6,
  AccessRelationProofNotEstablished = 7,
  NonContiguousAccess = 8,
  AliasProofNotEstablished = 9,
  DependenceProofNotEstablished = 10,
  LoopCarriedMemoryDependence = 11,
  AlignmentProofNotEstablished = 12,
  UnsupportedReduction = 13,
  StrictFloatingReduction = 14,
  ProviderMaterializationRejected = 15,
  FabricCapabilityUnavailable = 16,
  UnsupportedTail = 17,
  NonUnitPhysicalStride = 18,
  HeterogeneousElementWidth = 19,
  IntegerOverflowReduction = 20,
  NonLocalMemoryRoot = 21,
  VectorLoweringUnavailable = 22,
  UnsupportedPhysicalOffset = 23,
};

enum class StructuredScopAccessKind : std::uint32_t {
  Read = 0,
  Write = 1,
};

enum class StructuredReductionSchedule : std::uint32_t {
  None = 0,
  IntegerAssociative = 1,
  FloatingReassociated = 2,
};

struct StructuredScopAccessView final {
  StructuredScopAccessKind kind;
  std::uint64_t statementOrdinal = 0;
  std::uint64_t relationDimensionCount = 0;
  std::uint64_t relationSymbolCount = 0;
  std::uint64_t relationConstraintCount = 0;
  std::uint64_t elementBytes = 0;
  std::uint64_t alignmentBytes = 0;
  std::uint64_t memoryBoundaryArgument = 0;
  std::optional<std::uint64_t> storedStatementOrdinal;
};

struct StructuredScopComputeView final {
  std::uint64_t statementOrdinal = 0;
  dataflow::OperationSchemaId schema;
  dataflow::SemanticPayload payload;
  std::vector<std::optional<std::uint64_t>> operandStatements;
};

/// Invocation-local proof summary for the closed exact vector SCoP domain.
/// The MLIR Affine/Presburger relations remain transient provider values; this
/// view records which exact source entities those providers proved and the
/// typed facts consumed by canonical schedule generation.
struct ExactStructuredScopView final {
  explicit ExactStructuredScopView(StructuredEntityRef loop)
      : loop(std::move(loop)) {}

  StructuredEntityRef loop;
  std::string ownerSymbol;
  std::uint64_t loopOrdinalInOwner = 0;
  std::uint64_t statementCount = 0;
  std::uint64_t parameterCount = 0;
  std::uint64_t domainConstraintCount = 0;
  std::vector<StructuredScopAccessView> accesses;
  std::vector<StructuredScopComputeView> computes;
  StructuredReductionSchedule reductionSchedule =
      StructuredReductionSchedule::None;
  std::uint64_t reductionCount = 0;
  std::optional<mlir::arith::AtomicRMWKind> reductionKind;
  std::uint64_t minimumAlignmentBytes = 0;
  std::uint64_t maximumElementBytes = 0;
  std::optional<std::uint64_t> constantTripCount;
};

struct StructuredScopRefusal final {
  StructuredEntityRef loop;
  StructuredScopRefusalKind kind;
};

using StructuredScopAnalysisOutcome =
    std::variant<ExactStructuredScopView, StructuredScopRefusal>;

/// Projects one exact affine loop through MLIR Affine/Presburger dependence
/// and access analysis plus MLIR alias analysis. The admitted domain is a
/// rank-one, zero-based, unit-stride SCoP with direct contiguous affine memory
/// accesses, explicit alignment, no nested control, and only provider-proven
/// parallel dependences or supported reductions.
llvm::Expected<StructuredScopAnalysisOutcome>
analyzeExactStructuredScop(const StructuredProgramCandidate &parent,
                           const StructuredEntityRef &loop);

/// Mechanically projects one selected SCF loop and its direct memref accesses
/// into the Affine spelling consumed by the upstream analysis and vectorizer.
/// The caller owns a private clone; sibling operations are not rewritten.
llvm::Expected<mlir::affine::AffineForOp>
projectExactStructuredScopToAffine(mlir::Operation *loop);

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDSCOP_H
