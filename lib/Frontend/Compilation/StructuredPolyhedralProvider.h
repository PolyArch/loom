#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDPOLYHEDRALPROVIDER_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDPOLYHEDRALPROVIDER_H

#include "Frontend/Compilation/StructuredScop.h"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace loom::frontend::detail {

inline constexpr std::uint64_t maximumPinnedIslStatementCount = 1024;

struct PolyhedralStatementDomain final {
  std::uint64_t statementOrdinal = 0;
  const mlir::affine::FlatAffineValueConstraints *domain = nullptr;
};

struct PolyhedralDependenceRelation final {
  std::uint64_t sourceStatementOrdinal = 0;
  std::uint64_t destinationStatementOrdinal = 0;
  std::uint64_t sourceDimensionCount = 0;
  std::uint64_t destinationDimensionCount = 0;
  /// Null denotes exact same-iteration precedence over the common loop
  /// prefix. A non-null relation is the MLIR dependence polyhedron.
  const mlir::affine::FlatAffineValueConstraints *relation = nullptr;
};

struct PolyhedralScheduleProviderView final {
  std::uint64_t parameterCount = 0;
  StructuredPolyhedralScheduleForm form =
      StructuredPolyhedralScheduleForm::General;
  std::uint64_t scheduleBandCount = 0;
  std::uint64_t scheduleDimensionCount = 0;
  std::uint64_t coincidentDimensionCount = 0;
  std::vector<StructuredPolyhedralStatementScheduleView> statementSchedules;
  /// Exact parameter order used by the invocation-local ISL spaces.
  std::vector<mlir::Value> parameters;
};

enum class PolyhedralScheduleProviderRefusalKind : std::uint32_t {
  DomainNotAdmitted = 0,
  ScheduleNotEstablished = 1,
  OperationBudgetExhausted = 2,
};

using PolyhedralScheduleProviderOutcome =
    std::variant<PolyhedralScheduleProviderView,
                 PolyhedralScheduleProviderRefusalKind>;

/// Computes one bounded ephemeral ISL schedule from MLIR-owned exact domains
/// and dependence relations. No ISL object or spelling escapes this call.
llvm::Expected<PolyhedralScheduleProviderOutcome> computePinnedIslSchedule(
    llvm::ArrayRef<PolyhedralStatementDomain> statements,
    llvm::ArrayRef<PolyhedralDependenceRelation> dependences);

} // namespace loom::frontend::detail

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDPOLYHEDRALPROVIDER_H
