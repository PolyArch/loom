#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDPOLYHEDRALPROVIDER_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDPOLYHEDRALPROVIDER_H

#include "Frontend/Compilation/StructuredScop.h"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
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
  /// The same provider schedule with every outermost band tiled by one
  /// requested factor, in requested factor order. A factor whose tiled
  /// relation the provider could not prove valid is absent.
  std::vector<StructuredPolyhedralTiledScheduleView> tiledSchedules;
};

enum class PolyhedralScheduleProviderRefusalKind : std::uint32_t {
  DomainNotAdmitted = 0,
  ScheduleNotEstablished = 1,
  OperationBudgetExhausted = 2,
};

using PolyhedralScheduleProviderOutcome =
    std::variant<PolyhedralScheduleProviderView,
                 PolyhedralScheduleProviderRefusalKind>;

enum class PolyhedralAstExpressionKind : std::uint32_t {
  Integer = 0,
  Identifier = 1,
  And = 2,
  AndThen = 3,
  Or = 4,
  OrElse = 5,
  Maximum = 6,
  Minimum = 7,
  Negate = 8,
  Add = 9,
  Subtract = 10,
  Multiply = 11,
  Divide = 12,
  FloorDivide = 13,
  PositiveDivide = 14,
  PositiveRemainder = 15,
  ZeroRemainder = 16,
  Conditional = 17,
  Select = 18,
  Equal = 19,
  LessEqual = 20,
  Less = 21,
  GreaterEqual = 22,
  Greater = 23,
  Call = 24,
};

/// One invocation-local expression mechanically derived from the frozen
/// schedule map. It is never serialized or used as candidate identity.
struct PolyhedralAstExpression final {
  PolyhedralAstExpressionKind kind = PolyhedralAstExpressionKind::Integer;
  std::int64_t integer = 0;
  std::string identifier;
  std::vector<PolyhedralAstExpression> operands;
};

struct PolyhedralAstNode;

struct PolyhedralAstFor final {
  std::string iterator;
  PolyhedralAstExpression initial;
  PolyhedralAstExpression condition;
  PolyhedralAstExpression increment;
  std::unique_ptr<PolyhedralAstNode> body;
};

struct PolyhedralAstIf final {
  PolyhedralAstExpression condition;
  std::unique_ptr<PolyhedralAstNode> thenNode;
  std::unique_ptr<PolyhedralAstNode> elseNode;
};

struct PolyhedralAstBlock final {
  std::vector<PolyhedralAstNode> children;
};

struct PolyhedralAstUser final {
  PolyhedralAstExpression call;
};

struct PolyhedralAstNode final {
  std::variant<PolyhedralAstFor, PolyhedralAstIf, PolyhedralAstBlock,
               PolyhedralAstUser>
      value;
};

using PolyhedralAstBuildOutcome =
    std::variant<PolyhedralAstNode, StructuredScopRefusalKind>;

/// Computes one bounded ephemeral ISL schedule from MLIR-owned exact domains
/// and dependence relations, plus one tiled variant per requested factor. No
/// ISL object or spelling escapes this call.
llvm::Expected<PolyhedralScheduleProviderOutcome> computePinnedIslSchedule(
    llvm::ArrayRef<PolyhedralStatementDomain> statements,
    llvm::ArrayRef<PolyhedralDependenceRelation> dependences,
    llvm::ArrayRef<std::uint64_t> tileFactors = {});

/// Reconstructs and independently checks the frozen schedule relation before
/// deriving the bounded ISL AST consumed by ordinary-MLIR materialization.
llvm::Expected<PolyhedralAstBuildOutcome>
buildPinnedIslAst(const StructuredPolyhedralScopView &scop);

} // namespace loom::frontend::detail

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDPOLYHEDRALPROVIDER_H
