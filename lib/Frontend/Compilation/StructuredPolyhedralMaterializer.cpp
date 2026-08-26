#include "StructuredPolyhedralMaterializer.h"

#include "StructuredPolyhedralProvider.h"

#include "Common/IndexWidth.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::frontend::detail {
namespace {

llvm::Error materializerError(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "structured_polyhedral_materializer_invalid: " + message);
}

bool isComparison(PolyhedralAstExpressionKind kind) {
  switch (kind) {
  case PolyhedralAstExpressionKind::Equal:
  case PolyhedralAstExpressionKind::LessEqual:
  case PolyhedralAstExpressionKind::Less:
  case PolyhedralAstExpressionKind::GreaterEqual:
  case PolyhedralAstExpressionKind::Greater:
    return true;
  default:
    return false;
  }
}

bool validateExpression(const PolyhedralAstExpression &expression,
                        bool allowCall, bool allowZeroRemainder = false) {
  const std::size_t count = expression.operands.size();
  switch (expression.kind) {
  case PolyhedralAstExpressionKind::Integer:
    return expression.identifier.empty() && count == 0;
  case PolyhedralAstExpressionKind::Identifier:
    return !expression.identifier.empty() && count == 0;
  case PolyhedralAstExpressionKind::Negate:
    break;
  case PolyhedralAstExpressionKind::Maximum:
  case PolyhedralAstExpressionKind::Minimum:
    if (count == 0)
      return false;
    break;
  case PolyhedralAstExpressionKind::Select:
  case PolyhedralAstExpressionKind::Conditional:
    if (count != 3)
      return false;
    break;
  case PolyhedralAstExpressionKind::Call:
    if (!allowCall || count == 0 ||
        expression.operands.front().kind !=
            PolyhedralAstExpressionKind::Identifier)
      return false;
    break;
  case PolyhedralAstExpressionKind::ZeroRemainder:
    if (!allowZeroRemainder)
      return false;
    [[fallthrough]];
  default:
    if (count != 2)
      return false;
    break;
  }
  if (expression.kind == PolyhedralAstExpressionKind::Negate && count != 1)
    return false;
  const bool positiveConstantDivisor =
      expression.kind == PolyhedralAstExpressionKind::Divide ||
      expression.kind == PolyhedralAstExpressionKind::FloorDivide ||
      expression.kind == PolyhedralAstExpressionKind::PositiveDivide ||
      expression.kind == PolyhedralAstExpressionKind::PositiveRemainder ||
      expression.kind == PolyhedralAstExpressionKind::ZeroRemainder;
  if (positiveConstantDivisor &&
      (expression.operands[1].kind != PolyhedralAstExpressionKind::Integer ||
       expression.operands[1].integer <= 0))
    return false;
  for (std::size_t index = 0; index != count; ++index) {
    const bool zeroRemainderOperand =
        expression.kind == PolyhedralAstExpressionKind::Equal &&
        expression.operands[1 - index].kind ==
            PolyhedralAstExpressionKind::Integer &&
        expression.operands[1 - index].integer == 0;
    if (!validateExpression(expression.operands[index], false,
                            zeroRemainderOperand))
      return false;
  }
  return true;
}

bool expressionReturnsBoolean(const PolyhedralAstExpression &expression) {
  if (isComparison(expression.kind) ||
      expression.kind == PolyhedralAstExpressionKind::And ||
      expression.kind == PolyhedralAstExpressionKind::AndThen ||
      expression.kind == PolyhedralAstExpressionKind::Or ||
      expression.kind == PolyhedralAstExpressionKind::OrElse)
    return true;
  if ((expression.kind == PolyhedralAstExpressionKind::Select ||
       expression.kind == PolyhedralAstExpressionKind::Conditional) &&
      expression.operands.size() == 3)
    return expressionReturnsBoolean(expression.operands[1]) &&
           expressionReturnsBoolean(expression.operands[2]);
  return false;
}

bool expressionsEqual(const PolyhedralAstExpression &lhs,
                      const PolyhedralAstExpression &rhs) {
  if (lhs.kind != rhs.kind || lhs.integer != rhs.integer ||
      lhs.identifier != rhs.identifier ||
      lhs.operands.size() != rhs.operands.size())
    return false;
  return llvm::all_of(
      llvm::zip_equal(lhs.operands, rhs.operands), [](const auto &pair) {
        return expressionsEqual(std::get<0>(pair), std::get<1>(pair));
      });
}

std::optional<std::uint64_t> statementOrdinal(llvm::StringRef identifier) {
  if (!identifier.consume_front("S") || identifier.empty())
    return std::nullopt;
  std::uint64_t ordinal = 0;
  if (identifier.getAsInteger(10, ordinal))
    return std::nullopt;
  return ordinal;
}

bool validateAst(const PolyhedralAstNode &node,
                 const StructuredPolyhedralScopView &scop) {
  if (const auto *loop = std::get_if<PolyhedralAstFor>(&node.value)) {
    if (loop->iterator.empty() || !loop->body ||
        !validateExpression(loop->initial, false) ||
        !validateExpression(loop->increment, false) ||
        !validateExpression(loop->condition, false) ||
        loop->increment.kind != PolyhedralAstExpressionKind::Integer ||
        loop->increment.integer <= 0 ||
        (loop->condition.kind != PolyhedralAstExpressionKind::Less &&
         loop->condition.kind != PolyhedralAstExpressionKind::LessEqual) ||
        loop->condition.operands.size() != 2 ||
        loop->condition.operands.front().kind !=
            PolyhedralAstExpressionKind::Identifier ||
        loop->condition.operands.front().identifier != loop->iterator)
      return false;
    return validateAst(*loop->body, scop);
  }
  if (const auto *branch = std::get_if<PolyhedralAstIf>(&node.value)) {
    return branch->thenNode && validateExpression(branch->condition, false) &&
           expressionReturnsBoolean(branch->condition) &&
           validateAst(*branch->thenNode, scop) &&
           (!branch->elseNode || validateAst(*branch->elseNode, scop));
  }
  if (const auto *block = std::get_if<PolyhedralAstBlock>(&node.value))
    return llvm::all_of(block->children, [&](const auto &child) {
      return validateAst(child, scop);
    });
  const auto &user = std::get<PolyhedralAstUser>(node.value);
  if (!validateExpression(user.call, true))
    return false;
  const auto ordinal = statementOrdinal(user.call.operands.front().identifier);
  return ordinal && *ordinal < scop.statements.size() &&
         user.call.operands.size() ==
             scop.statements[*ordinal].domain.dimensions.size() + 1;
}

void collectExpressionIdentifiers(const PolyhedralAstExpression &expression,
                                  llvm::StringSet<> &identifiers) {
  if (expression.kind == PolyhedralAstExpressionKind::Identifier)
    identifiers.insert(expression.identifier);
  for (const PolyhedralAstExpression &operand : expression.operands)
    collectExpressionIdentifiers(operand, identifiers);
}

void collectAstIdentifiers(const PolyhedralAstNode &node,
                           llvm::StringSet<> &identifiers) {
  if (const auto *loop = std::get_if<PolyhedralAstFor>(&node.value)) {
    collectExpressionIdentifiers(loop->initial, identifiers);
    collectExpressionIdentifiers(loop->condition, identifiers);
    collectExpressionIdentifiers(loop->increment, identifiers);
    collectAstIdentifiers(*loop->body, identifiers);
    return;
  }
  if (const auto *branch = std::get_if<PolyhedralAstIf>(&node.value)) {
    collectExpressionIdentifiers(branch->condition, identifiers);
    collectAstIdentifiers(*branch->thenNode, identifiers);
    if (branch->elseNode)
      collectAstIdentifiers(*branch->elseNode, identifiers);
    return;
  }
  if (const auto *block = std::get_if<PolyhedralAstBlock>(&node.value)) {
    for (const PolyhedralAstNode &child : block->children)
      collectAstIdentifiers(child, identifiers);
    return;
  }
  collectExpressionIdentifiers(std::get<PolyhedralAstUser>(node.value).call,
                               identifiers);
}

struct PlannedStatementValue final {
  std::vector<PolyhedralAstExpression> coordinates;
  std::vector<StructuredEntityRef> dimensions;
};

struct SignedInterval final {
  __int128 lower = 0;
  __int128 upper = 0;
};

struct PreflightScope final {
  llvm::StringMap<SignedInterval> identifierRanges;
  llvm::DenseMap<std::uint64_t, std::vector<PlannedStatementValue>>
      statementValues;
};

class AstPreflight final {
public:
  AstPreflight(const StructuredPolyhedralScopView &scop, unsigned indexWidth)
      : scop_(scop), indexWidth_(indexWidth), scheduleWidth_(indexWidth) {}

  std::optional<unsigned> run(const PolyhedralAstNode &ast) {
    if (indexWidth_ != 32 && indexWidth_ != 64)
      return std::nullopt;
    PreflightScope scope;
    for (std::uint64_t ordinal = 0; ordinal != scop_.parameters.size();
         ++ordinal)
      scope.identifierRanges.try_emplace("p" + std::to_string(ordinal),
                                         fullSignedRange(indexWidth_));
    if (!visit(ast, scope))
      return std::nullopt;
    return scheduleWidth_;
  }

private:
  static SignedInterval fullSignedRange(unsigned width) {
    const __int128 magnitude = static_cast<__int128>(1) << (width - 1);
    return {-magnitude, magnitude - 1};
  }

  static bool fitsSigned(const SignedInterval &range, unsigned width) {
    const SignedInterval limits = fullSignedRange(width);
    return range.lower >= limits.lower && range.upper <= limits.upper;
  }

  bool observe(const SignedInterval &range) {
    if (range.lower > range.upper || !fitsSigned(range, 64))
      return false;
    if (!fitsSigned(range, 32))
      scheduleWidth_ = 64;
    return true;
  }

  static __int128 floorDivide(__int128 value, __int128 divisor) {
    __int128 quotient = value / divisor;
    if (value % divisor < 0)
      --quotient;
    return quotient;
  }

  std::optional<SignedInterval>
  expressionRange(const PolyhedralAstExpression &expression,
                  const PreflightScope &scope) {
    if (expression.kind == PolyhedralAstExpressionKind::Integer) {
      SignedInterval range{expression.integer, expression.integer};
      return observe(range) ? std::optional<SignedInterval>(range)
                            : std::nullopt;
    }
    if (expression.kind == PolyhedralAstExpressionKind::Identifier) {
      auto found = scope.identifierRanges.find(expression.identifier);
      if (found == scope.identifierRanges.end() || !observe(found->second))
        return std::nullopt;
      return found->second;
    }
    if (expression.kind == PolyhedralAstExpressionKind::Call)
      return std::nullopt;
    std::vector<SignedInterval> operands;
    operands.reserve(expression.operands.size());
    for (const PolyhedralAstExpression &operand : expression.operands) {
      auto range = expressionRange(operand, scope);
      if (!range)
        return std::nullopt;
      operands.push_back(*range);
    }
    std::optional<SignedInterval> result;
    switch (expression.kind) {
    case PolyhedralAstExpressionKind::And:
    case PolyhedralAstExpressionKind::AndThen:
    case PolyhedralAstExpressionKind::Or:
    case PolyhedralAstExpressionKind::OrElse:
    case PolyhedralAstExpressionKind::Equal:
    case PolyhedralAstExpressionKind::LessEqual:
    case PolyhedralAstExpressionKind::Less:
    case PolyhedralAstExpressionKind::GreaterEqual:
    case PolyhedralAstExpressionKind::Greater:
      result = SignedInterval{0, 1};
      break;
    case PolyhedralAstExpressionKind::Negate:
      result = SignedInterval{-operands.front().upper, -operands.front().lower};
      break;
    case PolyhedralAstExpressionKind::Add:
      result = SignedInterval{operands[0].lower + operands[1].lower,
                              operands[0].upper + operands[1].upper};
      break;
    case PolyhedralAstExpressionKind::Subtract:
      result = SignedInterval{operands[0].lower - operands[1].upper,
                              operands[0].upper - operands[1].lower};
      break;
    case PolyhedralAstExpressionKind::Multiply: {
      const std::array<__int128, 4> products = {
          operands[0].lower * operands[1].lower,
          operands[0].lower * operands[1].upper,
          operands[0].upper * operands[1].lower,
          operands[0].upper * operands[1].upper};
      result =
          SignedInterval{*std::min_element(products.begin(), products.end()),
                         *std::max_element(products.begin(), products.end())};
      break;
    }
    case PolyhedralAstExpressionKind::Divide:
      result = SignedInterval{operands[0].lower / operands[1].lower,
                              operands[0].upper / operands[1].lower};
      break;
    case PolyhedralAstExpressionKind::FloorDivide:
      result =
          SignedInterval{floorDivide(operands[0].lower, operands[1].lower),
                         floorDivide(operands[0].upper, operands[1].lower)};
      break;
    case PolyhedralAstExpressionKind::PositiveDivide:
      result = SignedInterval{0, std::max<__int128>(0, operands[0].upper) /
                                     operands[1].lower};
      break;
    case PolyhedralAstExpressionKind::PositiveRemainder:
      result = SignedInterval{0, operands[1].lower - 1};
      break;
    case PolyhedralAstExpressionKind::ZeroRemainder:
      result = SignedInterval{1 - operands[1].lower, operands[1].lower - 1};
      break;
    case PolyhedralAstExpressionKind::Maximum: {
      __int128 lower = operands.front().lower;
      __int128 upper = operands.front().upper;
      for (const SignedInterval &operand : llvm::drop_begin(operands)) {
        lower = std::max(lower, operand.lower);
        upper = std::max(upper, operand.upper);
      }
      result = SignedInterval{lower, upper};
      break;
    }
    case PolyhedralAstExpressionKind::Minimum: {
      __int128 lower = operands.front().lower;
      __int128 upper = operands.front().upper;
      for (const SignedInterval &operand : llvm::drop_begin(operands)) {
        lower = std::min(lower, operand.lower);
        upper = std::min(upper, operand.upper);
      }
      result = SignedInterval{lower, upper};
      break;
    }
    case PolyhedralAstExpressionKind::Conditional:
    case PolyhedralAstExpressionKind::Select:
      result = SignedInterval{std::min(operands[1].lower, operands[2].lower),
                              std::max(operands[1].upper, operands[2].upper)};
      break;
    case PolyhedralAstExpressionKind::Integer:
    case PolyhedralAstExpressionKind::Identifier:
    case PolyhedralAstExpressionKind::Call:
      return std::nullopt;
    }
    if (!result || !observe(*result))
      return std::nullopt;
    return *result;
  }

  bool
  scalarInputsAvailable(std::uint64_t ordinal,
                        llvm::ArrayRef<PolyhedralAstExpression> coordinates,
                        const PreflightScope &scope) const {
    for (const StructuredPolyhedralDependenceView &dependence :
         scop_.dependences) {
      if (dependence.kind != StructuredPolyhedralDependenceKind::ScalarSsa ||
          dependence.destinationStatementOrdinal != ordinal)
        continue;
      auto materialized =
          scope.statementValues.find(dependence.sourceStatementOrdinal);
      if (materialized == scope.statementValues.end())
        return false;
      const auto &consumerDimensions =
          scop_.statements[ordinal].domain.dimensions;
      const bool matches = llvm::any_of(
          llvm::reverse(materialized->second),
          [&](const PlannedStatementValue &candidate) {
            if (candidate.coordinates.size() > coordinates.size() ||
                candidate.dimensions.size() > consumerDimensions.size() ||
                !std::equal(candidate.dimensions.begin(),
                            candidate.dimensions.end(),
                            consumerDimensions.begin()))
              return false;
            return llvm::all_of(
                llvm::zip_equal(
                    candidate.coordinates,
                    coordinates.take_front(candidate.coordinates.size())),
                [](const auto &pair) {
                  return expressionsEqual(std::get<0>(pair), std::get<1>(pair));
                });
          });
      if (!matches)
        return false;
    }
    return true;
  }

  bool visit(const PolyhedralAstNode &node, PreflightScope &scope) {
    if (const auto *loop = std::get_if<PolyhedralAstFor>(&node.value)) {
      auto lower = expressionRange(loop->initial, scope);
      auto step = expressionRange(loop->increment, scope);
      auto upper = expressionRange(loop->condition.operands[1], scope);
      if (!lower || !step || !upper)
        return false;
      const __int128 stepValue = loop->increment.integer;
      SignedInterval iteratorRange{lower->lower,
                                   std::max(lower->upper, upper->upper)};
      if (!observe(iteratorRange))
        return false;
      const bool inclusive =
          loop->condition.kind == PolyhedralAstExpressionKind::LessEqual;
      const SignedInterval exclusiveUpper =
          inclusive ? SignedInterval{upper->lower + 1, upper->upper + 1}
                    : *upper;
      if (!observe(exclusiveUpper))
        return false;
      SignedInterval finalIncrement{iteratorRange.lower + stepValue,
                                    upper->upper + stepValue -
                                        (inclusive ? 0 : 1)};
      if (finalIncrement.lower > finalIncrement.upper)
        std::swap(finalIncrement.lower, finalIncrement.upper);
      if (!observe(finalIncrement))
        return false;
      PreflightScope nested = scope;
      nested.identifierRanges.insert_or_assign(loop->iterator, iteratorRange);
      if (!expressionRange(loop->condition, nested))
        return false;
      return visit(*loop->body, nested);
    }
    if (const auto *branch = std::get_if<PolyhedralAstIf>(&node.value)) {
      if (!expressionRange(branch->condition, scope))
        return false;
      PreflightScope thenScope = scope;
      if (!visit(*branch->thenNode, thenScope))
        return false;
      if (branch->elseNode) {
        PreflightScope elseScope = scope;
        if (!visit(*branch->elseNode, elseScope))
          return false;
      }
      return true;
    }
    if (const auto *block = std::get_if<PolyhedralAstBlock>(&node.value)) {
      for (const PolyhedralAstNode &child : block->children)
        if (!visit(child, scope))
          return false;
      return true;
    }
    const auto &user = std::get<PolyhedralAstUser>(node.value);
    const auto ordinal =
        statementOrdinal(user.call.operands.front().identifier);
    if (!ordinal || *ordinal >= scop_.statements.size())
      return false;
    llvm::ArrayRef<PolyhedralAstExpression> coordinates(user.call.operands);
    coordinates = coordinates.drop_front();
    for (const PolyhedralAstExpression &coordinate : coordinates) {
      auto range = expressionRange(coordinate, scope);
      if (!range || !fitsSigned(*range, indexWidth_))
        return false;
    }
    if (!scalarInputsAvailable(*ordinal, coordinates, scope))
      return false;
    scope.statementValues[*ordinal].push_back(
        PlannedStatementValue{std::vector<PolyhedralAstExpression>(
                                  coordinates.begin(), coordinates.end()),
                              scop_.statements[*ordinal].domain.dimensions});
    return true;
  }

  const StructuredPolyhedralScopView &scop_;
  unsigned indexWidth_ = 0;
  unsigned scheduleWidth_ = 0;
};

struct MaterializationPlan final {
  PolyhedralAstNode ast;
  unsigned scheduleWidth = 0;
};

using MaterializationPlanOutcome =
    std::variant<MaterializationPlan, StructuredScopRefusalKind>;

llvm::Expected<MaterializationPlanOutcome>
buildMaterializationPlan(mlir::Operation *root,
                         const StructuredPolyhedralScopView &scop) {
  auto astOutcome = buildPinnedIslAst(scop);
  if (!astOutcome)
    return astOutcome.takeError();
  if (auto *providerRefusal =
          std::get_if<StructuredScopRefusalKind>(&*astOutcome))
    return *providerRefusal;
  PolyhedralAstNode ast = std::move(std::get<PolyhedralAstNode>(*astOutcome));
  if (!validateAst(ast, scop))
    return StructuredScopRefusalKind::PolyhedralMaterializationUnavailable;
  auto indexWidth = getIndexBitWidth(root);
  if (!indexWidth)
    return indexWidth.takeError();
  auto scheduleWidth = AstPreflight(scop, *indexWidth).run(ast);
  if (!scheduleWidth)
    return StructuredScopRefusalKind::PolyhedralMaterializationUnavailable;
  return MaterializationPlan{std::move(ast), *scheduleWidth};
}

bool valueDefinedInside(mlir::Value value, mlir::Operation *root) {
  mlir::Region *region = value.getParentRegion();
  if (!region || root->getNumRegions() == 0)
    return false;
  mlir::Region &rootRegion = root->getRegion(0);
  return region == &rootRegion || rootRegion.isAncestor(region);
}

struct MaterializedStatementValue final {
  mlir::Value value;
  std::vector<PolyhedralAstExpression> coordinates;
  std::vector<StructuredEntityRef> dimensions;
};

struct MaterializationScope final {
  llvm::StringMap<mlir::Value> iterators;
  llvm::DenseMap<mlir::Value, std::vector<MaterializedStatementValue>>
      statementValues;
};

class AstMaterializer final {
public:
  AstMaterializer(
      mlir::Operation *root, const StructuredPolyhedralScopView &scop,
      const StructuredProgramCandidateView &parentView,
      const mlir::IRMapping &cloneMapping, unsigned scheduleWidth,
      llvm::SmallVectorImpl<mlir::Operation *> &materializedOperations)
      : root_(root), scop_(scop), parentView_(parentView),
        cloneMapping_(cloneMapping), builder_(root), location_(root->getLoc()),
        scheduleType_(builder_.getIntegerType(scheduleWidth)),
        materializedOperations_(materializedOperations) {}

  llvm::Expected<bool> run(const PolyhedralAstNode &ast) {
    if (llvm::Error error = prepare())
      return std::move(error);
    mlir::Operation *previous = root_->getPrevNode();
    MaterializationScope scope;
    llvm::StringSet<> referencedIdentifiers;
    collectAstIdentifiers(ast, referencedIdentifiers);
    for (auto [ordinal, parameter] : llvm::enumerate(parameters_)) {
      const std::string identifier = "p" + std::to_string(ordinal);
      if (!referencedIdentifiers.contains(identifier))
        continue;
      mlir::Value extended = mlir::arith::IndexCastOp::create(
          builder_, location_, scheduleType_, parameter);
      scope.iterators.try_emplace(identifier, extended);
    }
    if (llvm::Error error = emitNode(ast, scope)) {
      bool unavailable = false;
      llvm::Error remaining = llvm::handleErrors(
          std::move(error),
          [&](const ScalarExpansionUnavailable &) { unavailable = true; });
      if (remaining)
        return std::move(remaining);
      if (unavailable)
        return false;
    }
    mlir::Operation *operation =
        previous ? previous->getNextNode() : &root_->getBlock()->front();
    while (operation != root_) {
      operation->walk([&](mlir::Operation *nested) {
        materializedOperations_.push_back(nested);
      });
      operation = operation->getNextNode();
    }
    root_->erase();
    return true;
  }

private:
  class ScalarExpansionUnavailable final
      : public llvm::ErrorInfo<ScalarExpansionUnavailable> {
  public:
    static char ID;
    void log(llvm::raw_ostream &stream) const override {
      stream << "polyhedral schedule requires scalar expansion";
    }
    std::error_code convertToErrorCode() const override {
      return llvm::inconvertibleErrorCode();
    }
  };

  llvm::Error prepare() {
    if (!root_ || root_->getNumRegions() != 1 ||
        scop_.statements.size() != scop_.schedule.statementSchedules.size())
      return materializerError(
          "the SCoP structure changed before materialization");
    statements_.reserve(scop_.statements.size());
    dimensions_.reserve(scop_.statements.size());
    for (const StructuredPolyhedralStatementView &statement :
         scop_.statements) {
      auto source = parentView_.resolve(statement.operation);
      if (!source)
        return source.takeError();
      mlir::Operation *cloned = cloneMapping_.lookupOrNull(source->operation);
      if (!cloned || !root_->isAncestor(cloned) || cloned->getNumRegions() != 0)
        return materializerError(
            "a frozen statement is absent from the selected loop clone");
      statements_.push_back(cloned);
      std::vector<mlir::Value> dimensions;
      dimensions.reserve(statement.domain.dimensions.size());
      for (const StructuredEntityRef &dimension : statement.domain.dimensions) {
        auto sourceDimension = parentView_.resolve(dimension);
        if (!sourceDimension)
          return sourceDimension.takeError();
        mlir::Value clonedDimension =
            cloneMapping_.lookupOrNull(sourceDimension->value);
        if (!clonedDimension || !valueDefinedInside(clonedDimension, root_))
          return materializerError(
              "a frozen statement dimension is absent from the loop clone");
        dimensions.push_back(clonedDimension);
      }
      dimensions_.push_back(std::move(dimensions));
    }
    parameters_.reserve(scop_.parameters.size());
    for (const StructuredEntityRef &parameter : scop_.parameters) {
      auto source = parentView_.resolve(parameter);
      if (!source)
        return source.takeError();
      mlir::Value cloned = cloneMapping_.lookupOrNull(source->value);
      if (!cloned || valueDefinedInside(cloned, root_) ||
          !llvm::isa<mlir::IndexType>(cloned.getType()))
        return materializerError(
            "a frozen schedule parameter is not an external index value");
      parameters_.push_back(cloned);
    }
    return llvm::Error::success();
  }

  llvm::Expected<mlir::Value>
  emitExpression(const PolyhedralAstExpression &expression,
                 MaterializationScope &scope) {
    if (expression.kind == PolyhedralAstExpressionKind::Integer)
      return mlir::arith::ConstantIntOp::create(
                 builder_, location_, scheduleType_, expression.integer)
          .getResult();
    if (expression.kind == PolyhedralAstExpressionKind::Identifier) {
      auto found = scope.iterators.find(expression.identifier);
      if (found == scope.iterators.end())
        return materializerError("an AST identifier has no materialized value");
      return found->second;
    }
    if (expression.kind == PolyhedralAstExpressionKind::Call)
      return materializerError(
          "a statement call was used as a value expression");

    if (expression.kind == PolyhedralAstExpressionKind::AndThen ||
        expression.kind == PolyhedralAstExpressionKind::OrElse ||
        expression.kind == PolyhedralAstExpressionKind::Conditional) {
      const bool conditional =
          expression.kind == PolyhedralAstExpressionKind::Conditional;
      if (expression.operands.size() != (conditional ? 3 : 2))
        return materializerError("a lazy AST expression has invalid arity");
      auto condition = emitExpression(expression.operands.front(), scope);
      if (!condition)
        return condition.takeError();
      if (!condition->getType().isInteger(1))
        return materializerError("a lazy AST condition is not boolean");
      const bool booleanResult =
          !conditional || (expressionReturnsBoolean(expression.operands[1]) &&
                           expressionReturnsBoolean(expression.operands[2]));
      const bool mixedConditional =
          conditional && (expressionReturnsBoolean(expression.operands[1]) !=
                          expressionReturnsBoolean(expression.operands[2]));
      if (mixedConditional)
        return materializerError("a conditional AST expression changes type");
      mlir::Type resultType = booleanResult ? mlir::Type(builder_.getI1Type())
                                            : mlir::Type(scheduleType_);
      auto branch = mlir::scf::IfOp::create(
          builder_, location_, mlir::TypeRange{resultType}, *condition,
          /*addThenBlock=*/true, /*addElseBlock=*/true);
      const PolyhedralAstExpression *thenExpression = nullptr;
      const PolyhedralAstExpression *elseExpression = nullptr;
      bool thenConstant = false;
      bool elseConstant = false;
      if (conditional) {
        thenExpression = &expression.operands[1];
        elseExpression = &expression.operands[2];
      } else if (expression.kind == PolyhedralAstExpressionKind::AndThen) {
        thenExpression = &expression.operands[1];
      } else {
        thenConstant = true;
        elseExpression = &expression.operands[1];
      }
      const auto materializeBranch =
          [&](mlir::Block &block,
              const PolyhedralAstExpression *selectedExpression,
              bool constant) -> llvm::Error {
        mlir::OpBuilder::InsertionGuard guard(builder_);
        builder_.setInsertionPoint(block.getTerminator());
        mlir::Value value;
        if (selectedExpression) {
          auto selected = emitExpression(*selectedExpression, scope);
          if (!selected)
            return selected.takeError();
          value = *selected;
        } else {
          value = mlir::arith::ConstantIntOp::create(builder_, location_,
                                                     constant, 1);
        }
        if (value.getType() != resultType)
          return materializerError("a lazy AST branch changes result type");
        block.getTerminator()->setOperands(value);
        return llvm::Error::success();
      };
      if (llvm::Error error = materializeBranch(branch.getThenRegion().front(),
                                                thenExpression, thenConstant))
        return std::move(error);
      if (llvm::Error error = materializeBranch(branch.getElseRegion().front(),
                                                elseExpression, elseConstant))
        return std::move(error);
      return branch.getResult(0);
    }

    std::vector<mlir::Value> operands;
    operands.reserve(expression.operands.size());
    for (const PolyhedralAstExpression &operand : expression.operands) {
      auto value = emitExpression(operand, scope);
      if (!value)
        return value.takeError();
      operands.push_back(*value);
    }
    const auto requireScheduleOperands = [&]() -> llvm::Error {
      if (llvm::any_of(operands, [](mlir::Value value) {
            return !value.getType().isIntOrIndex();
          }))
        return materializerError(
            "an arithmetic AST expression is not integer typed");
      if (llvm::any_of(operands, [&](mlir::Value value) {
            return value.getType() != scheduleType_;
          }))
        return materializerError(
            "an arithmetic AST expression changed schedule width");
      return llvm::Error::success();
    };
    const auto binaryIndex = [&]() -> llvm::Error {
      if (operands.size() != 2)
        return materializerError("a binary AST expression has invalid arity");
      return requireScheduleOperands();
    };

    switch (expression.kind) {
    case PolyhedralAstExpressionKind::And:
    case PolyhedralAstExpressionKind::Or: {
      if (operands.size() != 2 || llvm::any_of(operands, [](mlir::Value value) {
            return !value.getType().isInteger(1);
          }))
        return materializerError(
            "a boolean AST expression has invalid operands");
      return expression.kind == PolyhedralAstExpressionKind::And
                 ? mlir::arith::AndIOp::create(builder_, location_, operands[0],
                                               operands[1])
                       .getResult()
                 : mlir::arith::OrIOp::create(builder_, location_, operands[0],
                                              operands[1])
                       .getResult();
    }
    case PolyhedralAstExpressionKind::Maximum:
    case PolyhedralAstExpressionKind::Minimum: {
      if (operands.empty())
        return materializerError("a min/max AST expression has no operand");
      if (llvm::Error error = requireScheduleOperands())
        return std::move(error);
      mlir::Value result = operands.front();
      for (mlir::Value operand : llvm::drop_begin(operands))
        result = expression.kind == PolyhedralAstExpressionKind::Maximum
                     ? mlir::arith::MaxSIOp::create(builder_, location_, result,
                                                    operand)
                           .getResult()
                     : mlir::arith::MinSIOp::create(builder_, location_, result,
                                                    operand)
                           .getResult();
      return result;
    }
    case PolyhedralAstExpressionKind::Negate: {
      if (operands.size() != 1)
        return materializerError("a negated AST expression has invalid arity");
      if (llvm::Error error = requireScheduleOperands())
        return std::move(error);
      mlir::Value zero = mlir::arith::ConstantIntOp::create(builder_, location_,
                                                            scheduleType_, 0);
      return mlir::arith::SubIOp::create(builder_, location_, zero, operands[0])
          .getResult();
    }
    case PolyhedralAstExpressionKind::Add:
    case PolyhedralAstExpressionKind::Subtract:
    case PolyhedralAstExpressionKind::Multiply:
    case PolyhedralAstExpressionKind::Divide:
    case PolyhedralAstExpressionKind::FloorDivide:
    case PolyhedralAstExpressionKind::PositiveDivide:
    case PolyhedralAstExpressionKind::PositiveRemainder:
    case PolyhedralAstExpressionKind::ZeroRemainder: {
      if (llvm::Error error = binaryIndex())
        return std::move(error);
      switch (expression.kind) {
      case PolyhedralAstExpressionKind::Add:
        return mlir::arith::AddIOp::create(builder_, location_, operands[0],
                                           operands[1])
            .getResult();
      case PolyhedralAstExpressionKind::Subtract:
        return mlir::arith::SubIOp::create(builder_, location_, operands[0],
                                           operands[1])
            .getResult();
      case PolyhedralAstExpressionKind::Multiply:
        return mlir::arith::MulIOp::create(builder_, location_, operands[0],
                                           operands[1])
            .getResult();
      case PolyhedralAstExpressionKind::FloorDivide: {
        mlir::Value quotient = mlir::arith::DivSIOp::create(
            builder_, location_, operands[0], operands[1]);
        mlir::Value remainder = mlir::arith::RemSIOp::create(
            builder_, location_, operands[0], operands[1]);
        mlir::Value zero = mlir::arith::ConstantIntOp::create(
            builder_, location_, scheduleType_, 0);
        mlir::Value one = mlir::arith::ConstantIntOp::create(
            builder_, location_, scheduleType_, 1);
        mlir::Value nonzero = mlir::arith::CmpIOp::create(
            builder_, location_, mlir::arith::CmpIPredicate::ne, remainder,
            zero);
        mlir::Value negativeRemainder = mlir::arith::CmpIOp::create(
            builder_, location_, mlir::arith::CmpIPredicate::slt, remainder,
            zero);
        mlir::Value adjust = mlir::arith::AndIOp::create(
            builder_, location_, nonzero, negativeRemainder);
        mlir::Value decremented =
            mlir::arith::SubIOp::create(builder_, location_, quotient, one);
        return mlir::arith::SelectOp::create(builder_, location_, adjust,
                                             decremented, quotient)
            .getResult();
      }
      case PolyhedralAstExpressionKind::PositiveRemainder:
      case PolyhedralAstExpressionKind::ZeroRemainder:
        return mlir::arith::RemSIOp::create(builder_, location_, operands[0],
                                            operands[1])
            .getResult();
      default:
        return mlir::arith::DivSIOp::create(builder_, location_, operands[0],
                                            operands[1])
            .getResult();
      }
    }
    case PolyhedralAstExpressionKind::Select: {
      if (operands.size() != 3 || !operands[0].getType().isInteger(1) ||
          operands[1].getType() != operands[2].getType())
        return materializerError(
            "a select AST expression has invalid operands");
      return mlir::arith::SelectOp::create(builder_, location_, operands[0],
                                           operands[1], operands[2])
          .getResult();
    }
    case PolyhedralAstExpressionKind::Equal:
    case PolyhedralAstExpressionKind::LessEqual:
    case PolyhedralAstExpressionKind::Less:
    case PolyhedralAstExpressionKind::GreaterEqual:
    case PolyhedralAstExpressionKind::Greater: {
      if (llvm::Error error = binaryIndex())
        return std::move(error);
      mlir::arith::CmpIPredicate predicate;
      switch (expression.kind) {
      case PolyhedralAstExpressionKind::Equal:
        predicate = mlir::arith::CmpIPredicate::eq;
        break;
      case PolyhedralAstExpressionKind::LessEqual:
        predicate = mlir::arith::CmpIPredicate::sle;
        break;
      case PolyhedralAstExpressionKind::Less:
        predicate = mlir::arith::CmpIPredicate::slt;
        break;
      case PolyhedralAstExpressionKind::GreaterEqual:
        predicate = mlir::arith::CmpIPredicate::sge;
        break;
      default:
        predicate = mlir::arith::CmpIPredicate::sgt;
        break;
      }
      return mlir::arith::CmpIOp::create(builder_, location_, predicate,
                                         operands[0], operands[1])
          .getResult();
    }
    case PolyhedralAstExpressionKind::Integer:
    case PolyhedralAstExpressionKind::Identifier:
    case PolyhedralAstExpressionKind::AndThen:
    case PolyhedralAstExpressionKind::OrElse:
    case PolyhedralAstExpressionKind::Conditional:
    case PolyhedralAstExpressionKind::Call:
      break;
    }
    return materializerError("an AST expression has no MLIR realization");
  }

  llvm::Error emitUser(const PolyhedralAstUser &user,
                       MaterializationScope &scope) {
    if (user.call.kind != PolyhedralAstExpressionKind::Call ||
        user.call.operands.empty())
      return materializerError("an AST user node has no statement call");
    const PolyhedralAstExpression &callee = user.call.operands.front();
    if (callee.kind != PolyhedralAstExpressionKind::Identifier)
      return materializerError("an AST statement call has no tuple identifier");
    const auto ordinal = statementOrdinal(callee.identifier);
    if (!ordinal || *ordinal >= statements_.size() ||
        user.call.operands.size() != dimensions_[*ordinal].size() + 1)
      return materializerError("an AST statement call has invalid coordinates");

    mlir::IRMapping mapping;
    std::vector<PolyhedralAstExpression> coordinates;
    coordinates.reserve(dimensions_[*ordinal].size());
    for (auto [sourceDimension, expression] : llvm::zip_equal(
             dimensions_[*ordinal], llvm::drop_begin(user.call.operands))) {
      auto coordinate = emitExpression(expression, scope);
      if (!coordinate)
        return coordinate.takeError();
      // AST calls are inverse points in the exact source statement domain.
      mlir::Value sourceCoordinate = mlir::arith::IndexCastOp::create(
          builder_, location_, builder_.getIndexType(), *coordinate);
      mapping.map(sourceDimension, sourceCoordinate);
      coordinates.push_back(expression);
    }
    mlir::Operation *source = statements_[*ordinal];
    for (mlir::Value operand : source->getOperands()) {
      auto materialized = scope.statementValues.find(operand);
      if (materialized != scope.statementValues.end()) {
        auto matching = llvm::find_if(
            llvm::reverse(materialized->second),
            [&](const MaterializedStatementValue &candidate) {
              if (candidate.coordinates.size() > coordinates.size())
                return false;
              const auto &consumerDimensions =
                  scop_.statements[*ordinal].domain.dimensions;
              if (candidate.dimensions.size() > consumerDimensions.size() ||
                  !std::equal(candidate.dimensions.begin(),
                              candidate.dimensions.end(),
                              consumerDimensions.begin()))
                return false;
              return llvm::all_of(
                  llvm::zip_equal(
                      candidate.coordinates,
                      llvm::ArrayRef<PolyhedralAstExpression>(coordinates)
                          .take_front(candidate.coordinates.size())),
                  [](const auto &pair) {
                    return expressionsEqual(std::get<0>(pair),
                                            std::get<1>(pair));
                  });
            });
        if (matching == materialized->second.rend())
          return llvm::make_error<ScalarExpansionUnavailable>();
        if (!mapping.contains(operand))
          mapping.map(operand, matching->value);
        continue;
      }
      if (valueDefinedInside(operand, root_) && !mapping.contains(operand))
        return llvm::make_error<ScalarExpansionUnavailable>();
    }
    auto clonedResults = materializeStatement(source, mapping);
    if (!clonedResults)
      return clonedResults.takeError();
    if (clonedResults->size() != source->getNumResults())
      return materializerError("a scheduled statement did not clone exactly");
    for (auto [sourceResult, clonedResult] :
         llvm::zip_equal(source->getResults(), *clonedResults))
      scope.statementValues[sourceResult].push_back(MaterializedStatementValue{
          clonedResult, coordinates,
          scop_.statements[*ordinal].domain.dimensions});
    return llvm::Error::success();
  }

  llvm::Expected<llvm::SmallVector<mlir::Value>>
  materializeStatement(mlir::Operation *source, mlir::IRMapping &mapping) {
    const auto mapped = [&](mlir::Value value) {
      return mapping.lookupOrDefault(value);
    };
    const auto expandedIndices = [&](mlir::AffineMap map,
                                     mlir::ValueRange operands)
        -> llvm::Expected<llvm::SmallVector<mlir::Value>> {
      llvm::SmallVector<mlir::Value> mappedOperands;
      mappedOperands.reserve(operands.size());
      for (mlir::Value operand : operands)
        mappedOperands.push_back(mapped(operand));
      auto expanded = mlir::affine::expandAffineMap(builder_, source->getLoc(),
                                                    map, mappedOperands);
      if (!expanded)
        return materializerError("an affine statement index did not expand");
      return llvm::SmallVector<mlir::Value>(expanded->begin(), expanded->end());
    };

    if (auto load = llvm::dyn_cast<mlir::affine::AffineLoadOp>(source)) {
      auto indices =
          expandedIndices(load.getAffineMap(), load.getMapOperands());
      if (!indices)
        return indices.takeError();
      mlir::Value result = mlir::memref::LoadOp::create(
          builder_, load.getLoc(), mapped(load.getMemRef()), *indices);
      return llvm::SmallVector<mlir::Value>{result};
    }
    if (auto store = llvm::dyn_cast<mlir::affine::AffineStoreOp>(source)) {
      auto indices =
          expandedIndices(store.getAffineMap(), store.getMapOperands());
      if (!indices)
        return indices.takeError();
      mlir::memref::StoreOp::create(builder_, store.getLoc(),
                                    mapped(store.getValueToStore()),
                                    mapped(store.getMemRef()), *indices);
      return llvm::SmallVector<mlir::Value>{};
    }
    if (auto apply = llvm::dyn_cast<mlir::affine::AffineApplyOp>(source)) {
      auto results =
          expandedIndices(apply.getAffineMap(), apply.getMapOperands());
      if (!results)
        return results.takeError();
      if (results->size() != 1)
        return materializerError("an affine.apply changed result cardinality");
      return std::move(*results);
    }
    if (auto maximum = llvm::dyn_cast<mlir::affine::AffineMaxOp>(source)) {
      auto values =
          expandedIndices(maximum.getAffineMap(), maximum.getMapOperands());
      if (!values)
        return values.takeError();
      if (values->empty())
        return materializerError("an affine.max has no expression");
      mlir::Value result = values->front();
      for (mlir::Value value : llvm::drop_begin(*values))
        result = mlir::arith::MaxSIOp::create(builder_, maximum.getLoc(),
                                              result, value);
      return llvm::SmallVector<mlir::Value>{result};
    }
    if (auto minimum = llvm::dyn_cast<mlir::affine::AffineMinOp>(source)) {
      auto values =
          expandedIndices(minimum.getAffineMap(), minimum.getMapOperands());
      if (!values)
        return values.takeError();
      if (values->empty())
        return materializerError("an affine.min has no expression");
      mlir::Value result = values->front();
      for (mlir::Value value : llvm::drop_begin(*values))
        result = mlir::arith::MinSIOp::create(builder_, minimum.getLoc(),
                                              result, value);
      return llvm::SmallVector<mlir::Value>{result};
    }

    mlir::Operation *cloned = builder_.clone(*source, mapping);
    if (!cloned)
      return materializerError("a scheduled statement did not clone");
    return llvm::SmallVector<mlir::Value>(cloned->getResults());
  }

  llvm::Error emitNode(const PolyhedralAstNode &node,
                       MaterializationScope &scope) {
    if (const auto *loop = std::get_if<PolyhedralAstFor>(&node.value)) {
      auto lower = emitExpression(loop->initial, scope);
      auto step = emitExpression(loop->increment, scope);
      if (!lower)
        return lower.takeError();
      if (!step)
        return step.takeError();
      if (loop->condition.operands.size() != 2 ||
          loop->condition.operands.front().kind !=
              PolyhedralAstExpressionKind::Identifier ||
          loop->condition.operands.front().identifier != loop->iterator)
        return materializerError("an AST loop condition changed its iterator");
      const bool inclusive =
          loop->condition.kind == PolyhedralAstExpressionKind::LessEqual;
      if (!inclusive &&
          loop->condition.kind != PolyhedralAstExpressionKind::Less)
        return materializerError("an AST loop condition is not an upper bound");
      auto upper = emitExpression(loop->condition.operands[1], scope);
      if (!upper)
        return upper.takeError();
      if (inclusive) {
        mlir::Value one = mlir::arith::ConstantIntOp::create(
            builder_, location_, scheduleType_, 1);
        *upper = mlir::arith::AddIOp::create(builder_, location_, *upper, one);
      }
      auto materializedLoop =
          mlir::scf::ForOp::create(builder_, location_, *lower, *upper, *step);
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPointToStart(materializedLoop.getBody());
      MaterializationScope nested = scope;
      nested.iterators.insert_or_assign(loop->iterator,
                                        materializedLoop.getInductionVar());
      return emitNode(*loop->body, nested);
    }
    if (const auto *branch = std::get_if<PolyhedralAstIf>(&node.value)) {
      auto condition = emitExpression(branch->condition, scope);
      if (!condition)
        return condition.takeError();
      if (!condition->getType().isInteger(1))
        return materializerError("an AST conditional is not boolean");
      auto materializedIf = mlir::scf::IfOp::create(
          builder_, location_, *condition, static_cast<bool>(branch->elseNode));
      {
        mlir::OpBuilder::InsertionGuard guard(builder_);
        builder_.setInsertionPointToStart(
            &materializedIf.getThenRegion().front());
        MaterializationScope thenScope = scope;
        if (llvm::Error error = emitNode(*branch->thenNode, thenScope))
          return error;
      }
      if (branch->elseNode) {
        mlir::OpBuilder::InsertionGuard guard(builder_);
        builder_.setInsertionPointToStart(
            &materializedIf.getElseRegion().front());
        MaterializationScope elseScope = scope;
        if (llvm::Error error = emitNode(*branch->elseNode, elseScope))
          return error;
      }
      return llvm::Error::success();
    }
    if (const auto *block = std::get_if<PolyhedralAstBlock>(&node.value)) {
      for (const PolyhedralAstNode &child : block->children)
        if (llvm::Error error = emitNode(child, scope))
          return error;
      return llvm::Error::success();
    }
    return emitUser(std::get<PolyhedralAstUser>(node.value), scope);
  }

  mlir::Operation *root_ = nullptr;
  const StructuredPolyhedralScopView &scop_;
  const StructuredProgramCandidateView &parentView_;
  const mlir::IRMapping &cloneMapping_;
  mlir::OpBuilder builder_;
  mlir::Location location_;
  mlir::IntegerType scheduleType_;
  llvm::SmallVectorImpl<mlir::Operation *> &materializedOperations_;
  std::vector<mlir::Operation *> statements_;
  std::vector<std::vector<mlir::Value>> dimensions_;
  std::vector<mlir::Value> parameters_;
};

char AstMaterializer::ScalarExpansionUnavailable::ID = 0;

} // namespace

llvm::Expected<std::optional<StructuredScopRefusalKind>>
classifyPinnedIslScheduleMaterialization(
    mlir::Operation *root, const StructuredPolyhedralScopView &scop) {
  auto plan = buildMaterializationPlan(root, scop);
  if (!plan)
    return plan.takeError();
  if (auto *refusal = std::get_if<StructuredScopRefusalKind>(&*plan))
    return *refusal;
  return std::nullopt;
}

llvm::Expected<std::optional<StructuredScopRefusalKind>>
materializePinnedIslSchedule(
    mlir::Operation *root, const StructuredPolyhedralScopView &scop,
    const StructuredProgramCandidateView &parentView,
    const mlir::IRMapping &cloneMapping,
    llvm::SmallVectorImpl<mlir::Operation *> &materializedOperations) {
  auto outcome = buildMaterializationPlan(root, scop);
  if (!outcome)
    return outcome.takeError();
  if (auto *refusal = std::get_if<StructuredScopRefusalKind>(&*outcome))
    return *refusal;
  MaterializationPlan &plan = std::get<MaterializationPlan>(*outcome);
  auto materialized =
      AstMaterializer(root, scop, parentView, cloneMapping, plan.scheduleWidth,
                      materializedOperations)
          .run(plan.ast);
  if (!materialized)
    return materialized.takeError();
  if (!*materialized)
    return StructuredScopRefusalKind::PolyhedralMaterializationUnavailable;
  return std::nullopt;
}

} // namespace loom::frontend::detail
