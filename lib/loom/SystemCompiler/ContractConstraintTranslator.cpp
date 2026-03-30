#include "loom/SystemCompiler/ContractConstraintTranslator.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace loom {

//===----------------------------------------------------------------------===//
// Symbolic expression evaluator
//===----------------------------------------------------------------------===//

namespace {

/// Tokenizer for symbolic expressions.
enum class TokenKind { Number, Ident, Plus, Minus, Star, Slash, LParen, RParen, End };

struct Token {
  TokenKind kind;
  std::string text;
  int64_t numVal = 0;
};

class Lexer {
public:
  explicit Lexer(const std::string &input) : input_(input), pos_(0) {}

  Token next() {
    skipWhitespace();
    if (pos_ >= input_.size())
      return {TokenKind::End, "", 0};

    char ch = input_[pos_];

    if (std::isdigit(ch))
      return lexNumber();
    if (std::isalpha(ch) || ch == '_')
      return lexIdent();
    ++pos_;
    switch (ch) {
    case '+': return {TokenKind::Plus, "+", 0};
    case '-': return {TokenKind::Minus, "-", 0};
    case '*': return {TokenKind::Star, "*", 0};
    case '/': return {TokenKind::Slash, "/", 0};
    case '(': return {TokenKind::LParen, "(", 0};
    case ')': return {TokenKind::RParen, ")", 0};
    default:
      return {TokenKind::End, std::string(1, ch), 0};
    }
  }

private:
  void skipWhitespace() {
    while (pos_ < input_.size() && std::isspace(input_[pos_]))
      ++pos_;
  }

  Token lexNumber() {
    size_t start = pos_;
    while (pos_ < input_.size() && std::isdigit(input_[pos_]))
      ++pos_;
    std::string text = input_.substr(start, pos_ - start);
    return {TokenKind::Number, text, std::stoll(text)};
  }

  Token lexIdent() {
    size_t start = pos_;
    while (pos_ < input_.size() &&
           (std::isalnum(input_[pos_]) || input_[pos_] == '_'))
      ++pos_;
    return {TokenKind::Ident, input_.substr(start, pos_ - start), 0};
  }

  std::string input_;
  size_t pos_;
};

/// Recursive descent parser for arithmetic expressions.
/// Grammar:
///   expr   -> term (('+' | '-') term)*
///   term   -> factor (('*' | '/') factor)*
///   factor -> NUMBER | IDENT | '(' expr ')' | '-' factor
class ExprParser {
public:
  ExprParser(const std::string &input,
             const std::map<std::string, int64_t> &params)
      : lexer_(input), params_(params) {
    advance();
  }

  SymbolicEvalResult parse() {
    auto result = parseExpr();
    if (!result.ok())
      return result;
    if (current_.kind != TokenKind::End) {
      return {std::nullopt, "unexpected token '" + current_.text +
                                "' after expression"};
    }
    return result;
  }

private:
  void advance() { current_ = lexer_.next(); }

  SymbolicEvalResult parseExpr() {
    auto left = parseTerm();
    if (!left.ok())
      return left;

    while (current_.kind == TokenKind::Plus ||
           current_.kind == TokenKind::Minus) {
      bool isAdd = (current_.kind == TokenKind::Plus);
      advance();
      auto right = parseTerm();
      if (!right.ok())
        return right;
      if (isAdd)
        left.value = *left.value + *right.value;
      else
        left.value = *left.value - *right.value;
    }
    return left;
  }

  SymbolicEvalResult parseTerm() {
    auto left = parseFactor();
    if (!left.ok())
      return left;

    while (current_.kind == TokenKind::Star ||
           current_.kind == TokenKind::Slash) {
      bool isMul = (current_.kind == TokenKind::Star);
      advance();
      auto right = parseFactor();
      if (!right.ok())
        return right;
      if (isMul) {
        left.value = *left.value * *right.value;
      } else {
        if (*right.value == 0)
          return {std::nullopt, "division by zero"};
        left.value = *left.value / *right.value;
      }
    }
    return left;
  }

  SymbolicEvalResult parseFactor() {
    // Unary minus
    if (current_.kind == TokenKind::Minus) {
      advance();
      auto inner = parseFactor();
      if (!inner.ok())
        return inner;
      inner.value = -(*inner.value);
      return inner;
    }

    if (current_.kind == TokenKind::Number) {
      int64_t val = current_.numVal;
      advance();
      return {val, ""};
    }

    if (current_.kind == TokenKind::Ident) {
      std::string name = current_.text;
      advance();
      auto it = params_.find(name);
      if (it == params_.end())
        return {std::nullopt,
                "unknown variable '" + name + "' is not in the parameter map"};
      return {it->second, ""};
    }

    if (current_.kind == TokenKind::LParen) {
      advance();
      auto inner = parseExpr();
      if (!inner.ok())
        return inner;
      if (current_.kind != TokenKind::RParen)
        return {std::nullopt, "expected ')'"};
      advance();
      return inner;
    }

    return {std::nullopt, "unexpected token '" + current_.text + "'"};
  }

  Lexer lexer_;
  const std::map<std::string, int64_t> &params_;
  Token current_;
};

} // anonymous namespace

SymbolicEvalResult
evaluateSymbolicExpr(const std::string &expr,
                     const std::map<std::string, int64_t> &params) {
  if (expr.empty())
    return {std::nullopt, "empty expression"};
  ExprParser parser(expr, params);
  return parser.parse();
}

//===----------------------------------------------------------------------===//
// ContractConstraintTranslator
//===----------------------------------------------------------------------===//

ContractConstraintTranslator::ContractConstraintTranslator(
    std::map<std::string, int64_t> params)
    : params_(std::move(params)) {}

ConstraintSet
ContractConstraintTranslator::translate(
    const std::vector<TDCEdgeSpec> &edges,
    const std::vector<TDCPathSpec> &paths) const {
  ConstraintSet out;

  for (const auto &edge : edges)
    translateEdge(edge, out);
  for (const auto &path : paths)
    translatePath(path, out);

  return out;
}

void ContractConstraintTranslator::translateEdge(const TDCEdgeSpec &edge,
                                                  ConstraintSet &out) const {
  // Ordering -> SchedulingConstraint (only for FIFO)
  if (edge.ordering.has_value() && *edge.ordering == Ordering::FIFO) {
    SchedulingConstraint sc;
    sc.producer = edge.producerKernel;
    sc.consumer = edge.consumerKernel;
    out.scheduling.push_back(std::move(sc));
  }

  // Throughput -> RateConstraint
  if (edge.throughput.has_value() && !edge.throughput->empty()) {
    auto evalResult = evaluateSymbolicExpr(*edge.throughput, params_);
    if (evalResult.ok()) {
      RateConstraint rc;
      rc.edgeProducer = edge.producerKernel;
      rc.edgeConsumer = edge.consumerKernel;
      rc.minRate = *evalResult.value;
      out.rate.push_back(std::move(rc));
    } else {
      out.diagnostics.push_back(
          "failed to evaluate throughput expression '" + *edge.throughput +
          "' for edge " + edge.producerKernel + "->" +
          edge.consumerKernel + ": " + evalResult.error);
    }
  }

  // Placement -> MemoryConstraint (only when not AUTO)
  if (edge.placement.has_value() && *edge.placement != Placement::AUTO) {
    MemoryConstraint mc;
    mc.edgeProducer = edge.producerKernel;
    mc.edgeConsumer = edge.consumerKernel;
    switch (*edge.placement) {
    case Placement::LOCAL_SPM:
      mc.level = MemoryLevel::LOCAL_SPM;
      break;
    case Placement::SHARED_L2:
      mc.level = MemoryLevel::SHARED_L2;
      break;
    case Placement::EXTERNAL:
      mc.level = MemoryLevel::EXTERNAL;
      break;
    case Placement::AUTO:
      break; // unreachable
    }
    out.memory.push_back(std::move(mc));
  }

  // Shape -> TilingConstraint
  if (edge.shape.has_value() && !edge.shape->empty()) {
    auto resolved = resolveShape(*edge.shape);
    if (!resolved.empty()) {
      TilingConstraint tc;
      tc.edgeProducer = edge.producerKernel;
      tc.edgeConsumer = edge.consumerKernel;
      tc.dimensions = std::move(resolved);
      out.tiling.push_back(std::move(tc));
    } else {
      out.diagnostics.push_back(
          "failed to resolve shape expression '" + *edge.shape +
          "' for edge " + edge.producerKernel + "->" +
          edge.consumerKernel);
    }
  }
}

void ContractConstraintTranslator::translatePath(const TDCPathSpec &path,
                                                  ConstraintSet &out) const {
  if (path.latency.empty())
    return;

  auto evalResult = evaluateSymbolicExpr(path.latency, params_);
  if (evalResult.ok()) {
    PathLatencyConstraint plc;
    plc.startProducer = path.startProducer;
    plc.startConsumer = path.startConsumer;
    plc.endProducer = path.endProducer;
    plc.endConsumer = path.endConsumer;
    plc.maxCycles = *evalResult.value;
    out.pathLatency.push_back(std::move(plc));
  } else {
    out.diagnostics.push_back(
        "failed to evaluate latency expression '" + path.latency +
        "' for path " + path.startProducer + "->" + path.startConsumer +
        "..." + path.endProducer + "->" + path.endConsumer + ": " +
        evalResult.error);
  }
}

std::vector<int64_t>
ContractConstraintTranslator::resolveShape(const std::string &shapeExpr) const {
  auto dimStrings = parseShapeExpr(shapeExpr);
  std::vector<int64_t> result;
  result.reserve(dimStrings.size());

  for (const auto &dimStr : dimStrings) {
    auto evalResult = evaluateSymbolicExpr(dimStr, params_);
    if (!evalResult.ok())
      return {}; // Return empty to signal failure
    result.push_back(*evalResult.value);
  }

  return result;
}

std::map<EdgeKey, uint8_t>
ContractConstraintTranslator::computePruningMasks(
    const std::vector<TDCEdgeSpec> &edges) const {
  std::map<EdgeKey, uint8_t> masks;
  for (const auto &edge : edges) {
    EdgeKey key{edge.producerKernel, edge.consumerKernel};
    masks[key] = computePruningMask(edge);
  }
  return masks;
}

uint8_t
ContractConstraintTranslator::computePruningMask(
    const TDCEdgeSpec &edgeSpec) const {
  uint8_t mask = 0;

  // Bit 0: ordering locked (FIFO or SYMBOLIC specified)
  if (edgeSpec.ordering.has_value() &&
      *edgeSpec.ordering != Ordering::UNORDERED) {
    mask |= (1u << PRUNING_ORDERING_LOCKED);
  }

  // Bit 1: throughput floor set
  if (edgeSpec.throughput.has_value() && !edgeSpec.throughput->empty()) {
    mask |= (1u << PRUNING_THROUGHPUT_FLOOR);
  }

  // Bit 2: placement locked (anything except AUTO)
  if (edgeSpec.placement.has_value() &&
      *edgeSpec.placement != Placement::AUTO) {
    mask |= (1u << PRUNING_PLACEMENT_LOCKED);
  }

  // Bit 3: shape locked
  if (edgeSpec.shape.has_value() && !edgeSpec.shape->empty()) {
    mask |= (1u << PRUNING_SHAPE_LOCKED);
  }

  return mask;
}

//===----------------------------------------------------------------------===//
// Legacy flat TranslatedConstraint API (backward compatibility)
//===----------------------------------------------------------------------===//

std::vector<TranslatedConstraint>
translateEdgeConstraints(const TDCEdgeSpec &edgeSpec) {
  std::vector<TranslatedConstraint> constraints;
  std::string edgeLabel =
      edgeSpec.producerKernel + "->" + edgeSpec.consumerKernel;

  if (edgeSpec.ordering.has_value()) {
    TranslatedConstraint c;
    c.label = "ordering:" + std::string(orderingToString(*edgeSpec.ordering)) +
              ":" + edgeLabel;
    c.dimension = "ordering";
    c.enumValue = orderingToString(*edgeSpec.ordering);
    constraints.push_back(std::move(c));
  }

  if (edgeSpec.throughput.has_value()) {
    TranslatedConstraint c;
    c.label = "throughput:" + edgeLabel;
    c.dimension = "throughput";
    c.expression = *edgeSpec.throughput;
    constraints.push_back(std::move(c));
  }

  if (edgeSpec.placement.has_value()) {
    TranslatedConstraint c;
    c.label =
        "placement:" +
        std::string(placementToString(*edgeSpec.placement)) + ":" + edgeLabel;
    c.dimension = "placement";
    c.enumValue = placementToString(*edgeSpec.placement);
    constraints.push_back(std::move(c));
  }

  if (edgeSpec.shape.has_value()) {
    TranslatedConstraint c;
    c.label = "shape:" + edgeLabel;
    c.dimension = "shape";
    c.expression = *edgeSpec.shape;
    constraints.push_back(std::move(c));
  }

  return constraints;
}

std::vector<TranslatedConstraint>
translatePathConstraints(const TDCPathSpec &pathSpec) {
  std::vector<TranslatedConstraint> constraints;

  if (!pathSpec.latency.empty()) {
    std::string pathLabel = pathSpec.startProducer + "->" +
                            pathSpec.startConsumer + "..." +
                            pathSpec.endProducer + "->" + pathSpec.endConsumer;
    TranslatedConstraint c;
    c.label = "latency:" + pathLabel;
    c.dimension = "latency";
    c.expression = pathSpec.latency;
    constraints.push_back(std::move(c));
  }

  return constraints;
}

std::vector<TranslatedConstraint>
translateAllConstraints(const std::vector<TDCEdgeSpec> &edges,
                        const std::vector<TDCPathSpec> &paths) {
  std::vector<TranslatedConstraint> all;

  for (const auto &edge : edges) {
    auto edgeConstraints = translateEdgeConstraints(edge);
    all.insert(all.end(), std::make_move_iterator(edgeConstraints.begin()),
               std::make_move_iterator(edgeConstraints.end()));
  }

  for (const auto &path : paths) {
    auto pathConstraints = translatePathConstraints(path);
    all.insert(all.end(), std::make_move_iterator(pathConstraints.begin()),
               std::make_move_iterator(pathConstraints.end()));
  }

  return all;
}

TDCEdgeSpec contractSpecToEdgeSpec(const ContractSpec &legacy) {
  TDCEdgeSpec spec;
  spec.producerKernel = legacy.producerKernel;
  spec.consumerKernel = legacy.consumerKernel;
  spec.dataTypeName = legacy.dataTypeName;
  spec.ordering = legacy.ordering;
  spec.placement = legacy.visibility;
  return spec;
}

} // namespace loom
