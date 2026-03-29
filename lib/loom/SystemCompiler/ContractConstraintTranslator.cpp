#include "loom/SystemCompiler/ContractConstraintTranslator.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace loom {

//===----------------------------------------------------------------------===//
// Symbolic Expression Evaluator -- Recursive Descent Parser
//
// Grammar:
//   expr    -> term (('+' | '-') term)*
//   term    -> factor (('*' | '/') factor)*
//   factor  -> '(' expr ')' | NUMBER | IDENT
//   NUMBER  -> [0-9]+
//   IDENT   -> [a-zA-Z_][a-zA-Z_0-9]*
//===----------------------------------------------------------------------===//

namespace {

/// Token types for the expression lexer.
enum class TokenKind { NUMBER, IDENT, PLUS, MINUS, STAR, SLASH, LPAREN, RPAREN, END, INVALID };

struct Token {
  TokenKind kind = TokenKind::END;
  std::string text;
  int64_t numVal = 0;
};

/// Simple lexer for arithmetic expressions.
class ExprLexer {
public:
  explicit ExprLexer(const std::string &input) : src_(input), pos_(0) {}

  Token next() {
    skipWhitespace();
    if (pos_ >= src_.size())
      return {TokenKind::END, "", 0};

    char ch = src_[pos_];

    // Single-character tokens.
    if (ch == '+') { ++pos_; return {TokenKind::PLUS, "+", 0}; }
    if (ch == '-') { ++pos_; return {TokenKind::MINUS, "-", 0}; }
    if (ch == '*') { ++pos_; return {TokenKind::STAR, "*", 0}; }
    if (ch == '/') { ++pos_; return {TokenKind::SLASH, "/", 0}; }
    if (ch == '(') { ++pos_; return {TokenKind::LPAREN, "(", 0}; }
    if (ch == ')') { ++pos_; return {TokenKind::RPAREN, ")", 0}; }

    // Number literal.
    if (std::isdigit(static_cast<unsigned char>(ch))) {
      size_t start = pos_;
      while (pos_ < src_.size() &&
             std::isdigit(static_cast<unsigned char>(src_[pos_])))
        ++pos_;
      std::string numStr = src_.substr(start, pos_ - start);
      int64_t val = 0;
      try {
        val = std::stoll(numStr);
      } catch (...) {
        return {TokenKind::INVALID, numStr, 0};
      }
      return {TokenKind::NUMBER, numStr, val};
    }

    // Identifier.
    if (std::isalpha(static_cast<unsigned char>(ch)) || ch == '_') {
      size_t start = pos_;
      while (pos_ < src_.size() &&
             (std::isalnum(static_cast<unsigned char>(src_[pos_])) ||
              src_[pos_] == '_'))
        ++pos_;
      std::string ident = src_.substr(start, pos_ - start);
      return {TokenKind::IDENT, ident, 0};
    }

    ++pos_;
    return {TokenKind::INVALID, std::string(1, ch), 0};
  }

private:
  void skipWhitespace() {
    while (pos_ < src_.size() &&
           std::isspace(static_cast<unsigned char>(src_[pos_])))
      ++pos_;
  }

  std::string src_;
  size_t pos_;
};

/// Recursive descent parser for arithmetic expressions.
class ExprParser {
public:
  ExprParser(const std::string &input, const ParameterMap &params)
      : lexer_(input), params_(params) {
    advance();
  }

  EvalResult parse() {
    EvalResult result = parseExpr();
    if (!result.ok())
      return result;
    if (current_.kind != TokenKind::END) {
      result.error = "unexpected token '" + current_.text +
                     "' after expression";
      return result;
    }
    return result;
  }

private:
  void advance() { current_ = lexer_.next(); }

  EvalResult parseExpr() {
    EvalResult lhs = parseTerm();
    if (!lhs.ok())
      return lhs;

    while (current_.kind == TokenKind::PLUS ||
           current_.kind == TokenKind::MINUS) {
      TokenKind op = current_.kind;
      advance();
      EvalResult rhs = parseTerm();
      if (!rhs.ok())
        return rhs;
      if (op == TokenKind::PLUS)
        lhs.value += rhs.value;
      else
        lhs.value -= rhs.value;
    }
    return lhs;
  }

  EvalResult parseTerm() {
    EvalResult lhs = parseFactor();
    if (!lhs.ok())
      return lhs;

    while (current_.kind == TokenKind::STAR ||
           current_.kind == TokenKind::SLASH) {
      TokenKind op = current_.kind;
      advance();
      EvalResult rhs = parseFactor();
      if (!rhs.ok())
        return rhs;
      if (op == TokenKind::STAR) {
        lhs.value *= rhs.value;
      } else {
        if (rhs.value == 0) {
          lhs.error = "division by zero";
          return lhs;
        }
        lhs.value /= rhs.value;
      }
    }
    return lhs;
  }

  EvalResult parseFactor() {
    EvalResult result;

    // Parenthesized sub-expression.
    if (current_.kind == TokenKind::LPAREN) {
      advance();
      result = parseExpr();
      if (!result.ok())
        return result;
      if (current_.kind != TokenKind::RPAREN) {
        result.error = "expected ')' but got '" + current_.text + "'";
        return result;
      }
      advance();
      return result;
    }

    // Integer literal.
    if (current_.kind == TokenKind::NUMBER) {
      result.value = current_.numVal;
      advance();
      return result;
    }

    // Named parameter.
    if (current_.kind == TokenKind::IDENT) {
      std::string name = current_.text;
      auto it = params_.find(name);
      if (it == params_.end()) {
        result.error =
            "unknown variable '" + name + "' not in parameter map";
        return result;
      }
      result.value = it->second;
      advance();
      return result;
    }

    // Unary minus.
    if (current_.kind == TokenKind::MINUS) {
      advance();
      result = parseFactor();
      if (result.ok())
        result.value = -result.value;
      return result;
    }

    result.error = "unexpected token '" + current_.text + "'";
    return result;
  }

  ExprLexer lexer_;
  const ParameterMap &params_;
  Token current_;
};

} // namespace

//===----------------------------------------------------------------------===//
// evaluateSymbolicExpr
//===----------------------------------------------------------------------===//

EvalResult evaluateSymbolicExpr(const std::string &expr,
                                const ParameterMap &params) {
  if (expr.empty())
    return {0, "empty expression"};

  ExprParser parser(expr, params);
  return parser.parse();
}

//===----------------------------------------------------------------------===//
// parseShapeDimensions
//===----------------------------------------------------------------------===//

std::vector<std::string> parseShapeDimensions(const std::string &shapeExpr) {
  std::vector<std::string> dims;
  if (shapeExpr.empty())
    return dims;

  // Strip outer brackets if present.
  std::string inner = shapeExpr;
  size_t start = inner.find('[');
  size_t end = inner.rfind(']');
  if (start != std::string::npos && end != std::string::npos && end > start) {
    inner = inner.substr(start + 1, end - start - 1);
  }

  // Split by comma, trimming whitespace.
  std::istringstream stream(inner);
  std::string token;
  while (std::getline(stream, token, ',')) {
    // Trim leading and trailing whitespace.
    size_t first = token.find_first_not_of(" \t\n\r");
    if (first == std::string::npos)
      continue;
    size_t last = token.find_last_not_of(" \t\n\r");
    dims.push_back(token.substr(first, last - first + 1));
  }
  return dims;
}

//===----------------------------------------------------------------------===//
// ContractConstraintTranslator
//===----------------------------------------------------------------------===//

void ContractConstraintTranslator::emitDiag(
    TranslatorDiagnostic::Severity sev, const std::string &msg) {
  diagnostics_.push_back({sev, msg});
}

void ContractConstraintTranslator::translateEdgeSpec(
    const TDCEdgeSpec &spec, const ParameterMap &params, ConstraintSet &out) {

  // Ordering dimension.
  if (spec.ordering.has_value()) {
    TDCOrdering ord = spec.ordering.value();
    if (ord == TDCOrdering::FIFO) {
      SchedulingConstraint sc;
      sc.producer = spec.producerKernel;
      sc.consumer = spec.consumerKernel;
      out.schedulingConstraints.push_back(sc);
    }
    // UNORDERED: emit nothing -- compiler is free.
  }

  // Throughput dimension.
  if (spec.throughput.has_value()) {
    const std::string &expr = spec.throughput.value();
    EvalResult eval = evaluateSymbolicExpr(expr, params);
    if (eval.ok()) {
      RateConstraint rc;
      rc.edgeProducer = spec.producerKernel;
      rc.edgeConsumer = spec.consumerKernel;
      rc.minRate = eval.value;
      out.rateConstraints.push_back(rc);
    } else {
      emitDiag(TranslatorDiagnostic::ERROR,
               "failed to evaluate throughput expression '" + expr +
                   "' for edge " + spec.producerKernel + " -> " +
                   spec.consumerKernel + ": " + eval.error);
    }
  }

  // Placement dimension.
  if (spec.placement.has_value()) {
    TDCPlacement pl = spec.placement.value();
    if (pl != TDCPlacement::AUTO) {
      MemoryConstraint mc;
      mc.edgeProducer = spec.producerKernel;
      mc.edgeConsumer = spec.consumerKernel;
      switch (pl) {
      case TDCPlacement::LOCAL_SPM:
        mc.level = MemoryLevel::LOCAL_SPM;
        break;
      case TDCPlacement::SHARED_L2:
        mc.level = MemoryLevel::SHARED_L2;
        break;
      case TDCPlacement::EXTERNAL:
        mc.level = MemoryLevel::EXTERNAL;
        break;
      case TDCPlacement::AUTO:
        break; // unreachable due to outer check
      }
      out.memoryConstraints.push_back(mc);
    }
    // AUTO: emit nothing -- compiler is free.
  }

  // Shape dimension.
  if (spec.shape.has_value()) {
    const std::string &shapeStr = spec.shape.value();
    std::vector<std::string> dimExprs = parseShapeDimensions(shapeStr);
    if (dimExprs.empty()) {
      emitDiag(TranslatorDiagnostic::WARNING,
               "empty shape expression for edge " + spec.producerKernel +
                   " -> " + spec.consumerKernel);
      return;
    }

    TilingConstraint tc;
    tc.edgeProducer = spec.producerKernel;
    tc.edgeConsumer = spec.consumerKernel;

    bool allOk = true;
    for (const auto &dimExpr : dimExprs) {
      EvalResult eval = evaluateSymbolicExpr(dimExpr, params);
      if (eval.ok()) {
        tc.dimensions.push_back(eval.value);
      } else {
        emitDiag(TranslatorDiagnostic::ERROR,
                 "failed to evaluate shape dimension '" + dimExpr +
                     "' for edge " + spec.producerKernel + " -> " +
                     spec.consumerKernel + ": " + eval.error);
        allOk = false;
        break;
      }
    }

    if (allOk) {
      out.tilingConstraints.push_back(tc);
    }
  }
}

void ContractConstraintTranslator::translatePathSpec(
    const TDCPathSpec &spec, const ParameterMap &params, ConstraintSet &out) {
  EvalResult eval = evaluateSymbolicExpr(spec.latency, params);
  if (eval.ok()) {
    PathLatencyConstraint plc;
    plc.startProducer = spec.startProducer;
    plc.startConsumer = spec.startConsumer;
    plc.endProducer = spec.endProducer;
    plc.endConsumer = spec.endConsumer;
    plc.maxCycles = eval.value;
    out.pathLatencyConstraints.push_back(plc);
  } else {
    emitDiag(TranslatorDiagnostic::ERROR,
             "failed to evaluate path latency expression '" + spec.latency +
                 "' for path " + spec.startProducer + "->" +
                 spec.startConsumer + " ... " + spec.endProducer + "->" +
                 spec.endConsumer + ": " + eval.error);
  }
}

ConstraintSet ContractConstraintTranslator::translate(
    const std::vector<TDCEdgeSpec> &edgeSpecs,
    const std::vector<TDCPathSpec> &pathSpecs, const ParameterMap &params) {
  clearDiagnostics();
  ConstraintSet result;

  for (const auto &edgeSpec : edgeSpecs) {
    translateEdgeSpec(edgeSpec, params, result);
  }

  for (const auto &pathSpec : pathSpecs) {
    translatePathSpec(pathSpec, params, result);
  }

  return result;
}

std::vector<PruningMask> ContractConstraintTranslator::computePruningMasks(
    const std::vector<TDCEdgeSpec> &edgeSpecs) {
  std::vector<PruningMask> masks;
  masks.reserve(edgeSpecs.size());

  for (const auto &spec : edgeSpecs) {
    PruningMask pm;
    pm.edgeProducer = spec.producerKernel;
    pm.edgeConsumer = spec.consumerKernel;
    pm.mask = 0;

    if (spec.ordering.has_value() &&
        spec.ordering.value() == TDCOrdering::FIFO) {
      pm.mask |= PruningMask::ORDERING_LOCKED;
    }

    if (spec.throughput.has_value()) {
      pm.mask |= PruningMask::THROUGHPUT_LOCKED;
    }

    if (spec.placement.has_value() &&
        spec.placement.value() != TDCPlacement::AUTO) {
      pm.mask |= PruningMask::PLACEMENT_LOCKED;
    }

    if (spec.shape.has_value()) {
      pm.mask |= PruningMask::SHAPE_LOCKED;
    }

    masks.push_back(pm);
  }

  return masks;
}

} // namespace loom
