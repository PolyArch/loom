#include "Dataflow/IR/DataflowActorSemantics.h"

#include "Dataflow/IR/OperationSchema.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <optional>
#include <string>
#include <system_error>

namespace {

using dataflow::semantics::getStreamActivation;

std::string typeToString(mlir::Type type) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  type.print(stream);
  return storage;
}

llvm::Error requireStreamBitWidth(unsigned bitWidth) {
  if (bitWidth >= 1)
    return llvm::Error::success();
  return llvm::createStringError(
      std::errc::invalid_argument,
      "dataflow.stream integer bit width must be nonzero, got %u", bitWidth);
}

bool evaluateStreamPredicate(const llvm::APInt &current,
                             const llvm::APInt &limit,
                             mlir::arith::CmpIPredicate predicate) {
  static_assert(mlir::arith::getMaxEnumValForCmpIPredicate() + 1 == 10,
                "audit dataflow.stream predicate semantics");
  return mlir::arith::applyCmpPredicate(predicate, current, limit);
}

llvm::Expected<llvm::APInt>
evaluateStreamStep(const llvm::APInt &current, const llvm::APInt &step,
                   dataflow::StreamStepKind stepKind) {
  static_assert(dataflow::getMaxEnumValForStreamStepKind() + 1 == 8,
                "audit dataflow.stream step semantics");
  switch (stepKind) {
  case dataflow::StreamStepKind::Add:
    return current + step;
  case dataflow::StreamStepKind::Sub:
    return current - step;
  case dataflow::StreamStepKind::Mul:
    return current * step;
  case dataflow::StreamStepKind::SDiv:
    if (step.isZero())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "arith.divsi divisor must be non-zero");
    if (current.isMinSignedValue() && step.isAllOnes())
      return llvm::createStringError(std::errc::result_out_of_range,
                                     "arith.divsi signed overflow");
    return current.sdiv(step);
  case dataflow::StreamStepKind::UDiv:
    if (step.isZero())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "arith.divui divisor must be non-zero");
    return current.udiv(step);
  case dataflow::StreamStepKind::ShL:
  case dataflow::StreamStepKind::AShr:
  case dataflow::StreamStepKind::LShr: {
    // The shift amount is unsigned. getLimitedValue saturates at the bit
    // width without narrowing, so wide step values are rejected
    // deterministically instead of tripping APInt narrow getters.
    const std::uint64_t amount = step.getLimitedValue(step.getBitWidth());
    if (amount >= step.getBitWidth()) {
      llvm::SmallString<40> decimal;
      step.toString(decimal, 10, /*Signed=*/false);
      return llvm::createStringError(
          std::errc::invalid_argument,
          "dataflow.stream shift amount must be in [0, %u), got %s",
          step.getBitWidth(), decimal.c_str());
    }
    if (stepKind == dataflow::StreamStepKind::ShL)
      return current.shl(static_cast<unsigned>(amount));
    if (stepKind == dataflow::StreamStepKind::AShr)
      return current.ashr(static_cast<unsigned>(amount));
    return current.lshr(static_cast<unsigned>(amount));
  }
  }
  llvm_unreachable("unknown dataflow.stream step kind");
}

std::optional<int64_t> getIntegerConstant(mlir::Value value) {
  auto narrow = [](const mlir::APInt &integer) -> std::optional<int64_t> {
    if (!integer.isSignedIntN(64))
      return std::nullopt;
    return integer.getSExtValue();
  };
  if (auto constant = value.getDefiningOp<dataflow::ConstantOp>()) {
    if (auto boolean = llvm::dyn_cast<mlir::BoolAttr>(constant.getConstValue()))
      return boolean.getValue();
    if (auto integer =
            llvm::dyn_cast<mlir::IntegerAttr>(constant.getConstValue()))
      return narrow(integer.getValue());
  }
  mlir::APInt integer;
  if (mlir::matchPattern(value, mlir::m_ConstantInt(&integer)))
    return narrow(integer);
  return std::nullopt;
}

std::optional<bool> getKnownBool(mlir::Value value,
                                 llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second || !value.getType().isInteger(1))
    return std::nullopt;
  auto integer = getIntegerConstant(value);
  if (integer)
    return *integer != 0;
  if (auto invariant = value.getDefiningOp<dataflow::InvariantOp>())
    return getKnownBool(invariant.getInit(), visited);
  if (auto gate = value.getDefiningOp<dataflow::GateOp>()) {
    if (value != gate.getAfterValue())
      return std::nullopt;
    return getKnownBool(gate.getBeforeValue(), visited);
  }
  if (auto result = llvm::dyn_cast<mlir::OpResult>(value)) {
    if (auto sync = llvm::dyn_cast<dataflow::SyncOp>(result.getOwner());
        sync && result.getResultNumber() < sync.getInputs().size())
      return getKnownBool(sync.getInputs()[result.getResultNumber()], visited);
    if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(result.getOwner()))
      return getKnownBool(demux.getInput(), visited);
  }
  return std::nullopt;
}

std::optional<bool> getKnownBool(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  return getKnownBool(value, visited);
}

mlir::Value unwrapSynchronizedSelection(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    auto result = llvm::dyn_cast<mlir::OpResult>(value);
    auto sync =
        result ? llvm::dyn_cast<dataflow::SyncOp>(result.getOwner()) : nullptr;
    if (sync && result.getResultNumber() < sync.getInputs().size()) {
      value = sync.getInputs()[result.getResultNumber()];
      continue;
    }
    auto mux = value.getDefiningOp<dataflow::MuxOp>();
    if (!mux || mux.getInputs().size() != 2 || !value.getType().isInteger(1))
      break;
    auto falseValue = getKnownBool(mux.getInputs()[0]);
    auto trueValue = getKnownBool(mux.getInputs()[1]);
    if (!falseValue || !trueValue || *falseValue || !*trueValue)
      break;
    value = mux.getSel();
  }
  return value;
}

bool haveSynchronizedValueCorrespondence(mlir::Value lhs, mlir::Value rhs) {
  return unwrapSynchronizedSelection(lhs) == unwrapSynchronizedSelection(rhs);
}

mlir::Value graphStartForArgument(mlir::Value value) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || argument.getOwner()->getNumArguments() == 0)
    return {};
  auto graph = llvm::dyn_cast_or_null<dataflow::GraphOp>(
      argument.getOwner()->getParentOp());
  if (!graph || argument.getOwner() != &graph.getBody().front())
    return {};
  if (argument.getArgNumber() != 0 &&
      graph.getInputPortKind(argument.getArgNumber() - 1) ==
          dataflow::GraphPortKind::Stream)
    return {};
  return graph.getStart();
}

mlir::Value getActivationSource(mlir::Value value,
                                llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return {};
  if (mlir::Value start = graphStartForArgument(value))
    return start;

  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return {};
  if (auto constant = llvm::dyn_cast<dataflow::ConstantOp>(def))
    return constant.getCtrl();
  if (auto gate = llvm::dyn_cast<dataflow::GateOp>(def))
    return value == gate.getAfterValue() ? gate.getBeforeCond() : mlir::Value{};
  if (llvm::isa<mlir::arith::ConstantOp>(def))
    return {};
  if (llvm::isa<dataflow::StreamOp, dataflow::CarryOp, dataflow::InvariantOp,
                dataflow::GateOp, dataflow::DemuxOp, dataflow::ParallelizeOp,
                dataflow::SerializeOp, dataflow::UnpackOp>(def))
    return {};
  if (!dataflow::isCanonicalDataflowActor(def))
    return {};

  mlir::Value common;
  for (mlir::Value operand : def->getOperands()) {
    if (operand.getDefiningOp<mlir::arith::ConstantOp>())
      continue;
    llvm::DenseSet<mlir::Value> branchVisited = visited;
    mlir::Value activation = getActivationSource(operand, branchVisited);
    if (!activation)
      return {};
    if (common && activation != common)
      return {};
    common = activation;
  }
  return common;
}

mlir::Value getActivationSource(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  return getActivationSource(value, visited);
}

mlir::Value unwrapSelectorOrdinal(mlir::Value selector, unsigned arity) {
  if (arity < 2)
    return {};
  if (arity == 2) {
    if (!selector.getType().isInteger(1))
      return {};
    if (auto trunc = selector.getDefiningOp<mlir::arith::TruncIOp>())
      return trunc.getIn();
    if (auto compare = selector.getDefiningOp<mlir::arith::CmpIOp>()) {
      auto unwrapComparison = [&](mlir::Value ordinal, mlir::Value constant,
                                  int64_t expected) -> mlir::Value {
        auto known = getIntegerConstant(constant);
        return known && *known == expected ? ordinal : mlir::Value{};
      };
      switch (compare.getPredicate()) {
      case mlir::arith::CmpIPredicate::eq:
        if (mlir::Value ordinal =
                unwrapComparison(compare.getLhs(), compare.getRhs(), 1))
          return ordinal;
        return unwrapComparison(compare.getRhs(), compare.getLhs(), 1);
      case mlir::arith::CmpIPredicate::ne:
        if (mlir::Value ordinal =
                unwrapComparison(compare.getLhs(), compare.getRhs(), 0))
          return ordinal;
        return unwrapComparison(compare.getRhs(), compare.getLhs(), 0);
      case mlir::arith::CmpIPredicate::sge:
      case mlir::arith::CmpIPredicate::uge:
        if (auto threshold = getIntegerConstant(compare.getRhs());
            threshold && *threshold > 0)
          return compare.getLhs();
        return {};
      default:
        break;
      }
    }
    return selector;
  }
  if (!llvm::isa<mlir::IndexType>(selector.getType()))
    return {};
  if (auto cast = selector.getDefiningOp<mlir::arith::IndexCastOp>())
    return cast.getIn();
  return selector;
}

mlir::Value unwrapPhaseProjection(mlir::Value value,
                                  dataflow::StreamOp stream) {
  auto result = llvm::dyn_cast<mlir::OpResult>(value);
  auto demux =
      result ? llvm::dyn_cast<dataflow::DemuxOp>(result.getOwner()) : nullptr;
  if (!demux || demux.getOutputs().size() != 2 ||
      result.getResultNumber() != 1 || demux.getSel() != stream.getPhase())
    return {};
  auto invariant = demux.getInput().getDefiningOp<dataflow::InvariantOp>();
  if (!invariant || invariant.getCond() != stream.getPhase() ||
      invariant.getOutput() != demux.getInput())
    return {};
  return invariant.getInit();
}

void collectRootStreams(mlir::Value value,
                        llvm::DenseSet<mlir::Operation *> &roots,
                        llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return;
  if (auto stream = value.getDefiningOp<dataflow::StreamOp>()) {
    if (value == stream.getIv())
      roots.insert(stream);
    return;
  }
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return;
  if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def)) {
    auto result = llvm::cast<mlir::OpResult>(value);
    auto stream = demux.getSel().getDefiningOp<dataflow::StreamOp>();
    if (stream && demux.getSel() == stream.getPhase() &&
        result.getResultNumber() == 1 && unwrapPhaseProjection(value, stream))
      return;
  }
  for (mlir::Value operand : def->getOperands())
    collectRootStreams(operand, roots, visited);
}

dataflow::StreamOp findRootStream(mlir::Value value) {
  llvm::DenseSet<mlir::Operation *> roots;
  llvm::DenseSet<mlir::Value> visited;
  collectRootStreams(value, roots, visited);
  if (roots.size() != 1)
    return {};
  return llvm::cast<dataflow::StreamOp>(*roots.begin());
}

bool isBodyAligned(mlir::Value value, dataflow::StreamOp stream,
                   llvm::DenseSet<mlir::Value> &visited) {
  if (value == stream.getIv())
    return true;
  if (!value || !visited.insert(value).second)
    return false;
  if (unwrapPhaseProjection(value, stream))
    return true;

  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (auto constant = llvm::dyn_cast<dataflow::ConstantOp>(def))
    return isBodyAligned(constant.getCtrl(), stream, visited);
  if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(def)) {
    auto result = llvm::cast<mlir::OpResult>(value);
    if (demux.getOutputs().size() == 2 && result.getResultNumber() == 1 &&
        demux.getSel() == stream.getPhase()) {
      auto invariant = demux.getInput().getDefiningOp<dataflow::InvariantOp>();
      return invariant && invariant.getCond() == stream.getPhase();
    }
  }
  if (llvm::isa<dataflow::StreamOp, dataflow::CarryOp, dataflow::InvariantOp,
                dataflow::GateOp, dataflow::MuxOp, dataflow::DemuxOp>(def))
    return false;
  if (def->getNumOperands() == 0)
    return false;
  return llvm::all_of(def->getOperands(), [&](mlir::Value operand) {
    llvm::DenseSet<mlir::Value> branchVisited = visited;
    return isBodyAligned(operand, stream, branchVisited);
  });
}

bool isBodyAligned(mlir::Value value, dataflow::StreamOp stream) {
  llvm::DenseSet<mlir::Value> visited;
  return isBodyAligned(value, stream, visited);
}

struct LinearExpr {
  LinearExpr() = default;
  explicit LinearExpr(int64_t constant) : constant(constant) {}

  int64_t constant = 0;
  llvm::DenseMap<mlir::Value, int64_t> terms;
  bool valid = true;

  void addTerm(mlir::Value value, int64_t coefficient) {
    if (!valid || coefficient == 0)
      return;
    int64_t &current = terms[value];
    int64_t sum;
    if (llvm::AddOverflow(current, coefficient, sum)) {
      valid = false;
      return;
    }
    current = sum;
    if (current == 0)
      terms.erase(value);
  }

  void add(const LinearExpr &other, int64_t scale = 1) {
    if (!other.valid) {
      valid = false;
      return;
    }
    int64_t scaled;
    int64_t sum;
    if (llvm::MulOverflow(scale, other.constant, scaled) ||
        llvm::AddOverflow(constant, scaled, sum)) {
      valid = false;
      return;
    }
    constant = sum;
    for (auto [value, coefficient] : other.terms) {
      if (llvm::MulOverflow(scale, coefficient, scaled)) {
        valid = false;
        return;
      }
      addTerm(value, scaled);
    }
  }
};

LinearExpr operator-(LinearExpr lhs, const LinearExpr &rhs) {
  lhs.add(rhs, -1);
  return lhs;
}

LinearExpr scale(LinearExpr value, int64_t factor) {
  int64_t scaled;
  if (!value.valid || llvm::MulOverflow(value.constant, factor, scaled)) {
    value.valid = false;
    return value;
  }
  value.constant = scaled;
  llvm::SmallVector<mlir::Value, 4> zeroTerms;
  for (auto &entry : value.terms) {
    if (llvm::MulOverflow(entry.second, factor, scaled)) {
      value.valid = false;
      return value;
    }
    entry.second = scaled;
    if (scaled == 0)
      zeroTerms.push_back(entry.first);
  }
  for (mlir::Value term : zeroTerms)
    value.terms.erase(term);
  return value;
}

LinearExpr parseLinear(mlir::Value value, dataflow::StreamOp stream,
                       llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return [] {
      LinearExpr result;
      result.valid = false;
      return result;
    }();
  if (value == stream.getIv()) {
    LinearExpr result;
    result.addTerm(value, 1);
    return result;
  }
  if (mlir::Value projected = unwrapPhaseProjection(value, stream)) {
    llvm::DenseSet<mlir::Value> branchVisited = visited;
    return parseLinear(projected, stream, branchVisited);
  }
  if (auto integer = getIntegerConstant(value))
    return LinearExpr(*integer);

  mlir::Operation *def = value.getDefiningOp();
  if (!def) {
    LinearExpr result;
    result.addTerm(value, 1);
    return result;
  }
  if (auto add = llvm::dyn_cast<mlir::arith::AddIOp>(def)) {
    llvm::DenseSet<mlir::Value> lhsVisited = visited;
    llvm::DenseSet<mlir::Value> rhsVisited = visited;
    auto lhs = parseLinear(add.getLhs(), stream, lhsVisited);
    lhs.add(parseLinear(add.getRhs(), stream, rhsVisited));
    return lhs;
  }
  if (auto sub = llvm::dyn_cast<mlir::arith::SubIOp>(def)) {
    llvm::DenseSet<mlir::Value> lhsVisited = visited;
    llvm::DenseSet<mlir::Value> rhsVisited = visited;
    auto lhs = parseLinear(sub.getLhs(), stream, lhsVisited);
    lhs.add(parseLinear(sub.getRhs(), stream, rhsVisited), -1);
    return lhs;
  }
  if (auto mul = llvm::dyn_cast<mlir::arith::MulIOp>(def)) {
    if (auto lhs = getIntegerConstant(mul.getLhs())) {
      llvm::DenseSet<mlir::Value> branchVisited = visited;
      return scale(parseLinear(mul.getRhs(), stream, branchVisited), *lhs);
    }
    if (auto rhs = getIntegerConstant(mul.getRhs())) {
      llvm::DenseSet<mlir::Value> branchVisited = visited;
      return scale(parseLinear(mul.getLhs(), stream, branchVisited), *rhs);
    }
  }
  if (auto cast = llvm::dyn_cast<mlir::arith::IndexCastOp>(def))
    return parseLinear(cast.getIn(), stream, visited);
  if (auto cast = llvm::dyn_cast<mlir::arith::ExtSIOp>(def))
    return parseLinear(cast.getIn(), stream, visited);
  if (auto cast = llvm::dyn_cast<mlir::arith::ExtUIOp>(def))
    return parseLinear(cast.getIn(), stream, visited);

  LinearExpr result;
  result.addTerm(value, 1);
  return result;
}

LinearExpr parseLinear(mlir::Value value, dataflow::StreamOp stream) {
  llvm::DenseSet<mlir::Value> visited;
  return parseLinear(value, stream, visited);
}

bool isKnownNonNegative(mlir::Value value,
                        llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return false;
  if (auto integer = getIntegerConstant(value))
    return *integer >= 0;
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (auto cast = llvm::dyn_cast<mlir::arith::IndexCastOp>(def))
    return isKnownNonNegative(cast.getIn(), visited);
  if (llvm::isa<mlir::arith::DivUIOp, mlir::arith::RemUIOp>(def))
    return true;
  if (auto select = llvm::dyn_cast<mlir::arith::SelectOp>(def)) {
    llvm::DenseSet<mlir::Value> trueVisited = visited;
    llvm::DenseSet<mlir::Value> falseVisited = visited;
    return isKnownNonNegative(select.getTrueValue(), trueVisited) &&
           isKnownNonNegative(select.getFalseValue(), falseVisited);
  }
  if (llvm::isa<mlir::arith::AddIOp, mlir::arith::MulIOp>(def)) {
    llvm::DenseSet<mlir::Value> lhsVisited = visited;
    llvm::DenseSet<mlir::Value> rhsVisited = visited;
    return isKnownNonNegative(def->getOperand(0), lhsVisited) &&
           isKnownNonNegative(def->getOperand(1), rhsVisited);
  }
  return false;
}

bool isKnownNonNegative(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  return isKnownNonNegative(value, visited);
}

bool isKnownNonNegative(const LinearExpr &expr) {
  if (!expr.valid || expr.constant < 0)
    return false;
  return llvm::all_of(expr.terms, [](const auto &entry) {
    return entry.second >= 0 && isKnownNonNegative(entry.first);
  });
}

bool equalsConstant(const LinearExpr &expr, int64_t value) {
  return expr.valid && expr.terms.empty() && expr.constant == value;
}

struct Interval {
  LinearExpr lower;
  LinearExpr upper;
  bool valid = true;

  void tightenLower(const LinearExpr &candidate) {
    if (!valid || !candidate.valid) {
      valid = false;
      return;
    }
    if (isKnownNonNegative(candidate - lower)) {
      lower = candidate;
      return;
    }
    if (!isKnownNonNegative(lower - candidate))
      valid = false;
  }

  void tightenUpper(const LinearExpr &candidate) {
    if (!valid || !candidate.valid) {
      valid = false;
      return;
    }
    if (isKnownNonNegative(upper - candidate)) {
      upper = candidate;
      return;
    }
    if (!isKnownNonNegative(candidate - upper))
      valid = false;
  }
};

struct Threshold {
  LinearExpr value;
  bool trueIsLess = true;
};

std::optional<Threshold> parseThreshold(mlir::Value condition,
                                        dataflow::StreamOp stream) {
  auto compare = condition.getDefiningOp<mlir::arith::CmpIOp>();
  if (!compare)
    return std::nullopt;
  auto predicate = compare.getPredicate();
  bool trueIsLess = predicate == mlir::arith::CmpIPredicate::slt ||
                    predicate == mlir::arith::CmpIPredicate::ult;
  bool trueIsGreaterEqual = predicate == mlir::arith::CmpIPredicate::sge ||
                            predicate == mlir::arith::CmpIPredicate::uge;
  if (!trueIsLess && !trueIsGreaterEqual)
    return std::nullopt;

  LinearExpr lhs = parseLinear(compare.getLhs(), stream);
  LinearExpr rhs = parseLinear(compare.getRhs(), stream);
  mlir::Value iv = stream.getIv();
  int64_t coefficient;
  if (llvm::SubOverflow(lhs.terms.lookup(iv), rhs.terms.lookup(iv),
                        coefficient) ||
      coefficient != 1)
    return std::nullopt;
  lhs.addTerm(iv, -lhs.terms.lookup(iv));
  rhs.addTerm(iv, -rhs.terms.lookup(iv));
  LinearExpr threshold = rhs - lhs;
  if (!threshold.valid)
    return std::nullopt;
  return Threshold{std::move(threshold), trueIsLess};
}

void collectLaneIntervals(mlir::Value route, dataflow::StreamOp stream,
                          unsigned lane, Interval interval,
                          llvm::SmallVectorImpl<Interval> &matches,
                          llvm::DenseSet<mlir::Value> &visited) {
  if (!route || !interval.valid || !visited.insert(route).second)
    return;
  if (auto cast = route.getDefiningOp<mlir::arith::IndexCastOp>()) {
    collectLaneIntervals(cast.getIn(), stream, lane, std::move(interval),
                         matches, visited);
    return;
  }
  if (auto integer = getIntegerConstant(route)) {
    if (*integer == static_cast<int64_t>(lane))
      matches.push_back(std::move(interval));
    return;
  }
  if (route == stream.getIv()) {
    LinearExpr lower(static_cast<int64_t>(lane));
    LinearExpr upper(static_cast<int64_t>(lane) + 1);
    interval.tightenLower(lower);
    interval.tightenUpper(upper);
    if (interval.valid)
      matches.push_back(std::move(interval));
    return;
  }
  auto select = route.getDefiningOp<mlir::arith::SelectOp>();
  if (!select)
    return;
  auto threshold = parseThreshold(select.getCondition(), stream);
  if (!threshold)
    return;

  Interval trueInterval = interval;
  Interval falseInterval = interval;
  if (threshold->trueIsLess) {
    trueInterval.tightenUpper(threshold->value);
    falseInterval.tightenLower(threshold->value);
  } else {
    trueInterval.tightenLower(threshold->value);
    falseInterval.tightenUpper(threshold->value);
  }
  llvm::DenseSet<mlir::Value> trueVisited = visited;
  llvm::DenseSet<mlir::Value> falseVisited = visited;
  collectLaneIntervals(select.getTrueValue(), stream, lane,
                       std::move(trueInterval), matches, trueVisited);
  collectLaneIntervals(select.getFalseValue(), stream, lane,
                       std::move(falseInterval), matches, falseVisited);
}

bool routeVisitsLaneOnce(mlir::Value route, unsigned lane) {
  dataflow::StreamOp stream = findRootStream(route);
  if (!stream || !isBodyAligned(route, stream) ||
      stream.getStepKind() != dataflow::StreamStepKind::Add ||
      (stream.getPredicate() != mlir::arith::CmpIPredicate::slt &&
       stream.getPredicate() != mlir::arith::CmpIPredicate::ult))
    return false;
  auto step = getIntegerConstant(stream.getStep());
  if (!step || *step != 1)
    return false;

  Interval domain{parseLinear(stream.getInit(), stream),
                  parseLinear(stream.getLimit(), stream), true};
  if (!domain.lower.valid || !domain.upper.valid)
    return false;
  llvm::SmallVector<Interval, 2> matches;
  llvm::DenseSet<mlir::Value> visited;
  collectLaneIntervals(route, stream, lane, std::move(domain), matches,
                       visited);
  return matches.size() == 1 &&
         equalsConstant(matches.front().upper - matches.front().lower, 1);
}

bool selectorPredicateVisitsLaneOnce(mlir::Value selector, unsigned lane) {
  if (lane > 1)
    return false;
  auto compare = selector.getDefiningOp<mlir::arith::CmpIOp>();
  if (!compare)
    return false;
  dataflow::StreamOp stream = findRootStream(compare.getLhs());
  auto threshold = stream ? parseThreshold(selector, stream) : std::nullopt;
  if (!stream || !threshold ||
      stream.getStepKind() != dataflow::StreamStepKind::Add ||
      (stream.getPredicate() != mlir::arith::CmpIPredicate::slt &&
       stream.getPredicate() != mlir::arith::CmpIPredicate::ult) ||
      getIntegerConstant(stream.getStep()) != 1)
    return false;

  Interval interval{parseLinear(stream.getInit(), stream),
                    parseLinear(stream.getLimit(), stream), true};
  const bool trueLane = lane == 1;
  if (trueLane == threshold->trueIsLess)
    interval.tightenUpper(threshold->value);
  else
    interval.tightenLower(threshold->value);
  return interval.valid && equalsConstant(interval.upper - interval.lower, 1);
}

struct SelectorDescription {
  mlir::Value routeSelector;
  mlir::Value route;
  mlir::Value activity;
  dataflow::StreamOp stream;
  unsigned arity = 0;
};

std::optional<SelectorDescription> describeSelector(mlir::Value selector,
                                                    unsigned arity) {
  mlir::Value ordinal = unwrapSelectorOrdinal(selector, arity);
  if (!ordinal)
    return std::nullopt;

  mlir::Value route = ordinal;
  mlir::Value activity;
  if (auto result = llvm::dyn_cast<mlir::OpResult>(ordinal)) {
    if (auto filter = llvm::dyn_cast<dataflow::DemuxOp>(result.getOwner());
        filter && filter.getOutputs().size() == 2 &&
        result.getResultNumber() == 1) {
      route = filter.getInput();
      activity = filter.getSel();
    }
  }
  dataflow::StreamOp stream = findRootStream(route);
  if (!stream || !isBodyAligned(route, stream))
    return std::nullopt;
  mlir::Value routeSelector = selector;
  if (activity) {
    auto activities = activity.getDefiningOp<dataflow::MuxOp>();
    if (!activities || activities.getInputs().size() != arity ||
        unwrapSelectorOrdinal(activities.getSel(), arity) != route)
      return std::nullopt;
    routeSelector = activities.getSel();
  }
  return SelectorDescription{routeSelector, route, activity, stream, arity};
}

bool isGraphStreamInput(mlir::Value value) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || argument.getArgNumber() == 0)
    return false;
  auto graph = llvm::dyn_cast_or_null<dataflow::GraphOp>(
      argument.getOwner()->getParentOp());
  return graph && argument.getOwner() == &graph.getBody().front() &&
         graph.getInputPortKind(argument.getArgNumber() - 1) ==
             dataflow::GraphPortKind::Stream;
}

struct StaticSelectiveRouterLeg {
  mlir::Value selector;
  mlir::Value ordinal;
  mlir::Value event;
  std::int64_t threshold = 0;
  unsigned lane = 0;
};

struct SelectiveRouterLeafDescription {
  mlir::Value activation;
  mlir::Value route;
  mlir::Value activity;
  mlir::Value activityEvent;
  mlir::Value synchronization;
  unsigned lane = 0;
};

std::optional<StaticSelectiveRouterLeg>
describeStaticSelectiveRouterLeg(dataflow::DemuxOp payload, unsigned lane) {
  if (!payload || payload.getOutputs().size() != 2 || lane > 1)
    return std::nullopt;
  auto compare = payload.getSel().getDefiningOp<mlir::arith::CmpIOp>();
  if (!compare || compare.getPredicate() != mlir::arith::CmpIPredicate::uge)
    return std::nullopt;
  auto threshold = compare.getRhs().getDefiningOp<dataflow::ConstantOp>();
  auto value = getIntegerConstant(compare.getRhs());
  if (!threshold || !value || *value <= 0 ||
      threshold.getValue().getType() != compare.getLhs().getType())
    return std::nullopt;
  return StaticSelectiveRouterLeg{payload.getSel(), compare.getLhs(),
                                  threshold.getCtrl(), *value, lane};
}

std::optional<mlir::Value> findRoutedResult(mlir::Value selector,
                                            mlir::Value input, unsigned lane) {
  mlir::Value found;
  for (mlir::Operation *user : input.getUsers()) {
    auto demux = llvm::dyn_cast<dataflow::DemuxOp>(user);
    if (!demux || demux.getSel() != selector || demux.getInput() != input ||
        demux.getOutputs().size() != 2)
      continue;
    if (found)
      return std::nullopt;
    found = demux.getOutputs()[lane];
  }
  return found ? std::optional<mlir::Value>{found} : std::nullopt;
}

std::optional<mlir::Value> findRightLocalOrdinal(mlir::Value routedOrdinal,
                                                 mlir::Value event,
                                                 std::int64_t threshold) {
  mlir::Value found;
  for (mlir::Operation *user : routedOrdinal.getUsers()) {
    auto subtract = llvm::dyn_cast<mlir::arith::SubIOp>(user);
    if (!subtract || subtract.getLhs() != routedOrdinal ||
        getIntegerConstant(subtract.getRhs()) != threshold)
      continue;
    auto constant = subtract.getRhs().getDefiningOp<dataflow::ConstantOp>();
    if (!constant || constant.getCtrl() != event)
      continue;
    if (found)
      return std::nullopt;
    found = subtract.getResult();
  }
  return found ? std::optional<mlir::Value>{found} : std::nullopt;
}

std::optional<std::int64_t> getRouterThreshold(mlir::Value selector,
                                               mlir::Value ordinal,
                                               mlir::Value event) {
  auto compare = selector.getDefiningOp<mlir::arith::CmpIOp>();
  if (!compare || compare.getPredicate() != mlir::arith::CmpIPredicate::uge ||
      compare.getLhs() != ordinal)
    return std::nullopt;
  auto constant = compare.getRhs().getDefiningOp<dataflow::ConstantOp>();
  auto threshold = getIntegerConstant(compare.getRhs());
  if (!constant || constant.getCtrl() != event || !threshold || *threshold <= 0)
    return std::nullopt;
  return threshold;
}

struct RouterLeaf {
  mlir::Value value;
  mlir::Value event;
};

bool hasPayloadRouterChild(mlir::Value payload) {
  return llvm::any_of(payload.getUsers(), [&](mlir::Operation *user) {
    auto demux = llvm::dyn_cast<dataflow::DemuxOp>(user);
    return demux && demux.getInput() == payload &&
           demux.getOutputs().size() == 2 &&
           describeStaticSelectiveRouterLeg(demux, 0).has_value();
  });
}

bool collectPayloadRouterLeaves(mlir::Value payload, mlir::Value ordinal,
                                mlir::Value event,
                                llvm::SmallVectorImpl<RouterLeaf> &leaves,
                                bool requireNode = false) {
  dataflow::DemuxOp router;
  std::int64_t threshold = 0;
  for (mlir::Operation *user : payload.getUsers()) {
    auto demux = llvm::dyn_cast<dataflow::DemuxOp>(user);
    if (!demux || demux.getInput() != payload || demux.getOutputs().size() != 2)
      continue;
    auto candidate = getRouterThreshold(demux.getSel(), ordinal, event);
    if (!candidate)
      continue;
    if (router)
      return false;
    router = demux;
    threshold = *candidate;
  }
  if (!router) {
    if (requireNode)
      return false;
    leaves.push_back({payload, event});
    return true;
  }

  const std::size_t leftBegin = leaves.size();
  if (hasPayloadRouterChild(router.getOutputs()[0])) {
    auto leftEvent = findRoutedResult(router.getSel(), event, 0);
    auto leftOrdinal = findRoutedResult(router.getSel(), ordinal, 0);
    if (!leftEvent || !leftOrdinal ||
        !collectPayloadRouterLeaves(router.getOutputs()[0], *leftOrdinal,
                                    *leftEvent, leaves))
      return false;
  } else {
    leaves.push_back({router.getOutputs()[0], {}});
  }
  const std::size_t leftCount = leaves.size() - leftBegin;
  const std::size_t rightBegin = leaves.size();
  if (hasPayloadRouterChild(router.getOutputs()[1])) {
    auto rightEvent = findRoutedResult(router.getSel(), event, 1);
    auto routedRightOrdinal = findRoutedResult(router.getSel(), ordinal, 1);
    if (!rightEvent || !routedRightOrdinal)
      return false;
    auto rightOrdinal =
        findRightLocalOrdinal(*routedRightOrdinal, *rightEvent, threshold);
    if (!rightOrdinal ||
        !collectPayloadRouterLeaves(router.getOutputs()[1], *rightOrdinal,
                                    *rightEvent, leaves))
      return false;
  } else {
    leaves.push_back({router.getOutputs()[1], {}});
  }
  const std::size_t rightCount = leaves.size() - rightBegin;
  const std::size_t total = leftCount + rightCount;
  return leftCount == static_cast<std::size_t>(threshold) &&
         leftCount == (total + 1) / 2;
}

bool collectActivityRouterLeaves(mlir::Value activity, mlir::Value ordinal,
                                 mlir::Value event,
                                 llvm::SmallVectorImpl<RouterLeaf> &leaves) {
  auto router = activity.getDefiningOp<dataflow::MuxOp>();
  auto threshold = router && router.getInputs().size() == 2
                       ? getRouterThreshold(router.getSel(), ordinal, event)
                       : std::nullopt;
  if (!threshold) {
    leaves.push_back({activity, event});
    return true;
  }

  auto leftEvent = findRoutedResult(router.getSel(), event, 0);
  auto rightEvent = findRoutedResult(router.getSel(), event, 1);
  if (!leftEvent || !rightEvent)
    return false;

  auto collectChild = [&](mlir::Value child, mlir::Value childEvent,
                          unsigned lane) {
    auto childRouter = child.getDefiningOp<dataflow::MuxOp>();
    auto childCompare =
        childRouter && childRouter.getInputs().size() == 2
            ? childRouter.getSel().getDefiningOp<mlir::arith::CmpIOp>()
            : nullptr;
    auto childThreshold =
        childCompare
            ? childCompare.getRhs().getDefiningOp<dataflow::ConstantOp>()
            : nullptr;
    bool nested =
        childCompare && childThreshold &&
        childCompare.getPredicate() == mlir::arith::CmpIPredicate::uge &&
        childThreshold.getCtrl() == childEvent;
    if (!nested) {
      leaves.push_back({child, childEvent});
      return true;
    }

    auto routedOrdinal = findRoutedResult(router.getSel(), ordinal, lane);
    if (!routedOrdinal)
      return false;
    mlir::Value childOrdinal = *routedOrdinal;
    if (lane == 1) {
      auto local = findRightLocalOrdinal(childOrdinal, childEvent, *threshold);
      if (!local)
        return false;
      childOrdinal = *local;
    }
    return getRouterThreshold(childRouter.getSel(), childOrdinal, childEvent) &&
           collectActivityRouterLeaves(child, childOrdinal, childEvent, leaves);
  };

  const std::size_t leftBegin = leaves.size();
  if (!collectChild(router.getInputs()[0], *leftEvent, 0))
    return false;
  const std::size_t leftCount = leaves.size() - leftBegin;
  const std::size_t rightBegin = leaves.size();
  if (!collectChild(router.getInputs()[1], *rightEvent, 1))
    return false;
  const std::size_t rightCount = leaves.size() - rightBegin;
  const std::size_t total = leftCount + rightCount;
  return leftCount == static_cast<std::size_t>(*threshold) &&
         leftCount == (total + 1) / 2;
}

bool activityIsTrueForEvent(mlir::Value value, mlir::Value event,
                            mlir::Value branchSelector,
                            std::optional<unsigned> branchLane,
                            llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return false;
  if (auto known = getKnownBool(value)) {
    auto constant = value.getDefiningOp<dataflow::ConstantOp>();
    return constant && *known && constant.getCtrl() == event;
  }
  auto mux = value.getDefiningOp<dataflow::MuxOp>();
  if (!mux)
    return false;
  if (auto selected = getKnownBool(mux.getSel())) {
    unsigned lane = *selected ? 1 : 0;
    llvm::DenseSet<mlir::Value> branchVisited = visited;
    return activityIsTrueForEvent(mux.getInputs()[lane], event, branchSelector,
                                  branchLane, branchVisited);
  }
  if (branchLane &&
      haveSynchronizedValueCorrespondence(mux.getSel(), branchSelector)) {
    if (*branchLane >= mux.getInputs().size())
      return false;
    llvm::DenseSet<mlir::Value> branchVisited = visited;
    return activityIsTrueForEvent(mux.getInputs()[*branchLane], event,
                                  branchSelector, branchLane, branchVisited);
  }
  return llvm::all_of(mux.getInputs(), [&](mlir::Value input) {
    llvm::DenseSet<mlir::Value> branchVisited = visited;
    return activityIsTrueForEvent(input, event, branchSelector, branchLane,
                                  branchVisited);
  });
}

bool activityIsTrueForEvent(mlir::Value value, mlir::Value event,
                            mlir::Value branchSelector = {},
                            std::optional<unsigned> branchLane = std::nullopt) {
  if (!value)
    return true;
  llvm::DenseSet<mlir::Value> visited;
  return activityIsTrueForEvent(value, event, branchSelector, branchLane,
                                visited);
}

bool activityIsTotalForEvent(mlir::Value value, mlir::Value event,
                             llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return false;
  if (auto constant = value.getDefiningOp<dataflow::ConstantOp>())
    return value.getType().isInteger(1) && constant.getCtrl() == event &&
           llvm::isa<mlir::BoolAttr>(constant.getConstValue());

  auto mux = value.getDefiningOp<dataflow::MuxOp>();
  if (!mux || mux.getInputs().size() != 2)
    return false;
  auto selector = llvm::dyn_cast<mlir::OpResult>(mux.getSel());
  auto sync = selector ? llvm::dyn_cast<dataflow::SyncOp>(selector.getOwner())
                       : nullptr;
  if (!sync || sync.getInputs().size() != 2 || sync.getOutputs().size() != 2 ||
      selector.getResultNumber() != 0)
    return false;
  mlir::Value synchronizedEvent = sync.getOutputs()[1];
  mlir::Value predecessor = sync.getInputs()[1];
  auto descendsFromEvent = [&](mlir::Value candidate) {
    while (candidate != event) {
      auto result = llvm::dyn_cast<mlir::OpResult>(candidate);
      auto parent = result ? llvm::dyn_cast<dataflow::SyncOp>(result.getOwner())
                           : nullptr;
      if (!parent || result.getResultNumber() != 1 ||
          parent.getInputs().size() != 2)
        return false;
      candidate = parent.getInputs()[1];
    }
    return true;
  };
  if (!descendsFromEvent(predecessor))
    return false;
  auto inactiveAt = [&](mlir::Value candidate) {
    auto constant = candidate.getDefiningOp<dataflow::ConstantOp>();
    return constant && candidate.getType().isInteger(1) &&
           constant.getCtrl() == synchronizedEvent &&
           llvm::isa<mlir::BoolAttr>(constant.getConstValue());
  };
  for (auto [inactive, active] :
       {std::pair{mux.getInputs()[0], mux.getInputs()[1]},
        std::pair{mux.getInputs()[1], mux.getInputs()[0]}}) {
    if (!inactiveAt(inactive))
      continue;
    llvm::DenseSet<mlir::Value> branchVisited = visited;
    if (activityIsTotalForEvent(active, event, branchVisited))
      return true;
  }
  return false;
}

std::optional<mlir::Value>
describeSelectiveRouterActivityEvent(mlir::Value activity) {
  auto root = activity.getDefiningOp<dataflow::MuxOp>();
  if (!root || root.getInputs().size() != 2)
    return std::nullopt;
  auto compare = root.getSel().getDefiningOp<mlir::arith::CmpIOp>();
  auto threshold =
      compare ? getIntegerConstant(compare.getRhs()) : std::nullopt;
  auto controlledThreshold =
      compare ? compare.getRhs().getDefiningOp<dataflow::ConstantOp>()
              : nullptr;
  if (!compare || compare.getPredicate() != mlir::arith::CmpIPredicate::uge ||
      !threshold || *threshold <= 0 || !controlledThreshold)
    return std::nullopt;
  mlir::Value route = compare.getLhs();
  mlir::Value event = controlledThreshold.getCtrl();
  llvm::SmallVector<RouterLeaf, 8> leaves;
  if (!collectActivityRouterLeaves(activity, route, event, leaves) ||
      leaves.size() < 2)
    return std::nullopt;
  for (const RouterLeaf &leaf : leaves) {
    llvm::DenseSet<mlir::Value> visited;
    if (!activityIsTotalForEvent(leaf.value, leaf.event, visited))
      return std::nullopt;
  }
  dataflow::StreamOp stream = findRootStream(route);
  auto activation = stream ? getStreamActivation(stream) : std::nullopt;
  if (!activation || unwrapPhaseProjection(event, stream) != *activation)
    return std::nullopt;
  return event;
}

std::optional<mlir::Value>
describeSingleStreamActivityEvent(mlir::Value activity) {
  if (auto constant = activity.getDefiningOp<dataflow::ConstantOp>()) {
    if (!activity.getType().isInteger(1) ||
        !llvm::isa<mlir::BoolAttr>(constant.getConstValue()))
      return std::nullopt;
    return constant.getCtrl();
  }

  auto mux = activity.getDefiningOp<dataflow::MuxOp>();
  if (!mux || mux.getInputs().size() != 2)
    return std::nullopt;
  auto selector = llvm::dyn_cast<mlir::OpResult>(mux.getSel());
  auto sync = selector ? llvm::dyn_cast<dataflow::SyncOp>(selector.getOwner())
                       : nullptr;
  if (!sync || selector.getResultNumber() != 0 ||
      sync.getInputs().size() != 2 || sync.getOutputs().size() != 2)
    return std::nullopt;
  mlir::Value event = sync.getInputs()[1];
  llvm::DenseSet<mlir::Value> visited;
  return activityIsTotalForEvent(activity, event, visited)
             ? std::optional<mlir::Value>{event}
             : std::nullopt;
}

std::optional<mlir::Value>
describeSelectiveRouterPublicationEvent(mlir::Value publication) {
  auto root = publication.getDefiningOp<dataflow::MuxOp>();
  if (!root || root.getInputs().size() != 2)
    return std::nullopt;
  auto compare = root.getSel().getDefiningOp<mlir::arith::CmpIOp>();
  auto threshold =
      compare ? getIntegerConstant(compare.getRhs()) : std::nullopt;
  auto controlledThreshold =
      compare ? compare.getRhs().getDefiningOp<dataflow::ConstantOp>()
              : nullptr;
  if (!compare || compare.getPredicate() != mlir::arith::CmpIPredicate::uge ||
      !threshold || *threshold <= 0 || !controlledThreshold)
    return std::nullopt;

  mlir::Value route = compare.getLhs();
  mlir::Value event = controlledThreshold.getCtrl();
  llvm::SmallVector<RouterLeaf, 8> leaves;
  if (!collectActivityRouterLeaves(publication, route, event, leaves) ||
      leaves.size() < 2)
    return std::nullopt;

  std::optional<unsigned> resultOrdinal;
  for (const RouterLeaf &leaf : leaves) {
    auto result = llvm::dyn_cast<mlir::OpResult>(leaf.value);
    auto sync =
        result ? llvm::dyn_cast<dataflow::SyncOp>(result.getOwner()) : nullptr;
    if (!sync || sync.getInputs().size() != 2 ||
        sync.getOutputs().size() != 2 || sync.getInputs()[0] != leaf.event)
      return std::nullopt;
    unsigned ordinal = result.getResultNumber();
    if (ordinal > 1 || (resultOrdinal && *resultOrdinal != ordinal))
      return std::nullopt;
    resultOrdinal = ordinal;
  }
  return event;
}

std::optional<mlir::Value>
describeSingleStreamPublicationEvent(mlir::Value publication) {
  auto result = llvm::dyn_cast<mlir::OpResult>(publication);
  auto sync =
      result ? llvm::dyn_cast<dataflow::SyncOp>(result.getOwner()) : nullptr;
  if (!sync || sync.getInputs().size() != 2 || sync.getOutputs().size() != 2 ||
      result.getResultNumber() > 1)
    return std::nullopt;
  auto activeResult = llvm::dyn_cast<mlir::OpResult>(sync.getInputs()[0]);
  auto activityGate =
      activeResult ? llvm::dyn_cast<dataflow::DemuxOp>(activeResult.getOwner())
                   : nullptr;
  if (!activityGate || activityGate.getOutputs().size() != 2 ||
      activeResult.getResultNumber() != 1)
    return std::nullopt;
  auto activityEvent = describeSingleStreamActivityEvent(activityGate.getSel());
  if (!activityEvent || *activityEvent != activityGate.getInput())
    return std::nullopt;
  return activeResult;
}

std::optional<SelectiveRouterLeafDescription>
describeSelectiveRouterLeaf(mlir::Value value) {
  llvm::SmallVector<StaticSelectiveRouterLeg, 8> reversePath;
  mlir::Value input = value;
  while (auto result = llvm::dyn_cast<mlir::OpResult>(input)) {
    auto payload = llvm::dyn_cast<dataflow::DemuxOp>(result.getOwner());
    if (!payload)
      break;
    auto leg =
        describeStaticSelectiveRouterLeg(payload, result.getResultNumber());
    if (!leg) {
      if (reversePath.empty())
        return std::nullopt;
      break;
    }
    reversePath.push_back(*leg);
    input = payload.getInput();
  }
  if (reversePath.empty())
    return std::nullopt;
  std::reverse(reversePath.begin(), reversePath.end());
  const StaticSelectiveRouterLeg &root = reversePath.front();
  if (!isGraphStreamInput(input) && input != root.event &&
      input != root.ordinal)
    return std::nullopt;
  mlir::Value synchronization =
      input == root.event || input == root.ordinal ? root.event : mlir::Value{};

  llvm::SmallVector<RouterLeaf, 8> payloadLeaves;
  if (!collectPayloadRouterLeaves(input, root.ordinal, root.event,
                                  payloadLeaves, /*requireNode=*/true))
    return std::nullopt;
  auto leaf = llvm::find_if(payloadLeaves, [&](const RouterLeaf &entry) {
    return entry.value == value;
  });
  if (leaf == payloadLeaves.end())
    return std::nullopt;
  const unsigned lane =
      static_cast<unsigned>(std::distance(payloadLeaves.begin(), leaf));

  mlir::Value route = root.ordinal;
  mlir::Value event = root.event;
  mlir::Value activity;
  auto routeResult = llvm::dyn_cast<mlir::OpResult>(route);
  auto eventResult = llvm::dyn_cast<mlir::OpResult>(event);
  auto routeFilter =
      routeResult ? llvm::dyn_cast<dataflow::DemuxOp>(routeResult.getOwner())
                  : nullptr;
  auto eventFilter =
      eventResult ? llvm::dyn_cast<dataflow::DemuxOp>(eventResult.getOwner())
                  : nullptr;
  if (routeFilter) {
    if (!eventFilter || routeFilter.getOutputs().size() != 2 ||
        eventFilter.getOutputs().size() != 2 ||
        routeResult.getResultNumber() != 1 ||
        eventResult.getResultNumber() != 1 ||
        routeFilter.getSel() != eventFilter.getSel())
      return std::nullopt;
    activity = routeFilter.getSel();
    route = routeFilter.getInput();
    event = eventFilter.getInput();
  }

  dataflow::StreamOp stream = findRootStream(route);
  auto activation = stream ? getStreamActivation(stream) : std::nullopt;
  if (!stream || !activation ||
      unwrapPhaseProjection(event, stream) != *activation)
    return std::nullopt;

  mlir::Value leafActivity;
  mlir::Value activityEvent;
  if (activity) {
    llvm::SmallVector<RouterLeaf, 8> activityLeaves;
    if (!collectActivityRouterLeaves(activity, route, event, activityLeaves) ||
        activityLeaves.size() != payloadLeaves.size())
      return std::nullopt;
    leafActivity = activityLeaves[lane].value;
    activityEvent = activityLeaves[lane].event;
  }
  return SelectiveRouterLeafDescription{
      *activation, route, leafActivity, activityEvent, synchronization, lane};
}

bool controlMatchesLane(mlir::Value control,
                        const SelectorDescription &description, unsigned lane) {
  auto result = llvm::dyn_cast<mlir::OpResult>(control);
  auto demux =
      result ? llvm::dyn_cast<dataflow::DemuxOp>(result.getOwner()) : nullptr;
  if (!demux || demux.getSel() != description.routeSelector ||
      demux.getOutputs().size() != description.arity ||
      result.getResultNumber() != lane)
    return false;
  return isBodyAligned(demux.getInput(), description.stream);
}

mlir::Value getLaneActivity(const SelectorDescription &description,
                            unsigned lane) {
  if (!description.activity)
    return {};
  auto activities = description.activity.getDefiningOp<dataflow::MuxOp>();
  if (!activities || activities.getInputs().size() != description.arity ||
      lane >= description.arity)
    return {};
  mlir::Value activityRoute =
      unwrapSelectorOrdinal(activities.getSel(), description.arity);
  if (activityRoute != description.route)
    return {};
  return activities.getInputs()[lane];
}

bool activityIsTrue(mlir::Value value, const SelectorDescription &description,
                    unsigned lane, mlir::Value branchSelector,
                    std::optional<unsigned> branchLane,
                    llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return false;
  if (auto known = getKnownBool(value)) {
    auto constant = value.getDefiningOp<dataflow::ConstantOp>();
    if (!constant)
      return *known;
    return *known && controlMatchesLane(constant.getCtrl(), description, lane);
  }
  auto mux = value.getDefiningOp<dataflow::MuxOp>();
  if (!mux)
    return false;
  if (auto selected = getKnownBool(mux.getSel())) {
    unsigned selectedLane = *selected ? 1 : 0;
    if (selectedLane >= mux.getInputs().size())
      return false;
    return activityIsTrue(mux.getInputs()[selectedLane], description, lane,
                          branchSelector, branchLane, visited);
  }
  if (branchLane &&
      haveSynchronizedValueCorrespondence(mux.getSel(), branchSelector)) {
    if (*branchLane >= mux.getInputs().size())
      return false;
    return activityIsTrue(mux.getInputs()[*branchLane], description, lane,
                          branchSelector, branchLane, visited);
  }
  return llvm::all_of(mux.getInputs(), [&](mlir::Value input) {
    llvm::DenseSet<mlir::Value> branchVisited = visited;
    return activityIsTrue(input, description, lane, branchSelector, branchLane,
                          branchVisited);
  });
}

bool activityIsTrue(const SelectorDescription &description, unsigned lane,
                    mlir::Value branchSelector = {},
                    std::optional<unsigned> branchLane = std::nullopt) {
  if (!description.activity)
    return true;
  mlir::Value activity = getLaneActivity(description, lane);
  llvm::DenseSet<mlir::Value> visited;
  return activityIsTrue(activity, description, lane, branchSelector, branchLane,
                        visited);
}

std::optional<mlir::Value>
findSynchronization(mlir::Value value, mlir::Value branchSelector,
                    std::optional<unsigned> branchLane,
                    llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return std::nullopt;
  auto mux = value.getDefiningOp<dataflow::MuxOp>();
  if (!mux)
    return std::nullopt;
  if (branchLane &&
      haveSynchronizedValueCorrespondence(mux.getSel(), branchSelector)) {
    if (*branchLane >= mux.getInputs().size())
      return std::nullopt;
    auto nested = findSynchronization(mux.getInputs()[*branchLane],
                                      branchSelector, branchLane, visited);
    return nested ? nested : std::optional<mlir::Value>{branchSelector};
  }
  if (auto selected = getKnownBool(mux.getSel())) {
    unsigned selectedLane = *selected ? 1 : 0;
    if (selectedLane >= mux.getInputs().size())
      return std::nullopt;
    auto nested = findSynchronization(mux.getInputs()[selectedLane],
                                      branchSelector, branchLane, visited);
    return nested ? nested : std::optional<mlir::Value>{mux.getSel()};
  }
  return std::nullopt;
}

} // namespace

llvm::Expected<dataflow::semantics::MemoryAccessType>
dataflow::semantics::analyzeMemoryAccessType(mlir::MemRefType memoryType,
                                             mlir::Type dataType,
                                             mlir::Type addressType,
                                             mlir::Operation *scope,
                                             mlir::Type maskType) {
  mlir::Type elementType = memoryType.getElementType();
  MemoryAccessType access;
  access.elementType = elementType;
  access.dataType = dataType;
  access.addressType = addressType;
  // Data that exactly equals the memory element type is one element access,
  // including when that element is itself a vector. A vector-valued element is
  // still a semantic vector, so it passes the same positive fixed-rank
  // validator, but it contributes no access lane shape: the access keeps its
  // element geometry and one logical address.
  if (dataType == elementType) {
    if (llvm::isa<mlir::VectorType>(elementType))
      if (llvm::Expected<mlir::VectorType> element =
              analyzeFixedRankDataVector(elementType, VectorRank::AnyFixed);
          !element)
        return element.takeError();
    if (maskType)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "mask is only valid for a vector memory access");
  } else if (auto pointer =
                 llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(dataType)) {
    if (maskType)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "mask is not valid for a scalar pointer memory access");
    auto storage = llvm::dyn_cast<mlir::IntegerType>(elementType);
    if (!storage || storage.getWidth() == 0)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "pointer data requires a nonzero-width integer storage element");
    auto layout = loom::resolvePointerLayout(scope, pointer.getAddressSpace());
    if (!layout)
      return layout.takeError();
    if (storage.getWidth() != layout->representationBits)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "pointer data requires an integer storage element of exactly "
          "P(AS) bits");
    access.dataPointerLayout = *layout;
  } else if (llvm::isa<mlir::VectorType>(dataType)) {
    auto vector = analyzeFixedRankDataVector(dataType, VectorRank::AnyFixed);
    if (!vector)
      return vector.takeError();
    if (vector->getElementType() != elementType)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "data vector element type %s must match memory element type %s",
          typeToString(vector->getElementType()).c_str(),
          typeToString(elementType).c_str());
    if (maskType)
      if (llvm::Error error = validateVectorMaskType(*vector, maskType))
        return std::move(error);
    access.vectorType = *vector;
  } else {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to verify that 'data' type matches memref element type");
  }

  if (llvm::isa<mlir::IndexType>(addressType))
    return access;

  if (auto pointer = llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(addressType)) {
    auto layout = loom::resolvePointerLayout(scope, pointer.getAddressSpace());
    if (!layout)
      return layout.takeError();
    access.addressForm = MemoryAddressForm::PointerAddressed;
    access.pointerLayout = *layout;
    return access;
  }

  auto addressVector = llvm::dyn_cast<mlir::VectorType>(addressType);
  if (!addressVector)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "operand #1 must be index, LLVM pointer, or a fixed-size vector of "
        "one of those types");
  if (addressVector.isScalable())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "address vector must be a fixed-size "
                                   "vector");
  const bool indexAddress =
      llvm::isa<mlir::IndexType>(addressVector.getElementType());
  auto pointerAddress = llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(
      addressVector.getElementType());
  if (!indexAddress && !pointerAddress)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "address vector element type must be "
                                   "'index' or an LLVM pointer");
  if (!access.isVector())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "vector address requires a fixed-size vector data type");
  if (addressVector.getShape() != access.vectorType.getShape())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "address vector shape '%s' must match data vector shape '%s'",
        typeToString(addressVector).c_str(),
        typeToString(access.vectorType).c_str());
  access.addressVectorType = addressVector;
  if (pointerAddress) {
    auto layout =
        loom::resolvePointerLayout(scope, pointerAddress.getAddressSpace());
    if (!layout)
      return layout.takeError();
    access.addressForm = MemoryAddressForm::PointerAddressed;
    access.pointerLayout = *layout;
  }
  return access;
}

llvm::Expected<dataflow::semantics::StreamTransition>
dataflow::semantics::evaluateStreamTransition(
    const StreamSemanticState &state, const StreamSemanticConfig &config,
    std::optional<StreamActivation> activation) {
  StreamTransition transition;
  transition.nextState = state;

  // A stream in Idle fires only once its full activation is present; a running
  // stream advances from its recorded state and consumes no operand. Both cases
  // of a state share these consumed heads, so gate operand readiness on the
  // state's close case before the continuation predicate selects the case.
  if (state.mode == StreamMode::Idle && !activation) {
    transition.firing = makeSemanticFiringDecision(
        streamCaseDescriptor(StreamCase::StartClose).consumedInputs,
        SemanticInputMask{0});
    return transition;
  }

  const StreamSemanticState active =
      state.mode == StreamMode::Idle
          ? StreamSemanticState{StreamMode::Running, activation->init,
                                activation->limit, activation->step}
          : state;

  if (llvm::Error width = requireStreamBitWidth(config.bitWidth))
    return std::move(width);
  if (active.current.getBitWidth() != config.bitWidth ||
      active.limit.getBitWidth() != config.bitWidth ||
      active.step.getBitWidth() != config.bitWidth)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "dataflow.stream operand bit width does not match declared width %u",
        config.bitWidth);

  const bool cont =
      evaluateStreamPredicate(active.current, active.limit, config.predicate);
  const StreamCaseDescriptor descriptor = streamCaseDescriptor(
      state.mode == StreamMode::Idle
          ? (cont ? StreamCase::StartTrue : StreamCase::StartClose)
          : (cont ? StreamCase::ContinueTrue : StreamCase::ContinueClose));

  transition.firing = makeSemanticFiringDecision(descriptor.consumedInputs,
                                                 descriptor.consumedInputs);
  transition.emitPhase = descriptor.emitPhase;
  transition.phase = descriptor.phase;
  // The descriptor names the IV output source; the evaluator only resolves the
  // named source to its dynamic recurrence payload, as the invariant evaluator
  // resolves its own latched and init sources.
  if (descriptor.ivSource == StreamOutputSource::Current) {
    transition.emitIv = true;
    transition.iv = active.current;
  }
  if (descriptor.nextMode == StreamMode::Running) {
    auto next =
        evaluateStreamStep(active.current, active.step, config.stepKind);
    if (!next)
      return next.takeError();
    transition.nextState = StreamSemanticState{StreamMode::Running, *next,
                                               active.limit, active.step};
  } else {
    transition.nextState = StreamSemanticState{};
  }
  return transition;
}

dataflow::semantics::CarryTransition
dataflow::semantics::evaluateCarryTransition(CarrySemanticState state,
                                             std::optional<bool> phase,
                                             bool initAvailable,
                                             bool nextAvailable) {
  CarryTransition transition;
  transition.nextState = state;
  if (state == CarrySemanticState::Initial) {
    const CarryCaseDescriptor descriptor =
        carryCaseDescriptor(selectCarryCase(state, false));
    transition.firing = makeSemanticFiringDecision(
        descriptor.consumedInputs,
        initAvailable ? descriptor.consumedInputs : SemanticInputMask{0});
    if (transition.firing.ready) {
      transition.forwardedInput = descriptor.forwardedInput;
      transition.nextState = descriptor.nextState;
    }
    return transition;
  }

  // The running phase head selects the next or close case; both consume it, so
  // block on that shared head until it arrives.
  if (!phase) {
    transition.firing = makeSemanticFiringDecision(
        carryCaseDescriptor(selectCarryCase(state, false)).consumedInputs,
        SemanticInputMask{0});
    return transition;
  }
  const CarryCaseDescriptor descriptor =
      carryCaseDescriptor(selectCarryCase(state, *phase));
  const SemanticInputMask available =
      semanticInput(CarryInput::Phase) |
      (nextAvailable ? semanticInput(CarryInput::Next) : SemanticInputMask{0});
  transition.firing =
      makeSemanticFiringDecision(descriptor.consumedInputs, available);
  if (!transition.firing.ready)
    return transition;
  transition.forwardedInput = descriptor.forwardedInput;
  transition.nextState = descriptor.nextState;
  return transition;
}

dataflow::semantics::InvariantTransition
dataflow::semantics::evaluateInvariantTransition(InvariantSemanticState state,
                                                 std::optional<bool> phase,
                                                 bool initAvailable) {
  InvariantTransition transition;
  transition.nextState = state;
  if (state == InvariantSemanticState::Initial) {
    const InvariantCaseDescriptor descriptor =
        invariantCaseDescriptor(selectInvariantCase(state, false));
    transition.firing = makeSemanticFiringDecision(
        descriptor.consumedInputs,
        initAvailable ? descriptor.consumedInputs : SemanticInputMask{0});
    if (transition.firing.ready) {
      transition.output = descriptor.output;
      transition.latchInput = descriptor.latchInput;
      transition.clearLatch = descriptor.clearLatch;
      transition.nextState = descriptor.nextState;
    }
    return transition;
  }

  // The running phase head selects replay or close; both consume only it, so
  // block on that shared head until it arrives.
  if (!phase) {
    transition.firing = makeSemanticFiringDecision(
        invariantCaseDescriptor(selectInvariantCase(state, false))
            .consumedInputs,
        SemanticInputMask{0});
    return transition;
  }
  const InvariantCaseDescriptor descriptor =
      invariantCaseDescriptor(selectInvariantCase(state, *phase));
  transition.firing = makeSemanticFiringDecision(descriptor.consumedInputs,
                                                 descriptor.consumedInputs);
  transition.output = descriptor.output;
  transition.latchInput = descriptor.latchInput;
  transition.clearLatch = descriptor.clearLatch;
  transition.nextState = descriptor.nextState;
  return transition;
}

dataflow::semantics::GateTransition dataflow::semantics::evaluateGateTransition(
    GateSemanticState state, std::optional<bool> phase, bool valueAvailable) {
  GateTransition transition;
  transition.nextState = state;

  // Every gate case consumes the condition and value heads together; require
  // both, as the state's control-only case states, before the condition selects
  // the case.
  const SemanticInputMask required =
      gateCaseDescriptor(selectGateCase(state, false)).consumedInputs;
  SemanticInputMask available =
      valueAvailable ? semanticInput(GateInput::Value) : SemanticInputMask{0};
  if (phase)
    available |= semanticInput(GateInput::Phase);
  transition.firing = makeSemanticFiringDecision(required, available);
  if (!transition.firing.ready)
    return transition;

  const GateCaseDescriptor descriptor =
      gateCaseDescriptor(selectGateCase(state, *phase));
  transition.emitPhase = descriptor.emitPhase;
  transition.phase = descriptor.phase;
  transition.forwardedInput = descriptor.forwardedInput;
  transition.nextState = descriptor.nextState;
  return transition;
}

dataflow::semantics::ParallelizeTransition
dataflow::semantics::evaluateParallelizeTransition(
    const ParallelizeSemanticState &state, std::uint64_t vectorLength,
    std::optional<bool> scalarPhase, bool dataAvailable) {
  assert(vectorLength != 0 && state.pendingItems < vectorLength);
  ParallelizeTransition transition;
  transition.nextState = state;

  SemanticInputMask required = semanticInput(ParallelizeInput::Phase);
  SemanticInputMask available = scalarPhase ? required : SemanticInputMask{0};
  if (scalarPhase && *scalarPhase) {
    required |= semanticInput(ParallelizeInput::Data);
    if (dataAvailable)
      available |= semanticInput(ParallelizeInput::Data);
  }
  transition.firing = makeSemanticFiringDecision(required, available);
  if (!transition.firing.ready)
    return transition;

  if (!*scalarPhase) {
    if (state.pendingItems != 0) {
      transition.emitGroup = true;
      transition.activeItems = state.pendingItems;
      transition.emitTruePhase = true;
    }
    transition.emitFalsePhase = true;
    transition.nextState = {};
    return transition;
  }

  transition.nextState.pendingItems = state.pendingItems + 1;
  if (transition.nextState.pendingItems == vectorLength) {
    transition.emitGroup = true;
    transition.activeItems = vectorLength;
    transition.emitTruePhase = true;
    transition.nextState = {};
  }
  return transition;
}

dataflow::semantics::SerializeTransition
dataflow::semantics::evaluateSerializeTransition(std::optional<bool> groupPhase,
                                                 bool vectorAvailable,
                                                 bool maskAvailable) {
  SerializeTransition transition;
  SemanticInputMask required = semanticInput(SerializeInput::Phase);
  SemanticInputMask available = groupPhase ? required : SemanticInputMask{0};
  if (groupPhase && *groupPhase) {
    required |= semanticInput(SerializeInput::Vector) |
                semanticInput(SerializeInput::Mask);
    if (vectorAvailable)
      available |= semanticInput(SerializeInput::Vector);
    if (maskAvailable)
      available |= semanticInput(SerializeInput::Mask);
  }
  transition.firing = makeSemanticFiringDecision(required, available);
  if (!transition.firing.ready)
    return transition;
  transition.emitActiveItems = *groupPhase;
  transition.emitFalsePhase = !*groupPhase;
  return transition;
}

std::optional<mlir::Value>
dataflow::semantics::getStreamActivation(dataflow::StreamOp stream) {
  if (!stream)
    return std::nullopt;
  mlir::Value init = getActivationSource(stream.getInit());
  mlir::Value limit = getActivationSource(stream.getLimit());
  mlir::Value step = getActivationSource(stream.getStep());
  if (!init || init != limit || init != step)
    return std::nullopt;
  return init;
}

std::optional<mlir::Value>
dataflow::semantics::getCloseActivation(mlir::Value value) {
  auto result = llvm::dyn_cast<mlir::OpResult>(value);
  auto close =
      result ? llvm::dyn_cast<dataflow::DemuxOp>(result.getOwner()) : nullptr;
  if (!close || close.getOutputs().size() != 2 || result.getResultNumber() != 0)
    return std::nullopt;
  auto stream = close.getSel().getDefiningOp<dataflow::StreamOp>();
  if (!stream || close.getSel() != stream.getPhase())
    return std::nullopt;
  auto invariant = close.getInput().getDefiningOp<dataflow::InvariantOp>();
  if (!invariant || invariant.getCond() != stream.getPhase())
    return std::nullopt;
  auto activation = getStreamActivation(stream);
  if (!activation || invariant.getInit() != *activation)
    return std::nullopt;
  return activation;
}

std::optional<bool> dataflow::semantics::gateClosesWhenSelected(
    dataflow::GateOp gate, mlir::Value selector, unsigned lane) {
  if (!gate || lane > 1)
    return std::nullopt;
  auto stream = gate.getBeforeCond().getDefiningOp<dataflow::StreamOp>();
  auto predicate = selector.getDefiningOp<mlir::arith::CmpIOp>();
  if (!stream || gate.getBeforeCond() != stream.getPhase() || !predicate ||
      predicate.getPredicate() != stream.getPredicate() ||
      predicate.getLhs() != stream.getInit() ||
      predicate.getRhs() != stream.getLimit())
    return std::nullopt;
  return lane == 1;
}

bool dataflow::semantics::gateAlwaysCloses(dataflow::GateOp gate) {
  if (!gate)
    return false;
  auto stream = gate.getBeforeCond().getDefiningOp<dataflow::StreamOp>();
  if (!stream || gate.getBeforeCond() != stream.getPhase())
    return false;

  auto getConstant = [](mlir::Value value) -> std::optional<mlir::APInt> {
    if (auto constant = value.getDefiningOp<dataflow::ConstantOp>())
      if (auto integer =
              llvm::dyn_cast<mlir::IntegerAttr>(constant.getConstValue()))
        return integer.getValue();
    mlir::APInt integer;
    if (mlir::matchPattern(value, mlir::m_ConstantInt(&integer)))
      return integer;
    return std::nullopt;
  };
  auto init = getConstant(stream.getInit());
  auto limit = getConstant(stream.getLimit());
  if (!init || !limit || init->getBitWidth() != limit->getBitWidth())
    return false;
  return evaluateStreamPredicate(*init, *limit, stream.getPredicate());
}

std::optional<dataflow::GateOp>
dataflow::semantics::getGateCloseProjection(mlir::Value value) {
  auto result = llvm::dyn_cast<mlir::OpResult>(value);
  auto demux =
      result ? llvm::dyn_cast<dataflow::DemuxOp>(result.getOwner()) : nullptr;
  if (!demux || demux.getOutputs().size() != 2 || result.getResultNumber() != 0)
    return std::nullopt;
  auto gate = demux.getSel().getDefiningOp<dataflow::GateOp>();
  if (!gate || demux.getSel() != gate.getAfterCond() ||
      demux.getInput() != gate.getAfterValue())
    return std::nullopt;
  return gate;
}

std::optional<mlir::Value>
dataflow::semantics::getSelectorActivation(mlir::Value selector,
                                           unsigned arity) {
  auto description = describeSelector(selector, arity);
  if (!description)
    return std::nullopt;
  return getStreamActivation(description->stream);
}

std::optional<mlir::Value>
dataflow::semantics::getSelectiveRouterLeafActivation(
    mlir::Value value, mlir::Value branchSelector,
    std::optional<unsigned> branchLane) {
  auto description = describeSelectiveRouterLeaf(value);
  if (!description ||
      !routeVisitsLaneOnce(description->route, description->lane) ||
      !activityIsTrueForEvent(description->activity, description->activityEvent,
                              branchSelector, branchLane))
    return std::nullopt;
  return description->activation;
}

std::optional<mlir::Value>
dataflow::semantics::getSelectiveRouterLeafSynchronization(
    mlir::Value value, mlir::Value branchSelector,
    std::optional<unsigned> branchLane) {
  auto description = describeSelectiveRouterLeaf(value);
  if (!description ||
      !activityIsTrueForEvent(description->activity, description->activityEvent,
                              branchSelector, branchLane))
    return std::nullopt;
  if (description->synchronization)
    return description->synchronization;
  if (!description->activity)
    return std::nullopt;
  llvm::DenseSet<mlir::Value> visited;
  return findSynchronization(description->activity, branchSelector, branchLane,
                             visited);
}

std::optional<mlir::Value>
dataflow::semantics::getStreamActivityEvent(mlir::Value value) {
  if (auto event = describeSelectiveRouterActivityEvent(value))
    return event;
  return describeSingleStreamActivityEvent(value);
}

bool dataflow::semantics::haveEquivalentSynchronizedSelectionCorrespondence(
    mlir::Value lhs, mlir::Value rhs) {
  return haveSynchronizedValueCorrespondence(lhs, rhs);
}

std::optional<mlir::Value>
dataflow::semantics::getStreamPublicationEvent(mlir::Value value) {
  if (auto event = describeSelectiveRouterPublicationEvent(value))
    return event;
  return describeSingleStreamPublicationEvent(value);
}

bool dataflow::semantics::selectorSelectsLaneOncePerActivation(
    mlir::Value selector, unsigned arity, unsigned lane) {
  if (lane >= arity)
    return false;
  auto description = describeSelector(selector, arity);
  bool route =
      description && (routeVisitsLaneOnce(description->route, lane) ||
                      (arity == 2 && selectorPredicateVisitsLaneOnce(
                                         description->routeSelector, lane)));
  bool activity = description && activityIsTrue(*description, lane);
  return description && route && activity;
}

std::optional<mlir::Value> dataflow::semantics::getSelectorLaneEventActivation(
    mlir::Value selector, unsigned arity, unsigned lane, mlir::Value event) {
  if (!selectorSelectsLaneOncePerActivation(selector, arity, lane))
    return std::nullopt;
  auto description = describeSelector(selector, arity);
  auto activation =
      description ? getStreamActivation(description->stream) : std::nullopt;
  mlir::Value projection =
      description ? unwrapPhaseProjection(event, description->stream)
                  : mlir::Value{};
  if (!activation || projection != *activation)
    return std::nullopt;
  return activation;
}

bool dataflow::semantics::selectorSelectsEveryLaneOncePerActivation(
    mlir::Value selector, unsigned arity) {
  return arity > 1 &&
         llvm::all_of(llvm::seq<unsigned>(0, arity), [&](unsigned lane) {
           return selectorSelectsLaneOncePerActivation(selector, arity, lane);
         });
}

bool dataflow::semantics::selectorLaneActiveWhenSelected(
    mlir::Value scheduleSelector, unsigned arity, unsigned scheduleLane,
    mlir::Value branchSelector, unsigned branchLane) {
  if (scheduleLane >= arity)
    return false;
  auto description = describeSelector(scheduleSelector, arity);
  bool active = description && activityIsTrue(*description, scheduleLane,
                                              branchSelector, branchLane);
  return active;
}

std::optional<mlir::Value> dataflow::semantics::getSelectorLaneSynchronization(
    mlir::Value scheduleSelector, unsigned arity, unsigned scheduleLane,
    mlir::Value branchSelector, std::optional<unsigned> branchLane) {
  if (scheduleLane >= arity)
    return std::nullopt;
  auto description = describeSelector(scheduleSelector, arity);
  if (!description || !description->activity)
    return std::nullopt;
  mlir::Value activity = getLaneActivity(*description, scheduleLane);
  llvm::DenseSet<mlir::Value> visited;
  return findSynchronization(activity, branchSelector, branchLane, visited);
}
