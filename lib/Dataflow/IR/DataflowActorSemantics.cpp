#include "Dataflow/IR/DataflowActorSemantics.h"

#include "Dataflow/IR/DataflowInterfaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <optional>
#include <string>
#include <system_error>

namespace {

using dataflow::semantics::getStreamActivation;

bool isSupportedVectorElementType(mlir::Type type) {
  if (auto integer = llvm::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth() != 0;
  return llvm::isa<mlir::FloatType>(type);
}

std::string typeToString(mlir::Type type) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  type.print(stream);
  return storage;
}

llvm::Error requireStreamBitWidth(unsigned bitWidth) {
  if (bitWidth >= 1 && bitWidth <= 64)
    return llvm::Error::success();
  return llvm::createStringError(
      std::errc::invalid_argument,
      "dataflow.stream integer bit width must be in [1, 64], got %u", bitWidth);
}

bool evaluateStreamPredicate(const llvm::APInt &current,
                             const llvm::APInt &limit,
                             mlir::arith::CmpIPredicate predicate) {
  static_assert(mlir::arith::getMaxEnumValForCmpIPredicate() + 1 == 10,
                "audit dataflow.stream predicate semantics");
  return mlir::arith::applyCmpPredicate(predicate, current, limit);
}

llvm::APInt streamBits(unsigned bitWidth, std::int64_t value) {
  return llvm::APInt(bitWidth, static_cast<std::uint64_t>(value),
                     /*isSigned=*/false, /*implicitTrunc=*/true);
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
    std::int64_t amount = step.getSExtValue();
    if (amount < 0 || static_cast<std::uint64_t>(amount) >= step.getBitWidth())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "dataflow.stream shift amount must be in [0, %u), got %lld",
          step.getBitWidth(), static_cast<long long>(amount));
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
  if (auto result = llvm::dyn_cast<mlir::OpResult>(value))
    if (auto demux = llvm::dyn_cast<dataflow::DemuxOp>(result.getOwner()))
      return getKnownBool(demux.getInput(), visited);
  return std::nullopt;
}

std::optional<bool> getKnownBool(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  return getKnownBool(value, visited);
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
  if (!dataflow::isCanonicalDataflowActor(def) &&
      !llvm::isa<mlir::UnrealizedConversionCastOp>(def))
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
  if (branchLane && mux.getSel() == branchSelector) {
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
  if (branchLane && mux.getSel() == branchSelector) {
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

llvm::Expected<mlir::VectorType>
dataflow::semantics::analyzeFixedRankOneDataVector(mlir::Type type) {
  auto vector = llvm::dyn_cast<mlir::VectorType>(type);
  if (!vector || vector.getRank() != 1 || vector.isScalable())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "data vector must be a fixed-size rank-1 vector");
  if (!isSupportedVectorElementType(vector.getElementType()))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "data vector element type must be a nonzero-width integer or "
        "floating-point type");
  return vector;
}

llvm::Error
dataflow::semantics::validateVectorMaskType(mlir::VectorType dataVector,
                                            mlir::Type maskType) {
  auto mask = llvm::dyn_cast<mlir::VectorType>(maskType);
  if (!mask || mask.getRank() != 1 || mask.isScalable())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mask vector must be a fixed-size rank-1 vector");
  if (!mask.getElementType().isInteger(1))
    return llvm::createStringError(std::errc::invalid_argument,
                                   "mask vector element type must be 'i1'");
  if (mask.getShape() != dataVector.getShape())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "mask vector shape '%s' must match data vector shape '%s'",
        typeToString(mask).c_str(), typeToString(dataVector).c_str());
  return llvm::Error::success();
}

llvm::Expected<dataflow::semantics::MemoryAccessType>
dataflow::semantics::analyzeMemoryAccessType(mlir::MemRefType memoryType,
                                             mlir::Type dataType,
                                             mlir::Type addressType,
                                             mlir::Type maskType) {
  mlir::Type elementType = memoryType.getElementType();
  MemoryAccessType access;
  access.elementType = elementType;
  if (dataType == elementType) {
    if (maskType)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "mask is only valid for a vector memory access");
  } else if (llvm::isa<mlir::VectorType>(dataType)) {
    auto vector = analyzeFixedRankOneDataVector(dataType);
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

  auto addressVector = llvm::dyn_cast<mlir::VectorType>(addressType);
  if (!addressVector)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "operand #1 must be index or a fixed-size rank-1 vector of index");
  if (addressVector.getRank() != 1 || addressVector.isScalable())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "address vector must be a fixed-size rank-1 vector");
  if (!llvm::isa<mlir::IndexType>(addressVector.getElementType()))
    return llvm::createStringError(std::errc::invalid_argument,
                                   "address vector element type must be "
                                   "'index'");
  if (!access.isVector())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "vector address requires a fixed-size rank-1 vector data type");
  if (addressVector.getShape() != access.vectorType.getShape())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "address vector shape '%s' must match data vector shape '%s'",
        typeToString(addressVector).c_str(),
        typeToString(access.vectorType).c_str());
  access.addressVectorType = addressVector;
  return access;
}

llvm::Expected<dataflow::semantics::StreamTransition>
dataflow::semantics::evaluateStreamTransition(
    const StreamSemanticState &state, const StreamSemanticConfig &config,
    std::optional<StreamActivation> activation) {
  StreamTransition transition;
  transition.nextState = state;

  StreamSemanticState active = state;
  if (state.mode == StreamMode::Idle) {
    const SemanticInputMask required = semanticInput(StreamInput::Init) |
                                       semanticInput(StreamInput::Limit) |
                                       semanticInput(StreamInput::Step);
    transition.firing = makeSemanticFiringDecision(
        required, activation ? required : SemanticInputMask{0});
    if (!transition.firing.ready)
      return transition;
    active = StreamSemanticState{StreamMode::Running, activation->init,
                                 activation->limit, activation->step};
  } else {
    transition.firing = makeSemanticFiringDecision(0, 0);
  }

  if (llvm::Error width = requireStreamBitWidth(config.bitWidth))
    return std::move(width);
  llvm::APInt current = streamBits(config.bitWidth, active.current);
  llvm::APInt limit = streamBits(config.bitWidth, active.limit);
  bool cont = evaluateStreamPredicate(current, limit, config.predicate);
  transition.emitPhase = true;
  transition.phase = cont;
  if (!cont) {
    transition.nextState = StreamSemanticState{};
    return transition;
  }

  llvm::APInt step = streamBits(config.bitWidth, active.step);
  auto next = evaluateStreamStep(current, step, config.stepKind);
  if (!next)
    return next.takeError();
  transition.emitIv = true;
  transition.iv = active.current;
  transition.nextState = StreamSemanticState{
      StreamMode::Running, next->getSExtValue(), active.limit, active.step};
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
    const SemanticInputMask required = semanticInput(CarryInput::Init);
    transition.firing = makeSemanticFiringDecision(
        required, initAvailable ? required : SemanticInputMask{0});
    if (transition.firing.ready) {
      transition.nextState = CarrySemanticState::Running;
      transition.forwardedInput = CarryInput::Init;
    }
    return transition;
  }

  SemanticInputMask required = semanticInput(CarryInput::Phase);
  SemanticInputMask available = phase ? required : SemanticInputMask{0};
  if (phase && *phase) {
    required |= semanticInput(CarryInput::Next);
    if (nextAvailable)
      available |= semanticInput(CarryInput::Next);
  }
  transition.firing = makeSemanticFiringDecision(required, available);
  if (!transition.firing.ready)
    return transition;
  if (*phase)
    transition.forwardedInput = CarryInput::Next;
  else
    transition.nextState = CarrySemanticState::Initial;
  return transition;
}

dataflow::semantics::InvariantTransition
dataflow::semantics::evaluateInvariantTransition(InvariantSemanticState state,
                                                 std::optional<bool> phase,
                                                 bool initAvailable) {
  InvariantTransition transition;
  transition.nextState = state;
  if (state == InvariantSemanticState::Initial) {
    const SemanticInputMask required = semanticInput(InvariantInput::Init);
    transition.firing = makeSemanticFiringDecision(
        required, initAvailable ? required : SemanticInputMask{0});
    if (transition.firing.ready) {
      transition.nextState = InvariantSemanticState::Running;
      transition.output = InvariantOutputSource::InitInput;
      transition.latchInput = InvariantInput::Init;
    }
    return transition;
  }

  const SemanticInputMask required = semanticInput(InvariantInput::Phase);
  transition.firing = makeSemanticFiringDecision(
      required, phase ? required : SemanticInputMask{0});
  if (!transition.firing.ready)
    return transition;
  if (*phase) {
    transition.output = InvariantOutputSource::Latched;
  } else {
    transition.nextState = InvariantSemanticState::Initial;
    transition.clearLatch = true;
  }
  return transition;
}

dataflow::semantics::GateTransition dataflow::semantics::evaluateGateTransition(
    GateSemanticState state, std::optional<bool> phase, bool valueAvailable) {
  GateTransition transition;
  transition.nextState = state;
  const SemanticInputMask required =
      semanticInput(GateInput::Phase) | semanticInput(GateInput::Value);
  SemanticInputMask available =
      valueAvailable ? semanticInput(GateInput::Value) : SemanticInputMask{0};
  if (phase)
    available |= semanticInput(GateInput::Phase);
  transition.firing = makeSemanticFiringDecision(required, available);
  if (!transition.firing.ready)
    return transition;

  if (state == GateSemanticState::Closed) {
    if (*phase) {
      transition.nextState = GateSemanticState::Open;
      transition.forwardedInput = GateInput::Value;
    }
    return transition;
  }

  transition.emitPhase = true;
  transition.phase = *phase;
  if (*phase) {
    transition.forwardedInput = GateInput::Value;
  } else {
    transition.nextState = GateSemanticState::Closed;
  }
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

bool dataflow::semantics::isStatelessOneTokenVectorBoundary(
    mlir::Operation *op) {
  return op && llvm::isa<dataflow::PackOp, dataflow::UnpackOp>(op);
}

std::optional<mlir::Value>
dataflow::semantics::getVectorBoundaryInputPhase(mlir::Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto parallelize = llvm::dyn_cast<dataflow::ParallelizeOp>(op))
    return parallelize.getScalarPhase();
  if (auto serialize = llvm::dyn_cast<dataflow::SerializeOp>(op))
    return serialize.getGroupPhase();
  return std::nullopt;
}

std::optional<mlir::Value>
dataflow::semantics::getVectorBoundaryOutputPhase(mlir::Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto parallelize = llvm::dyn_cast<dataflow::ParallelizeOp>(op))
    return parallelize.getGroupPhase();
  if (auto serialize = llvm::dyn_cast<dataflow::SerializeOp>(op))
    return serialize.getScalarPhase();
  return std::nullopt;
}

mlir::ValueRange dataflow::semantics::getVectorBoundaryTruePhaseInputPayloads(
    mlir::Operation *op) {
  if (!op || !llvm::isa<dataflow::ParallelizeOp, dataflow::SerializeOp>(op))
    return {};
  return op->getOperands().drop_back();
}

bool dataflow::semantics::isVectorBoundaryTruePhaseOutputPayload(
    mlir::Value value, mlir::Value phase) {
  mlir::Operation *def = value.getDefiningOp();
  if (auto parallelize = llvm::dyn_cast_or_null<dataflow::ParallelizeOp>(def))
    return phase == parallelize.getGroupPhase() &&
           (value == parallelize.getVector() || value == parallelize.getMask());
  if (auto serialize = llvm::dyn_cast_or_null<dataflow::SerializeOp>(def))
    return phase == serialize.getScalarPhase() && value == serialize.getData();
  return false;
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

bool dataflow::semantics::selectorSelectsLaneOncePerActivation(
    mlir::Value selector, unsigned arity, unsigned lane) {
  if (lane >= arity)
    return false;
  auto description = describeSelector(selector, arity);
  bool route = description && routeVisitsLaneOnce(description->route, lane);
  bool activity = description && activityIsTrue(*description, lane);
  return description && route && activity;
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
