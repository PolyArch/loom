#include "Frontend/Lowering/GraphParallelLowering.h"
#include "Frontend/Lowering/GraphMemoryAddressing.h"
#include "GraphRegionLowering.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace loom {
namespace lowering {

namespace {

std::optional<int64_t> getConstantIndex(::mlir::Value value) {
  ::mlir::APInt constant;
  if (!::mlir::matchPattern(value, ::mlir::m_ConstantInt(&constant)) ||
      !constant.isSignedIntN(64))
    return std::nullopt;
  return constant.getSExtValue();
}

std::optional<FixedParallelDomain>
getFixedParallelDomain(::mlir::scf::ForallOp forall) {
  auto lower = ::mlir::getConstantIntValues(forall.getMixedLowerBound());
  auto upper = ::mlir::getConstantIntValues(forall.getMixedUpperBound());
  auto step = ::mlir::getConstantIntValues(forall.getMixedStep());
  if (!lower || !upper || !step)
    return std::nullopt;
  return FixedParallelDomain{std::move(*lower), std::move(*upper),
                             std::move(*step)};
}

std::optional<FixedParallelDomain>
getFixedParallelDomain(::mlir::scf::ParallelOp parallel) {
  FixedParallelDomain domain;
  for (::mlir::Value value : parallel.getLowerBound()) {
    auto constant = getConstantIndex(value);
    if (!constant)
      return std::nullopt;
    domain.lower.push_back(*constant);
  }
  for (::mlir::Value value : parallel.getUpperBound()) {
    auto constant = getConstantIndex(value);
    if (!constant)
      return std::nullopt;
    domain.upper.push_back(*constant);
  }
  for (::mlir::Value value : parallel.getStep()) {
    auto constant = getConstantIndex(value);
    if (!constant)
      return std::nullopt;
    domain.step.push_back(*constant);
  }
  return domain;
}

bool forEachParallelPointUntil(
    const FixedParallelDomain &domain, unsigned dimension,
    ::llvm::SmallVectorImpl<int64_t> &point,
    ::llvm::function_ref<bool(::llvm::ArrayRef<int64_t>)> callback) {
  if (dimension == domain.lower.size())
    return callback(point);

  int64_t value = domain.lower[dimension];
  while (value < domain.upper[dimension]) {
    point.push_back(value);
    bool keepGoing =
        forEachParallelPointUntil(domain, dimension + 1, point, callback);
    point.pop_back();
    if (!keepGoing)
      return false;
    if (value > std::numeric_limits<int64_t>::max() - domain.step[dimension])
      break;
    value += domain.step[dimension];
  }
  return true;
}

bool hasSpatialCarrierAncestor(::mlir::Operation *op) {
  for (::mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent->getName().getStringRef() == "loom.spatial_region")
      return true;
  return false;
}

struct ParallelMemoryAccess {
  ::mlir::Operation *op;
  ::mlir::Value memory;
  ::llvm::SmallVector<::mlir::Value, 4> address;
  ::mlir::Type llvmAccessType;
  bool writes;
  bool atomic;
};

std::optional<ParallelMemoryAccess>
getLlvmMemoryAccess(::mlir::Operation *op, ::mlir::Value pointer,
                    ::mlir::Type accessType, bool writes, bool atomic) {
  return ParallelMemoryAccess{op, pointer, {}, accessType, writes, atomic};
}

// A canonical actor that graph-region lowering cannot lower or move has no
// per-lane completion event either. This queries the shared leaf
// classification instead of keeping its own notion of what is unsupported.
bool hasUnsupportedParallelCompletion(::mlir::Operation *op) {
  return ::dataflow::isCanonicalDataflowActor(op) &&
         ::loom::lowering::classifyGraphLoweringLeaf(op) ==
             ::loom::lowering::GraphLeafLowering::Unsupported;
}

std::optional<ParallelMemoryAccess>
getParallelMemoryAccess(::mlir::Operation *op) {
  auto atomic = [&]() {
    auto contract = ::dataflow::semantics::getMemoryActorContract(op);
    return contract && contract->atomic;
  };

  if (auto load = ::llvm::dyn_cast<::mlir::memref::LoadOp>(op))
    return ParallelMemoryAccess{
        op,
        load.getMemRef(),
        {load.getIndices().begin(), load.getIndices().end()},
        mlir::Type{},
        /*writes=*/false,
        /*atomic=*/false};
  if (auto store = ::llvm::dyn_cast<::mlir::memref::StoreOp>(op))
    return ParallelMemoryAccess{
        op,
        store.getMemRef(),
        {store.getIndices().begin(), store.getIndices().end()},
        mlir::Type{},
        /*writes=*/true,
        /*atomic=*/false};
  if (auto load = ::llvm::dyn_cast<::dataflow::LoadOp>(op))
    return ParallelMemoryAccess{op,
                                load.getMem(),
                                {load.getAddr()},
                                mlir::Type{},
                                /*writes=*/false,
                                atomic()};
  if (auto store = ::llvm::dyn_cast<::dataflow::StoreOp>(op))
    return ParallelMemoryAccess{op,
                                store.getMem(),
                                {store.getAddr()},
                                mlir::Type{},
                                /*writes=*/true,
                                atomic()};
  if (auto load = ::llvm::dyn_cast<::mlir::LLVM::LoadOp>(op))
    return getLlvmMemoryAccess(op, load.getAddr(), load.getType(),
                               /*writes=*/false,
                               load.getOrdering() !=
                                   ::mlir::LLVM::AtomicOrdering::not_atomic);
  if (auto store = ::llvm::dyn_cast<::mlir::LLVM::StoreOp>(op))
    return getLlvmMemoryAccess(op, store.getAddr(), store.getValue().getType(),
                               /*writes=*/true,
                               store.getOrdering() !=
                                   ::mlir::LLVM::AtomicOrdering::not_atomic);
  return std::nullopt;
}

bool hasUnmodeledWriteEffect(::mlir::Operation *op) {
  if (op->getNumRegions() != 0 ||
      ::llvm::isa<::mlir::memref::AllocOp, ::mlir::memref::AllocaOp>(op))
    return false;
  if (auto contract = ::dataflow::semantics::getMemoryActorContract(op);
      contract && contract->atomic)
    return false;
  auto effects = ::llvm::dyn_cast<::mlir::MemoryEffectOpInterface>(op);
  if (!effects)
    return false;
  ::llvm::SmallVector<::mlir::MemoryEffects::EffectInstance, 4> instances;
  effects.getEffects(instances);
  return ::llvm::any_of(instances, [](const auto &effect) {
    return ::llvm::isa<::mlir::MemoryEffects::Write,
                       ::mlir::MemoryEffects::Free>(effect.getEffect());
  });
}

std::optional<::mlir::Value> mapSpatialArgument(::mlir::Value value) {
  auto argument = ::llvm::dyn_cast<::mlir::BlockArgument>(value);
  if (!argument)
    return std::nullopt;
  ::mlir::Operation *parent = argument.getOwner()->getParentOp();
  if (!parent || parent->getName().getStringRef() != "loom.spatial_region" ||
      argument.getArgNumber() >= parent->getNumOperands())
    return std::nullopt;
  return parent->getOperand(argument.getArgNumber());
}

::mlir::Value mapSpatialArguments(::mlir::Value value) {
  ::llvm::DenseSet<::mlir::Value> visited;
  while (value && visited.insert(value).second) {
    auto mapped = mapSpatialArgument(value);
    if (!mapped)
      break;
    value = *mapped;
  }
  return value;
}

class MemoryRootResolver {
public:
  std::optional<::mlir::Value> resolve(::mlir::Value value) {
    ::llvm::DenseSet<::mlir::Value> visited;
    while (value && visited.insert(value).second) {
      if (auto mapped = mapSpatialArgument(value)) {
        value = *mapped;
        continue;
      }
      if (auto argument = ::llvm::dyn_cast<::mlir::BlockArgument>(value))
        return resolveBoundaryArgument(argument);

      ::mlir::Operation *def = value.getDefiningOp();
      if (!def)
        return std::nullopt;
      if (::llvm::isa<::mlir::memref::AllocOp, ::mlir::memref::AllocaOp>(def))
        return value;
      if (auto global = ::llvm::dyn_cast<::mlir::memref::GetGlobalOp>(def))
        return globalRoots.try_emplace(global.getNameAttr(), value)
            .first->second;
      if (auto view = ::llvm::dyn_cast<::mlir::ViewLikeOpInterface>(def)) {
        value = view.getViewSource();
        continue;
      }
      if (auto gep = ::llvm::dyn_cast<::mlir::LLVM::GEPOp>(def)) {
        value = gep.getBase();
        continue;
      }
      return std::nullopt;
    }
    return std::nullopt;
  }

private:
  ::llvm::DenseMap<::mlir::Operation *, ::mlir::Value> sharedBoundaryRoots;
  ::llvm::DenseMap<::mlir::FlatSymbolRefAttr, ::mlir::Value> globalRoots;

  std::optional<::mlir::Value>
  resolveBoundaryArgument(::mlir::BlockArgument argument) {
    ::mlir::Operation *owner = argument.getOwner()->getParentOp();
    unsigned inputIndex = argument.getArgNumber();
    ::mlir::DictionaryAttr attrs;

    if (auto graph = ::llvm::dyn_cast_or_null<::dataflow::GraphOp>(owner)) {
      if (inputIndex == 0 ||
          inputIndex > graph.getFunctionType().getNumInputs())
        return std::nullopt;
      --inputIndex;
      attrs =
          ::mlir::function_interface_impl::getArgAttrDict(graph, inputIndex);
    } else if (auto thread =
                   ::llvm::dyn_cast_or_null<::dataflow::ThreadOp>(owner)) {
      if (inputIndex >= thread.getFunctionType().getNumInputs())
        return std::nullopt;
      attrs =
          ::mlir::function_interface_impl::getArgAttrDict(thread, inputIndex);
    } else if (auto function =
                   ::llvm::dyn_cast_or_null<::mlir::LLVM::LLVMFuncOp>(owner)) {
      if (inputIndex >= function.getFunctionType().getNumParams())
        return std::nullopt;
      attrs =
          ::mlir::function_interface_impl::getArgAttrDict(function, inputIndex);
    } else {
      return std::nullopt;
    }

    if (attrs && attrs.contains("llvm.noalias"))
      return argument;
    return sharedBoundaryRoots.try_emplace(owner, argument).first->second;
  }
};

struct LinearExpression {
  enum class TransformKind {
    SignedDivide,
    SignedBitProjection,
    UnsignedBitProjection
  };
  struct Transform {
    TransformKind kind;
    ::llvm::APInt operand;

    friend bool operator==(const Transform &lhs, const Transform &rhs) {
      return lhs.kind == rhs.kind && lhs.operand == rhs.operand;
    }
  };

  LinearExpression(unsigned width, unsigned laneCount)
      : constant(width, 0), lanes(laneCount, ::llvm::APInt(width, 0)) {}

  ::llvm::APInt constant;
  ::llvm::SmallVector<::llvm::APInt, 4> lanes;
  ::llvm::DenseMap<::mlir::Value, ::llvm::APInt> symbols;
  ::llvm::DenseMap<::mlir::Value, ::llvm::APInt> descendantLanes;
  ::llvm::SmallVector<Transform, 2> transforms;
};

bool isParallelInductionVariable(::mlir::Operation *op, ::mlir::Value value) {
  if (auto parallel = ::llvm::dyn_cast<::mlir::scf::ParallelOp>(op))
    return ::llvm::is_contained(parallel.getInductionVars(), value);
  if (auto forall = ::llvm::dyn_cast<::mlir::scf::ForallOp>(op))
    return ::llvm::is_contained(forall.getInductionVars(), value);
  return false;
}

class LinearAddressBuilder {
public:
  LinearAddressBuilder(
      ::mlir::Operation *parallel, ::mlir::ValueRange inductionVars,
      unsigned width,
      const ::llvm::DenseSet<::mlir::Operation *> &provenParallelOps)
      : parallel(parallel),
        inductionVars(inductionVars.begin(), inductionVars.end()), width(width),
        provenParallelOps(provenParallelOps) {}

  std::optional<LinearExpression> build(::mlir::Value value) {
    if (auto found = cache.find(value); found != cache.end())
      return found->second;
    if (failed.contains(value) || !active.insert(value).second)
      return std::nullopt;

    auto finish = [&](std::optional<LinearExpression> expression) {
      active.erase(value);
      if (expression)
        cache.try_emplace(value, *expression);
      else
        failed.insert(value);
      return expression;
    };

    LinearExpression expression(width, inductionVars.size());
    for (auto [index, iv] : ::llvm::enumerate(inductionVars)) {
      if (value != iv)
        continue;
      expression.lanes[index] = ::llvm::APInt(width, 1);
      return finish(std::move(expression));
    }

    ::mlir::APInt constant;
    if (::mlir::matchPattern(value, ::mlir::m_ConstantInt(&constant))) {
      expression.constant = constant.sextOrTrunc(width);
      return finish(std::move(expression));
    }

    if (auto mapped = mapSpatialArgument(value))
      return finish(build(*mapped));

    if (auto argument = ::llvm::dyn_cast<::mlir::BlockArgument>(value)) {
      ::mlir::Operation *owner = argument.getOwner()->getParentOp();
      if (owner && parallel->isAncestor(owner)) {
        if (provenParallelOps.contains(owner) &&
            isParallelInductionVariable(owner, value)) {
          addCoefficient(expression.descendantLanes, value,
                         ::llvm::APInt(width, 1));
          return finish(std::move(expression));
        }
        return finish(std::nullopt);
      }
      addSymbol(expression, value, ::llvm::APInt(width, 1));
      return finish(std::move(expression));
    }

    ::mlir::Operation *def = value.getDefiningOp();
    if (!def || !parallel->isAncestor(def)) {
      addSymbol(expression, value, ::llvm::APInt(width, 1));
      return finish(std::move(expression));
    }

    if (auto add = ::llvm::dyn_cast<::mlir::arith::AddIOp>(def))
      return finish(combine(add.getLhs(), add.getRhs(), /*subtract=*/false));
    if (auto sub = ::llvm::dyn_cast<::mlir::arith::SubIOp>(def))
      return finish(combine(sub.getLhs(), sub.getRhs(), /*subtract=*/true));
    if (auto mul = ::llvm::dyn_cast<::mlir::arith::MulIOp>(def))
      return finish(multiply(mul.getLhs(), mul.getRhs()));
    if (auto cast = ::llvm::dyn_cast<::mlir::arith::IndexCastOp>(def))
      return finish(projectIndexCast(cast.getIn(), cast.getType(), false));
    if (auto cast = ::llvm::dyn_cast<::mlir::arith::IndexCastUIOp>(def))
      return finish(projectIndexCast(cast.getIn(), cast.getType(), true));
    if (auto truncate = ::llvm::dyn_cast<::mlir::arith::TruncIOp>(def)) {
      auto expression = build(truncate.getIn());
      auto resultType =
          ::llvm::dyn_cast<::mlir::IntegerType>(truncate.getType());
      if (!expression || !resultType)
        return finish(std::nullopt);
      expression->transforms.push_back(
          {LinearExpression::TransformKind::SignedBitProjection,
           ::llvm::APInt(width, resultType.getWidth())});
      return finish(std::move(expression));
    }
    if (auto divide = ::llvm::dyn_cast<::mlir::arith::DivSIOp>(def)) {
      ::mlir::APInt divisor;
      if (!::mlir::matchPattern(divide.getRhs(),
                                ::mlir::m_ConstantInt(&divisor)) ||
          divisor.isZero() || divisor.isNegative())
        return finish(std::nullopt);
      auto expression = build(divide.getLhs());
      if (!expression || !expression->symbols.empty() ||
          !expression->descendantLanes.empty())
        return finish(std::nullopt);
      expression->transforms.push_back(
          {LinearExpression::TransformKind::SignedDivide,
           divisor.sextOrTrunc(width)});
      return finish(std::move(expression));
    }

    if (!::mlir::isPure(def) || dependsOnLane(value))
      return finish(std::nullopt);
    addSymbol(expression, value, ::llvm::APInt(width, 1));
    return finish(std::move(expression));
  }

private:
  ::mlir::Operation *parallel;
  ::llvm::SmallVector<::mlir::Value, 4> inductionVars;
  unsigned width;
  const ::llvm::DenseSet<::mlir::Operation *> &provenParallelOps;
  ::llvm::DenseMap<::mlir::Value, LinearExpression> cache;
  ::llvm::DenseSet<::mlir::Value> failed;
  ::llvm::DenseSet<::mlir::Value> active;
  ::llvm::DenseMap<::mlir::Value, bool> dependencyCache;
  ::llvm::DenseSet<::mlir::Value> dependencyActive;

  std::optional<unsigned> integerWidth(::mlir::Type type) const {
    if (::llvm::isa<::mlir::IndexType>(type))
      return width;
    if (auto integer = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      return integer.getWidth();
    return std::nullopt;
  }

  std::optional<LinearExpression> projectIndexCast(::mlir::Value input,
                                                   ::mlir::Type resultType,
                                                   bool unsignedExtension) {
    auto sourceWidth = integerWidth(input.getType());
    auto destinationWidth = integerWidth(resultType);
    auto expression = build(input);
    if (!sourceWidth || !destinationWidth || !expression)
      return std::nullopt;
    if (*destinationWidth < *sourceWidth) {
      expression->transforms.push_back(
          {LinearExpression::TransformKind::SignedBitProjection,
           ::llvm::APInt(width, std::min(*destinationWidth, width))});
    } else if (unsignedExtension && *sourceWidth < *destinationWidth &&
               *sourceWidth < width) {
      expression->transforms.push_back(
          {LinearExpression::TransformKind::UnsignedBitProjection,
           ::llvm::APInt(width, *sourceWidth)});
    }
    return expression;
  }

  void
  addCoefficient(::llvm::DenseMap<::mlir::Value, ::llvm::APInt> &coefficients,
                 ::mlir::Value symbol, ::llvm::APInt coefficient) {
    if (coefficient.isZero())
      return;
    auto found = coefficients.find(symbol);
    if (found == coefficients.end()) {
      coefficients.try_emplace(symbol, std::move(coefficient));
      return;
    }
    found->second += coefficient;
    if (found->second.isZero())
      coefficients.erase(found);
  }

  void addSymbol(LinearExpression &expression, ::mlir::Value symbol,
                 ::llvm::APInt coefficient) {
    addCoefficient(expression.symbols, symbol, std::move(coefficient));
  }

  void add(LinearExpression &target, const LinearExpression &source,
           bool subtract) {
    target.constant = subtract ? target.constant - source.constant
                               : target.constant + source.constant;
    for (unsigned index = 0; index < target.lanes.size(); ++index)
      target.lanes[index] = subtract
                                ? target.lanes[index] - source.lanes[index]
                                : target.lanes[index] + source.lanes[index];
    for (const auto &[symbol, coefficient] : source.symbols)
      addSymbol(target, symbol, subtract ? -coefficient : coefficient);
    for (const auto &[lane, coefficient] : source.descendantLanes)
      addCoefficient(target.descendantLanes, lane,
                     subtract ? -coefficient : coefficient);
  }

  std::optional<LinearExpression> combine(::mlir::Value lhs, ::mlir::Value rhs,
                                          bool subtract) {
    auto left = build(lhs);
    auto right = build(rhs);
    if (!left || !right || !left->transforms.empty() ||
        !right->transforms.empty())
      return std::nullopt;
    add(*left, *right, subtract);
    return left;
  }

  std::optional<LinearExpression> multiply(::mlir::Value lhs,
                                           ::mlir::Value rhs) {
    auto left = build(lhs);
    auto right = build(rhs);
    if (!left || !right || !left->transforms.empty() ||
        !right->transforms.empty())
      return std::nullopt;

    auto isConstant = [](const LinearExpression &expression) {
      return expression.symbols.empty() && expression.descendantLanes.empty() &&
             ::llvm::all_of(expression.lanes, [](const ::llvm::APInt &value) {
               return value.isZero();
             });
    };
    if (!isConstant(*left) && !isConstant(*right))
      return std::nullopt;
    if (!isConstant(*left))
      std::swap(left, right);

    ::llvm::APInt scale = left->constant;
    right->constant *= scale;
    for (::llvm::APInt &coefficient : right->lanes)
      coefficient *= scale;
    for (auto &[symbol, coefficient] : right->symbols) {
      (void)symbol;
      coefficient *= scale;
    }
    for (auto &[lane, coefficient] : right->descendantLanes) {
      (void)lane;
      coefficient *= scale;
    }
    return right;
  }

  bool dependsOnLane(::mlir::Value value) {
    if (auto found = dependencyCache.find(value);
        found != dependencyCache.end())
      return found->second;
    if (!dependencyActive.insert(value).second)
      return true;

    bool depends = ::llvm::is_contained(inductionVars, value);
    if (!depends) {
      if (auto mapped = mapSpatialArgument(value)) {
        depends = dependsOnLane(*mapped);
      } else if (auto argument =
                     ::llvm::dyn_cast<::mlir::BlockArgument>(value)) {
        ::mlir::Operation *owner = argument.getOwner()->getParentOp();
        depends = owner && parallel->isAncestor(owner);
      } else if (::mlir::Operation *def = value.getDefiningOp();
                 def && parallel->isAncestor(def)) {
        depends =
            def->getNumRegions() != 0 ||
            ::llvm::any_of(def->getOperands(), [&](::mlir::Value operand) {
              return dependsOnLane(operand);
            });
      }
    }

    dependencyActive.erase(value);
    dependencyCache.try_emplace(value, depends);
    return depends;
  }
};

bool sameSymbols(const LinearExpression &lhs, const LinearExpression &rhs) {
  if (lhs.symbols.size() != rhs.symbols.size() ||
      lhs.transforms != rhs.transforms)
    return false;
  for (const auto &[symbol, coefficient] : lhs.symbols) {
    auto found = rhs.symbols.find(symbol);
    if (found == rhs.symbols.end() || found->second != coefficient)
      return false;
  }
  return true;
}

::llvm::APInt evaluateLaneConstant(const LinearExpression &expression,
                                   ::llvm::ArrayRef<int64_t> point) {
  ::llvm::APInt result = expression.constant;
  for (auto [coefficient, coordinate] :
       ::llvm::zip_equal(expression.lanes, point))
    result += coefficient * ::llvm::APInt(result.getBitWidth(),
                                          static_cast<uint64_t>(coordinate),
                                          /*isSigned=*/true);
  for (const LinearExpression::Transform &transform : expression.transforms) {
    switch (transform.kind) {
    case LinearExpression::TransformKind::SignedDivide:
      result = result.sdiv(transform.operand);
      break;
    case LinearExpression::TransformKind::SignedBitProjection: {
      const unsigned targetWidth =
          static_cast<unsigned>(transform.operand.getZExtValue());
      result =
          result.trunc(targetWidth).sext(expression.constant.getBitWidth());
      break;
    }
    case LinearExpression::TransformKind::UnsignedBitProjection: {
      const unsigned targetWidth =
          static_cast<unsigned>(transform.operand.getZExtValue());
      result =
          result.trunc(targetWidth).zext(expression.constant.getBitWidth());
      break;
    }
    }
  }
  return result;
}

struct ByteAddressTermExpression {
  LinearExpression index;
  std::int64_t byteStride = 0;
  std::int64_t elementScale = 0;
  unsigned exactSignedDivideShift = 0;
};

struct ByteAccessExpression {
  const ParallelMemoryAccess *access = nullptr;
  ::llvm::SmallVector<ByteAddressTermExpression, 4> terms;
  std::int64_t byteBias = 0;
  std::int64_t elementBias = 0;
  std::uint64_t accessByteCount = 0;
  bool exactCanonicalElementProjection = false;
};

using ByteSymbolProjection = ::llvm::DenseMap<::mlir::Value, ::llvm::APInt>;

std::optional<ByteSymbolProjection>
projectByteAddressSymbols(const ByteAccessExpression &address, unsigned width) {
  ByteSymbolProjection projection;
  for (const ByteAddressTermExpression &term : address.terms) {
    if (!term.index.descendantLanes.empty() ||
        (!term.index.transforms.empty() && !term.index.symbols.empty()))
      return std::nullopt;
    if (!term.index.transforms.empty())
      continue;
    ::llvm::APInt stride(width, static_cast<std::uint64_t>(term.byteStride),
                         /*isSigned=*/true);
    for (const auto &[symbol, coefficient] : term.index.symbols) {
      ::llvm::APInt scaled = coefficient.sextOrTrunc(width) * stride;
      auto found = projection.find(symbol);
      if (found == projection.end()) {
        if (!scaled.isZero())
          projection.try_emplace(symbol, std::move(scaled));
        continue;
      }
      found->second += scaled;
      if (found->second.isZero())
        projection.erase(found);
    }
  }
  return projection;
}

bool sameByteAddressSymbols(const ByteSymbolProjection &lhs,
                            const ByteSymbolProjection &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (const auto &[symbol, coefficient] : lhs) {
    auto found = rhs.find(symbol);
    if (found == rhs.end() || found->second != coefficient)
      return false;
  }
  return true;
}

::llvm::APInt evaluateByteAddress(const ByteAccessExpression &address,
                                  ::llvm::ArrayRef<int64_t> point,
                                  unsigned width) {
  ::llvm::APInt result(width, static_cast<std::uint64_t>(address.byteBias),
                       /*isSigned=*/true);
  for (const ByteAddressTermExpression &term : address.terms) {
    ::llvm::APInt index =
        evaluateLaneConstant(term.index, point).sextOrTrunc(width);
    ::llvm::APInt stride(width, static_cast<std::uint64_t>(term.byteStride),
                         /*isSigned=*/true);
    result += index * stride;
  }
  return result;
}

bool hasExactCanonicalElementProjection(
    const ResolvedLinearMemoryAddress &address) {
  if (address.elementTerms.empty()) {
    auto indexType = ::llvm::dyn_cast<::mlir::IntegerType>(address.indexType);
    return indexType &&
           ::llvm::isIntN(indexType.getWidth(), address.elementBias);
  }
  return address.elementTerms.size() == 1 && address.elementBias == 0 &&
         address.elementTerms.front().scale == 1 &&
         address.elementTerms.front().exactSignedDivideShift == 0;
}

bool hasRepresentableCanonicalElementArithmetic(
    const ByteAccessExpression &address, ::llvm::ArrayRef<int64_t> point,
    unsigned comparisonWidth, unsigned canonicalWidth) {
  ::llvm::APInt result(comparisonWidth, 0);
  for (const ByteAddressTermExpression &term : address.terms) {
    ::llvm::APInt contribution =
        evaluateLaneConstant(term.index, point).sextOrTrunc(comparisonWidth);
    if (term.exactSignedDivideShift != 0)
      contribution = contribution.ashr(term.exactSignedDivideShift);
    contribution *= ::llvm::APInt(comparisonWidth,
                                  static_cast<std::uint64_t>(term.elementScale),
                                  /*isSigned=*/true);
    if (!contribution.isSignedIntN(canonicalWidth))
      return false;
    result += contribution;
    if (!result.isSignedIntN(canonicalWidth))
      return false;
  }
  ::llvm::APInt bias(comparisonWidth,
                     static_cast<std::uint64_t>(address.elementBias),
                     /*isSigned=*/true);
  if (!bias.isSignedIntN(canonicalWidth))
    return false;
  result += bias;
  return result.isSignedIntN(canonicalWidth);
}

bool hasDynamicByteLaneSeparation(const ByteAccessExpression &address,
                                  unsigned laneCount, unsigned width) {
  if (laneCount != 1 || !address.exactCanonicalElementProjection)
    return false;
  ::llvm::APInt stride(width, 0);
  for (const ByteAddressTermExpression &term : address.terms) {
    if (!term.index.transforms.empty() || !term.index.descendantLanes.empty())
      return false;
    ::llvm::APInt byteStride(width, static_cast<std::uint64_t>(term.byteStride),
                             /*isSigned=*/true);
    stride += term.index.lanes.front().sextOrTrunc(width) * byteStride;
  }
  ::llvm::APInt accessBytes(width, address.accessByteCount);
  return !stride.isZero() && stride.abs().uge(accessBytes);
}

struct ByteInterval {
  ::llvm::APInt begin;
  ::llvm::APInt end;
  std::uint64_t lane = 0;
  bool writes = false;
  bool atomic = false;
};

struct AddressKeyHash {
  std::size_t operator()(const std::vector<::llvm::APInt> &address) const {
    ::llvm::hash_code hash = ::llvm::hash_value(address.size());
    for (const ::llvm::APInt &coordinate : address)
      hash = ::llvm::hash_combine(hash, coordinate);
    return hash;
  }
};

struct SeenAddress {
  uint64_t firstLane;
  bool multipleLanes = false;
  bool writes = false;
  bool allAtomic = true;
};

struct ParallelCheckInfo {
  ::mlir::Operation *op;
  std::optional<FixedParallelDomain> domain;
  bool owned;
  ::llvm::SmallVector<ParallelMemoryAccess, 8> accesses;
  ::mlir::Operation *unmodeledWrite = nullptr;
};

::mlir::LogicalResult checkLaneMemoryLegality(
    ParallelCheckInfo &info,
    const ::llvm::DenseSet<::mlir::Operation *> &provenParallelOps) {
  if (info.unmodeledWrite)
    return info.unmodeledWrite->emitError(
        "loom-lower-graph-memory: parallel lane effect has no disjoint, "
        "atomic, reduction, or ordered proof");

  bool anyWrite =
      ::llvm::any_of(info.accesses, [](const ParallelMemoryAccess &access) {
        return access.writes;
      });

  bool allAtomic =
      ::llvm::all_of(info.accesses, [](const ParallelMemoryAccess &access) {
        return access.atomic;
      });
  MemoryRootResolver roots;
  ::llvm::DenseMap<::mlir::Value,
                   ::llvm::SmallVector<const ParallelMemoryAccess *, 4>>
      accessesByRoot;
  bool unresolved = false;
  for (const ParallelMemoryAccess &access : info.accesses) {
    auto root = roots.resolve(access.memory);
    if (!root) {
      unresolved = true;
      continue;
    }
    accessesByRoot[*root].push_back(&access);
  }
  if (unresolved && anyWrite && !allAtomic)
    return info.op->emitError(
        "loom-lower-graph-memory: parallel memory root is unresolved");

  ::llvm::Expected<unsigned> indexBits = ::loom::getIndexBitWidth(info.op);
  if (!indexBits)
    return info.op->emitError("loom-lower-graph-memory: ")
           << ::llvm::toString(indexBits.takeError());

  ::llvm::SmallVector<::mlir::Value, 4> inductionVars;
  if (auto parallel = ::llvm::dyn_cast<::mlir::scf::ParallelOp>(info.op)) {
    auto vars = parallel.getInductionVars();
    inductionVars.append(vars.begin(), vars.end());
  } else {
    auto forall = ::mlir::cast<::mlir::scf::ForallOp>(info.op);
    auto vars = forall.getInductionVars();
    inductionVars.append(vars.begin(), vars.end());
  }
  LinearAddressBuilder expressions(info.op, inductionVars, *indexBits,
                                   provenParallelOps);

  for (auto &[root, rootAccesses] : accessesByRoot) {
    (void)root;
    bool rootWrites =
        ::llvm::any_of(rootAccesses, [](const ParallelMemoryAccess *access) {
          return access->writes;
        });
    bool rootAtomic =
        ::llvm::all_of(rootAccesses, [](const ParallelMemoryAccess *access) {
          return access->atomic;
        });

    bool hasLlvmAccess =
        ::llvm::any_of(rootAccesses, [](const ParallelMemoryAccess *access) {
          return static_cast<bool>(access->llvmAccessType);
        });
    if (!hasLlvmAccess) {
      ::mlir::Value memory = mapSpatialArguments(rootAccesses.front()->memory);
      if (::llvm::any_of(rootAccesses, [&](const ParallelMemoryAccess *access) {
            return mapSpatialArguments(access->memory) != memory;
          }))
        return info.op->emitError(
            "loom-lower-graph-memory: parallel accesses disagree on their "
            "resolved memory base");
    }
    if (hasLlvmAccess) {
      if (::llvm::any_of(rootAccesses, [](const ParallelMemoryAccess *access) {
            return !access->llvmAccessType;
          }))
        return info.op->emitError(
            "loom-lower-graph-memory: parallel memory root mixes LLVM byte "
            "addresses with non-LLVM indices");
      if (*indexBits > (mlir::IntegerType::kMaxWidth - 1) / 2)
        return info.op->emitError(
            "loom-lower-graph-memory: parallel byte-address comparison width "
            "is invalid");
      const unsigned comparisonWidth = *indexBits * 2 + 1;
      ::llvm::SmallVector<ByteAccessExpression, 8> byteAddresses;
      byteAddresses.reserve(rootAccesses.size());
      for (const ParallelMemoryAccess *access : rootAccesses) {
        auto resolved = resolveLinearMemoryAddress(
            access->memory, access->llvmAccessType, *indexBits);
        if (!resolved)
          return access->op->emitError(
              "loom-lower-graph-memory: LLVM memory access has no exact "
              "DataLayout byte-address projection");
        auto resolvedRoot = roots.resolve(resolved->root);
        if (!resolvedRoot || *resolvedRoot != root)
          return access->op->emitError(
              "loom-lower-graph-memory: LLVM byte-address projection changed "
              "its resolved memory root");

        ByteAccessExpression address;
        address.access = access;
        address.byteBias = resolved->byteBias;
        address.elementBias = resolved->elementBias;
        address.accessByteCount = resolved->accessByteCount;
        address.exactCanonicalElementProjection =
            hasExactCanonicalElementProjection(*resolved);
        address.terms.reserve(resolved->terms.size());
        for (auto [byteTerm, elementTerm] :
             ::llvm::zip_equal(resolved->terms, resolved->elementTerms)) {
          auto expression = expressions.build(byteTerm.index);
          if (!expression)
            return access->op->emitError(
                "loom-lower-graph-memory: LLVM byte address has no affine "
                "lane projection");
          address.terms.push_back({std::move(*expression), byteTerm.byteStride,
                                   elementTerm.scale,
                                   elementTerm.exactSignedDivideShift});
        }
        byteAddresses.push_back(std::move(address));
      }

      if (!info.domain) {
        if (::llvm::any_of(byteAddresses,
                           [](const ByteAccessExpression &value) {
                             return !value.exactCanonicalElementProjection;
                           }))
          return info.op->emitError(
              "loom-lower-graph-memory: dynamic LLVM element address has no "
              "exact canonical-index projection");
        if (rootWrites && !rootAtomic &&
            (byteAddresses.size() != 1 ||
             !hasDynamicByteLaneSeparation(
                 byteAddresses.front(), inductionVars.size(), comparisonWidth)))
          return info.op->emitError(
              "loom-lower-graph-memory: dynamic parallel LLVM memory effect "
              "has no exact byte-disjoint lane projection");
        continue;
      }

      auto referenceSymbols =
          projectByteAddressSymbols(byteAddresses.front(), comparisonWidth);
      if (!referenceSymbols)
        return info.op->emitError(
            "loom-lower-graph-memory: LLVM byte address has an incomparable "
            "symbolic projection");
      for (const ByteAccessExpression &address :
           ::llvm::drop_begin(byteAddresses)) {
        auto symbols = projectByteAddressSymbols(address, comparisonWidth);
        if (!symbols || !sameByteAddressSymbols(*referenceSymbols, *symbols))
          return info.op->emitError(
              "loom-lower-graph-memory: LLVM byte addresses have different "
              "symbolic projections");
      }
      if (!referenceSymbols->empty() &&
          ::llvm::any_of(byteAddresses, [](const ByteAccessExpression &value) {
            return !value.exactCanonicalElementProjection;
          }))
        return info.op->emitError(
            "loom-lower-graph-memory: symbolic LLVM element address has no "
            "exact canonical-index projection");

      ::llvm::SmallVector<ByteInterval, 16> intervals;
      std::uint64_t lane = 0;
      bool unrepresentableElementIndex = false;
      ::llvm::SmallVector<int64_t, 4> point;
      (void)forEachParallelPointUntil(
          *info.domain, 0, point, [&](::llvm::ArrayRef<int64_t> coordinates) {
            for (const ByteAccessExpression &address : byteAddresses) {
              ::llvm::APInt begin =
                  evaluateByteAddress(address, coordinates, comparisonWidth);
              if (!address.exactCanonicalElementProjection &&
                  !hasRepresentableCanonicalElementArithmetic(
                      address, coordinates, comparisonWidth, *indexBits)) {
                unrepresentableElementIndex = true;
                return false;
              }
              ::llvm::APInt end =
                  begin +
                  ::llvm::APInt(comparisonWidth, address.accessByteCount);
              intervals.push_back(ByteInterval{std::move(begin), std::move(end),
                                               lane, address.access->writes,
                                               address.access->atomic});
            }
            ++lane;
            return true;
          });
      if (unrepresentableElementIndex)
        return info.op->emitError(
            "loom-lower-graph-memory: LLVM element address exceeds the "
            "selected canonical index width");
      if (!rootWrites || rootAtomic)
        continue;
      ::llvm::sort(intervals,
                   [](const ByteInterval &lhs, const ByteInterval &rhs) {
                     if (lhs.begin != rhs.begin)
                       return lhs.begin.slt(rhs.begin);
                     if (lhs.end != rhs.end)
                       return lhs.end.slt(rhs.end);
                     return lhs.lane < rhs.lane;
                   });
      ::llvm::SmallVector<ByteInterval, 8> active;
      bool overlap = false;
      for (const ByteInterval &interval : intervals) {
        ::llvm::erase_if(active, [&](const ByteInterval &candidate) {
          return candidate.end.sle(interval.begin);
        });
        for (const ByteInterval &candidate : active) {
          if (candidate.lane != interval.lane &&
              (candidate.writes || interval.writes) &&
              (!candidate.atomic || !interval.atomic)) {
            overlap = true;
            break;
          }
        }
        if (overlap)
          break;
        active.push_back(interval);
      }
      if (overlap)
        return info.op->emitError(
            "loom-lower-graph-memory: parallel lanes have overlapping plain "
            "memory byte ranges");
      continue;
    }

    if (!rootWrites || rootAtomic)
      continue;

    struct AccessExpressions {
      const ParallelMemoryAccess *access;
      ::llvm::SmallVector<LinearExpression, 4> address;
    };
    ::llvm::SmallVector<AccessExpressions, 8> analyzed;
    for (const ParallelMemoryAccess *access : rootAccesses) {
      if (::llvm::any_of(access->address, [](const ::mlir::Value value) {
            return !value.getType().isIntOrIndex();
          }))
        return info.op->emitError(
            "loom-lower-graph-memory: parallel address is not integer-typed");

      AccessExpressions one{access, {}};
      for (::mlir::Value address : access->address) {
        auto expression = expressions.build(address);
        if (!expression)
          return info.op->emitError(
              "loom-lower-graph-memory: parallel address has no affine lane "
              "projection");
        one.address.push_back(std::move(*expression));
      }
      analyzed.push_back(std::move(one));
    }

    unsigned rank = analyzed.front().address.size();
    if (::llvm::any_of(analyzed, [&](const AccessExpressions &access) {
          return access.address.size() != rank;
        }))
      return info.op->emitError(
          "loom-lower-graph-memory: parallel accesses have different address "
          "ranks");

    ::llvm::SmallVector<unsigned, 4> comparableDimensions;
    for (unsigned dimension = 0; dimension < rank; ++dimension) {
      const LinearExpression &reference = analyzed.front().address[dimension];
      if (reference.descendantLanes.empty() &&
          ::llvm::all_of(analyzed, [&](const AccessExpressions &access) {
            return access.address[dimension].descendantLanes.empty() &&
                   sameSymbols(reference, access.address[dimension]);
          }))
        comparableDimensions.push_back(dimension);
    }
    if (comparableDimensions.empty()) {
      auto diagnostic = info.op->emitError(
          "loom-lower-graph-memory: parallel addresses have no comparable "
          "lane projection; address rank is ");
      diagnostic << rank;
      return ::mlir::failure();
    }

    if (!info.domain) {
      if (analyzed.size() != 1)
        return info.op->emitError(
            "loom-lower-graph-memory: dynamic parallel plain-memory effects "
            "require one injective access per memory root");
      ::llvm::SmallVector<bool, 4> projected(inductionVars.size(), false);
      for (const LinearExpression &expression : analyzed.front().address) {
        if (!expression.transforms.empty() ||
            !expression.descendantLanes.empty())
          continue;
        std::optional<unsigned> lane;
        bool unitProjection = true;
        for (auto [ordinal, coefficient] :
             ::llvm::enumerate(expression.lanes)) {
          if (coefficient.isZero())
            continue;
          if (lane || (!coefficient.isOne() && !coefficient.isAllOnes())) {
            unitProjection = false;
            break;
          }
          lane = ordinal;
        }
        if (unitProjection && lane && !projected[*lane])
          projected[*lane] = true;
      }
      if (!::llvm::all_of(projected, [](bool value) { return value; }))
        return info.op->emitError(
            "loom-lower-graph-memory: dynamic parallel plain-memory effects "
            "have no exact injective lane projection");
      continue;
    }

    std::unordered_map<std::vector<::llvm::APInt>, SeenAddress, AddressKeyHash>
        seen;
    uint64_t lane = 0;
    bool overlap = false;
    ::llvm::SmallVector<int64_t, 4> point;
    (void)forEachParallelPointUntil(
        *info.domain, 0, point, [&](::llvm::ArrayRef<int64_t> coordinates) {
          uint64_t currentLane = lane++;
          for (const AccessExpressions &access : analyzed) {
            std::vector<::llvm::APInt> key;
            key.reserve(comparableDimensions.size());
            for (unsigned dimension : comparableDimensions)
              key.push_back(
                  evaluateLaneConstant(access.address[dimension], coordinates));

            auto [found, inserted] = seen.try_emplace(
                std::move(key),
                SeenAddress{currentLane, /*multipleLanes=*/false,
                            access.access->writes, access.access->atomic});
            if (inserted)
              continue;

            SeenAddress &previous = found->second;
            bool differentLane =
                previous.firstLane != currentLane || previous.multipleLanes;
            if (differentLane && (previous.writes || access.access->writes) &&
                (!previous.allAtomic || !access.access->atomic)) {
              overlap = true;
              return false;
            }
            if (previous.firstLane != currentLane)
              previous.multipleLanes = true;
            previous.writes |= access.access->writes;
            previous.allAtomic &= access.access->atomic;
          }
          return true;
        });
    if (overlap)
      return info.op->emitError(
          "loom-lower-graph-memory: parallel lanes have overlapping plain "
          "memory effects");
  }
  return ::mlir::success();
}

} // namespace

std::optional<FixedParallelDomain>
getFixedParallelDomain(::mlir::Operation *op) {
  if (auto forall = ::llvm::dyn_cast<::mlir::scf::ForallOp>(op))
    return getFixedParallelDomain(forall);
  if (auto parallel = ::llvm::dyn_cast<::mlir::scf::ParallelOp>(op))
    return getFixedParallelDomain(parallel);
  return std::nullopt;
}

void forEachParallelPoint(
    const FixedParallelDomain &domain,
    ::llvm::function_ref<void(::llvm::ArrayRef<int64_t>)> callback) {
  ::llvm::SmallVector<int64_t, 4> point;
  (void)forEachParallelPointUntil(domain, 0, point,
                                  [&](::llvm::ArrayRef<int64_t> coordinates) {
                                    callback(coordinates);
                                    return true;
                                  });
}

namespace {

::mlir::LogicalResult
checkParallelPreconditions(::llvm::ArrayRef<::mlir::Operation *> parallelOps,
                           bool requireFixedDomain, bool selectedOwnership) {
  if (parallelOps.empty())
    return ::mlir::success();

  ::llvm::DenseMap<::mlir::Operation *, ParallelCheckInfo> checks;
  ::llvm::DenseSet<::mlir::Operation *> provenParallelOps;
  for (::mlir::Operation *op : parallelOps) {
    if (op->hasAttr("loom.parallel_group") ||
        op->hasAttr("loom.parallel_schedule"))
      return op->emitError(
          "loom-lower-graph-memory: parallel SCF carries unsupported author "
          "metadata");

    if (auto mapping = op->getAttrOfType<::mlir::ArrayAttr>("mapping");
        mapping && !mapping.empty())
      return op->emitError(
          "loom-lower-graph-memory: graph-owned parallel SCF must not retain "
          "an execution-resource mapping");

    if (auto forall = ::llvm::dyn_cast<::mlir::scf::ForallOp>(op)) {
      auto inParallel = forall.getTerminator();
      if (!forall.getOutputs().empty() || forall.getNumResults() != 0 ||
          inParallel.getRegion().empty() ||
          !inParallel.getRegion().front().empty())
        return forall.emitError(
            "loom-lower-graph-memory: graph-owned scf.forall must be in "
            "effect form before fixed-lane lowering");
    } else {
      auto parallel = ::mlir::cast<::mlir::scf::ParallelOp>(op);
      auto reduce = ::mlir::cast<::mlir::scf::ReduceOp>(
          parallel.getBody()->getTerminator());
      if (!parallel.getInitVals().empty() || parallel.getNumResults() != 0 ||
          !reduce.getOperands().empty() || !reduce.getReductions().empty())
        return parallel.emitError(
            "loom-lower-graph-memory: graph-owned scf.parallel reductions "
            "must be normalized before fixed-lane lowering");
    }

    std::optional<FixedParallelDomain> domain;
    if (auto forall = ::llvm::dyn_cast<::mlir::scf::ForallOp>(op))
      domain = getFixedParallelDomain(forall);
    else if (auto parallel = ::llvm::dyn_cast<::mlir::scf::ParallelOp>(op))
      domain = getFixedParallelDomain(parallel);
    if (requireFixedDomain && !domain)
      return op->emitError(
          "loom-lower-graph-memory: selected graph-owned parallel SCF "
          "requires a fixed compile-time lane domain");
    if (domain &&
        ::llvm::any_of(domain->step, [](int64_t step) { return step <= 0; }))
      return op->emitError(
          "loom-lower-graph-memory: selected graph-owned parallel SCF "
          "requires positive fixed lane steps");

    bool graphOwned =
        static_cast<bool>(op->getParentOfType<::dataflow::GraphOp>());
    checks.try_emplace(op, ParallelCheckInfo{op,
                                             std::move(domain),
                                             selectedOwnership || graphOwned ||
                                                 hasSpatialCarrierAncestor(op),
                                             {},
                                             nullptr});
    provenParallelOps.insert(op);
  }

  ::mlir::Operation *root = parallelOps.front();
  while (root->getParentOp())
    root = root->getParentOp();
  ::mlir::WalkResult effects = root->walk([&](::mlir::Operation *nested) {
    if (nested->getName().getStringRef() == "loom.spatial_region") {
      for (::mlir::Operation *parent = nested->getParentOp(); parent;
           parent = parent->getParentOp()) {
        auto found = checks.find(parent);
        if (found != checks.end())
          found->second.owned = true;
      }
    }

    if (hasUnsupportedParallelCompletion(nested)) {
      for (::mlir::Operation *parent = nested->getParentOp(); parent;
           parent = parent->getParentOp()) {
        if (!checks.contains(parent))
          continue;
        nested->emitError() << "loom-lower-graph-memory: parallel actor '"
                            << nested->getName().getStringRef()
                            << "' has no completion lowering";
        return ::mlir::WalkResult::interrupt();
      }
    }

    auto access = getParallelMemoryAccess(nested);
    bool unmodeledWrite = !access && hasUnmodeledWriteEffect(nested);
    if (!access && !unmodeledWrite)
      return ::mlir::WalkResult::advance();
    for (::mlir::Operation *parent = nested->getParentOp(); parent;
         parent = parent->getParentOp()) {
      auto found = checks.find(parent);
      if (found == checks.end())
        continue;
      if (access)
        found->second.accesses.push_back(*access);
      else if (!found->second.unmodeledWrite)
        found->second.unmodeledWrite = nested;
    }
    return ::mlir::WalkResult::advance();
  });
  if (effects.wasInterrupted())
    return ::mlir::failure();

  for (auto &entry : checks) {
    ParallelCheckInfo &info = entry.second;
    if (!info.owned)
      return info.op->emitError(
          "loom-lower-graph-memory: raw parallel SCF has no compiler-owned "
          "graph or spatial structure");
    if (::mlir::failed(checkLaneMemoryLegality(info, provenParallelOps)))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

} // namespace

::mlir::LogicalResult checkGraphOwnedParallelPreconditions(
    ::llvm::ArrayRef<::mlir::Operation *> parallelOps) {
  return checkParallelPreconditions(parallelOps,
                                    /*requireFixedDomain=*/true,
                                    /*selectedOwnership=*/false);
}

::mlir::LogicalResult
checkLogicalThreadParallelPreconditions(::mlir::Operation *forall) {
  if (!::llvm::isa_and_nonnull<::mlir::scf::ForallOp>(forall))
    return ::mlir::failure();
  return checkParallelPreconditions({forall}, /*requireFixedDomain=*/false,
                                    /*selectedOwnership=*/true);
}

} // namespace lowering
} // namespace loom
