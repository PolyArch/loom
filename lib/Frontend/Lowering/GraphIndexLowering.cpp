#include "GraphIndexLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace {

// An index attribute holds its value in one fixed-width APInt, so a constant
// the canonical index admits but that storage cannot hold exactly has no index
// attribute. The value is carried exactly or not at all; it is never narrowed
// through a host integer.
std::optional<::mlir::IntegerAttr>
getExactIndexAttr(::mlir::OpBuilder &builder, const ::llvm::APInt &value) {
  constexpr unsigned storage = ::mlir::IndexType::kInternalStorageBitWidth;
  if (value.getSignificantBits() > storage)
    return std::nullopt;
  return builder.getIntegerAttr(builder.getIndexType(),
                                value.sextOrTrunc(storage));
}

class IndexMaterializationTransaction {
public:
  explicit IndexMaterializationTransaction(::mlir::Operation *anchor)
      : anchor(anchor), originalPrevious(anchor->getPrevNode()) {}

  ~IndexMaterializationTransaction() {
    if (committed)
      return;
    while (::mlir::Operation *created = anchor->getPrevNode()) {
      if (created == originalPrevious)
        break;
      created->erase();
    }
  }

  void commit() { committed = true; }

private:
  ::mlir::Operation *anchor;
  ::mlir::Operation *originalPrevious;
  bool committed = false;
};

::mlir::Value materializeIndexDomainValue(
    ::mlir::Value value, ::mlir::OpBuilder &builder,
    ::llvm::DenseMap<::mlir::Value, ::mlir::Value> &cache, unsigned indexBits) {
  if (::llvm::isa<::mlir::IndexType>(value.getType()))
    return value;
  if (!::llvm::isa<::mlir::IntegerType>(value.getType()))
    return {};
  if (auto it = cache.find(value); it != cache.end())
    return it->second;

  if (auto cast = value.getDefiningOp<::mlir::arith::IndexCastOp>()) {
    ::mlir::Value input = cast.getIn();
    if (::llvm::isa<::mlir::IndexType>(input.getType())) {
      cache[value] = input;
      return input;
    }
  }

  auto materializeBinary = [&](auto op, auto create) -> ::mlir::Value {
    ::mlir::Value lhs =
        materializeIndexDomainValue(op.getLhs(), builder, cache, indexBits);
    if (!lhs)
      return {};
    ::mlir::Value rhs =
        materializeIndexDomainValue(op.getRhs(), builder, cache, indexBits);
    if (!rhs)
      return {};
    ::mlir::Value result = create(lhs, rhs);
    cache[value] = result;
    return result;
  };

  if (auto add = value.getDefiningOp<::mlir::arith::AddIOp>())
    return materializeBinary(add, [&](::mlir::Value lhs, ::mlir::Value rhs) {
      return ::mlir::arith::AddIOp::create(builder, add.getLoc(), lhs, rhs)
          .getResult();
    });
  if (auto sub = value.getDefiningOp<::mlir::arith::SubIOp>())
    return materializeBinary(sub, [&](::mlir::Value lhs, ::mlir::Value rhs) {
      return ::mlir::arith::SubIOp::create(builder, sub.getLoc(), lhs, rhs)
          .getResult();
    });
  if (auto mul = value.getDefiningOp<::mlir::arith::MulIOp>())
    return materializeBinary(mul, [&](::mlir::Value lhs, ::mlir::Value rhs) {
      return ::mlir::arith::MulIOp::create(builder, mul.getLoc(), lhs, rhs)
          .getResult();
    });
  if (auto shl = value.getDefiningOp<::mlir::arith::ShLIOp>())
    return materializeBinary(shl, [&](::mlir::Value lhs, ::mlir::Value rhs) {
      return ::mlir::arith::ShLIOp::create(builder, shl.getLoc(), lhs, rhs)
          .getResult();
    });
  if (auto shr = value.getDefiningOp<::mlir::arith::ShRUIOp>())
    return materializeBinary(shr, [&](::mlir::Value lhs, ::mlir::Value rhs) {
      return ::mlir::arith::ShRUIOp::create(builder, shr.getLoc(), lhs, rhs)
          .getResult();
    });
  if (auto andi = value.getDefiningOp<::mlir::arith::AndIOp>())
    return materializeBinary(andi, [&](::mlir::Value lhs, ::mlir::Value rhs) {
      return ::mlir::arith::AndIOp::create(builder, andi.getLoc(), lhs, rhs)
          .getResult();
    });

  if (auto zext = value.getDefiningOp<::mlir::arith::ExtUIOp>()) {
    auto sourceType =
        ::llvm::dyn_cast<::mlir::IntegerType>(zext.getIn().getType());
    auto resultType =
        ::llvm::dyn_cast<::mlir::IntegerType>(zext.getOut().getType());
    // The extension is redundant in the index domain only when its source
    // already spans the canonical index. A narrower source still carries the
    // zero extension, because converting it directly would sign-extend it.
    if (sourceType && resultType && sourceType.getWidth() == indexBits &&
        resultType.getWidth() >= sourceType.getWidth()) {
      ::mlir::Value input =
          materializeIndexDomainValue(zext.getIn(), builder, cache, indexBits);
      if (!input)
        return {};
      cache[value] = input;
      return input;
    }
  }

  if (auto constant = value.getDefiningOp<::dataflow::ConstantOp>()) {
    auto typed = ::llvm::dyn_cast<::mlir::TypedAttr>(constant.getConstValue());
    auto integer = typed ? ::llvm::dyn_cast<::mlir::IntegerAttr>(typed)
                         : ::mlir::IntegerAttr{};
    if (integer) {
      std::optional<::mlir::IntegerAttr> indexAttr =
          getExactIndexAttr(builder, integer.getValue());
      if (!indexAttr)
        return {};
      auto indexConstant = ::dataflow::ConstantOp::create(
          builder, constant.getLoc(), builder.getIndexType(),
          constant.getCtrl(), ::mlir::cast<::mlir::Attribute>(*indexAttr));
      cache[value] = indexConstant.getValue();
      return indexConstant.getValue();
    }
  }

  if (auto invariant = value.getDefiningOp<::dataflow::InvariantOp>()) {
    bool projected = !value.use_empty();
    for (::mlir::OpOperand &use : value.getUses()) {
      auto gate = ::llvm::dyn_cast<::dataflow::GateOp>(use.getOwner());
      if (!gate || gate.getBeforeValue() != value ||
          gate.getBeforeCond() != invariant.getCond()) {
        projected = false;
        break;
      }
    }
    if (!projected)
      return {};
    ::mlir::Value init = materializeIndexDomainValue(invariant.getInit(),
                                                     builder, cache, indexBits);
    if (!init)
      return {};
    auto indexInvariant = ::dataflow::InvariantOp::create(
        builder, invariant.getLoc(), builder.getIndexType(),
        invariant.getCond(), init);
    cache[value] = indexInvariant.getOutput();
    return indexInvariant.getOutput();
  }

  if (auto gate = value.getDefiningOp<::dataflow::GateOp>()) {
    if (gate.getAfterValue() != value)
      return {};
    ::mlir::Value before = materializeIndexDomainValue(
        gate.getBeforeValue(), builder, cache, indexBits);
    if (!before)
      return {};
    auto indexGate = ::dataflow::GateOp::create(
        builder, gate.getLoc(), builder.getI1Type(), builder.getIndexType(),
        gate.getBeforeCond(), before);
    cache[value] = indexGate.getAfterValue();
    return indexGate.getAfterValue();
  }

  if (::llvm::isa_and_nonnull<::dataflow::CarryOp, ::dataflow::DemuxOp>(
          value.getDefiningOp()))
    return {};

  auto indexCast = ::mlir::arith::IndexCastOp::create(
      builder, value.getLoc(), builder.getIndexType(), value);
  cache[value] = indexCast.getResult();
  return indexCast.getResult();
}

bool isDataflowMemoryAddressUse(::mlir::OpOperand &use) {
  ::mlir::Operation *owner = use.getOwner();
  return (::llvm::isa<::dataflow::LoadOp, ::dataflow::StoreOp>(owner)) &&
         use.getOperandNumber() == 1;
}

bool valueFeedsOnlyMemoryAddress(::mlir::Value value,
                                 ::llvm::SmallPtrSetImpl<::mlir::Value> &seen) {
  if (value.use_empty() || !seen.insert(value).second)
    return false;
  bool sawAddress = false;
  for (::mlir::OpOperand &use : value.getUses()) {
    if (isDataflowMemoryAddressUse(use)) {
      sawAddress = true;
      continue;
    }
    auto select = ::llvm::dyn_cast<::mlir::arith::SelectOp>(use.getOwner());
    if (select && use.getOperandNumber() != 0 &&
        valueFeedsOnlyMemoryAddress(select.getResult(), seen)) {
      sawAddress = true;
      continue;
    }
    return false;
  }
  return sawAddress;
}

bool valueFeedsOnlyMemoryAddress(::mlir::Value value) {
  ::llvm::SmallPtrSet<::mlir::Value, 8> seen;
  return valueFeedsOnlyMemoryAddress(value, seen);
}

bool isMemoryAddressIndexCast(::mlir::arith::IndexCastOp cast) {
  return ::llvm::isa<::mlir::IndexType>(cast.getType()) &&
         ::llvm::isa<::mlir::IntegerType>(cast.getIn().getType()) &&
         valueFeedsOnlyMemoryAddress(cast.getResult());
}

void collectDirectMemoryAddressCasts(
    ::mlir::Value value,
    ::llvm::SmallVectorImpl<::mlir::arith::IndexCastOp> &addressCasts) {
  for (::mlir::OpOperand &use : value.getUses()) {
    auto cast = ::llvm::dyn_cast<::mlir::arith::IndexCastOp>(use.getOwner());
    if (cast && isMemoryAddressIndexCast(cast))
      addressCasts.push_back(cast);
  }
}

bool isPredicateControlUse(::mlir::OpOperand &use) {
  ::mlir::StringRef name = use.getOwner()->getName().getStringRef();
  unsigned operand = use.getOperandNumber();
  if (name == "arith.select" || name == "dataflow.mux" ||
      name == "dataflow.demux" || name == "dataflow.gate")
    return operand == 0;
  if (name == "dataflow.carry" || name == "dataflow.invariant")
    return operand == 0;
  return false;
}

bool valueFeedsOnlyPredicateControls(::mlir::Value value) {
  return !value.use_empty() &&
         ::llvm::all_of(value.getUses(), isPredicateControlUse);
}

bool rewriteOneIndexDomainCmp(::mlir::arith::CmpIOp cmp,
                              ::mlir::OpBuilder &builder, unsigned indexBits) {
  if (!::llvm::isa<::mlir::IntegerType>(cmp.getLhs().getType()) ||
      !::llvm::isa<::mlir::IntegerType>(cmp.getRhs().getType()) ||
      !valueFeedsOnlyPredicateControls(cmp.getResult()))
    return false;

  ::llvm::SmallVector<::mlir::arith::IndexCastOp, 4> lhsAddressCasts;
  ::llvm::SmallVector<::mlir::arith::IndexCastOp, 4> rhsAddressCasts;
  collectDirectMemoryAddressCasts(cmp.getLhs(), lhsAddressCasts);
  collectDirectMemoryAddressCasts(cmp.getRhs(), rhsAddressCasts);
  if (lhsAddressCasts.empty() && rhsAddressCasts.empty())
    return false;

  ::mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(cmp);
  IndexMaterializationTransaction transaction(cmp);
  ::llvm::DenseMap<::mlir::Value, ::mlir::Value> cache;
  ::mlir::Value lhs =
      materializeIndexDomainValue(cmp.getLhs(), builder, cache, indexBits);
  ::mlir::Value rhs =
      materializeIndexDomainValue(cmp.getRhs(), builder, cache, indexBits);
  if (!lhs || !rhs)
    return false;
  transaction.commit();

  auto replacement = ::mlir::arith::CmpIOp::create(
      builder, cmp.getLoc(), cmp.getPredicate(), lhs, rhs);
  cmp.getResult().replaceAllUsesWith(replacement.getResult());
  auto replaceAddressCasts =
      [](::mlir::Value replacement,
         ::llvm::ArrayRef<::mlir::arith::IndexCastOp> casts) {
        for (::mlir::arith::IndexCastOp cast : casts) {
          if (!cast.getOperation()->getBlock())
            continue;
          if (::mlir::Operation *def = replacement.getDefiningOp()) {
            if (def->getBlock() != cast->getBlock() ||
                cast->isBeforeInBlock(def))
              continue;
          }
          cast.replaceAllUsesWith(replacement);
          cast.erase();
        }
      };
  replaceAddressCasts(lhs, lhsAddressCasts);
  replaceAddressCasts(rhs, rhsAddressCasts);
  cmp.erase();
  return true;
}

bool rewriteIndexDomainCmps(::dataflow::GraphOp graph,
                            ::mlir::OpBuilder &builder, unsigned indexBits) {
  ::llvm::SmallVector<::mlir::arith::CmpIOp, 8> comparisons;
  graph.getBody().walk([&](::mlir::arith::CmpIOp cmp) {
    if (::llvm::isa<::mlir::IntegerType>(cmp.getLhs().getType()) &&
        ::llvm::isa<::mlir::IntegerType>(cmp.getRhs().getType()))
      comparisons.push_back(cmp);
  });
  bool changed = false;
  for (::mlir::arith::CmpIOp cmp : comparisons)
    if (cmp.getOperation()->getBlock())
      changed |= rewriteOneIndexDomainCmp(cmp, builder, indexBits);
  return changed;
}

bool rewriteAddressIndexCasts(::dataflow::GraphOp graph,
                              ::mlir::OpBuilder &builder, unsigned indexBits) {
  ::llvm::SmallVector<::mlir::arith::IndexCastOp, 8> casts;
  graph.getBody().walk([&](::mlir::arith::IndexCastOp cast) {
    if (::llvm::isa<::mlir::IndexType>(cast.getType()) &&
        ::llvm::isa<::mlir::IntegerType>(cast.getIn().getType()))
      casts.push_back(cast);
  });
  bool changed = false;
  for (::mlir::arith::IndexCastOp cast : casts) {
    if (!cast.getOperation()->getBlock())
      continue;
    ::mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(cast);
    IndexMaterializationTransaction transaction(cast);
    ::llvm::DenseMap<::mlir::Value, ::mlir::Value> cache;
    ::mlir::Value indexValue =
        materializeIndexDomainValue(cast.getIn(), builder, cache, indexBits);
    if (!indexValue || indexValue == cast.getResult())
      continue;
    transaction.commit();
    cast.replaceAllUsesWith(indexValue);
    cast.erase();
    changed = true;
  }
  return changed;
}

void eraseDeadIndexArithmetic(::dataflow::GraphOp graph) {
  bool changed = true;
  while (changed) {
    changed = false;
    ::llvm::SmallVector<::mlir::Operation *, 8> deadOps;
    graph.getBody().walk([&](::mlir::Operation *op) {
      if (!op->use_empty())
        return;
      if (::llvm::isa<::mlir::arith::IndexCastOp, ::mlir::arith::AddIOp,
                      ::mlir::arith::SubIOp, ::mlir::arith::MulIOp,
                      ::mlir::arith::ShLIOp, ::mlir::arith::ShRUIOp,
                      ::mlir::arith::AndIOp, ::mlir::arith::ExtUIOp,
                      ::dataflow::ConstantOp, ::dataflow::InvariantOp,
                      ::dataflow::GateOp>(op))
        deadOps.push_back(op);
    });
    for (::mlir::Operation *op : deadOps) {
      op->erase();
      changed = true;
    }
  }
}

} // namespace

namespace loom {
namespace lowering {

void lowerGraphIndexDomains(::dataflow::GraphOp graph, unsigned indexBits) {
  ::mlir::OpBuilder builder(graph.getContext());
  bool changed = rewriteIndexDomainCmps(graph, builder, indexBits);
  changed |= rewriteAddressIndexCasts(graph, builder, indexBits);
  if (changed)
    eraseDeadIndexArithmetic(graph);
}

} // namespace lowering
} // namespace loom
