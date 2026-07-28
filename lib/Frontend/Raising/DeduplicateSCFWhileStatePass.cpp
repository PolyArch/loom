// Remove only provably equivalent state lanes from scf.while.
//
// CFG structuring can preserve two PHIs for the same recurrence when one SSA
// name advances the value and another name carries an alias used by the body.
// If both lanes start from the same value, publish the same condition value,
// and feed that value back through their respective identity yields, they are
// the same state for every iteration and on loop exit. Keeping both lanes
// obscures pointer-induction and dependence analysis without preserving any
// source distinction.

#include "Frontend/Raising/Passes.h"

#include "CallableRegions.h"
#include "ExactRewrite.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace {

struct EquivalentStateClasses final {
  ::llvm::SmallVector<unsigned, 8> representative;
  ::llvm::SmallVector<unsigned, 8> kept;
  ::llvm::SmallVector<unsigned, 8> newOrdinal;
};

std::optional<EquivalentStateClasses>
deriveEquivalentStateClasses(::mlir::scf::WhileOp loop) {
  ::mlir::Block *before = loop.getBeforeBody();
  ::mlir::Block *after = loop.getAfterBody();
  ::mlir::scf::ConditionOp condition = loop.getConditionOp();
  ::mlir::scf::YieldOp yield = loop.getYieldOp();
  const unsigned count = loop.getInits().size();
  if (count < 2 || before->getNumArguments() != count ||
      condition.getArgs().size() != count ||
      after->getNumArguments() != count || yield.getResults().size() != count ||
      loop.getNumResults() != count)
    return std::nullopt;

  EquivalentStateClasses classes;
  classes.representative.resize(count);
  bool changed = false;
  for (unsigned lane = 0; lane < count; ++lane) {
    classes.representative[lane] = lane;
    if (yield.getResults()[lane] != after->getArgument(lane))
      continue;
    for (unsigned candidate = 0; candidate < lane; ++candidate) {
      const unsigned representative = classes.representative[candidate];
      if (yield.getResults()[representative] !=
              after->getArgument(representative) ||
          loop.getInits()[representative] != loop.getInits()[lane] ||
          condition.getArgs()[representative] != condition.getArgs()[lane] ||
          before->getArgument(representative).getType() !=
              before->getArgument(lane).getType() ||
          after->getArgument(representative).getType() !=
              after->getArgument(lane).getType() ||
          loop.getResult(representative).getType() !=
              loop.getResult(lane).getType())
        continue;
      classes.representative[lane] = representative;
      changed = true;
      break;
    }
  }
  if (!changed)
    return std::nullopt;

  classes.newOrdinal.resize(count);
  for (unsigned lane = 0; lane < count; ++lane) {
    if (classes.representative[lane] != lane)
      continue;
    classes.newOrdinal[lane] = classes.kept.size();
    classes.kept.push_back(lane);
  }
  for (unsigned lane = 0; lane < count; ++lane)
    classes.newOrdinal[lane] = classes.newOrdinal[classes.representative[lane]];
  return classes;
}

struct DeduplicateSCFWhileState final
    : public ::mlir::OpRewritePattern<::mlir::scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::scf::WhileOp loop,
                  ::mlir::PatternRewriter &rewriter) const override {
    std::optional<EquivalentStateClasses> classes =
        deriveEquivalentStateClasses(loop);
    if (!classes)
      return ::mlir::failure();

    ::llvm::SmallVector<::mlir::Value, 8> inits;
    ::llvm::SmallVector<::mlir::Type, 8> resultTypes;
    inits.reserve(classes->kept.size());
    resultTypes.reserve(classes->kept.size());
    for (unsigned lane : classes->kept) {
      inits.push_back(loop.getInits()[lane]);
      resultTypes.push_back(loop.getResult(lane).getType());
    }

    auto buildBefore = [&](::mlir::OpBuilder &builder, ::mlir::Location,
                           ::mlir::ValueRange arguments) {
      ::mlir::IRMapping mapping;
      for (unsigned lane = 0; lane < classes->representative.size(); ++lane)
        mapping.map(loop.getBeforeBody()->getArgument(lane),
                    arguments[classes->newOrdinal[lane]]);
      for (::mlir::Operation &operation :
           loop.getBeforeBody()->without_terminator())
        builder.clone(operation, mapping);

      ::llvm::SmallVector<::mlir::Value, 8> next;
      next.reserve(classes->kept.size());
      for (unsigned lane : classes->kept)
        next.push_back(
            mapping.lookupOrDefault(loop.getConditionOp().getArgs()[lane]));
      ::mlir::scf::ConditionOp::create(
          builder, loop.getConditionOp().getLoc(),
          mapping.lookupOrDefault(loop.getConditionOp().getCondition()), next);
    };

    auto buildAfter = [&](::mlir::OpBuilder &builder, ::mlir::Location,
                          ::mlir::ValueRange arguments) {
      ::mlir::IRMapping mapping;
      for (unsigned lane = 0; lane < classes->representative.size(); ++lane)
        mapping.map(loop.getAfterBody()->getArgument(lane),
                    arguments[classes->newOrdinal[lane]]);
      for (::mlir::Operation &operation :
           loop.getAfterBody()->without_terminator())
        builder.clone(operation, mapping);

      ::llvm::SmallVector<::mlir::Value, 8> next;
      next.reserve(classes->kept.size());
      for (unsigned lane : classes->kept)
        next.push_back(
            mapping.lookupOrDefault(loop.getYieldOp().getResults()[lane]));
      ::mlir::scf::YieldOp::create(builder, loop.getYieldOp().getLoc(), next);
    };

    auto replacement = ::mlir::scf::WhileOp::create(
        rewriter, loop.getLoc(), resultTypes, inits, buildBefore, buildAfter);
    replacement->setDiscardableAttrs(loop->getDiscardableAttrDictionary());

    ::llvm::SmallVector<::mlir::Value, 8> results;
    results.reserve(classes->representative.size());
    for (unsigned lane = 0; lane < classes->representative.size(); ++lane)
      results.push_back(replacement.getResult(classes->newOrdinal[lane]));
    rewriter.replaceOp(loop, results);
    return ::mlir::success();
  }
};

struct DeduplicateSCFWhileStatePass final
    : public ::mlir::PassWrapper<DeduplicateSCFWhileStatePass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeduplicateSCFWhileStatePass)

  ::llvm::StringRef getArgument() const final {
    return "loom-deduplicate-scf-while-state";
  }
  ::llvm::StringRef getDescription() const final {
    return "Remove exactly equivalent scf.while carried-state lanes";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    ::mlir::RewritePatternSet patterns(&getContext());
    patterns.add<DeduplicateSCFWhileState>(&getContext());
    ::mlir::FrozenRewritePatternSet frozen(std::move(patterns));
    (void)loom::raising::forEachCallableRegion(
        getOperation(), [&](::mlir::Region &region) {
          loom::raising::applyExactPatternsOnce(region, frozen);
          return ::mlir::success();
        });
  }
};

} // namespace

namespace loom {
namespace raising {

std::unique_ptr<::mlir::Pass> createDeduplicateSCFWhileStatePass() {
  return std::make_unique<DeduplicateSCFWhileStatePass>();
}

void registerDeduplicateSCFWhileStatePass() {
  static bool once = []() {
    ::mlir::PassRegistration<DeduplicateSCFWhileStatePass>();
    return true;
  }();
  (void)once;
}

} // namespace raising
} // namespace loom
