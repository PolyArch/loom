#include "Fabric/Tech/Passes.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/SubgraphMatcher.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace {

struct MapSubgraphToFusPass
    : public ::mlir::PassWrapper<MapSubgraphToFusPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MapSubgraphToFusPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-map-subgraph-to-fus";
  }
  ::llvm::StringRef getDescription() const final {
    return "For each dataflow.subgraph annotated 'loom.is_pattern', try to "
           "match it against every fabric.fu in the module and annotate the "
           "subgraph with the resulting sw_configs. Patterns that match "
           "no FU are tagged with 'loom.unmatched'.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::dataflow::DataflowDialect,
                    ::mlir::arith::ArithDialect,
                    ::mlir::math::MathDialect>();
  }

  // Identify the symbolic name of an FU. We use the enclosing func.func
  // name and the FU's index within that function so users can disambiguate
  // multi-FU functions.
  static std::string nameForFu(::fabric::FuOp fu, unsigned indexInParent) {
    std::string s;
    ::llvm::raw_string_ostream os(s);
    if (auto func = fu->getParentOfType<::mlir::func::FuncOp>())
      os << "@" << func.getName();
    else
      os << "<anon>";
    os << "#" << indexInParent;
    return s;
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    auto *ctx = &getContext();

    ::llvm::SmallVector<::fabric::FuOp> fus;
    ::llvm::SmallVector<unsigned> fuIndexInParent;
    {
      ::llvm::DenseMap<::mlir::Operation *, unsigned> perFunc;
      module.walk([&](::fabric::FuOp fu) {
        auto *parent = fu->getParentOp();
        unsigned &n = perFunc[parent];
        fus.push_back(fu);
        fuIndexInParent.push_back(n++);
      });
    }

    ::llvm::SmallVector<::dataflow::SubgraphOp> patterns;
    module.walk([&](::dataflow::SubgraphOp sg) {
      if (sg->hasAttr("loom.is_pattern"))
        patterns.push_back(sg);
    });

    if (patterns.empty())
      return;

    auto tempModule = ::mlir::ModuleOp::create(::mlir::UnknownLoc::get(ctx));

    for (::dataflow::SubgraphOp pat : patterns) {
      // Strip prior annotations, if any, before re-running.
      pat->removeAttr("loom.matched_fu");
      pat->removeAttr("loom.match_config");
      pat->removeAttr("loom.unmatched");

      bool found = false;
      for (auto [i, fu] : ::llvm::enumerate(fus)) {
        // Clear the scratch module body before each FU query.
        tempModule.getBody()->clear();

        auto r = ::fabric::mapPatternToFu(pat, fu, tempModule);
        if (r.matched) {
          pat->setAttr(
              "loom.matched_fu",
              ::mlir::StringAttr::get(ctx, nameForFu(fu, fuIndexInParent[i])));
          pat->setAttr("loom.match_config",
                       ::mlir::StringAttr::get(ctx, r.configDescription));
          found = true;
          break;
        }
      }
      if (!found)
        pat->setAttr("loom.unmatched", ::mlir::UnitAttr::get(ctx));
      tempModule.getBody()->clear();
    }

    tempModule.erase();
  }
};

} // namespace

namespace fabric {

std::unique_ptr<::mlir::Pass> createMapSubgraphToFusPass() {
  return std::make_unique<MapSubgraphToFusPass>();
}

} // namespace fabric
