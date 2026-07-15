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

#include <cstdint>

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
           "subgraph with its selected encoding and mapping witness. Patterns "
           "that match no FU are tagged with 'loom.unmatched'.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::dataflow::DataflowDialect,
                    ::mlir::arith::ArithDialect,
                    ::mlir::math::MathDialect>();
  }

  // Identify the symbolic name of an FU. After migration, every fabric.fu
  // is nested inside a fabric.pe inside a fabric.module; we use the
  // enclosing fabric.module's `sym_name` as the stable identifier.
  // The legacy func::FuncOp lookup is retained as a fallback so that any
  // residual non-migrated input still produces a usable name.
  // The "indexInParent" still keys off the FU's immediate parent op
  // (the fabric.pe after migration), giving each FU within the same
  // PE a distinct index.
  static std::string nameForFu(::fabric::FuOp fu, unsigned indexInParent) {
    std::string s;
    ::llvm::raw_string_ostream os(s);
    if (auto mod = fu->getParentOfType<::fabric::ModuleOp>())
      os << "@" << mod.getSymName();
    else if (auto func = fu->getParentOfType<::mlir::func::FuncOp>())
      os << "@" << func.getName();
    else
      os << "<anon>";
    os << "#" << indexInParent;
    return s;
  }

  static ::mlir::DenseI32ArrayAttr portCorrespondenceAttr(
      ::mlir::MLIRContext *ctx,
      ::llvm::ArrayRef<std::pair<unsigned, unsigned>> correspondence) {
    ::llvm::SmallVector<int32_t, 8> values;
    values.reserve(correspondence.size() * 2);
    for (auto [softwarePort, fuPort] : correspondence) {
      values.push_back(static_cast<int32_t>(softwarePort));
      values.push_back(static_cast<int32_t>(fuPort));
    }
    return ::mlir::DenseI32ArrayAttr::get(ctx, values);
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

    for (::dataflow::SubgraphOp pat : patterns) {
      // Strip prior annotations, if any, before re-running.
      pat->removeAttr("loom.matched_fu");
      pat->removeAttr("loom.matched_encoding");
      pat->removeAttr("loom.actor_to_fabric_op");
      pat->removeAttr("loom.input_port_correspondence");
      pat->removeAttr("loom.output_port_correspondence");
      pat->removeAttr("loom.unmatched");

      bool found = false;
      for (auto [i, fu] : ::llvm::enumerate(fus)) {
        auto r = ::fabric::mapPatternToFu(pat, fu);
        if (r.matched) {
          ::llvm::SmallVector<int32_t, 8> actorToFabricOp;
          for (unsigned resource : r.actorToFabricOp)
            actorToFabricOp.push_back(static_cast<int32_t>(resource));
          pat->setAttr("loom.actor_to_fabric_op",
                       ::mlir::DenseI32ArrayAttr::get(ctx, actorToFabricOp));
          pat->setAttr("loom.input_port_correspondence",
                       portCorrespondenceAttr(ctx, r.inputPorts));
          pat->setAttr("loom.matched_encoding",
                       ::mlir::IntegerAttr::get(
                           ::mlir::IntegerType::get(ctx, 64), r.encodingIndex));
          pat->setAttr(
              "loom.matched_fu",
              ::mlir::StringAttr::get(ctx, nameForFu(fu, fuIndexInParent[i])));
          pat->setAttr("loom.output_port_correspondence",
                       portCorrespondenceAttr(ctx, r.outputPorts));
          found = true;
          break;
        }
      }
      if (!found)
        pat->setAttr("loom.unmatched", ::mlir::UnitAttr::get(ctx));
    }
  }
};

} // namespace

namespace fabric {

std::unique_ptr<::mlir::Pass> createMapSubgraphToFusPass() {
  return std::make_unique<MapSubgraphToFusPass>();
}

} // namespace fabric
