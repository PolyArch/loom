// Expand SpatialCore-owned `memref.copy` into nested structured scalar
// load/store loops over the shared logical index domain. Each endpoint keeps
// its own exact layout for the later scalar-address lowering.

#include "Frontend/Lowering/Passes.h"

#include "RankedMemRefLowering.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>

namespace {

constexpr std::int64_t kLoopLowerBound = 0;
constexpr std::int64_t kLoopStep = 1;

void expandCopy(::mlir::memref::CopyOp copy) {
  auto type = ::llvm::cast<::mlir::MemRefType>(copy.getSource().getType());
  ::mlir::OpBuilder builder(copy);
  ::mlir::Location loc = copy.getLoc();
  ::llvm::SmallVector<::mlir::Value, 4> lowerBounds;
  ::llvm::SmallVector<::mlir::Value, 4> upperBounds;
  ::llvm::SmallVector<::mlir::Value, 4> steps;
  for (std::int64_t extent : type.getShape()) {
    lowerBounds.push_back(
        ::mlir::arith::ConstantIndexOp::create(builder, loc, kLoopLowerBound));
    upperBounds.push_back(
        ::mlir::arith::ConstantIndexOp::create(builder, loc, extent));
    steps.push_back(
        ::mlir::arith::ConstantIndexOp::create(builder, loc, kLoopStep));
  }
  ::mlir::scf::buildLoopNest(
      builder, loc, lowerBounds, upperBounds, steps,
      [&](::mlir::OpBuilder &body, ::mlir::Location bodyLoc,
          ::mlir::ValueRange indices) {
        ::mlir::Value element = ::mlir::memref::LoadOp::create(
            body, bodyLoc, copy.getSource(), indices);
        ::mlir::memref::StoreOp::create(body, bodyLoc, element,
                                        copy.getTarget(), indices);
      });
  copy.erase();
}

struct ExpandGraphMemrefCopyPass
    : public ::mlir::PassWrapper<ExpandGraphMemrefCopyPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandGraphMemrefCopyPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-expand-graph-memref-copy";
  }
  ::llvm::StringRef getDescription() const final {
    return "Expand memref.copy inside dataflow.graph bodies into a structured "
           "memref.load/memref.store element loop.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::memref::MemRefDialect,
                    ::mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    ::llvm::SmallVector<::mlir::memref::CopyOp, 8> copiesToExpand;
    for (auto graph : getOperation().getOps<::dataflow::GraphOp>()) {
      if (graph.isExternal())
        continue;
      ::llvm::SmallVector<::mlir::memref::CopyOp, 4> copies;
      graph.getBody().walk(
          [&](::mlir::memref::CopyOp copy) { copies.push_back(copy); });
      if (copies.empty())
        continue;

      ::llvm::Expected<unsigned> indexBits = ::loom::getIndexBitWidth(graph);
      if (!indexBits) {
        graph.emitError("loom-expand-graph-memref-copy: ")
            << ::llvm::toString(indexBits.takeError());
        signalPassFailure();
        return;
      }

      for (::mlir::memref::CopyOp copy : copies) {
        if (::mlir::failed(::loom::lowering::detail::checkRankedMemRefCopy(
                copy, *indexBits))) {
          signalPassFailure();
          return;
        }
        auto type =
            ::llvm::cast<::mlir::MemRefType>(copy.getSource().getType());
        if (type.getRank() > 0 && !::llvm::isIntN(*indexBits, kLoopStep)) {
          copy.emitOpError(
              "loom-expand-graph-memref-copy: cannot expand memref.copy "
              "into a structured load/store loop; bound ")
              << kLoopStep
              << " is not representable in the graph's resolved signed index "
                 "domain 'i"
              << *indexBits << "'";
          signalPassFailure();
          return;
        }
      }
      for (::mlir::memref::CopyOp copy : copies)
        copiesToExpand.push_back(copy);
    }
    for (::mlir::memref::CopyOp copy : copiesToExpand)
      expandCopy(copy);
  }
};

} // namespace

namespace loom {
namespace lowering {

std::unique_ptr<::mlir::Pass> createExpandGraphMemrefCopyPass() {
  return std::make_unique<ExpandGraphMemrefCopyPass>();
}

void registerExpandGraphMemrefCopyPass() {
  static bool once = []() {
    ::mlir::PassRegistration<ExpandGraphMemrefCopyPass>();
    return true;
  }();
  (void)once;
}

} // namespace lowering
} // namespace loom
