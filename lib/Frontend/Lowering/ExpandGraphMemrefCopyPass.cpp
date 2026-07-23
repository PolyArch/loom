// Expand SpatialCore-owned `memref.copy` into nested structured scalar
// load/store loops. Only static identity layouts are admitted because layout
// interpretation has no independent owner in this pass.

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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace {

constexpr std::int64_t kLoopLowerBound = 0;
constexpr std::int64_t kLoopStep = 1;

void buildCopyNest(::mlir::OpBuilder &builder, ::mlir::Location loc,
                   ::mlir::memref::CopyOp copy,
                   ::llvm::ArrayRef<std::int64_t> shape, unsigned dimension,
                   ::llvm::SmallVectorImpl<::mlir::Value> &indices) {
  if (dimension == shape.size()) {
    ::mlir::Value element =
        ::mlir::memref::LoadOp::create(builder, loc, copy.getSource(), indices);
    ::mlir::memref::StoreOp::create(builder, loc, element, copy.getTarget(),
                                    indices);
    return;
  }

  ::mlir::Value lower =
      ::mlir::arith::ConstantIndexOp::create(builder, loc, kLoopLowerBound);
  ::mlir::Value upper =
      ::mlir::arith::ConstantIndexOp::create(builder, loc, shape[dimension]);
  ::mlir::Value step =
      ::mlir::arith::ConstantIndexOp::create(builder, loc, kLoopStep);
  ::mlir::scf::ForOp::create(
      builder, loc, lower, upper, step, ::mlir::ValueRange{},
      [&](::mlir::OpBuilder &body, ::mlir::Location bodyLoc,
          ::mlir::Value index, ::mlir::ValueRange) {
        indices.push_back(index);
        buildCopyNest(body, bodyLoc, copy, shape, dimension + 1, indices);
        indices.pop_back();
        ::mlir::scf::YieldOp::create(body, bodyLoc);
      });
}

void expandCopy(::mlir::memref::CopyOp copy) {
  auto type = ::llvm::cast<::mlir::MemRefType>(copy.getSource().getType());
  ::mlir::OpBuilder builder(copy);
  ::mlir::Location loc = copy.getLoc();
  ::llvm::SmallVector<::mlir::Value, 4> indices;
  buildCopyNest(builder, loc, copy, type.getShape(), 0, indices);
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

      for (::mlir::memref::CopyOp copy : copies)
        if (::mlir::failed(::loom::lowering::detail::checkRankedMemRefCopy(
                copy, *indexBits))) {
          signalPassFailure();
          return;
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
