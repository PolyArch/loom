// Expand SpatialCore-owned `memref.copy` inside `dataflow.graph` bodies into a
// structured element loop of `memref.load` / `memref.store`.
//
// `memref.copy` carries no independent Dataflow software semantics, so the
// canonical program has no bulk-transfer actor for it. The expansion is purely
// mechanical: the resulting accesses are the ones the graph-memory owner
// already lowers to `dataflow.load` / `dataflow.store` plus the ctrl/done
// memory-event network, so element type, index space, layout and subview
// addressing, alias roots, and graph completion all keep their existing single
// owner. Loads and stores address the original source and target values, so a
// non-identity layout stays in the memref type and is applied by the access
// rather than re-derived here.
//
// Overlap: the operation documentation requires the same element type and shape
// and permits differing layouts, but says nothing about overlapping operands.
// The official lowering does. `MemRefCopyOpLowering` selects `llvm.memcpy`
// whenever both operands have a contiguous layout, deciding from the layout
// alone with no aliasing check, and `llvm.memcpy` requires operands that are
// equal or non-overlapping; that dispatch would be unsound if an overlapping
// copy were defined. This expansion does not depend on that reading either: the
// generic official path, the `memrefCopy` runtime helper, walks the logical
// index space in increasing order and applies each operand's own layout, which
// is exactly the order emitted here. The syntactic self-copy and empty-shape
// cases are erased by the upstream canonicalizer that runs before this pass.
//
// This pass runs on SpatialCore graph bodies only. An InstructionCore copy
// outside a `dataflow.graph` keeps its upstream semantics.
//
// Capability gate: the expansion materializes the loop bounds itself, and a
// structured `index` loop is lowered to a `dataflow.stream` over the index
// width resolved at that graph, compared with a signed predicate. That width
// comes from `loom::getIndexBitWidth`, the one resolution of the fact, read
// once per graph before any of its copies expand, so an unusable declaration is
// reported before the loops exist rather than diagnosed later by the
// graph-memory owner. A copy whose bounds do not fit that domain, or whose
// operands are not ranked, statically shaped and rank-one, fails here inside
// the publication transaction, so no partial module is published. This bounds
// what the expansion can emit today; it is not a semantic rule about
// `memref.copy`.

#include "Frontend/Lowering/Passes.h"

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

// The generated loop always counts up from zero by one, so these are both the
// emitted bounds and the values the capability gate checks.
constexpr std::int64_t kLoopLowerBound = 0;
constexpr std::int64_t kLoopStep = 1;

// `memref.copy` verifies same element type and same shape, so one ranked
// static extent describes both operands. `indexBits` is the index width the
// caller already resolved for the owning graph; the expansion never resolves it
// again.
::mlir::LogicalResult expandCopy(::mlir::memref::CopyOp copy,
                                 unsigned indexBits) {
  auto source =
      ::llvm::dyn_cast<::mlir::MemRefType>(copy.getSource().getType());
  auto target =
      ::llvm::dyn_cast<::mlir::MemRefType>(copy.getTarget().getType());
  if (!source || !target || source.getRank() != 1 || target.getRank() != 1 ||
      !source.hasStaticShape() || !target.hasStaticShape())
    return copy.emitOpError(
        "loom-expand-graph-memref-copy: cannot expand memref.copy into a "
        "structured load/store loop; source and target must be ranked, "
        "statically shaped, rank-one memrefs");

  std::int64_t extent = source.getDimSize(0);
  for (std::int64_t bound : {kLoopLowerBound, kLoopStep, extent})
    if (!::llvm::isIntN(indexBits, bound))
      return copy.emitOpError(
                 "loom-expand-graph-memref-copy: cannot expand memref.copy "
                 "into a structured load/store loop; bound ")
             << bound << " is not representable in the graph's resolved signed "
             << "index domain 'i" << indexBits << "'";

  ::mlir::OpBuilder builder(copy);
  ::mlir::Location loc = copy.getLoc();
  ::mlir::Value lower =
      ::mlir::arith::ConstantIndexOp::create(builder, loc, kLoopLowerBound);
  ::mlir::Value upper =
      ::mlir::arith::ConstantIndexOp::create(builder, loc, extent);
  ::mlir::Value step =
      ::mlir::arith::ConstantIndexOp::create(builder, loc, kLoopStep);
  ::mlir::scf::ForOp::create(
      builder, loc, lower, upper, step, ::mlir::ValueRange{},
      [&](::mlir::OpBuilder &body, ::mlir::Location bodyLoc,
          ::mlir::Value index, ::mlir::ValueRange) {
        ::mlir::Value element = ::mlir::memref::LoadOp::create(
            body, bodyLoc, copy.getSource(), ::mlir::ValueRange{index});
        ::mlir::memref::StoreOp::create(body, bodyLoc, element,
                                        copy.getTarget(),
                                        ::mlir::ValueRange{index});
        ::mlir::scf::YieldOp::create(body, bodyLoc);
      });
  copy.erase();
  return ::mlir::success();
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
    for (auto graph : getOperation().getOps<::dataflow::GraphOp>()) {
      if (graph.isExternal())
        continue;
      ::llvm::SmallVector<::mlir::memref::CopyOp, 4> copies;
      graph.getBody().walk(
          [&](::mlir::memref::CopyOp copy) { copies.push_back(copy); });
      if (copies.empty())
        continue;

      // One resolution of this graph's index width, before any of its copies
      // expand, so an unusable declaration stops the transaction here.
      ::llvm::Expected<unsigned> indexBits = ::loom::getIndexBitWidth(graph);
      if (!indexBits) {
        graph.emitError("loom-expand-graph-memref-copy: ")
            << ::llvm::toString(indexBits.takeError());
        signalPassFailure();
        return;
      }

      for (::mlir::memref::CopyOp copy : copies)
        if (::mlir::failed(expandCopy(copy, *indexBits))) {
          signalPassFailure();
          return;
        }
    }
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
