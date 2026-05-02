#include "Fabric/Tech/Synthesizer/CoverageVerifier.h"

#include "Fabric/Tech/Synthesizer/Parallel.h"
#include "Fabric/Tech/SubgraphEnumerator.h"
#include "Fabric/Tech/SubgraphMatcher.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <optional>

namespace loom::fabric::tech {

CoverageVerifier::CoverageVerifier(const ::loom::SynthConfig &cfg)
    : parallelMatch(cfg.coverageVerifierParallelMatch),
      parallelismWorkers(cfg.parallelismWorkers) {}

CoverageReport
CoverageVerifier::verify(::fabric::FuOp fu,
                         ::llvm::ArrayRef<::dataflow::SubgraphOp> inputs) {
  CoverageReport report;
  // Pre-size matchIndex so each parallel worker writes a distinct slot.
  report.matchIndex.assign(inputs.size(), std::nullopt);

  if (!fu)
    return report;

  // Locate the wrapper func.func that lexically contains `fu`. The
  // enumerator emits its candidates as siblings of this wrapper in the
  // FU's enclosing module, which is precisely why the verifier must
  // route them into a scratch module instead.
  auto userWrapper = fu->getParentOfType<::mlir::func::FuncOp>();
  if (!userWrapper)
    return report;

  ::mlir::MLIRContext *ctx = fu.getContext();

  // RAII-managed scratch module. Its destructor erases all nested ops,
  // so any candidate func.func appended by the enumerator goes away on
  // return without ever touching the user's module.
  ::mlir::OwningOpRef<::mlir::ModuleOp> scratch(
      ::mlir::ModuleOp::create(::mlir::UnknownLoc::get(ctx)));

  // Clone the user's wrapper func.func (with its FU body, since
  // fabric.fu is IsolatedFromAbove and so the wrapper carries
  // everything the enumerator needs) into the scratch module. We use
  // an OpBuilder anchored at the scratch module's body so the cloned
  // op is appended in-order.
  ::mlir::OpBuilder builder(scratch->getBodyRegion());
  ::mlir::Operation *clonedWrapperOp = builder.clone(*userWrapper);

  // Find the cloned fabric.fu inside that wrapper. With one-FU-per-
  // wrapper (the canonical synthesis layout) this is just the first
  // fabric.fu inside the cloned op; if a wrapper held multiple FUs we
  // still want the one that mirrors the input `fu`'s position, but the
  // verifier contract takes one `fu` so any ambiguity here would be a
  // caller bug. We pick the first FuOp seen by walk; this matches the
  // single-FU contract.
  ::fabric::FuOp clonedFu;
  clonedWrapperOp->walk([&](::fabric::FuOp candidate) {
    if (!clonedFu)
      clonedFu = candidate;
  });
  if (!clonedFu)
    return report;

  // Run the enumerator. The scratch module is the destination for the
  // appended candidate `func.func`s; the user's module is untouched.
  ::llvm::StringRef unsupported;
  auto candidates = ::fabric::enumerateFuSubgraphs(
      clonedFu, scratch.get(), "candidate", &unsupported);
  // If the enumerator skipped the FU because it contains an
  // unsupported op, no candidate matches anything; leave matchIndex
  // all-nullopt. Caller can read `report.allCovered() == false` for
  // any non-empty input list.

  // Collect candidate subgraphs in stable enumerator order. We index
  // into `candidates` directly rather than re-walking the scratch
  // module: the enumerator's documented contract is "appended in
  // monotonically increasing idx order", so `candidates[i].subgraph`
  // is the i'th match candidate exactly.
  ::llvm::SmallVector<::dataflow::SubgraphOp, 8> candidateSubgraphs;
  candidateSubgraphs.reserve(candidates.size());
  for (auto &c : candidates)
    candidateSubgraphs.push_back(c.subgraph);

  // Per-input matching loop. Reads are safe across threads (every
  // worker only calls `subgraphsIsomorphic` which doesn't mutate
  // either operand); writes target distinct slots in `matchIndex`.
  auto matchOne = [&](size_t i) {
    ::dataflow::SubgraphOp pat = inputs[i];
    if (!pat)
      return;
    for (size_t j = 0; j < candidateSubgraphs.size(); ++j) {
      if (::fabric::subgraphsIsomorphic(pat, candidateSubgraphs[j])) {
        report.matchIndex[i] = j;
        return;
      }
    }
  };

  if (parallelMatch && inputs.size() > 1) {
    WorkerPool pool(parallelismWorkers);
    ::llvm::SmallVector<size_t, 8> indices;
    indices.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
      indices.push_back(i);
    pool.parallelFor(indices, matchOne);
  } else {
    for (size_t i = 0; i < inputs.size(); ++i)
      matchOne(i);
  }

  // `scratch`'s destructor runs on return, deterministically dropping
  // every cloned op and every appended candidate.
  return report;
}

} // namespace loom::fabric::tech
