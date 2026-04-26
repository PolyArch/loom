#include "Fabric/Tech/Partitioner/CandidateCache.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/TemplateLibrary.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"

#include <algorithm>

namespace fabric {

namespace {

// One entry of the worklist seen by worker threads: the op pointer plus the
// program-position index used to write back into a pre-allocated slot. We
// also pre-resolve the op name so the worker does not need to call into MLIR
// internals beyond the immutable template library lookup.
struct WorkItem {
  ::mlir::Operation *op;
  ::llvm::StringRef name;
  unsigned slot;
};

} // namespace

CandidateCache CandidateCache::build(::dataflow::GraphOp graph,
                                     const TemplateLibrary &lib,
                                     unsigned threadCount) {
  CandidateCache out;

  // Walk the body in program order, materializing one work item per
  // non-terminator op. The slot index doubles as the eventual program-order
  // index in `out.cache`.
  ::llvm::SmallVector<WorkItem> work;
  ::mlir::Block &body = graph.getBody().front();
  for (::mlir::Operation &op : body) {
    if (::mlir::isa<::dataflow::YieldOp>(op))
      continue;
    WorkItem wi;
    wi.op = &op;
    wi.name = op.getName().getStringRef();
    wi.slot = static_cast<unsigned>(work.size());
    work.push_back(wi);
  }

  out.cache.resize(work.size());
  for (const WorkItem &wi : work) {
    out.cache[wi.slot].root = wi.op;
    out.opIndex[wi.op] = wi.slot;
  }

  if (work.empty())
    return out;

  // Per-op match closure. Pure read of the template library; safe to run
  // concurrently as long as each invocation writes only to its own slot.
  auto matchOne = [&](const WorkItem &wi) {
    auto &slot = out.cache[wi.slot].templateIds;
    if (!::fabric::isFabricOpSupported(wi.name))
      return;
    ::llvm::ArrayRef<unsigned> ids = lib.templatesByRootOp(wi.name);
    slot.reserve(ids.size());
    for (unsigned id : ids)
      slot.push_back(id);
    std::sort(slot.begin(), slot.end());
  };

  // Decide the effective worker count. A threshold of 2+ items per worker
  // is enforced so that small graphs do not pay thread-pool startup cost.
  unsigned hw = ::llvm::hardware_concurrency().compute_thread_count();
  if (hw == 0)
    hw = 1;
  unsigned requested = threadCount == 0 ? hw : threadCount;
  if (requested > work.size())
    requested = static_cast<unsigned>(work.size());
  if (requested == 0)
    requested = 1;

  if (requested <= 1) {
    for (const WorkItem &wi : work)
      matchOne(wi);
  } else {
    ::llvm::StdThreadPool pool(::llvm::hardware_concurrency(requested));
    // Chunk the worklist roughly evenly so each task amortizes the
    // synchronization overhead. The chunk boundaries are deterministic
    // because they are derived from the worklist size and `requested`.
    const unsigned numTasks = requested;
    const size_t total = work.size();
    const size_t chunk = (total + numTasks - 1) / numTasks;
    for (unsigned t = 0; t < numTasks; ++t) {
      size_t lo = static_cast<size_t>(t) * chunk;
      size_t hi = std::min(lo + chunk, total);
      if (lo >= hi)
        break;
      pool.async([&, lo, hi]() {
        for (size_t i = lo; i < hi; ++i)
          matchOne(work[i]);
      });
    }
    pool.wait();
  }

  return out;
}

::llvm::ArrayRef<unsigned>
CandidateCache::templatesForOp(::mlir::Operation *op) const {
  auto it = opIndex.find(op);
  if (it == opIndex.end())
    return {};
  return cache[it->second].templateIds;
}

} // namespace fabric
