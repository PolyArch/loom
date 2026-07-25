#include "Fabric/Tech/Parallel.h"

#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"

#include <thread>

namespace loom::fabric::tech {

static unsigned resolveWorkerCount(unsigned requested) {
  if (requested != 0)
    return requested;
  unsigned hw = std::thread::hardware_concurrency();
  // Fallback when std::thread::hardware_concurrency() returns 0 (per the
  // standard, that means "not computable or not well defined").
  return hw == 0 ? 1u : hw;
}

WorkerPool::WorkerPool(unsigned workers)
    : pool(nullptr), resolvedWorkers(resolveWorkerCount(workers)) {
  if (resolvedWorkers > 1) {
    pool = std::make_unique<::llvm::DefaultThreadPool>(
        ::llvm::hardware_concurrency(resolvedWorkers));
  }
}

WorkerPool::~WorkerPool() = default;

void WorkerPool::parallelFor(::llvm::ArrayRef<size_t> indices,
                             ::llvm::function_ref<void(size_t)> fn) {
  if (resolvedWorkers <= 1 || !pool) {
    for (size_t idx : indices)
      fn(idx);
    return;
  }
  ::llvm::ThreadPoolTaskGroup group(*pool);
  for (size_t idx : indices)
    group.async([idx, fn]() { fn(idx); });
  // ~ThreadPoolTaskGroup waits for completion.
}

} // namespace loom::fabric::tech
