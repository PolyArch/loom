#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_PARALLEL_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_PARALLEL_H

// Thin wrapper over llvm::ThreadPool used by the
// loom-synthesize-configured-functions pass for cross-group, coverage-verifier,
// and strategy-internal parallelism.
//
// Contract for callers:
//
//   * Closures must capture by value. Callers may pass `OwningOpRef`s into
//     a worker only by moving them into the closure's captured state; they
//     must not reference stack-local MLIR objects whose lifetime ends before
//     the worker runs.
//   * MLIR mutation is forbidden inside `parallelMap` / `parallelFor`
//     closures. Each worker builds its candidate FU in a thread-local
//     scratch `MLIRContext` / `OwningOpRef`; the pass main thread splices
//     workers' detached results into the user's `ModuleOp` serially via
//     `runSerialInOrder`. See the spec section "Parallelism plan" /
//     "MLIR mutation is never parallel" for the full rule.
//   * `workers == 0` means `std::thread::hardware_concurrency()` (auto).
//     `workers == 1` bypasses the underlying thread pool and runs each
//     closure inline on the calling thread; this keeps single-thread
//     test invocations deterministic and avoids spawning a worker we
//     would never use.
//   * Result vectors returned by `parallelMap` always preserve input
//     index order regardless of completion order.
//
// Template definitions live in this header (templates can't sit in a
// translation unit without explicit instantiation lists). Keep them small.

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"

#include <cstddef>
#include <memory>
#include <type_traits>

namespace loom::fabric::tech {

class WorkerPool {
public:
  // workers == 0 means std::thread::hardware_concurrency().
  explicit WorkerPool(unsigned workers = 0);
  ~WorkerPool();

  WorkerPool(const WorkerPool &) = delete;
  WorkerPool &operator=(const WorkerPool &) = delete;

  // Returns the actual number of workers (after auto-detect resolves 0).
  unsigned numWorkers() const { return resolvedWorkers; }

  // Run `fn` once per element of `inputs`, in parallel. The returned
  // vector preserves input index order regardless of completion order.
  // The closure must be thread-safe and capture by value (no MLIR
  // mutation). When `numWorkers() == 1`, runs inline on the calling
  // thread. `R` must be default-constructible (slots are pre-allocated
  // so each worker can write its result by index).
  template <class T, class R>
  ::llvm::SmallVector<R, 8> parallelMap(::llvm::ArrayRef<T> inputs,
                                        ::llvm::function_ref<R(const T &)> fn) {
    static_assert(std::is_default_constructible_v<R>,
                  "WorkerPool::parallelMap requires default-constructible R");
    ::llvm::SmallVector<R, 8> results;
    results.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
      results.emplace_back();
    if (resolvedWorkers <= 1 || !pool) {
      for (size_t i = 0; i < inputs.size(); ++i)
        results[i] = fn(inputs[i]);
      return results;
    }
    {
      ::llvm::ThreadPoolTaskGroup group(*pool);
      for (size_t i = 0; i < inputs.size(); ++i) {
        const T *itemPtr = &inputs[i];
        R *slotPtr = &results[i];
        group.async([slotPtr, itemPtr, fn]() { *slotPtr = fn(*itemPtr); });
      }
    }
    return results;
  }

  // Convenience: void-returning variant. Waits for all to finish. Useful
  // for fire-and-forget side-effect tasks that internally collect results
  // via shared mutable state protected by the caller's own synchronization.
  void parallelFor(::llvm::ArrayRef<size_t> indices,
                   ::llvm::function_ref<void(size_t)> fn);

  // Convenience: index-stable serial fold. Just `for (i ... ) fn(i);` but
  // with a clear name expressing intent. Used at MLIR-mutation boundaries
  // where the spec's "MLIR mutation is never parallel" rule forbids
  // parallelMap.
  template <class T>
  void runSerialInOrder(::llvm::ArrayRef<T> inputs,
                        ::llvm::function_ref<void(const T &)> fn) {
    for (const T &item : inputs)
      fn(item);
  }

private:
  // null when resolvedWorkers == 1 (inline-execution fast path).
  std::unique_ptr<::llvm::DefaultThreadPool> pool;
  unsigned resolvedWorkers;
};

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_PARALLEL_H
