// Tiny CLI used by lit tests to exercise the WorkerPool wrapper around
// llvm::ThreadPool.
//
// Usage:
//   loom-parallel-test --workers <N> --map <N>            # parallelMap squares
//   loom-parallel-test --workers <N> --for <N>            # parallelFor atomic
//   sum loom-parallel-test --workers <N> --serial <N>         #
//   runSerialInOrder loom-parallel-test --workers <N> --workers-effective  #
//   numWorkers()
//
// At least one of --map / --for / --serial / --workers-effective must be
// supplied; mode flags are evaluated in that order on a single invocation
// (they may be combined for stress tests, though lit tests use one mode at
// a time).

#include "Fabric/Tech/Parallel.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

static ::llvm::cl::opt<unsigned>
    workers("workers", ::llvm::cl::desc("Worker count (0 = auto-detect)"),
            ::llvm::cl::init(0));

static ::llvm::cl::opt<int> mapMode(
    "map", ::llvm::cl::desc("parallelMap [0..N): print squares in input order"),
    ::llvm::cl::init(-1));

static ::llvm::cl::opt<int>
    forMode("for",
            ::llvm::cl::desc("parallelFor [0..N): atomic-sum and print total"),
            ::llvm::cl::init(-1));

static ::llvm::cl::opt<int> serialMode(
    "serial", ::llvm::cl::desc("runSerialInOrder [0..N): print one line per i"),
    ::llvm::cl::init(-1));

static ::llvm::cl::opt<bool>
    workersEffective("workers-effective",
                     ::llvm::cl::desc("Print numWorkers() after auto-detect"),
                     ::llvm::cl::init(false));

namespace {

void runMap(::loom::fabric::tech::WorkerPool &pool, int n) {
  ::llvm::SmallVector<int64_t, 16> inputs;
  inputs.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    inputs.push_back(i);
  auto results = pool.parallelMap<int64_t, int64_t>(
      ::llvm::ArrayRef<int64_t>(inputs),
      ::llvm::function_ref<int64_t(const int64_t &)>(
          [](const int64_t &x) -> int64_t { return x * x; }));
  for (size_t i = 0; i < results.size(); ++i)
    ::llvm::outs() << "result[" << i << "]=" << results[i] << "\n";
}

void runFor(::loom::fabric::tech::WorkerPool &pool, int n) {
  ::llvm::SmallVector<size_t, 16> indices;
  indices.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    indices.push_back(static_cast<size_t>(i));
  std::atomic<int64_t> sum{0};
  pool.parallelFor(::llvm::ArrayRef<size_t>(indices),
                   ::llvm::function_ref<void(size_t)>([&sum](size_t idx) {
                     sum.fetch_add(static_cast<int64_t>(idx),
                                   std::memory_order_relaxed);
                   }));
  ::llvm::outs() << "sum=" << sum.load(std::memory_order_relaxed) << "\n";
}

void runSerial(::loom::fabric::tech::WorkerPool &pool, int n) {
  ::llvm::SmallVector<int64_t, 16> inputs;
  inputs.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    inputs.push_back(i);
  pool.runSerialInOrder<int64_t>(
      ::llvm::ArrayRef<int64_t>(inputs),
      ::llvm::function_ref<void(const int64_t &)>(
          [](const int64_t &x) { ::llvm::outs() << "serial[" << x << "]\n"; }));
}

} // namespace

int main(int argc, char **argv) {
  ::llvm::cl::ParseCommandLineOptions(
      argc, argv, "loom-parallel-test: drive WorkerPool from lit tests\n");
  ::loom::fabric::tech::WorkerPool pool(workers.getValue());
  bool didSomething = false;
  if (mapMode.getValue() >= 0) {
    runMap(pool, mapMode.getValue());
    didSomething = true;
  }
  if (forMode.getValue() >= 0) {
    runFor(pool, forMode.getValue());
    didSomething = true;
  }
  if (serialMode.getValue() >= 0) {
    runSerial(pool, serialMode.getValue());
    didSomething = true;
  }
  if (workersEffective.getValue()) {
    ::llvm::outs() << "workers=" << pool.numWorkers() << "\n";
    didSomething = true;
  }
  if (!didSomething) {
    ::llvm::errs() << "error: one of --map / --for / --serial / "
                      "--workers-effective is required\n";
    return 1;
  }
  return 0;
}
