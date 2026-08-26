#include "ExecutionMatrixTestSupport.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

int main(int argc, char **argv) {
  if (argc == 4 && llvm::StringRef(argv[1]) == "paired-spatial-cgra-batch") {
    std::uint64_t warmupRuns = 0;
    std::uint64_t measurementRuns = 0;
    if (llvm::StringRef(argv[2]).getAsInteger(10, warmupRuns) ||
        llvm::StringRef(argv[3]).getAsInteger(10, measurementRuns)) {
      llvm::errs() << "paired Spatial run counts must be unsigned integers\n";
      return EXIT_FAILURE;
    }
    loom::system_test::runPairedSpatialCgraBatch(warmupRuns, measurementRuns);
    return EXIT_SUCCESS;
  }
  if (argc == 3 && llvm::StringRef(argv[1]) == "deterministic-system-replay") {
    loom::system_test::verifyDeterministicSystemReplay(argv[2]);
    return EXIT_SUCCESS;
  }
  if (argc != 2 && argc != 3) {
    llvm::errs() << "usage: " << argv[0]
                 << " <spatial-dfg|spatial-cgra|spatial-rtl|system-dfg|"
                    "system-cgra|system-rtl|paired-spatial-cgra|"
                    "paired-system-cgra> [gem5-readiness]\n"
                 << "       " << argv[0]
                 << " deterministic-system-replay <gem5-readiness>\n";
    llvm::errs() << "       " << argv[0]
                 << " paired-spatial-cgra-batch <warmup-runs> "
                    "<measurement-runs>\n";
    return EXIT_FAILURE;
  }
  auto cell = loom::system_test::parseExecutionMatrixCell(argv[1]);
  if (!cell) {
    llvm::errs() << llvm::toString(cell.takeError()) << '\n';
    return EXIT_FAILURE;
  }
  loom::system_test::runExecutionMatrixCell(
      *cell, argc == 3 ? llvm::StringRef(argv[2]) : llvm::StringRef());
  return EXIT_SUCCESS;
}
