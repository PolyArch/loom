#include "ExecutionMatrixTestSupport.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

int main(int argc, char **argv) {
  if (argc == 3 && llvm::StringRef(argv[1]) == "deterministic-system-replay") {
    loom::system_test::verifyDeterministicSystemReplay(argv[2]);
    return EXIT_SUCCESS;
  }
  if (argc != 2 && argc != 3) {
    llvm::errs() << "usage: " << argv[0]
                 << " <spatial-dfg|spatial-cgra|spatial-rtl|system-dfg|"
                    "system-cgra|system-rtl> [gem5-readiness]\n"
                 << "       " << argv[0]
                 << " deterministic-system-replay <gem5-readiness>\n";
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
