#include "Simulator/CycleSummary.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

static llvm::cl::opt<std::string>
    primitiveCoveragePath("primitive-coverage",
                          llvm::cl::desc("dataflow primitive coverage CSV"),
                          llvm::cl::Required);

static llvm::cl::opt<std::string>
    outputPath("output", llvm::cl::desc("sim cycle summary CSV"),
               llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "loom-sim-cycle-summary: emit simulator cycle summary diagnostics\n");

  loom::sim::CycleSummaryOptions options;

  auto rowsOrErr =
      loom::sim::summarizePrimitiveCoverage(primitiveCoveragePath, options);
  if (!rowsOrErr) {
    llvm::errs() << "error: " << llvm::toString(rowsOrErr.takeError())
                 << "\n";
    return 1;
  }
  if (llvm::Error err =
          loom::sim::writeCycleSummaryCsv(outputPath, *rowsOrErr)) {
    llvm::errs() << "error: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }
  return 0;
}
