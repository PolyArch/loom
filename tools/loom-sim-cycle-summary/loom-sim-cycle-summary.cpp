#include "Simulator/CycleSummary.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

static llvm::cl::opt<std::string>
    primitiveCoveragePath("primitive-coverage",
                          llvm::cl::desc("dataflow primitive coverage CSV"));

static llvm::cl::list<std::string>
    dfgReportPaths("dfg-report", llvm::cl::desc("DFG simulation report JSON"),
                   llvm::cl::ZeroOrMore);

static llvm::cl::list<std::string>
    cgraReportPaths("cgra-report",
                    llvm::cl::desc("CGRA simulation report JSON"),
                    llvm::cl::ZeroOrMore);

static llvm::cl::opt<std::string>
    outputPath("output", llvm::cl::desc("sim cycle summary CSV"),
               llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "loom-sim-cycle-summary: emit simulator cycle summary diagnostics\n");

  loom::sim::CycleSummaryOptions options;

  llvm::Expected<llvm::SmallVector<loom::sim::CycleSummaryRow>> rowsOrErr =
      dfgReportPaths.empty()
          ? (primitiveCoveragePath.empty()
                 ? llvm::Expected<
                       llvm::SmallVector<loom::sim::CycleSummaryRow>>(
                       loom::sim::scaffoldCycleSummaryRows())
                 : loom::sim::summarizePrimitiveCoverage(primitiveCoveragePath,
                                                         options))
          : loom::sim::summarizeSimulationReports(dfgReportPaths,
                                                  cgraReportPaths);
  if (!rowsOrErr) {
    llvm::errs() << "error: " << llvm::toString(rowsOrErr.takeError()) << "\n";
    return 1;
  }
  if (llvm::Error err =
          loom::sim::writeCycleSummaryCsv(outputPath, *rowsOrErr)) {
    llvm::errs() << "error: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }
  return 0;
}
