#include "Simulator/CGRASimulator.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

static llvm::cl::opt<std::string>
    dfgReportPath("dfg-report", llvm::cl::desc("DFG simulation report JSON"),
                  llvm::cl::Required);

static llvm::cl::opt<std::string>
    mappingArtifactPath("mapping-artifact",
                        llvm::cl::desc("PnR mapping artifact JSON"),
                        llvm::cl::Required);

static llvm::cl::opt<std::string>
    hardwareMlirPath("hardware-mlir",
                     llvm::cl::desc("Fabric ADG or module MLIR input"),
                     llvm::cl::init(""));

static llvm::cl::opt<std::string>
    outputPath("output", llvm::cl::desc("CGRA simulation report JSON"),
               llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "loom-cgra-sim: estimate hardware-aware cycles from DFG and mapping "
      "evidence\n");

  loom::sim::CGRASimOptions options;
  options.dfgReportPath = dfgReportPath;
  options.mappingArtifactPath = mappingArtifactPath;
  options.hardwareMlirPath = hardwareMlirPath;

  llvm::Expected<loom::sim::CGRASimReport> reportOrErr =
      loom::sim::runCGRASimulation(options);
  if (!reportOrErr) {
    llvm::errs() << "error: " << llvm::toString(reportOrErr.takeError())
                 << "\n";
    return 1;
  }
  if (llvm::Error err =
          loom::sim::writeCGRASimReportJson(outputPath, *reportOrErr)) {
    llvm::errs() << "error: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }
  return 0;
}
