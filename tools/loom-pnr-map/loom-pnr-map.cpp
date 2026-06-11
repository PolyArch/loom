#include "PnR/Mapping.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

static llvm::cl::opt<std::string>
    dfgMlirPath("dfg-mlir", llvm::cl::desc("dataflow MLIR input"),
                llvm::cl::Required);

static llvm::cl::opt<std::string>
    graphName("graph", llvm::cl::desc("dataflow.graph.func symbol"),
              llvm::cl::Required);

static llvm::cl::opt<std::string>
    hardwareMlirPath("hardware-mlir", llvm::cl::desc("fabric MLIR input"),
                     llvm::cl::Required);

static llvm::cl::opt<std::string>
    hardwareName("hardware", llvm::cl::desc("fabric.module symbol"),
                 llvm::cl::Required);

static llvm::cl::opt<std::string>
    workloadName("workload", llvm::cl::desc("workload name"),
                 llvm::cl::init(""));

static llvm::cl::opt<std::string>
    outputPath("output", llvm::cl::desc("mapping summary CSV"),
               llvm::cl::Required);

static llvm::cl::opt<std::string>
    artifactPath("artifact", llvm::cl::desc("mapping artifact JSON"),
                 llvm::cl::init(""));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "loom-pnr-map: map a dataflow graph onto a fabric hardware template\n");

  loom::pnr::MappingOptions options;
  options.dfgMlirPath = dfgMlirPath;
  options.graphName = graphName;
  options.hardwareMlirPath = hardwareMlirPath;
  options.hardwareName = hardwareName;
  options.workload = workloadName;

  llvm::Expected<loom::pnr::MappingSummary> summaryOrErr =
      loom::pnr::createMapping(options);
  if (!summaryOrErr) {
    llvm::errs() << "error: " << llvm::toString(summaryOrErr.takeError())
                 << "\n";
    return 1;
  }

  if (llvm::Error err = loom::pnr::writeMappingCsv(outputPath, {*summaryOrErr})) {
    llvm::errs() << "error: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }
  if (!artifactPath.empty()) {
    if (llvm::Error err =
            loom::pnr::writeMappingJson(artifactPath, *summaryOrErr)) {
      llvm::errs() << "error: " << llvm::toString(std::move(err)) << "\n";
      return 1;
    }
  }
  return 0;
}
