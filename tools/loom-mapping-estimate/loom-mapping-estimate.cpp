#include "PnR/MappingEstimator.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

static llvm::cl::opt<std::string>
    mappingArtifactPath("mapping-artifact",
                        llvm::cl::desc("PnR mapping artifact JSON"),
                        llvm::cl::Required);

static llvm::cl::opt<std::string>
    hardwareMlirPath("hardware-mlir",
                     llvm::cl::desc("Fabric ADG or module MLIR input"),
                     llvm::cl::init(""));

static llvm::cl::opt<std::string>
    outputPath("output", llvm::cl::desc("Mapping estimate report JSON"),
               llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "loom-mapping-estimate: validate a PnR artifact and emit "
      "a static mapping cost estimate\n");

  loom::pnr::MappingEstimateOptions options;
  options.mappingArtifactPath = mappingArtifactPath;
  options.hardwareMlirPath = hardwareMlirPath;

  llvm::Expected<loom::pnr::MappingEstimateReport> reportOrErr =
      loom::pnr::estimateMapping(options);
  if (!reportOrErr) {
    llvm::errs() << "error: " << llvm::toString(reportOrErr.takeError())
                 << "\n";
    return 1;
  }
  if (llvm::Error err =
          loom::pnr::writeMappingEstimateReportJson(outputPath, *reportOrErr)) {
    llvm::errs() << "error: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }
  return 0;
}
