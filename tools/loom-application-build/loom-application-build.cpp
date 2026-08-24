#include "Application/ProductBuild.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

llvm::cl::opt<std::string> driverArgumentsOutput(
    "driver-arguments-output",
    llvm::cl::desc("Write the System-derived compiler arguments"),
    llvm::cl::value_desc("path"));
llvm::cl::opt<std::string> finalLinkOutput(
    "final-link-output",
    llvm::cl::desc("Consume one completed compiler final-link output"),
    llvm::cl::value_desc("path"));
llvm::cl::opt<std::string>
    deploymentOutput("deployment-output",
                     llvm::cl::desc("Publish the Deployment package"),
                     llvm::cl::value_desc("path"), llvm::cl::Required);
llvm::cl::opt<std::string>
    accelerationProfile("acceleration-profile",
                        llvm::cl::desc("Resolved configuration selector"),
                        llvm::cl::value_desc("selector"));
llvm::cl::opt<std::string> hardwarePath("hardware",
                                        llvm::cl::desc("External Fabric MLIR"),
                                        llvm::cl::value_desc("path"));
llvm::cl::opt<std::string>
    visualizationPath("visualization",
                      llvm::cl::desc("Visualization export destination"),
                      llvm::cl::value_desc("path"));
llvm::cl::opt<std::string>
    localToolConfigPath("local-config",
                        llvm::cl::desc("Explicit local tool configuration"),
                        llvm::cl::value_desc("path"));
llvm::cl::list<std::string> operatorProtocolSymbols(
    "operator-protocol-symbol",
    llvm::cl::desc("Select a defined callable as an operator protocol root"),
    llvm::cl::value_desc("symbol"), llvm::cl::ZeroOrMore);
llvm::cl::opt<std::uint64_t> mappingTechCandidateLimit(
    "mapping-tech-candidate-limit",
    llvm::cl::desc("maximum TechMapping candidates admitted to Spatial PnR "
                   "for each target Module"),
    llvm::cl::init(loom::application::defaultProductTechCandidateLimit));
llvm::cl::opt<std::uint64_t> mappingWallTimeLimitMilliseconds(
    "mapping-wall-time-limit-ms",
    llvm::cl::desc("cooperative pre-Mapping and Mapping wall-time limit"),
    llvm::cl::init(
        loom::application::defaultProductMappingWallTimeLimitMilliseconds));
llvm::cl::opt<std::string> mappingStoppingPolicy(
    "mapping-stopping-policy",
    llvm::cl::desc(
        "Mapping stopping policy: first_verified or bounded_quality"),
    llvm::cl::init("first_verified"));
llvm::cl::opt<std::string> mappingSpectrumEndpoint(
    "mapping-spectrum-endpoint",
    llvm::cl::desc("Spectrum ranking focus: automatic, max_temporal, "
                   "max_spatial, or intermediate"),
    llvm::cl::init("automatic"));

llvm::Expected<loom::application::ProductBuildOptions> productOptions() {
  auto stoppingPolicy = loom::application::parseProductMappingStoppingPolicy(
      mappingStoppingPolicy);
  if (!stoppingPolicy)
    return stoppingPolicy.takeError();
  auto spectrumEndpoint =
      loom::application::parseProductMappingSpectrumEndpoint(
          mappingSpectrumEndpoint);
  if (!spectrumEndpoint)
    return spectrumEndpoint.takeError();
  return loom::application::ProductBuildOptions{
      deploymentOutput,
      accelerationProfile,
      hardwarePath,
      visualizationPath,
      localToolConfigPath,
      std::vector<std::string>(operatorProtocolSymbols.begin(),
                               operatorProtocolSymbols.end()),
      mappingTechCandidateLimit,
      mappingWallTimeLimitMilliseconds,
      *stoppingPolicy,
      *spectrumEndpoint};
}

llvm::Error writeDriverArguments(llvm::ArrayRef<std::string> arguments) {
  std::error_code error;
  llvm::raw_fd_ostream output(driverArgumentsOutput, error,
                              llvm::sys::fs::OF_None);
  if (error)
    return llvm::createStringError(error, "cannot open driver argument output");
  for (const std::string &argument : arguments) {
    output << argument;
    output.write('\0');
  }
  output.close();
  if (output.has_error())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot write driver argument output");
  return llvm::Error::success();
}

int reportError(llvm::Error error) {
  llvm::errs() << "loom-application-build: error: "
               << llvm::toString(std::move(error)) << '\n';
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Loom application build helper\n");
  const bool projectsArguments = !driverArgumentsOutput.empty();
  const bool buildsDeployment = !finalLinkOutput.empty();
  if (projectsArguments == buildsDeployment) {
    llvm::errs() << "loom-application-build: error: select exactly one "
                    "product action\n";
    return 1;
  }

  auto options = productOptions();
  if (!options)
    return reportError(options.takeError());
  auto invocation =
      loom::application::ProductBuildInvocation::create(std::move(*options));
  if (!invocation)
    return reportError(invocation.takeError());
  if (projectsArguments) {
    if (llvm::Error error =
            writeDriverArguments((*invocation)->compilerArguments()))
      return reportError(std::move(error));
    return 0;
  }
  if (llvm::Error error = (*invocation)->buildFromFinalLink(finalLinkOutput))
    return reportError(std::move(error));
  return 0;
}
