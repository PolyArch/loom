#include "Application/HostRunner.h"
#include "Application/Manifest.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

namespace {

llvm::cl::opt<std::string>
    manifestPath("manifest", llvm::cl::desc("Application portfolio manifest"),
                 llvm::cl::value_desc("path"), llvm::cl::Required);
llvm::cl::opt<std::string>
    repositoryRoot("repository-root",
                   llvm::cl::desc("Absolute repository root"),
                   llvm::cl::value_desc("path"), llvm::cl::Required);
llvm::cl::opt<std::string> cacheRoot("cache-root",
                                     llvm::cl::desc("Absolute cache root"),
                                     llvm::cl::value_desc("path"));
llvm::cl::opt<std::string>
    applicationIdentity("application",
                        llvm::cl::desc("Exact application identity"),
                        llvm::cl::value_desc("identity"));
llvm::cl::opt<std::string> inputName("input",
                                     llvm::cl::desc("Exact named input"),
                                     llvm::cl::value_desc("name"));
llvm::cl::opt<std::string>
    executionSelection("selection",
                       llvm::cl::desc("Exact manifest execution tier"),
                       llvm::cl::value_desc("smoke|validation|scale_eda"));
llvm::cl::opt<std::string>
    compilerExecutable("compiler",
                       llvm::cl::desc("Explicit clang or clang++ executable"),
                       llvm::cl::value_desc("path"));

int reportError(llvm::Error error) {
  llvm::errs() << "loom-application-host-run: error: "
               << llvm::toString(std::move(error)) << '\n';
  return EXIT_FAILURE;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Loom bounded application host runner\n");

  auto manifest = loom::application::loadApplicationManifest(manifestPath);
  if (!manifest)
    return reportError(manifest.takeError());

  const bool hasExactSelection =
      !applicationIdentity.empty() || !inputName.empty();
  if ((!executionSelection.empty() && hasExactSelection) ||
      (executionSelection.empty() &&
       (applicationIdentity.empty() || inputName.empty()))) {
    llvm::errs()
        << "loom-application-host-run: error: select either --selection or "
           "the complete --application/--input pair\n";
    return EXIT_FAILURE;
  }

  if (!executionSelection.empty()) {
    auto selection =
        loom::application::parseExecutionSelection(executionSelection);
    if (!selection)
      return reportError(selection.takeError());
    loom::application::ApplicationHostSelectionRunRequest request{
        *selection, repositoryRoot, std::nullopt, std::nullopt};
    if (!cacheRoot.empty())
      request.cacheRoot = cacheRoot;
    if (!compilerExecutable.empty())
      request.compilerExecutable = compilerExecutable;
    auto report =
        loom::application::runApplicationSelectionOnHost(*manifest, request);
    if (!report)
      return reportError(report.takeError());
    loom::application::writeApplicationHostSelectionRunReportJson(llvm::outs(),
                                                                  *report);
    for (const loom::application::ApplicationHostRunReport &member :
         report->reports)
      if (!member.diagnostic.empty())
        llvm::errs() << "loom-application-host-run: " << member.diagnostic
                     << '\n';
    return loom::application::applicationHostSelectionRunSucceeded(*report)
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
  }

  loom::application::ApplicationHostRunRequest request{
      applicationIdentity, inputName, repositoryRoot, std::nullopt,
      std::nullopt};
  if (!cacheRoot.empty())
    request.cacheRoot = cacheRoot;
  if (!compilerExecutable.empty())
    request.compilerExecutable = compilerExecutable;

  auto report =
      loom::application::runApplicationInputOnHost(*manifest, request);
  if (!report)
    return reportError(report.takeError());
  loom::application::writeApplicationHostRunReportJson(llvm::outs(), *report);
  if (!report->diagnostic.empty())
    llvm::errs() << "loom-application-host-run: " << report->diagnostic << '\n';
  return loom::application::applicationHostRunSucceeded(*report) ? EXIT_SUCCESS
                                                                 : EXIT_FAILURE;
}
