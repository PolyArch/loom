// Tiny CLI used by lit tests to exercise resolved configuration loading.
//
// Usage: loom-config-test [output option] [path]

#include "Common/ResolvedConfig.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

static ::llvm::cl::opt<std::string> inputPath(::llvm::cl::Positional,
                                              ::llvm::cl::desc("<config-path>"),
                                              ::llvm::cl::init(""));

static ::llvm::cl::opt<bool>
    resolvedJson("resolved-json",
                 ::llvm::cl::desc("dump canonical resolved config JSON"),
                 ::llvm::cl::init(false));

static ::llvm::cl::opt<bool>
    resolvedFingerprint("resolved-fingerprint",
                        ::llvm::cl::desc("dump resolved config fingerprint"),
                        ::llvm::cl::init(false));

static ::llvm::cl::opt<bool> componentFingerprint(
    "component-fingerprint",
    ::llvm::cl::desc("dump typed component config view fingerprint"),
    ::llvm::cl::init(false));

static ::llvm::cl::opt<std::string>
    componentView("component-view",
                  ::llvm::cl::desc("component config view identity"),
                  ::llvm::cl::init(""));

int main(int argc, char **argv) {
  ::llvm::cl::ParseCommandLineOptions(argc, argv,
                                      "loom-config-test: parse and dump a "
                                      "resolved configuration file\n");
  if (!(resolvedJson || resolvedFingerprint || componentFingerprint)) {
    ::llvm::errs() << "error: expected a resolved config output option\n";
    return 1;
  }
  if (componentFingerprint && componentView.empty()) {
    ::llvm::errs() << "error: --component-fingerprint requires "
                      "--component-view <view-id>\n";
    return 1;
  }

  ::llvm::Expected<::loom::ResolvedConfig> cfg =
      inputPath.empty() ? ::llvm::Expected<::loom::ResolvedConfig>(
                              ::loom::defaultResolvedConfig())
                        : ::loom::loadResolvedConfig(inputPath);
  if (!cfg) {
    ::llvm::errs() << "error: " << ::llvm::toString(cfg.takeError()) << "\n";
    return 1;
  }

  if (resolvedJson)
    ::llvm::outs() << ::loom::canonicalResolvedConfigJson(*cfg) << "\n";
  if (resolvedFingerprint)
    ::llvm::outs() << ::loom::resolvedConfigFingerprint(*cfg) << "\n";
  if (componentFingerprint)
    ::llvm::outs() << ::loom::componentConfigFingerprint(*cfg, componentView)
                   << "\n";
  return 0;
}
