// Tiny CLI used by lit tests to exercise the TechMapConfig YAML/TOML loader.
//
// Usage: loom-config-test <path>
// Prints one key=value pair per line in a stable order, or `error: ...` to
// stderr on failure (and exits non-zero).

#include "Common/Config.h"
#include "Common/ResolvedConfig.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>

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
                                      "TechMapConfig file\n");
  if (resolvedJson || resolvedFingerprint || componentFingerprint) {
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

  if (inputPath.empty()) {
    ::llvm::errs() << "error: missing <config-path>\n";
    return 1;
  }
  auto cfg = ::loom::loadTechMapConfig(inputPath.getValue());
  if (!cfg) {
    ::llvm::errs() << "error: " << ::llvm::toString(cfg.takeError()) << "\n";
    return 1;
  }
  ::llvm::outs() << "algorithm=" << cfg->algorithm << "\n";
  ::llvm::outs() << "alpha=" << cfg->alpha << "\n";
  ::llvm::outs() << "beta=" << cfg->beta << "\n";
  ::llvm::outs() << "gamma=" << cfg->gamma << "\n";
  ::llvm::outs() << "beam_width=" << cfg->beamWidth << "\n";
  ::llvm::outs() << "sa_steps=" << cfg->saSteps << "\n";
  ::llvm::outs() << "sa_seed=" << cfg->saSeed << "\n";
  ::llvm::outs() << "threads=" << cfg->threads << "\n";
  return 0;
}
