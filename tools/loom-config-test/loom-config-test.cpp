// Tiny CLI used by lit tests to exercise resolved configuration loading.
//
// Usage: loom-config-test [output option] [--loom-accel-profile=<selector>]

#include "Common/ResolvedConfig.h"

#include "Common/ArtifactText.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

static ::llvm::cl::opt<std::string> accelerationProfile(
    "loom-accel-profile",
    ::llvm::cl::desc("builtin acceleration preset or configuration path"),
    ::llvm::cl::value_desc("preset-or-path"), ::llvm::cl::init(""));

static ::llvm::cl::opt<bool>
    resolvedJson("resolved-json",
                 ::llvm::cl::desc("dump canonical resolved config JSON"),
                 ::llvm::cl::init(false));

static ::llvm::cl::opt<bool>
    resolvedIdentity("resolved-identity",
                     ::llvm::cl::desc("dump resolved config ArtifactIdentity"),
                     ::llvm::cl::init(false));

int main(int argc, char **argv) {
  ::llvm::cl::ParseCommandLineOptions(argc, argv,
                                      "loom-config-test: parse and dump a "
                                      "resolved configuration file\n");
  if (!(resolvedJson || resolvedIdentity)) {
    ::llvm::errs() << "error: expected a resolved config output option\n";
    return 1;
  }

  ::llvm::Expected<::loom::ResolvedConfig> cfg =
      ::loom::resolveConfigProfile(accelerationProfile);
  if (!cfg) {
    ::llvm::errs() << "error: " << ::llvm::toString(cfg.takeError()) << "\n";
    return 1;
  }

  if (resolvedJson)
    ::llvm::outs() << ::loom::canonicalResolvedConfigJson(*cfg) << "\n";
  if (resolvedIdentity)
    ::llvm::outs() << ::loom::formatArtifactIdentityHex(
                          ::loom::resolvedConfigIdentity(*cfg))
                   << "\n";
  return 0;
}
