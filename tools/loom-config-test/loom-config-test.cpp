// Tiny CLI used by lit tests to exercise the TechMapConfig YAML/TOML loader.
//
// Usage: loom-config-test <path>
// Prints one key=value pair per line in a stable order, or `error: ...` to
// stderr on failure (and exits non-zero).

#include "Common/Config.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>

static ::llvm::cl::opt<std::string>
    inputPath(::llvm::cl::Positional, ::llvm::cl::desc("<config-path>"),
              ::llvm::cl::Required);

int main(int argc, char **argv) {
  ::llvm::cl::ParseCommandLineOptions(argc, argv,
                                       "loom-config-test: parse and dump a "
                                       "TechMapConfig file\n");
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
