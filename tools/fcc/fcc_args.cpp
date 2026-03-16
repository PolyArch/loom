#include "fcc_args.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm;

namespace fcc {

static cl::list<std::string> inputSources(cl::Positional,
                                           cl::desc("<source files>"),
                                           cl::OneOrMore);

static cl::opt<std::string> outputDir("o", cl::desc("Output directory"),
                                       cl::Required);

static cl::list<std::string> includePaths("I", cl::desc("Include path"),
                                           cl::Prefix);

static cl::opt<std::string> adgPath("adg", cl::desc("Path to .fabric.mlir ADG"),
                                     cl::init(""));

static cl::opt<bool> simulate("simulate",
                               cl::desc("Run standalone simulator after mapping"),
                               cl::init(false));

static cl::opt<unsigned> simMaxCycles("sim-max-cycles",
                                       cl::desc("Max simulation cycles"),
                                       cl::init(1000000));

static cl::opt<unsigned> mapperBudget("mapper-budget",
                                       cl::desc("Mapper time budget (seconds)"),
                                       cl::init(60));

static cl::opt<unsigned> mapperSeed("mapper-seed",
                                     cl::desc("Deterministic seed"),
                                     cl::init(0));

bool parseArgs(int argc, char **argv, FccArgs &args) {
  cl::ParseCommandLineOptions(argc, argv, "fcc - fabric compiler\n");

  args.sources.assign(inputSources.begin(), inputSources.end());
  args.outputDir = outputDir;
  args.includePaths.assign(includePaths.begin(), includePaths.end());
  args.adgPath = adgPath;
  args.simulate = simulate;
  args.simMaxCycles = simMaxCycles;
  args.mapperBudget = mapperBudget;
  args.mapperSeed = mapperSeed;

  // Derive base name from first source
  if (!args.sources.empty()) {
    StringRef stem = sys::path::stem(args.sources[0]);
    args.baseName = stem.str();
  }

  // Ensure output directory exists
  if (std::error_code ec = sys::fs::create_directories(args.outputDir)) {
    errs() << "fcc: cannot create output directory '" << args.outputDir
           << "': " << ec.message() << "\n";
    return false;
  }

  return true;
}

} // namespace fcc
