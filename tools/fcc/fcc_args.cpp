#include "fcc_args.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm;

namespace fcc {

static cl::list<std::string> inputSources(cl::Positional,
                                           cl::desc("<source files>"),
                                           cl::ZeroOrMore);

static cl::opt<std::string> outputDir("o", cl::desc("Output directory"),
                                       cl::Required);

static cl::list<std::string> includePaths("I", cl::desc("Include path"),
                                           cl::Prefix);

static cl::opt<std::string> adgPath("adg", cl::desc("Path to .fabric.mlir ADG"),
                                     cl::init(""));

static cl::opt<std::string> dfgPathOpt("dfg",
                                        cl::desc("Path to pre-built DFG .mlir (skip frontend)"),
                                        cl::init(""));

static cl::opt<bool> vizOnlyOpt("viz-only",
                                cl::desc("Visualize ADG/DFG side-by-side without mapping"),
                                cl::init(false));

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
  args.dfgPath = dfgPathOpt;
  args.vizOnly = vizOnlyOpt;
  args.simulate = simulate;
  args.simMaxCycles = simMaxCycles;
  args.mapperBudget = mapperBudget;
  args.mapperSeed = mapperSeed;

  // --viz-only needs at least --dfg or --adg
  if (args.vizOnly) {
    if (args.dfgPath.empty() && args.adgPath.empty()) {
      errs() << "fcc: --viz-only needs at least --dfg or --adg\n";
      return false;
    }
  } else {
    // Normal mode: need sources or --dfg
    if (args.sources.empty() && args.dfgPath.empty()) {
      errs() << "fcc: no input sources and no --dfg specified\n";
      return false;
    }
    // --dfg requires --adg (for mapping)
    if (!args.dfgPath.empty() && args.adgPath.empty()) {
      errs() << "fcc: --dfg requires --adg\n";
      return false;
    }
  }

  // Derive base name
  if (!args.sources.empty()) {
    StringRef stem = sys::path::stem(args.sources[0]);
    args.baseName = stem.str();
  } else if (!args.dfgPath.empty()) {
    StringRef stem = sys::path::stem(args.dfgPath);
    if (stem.ends_with(".dfg"))
      stem = stem.drop_back(4);
    args.baseName = stem.str();
  } else if (!args.adgPath.empty()) {
    StringRef stem = sys::path::stem(args.adgPath);
    if (stem.ends_with(".fabric"))
      stem = stem.drop_back(7);
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
