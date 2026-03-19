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

static cl::opt<std::string> mapJsonPathOpt(
    "map-json",
    cl::desc("Path to existing mapping JSON for visualization regeneration"),
    cl::init(""));

static cl::opt<bool> vizOnlyOpt("viz-only",
                                cl::desc("Visualize ADG/DFG side-by-side without mapping"),
                                cl::init(false));

static cl::opt<std::string> vizLayoutOpt(
    "viz-layout",
    cl::desc("Visualization layout mode: default or neato"),
    cl::init("default"));

static cl::opt<bool> simulate("simulate",
                               cl::desc("Run standalone simulator after mapping"),
                               cl::init(false));

static cl::opt<std::string> simBundlePath(
    "sim-bundle",
    cl::desc("Path to simulation bundle JSON with explicit inputs and expected results"),
    cl::init(""));

static cl::opt<unsigned> simMaxCycles("sim-max-cycles",
                                       cl::desc("Max simulation cycles"),
                                       cl::init(1000000));

static cl::opt<std::string> runtimeManifestPath(
    "runtime-manifest",
    cl::desc("Path to mapped-case runtime manifest JSON"),
    cl::init(""));

static cl::opt<std::string> runtimeRequestPath(
    "runtime-request",
    cl::desc("Path to runtime invocation request JSON"),
    cl::init(""));

static cl::opt<std::string> runtimeResultPath(
    "runtime-result",
    cl::desc("Path to runtime invocation result JSON"),
    cl::init(""));

static cl::opt<std::string> runtimeTracePath(
    "runtime-trace",
    cl::desc("Path to runtime replay trace artifact"),
    cl::init(""));

static cl::opt<std::string> runtimeStatPath(
    "runtime-stat",
    cl::desc("Path to runtime replay stat artifact"),
    cl::init(""));

static cl::opt<unsigned> mapperBudget("mapper-budget",
                                       cl::desc("Mapper time budget (seconds)"),
                                       cl::init(60));

static cl::opt<unsigned> mapperSeed("mapper-seed",
                                     cl::desc("Deterministic seed"),
                                     cl::init(0));

static cl::opt<unsigned> mapperLanes(
    "mapper-lanes",
    cl::desc("Number of parallel placement-and-route lanes (0 = auto)"),
    cl::init(0));

static cl::opt<unsigned> mapperInterleavedRounds(
    "mapper-interleaved-rounds",
    cl::desc("Number of place-route interleaving rounds"),
    cl::init(4));

static cl::opt<unsigned> mapperSelectiveRipupPasses(
    "mapper-selective-ripup-passes",
    cl::desc("Number of failed-edge selective rip-up passes per routing round"),
    cl::init(3));

static cl::opt<unsigned> mapperPlacementMoveRadius(
    "mapper-placement-move-radius",
    cl::desc("Detailed placement move radius in Manhattan distance (0 = unrestricted)"),
    cl::init(3));

static cl::opt<unsigned> mapperCpSatGlobalNodeLimit(
    "mapper-cpsat-global-node-limit",
    cl::desc("Maximum node count for CP-SAT global placement"),
    cl::init(24));

static cl::opt<unsigned> mapperCpSatNeighborhoodNodeLimit(
    "mapper-cpsat-neighborhood-node-limit",
    cl::desc("Maximum node count for CP-SAT neighborhood repair"),
    cl::init(8));

static cl::opt<double> mapperCpSatTimeLimitSeconds(
    "mapper-cpsat-time-limit",
    cl::desc("Per-solve CP-SAT time limit in seconds"),
    cl::init(0.75));

static cl::opt<bool> mapperEnableCpSat(
    "mapper-enable-cpsat",
    cl::desc("Enable OR-Tools CP-SAT placement refinement"),
    cl::init(true));

static cl::opt<double> mapperRoutingHeuristicWeight(
    "mapper-routing-heuristic-weight",
    cl::desc("Weighted A* heuristic multiplier (Chebyshev distance weight)"),
    cl::init(1.5));

static cl::opt<unsigned> mapperNegotiatedRoutingPasses(
    "mapper-negotiated-routing-passes",
    cl::desc("Number of negotiated congestion routing iterations (0 = disable)"),
    cl::init(12));

static cl::opt<double> mapperCongestionHistoryFactor(
    "mapper-congestion-history-factor",
    cl::desc("PathFinder congestion history increment"),
    cl::init(1.0));

static cl::opt<double> mapperCongestionHistoryScale(
    "mapper-congestion-history-scale",
    cl::desc("PathFinder congestion history scaling per iteration"),
    cl::init(1.5));

static cl::opt<double> mapperCongestionPresentFactor(
    "mapper-congestion-present-factor",
    cl::desc("PathFinder present-demand cost weight"),
    cl::init(1.0));

static cl::opt<double> mapperCongestionPlacementWeight(
    "mapper-congestion-placement-weight",
    cl::desc("Weight of congestion penalty in placement scoring"),
    cl::init(0.3));

bool parseArgs(int argc, char **argv, FccArgs &args) {
  cl::ParseCommandLineOptions(argc, argv, "fcc - fabric compiler\n");

  args.sources.assign(inputSources.begin(), inputSources.end());
  args.outputDir = outputDir;
  args.includePaths.assign(includePaths.begin(), includePaths.end());
  args.adgPath = adgPath;
  args.dfgPath = dfgPathOpt;
  args.mapJsonPath = mapJsonPathOpt;
  args.vizOnly = vizOnlyOpt;
  if (vizLayoutOpt == "default")
    args.vizLayout = VizLayoutMode::Default;
  else if (vizLayoutOpt == "neato")
    args.vizLayout = VizLayoutMode::Neato;
  else {
    errs() << "fcc: --viz-layout must be 'default' or 'neato'\n";
    return false;
  }
  args.simulate = simulate;
  args.simBundlePath = simBundlePath;
  args.simMaxCycles = simMaxCycles;
  args.runtimeManifestPath = runtimeManifestPath;
  args.runtimeRequestPath = runtimeRequestPath;
  args.runtimeResultPath = runtimeResultPath;
  args.runtimeTracePath = runtimeTracePath;
  args.runtimeStatPath = runtimeStatPath;
  args.mapperBudget = mapperBudget;
  args.mapperSeed = mapperSeed;
  args.mapperLanes = mapperLanes;
  args.mapperInterleavedRounds = mapperInterleavedRounds;
  args.mapperSelectiveRipupPasses = mapperSelectiveRipupPasses;
  args.mapperPlacementMoveRadius = mapperPlacementMoveRadius;
  args.mapperCpSatGlobalNodeLimit = mapperCpSatGlobalNodeLimit;
  args.mapperCpSatNeighborhoodNodeLimit = mapperCpSatNeighborhoodNodeLimit;
  args.mapperCpSatTimeLimitSeconds = mapperCpSatTimeLimitSeconds;
  args.mapperEnableCpSat = mapperEnableCpSat;
  args.mapperRoutingHeuristicWeight = mapperRoutingHeuristicWeight;
  args.mapperNegotiatedRoutingPasses = mapperNegotiatedRoutingPasses;
  args.mapperCongestionHistoryFactor = mapperCongestionHistoryFactor;
  args.mapperCongestionHistoryScale = mapperCongestionHistoryScale;
  args.mapperCongestionPresentFactor = mapperCongestionPresentFactor;
  args.mapperCongestionPlacementWeight = mapperCongestionPlacementWeight;

  // --viz-only needs at least --dfg or --adg
  if (!args.runtimeManifestPath.empty()) {
    if (args.runtimeRequestPath.empty() || args.runtimeResultPath.empty()) {
      errs() << "fcc: --runtime-manifest requires --runtime-request and "
                "--runtime-result\n";
      return false;
    }
  } else if (args.vizOnly) {
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
  } else if (!args.runtimeManifestPath.empty()) {
    StringRef stem = sys::path::stem(args.runtimeManifestPath);
    if (stem.ends_with(".runtime"))
      stem = stem.drop_back(8);
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
