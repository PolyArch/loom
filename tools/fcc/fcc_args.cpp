#include "fcc_args.h"
#include "mapper_config.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <cmath>
#include <string>
#include <type_traits>

using namespace llvm;

namespace fcc {

namespace {

template <typename T> bool mapperValuesDiffer(const T &lhs, const T &rhs) {
  if constexpr (std::is_floating_point_v<T>)
    return std::abs(lhs - rhs) > 1.0e-9;
  return lhs != rhs;
}

template <typename T> std::string formatMapperValue(const T &value) {
  if constexpr (std::is_same_v<T, bool>)
    return value ? "true" : "false";
  return std::to_string(value);
}

template <> std::string formatMapperValue<std::string>(const std::string &value) {
  return value;
}

template <typename T, typename U>
void applyMapperCliOverride(llvm::StringRef optionName, const T &cliValue,
                            unsigned occurrences, U &mappedValue) {
  if (occurrences == 0)
    return;
  U promotedValue = static_cast<U>(cliValue);
  if (mapperValuesDiffer(mappedValue, promotedValue)) {
    errs() << "fcc: warning: CLI option --" << optionName
           << " overrides mapper base config value "
           << formatMapperValue(mappedValue) << " -> "
           << formatMapperValue(promotedValue) << "\n";
  }
  mappedValue = promotedValue;
}

} // namespace

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

static cl::opt<bool> genSV("gen-sv",
                             cl::desc("Generate synthesizable SystemVerilog from ADG"),
                             cl::init(false));

static cl::opt<std::string> fpIpProfile(
    "fp-ip-profile",
    cl::desc("Floating-point IP profile for synthesis (enables Tier 3 transcendental FP ops)"),
    cl::init(""));

static cl::opt<std::string> tracePortDump(
    "trace-port-dump",
    cl::desc("Dump per-cycle port traces for the named module during simulation"),
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

static cl::opt<double> mapperSnapshotIntervalSeconds(
    "mapper-snapshot-interval-seconds",
    cl::desc("Emit periodic mapper snapshots every N seconds (-1 = disabled)"),
    cl::init(-1.0));

static cl::opt<int> mapperSnapshotIntervalRounds(
    "mapper-snapshot-interval-rounds",
    cl::desc("Emit periodic mapper snapshots every N progress rounds (-1 = disabled)"),
    cl::init(-1));

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

static cl::opt<std::string> mapperBaseConfigPathOpt(
    "mapper-base-config",
    cl::desc("Path to mapper base config YAML (default: repository built-in template)"),
    cl::init(""));

bool parseArgs(int argc, char **argv, FccArgs &args) {
  cl::ParseCommandLineOptions(argc, argv, "fcc - fabric compiler\n");

  args.sources.assign(inputSources.begin(), inputSources.end());
  args.outputDir = outputDir.getValue();
  args.includePaths.assign(includePaths.begin(), includePaths.end());
  args.adgPath = adgPath.getValue();
  args.dfgPath = dfgPathOpt.getValue();
  args.mapJsonPath = mapJsonPathOpt.getValue();
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
  args.simBundlePath = simBundlePath.getValue();
  args.simMaxCycles = simMaxCycles;
  args.runtimeManifestPath = runtimeManifestPath.getValue();
  args.runtimeRequestPath = runtimeRequestPath.getValue();
  args.runtimeResultPath = runtimeResultPath.getValue();
  args.runtimeTracePath = runtimeTracePath.getValue();
  args.runtimeStatPath = runtimeStatPath.getValue();
  args.genSV = genSV;
  args.fpIpProfile = fpIpProfile.getValue();
  args.tracePortDump = tracePortDump.getValue();

  args.mapperBaseConfigPath = mapperBaseConfigPathOpt.getValue();
  if (mapperBaseConfigPathOpt.getNumOccurrences() > 0)
    args.mapperResolvedBaseConfigPath = mapperBaseConfigPathOpt.getValue();
  else
    args.mapperResolvedBaseConfigPath = getDefaultMapperBaseConfigPath();
  if (args.mapperResolvedBaseConfigPath.empty()) {
    errs() << "fcc: no default mapper base config path is available\n";
    return false;
  }
  std::string mapperConfigError;
  if (!loadMapperBaseConfig(args.mapperResolvedBaseConfigPath,
                            args.mapperOptions, mapperConfigError)) {
    errs() << "fcc: " << mapperConfigError << "\n";
    return false;
  }
  applyMapperCliOverride("mapper-budget", mapperBudget,
                         mapperBudget.getNumOccurrences(),
                         args.mapperOptions.budgetSeconds);
  applyMapperCliOverride("mapper-seed", mapperSeed,
                         mapperSeed.getNumOccurrences(), args.mapperOptions.seed);
  applyMapperCliOverride("mapper-lanes", mapperLanes,
                         mapperLanes.getNumOccurrences(),
                         args.mapperOptions.lanes);
  applyMapperCliOverride("mapper-snapshot-interval-seconds",
                         mapperSnapshotIntervalSeconds,
                         mapperSnapshotIntervalSeconds.getNumOccurrences(),
                         args.mapperOptions.snapshotIntervalSeconds);
  applyMapperCliOverride("mapper-snapshot-interval-rounds",
                         mapperSnapshotIntervalRounds,
                         mapperSnapshotIntervalRounds.getNumOccurrences(),
                         args.mapperOptions.snapshotIntervalRounds);
  applyMapperCliOverride("mapper-interleaved-rounds",
                         mapperInterleavedRounds,
                         mapperInterleavedRounds.getNumOccurrences(),
                         args.mapperOptions.interleavedRounds);
  applyMapperCliOverride("mapper-selective-ripup-passes",
                         mapperSelectiveRipupPasses,
                         mapperSelectiveRipupPasses.getNumOccurrences(),
                         args.mapperOptions.selectiveRipupPasses);
  applyMapperCliOverride("mapper-placement-move-radius",
                         mapperPlacementMoveRadius,
                         mapperPlacementMoveRadius.getNumOccurrences(),
                         args.mapperOptions.placementMoveRadius);
  applyMapperCliOverride("mapper-cpsat-global-node-limit",
                         mapperCpSatGlobalNodeLimit,
                         mapperCpSatGlobalNodeLimit.getNumOccurrences(),
                         args.mapperOptions.cpSatGlobalNodeLimit);
  applyMapperCliOverride("mapper-cpsat-neighborhood-node-limit",
                         mapperCpSatNeighborhoodNodeLimit,
                         mapperCpSatNeighborhoodNodeLimit.getNumOccurrences(),
                         args.mapperOptions.cpSatNeighborhoodNodeLimit);
  applyMapperCliOverride("mapper-cpsat-time-limit",
                         mapperCpSatTimeLimitSeconds,
                         mapperCpSatTimeLimitSeconds.getNumOccurrences(),
                         args.mapperOptions.cpSatTimeLimitSeconds);
  applyMapperCliOverride("mapper-enable-cpsat", mapperEnableCpSat,
                         mapperEnableCpSat.getNumOccurrences(),
                         args.mapperOptions.enableCPSat);
  applyMapperCliOverride("mapper-routing-heuristic-weight",
                         mapperRoutingHeuristicWeight,
                         mapperRoutingHeuristicWeight.getNumOccurrences(),
                         args.mapperOptions.routingHeuristicWeight);
  applyMapperCliOverride("mapper-negotiated-routing-passes",
                         mapperNegotiatedRoutingPasses,
                         mapperNegotiatedRoutingPasses.getNumOccurrences(),
                         args.mapperOptions.negotiatedRoutingPasses);
  applyMapperCliOverride("mapper-congestion-history-factor",
                         mapperCongestionHistoryFactor,
                         mapperCongestionHistoryFactor.getNumOccurrences(),
                         args.mapperOptions.congestionHistoryFactor);
  applyMapperCliOverride("mapper-congestion-history-scale",
                         mapperCongestionHistoryScale,
                         mapperCongestionHistoryScale.getNumOccurrences(),
                         args.mapperOptions.congestionHistoryScale);
  applyMapperCliOverride("mapper-congestion-present-factor",
                         mapperCongestionPresentFactor,
                         mapperCongestionPresentFactor.getNumOccurrences(),
                         args.mapperOptions.congestionPresentFactor);
  applyMapperCliOverride("mapper-congestion-placement-weight",
                         mapperCongestionPlacementWeight,
                         mapperCongestionPlacementWeight.getNumOccurrences(),
                         args.mapperOptions.congestionPlacementWeight);
  std::string mapperOptionError;
  if (!validateMapperOptions(args.mapperOptions, mapperOptionError)) {
    errs() << "fcc: invalid mapper options: " << mapperOptionError << "\n";
    return false;
  }

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
  } else if (args.genSV) {
    // --gen-sv mode: only needs --adg (no sources or --dfg required)
    if (args.adgPath.empty()) {
      errs() << "fcc: --gen-sv requires --adg\n";
      return false;
    }
  } else if (args.simulate && !args.adgPath.empty() &&
             args.sources.empty() && args.dfgPath.empty()) {
    // ADG-only simulation mode: --simulate --adg without sources/DFG
    // Used for golden trace generation via --trace-port-dump
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
    args.baseName = sys::path::stem(args.sources[0]).str();
  } else if (!args.runtimeManifestPath.empty()) {
    args.baseName = sys::path::stem(args.runtimeManifestPath).str();
    if (StringRef(args.baseName).ends_with(".runtime"))
      args.baseName.resize(args.baseName.size() - 8);
  } else if (!args.dfgPath.empty()) {
    args.baseName = sys::path::stem(args.dfgPath).str();
    if (StringRef(args.baseName).ends_with(".dfg"))
      args.baseName.resize(args.baseName.size() - 4);
  } else if (!args.adgPath.empty()) {
    args.baseName = sys::path::stem(args.adgPath).str();
    if (StringRef(args.baseName).ends_with(".fabric"))
      args.baseName.resize(args.baseName.size() - 7);
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
