#ifndef FCC_TOOLS_FCC_ARGS_H
#define FCC_TOOLS_FCC_ARGS_H

#include "fcc/Viz/VizExporter.h"

#include <string>
#include <vector>

namespace fcc {

struct FccArgs {
  // Input sources
  std::vector<std::string> sources;

  // Output directory
  std::string outputDir;

  // Include paths (forwarded to clang)
  std::vector<std::string> includePaths;

  // ADG path (fabric MLIR)
  std::string adgPath;

  // DFG path (handshake MLIR) - skip frontend when provided
  std::string dfgPath;

  // Viz-only mode: just visualize ADG/DFG side-by-side, no mapping
  bool vizOnly = false;
  VizLayoutMode vizLayout = VizLayoutMode::Default;

  // Simulation
  bool simulate = false;
  unsigned simMaxCycles = 1000000;

  // Mapper
  unsigned mapperBudget = 60;
  unsigned mapperSeed = 0;
  unsigned mapperLanes = 0;
  unsigned mapperInterleavedRounds = 4;
  unsigned mapperSelectiveRipupPasses = 3;
  unsigned mapperPlacementMoveRadius = 3;
  unsigned mapperCpSatGlobalNodeLimit = 24;
  unsigned mapperCpSatNeighborhoodNodeLimit = 8;
  double mapperCpSatTimeLimitSeconds = 0.75;
  bool mapperEnableCpSat = true;

  // Derived: base name of first source (e.g. "vecadd" from "vecadd.c")
  std::string baseName;
};

// Parse command line arguments. Returns true on success.
bool parseArgs(int argc, char **argv, FccArgs &args);

} // namespace fcc

#endif
