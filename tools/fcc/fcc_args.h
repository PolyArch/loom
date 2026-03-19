#ifndef FCC_TOOLS_FCC_ARGS_H
#define FCC_TOOLS_FCC_ARGS_H

#include "fcc/Mapper/MapperOptions.h"
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

  // Existing mapping JSON for visualization-only regeneration
  std::string mapJsonPath;

  // Viz-only mode: just visualize ADG/DFG side-by-side, no mapping
  bool vizOnly = false;
  VizLayoutMode vizLayout = VizLayoutMode::Default;

  // Simulation
  bool simulate = false;
  std::string simBundlePath;
  unsigned simMaxCycles = 1000000;
  std::string runtimeManifestPath;
  std::string runtimeRequestPath;
  std::string runtimeResultPath;
  std::string runtimeTracePath;
  std::string runtimeStatPath;

  // Mapper
  std::string mapperBaseConfigPath;
  std::string mapperResolvedBaseConfigPath;
  MapperOptions mapperOptions;

  // Derived: base name of first source (e.g. "vecadd" from "vecadd.c")
  std::string baseName;
};

// Parse command line arguments. Returns true on success.
bool parseArgs(int argc, char **argv, FccArgs &args);

} // namespace fcc

#endif
