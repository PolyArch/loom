//===-- ConfigGen.h - Configuration bitstream generation -----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Generates per-node configuration fragments and assembles them into a flat
// binary config_mem image. Outputs .config.bin, _addr.h, and .mapping.json.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_CONFIGGEN_H
#define LOOM_MAPPER_CONFIGGEN_H

#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

#include <string>
#include <vector>

namespace loom {

class ConfigGen {
public:
  /// Generate all output files from a completed mapping.
  /// basePath: output path prefix (e.g., "output/kernel")
  /// dumpMapping: if true, also write .mapping.json
  bool generate(const MappingState &state, const Graph &dfg, const Graph &adg,
                const std::string &basePath, bool dumpMapping,
                const std::string &profile, int seed);

  /// Write the binary config image.
  bool writeBinary(const std::string &path);

  /// Write the C address header.
  bool writeAddrHeader(const std::string &path);

  /// Write the JSON mapping report.
  bool writeMappingJson(const MappingState &state, const Graph &dfg,
                        const Graph &adg, const std::string &path,
                        const std::string &profile, int seed);

private:
  /// Per-node config fragment: node name -> (word offset, word count, data).
  struct NodeConfig {
    std::string name;
    uint32_t wordOffset = 0;
    uint32_t wordCount = 0;
    std::vector<uint32_t> words;
  };

  std::vector<NodeConfig> nodeConfigs;
  std::vector<uint8_t> configBlob;
  uint32_t totalConfigWords = 0;
  uint32_t wordWidthBits = 32;
};

} // namespace loom

#endif // LOOM_MAPPER_CONFIGGEN_H
