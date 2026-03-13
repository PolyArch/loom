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

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Operation.h"

#include <string>
#include <vector>

namespace loom {

class ConfigGen {
public:
  /// Per-module config address slice (public for simulator consumption).
  struct ConfigSlice {
    std::string name;
    uint32_t wordOffset = 0;
    uint32_t wordCount = 0;
  };

  /// Generate all output files from a completed mapping.
  /// basePath: output path prefix (e.g., "output/kernel")
  /// Always writes: .config.bin, _addr.h, .map.json, .map.txt
  bool generate(const MappingState &state, const Graph &dfg, const Graph &adg,
                const std::string &basePath,
                const std::string &profile, int seed);

  /// Get per-module config slices (valid after generate()).
  /// Each entry maps to an ADG hardware node by position.
  const std::vector<ConfigSlice> &getConfigSlices() const {
    return configSlices_;
  }

  /// Get the total config blob size in bytes (valid after generate()).
  size_t getConfigBlobSize() const { return configBlob.size(); }

  /// Write the binary config image.
  bool writeBinary(const std::string &path);

  /// Write the C address header.
  bool writeAddrHeader(const std::string &path);

  /// Write the JSON mapping report (.map.json).
  bool writeMapJson(const MappingState &state, const Graph &dfg,
                    const Graph &adg, const std::string &path,
                    const std::string &profile, int seed);

  /// Write the human-readable mapping report (.map.txt).
  bool writeMapText(const MappingState &state, const Graph &dfg,
                    const Graph &adg, const std::string &path);

  /// Write a configured copy of the fabric MLIR with runtime configuration
  /// attributes set on all configurable ops according to the mapping result:
  /// route_table on switches, instruction_mem on temporal PE instances,
  /// output_tag on tagged PE instances, and tag on add_tag ops.
  bool writeConfiguredFabric(
      const MappingState &state, const Graph &dfg, const Graph &adg,
      const llvm::DenseMap<mlir::Operation *, IdIndex> &opMap,
      mlir::Operation *adgModule, const std::string &path);

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

  /// Public-facing config slice metadata (populated by generate()).
  std::vector<ConfigSlice> configSlices_;
};

} // namespace loom

#endif // LOOM_MAPPER_CONFIGGEN_H
