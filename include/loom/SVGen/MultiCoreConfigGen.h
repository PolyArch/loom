#ifndef LOOM_SVGEN_MULTICORECONFIGGEN_H
#define LOOM_SVGEN_MULTICORECONFIGGEN_H

#include "loom/SVGen/MultiCoreSVGen.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace svgen {

/// Configuration data for a single core within the multi-core system image.
struct CoreConfigEntry {
  /// Core instance name (unique across the system).
  std::string coreInstanceName;

  /// Core type name (shared across instances of the same type).
  std::string coreType;

  /// Raw configuration blob for this core.
  std::vector<uint8_t> configBlob;

  /// Per-hardware-node configuration slices.
  std::vector<ConfigGen::ConfigSlice> slices;

  /// Byte offset of this core's config within the full system image.
  uint32_t baseOffset = 0;
};

/// System-level configuration (NoC routing tables, DMA config, L2 mapping).
struct SystemConfigEntry {
  /// Raw system-level configuration blob.
  std::vector<uint8_t> systemConfigBlob;

  /// Byte offset within the full system image.
  uint32_t baseOffset = 0;
};

/// Multi-core configuration image assembled from per-core configs and
/// system-level configuration.
///
/// Binary layout:
///   [Header: 8 bytes]
///     - uint32_t numCores
///     - uint32_t perCoreConfigSize (max across all cores, for uniform stride)
///   [Per-core config blobs, each padded to perCoreConfigSize]
///   [System config blob]
struct MultiCoreConfigImage {
  /// Per-core configuration entries.
  std::vector<CoreConfigEntry> coreConfigs;

  /// System-level configuration entry.
  SystemConfigEntry systemConfig;

  /// Fully assembled binary image.
  std::vector<uint8_t> fullImage;

  /// Total size of the image in 32-bit words.
  uint32_t totalWords = 0;

  /// Write the full image to a binary file.
  bool writeBinary(const std::string &path) const;

  /// Write a JSON manifest describing the image layout.
  bool writeJSON(const std::string &path) const;

  /// Write a C header file with the image as a const array.
  bool writeHeader(const std::string &path) const;
};

/// Generate a multi-core configuration image from the compilation description.
///
/// For each core descriptor, extracts the aggregateConfigBlob and configSlices,
/// then assembles them into a single binary image with a header.
///
/// \param compilation  The multi-core compilation description.
/// \returns            The assembled MultiCoreConfigImage.
MultiCoreConfigImage
generateMultiCoreConfig(const MultiCoreCompilationDesc &compilation);

} // namespace svgen
} // namespace loom

#endif // LOOM_SVGEN_MULTICORECONFIGGEN_H
