#include "loom/SVGen/MultiCoreConfigGen.h"
#include "loom/SVGen/SVEmitter.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace loom {
namespace svgen {

namespace {

/// Write a uint32_t in little-endian format to a byte vector at the given
/// offset.
static void writeU32LE(std::vector<uint8_t> &buf, size_t offset,
                       uint32_t value) {
  if (offset + 4 > buf.size())
    buf.resize(offset + 4, 0);
  buf[offset + 0] = static_cast<uint8_t>(value & 0xFF);
  buf[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFF);
  buf[offset + 2] = static_cast<uint8_t>((value >> 16) & 0xFF);
  buf[offset + 3] = static_cast<uint8_t>((value >> 24) & 0xFF);
}

/// Round up to the next multiple of 4 (word alignment).
static uint32_t alignToWord(uint32_t size) { return (size + 3) & ~3u; }

} // namespace

//===----------------------------------------------------------------------===//
// MultiCoreConfigImage output methods
//===----------------------------------------------------------------------===//

bool MultiCoreConfigImage::writeBinary(const std::string &path) const {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "multi-core-configgen: cannot write binary " << path
                 << ": " << ec.message() << "\n";
    return false;
  }
  os.write(reinterpret_cast<const char *>(fullImage.data()), fullImage.size());
  return true;
}

bool MultiCoreConfigImage::writeJSON(const std::string &path) const {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "multi-core-configgen: cannot write JSON " << path << ": "
                 << ec.message() << "\n";
    return false;
  }

  os << "{\n";
  os << "  \"totalWords\": " << totalWords << ",\n";
  os << "  \"totalBytes\": " << fullImage.size() << ",\n";
  os << "  \"numCores\": " << coreConfigs.size() << ",\n";
  os << "  \"cores\": [\n";

  for (unsigned i = 0; i < coreConfigs.size(); ++i) {
    const auto &cc = coreConfigs[i];
    os << "    {\n";
    os << "      \"coreInstanceName\": \"" << cc.coreInstanceName << "\",\n";
    os << "      \"coreType\": \"" << cc.coreType << "\",\n";
    os << "      \"baseOffset\": " << cc.baseOffset << ",\n";
    os << "      \"configBytes\": " << cc.configBlob.size() << ",\n";
    os << "      \"numSlices\": " << cc.slices.size() << ",\n";
    os << "      \"slices\": [\n";

    for (unsigned j = 0; j < cc.slices.size(); ++j) {
      const auto &sl = cc.slices[j];
      os << "        {\"name\": \"" << sl.name << "\", \"kind\": \"" << sl.kind
         << "\", \"wordOffset\": " << sl.wordOffset
         << ", \"wordCount\": " << sl.wordCount << "}";
      if (j + 1 < cc.slices.size())
        os << ",";
      os << "\n";
    }

    os << "      ]\n";
    os << "    }";
    if (i + 1 < coreConfigs.size())
      os << ",";
    os << "\n";
  }

  os << "  ],\n";
  os << "  \"systemConfig\": {\n";
  os << "    \"baseOffset\": " << systemConfig.baseOffset << ",\n";
  os << "    \"configBytes\": " << systemConfig.systemConfigBlob.size()
     << "\n";
  os << "  }\n";
  os << "}\n";

  return true;
}

bool MultiCoreConfigImage::writeHeader(const std::string &path) const {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "multi-core-configgen: cannot write header " << path
                 << ": " << ec.message() << "\n";
    return false;
  }

  os << "#ifndef TAPESTRY_SYSTEM_CONFIG_H\n";
  os << "#define TAPESTRY_SYSTEM_CONFIG_H\n\n";
  os << "#include <stdint.h>\n\n";
  os << "#define TAPESTRY_NUM_CORES " << coreConfigs.size() << "\n";
  os << "#define TAPESTRY_CONFIG_TOTAL_WORDS " << totalWords << "\n";
  os << "#define TAPESTRY_CONFIG_TOTAL_BYTES " << fullImage.size() << "\n\n";

  // Per-core base offset defines.
  for (unsigned i = 0; i < coreConfigs.size(); ++i) {
    std::string upper =
        SVEmitter::sanitizeName(coreConfigs[i].coreInstanceName);
    // Convert to uppercase for preprocessor defines.
    for (char &c : upper)
      c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
    os << "#define TAPESTRY_CORE_" << upper << "_OFFSET "
       << coreConfigs[i].baseOffset << "\n";
  }
  os << "\n";

  // Full image as a const array.
  os << "static const uint8_t tapestry_system_config[" << fullImage.size()
     << "] = {\n";
  for (size_t i = 0; i < fullImage.size(); ++i) {
    if (i % 16 == 0)
      os << "  ";
    os << "0x";
    os.write_hex(fullImage[i]);
    if (i + 1 < fullImage.size())
      os << ", ";
    if (i % 16 == 15 || i + 1 == fullImage.size())
      os << "\n";
  }
  os << "};\n\n";
  os << "#endif // TAPESTRY_SYSTEM_CONFIG_H\n";

  return true;
}

//===----------------------------------------------------------------------===//
// Multi-core config image generation
//===----------------------------------------------------------------------===//

MultiCoreConfigImage
generateMultiCoreConfig(const MultiCoreCompilationDesc &compilation) {
  MultiCoreConfigImage image;

  llvm::outs() << "multi-core-configgen: assembling config image for "
               << compilation.coreDescs.size() << " cores\n";

  if (compilation.coreDescs.empty()) {
    llvm::errs() << "multi-core-configgen: no core descriptions\n";
    return image;
  }

  // Collect per-core config blobs and compute max size for uniform stride.
  uint32_t maxCoreConfigSize = 0;
  for (const auto &coreDesc : compilation.coreDescs) {
    uint32_t blobSize =
        static_cast<uint32_t>(coreDesc.aggregateConfigBlob.size());
    uint32_t aligned = alignToWord(blobSize);
    if (aligned > maxCoreConfigSize)
      maxCoreConfigSize = aligned;
  }

  // If no core has config data, use a minimum stride of 4 bytes.
  if (maxCoreConfigSize == 0)
    maxCoreConfigSize = 4;

  uint32_t numCores = static_cast<uint32_t>(compilation.coreDescs.size());

  // Header: 8 bytes (2 x uint32).
  const uint32_t headerSize = 8;

  // Compute per-core base offsets.
  uint32_t offset = headerSize;
  for (const auto &coreDesc : compilation.coreDescs) {
    CoreConfigEntry entry;
    entry.coreInstanceName = coreDesc.coreInstanceName;
    entry.coreType = coreDesc.coreType;
    entry.configBlob = coreDesc.aggregateConfigBlob;
    entry.slices = coreDesc.configSlices;
    entry.baseOffset = offset;

    // Adjust slice word offsets to be relative to the system image.
    uint32_t coreWordBase = offset / 4;
    for (auto &slice : entry.slices) {
      slice.wordOffset += coreWordBase;
    }

    image.coreConfigs.push_back(std::move(entry));
    offset += maxCoreConfigSize;
  }

  // System config section (placeholder for NoC routing tables, DMA config).
  // The offset and structure are maintained for forward compatibility even
  // though the system config blob is currently empty.
  image.systemConfig.baseOffset = offset;

  // Assemble the full binary image.
  uint32_t totalSize =
      offset +
      static_cast<uint32_t>(image.systemConfig.systemConfigBlob.size());
  totalSize = alignToWord(totalSize);
  // Ensure minimum image size of header + core sections.
  if (totalSize < headerSize)
    totalSize = headerSize;
  image.fullImage.resize(totalSize, 0);

  // Write header.
  writeU32LE(image.fullImage, 0, numCores);
  writeU32LE(image.fullImage, 4, maxCoreConfigSize);

  // Write per-core config blobs.
  for (const auto &cc : image.coreConfigs) {
    if (!cc.configBlob.empty()) {
      std::memcpy(image.fullImage.data() + cc.baseOffset, cc.configBlob.data(),
                  cc.configBlob.size());
    }
  }

  // Write system config blob (if any).
  if (!image.systemConfig.systemConfigBlob.empty()) {
    std::memcpy(image.fullImage.data() + image.systemConfig.baseOffset,
                image.systemConfig.systemConfigBlob.data(),
                image.systemConfig.systemConfigBlob.size());
  }

  image.totalWords = totalSize / 4;

  llvm::outs() << "multi-core-configgen: assembled image: " << numCores
               << " cores, " << maxCoreConfigSize << " bytes/core stride, "
               << totalSize << " total bytes (" << image.totalWords
               << " words)\n";

  return image;
}

} // namespace svgen
} // namespace loom
