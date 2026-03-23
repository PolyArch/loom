#ifndef LOOM_MULTICORESIM_SPMMODEL_H
#define LOOM_MULTICORESIM_SPMMODEL_H

#include <cstdint>
#include <vector>

namespace loom {
namespace mcsim {

/// Per-core scratchpad memory model.
///
/// Provides a flat byte-addressable memory with zero-wait-state access
/// for the local core. All accesses are single-cycle.
class SPMModel {
public:
  SPMModel(unsigned coreId, uint64_t sizeBytes);

  /// Read a value from the SPM at the given byte offset.
  /// sizeBytes must be 1, 2, 4, or 8.
  uint64_t read(uint64_t offset, unsigned sizeBytes) const;

  /// Write a value to the SPM at the given byte offset.
  /// sizeBytes must be 1, 2, 4, or 8.
  void write(uint64_t offset, uint64_t data, unsigned sizeBytes);

  /// Read a contiguous block of bytes (for DMA bulk transfers).
  std::vector<uint8_t> readBlock(uint64_t offset, uint64_t sizeBytes) const;

  /// Write a contiguous block of bytes (for DMA bulk transfers).
  void writeBlock(uint64_t offset, const std::vector<uint8_t> &data);

  /// Initialize the SPM contents from a byte vector.
  void initialize(const std::vector<uint8_t> &data);

  uint64_t getCapacity() const { return sizeBytes_; }
  unsigned getCoreId() const { return coreId_; }

  /// Return the highest written byte offset + 1 (approximate usage tracking).
  uint64_t getUsed() const { return highWaterMark_; }

private:
  unsigned coreId_;
  uint64_t sizeBytes_;
  std::vector<uint8_t> storage_;
  uint64_t highWaterMark_ = 0;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_SPMMODEL_H
