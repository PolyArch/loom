#ifndef LOOM_MULTICORESIM_L2BANKMODEL_H
#define LOOM_MULTICORESIM_L2BANKMODEL_H

#include <cstdint>
#include <deque>
#include <vector>

namespace loom {
namespace mcsim {

/// Shared L2 bank model with busy tracking and conflict detection.
///
/// Each bank occupies a fixed number of cycles per access. If a new access
/// arrives while the bank is busy, it is queued and the additional latency
/// is counted as a bank conflict.
class L2BankModel {
public:
  L2BankModel(unsigned bankId, uint64_t sizeBytes, unsigned accessLatency,
              unsigned widthBytes);

  /// Read from the bank at the given bank-local offset.
  /// Returns the number of cycles until data is ready.
  unsigned read(uint64_t bankOffset, unsigned sizeBytes, uint64_t globalCycle);

  /// Write to the bank at the given bank-local offset.
  /// Returns the number of cycles until the write completes.
  unsigned write(uint64_t bankOffset, uint64_t data, unsigned sizeBytes,
                 uint64_t globalCycle);

  /// Read a block of bytes from the bank (for DMA bulk transfers).
  std::vector<uint8_t> readBlock(uint64_t bankOffset,
                                 uint64_t sizeBytes) const;

  /// Write a block of bytes to the bank (for DMA bulk transfers).
  void writeBlock(uint64_t bankOffset, const std::vector<uint8_t> &data);

  /// Check if the bank is currently processing an access.
  bool isBusy(uint64_t globalCycle) const;

  /// Advance the bank state by one cycle.
  void tick(uint64_t globalCycle);

  /// Initialize bank contents from a byte vector.
  void initialize(const std::vector<uint8_t> &data);

  unsigned getBankId() const { return bankId_; }
  uint64_t getCapacity() const { return sizeBytes_; }
  uint64_t getAccessCount() const { return accessCount_; }
  uint64_t getConflictCount() const { return conflictCount_; }

private:
  /// Compute the actual latency for an access, accounting for bank busy state.
  unsigned computeAccessLatency(uint64_t globalCycle);

  unsigned bankId_;
  uint64_t sizeBytes_;
  unsigned accessLatency_;
  unsigned widthBytes_;
  std::vector<uint8_t> storage_;

  /// Cycle at which the bank becomes free.
  uint64_t busyUntilCycle_ = 0;

  /// Statistics.
  uint64_t accessCount_ = 0;
  uint64_t conflictCount_ = 0;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_L2BANKMODEL_H
