#ifndef LOOM_MULTICORESIM_DRAMMODEL_H
#define LOOM_MULTICORESIM_DRAMMODEL_H

#include <cstdint>
#include <deque>
#include <vector>

namespace loom {
namespace mcsim {

/// Simple DRAM model with fixed latency and bandwidth limiting.
///
/// Models external DRAM as a large byte-addressable memory with a fixed
/// access latency and a bandwidth cap. Requests beyond the bandwidth limit
/// are queued and serviced in order.
class DRAMModel {
public:
  DRAMModel(uint64_t sizeBytes, unsigned accessLatency,
            unsigned bandwidthBytesPerCycle);

  /// Describes a pending DRAM request in the queue.
  struct PendingRequest {
    uint64_t requestId;
    uint64_t address;
    uint64_t sizeBytes;
    bool isWrite;
    std::vector<uint8_t> writeData;
    uint64_t submitCycle;
    uint64_t completionCycle;
  };

  /// Submit a read request. Returns the request ID.
  uint64_t submitRead(uint64_t address, uint64_t sizeBytes,
                      uint64_t globalCycle);

  /// Submit a write request. Returns the request ID.
  uint64_t submitWrite(uint64_t address, const std::vector<uint8_t> &data,
                       uint64_t globalCycle);

  /// Check if a request has completed.
  bool isComplete(uint64_t requestId) const;

  /// Retrieve the read data for a completed read request.
  std::vector<uint8_t> getReadData(uint64_t requestId) const;

  /// Advance the DRAM state by one cycle.
  void tick(uint64_t globalCycle);

  /// Initialize DRAM contents at the given base address.
  void initialize(uint64_t baseAddr, const std::vector<uint8_t> &data);

  /// Direct read for initialization/debug (bypasses latency model).
  std::vector<uint8_t> directRead(uint64_t address, uint64_t sizeBytes) const;

  /// Direct write for initialization/debug (bypasses latency model).
  void directWrite(uint64_t address, const std::vector<uint8_t> &data);

  uint64_t getCapacity() const { return sizeBytes_; }
  uint64_t getReadCount() const { return readCount_; }
  uint64_t getWriteCount() const { return writeCount_; }
  uint64_t getTotalBytesTransferred() const { return totalBytesTransferred_; }

private:
  /// Compute the completion cycle for a new request.
  uint64_t computeCompletionCycle(uint64_t sizeBytes, uint64_t globalCycle);

  uint64_t sizeBytes_;
  unsigned accessLatency_;
  unsigned bandwidthBytesPerCycle_;
  std::vector<uint8_t> storage_;

  uint64_t nextRequestId_ = 1;

  /// Queue of in-flight requests.
  std::deque<PendingRequest> pendingRequests_;

  /// Completed requests (kept until data is retrieved).
  std::deque<PendingRequest> completedRequests_;

  /// The earliest cycle at which the DRAM bus is available for new transfers.
  uint64_t busBusyUntilCycle_ = 0;

  /// Statistics.
  uint64_t readCount_ = 0;
  uint64_t writeCount_ = 0;
  uint64_t totalBytesTransferred_ = 0;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_DRAMMODEL_H
