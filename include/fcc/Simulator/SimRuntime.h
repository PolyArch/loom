#ifndef FCC_SIMULATOR_SIMRUNTIME_H
#define FCC_SIMULATOR_SIMRUNTIME_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace fcc {
namespace sim {

enum class MemoryRequestKind : uint8_t {
  Load = 0,
  Store = 1,
};

struct MemoryCompletion {
  uint64_t requestId = 0;
  MemoryRequestKind kind = MemoryRequestKind::Load;
  unsigned regionId = 0;
  uint32_t ownerNodeId = 0;
  uint64_t data = 0;
  uint16_t tag = 0;
  bool hasTag = false;
};

struct MemoryRequestRecord {
  uint64_t requestId = 0;
  MemoryRequestKind kind = MemoryRequestKind::Load;
  unsigned regionId = 0;
  uint32_t ownerNodeId = 0;
  uint64_t byteAddr = 0;
  uint64_t data = 0;
  unsigned byteWidth = 0;
  uint16_t tag = 0;
  bool hasTag = false;
};

class SimRuntimeServices {
public:
  virtual ~SimRuntimeServices() = default;

  virtual uint64_t getCurrentCycle() const = 0;
  virtual unsigned getMemoryLatencyCycles(bool isExtMemory) const = 0;

  virtual bool issueMemoryLoad(uint32_t ownerNodeId, unsigned regionId,
                               uint64_t byteAddr, unsigned byteWidth,
                               uint16_t tag, bool hasTag,
                               uint64_t &requestId,
                               std::string &error) = 0;

  virtual bool issueMemoryStore(uint32_t ownerNodeId, unsigned regionId,
                                uint64_t byteAddr, uint64_t data,
                                unsigned byteWidth, uint16_t tag, bool hasTag,
                                uint64_t &requestId,
                                std::string &error) = 0;

  virtual bool takeMemoryCompletion(uint64_t requestId,
                                    MemoryCompletion &completion) = 0;

  virtual bool hasOutstandingMemoryRequest(uint64_t requestId) const = 0;
  virtual bool regionHasOutstandingRequests(unsigned regionId) const = 0;
  virtual bool bindMemoryRegion(unsigned regionId, uint8_t *data,
                                size_t sizeBytes,
                                std::string &error) = 0;
};

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_SIMRUNTIME_H
