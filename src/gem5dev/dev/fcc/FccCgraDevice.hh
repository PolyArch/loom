#ifndef __DEV_FCC_FCCCGRADEVICE_HH__
#define __DEV_FCC_FCCCGRADEVICE_HH__

#include "dev/dma_device.hh"
#include "fcc/Simulator/SimRuntime.h"
#include "params/FccCgraDevice.hh"
#include "sim/eventq.hh"

#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace fcc {
namespace sim {
class CycleKernel;
struct RuntimeImage;
} // namespace sim
} // namespace fcc

namespace gem5
{

class FccCgraDevice : public DmaDevice
{
  public:
    using Params = FccCgraDeviceParams;

    explicit FccCgraDevice(const Params &p);

    AddrRangeList getAddrRanges() const override;
    Tick read(PacketPtr pkt) override;
    Tick write(PacketPtr pkt) override;

  private:
    enum class DirectPhase : uint8_t {
        Idle,
        ConfigLoad,
        Running,
        MemoryIO,
    };

    struct OutputSlotData {
        std::vector<uint64_t> data;
        std::vector<uint16_t> tags;
    };

    struct PendingKernelTransfer {
        fcc::sim::MemoryRequestRecord request;
        unsigned slot = 0;
        Addr physAddr = 0;
        std::vector<uint8_t> bytes;
    };

    const Addr pioAddr_;
    const Tick pioDelay_;
    const Addr pioSize_;
    const Tick accelCycleLatency_;
    const uint64_t maxBatchCycles_;
    const uint64_t maxInflightMemReqs_;
    std::string runtimeManifest;
    std::string simImagePath;
    std::string workDir;
    bool directRuntimeReady = false;

    uint32_t statusReg = 0;
    uint32_t errorCode = 0;
    uint64_t cycleCount = 0;
    uint32_t selectedOutputPort = 0;
    uint32_t selectedOutputIndex = 0;
    uint64_t invocationCount = 0;

    std::vector<uint32_t> configWords;
    uint64_t configBase = 0;
    uint32_t configWordCount = 0;
    uint64_t lastConfigLoadCycles_ = 0;
    uint64_t lastConfigLoadDmaRequestCount_ = 0;
    uint64_t lastConfigLoadDmaReadBytes_ = 0;
    Tick lastConfigLoadStartTick_ = 0;
    Tick lastConfigLoadEndTick_ = 0;
    Tick activeInvocationStartTick_ = 0;
    Tick memoryIoTicks_ = 0;
    Tick memoryIoWindowStartTick_ = 0;
    uint32_t memBase[8] = {};
    uint32_t memSize[8] = {};
    uint64_t scalarArgs[8] = {};
    std::vector<uint8_t> pendingConfigBytes_;

    DirectPhase directPhase_ = DirectPhase::Idle;
    std::filesystem::path activeInvocationDir_;
    std::filesystem::path activeReplyDir_;
    std::string activeErrorMessage_;
    std::unordered_map<unsigned, OutputSlotData> outputs;
    std::vector<int> regionSlotById_;
    std::deque<PendingKernelTransfer> pendingKernelTransfers_;
    std::unordered_map<uint64_t, PendingKernelTransfer> inflightKernelTransfers_;
    std::unique_ptr<fcc::sim::RuntimeImage> runtimeImage;
    std::unique_ptr<fcc::sim::CycleKernel> kernel;
    EventFunctionWrapper configLoadEvent_;
    EventFunctionWrapper invokeEvent_;

    void resetDevice();
    uint32_t readRegister32(Addr offset);
    void writeRegister32(Addr offset, uint32_t value);
    bool runInvocation();
    bool beginConfigLoad();
    void onConfigLoadComplete();
    bool startDirectInvocation(const std::filesystem::path &invocationDir,
                               const std::filesystem::path &replyDir);
    void onDirectInvoke();
    void issueKernelTransfers();
    void onKernelTransferComplete(uint64_t requestId);
    void scheduleNextKernelBatch(uint64_t cyclesAdvanced);
    void finishDirectInvocation(bool success,
                                const std::string &errorMessage);
    bool initializeDirectRuntime();
    bool writeDirectReplyArtifacts(
        const std::filesystem::path &replyDir, bool success,
        const std::string &errorMessage,
        const std::vector<unsigned> &outputSlots,
        const std::vector<unsigned> &memorySlots) const;
};

} // namespace gem5

#endif // __DEV_FCC_FCCCGRADEVICE_HH__
