#ifndef __DEV_FCC_FCCCGRADEVICE_HH__
#define __DEV_FCC_FCCCGRADEVICE_HH__

#include "dev/dma_device.hh"
#include "params/FccCgraDevice.hh"
#include "sim/eventq.hh"

#include <cstdint>
#include <filesystem>
#include <memory>
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
        Reading,
        Running,
        Writing,
    };

    struct OutputSlotData {
        std::vector<uint64_t> data;
        std::vector<uint16_t> tags;
    };

    struct PendingRegionTransfer {
        unsigned slot = 0;
        unsigned regionId = 0;
        size_t sizeBytes = 0;
    };

    const Addr pioAddr_;
    const Tick pioDelay_;
    const Addr pioSize_;
    std::string runtimeManifest;
    std::string simImagePath;
    std::string fccBinary;
    std::string bridgeScript;
    std::string workDir;
    bool directRuntimeReady = false;

    uint32_t statusReg = 0;
    uint32_t errorCode = 0;
    uint32_t configAddr = 0;
    uint64_t cycleCount = 0;
    uint32_t selectedOutputPort = 0;
    uint32_t selectedOutputIndex = 0;
    uint64_t invocationCount = 0;

    std::vector<uint32_t> configWords;
    uint32_t memBase[8] = {};
    uint32_t memSize[8] = {};
    uint64_t scalarArgs[8] = {};

    DirectPhase directPhase_ = DirectPhase::Idle;
    std::filesystem::path activeInvocationDir_;
    std::filesystem::path activeReplyDir_;
    std::vector<PendingRegionTransfer> pendingReads_;
    std::vector<PendingRegionTransfer> pendingWrites_;
    size_t pendingReadIndex_ = 0;
    size_t pendingWriteIndex_ = 0;
    std::string activeErrorMessage_;
    std::unordered_map<unsigned, OutputSlotData> outputs;
    std::vector<std::vector<uint8_t>> regionBuffers;
    std::unique_ptr<fcc::sim::RuntimeImage> runtimeImage;
    std::unique_ptr<fcc::sim::CycleKernel> kernel;
    EventFunctionWrapper invokeEvent_;
    EventFunctionWrapper dmaReadDoneEvent_;
    EventFunctionWrapper dmaWriteDoneEvent_;

    void resetDevice();
    uint32_t readRegister32(Addr offset);
    void writeRegister32(Addr offset, uint32_t value);
    bool runInvocation();
    bool startDirectInvocation(const std::filesystem::path &invocationDir,
                               const std::filesystem::path &replyDir);
    void issueNextDirectRead();
    void issueNextDirectWrite();
    void onDirectReadComplete();
    void onDirectInvoke();
    void onDirectWriteComplete();
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
