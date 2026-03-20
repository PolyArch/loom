#ifndef __DEV_FCC_DMA_SPIKE_DEVICE_HH__
#define __DEV_FCC_DMA_SPIKE_DEVICE_HH__

#include <vector>

#include "base/addr_range.hh"
#include "dev/dma_device.hh"
#include "params/FccDmaSpikeDevice.hh"
#include "sim/eventq.hh"

namespace gem5
{

class FccDmaSpikeDevice : public DmaDevice
{
  public:
    PARAMS(FccDmaSpikeDevice);
    explicit FccDmaSpikeDevice(const Params &p);

    AddrRangeList getAddrRanges() const override;
    Tick read(PacketPtr pkt) override;
    Tick write(PacketPtr pkt) override;

  private:
    static constexpr uint32_t kStatusBusy = 1u << 0;
    static constexpr uint32_t kStatusDone = 1u << 1;
    static constexpr uint32_t kStatusError = 1u << 2;

    static constexpr Addr kRegStatus = 0x00;
    static constexpr Addr kRegControl = 0x04;
    static constexpr Addr kRegSrcLo = 0x08;
    static constexpr Addr kRegSrcHi = 0x0C;
    static constexpr Addr kRegDstLo = 0x10;
    static constexpr Addr kRegDstHi = 0x14;
    static constexpr Addr kRegSize = 0x18;
    static constexpr Addr kRegChecksumLo = 0x1C;
    static constexpr Addr kRegChecksumHi = 0x20;
    static constexpr Addr kRegErrorCode = 0x24;

    static constexpr uint32_t kCtrlStart = 1u << 0;
    static constexpr uint32_t kCtrlReset = 1u << 1;

    void resetDevice();
    void startTransfer();
    void finishError(uint32_t code);
    void onReadComplete();
    void onWriteComplete();
    uint32_t readReg32(Addr offset) const;
    void writeReg32(Addr offset, uint32_t value);
    void recomputeChecksum();

    const Addr pioAddr_;
    const Tick pioDelay_;
    const Addr pioSize_;

    uint32_t statusReg_ = 0;
    uint32_t errorCode_ = 0;
    uint64_t srcAddr_ = 0;
    uint64_t dstAddr_ = 0;
    uint32_t transferSize_ = 0;
    uint64_t checksum_ = 0;

    std::vector<uint8_t> dmaBuffer_;
    EventFunctionWrapper readDoneEvent_;
    EventFunctionWrapper writeDoneEvent_;
};

} // namespace gem5

#endif // __DEV_FCC_DMA_SPIKE_DEVICE_HH__
