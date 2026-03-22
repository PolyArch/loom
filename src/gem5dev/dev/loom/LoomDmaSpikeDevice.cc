#include "dev/loom/LoomDmaSpikeDevice.hh"

#include "base/logging.hh"
#include "mem/packet.hh"

namespace gem5
{

LoomDmaSpikeDevice::LoomDmaSpikeDevice(const Params &p)
    : DmaDevice(p), pioAddr_(p.pio_addr), pioDelay_(p.pio_latency),
      pioSize_(p.pio_size),
      readDoneEvent_([this] { onReadComplete(); }, name() + ".readDone"),
      writeDoneEvent_([this] { onWriteComplete(); }, name() + ".writeDone")
{
    resetDevice();
}

AddrRangeList
LoomDmaSpikeDevice::getAddrRanges() const
{
    return {RangeSize(pioAddr_, pioSize_)};
}

Tick
LoomDmaSpikeDevice::read(PacketPtr pkt)
{
    if (pkt->getSize() != 4)
        panic("LoomDmaSpikeDevice only supports 32-bit MMIO accesses");
    Addr offset = pkt->getAddr() - pioAddr_;
    pkt->setUintX(readReg32(offset), ByteOrder::little);
    pkt->makeAtomicResponse();
    return pioDelay_;
}

Tick
LoomDmaSpikeDevice::write(PacketPtr pkt)
{
    if (pkt->getSize() != 4)
        panic("LoomDmaSpikeDevice only supports 32-bit MMIO accesses");
    Addr offset = pkt->getAddr() - pioAddr_;
    writeReg32(offset, static_cast<uint32_t>(pkt->getUintX(ByteOrder::little)));
    pkt->makeAtomicResponse();
    return pioDelay_;
}

void
LoomDmaSpikeDevice::resetDevice()
{
    statusReg_ = 0;
    errorCode_ = 0;
    srcAddr_ = 0;
    dstAddr_ = 0;
    transferSize_ = 0;
    checksum_ = 0;
    dmaBuffer_.clear();
}

uint32_t
LoomDmaSpikeDevice::readReg32(Addr offset) const
{
    switch (offset) {
      case kRegStatus:
        return statusReg_;
      case kRegSrcLo:
        return static_cast<uint32_t>(srcAddr_ & 0xffffffffULL);
      case kRegSrcHi:
        return static_cast<uint32_t>(srcAddr_ >> 32);
      case kRegDstLo:
        return static_cast<uint32_t>(dstAddr_ & 0xffffffffULL);
      case kRegDstHi:
        return static_cast<uint32_t>(dstAddr_ >> 32);
      case kRegSize:
        return transferSize_;
      case kRegChecksumLo:
        return static_cast<uint32_t>(checksum_ & 0xffffffffULL);
      case kRegChecksumHi:
        return static_cast<uint32_t>(checksum_ >> 32);
      case kRegErrorCode:
        return errorCode_;
      default:
        return 0;
    }
}

void
LoomDmaSpikeDevice::writeReg32(Addr offset, uint32_t value)
{
    if (offset == kRegControl) {
        if (value & kCtrlReset) {
            resetDevice();
            return;
        }
        if (value & kCtrlStart) {
            startTransfer();
            return;
        }
        return;
    }

    if ((statusReg_ & kStatusBusy) != 0)
        return;

    switch (offset) {
      case kRegSrcLo:
        srcAddr_ = (srcAddr_ & 0xffffffff00000000ULL) | value;
        return;
      case kRegSrcHi:
        srcAddr_ = (srcAddr_ & 0x00000000ffffffffULL) |
                   (static_cast<uint64_t>(value) << 32);
        return;
      case kRegDstLo:
        dstAddr_ = (dstAddr_ & 0xffffffff00000000ULL) | value;
        return;
      case kRegDstHi:
        dstAddr_ = (dstAddr_ & 0x00000000ffffffffULL) |
                   (static_cast<uint64_t>(value) << 32);
        return;
      case kRegSize:
        transferSize_ = value;
        return;
      default:
        return;
    }
}

void
LoomDmaSpikeDevice::finishError(uint32_t code)
{
    statusReg_ = kStatusError;
    errorCode_ = code;
}

void
LoomDmaSpikeDevice::startTransfer()
{
    if ((statusReg_ & kStatusBusy) != 0)
        return;
    if (transferSize_ == 0) {
        finishError(1);
        return;
    }
    dmaBuffer_.assign(transferSize_, 0);
    checksum_ = 0;
    errorCode_ = 0;
    statusReg_ = kStatusBusy;
    dmaRead(srcAddr_, static_cast<int>(transferSize_), &readDoneEvent_,
            dmaBuffer_.data(), 0);
}

void
LoomDmaSpikeDevice::recomputeChecksum()
{
    uint64_t accum = 0;
    for (uint8_t byte : dmaBuffer_)
        accum += static_cast<uint64_t>(byte);
    checksum_ = accum;
}

void
LoomDmaSpikeDevice::onReadComplete()
{
    if ((statusReg_ & kStatusBusy) == 0)
        return;
    recomputeChecksum();
    dmaWrite(dstAddr_, static_cast<int>(transferSize_), &writeDoneEvent_,
             dmaBuffer_.data(), 0);
}

void
LoomDmaSpikeDevice::onWriteComplete()
{
    if ((statusReg_ & kStatusBusy) == 0)
        return;
    statusReg_ = kStatusDone;
}

} // namespace gem5
