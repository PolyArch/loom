#ifndef __DEV_FCC_FCCCGRADEVICE_HH__
#define __DEV_FCC_FCCCGRADEVICE_HH__

#include "dev/io_device.hh"
#include "params/FccCgraDevice.hh"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace gem5
{

class FccCgraDevice : public BasicPioDevice
{
  public:
    using Params = FccCgraDeviceParams;

    explicit FccCgraDevice(const Params &p);

    Tick read(PacketPtr pkt) override;
    Tick write(PacketPtr pkt) override;

  private:
    struct OutputSlotData {
        std::vector<uint64_t> data;
        std::vector<uint16_t> tags;
    };

    std::string runtimeManifest;
    std::string fccBinary;
    std::string bridgeScript;
    std::string workDir;

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

    std::unordered_map<unsigned, OutputSlotData> outputs;

    void resetDevice();
    uint32_t readRegister32(Addr offset);
    void writeRegister32(Addr offset, uint32_t value);
    bool runInvocation();
};

} // namespace gem5

#endif // __DEV_FCC_FCCCGRADEVICE_HH__
