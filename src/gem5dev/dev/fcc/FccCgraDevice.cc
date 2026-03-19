#include "dev/fcc/FccCgraDevice.hh"

#include <filesystem>
#include <fstream>
#include <sstream>

#include "base/logging.hh"
#include "mem/packet.hh"
#include "sim/system.hh"

namespace gem5
{

namespace
{

constexpr uint32_t kStatusBusy = 1u << 0;
constexpr uint32_t kStatusDone = 1u << 1;
constexpr uint32_t kStatusError = 1u << 2;

constexpr Addr kRegStatus = 0x00;
constexpr Addr kRegControl = 0x04;
constexpr Addr kRegConfigAddr = 0x08;
constexpr Addr kRegConfigData = 0x0C;
constexpr Addr kRegConfigSize = 0x10;
constexpr Addr kRegMemBase0 = 0x20;
constexpr Addr kRegMemSize0 = 0x24;
constexpr Addr kRegArg0 = 0x80;
constexpr Addr kRegOutputPort = 0x100;
constexpr Addr kRegOutputIndex = 0x104;
constexpr Addr kRegOutputCount = 0x108;
constexpr Addr kRegOutputDataLo = 0x10C;
constexpr Addr kRegOutputDataHi = 0x110;
constexpr Addr kRegOutputTag = 0x114;
constexpr Addr kRegCycleCount = 0xF0;
constexpr Addr kRegErrorCode = 0xF4;

constexpr uint32_t kCtrlStart = 1u << 0;
constexpr uint32_t kCtrlReset = 1u << 1;

static std::string shellQuote(const std::string &value)
{
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'')
            quoted += "'\\''";
        else
            quoted += c;
    }
    quoted += "'";
    return quoted;
}

static bool readBinaryFile(const std::filesystem::path &path,
                           std::vector<uint8_t> &bytes)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    bytes.assign(std::istreambuf_iterator<char>(in),
                 std::istreambuf_iterator<char>());
    return true;
}

static bool readU64File(const std::filesystem::path &path,
                        std::vector<uint64_t> &values)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    values.clear();
    while (true) {
        uint64_t value = 0;
        in.read(reinterpret_cast<char *>(&value), sizeof(value));
        if (!in)
            break;
        values.push_back(value);
    }
    return true;
}

static bool readU16File(const std::filesystem::path &path,
                        std::vector<uint16_t> &values)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    values.clear();
    while (true) {
        uint16_t value = 0;
        in.read(reinterpret_cast<char *>(&value), sizeof(value));
        if (!in)
            break;
        values.push_back(value);
    }
    return true;
}

} // namespace

FccCgraDevice::FccCgraDevice(const Params &p)
    : BasicPioDevice(p, p.pio_size), runtimeManifest(p.runtime_manifest),
      fccBinary(p.fcc_binary), bridgeScript(p.bridge_script),
      workDir(p.work_dir)
{
    resetDevice();
}

void
FccCgraDevice::resetDevice()
{
    statusReg = 0;
    errorCode = 0;
    configAddr = 0;
    cycleCount = 0;
    selectedOutputPort = 0;
    selectedOutputIndex = 0;
    configWords.clear();
    outputs.clear();
    for (unsigned i = 0; i < 8; ++i) {
        memBase[i] = 0;
        memSize[i] = 0;
        scalarArgs[i] = 0;
    }
}

uint32_t
FccCgraDevice::readRegister32(Addr offset)
{
    switch (offset) {
      case kRegStatus:
        return statusReg;
      case kRegConfigSize:
        return static_cast<uint32_t>(configWords.size());
      case kRegOutputCount: {
        auto it = outputs.find(selectedOutputPort);
        if (it == outputs.end())
            return 0;
        return static_cast<uint32_t>(it->second.data.size());
      }
      case kRegOutputDataLo: {
        auto it = outputs.find(selectedOutputPort);
        if (it == outputs.end() ||
            selectedOutputIndex >= it->second.data.size()) {
            return 0;
        }
        return static_cast<uint32_t>(it->second.data[selectedOutputIndex] &
                                     0xffffffffu);
      }
      case kRegOutputDataHi: {
        auto it = outputs.find(selectedOutputPort);
        if (it == outputs.end() ||
            selectedOutputIndex >= it->second.data.size()) {
            return 0;
        }
        return static_cast<uint32_t>(it->second.data[selectedOutputIndex] >> 32);
      }
      case kRegOutputTag: {
        auto it = outputs.find(selectedOutputPort);
        if (it == outputs.end() ||
            selectedOutputIndex >= it->second.tags.size()) {
            return 0;
        }
        return static_cast<uint32_t>(it->second.tags[selectedOutputIndex]);
      }
      case kRegCycleCount:
        return static_cast<uint32_t>(cycleCount & 0xffffffffu);
      case kRegErrorCode:
        return errorCode;
      default:
        break;
    }

    if (offset >= kRegMemBase0 && offset < kRegArg0) {
        unsigned slot = static_cast<unsigned>((offset - kRegMemBase0) / 0x10);
        unsigned lane = static_cast<unsigned>((offset - kRegMemBase0) % 0x10);
        if (slot < 8) {
            if (lane == 0)
                return memBase[slot];
            if (lane == 4)
                return memSize[slot];
        }
    }

    if (offset >= kRegArg0 && offset < kRegOutputPort) {
        unsigned slot = static_cast<unsigned>((offset - kRegArg0) / 0x08);
        unsigned lane = static_cast<unsigned>((offset - kRegArg0) % 0x08);
        if (slot < 8) {
            if (lane == 0)
                return static_cast<uint32_t>(scalarArgs[slot] & 0xffffffffu);
            if (lane == 4)
                return static_cast<uint32_t>(scalarArgs[slot] >> 32);
        }
    }

    return 0;
}

void
FccCgraDevice::writeRegister32(Addr offset, uint32_t value)
{
    if (offset == kRegControl) {
        if (value & kCtrlReset) {
            resetDevice();
            return;
        }
        if (value & kCtrlStart) {
            runInvocation();
            return;
        }
        return;
    }

    if (offset == kRegConfigAddr) {
        configAddr = value;
        return;
    }
    if (offset == kRegConfigData) {
        if (configWords.size() <= configAddr)
            configWords.resize(configAddr + 1, 0);
        configWords[configAddr] = value;
        return;
    }
    if (offset == kRegOutputPort) {
        selectedOutputPort = value;
        return;
    }
    if (offset == kRegOutputIndex) {
        selectedOutputIndex = value;
        return;
    }

    if (offset >= kRegMemBase0 && offset < kRegArg0) {
        unsigned slot = static_cast<unsigned>((offset - kRegMemBase0) / 0x10);
        unsigned lane = static_cast<unsigned>((offset - kRegMemBase0) % 0x10);
        if (slot < 8) {
            if (lane == 0)
                memBase[slot] = value;
            else if (lane == 4)
                memSize[slot] = value;
        }
        return;
    }

    if (offset >= kRegArg0 && offset < kRegOutputPort) {
        unsigned slot = static_cast<unsigned>((offset - kRegArg0) / 0x08);
        unsigned lane = static_cast<unsigned>((offset - kRegArg0) % 0x08);
        if (slot < 8) {
            if (lane == 0)
                scalarArgs[slot] = (scalarArgs[slot] & 0xffffffff00000000ULL) |
                                   static_cast<uint64_t>(value);
            else if (lane == 4)
                scalarArgs[slot] = (scalarArgs[slot] & 0x00000000ffffffffULL) |
                                   (static_cast<uint64_t>(value) << 32);
        }
    }
}

bool
FccCgraDevice::runInvocation()
{
    statusReg = kStatusBusy;
    errorCode = 0;
    outputs.clear();

    try {
        std::filesystem::create_directories(workDir);
    } catch (...) {
        statusReg = kStatusError;
        errorCode = 1;
        return false;
    }

    std::filesystem::path invocationDir =
        std::filesystem::path(workDir) /
        ("invoke-" + std::to_string(invocationCount++));
    std::filesystem::path replyDir = invocationDir / "reply";
    std::filesystem::path helperDir = invocationDir / "helper";
    std::filesystem::path requestPath = invocationDir / "request.json";
    try {
        std::filesystem::create_directories(replyDir);
        std::filesystem::create_directories(helperDir);
    } catch (...) {
        statusReg = kStatusError;
        errorCode = 1;
        return false;
    }

    std::ofstream request(requestPath);
    if (!request) {
        statusReg = kStatusError;
        errorCode = 2;
        return false;
    }

    request << "{\n";
    request << "  \"start_token_count\": 1,\n";
    request << "  \"config_words\": [";
    for (size_t i = 0; i < configWords.size(); ++i) {
        if (i)
            request << ", ";
        request << configWords[i];
    }
    request << "],\n";
    request << "  \"scalar_args\": [\n";
    for (unsigned slot = 0; slot < 8; ++slot) {
        request << "    {\"slot\": " << slot << ", \"data\": ["
                << scalarArgs[slot] << "]}";
        request << (slot == 7 ? "\n" : ",\n");
    }
    request << "  ],\n";
    request << "  \"memory_regions\": [\n";
    bool firstRegion = true;
    for (unsigned slot = 0; slot < 8; ++slot) {
        if (memSize[slot] == 0)
            continue;
        std::vector<uint8_t> bytes(memSize[slot], 0);
        sys->physProxy.readBlob(memBase[slot], bytes.data(), bytes.size());
        if (!firstRegion)
            request << ",\n";
        firstRegion = false;
        request << "    {\"slot\": " << slot << ", \"bytes\": [";
        for (size_t i = 0; i < bytes.size(); ++i) {
            if (i)
                request << ", ";
            request << static_cast<unsigned>(bytes[i]);
        }
        request << "]}";
    }
    if (!firstRegion)
        request << "\n";
    request << "  ]\n";
    request << "}\n";
    request.close();

    std::ostringstream cmd;
    cmd << "python3 " << shellQuote(bridgeScript) << " --fcc "
        << shellQuote(fccBinary) << " --runtime-manifest "
        << shellQuote(runtimeManifest) << " --request "
        << shellQuote(requestPath.string()) << " --reply-dir "
        << shellQuote(replyDir.string()) << " --work-dir "
        << shellQuote(helperDir.string());

    int rc = std::system(cmd.str().c_str());
    std::filesystem::path metaPath = replyDir / "reply.meta";
    if (rc != 0 || !std::filesystem::exists(metaPath)) {
        statusReg = kStatusError;
        errorCode = 3;
        return false;
    }

    bool success = false;
    std::vector<unsigned> outputSlots;
    std::vector<unsigned> memorySlots;
    {
        std::ifstream meta(metaPath);
        std::string line;
        while (std::getline(meta, line)) {
            auto pos = line.find('=');
            if (pos == std::string::npos)
                continue;
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            if (key == "success") {
                success = value == "1";
            } else if (key == "cycle_count") {
                cycleCount = std::stoull(value);
            } else if (key == "error_message") {
                if (!success && !value.empty())
                    errorCode = 4;
            } else if (key == "output_slot") {
                outputSlots.push_back(static_cast<unsigned>(std::stoul(value)));
            } else if (key == "memory_slot") {
                memorySlots.push_back(static_cast<unsigned>(std::stoul(value)));
            }
        }
    }

    for (unsigned slot : outputSlots) {
        OutputSlotData slotData;
        if (!readU64File(replyDir / ("output.slot" + std::to_string(slot) +
                                     ".data.bin"),
                         slotData.data)) {
            statusReg = kStatusError;
            errorCode = 5;
            return false;
        }
        readU16File(replyDir / ("output.slot" + std::to_string(slot) +
                                ".tags.bin"),
                    slotData.tags);
        outputs[slot] = std::move(slotData);
    }

    for (unsigned slot : memorySlots) {
        std::vector<uint8_t> bytes;
        if (!readBinaryFile(replyDir / ("memory.slot" + std::to_string(slot) +
                                        ".bin"),
                            bytes)) {
            statusReg = kStatusError;
            errorCode = 6;
            return false;
        }
        if (slot < 8 && memSize[slot] != 0) {
            size_t size = std::min<size_t>(bytes.size(), memSize[slot]);
            sys->physProxy.writeBlob(memBase[slot], bytes.data(), size);
        }
    }

    statusReg = success ? kStatusDone : kStatusError;
    if (!success && errorCode == 0)
        errorCode = 7;
    return success;
}

Tick
FccCgraDevice::read(PacketPtr pkt)
{
    const Addr offset = pkt->getAddr() - pioAddr;
    if (pkt->getSize() != 4)
        panic("FccCgraDevice only supports 32-bit MMIO accesses");
    pkt->setUintX(readRegister32(offset), ByteOrder::little);
    pkt->makeAtomicResponse();
    return pioDelay;
}

Tick
FccCgraDevice::write(PacketPtr pkt)
{
    const Addr offset = pkt->getAddr() - pioAddr;
    if (pkt->getSize() != 4)
        panic("FccCgraDevice only supports 32-bit MMIO accesses");
    writeRegister32(offset,
                    static_cast<uint32_t>(pkt->getUintX(ByteOrder::little)));
    pkt->makeAtomicResponse();
    return pioDelay;
}

} // namespace gem5
