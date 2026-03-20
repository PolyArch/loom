#include "dev/fcc/FccCgraDevice.hh"

#include "fcc/Simulator/CycleKernel.h"
#include "fcc/Simulator/RuntimeImage.h"
#include "fcc/Simulator/SimModule.h"

#include "base/addr_range.hh"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
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
constexpr uint64_t kMaxInvocationCycles = 1000000;

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

static bool writeBinaryFile(const std::filesystem::path &path,
                            const uint8_t *bytes, size_t size)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
        return false;
    out.write(reinterpret_cast<const char *>(bytes),
              static_cast<std::streamsize>(size));
    return static_cast<bool>(out);
}

template <typename T>
static bool writeScalarVectorFile(const std::filesystem::path &path,
                                  const std::vector<T> &values)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
        return false;
    for (const T &value : values) {
        out.write(reinterpret_cast<const char *>(&value), sizeof(T));
        if (!out)
            return false;
    }
    return true;
}

static bool writeTraceJson(const std::filesystem::path &path,
                           const fcc::sim::TraceDocument &doc)
{
    std::ofstream out(path);
    if (!out)
        return false;
    out << "{\n";
    out << "  \"version\": " << doc.version << ",\n";
    out << "  \"trace_kind\": \"fcc_cycle_trace\",\n";
    out << "  \"producer\": \"fcc_gem5_device\",\n";
    out << "  \"epoch_id\": " << doc.epochId << ",\n";
    out << "  \"invocation_id\": " << doc.invocationId << ",\n";
    out << "  \"core_id\": " << doc.coreId << ",\n";
    out << "  \"modules\": [\n";
    for (size_t idx = 0; idx < doc.modules.size(); ++idx) {
        const auto &module = doc.modules[idx];
        out << "    {\"hw_node_id\": " << module.hwNodeId
            << ", \"kind\": " << std::quoted(module.kind)
            << ", \"name\": " << std::quoted(module.name)
            << ", \"component_name\": " << std::quoted(module.componentName)
            << ", \"function_unit_name\": "
            << std::quoted(module.functionUnitName)
            << ", \"boundary_ordinal\": " << module.boundaryOrdinal << "}";
        if (idx + 1 != doc.modules.size())
            out << ",";
        out << "\n";
    }
    out << "  ],\n";
    out << "  \"events\": [\n";
    for (size_t idx = 0; idx < doc.events.size(); ++idx) {
        const auto &event = doc.events[idx];
        out << "    {\"cycle\": " << event.cycle
            << ", \"phase\": "
            << std::quoted(fcc::sim::simPhaseName(event.phase))
            << ", \"epoch_id\": " << event.epochId
            << ", \"invocation_id\": " << event.invocationId
            << ", \"core_id\": " << event.coreId
            << ", \"hw_node_id\": " << event.hwNodeId
            << ", \"event_kind\": "
            << std::quoted(fcc::sim::eventKindName(event.eventKind))
            << ", \"lane\": " << static_cast<unsigned>(event.lane)
            << ", \"flags\": " << event.flags
            << ", \"arg0\": " << event.arg0
            << ", \"arg1\": " << event.arg1 << "}";
        if (idx + 1 != doc.events.size())
            out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    return true;
}

static bool writeStatText(const std::filesystem::path &path,
                          bool success,
                          gem5::Tick cycleCount,
                          fcc::sim::BoundaryReason reason,
                          const std::string &errorMessage)
{
    std::ofstream out(path);
    if (!out)
        return false;
    out << "success: " << (success ? "true" : "false") << "\n";
    out << "termination: " << fcc::sim::boundaryReasonName(reason) << "\n";
    out << "total_cycles: " << cycleCount << "\n";
    out << "config_cycles: 0\n";
    out << "total_config_writes: 0\n";
    out << "error_message: "
        << (errorMessage.empty() ? "<none>" : errorMessage) << "\n";
    return true;
}

} // namespace

FccCgraDevice::FccCgraDevice(const Params &p)
    : DmaDevice(p), pioAddr_(p.pio_addr), pioDelay_(p.pio_latency),
      pioSize_(p.pio_size), runtimeManifest(p.runtime_manifest),
      simImagePath(p.sim_image), fccBinary(p.fcc_binary),
      bridgeScript(p.bridge_script), workDir(p.work_dir),
      runtimeImage(std::make_unique<fcc::sim::RuntimeImage>()),
      kernel(std::make_unique<fcc::sim::CycleKernel>()),
      invokeEvent_([this] { onDirectInvoke(); }, name() + ".invoke"),
      dmaReadDoneEvent_([this] { onDirectReadComplete(); },
                        name() + ".dmaReadDone"),
      dmaWriteDoneEvent_([this] { onDirectWriteComplete(); },
                         name() + ".dmaWriteDone")
{
    resetDevice();
    directRuntimeReady = initializeDirectRuntime();
}

AddrRangeList
FccCgraDevice::getAddrRanges() const
{
    return {RangeSize(pioAddr_, pioSize_)};
}

void
FccCgraDevice::resetDevice()
{
    if (invokeEvent_.scheduled())
        deschedule(invokeEvent_);
    if (dmaReadDoneEvent_.scheduled())
        deschedule(dmaReadDoneEvent_);
    if (dmaWriteDoneEvent_.scheduled())
        deschedule(dmaWriteDoneEvent_);
    statusReg = 0;
    errorCode = 0;
    configAddr = 0;
    cycleCount = 0;
    selectedOutputPort = 0;
    selectedOutputIndex = 0;
    configWords.clear();
    outputs.clear();
    directPhase_ = DirectPhase::Idle;
    activeInvocationDir_.clear();
    activeReplyDir_.clear();
    pendingReads_.clear();
    pendingWrites_.clear();
    pendingReadIndex_ = 0;
    pendingWriteIndex_ = 0;
    activeErrorMessage_.clear();
    if (directRuntimeReady && runtimeImage) {
        configWords = runtimeImage->configImage.words;
    }
    for (unsigned i = 0; i < 8; ++i) {
        memBase[i] = 0;
        memSize[i] = 0;
        scalarArgs[i] = 0;
    }
}

bool
FccCgraDevice::initializeDirectRuntime()
{
    if (simImagePath.empty())
        return false;
    if (!runtimeImage || !kernel)
        return false;
    std::string error;
    if (!fcc::sim::loadRuntimeImageBinary(simImagePath, *runtimeImage, error)) {
        warn("FccCgraDevice: failed to load sim image %s: %s",
             simImagePath, error);
        return false;
    }
    return true;
}

bool
FccCgraDevice::startDirectInvocation(const std::filesystem::path &invocationDir,
                                     const std::filesystem::path &replyDir)
{
    using namespace fcc::sim;

    if (!runtimeImage || !kernel) {
        statusReg = kStatusError;
        errorCode = 8;
        return false;
    }

    RuntimeImage image = *runtimeImage;
    if (!configWords.empty())
        image.configImage.words = configWords;

    kernel->resetAll();
    if (!kernel->build(image.staticModel) || !kernel->configure(image.configImage)) {
        statusReg = kStatusError;
        errorCode = 9;
        return false;
    }

    activeInvocationDir_ = invocationDir;
    activeReplyDir_ = replyDir;
    pendingReads_.clear();
    pendingWrites_.clear();
    pendingReadIndex_ = 0;
    pendingWriteIndex_ = 0;
    activeErrorMessage_.clear();
    directPhase_ = DirectPhase::Reading;

    const uint64_t invocationId = invocationCount - 1;
    kernel->setInvocationContext(1, invocationId);

    if (image.controlImage.startTokenPort >= 0) {
        SimToken token;
        token.data = 1;
        kernel->setInputTokens(
            static_cast<unsigned>(image.controlImage.startTokenPort), {token});
    }

    for (const auto &binding : image.controlImage.scalarBindings) {
        if (binding.slot >= 8)
            continue;
        SimToken token;
        token.data = scalarArgs[binding.slot];
        kernel->setInputTokens(binding.portIdx, {token});
    }

    regionBuffers.clear();
    regionBuffers.resize(image.controlImage.memoryBindings.size());
    std::set<unsigned> seenSlots;
    for (const auto &binding : image.controlImage.memoryBindings) {
        if (binding.slot >= 8 || memSize[binding.slot] == 0)
            continue;
        regionBuffers[binding.regionId].assign(memSize[binding.slot], 0);
        pendingReads_.push_back(
            {binding.slot, binding.regionId, regionBuffers[binding.regionId].size()});
        if (seenSlots.insert(binding.slot).second) {
            pendingWrites_.push_back(
                {binding.slot, binding.regionId, regionBuffers[binding.regionId].size()});
        }
    }

    if (pendingReads_.empty()) {
        directPhase_ = DirectPhase::Running;
        schedule(invokeEvent_, curTick() + 1);
    } else {
        issueNextDirectRead();
    }
    return true;
}

void
FccCgraDevice::issueNextDirectRead()
{
    if (pendingReadIndex_ >= pendingReads_.size()) {
        directPhase_ = DirectPhase::Running;
        schedule(invokeEvent_, curTick() + 1);
        return;
    }
    const auto &transfer = pendingReads_[pendingReadIndex_];
    dmaRead(memBase[transfer.slot], static_cast<int>(transfer.sizeBytes),
            &dmaReadDoneEvent_, regionBuffers[transfer.regionId].data(), 0);
}

void
FccCgraDevice::onDirectReadComplete()
{
    if (directPhase_ != DirectPhase::Reading)
        return;
    ++pendingReadIndex_;
    issueNextDirectRead();
}

void
FccCgraDevice::issueNextDirectWrite()
{
    if (pendingWriteIndex_ >= pendingWrites_.size()) {
        finishDirectInvocation(activeErrorMessage_.empty(), activeErrorMessage_);
        return;
    }
    const auto &transfer = pendingWrites_[pendingWriteIndex_];
    dmaWrite(memBase[transfer.slot], static_cast<int>(transfer.sizeBytes),
             &dmaWriteDoneEvent_, regionBuffers[transfer.regionId].data(), 0);
}

void
FccCgraDevice::onDirectWriteComplete()
{
    if (directPhase_ != DirectPhase::Writing)
        return;
    ++pendingWriteIndex_;
    issueNextDirectWrite();
}

void
FccCgraDevice::onDirectInvoke()
{
    using namespace fcc::sim;

    if (directPhase_ != DirectPhase::Running || !runtimeImage || !kernel)
        return;

    for (const auto &binding : runtimeImage->controlImage.memoryBindings) {
        if (binding.slot >= 8 || memSize[binding.slot] == 0 ||
            binding.regionId >= regionBuffers.size())
            continue;
        std::string err = kernel->setMemoryRegionBacking(
            binding.regionId, regionBuffers[binding.regionId].data(),
            regionBuffers[binding.regionId].size());
        if (!err.empty()) {
            finishDirectInvocation(false, err);
            return;
        }
    }

    BoundaryReason reason = kernel->runUntilBoundary(kMaxInvocationCycles);
    cycleCount = kernel->getCurrentCycle();

    bool success = (reason == BoundaryReason::InvocationDone);
    activeErrorMessage_.clear();
    if (!success) {
        activeErrorMessage_ = fcc::sim::boundaryReasonName(reason);
    } else if (!kernel->validateSuccessfulTermination(activeErrorMessage_)) {
        success = false;
    }

    outputs.clear();
    for (const auto &binding : runtimeImage->controlImage.outputBindings) {
        OutputSlotData slotData;
        const auto &tokens = kernel->getOutputTokens(binding.portIdx);
        for (const SimToken &token : tokens) {
            slotData.data.push_back(token.data);
            slotData.tags.push_back(token.hasTag ? token.tag : 0);
        }
        outputs[binding.slot] = std::move(slotData);
    }

    directPhase_ = DirectPhase::Writing;
    pendingWriteIndex_ = 0;
    if (pendingWrites_.empty()) {
        finishDirectInvocation(success, activeErrorMessage_);
        return;
    }
    if (!success) {
        finishDirectInvocation(false, activeErrorMessage_);
        return;
    }
    issueNextDirectWrite();
}

void
FccCgraDevice::finishDirectInvocation(bool success,
                                      const std::string &errorMessage)
{
    std::set<unsigned> outputSlotSet;
    for (const auto &entry : outputs)
        outputSlotSet.insert(entry.first);
    std::vector<unsigned> outputSlots(outputSlotSet.begin(), outputSlotSet.end());

    std::vector<unsigned> memorySlots;
    memorySlots.reserve(pendingWrites_.size());
    for (const auto &transfer : pendingWrites_) {
        if (std::find(memorySlots.begin(), memorySlots.end(), transfer.slot) ==
            memorySlots.end()) {
            memorySlots.push_back(transfer.slot);
        }
    }

    const std::filesystem::path tracePath = activeReplyDir_ / "trace.json";
    const std::filesystem::path statPath = activeReplyDir_ / "stat.txt";
    bool wroteArtifacts =
        writeTraceJson(tracePath, kernel->getTraceDocument()) &&
        writeStatText(statPath, success, cycleCount,
                      success ? fcc::sim::BoundaryReason::InvocationDone
                              : fcc::sim::BoundaryReason::Deadlock,
                      errorMessage) &&
        writeDirectReplyArtifacts(activeReplyDir_, success, errorMessage,
                                  outputSlots, memorySlots);

    if (wroteArtifacts) {
        for (unsigned slot : outputSlots) {
            const auto &slotData = outputs.at(slot);
            wroteArtifacts =
                writeScalarVectorFile<uint64_t>(
                    activeReplyDir_ /
                        ("output.slot" + std::to_string(slot) + ".data.bin"),
                    slotData.data) &&
                writeScalarVectorFile<uint16_t>(
                    activeReplyDir_ /
                        ("output.slot" + std::to_string(slot) + ".tags.bin"),
                    slotData.tags);
            if (!wroteArtifacts)
                break;
        }
    }
    if (wroteArtifacts) {
        for (const auto &transfer : pendingWrites_) {
            wroteArtifacts = writeBinaryFile(
                activeReplyDir_ /
                    ("memory.slot" + std::to_string(transfer.slot) + ".bin"),
                regionBuffers[transfer.regionId].data(),
                regionBuffers[transfer.regionId].size());
            if (!wroteArtifacts)
                break;
        }
    }

    directPhase_ = DirectPhase::Idle;
    if (!wroteArtifacts) {
        statusReg = kStatusError;
        errorCode = 11;
        return;
    }
    statusReg = success ? kStatusDone : kStatusError;
    if (!success)
        errorCode = 14;
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
    if ((statusReg & kStatusBusy) != 0)
        return false;
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

    if (directRuntimeReady)
        return startDirectInvocation(invocationDir, replyDir);

    if (fccBinary.empty() || bridgeScript.empty()) {
        statusReg = kStatusError;
        errorCode = 15;
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

bool
FccCgraDevice::writeDirectReplyArtifacts(
    const std::filesystem::path &replyDir, bool success,
    const std::string &errorMessage, const std::vector<unsigned> &outputSlots,
    const std::vector<unsigned> &memorySlots) const
{
    std::ofstream meta(replyDir / "reply.meta");
    if (!meta)
        return false;
    meta << "success=" << (success ? 1 : 0) << "\n";
    meta << "cycle_count=" << cycleCount << "\n";
    meta << "trace_path=" << (replyDir / "trace.json").string() << "\n";
    meta << "stat_path=" << (replyDir / "stat.txt").string() << "\n";
    if (!errorMessage.empty())
        meta << "error_message=" << errorMessage << "\n";
    for (unsigned slot : outputSlots)
        meta << "output_slot=" << slot << "\n";
    for (unsigned slot : memorySlots)
        meta << "memory_slot=" << slot << "\n";
    return true;
}

Tick
FccCgraDevice::read(PacketPtr pkt)
{
    const Addr offset = pkt->getAddr() - pioAddr_;
    if (pkt->getSize() != 4)
        panic("FccCgraDevice only supports 32-bit MMIO accesses");
    pkt->setUintX(readRegister32(offset), ByteOrder::little);
    pkt->makeAtomicResponse();
    return pioDelay_;
}

Tick
FccCgraDevice::write(PacketPtr pkt)
{
    const Addr offset = pkt->getAddr() - pioAddr_;
    if (pkt->getSize() != 4)
        panic("FccCgraDevice only supports 32-bit MMIO accesses");
    writeRegister32(offset,
                    static_cast<uint32_t>(pkt->getUintX(ByteOrder::little)));
    pkt->makeAtomicResponse();
    return pioDelay_;
}

} // namespace gem5
