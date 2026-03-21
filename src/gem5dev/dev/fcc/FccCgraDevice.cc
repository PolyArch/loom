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
constexpr Addr kRegConfigBaseLo = 0x08;
constexpr Addr kRegConfigBaseHi = 0x0C;
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
constexpr uint32_t kCtrlLoadConfig = 1u << 2;
constexpr uint64_t kMaxInvocationCycles = 1000000;

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

static void writeEscapedString(std::ostream &out, const std::string &value)
{
    out << '"';
    for (char ch : value) {
        switch (ch) {
          case '\\':
            out << "\\\\";
            break;
          case '"':
            out << "\\\"";
            break;
          case '\n':
            out << "\\n";
            break;
          case '\r':
            out << "\\r";
            break;
          case '\t':
            out << "\\t";
            break;
          default:
            out << ch;
            break;
        }
    }
    out << '"';
}

static bool writeAcceleratorStatsJson(
    const std::filesystem::path &path, const fcc::sim::AcceleratorStats &stats,
    bool success, fcc::sim::BoundaryReason reason,
    const std::string &errorMessage)
{
    std::ofstream out(path);
    if (!out)
        return false;

    out << "{\n";
    out << "  \"success\": " << (success ? "true" : "false") << ",\n";
    out << "  \"termination\": ";
    writeEscapedString(out, fcc::sim::boundaryReasonName(reason));
    out << ",\n";
    out << "  \"error_message\": ";
    writeEscapedString(out, errorMessage.empty() ? std::string("<none>") : errorMessage);
    out << ",\n";
    out << "  \"total_cycles\": " << stats.totalCycles << ",\n";
    out << "  \"kernel_cycles\": " << stats.kernelCycles << ",\n";
    out << "  \"device_elapsed_ticks\": " << stats.deviceElapsedTicks << ",\n";
    out << "  \"memory_io_ticks\": " << stats.memoryIoTicks << ",\n";
    out << "  \"load_request_count\": " << stats.loadRequestCount << ",\n";
    out << "  \"store_request_count\": " << stats.storeRequestCount << ",\n";
    out << "  \"load_bytes\": " << stats.loadBytes << ",\n";
    out << "  \"store_bytes\": " << stats.storeBytes << ",\n";

    out << "  \"config_load\": {\n";
    out << "    \"word_count\": " << stats.configLoad.wordCount << ",\n";
    out << "    \"byte_count\": " << stats.configLoad.byteCount << ",\n";
    out << "    \"words_per_cycle\": " << stats.configLoad.wordsPerCycle << ",\n";
    out << "    \"start_cycle\": " << stats.configLoad.startCycle << ",\n";
    out << "    \"end_cycle\": " << stats.configLoad.endCycle << ",\n";
    out << "    \"cycles\": " << stats.configLoad.cycles << ",\n";
    out << "    \"dma_request_count\": " << stats.configLoad.dmaRequestCount << ",\n";
    out << "    \"dma_read_bytes\": " << stats.configLoad.dmaReadBytes << ",\n";
    out << "    \"dma_start_tick\": " << stats.configLoad.dmaStartTick << ",\n";
    out << "    \"dma_end_tick\": " << stats.configLoad.dmaEndTick << ",\n";
    out << "    \"dma_elapsed_ticks\": " << stats.configLoad.dmaElapsedTicks << ",\n";
    out << "    \"kernel_launch_cycle\": " << stats.configLoad.kernelLaunchCycle << ",\n";
    out << "    \"kernel_first_active_cycle\": "
        << stats.configLoad.kernelFirstActiveCycle << ",\n";
    out << "    \"config_exec_overlap_cycles\": "
        << stats.configLoad.configExecOverlapCycles << ",\n";
    out << "    \"config_exec_exposed_cycles\": "
        << stats.configLoad.configExecExposedCycles << ",\n";
    out << "    \"config_overlap_efficiency\": "
        << stats.configLoad.configOverlapEfficiency << "\n";
    out << "  },\n";

    out << "  \"static_utilization\": {\n";
    out << "    \"total_modules\": " << stats.staticUtilization.totalModules << ",\n";
    out << "    \"configured_modules\": "
        << stats.staticUtilization.configuredModules << ",\n";
    out << "    \"total_function_units\": "
        << stats.staticUtilization.totalFunctionUnits << ",\n";
    out << "    \"mapped_function_units\": "
        << stats.staticUtilization.mappedFunctionUnits << ",\n";
    out << "    \"total_spatial_pes\": "
        << stats.staticUtilization.totalSpatialPEs << ",\n";
    out << "    \"used_spatial_pes\": "
        << stats.staticUtilization.usedSpatialPEs << ",\n";
    out << "    \"total_temporal_pes\": "
        << stats.staticUtilization.totalTemporalPEs << ",\n";
    out << "    \"used_temporal_pes\": "
        << stats.staticUtilization.usedTemporalPEs << ",\n";
    out << "    \"configured_module_ratio\": "
        << stats.staticUtilization.configuredModuleRatio << ",\n";
    out << "    \"mapped_function_unit_ratio\": "
        << stats.staticUtilization.mappedFunctionUnitRatio << ",\n";
    out << "    \"used_spatial_pe_ratio\": "
        << stats.staticUtilization.usedSpatialPERatio << ",\n";
    out << "    \"used_temporal_pe_ratio\": "
        << stats.staticUtilization.usedTemporalPERatio << "\n";
    out << "  },\n";

    out << "  \"dynamic_utilization\": {\n";
    out << "    \"kernel_cycles\": " << stats.dynamicUtilization.kernelCycles << ",\n";
    out << "    \"active_cycles\": " << stats.dynamicUtilization.activeCycles << ",\n";
    out << "    \"idle_cycles\": " << stats.dynamicUtilization.idleCycles << ",\n";
    out << "    \"fabric_active_cycles\": "
        << stats.dynamicUtilization.fabricActiveCycles << ",\n";
    out << "    \"need_mem_issue_cycles\": "
        << stats.dynamicUtilization.needMemIssueCycles << ",\n";
    out << "    \"wait_mem_resp_cycles\": "
        << stats.dynamicUtilization.waitMemRespCycles << ",\n";
    out << "    \"budget_boundary_count\": "
        << stats.dynamicUtilization.budgetBoundaryCount << ",\n";
    out << "    \"deadlock_boundary_count\": "
        << stats.dynamicUtilization.deadlockBoundaryCount << ",\n";
    out << "    \"max_inflight_memory_requests\": "
        << stats.dynamicUtilization.maxInflightMemoryRequests << ",\n";
    out << "    \"active_cycle_ratio\": "
        << stats.dynamicUtilization.activeCycleRatio << ",\n";
    out << "    \"fabric_active_ratio\": "
        << stats.dynamicUtilization.fabricActiveRatio << ",\n";
    out << "    \"mem_issue_ratio\": "
        << stats.dynamicUtilization.memIssueRatio << ",\n";
    out << "    \"mem_wait_ratio\": "
        << stats.dynamicUtilization.memWaitRatio << "\n";
    out << "  },\n";

    out << "  \"config_slices\": [\n";
    for (size_t idx = 0; idx < stats.configSlices.size(); ++idx) {
        const auto &slice = stats.configSlices[idx];
        out << "    {\"name\": ";
        writeEscapedString(out, slice.name);
        out << ", \"kind\": ";
        writeEscapedString(out, slice.kind);
        out << ", \"hw_node_id\": " << slice.hwNodeId
            << ", \"word_offset\": " << slice.wordOffset
            << ", \"word_count\": " << slice.wordCount
            << ", \"start_cycle\": " << slice.startCycle
            << ", \"end_cycle\": " << slice.endCycle << "}";
        if (idx + 1 != stats.configSlices.size())
            out << ",";
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"memory_regions\": [\n";
    for (size_t idx = 0; idx < stats.memoryRegions.size(); ++idx) {
        const auto &region = stats.memoryRegions[idx];
        out << "    {\"region_id\": " << region.regionId
            << ", \"slot\": " << region.slot
            << ", \"load_request_count\": " << region.loadRequestCount
            << ", \"store_request_count\": " << region.storeRequestCount
            << ", \"load_bytes\": " << region.loadBytes
            << ", \"store_bytes\": " << region.storeBytes
            << ", \"has_first_request_cycle\": "
            << (region.hasFirstRequestCycle ? "true" : "false")
            << ", \"first_request_cycle\": " << region.firstRequestCycle
            << ", \"has_last_completion_cycle\": "
            << (region.hasLastCompletionCycle ? "true" : "false")
            << ", \"last_completion_cycle\": " << region.lastCompletionCycle
            << "}";
        if (idx + 1 != stats.memoryRegions.size())
            out << ",";
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"modules\": [\n";
    for (size_t idx = 0; idx < stats.modules.size(); ++idx) {
        const auto &module = stats.modules[idx];
        out << "    {\"hw_node_id\": " << module.hwNodeId
            << ", \"name\": ";
        writeEscapedString(out, module.name);
        out << ", \"kind\": ";
        writeEscapedString(out, module.kind);
        out << ", \"component_name\": ";
        writeEscapedString(out, module.componentName);
        out << ", \"function_unit_name\": ";
        writeEscapedString(out, module.functionUnitName);
        out << ", \"configured\": " << (module.configured ? "true" : "false")
            << ", \"statically_used\": " << (module.staticallyUsed ? "true" : "false")
            << ", \"dynamically_used\": " << (module.dynamicallyUsed ? "true" : "false")
            << ", \"config_ready_cycle\": " << module.configReadyCycle
            << ", \"has_first_use_cycle\": "
            << (module.hasFirstUseCycle ? "true" : "false")
            << ", \"first_use_cycle\": " << module.firstUseCycle
            << ", \"config_slack_cycles\": " << module.configSlackCycles
            << ", \"active_cycles\": " << module.activeCycles
            << ", \"dynamic_utilization\": " << module.dynamicUtilization
            << ", \"stall_cycles_in\": " << module.stallCyclesIn
            << ", \"stall_cycles_out\": " << module.stallCyclesOut
            << ", \"tokens_in\": " << module.tokensIn
            << ", \"tokens_out\": " << module.tokensOut
            << ", \"logical_fire_count\": " << module.logicalFireCount
            << ", \"input_capture_count\": " << module.inputCaptureCount
            << ", \"output_transfer_count\": " << module.outputTransferCount
            << ", \"output_busy_cycles\": " << module.outputBusyCycles
            << ", \"input_latched_cycles\": " << module.inputLatchedCycles
            << ", \"counters\": {";
        for (size_t cidx = 0; cidx < module.counters.size(); ++cidx) {
            const auto &counter = module.counters[cidx];
            writeEscapedString(out, counter.name);
            out << ": " << counter.value;
            if (cidx + 1 != module.counters.size())
                out << ", ";
        }
        out << "}}";
        if (idx + 1 != stats.modules.size())
            out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    return true;
}

static bool writeStatText(const std::filesystem::path &path,
                          bool success,
                          const fcc::sim::AcceleratorStats &stats,
                          fcc::sim::BoundaryReason reason,
                          const std::string &errorMessage)
{
    std::ofstream out(path);
    if (!out)
        return false;
    out << "success: " << (success ? "true" : "false") << "\n";
    out << "termination: " << fcc::sim::boundaryReasonName(reason) << "\n";
    out << "total_cycles: " << stats.totalCycles << "\n";
    out << "kernel_cycles: " << stats.kernelCycles << "\n";
    out << "device_elapsed_ticks: " << stats.deviceElapsedTicks << "\n";
    out << "memory_io_ticks: " << stats.memoryIoTicks << "\n";
    out << "config_cycles: " << stats.configLoad.cycles << "\n";
    out << "total_config_writes: " << stats.configLoad.wordCount << "\n";
    out << "config_dma_request_count: " << stats.configLoad.dmaRequestCount << "\n";
    out << "config_dma_read_bytes: " << stats.configLoad.dmaReadBytes << "\n";
    out << "config_dma_elapsed_ticks: " << stats.configLoad.dmaElapsedTicks << "\n";
    out << "config_exec_overlap_cycles: "
        << stats.configLoad.configExecOverlapCycles << "\n";
    out << "config_exec_exposed_cycles: "
        << stats.configLoad.configExecExposedCycles << "\n";
    out << "config_overlap_efficiency: "
        << stats.configLoad.configOverlapEfficiency << "\n";
    out << "load_request_count: " << stats.loadRequestCount << "\n";
    out << "store_request_count: " << stats.storeRequestCount << "\n";
    out << "load_bytes: " << stats.loadBytes << "\n";
    out << "store_bytes: " << stats.storeBytes << "\n";
    out << "mapped_function_units: "
        << stats.staticUtilization.mappedFunctionUnits << "\n";
    out << "used_spatial_pes: "
        << stats.staticUtilization.usedSpatialPEs << "\n";
    out << "used_temporal_pes: "
        << stats.staticUtilization.usedTemporalPEs << "\n";
    out << "configured_module_ratio: "
        << stats.staticUtilization.configuredModuleRatio << "\n";
    out << "mapped_function_unit_ratio: "
        << stats.staticUtilization.mappedFunctionUnitRatio << "\n";
    out << "used_spatial_pe_ratio: "
        << stats.staticUtilization.usedSpatialPERatio << "\n";
    out << "used_temporal_pe_ratio: "
        << stats.staticUtilization.usedTemporalPERatio << "\n";
    out << "fabric_active_cycles: "
        << stats.dynamicUtilization.fabricActiveCycles << "\n";
    out << "active_cycle_ratio: "
        << stats.dynamicUtilization.activeCycleRatio << "\n";
    out << "fabric_active_ratio: "
        << stats.dynamicUtilization.fabricActiveRatio << "\n";
    out << "need_mem_issue_cycles: "
        << stats.dynamicUtilization.needMemIssueCycles << "\n";
    out << "mem_issue_ratio: "
        << stats.dynamicUtilization.memIssueRatio << "\n";
    out << "wait_mem_resp_cycles: "
        << stats.dynamicUtilization.waitMemRespCycles << "\n";
    out << "mem_wait_ratio: "
        << stats.dynamicUtilization.memWaitRatio << "\n";
    out << "max_inflight_memory_requests: "
        << stats.dynamicUtilization.maxInflightMemoryRequests << "\n";
    out << "error_message: "
        << (errorMessage.empty() ? "<none>" : errorMessage) << "\n";
    return true;
}

} // namespace

FccCgraDevice::FccCgraDevice(const Params &p)
    : DmaDevice(p), pioAddr_(p.pio_addr), pioDelay_(p.pio_latency),
      pioSize_(p.pio_size), accelCycleLatency_(p.accel_cycle_latency),
      maxBatchCycles_(std::max<uint64_t>(1, p.max_batch_cycles)),
      maxInflightMemReqs_(std::max<uint64_t>(1, p.max_inflight_mem_reqs)),
      runtimeManifest(p.runtime_manifest), simImagePath(p.sim_image),
      workDir(p.work_dir),
      runtimeImage(std::make_unique<fcc::sim::RuntimeImage>()),
      kernel(std::make_unique<fcc::sim::CycleKernel>()),
      configLoadEvent_([this] { onConfigLoadComplete(); },
                       name() + ".config_load"),
      invokeEvent_([this] { onDirectInvoke(); }, name() + ".invoke")
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
    statusReg = 0;
    errorCode = 0;
    cycleCount = 0;
    selectedOutputPort = 0;
    selectedOutputIndex = 0;
    configBase = 0;
    configWordCount = 0;
    lastConfigLoadCycles_ = 0;
    lastConfigLoadDmaRequestCount_ = 0;
    lastConfigLoadDmaReadBytes_ = 0;
    pendingConfigBytes_.clear();
    if (configLoadEvent_.scheduled())
        deschedule(configLoadEvent_);
    configWords.clear();
    outputs.clear();
    directPhase_ = DirectPhase::Idle;
    activeInvocationDir_.clear();
    activeReplyDir_.clear();
    activeErrorMessage_.clear();
    regionSlotById_.clear();
    pendingKernelTransfers_.clear();
    inflightKernelTransfers_.clear();
    if (directRuntimeReady && runtimeImage) {
        configWords = runtimeImage->configImage.words;
        configWordCount = static_cast<uint32_t>(configWords.size());
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
    activeErrorMessage_.clear();
    regionSlotById_.clear();
    pendingKernelTransfers_.clear();
    inflightKernelTransfers_.clear();
    directPhase_ = DirectPhase::Running;
    kernel->setUseExternalMemoryService(true);

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

    unsigned maxRegionId = 0;
    for (const auto &binding : image.controlImage.memoryBindings)
        maxRegionId = std::max(maxRegionId, binding.regionId);
    regionSlotById_.assign(maxRegionId + 1, -1);
    for (const auto &binding : image.controlImage.memoryBindings) {
        if (binding.slot >= 8 || memSize[binding.slot] == 0)
            continue;
        if (binding.regionId >= regionSlotById_.size())
            regionSlotById_.resize(binding.regionId + 1, -1);
        regionSlotById_[binding.regionId] = static_cast<int>(binding.slot);
        std::string err = kernel->bindExternalMemoryRegion(
            binding.regionId, memBase[binding.slot], memSize[binding.slot]);
        if (!err.empty()) {
            statusReg = kStatusError;
            errorCode = 10;
            return false;
        }
    }

    scheduleNextKernelBatch(1);
    return true;
}

void
FccCgraDevice::onDirectInvoke()
{
    using namespace fcc::sim;

    if (directPhase_ != DirectPhase::Running || !runtimeImage || !kernel)
        return;

    const uint64_t cycleBefore = kernel->getCurrentCycle();
    const uint64_t batchLimit =
        std::min<uint64_t>(kMaxInvocationCycles, maxBatchCycles_);
    BoundaryReason reason = kernel->runUntilBoundary(batchLimit);
    cycleCount = kernel->getCurrentCycle();
    const uint64_t cyclesAdvanced = cycleCount - cycleBefore;

    if (reason == BoundaryReason::BudgetHit) {
        scheduleNextKernelBatch(cyclesAdvanced);
        return;
    }
    if (reason == BoundaryReason::NeedMemIssue) {
        pendingKernelTransfers_.clear();
        for (const auto &request : kernel->drainOutgoingMemoryRequests()) {
            if (request.regionId >= regionSlotById_.size() ||
                regionSlotById_[request.regionId] < 0) {
                finishDirectInvocation(false,
                                       "gem5 device missing region slot binding");
                return;
            }
            PendingKernelTransfer transfer;
            transfer.request = request;
            transfer.slot = static_cast<unsigned>(regionSlotById_[request.regionId]);
            transfer.physAddr = request.byteAddr;
            transfer.bytes.assign(request.byteWidth, 0);
            if (request.kind == MemoryRequestKind::Store) {
                for (unsigned byte = 0; byte < request.byteWidth; ++byte)
                    transfer.bytes[byte] = static_cast<uint8_t>(
                        (request.data >> (byte * 8)) & 0xffu);
            }
            pendingKernelTransfers_.push_back(std::move(transfer));
        }
        if (pendingKernelTransfers_.empty()) {
            finishDirectInvocation(
                false,
                "cycle kernel requested memory issue but provided no request");
            return;
        }
        directPhase_ = DirectPhase::MemoryIO;
        memoryIoWindowStartTick_ = curTick();
        issueKernelTransfers();
        return;
    }
    if (reason == BoundaryReason::WaitMemResp)
        return;

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

    finishDirectInvocation(success, activeErrorMessage_);
}

void
FccCgraDevice::scheduleNextKernelBatch(uint64_t cyclesAdvanced)
{
    const uint64_t effectiveCycles = std::max<uint64_t>(1, cyclesAdvanced);
    const Tick delay = accelCycleLatency_ * effectiveCycles;
    schedule(invokeEvent_, curTick() + delay);
}

void
FccCgraDevice::issueKernelTransfers()
{
    if (directPhase_ != DirectPhase::MemoryIO)
        return;
    while (!pendingKernelTransfers_.empty() &&
           inflightKernelTransfers_.size() < maxInflightMemReqs_) {
        PendingKernelTransfer transfer = std::move(pendingKernelTransfers_.front());
        pendingKernelTransfers_.pop_front();
        const uint64_t requestId = transfer.request.requestId;
        auto inserted =
            inflightKernelTransfers_.emplace(requestId, std::move(transfer));
        auto &active = inserted.first->second;
        auto *doneEvent = new EventFunctionWrapper(
            [this, requestId] { onKernelTransferComplete(requestId); }, name(),
            true);
        if (active.request.kind == fcc::sim::MemoryRequestKind::Load) {
            dmaRead(active.physAddr, static_cast<int>(active.bytes.size()),
                    doneEvent, active.bytes.data(), 0);
        } else {
            dmaWrite(active.physAddr, static_cast<int>(active.bytes.size()),
                     doneEvent, active.bytes.data(), 0);
        }
    }
    if (!pendingKernelTransfers_.empty() || !inflightKernelTransfers_.empty())
        return;
    if (pendingKernelTransfers_.empty() && inflightKernelTransfers_.empty()) {
        if (memoryIoWindowStartTick_ != 0) {
            memoryIoTicks_ += curTick() - memoryIoWindowStartTick_;
            memoryIoWindowStartTick_ = 0;
        }
        directPhase_ = DirectPhase::Running;
        scheduleNextKernelBatch(1);
    }
}

void
FccCgraDevice::onKernelTransferComplete(uint64_t requestId)
{
    if (directPhase_ != DirectPhase::MemoryIO)
        return;
    auto it = inflightKernelTransfers_.find(requestId);
    if (it == inflightKernelTransfers_.end())
        return;
    PendingKernelTransfer transfer = std::move(it->second);
    inflightKernelTransfers_.erase(it);

    fcc::sim::MemoryCompletion completion;
    completion.requestId = transfer.request.requestId;
    completion.kind = transfer.request.kind;
    completion.regionId = transfer.request.regionId;
    completion.ownerNodeId = transfer.request.ownerNodeId;
    completion.tag = transfer.request.tag;
    completion.hasTag = transfer.request.hasTag;
    if (transfer.request.kind == fcc::sim::MemoryRequestKind::Load) {
        uint64_t value = 0;
        for (size_t byte = 0; byte < transfer.bytes.size(); ++byte)
            value |= uint64_t(transfer.bytes[byte]) << (byte * 8);
        completion.data = value;
    }
    kernel->pushMemoryCompletion(completion);
    issueKernelTransfers();
}

void
FccCgraDevice::finishDirectInvocation(bool success,
                                      const std::string &errorMessage)
{
    fcc::sim::AcceleratorStats accelStats;
    if (kernel) {
        accelStats = kernel->buildAcceleratorStats(
            /*configLoadStartCycle=*/0, lastConfigLoadCycles_,
            /*kernelLaunchCycle=*/lastConfigLoadCycles_,
            lastConfigLoadDmaRequestCount_, lastConfigLoadDmaReadBytes_);
        accelStats.configLoad.dmaStartTick = lastConfigLoadStartTick_;
        accelStats.configLoad.dmaEndTick = lastConfigLoadEndTick_;
        accelStats.configLoad.dmaElapsedTicks =
            (lastConfigLoadEndTick_ >= lastConfigLoadStartTick_)
                ? (lastConfigLoadEndTick_ - lastConfigLoadStartTick_)
                : 0;
        accelStats.deviceElapsedTicks =
            (curTick() >= activeInvocationStartTick_)
                ? (curTick() - activeInvocationStartTick_)
                : 0;
        accelStats.memoryIoTicks = memoryIoTicks_;
        for (auto &region : accelStats.memoryRegions) {
            if (region.regionId < regionSlotById_.size())
                region.slot = regionSlotById_[region.regionId];
        }
    }

    std::set<unsigned> outputSlotSet;
    for (const auto &entry : outputs)
        outputSlotSet.insert(entry.first);
    std::vector<unsigned> outputSlots(outputSlotSet.begin(), outputSlotSet.end());

    std::set<unsigned> memorySlotSet;
    for (int slot : regionSlotById_) {
        if (slot >= 0 && static_cast<unsigned>(slot) < 8 && memSize[slot] != 0)
            memorySlotSet.insert(static_cast<unsigned>(slot));
    }
    std::vector<unsigned> memorySlots(memorySlotSet.begin(), memorySlotSet.end());

    const std::filesystem::path tracePath = activeReplyDir_ / "trace.json";
    const std::filesystem::path statPath = activeReplyDir_ / "stat.txt";
    const std::filesystem::path statJsonPath = activeReplyDir_ / "stat.json";
    bool wroteArtifacts =
        writeTraceJson(tracePath, kernel->getTraceDocument()) &&
        writeStatText(statPath, success, accelStats,
                      success ? fcc::sim::BoundaryReason::InvocationDone
                              : fcc::sim::BoundaryReason::Deadlock,
                      errorMessage) &&
        writeAcceleratorStatsJson(
            statJsonPath, accelStats, success,
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
        for (unsigned slot : memorySlots) {
            std::vector<uint8_t> bytes(memSize[slot], 0);
            sys->physProxy.readBlob(memBase[slot], bytes.data(), bytes.size());
            wroteArtifacts =
                writeBinaryFile(activeReplyDir_ /
                                    ("memory.slot" + std::to_string(slot) + ".bin"),
                                bytes.data(), bytes.size());
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
      case kRegConfigBaseLo:
        return static_cast<uint32_t>(configBase & 0xffffffffu);
      case kRegConfigBaseHi:
        return static_cast<uint32_t>(configBase >> 32);
      case kRegConfigSize:
        return configWordCount;
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
        if (value & kCtrlLoadConfig) {
            beginConfigLoad();
            return;
        }
        if (value & kCtrlStart) {
            runInvocation();
            return;
        }
        return;
    }

    if (offset == kRegConfigBaseLo) {
        configBase = (configBase & 0xffffffff00000000ULL) |
                     static_cast<uint64_t>(value);
        return;
    }
    if (offset == kRegConfigBaseHi) {
        configBase = (configBase & 0x00000000ffffffffULL) |
                     (static_cast<uint64_t>(value) << 32);
        return;
    }
    if (offset == kRegConfigSize) {
        configWordCount = value;
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
FccCgraDevice::beginConfigLoad()
{
    if ((statusReg & kStatusBusy) != 0)
        return false;
    if (directPhase_ != DirectPhase::Idle)
        return false;
    if (configWordCount == 0) {
        configWords.clear();
        lastConfigLoadCycles_ = 0;
        lastConfigLoadDmaRequestCount_ = 0;
        lastConfigLoadDmaReadBytes_ = 0;
        lastConfigLoadStartTick_ = curTick();
        lastConfigLoadEndTick_ = curTick();
        statusReg = 0;
        errorCode = 0;
        return true;
    }

    statusReg = kStatusBusy;
    errorCode = 0;
    directPhase_ = DirectPhase::ConfigLoad;
    lastConfigLoadStartTick_ = curTick();
    lastConfigLoadDmaRequestCount_ = 1;
    lastConfigLoadDmaReadBytes_ =
        static_cast<uint64_t>(configWordCount) * sizeof(uint32_t);
    pendingConfigBytes_.assign(static_cast<size_t>(configWordCount) * sizeof(uint32_t),
                               0);
    dmaRead(configBase, static_cast<int>(pendingConfigBytes_.size()),
            &configLoadEvent_, pendingConfigBytes_.data(), 0);
    return true;
}

void
FccCgraDevice::onConfigLoadComplete()
{
    if (directPhase_ != DirectPhase::ConfigLoad)
        return;
    configWords.clear();
    configWords.reserve(configWordCount);
    for (uint32_t idx = 0; idx < configWordCount; ++idx) {
        size_t base = static_cast<size_t>(idx) * sizeof(uint32_t);
        uint32_t word = static_cast<uint32_t>(pendingConfigBytes_[base]) |
                        (static_cast<uint32_t>(pendingConfigBytes_[base + 1]) << 8) |
                        (static_cast<uint32_t>(pendingConfigBytes_[base + 2]) << 16) |
                        (static_cast<uint32_t>(pendingConfigBytes_[base + 3]) << 24);
        configWords.push_back(word);
    }
    const uint64_t wordsPerCycle =
        kernel ? std::max<unsigned>(1, kernel->getConfigWordsPerCycle()) : 1;
    lastConfigLoadCycles_ =
        (static_cast<uint64_t>(configWordCount) + wordsPerCycle - 1) /
        wordsPerCycle;
    lastConfigLoadEndTick_ = curTick();
    pendingConfigBytes_.clear();
    directPhase_ = DirectPhase::Idle;
    statusReg = 0;
}

bool
FccCgraDevice::runInvocation()
{
    if ((statusReg & kStatusBusy) != 0)
        return false;
    statusReg = kStatusBusy;
    errorCode = 0;
    outputs.clear();
    activeInvocationStartTick_ = curTick();
    memoryIoTicks_ = 0;
    memoryIoWindowStartTick_ = 0;

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
    try {
        std::filesystem::create_directories(replyDir);
    } catch (...) {
        statusReg = kStatusError;
        errorCode = 1;
        return false;
    }

    if (!directRuntimeReady) {
        statusReg = kStatusError;
        errorCode = 15;
        return false;
    }
    return startDirectInvocation(invocationDir, replyDir);
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
    meta << "stat_json_path=" << (replyDir / "stat.json").string() << "\n";
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
