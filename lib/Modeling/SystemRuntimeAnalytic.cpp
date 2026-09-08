#include "Evaluation/Models/SystemRuntimeAnalytic.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/MemoryConsistencyContract.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Hardware/Configuration/PackedConfigurationABI.h"
#include "Runtime/Gem5SimulationBinding.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <mutex>
#include <variant>

namespace loom::evaluation::models {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_runtime_model_invalid: " +
                                     message.str());
}

llvm::Error overflow(llvm::StringRef context) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_runtime_model_overflow: %s",
                                 context.str().c_str());
}

// The launch protocol shape owned by docs/spec-runtime-abi.md. The host
// submits one Thread Dispatch per activation (target select, reset,
// invocation address and size, start, then occurrence and status reads); the
// AccCore InstructionCore enters, programs the bridge descriptor and start,
// polls completion, and signals the dispatch device. These counts are the
// model's reading of that sequence; latencies come from the platform policy.
constexpr std::uint64_t kHostDispatchPioOperations = 10;
constexpr std::uint64_t kHostDispatchInstructionLeaves = 40;
constexpr std::uint64_t kInstructionCoreEntryInstructionLeaves = 30;
constexpr std::uint64_t kBridgeProgrammingPioOperations = 8;
constexpr std::uint64_t kCompletionPioOperations = 4;
// Cycles one executable leaf occupies a cached single-issue in-order core and
// a two-wide out-of-order core. The model identity pins these assumptions.
constexpr std::uint64_t kInOrderCyclesPerLeaf = 3;
constexpr std::uint64_t kOutOfOrderCyclesPerLeaf = 1;
constexpr std::uint64_t kFemtosecondsPerPicosecond = 1000;

llvm::Expected<std::uint64_t> checkedMul(std::uint64_t lhs, std::uint64_t rhs,
                                         llvm::StringRef context) {
  const auto product = llvm::checkedMulUnsigned(lhs, rhs);
  if (!product)
    return overflow(context);
  return *product;
}

llvm::Expected<std::uint64_t> checkedAdd(std::uint64_t lhs, std::uint64_t rhs,
                                         llvm::StringRef context) {
  const auto sum = llvm::checkedAddUnsigned(lhs, rhs);
  if (!sum)
    return overflow(context);
  return *sum;
}

std::uint64_t ceilDiv(std::uint64_t value, std::uint64_t divisor) {
  return value / divisor + (value % divisor != 0 ? 1 : 0);
}

llvm::Expected<std::uint64_t>
clockPeriodPicoseconds(const fabric::FabricSystemRootView &system,
                       const fabric::ClockDomainRef &clock) {
  const auto *domain = system.hardwareDomainContract(clock.underlying());
  const auto *contract =
      domain ? std::get_if<fabric::ClockDomainContractRecord>(
                   &domain->contract())
             : nullptr;
  if (!contract)
    return invalid("service clock does not resolve to a Clock domain");
  const std::uint64_t period =
      contract->periodFs() / kFemtosecondsPerPicosecond;
  if (period == 0)
    return invalid("service clock period is below one picosecond");
  return period;
}

/// Configuration payload bytes per AccCore, memoized by the immutable Fabric
/// identity because the packed ConfigurationABI derivation is expensive and
/// deterministic.
llvm::Expected<std::uint64_t>
configurationBytesPerCore(const fabric::FinalizedFabricRoot &fabricRoot,
                          std::uint64_t accCoreCount) {
  using Key = std::array<std::uint8_t, ArtifactIdentity::byteSize>;
  static std::mutex mutex;
  static std::map<Key, std::uint64_t> memo;
  Key key{};
  const auto bytes = fabricRoot.reference().artifact.bytes();
  std::copy(bytes.begin(), bytes.end(), key.begin());
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (auto found = memo.find(key); found != memo.end())
      return found->second;
  }
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto draft = hardware::derivePackedConfigurationABIDraft(fabricRoot, context);
  if (!draft)
    return draft.takeError();
  std::uint64_t bits = 0;
  for (const hardware::ProgrammingUnitDraft &unit : draft->programmingUnits) {
    auto sum = checkedAdd(bits, unit.payloadBitCount, "configuration bits");
    if (!sum)
      return sum.takeError();
    bits = *sum;
  }
  const std::uint64_t perCore =
      ceilDiv(ceilDiv(bits, 8), std::max<std::uint64_t>(1, accCoreCount));
  std::lock_guard<std::mutex> lock(mutex);
  memo.try_emplace(key, perCore);
  return perCore;
}

} // namespace

llvm::StringRef toString(AnalyticLaunchBottleneck bottleneck) {
  switch (bottleneck) {
  case AnalyticLaunchBottleneck::Launch:
    return "launch";
  case AnalyticLaunchBottleneck::Compute:
    return "compute";
  case AnalyticLaunchBottleneck::MemoryBandwidth:
    return "memory_bandwidth";
  case AnalyticLaunchBottleneck::MemoryLatency:
    return "memory_latency";
  }
  llvm_unreachable("unknown analytic launch bottleneck");
}

llvm::Expected<SystemPlatformModel>
projectSystemPlatformModel(const fabric::FinalizedFabricRoot &fabricRoot) {
  auto systemRoot = fabric::requireSystemRoot(fabricRoot.view());
  if (!systemRoot)
    return systemRoot.takeError();
  const fabric::FabricSystemRootView &system = *systemRoot;
  const auto hosts = system.artifact().hostCoreOccurrences();
  if (hosts.size() != 1)
    return invalid("System runtime model requires exactly one HostCore");
  const auto *host = system.instructionCoreMicroarchitecture(hosts.front());
  if (!host)
    return invalid("HostCore has no microarchitectural realization");
  SystemPlatformModel platform;
  platform.hostCyclesPerInstructionLeaf =
      host->kind() == fabric::InstructionCoreRealizationKind::InOrder
          ? kInOrderCyclesPerLeaf
          : kOutOfOrderCyclesPerLeaf;
  platform.accCoreCount = system.artifact().accCoreOccurrences().size();
  if (platform.accCoreCount == 0)
    return invalid("System runtime model requires at least one AccCore");

  const fabric::CanonicalServiceCapabilityRecord *capability = nullptr;
  const ::fabric::MemoryServiceContractRecord *service = nullptr;
  for (const auto &attachment : system.spatialAttachments()) {
    if (!attachment.serviceEndpoint)
      continue;
    const auto *capabilities =
        system.serviceEndpointCapabilities(*attachment.serviceEndpoint);
    if (!capabilities ||
        capabilities->plane() != fabric::CanonicalServiceEndpointPlane::Memory ||
        capabilities->capabilities().empty())
      continue;
    const auto *owner = system.serviceEndpointOwner(*attachment.serviceEndpoint);
    const auto *memoryService =
        owner ? std::get_if<fabric::FabricMemoryServiceRef>(
                    &owner->owner().payload)
              : nullptr;
    const auto *memoryRef =
        memoryService ? std::get_if<fabric::SystemMemoryServiceRef>(
                            &memoryService->payload)
                      : nullptr;
    const auto *record = memoryRef ? system.memoryService(*memoryRef) : nullptr;
    if (!record)
      continue;
    capability = &capabilities->capabilities().front();
    service = record;
    break;
  }
  if (!capability || !service)
    return invalid("no SpatialCore memory attachment reaches a System memory "
                   "service");
  const fabric::ServiceRateContractRecord &rate = capability->rate();
  auto period = clockPeriodPicoseconds(system, rate.rateClock());
  if (!period)
    return period.takeError();
  platform.clockPeriodPicoseconds = *period;
  std::uint64_t beatBits = 0;
  for (const auto &declaration : service->capabilities())
    beatBits = std::max(beatBits, declaration.serviceBeatWidthBits);
  if (beatBits < 8)
    return invalid("System memory service declares no service beat width");
  if (rate.operationsPerWindow() == 0 || rate.windowTicks() == 0)
    return invalid("System memory service rate window is empty");
  auto windowPicoseconds =
      checkedMul(rate.windowTicks(), *period, "memory service window");
  if (!windowPicoseconds)
    return windowPicoseconds.takeError();
  auto windowBytes = checkedMul(rate.operationsPerWindow(), beatBits / 8,
                                "memory service window bytes");
  if (!windowBytes)
    return windowBytes.takeError();
  platform.memoryServicePicosecondsPerByte =
      std::max<std::uint64_t>(1, ceilDiv(*windowPicoseconds, *windowBytes));
  // Every SpatialCore reaches the shared memory through its private access
  // cache, so a request is one line fill and the miss-status entries bound
  // the requests in flight; the endpoint's own outstanding limit still caps
  // them. The most constrained AccCore bounds the model.
  platform.accCoreRequestBytes = 0;
  platform.accCoreOutstandingRequests =
      std::max<std::uint64_t>(1, rate.maxOutstanding());
  for (const auto core : system.artifact().accCoreOccurrences()) {
    const auto *access = system.spatialMemoryAccess(core);
    if (!access)
      return invalid("AccCore declares no Spatial memory access realization");
    const std::uint64_t line = access->cache().lineBytes();
    platform.accCoreRequestBytes = platform.accCoreRequestBytes == 0
                                       ? line
                                       : std::min(platform.accCoreRequestBytes,
                                                  line);
    platform.accCoreOutstandingRequests =
        std::min<std::uint64_t>(platform.accCoreOutstandingRequests,
                                std::max<std::uint32_t>(
                                    1, access->cache().missStatusEntries()));
  }
  if (const auto *bounded =
          std::get_if<::fabric::BoundedCompletion>(&rate.progress())) {
    auto progressPeriod =
        clockPeriodPicoseconds(system, bounded->progressClock);
    if (!progressPeriod)
      return progressPeriod.takeError();
    auto latency = checkedMul(bounded->maxIssueToRetireTicks, *progressPeriod,
                              "memory service completion");
    if (!latency)
      return latency.takeError();
    platform.memoryLatencyPicoseconds = *latency;
  } else {
    platform.memoryLatencyPicoseconds = *windowPicoseconds;
  }

  const runtime::Gem5BuiltinPlatformPolicy policy;
  auto leafPicoseconds = checkedMul(platform.hostCyclesPerInstructionLeaf,
                                    *period, "instruction leaf period");
  if (!leafPicoseconds)
    return leafPicoseconds.takeError();
  auto dispatchLeaves = checkedMul(kHostDispatchInstructionLeaves,
                                   *leafPicoseconds, "host dispatch leaves");
  if (!dispatchLeaves)
    return dispatchLeaves.takeError();
  auto dispatchPio = checkedMul(kHostDispatchPioOperations,
                                policy.processorClockPeriodTicks,
                                "host dispatch PIO");
  if (!dispatchPio)
    return dispatchPio.takeError();
  auto dispatch = checkedAdd(*dispatchLeaves, *dispatchPio, "host dispatch");
  if (!dispatch)
    return dispatch.takeError();
  platform.launchDispatchPicoseconds = *dispatch;
  auto entryLeaves = checkedMul(kInstructionCoreEntryInstructionLeaves,
                                *leafPicoseconds, "InstructionCore entry");
  if (!entryLeaves)
    return entryLeaves.takeError();
  auto bridgePio = checkedMul(
      kBridgeProgrammingPioOperations + kCompletionPioOperations,
      policy.spatialBridgeLatencyTicks, "bridge PIO");
  if (!bridgePio)
    return bridgePio.takeError();
  auto fixed = checkedAdd(*entryLeaves, *bridgePio, "launch fixed cost");
  if (!fixed)
    return fixed.takeError();
  platform.launchFixedPicoseconds = *fixed;
  auto configuration =
      configurationBytesPerCore(fabricRoot, platform.accCoreCount);
  if (!configuration)
    return configuration.takeError();
  platform.configurationBytesPerCore = *configuration;
  return platform;
}

llvm::Expected<std::uint64_t>
estimateConfigurationLoadPicoseconds(const SystemPlatformModel &platform,
                                     std::uint64_t accCores) {
  if (accCores == 0 || platform.configurationBytesPerCore == 0)
    return std::uint64_t{0};
  if (platform.accCoreOutstandingRequests == 0 ||
      platform.accCoreRequestBytes == 0)
    return invalid("platform model has no SpatialCore memory request shape");
  auto bytes = checkedMul(platform.configurationBytesPerCore, accCores,
                          "configuration payload");
  if (!bytes)
    return bytes.takeError();
  auto service = checkedMul(*bytes, platform.memoryServicePicosecondsPerByte,
                            "configuration service");
  if (!service)
    return service.takeError();
  const std::uint64_t requests =
      ceilDiv(platform.configurationBytesPerCore, platform.accCoreRequestBytes);
  auto chain = checkedMul(ceilDiv(requests, platform.accCoreOutstandingRequests),
                          platform.memoryLatencyPicoseconds,
                          "configuration request chain");
  if (!chain)
    return chain.takeError();
  return std::max(*service, *chain);
}

llvm::Expected<AnalyticLaunchDuration>
estimateLaunchDuration(const SystemPlatformModel &platform,
                       const AnalyticLaunchEstimate &launch,
                       std::uint64_t accCores) {
  if (launch.activations == 0)
    return AnalyticLaunchDuration{0, AnalyticLaunchBottleneck::Launch};
  if (platform.accCoreOutstandingRequests == 0 ||
      platform.accCoreRequestBytes == 0)
    return invalid("platform model has no SpatialCore memory request shape");
  const std::uint64_t cores = std::max<std::uint64_t>(
      1, std::min(accCores, launch.activations));
  const std::uint64_t activationsPerCore = ceilDiv(launch.activations, cores);

  auto compute = checkedMul(launch.computeCyclesPerActivation,
                            platform.clockPeriodPicoseconds, "compute time");
  if (!compute)
    return compute.takeError();
  auto wireService = checkedMul(launch.boundaryPayloadBytesPerActivation,
                                platform.memoryServicePicosecondsPerByte,
                                "wire service");
  if (!wireService)
    return wireService.takeError();
  auto wire = checkedAdd(*wireService, platform.memoryLatencyPicoseconds,
                         "wire fetch");
  if (!wire)
    return wire.takeError();
  auto service = checkedMul(launch.externalMemoryBytesPerActivation,
                            platform.memoryServicePicosecondsPerByte,
                            "memory service");
  if (!service)
    return service.takeError();
  auto bandwidth = checkedMul(*service, cores, "shared memory service");
  if (!bandwidth)
    return bandwidth.takeError();
  const std::uint64_t requests = ceilDiv(
      launch.externalMemoryBytesPerActivation, platform.accCoreRequestBytes);
  auto chain = checkedMul(ceilDiv(requests, platform.accCoreOutstandingRequests),
                          platform.memoryLatencyPicoseconds,
                          "memory request chain");
  if (!chain)
    return chain.takeError();

  AnalyticLaunchDuration result;
  std::uint64_t point = *compute;
  result.bottleneck = AnalyticLaunchBottleneck::Compute;
  if (*bandwidth > point) {
    point = *bandwidth;
    result.bottleneck = AnalyticLaunchBottleneck::MemoryBandwidth;
  }
  if (*chain > point) {
    point = *chain;
    result.bottleneck = AnalyticLaunchBottleneck::MemoryLatency;
  }
  auto fixed = checkedAdd(platform.launchFixedPicoseconds, *wire,
                          "launch fixed cost");
  if (!fixed)
    return fixed.takeError();
  if (*fixed >= point)
    result.bottleneck = AnalyticLaunchBottleneck::Launch;
  auto activation = checkedAdd(*fixed, point, "activation time");
  if (!activation)
    return activation.takeError();
  auto perCore = checkedMul(activationsPerCore, *activation, "per-core chain");
  if (!perCore)
    return perCore.takeError();
  auto dispatch = checkedMul(launch.activations,
                             platform.launchDispatchPicoseconds,
                             "host dispatch chain");
  if (!dispatch)
    return dispatch.takeError();
  auto total = checkedAdd(*dispatch, *perCore, "launch duration");
  if (!total)
    return total.takeError();
  result.picoseconds = *total;
  return result;
}

llvm::Expected<std::uint64_t>
estimateHostResidualPicoseconds(const SystemPlatformModel &platform,
                                std::uint64_t instructionLeaves) {
  auto cycles = checkedMul(instructionLeaves,
                           platform.hostCyclesPerInstructionLeaf,
                           "host residual cycles");
  if (!cycles)
    return cycles.takeError();
  return checkedMul(*cycles, platform.clockPeriodPicoseconds,
                    "host residual time");
}

void appendAnalyticLaunchEstimates(
    std::vector<std::uint8_t> &bytes,
    llvm::ArrayRef<AnalyticLaunchEstimate> launches) {
  const auto appendU64 = [&](std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  };
  appendU64(launches.size());
  for (const AnalyticLaunchEstimate &launch : launches) {
    const auto artifact = launch.launch.artifact.bytes();
    appendU64(artifact.size());
    bytes.insert(bytes.end(), artifact.begin(), artifact.end());
    appendU64(launch.launch.entity.value());
    appendU64(launch.activations);
    appendU64(launch.computeCyclesPerActivation);
    appendU64(launch.externalMemoryBytesPerActivation);
    appendU64(launch.boundaryPayloadBytesPerActivation);
  }
}

} // namespace loom::evaluation::models
