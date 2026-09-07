#include "Evaluation/Models/SystemRuntimeAnalytic.h"

#include "Fabric/IR/MemoryConsistencyContract.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Runtime/Gem5SimulationBinding.h"

#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <limits>
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
projectSystemPlatformModel(const fabric::FabricSystemRootView &system) {
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
  platform.accCoreRequestBytes = beatBits / 8;
  if (rate.operationsPerWindow() == 0 || rate.windowTicks() == 0)
    return invalid("System memory service rate window is empty");
  auto windowPicoseconds =
      checkedMul(rate.windowTicks(), *period, "memory service window");
  if (!windowPicoseconds)
    return windowPicoseconds.takeError();
  auto windowBytes = checkedMul(rate.operationsPerWindow(),
                                platform.accCoreRequestBytes,
                                "memory service window bytes");
  if (!windowBytes)
    return windowBytes.takeError();
  platform.memoryServicePicosecondsPerByte =
      std::max<std::uint64_t>(1, ceilDiv(*windowPicoseconds, *windowBytes));
  platform.accCoreOutstandingRequests =
      std::max<std::uint64_t>(1, rate.maxOutstanding());
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
  return platform;
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
