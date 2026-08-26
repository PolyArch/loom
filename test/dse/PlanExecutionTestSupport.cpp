#include "PlanExecutionTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/ComponentViewDigest.h"
#include "Common/TimeoutBudgets.h"
#include "DSE/CandidateGenerator.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace loom::dse::test_support {
namespace {

constexpr ArtifactSchemaDescriptor sourceSchema{
    "loom.test.plan_executor_source", SchemaVersion{1, 0}};
constexpr ArtifactSchemaDescriptor candidateSchema{
    "loom.test.plan_executor_candidate", SchemaVersion{1, 0}};
constexpr std::array<std::uint8_t, 4> configSchema = {0x50, 0x45, 0x58, 0x45};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "plan execution test invalid: " + message);
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  if (bytes != llvm::ArrayRef<std::uint8_t>({0x01}))
    return invalid("generator config is not canonical");
  return validateComponentViewDigest(configSchema, bytes, digest);
}

constexpr std::array<CandidateGeneratorInputSlotDescriptor, 1> inputs = {{
    {CandidateGeneratorInputSlotRef(0), "source", PlanValueRole::CandidateSet,
     &sourceSchema, PlanValueCardinality::ExactlyOne},
}};
constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputs = {{
    {CandidateGeneratorOutputSlotRef(0), "candidate",
     PlanValueRole::CandidateSet, &candidateSchema,
     PlanValueCardinality::NonEmptySet},
}};
constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "candidate_attempt"},
}};

const CandidateGeneratorDescriptor descriptor{
    CandidateGeneratorKind(0x7fff7600),
    "test.plan_executor",
    "loom.test.plan_executor.v1",
    inputs,
    outputs,
    ResolvedDseConfigViewContract{configSchema, validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    nullptr,
    ProviderForm::InProcess};

std::atomic_uint64_t providerCalls{0};
std::atomic_uint64_t activeProviders{0};
std::atomic_uint64_t maximumActiveProviders{0};
std::atomic_uint64_t requiredConcurrentProviders{1};
std::atomic_bool waitForStopRequest{false};
std::atomic_bool observedStopRequest{false};
std::atomic_uint64_t observedCpuBudgetCores{0};
std::atomic_uint64_t observedMemoryBudgetBytes{0};
std::mutex concurrencyMutex;
std::condition_variable concurrencyChanged;

void observeMaximum(std::uint64_t active) {
  std::uint64_t observed =
      maximumActiveProviders.load(std::memory_order_relaxed);
  while (observed < active &&
         !maximumActiveProviders.compare_exchange_weak(
             observed, active, std::memory_order_relaxed)) {
  }
}

llvm::Expected<CandidateGeneratorProviderResult>
generate(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
         const ResolvedCandidateGeneratorBinding &binding,
         const ArtifactStore &store, const BlobStore &,
         const CandidateGeneratorInvocationView &invocation) {
  if (binding.descriptorRef() != descriptor.reference() ||
      inputBindings.size() != 1 ||
      inputBindings.front().slot != CandidateGeneratorInputSlotRef(0) ||
      inputBindings.front().artifacts.size() != 1)
    return invalid("provider received the wrong exact invocation");

  providerCalls.fetch_add(1, std::memory_order_relaxed);
  observedCpuBudgetCores.store(
      invocation.executionBudget().cpuCores.value_or(0),
      std::memory_order_relaxed);
  observedMemoryBudgetBytes.store(
      invocation.executionBudget().memoryBytes.value_or(0),
      std::memory_order_relaxed);
  const std::uint64_t active =
      activeProviders.fetch_add(1, std::memory_order_relaxed) + 1;
  observeMaximum(active);
  concurrencyChanged.notify_all();
  if (requiredConcurrentProviders.load(std::memory_order_relaxed) > 1) {
    std::unique_lock<std::mutex> lock(concurrencyMutex);
    const bool rendezvous = concurrencyChanged.wait_for(
        lock, loom::timeout::duration(loom::timeout::Tier::UltraFast), [] {
          return activeProviders.load(std::memory_order_relaxed) >=
                 requiredConcurrentProviders.load(std::memory_order_relaxed);
        });
    if (!rendezvous) {
      activeProviders.fetch_sub(1, std::memory_order_relaxed);
      concurrencyChanged.notify_all();
      return invalid("parallel provider rendezvous timed out");
    }
  }
  if (waitForStopRequest.load(std::memory_order_relaxed)) {
    const auto deadline =
        std::chrono::steady_clock::now() +
        loom::timeout::duration(loom::timeout::Tier::UltraFast);
    while (!invocation.stopRequested() &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool stopped = invocation.stopRequested();
    observedStopRequest.store(stopped, std::memory_order_relaxed);
    activeProviders.fetch_sub(1, std::memory_order_relaxed);
    concurrencyChanged.notify_all();
    if (!stopped)
      return invalid("provider execution-control stop was not observable");
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{
            CandidateGeneratorIncompleteReason::CancelledOrTimeout,
            {{CandidateGeneratorOutputSlotRef(0), {}}},
            {}},
        {{CandidateGeneratorWorkUnitRef(0), 1, 0}}};
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  activeProviders.fetch_sub(1, std::memory_order_relaxed);
  concurrencyChanged.notify_all();

  auto identity = store.put(
      candidateSchema, CanonicalSemanticBytes(std::vector<std::uint8_t>{0x41}));
  if (!identity)
    return identity.takeError();
  ArtifactRootReference candidate{candidateSchema.identity.str(),
                                  candidateSchema.version, *identity};
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{
          {{CandidateGeneratorOutputSlotRef(0), {candidate}}},
          {{CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
            CandidateGeneratorOutputSlotRef(0),
            candidate,
            {},
            {}}}},
      {{CandidateGeneratorWorkUnitRef(0), 1, 1}}};
}

llvm::Expected<ArtifactRootReference>
publishSource(const ArtifactStore &store) {
  auto identity = store.put(
      sourceSchema, CanonicalSemanticBytes(std::vector<std::uint8_t>{0x11}));
  if (!identity)
    return identity.takeError();
  return ArtifactRootReference{sourceSchema.identity.str(),
                               sourceSchema.version, *identity};
}

} // namespace

llvm::Error registerPlanExecutionTestGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(CandidateGeneratorProvider{
      descriptor.reference(), CandidateGeneratorInProcessProvider{generate}});
}

llvm::Expected<PlanExecutionFixture>
makePlanExecutionFixture(const ArtifactStore &store, std::size_t nodeCount,
                         llvm::StringRef producerIdentity) {
  auto source = publishSource(store);
  if (!source)
    return source.takeError();
  auto digest = computeComponentViewDigest(configSchema, {0x01});
  if (!digest)
    return digest.takeError();
  ResolvedConfig config = defaultResolvedConfig();
  config.dse.planNodes.reserve(nodeCount);
  for (std::size_t index = 0; index != nodeCount; ++index)
    config.dse.planNodes.push_back(
        GeneratePlanNodeDefinition{descriptor.reference(),
                                   {ExactPlanArtifacts{{*source}}},
                                   {0x01},
                                   *digest});
  auto configIdentity = store.put(ResolvedConfig::artifactSchema,
                                  canonicalResolvedConfigBytes(config));
  if (!configIdentity)
    return configIdentity.takeError();
  if (*configIdentity != resolvedConfigIdentity(config))
    return invalid("resolved config publication changed its identity");
  auto view = projectResolvedDseConfigView(config);
  if (!view)
    return view.takeError();
  auto producer = DseProducerSemanticBuildIdentity::get(producerIdentity);
  if (!producer)
    return producer.takeError();
  auto closure =
      DseRunClosure::get(std::move(*producer), {*source}, config, {}, store);
  if (!closure)
    return closure.takeError();
  return PlanExecutionFixture{std::move(config), std::move(*view),
                              std::move(*closure), std::move(*source)};
}

void resetPlanExecutionProviderObservations() {
  providerCalls.store(0, std::memory_order_relaxed);
  activeProviders.store(0, std::memory_order_relaxed);
  maximumActiveProviders.store(0, std::memory_order_relaxed);
  requiredConcurrentProviders.store(1, std::memory_order_relaxed);
  waitForStopRequest.store(false, std::memory_order_relaxed);
  observedStopRequest.store(false, std::memory_order_relaxed);
  observedCpuBudgetCores.store(0, std::memory_order_relaxed);
  observedMemoryBudgetBytes.store(0, std::memory_order_relaxed);
}

void requireConcurrentPlanExecutionProviders(std::uint64_t count) {
  requiredConcurrentProviders.store(count, std::memory_order_relaxed);
}

void requirePlanExecutionProviderStopObservation() {
  waitForStopRequest.store(true, std::memory_order_relaxed);
}

bool waitForActivePlanExecutionProvider() {
  std::unique_lock<std::mutex> lock(concurrencyMutex);
  return concurrencyChanged.wait_for(
      lock, loom::timeout::duration(loom::timeout::Tier::UltraFast),
      [] { return activeProviders.load(std::memory_order_relaxed) != 0; });
}

std::uint64_t planExecutionProviderCalls() {
  return providerCalls.load(std::memory_order_relaxed);
}

std::uint64_t maximumConcurrentPlanExecutionProviders() {
  return maximumActiveProviders.load(std::memory_order_relaxed);
}

bool planExecutionProviderObservedStop() {
  return observedStopRequest.load(std::memory_order_relaxed);
}

std::uint64_t planExecutionProviderCpuBudgetCores() {
  return observedCpuBudgetCores.load(std::memory_order_relaxed);
}

std::uint64_t planExecutionProviderMemoryBudgetBytes() {
  return observedMemoryBudgetBytes.load(std::memory_order_relaxed);
}

} // namespace loom::dse::test_support
