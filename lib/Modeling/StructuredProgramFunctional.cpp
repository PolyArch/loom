#include "Evaluation/Models/StructuredProgramFunctional.h"
#include "Evaluation/ProductionRegistry.h"
#include "StructuredEvaluationInvocationCacheInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/MappingDebugLog.h"
#include "Config/ResolvedConfig.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/StandardFindings.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Simulator/NativeSimulationOracle.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SourceBackedDfgValidation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::evaluation::models {
namespace {

constexpr BuiltinEvaluationCase kCase =
    BuiltinEvaluationCase::StructuredProgramFunctionalComparison;
constexpr BuiltinEvaluationModel kModel =
    BuiltinEvaluationModel::StructuredProgramFunctional;
constexpr CaseSubjectRoleRef kCandidateRole(0);
constexpr ScopeFormRef kWholeExactCaseScope(0);

EvaluationCaseSignatureRef caseSignatureRef() {
  return llvm::cantFail(EvaluationCaseSignatureRef::get(
      evaluationSchemaVersion(), builtinEvaluationCaseKind(kCase)));
}

const ArtifactSchemaDescriptor *const kStructuredSchemas[] = {
    &frontend::structuredProgramArtifactSchema};
const ArtifactSchemaDescriptor *const kWorkloadSchemas[] = {
    &sim::simulationWorkloadSchema};
const ArtifactSchemaDescriptor *const kRuntimeInputSchemas[] = {
    &sim::simulationRuntimeInputSchema};

const CaseSubjectRoleDescriptor kSubjectRoles[] = {
    {kCandidateRole, "selected_structured_program",
     SubjectRoleCardinality::ExactlyOne, kStructuredSchemas, nullptr}};

llvm::Error verifyWorkloadCompatibility(
    const EvaluationCase &, const EvaluationSubjectBindings &,
    const std::optional<ArtifactRootReference> &workload,
    const std::optional<ArtifactRootReference> &runtimeInput,
    const CaseArtifactResolution &resolution, const ArtifactStore &,
    const BlobStore &) {
  if (!workload || !runtimeInput)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: workload inputs are not total");
  const CaseArtifactResolution::Entry *workloadEntry =
      resolution.find(*workload);
  const CaseArtifactResolution::Entry *runtimeEntry =
      resolution.find(*runtimeInput);
  if (!workloadEntry || !runtimeEntry ||
      !CaseArtifactResolution::reaches(*runtimeEntry, *workload))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: runtime input does not reach "
        "its exact workload");
  const bool hasSource = llvm::any_of(
      workloadEntry->dependencyClosure,
      [](const ArtifactRootReference &reference) {
        return reference.schemaIdentity ==
                   frontend::structuredProgramArtifactSchema.identity &&
               reference.schemaVersion ==
                   frontend::structuredProgramArtifactSchema.version;
      });
  if (!hasSource)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: workload has no exact source "
        "program");
  return llvm::Error::success();
}

const EvaluationCaseSignatureDescriptor kCaseSignature{
    builtinEvaluationCaseKind(kCase),
    "structured_program_functional_comparison",
    "One exact Structured candidate compared with its workload-owned source.",
    kSubjectRoles,
    ArtifactRequirement::Required,
    kWorkloadSchemas,
    ArtifactRequirement::Required,
    kRuntimeInputSchemas,
    &verifyWorkloadCompatibility,
    AbsentReferenceCycle{},
    {}};

const ScopeFormRef kWholeCaseScopeForms[] = {kWholeExactCaseScope};
const FindingCapability kFindingCapabilities[] = {
    {standard_findings::FunctionalMismatch, kWholeCaseScopeForms,
     findingResultFormMask(FindingResultForm::Absent) |
         findingResultFormMask(FindingResultForm::Present) |
         findingResultFormMask(FindingResultForm::NotApplicable)}};
const ModeledPhenomenon kModeledPhenomena[] = {
    ModeledPhenomenon::StructuredProgram};

struct EmptyFunctionalConfig final {};

bool haveEquivalentSourceObservations(
    const sim::NativeStructuredProgramObservations &lhs,
    const sim::NativeStructuredProgramObservations &rhs) {
  if (!sim::haveEquivalentFunctionalObservations(lhs, rhs) ||
      lhs.blockActivations.size() != rhs.blockActivations.size())
    return false;
  return llvm::equal(
      lhs.blockActivations, rhs.blockActivations,
      [](const sim::NativeStructuredBlockActivation &lhsActivation,
         const sim::NativeStructuredBlockActivation &rhsActivation) {
        return lhsActivation.block == rhsActivation.block &&
               lhsActivation.activations == rhsActivation.activations;
      });
}

llvm::Error primeSourceObservationCacheImpl(
    const ArtifactRootReference &source,
    const ArtifactRootReference &workloadReference,
    const ArtifactRootReference &runtimeInputReference,
    const sim::NativeStructuredProgramObservations &observations) {
  StructuredEvaluationInvocationCache *cache =
      detail::currentStructuredEvaluationCache();
  if (!cache)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: source observation priming "
        "requires an active invocation cache");
  auto &impl = detail::StructuredEvaluationCacheAccess::impl(*cache);
  const detail::StructuredSourceObservationCacheKey key{
      source, workloadReference, runtimeInputReference};
  auto value = std::make_shared<const sim::NativeStructuredProgramObservations>(
      observations);
  using SourceFlightMap = decltype(impl.sourceObservationFlights);
  std::unique_ptr<detail::CacheFlightGuard<
      detail::StructuredSourceObservationCacheKey, SourceFlightMap>>
      sourceFlight;
  std::unique_lock<std::mutex> lock(impl.mutex);
  while (true) {
    auto existing = impl.sourceObservations.find(key);
    if (existing != impl.sourceObservations.end()) {
      if (!haveEquivalentSourceObservations(*existing->second, observations))
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "structured_functional_model_invalid: nondeterministic source "
            "observations");
      return llvm::Error::success();
    }
    auto flight = impl.sourceObservationFlights.find(key);
    if (flight == impl.sourceObservationFlights.end()) {
      auto entry = std::make_shared<detail::CacheFlightEntry>();
      impl.sourceObservationFlights.emplace(key, entry);
      sourceFlight = std::make_unique<detail::CacheFlightGuard<
          detail::StructuredSourceObservationCacheKey, SourceFlightMap>>(
          impl.sourceObservationFlights, impl.mutex, impl.flightChanged, key,
          std::move(entry));
      break;
    }
    auto entry = flight->second;
    impl.sourceObservationSingleFlightWaitCount.fetch_add(
        1, std::memory_order_relaxed);
    impl.flightChanged.wait(lock, [&] { return entry->complete; });
  }
  if (impl.sourceObservations.size() >=
          impl.limits.maximumSourceObservationEntries &&
      impl.sourceObservations.find(key) == impl.sourceObservations.end()) {
    impl.capacityBypassCount.fetch_add(1, std::memory_order_relaxed);
    return llvm::createStringError(
        std::make_error_code(std::errc::no_buffer_space),
        "structured_functional_model_incomplete: source observation cache "
        "capacity exhausted");
  }
  auto [found, inserted] =
      impl.sourceObservations.try_emplace(key, std::move(value));
  if (!inserted) {
    if (!haveEquivalentSourceObservations(*found->second, observations))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_functional_model_invalid: nondeterministic source "
          "observations");
    return llvm::Error::success();
  }
  impl.sourceObservationPrimeCount.fetch_add(1, std::memory_order_relaxed);
  return llvm::Error::success();
}

using ReplayResultKind = detail::StructuredReplayResultKind;
using CachedReplayResult = detail::StructuredCachedReplayResult;

detail::StructuredFunctionalCacheKey
replayCacheKey(const ArtifactRootReference &candidate,
               const ArtifactRootReference &workload,
               const ArtifactRootReference &runtimeInput) {
  return {candidate, workload, runtimeInput};
}

bool cachedReplayMeetsLimits(
    const CachedReplayResult &cached,
    const sim::SourceBackedDfgValidationLimits &limits) {
  if (!cached.replay ||
      cached.kind == ReplayResultKind::CancelledOrTimeout)
    return false;
  return cached.replay->wavefrontSteps <= limits.maxWavefrontSteps &&
         cached.replay->eventCount <= limits.maxEventCount &&
         cached.replay->memoryBytesCompared <= limits.maxRetainedCaptureBytes;
}

llvm::Expected<CachedReplayResult> classifyReplayResult(
    llvm::Expected<sim::SourceBackedDfgValidationResult> replay) {
  if (replay) {
    ReplayResultKind kind = ReplayResultKind::Unsupported;
    switch (replay->status) {
    case sim::SourceBackedDfgValidationStatus::Equivalent:
      kind = ReplayResultKind::Equivalent;
      break;
    case sim::SourceBackedDfgValidationStatus::Mismatch:
      kind = ReplayResultKind::Mismatch;
      break;
    case sim::SourceBackedDfgValidationStatus::Inapplicable:
      kind = ReplayResultKind::Inapplicable;
      break;
    }
    return CachedReplayResult{kind, std::move(*replay)};
  }

  std::error_code code;
  std::string message;
  llvm::raw_string_ostream stream(message);
  llvm::handleAllErrors(std::move(replay).takeError(),
                        [&](const llvm::ErrorInfoBase &failure) {
                          code = failure.convertToErrorCode();
                          failure.log(stream);
                        });
  stream.flush();
  if (code == std::make_error_code(std::errc::not_supported)) {
    mapping_debug::emit(
        mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
        mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
          fields["failure_scope"] = "structured_functional_replay";
          fields["closure_status"] = "proven_infeasible";
          fields["diagnostic"] = message;
        });
    return CachedReplayResult{ReplayResultKind::Unsupported, std::nullopt};
  }
  if (code == std::make_error_code(std::errc::timed_out)) {
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::DataflowLowering,
        mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
          fields["failure_scope"] = "structured_functional_replay";
          fields["closure_status"] = "cancelled_or_timeout";
          fields["diagnostic"] = message;
        });
    return CachedReplayResult{ReplayResultKind::CancelledOrTimeout,
                              std::nullopt};
  }
  return llvm::createStringError(code ? code : llvm::inconvertibleErrorCode(),
                                 "%s", message.c_str());
}

llvm::ArrayRef<std::uint8_t> configSchemaBytes() {
  static constexpr llvm::StringLiteral descriptor =
      "loom.structured_functional.config.1.0";
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<OwnerValue> projectConfig(const ResolvedConfig &) {
  return OwnerValue::get(EmptyFunctionalConfig{});
}

llvm::Expected<std::vector<std::uint8_t>>
encodeConfig(const OwnerValue &value) {
  if (!value.getIf<EmptyFunctionalConfig>())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured functional config has the wrong owner type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue> adoptConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                       const ComponentViewDigest &) {
  if (!bytes.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured functional config must be empty");
  return OwnerValue::get(EmptyFunctionalConfig{});
}

const ResolvedModelConfigViewContract kConfigView{
    configSchemaBytes(), &projectConfig, &encodeConfig, &adoptConfig};

const EvaluationModelDescriptor kModelDescriptor{
    builtinEvaluationModelKind(kModel),
    "structured_program_functional",
    "loom.structured_program.functional.v1",
    caseSignatureRef(),
    {},
    {},
    kFindingCapabilities,
    {},
    {},
    kConfigView,
    kModeledPhenomena,
    EvaluationExecutionMethod::Simulation,
    {},
    DeterminismContract::Deterministic,
    {},
    ProviderForm::InProcess};

llvm::Expected<EvaluationModelResult> classifyNativeFailure(llvm::Error error) {
  std::error_code code;
  std::string message;
  llvm::raw_string_ostream stream(message);
  llvm::handleAllErrors(std::move(error),
                        [&](const llvm::ErrorInfoBase &failure) {
                          code = failure.convertToErrorCode();
                          failure.log(stream);
                        });
  stream.flush();
  if (code == std::make_error_code(std::errc::not_supported))
    return EvaluationModelResult{
        {}, UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  if (code == std::make_error_code(std::errc::io_error))
    return EvaluationModelResult{
        {}, ExecutionFailedEvidence{OutcomeReason::ToolFailure}};
  return llvm::createStringError(code ? code : llvm::inconvertibleErrorCode(),
                                 "%s", message.c_str());
}

llvm::Expected<std::shared_ptr<const sim::NativeStructuredProgramObservations>>
sourceObservationsFor(
    const ArtifactRootReference &source,
    const ArtifactRootReference &workloadReference,
    const ArtifactRootReference &runtimeInputReference,
    const sim::ImportedStructuredProgramSimulationInputs &inputs) {
  StructuredEvaluationInvocationCache *cache =
      detail::currentStructuredEvaluationCache();
  detail::StructuredSourceObservationCacheKey key{source, workloadReference,
                                                  runtimeInputReference};
  std::unique_ptr<detail::CacheFlightGuard<
      detail::StructuredSourceObservationCacheKey,
      decltype(std::declval<StructuredEvaluationInvocationCache::Impl &>()
                   .sourceObservationFlights)>>
      sourceFlight;
  if (cache) {
    auto &impl = detail::StructuredEvaluationCacheAccess::impl(*cache);
    using SourceFlightMap = decltype(impl.sourceObservationFlights);
    std::unique_lock<std::mutex> lock(impl.mutex);
    while (true) {
      auto found = impl.sourceObservations.find(key);
      if (found != impl.sourceObservations.end()) {
        impl.sourceObservationHitCount.fetch_add(1,
                                                std::memory_order_relaxed);
        return found->second;
      }
      auto flight = impl.sourceObservationFlights.find(key);
      if (flight == impl.sourceObservationFlights.end()) {
        auto entry = std::make_shared<detail::CacheFlightEntry>();
        impl.sourceObservationFlights.emplace(key, entry);
        impl.sourceObservationMissCount.fetch_add(1,
                                                 std::memory_order_relaxed);
        sourceFlight = std::make_unique<detail::CacheFlightGuard<
            detail::StructuredSourceObservationCacheKey, SourceFlightMap>>(
            impl.sourceObservationFlights, impl.mutex, impl.flightChanged, key,
            std::move(entry));
        break;
      }
      auto entry = flight->second;
      impl.sourceObservationSingleFlightWaitCount.fetch_add(
          1, std::memory_order_relaxed);
      impl.flightChanged.wait(lock, [&] { return entry->complete; });
    }
  }

  auto observations = sim::executeNativeStructuredProgram(
      inputs.structuredProgram, inputs.workload, inputs.runtimeInput);
  if (!observations)
    return observations.takeError();
  auto value = std::make_shared<const sim::NativeStructuredProgramObservations>(
      std::move(*observations));
  if (!cache)
    return value;
  auto &impl = detail::StructuredEvaluationCacheAccess::impl(*cache);
  std::lock_guard<std::mutex> lock(impl.mutex);
  if (auto existing = impl.sourceObservations.find(key);
      existing != impl.sourceObservations.end()) {
    if (!haveEquivalentSourceObservations(*existing->second, *value))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_functional_model_invalid: nondeterministic source "
          "observations");
    impl.sourceObservationHitCount.fetch_add(1, std::memory_order_relaxed);
    return existing->second;
  }
  if (impl.sourceObservations.size() >=
      impl.limits.maximumSourceObservationEntries) {
    impl.capacityBypassCount.fetch_add(1, std::memory_order_relaxed);
    return value;
  }
  auto [found, inserted] = impl.sourceObservations.try_emplace(key, value);
  if (!inserted && !haveEquivalentSourceObservations(*found->second, *value))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: nondeterministic source "
        "observations");
  if (inserted)
    impl.sourceObservationPrimeCount.fetch_add(1, std::memory_order_relaxed);
  return found->second;
}

llvm::Expected<EvaluationModelResult>
evaluate(const EvaluationRequest &request, const CaseArtifactResolution &,
         const ArtifactStore &artifactStore, const BlobStore &) {
  llvm::ArrayRef<ArtifactRootReference> candidates =
      request.subjectBindings().subjects(kCandidateRole);
  if (candidates.size() != 1 || !request.workload() || !request.runtimeInput())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: exact case inputs are not "
        "total");
  auto candidate =
      frontend::importStructuredProgram(candidates.front(), artifactStore);
  if (!candidate)
    return candidate.takeError();
  auto inputs = sim::importStructuredProgramSimulationInputs(
      *request.workload(), *request.runtimeInput(), artifactStore);
  if (!inputs)
    return inputs.takeError();
  const ArtifactRootReference source{
      frontend::structuredProgramArtifactSchema.identity.str(),
      frontend::structuredProgramArtifactSchema.version,
      inputs->structuredProgram.identity()};
  bool mismatch = false;
  std::shared_ptr<const CachedReplayResult> replay;
  if (candidate->identity() == inputs->structuredProgram.identity()) {
    auto sourceObservations = sourceObservationsFor(
        source, *request.workload(), *request.runtimeInput(), *inputs);
    if (!sourceObservations)
      return classifyNativeFailure(sourceObservations.takeError());
  } else {
    const detail::StructuredFunctionalCacheKey key = replayCacheKey(
        candidates.front(), *request.workload(), *request.runtimeInput());
    StructuredEvaluationInvocationCache *cache =
        detail::currentStructuredEvaluationCache();
    using FunctionalFlightMap =
        decltype(std::declval<StructuredEvaluationInvocationCache::Impl &>()
                     .functionalFlights);
    std::unique_ptr<detail::CacheFlightGuard<
        detail::StructuredFunctionalCacheKey, FunctionalFlightMap>>
        functionalFlight;
    if (cache) {
      auto &impl = detail::StructuredEvaluationCacheAccess::impl(*cache);
      using FlightMap = decltype(impl.functionalFlights);
      std::unique_lock<std::mutex> lock(impl.mutex);
      while (true) {
        auto found = impl.functionalResults.find(key);
        if (found != impl.functionalResults.end()) {
          replay = found->second;
          impl.functionalHitCount.fetch_add(1, std::memory_order_relaxed);
          break;
        }
        auto flight = impl.functionalFlights.find(key);
        if (flight == impl.functionalFlights.end()) {
          auto entry = std::make_shared<detail::CacheFlightEntry>();
          impl.functionalFlights.emplace(key, entry);
          impl.functionalMissCount.fetch_add(1, std::memory_order_relaxed);
          functionalFlight = std::make_unique<detail::CacheFlightGuard<
              detail::StructuredFunctionalCacheKey, FlightMap>>(
              impl.functionalFlights, impl.mutex, impl.flightChanged, key,
              std::move(entry));
          break;
        }
        auto entry = flight->second;
        impl.functionalSingleFlightWaitCount.fetch_add(
            1, std::memory_order_relaxed);
        impl.flightChanged.wait(lock, [&] { return entry->complete; });
      }
    }
    if (replay) {
      if (replay->kind == ReplayResultKind::Unsupported)
        return EvaluationModelResult{
            {},
            UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
      if (replay->kind == ReplayResultKind::CancelledOrTimeout)
        return EvaluationModelResult{
            {}, CancelledOrTimeoutEvidence{
                    OutcomeReason::ExecutionLimitReached}};
      mismatch = replay->kind == ReplayResultKind::Mismatch;
    } else {
      auto sourceObservations = sourceObservationsFor(
          source, *request.workload(), *request.runtimeInput(), *inputs);
      if (!sourceObservations)
        return classifyNativeFailure(sourceObservations.takeError());
      auto selectedObservations = sim::executeSelectedStructuredProgram(
          *candidate, inputs->structuredProgram, inputs->workload,
          inputs->runtimeInput);
      if (!selectedObservations)
        return classifyNativeFailure(selectedObservations.takeError());
      mismatch = !sim::haveEquivalentFunctionalObservations(
          **sourceObservations, *selectedObservations);
      auto computedReplay = std::make_shared<const CachedReplayResult>(
          CachedReplayResult{mismatch ? ReplayResultKind::Mismatch
                                      : ReplayResultKind::Equivalent,
                             std::nullopt});
      if (cache) {
        auto &impl = detail::StructuredEvaluationCacheAccess::impl(*cache);
        std::lock_guard<std::mutex> lock(impl.mutex);
        if (auto existing = impl.functionalResults.find(key);
            existing != impl.functionalResults.end()) {
          if (!(*existing->second == *computedReplay))
            return llvm::createStringError(
                llvm::inconvertibleErrorCode(),
                "structured_functional_model_invalid: nondeterministic "
                "replay");
          replay = existing->second;
        } else if (impl.functionalResults.size() <
                   impl.limits.maximumFunctionalEntries) {
          auto [found, inserted] =
              impl.functionalResults.try_emplace(key, computedReplay);
          if (!inserted && !(*found->second == *computedReplay))
            return llvm::createStringError(
                llvm::inconvertibleErrorCode(),
                "structured_functional_model_invalid: nondeterministic "
                "replay");
          replay = found->second;
          if (inserted)
            impl.functionalPrimeCount.fetch_add(1, std::memory_order_relaxed);
        } else {
          impl.capacityBypassCount.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (!mismatch)
        return EvaluationModelResult{
            {},
            UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
    }
  }

  std::vector<FindingResult> findings;
  findings.reserve(request.findingRequests().size());
  for (const FindingRequest &finding : request.findingRequests()) {
    if (finding.query().kind != standard_findings::FunctionalMismatch)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_functional_model_invalid: unsupported finding");
    if (mismatch) {
      findings.push_back(FindingResult{PresentFinding{{FindingOccurrence::get(
          standard_findings::FunctionalMismatchOccurrence{})}}});
    } else if (replay && replay->kind == ReplayResultKind::Inapplicable) {
      findings.push_back(FindingResult{
          NotApplicableFinding{NotApplicableReason::UndefinedForSubject}});
    } else {
      findings.push_back(FindingResult{AbsentFinding{}});
    }
  }
  return EvaluationModelResult{{}, CompletedEvidence{{}, std::move(findings)}};
}

const EvaluationModelProvider kProvider{
    kModelDescriptor.reference(), EvaluationModelInProcessProvider{&evaluate}};

llvm::Expected<CaseArtifactResolution>
resolveCase(const ArtifactRootReference &candidate,
            const ArtifactRootReference &source,
            const ArtifactRootReference &workload,
            const ArtifactRootReference &runtimeInput,
            const ArtifactStore &artifactStore) {
  std::map<ArtifactRootReference, std::vector<ArtifactRootReference>,
           decltype(&artifactRootReferenceLess)>
      entries(&artifactRootReferenceLess);
  entries[candidate];
  entries[source];
  entries[workload] = {source};
  entries[runtimeInput] = {source, workload};
  std::vector<CaseArtifactResolution::Entry> resolved;
  resolved.reserve(entries.size());
  for (auto &[reference, closure] : entries) {
    if (auto bytes = artifactStore.get(reference); !bytes)
      return bytes.takeError();
    llvm::sort(closure, artifactRootReferenceLess);
    closure.erase(std::unique(closure.begin(), closure.end()), closure.end());
    resolved.push_back({reference, std::move(closure)});
  }
  return CaseArtifactResolution::get(std::move(resolved));
}

} // namespace

llvm::Error primeStructuredProgramSourceObservations(
    const ArtifactRootReference &source, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput,
    const sim::NativeStructuredProgramObservations &observations) {
  return primeSourceObservationCacheImpl(source, workload, runtimeInput,
                                         observations);
}

llvm::Error registerStructuredProgramFunctionalModel() {
  if (llvm::Error error = standard_findings::registerStandardFindings())
    return error;
  if (llvm::Error error = registerEvaluationCaseSignature(kCaseSignature))
    return error;
  if (llvm::Error error = registerEvaluationModelDescriptor(kModelDescriptor))
    return error;
  return registerEvaluationModelProvider(kProvider);
}

EvaluationModelDescriptorRef structuredProgramFunctionalModelDescriptorRef() {
  return kModelDescriptor.reference();
}

CaseSubjectRoleRef structuredProgramFunctionalCandidateRole() {
  return kCandidateRole;
}

llvm::Error primeStructuredProgramFunctionalReplay(
    const ArtifactRootReference &candidateReference,
    const StructuredProgramFunctionalReplayInvocation &invocation,
    const ArtifactStore &artifactStore) {
  if (llvm::Error error = registerStructuredProgramFunctionalModel())
    return error;
  StructuredEvaluationInvocationCache *cache =
      detail::currentStructuredEvaluationCache();
  if (!cache)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: replay priming requires an "
        "active invocation cache");
  if (candidateReference.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      candidateReference.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      candidateReference.artifact !=
          invocation.candidate.structuredProgram.identity())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: replay candidate mismatch");
  if (auto stored = artifactStore.get(candidateReference); !stored)
    return stored.takeError();
  auto inputs = sim::importStructuredProgramSimulationInputs(
      invocation.workload, invocation.runtimeInput, artifactStore);
  if (!inputs)
    return inputs.takeError();
  if (inputs->structuredProgram.identity() !=
          invocation.sourceProgram.identity() ||
      inputs->workload.identity() != invocation.simulationWorkload.identity() ||
      inputs->runtimeInput.identity() !=
          invocation.simulationRuntimeInput.identity() ||
      invocation.workload.artifact !=
          invocation.simulationWorkload.identity() ||
      invocation.runtimeInput.artifact !=
          invocation.simulationRuntimeInput.identity())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: replay invocation mismatch");

  const detail::StructuredFunctionalCacheKey key = replayCacheKey(
      candidateReference, invocation.workload, invocation.runtimeInput);
  auto &cacheImpl = detail::StructuredEvaluationCacheAccess::impl(*cache);
  using FunctionalFlightMap = decltype(cacheImpl.functionalFlights);
  std::unique_ptr<detail::CacheFlightGuard<
      detail::StructuredFunctionalCacheKey, FunctionalFlightMap>>
      functionalFlight;
  {
    std::unique_lock<std::mutex> lock(cacheImpl.mutex);
    while (true) {
      if (auto found = cacheImpl.functionalResults.find(key);
          found != cacheImpl.functionalResults.end()) {
        if (cachedReplayMeetsLimits(*found->second, invocation.limits))
          return llvm::Error::success();
        cacheImpl.functionalResults.erase(found);
      }
      auto flight = cacheImpl.functionalFlights.find(key);
      if (flight == cacheImpl.functionalFlights.end()) {
        auto entry = std::make_shared<detail::CacheFlightEntry>();
        cacheImpl.functionalFlights.emplace(key, entry);
        functionalFlight = std::make_unique<detail::CacheFlightGuard<
            detail::StructuredFunctionalCacheKey, FunctionalFlightMap>>(
            cacheImpl.functionalFlights, cacheImpl.mutex,
            cacheImpl.flightChanged, key, std::move(entry));
        break;
      }
      auto entry = flight->second;
      cacheImpl.functionalSingleFlightWaitCount.fetch_add(
          1, std::memory_order_relaxed);
      cacheImpl.flightChanged.wait(lock, [&] { return entry->complete; });
    }
  }

  const ArtifactRootReference sourceReference{
      frontend::structuredProgramArtifactSchema.identity.str(),
      frontend::structuredProgramArtifactSchema.version,
      invocation.sourceProgram.identity()};
  if (llvm::Error error = primeStructuredProgramSourceObservations(
          sourceReference, invocation.workload, invocation.runtimeInput,
          invocation.sourceObservations))
    return error;

  const auto publishReplayCase =
      [&](const sim::CanonicalSimulationWorkload &replayWorkload,
          const sim::CanonicalSimulationRuntimeInput &replayInput)
      -> llvm::Expected<sim::SourceBackedDfgReplayCaseReference> {
    auto workloadReference =
        sim::publishSimulationWorkload(replayWorkload, artifactStore);
    if (!workloadReference)
      return workloadReference.takeError();
    auto runtimeInputReference =
        sim::publishSimulationRuntimeInput(replayInput, artifactStore);
    if (!runtimeInputReference)
      return runtimeInputReference.takeError();
    return sim::SourceBackedDfgReplayCaseReference{
        std::move(*workloadReference), std::move(*runtimeInputReference)};
  };
  auto classified = classifyReplayResult(sim::validateSourceBackedDfgReplay(
      invocation.sourceProgram, invocation.candidate,
      invocation.simulationWorkload, invocation.simulationRuntimeInput,
      invocation.limits, &invocation.sourceObservations, publishReplayCase));
  if (!classified)
    return classified.takeError();
  mapping_debug::emit(
      mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
      mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
        llvm::StringRef kind;
        switch (classified->kind) {
        case ReplayResultKind::Equivalent:
          kind = "equivalent";
          break;
        case ReplayResultKind::Mismatch:
          kind = "mismatch";
          break;
        case ReplayResultKind::Inapplicable:
          kind = "inapplicable";
          break;
        case ReplayResultKind::Unsupported:
          kind = "unsupported";
          break;
        case ReplayResultKind::CancelledOrTimeout:
          kind = "cancelled_or_timeout";
          break;
        }
        fields["context_kind"] = "structured_functional_replay";
        fields["replay_kind"] = kind;
      });
  auto value =
      std::make_shared<const CachedReplayResult>(std::move(*classified));
  std::lock_guard<std::mutex> lock(cacheImpl.mutex);
  if (auto existing = cacheImpl.functionalResults.find(key);
      existing != cacheImpl.functionalResults.end()) {
    if (!(*existing->second == *value))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_functional_model_invalid: nondeterministic replay");
    return llvm::Error::success();
  }
  if (cacheImpl.functionalResults.size() >=
      cacheImpl.limits.maximumFunctionalEntries) {
    cacheImpl.capacityBypassCount.fetch_add(1, std::memory_order_relaxed);
    return llvm::createStringError(
        std::make_error_code(std::errc::no_buffer_space),
        "structured_functional_model_incomplete: replay cache capacity "
        "exhausted");
  }
  auto [found, inserted] = cacheImpl.functionalResults.try_emplace(key, value);
  if (!inserted && !(*found->second == *value))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: nondeterministic replay");
  if (inserted)
    cacheImpl.functionalPrimeCount.fetch_add(1, std::memory_order_relaxed);
  return llvm::Error::success();
}

llvm::Expected<sim::SourceBackedDfgValidationResult>
getPrimedStructuredProgramFunctionalReplay(
    const ArtifactRootReference &candidate,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput) {
  StructuredEvaluationInvocationCache *cache =
      detail::currentStructuredEvaluationCache();
  if (!cache)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "structured_functional_model_invalid: replay lookup requires an "
        "active invocation cache");
  const detail::StructuredFunctionalCacheKey key =
      replayCacheKey(candidate, workload, runtimeInput);
  auto &impl = detail::StructuredEvaluationCacheAccess::impl(*cache);
  std::unique_lock<std::mutex> lock(impl.mutex);
  while (true) {
    auto found = impl.functionalResults.find(key);
    if (found != impl.functionalResults.end()) {
      impl.functionalHitCount.fetch_add(1, std::memory_order_relaxed);
      if (!found->second->replay)
        return llvm::createStringError(
            found->second->kind == ReplayResultKind::CancelledOrTimeout
                ? std::make_error_code(std::errc::timed_out)
                : std::make_error_code(std::errc::not_supported),
            found->second->kind == ReplayResultKind::CancelledOrTimeout
                ? "structured_functional_model_incomplete: replay timed out"
                : "structured_functional_model_unsupported: replay provider "
                  "unavailable");
      return *found->second->replay;
    }
    auto flight = impl.functionalFlights.find(key);
    if (flight == impl.functionalFlights.end()) {
      impl.functionalMissCount.fetch_add(1, std::memory_order_relaxed);
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "structured_functional_model_invalid: replay was not primed");
    }
    auto entry = flight->second;
    impl.functionalSingleFlightWaitCount.fetch_add(
        1, std::memory_order_relaxed);
    impl.flightChanged.wait(lock, [&] { return entry->complete; });
  }
}

llvm::Expected<PreparedStructuredProgramFunctionalEvaluation>
prepareStructuredProgramFunctionalEvaluation(
    const ArtifactRootReference &candidate,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &artifactStore, const BlobStore &blobStore) {
  if (llvm::Error error = registerStructuredProgramFunctionalModel())
    return std::move(error);
  auto inputs = sim::importStructuredProgramSimulationInputs(
      workload, runtimeInput, artifactStore);
  if (!inputs)
    return inputs.takeError();
  const ArtifactRootReference source{
      frontend::structuredProgramArtifactSchema.identity.str(),
      frontend::structuredProgramArtifactSchema.version,
      inputs->structuredProgram.identity()};
  auto resolution =
      resolveCase(candidate, source, workload, runtimeInput, artifactStore);
  if (!resolution)
    return resolution.takeError();
  auto bindings =
      EvaluationSubjectBindings::get({{kCandidateRole, {candidate}}});
  if (!bindings)
    return bindings.takeError();
  auto evaluationCase = EvaluationCase::get(
      caseSignatureRef(), std::move(*bindings), workload, runtimeInput, {},
      *resolution, artifactStore, blobStore);
  if (!evaluationCase)
    return evaluationCase.takeError();
  auto finding = FindingRequest::get(
      FindingQuery{standard_findings::FunctionalMismatch,
                   EvaluationScope{kWholeExactCaseScope, {}}},
      {}, *evaluationCase, *resolution, artifactStore);
  if (!finding)
    return finding.takeError();
  auto modelBinding =
      ResolvedModelBinding::project(kModelDescriptor.reference(), {}, config);
  if (!modelBinding)
    return modelBinding.takeError();
  auto request = EvaluationRequest::get(*evaluationCase, {}, {*finding},
                                        std::move(*modelBinding), 0,
                                        *resolution, artifactStore, blobStore);
  if (!request)
    return request.takeError();
  auto published = publishEvaluationRequest(*request, artifactStore);
  if (!published)
    return published.takeError();
  return PreparedStructuredProgramFunctionalEvaluation{
      std::move(*request), std::move(*resolution), kCandidateRole,
      FindingRequestOrdinal(0)};
}

} // namespace loom::evaluation::models
