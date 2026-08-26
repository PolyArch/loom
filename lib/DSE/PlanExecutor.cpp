#include "DSE/PlanExecutor.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "DSE/CandidateGeneratorRecovery.h"
#include "DSE/ResolvedConfigView.h"
#include "Evaluation/ModelDescriptor.h"
#include "Evaluation/ModelProvider.h"
#include "ExternalTool/InvocationBundle.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "PnR/PnrDerivedContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral kEvaluationRegistryIdentity =
    "loom.evaluation.registry";
constexpr llvm::StringLiteral kInvocationDirectory = "external-invocations";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dse_plan_executor_invalid: " + message);
}

bool bindingLess(const WorkUnitResourceBinding &lhs,
                 const WorkUnitResourceBinding &rhs) {
  return lhs.key < rhs.key;
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

BlobDigest workUnitDigest(const WorkUnitKey &key) {
  constexpr llvm::StringLiteral domain = "loom.dse.work_unit.v1";
  std::vector<std::uint8_t> bytes(domain.bytes_begin(), domain.bytes_end());
  bytes.push_back(0);
  appendU64(bytes, key.planNodeOrdinal());
  appendU64(bytes, key.descriptor().ownerRegistryIdentity().size());
  bytes.insert(bytes.end(),
               key.descriptor().ownerRegistryIdentity().bytes_begin(),
               key.descriptor().ownerRegistryIdentity().bytes_end());
  appendU32(bytes, key.descriptor().ownerRegistryVersion().major);
  appendU32(bytes, key.descriptor().ownerRegistryVersion().minor);
  appendU32(bytes, key.descriptor().ownerLocalKind());
  appendU64(bytes, key.stableOrdinal());
  return computeBlobDigest(bytes);
}

llvm::Expected<std::string> bundleDestination(const ExecutionJournal &journal,
                                              const WorkUnitKey &key) {
  std::filesystem::path directory(journal.localRunRoot().str());
  directory /= kInvocationDirectory.str();
  std::error_code error;
  std::filesystem::create_directory(directory, error);
  if (error)
    return llvm::createStringError(error, "cannot create invocation directory");
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(directory, error);
  if (error)
    return llvm::createStringError(error,
                                   "cannot inspect invocation directory");
  if (!std::filesystem::is_directory(status) ||
      std::filesystem::is_symlink(status))
    return invalid("invocation directory is not a non-symlink directory");
  directory /= formatBlobDigestHex(workUnitDigest(key));
  return directory.string();
}

llvm::Expected<std::uint64_t> terminalUnixNanoseconds() {
  const auto duration = std::chrono::system_clock::now().time_since_epoch();
  const auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
  if (nanoseconds <= 0)
    return invalid("system clock cannot represent a positive terminal time");
  return static_cast<std::uint64_t>(nanoseconds);
}

llvm::Expected<std::uint64_t>
activeNanoseconds(std::chrono::steady_clock::time_point begin) {
  const auto amount = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
  if (amount < 0)
    return invalid("monotonic clock moved backwards");
  return static_cast<std::uint64_t>(amount);
}

std::vector<ArtifactRootReference>
candidateResultRoots(const CandidateGeneratorProviderResult &result) {
  std::vector<ArtifactRootReference> roots;
  std::visit(
      [&](const auto &outcome) {
        using T = std::decay_t<decltype(outcome)>;
        const auto &bindings =
            [&]() -> const std::vector<CandidateGeneratorOutputBinding> & {
          if constexpr (std::is_same_v<T, CompletedCandidateGeneratorResult>)
            return outcome.outputBindings;
          else
            return outcome.retainedOutputBindings;
        }();
        for (const CandidateGeneratorOutputBinding &binding : bindings)
          roots.insert(roots.end(), binding.artifacts.begin(),
                       binding.artifacts.end());
      },
      result.outcome);
  llvm::sort(roots, artifactRootReferenceLess);
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
  return roots;
}

JournalWorkUnitStatus
candidateJournalStatus(const CandidateGeneratorProviderResult &result) {
  const auto *incomplete =
      std::get_if<IncompleteCandidateGeneratorResult>(&result.outcome);
  if (!incomplete)
    return JournalWorkUnitStatus::Completed;
  switch (incomplete->reason) {
  case CandidateGeneratorIncompleteReason::ProviderUnavailable:
  case CandidateGeneratorIncompleteReason::Unsupported:
    return JournalWorkUnitStatus::Unsupported;
  case CandidateGeneratorIncompleteReason::ExecutionFailed:
    return JournalWorkUnitStatus::Failed;
  case CandidateGeneratorIncompleteReason::CancelledOrTimeout:
    return JournalWorkUnitStatus::TimedOut;
  case CandidateGeneratorIncompleteReason::ProofNotEstablished:
  case CandidateGeneratorIncompleteReason::SemanticLimitReached:
    return JournalWorkUnitStatus::Completed;
  }
  llvm_unreachable("unknown candidate generator outcome");
}

JournalWorkUnitStatus
evidenceJournalStatus(const evaluation::EvaluationEvidence &evidence) {
  switch (evidence.outcomeKind()) {
  case evaluation::EvidenceOutcomeKind::Completed:
    return JournalWorkUnitStatus::Completed;
  case evaluation::EvidenceOutcomeKind::Unsupported:
    return JournalWorkUnitStatus::Unsupported;
  case evaluation::EvidenceOutcomeKind::ExecutionFailed:
    return JournalWorkUnitStatus::Failed;
  case evaluation::EvidenceOutcomeKind::CancelledOrTimeout:
    return JournalWorkUnitStatus::TimedOut;
  }
  llvm_unreachable("unknown Evidence outcome");
}

CandidateGeneratorProviderResult
makeIncompleteCandidateResult(const CandidateGeneratorDescriptor &descriptor,
                              CandidateGeneratorIncompleteReason reason) {
  std::vector<CandidateGeneratorOutputBinding> outputs;
  outputs.reserve(descriptor.outputSlots.size());
  for (const CandidateGeneratorOutputSlotDescriptor &slot :
       descriptor.outputSlots)
    outputs.push_back({slot.slot, {}});
  std::vector<CandidateGeneratorWorkUnitSummary> work;
  work.reserve(descriptor.workUnits.size());
  for (const CandidateGeneratorWorkUnitDescriptor &unit : descriptor.workUnits)
    work.push_back({unit.unit, 0, 0});
  return CandidateGeneratorProviderResult{
      IncompleteCandidateGeneratorResult{reason, std::move(outputs), {}},
      std::move(work)};
}

struct ProviderExecutionStopContext final {
  const ExecutionJournal *journal = nullptr;
  std::optional<std::uint64_t> notAfterUnixNanoseconds;
};

bool providerExecutionStopRequested(const void *opaque) {
  const auto &context =
      *static_cast<const ProviderExecutionStopContext *>(opaque);
  if (context.journal->gracefulStopRequested())
    return true;
  if (!context.notAfterUnixNanoseconds)
    return false;
  const auto elapsed = std::chrono::system_clock::now().time_since_epoch();
  const auto now =
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  return now <= 0 ||
         static_cast<std::uint64_t>(now) >= *context.notAfterUnixNanoseconds;
}

std::optional<std::chrono::steady_clock::duration>
providerExecutionRemainingTime(const void *opaque) {
  const auto &context =
      *static_cast<const ProviderExecutionStopContext *>(opaque);
  if (context.journal->gracefulStopRequested())
    return std::chrono::steady_clock::duration::zero();
  if (!context.notAfterUnixNanoseconds)
    return std::nullopt;
  const auto elapsed = std::chrono::system_clock::now().time_since_epoch();
  const auto now =
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  if (now <= 0 ||
      static_cast<std::uint64_t>(now) >= *context.notAfterUnixNanoseconds)
    return std::chrono::steady_clock::duration::zero();
  const auto remaining =
      *context.notAfterUnixNanoseconds - static_cast<std::uint64_t>(now);
  return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::nanoseconds(remaining));
}

ExecutionControlView
providerExecutionControl(const ProviderExecutionStopContext &context) {
  return ExecutionControlView{&context, providerExecutionStopRequested,
                              providerExecutionRemainingTime};
}

llvm::Expected<std::uint64_t>
addResourceAmount(std::uint64_t lhs, std::uint64_t rhs, llvm::StringRef field) {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
    return invalid(field + " resource claim overflows uint64");
  return lhs + rhs;
}

llvm::Expected<std::vector<CountedSiteResource>>
combineCountedResources(llvm::ArrayRef<CountedSiteResource> lhs,
                        llvm::ArrayRef<CountedSiteResource> rhs) {
  std::map<SiteResourceKey, std::uint64_t> combined;
  for (const CountedSiteResource &resource : lhs)
    combined.emplace(resource.key, resource.units);
  for (const CountedSiteResource &resource : rhs) {
    auto [found, inserted] = combined.emplace(resource.key, resource.units);
    if (!inserted)
      found->second = std::max(found->second, resource.units);
  }
  std::vector<CountedSiteResource> result;
  result.reserve(combined.size());
  for (const auto &[key, units] : combined)
    result.push_back({key, units});
  return result;
}

llvm::Expected<SiteResourceClaim>
combineEvidenceLifecycleClaims(const SiteResourceClaim &inProcess,
                               const SiteResourceClaim &external) {
  auto memory = addResourceAmount(inProcess.memoryBytes(),
                                  external.memoryBytes(), "memory");
  if (!memory)
    return memory.takeError();
  auto tools = combineCountedResources(inProcess.externalTools(),
                                       external.externalTools());
  if (!tools)
    return tools.takeError();
  auto licenses =
      combineCountedResources(inProcess.licenses(), external.licenses());
  if (!licenses)
    return licenses.takeError();
  return SiteResourceClaim::get(
      std::max(inProcess.cpuCores(), external.cpuCores()), *memory,
      std::max(inProcess.scratchBytes(), external.scratchBytes()), *tools,
      *licenses);
}

llvm::Expected<std::optional<JournalWorkUnitRecord>>
findOrQueue(ExecutionJournal &journal, const WorkUnitKey &key) {
  auto found = journal.find(key);
  if (!found)
    return found.takeError();
  if (*found)
    return found;
  if (llvm::Error error = journal.queue(key))
    return std::move(error);
  return journal.find(key);
}

static llvm::Expected<std::optional<CandidateGeneratorProviderResult>>
tryImportPreparedCandidateImpl(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const external_tool::ExternalToolInvocationExecutionObservation *execution,
    const ArtifactStore &store, const BlobStore &blobs) {
  auto imported =
      execution ? importCandidateGeneratorInvocation(inputs, binding, prepared,
                                                     *execution, store, blobs)
                : importCandidateGeneratorInvocation(inputs, binding, prepared,
                                                     store, blobs);
  if (imported)
    return std::optional<CandidateGeneratorProviderResult>(
        std::move(*imported));
  bool incomplete = false;
  llvm::Error remaining = llvm::handleErrors(
      imported.takeError(),
      [&](const external_tool::IncompleteExternalToolInvocationError &) {
        incomplete = true;
      });
  if (remaining)
    return std::move(remaining);
  if (!incomplete)
    return invalid("candidate import lost its failure");
  return std::optional<CandidateGeneratorProviderResult>{};
}

llvm::Expected<std::optional<CandidateGeneratorProviderResult>>
tryImportPreparedCandidate(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const ArtifactStore &store, const BlobStore &blobs) {
  return tryImportPreparedCandidateImpl(inputs, binding, prepared, nullptr,
                                        store, blobs);
}

llvm::Expected<std::optional<CandidateGeneratorProviderResult>>
tryImportPreparedCandidate(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const external_tool::ExternalToolInvocationExecutionObservation &execution,
    const ArtifactStore &store, const BlobStore &blobs) {
  return tryImportPreparedCandidateImpl(inputs, binding, prepared, &execution,
                                        store, blobs);
}

llvm::Expected<std::optional<evaluation::EvaluationEvidence>>
tryImportPreparedEvidence(
    const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    const evaluation::EvaluationModelPreparedInvocation &prepared,
    const ArtifactStore &store, const BlobStore &blobs) {
  auto imported = evaluation::importEvaluationModelInvocation(
      request, resolution, prepared, store, blobs);
  if (imported)
    return std::optional<evaluation::EvaluationEvidence>(std::move(*imported));
  bool incomplete = false;
  llvm::Error remaining = llvm::handleErrors(
      imported.takeError(),
      [&](const external_tool::IncompleteExternalToolInvocationError &) {
        incomplete = true;
      });
  if (remaining)
    return std::move(remaining);
  if (!incomplete)
    return invalid("Evidence import lost its failure");
  return std::optional<evaluation::EvaluationEvidence>{};
}

llvm::Expected<std::optional<evaluation::EvaluationEvidence>>
tryImportPreparedEvidence(
    const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    const evaluation::EvaluationModelPreparedInvocation &prepared,
    const external_tool::ExternalToolInvocationExecutionObservation &execution,
    const ArtifactStore &store, const BlobStore &blobs) {
  auto imported = evaluation::importEvaluationModelInvocation(
      request, resolution, prepared, execution, store, blobs);
  if (imported)
    return std::optional<evaluation::EvaluationEvidence>(std::move(*imported));
  bool incomplete = false;
  llvm::Error remaining = llvm::handleErrors(
      imported.takeError(),
      [&](const external_tool::IncompleteExternalToolInvocationError &) {
        incomplete = true;
      });
  if (remaining)
    return std::move(remaining);
  if (!incomplete)
    return invalid("receipt-bound Evidence import lost its failure");
  return std::optional<evaluation::EvaluationEvidence>{};
}

class RecoverablePlanWorkExecutor final : public detail::DsePlanWorkExecutor {
public:
  RecoverablePlanWorkExecutor(ExecutionJournal &journal,
                              SiteScheduler &scheduler,
                              const PlanExecutionPolicy &policy)
      : journal_(journal), scheduler_(scheduler), policy_(policy),
        fabricImportAttachment_(
            fabric::FabricArtifactImportSession::currentAttachment()),
        derivedContextAttachment_(
            pnr::PnrDerivedContextSession::currentAttachment()) {}

  bool shouldStopBeforeDispatch() const override {
    return journal_.gracefulStopRequested() ||
           (policy_.maximumDispatches() &&
            dispatched_.load(std::memory_order_relaxed) >=
                *policy_.maximumDispatches());
  }

  llvm::Expected<CandidateGeneratorProviderResult>
  executeGenerate(std::uint64_t planNodeOrdinal,
                  llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
                  llvm::ArrayRef<CandidateGeneratorOutputDemand> outputDemands,
                  const ResolvedCandidateGeneratorBinding &binding,
                  const ArtifactStore &store, const BlobStore &blobs) override;

  llvm::Expected<std::vector<CandidateGeneratorProviderResult>>
  executeGenerateBatch(llvm::ArrayRef<detail::DseGenerateExecutionTask> tasks,
                       const ArtifactStore &store,
                       const BlobStore &blobs) override;

  llvm::Error beginPromotion(
      std::uint64_t planNodeOrdinal,
      llvm::ArrayRef<ArtifactRootReference> candidates,
      llvm::ArrayRef<EvidenceObligationTemplateRef> obligations) override;

  llvm::Expected<std::vector<PromotionEvidenceExecutionResult>>
  execute(llvm::ArrayRef<PromotionEvidenceExecutionTask> tasks,
          const ArtifactStore &store, const BlobStore &blobs) override;

private:
  llvm::Expected<WorkUnitKey>
  generateKey(std::uint64_t planNodeOrdinal,
              CandidateGeneratorDescriptorRef descriptor) const;
  llvm::Expected<WorkUnitKey>
  evidenceKey(const PromotionEvidenceExecutionTask &task) const;
  llvm::Expected<SiteResourceClaim>
  resourceClaim(const WorkUnitKey &key,
                const BlobDigest *externalBinding) const;
  const WorkUnitResourceBinding *resourceBinding(const WorkUnitKey &key) const;
  llvm::Expected<SiteResourceClaim>
  evidenceLifecycleClaim(const WorkUnitKey &key,
                         const BlobDigest *externalBinding) const;
  llvm::Expected<
      std::optional<external_tool::ExternalToolInvocationExecutionObservation>>
  executePreparedInvocationUnderLease(
      const WorkUnitKey &key,
      const external_tool::PreparedExternalToolInvocation &prepared,
      ExecutionControlView executionControl);
  llvm::Expected<
      std::optional<external_tool::ExternalToolInvocationExecutionObservation>>
  executePreparedInvocation(
      const WorkUnitKey &key,
      const external_tool::PreparedExternalToolInvocation &prepared,
      bool reserveNewDispatch);
  llvm::Error
  settleStoppedExternalExecution(const WorkUnitKey &key,
                                 std::uint64_t activeWallTimeNanoseconds = 0);
  llvm::Expected<PromotionEvidenceExecutionResult>
  executeEvidence(const PromotionEvidenceExecutionTask &task,
                  const ArtifactStore &store, const BlobStore &blobs);
  llvm::Expected<evaluation::EvaluationEvidence>
  importFinalizedEvidence(const JournalWorkUnitRecord &record,
                          const PromotionEvidenceExecutionTask &task,
                          const ArtifactStore &store,
                          const BlobStore &blobs) const;
  bool executionStopRequested() const;
  bool reserveDispatch();

  ExecutionJournal &journal_;
  SiteScheduler &scheduler_;
  const PlanExecutionPolicy &policy_;
  std::uint64_t promotionNodeOrdinal_ = 0;
  std::vector<ArtifactRootReference> promotionCandidates_;
  std::vector<EvidenceObligationTemplateRef> promotionObligations_;
  std::atomic_uint64_t dispatched_{0};
  fabric::FabricArtifactImportSession::Attachment fabricImportAttachment_;
  pnr::PnrDerivedContextSession::Attachment derivedContextAttachment_;
};

bool RecoverablePlanWorkExecutor::executionStopRequested() const {
  const ProviderExecutionStopContext context{
      &journal_, policy_.dispatchNotAfterUnixNanoseconds()};
  return providerExecutionStopRequested(&context);
}

bool RecoverablePlanWorkExecutor::reserveDispatch() {
  if (journal_.gracefulStopRequested())
    return false;
  if (policy_.dispatchNotAfterUnixNanoseconds()) {
    const auto elapsed = std::chrono::system_clock::now().time_since_epoch();
    const auto now =
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    if (now <= 0 || static_cast<std::uint64_t>(now) >=
                        *policy_.dispatchNotAfterUnixNanoseconds())
      return false;
  }
  if (!policy_.maximumDispatches()) {
    dispatched_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  std::uint64_t observed = dispatched_.load(std::memory_order_relaxed);
  while (observed < *policy_.maximumDispatches()) {
    if (dispatched_.compare_exchange_weak(observed, observed + 1,
                                          std::memory_order_relaxed))
      return true;
  }
  return false;
}

llvm::Expected<WorkUnitKey> RecoverablePlanWorkExecutor::generateKey(
    std::uint64_t planNodeOrdinal,
    CandidateGeneratorDescriptorRef descriptor) const {
  auto owner = WorkUnitDescriptorRef::get(
      candidateGeneratorDescriptorSchema.identity,
      candidateGeneratorDescriptorSchema.version, descriptor.kind().ordinal());
  if (!owner)
    return owner.takeError();
  return WorkUnitKey::get(planNodeOrdinal, std::move(*owner), 0);
}

llvm::Expected<WorkUnitKey> RecoverablePlanWorkExecutor::evidenceKey(
    const PromotionEvidenceExecutionTask &task) const {
  auto candidate = llvm::lower_bound(promotionCandidates_, task.candidate,
                                     artifactRootReferenceLess);
  if (candidate == promotionCandidates_.end() || *candidate != task.candidate)
    return invalid(
        "Evidence task candidate is outside the active Promote node");
  auto obligation = llvm::lower_bound(
      promotionObligations_, task.obligationTemplate,
      [](EvidenceObligationTemplateRef lhs, EvidenceObligationTemplateRef rhs) {
        return lhs.ordinal() < rhs.ordinal();
      });
  if (obligation == promotionObligations_.end() ||
      obligation->ordinal() != task.obligationTemplate.ordinal())
    return invalid(
        "Evidence task obligation is outside the active Promote node");
  const std::uint64_t candidateOrdinal =
      static_cast<std::uint64_t>(candidate - promotionCandidates_.begin());
  const std::uint64_t obligationOrdinal =
      static_cast<std::uint64_t>(obligation - promotionObligations_.begin());
  const std::uint64_t obligationCount = promotionObligations_.size();
  if (obligationCount != 0 &&
      candidateOrdinal >
          (std::numeric_limits<std::uint64_t>::max() - obligationOrdinal) /
              obligationCount)
    return invalid("Evidence work ordinal overflows uint64");
  const std::uint64_t stableOrdinal =
      candidateOrdinal * obligationCount + obligationOrdinal;
  const evaluation::EvaluationModelDescriptorRef descriptor =
      task.request.modelBinding().descriptorRef();
  auto owner = WorkUnitDescriptorRef::get(kEvaluationRegistryIdentity,
                                          descriptor.schemaVersion(),
                                          descriptor.modelKind().ordinal());
  if (!owner)
    return owner.takeError();
  return WorkUnitKey::get(promotionNodeOrdinal_, std::move(*owner),
                          stableOrdinal);
}

llvm::Expected<SiteResourceClaim> RecoverablePlanWorkExecutor::resourceClaim(
    const WorkUnitKey &key, const BlobDigest *externalBinding) const {
  const WorkUnitResourceBinding *selected = resourceBinding(key);
  if (selected) {
    if (externalBinding) {
      const SiteResourceKey tool =
          SiteResourceKey::externalToolBinding(*externalBinding);
      if (!llvm::any_of(selected->claim.externalTools(),
                        [&](const CountedSiteResource &resource) {
                          return resource.key == tool && resource.units != 0;
                        }))
        return invalid("external resource override omits its exact tool "
                       "binding claim");
      if (policy_.externalSite() && policy_.externalSite()->claimLicense) {
        const SiteResourceKey license =
            SiteResourceKey::licenseBinding(*externalBinding);
        if (!llvm::any_of(selected->claim.licenses(),
                          [&](const CountedSiteResource &resource) {
                            return resource.key == license &&
                                   resource.units != 0;
                          }))
          return invalid("external resource override omits its exact license "
                         "binding claim");
      }
    }
    return selected->claim;
  }
  if (!externalBinding)
    return policy_.inProcessClaim();
  if (!policy_.externalSite())
    return invalid("external provider has no execution site");
  const ExternalExecutionSite &site = *policy_.externalSite();
  const CountedSiteResource tool{
      SiteResourceKey::externalToolBinding(*externalBinding), 1};
  std::vector<CountedSiteResource> licenses;
  if (site.claimLicense)
    licenses.push_back({SiteResourceKey::licenseBinding(*externalBinding), 1});
  return SiteResourceClaim::get(site.cpuCores, site.memoryBytes,
                                site.scratchBytes, {tool}, licenses);
}

const WorkUnitResourceBinding *
RecoverablePlanWorkExecutor::resourceBinding(const WorkUnitKey &key) const {
  auto selected = llvm::lower_bound(
      policy_.resourceBindings(), key,
      [](const WorkUnitResourceBinding &binding, const WorkUnitKey &candidate) {
        return binding.key < candidate;
      });
  return selected != policy_.resourceBindings().end() && selected->key == key
             ? &*selected
             : nullptr;
}

llvm::Expected<SiteResourceClaim>
RecoverablePlanWorkExecutor::evidenceLifecycleClaim(
    const WorkUnitKey &key, const BlobDigest *externalBinding) const {
  if (resourceBinding(key))
    return resourceClaim(key, externalBinding);
  if (!policy_.externalSite() ||
      policy_.externalSite()->disposition !=
          ExternalAttemptDisposition::ExecutePrepared)
    return policy_.inProcessClaim();
  const ExternalExecutionSite &site = *policy_.externalSite();
  llvm::Expected<SiteResourceClaim> external =
      externalBinding ? resourceClaim(key, externalBinding)
                      : SiteResourceClaim::get(site.cpuCores, site.memoryBytes,
                                               site.scratchBytes);
  if (!external)
    return external.takeError();
  return combineEvidenceLifecycleClaims(policy_.inProcessClaim(), *external);
}

llvm::Expected<
    std::optional<external_tool::ExternalToolInvocationExecutionObservation>>
RecoverablePlanWorkExecutor::executePreparedInvocationUnderLease(
    const WorkUnitKey &key,
    const external_tool::PreparedExternalToolInvocation &prepared,
    ExecutionControlView executionControl) {
  if (llvm::Error error = journal_.beginPreparedExecution(key))
    return std::move(error);
  const auto begin = std::chrono::steady_clock::now();
  auto execution = external_tool::executeExternalToolInvocationBundleObserved(
      prepared, executionControl);
  auto active = activeNanoseconds(begin);
  if (!active)
    return active.takeError();
  auto end = terminalUnixNanoseconds();
  if (!end)
    return end.takeError();
  llvm::Error intervalError = journal_.recordPreparedExecutionInterval(
      key, *active, *end,
      execution
          ? std::optional<
                external_tool::ExternalToolInvocationExecutionObservation>(
                *execution)
          : std::nullopt);
  if (!execution) {
    bool admissionStopped = false;
    llvm::Error remaining = llvm::handleErrors(
        execution.takeError(),
        [&](const external_tool::ExternalToolExecutionAdmissionStoppedError &) {
          admissionStopped = true;
        });
    if (remaining)
      return llvm::joinErrors(std::move(remaining), std::move(intervalError));
    if (intervalError)
      return std::move(intervalError);
    if (!admissionStopped)
      return invalid("external execution admission lost its typed failure");
    return std::optional<
        external_tool::ExternalToolInvocationExecutionObservation>{};
  }
  if (intervalError)
    return std::move(intervalError);
  return std::optional<
      external_tool::ExternalToolInvocationExecutionObservation>(
      std::move(*execution));
}

llvm::Expected<
    std::optional<external_tool::ExternalToolInvocationExecutionObservation>>
RecoverablePlanWorkExecutor::executePreparedInvocation(
    const WorkUnitKey &key,
    const external_tool::PreparedExternalToolInvocation &prepared,
    bool reserveNewDispatch) {
  if (!policy_.externalSite() ||
      policy_.externalSite()->disposition !=
          ExternalAttemptDisposition::ExecutePrepared ||
      (reserveNewDispatch && !reserveDispatch()))
    return std::optional<
        external_tool::ExternalToolInvocationExecutionObservation>{};
  auto bindingDigest =
      external_tool::deriveExternalToolExecutionBindingDigest(prepared);
  if (!bindingDigest)
    return bindingDigest.takeError();
  auto claim = resourceClaim(key, &*bindingDigest);
  if (!claim)
    return claim.takeError();
  const ProviderExecutionStopContext stopContext{
      &journal_, policy_.dispatchNotAfterUnixNanoseconds()};
  const ExecutionControlView executionControl =
      providerExecutionControl(stopContext);
  auto acquired = scheduler_.acquire(key, *claim, executionControl);
  if (!acquired)
    return acquired.takeError();
  if (!*acquired)
    return std::optional<
        external_tool::ExternalToolInvocationExecutionObservation>{};
  SiteResourceLease lease = std::move(**acquired);
  auto execution =
      executePreparedInvocationUnderLease(key, prepared, executionControl);
  if (!execution)
    return execution.takeError();
  return std::move(*execution);
}

llvm::Error RecoverablePlanWorkExecutor::settleStoppedExternalExecution(
    const WorkUnitKey &key, std::uint64_t activeWallTimeNanoseconds) {
  auto terminalTime = terminalUnixNanoseconds();
  if (!terminalTime)
    return terminalTime.takeError();
  return journal_.markTerminal(key, JournalWorkUnitStatus::TimedOut,
                               activeWallTimeNanoseconds, *terminalTime, {});
}

llvm::Expected<CandidateGeneratorProviderResult>
RecoverablePlanWorkExecutor::executeGenerate(
    std::uint64_t planNodeOrdinal,
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    llvm::ArrayRef<CandidateGeneratorOutputDemand> outputDemands,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &blobs) {
  const CandidateGeneratorDescriptor *descriptor =
      binding.descriptorRef().descriptor();
  if (!descriptor)
    return invalid("Generate binding lost its descriptor");
  auto key = generateKey(planNodeOrdinal, binding.descriptorRef());
  if (!key)
    return key.takeError();
  auto record = findOrQueue(journal_, *key);
  if (!record)
    return record.takeError();
  if (!*record)
    return invalid("queued Generate work unit cannot be reloaded");

  if ((*record)->status == JournalWorkUnitStatus::Prepared) {
    if (descriptor->providerForm != ProviderForm::ExternalPrepareImport ||
        !(*record)->preparedInvocation)
      return invalid("prepared Generate record has the wrong provider form");
    auto imported = tryImportPreparedCandidate(
        inputs, binding, *(*record)->preparedInvocation, store, blobs);
    if (!imported)
      return imported.takeError();
    if (!*imported) {
      auto executed =
          executePreparedInvocation(*key, *(*record)->preparedInvocation, true);
      if (!executed)
        return executed.takeError();
      if (!*executed)
        return makeIncompleteCandidateResult(
            *descriptor,
            CandidateGeneratorIncompleteReason::CancelledOrTimeout);
      if ((**executed).exitCode ==
          external_tool::externalToolExecutionStoppedExitCode) {
        if (llvm::Error error = settleStoppedExternalExecution(*key))
          return std::move(error);
        return makeIncompleteCandidateResult(
            *descriptor,
            CandidateGeneratorIncompleteReason::CancelledOrTimeout);
      }
      imported = tryImportPreparedCandidate(inputs, binding,
                                            *(*record)->preparedInvocation,
                                            **executed, store, blobs);
      if (!imported)
        return imported.takeError();
      if (!*imported)
        return makeIncompleteCandidateResult(
            *descriptor,
            CandidateGeneratorIncompleteReason::CancelledOrTimeout);
      (**imported).dispatched = (*executed)->invokedExternalTool;
    }
    std::vector<ArtifactRootReference> roots = candidateResultRoots(**imported);
    auto terminalTime = terminalUnixNanoseconds();
    if (!terminalTime)
      return terminalTime.takeError();
    if (llvm::Error error = journal_.markTerminal(
            *key, candidateJournalStatus(**imported), 0, *terminalTime, roots))
      return std::move(error);
    return std::move(**imported);
  }

  const bool terminal = (*record)->status == JournalWorkUnitStatus::Completed ||
                        (*record)->status == JournalWorkUnitStatus::Failed ||
                        (*record)->status == JournalWorkUnitStatus::TimedOut ||
                        (*record)->status == JournalWorkUnitStatus::Unsupported;
  if (terminal) {
    llvm::Expected<CandidateGeneratorProviderResult> replay =
        [&]() -> llvm::Expected<CandidateGeneratorProviderResult> {
      if (descriptor->providerForm == ProviderForm::ExternalPrepareImport) {
        if (!(*record)->preparedInvocation)
          return invalid("terminal external Generate record lost its attempt");
        if ((*record)->finalizedWorkRecord)
          return invalid(
              "terminal external Generate record has an in-process recovery "
              "record");
        auto imported = tryImportPreparedCandidate(
            inputs, binding, *(*record)->preparedInvocation, store, blobs);
        if (!imported)
          return imported.takeError();
        if (!*imported) {
          if ((*record)->status == JournalWorkUnitStatus::TimedOut)
            return makeIncompleteCandidateResult(
                *descriptor,
                CandidateGeneratorIncompleteReason::CancelledOrTimeout);
          return invalid("terminal Generate attempt is incomplete");
        }
        return std::move(**imported);
      }
      if ((*record)->preparedInvocation)
        return invalid(
            "terminal in-process Generate record has an external attempt");
      if (!(*record)->finalizedWorkRecord)
        return invalid(
            "terminal in-process Generate work has no owner recovery record");
      const OwnerFinalizedWorkRecordRef &reference =
          *(*record)->finalizedWorkRecord;
      if (reference.schemaIdentity !=
              candidateGeneratorFinalizedWorkRecordSchemaIdentity ||
          reference.schemaVersion !=
              candidateGeneratorFinalizedWorkRecordSchemaVersion)
        return invalid(
            "terminal in-process Generate work has a foreign recovery owner");
      auto recovered = importCandidateGeneratorFinalizedWorkRecord(
          reference.payloadDigest, journal_.runKey(), *key, inputs, binding,
          store, blobs);
      if (!recovered)
        return recovered.takeError();
      const bool completed =
          std::holds_alternative<CompletedCandidateGeneratorResult>(
              recovered->outcome);
      const auto &outputs =
          completed
              ? std::get<CompletedCandidateGeneratorResult>(recovered->outcome)
                    .outputBindings
              : std::get<IncompleteCandidateGeneratorResult>(recovered->outcome)
                    .retainedOutputBindings;
      const auto &edges =
          completed
              ? std::get<CompletedCandidateGeneratorResult>(recovered->outcome)
                    .lineageEdges
              : std::get<IncompleteCandidateGeneratorResult>(recovered->outcome)
                    .lineageEdges;
      if (llvm::Error error = validateCanonicalCandidateGeneratorInvocation(
              inputs, binding, outputs, edges, completed, store))
        return std::move(error);
      if (llvm::Error error = validateCandidateGeneratorWorkSummary(
              binding.descriptorRef(), recovered->workSummary))
        return std::move(error);
      return recovered;
    }();
    if (!replay)
      return replay.takeError();
    if (candidateJournalStatus(*replay) != (*record)->status)
      return invalid("replayed Generate status differs from the Journal");
    if (candidateResultRoots(*replay) != (*record)->finalizedOutputs)
      return invalid("replayed Generate roots differ from the Journal");
    return replay;
  }
  if ((*record)->status != JournalWorkUnitStatus::Queued)
    return invalid("Generate work unit is already running");
  if (descriptor->providerForm == ProviderForm::ExternalPrepareImport &&
      !policy_.externalSite())
    return makeIncompleteCandidateResult(
        *descriptor, CandidateGeneratorIncompleteReason::ProviderUnavailable);
  if (!reserveDispatch())
    return makeIncompleteCandidateResult(
        *descriptor, CandidateGeneratorIncompleteReason::CancelledOrTimeout);

  if (descriptor->providerForm == ProviderForm::InProcess) {
    auto claim = resourceClaim(*key, nullptr);
    if (!claim)
      return claim.takeError();
    const ProviderExecutionStopContext stopContext{
        &journal_, policy_.dispatchNotAfterUnixNanoseconds()};
    const ExecutionControlView executionControl =
        providerExecutionControl(stopContext);
    auto acquired = scheduler_.acquire(*key, *claim, executionControl);
    if (!acquired)
      return acquired.takeError();
    if (!*acquired)
      return makeIncompleteCandidateResult(
          *descriptor, CandidateGeneratorIncompleteReason::CancelledOrTimeout);
    SiteResourceLease lease = std::move(**acquired);
    if (llvm::Error error = journal_.markRunning(*key))
      return std::move(error);
    const auto begin = std::chrono::steady_clock::now();
    const ExecutionResourceBudget executionBudget{
        lease.claim().cpuCores() == 0
            ? std::nullopt
            : std::optional<std::uint64_t>(lease.claim().cpuCores()),
        lease.claim().memoryBytes() == 0
            ? std::nullopt
            : std::optional<std::uint64_t>(lease.claim().memoryBytes())};
    const CandidateGeneratorInvocationView invocation(
        executionControl, outputDemands, executionBudget);
    auto generated =
        invokeCandidateGenerator(inputs, binding, store, blobs, invocation);
    if (!generated) {
      llvm::Error reset = journal_.queue(*key);
      return llvm::joinErrors(generated.takeError(), std::move(reset));
    }
    auto active = activeNanoseconds(begin);
    if (!active)
      return active.takeError();
    auto terminalTime = terminalUnixNanoseconds();
    if (!terminalTime)
      return terminalTime.takeError();
    std::vector<ArtifactRootReference> roots = candidateResultRoots(*generated);
    auto recovery = publishCandidateGeneratorFinalizedWorkRecord(
        journal_.runKey(), *key, inputs, binding, *generated, store, blobs);
    if (!recovery) {
      llvm::Error reset = journal_.queue(*key);
      return llvm::joinErrors(recovery.takeError(), std::move(reset));
    }
    OwnerFinalizedWorkRecordRef recoveryRef{
        candidateGeneratorFinalizedWorkRecordSchemaIdentity.str(),
        candidateGeneratorFinalizedWorkRecordSchemaVersion, *recovery};
    if (llvm::Error error = journal_.markTerminal(
            *key, candidateJournalStatus(*generated), *active, *terminalTime,
            roots, std::move(recoveryRef)))
      return std::move(error);
    return generated;
  }

  const ProviderExecutionStopContext preparationStopContext{
      &journal_, policy_.dispatchNotAfterUnixNanoseconds()};
  auto acquiredPreparation =
      scheduler_.acquire(*key, policy_.inProcessClaim(),
                         providerExecutionControl(preparationStopContext));
  if (!acquiredPreparation)
    return acquiredPreparation.takeError();
  if (!*acquiredPreparation)
    return makeIncompleteCandidateResult(
        *descriptor, CandidateGeneratorIncompleteReason::CancelledOrTimeout);
  SiteResourceLease preparationLease = std::move(**acquiredPreparation);
  if (llvm::Error error = journal_.markRunning(*key))
    return std::move(error);
  const auto preparationBegin = std::chrono::steady_clock::now();
  auto destination = bundleDestination(journal_, *key);
  if (!destination) {
    llvm::Error reset = journal_.queue(*key);
    return llvm::joinErrors(destination.takeError(), std::move(reset));
  }
  external_tool::ExternalToolPreparationContext context{
      policy_.externalSite()->localToolConfig, std::move(*destination)};
  auto prepared = prepareCandidateGeneratorInvocation(inputs, binding, store,
                                                      blobs, context);
  if (!prepared) {
    llvm::Error reset = journal_.queue(*key);
    return llvm::joinErrors(prepared.takeError(), std::move(reset));
  }
  auto preparationActive = activeNanoseconds(preparationBegin);
  if (!preparationActive)
    return preparationActive.takeError();
  auto preparationEnd = terminalUnixNanoseconds();
  if (!preparationEnd)
    return preparationEnd.takeError();
  if (llvm::Error error = journal_.recordPrepared(
          *key, *prepared, *preparationActive, *preparationEnd))
    return std::move(error);
  preparationLease.release();
  if (policy_.externalSite()->disposition ==
      ExternalAttemptDisposition::PrepareOnly)
    return makeIncompleteCandidateResult(
        *descriptor, CandidateGeneratorIncompleteReason::CancelledOrTimeout);
  auto executed = executePreparedInvocation(*key, *prepared, false);
  if (!executed)
    return executed.takeError();
  if (!*executed)
    return makeIncompleteCandidateResult(
        *descriptor, CandidateGeneratorIncompleteReason::CancelledOrTimeout);
  if ((**executed).exitCode ==
      external_tool::externalToolExecutionStoppedExitCode) {
    if (llvm::Error error = settleStoppedExternalExecution(*key))
      return std::move(error);
    return makeIncompleteCandidateResult(
        *descriptor, CandidateGeneratorIncompleteReason::CancelledOrTimeout);
  }
  auto imported = tryImportPreparedCandidate(inputs, binding, *prepared,
                                             **executed, store, blobs);
  if (!imported)
    return imported.takeError();
  if (!*imported)
    return makeIncompleteCandidateResult(
        *descriptor, CandidateGeneratorIncompleteReason::CancelledOrTimeout);
  (**imported).dispatched = (*executed)->invokedExternalTool;
  auto terminalTime = terminalUnixNanoseconds();
  if (!terminalTime)
    return terminalTime.takeError();
  std::vector<ArtifactRootReference> roots = candidateResultRoots(**imported);
  if (llvm::Error error = journal_.markTerminal(
          *key, candidateJournalStatus(**imported), 0, *terminalTime, roots))
    return std::move(error);
  return std::move(**imported);
}

llvm::Expected<std::vector<CandidateGeneratorProviderResult>>
RecoverablePlanWorkExecutor::executeGenerateBatch(
    llvm::ArrayRef<detail::DseGenerateExecutionTask> tasks,
    const ArtifactStore &store, const BlobStore &blobs) {
  if (tasks.empty())
    return std::vector<CandidateGeneratorProviderResult>{};
  const std::size_t workerCount = static_cast<std::size_t>(
      std::min<std::uint64_t>(policy_.workerCount(), tasks.size()));
  if (workerCount == 1 || policy_.maximumDispatches()) {
    std::vector<CandidateGeneratorProviderResult> results;
    results.reserve(tasks.size());
    for (const detail::DseGenerateExecutionTask &task : tasks) {
      auto result =
          executeGenerate(task.planNodeOrdinal, task.inputs, task.outputDemands,
                          task.binding, store, blobs);
      if (!result)
        return result.takeError();
      results.push_back(std::move(*result));
    }
    return results;
  }

  using WorkResult = llvm::Expected<CandidateGeneratorProviderResult>;
  std::vector<std::unique_ptr<WorkResult>> results(tasks.size());
  std::atomic_size_t next{0};
  llvm::DefaultThreadPool pool(llvm::heavyweight_hardware_concurrency(
      static_cast<unsigned>(workerCount)));
  for (std::size_t worker = 0; worker != workerCount; ++worker)
    pool.async([&] {
      std::unique_ptr<fabric::FabricArtifactImportSession> importSession;
      std::unique_ptr<pnr::PnrDerivedContextSession> derivedContextSession;
      if (fabricImportAttachment_)
        importSession = std::make_unique<fabric::FabricArtifactImportSession>(
            fabricImportAttachment_);
      if (derivedContextAttachment_)
        derivedContextSession = std::make_unique<pnr::PnrDerivedContextSession>(
            derivedContextAttachment_);
      while (true) {
        const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= tasks.size())
          break;
        const detail::DseGenerateExecutionTask &task = tasks[index];
        results[index] = std::make_unique<WorkResult>(
            executeGenerate(task.planNodeOrdinal, task.inputs,
                            task.outputDemands, task.binding, store, blobs));
      }
    });
  pool.wait();

  std::vector<CandidateGeneratorProviderResult> ordered;
  ordered.reserve(tasks.size());
  for (std::unique_ptr<WorkResult> &result : results) {
    if (!result)
      return invalid("Generate worker did not publish a result");
    if (!*result)
      return result->takeError();
    ordered.push_back(std::move(**result));
  }
  return ordered;
}

llvm::Error RecoverablePlanWorkExecutor::beginPromotion(
    std::uint64_t planNodeOrdinal,
    llvm::ArrayRef<ArtifactRootReference> candidates,
    llvm::ArrayRef<EvidenceObligationTemplateRef> obligations) {
  if (!llvm::is_sorted(candidates, artifactRootReferenceLess) ||
      std::adjacent_find(candidates.begin(), candidates.end()) !=
          candidates.end())
    return invalid("Promote candidates are not canonical and unique");
  const auto obligationLess = [](EvidenceObligationTemplateRef lhs,
                                 EvidenceObligationTemplateRef rhs) {
    return lhs.ordinal() < rhs.ordinal();
  };
  if (!llvm::is_sorted(obligations, obligationLess) ||
      std::adjacent_find(obligations.begin(), obligations.end()) !=
          obligations.end())
    return invalid("Promote obligations are not canonical and unique");
  promotionNodeOrdinal_ = planNodeOrdinal;
  promotionCandidates_.assign(candidates.begin(), candidates.end());
  promotionObligations_.assign(obligations.begin(), obligations.end());
  return llvm::Error::success();
}

llvm::Expected<evaluation::EvaluationEvidence>
RecoverablePlanWorkExecutor::importFinalizedEvidence(
    const JournalWorkUnitRecord &record,
    const PromotionEvidenceExecutionTask &task, const ArtifactStore &store,
    const BlobStore &blobs) const {
  if (record.finalizedOutputs.size() != 1 ||
      record.finalizedOutputs.front().schemaIdentity !=
          evaluation::EvaluationEvidence::artifactSchema.identity ||
      record.finalizedOutputs.front().schemaVersion !=
          evaluation::EvaluationEvidence::artifactSchema.version)
    return invalid("terminal Evidence work has no exact Evidence root");
  auto evidence = evaluation::importEvaluationEvidence(
      record.finalizedOutputs.front(), *task.resolution, store, blobs);
  if (!evidence)
    return evidence.takeError();
  if (evidence->requestRef() !=
      evaluation::evaluationRequestReference(task.request))
    return invalid("terminal Evidence belongs to another Request");
  return evidence;
}

llvm::Expected<PromotionEvidenceExecutionResult>
RecoverablePlanWorkExecutor::executeEvidence(
    const PromotionEvidenceExecutionTask &task, const ArtifactStore &store,
    const BlobStore &blobs) {
  auto key = evidenceKey(task);
  if (!key)
    return key.takeError();
  auto record = findOrQueue(journal_, *key);
  if (!record)
    return record.takeError();
  if (!*record)
    return invalid("queued Evidence work unit cannot be reloaded");

  const evaluation::EvaluationModelDescriptor *descriptor =
      task.request.modelBinding().descriptorRef().descriptor();
  if (!descriptor)
    return invalid("Evidence Request lost its model descriptor");

  if ((*record)->status == JournalWorkUnitStatus::Prepared) {
    if (descriptor->providerForm != ProviderForm::ExternalPrepareImport ||
        !(*record)->preparedInvocation)
      return invalid("prepared Evidence record has the wrong provider form");
    const auto lifecycleBegin = std::chrono::steady_clock::now();
    const ProviderExecutionStopContext stopContext{
        &journal_, policy_.dispatchNotAfterUnixNanoseconds()};
    const ExecutionControlView executionControl =
        providerExecutionControl(stopContext);
    auto lifecycleClaim = evidenceLifecycleClaim(*key, nullptr);
    if (!lifecycleClaim)
      return lifecycleClaim.takeError();
    auto acquired = scheduler_.acquire(*key, *lifecycleClaim, executionControl);
    if (!acquired)
      return acquired.takeError();
    if (!*acquired)
      return PromotionEvidenceExecutionResult{
          PromotionAcquisitionIncompleteReason::CancelledOrTimeout};
    SiteResourceLease lifecycleLease = std::move(**acquired);
    auto live = evaluation::bindPreparedEvaluationModelInvocation(
        task.request, *task.resolution, *(*record)->preparedInvocation, store,
        blobs);
    if (!live)
      return live.takeError();
    auto imported = tryImportPreparedEvidence(task.request, *task.resolution,
                                              *live, store, blobs);
    if (!imported)
      return imported.takeError();
    if (!*imported) {
      if (!policy_.externalSite() ||
          policy_.externalSite()->disposition !=
              ExternalAttemptDisposition::ExecutePrepared ||
          !reserveDispatch())
        return PromotionEvidenceExecutionResult{
            executionStopRequested()
                ? PromotionAcquisitionIncompleteReason::CancelledOrTimeout
                : PromotionAcquisitionIncompleteReason::ProviderUnavailable};
      auto bindingDigest =
          external_tool::deriveExternalToolExecutionBindingDigest(
              *(*record)->preparedInvocation);
      if (!bindingDigest)
        return bindingDigest.takeError();
      auto boundClaim = evidenceLifecycleClaim(*key, &*bindingDigest);
      if (!boundClaim)
        return boundClaim.takeError();
      auto bound = scheduler_.bindCountedResources(lifecycleLease, *boundClaim,
                                                   executionControl);
      if (!bound)
        return bound.takeError();
      if (!*bound)
        return PromotionEvidenceExecutionResult{
            PromotionAcquisitionIncompleteReason::CancelledOrTimeout};
      auto executed = executePreparedInvocationUnderLease(
          *key, *(*record)->preparedInvocation, executionControl);
      if (!executed)
        return executed.takeError();
      if (!*executed)
        return PromotionEvidenceExecutionResult{
            PromotionAcquisitionIncompleteReason::CancelledOrTimeout};
      if ((**executed).exitCode ==
          external_tool::externalToolExecutionStoppedExitCode) {
        auto active = activeNanoseconds(lifecycleBegin);
        if (!active)
          return active.takeError();
        if (llvm::Error error = settleStoppedExternalExecution(*key, *active))
          return std::move(error);
        return PromotionEvidenceExecutionResult{
            PromotionAcquisitionIncompleteReason::CancelledOrTimeout};
      }
      imported = tryImportPreparedEvidence(task.request, *task.resolution,
                                           *live, **executed, store, blobs);
      if (!imported)
        return imported.takeError();
      if (!*imported)
        return PromotionEvidenceExecutionResult{
            PromotionAcquisitionIncompleteReason::ProviderUnavailable};
    }
    auto root = evaluation::publishEvaluationEvidence(**imported, store);
    if (!root)
      return root.takeError();
    auto terminalTime = terminalUnixNanoseconds();
    if (!terminalTime)
      return terminalTime.takeError();
    auto active = activeNanoseconds(lifecycleBegin);
    if (!active)
      return active.takeError();
    const std::array<ArtifactRootReference, 1> outputs = {*root};
    if (llvm::Error error =
            journal_.markTerminal(*key, evidenceJournalStatus(**imported),
                                  *active, *terminalTime, outputs))
      return std::move(error);
    return PromotionEvidenceExecutionResult{std::move(**imported)};
  }

  const bool terminal = (*record)->status == JournalWorkUnitStatus::Completed ||
                        (*record)->status == JournalWorkUnitStatus::Failed ||
                        (*record)->status == JournalWorkUnitStatus::TimedOut ||
                        (*record)->status == JournalWorkUnitStatus::Unsupported;
  if (terminal) {
    if ((*record)->status == JournalWorkUnitStatus::TimedOut &&
        (*record)->preparedInvocation && (*record)->finalizedOutputs.empty())
      return PromotionEvidenceExecutionResult{
          PromotionAcquisitionIncompleteReason::CancelledOrTimeout};
    auto evidence = importFinalizedEvidence(**record, task, store, blobs);
    if (!evidence)
      return evidence.takeError();
    if (evidenceJournalStatus(*evidence) != (*record)->status)
      return invalid("imported Evidence status differs from the Journal");
    return PromotionEvidenceExecutionResult{std::move(*evidence)};
  }
  if ((*record)->status != JournalWorkUnitStatus::Queued)
    return invalid("Evidence work unit is already running");
  if (descriptor->providerForm == ProviderForm::ExternalPrepareImport &&
      !policy_.externalSite())
    return PromotionEvidenceExecutionResult{
        PromotionAcquisitionIncompleteReason::ProviderUnavailable};
  if (!reserveDispatch())
    return PromotionEvidenceExecutionResult{
        executionStopRequested()
            ? PromotionAcquisitionIncompleteReason::CancelledOrTimeout
            : PromotionAcquisitionIncompleteReason::ProviderUnavailable};

  std::optional<evaluation::EvaluationEvidence> evidence;
  std::uint64_t active = 0;
  const auto lifecycleBegin = std::chrono::steady_clock::now();
  if (descriptor->providerForm == ProviderForm::InProcess) {
    auto claim = resourceClaim(*key, nullptr);
    if (!claim)
      return claim.takeError();
    const ProviderExecutionStopContext stopContext{
        &journal_, policy_.dispatchNotAfterUnixNanoseconds()};
    auto acquired =
        scheduler_.acquire(*key, *claim, providerExecutionControl(stopContext));
    if (!acquired)
      return acquired.takeError();
    if (!*acquired)
      return PromotionEvidenceExecutionResult{
          PromotionAcquisitionIncompleteReason::CancelledOrTimeout};
    SiteResourceLease lease = std::move(**acquired);
    if (llvm::Error error = journal_.markRunning(*key))
      return std::move(error);
    const auto begin = std::chrono::steady_clock::now();
    auto evaluated = evaluation::evaluateRequest(task.request, *task.resolution,
                                                 store, blobs);
    if (!evaluated) {
      llvm::Error reset = journal_.queue(*key);
      return llvm::joinErrors(evaluated.takeError(), std::move(reset));
    }
    evidence = std::move(*evaluated);
    auto measured = activeNanoseconds(begin);
    if (!measured)
      return measured.takeError();
    active = *measured;
  } else {
    const ProviderExecutionStopContext preparationStopContext{
        &journal_, policy_.dispatchNotAfterUnixNanoseconds()};
    auto preparationClaim = evidenceLifecycleClaim(*key, nullptr);
    if (!preparationClaim)
      return preparationClaim.takeError();
    auto acquiredPreparation =
        scheduler_.acquire(*key, *preparationClaim,
                           providerExecutionControl(preparationStopContext));
    if (!acquiredPreparation)
      return acquiredPreparation.takeError();
    if (!*acquiredPreparation)
      return PromotionEvidenceExecutionResult{
          PromotionAcquisitionIncompleteReason::CancelledOrTimeout};
    SiteResourceLease preparationLease = std::move(**acquiredPreparation);
    if (llvm::Error error = journal_.markRunning(*key))
      return std::move(error);
    const auto preparationBegin = std::chrono::steady_clock::now();
    auto destination = bundleDestination(journal_, *key);
    if (!destination) {
      llvm::Error reset = journal_.queue(*key);
      return llvm::joinErrors(destination.takeError(), std::move(reset));
    }
    external_tool::ExternalToolPreparationContext context{
        policy_.externalSite()->localToolConfig, std::move(*destination)};
    auto preparation = evaluation::prepareEvaluationModelInvocation(
        task.request, *task.resolution, store, blobs, context);
    if (!preparation) {
      llvm::Error reset = journal_.queue(*key);
      return llvm::joinErrors(preparation.takeError(), std::move(reset));
    }
    if (auto *unsupported =
            std::get_if<evaluation::EvaluationEvidence>(&*preparation)) {
      evidence = std::move(*unsupported);
      auto preparationActive = activeNanoseconds(preparationBegin);
      if (!preparationActive)
        return preparationActive.takeError();
      active = *preparationActive;
    } else {
      auto prepared = std::get<evaluation::EvaluationModelPreparedInvocation>(
          std::move(*preparation));
      const external_tool::PreparedExternalToolInvocation &external =
          prepared.externalInvocation();
      auto preparationActive = activeNanoseconds(preparationBegin);
      if (!preparationActive)
        return preparationActive.takeError();
      auto preparationEnd = terminalUnixNanoseconds();
      if (!preparationEnd)
        return preparationEnd.takeError();
      if (llvm::Error error = journal_.recordPrepared(
              *key, external, *preparationActive, *preparationEnd))
        return std::move(error);
      if (policy_.externalSite()->disposition ==
          ExternalAttemptDisposition::PrepareOnly)
        return PromotionEvidenceExecutionResult{
            PromotionAcquisitionIncompleteReason::ProviderUnavailable};
      auto bindingDigest =
          external_tool::deriveExternalToolExecutionBindingDigest(external);
      if (!bindingDigest)
        return bindingDigest.takeError();
      auto boundClaim = evidenceLifecycleClaim(*key, &*bindingDigest);
      if (!boundClaim)
        return boundClaim.takeError();
      auto bound = scheduler_.bindCountedResources(
          preparationLease, *boundClaim,
          providerExecutionControl(preparationStopContext));
      if (!bound)
        return bound.takeError();
      if (!*bound)
        return PromotionEvidenceExecutionResult{
            PromotionAcquisitionIncompleteReason::CancelledOrTimeout};
      auto executed = executePreparedInvocationUnderLease(
          *key, external, providerExecutionControl(preparationStopContext));
      if (!executed)
        return executed.takeError();
      if (!*executed)
        return PromotionEvidenceExecutionResult{
            PromotionAcquisitionIncompleteReason::CancelledOrTimeout};
      if ((**executed).exitCode ==
          external_tool::externalToolExecutionStoppedExitCode) {
        if (llvm::Error error = settleStoppedExternalExecution(*key))
          return std::move(error);
        return PromotionEvidenceExecutionResult{
            PromotionAcquisitionIncompleteReason::CancelledOrTimeout};
      }
      auto imported = tryImportPreparedEvidence(
          task.request, *task.resolution, prepared, **executed, store, blobs);
      if (!imported)
        return imported.takeError();
      if (!*imported)
        return PromotionEvidenceExecutionResult{
            PromotionAcquisitionIncompleteReason::ProviderUnavailable};
      evidence = std::move(**imported);
      active = 0;
    }
  }

  auto root = evaluation::publishEvaluationEvidence(*evidence, store);
  if (!root)
    return root.takeError();
  auto terminalTime = terminalUnixNanoseconds();
  if (!terminalTime)
    return terminalTime.takeError();
  auto lifecycleActive = activeNanoseconds(lifecycleBegin);
  if (!lifecycleActive)
    return lifecycleActive.takeError();
  active = *lifecycleActive;
  const std::array<ArtifactRootReference, 1> outputs = {*root};
  if (llvm::Error error =
          journal_.markTerminal(*key, evidenceJournalStatus(*evidence), active,
                                *terminalTime, outputs))
    return std::move(error);
  return PromotionEvidenceExecutionResult{std::move(*evidence)};
}

llvm::Expected<std::vector<PromotionEvidenceExecutionResult>>
RecoverablePlanWorkExecutor::execute(
    llvm::ArrayRef<PromotionEvidenceExecutionTask> tasks,
    const ArtifactStore &store, const BlobStore &blobs) {
  if (tasks.empty())
    return std::vector<PromotionEvidenceExecutionResult>{};
  const std::size_t workerCount = static_cast<std::size_t>(
      std::min<std::uint64_t>(policy_.workerCount(), tasks.size()));
  if (workerCount == 1 || policy_.maximumDispatches()) {
    std::vector<PromotionEvidenceExecutionResult> results;
    results.reserve(tasks.size());
    for (const PromotionEvidenceExecutionTask &task : tasks) {
      auto result = executeEvidence(task, store, blobs);
      if (!result)
        return result.takeError();
      results.push_back(std::move(*result));
    }
    return results;
  }

  using WorkResult = llvm::Expected<PromotionEvidenceExecutionResult>;
  std::vector<std::unique_ptr<WorkResult>> results(tasks.size());
  std::atomic_size_t next{0};
  llvm::DefaultThreadPool pool(llvm::heavyweight_hardware_concurrency(
      static_cast<unsigned>(workerCount)));
  for (std::size_t worker = 0; worker != workerCount; ++worker)
    pool.async([&] {
      std::unique_ptr<fabric::FabricArtifactImportSession> importSession;
      std::unique_ptr<pnr::PnrDerivedContextSession> derivedContextSession;
      if (fabricImportAttachment_)
        importSession = std::make_unique<fabric::FabricArtifactImportSession>(
            fabricImportAttachment_);
      if (derivedContextAttachment_)
        derivedContextSession = std::make_unique<pnr::PnrDerivedContextSession>(
            derivedContextAttachment_);
      while (true) {
        const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= tasks.size())
          break;
        results[index] = std::make_unique<WorkResult>(
            executeEvidence(tasks[index], store, blobs));
      }
    });
  pool.wait();

  std::vector<PromotionEvidenceExecutionResult> ordered;
  ordered.reserve(tasks.size());
  for (std::unique_ptr<WorkResult> &result : results) {
    if (!result)
      return invalid("Evidence worker did not publish a result");
    if (!*result)
      return result->takeError();
    ordered.push_back(std::move(**result));
  }
  return ordered;
}

} // namespace

llvm::Expected<PlanExecutionPolicy> PlanExecutionPolicy::get(
    std::uint64_t workerCount, SiteResourceClaim inProcessClaim,
    std::optional<ExternalExecutionSite> externalSite,
    llvm::ArrayRef<WorkUnitResourceBinding> resourceBindings,
    std::optional<std::uint64_t> maximumDispatches,
    std::optional<std::uint64_t> dispatchNotAfterUnixNanoseconds) {
  if (workerCount == 0)
    return invalid("execution policy requires a positive worker count");
  if (dispatchNotAfterUnixNanoseconds && *dispatchNotAfterUnixNanoseconds == 0)
    return invalid("dispatch deadline must be positive when present");
  if (externalSite && static_cast<std::uint32_t>(externalSite->disposition) >
                          static_cast<std::uint32_t>(
                              ExternalAttemptDisposition::ExecutePrepared))
    return invalid("execution policy has an unknown external disposition");
  std::vector<WorkUnitResourceBinding> bindings(resourceBindings.begin(),
                                                resourceBindings.end());
  if (!llvm::is_sorted(bindings, bindingLess))
    return invalid("resource bindings are not in canonical WorkUnitKey order");
  for (std::size_t index = 1; index < bindings.size(); ++index)
    if (bindings[index - 1].key == bindings[index].key)
      return invalid("resource bindings contain a duplicate WorkUnitKey");
  return PlanExecutionPolicy(
      workerCount, std::move(inProcessClaim), std::move(externalSite),
      std::move(bindings), maximumDispatches, dispatchNotAfterUnixNanoseconds);
}

llvm::Error stopDseExecution(ExecutionJournal &journal,
                             GracefulStopPolicy policy) {
  if (policy != GracefulStopPolicy::FinishAtomicOwnerBoundary)
    return invalid("unknown graceful stop policy");
  return journal.requestGracefulStop();
}

llvm::Expected<DsePlanExecutionResult>
executeDsePlan(const ResolvedDseConfigView &view, const DseRunClosure &closure,
               ExecutionJournal &journal, SiteScheduler &scheduler,
               const PlanExecutionPolicy &policy, const ArtifactStore &store,
               const BlobStore &blobs) {
  if (journal.runKey() != closure.runKey())
    return invalid("ExecutionJournal belongs to another run closure");
  if (journal.resolvedDseConfigViewDigest() != view.digest())
    return invalid("ExecutionJournal belongs to another resolved DSE plan");
  RecoverablePlanWorkExecutor executor(journal, scheduler, policy);
  return detail::executeDsePlanWithWorkExecutor(view, store, blobs, &executor,
                                                {});
}

llvm::Expected<DsePlanExecutionResult>
resumeDsePlan(const ResolvedDseConfigView &view, const DseRunClosure &closure,
              ExecutionJournal &journal, SiteScheduler &scheduler,
              const PlanExecutionPolicy &policy, const ArtifactStore &store,
              const BlobStore &blobs,
              InvocationManifestRetention manifestRetention) {
  if (manifestRetention != InvocationManifestRetention::Release &&
      manifestRetention != InvocationManifestRetention::Retain)
    return invalid("unknown invocation manifest retention policy");
  if (journal.runKey() != closure.runKey())
    return invalid("ExecutionJournal belongs to another run closure");
  if (journal.resolvedDseConfigViewDigest() != view.digest())
    return invalid("ExecutionJournal belongs to another resolved DSE plan");
  if (llvm::Error error = journal.beginResume())
    return std::move(error);
  auto execution =
      executeDsePlan(view, closure, journal, scheduler, policy, store, blobs);
  if (!execution)
    return llvm::joinErrors(execution.takeError(),
                            journal.releaseInvocationOccurrence());
  if (manifestRetention == InvocationManifestRetention::Release) {
    llvm::Error releaseError = journal.releaseInvocationOccurrence();
    if (releaseError)
      return std::move(releaseError);
  }
  return execution;
}

} // namespace loom::dse
