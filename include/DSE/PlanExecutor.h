#ifndef LOOM_DSE_PLANEXECUTOR_H
#define LOOM_DSE_PLANEXECUTOR_H

#include "DSE/ExecutionJournal.h"
#include "DSE/Plan.h"
#include "DSE/SiteScheduler.h"
#include "ExternalTool/LocalConfig.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {

enum class ExternalAttemptDisposition : std::uint32_t {
  PrepareOnly = 0,
  ExecutePrepared = 1,
};

struct ExternalExecutionSite final {
  external_tool::LocalToolConfig localToolConfig;
  ExternalAttemptDisposition disposition =
      ExternalAttemptDisposition::ExecutePrepared;
  std::uint64_t cpuCores = 1;
  std::uint64_t memoryBytes = 0;
  std::uint64_t scratchBytes = 0;
  bool claimLicense = false;
};

struct WorkUnitResourceBinding final {
  WorkUnitKey key;
  SiteResourceClaim claim;
};

/// Operational inputs for one execution occurrence. None of these values
/// enter DseRunKey, candidate identity, Evidence, or selection.
class PlanExecutionPolicy final {
public:
  static llvm::Expected<PlanExecutionPolicy>
  get(std::uint64_t workerCount, SiteResourceClaim inProcessClaim,
      std::optional<ExternalExecutionSite> externalSite = std::nullopt,
      llvm::ArrayRef<WorkUnitResourceBinding> resourceBindings = {},
      std::optional<std::uint64_t> maximumDispatches = std::nullopt,
      std::optional<std::uint64_t> dispatchNotAfterUnixNanoseconds =
          std::nullopt);

  std::uint64_t workerCount() const { return workerCount_; }
  const SiteResourceClaim &inProcessClaim() const { return inProcessClaim_; }
  const std::optional<ExternalExecutionSite> &externalSite() const {
    return externalSite_;
  }
  llvm::ArrayRef<WorkUnitResourceBinding> resourceBindings() const {
    return resourceBindings_;
  }
  std::optional<std::uint64_t> maximumDispatches() const {
    return maximumDispatches_;
  }
  std::optional<std::uint64_t> dispatchNotAfterUnixNanoseconds() const {
    return dispatchNotAfterUnixNanoseconds_;
  }

private:
  PlanExecutionPolicy(
      std::uint64_t workerCount, SiteResourceClaim inProcessClaim,
      std::optional<ExternalExecutionSite> externalSite,
      std::vector<WorkUnitResourceBinding> resourceBindings,
      std::optional<std::uint64_t> maximumDispatches,
      std::optional<std::uint64_t> dispatchNotAfterUnixNanoseconds)
      : workerCount_(workerCount), inProcessClaim_(std::move(inProcessClaim)),
        externalSite_(std::move(externalSite)),
        resourceBindings_(std::move(resourceBindings)),
        maximumDispatches_(maximumDispatches),
        dispatchNotAfterUnixNanoseconds_(dispatchNotAfterUnixNanoseconds) {}

  std::uint64_t workerCount_ = 0;
  SiteResourceClaim inProcessClaim_;
  std::optional<ExternalExecutionSite> externalSite_;
  std::vector<WorkUnitResourceBinding> resourceBindings_;
  std::optional<std::uint64_t> maximumDispatches_;
  std::optional<std::uint64_t> dispatchNotAfterUnixNanoseconds_;
};

using DsePlanExecutionResult = DsePlanExecutionOutcome;

enum class GracefulStopPolicy : std::uint32_t {
  FinishAtomicOwnerBoundary = 0,
};

llvm::Error stopDseExecution(ExecutionJournal &journal,
                             GracefulStopPolicy policy);

llvm::Expected<DsePlanExecutionResult>
executeDsePlan(const ResolvedDseConfigView &view, const DseRunClosure &closure,
               ExecutionJournal &journal, SiteScheduler &scheduler,
               const PlanExecutionPolicy &policy, const ArtifactStore &store,
               const BlobStore &blobs);

enum class InvocationManifestRetention : std::uint8_t {
  Release,
  Retain,
};

llvm::Expected<DsePlanExecutionResult>
resumeDsePlan(const ResolvedDseConfigView &view, const DseRunClosure &closure,
              ExecutionJournal &journal, SiteScheduler &scheduler,
              const PlanExecutionPolicy &policy, const ArtifactStore &store,
              const BlobStore &blobs,
              InvocationManifestRetention manifestRetention);

namespace detail {

struct DseGenerateExecutionTask final {
  std::uint64_t planNodeOrdinal = 0;
  std::vector<CandidateGeneratorInputBinding> inputs;
  std::vector<CandidateGeneratorOutputDemand> outputDemands;
  ResolvedCandidateGeneratorBinding binding;
};

class DsePlanWorkExecutor : public PromotionEvidenceExecutor {
public:
  ~DsePlanWorkExecutor() override = default;

  virtual bool shouldStopBeforeDispatch() const = 0;
  virtual llvm::Expected<CandidateGeneratorProviderResult>
  executeGenerate(std::uint64_t planNodeOrdinal,
                  llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
                  llvm::ArrayRef<CandidateGeneratorOutputDemand> outputDemands,
                  const ResolvedCandidateGeneratorBinding &binding,
                  const ArtifactStore &store, const BlobStore &blobs) = 0;
  virtual llvm::Expected<std::vector<CandidateGeneratorProviderResult>>
  executeGenerateBatch(llvm::ArrayRef<DseGenerateExecutionTask> tasks,
                       const ArtifactStore &store, const BlobStore &blobs) = 0;
  virtual llvm::Error
  beginPromotion(std::uint64_t planNodeOrdinal,
                 llvm::ArrayRef<ArtifactRootReference> candidates,
                 llvm::ArrayRef<EvidenceObligationTemplateRef> obligations) = 0;
};

llvm::Expected<DsePlanExecutionOutcome> executeDsePlanWithWorkExecutor(
    const ResolvedDseConfigView &view, const ArtifactStore &store,
    const BlobStore &blobs, DsePlanWorkExecutor *executor,
    ExecutionControlView executionControl = {});

} // namespace detail
} // namespace loom::dse

#endif // LOOM_DSE_PLANEXECUTOR_H
