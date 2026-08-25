#ifndef LOOM_RUNTIME_DYNAMICWORKEXECUTION_H
#define LOOM_RUNTIME_DYNAMICWORKEXECUTION_H

#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Simulator/CGRASimulator.h"
#include "Simulator/DynamicWorkScheduler.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::runtime {

enum class DynamicWorkExecutionUnsupportedReason : std::uint8_t {
  StableItemDomainUnavailable,
  SelectedGraphUnavailable,
  ThreadBodyUnavailable,
  GraphBoundaryUnavailable,
  ScalarPayloadUnavailable,
};

class DynamicWorkExecutionUnsupported final
    : public llvm::ErrorInfo<DynamicWorkExecutionUnsupported> {
public:
  static char ID;

  DynamicWorkExecutionUnsupported(DynamicWorkExecutionUnsupportedReason reason,
                                  std::string message)
      : reason_(reason), message_(std::move(message)) {}

  DynamicWorkExecutionUnsupportedReason reason() const { return reason_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  DynamicWorkExecutionUnsupportedReason reason_;
  std::string message_;
};

class DynamicWorkCgraExecutionIncomplete final
    : public llvm::ErrorInfo<DynamicWorkCgraExecutionIncomplete> {
public:
  static char ID;

  explicit DynamicWorkCgraExecutionIncomplete(
      sim::CgraSimulationOutcome outcome)
      : outcome_(std::move(outcome)) {}

  const sim::CgraSimulationOutcome &outcome() const { return outcome_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  sim::CgraSimulationOutcome outcome_;
};

struct DynamicWorkExecutionRequest final {
  std::uint32_t workerCount = 0;
  std::size_t queueCapacityPerWorker = 0;
  std::vector<std::uint8_t> rootPayload;
};

struct DynamicWorkSelectedServicePlan final {
  mapping::SystemServiceObligationKey obligation;
  mapping::ServicePlanSelectionKey selection;
  std::uint64_t planOrdinal = 0;
};

struct DynamicWorkExecutionAssignment final {
  sim::WorkItemId item;
  std::uint32_t workerOrdinal = 0;
  llvm::ArrayRef<std::uint8_t> payload;
  mapping::InstructionExecutionContextKey instructionContext;
  std::optional<mapping::SelectedSystemSpatialContext> spatialContext;
  std::vector<DynamicWorkSelectedServicePlan> servicePlans;
};

enum class DynamicWorkExecutionAction : std::uint8_t {
  Complete,
  RequestCancellation,
};

using DynamicWorkSelectedBodyExecutor =
    std::function<llvm::Expected<DynamicWorkExecutionAction>(
        const DynamicWorkExecutionAssignment &)>;

struct DynamicWorkExecutionResult final {
  sim::ThreadDispatchOccurrenceId dispatchOccurrence{0};
  sim::RetirementEffect joinEffect = sim::RetirementEffect::DomainStillActive;
  bool cancelled = false;
  std::vector<sim::DynamicWorkScheduleAction> replay;
};

struct DynamicWorkCgraExecutionRequest final {
  DynamicWorkExecutionRequest dispatch;
  std::uint64_t maxEventFrames = 0;
};

struct DynamicWorkCgraExecutionResult final {
  DynamicWorkExecutionResult dispatch;
  mapping::InstructionExecutionContextKey instructionContext;
  mapping::SelectedSystemSpatialContext spatialContext;
  std::vector<DynamicWorkSelectedServicePlan> servicePlans;
  sim::RetiredCgraSimulation execution;
};

/// Coordinates the bounded root-only DynamicWork profile. SystemMapping
/// selects persistent Instruction, optional Spatial, and service-plan targets
/// from the Dataflow-owned stable root key. Scheduler worker assignment
/// remains transient.
class DynamicWorkExecutionSession final {
public:
  DynamicWorkExecutionSession() = default;

  DynamicWorkExecutionSession(const DynamicWorkExecutionSession &) = delete;
  DynamicWorkExecutionSession &
  operator=(const DynamicWorkExecutionSession &) = delete;

  /// Invokes one external synchronous execution owner with the verified
  /// selection. Complete is that owner's report, not independent body
  /// execution evidence. The result proves only responsibility retirement.
  llvm::Expected<DynamicWorkExecutionResult>
  executeRoot(const dataflow::CanonicalDataflowProgramView &dataflow,
              const mapping::FinalizedSystemMapping &systemMapping,
              dataflow::RootThreadLaunchRef root,
              DynamicWorkExecutionRequest request,
              DynamicWorkSelectedBodyExecutor executor);

  /// Executes the closed direct-CGRA profile without a caller-supplied body
  /// executor. The work-item payload must be one byte-addressable scalar
  /// integer forwarded unchanged to the sole direct graph value input.
  llvm::Expected<DynamicWorkCgraExecutionResult>
  executeRootCgra(const dataflow::CanonicalDataflowArtifact &dataflowArtifact,
                  const mapping::FinalizedSystemMapping &systemMapping,
                  dataflow::RootThreadLaunchRef root,
                  DynamicWorkCgraExecutionRequest request,
                  const ::loom::ArtifactStore &artifacts);

private:
  llvm::Expected<sim::ThreadDispatchOccurrenceId> allocateDispatchOccurrence();

  std::mutex mutex_;
  std::uint64_t nextDispatchOccurrence_ = 1;
};

} // namespace loom::runtime

#endif // LOOM_RUNTIME_DYNAMICWORKEXECUTION_H
