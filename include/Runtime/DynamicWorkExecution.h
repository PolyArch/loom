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

enum class DynamicWorkExecutionIncompleteReason : std::uint8_t {
  QueueCapacity,
};

class DynamicWorkExecutionIncomplete final
    : public llvm::ErrorInfo<DynamicWorkExecutionIncomplete> {
public:
  static char ID;

  DynamicWorkExecutionIncomplete(DynamicWorkExecutionIncompleteReason reason,
                                 sim::WorkItemId item)
      : reason_(reason), item_(std::move(item)) {}

  DynamicWorkExecutionIncompleteReason reason() const { return reason_; }
  const sim::WorkItemId &item() const { return item_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  DynamicWorkExecutionIncompleteReason reason_;
  sim::WorkItemId item_;
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

struct DynamicWorkItemExecution final {
  DynamicWorkExecutionAction action = DynamicWorkExecutionAction::Complete;
  std::vector<std::vector<std::uint8_t>> childPayloads;
};

using DynamicWorkSelectedBodyExecutor =
    std::function<llvm::Expected<DynamicWorkItemExecution>(
        const DynamicWorkExecutionAssignment &)>;

struct DynamicWorkExecutionResult final {
  sim::ThreadDispatchOccurrenceId dispatchOccurrence{0};
  sim::RetirementEffect joinEffect = sim::RetirementEffect::DomainStillActive;
  bool cancelled = false;
  std::uint64_t processedItemCount = 0;
  std::uint64_t publishedChildCount = 0;
  std::uint64_t completedItemCount = 0;
  std::uint64_t cancelledItemCount = 0;
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

/// Coordinates the bounded DynamicWork profile. SystemMapping selects
/// persistent Instruction, optional Spatial, and service-plan targets from the
/// Dataflow-owned domain execution class. WorkItemId, scheduler worker
/// assignment, and deque placement remain transient.
class DynamicWorkExecutionSession final {
public:
  DynamicWorkExecutionSession() = default;

  DynamicWorkExecutionSession(const DynamicWorkExecutionSession &) = delete;
  DynamicWorkExecutionSession &
  operator=(const DynamicWorkExecutionSession &) = delete;

  /// Invokes one external synchronous execution owner for every item in the
  /// responsibility domain. Child publication is atomic per returned item
  /// result. Complete is the owner's report, not independent body execution
  /// evidence; the result proves the selected Mapping and responsibility join.
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
