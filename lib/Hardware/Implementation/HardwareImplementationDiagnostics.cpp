#include "HardwareImplementationDiagnostics.h"

#include "Common/InvocationDiagnosticLog.h"

#include "llvm/Support/JSON.h"

#include <utility>

namespace loom::hardware::detail {

HardwareImplementationStageTracker::HardwareImplementationStageTracker(
    llvm::StringRef operation)
    : operation_(operation.str()) {
  if (!invocationDiagnosticEnabled(DiagnosticVerbosity::Summary))
    return;
  resources_.emplace();
  emit("begin");
}

HardwareImplementationStageTracker::~HardwareImplementationStageTracker() {
  if (resources_ && !finished_)
    emit("incomplete");
}

void HardwareImplementationStageTracker::finish() {
  if (finished_)
    return;
  emit("end");
  finished_ = true;
}

void HardwareImplementationStageTracker::emit(
    llvm::StringRef boundary) const {
  if (!resources_)
    return;
  const ExecutionResourceStatistics observation = resources_->observe();
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary,
      InvocationDiagnosticStage::HardwareConfiguration,
      InvocationDiagnosticEvent::Statistics, [&] {
        llvm::json::Object fields{
            {"statistics_kind", "hardware_implementation_finalization_stage"},
            {"operation", operation_},
            {"boundary", boundary.str()},
            {"active_wall_time_ns", observation.activeWallTimeNanoseconds},
            {"allocated_memory_bytes", observation.allocatedMemoryBytes},
            {"resource_observation_scope", "process"},
            {"peak_resident_scope", "process_lifetime"}};
        if (observation.currentResidentMemoryBytes)
          fields["current_resident_memory_bytes"] =
              *observation.currentResidentMemoryBytes;
        if (observation.peakResidentMemoryBytes)
          fields["peak_resident_memory_bytes"] =
              *observation.peakResidentMemoryBytes;
        if (observation.processCpuTimeDeltaNanoseconds)
          fields["process_cpu_time_delta_ns"] =
              *observation.processCpuTimeDeltaNanoseconds;
        return llvm::json::Value(std::move(fields));
      });
}

} // namespace loom::hardware::detail
