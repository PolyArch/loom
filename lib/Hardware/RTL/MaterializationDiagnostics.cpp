#include "Hardware/RTL/MaterializationDiagnostics.h"

#include "Common/InvocationDiagnosticLog.h"

#include "circt/Dialect/HW/HWOps.h"
#include "llvm/Support/JSON.h"

#include <limits>

namespace loom::hardware::rtl {
namespace {

struct RtlIrInventory final {
  std::uint64_t operationCount = 0;
  std::uint64_t moduleCount = 0;
  std::uint64_t generatedModuleCount = 0;
};

void increment(std::uint64_t &value) {
  if (value != std::numeric_limits<std::uint64_t>::max())
    ++value;
}

RtlIrInventory inventory(mlir::ModuleOp module) {
  RtlIrInventory result;
  module.walk([&](mlir::Operation *operation) {
    increment(result.operationCount);
    if (llvm::isa<circt::hw::HWModuleOp, circt::hw::HWModuleExternOp>(
            operation))
      increment(result.moduleCount);
    else if (llvm::isa<circt::hw::HWModuleGeneratedOp>(operation))
      increment(result.generatedModuleCount);
  });
  return result;
}

} // namespace

RtlMaterializationStageTracker::RtlMaterializationStageTracker(
    llvm::StringRef operation, llvm::StringRef materializationKey,
    std::optional<mlir::ModuleOp> module)
    : operation_(operation.str()),
      materializationKey_(materializationKey.str()) {
  if (!invocationDiagnosticEnabled(DiagnosticVerbosity::Summary))
    return;
  resources_.emplace();
  emit("begin", module, std::nullopt);
}

RtlMaterializationStageTracker::~RtlMaterializationStageTracker() {
  if (resources_ && !finished_)
    emit("incomplete", std::nullopt, std::nullopt);
}

void RtlMaterializationStageTracker::finish(
    std::optional<mlir::ModuleOp> module,
    std::optional<std::uint64_t> emittedBytes) {
  if (finished_)
    return;
  emit("end", module, emittedBytes);
  finished_ = true;
}

void RtlMaterializationStageTracker::emit(
    llvm::StringRef boundary, std::optional<mlir::ModuleOp> module,
    std::optional<std::uint64_t> emittedBytes) {
  if (!resources_)
    return;
  const ExecutionResourceStatistics observation = resources_->observe();
  const std::optional<RtlIrInventory> ir =
      module ? std::optional<RtlIrInventory>(inventory(*module)) : std::nullopt;
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary,
      InvocationDiagnosticStage::HardwareConfiguration,
      InvocationDiagnosticEvent::Statistics, [&] {
        llvm::json::Object fields{
            {"statistics_kind", "rtl_materialization_stage"},
            {"operation", operation_},
            {"materialization_key", materializationKey_},
            {"boundary", boundary},
            {"active_wall_time_ns", observation.activeWallTimeNanoseconds},
            {"resource_observation_scope", "process"},
            {"peak_resident_scope", "process_lifetime"},
            {"allocated_memory_bytes", observation.allocatedMemoryBytes}};
        if (observation.processCpuTimeDeltaNanoseconds)
          fields["process_cpu_time_delta_ns"] =
              *observation.processCpuTimeDeltaNanoseconds;
        if (observation.currentResidentMemoryBytes)
          fields["current_resident_memory_bytes"] =
              *observation.currentResidentMemoryBytes;
        if (observation.peakResidentMemoryBytes)
          fields["peak_resident_memory_bytes"] =
              *observation.peakResidentMemoryBytes;
        if (ir) {
          fields["mlir_operation_count"] = ir->operationCount;
          fields["mlir_module_count"] = ir->moduleCount;
          fields["mlir_generated_module_count"] = ir->generatedModuleCount;
        }
        if (emittedBytes)
          fields["emitted_bytes"] = *emittedBytes;
        return llvm::json::Value(std::move(fields));
      });
}

} // namespace loom::hardware::rtl
