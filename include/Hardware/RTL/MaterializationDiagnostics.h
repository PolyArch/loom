#ifndef LOOM_HARDWARE_RTL_MATERIALIZATIONDIAGNOSTICS_H
#define LOOM_HARDWARE_RTL_MATERIALIZATIONDIAGNOSTICS_H

#include "Common/ExecutionControl.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace loom::hardware::rtl {

/// Invocation-local resource accounting for one RTL materialization stage.
/// Diagnostics explain execution only and never affect Artifact identity.
class RtlMaterializationStageTracker final {
public:
  explicit RtlMaterializationStageTracker(
      llvm::StringRef operation, llvm::StringRef materializationKey = {},
      std::optional<mlir::ModuleOp> module = std::nullopt);
  ~RtlMaterializationStageTracker();

  RtlMaterializationStageTracker(const RtlMaterializationStageTracker &) =
      delete;
  RtlMaterializationStageTracker &
  operator=(const RtlMaterializationStageTracker &) = delete;

  void finish(std::optional<mlir::ModuleOp> module = std::nullopt,
              std::optional<std::uint64_t> emittedBytes = std::nullopt);

private:
  void emit(llvm::StringRef boundary, std::optional<mlir::ModuleOp> module,
            std::optional<std::uint64_t> emittedBytes);

  std::string operation_;
  std::string materializationKey_;
  std::optional<ExecutionResourceTracker> resources_;
  bool finished_ = false;
};

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_MATERIALIZATIONDIAGNOSTICS_H
