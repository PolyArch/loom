#ifndef LOOM_LIB_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATIONDIAGNOSTICS_H
#define LOOM_LIB_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATIONDIAGNOSTICS_H

#include "Common/ExecutionControl.h"

#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace loom::hardware::detail {

class HardwareImplementationStageTracker final {
public:
  explicit HardwareImplementationStageTracker(llvm::StringRef operation);
  ~HardwareImplementationStageTracker();

  HardwareImplementationStageTracker(
      const HardwareImplementationStageTracker &) = delete;
  HardwareImplementationStageTracker &
  operator=(const HardwareImplementationStageTracker &) = delete;

  void finish();

private:
  void emit(llvm::StringRef boundary) const;

  std::string operation_;
  std::optional<ExecutionResourceTracker> resources_;
  bool finished_ = false;
};

} // namespace loom::hardware::detail

#endif // LOOM_LIB_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATIONDIAGNOSTICS_H
