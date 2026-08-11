#ifndef LOOM_EVALUATION_MODELS_MAPPEDRTLSIMULATIONCONFIG_H
#define LOOM_EVALUATION_MODELS_MAPPEDRTLSIMULATIONCONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <system_error>

namespace loom::evaluation::models {

/// The semantic machine-independent binding selected by model kind 21.
/// Executable paths, modules, containers, scratch paths, and limits remain in
/// the machine-local invocation context.
struct MappedRtlSimulatorBinding final {
  std::string stableHdlSimulatorBuildIdentity;

  friend bool operator==(const MappedRtlSimulatorBinding &lhs,
                         const MappedRtlSimulatorBinding &rhs) {
    return lhs.stableHdlSimulatorBuildIdentity ==
           rhs.stableHdlSimulatorBuildIdentity;
  }
  friend bool operator!=(const MappedRtlSimulatorBinding &lhs,
                         const MappedRtlSimulatorBinding &rhs) {
    return !(lhs == rhs);
  }
};

inline llvm::Error
validateMappedRtlSimulatorBinding(const MappedRtlSimulatorBinding &binding) {
  const llvm::StringRef identity(binding.stableHdlSimulatorBuildIdentity);
  if (identity.empty() || identity.trim() != identity)
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "mapped_rtl_simulation_invalid: HDL simulator build identity is not "
        "one normalized line");
  for (unsigned char character : identity.bytes())
    if (character < 0x20 || character > 0x7e)
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "mapped_rtl_simulation_invalid: HDL simulator build identity is "
          "not one normalized line");
  return llvm::Error::success();
}

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_MAPPEDRTLSIMULATIONCONFIG_H
