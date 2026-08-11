#ifndef LOOM_EVALUATION_MODELS_OPENROADSTATICFPACONFIG_H
#define LOOM_EVALUATION_MODELS_OPENROADSTATICFPACONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <system_error>

namespace loom::evaluation::models {

struct OpenRoadStaticFpaProviderBinding final {
  std::string stableProviderBuildIdentity;

  friend bool operator==(const OpenRoadStaticFpaProviderBinding &lhs,
                         const OpenRoadStaticFpaProviderBinding &rhs) {
    return lhs.stableProviderBuildIdentity == rhs.stableProviderBuildIdentity;
  }
};

inline llvm::Error validateOpenRoadStaticFpaProviderBinding(
    const OpenRoadStaticFpaProviderBinding &binding) {
  const llvm::StringRef identity(binding.stableProviderBuildIdentity);
  if (identity.empty() || identity.trim() != identity)
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "openroad_static_fpa_invalid: provider build identity is not one "
        "normalized line");
  for (unsigned char character : identity.bytes())
    if (character < 0x20 || character > 0x7e)
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "openroad_static_fpa_invalid: provider build identity is not one "
          "normalized line");
  return llvm::Error::success();
}

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_OPENROADSTATICFPACONFIG_H
