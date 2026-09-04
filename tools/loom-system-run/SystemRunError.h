#ifndef LOOM_TOOLS_LOOM_SYSTEM_RUN_SYSTEMRUNERROR_H
#define LOOM_TOOLS_LOOM_SYSTEM_RUN_SYSTEMRUNERROR_H

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <system_error>

namespace loom::system_run {

inline llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "system_run_invalid: " + message);
}

} // namespace loom::system_run

#endif // LOOM_TOOLS_LOOM_SYSTEM_RUN_SYSTEMRUNERROR_H
