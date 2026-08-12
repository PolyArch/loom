#ifndef LOOM_COMMON_DIAGNOSTICVERBOSITY_H
#define LOOM_COMMON_DIAGNOSTICVERBOSITY_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace loom {

enum class DiagnosticVerbosity : std::uint8_t {
  Disabled = 0,
  Summary = 1,
  Decision = 2,
  Detail = 3,
};

inline constexpr llvm::StringLiteral diagnosticVerbosityEnvironment =
    "LOOM_VERBOSE_LEVEL";
inline constexpr llvm::StringLiteral diagnosticVerbosityArgumentPrefix =
    "+LOOM_VERBOSE_LEVEL=";

/// Returns the process-wide invocation verbosity parsed once from the
/// Common-owned environment binding.
DiagnosticVerbosity diagnosticVerbosity();

bool diagnosticVerbosityEnabled(DiagnosticVerbosity minimum);

/// Returns the presentation-only plusarg projected into supported external
/// execution commands. Level zero has no argument.
std::optional<std::string> diagnosticVerbosityArgument();

/// Recognizes exactly the closed presentation argument emitted above.
bool isDiagnosticVerbosityArgument(llvm::StringRef argument);

/// Recognizes the reserved external-command spelling, including malformed
/// values that callers may not author directly.
bool isDiagnosticVerbosityBinding(llvm::StringRef argument);

} // namespace loom

#endif // LOOM_COMMON_DIAGNOSTICVERBOSITY_H
