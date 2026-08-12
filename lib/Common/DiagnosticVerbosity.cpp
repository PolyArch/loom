#include "Common/DiagnosticVerbosity.h"

#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstdlib>

namespace loom {
namespace {

DiagnosticVerbosity parseDiagnosticVerbosity() {
  const char *binding = std::getenv(diagnosticVerbosityEnvironment.data());
  if (!binding || !*binding)
    return DiagnosticVerbosity::Disabled;

  llvm::StringRef text(binding);
  if (!std::all_of(text.begin(), text.end(), [](char character) {
        return character >= '0' && character <= '9';
      }))
    return DiagnosticVerbosity::Disabled;
  text = text.drop_while([](char character) { return character == '0'; });
  if (text.empty())
    return DiagnosticVerbosity::Disabled;
  if (text.size() != 1 || text.front() >= '3')
    return DiagnosticVerbosity::Detail;
  return text.front() == '1' ? DiagnosticVerbosity::Summary
                             : DiagnosticVerbosity::Decision;
}

} // namespace

DiagnosticVerbosity diagnosticVerbosity() {
  static const DiagnosticVerbosity parsed = parseDiagnosticVerbosity();
  return parsed;
}

bool diagnosticVerbosityEnabled(DiagnosticVerbosity minimum) {
  return static_cast<std::uint8_t>(diagnosticVerbosity()) >=
         static_cast<std::uint8_t>(minimum);
}

std::optional<std::string> diagnosticVerbosityArgument() {
  const DiagnosticVerbosity level = diagnosticVerbosity();
  if (level == DiagnosticVerbosity::Disabled)
    return std::nullopt;
  return "+" + diagnosticVerbosityEnvironment.str() + "=" +
         std::to_string(static_cast<std::uint8_t>(level));
}

bool isDiagnosticVerbosityArgument(llvm::StringRef argument) {
  if (!isDiagnosticVerbosityBinding(argument))
    return false;
  argument = argument.drop_front(diagnosticVerbosityArgumentPrefix.size());
  return argument == "1" || argument == "2" || argument == "3";
}

bool isDiagnosticVerbosityBinding(llvm::StringRef argument) {
  return argument.starts_with(diagnosticVerbosityArgumentPrefix);
}

} // namespace loom
