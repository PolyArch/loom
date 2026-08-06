#ifndef LOOM_LIB_EXTERNALTOOL_SHELLRENDERINGINTERNAL_H
#define LOOM_LIB_EXTERNALTOOL_SHELLRENDERINGINTERNAL_H

#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

namespace loom::external_tool::detail {

/// Single-quotes one argument for generated Bash probe and bundle scripts.
/// This is the only shell quoting implementation inside the ExternalTool
/// library.
inline std::string shellQuote(llvm::StringRef value) {
  std::string result = "'";
  for (char character : value) {
    if (character == '\'')
      result += "'\\''";
    else
      result += character;
  }
  result += "'";
  return result;
}

/// Renders the PolyArch/container run protocol for one command inside the
/// resolved runtime. workdirShellExpression must be an already-quoted path
/// or a simple variable expansion owned by the caller's script.
inline std::string renderPolyArchContainerInvocation(
    llvm::StringRef containerExecutable, llvm::StringRef os,
    llvm::StringRef workdirShellExpression,
    const std::vector<std::string> &arguments) {
  std::string rendered = shellQuote(containerExecutable);
  rendered += " 'run' '--os' " + shellQuote(os);
  rendered += " '--workdir' " + workdirShellExpression.str() +
              " '--env' 'INHERIT' '--'";
  for (const std::string &argument : arguments)
    rendered += " " + shellQuote(argument);
  return rendered;
}

} // namespace loom::external_tool::detail

#endif // LOOM_LIB_EXTERNALTOOL_SHELLRENDERINGINTERNAL_H
