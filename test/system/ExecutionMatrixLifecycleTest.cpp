#include "ExecutionMatrixLifecycle.h"

#include <vector>

int main() {
  using loom::external_tool::ExternalToolCommandExecutionObservation;
  using namespace loom::system_test;

  std::vector<ExternalToolCommandExecutionObservation> commands;
  for (std::uint64_t ordinal = 0; ordinal != 5; ++ordinal)
    commands.push_back({ordinal, (ordinal + 1) * 1000, 0});
  emitExecutionMatrixExternalCommands(
      {ExecutionMatrixCell::SystemRtl, ExecutionMatrixAttemptKind::Ordinary},
      commands);
  emitExecutionMatrixExternalCommands(
      {ExecutionMatrixCell::SystemRtl, ExecutionMatrixAttemptKind::Diagnostic},
      commands);
  return 0;
}
