#include "ExecutionMatrixLifecycle.h"

#include "llvm/Support/Error.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "execution matrix lifecycle test failure: " << message << '\n';
  std::exit(1);
}

void requireDiagnosticInvocation(
    llvm::StringRef spelling,
    loom::system_test::ExecutionMatrixCell expectedCell) {
  auto invocation = loom::system_test::parseExecutionMatrixInvocation(spelling);
  if (!invocation)
    fail(llvm::toString(invocation.takeError()));
  if (invocation->cell != expectedCell ||
      invocation->attempt !=
          loom::system_test::ExecutionMatrixAttemptKind::Diagnostic ||
      loom::system_test::executionMatrixInvocationName(*invocation) != spelling)
    fail("standalone diagnostic invocation changed its typed identity");
}

} // namespace

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
  requireDiagnosticInvocation("diagnostic-system-cgra",
                              ExecutionMatrixCell::SystemCgra);
  requireDiagnosticInvocation("diagnostic-system-rtl",
                              ExecutionMatrixCell::SystemRtl);
  ExecutionMatrixLifecycleRecorder pairLifecycle;
  {
    ExecutionMatrixLifecycleTimer timer(
        pairLifecycle, ExecutionMatrixLifecycleOperation::Gem5Readiness);
  }
  if (pairLifecycle.operationCount(
          ExecutionMatrixLifecycleOperation::Gem5Readiness) != 1)
    fail("attempt-pair lifecycle did not retain its measured operation");
  pairLifecycle.emitAttemptPair(ExecutionMatrixCell::SystemCgra);
  return 0;
}
