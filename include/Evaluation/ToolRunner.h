#ifndef LOOM_EVALUATION_TOOL_RUNNER_H
#define LOOM_EVALUATION_TOOL_RUNNER_H

#include "Common/Artifact.h"

#include "llvm/Support/Error.h"

#include <chrono>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace loom::evaluation {

struct EnvironmentVariable {
  std::string name;
  std::string value;
};

struct MaterializedInputArtifact {
  ArtifactIdentity artifact;
  std::string path;
};

struct ToolInvocation {
  std::string toolBindingIdentity;
  std::string executablePath;
  std::vector<std::string> argv;
  std::vector<EnvironmentVariable> environmentOverlay;
  std::string scratchDirectory;
  std::vector<MaterializedInputArtifact> inputs;
  std::vector<std::string> declaredOutputs;
  std::optional<std::chrono::milliseconds> timeout;
  std::function<bool()> cancellationRequested;
  std::vector<std::string> resourceLeaseBindingIdentities;
  std::vector<std::string> licenseLeaseBindingIdentities;
};

enum class ToolRunStatus {
  LaunchFailure,
  Exited,
  Signaled,
  TimedOut,
  Cancelled,
  InfrastructureFailure,
};

struct ToolRunOutcome {
  ToolRunStatus status;
  std::optional<int> exitCode;
  std::optional<int> terminationSignal;
  std::optional<int> launchErrorNumber;
  std::string launchErrorMessage;
  std::string standardOutput;
  std::string standardError;
  std::vector<std::string> producedFiles;
  std::optional<std::string> inventoryDiagnostic;
  std::optional<std::string> infrastructureDiagnostic;
  std::chrono::system_clock::time_point startedAt;
  std::chrono::system_clock::time_point endedAt;
  std::string toolBindingIdentity;
  std::vector<std::string> resourceLeaseBindingIdentities;
  std::vector<std::string> licenseLeaseBindingIdentities;
};

llvm::Expected<ToolRunOutcome> runTool(const ToolInvocation &invocation);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_TOOL_RUNNER_H
