#ifndef LOOM_EXTERNALTOOL_INVOCATIONBUNDLE_H
#define LOOM_EXTERNALTOOL_INVOCATIONBUNDLE_H

#include "ExternalTool/Binding.h"
#include "ExternalTool/ExternalFile.h"
#include "ExternalTool/RuntimeBinding.h"
#include "ExternalTool/ShellProbe.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <vector>

namespace loom::external_tool {

struct MaterializedBundleFile {
  std::string relativePath;
  std::string contents;
  std::optional<std::string> sourceArtifactIdentity;
  bool executable = false;
};

struct ExternalToolInvocationBundleSpec {
  std::string providerIdentity;
  std::string semanticBindingIdentity;
  std::string resultImporterIdentity;
  ResolvedToolBinding tool;
  ToolVersionProbe toolVersionProbe;
  InvocationRuntimeBinding runtime;
  ToolVersionProbe containerVersionProbe;
  std::vector<std::vector<std::string>> commands;
  std::vector<std::string> inheritEnvironment;
  std::vector<std::string> declaredOutputs;
  std::vector<MaterializedBundleFile> files;
  std::vector<ResolvedExternalFile> externalFiles;
};

enum class InvocationCompletionStatus {
  Success,
  MissingEnvironment,
  ModuleActivationFailed,
  VersionMismatch,
  BundleContentMismatch,
  ToolExit,
  MissingOutput,
};

struct InvocationCompletion {
  InvocationCompletionStatus status;
  int exitCode;
};

llvm::Error finalizeExternalToolInvocationBundle(
    llvm::StringRef bundleRoot,
    const ExternalToolInvocationBundleSpec &specification);

llvm::Expected<int>
executeExternalToolInvocationBundle(llvm::StringRef bundleRoot);

llvm::Expected<InvocationCompletion>
loadExternalToolInvocationCompletion(llvm::StringRef bundleRoot);

} // namespace loom::external_tool

#endif // LOOM_EXTERNALTOOL_INVOCATIONBUNDLE_H
