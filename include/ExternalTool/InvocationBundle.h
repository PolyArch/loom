#ifndef LOOM_EXTERNALTOOL_INVOCATIONBUNDLE_H
#define LOOM_EXTERNALTOOL_INVOCATIONBUNDLE_H

#include "ExternalTool/Binding.h"
#include "ExternalTool/ExternalFile.h"
#include "ExternalTool/RuntimeBinding.h"
#include "ExternalTool/ShellProbe.h"

#include "Common/Artifact.h"
#include "Common/BlobDigest.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::external_tool {

struct MaterializedBundleFile {
  std::string relativePath;
  std::string contents;
  std::optional<ArtifactRootReference> sourceArtifact;
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
  BlobDigest manifestDigest;
  std::vector<BlobDigest> outputDigests;
};

struct ExternalToolInvocationSemanticInput final {
  std::string relativePath;
  ArtifactRootReference sourceArtifact;
  BlobDigest contentDigest;

  friend bool operator==(const ExternalToolInvocationSemanticInput &lhs,
                         const ExternalToolInvocationSemanticInput &rhs) {
    return lhs.relativePath == rhs.relativePath &&
           lhs.sourceArtifact == rhs.sourceArtifact &&
           lhs.contentDigest == rhs.contentDigest;
  }
};

struct ExternalToolInvocationExternalInput final {
  std::string providerInputSlot;
  ExternalFileFingerprint fingerprint;

  friend bool operator==(const ExternalToolInvocationExternalInput &lhs,
                         const ExternalToolInvocationExternalInput &rhs) {
    return lhs.providerInputSlot == rhs.providerInputSlot &&
           lhs.fingerprint == rhs.fingerprint;
  }
};

struct ExternalToolInvocationImportExpectation final {
  std::string providerIdentity;
  std::string semanticBindingIdentity;
  std::string resultImporterIdentity;
  std::vector<ExternalToolInvocationSemanticInput> semanticInputs;
  std::vector<ExternalToolInvocationExternalInput> externalInputs;
  std::vector<std::string> declaredOutputs;
};

class ImportedExternalToolInvocationBundle final {
private:
  ImportedExternalToolInvocationBundle(
      std::vector<std::pair<std::string, std::string>> outputs)
      : outputs_(std::move(outputs)) {}

  std::vector<std::pair<std::string, std::string>> outputs_;

  friend llvm::Expected<ImportedExternalToolInvocationBundle>
  importExternalToolInvocationBundle(
      llvm::StringRef bundleRoot,
      const ExternalToolInvocationImportExpectation &expectation);
  friend llvm::Expected<std::string> readExternalToolInvocationDeclaredOutput(
      const ImportedExternalToolInvocationBundle &bundle,
      llvm::StringRef relativePath);
};

llvm::Error finalizeExternalToolInvocationBundle(
    llvm::StringRef bundleRoot,
    const ExternalToolInvocationBundleSpec &specification);

llvm::Expected<int>
executeExternalToolInvocationBundle(llvm::StringRef bundleRoot);

llvm::Expected<InvocationCompletion>
loadExternalToolInvocationCompletion(llvm::StringRef bundleRoot);

/// Strictly imports one canonical, successfully completed invocation attempt,
/// binds it to the caller's exact semantic expectations, and snapshots all
/// declared ordinary output bytes from the same bundle directory.
llvm::Expected<ImportedExternalToolInvocationBundle>
importExternalToolInvocationBundle(
    llvm::StringRef bundleRoot,
    const ExternalToolInvocationImportExpectation &expectation);

/// Reads one declared output from the immutable import snapshot.
llvm::Expected<std::string> readExternalToolInvocationDeclaredOutput(
    const ImportedExternalToolInvocationBundle &bundle,
    llvm::StringRef relativePath);

} // namespace loom::external_tool

#endif // LOOM_EXTERNALTOOL_INVOCATIONBUNDLE_H
