#ifndef LOOM_LIB_EXTERNALTOOL_INVOCATIONBUNDLEINTERNAL_H
#define LOOM_LIB_EXTERNALTOOL_INVOCATIONBUNDLEINTERNAL_H

#include "ExternalTool/InvocationBundle.h"

#include "Common/BlobDigest.h"

#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>
#include <vector>

namespace loom::external_tool {

/// The bundle-internal layout names shared by the manifest codec, the
/// run-script renderer, finalization, execution, and strict import.
constexpr llvm::StringLiteral kManifestName = "tool-invocation.json";
constexpr llvm::StringLiteral kRunScriptName = "run.sh";
constexpr llvm::StringLiteral kCompletionPath = "outputs/completion.json";
constexpr llvm::StringLiteral kStdoutPath = "outputs/stdout.log";
constexpr llvm::StringLiteral kStderrPath = "outputs/stderr.log";
constexpr llvm::StringLiteral kToolVersionPath = "outputs/.loom-tool-version";

struct ManifestMaterializedFile final {
  std::string relativePath;
  bool executable;
  BlobDigest contentDigest;
  std::optional<ArtifactRootReference> sourceArtifact;
};

struct InvocationManifestData final {
  std::string providerIdentity;
  SemanticInvocationClosure semanticClosure;
  std::string resultImporterIdentity;
  ResolvedToolBinding tool;
  ToolVersionProbe toolVersionProbe;
  InvocationRuntimeBinding runtime;
  ToolVersionProbe containerVersionProbe;
  std::vector<std::vector<std::string>> commands;
  std::vector<std::string> inheritEnvironment;
  std::vector<ManifestMaterializedFile> materializedFiles;
  std::vector<ResolvedExternalFile> externalFiles;
  std::vector<std::string> declaredOutputs;
};

/// The single content digest of in-memory bundle bytes, used for manifests,
/// materialized files, and declared outputs.
BlobDigest contentDigest(llvm::StringRef contents);

/// The canonical manifest JSON bytes of one invocation bundle.
std::string serializeManifest(const InvocationManifestData &manifest);

/// The deterministic run.sh bytes of one invocation bundle.
std::string renderRunScript(const InvocationManifestData &manifest);

} // namespace loom::external_tool

#endif // LOOM_LIB_EXTERNALTOOL_INVOCATIONBUNDLEINTERNAL_H
