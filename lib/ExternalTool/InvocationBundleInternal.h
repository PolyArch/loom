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
constexpr llvm::StringLiteral kTypedClosureManifestVersion = "2.0";
constexpr llvm::StringLiteral kExternalFileTreeManifestVersion = "2.1";
constexpr llvm::StringLiteral kToolProducedExecutableManifestVersion = "2.2";
constexpr llvm::StringLiteral kCurrentManifestVersion =
    externalToolInvocationManifestVersion;

struct ManifestMaterializedFile final {
  std::string relativePath;
  bool executable;
  BlobDigest contentDigest;
  std::optional<ArtifactRootReference> sourceArtifact;
};

struct InvocationManifestData final {
  ExternalToolSemanticContract semanticContract;
  ResolvedToolBinding tool;
  ToolVersionProbe toolVersionProbe;
  InvocationRuntimeBinding runtime;
  ToolVersionProbe containerVersionProbe;
  std::vector<std::vector<std::string>> commands;
  std::vector<std::string> inheritEnvironment;
  std::vector<ManifestMaterializedFile> materializedFiles;
  std::vector<ResolvedExternalFile> externalFiles;
  std::vector<ResolvedExternalFileTree> externalFileTrees;
  std::vector<std::string> declaredOutputs;
  std::vector<std::string> toolProducedExecutables;
};

/// The single content digest of in-memory bundle bytes, used for manifests,
/// materialized files, and declared outputs.
BlobDigest contentDigest(llvm::StringRef contents);

/// The sole canonical text codec used at the shell/JSON boundary. Runtime
/// logic carries InvocationCompletionStatus rather than comparing spellings.
llvm::StringRef completionStatusSpelling(InvocationCompletionStatus status);

/// The canonical manifest JSON bytes of one invocation bundle.
std::string
serializeManifest(const InvocationManifestData &manifest,
                  llvm::StringRef version = kCurrentManifestVersion);

/// The deterministic run.sh bytes of one invocation bundle.
std::string renderRunScript(const InvocationManifestData &manifest);

} // namespace loom::external_tool

#endif // LOOM_LIB_EXTERNALTOOL_INVOCATIONBUNDLEINTERNAL_H
