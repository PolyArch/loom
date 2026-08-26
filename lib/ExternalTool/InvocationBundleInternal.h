#ifndef LOOM_LIB_EXTERNALTOOL_INVOCATIONBUNDLEINTERNAL_H
#define LOOM_LIB_EXTERNALTOOL_INVOCATIONBUNDLEINTERNAL_H

#include "ExternalTool/InvocationBundle.h"

#include "Common/BlobDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::external_tool {

inline constexpr llvm::StringLiteral kInvocationCompletionSchema =
    "loom.external_tool_completion";
inline constexpr llvm::StringLiteral kInvocationCompletionVersion = "2.0";

llvm::Error invocationBundleError(const llvm::Twine &message);

struct ExternalToolInvocationExecutionReceipt::State final {
  std::string bundleRoot;
  BlobDigest manifestDigest;
  BlobDigest attemptToken;
  int exitCode = 0;
  ExternalToolResultReusePolicy reusePolicy =
      ExternalToolResultReusePolicy::AllowExactReuse;
  ExternalToolResultCacheAvailability cacheAvailability =
      ExternalToolResultCacheAvailability::Disabled;
  ExternalToolResultCacheLookup cacheLookup =
      ExternalToolResultCacheLookup::NotAttempted;
  ExternalToolResultCacheDiscard cacheDiscard =
      ExternalToolResultCacheDiscard::NotAttempted;
  ExternalToolResultCachePublication cachePublication =
      ExternalToolResultCachePublication::NotAttempted;
  bool waitedForCacheKeyLock = false;
  bool invokedExternalTool = false;
  std::vector<ExternalToolCommandExecutionObservation> commandExecutions;
  std::optional<InvocationCompletion> completion;

  State(const PreparedExternalToolInvocation &prepared,
        const ExternalToolInvocationExecutionObservation &observation,
        std::optional<InvocationCompletion> completion)
      : bundleRoot(prepared.bundleRoot),
        manifestDigest(observation.manifestDigest),
        attemptToken(observation.attemptToken), exitCode(observation.exitCode),
        reusePolicy(observation.reusePolicy),
        cacheAvailability(observation.cacheAvailability),
        cacheLookup(observation.cacheLookup),
        cacheDiscard(observation.cacheDiscard),
        cachePublication(observation.cachePublication),
        waitedForCacheKeyLock(observation.waitedForCacheKeyLock),
        invokedExternalTool(observation.invokedExternalTool),
        commandExecutions(observation.commandExecutions),
        completion(std::move(completion)) {}

  bool
  matches(const PreparedExternalToolInvocation &prepared,
          const ExternalToolInvocationExecutionObservation &observation) const {
    return bundleRoot == prepared.bundleRoot &&
           manifestDigest == prepared.manifestDigest &&
           manifestDigest == observation.manifestDigest &&
           attemptToken == observation.attemptToken &&
           exitCode == observation.exitCode &&
           reusePolicy == observation.reusePolicy &&
           cacheAvailability == observation.cacheAvailability &&
           cacheLookup == observation.cacheLookup &&
           cacheDiscard == observation.cacheDiscard &&
           cachePublication == observation.cachePublication &&
           waitedForCacheKeyLock == observation.waitedForCacheKeyLock &&
           invokedExternalTool == observation.invokedExternalTool &&
           commandExecutions == observation.commandExecutions;
  }
};

struct ExternalToolInvocationExecutionReceiptAccess final {
  static ExternalToolInvocationExecutionReceipt
  create(const PreparedExternalToolInvocation &prepared,
         const ExternalToolInvocationExecutionObservation &observation,
         std::optional<InvocationCompletion> completion) {
    return ExternalToolInvocationExecutionReceipt(
        std::make_shared<const ExternalToolInvocationExecutionReceipt::State>(
            prepared, observation, std::move(completion)));
  }

  static std::shared_ptr<const ExternalToolInvocationExecutionReceipt::State>
  state(const ExternalToolInvocationExecutionReceipt &receipt) {
    return receipt.state_;
  }
};

struct ImportedExternalToolInvocationBundleAccess final {
  static ImportedExternalToolInvocationBundle
  create(std::vector<std::pair<std::string, std::string>> outputs) {
    return ImportedExternalToolInvocationBundle(std::move(outputs));
  }
};

/// The bundle-internal layout names shared by the manifest codec, the
/// run-script renderer, finalization, execution, and strict import.
constexpr llvm::StringLiteral kManifestName = "tool-invocation.json";
constexpr llvm::StringLiteral kRunScriptName = "run.sh";
constexpr llvm::StringLiteral kCompletionPath = "outputs/completion.json";
constexpr llvm::StringLiteral kStdoutPath = "outputs/stdout.log";
constexpr llvm::StringLiteral kStderrPath = "outputs/stderr.log";
constexpr llvm::StringLiteral kToolVersionPath = "outputs/.loom-tool-version";
constexpr llvm::StringLiteral kAttemptTokenPath = ".loom-attempt-token";
constexpr llvm::StringLiteral kCommandObservationsPath =
    ".loom-command-observations";
constexpr llvm::StringLiteral kCommandExecutionDirectory =
    ".loom-command-execution";
constexpr llvm::StringLiteral kTypedClosureManifestVersion = "2.0";
constexpr llvm::StringLiteral kExternalFileTreeManifestVersion = "2.1";
constexpr llvm::StringLiteral kToolProducedExecutableManifestVersion = "2.2";
constexpr llvm::StringLiteral kParallelCommandGroupManifestVersion = "2.3";
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
  std::vector<ExternalToolParallelCommandGroup> parallelCommandGroups;
  std::string version;
};

llvm::Error validateParallelCommandGroups(
    llvm::ArrayRef<ExternalToolParallelCommandGroup> groups,
    llvm::ArrayRef<std::vector<std::string>> commands,
    llvm::StringRef toolExecutable,
    llvm::ArrayRef<std::string> toolProducedExecutables);

llvm::Expected<std::vector<ExternalToolParallelCommandGroup>>
parseParallelCommandGroups(const llvm::json::Object &manifest);

void writeParallelCommandGroups(
    llvm::json::OStream &json,
    llvm::ArrayRef<ExternalToolParallelCommandGroup> groups);

llvm::Expected<std::vector<ExternalToolCommandExecutionObservation>>
loadCommandExecutionObservations(const PreparedExternalToolInvocation &prepared,
                                 const BlobDigest &attemptToken,
                                 std::uint64_t commandCount);

/// Opens the prepared root through the bundle integrity path and returns the
/// exact canonical manifest bytes and parsed typed manifest.
llvm::Expected<std::pair<std::string, InvocationManifestData>>
loadPreparedInvocationManifest(const PreparedExternalToolInvocation &prepared);

/// The sole canonical completion serializer shared by the generated launcher
/// and cache restoration.
std::string
serializeInvocationCompletion(InvocationCompletionStatus status, int exitCode,
                              const BlobDigest &manifestDigest,
                              const BlobDigest &attemptToken,
                              llvm::ArrayRef<BlobDigest> outputDigests);

llvm::Expected<InvocationCompletion>
parseInvocationCompletion(llvm::StringRef contents);

/// A present completion belongs to exactly one observed execution boundary.
/// Absence remains the canonical incomplete-attempt projection.
llvm::Error validateInvocationCompletionExecutionBoundary(
    const PreparedExternalToolInvocation &prepared,
    const BlobDigest &attemptToken, int exitCode,
    const std::optional<InvocationCompletion> &completion);

/// The one JSON codec for the exact version probe carried by a manifest and
/// by the persistent tool-version cache domain.
void writeToolVersionProbeJson(llvm::json::OStream &json,
                               const ToolVersionProbe &probe);

/// The single content digest of in-memory bundle bytes, used for manifests,
/// materialized files, and declared outputs.
BlobDigest contentDigest(llvm::StringRef contents);

/// The sole canonical text codec used at the shell/JSON boundary. Runtime
/// logic carries InvocationCompletionStatus rather than comparing spellings.
llvm::StringRef completionStatusSpelling(InvocationCompletionStatus status);

/// The canonical manifest JSON bytes of one invocation bundle.
std::string serializeManifest(const InvocationManifestData &manifest);

/// The deterministic run.sh bytes of one invocation bundle.
std::string renderRunScript(const InvocationManifestData &manifest);

} // namespace loom::external_tool

#endif // LOOM_LIB_EXTERNALTOOL_INVOCATIONBUNDLEINTERNAL_H
