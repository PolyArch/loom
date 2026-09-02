#ifndef LOOM_EXTERNALTOOL_INVOCATIONBUNDLE_H
#define LOOM_EXTERNALTOOL_INVOCATIONBUNDLE_H

#include "ExternalTool/Binding.h"
#include "ExternalTool/ExternalFile.h"
#include "ExternalTool/LocalConfig.h"
#include "ExternalTool/RuntimeBinding.h"
#include "ExternalTool/ShellProbe.h"

#include "Common/Artifact.h"
#include "Common/BlobDigest.h"
#include "Common/ExecutionControl.h"
#include "Common/ProviderForm.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::external_tool {

class ExternalToolInvocationExecutionReceipt final {
public:
  ExternalToolInvocationExecutionReceipt() = default;

private:
  struct State;

  explicit ExternalToolInvocationExecutionReceipt(
      std::shared_ptr<const State> state)
      : state_(std::move(state)) {}

  std::shared_ptr<const State> state_;

  friend struct ExternalToolInvocationExecutionReceiptAccess;
};

inline constexpr llvm::StringLiteral externalToolInvocationManifestSchema =
    "loom.external_tool_invocation";
inline constexpr llvm::StringLiteral externalToolInvocationManifestVersion =
    "2.4";

/// The CandidateGenerator closure of one semantic invocation: the exact
/// typed input bindings and the exact resolved binding as owner-codec
/// canonical bytes, plus the registry-derived binding identity. The bundle
/// stores and revalidates these bytes but never reinterprets them.
struct CandidateGeneratorInvocationClosure final {
  std::vector<std::uint8_t> typedInputBindings;
  std::vector<std::uint8_t> resolvedBinding;
  BlobDigest::Storage bindingIdentity{};

  friend bool operator==(const CandidateGeneratorInvocationClosure &lhs,
                         const CandidateGeneratorInvocationClosure &rhs) {
    return lhs.typedInputBindings == rhs.typedInputBindings &&
           lhs.resolvedBinding == rhs.resolvedBinding &&
           lhs.bindingIdentity == rhs.bindingIdentity;
  }
  friend bool operator!=(const CandidateGeneratorInvocationClosure &lhs,
                         const CandidateGeneratorInvocationClosure &rhs) {
    return !(lhs == rhs);
  }
};

/// The one exact semantic closure of an invocation bundle: a
/// CandidateGenerator closure (stable tag 0) or an exact EvaluationRequest
/// reference (stable tag 1).
using SemanticInvocationClosure =
    std::variant<CandidateGeneratorInvocationClosure, ArtifactRootReference>;

/// The complete semantic portion of one external-tool invocation. DSE or
/// Evaluation derives this value from its exact descriptor and typed closure;
/// adapters transport it without re-encoding any field.
struct ExternalToolSemanticContract final {
  std::string providerIdentity;
  SemanticInvocationClosure semanticClosure;
  std::string resultImporterIdentity;

  friend bool operator==(const ExternalToolSemanticContract &lhs,
                         const ExternalToolSemanticContract &rhs) {
    return lhs.providerIdentity == rhs.providerIdentity &&
           lhs.semanticClosure == rhs.semanticClosure &&
           lhs.resultImporterIdentity == rhs.resultImporterIdentity;
  }
  friend bool operator!=(const ExternalToolSemanticContract &lhs,
                         const ExternalToolSemanticContract &rhs) {
    return !(lhs == rhs);
  }
};

/// The ExternalTool-owned verification digest framing for one exact semantic
/// descriptor reference. Semantic owners supply the canonical reference bytes
/// and may request only the ExternalPrepareImport form.
llvm::Expected<std::string> deriveExternalToolResultImporterIdentity(
    llvm::ArrayRef<std::uint8_t> semanticDescriptorReferenceBytes,
    ProviderForm providerForm);

struct MaterializedBundleFile {
  std::string relativePath;
  std::string contents;
  std::optional<ArtifactRootReference> sourceArtifact;
  bool executable = false;
};

/// The nonsemantic preparation context for one external provider attempt:
/// the strictly adopted machine-local tool configuration and the destination
/// directory for the finalized bundle. Neither enters an Artifact, Request,
/// Evidence, or binding identity.
struct ExternalToolPreparationContext final {
  LocalToolConfig localConfig;
  std::string bundleDestination;
};

/// The ephemeral prepared handle of one finalized invocation bundle. It owns
/// no semantic closure; every import receives the full typed closure again
/// and recomputes the expected manifest. The digest is only an integrity and
/// lookup key.
struct PreparedExternalToolInvocation final {
  std::string bundleRoot;
  BlobDigest manifestDigest;
};

/// The three independently reviewable content domains of one reusable
/// successful external-tool result. Paths and attempt state are excluded.
struct ExternalToolResultCacheKey final {
  BlobDigest inputMaterialDigest;
  BlobDigest executionConfigurationDigest;
  BlobDigest toolVersionDigest;

  friend bool operator==(const ExternalToolResultCacheKey &lhs,
                         const ExternalToolResultCacheKey &rhs) {
    return lhs.inputMaterialDigest == rhs.inputMaterialDigest &&
           lhs.executionConfigurationDigest ==
               rhs.executionConfigurationDigest &&
           lhs.toolVersionDigest == rhs.toolVersionDigest;
  }
  friend bool operator!=(const ExternalToolResultCacheKey &lhs,
                         const ExternalToolResultCacheKey &rhs) {
    return !(lhs == rhs);
  }
};

enum class ExternalToolResultCacheAvailability {
  Disabled,
  Available,
  Unavailable,
};

enum class ExternalToolResultCacheLookup {
  NotAttempted,
  Hit,
  Miss,
};

enum class ExternalToolResultCacheDiscard {
  NotAttempted,
  Discarded,
  Failed,
};

enum class ExternalToolResultCachePublication {
  NotAttempted,
  Published,
  Failed,
};

/// The caller-owned reuse policy for one external-tool attempt. Requiring a
/// fresh result bypasses all persistent-cache keying and state transitions.
enum class ExternalToolResultReusePolicy {
  AllowExactReuse,
  RequireFresh,
};

/// One launcher-observed command execution. Wall time is operational attempt
/// state and never enters semantic identity or persistent result reuse.
struct ExternalToolCommandExecutionObservation final {
  std::uint64_t commandOrdinal = 0;
  std::uint64_t wallNanoseconds = 0;
  int exitCode = 0;

  friend bool operator==(const ExternalToolCommandExecutionObservation &lhs,
                         const ExternalToolCommandExecutionObservation &rhs) {
    return lhs.commandOrdinal == rhs.commandOrdinal &&
           lhs.wallNanoseconds == rhs.wallNanoseconds &&
           lhs.exitCode == rhs.exitCode;
  }
};

/// The exact cache and execution disposition of one invocation attempt. Cache
/// infrastructure failures remain non-fatal to the external tool, but are no
/// longer erased into diagnostics. A cache hit never invokes the external
/// tool; unsuccessful tool attempts are never published.
struct ExternalToolInvocationExecutionObservation final {
  BlobDigest manifestDigest;
  BlobDigest attemptToken;
  int exitCode;
  ExternalToolResultReusePolicy reusePolicy;
  ExternalToolResultCacheAvailability cacheAvailability;
  ExternalToolResultCacheLookup cacheLookup;
  ExternalToolResultCacheDiscard cacheDiscard;
  ExternalToolResultCachePublication cachePublication;
  bool waitedForCacheKeyLock;
  bool invokedExternalTool;
  std::vector<ExternalToolCommandExecutionObservation> commandExecutions = {};
  ExternalToolInvocationExecutionReceipt receipt = {};
};

/// Reserved operational result when execution control stops a prepared
/// invocation. External tools cannot return negative process exit codes.
inline constexpr int externalToolExecutionStoppedExitCode = -2;

/// Execution control stopped fence admission before a new durable generation
/// began. The prior token, completion, and declared outputs remain untouched.
class ExternalToolExecutionAdmissionStoppedError final
    : public llvm::ErrorInfo<ExternalToolExecutionAdmissionStoppedError> {
public:
  static char ID;

  void log(llvm::raw_ostream &os) const override {
    os << "external tool execution stopped before generation admission";
  }

  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
};

/// A manifest-frozen bounded fork-join group over adjacent independent frozen
/// tool commands. Commands outside every group retain ordered execution, and
/// each group boundary is a barrier.
struct ExternalToolParallelCommandGroup final {
  std::uint64_t beginCommandOrdinal = 0;
  std::uint64_t endCommandOrdinal = 0;
  std::uint64_t workerLimit = 0;

  friend bool operator==(const ExternalToolParallelCommandGroup &lhs,
                         const ExternalToolParallelCommandGroup &rhs) {
    return lhs.beginCommandOrdinal == rhs.beginCommandOrdinal &&
           lhs.endCommandOrdinal == rhs.endCommandOrdinal &&
           lhs.workerLimit == rhs.workerLimit;
  }
};

struct ExternalToolInvocationBundleSpec {
  ExternalToolSemanticContract semanticContract;
  ResolvedToolBinding tool;
  ToolVersionProbe toolVersionProbe;
  InvocationRuntimeBinding runtime;
  ToolVersionProbe containerVersionProbe;
  std::vector<std::vector<std::string>> commands;
  std::vector<std::string> inheritEnvironment;
  std::vector<std::string> declaredOutputs;
  std::vector<MaterializedBundleFile> files;
  std::vector<ResolvedExternalFile> externalFiles;
  std::vector<ResolvedExternalFileTree> externalFileTrees;
  /// Canonical work-relative programs that a preceding frozen-tool command
  /// must create before a later command may execute them.
  std::vector<std::string> toolProducedExecutables = {};
  /// Canonical sorted nonoverlapping independent command groups. This exact
  /// execution schedule is serialized and participates in result reuse.
  std::vector<ExternalToolParallelCommandGroup> parallelCommandGroups = {};
  /// Sorted-unique command ordinals that consume the Common-owned diagnostic
  /// verbosity. Finalization mechanically appends the presentation argument.
  /// This invocation-local projection metadata is not serialized.
  std::vector<std::uint64_t> diagnosticCommandOrdinals = {};
  /// Typed frozen auxiliary command owners. External data inputs never gain
  /// executable authority merely because their host path is executable.
  std::vector<ResolvedAuxiliaryToolExecutable> auxiliaryToolExecutables = {};
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

/// Exit codes reserved by the generated launcher itself. External tools keep
/// their native nonzero exit codes; only launcher-authored failures use this
/// closed domain.
enum class InvocationLauncherExitCode : int {
  LauncherFailure = 119,
  ToolProducedExecutableUnavailable = 120,
  BundleContentMismatch = 121,
  MissingOutput = 122,
  VersionMismatch = 123,
  ModuleActivationFailed = 124,
  MissingEnvironment = 125,
};

struct InvocationCompletion {
  InvocationCompletionStatus status;
  int exitCode;
  BlobDigest manifestDigest;
  BlobDigest attemptToken;
  std::vector<BlobDigest> outputDigests;

  friend bool operator==(const InvocationCompletion &lhs,
                         const InvocationCompletion &rhs) {
    return lhs.status == rhs.status && lhs.exitCode == rhs.exitCode &&
           lhs.manifestDigest == rhs.manifestDigest &&
           lhs.attemptToken == rhs.attemptToken &&
           lhs.outputDigests == rhs.outputDigests;
  }
  friend bool operator!=(const InvocationCompletion &lhs,
                         const InvocationCompletion &rhs) {
    return !(lhs == rhs);
  }
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

struct ExternalToolInvocationExternalFileTree final {
  std::string providerInputSlot;
  std::vector<ExternalFileTreeMember> members;

  friend bool operator==(const ExternalToolInvocationExternalFileTree &lhs,
                         const ExternalToolInvocationExternalFileTree &rhs) {
    return lhs.providerInputSlot == rhs.providerInputSlot &&
           lhs.members == rhs.members;
  }
};

struct ExternalToolInvocationImportExpectation final {
  ExternalToolSemanticContract semanticContract;
  std::vector<ExternalToolInvocationSemanticInput> semanticInputs;
  std::vector<ExternalToolInvocationExternalInput> externalInputs;
  std::vector<ExternalToolInvocationExternalFileTree> externalFileTrees;
  std::vector<std::string> declaredOutputs;
};

/// A prepared invocation has no atomically published completion record. This
/// carries no process-liveness, retry, polling, or lifecycle authority.
struct IncompleteExternalToolInvocationAttempt final {};

/// One valid non-success completion, preserving the script-owned status and
/// exact process exit code without exposing any declared output bytes.
struct FailedExternalToolInvocationAttempt final {
  InvocationCompletionStatus status;
  int exitCode;
};

class ImportedExternalToolInvocationBundle;

/// The closed result of importing one expectation-bound invocation attempt.
/// Integrity and expectation violations remain llvm::Error; only Success can
/// carry an immutable declared-output snapshot.
using ExternalToolInvocationAttemptOutcome =
    std::variant<IncompleteExternalToolInvocationAttempt,
                 FailedExternalToolInvocationAttempt,
                 ImportedExternalToolInvocationBundle>;

class ImportedExternalToolInvocationBundle final {
private:
  ImportedExternalToolInvocationBundle(
      std::vector<std::pair<std::string, std::string>> outputs)
      : outputs_(std::move(outputs)) {}

  std::vector<std::pair<std::string, std::string>> outputs_;

  friend llvm::Expected<ExternalToolInvocationAttemptOutcome>
  importExternalToolInvocationAttempt(
      const PreparedExternalToolInvocation &prepared,
      const ExternalToolInvocationImportExpectation &expectation);
  friend llvm::Expected<ExternalToolInvocationAttemptOutcome>
  importExternalToolInvocationAttempt(
      const PreparedExternalToolInvocation &prepared,
      const ExternalToolInvocationImportExpectation &expectation,
      const ExternalToolInvocationExecutionObservation &execution);
  friend llvm::Expected<std::string> readExternalToolInvocationDeclaredOutput(
      const ImportedExternalToolInvocationBundle &bundle,
      llvm::StringRef relativePath);
  friend struct ImportedExternalToolInvocationBundleAccess;
};

/// The success-only import wrapper's typed projection of an incomplete
/// attempt. Only an absent outputs/completion.json produces this error; a
/// missing manifest or present but malformed completion remains an ordinary
/// bundle integrity failure.
class IncompleteExternalToolInvocationError final
    : public llvm::ErrorInfo<IncompleteExternalToolInvocationError> {
public:
  static char ID;

  void log(llvm::raw_ostream &os) const override {
    os << "invocation is incomplete: the completion record is absent";
  }

  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
};

llvm::Expected<PreparedExternalToolInvocation>
finalizeExternalToolInvocationBundle(
    llvm::StringRef bundleRoot,
    const ExternalToolInvocationBundleSpec &specification);

/// Execution-only compatibility projection that discards the sealed receipt.
/// A current-process import must use the observed executor below instead.
llvm::Expected<int> executeExternalToolInvocationBundle(
    const PreparedExternalToolInvocation &prepared);

/// Executes according to the caller-owned reuse policy while preserving the
/// exact cache and external-tool disposition for journals and manifests.
/// Stopped fence admission returns
/// ExternalToolExecutionAdmissionStoppedError without starting a generation.
llvm::Expected<ExternalToolInvocationExecutionObservation>
executeExternalToolInvocationBundleObserved(
    const PreparedExternalToolInvocation &prepared,
    ExecutionControlView executionControl = {},
    ExternalToolResultReusePolicy reusePolicy =
        ExternalToolResultReusePolicy::AllowExactReuse);

/// Checks the public generation fields used by operational work accounting.
/// This does not prove that the ExternalTool executor produced the fields.
llvm::Error validateExternalToolInvocationExecutionObservation(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationExecutionObservation &observation);

/// Proves that the ExternalTool executor produced the complete observation for
/// the currently published generation and that its atomic completion record
/// has not changed since execution returned. Declared output bytes are owned
/// only by ImportedExternalToolInvocationBundle after strict import.
llvm::Error validateExternalToolInvocationExecutionReceipt(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationExecutionObservation &observation);

/// Derives the exact persistent-result cache key from one verified prepared
/// invocation. This reads and validates the key-bearing generated files and
/// resolved launcher bytes but neither looks up nor publishes a cache entry.
llvm::Expected<ExternalToolResultCacheKey> deriveExternalToolResultCacheKey(
    const PreparedExternalToolInvocation &prepared);

/// Mechanically derives the execution-resource identity of one exact resolved
/// tool and runtime binding. Invocation inputs, outputs, paths, and attempt
/// identity are excluded so independent users of the same binding contend on
/// the same scheduler capacity.
llvm::Expected<BlobDigest> deriveExternalToolExecutionBindingDigest(
    const ResolvedToolBinding &tool, const InvocationRuntimeBinding &runtime);

/// Verifies one prepared manifest and derives its exact execution-resource
/// identity through the same binding codec.
llvm::Expected<BlobDigest> deriveExternalToolExecutionBindingDigest(
    const PreparedExternalToolInvocation &prepared);

/// Diagnostic reader for the completion record of one prepared invocation:
/// the prepared manifest is verified through the shared integrity helper and
/// the record is parsed from the same open bundle root. It is a raw
/// diagnostic view only, not an import or execution authority.
llvm::Expected<InvocationCompletion> loadExternalToolInvocationCompletion(
    const PreparedExternalToolInvocation &prepared);

/// Imports one canonical attempt against the exact prepared handle and full
/// semantic expectation. A present completion must bind both the validated
/// manifest and the current attempt token before any outcome is exposed. Only
/// Success verifies and snapshots every declared ordinary output as owned
/// immutable bytes from the same directory.
llvm::Expected<ExternalToolInvocationAttemptOutcome>
importExternalToolInvocationAttempt(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationImportExpectation &expectation);

/// Receipt-aware strict import for a caller that just executed the bundle.
/// The sealed execution generation, completion record, and expectation-bound
/// declared outputs are checked as one import operation before an immutable
/// output snapshot can escape.
llvm::Expected<ExternalToolInvocationAttemptOutcome>
importExternalToolInvocationAttempt(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationImportExpectation &expectation,
    const ExternalToolInvocationExecutionObservation &execution);

/// Success-only compatibility wrapper over importExternalToolInvocationAttempt.
/// Incomplete and failed outcomes are projected back to import errors.
llvm::Expected<ImportedExternalToolInvocationBundle>
importExternalToolInvocationBundle(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationImportExpectation &expectation);

/// Success-only projection of the receipt-aware strict attempt importer.
llvm::Expected<ImportedExternalToolInvocationBundle>
importExternalToolInvocationBundle(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationImportExpectation &expectation,
    const ExternalToolInvocationExecutionObservation &execution);

/// Reads one declared output from the immutable import snapshot.
llvm::Expected<std::string> readExternalToolInvocationDeclaredOutput(
    const ImportedExternalToolInvocationBundle &bundle,
    llvm::StringRef relativePath);

} // namespace loom::external_tool

#endif // LOOM_EXTERNALTOOL_INVOCATIONBUNDLE_H
