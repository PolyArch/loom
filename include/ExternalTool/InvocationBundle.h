#ifndef LOOM_EXTERNALTOOL_INVOCATIONBUNDLE_H
#define LOOM_EXTERNALTOOL_INVOCATIONBUNDLE_H

#include "ExternalTool/Binding.h"
#include "ExternalTool/ExternalFile.h"
#include "ExternalTool/LocalConfig.h"
#include "ExternalTool/RuntimeBinding.h"
#include "ExternalTool/ShellProbe.h"

#include "Common/Artifact.h"
#include "Common/BlobDigest.h"
#include "Common/ProviderForm.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::external_tool {

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
  ExternalToolSemanticContract semanticContract;
  std::vector<ExternalToolInvocationSemanticInput> semanticInputs;
  std::vector<ExternalToolInvocationExternalInput> externalInputs;
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
  friend llvm::Expected<std::string> readExternalToolInvocationDeclaredOutput(
      const ImportedExternalToolInvocationBundle &bundle,
      llvm::StringRef relativePath);
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

llvm::Expected<PreparedExternalToolInvocation> finalizeExternalToolInvocationBundle(
    llvm::StringRef bundleRoot,
    const ExternalToolInvocationBundleSpec &specification);

llvm::Expected<int> executeExternalToolInvocationBundle(
    const PreparedExternalToolInvocation &prepared);

/// Diagnostic reader for the completion record of one prepared invocation:
/// the prepared manifest is verified through the shared integrity helper and
/// the record is parsed from the same open bundle root. It is a raw
/// diagnostic view only, not an import or execution authority.
llvm::Expected<InvocationCompletion>
loadExternalToolInvocationCompletion(
    const PreparedExternalToolInvocation &prepared);

/// Imports one canonical attempt against the exact prepared handle and full
/// semantic expectation. The completion must bind the validated manifest
/// before any outcome is exposed. Only Success verifies and snapshots every
/// declared ordinary output as owned immutable bytes from the same directory.
llvm::Expected<ExternalToolInvocationAttemptOutcome>
importExternalToolInvocationAttempt(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationImportExpectation &expectation);

/// Success-only compatibility wrapper over importExternalToolInvocationAttempt.
/// Incomplete and failed outcomes are projected back to import errors.
llvm::Expected<ImportedExternalToolInvocationBundle>
importExternalToolInvocationBundle(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationImportExpectation &expectation);

/// Reads one declared output from the immutable import snapshot.
llvm::Expected<std::string> readExternalToolInvocationDeclaredOutput(
    const ImportedExternalToolInvocationBundle &bundle,
    llvm::StringRef relativePath);

} // namespace loom::external_tool

#endif // LOOM_EXTERNALTOOL_INVOCATIONBUNDLE_H
