#include "ExternalTool/InvocationBundle.h"

#include "InvocationBundleInternal.h"

#include "Common/BlobDigest.h"
#include "Common/DiagnosticVerbosity.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace loom::external_tool {

char ExternalToolExecutionAdmissionStoppedError::ID = 0;
char IncompleteExternalToolInvocationError::ID = 0;

namespace {

/// The single path-domain predicate for bundle roots, shared by finalization
/// and by every open of the published directory.
llvm::Error validateBundleRootSpelling(llvm::StringRef bundleRoot) {
  if (bundleRoot.empty() || bundleRoot.contains('\0') ||
      !llvm::sys::path::is_absolute(bundleRoot))
    return invocationBundleError("bundle root must be an absolute path");
  const std::filesystem::path root(bundleRoot.str());
  if (root.lexically_normal() != root)
    return invocationBundleError("bundle root must be lexically normalized");
  return llvm::Error::success();
}

llvm::Error writeFile(const std::filesystem::path &path, llvm::StringRef data,
                      bool executable) {
  std::error_code directoryError;
  std::filesystem::create_directories(path.parent_path(), directoryError);
  if (directoryError)
    return invocationBundleError("could not create bundle directory: " +
                       directoryError.message());
  std::error_code outputError;
  llvm::raw_fd_ostream output(path.string(), outputError,
                              llvm::sys::fs::OF_None);
  if (outputError)
    return invocationBundleError("could not open bundle file: " + outputError.message());
  output.write(data.data(), data.size());
  output.close();
  if (output.has_error())
    return invocationBundleError("could not write bundle file");
  if (executable) {
    std::filesystem::permissions(path,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_write |
                                     std::filesystem::perms::owner_exec |
                                     std::filesystem::perms::group_read |
                                     std::filesystem::perms::group_exec,
                                 std::filesystem::perm_options::replace,
                                 outputError);
    if (outputError)
      return invocationBundleError("could not set bundle file permissions: " +
                         outputError.message());
  }
  return llvm::Error::success();
}

llvm::Expected<std::filesystem::path>
createStagingDirectory(const std::filesystem::path &bundleRoot) {
  for (unsigned attempt = 0; attempt != 32; ++attempt) {
    llvm::SmallString<256> model(
        (bundleRoot.string() + ".partial-%%%%%%").c_str());
    llvm::SmallString<256> candidate;
    llvm::sys::fs::createUniquePath(model, candidate, true);
    std::error_code error;
    if (std::filesystem::create_directory(candidate.str().str(), error))
      return std::filesystem::path(candidate.str().str());
    if (error != std::errc::file_exists)
      return invocationBundleError("could not create bundle staging directory: " +
                         error.message());
  }
  return invocationBundleError("could not allocate a bundle staging directory");
}

struct StagingCleanup {
  std::filesystem::path path;
  bool published = false;

  ~StagingCleanup() {
    if (published)
      return;
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

class BundleFileDescriptor final {
public:
  explicit BundleFileDescriptor(int value = -1) : value_(value) {}
  BundleFileDescriptor(const BundleFileDescriptor &) = delete;
  BundleFileDescriptor &operator=(const BundleFileDescriptor &) = delete;
  BundleFileDescriptor(BundleFileDescriptor &&other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  BundleFileDescriptor &operator=(BundleFileDescriptor &&other) noexcept {
    if (this != &other) {
      if (value_ >= 0)
        ::close(value_);
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  ~BundleFileDescriptor() {
    if (value_ >= 0)
      ::close(value_);
  }

  int get() const { return value_; }

private:
  int value_;
};

llvm::Error bundleSystemError(const llvm::Twine &message) {
  return invocationBundleError(message + ": " + std::strerror(errno));
}

bool sameObservedFile(const struct stat &lhs, const struct stat &rhs) {
  return lhs.st_dev == rhs.st_dev && lhs.st_ino == rhs.st_ino &&
         lhs.st_mode == rhs.st_mode && lhs.st_nlink == rhs.st_nlink &&
         lhs.st_size == rhs.st_size &&
         lhs.st_mtim.tv_sec == rhs.st_mtim.tv_sec &&
         lhs.st_mtim.tv_nsec == rhs.st_mtim.tv_nsec &&
         lhs.st_ctim.tv_sec == rhs.st_ctim.tv_sec &&
         lhs.st_ctim.tv_nsec == rhs.st_ctim.tv_nsec;
}

llvm::Expected<BundleFileDescriptor>
openBundleRoot(llvm::StringRef bundleRoot) {
  if (llvm::Error error = validateBundleRootSpelling(bundleRoot))
    return std::move(error);
  BundleFileDescriptor descriptor(
      ::open(bundleRoot.str().c_str(),
             O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
  if (descriptor.get() < 0)
    return bundleSystemError("could not open bundle root");
  return descriptor;
}

llvm::Expected<BundleFileDescriptor>
openOrdinaryBundleFile(int bundleRoot, llvm::StringRef relativePath) {
  auto normalized = normalizedRelativePath(relativePath, "bundle file");
  if (!normalized)
    return normalized.takeError();

  BundleFileDescriptor current(::fcntl(bundleRoot, F_DUPFD_CLOEXEC, 0));
  if (current.get() < 0)
    return bundleSystemError("could not duplicate bundle root descriptor");

  std::vector<std::string> components;
  for (const std::filesystem::path &component :
       std::filesystem::path(*normalized))
    components.push_back(component.string());
  for (std::size_t index = 0; index < components.size(); ++index) {
    const bool final = index + 1 == components.size();
    struct stat status{};
    if (::fstatat(current.get(), components[index].c_str(), &status,
                  AT_SYMLINK_NOFOLLOW) != 0)
      return bundleSystemError("could not inspect bundle file component '" +
                               components[index] + "'");
    if (S_ISLNK(status.st_mode))
      return invocationBundleError("bundle file path contains a symlink component");
    if (final && !S_ISREG(status.st_mode))
      return invocationBundleError("bundle file path must name an ordinary file");
    if (!final && !S_ISDIR(status.st_mode))
      return invocationBundleError("bundle file parent is not an ordinary directory");

    const int flags = final ? O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK
                            : O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_DIRECTORY;
    BundleFileDescriptor next(
        ::openat(current.get(), components[index].c_str(), flags));
    if (next.get() < 0)
      return bundleSystemError("could not open bundle file component '" +
                               components[index] + "'");
    current = std::move(next);
  }
  return current;
}

llvm::Expected<std::string>
readOrdinaryBundleFile(int bundleRoot, llvm::StringRef relativePath) {
  auto file = openOrdinaryBundleFile(bundleRoot, relativePath);
  if (!file)
    return file.takeError();
  struct stat before{};
  if (::fstat(file->get(), &before) != 0)
    return bundleSystemError("could not inspect opened bundle file");
  if (!S_ISREG(before.st_mode) || before.st_size < 0)
    return invocationBundleError("bundle file path must name an ordinary file");
  if (static_cast<std::uintmax_t>(before.st_size) >
      std::numeric_limits<std::size_t>::max())
    return invocationBundleError("bundle file is too large to import");

  std::string contents;
  contents.reserve(static_cast<std::size_t>(before.st_size));
  std::array<char, 64 * 1024> buffer{};
  while (true) {
    const ssize_t count = ::read(file->get(), buffer.data(), buffer.size());
    if (count == 0)
      break;
    if (count < 0) {
      if (errno == EINTR)
        continue;
      return bundleSystemError("could not read bundle file");
    }
    contents.append(buffer.data(), static_cast<std::size_t>(count));
  }

  struct stat after{};
  if (::fstat(file->get(), &after) != 0)
    return bundleSystemError("could not re-inspect opened bundle file");
  if (!sameObservedFile(before, after) ||
      contents.size() != static_cast<std::uintmax_t>(after.st_size))
    return invocationBundleError("bundle file changed while it was read");
  return contents;
}

/// Reads and parses the completion record through one already-open bundle
/// root descriptor; the single completion consumer shared by strict import
/// and the diagnostic reader.
llvm::Expected<InvocationCompletion> readCompletionFromRoot(int bundleRoot) {
  auto contents = readOrdinaryBundleFile(bundleRoot, kCompletionPath);
  if (!contents)
    return contents.takeError();
  return parseInvocationCompletion(*contents);
}

llvm::Expected<std::optional<InvocationCompletion>>
readOptionalCompletionFromRoot(int bundleRoot) {
  struct stat status{};
  if (::fstatat(bundleRoot, kCompletionPath.str().c_str(), &status,
                AT_SYMLINK_NOFOLLOW) != 0) {
    if (errno == ENOENT)
      return std::optional<InvocationCompletion>{};
    return bundleSystemError("could not inspect invocation completion");
  }
  auto completion = readCompletionFromRoot(bundleRoot);
  if (!completion)
    return completion.takeError();
  return std::optional<InvocationCompletion>(std::move(*completion));
}

llvm::Expected<BlobDigest> readAttemptTokenFromRoot(int bundleRoot) {
  auto tokenText = readOrdinaryBundleFile(bundleRoot, kAttemptTokenPath);
  if (!tokenText)
    return tokenText.takeError();
  auto token = parseBlobDigestHex(*tokenText);
  if (!token)
    return invocationBundleError("execution attempt token is malformed");
  return token;
}

/// One owning view of a prepared bundle: the open root descriptor plus the
/// exact manifest bytes, proven to digest to the prepared handle. Strict
/// import reads the manifest, completion, and every declared output through
/// this one descriptor.
struct PreparedBundleView final {
  BundleFileDescriptor root;
  std::string manifestBytes;
};

llvm::Expected<PreparedBundleView>
openPreparedBundle(const PreparedExternalToolInvocation &prepared) {
  auto root = openBundleRoot(prepared.bundleRoot);
  if (!root)
    return root.takeError();
  auto contents = readOrdinaryBundleFile(root->get(), kManifestName);
  if (!contents)
    return contents.takeError();
  if (contentDigest(*contents) != prepared.manifestDigest)
    return invocationBundleError(
        "invocation manifest does not match the prepared handle");
  return PreparedBundleView{std::move(*root), std::move(*contents)};
}

} // namespace

llvm::Expected<std::pair<std::string, InvocationManifestData>>
loadPreparedInvocationManifest(const PreparedExternalToolInvocation &prepared) {
  auto root = openBundleRoot(prepared.bundleRoot);
  if (!root)
    return root.takeError();
  return loadPreparedInvocationManifestFromRoot(prepared, root->get());
}

llvm::Expected<std::pair<std::string, InvocationManifestData>>
loadPreparedInvocationManifestFromRoot(
    const PreparedExternalToolInvocation &prepared, int bundleRoot) {
  if (llvm::Error error = validateBundleRootSpelling(prepared.bundleRoot))
    return std::move(error);
  auto manifestBytes = readOrdinaryBundleFile(bundleRoot, kManifestName);
  if (!manifestBytes)
    return manifestBytes.takeError();
  if (contentDigest(*manifestBytes) != prepared.manifestDigest)
    return invocationBundleError(
        "invocation manifest does not match the prepared handle");
  auto manifest = parseManifest(*manifestBytes);
  if (!manifest)
    return manifest.takeError();
  return std::make_pair(std::move(*manifestBytes), std::move(*manifest));
}

llvm::Expected<std::optional<InvocationCompletion>>
loadOptionalInvocationCompletionFromRoot(int bundleRoot) {
  return readOptionalCompletionFromRoot(bundleRoot);
}

llvm::Expected<BlobDigest> deriveExternalToolExecutionBindingDigest(
    const PreparedExternalToolInvocation &prepared) {
  auto bundle = openPreparedBundle(prepared);
  if (!bundle)
    return bundle.takeError();
  auto manifest = parseManifest(bundle->manifestBytes);
  if (!manifest)
    return manifest.takeError();
  return deriveExternalToolExecutionBindingDigest(manifest->tool,
                                                  manifest->runtime);
}

llvm::Expected<PreparedExternalToolInvocation>
finalizeExternalToolInvocationBundle(
    llvm::StringRef bundleRoot,
    const ExternalToolInvocationBundleSpec &specification) {
  std::uint64_t previousDiagnosticOrdinal = 0;
  bool hasPreviousDiagnosticOrdinal = false;
  for (const std::uint64_t ordinal : specification.diagnosticCommandOrdinals) {
    if (ordinal >= specification.commands.size())
      return invocationBundleError("diagnostic command ordinal is out of range");
    if (hasPreviousDiagnosticOrdinal && previousDiagnosticOrdinal >= ordinal)
      return invocationBundleError(
          "diagnostic command ordinals are not canonical sorted-unique");
    previousDiagnosticOrdinal = ordinal;
    hasPreviousDiagnosticOrdinal = true;
    // The presentation argument reaches the simulation, which is either a
    // tool-produced program or the frozen tool running its own snapshot.
    if (specification.commands[ordinal].empty() ||
        (specification.commands[ordinal].front() !=
             specification.tool.executable &&
         !llvm::is_contained(specification.toolProducedExecutables,
                             specification.commands[ordinal].front())))
      return invocationBundleError("diagnostic command is not the frozen tool or a "
                         "tool-produced executable");
  }
  for (const std::vector<std::string> &command : specification.commands)
    for (const std::string &argument : command)
      if (isDiagnosticVerbosityBinding(argument))
        return invocationBundleError(
            "diagnostic verbosity is owned by invocation finalization");

  if (llvm::Error error = validateSpecification(specification))
    return error;
  if (llvm::Error error = validateBundleRootSpelling(bundleRoot))
    return std::move(error);
  const std::filesystem::path root(bundleRoot.str());
  std::error_code statusError;
  if (std::filesystem::exists(root, statusError) || statusError)
    return invocationBundleError("bundle root already exists or is inaccessible");
  if (!std::filesystem::is_directory(root.parent_path(), statusError) ||
      statusError)
    return invocationBundleError("bundle parent must be an existing directory");
  InvocationManifestData manifest = makeManifest(specification);
  if (std::optional<std::string> argument = diagnosticVerbosityArgument())
    for (const std::uint64_t ordinal : specification.diagnosticCommandOrdinals)
      manifest.commands[ordinal].push_back(*argument);

  llvm::Expected<std::filesystem::path> staging = createStagingDirectory(root);
  if (!staging)
    return staging.takeError();
  StagingCleanup cleanup{*staging};
  std::error_code directoryError;
  std::filesystem::create_directories(*staging / "drivers", directoryError);
  std::filesystem::create_directories(*staging / "inputs", directoryError);
  std::filesystem::create_directories(*staging / "outputs", directoryError);
  // The tool scratch root exists before the first command, so a compiler
  // that creates its library or program one level below it needs no
  // pre-existing product.
  std::filesystem::create_directories(*staging / "work", directoryError);
  if (directoryError)
    return invocationBundleError("could not create bundle layout: " +
                       directoryError.message());
  for (const std::string &output : specification.declaredOutputs) {
    std::filesystem::create_directories((*staging / output).parent_path(),
                                        directoryError);
    if (directoryError)
      return invocationBundleError("could not create declared output directory: " +
                         directoryError.message());
  }
  for (const std::string &executable : specification.toolProducedExecutables) {
    std::filesystem::create_directories((*staging / executable).parent_path(),
                                        directoryError);
    if (directoryError)
      return invocationBundleError(
          "could not create tool-produced executable directory: " +
          directoryError.message());
  }

  for (const MaterializedBundleFile &file : specification.files)
    if (llvm::Error error = writeFile(*staging / file.relativePath,
                                      file.contents, file.executable))
      return error;
  const std::string manifestBytes = serializeManifest(manifest);
  if (llvm::Error error =
          writeFile(*staging / kManifestName.str(), manifestBytes, false))
    return error;
  if (llvm::Error error = writeFile(*staging / kRunScriptName.str(),
                                    renderRunScript(manifest), true))
    return error;

  std::error_code publishError;
  std::filesystem::rename(*staging, root, publishError);
  if (publishError)
    return invocationBundleError("could not publish bundle: " + publishError.message());
  cleanup.published = true;
  return PreparedExternalToolInvocation{bundleRoot.str(),
                                        contentDigest(manifestBytes)};
}

llvm::Error validateExternalToolInvocationExecutionObservation(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationExecutionObservation &observation) {
  auto bundle = openPreparedBundle(prepared);
  if (!bundle)
    return bundle.takeError();
  return validateExternalToolInvocationExecutionObservationFromRoot(
      prepared, observation, bundle->root.get());
}

llvm::Error validateExternalToolInvocationExecutionObservationFromRoot(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationExecutionObservation &observation,
    int bundleRoot) {
  if (observation.manifestDigest != prepared.manifestDigest)
    return invocationBundleError(
        "execution observation manifest differs from prepared invocation");
  if (!llvm::all_of(
          llvm::enumerate(observation.commandExecutions), [&](const auto row) {
            return row.index() == row.value().commandOrdinal &&
                   (observation.exitCode != 0 || row.value().exitCode == 0);
          }))
    return invocationBundleError(
        "execution observation command results are inconsistent");
  auto manifest = readOrdinaryBundleFile(bundleRoot, kManifestName);
  if (!manifest)
    return manifest.takeError();
  if (contentDigest(*manifest) != prepared.manifestDigest)
    return invocationBundleError(
        "invocation manifest does not match the prepared handle");
  auto token = readAttemptTokenFromRoot(bundleRoot);
  if (!token)
    return token.takeError();
  if (*token != observation.attemptToken)
    return invocationBundleError(
        "execution observation belongs to another attempt generation");
  return llvm::Error::success();
}

llvm::Expected<InvocationCompletion> loadExternalToolInvocationCompletion(
    const PreparedExternalToolInvocation &prepared) {
  auto bundle = openPreparedBundle(prepared);
  if (!bundle)
    return bundle.takeError();
  return readCompletionFromRoot(bundle->root.get());
}

namespace {

llvm::Expected<ExternalToolInvocationAttemptOutcome>
importExternalToolInvocationAttemptImpl(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationImportExpectation &expectation,
    const ExternalToolInvocationExecutionObservation *execution) {
  const auto receiptState =
      execution ? ExternalToolInvocationExecutionReceiptAccess::state(
                      execution->receipt)
                : nullptr;
  if (execution) {
    if (!receiptState)
      return invocationBundleError("execution observation has no executor receipt");
    if (!receiptState->matches(prepared, *execution))
      return invocationBundleError(
          "execution observation differs from its sealed executor state");
    if (llvm::Error error = validateInvocationCompletionExecutionBoundary(
            prepared, execution->attemptToken, execution->exitCode,
            receiptState->completion))
      return std::move(error);
  }

  auto bundle = openPreparedBundle(prepared);
  if (!bundle)
    return bundle.takeError();
  auto manifest = parseManifest(bundle->manifestBytes);
  if (!manifest)
    return manifest.takeError();
  if (manifest->semanticContract.providerIdentity !=
      expectation.semanticContract.providerIdentity)
    return invocationBundleError("invocation provider identity does not match importer");
  if (manifest->semanticContract.semanticClosure !=
      expectation.semanticContract.semanticClosure)
    return invocationBundleError("invocation semantic closure does not match importer");
  if (manifest->semanticContract.resultImporterIdentity !=
      expectation.semanticContract.resultImporterIdentity)
    return invocationBundleError("invocation result importer identity does not match");

  std::vector<ExternalToolInvocationSemanticInput> semanticInputs;
  for (const ManifestMaterializedFile &file : manifest->materializedFiles)
    if (file.sourceArtifact)
      semanticInputs.push_back(ExternalToolInvocationSemanticInput{
          file.relativePath, *file.sourceArtifact, file.contentDigest});
  if (semanticInputs != expectation.semanticInputs)
    return invocationBundleError("invocation semantic inputs do not match importer");
  std::vector<ExternalToolInvocationExternalInput> externalInputs;
  externalInputs.reserve(manifest->externalFiles.size());
  for (const ResolvedExternalFile &file : manifest->externalFiles)
    externalInputs.push_back(ExternalToolInvocationExternalInput{
        file.providerInputSlot, file.fingerprint});
  if (externalInputs != expectation.externalInputs)
    return invocationBundleError("invocation external inputs do not match importer");
  std::vector<ExternalToolInvocationExternalFileTree> externalFileTrees;
  externalFileTrees.reserve(manifest->externalFileTrees.size());
  for (const ResolvedExternalFileTree &tree : manifest->externalFileTrees)
    externalFileTrees.push_back(ExternalToolInvocationExternalFileTree{
        tree.providerInputSlot, tree.members});
  if (externalFileTrees != expectation.externalFileTrees)
    return invocationBundleError("invocation external file trees do not match importer");
  if (manifest->declaredOutputs != expectation.declaredOutputs)
    return invocationBundleError("invocation declared outputs do not match importer");

  const int bundleRoot = bundle->root.get();
  auto currentManifest = readOrdinaryBundleFile(bundleRoot, kManifestName);
  if (!currentManifest)
    return currentManifest.takeError();
  if (*currentManifest != bundle->manifestBytes)
    return invocationBundleError("invocation manifest changed during import");

  // Preserve the raw recovery projection for a never-started bundle: only an
  // absent completion is incomplete, and a token is required once a
  // completion exists. Receipt-aware import additionally anchors absence to
  // the executor's sealed generation.
  auto initialCompletion = readOptionalCompletionFromRoot(bundleRoot);
  if (!initialCompletion)
    return initialCompletion.takeError();
  if (!*initialCompletion && !execution)
    return ExternalToolInvocationAttemptOutcome(
        IncompleteExternalToolInvocationAttempt{});

  auto initialToken = readAttemptTokenFromRoot(bundleRoot);
  if (!initialToken)
    return initialToken.takeError();
  auto completion = readOptionalCompletionFromRoot(bundleRoot);
  if (!completion)
    return completion.takeError();
  if (receiptState) {
    if (*initialToken != execution->attemptToken ||
        *completion != receiptState->completion)
      return invocationBundleError(
          "invocation generation changed before declared outputs were read");
  } else if (!*completion) {
    return ExternalToolInvocationAttemptOutcome(
        IncompleteExternalToolInvocationAttempt{});
  }

  const auto validateFinalBoundary = [&]() -> llvm::Error {
    auto finalManifest = readOrdinaryBundleFile(bundleRoot, kManifestName);
    if (!finalManifest)
      return finalManifest.takeError();
    auto finalCompletion = readOptionalCompletionFromRoot(bundleRoot);
    if (!finalCompletion)
      return finalCompletion.takeError();
    auto finalToken = readAttemptTokenFromRoot(bundleRoot);
    if (!finalToken)
      return finalToken.takeError();
    const std::optional<InvocationCompletion> &expectedCompletion =
        receiptState ? receiptState->completion : *completion;
    const BlobDigest &expectedToken =
        receiptState ? execution->attemptToken : (*completion)->attemptToken;
    if (*finalManifest != bundle->manifestBytes ||
        *finalCompletion != expectedCompletion || *finalToken != expectedToken)
      return invocationBundleError(
          "invocation generation changed while declared outputs were read");
    return llvm::Error::success();
  };

  if (!*completion) {
    if (llvm::Error error = validateFinalBoundary())
      return std::move(error);
    return ExternalToolInvocationAttemptOutcome(
        IncompleteExternalToolInvocationAttempt{});
  }
  if ((*completion)->manifestDigest != contentDigest(bundle->manifestBytes))
    return invocationBundleError("completion does not bind the imported manifest");
  if ((*completion)->attemptToken != *initialToken)
    return invocationBundleError("completion belongs to another attempt generation");
  if ((*completion)->status != InvocationCompletionStatus::Success) {
    if (llvm::Error error = validateFinalBoundary())
      return std::move(error);
    return ExternalToolInvocationAttemptOutcome(
        FailedExternalToolInvocationAttempt{(*completion)->status,
                                            (*completion)->exitCode});
  }
  if ((*completion)->outputDigests.size() != manifest->declaredOutputs.size())
    return invocationBundleError("completion output digest count is invalid");

  std::vector<std::pair<std::string, std::string>> outputs;
  outputs.reserve(manifest->declaredOutputs.size());
  for (std::size_t index = 0; index < manifest->declaredOutputs.size();
       ++index) {
    const std::string &path = manifest->declaredOutputs[index];
    auto output = readOrdinaryBundleFile(bundleRoot, path);
    if (!output)
      return output.takeError();
    if (contentDigest(*output) != (*completion)->outputDigests[index])
      return invocationBundleError("declared output does not match completion digest");
    outputs.emplace_back(path, std::move(*output));
  }
  if (llvm::Error error = validateFinalBoundary())
    return std::move(error);
  return ExternalToolInvocationAttemptOutcome(
      ImportedExternalToolInvocationBundleAccess::create(std::move(outputs)));
}

} // namespace

llvm::Expected<ExternalToolInvocationAttemptOutcome>
importExternalToolInvocationAttempt(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationImportExpectation &expectation) {
  return importExternalToolInvocationAttemptImpl(prepared, expectation,
                                                 nullptr);
}

llvm::Expected<ExternalToolInvocationAttemptOutcome>
importExternalToolInvocationAttempt(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationImportExpectation &expectation,
    const ExternalToolInvocationExecutionObservation &execution) {
  return importExternalToolInvocationAttemptImpl(prepared, expectation,
                                                 &execution);
}

llvm::Expected<ImportedExternalToolInvocationBundle>
importExternalToolInvocationBundle(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationImportExpectation &expectation) {
  auto attempt = importExternalToolInvocationAttempt(prepared, expectation);
  if (!attempt)
    return attempt.takeError();
  if (std::holds_alternative<IncompleteExternalToolInvocationAttempt>(*attempt))
    return llvm::make_error<IncompleteExternalToolInvocationError>();
  if (std::holds_alternative<FailedExternalToolInvocationAttempt>(*attempt))
    return invocationBundleError("invocation did not complete successfully");
  return std::get<ImportedExternalToolInvocationBundle>(std::move(*attempt));
}

llvm::Expected<ImportedExternalToolInvocationBundle>
importExternalToolInvocationBundle(
    const PreparedExternalToolInvocation &prepared,
    const ExternalToolInvocationImportExpectation &expectation,
    const ExternalToolInvocationExecutionObservation &execution) {
  auto attempt =
      importExternalToolInvocationAttempt(prepared, expectation, execution);
  if (!attempt)
    return attempt.takeError();
  if (std::holds_alternative<IncompleteExternalToolInvocationAttempt>(*attempt))
    return llvm::make_error<IncompleteExternalToolInvocationError>();
  if (std::holds_alternative<FailedExternalToolInvocationAttempt>(*attempt))
    return invocationBundleError("invocation did not complete successfully");
  return std::get<ImportedExternalToolInvocationBundle>(std::move(*attempt));
}

llvm::Expected<std::string> readExternalToolInvocationDeclaredOutput(
    const ImportedExternalToolInvocationBundle &bundle,
    llvm::StringRef relativePath) {
  auto normalized = normalizedRelativePath(relativePath, "declared output");
  if (!normalized)
    return normalized.takeError();
  const auto found = llvm::find_if(bundle.outputs_, [&](const auto &output) {
    return output.first == *normalized;
  });
  if (found == bundle.outputs_.end())
    return invocationBundleError("output is not declared by the invocation manifest");
  return found->second;
}

} // namespace loom::external_tool
