#include "Application/SourceAdmission.h"

#include "Common/BlobDigest.h"
#include "Common/TimeoutBudgets.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include <unistd.h>

namespace loom::application {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "application_source_invalid: " + message);
}

llvm::Error failed(const llvm::Twine &message) {
  return llvm::createStringError(std::make_error_code(std::errc::io_error),
                                 "application_source_admission_failed: " +
                                     message);
}

struct CaptureFiles final {
  llvm::SmallString<128> output;
  llvm::SmallString<128> error;

  CaptureFiles() = default;
  CaptureFiles(const CaptureFiles &) = delete;
  CaptureFiles &operator=(const CaptureFiles &) = delete;
  CaptureFiles(CaptureFiles &&other) noexcept
      : output(std::move(other.output)), error(std::move(other.error)) {
    other.output.clear();
    other.error.clear();
  }
  ~CaptureFiles() {
    if (!output.empty())
      llvm::sys::fs::remove(output);
    if (!error.empty())
      llvm::sys::fs::remove(error);
  }
};

llvm::Expected<CaptureFiles> createCaptureFiles() {
  CaptureFiles files;
  int outputDescriptor = -1;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "loom-application-git", "stdout", outputDescriptor, files.output))
    return failed("cannot create Git stdout capture: " + error.message());
  ::close(outputDescriptor);
  int errorDescriptor = -1;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "loom-application-git", "stderr", errorDescriptor, files.error))
    return failed("cannot create Git stderr capture: " + error.message());
  ::close(errorDescriptor);
  return std::move(files);
}

llvm::Expected<std::string> readCapture(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false, false);
  if (!buffer)
    return failed("cannot read Git command capture: " +
                  buffer.getError().message());
  return (*buffer)->getBuffer().str();
}

struct GitCommandResult final {
  int status = -1;
  bool executionFailed = false;
  std::string output;
  std::string error;
};

llvm::Expected<GitCommandResult>
runGit(llvm::StringRef executable, llvm::ArrayRef<llvm::StringRef> arguments) {
  auto captures = createCaptureFiles();
  if (!captures)
    return captures.takeError();
  llvm::SmallVector<llvm::StringRef, 16> command;
  command.push_back(executable);
  command.append(arguments.begin(), arguments.end());
  const std::array<std::optional<llvm::StringRef>, 3> redirects = {
      llvm::StringRef(), captures->output.str(), captures->error.str()};
  std::string executionMessage;
  bool executionFailed = false;
  const int status = llvm::sys::ExecuteAndWait(
      executable, command, std::nullopt, redirects,
      static_cast<unsigned>(timeout::seconds(timeout::Tier::Fast)), 256,
      &executionMessage, &executionFailed);
  auto output = readCapture(captures->output);
  auto error = readCapture(captures->error);
  if (!output)
    return output.takeError();
  if (!error)
    return error.takeError();
  if (!executionMessage.empty()) {
    if (!error->empty())
      error->append("\n");
    error->append(executionMessage);
  }
  return GitCommandResult{status, executionFailed, std::move(*output),
                          std::move(*error)};
}

struct GitUnavailable final {};
struct GitlinkIndexEntry final {
  std::string commit;
};
using GitlinkIndexResult = std::variant<GitlinkIndexEntry, GitUnavailable>;

bool lowercaseHex(llvm::StringRef value) {
  return llvm::all_of(value, [](unsigned char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

llvm::Expected<GitlinkIndexResult>
readGitlinkIndex(llvm::StringRef git, llvm::StringRef repositoryRoot,
                 llvm::StringRef sourceRoot) {
  const llvm::SmallVector<llvm::StringRef, 8> arguments = {
      "-C",      repositoryRoot, "ls-files", "--error-unmatch",
      "--stage", "--",           sourceRoot};
  auto command = runGit(git, arguments);
  if (!command)
    return command.takeError();
  if (command->executionFailed || command->status < 0)
    return GitlinkIndexResult{GitUnavailable{}};
  if (command->status != 0)
    return invalid("Gitlink '" + sourceRoot +
                   "' is not an exact entry in the repository index: " +
                   llvm::StringRef(command->error).trim());

  llvm::StringRef record(command->output);
  record = record.trim();
  if (record.empty() || record.contains('\n'))
    return invalid("Gitlink '" + sourceRoot +
                   "' did not resolve to exactly one index entry");
  const auto fieldsAndPath = record.split('\t');
  if (fieldsAndPath.second != sourceRoot)
    return invalid("Gitlink index entry has a foreign path");
  llvm::StringRef fields = fieldsAndPath.first;
  if (!fields.consume_front("160000 "))
    return invalid("Gitlink '" + sourceRoot + "' does not have mode 160000");
  const auto hashAndStage = fields.split(' ');
  if (hashAndStage.second != "0" ||
      (hashAndStage.first.size() != 40 && hashAndStage.first.size() != 64) ||
      !lowercaseHex(hashAndStage.first))
    return invalid("Gitlink '" + sourceRoot +
                   "' has malformed commit or index stage metadata");
  return GitlinkIndexResult{GitlinkIndexEntry{hashAndStage.first.str()}};
}

bool isWithin(const std::filesystem::path &root,
              const std::filesystem::path &candidate) {
  const std::filesystem::path relative = candidate.lexically_relative(root);
  if (relative.empty() || relative.is_absolute())
    return false;
  const auto first = relative.begin();
  return first == relative.end() || *first != "..";
}

llvm::Expected<std::filesystem::path>
canonicalDirectory(const std::filesystem::path &path,
                   const std::filesystem::path &containmentRoot,
                   llvm::StringRef context) {
  std::error_code error;
  if (!std::filesystem::is_directory(path, error)) {
    if (error)
      return failed(context + " cannot be inspected: " + error.message());
    return invalid(context + " is not a directory");
  }
  std::filesystem::path canonical = std::filesystem::canonical(path, error);
  if (error)
    return failed(context + " cannot be canonicalized: " + error.message());
  if (!isWithin(containmentRoot, canonical))
    return invalid(context + " escapes its admitted root");
  return canonical;
}

llvm::Expected<std::filesystem::path>
canonicalFile(const std::filesystem::path &base, llvm::StringRef relative,
              const std::filesystem::path &containmentRoot,
              llvm::StringRef context) {
  const std::filesystem::path path = base / relative.str();
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error)) {
    if (error)
      return failed(context + " cannot be inspected: " + error.message());
    return invalid(context + " is not an existing regular file");
  }
  std::filesystem::path canonical = std::filesystem::canonical(path, error);
  if (error)
    return failed(context + " cannot be canonicalized: " + error.message());
  if (!isWithin(containmentRoot, canonical))
    return invalid(context + " escapes its admitted root");
  return canonical;
}

llvm::Error verifyFileDigest(const std::filesystem::path &path,
                             const BlobDigest &expected,
                             llvm::StringRef context) {
  auto buffer = llvm::MemoryBuffer::getFile(path.string(), false, false);
  if (!buffer)
    return failed(context + " cannot be read: " +
                  buffer.getError().message());
  const llvm::StringRef bytes = (*buffer)->getBuffer();
  const BlobDigest actual = computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size()));
  if (actual != expected)
    return invalid(context + " has digest " + formatBlobDigestHex(actual) +
                   " instead of " + formatBlobDigestHex(expected));
  return llvm::Error::success();
}

llvm::Expected<bool> pathExists(const std::filesystem::path &path,
                                llvm::StringRef context) {
  std::error_code error;
  const bool exists = std::filesystem::exists(path, error);
  if (error)
    return failed(context + " cannot be inspected: " + error.message());
  return exists;
}

const ApplicationDefinition *
findApplication(const ApplicationManifest &manifest, llvm::StringRef identity) {
  const auto applications = manifest.applications();
  const auto found = std::lower_bound(
      applications.begin(), applications.end(), identity,
      [](const ApplicationDefinition &application, llvm::StringRef key) {
        return application.identity < key;
      });
  if (found == applications.end() || found->identity != identity)
    return nullptr;
  return &*found;
}

using GitlinkCheckoutResult =
    std::variant<std::filesystem::path, GitUnavailable>;

/// A Gitlink checkout carries its own `.git` entry, a directory or a gitdir
/// file. Linked worktrees own no submodule checkouts and share the primary
/// worktree's, so a Gitlink without a checkout under `repositoryRoot`
/// resolves under the primary worktree Git reports for it; the resolved
/// checkout is still validated against this repository's index entry.
llvm::Expected<GitlinkCheckoutResult>
gitlinkCheckoutOwner(llvm::StringRef git,
                     const std::filesystem::path &repositoryRoot,
                     llvm::StringRef sourceRoot) {
  const auto hasCheckout = [&](const std::filesystem::path &root) {
    return pathExists(root / sourceRoot.str() / ".git", "Gitlink checkout");
  };
  auto local = hasCheckout(repositoryRoot);
  if (!local)
    return local.takeError();
  if (*local)
    return GitlinkCheckoutResult{repositoryRoot};
  const std::string repositoryText = repositoryRoot.string();
  const llvm::SmallVector<llvm::StringRef, 6> arguments = {
      "-C", repositoryText, "worktree", "list", "--porcelain"};
  auto worktrees = runGit(git, arguments);
  if (!worktrees)
    return worktrees.takeError();
  if (worktrees->executionFailed || worktrees->status != 0)
    return GitlinkCheckoutResult{GitUnavailable{}};
  llvm::StringRef primary =
      llvm::StringRef(worktrees->output).split('\n').first;
  if (!primary.consume_front("worktree "))
    return GitlinkCheckoutResult{GitUnavailable{}};
  std::error_code primaryError;
  const std::filesystem::path primaryRoot =
      std::filesystem::canonical(primary.trim().str(), primaryError);
  if (primaryError)
    return failed("primary worktree cannot be canonicalized: " +
                  primaryError.message());
  if (primaryRoot == repositoryRoot)
    return GitlinkCheckoutResult{GitUnavailable{}};
  auto shared = hasCheckout(primaryRoot);
  if (!shared)
    return shared.takeError();
  if (!*shared)
    return GitlinkCheckoutResult{GitUnavailable{}};
  return GitlinkCheckoutResult{primaryRoot};
}

llvm::Expected<GitlinkCheckoutResult>
validateGitlinkCheckout(llvm::StringRef git,
                        const std::filesystem::path &repositoryRoot,
                        const ApplicationDefinition &application,
                        llvm::StringRef expectedCommit) {
  auto owner =
      gitlinkCheckoutOwner(git, repositoryRoot, application.source.root);
  if (!owner)
    return owner.takeError();
  if (std::holds_alternative<GitUnavailable>(*owner))
    return GitlinkCheckoutResult{GitUnavailable{}};
  const std::filesystem::path &ownerRoot =
      std::get<std::filesystem::path>(*owner);
  auto canonical = canonicalDirectory(ownerRoot / application.source.root,
                                      ownerRoot, "Gitlink checkout");
  if (!canonical)
    return canonical.takeError();

  const std::string checkoutText = canonical->string();
  const llvm::SmallVector<llvm::StringRef, 6> headArguments = {
      "-C", checkoutText, "rev-parse", "--verify", "HEAD^{commit}"};
  auto head = runGit(git, headArguments);
  if (!head)
    return head.takeError();
  if (head->executionFailed || head->status < 0 || head->status != 0)
    return std::variant<std::filesystem::path, GitUnavailable>{
        GitUnavailable{}};
  const llvm::StringRef actualCommit = llvm::StringRef(head->output).trim();
  if (actualCommit != expectedCommit)
    return invalid("Gitlink checkout for '" + application.identity +
                   "' does not match the repository index commit");

  llvm::SmallVector<llvm::StringRef, 16> trackedArguments = {
      "-C", checkoutText, "ls-files", "--error-unmatch", "--"};
  for (const std::string &source : application.build.sources)
    trackedArguments.push_back(source);
  auto tracked = runGit(git, trackedArguments);
  if (!tracked)
    return tracked.takeError();
  if (tracked->executionFailed || tracked->status < 0)
    return std::variant<std::filesystem::path, GitUnavailable>{
        GitUnavailable{}};
  if (tracked->status != 0)
    return invalid("Gitlink source selection for '" + application.identity +
                   "' contains a file not owned by its pinned commit");

  llvm::SmallVector<llvm::StringRef, 16> cleanArguments = {
      "-C", checkoutText, "diff", "--quiet", "HEAD", "--"};
  for (const std::string &source : application.build.sources)
    cleanArguments.push_back(source);
  auto clean = runGit(git, cleanArguments);
  if (!clean)
    return clean.takeError();
  if (clean->executionFailed || clean->status < 0)
    return std::variant<std::filesystem::path, GitUnavailable>{
        GitUnavailable{}};
  if (clean->status == 1)
    return invalid("Gitlink source selection for '" + application.identity +
                   "' differs from its pinned commit");
  if (clean->status != 0)
    return invalid("Git could not validate the pinned source selection for '" +
                   application.identity + "'");
  return std::variant<std::filesystem::path, GitUnavailable>{
      std::move(*canonical)};
}

llvm::Expected<std::filesystem::path>
validateRepositorySource(const std::filesystem::path &repositoryRoot,
                         const ApplicationDefinition &application) {
  return canonicalDirectory(repositoryRoot / application.source.root,
                            repositoryRoot, "repository source root");
}

struct AdmittedApplicationFiles final {
  std::vector<std::string> sources;
  std::vector<AdmittedApplicationInput> inputs;
};

llvm::Expected<AdmittedApplicationFiles> resolveBuildAndOracleFiles(
    const std::filesystem::path &repositoryRoot,
    const std::filesystem::path &sourceRoot,
    const ApplicationDefinition &application,
    const WorkloadInputSelection *selectedInput = nullptr) {
  AdmittedApplicationFiles files;
  std::vector<std::filesystem::path> selectedSources;
  files.sources.reserve(application.build.sources.size());
  selectedSources.reserve(application.build.sources.size());
  for (const std::string &source : application.build.sources) {
    auto resolved =
        canonicalFile(sourceRoot, source, sourceRoot, "selected build source");
    if (!resolved)
      return resolved.takeError();
    files.sources.push_back(resolved->string());
    selectedSources.push_back(std::move(*resolved));
  }
  const auto validateOracle = [&](const WorkloadInputSelection &input)
      -> llvm::Expected<AdmittedApplicationInput> {
    auto oracle = canonicalFile(repositoryRoot, input.oracle.entry,
                                repositoryRoot, "oracle entry");
    if (!oracle)
      return oracle.takeError();
    if (llvm::Error error =
            verifyFileDigest(*oracle, input.oracle.digest, "oracle entry"))
      return std::move(error);
    if (llvm::is_contained(selectedSources, *oracle))
      return invalid("oracle entry for '" + application.identity +
                     "' is also a selected program source");
    return AdmittedApplicationInput{input.name, oracle->string(), {}};
  };
  if (selectedInput) {
    auto input = validateOracle(*selectedInput);
    if (!input)
      return input.takeError();
    files.inputs.push_back(std::move(*input));
    return files;
  }
  files.inputs.reserve(application.inputs.size());
  for (const WorkloadInputSelection &input : application.inputs) {
    auto admitted = validateOracle(input);
    if (!admitted)
      return admitted.takeError();
    files.inputs.push_back(std::move(*admitted));
  }
  return files;
}

using CachedInputAdmission = std::variant<std::vector<AdmittedCachedInput>,
                                          UnavailableApplicationSource>;

llvm::Expected<CachedInputAdmission>
validateCachedInputs(const std::filesystem::path &cacheRoot,
                     const ApplicationDefinition &application,
                     const WorkloadInputSelection *selectedInput = nullptr) {
  const auto validateInput = [&](const CachedInput &input)
      -> llvm::Expected<
          std::variant<AdmittedCachedInput, UnavailableApplicationSource>> {
    const std::filesystem::path path = cacheRoot / input.path;
    auto exists = pathExists(path, "cached input");
    if (!exists)
      return exists.takeError();
    if (!*exists)
      return UnavailableApplicationSource{SourceUnavailableReason::CachedInput,
                                          application.identity, input.path};
    auto canonical =
        canonicalFile(cacheRoot, input.path, cacheRoot, "cached input");
    if (!canonical)
      return canonical.takeError();
    auto buffer =
        llvm::MemoryBuffer::getFile(canonical->string(), false, false);
    if (!buffer)
      return failed("cannot read cached input '" + input.path +
                    "': " + buffer.getError().message());
    const llvm::StringRef bytes = (*buffer)->getBuffer();
    const BlobDigest actual = computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size()));
    if (actual != input.digest)
      return invalid("cached input '" + input.logicalName + "' for '" +
                     application.identity + "' has digest " +
                     formatBlobDigestHex(actual) + " instead of " +
                     formatBlobDigestHex(input.digest));
    return AdmittedCachedInput{input.logicalName, canonical->string()};
  };
  std::vector<AdmittedCachedInput> admitted;
  if (selectedInput) {
    admitted.reserve(selectedInput->cachedInputs.size());
    for (const std::string &logicalName : selectedInput->cachedInputs) {
      const auto found = llvm::find_if(
          application.cachedInputs, [&](const CachedInput &input) {
            return input.logicalName == logicalName;
          });
      if (found == application.cachedInputs.end())
        return invalid("input '" + selectedInput->name +
                       "' references unknown cached input '" + logicalName +
                       "'");
      auto unavailable = validateInput(*found);
      if (!unavailable)
        return unavailable.takeError();
      if (auto *missing =
              std::get_if<UnavailableApplicationSource>(&*unavailable))
        return std::move(*missing);
      admitted.push_back(
          std::get<AdmittedCachedInput>(std::move(*unavailable)));
    }
    return CachedInputAdmission{std::move(admitted)};
  }
  admitted.reserve(application.cachedInputs.size());
  for (const CachedInput &input : application.cachedInputs) {
    auto unavailable = validateInput(input);
    if (!unavailable)
      return unavailable.takeError();
    if (auto *missing =
            std::get_if<UnavailableApplicationSource>(&*unavailable))
      return std::move(*missing);
    admitted.push_back(std::get<AdmittedCachedInput>(std::move(*unavailable)));
  }
  return CachedInputAdmission{std::move(admitted)};
}

llvm::Error assignCachedInputs(
    const ApplicationDefinition &application,
    const WorkloadInputSelection *selectedInput,
    llvm::ArrayRef<AdmittedCachedInput> admittedCachedInputs,
    llvm::MutableArrayRef<AdmittedApplicationInput> admittedInputs) {
  const auto assign = [&](const WorkloadInputSelection &input,
                          AdmittedApplicationInput &admitted) -> llvm::Error {
    admitted.cachedInputs.reserve(input.cachedInputs.size());
    for (const std::string &logicalName : input.cachedInputs) {
      const auto found = llvm::find_if(
          admittedCachedInputs, [&](const AdmittedCachedInput &cached) {
            return cached.logicalName == logicalName;
          });
      if (found == admittedCachedInputs.end())
        return invalid("admitted cache projection lost logical input '" +
                       logicalName + "'");
      admitted.cachedInputs.push_back(*found);
    }
    return llvm::Error::success();
  };
  if (selectedInput) {
    if (admittedInputs.size() != 1 ||
        admittedInputs.front().inputName != selectedInput->name)
      return invalid("selected input admission changed identity");
    return assign(*selectedInput, admittedInputs.front());
  }
  if (admittedInputs.size() != application.inputs.size())
    return invalid("application input admission changed cardinality");
  for (auto [input, admitted] :
       llvm::zip_equal(application.inputs, admittedInputs))
    if (llvm::Error error = assign(input, admitted))
      return error;
  return llvm::Error::success();
}

} // namespace

llvm::StringRef toString(SourceUnavailableReason reason) {
  switch (reason) {
  case SourceUnavailableReason::GitExecutable:
    return "git_executable";
  case SourceUnavailableReason::GitlinkCheckout:
    return "gitlink_checkout";
  case SourceUnavailableReason::CacheRoot:
    return "cache_root";
  case SourceUnavailableReason::CachedInput:
    return "cached_input";
  }
  llvm_unreachable("unknown SourceUnavailableReason");
}

static llvm::Expected<std::vector<ApplicationSourceAdmissionOutcome>>
admitApplicationSourcesImpl(const ApplicationManifest &manifest,
                            llvm::ArrayRef<std::string> applicationIdentities,
                            llvm::StringRef repositoryRootText,
                            std::optional<llvm::StringRef> cacheRootText,
                            std::optional<llvm::StringRef> inputName) {
  if (applicationIdentities.empty())
    return invalid("explicit application selection is empty");
  for (std::size_t index = 1; index < applicationIdentities.size(); ++index)
    if (!(applicationIdentities[index - 1] < applicationIdentities[index]))
      return invalid(
          "explicit application selection is not canonical and unique");

  const std::filesystem::path repositoryInput(repositoryRootText.str());
  if (!repositoryInput.is_absolute())
    return invalid("repository root must be absolute");
  std::error_code repositoryError;
  if (!std::filesystem::is_directory(repositoryInput, repositoryError)) {
    if (repositoryError)
      return failed("repository root cannot be inspected: " +
                    repositoryError.message());
    return invalid("repository root is not an existing directory");
  }
  const std::filesystem::path repositoryRoot =
      std::filesystem::canonical(repositoryInput, repositoryError);
  if (repositoryError)
    return failed("repository root cannot be canonicalized: " +
                  repositoryError.message());

  bool requiresCache = false;
  std::vector<const ApplicationDefinition *> selected;
  selected.reserve(applicationIdentities.size());
  for (const std::string &identity : applicationIdentities) {
    const ApplicationDefinition *application =
        findApplication(manifest, identity);
    if (!application)
      return invalid("explicit selection names unknown application '" +
                     identity + "'");
    if (inputName) {
      const auto input = llvm::find_if(
          application->inputs, [&](const WorkloadInputSelection &candidate) {
            return candidate.name == *inputName;
          });
      if (input == application->inputs.end())
        return invalid("application '" + identity + "' has no input named '" +
                       *inputName + "'");
      requiresCache |= !input->cachedInputs.empty();
    } else {
      requiresCache |= !application->cachedInputs.empty();
    }
    selected.push_back(application);
  }

  std::optional<std::filesystem::path> cacheRoot;
  bool cacheUnavailable = false;
  if (requiresCache) {
    if (!cacheRootText) {
      cacheUnavailable = true;
    } else {
      const std::filesystem::path cacheInput(cacheRootText->str());
      if (!cacheInput.is_absolute())
        return invalid("cache root must be absolute");
      auto exists = pathExists(cacheInput, "cache root");
      if (!exists)
        return exists.takeError();
      if (!*exists) {
        cacheUnavailable = true;
      } else {
        std::error_code cacheError;
        if (!std::filesystem::is_directory(cacheInput, cacheError)) {
          if (cacheError)
            return failed("cache root cannot be inspected: " +
                          cacheError.message());
          return invalid("cache root is not a directory");
        }
        cacheRoot = std::filesystem::canonical(cacheInput, cacheError);
        if (cacheError)
          return failed("cache root cannot be canonicalized: " +
                        cacheError.message());
      }
    }
  }

  llvm::ErrorOr<std::string> git = llvm::sys::findProgramByName("git");
  std::vector<ApplicationSourceAdmissionOutcome> outcomes;
  outcomes.reserve(selected.size());
  for (const ApplicationDefinition *application : selected) {
    std::filesystem::path sourceRoot;
    if (application->source.kind == SourceKind::Gitlink) {
      if (!git) {
        outcomes.emplace_back(UnavailableApplicationSource{
            SourceUnavailableReason::GitExecutable, application->identity, {}});
        continue;
      }
      auto index = readGitlinkIndex(*git, repositoryRoot.string(),
                                    application->source.root);
      if (!index)
        return index.takeError();
      if (std::holds_alternative<GitUnavailable>(*index)) {
        outcomes.emplace_back(UnavailableApplicationSource{
            SourceUnavailableReason::GitExecutable, application->identity, {}});
        continue;
      }
      auto checkout =
          validateGitlinkCheckout(*git, repositoryRoot, *application,
                                  std::get<GitlinkIndexEntry>(*index).commit);
      if (!checkout)
        return checkout.takeError();
      if (std::holds_alternative<GitUnavailable>(*checkout)) {
        outcomes.emplace_back(UnavailableApplicationSource{
            SourceUnavailableReason::GitlinkCheckout, application->identity,
            application->source.root});
        continue;
      }
      sourceRoot = std::get<std::filesystem::path>(std::move(*checkout));
    } else {
      auto repositorySource =
          validateRepositorySource(repositoryRoot, *application);
      if (!repositorySource)
        return repositorySource.takeError();
      sourceRoot = std::move(*repositorySource);
    }

    const WorkloadInputSelection *selectedInput = nullptr;
    if (inputName) {
      const auto input = llvm::find_if(
          application->inputs, [&](const WorkloadInputSelection &candidate) {
            return candidate.name == *inputName;
          });
      selectedInput = &*input;
    }
    auto admittedFiles = resolveBuildAndOracleFiles(
        repositoryRoot, sourceRoot, *application, selectedInput);
    if (!admittedFiles)
      return admittedFiles.takeError();
    const bool selectedCache = selectedInput
                                   ? !selectedInput->cachedInputs.empty()
                                   : !application->cachedInputs.empty();
    if (selectedCache) {
      if (cacheUnavailable) {
        outcomes.emplace_back(UnavailableApplicationSource{
            SourceUnavailableReason::CacheRoot, application->identity,
            cacheRootText ? cacheRootText->str() : std::string{}});
        continue;
      }
      auto unavailable =
          validateCachedInputs(*cacheRoot, *application, selectedInput);
      if (!unavailable)
        return unavailable.takeError();
      if (auto *missing =
              std::get_if<UnavailableApplicationSource>(&*unavailable)) {
        outcomes.emplace_back(std::move(*missing));
        continue;
      }
      const auto &cachedInputs =
          std::get<std::vector<AdmittedCachedInput>>(*unavailable);
      if (llvm::Error error = assignCachedInputs(
              *application, selectedInput, cachedInputs, admittedFiles->inputs))
        return std::move(error);
    }
    outcomes.emplace_back(AdmittedApplicationSource{
        application->identity, repositoryRoot.generic_string(),
        sourceRoot.generic_string(), std::move(admittedFiles->sources),
        std::move(admittedFiles->inputs)});
  }
  return outcomes;
}

llvm::Expected<std::vector<ApplicationSourceAdmissionOutcome>>
admitApplicationSources(const ApplicationManifest &manifest,
                        llvm::ArrayRef<std::string> applicationIdentities,
                        llvm::StringRef repositoryRootText,
                        std::optional<llvm::StringRef> cacheRootText) {
  return admitApplicationSourcesImpl(manifest, applicationIdentities,
                                     repositoryRootText, cacheRootText,
                                     std::nullopt);
}

llvm::Expected<ApplicationSourceAdmissionOutcome> admitApplicationSource(
    const ApplicationManifest &manifest, llvm::StringRef applicationIdentity,
    llvm::StringRef inputName, llvm::StringRef repositoryRootText,
    std::optional<llvm::StringRef> cacheRootText) {
  auto outcomes =
      admitApplicationSourcesImpl(manifest, {applicationIdentity.str()},
                                  repositoryRootText, cacheRootText, inputName);
  if (!outcomes)
    return outcomes.takeError();
  if (outcomes->size() != 1)
    return invalid("selected application admission changed cardinality");
  return std::move(outcomes->front());
}

} // namespace loom::application
