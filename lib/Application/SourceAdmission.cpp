#include "Application/SourceAdmission.h"

#include "Common/BlobDigest.h"

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
  const int status =
      llvm::sys::ExecuteAndWait(executable, command, std::nullopt, redirects,
                                30, 256, &executionMessage, &executionFailed);
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

llvm::Expected<std::variant<std::filesystem::path, GitUnavailable>>
validateGitlinkCheckout(llvm::StringRef git,
                        const std::filesystem::path &repositoryRoot,
                        const ApplicationDefinition &application,
                        llvm::StringRef expectedCommit) {
  const std::filesystem::path checkout =
      repositoryRoot / application.source.root;
  auto exists = pathExists(checkout, "Gitlink checkout");
  if (!exists)
    return exists.takeError();
  if (!*exists)
    return std::variant<std::filesystem::path, GitUnavailable>{
        GitUnavailable{}};
  auto canonical =
      canonicalDirectory(checkout, repositoryRoot, "Gitlink checkout");
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

llvm::Error validateBuildAndOracles(const std::filesystem::path &repositoryRoot,
                                    const std::filesystem::path &sourceRoot,
                                    const ApplicationDefinition &application) {
  std::vector<std::filesystem::path> selectedSources;
  selectedSources.reserve(application.build.sources.size());
  for (const std::string &source : application.build.sources) {
    auto resolved =
        canonicalFile(sourceRoot, source, sourceRoot, "selected build source");
    if (!resolved)
      return resolved.takeError();
    selectedSources.push_back(std::move(*resolved));
  }
  for (const WorkloadInputSelection &input : application.inputs) {
    auto oracle = canonicalFile(repositoryRoot, input.oracle.entry,
                                repositoryRoot, "oracle entry");
    if (!oracle)
      return oracle.takeError();
    if (llvm::is_contained(selectedSources, *oracle))
      return invalid("oracle entry for '" + application.identity +
                     "' is also a selected program source");
  }
  return llvm::Error::success();
}

llvm::Expected<std::optional<UnavailableApplicationSource>>
validateCachedInputs(const std::filesystem::path &cacheRoot,
                     const ApplicationDefinition &application) {
  for (const CachedInput &input : application.cachedInputs) {
    const std::filesystem::path path = cacheRoot / input.path;
    auto exists = pathExists(path, "cached input");
    if (!exists)
      return exists.takeError();
    if (!*exists)
      return std::optional<UnavailableApplicationSource>{
          UnavailableApplicationSource{SourceUnavailableReason::CachedInput,
                                       application.identity, input.path}};
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
  }
  return std::optional<UnavailableApplicationSource>{};
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

llvm::Expected<std::vector<ApplicationSourceAdmissionOutcome>>
admitApplicationSources(const ApplicationManifest &manifest,
                        llvm::ArrayRef<std::string> applicationIdentities,
                        llvm::StringRef repositoryRootText,
                        std::optional<llvm::StringRef> cacheRootText) {
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
    requiresCache |= !application->cachedInputs.empty();
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

    if (llvm::Error error =
            validateBuildAndOracles(repositoryRoot, sourceRoot, *application))
      return std::move(error);
    if (!application->cachedInputs.empty()) {
      if (cacheUnavailable) {
        outcomes.emplace_back(UnavailableApplicationSource{
            SourceUnavailableReason::CacheRoot, application->identity,
            cacheRootText ? cacheRootText->str() : std::string{}});
        continue;
      }
      auto unavailable = validateCachedInputs(*cacheRoot, *application);
      if (!unavailable)
        return unavailable.takeError();
      if (*unavailable) {
        outcomes.emplace_back(std::move(**unavailable));
        continue;
      }
    }
    outcomes.emplace_back(AdmittedApplicationSource{
        application->identity, sourceRoot.generic_string()});
  }
  return outcomes;
}

} // namespace loom::application
