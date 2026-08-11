#include "ExternalTool/ExternalFile.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>

namespace loom::external_tool {
namespace {

class FileDescriptor final {
public:
  explicit FileDescriptor(int value = -1) : value_(value) {}
  FileDescriptor(const FileDescriptor &) = delete;
  FileDescriptor &operator=(const FileDescriptor &) = delete;
  FileDescriptor(FileDescriptor &&other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  FileDescriptor &operator=(FileDescriptor &&other) noexcept {
    if (this != &other) {
      if (value_ >= 0)
        ::close(value_);
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  ~FileDescriptor() {
    if (value_ >= 0)
      ::close(value_);
  }

  int get() const { return value_; }

private:
  int value_;
};

class DirectoryStream final {
public:
  explicit DirectoryStream(DIR *value) : value_(value) {}
  DirectoryStream(const DirectoryStream &) = delete;
  DirectoryStream &operator=(const DirectoryStream &) = delete;
  ~DirectoryStream() {
    if (value_)
      ::closedir(value_);
  }

  DIR *get() const { return value_; }

private:
  DIR *value_;
};

class DigestContext final {
public:
  DigestContext() : value_(EVP_MD_CTX_new()) {}
  DigestContext(const DigestContext &) = delete;
  DigestContext &operator=(const DigestContext &) = delete;
  ~DigestContext() { EVP_MD_CTX_free(value_); }

  EVP_MD_CTX *get() const { return value_; }

private:
  EVP_MD_CTX *value_;
};

struct IndexedFile final {
  std::string localKey;
  std::string path;
  ExternalFileFingerprint fingerprint;
};

struct IndexedFileTree final {
  std::string localKey;
  std::string path;
  std::vector<ExternalFileTreeMember> members;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "external_file_invalid: " + message);
}

llvm::Error systemError(const llvm::Twine &message) {
  return invalid(message + ": " + std::strerror(errno));
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

llvm::Expected<FileDescriptor> openAbsolutePath(llvm::StringRef spelling,
                                                bool directory) {
  if (spelling.empty() || spelling.find('\0') != llvm::StringRef::npos)
    return invalid("external path is empty or contains NUL");
  const std::filesystem::path path(spelling.str());
  if (!path.is_absolute() || path.lexically_normal() != path)
    return invalid("external path must be an absolute canonical path");

  FileDescriptor current(::open("/", O_RDONLY | O_CLOEXEC | O_DIRECTORY));
  if (current.get() < 0)
    return systemError("could not open filesystem root");

  std::vector<std::string> components;
  for (const std::filesystem::path &component : path.relative_path()) {
    const std::string name = component.string();
    if (name.empty() || name == "." || name == "..")
      return invalid("external path is not canonical");
    components.push_back(name);
  }
  if (components.empty())
    return invalid("external path must not name the filesystem root");

  for (std::size_t index = 0; index < components.size(); ++index) {
    const bool final = index + 1 == components.size();
    struct stat status{};
    if (::fstatat(current.get(), components[index].c_str(), &status,
                  AT_SYMLINK_NOFOLLOW) != 0)
      return systemError("could not inspect external file component '" +
                         components[index] + "'");
    if (S_ISLNK(status.st_mode))
      return invalid("external file path contains a symlink component");
    if (final && directory && !S_ISDIR(status.st_mode))
      return invalid("external file tree path must name an ordinary directory");
    if (final && !directory && !S_ISREG(status.st_mode))
      return invalid("external file path must name an ordinary file");
    if (!final && !S_ISDIR(status.st_mode))
      return invalid("external file parent is not an ordinary directory");

    const int flags = final && !directory
                          ? O_RDONLY | O_CLOEXEC | O_NOFOLLOW
                          : O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_DIRECTORY;
    FileDescriptor next(
        ::openat(current.get(), components[index].c_str(), flags));
    if (next.get() < 0)
      return systemError("could not open external file component '" +
                         components[index] + "'");
    current = std::move(next);
  }
  return current;
}

llvm::Expected<FileDescriptor> openOrdinaryFile(llvm::StringRef spelling) {
  return openAbsolutePath(spelling, false);
}

llvm::Expected<FileDescriptor> openOrdinaryDirectory(llvm::StringRef spelling) {
  return openAbsolutePath(spelling, true);
}

llvm::Expected<ExternalFileFingerprint>
fingerprintOpenFile(int descriptor, llvm::StringRef context) {
  struct stat before{};
  if (::fstat(descriptor, &before) != 0)
    return systemError("could not inspect " + context);
  if (!S_ISREG(before.st_mode))
    return invalid(context + " is not an ordinary file");

  DigestContext hash;
  if (!hash.get() || EVP_DigestInit_ex(hash.get(), EVP_sha256(), nullptr) != 1)
    return invalid("could not initialize the SHA-256 provider");
  std::array<std::uint8_t, 64 * 1024> buffer{};
  while (true) {
    const ssize_t count = ::read(descriptor, buffer.data(), buffer.size());
    if (count == 0)
      break;
    if (count < 0) {
      if (errno == EINTR)
        continue;
      return systemError("could not read " + context);
    }
    if (EVP_DigestUpdate(hash.get(), buffer.data(),
                         static_cast<std::size_t>(count)) != 1)
      return invalid("could not update the SHA-256 digest");
  }

  struct stat after{};
  if (::fstat(descriptor, &after) != 0)
    return systemError("could not re-inspect " + context);
  if (!sameObservedFile(before, after))
    return invalid(context + " changed while it was read");
  ExternalFileFingerprint::Storage digest{};
  unsigned digestSize = 0;
  if (EVP_DigestFinal_ex(hash.get(), digest.data(), &digestSize) != 1 ||
      digestSize != digest.size())
    return invalid("could not finalize the SHA-256 digest");
  return ExternalFileFingerprint::fromBytes(digest);
}

llvm::Error readFileTree(int directory, llvm::StringRef prefix,
                         std::vector<ExternalFileTreeMember> &members) {
  struct stat before{};
  if (::fstat(directory, &before) != 0)
    return systemError("could not inspect external file tree directory");
  if (!S_ISDIR(before.st_mode))
    return invalid("external file tree member is not a directory");

  const int duplicate = ::dup(directory);
  if (duplicate < 0)
    return systemError("could not duplicate external file tree directory");
  DirectoryStream stream(::fdopendir(duplicate));
  if (!stream.get()) {
    ::close(duplicate);
    return systemError("could not open external file tree directory stream");
  }
  std::vector<std::string> names;
  errno = 0;
  while (dirent *entry = ::readdir(stream.get())) {
    const llvm::StringRef name(entry->d_name);
    if (name == "." || name == "..")
      continue;
    names.push_back(name.str());
  }
  if (errno != 0)
    return systemError("could not enumerate external file tree directory");
  llvm::sort(names);

  for (const std::string &name : names) {
    struct stat status{};
    if (::fstatat(directory, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0)
      return systemError("could not inspect external file tree member '" +
                         name + "'");
    if (S_ISLNK(status.st_mode))
      return invalid("external file tree contains a symlink");
    const std::string relative =
        prefix.empty() ? name : (prefix + "/" + name).str();
    if (S_ISDIR(status.st_mode)) {
      FileDescriptor child(
          ::openat(directory, name.c_str(),
                   O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_DIRECTORY));
      if (child.get() < 0)
        return systemError("could not open external file tree directory '" +
                           relative + "'");
      if (llvm::Error error = readFileTree(child.get(), relative, members))
        return error;
      continue;
    }
    if (!S_ISREG(status.st_mode))
      return invalid("external file tree contains a special file");
    FileDescriptor file(
        ::openat(directory, name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (file.get() < 0)
      return systemError("could not open external file tree member '" +
                         relative + "'");
    auto fingerprint = fingerprintOpenFile(
        file.get(), "external file tree member '" + relative + "'");
    if (!fingerprint)
      return fingerprint.takeError();
    members.push_back({relative, std::move(*fingerprint)});
  }

  struct stat after{};
  if (::fstat(directory, &after) != 0)
    return systemError("could not re-inspect external file tree directory");
  if (!sameObservedFile(before, after))
    return invalid("external file tree changed while it was read");
  return llvm::Error::success();
}

llvm::Expected<std::vector<ExternalFileTreeMember>>
fingerprintFileTree(llvm::StringRef path) {
  auto directory = openOrdinaryDirectory(path);
  if (!directory)
    return directory.takeError();
  std::vector<ExternalFileTreeMember> members;
  if (llvm::Error error = readFileTree(directory->get(), {}, members))
    return std::move(error);
  if (members.empty())
    return invalid("external file tree must contain an ordinary file");
  llvm::sort(members, [](const ExternalFileTreeMember &lhs,
                         const ExternalFileTreeMember &rhs) {
    return lhs.relativePath < rhs.relativePath;
  });
  return members;
}

llvm::Error
validateFileTreeMembers(llvm::ArrayRef<ExternalFileTreeMember> members) {
  if (members.empty())
    return invalid("external file tree requirement must contain a member");
  std::string previous;
  for (const ExternalFileTreeMember &member : members) {
    const std::filesystem::path path(member.relativePath);
    bool containsParent = false;
    for (const std::filesystem::path &component : path)
      containsParent |= component == "..";
    if (member.relativePath.empty() ||
        member.relativePath.find('\0') != std::string::npos ||
        path.is_absolute() || path.lexically_normal() != path || path == "." ||
        containsParent)
      return invalid(
          "external file tree member must have a canonical relative path");
    if (!previous.empty() && previous == member.relativePath)
      return invalid(
          "external file tree requirement has a duplicate member path");
    if (!previous.empty() && previous > member.relativePath)
      return invalid(
          "external file tree members must be sorted by relative path");
    previous = member.relativePath;
  }
  return llvm::Error::success();
}

llvm::Error validateLocalKey(llvm::StringRef key) {
  if (key.empty())
    return invalid("external file local key must be nonempty");
  if (key.find('\0') != llvm::StringRef::npos)
    return invalid("external file local key contains NUL");
  return llvm::Error::success();
}

llvm::Error validateProviderSlot(llvm::StringRef slot) {
  if (slot.empty())
    return invalid("provider input slot must be nonempty");
  if (slot.find('\0') != llvm::StringRef::npos)
    return invalid("provider input slot contains NUL");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<ExternalFileFingerprint>
fingerprintExternalFile(llvm::StringRef path) {
  auto file = openOrdinaryFile(path);
  if (!file)
    return file.takeError();
  return fingerprintOpenFile(file->get(), "external file");
}

llvm::Error validateExternalFileTreeRequirement(
    const ExternalFileTreeRequirement &requirement) {
  if (llvm::Error error = validateProviderSlot(requirement.providerInputSlot))
    return error;
  return validateFileTreeMembers(requirement.members);
}

llvm::Expected<std::vector<ResolvedExternalFile>>
resolveExternalFiles(llvm::ArrayRef<ExternalFileRequirement> requirements,
                     const LocalToolConfig &config) {
  std::set<std::string> slots;
  for (const ExternalFileRequirement &requirement : requirements) {
    if (llvm::Error error = validateProviderSlot(requirement.providerInputSlot))
      return std::move(error);
    if (!slots.insert(requirement.providerInputSlot).second)
      return invalid("duplicate provider input slot '" +
                     requirement.providerInputSlot + "'");
  }

  std::set<std::string> paths;
  std::map<ExternalFileFingerprint::Storage, std::vector<IndexedFile>> index;
  for (const auto &[localKey, configuredPath] : config.externalFiles) {
    if (llvm::Error error = validateLocalKey(localKey))
      return std::move(error);
    const std::filesystem::path path(configuredPath);
    if (!path.is_absolute() || path.lexically_normal() != path)
      return invalid("external file path must be an absolute canonical path");
    const std::string canonicalPath = path.string();
    if (!paths.insert(canonicalPath).second)
      return invalid("external file map contains a duplicate canonical path");
    auto fingerprint = fingerprintExternalFile(canonicalPath);
    if (!fingerprint)
      return fingerprint.takeError();
    index[fingerprint->bytes()].push_back(
        IndexedFile{localKey, canonicalPath, *fingerprint});
  }

  for (auto &[fingerprint, files] : index)
    llvm::sort(files, [](const IndexedFile &lhs, const IndexedFile &rhs) {
      return std::tie(lhs.path, lhs.localKey) <
             std::tie(rhs.path, rhs.localKey);
    });

  std::vector<ResolvedExternalFile> result;
  result.reserve(requirements.size());
  for (const ExternalFileRequirement &requirement : requirements) {
    const auto found = index.find(requirement.fingerprint.bytes());
    if (found == index.end())
      return invalid(
          "no configured external file matches provider input slot '" +
          requirement.providerInputSlot + "' fingerprint " +
          formatExternalFileFingerprint(requirement.fingerprint));
    const IndexedFile &selected = found->second.front();
    result.push_back(ResolvedExternalFile{requirement.providerInputSlot,
                                          selected.localKey, selected.path,
                                          requirement.fingerprint});
  }
  return result;
}

llvm::Expected<std::vector<ResolvedExternalFileTree>> resolveExternalFileTrees(
    llvm::ArrayRef<ExternalFileTreeRequirement> requirements,
    const LocalToolConfig &config) {
  std::set<std::string> slots;
  for (const ExternalFileTreeRequirement &requirement : requirements) {
    if (llvm::Error error = validateExternalFileTreeRequirement(requirement))
      return error;
    if (!slots.insert(requirement.providerInputSlot).second)
      return invalid("duplicate provider input slot '" +
                     requirement.providerInputSlot + "'");
  }

  std::set<std::string> paths;
  std::vector<IndexedFileTree> index;
  index.reserve(config.externalFileTrees.size());
  for (const auto &[localKey, configuredPath] : config.externalFileTrees) {
    if (llvm::Error error = validateLocalKey(localKey))
      return std::move(error);
    const std::filesystem::path path(configuredPath);
    if (!path.is_absolute() || path.lexically_normal() != path)
      return invalid(
          "external file tree path must be an absolute canonical path");
    const std::string canonicalPath = path.string();
    if (!paths.insert(canonicalPath).second)
      return invalid(
          "external file tree map contains a duplicate canonical path");
    auto members = fingerprintFileTree(canonicalPath);
    if (!members)
      return members.takeError();
    index.push_back(
        IndexedFileTree{localKey, canonicalPath, std::move(*members)});
  }
  llvm::sort(index, [](const IndexedFileTree &lhs, const IndexedFileTree &rhs) {
    return std::tie(lhs.path, lhs.localKey) < std::tie(rhs.path, rhs.localKey);
  });

  std::vector<ResolvedExternalFileTree> result;
  result.reserve(requirements.size());
  for (const ExternalFileTreeRequirement &requirement : requirements) {
    const auto found = llvm::find_if(index, [&](const IndexedFileTree &tree) {
      return tree.members == requirement.members;
    });
    if (found == index.end())
      return invalid(
          "no configured external file tree matches provider input slot '" +
          requirement.providerInputSlot + "'");
    result.push_back(ResolvedExternalFileTree{requirement.providerInputSlot,
                                              found->localKey, found->path,
                                              requirement.members});
  }
  return result;
}

} // namespace loom::external_tool
