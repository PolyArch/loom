#include "ExternalTool/ExternalFile.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

struct IndexedFile final {
  std::string localKey;
  std::string path;
  ExternalFileFingerprint fingerprint;
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

llvm::Expected<FileDescriptor> openOrdinaryFile(llvm::StringRef spelling) {
  if (spelling.empty() || spelling.find('\0') != llvm::StringRef::npos)
    return invalid("external file path is empty or contains NUL");
  const std::filesystem::path path(spelling.str());
  if (!path.is_absolute() || path.lexically_normal() != path)
    return invalid("external file path must be an absolute canonical path");

  FileDescriptor current(::open("/", O_RDONLY | O_CLOEXEC | O_DIRECTORY));
  if (current.get() < 0)
    return systemError("could not open filesystem root");

  std::vector<std::string> components;
  for (const std::filesystem::path &component : path.relative_path()) {
    const std::string name = component.string();
    if (name.empty() || name == "." || name == "..")
      return invalid("external file path is not canonical");
    components.push_back(name);
  }
  if (components.empty())
    return invalid("external file path must name an ordinary file");

  for (std::size_t index = 0; index < components.size(); ++index) {
    const bool final = index + 1 == components.size();
    struct stat status {};
    if (::fstatat(current.get(), components[index].c_str(), &status,
                  AT_SYMLINK_NOFOLLOW) != 0)
      return systemError("could not inspect external file component '" +
                         components[index] + "'");
    if (S_ISLNK(status.st_mode))
      return invalid("external file path contains a symlink component");
    if (final && !S_ISREG(status.st_mode))
      return invalid("external file path must name an ordinary file");
    if (!final && !S_ISDIR(status.st_mode))
      return invalid("external file parent is not an ordinary directory");

    const int flags = final ? O_RDONLY | O_CLOEXEC | O_NOFOLLOW
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

llvm::Expected<ExternalFileFingerprint>
fingerprintFile(llvm::StringRef path) {
  auto file = openOrdinaryFile(path);
  if (!file)
    return file.takeError();
  struct stat before {};
  if (::fstat(file->get(), &before) != 0)
    return systemError("could not inspect opened external file");
  if (!S_ISREG(before.st_mode))
    return invalid("external file path must name an ordinary file");

  llvm::SHA256 hash;
  std::array<std::uint8_t, 64 * 1024> buffer{};
  while (true) {
    const ssize_t count = ::read(file->get(), buffer.data(), buffer.size());
    if (count == 0)
      break;
    if (count < 0) {
      if (errno == EINTR)
        continue;
      return systemError("could not read external file");
    }
    hash.update(llvm::ArrayRef<std::uint8_t>(
        buffer.data(), static_cast<std::size_t>(count)));
  }

  struct stat after {};
  if (::fstat(file->get(), &after) != 0)
    return systemError("could not re-inspect opened external file");
  if (!sameObservedFile(before, after))
    return invalid("external file changed while it was read");
  return ExternalFileFingerprint::fromBytes(hash.final());
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
ExternalFileFingerprint::fromBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != byteSize)
    return invalid("external file fingerprint requires exactly 32 bytes");
  Storage storage;
  std::copy(bytes.begin(), bytes.end(), storage.begin());
  return ExternalFileFingerprint(storage);
}

std::string
formatExternalFileFingerprint(const ExternalFileFingerprint &fingerprint) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(ExternalFileFingerprint::byteSize * 2);
  for (std::uint8_t byte : fingerprint.bytes()) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}

llvm::Expected<ExternalFileFingerprint>
parseExternalFileFingerprint(llvm::StringRef spelling) {
  if (spelling.size() != ExternalFileFingerprint::byteSize * 2)
    return invalid("external file fingerprint must use exactly 64 lowercase "
                   "hexadecimal characters");
  const auto nibble = [](char character) -> int {
    if (character >= '0' && character <= '9')
      return character - '0';
    if (character >= 'a' && character <= 'f')
      return character - 'a' + 10;
    return -1;
  };
  ExternalFileFingerprint::Storage bytes{};
  for (std::size_t index = 0; index < spelling.size(); index += 2) {
    const int high = nibble(spelling[index]);
    const int low = nibble(spelling[index + 1]);
    if (high < 0 || low < 0)
      return invalid("external file fingerprint must use lowercase "
                     "hexadecimal characters");
    bytes[index / 2] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return ExternalFileFingerprint::fromBytes(bytes);
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
    auto fingerprint = fingerprintFile(canonicalPath);
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
      return invalid("no configured external file matches provider input slot '" +
                     requirement.providerInputSlot + "' fingerprint " +
                     formatExternalFileFingerprint(requirement.fingerprint));
    const IndexedFile &selected = found->second.front();
    result.push_back(ResolvedExternalFile{
        requirement.providerInputSlot, selected.localKey, selected.path,
        requirement.fingerprint});
  }
  return result;
}

} // namespace loom::external_tool
