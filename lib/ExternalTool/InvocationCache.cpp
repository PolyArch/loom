#include "ExternalTool/InvocationBundle.h"

#include "InvocationBundleInternal.h"

#include "Common/ArtifactText.h"
#include "Common/BlobDigest.h"
#include "Common/DiagnosticVerbosity.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <optional>
#include <set>
#include <signal.h>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace loom::external_tool {
namespace {

constexpr llvm::StringLiteral kCacheRootEnvironment =
    "LOOM_EXTERNAL_TOOL_CACHE_ROOT";
constexpr llvm::StringLiteral kCacheMarkerName =
    ".loom-external-tool-result-cache";
constexpr llvm::StringLiteral kCacheMarkerContents =
    "loom.external_tool_result_cache 1.0\n";
constexpr llvm::StringLiteral kCacheInitializationLockName =
    ".loom-external-tool-result-cache.lock";
constexpr llvm::StringLiteral kCacheEntryName = "entry.json";
constexpr llvm::StringLiteral kCacheEntrySchema =
    "loom.external_tool_result_cache_entry";
constexpr llvm::StringLiteral kCacheEntryVersion = "1.2";

llvm::Error cacheError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "external_tool_cache_invalid: " + message);
}

llvm::Error cacheSystemError(const llvm::Twine &message) {
  return cacheError(message + ": " + std::strerror(errno));
}

void cacheDiagnostic(DiagnosticVerbosity level, llvm::StringRef event,
                     llvm::StringRef detail = {}) {
  if (!diagnosticVerbosityEnabled(level))
    return;
  llvm::errs() << "[loom.external-tool-cache] " << event;
  if (!detail.empty())
    llvm::errs() << ' ' << detail;
  llvm::errs() << '\n';
}

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

class CacheLock final {
public:
  explicit CacheLock(FileDescriptor descriptor)
      : descriptor_(std::move(descriptor)) {}
  CacheLock(const CacheLock &) = delete;
  CacheLock &operator=(const CacheLock &) = delete;
  CacheLock(CacheLock &&) = default;

private:
  FileDescriptor descriptor_;
};

struct CacheLockAcquisition final {
  CacheLock lock;
  bool waited;
};

struct CacheFile final {
  std::string relativePath;
  BlobDigest digest;
};

struct CacheEntry final {
  ExternalToolResultCacheKey key;
  std::vector<CacheFile> files;
};

struct CacheRootResolution final {
  std::optional<std::filesystem::path> root;
  bool executionStopped = false;
};

struct ScriptExecution final {
  int exitCode = 0;
  bool invoked = false;
};

bool waitForExecutionControl(ExecutionControlView executionControl) {
  if (executionControl.stopRequested())
    return false;
  auto delay = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::milliseconds(10));
  if (auto remaining = executionControl.remainingTime()) {
    if (*remaining <= std::chrono::steady_clock::duration::zero())
      return false;
    delay = std::min(
        delay,
        std::chrono::duration_cast<std::chrono::nanoseconds>(*remaining));
  }
  if (delay > std::chrono::steady_clock::duration::zero())
    std::this_thread::sleep_for(delay);
  return !executionControl.stopRequested();
}

llvm::Expected<bool> acquireExclusiveLock(int descriptor,
                                          ExecutionControlView executionControl,
                                          bool &waited) {
  while (::flock(descriptor, LOCK_EX | LOCK_NB) != 0) {
    if (errno == EINTR)
      continue;
    if (errno != EWOULDBLOCK && errno != EAGAIN)
      return cacheSystemError("cannot acquire cache lock");
    waited = true;
    if (!waitForExecutionControl(executionControl))
      return false;
  }
  return true;
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendBytes(std::vector<std::uint8_t> &bytes, llvm::StringRef value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.bytes_begin(), value.bytes_end());
}

BlobDigest domainDigest(llvm::StringRef domain, llvm::StringRef canonical) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(domain.size() + canonical.size() + 9);
  bytes.insert(bytes.end(), domain.bytes_begin(), domain.bytes_end());
  bytes.push_back(0);
  appendBytes(bytes, canonical);
  return computeBlobDigest(bytes);
}

void writeStringArray(llvm::json::OStream &json,
                      llvm::ArrayRef<std::string> values) {
  json.array([&] {
    for (const std::string &value : values)
      json.value(value);
  });
}

void writeSemanticClosure(llvm::json::OStream &json,
                          const SemanticInvocationClosure &closure) {
  json.object([&] {
    if (const auto *generator =
            std::get_if<CandidateGeneratorInvocationClosure>(&closure)) {
      json.attribute("form", "candidate_generator");
      json.attribute("typed_input_bindings",
                     llvm::toHex(generator->typedInputBindings, true));
      json.attribute("resolved_binding",
                     llvm::toHex(generator->resolvedBinding, true));
      json.attribute("binding_identity",
                     llvm::toHex(generator->bindingIdentity, true));
      return;
    }
    json.attribute("form", "evaluation");
    json.attributeBegin("request");
    writeArtifactRootReferenceJson(json,
                                   std::get<ArtifactRootReference>(closure));
    json.attributeEnd();
  });
}

using PathToken = std::pair<std::string, std::string>;

std::vector<PathToken>
pathTokens(const PreparedExternalToolInvocation &prepared,
           const InvocationManifestData &manifest) {
  std::vector<PathToken> result;
  auto add = [&](llvm::StringRef path, llvm::StringRef token) {
    if (!path.empty())
      result.emplace_back(path.str(), token.str());
  };
  add(prepared.bundleRoot, "${loom.bundle_root}");
  add(manifest.tool.executable, "${loom.tool_executable}");
  if (manifest.runtime.polyArchContainer)
    add(manifest.runtime.polyArchContainer->executable,
        "${loom.container_executable}");
  for (const ResolvedExternalFile &file : manifest.externalFiles)
    add(file.absolutePath,
        "${loom.external_file:" + file.providerInputSlot + "}");
  for (const ResolvedExternalFileTree &tree : manifest.externalFileTrees)
    add(tree.absolutePath,
        "${loom.external_tree:" + tree.providerInputSlot + "}");
  llvm::sort(result, [](const PathToken &lhs, const PathToken &rhs) {
    if (lhs.first.size() != rhs.first.size())
      return lhs.first.size() > rhs.first.size();
    return lhs < rhs;
  });
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

void writeNormalizedLocalPathSegments(llvm::json::OStream &json,
                                      llvm::StringRef value,
                                      llvm::ArrayRef<PathToken> tokens) {
  auto writeLiteral = [&](llvm::StringRef literal) {
    json.object([&] {
      json.attribute("literal_size", static_cast<std::int64_t>(literal.size()));
      json.attribute("literal_sha256",
                     formatBlobDigestHex(contentDigest(literal)));
    });
  };
  json.array([&] {
    std::size_t offset = 0;
    while (offset != value.size()) {
      std::size_t matchOffset = llvm::StringRef::npos;
      const PathToken *match = nullptr;
      for (const PathToken &candidate : tokens) {
        const std::size_t candidateOffset = value.find(candidate.first, offset);
        if (candidateOffset == llvm::StringRef::npos)
          continue;
        if (!match || candidateOffset < matchOffset ||
            (candidateOffset == matchOffset &&
             candidate.first.size() > match->first.size())) {
          matchOffset = candidateOffset;
          match = &candidate;
        }
      }
      if (!match) {
        writeLiteral(value.drop_front(offset));
        break;
      }
      if (matchOffset != offset)
        writeLiteral(value.slice(offset, matchOffset));
      json.object([&] { json.attribute("path_token", match->second); });
      offset = matchOffset + match->first.size();
    }
  });
}

llvm::Expected<std::string>
readOrdinaryFile(const std::filesystem::path &path) {
  struct stat before{};
  if (::lstat(path.c_str(), &before) != 0)
    return cacheSystemError("cannot inspect cache-bearing file");
  if (!S_ISREG(before.st_mode) || before.st_size < 0)
    return cacheError("cache-bearing path is not an ordinary file");
  FileDescriptor file(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (file.get() < 0)
    return cacheSystemError("cannot open cache-bearing file");
  std::string contents;
  contents.reserve(static_cast<std::size_t>(before.st_size));
  std::array<char, 64 * 1024> buffer{};
  while (true) {
    const ssize_t amount = ::read(file.get(), buffer.data(), buffer.size());
    if (amount == 0)
      break;
    if (amount < 0) {
      if (errno == EINTR)
        continue;
      return cacheSystemError("cannot read cache-bearing file");
    }
    contents.append(buffer.data(), static_cast<std::size_t>(amount));
  }
  struct stat after{};
  if (::fstat(file.get(), &after) != 0)
    return cacheSystemError("cannot re-inspect cache-bearing file");
  if (before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
      before.st_mode != after.st_mode || before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
      before.st_ctim.tv_nsec != after.st_ctim.tv_nsec ||
      contents.size() != static_cast<std::uintmax_t>(after.st_size))
    return cacheError("cache-bearing file changed while it was read");
  return contents;
}

llvm::Expected<BlobDigest>
digestOrdinaryFile(const std::filesystem::path &path) {
  struct stat before{};
  if (::lstat(path.c_str(), &before) != 0)
    return cacheSystemError("cannot inspect cache payload");
  if (!S_ISREG(before.st_mode) || before.st_size < 0)
    return cacheError("cache payload is not an ordinary file");
  FileDescriptor file(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (file.get() < 0)
    return cacheSystemError("cannot open cache payload");
  llvm::SHA256 digest;
  std::uintmax_t observedSize = 0;
  std::array<std::uint8_t, 64 * 1024> buffer{};
  while (true) {
    const ssize_t amount = ::read(file.get(), buffer.data(), buffer.size());
    if (amount == 0)
      break;
    if (amount < 0) {
      if (errno == EINTR)
        continue;
      return cacheSystemError("cannot read cache payload");
    }
    digest.update(llvm::ArrayRef<std::uint8_t>(
        buffer.data(), static_cast<std::size_t>(amount)));
    observedSize += static_cast<std::size_t>(amount);
  }
  struct stat after{};
  if (::fstat(file.get(), &after) != 0)
    return cacheSystemError("cannot re-inspect cache payload");
  if (before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
      before.st_mode != after.st_mode || before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
      before.st_ctim.tv_nsec != after.st_ctim.tv_nsec ||
      observedSize != static_cast<std::uintmax_t>(after.st_size))
    return cacheError("cache payload changed while it was hashed");
  return BlobDigest::fromBytes(digest.final());
}

llvm::Expected<std::string>
canonicalInputMaterial(const InvocationManifestData &manifest) {
  llvm::SmallString<4096> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output, 0);
  json.object([&] {
    json.attributeArray("semantic_inputs", [&] {
      for (const ManifestMaterializedFile &file : manifest.materializedFiles) {
        if (!file.sourceArtifact)
          continue;
        json.object([&] {
          json.attribute("path", file.relativePath);
          json.attributeBegin("source_artifact");
          writeArtifactRootReferenceJson(json, *file.sourceArtifact);
          json.attributeEnd();
          json.attribute("content_sha256",
                         formatBlobDigestHex(file.contentDigest));
        });
      }
    });
    json.attributeArray("external_files", [&] {
      for (const ResolvedExternalFile &file : manifest.externalFiles)
        json.object([&] {
          json.attribute("slot", file.providerInputSlot);
          json.attribute("content_sha256",
                         formatExternalFileFingerprint(file.fingerprint));
        });
    });
    json.attributeArray("external_file_trees", [&] {
      for (const ResolvedExternalFileTree &tree : manifest.externalFileTrees)
        json.object([&] {
          json.attribute("slot", tree.providerInputSlot);
          json.attributeArray("members", [&] {
            for (const ExternalFileTreeMember &member : tree.members)
              json.object([&] {
                json.attribute("path", member.relativePath);
                json.attribute("content_sha256", formatExternalFileFingerprint(
                                                     member.fingerprint));
              });
          });
        });
    });
  });
  return output.str().str();
}

llvm::Expected<std::string>
canonicalExecutionConfiguration(const PreparedExternalToolInvocation &prepared,
                                const InvocationManifestData &manifest) {
  const std::vector<PathToken> tokens = pathTokens(prepared, manifest);
  llvm::SmallString<8192> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output, 0);
  llvm::Error generatedFileError = llvm::Error::success();
  json.object([&] {
    json.attribute("provider_identity",
                   manifest.semanticContract.providerIdentity);
    json.attributeBegin("semantic_closure");
    writeSemanticClosure(json, manifest.semanticContract.semanticClosure);
    json.attributeEnd();
    json.attribute("result_importer_identity",
                   manifest.semanticContract.resultImporterIdentity);
    const std::set<std::string> producedExecutables(
        manifest.toolProducedExecutables.begin(),
        manifest.toolProducedExecutables.end());
    json.attributeArray("commands", [&] {
      for (const std::vector<std::string> &command : manifest.commands)
        json.array([&] {
          for (const auto &[index, argument] : llvm::enumerate(command))
            if (!(producedExecutables.count(command.front()) != 0 &&
                  index + 1 == command.size() &&
                  isDiagnosticVerbosityArgument(argument)))
              writeNormalizedLocalPathSegments(json, argument, tokens);
        });
    });
    json.attributeBegin("inherit_environment");
    writeStringArray(json, manifest.inheritEnvironment);
    json.attributeEnd();
    json.attributeArray("generated_files", [&] {
      for (const ManifestMaterializedFile &file : manifest.materializedFiles) {
        if (file.sourceArtifact)
          continue;
        auto contents = readOrdinaryFile(
            std::filesystem::path(prepared.bundleRoot) / file.relativePath);
        if (!contents) {
          generatedFileError = llvm::joinErrors(std::move(generatedFileError),
                                                contents.takeError());
          continue;
        }
        json.object([&] {
          json.attribute("path", file.relativePath);
          json.attribute("executable", file.executable);
          json.attributeBegin("contents");
          writeNormalizedLocalPathSegments(json, *contents, tokens);
          json.attributeEnd();
        });
      }
    });
    json.attributeBegin("declared_outputs");
    writeStringArray(json, manifest.declaredOutputs);
    json.attributeEnd();
    json.attributeBegin("tool_produced_executables");
    writeStringArray(json, manifest.toolProducedExecutables);
    json.attributeEnd();
  });
  if (generatedFileError)
    return std::move(generatedFileError);
  return output.str().str();
}

llvm::Expected<std::string>
canonicalToolVersion(const InvocationManifestData &manifest) {
  auto toolDigest = digestOrdinaryFile(manifest.tool.executable);
  if (!toolDigest)
    return toolDigest.takeError();
  std::optional<BlobDigest> containerDigest;
  if (manifest.runtime.polyArchContainer) {
    auto digest =
        digestOrdinaryFile(manifest.runtime.polyArchContainer->executable);
    if (!digest)
      return digest.takeError();
    containerDigest = *digest;
  }
  llvm::SmallString<2048> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output, 0);
  json.object([&] {
    json.attribute("tool_key", manifest.tool.toolKey);
    json.attribute("tool_version", manifest.tool.version);
    json.attribute("tool_executable_sha256", formatBlobDigestHex(*toolDigest));
    json.attributeBegin("tool_version_probe");
    writeToolVersionProbeJson(json, manifest.toolVersionProbe);
    json.attributeEnd();
    json.attribute("runtime",
                   manifest.runtime.kind == InvocationRuntimeKind::Host
                       ? "host"
                       : "polyarch_container");
    if (manifest.runtime.polyArchContainer) {
      json.attribute("container_key",
                     manifest.runtime.polyArchContainer->toolKey);
      json.attribute("container_version",
                     manifest.runtime.polyArchContainer->version);
      json.attribute("container_executable_sha256",
                     formatBlobDigestHex(*containerDigest));
      json.attribute("container_os", *manifest.runtime.os);
      json.attributeBegin("container_version_probe");
      writeToolVersionProbeJson(json, manifest.containerVersionProbe);
      json.attributeEnd();
    }
  });
  return output.str().str();
}

std::string keyText(const ExternalToolResultCacheKey &key) {
  return formatBlobDigestHex(key.inputMaterialDigest) + "/" +
         formatBlobDigestHex(key.executionConfigurationDigest) + "/" +
         formatBlobDigestHex(key.toolVersionDigest);
}

llvm::Expected<CacheRootResolution>
cacheRoot(ExecutionControlView executionControl) {
  const char *value = std::getenv(kCacheRootEnvironment.str().c_str());
  if (!value || !*value)
    return CacheRootResolution{};
  std::filesystem::path root(value);
  if (!root.is_absolute())
    return cacheError("cache root must be absolute");
  root = root.lexically_normal();
  std::error_code error;
  const bool existed = std::filesystem::exists(root, error);
  if (error)
    return cacheError("cannot inspect cache root: " + error.message());
  if (!existed) {
    std::filesystem::create_directories(root, error);
    if (error)
      return cacheError("cannot create cache root: " + error.message());
  }
  const auto status = std::filesystem::symlink_status(root, error);
  if (error || !std::filesystem::is_directory(status) ||
      std::filesystem::is_symlink(status))
    return cacheError("cache root is not an ordinary directory");
  struct stat rootStatus{};
  if (::lstat(root.c_str(), &rootStatus) != 0)
    return cacheSystemError("cannot inspect cache-root ownership");
  if (rootStatus.st_uid != ::geteuid())
    return cacheError("cache root is not owned by the current user");
  if (::chmod(root.c_str(), S_IRWXU) != 0)
    return cacheSystemError("cannot make cache root private");

  FileDescriptor initializationLock(
      ::open((root / kCacheInitializationLockName.str()).c_str(),
             O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600));
  if (initializationLock.get() < 0)
    return cacheSystemError("cannot open cache initialization lock");
  bool waited = false;
  auto locked =
      acquireExclusiveLock(initializationLock.get(), executionControl, waited);
  if (!locked)
    return locked.takeError();
  if (!*locked)
    return CacheRootResolution{std::nullopt, true};

  const std::filesystem::path marker = root / kCacheMarkerName.str();
  if (!std::filesystem::exists(marker, error)) {
    if (error)
      return cacheError("cannot inspect cache marker: " + error.message());
    for (const auto &entry : std::filesystem::directory_iterator(root, error)) {
      if (error)
        return cacheError("cannot inspect unmarked cache root: " +
                          error.message());
      if (entry.path().filename() != kCacheInitializationLockName.str())
        return cacheError("unmarked cache root is not empty");
    }
    const int descriptor =
        ::open(marker.c_str(),
               O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (descriptor >= 0) {
      FileDescriptor file(descriptor);
      llvm::StringRef remaining = kCacheMarkerContents;
      while (!remaining.empty()) {
        const ssize_t amount =
            ::write(file.get(), remaining.data(), remaining.size());
        if (amount < 0) {
          if (errno == EINTR)
            continue;
          return cacheSystemError("cannot write cache marker");
        }
        remaining = remaining.drop_front(static_cast<std::size_t>(amount));
      }
      if (::fsync(file.get()) != 0)
        return cacheSystemError("cannot sync cache marker");
    } else if (errno != EEXIST) {
      return cacheSystemError("cannot create cache marker");
    }
  }
  auto markerContents = readOrdinaryFile(marker);
  if (!markerContents)
    return markerContents.takeError();
  if (*markerContents != kCacheMarkerContents)
    return cacheError("cache root has an incompatible format marker");
  const std::set<std::string> allowedMembers{kCacheMarkerName.str(),
                                             kCacheInitializationLockName.str(),
                                             "command-entries",
                                             "command-locks",
                                             "entries",
                                             "locks"};
  for (const auto &entry : std::filesystem::directory_iterator(root, error)) {
    if (error)
      return cacheError("cannot inspect cache namespace: " + error.message());
    if (!allowedMembers.count(entry.path().filename().string()))
      return cacheError("cache root contains a foreign top-level member");
  }
  std::filesystem::create_directories(root / "entries", error);
  std::filesystem::create_directories(root / "locks", error);
  if (error)
    return cacheError("cannot create cache namespaces: " + error.message());
  for (llvm::StringRef name :
       {llvm::StringRef("entries"), llvm::StringRef("locks")}) {
    const std::filesystem::path namespacePath = root / name.str();
    const auto namespaceStatus =
        std::filesystem::symlink_status(namespacePath, error);
    if (error || !std::filesystem::is_directory(namespaceStatus) ||
        std::filesystem::is_symlink(namespaceStatus))
      return cacheError("cache namespace is not an ordinary directory");
    struct stat namespaceOwnership{};
    if (::lstat(namespacePath.c_str(), &namespaceOwnership) != 0)
      return cacheSystemError("cannot inspect cache-namespace ownership");
    if (namespaceOwnership.st_uid != ::geteuid())
      return cacheError("cache namespace is not owned by the current user");
    if (::chmod(namespacePath.c_str(), S_IRWXU) != 0)
      return cacheSystemError("cannot make cache namespace private");
  }
  return CacheRootResolution{root, false};
}

llvm::Expected<std::optional<CacheLockAcquisition>>
lockKey(const std::filesystem::path &root,
        const ExternalToolResultCacheKey &key,
        ExecutionControlView executionControl) {
  const std::string name =
      formatBlobDigestHex(key.inputMaterialDigest) + "." +
      formatBlobDigestHex(key.executionConfigurationDigest) + "." +
      formatBlobDigestHex(key.toolVersionDigest) + ".lock";
  FileDescriptor descriptor(::open((root / "locks" / name).c_str(),
                                   O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW,
                                   0600));
  if (descriptor.get() < 0)
    return cacheSystemError("cannot open cache-key lock");
  bool waited = false;
  auto locked =
      acquireExclusiveLock(descriptor.get(), executionControl, waited);
  if (!locked)
    return locked.takeError();
  if (!*locked)
    return std::optional<CacheLockAcquisition>{};
  return std::optional<CacheLockAcquisition>(
      CacheLockAcquisition{CacheLock(std::move(descriptor)), waited});
}

std::filesystem::path entryPath(const std::filesystem::path &root,
                                const ExternalToolResultCacheKey &key) {
  return root / "entries" / formatBlobDigestHex(key.inputMaterialDigest) /
         formatBlobDigestHex(key.executionConfigurationDigest) /
         formatBlobDigestHex(key.toolVersionDigest);
}

std::string serializeEntry(const CacheEntry &entry) {
  llvm::SmallString<4096> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output, 2);
  json.object([&] {
    json.attribute("schema", kCacheEntrySchema);
    json.attribute("version", kCacheEntryVersion);
    json.attribute("input_material_sha256",
                   formatBlobDigestHex(entry.key.inputMaterialDigest));
    json.attribute("execution_configuration_sha256",
                   formatBlobDigestHex(entry.key.executionConfigurationDigest));
    json.attribute("tool_version_sha256",
                   formatBlobDigestHex(entry.key.toolVersionDigest));
    json.attributeArray("files", [&] {
      for (const CacheFile &file : entry.files)
        json.object([&] {
          json.attribute("path", file.relativePath);
          json.attribute("content_sha256", formatBlobDigestHex(file.digest));
        });
    });
  });
  output << '\n';
  return output.str().str();
}

llvm::Expected<CacheEntry> parseEntry(llvm::StringRef contents) {
  auto parsed = llvm::json::parse(contents);
  if (!parsed)
    return cacheError("cache entry is malformed");
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object || object->size() != 6 ||
      object->getString("schema") != kCacheEntrySchema ||
      object->getString("version") != kCacheEntryVersion)
    return cacheError("cache entry has an invalid shape");
  auto parseDigest = [&](llvm::StringRef name) -> llvm::Expected<BlobDigest> {
    auto text = object->getString(name);
    if (!text)
      return cacheError("cache entry lacks " + name);
    return parseBlobDigestHex(*text);
  };
  auto input = parseDigest("input_material_sha256");
  if (!input)
    return input.takeError();
  auto configuration = parseDigest("execution_configuration_sha256");
  if (!configuration)
    return configuration.takeError();
  auto tool = parseDigest("tool_version_sha256");
  if (!tool)
    return tool.takeError();
  const llvm::json::Array *files = object->getArray("files");
  if (!files)
    return cacheError("cache entry lacks files");
  CacheEntry entry{{*input, *configuration, *tool}, {}};
  std::set<std::string> paths;
  for (const llvm::json::Value &value : *files) {
    const llvm::json::Object *file = value.getAsObject();
    if (!file || file->size() != 2)
      return cacheError("cache entry file has an invalid shape");
    auto path = file->getString("path");
    auto digest = file->getString("content_sha256");
    if (!path || !digest || path->empty() || !paths.insert(path->str()).second)
      return cacheError("cache entry file fields are invalid");
    const std::filesystem::path relative(path->str());
    if (relative.is_absolute() || relative.lexically_normal() != relative)
      return cacheError("cache entry file path is not canonical");
    auto parsedDigest = parseBlobDigestHex(*digest);
    if (!parsedDigest)
      return parsedDigest.takeError();
    entry.files.push_back({path->str(), *parsedDigest});
  }
  if (contents != serializeEntry(entry))
    return cacheError("cache entry is not canonical");
  return entry;
}

llvm::Expected<BlobDigest> digestFile(const std::filesystem::path &path) {
  return digestOrdinaryFile(path);
}

llvm::Error ensureSafeBundleParent(const std::filesystem::path &bundleRoot,
                                   llvm::StringRef relativePath) {
  const std::filesystem::path relative(relativePath.str());
  if (relative.is_absolute() || relative.lexically_normal() != relative)
    return cacheError("cache payload path is not canonical");
  std::filesystem::path current = bundleRoot;
  for (const std::filesystem::path &component : relative.parent_path()) {
    current /= component;
    std::error_code error;
    const auto status = std::filesystem::symlink_status(current, error);
    if (error || !std::filesystem::is_directory(status) ||
        std::filesystem::is_symlink(status))
      return cacheError("cache payload parent is not an ordinary directory");
  }
  return llvm::Error::success();
}

llvm::Error copyAtomically(const std::filesystem::path &source,
                           const std::filesystem::path &destination,
                           const BlobDigest &expected) {
  auto observed = digestFile(source);
  if (!observed)
    return observed.takeError();
  if (*observed != expected)
    return cacheError("cache payload digest does not match its entry");
  const std::filesystem::path temporary = destination.string() +
                                          ".loom-cache-partial." +
                                          std::to_string(::getpid());
  std::error_code error;
  std::filesystem::remove(temporary, error);
  error.clear();
  std::filesystem::copy_file(source, temporary,
                             std::filesystem::copy_options::none, error);
  if (error)
    return cacheError("cannot copy cache payload: " + error.message());
  auto copied = digestFile(temporary);
  if (!copied || *copied != expected) {
    std::filesystem::remove(temporary, error);
    if (!copied)
      return copied.takeError();
    return cacheError("copied cache payload changed");
  }
  std::filesystem::rename(temporary, destination, error);
  if (error) {
    std::filesystem::remove(temporary, error);
    return cacheError("cannot publish cache payload: " + error.message());
  }
  return llvm::Error::success();
}

std::vector<std::string>
cachedFilePaths(const InvocationManifestData &manifest) {
  return manifest.declaredOutputs;
}

llvm::Error
validateCacheableOutputClosure(const PreparedExternalToolInvocation &prepared,
                               const InvocationManifestData &manifest) {
  const std::filesystem::path outputs =
      std::filesystem::path(prepared.bundleRoot) / "outputs";
  std::error_code error;
  const std::filesystem::file_status rootStatus =
      std::filesystem::symlink_status(outputs, error);
  if (error || !std::filesystem::is_directory(rootStatus) ||
      std::filesystem::is_symlink(rootStatus))
    return cacheError("outputs is not an ordinary directory");

  std::set<std::string> expectedFiles{
      std::filesystem::path(kCompletionPath.str()).filename().string(),
      std::filesystem::path(kStdoutPath.str()).filename().string(),
      std::filesystem::path(kStderrPath.str()).filename().string()};
  std::set<std::string> expectedDirectories;
  for (const std::string &declared : manifest.declaredOutputs) {
    std::filesystem::path relative =
        std::filesystem::path(declared).lexically_relative("outputs");
    expectedFiles.insert(relative.generic_string());
    for (std::filesystem::path parent = relative.parent_path(); !parent.empty();
         parent = parent.parent_path())
      expectedDirectories.insert(parent.generic_string());
  }

  std::set<std::string> foundFiles;
  std::set<std::string> foundDirectories;
  for (std::filesystem::recursive_directory_iterator iterator(outputs, error),
       end;
       !error && iterator != end; iterator.increment(error)) {
    const std::filesystem::path path = iterator->path();
    const std::filesystem::file_status status =
        std::filesystem::symlink_status(path, error);
    if (error)
      break;
    const std::string relative =
        path.lexically_relative(outputs).generic_string();
    if (std::filesystem::is_symlink(status))
      return cacheError("outputs contains a symbolic link");
    if (std::filesystem::is_directory(status)) {
      if (!expectedDirectories.count(relative))
        return cacheError("outputs contains an undeclared directory");
      foundDirectories.insert(relative);
      continue;
    }
    if (!std::filesystem::is_regular_file(status) ||
        !expectedFiles.count(relative))
      return cacheError("outputs contains an undeclared entry");
    foundFiles.insert(relative);
  }
  if (error)
    return cacheError("cannot enumerate outputs: " + error.message());
  if (foundFiles != expectedFiles || foundDirectories != expectedDirectories)
    return cacheError("outputs does not match its declared closure");
  return llvm::Error::success();
}

llvm::Error writeAttemptFile(const std::filesystem::path &path,
                             llvm::StringRef contents) {
  const std::filesystem::path temporary =
      path.string() + ".loom-cache-partial." + std::to_string(::getpid());
  std::error_code error;
  {
    llvm::raw_fd_ostream output(temporary.string(), error,
                                llvm::sys::fs::OF_None);
    if (error)
      return cacheError("cannot create cache-hit attempt file: " +
                        error.message());
    output << contents;
    output.close();
    if (output.has_error())
      return cacheError("cannot write cache-hit attempt file");
  }
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary, error);
    return cacheError("cannot publish cache-hit attempt file: " +
                      error.message());
  }
  return llvm::Error::success();
}

llvm::Expected<bool>
restoreEntry(const std::filesystem::path &entryRoot,
             const PreparedExternalToolInvocation &prepared,
             const InvocationManifestData &manifest,
             const ExternalToolResultCacheKey &key) {
  auto entryText = readOrdinaryFile(entryRoot / kCacheEntryName.str());
  if (!entryText)
    return entryText.takeError();
  auto entry = parseEntry(*entryText);
  if (!entry)
    return entry.takeError();
  if (entry->key != key)
    return cacheError("cache entry key does not match its address");
  const std::vector<std::string> expectedPaths = cachedFilePaths(manifest);
  if (entry->files.size() != expectedPaths.size())
    return cacheError("cache entry file count does not match the invocation");
  for (std::size_t index = 0; index != expectedPaths.size(); ++index)
    if (entry->files[index].relativePath != expectedPaths[index])
      return cacheError("cache entry file membership does not match");

  const std::filesystem::path bundleRoot(prepared.bundleRoot);
  for (const CacheFile &file : entry->files) {
    if (llvm::Error error =
            ensureSafeBundleParent(bundleRoot, file.relativePath))
      return std::move(error);
    if (llvm::Error error =
            copyAtomically(entryRoot / "payload" / file.relativePath,
                           bundleRoot / file.relativePath, file.digest))
      return std::move(error);
  }
  std::vector<BlobDigest> outputs;
  outputs.reserve(manifest.declaredOutputs.size());
  for (std::size_t index = 0; index != manifest.declaredOutputs.size(); ++index)
    outputs.push_back(entry->files[index].digest);
  if (llvm::Error error =
          writeAttemptFile(bundleRoot / kStdoutPath.str(),
                           "loom external-tool result cache hit\n"))
    return std::move(error);
  if (llvm::Error error = writeAttemptFile(bundleRoot / kStderrPath.str(), {}))
    return std::move(error);
  const std::string completion = serializeInvocationCompletion(
      InvocationCompletionStatus::Success, 0, prepared.manifestDigest, outputs);
  if (llvm::Error error =
          writeAttemptFile(bundleRoot / kCompletionPath.str(), completion))
    return std::move(error);
  return true;
}

llvm::Expected<std::filesystem::path>
makeEntryStaging(const std::filesystem::path &entry) {
  std::error_code error;
  std::filesystem::create_directories(entry.parent_path(), error);
  if (error)
    return cacheError("cannot create cache entry parent: " + error.message());
  for (unsigned attempt = 0; attempt != 32; ++attempt) {
    llvm::SmallString<256> model((entry.string() + ".partial-%%%%%%").c_str());
    llvm::SmallString<256> candidate;
    llvm::sys::fs::createUniquePath(model, candidate, true);
    if (std::filesystem::create_directory(candidate.str().str(), error))
      return std::filesystem::path(candidate.str().str());
    if (error != std::errc::file_exists)
      return cacheError("cannot create cache staging: " + error.message());
  }
  return cacheError("cannot allocate cache staging");
}

llvm::Error publishEntry(const std::filesystem::path &entryRoot,
                         const PreparedExternalToolInvocation &prepared,
                         const InvocationManifestData &manifest,
                         const ExternalToolResultCacheKey &key) {
  auto completion = loadExternalToolInvocationCompletion(prepared);
  if (!completion)
    return completion.takeError();
  if (completion->status != InvocationCompletionStatus::Success ||
      completion->manifestDigest != prepared.manifestDigest ||
      completion->outputDigests.size() != manifest.declaredOutputs.size())
    return cacheError("only an exact successful completion is cacheable");
  if (llvm::Error error = validateCacheableOutputClosure(prepared, manifest))
    return error;

  auto staging = makeEntryStaging(entryRoot);
  if (!staging)
    return staging.takeError();
  struct Cleanup {
    std::filesystem::path path;
    bool published = false;
    ~Cleanup() {
      if (!published) {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
      }
    }
  } cleanup{*staging};

  CacheEntry entry{key, {}};
  const std::vector<std::string> files = cachedFilePaths(manifest);
  const std::filesystem::path bundleRoot(prepared.bundleRoot);
  for (const std::string &relative : files) {
    if (llvm::Error error = ensureSafeBundleParent(bundleRoot, relative))
      return error;
    auto digest = digestFile(bundleRoot / relative);
    if (!digest)
      return digest.takeError();
    const std::size_t outputIndex = entry.files.size();
    if (outputIndex < completion->outputDigests.size() &&
        *digest != completion->outputDigests[outputIndex])
      return cacheError("declared output changed before cache publication");
    const std::filesystem::path destination = *staging / "payload" / relative;
    std::error_code error;
    std::filesystem::create_directories(destination.parent_path(), error);
    if (error)
      return cacheError("cannot create cache payload parent: " +
                        error.message());
    std::filesystem::copy_file(bundleRoot / relative, destination,
                               std::filesystem::copy_options::none, error);
    if (error)
      return cacheError("cannot snapshot cache payload: " + error.message());
    auto copied = digestFile(destination);
    if (!copied || *copied != *digest) {
      if (!copied)
        return copied.takeError();
      return cacheError("cache payload changed while it was published");
    }
    entry.files.push_back({relative, *digest});
  }

  const std::string metadata = serializeEntry(entry);
  {
    std::error_code error;
    llvm::raw_fd_ostream output((*staging / kCacheEntryName.str()).string(),
                                error, llvm::sys::fs::OF_None);
    if (error)
      return cacheError("cannot create cache entry metadata: " +
                        error.message());
    output << metadata;
    output.close();
    if (output.has_error())
      return cacheError("cannot write cache entry metadata");
  }
  std::error_code error;
  std::filesystem::rename(*staging, entryRoot, error);
  if (error) {
    if (error == std::errc::file_exists)
      return llvm::Error::success();
    return cacheError("cannot publish cache entry: " + error.message());
  }
  cleanup.published = true;
  return llvm::Error::success();
}

enum class CacheScriptMode { Execute, Preflight, Postflight };

llvm::Error stopProcessGroup(const llvm::sys::ProcessInfo &process) {
  const pid_t processId = static_cast<pid_t>(process.Pid);
  if (::kill(-processId, SIGKILL) != 0) {
    if (errno != ESRCH)
      return cacheSystemError("cannot stop generated run-script process group");
    if (::kill(processId, SIGKILL) != 0 && errno != ESRCH)
      return cacheSystemError("cannot stop generated run script");
  }
  std::string message;
  const llvm::sys::ProcessInfo waited =
      llvm::sys::Wait(process, std::nullopt, &message);
  if (waited.Pid != process.Pid)
    return cacheError("cannot reap stopped generated run script: " + message);
  return llvm::Error::success();
}

llvm::Expected<ScriptExecution>
runScript(const PreparedExternalToolInvocation &prepared, CacheScriptMode mode,
          ExecutionControlView executionControl) {
  auto manifest = loadPreparedInvocationManifest(prepared);
  if (!manifest)
    return manifest.takeError();
  llvm::SmallString<256> script(prepared.bundleRoot);
  llvm::sys::path::append(script, kRunScriptName);
  llvm::ErrorOr<std::string> bash = llvm::sys::findProgramByName("bash");
  if (!bash)
    return cacheError("could not find bash: " + bash.getError().message());
  llvm::SmallVector<llvm::StringRef, 3> arguments{*bash, script};
  if (mode == CacheScriptMode::Preflight)
    arguments.push_back("--loom-cache-preflight");
  else if (mode == CacheScriptMode::Postflight)
    arguments.push_back("--loom-cache-postflight");
  if (executionControl.stopRequested())
    return ScriptExecution{externalToolExecutionStoppedExitCode, false};
  std::string message;
  bool executionFailed = false;
  const llvm::sys::ProcessInfo process =
      llvm::sys::ExecuteNoWait(*bash, arguments, std::nullopt, {}, 0, &message,
                               &executionFailed, nullptr, true);
  if (executionFailed || process.Pid == 0)
    return cacheError("could not execute generated run script: " + message);
  while (true) {
    if (executionControl.stopRequested()) {
      if (llvm::Error error = stopProcessGroup(process))
        return std::move(error);
      return ScriptExecution{externalToolExecutionStoppedExitCode, true};
    }
    const llvm::sys::ProcessInfo waited =
        llvm::sys::Wait(process, 0, &message, nullptr, true);
    if (waited.Pid == process.Pid) {
      if (waited.ReturnCode < 0)
        return cacheError("generated run script terminated abnormally: " +
                          message);
      return ScriptExecution{waited.ReturnCode, true};
    }
    if (waited.Pid != 0)
      return cacheError("could not wait for generated run script: " + message);
    if (!waitForExecutionControl(executionControl)) {
      if (llvm::Error error = stopProcessGroup(process))
        return std::move(error);
      return ScriptExecution{externalToolExecutionStoppedExitCode, true};
    }
  }
}

} // namespace

llvm::Expected<ExternalToolResultCacheKey> deriveExternalToolResultCacheKey(
    const PreparedExternalToolInvocation &prepared) {
  auto loaded = loadPreparedInvocationManifest(prepared);
  if (!loaded)
    return loaded.takeError();
  auto input = canonicalInputMaterial(loaded->second);
  if (!input)
    return input.takeError();
  auto configuration =
      canonicalExecutionConfiguration(prepared, loaded->second);
  if (!configuration)
    return configuration.takeError();
  auto toolVersion = canonicalToolVersion(loaded->second);
  if (!toolVersion)
    return toolVersion.takeError();
  return ExternalToolResultCacheKey{
      domainDigest("loom.external_tool_cache.input.v1", *input),
      domainDigest("loom.external_tool_cache.configuration.v1", *configuration),
      domainDigest("loom.external_tool_cache.tool.v1", *toolVersion)};
}

llvm::Expected<ExternalToolInvocationExecutionObservation>
executeExternalToolInvocationBundleObserved(
    const PreparedExternalToolInvocation &prepared,
    ExecutionControlView executionControl,
    ExternalToolResultReusePolicy reusePolicy) {
  const auto stopped =
      [&prepared, reusePolicy](ExternalToolResultCacheAvailability availability,
                               bool waited, bool invoked) {
        return ExternalToolInvocationExecutionObservation{
            prepared.manifestDigest,
            externalToolExecutionStoppedExitCode,
            reusePolicy,
            availability,
            ExternalToolResultCacheLookup::NotAttempted,
            ExternalToolResultCacheDiscard::NotAttempted,
            ExternalToolResultCachePublication::NotAttempted,
            waited,
            invoked};
      };
  auto executeWithoutCache =
      [&](ExternalToolResultCacheAvailability availability,
          bool waitedForCacheKeyLock = false)
      -> llvm::Expected<ExternalToolInvocationExecutionObservation> {
    auto execution =
        runScript(prepared, CacheScriptMode::Execute, executionControl);
    if (!execution)
      return execution.takeError();
    if (execution->exitCode == 0 &&
        reusePolicy == ExternalToolResultReusePolicy::RequireFresh) {
      auto postflight =
          runScript(prepared, CacheScriptMode::Postflight, executionControl);
      if (!postflight)
        return postflight.takeError();
      if (postflight->exitCode == externalToolExecutionStoppedExitCode)
        return stopped(availability, waitedForCacheKeyLock, execution->invoked);
      if (postflight->exitCode != 0)
        return cacheError(
            "fresh invocation inputs or tool changed during execution");
    }
    return ExternalToolInvocationExecutionObservation{
        prepared.manifestDigest,
        execution->exitCode,
        reusePolicy,
        availability,
        ExternalToolResultCacheLookup::NotAttempted,
        ExternalToolResultCacheDiscard::NotAttempted,
        ExternalToolResultCachePublication::NotAttempted,
        waitedForCacheKeyLock,
        execution->invoked};
  };
  if (executionControl.stopRequested())
    return stopped(reusePolicy == ExternalToolResultReusePolicy::RequireFresh
                       ? ExternalToolResultCacheAvailability::Disabled
                       : ExternalToolResultCacheAvailability::Unavailable,
                   false, false);
  if (reusePolicy == ExternalToolResultReusePolicy::RequireFresh)
    return executeWithoutCache(ExternalToolResultCacheAvailability::Disabled);

  auto root = cacheRoot(executionControl);
  if (!root) {
    cacheDiagnostic(DiagnosticVerbosity::Summary, "unavailable",
                    llvm::toString(root.takeError()));
    return executeWithoutCache(
        ExternalToolResultCacheAvailability::Unavailable);
  }
  if (root->executionStopped)
    return stopped(ExternalToolResultCacheAvailability::Unavailable, false,
                   false);
  if (!root->root)
    return executeWithoutCache(ExternalToolResultCacheAvailability::Disabled);

  auto key = deriveExternalToolResultCacheKey(prepared);
  if (!key) {
    cacheDiagnostic(DiagnosticVerbosity::Summary, "unavailable",
                    llvm::toString(key.takeError()));
    return executeWithoutCache(
        ExternalToolResultCacheAvailability::Unavailable);
  }
  const std::string keySpelling = keyText(*key);
  auto lock = lockKey(*root->root, *key, executionControl);
  if (!lock) {
    cacheDiagnostic(DiagnosticVerbosity::Summary, "unavailable",
                    llvm::toString(lock.takeError()));
    return executeWithoutCache(
        ExternalToolResultCacheAvailability::Unavailable);
  }
  if (!*lock)
    return stopped(ExternalToolResultCacheAvailability::Available, true, false);
  const bool waitedForCacheKeyLock = (**lock).waited;

  auto loaded = loadPreparedInvocationManifest(prepared);
  if (!loaded) {
    cacheDiagnostic(DiagnosticVerbosity::Summary, "unavailable",
                    llvm::toString(loaded.takeError()));
    return executeWithoutCache(ExternalToolResultCacheAvailability::Unavailable,
                               waitedForCacheKeyLock);
  }
  const std::filesystem::path entry = entryPath(*root->root, *key);
  std::error_code statusError;
  const bool entryExists = std::filesystem::exists(entry, statusError);
  if (statusError) {
    cacheDiagnostic(DiagnosticVerbosity::Summary, "unavailable",
                    statusError.message());
    return executeWithoutCache(ExternalToolResultCacheAvailability::Unavailable,
                               waitedForCacheKeyLock);
  }
  ExternalToolResultCacheDiscard discard =
      ExternalToolResultCacheDiscard::NotAttempted;
  if (entryExists) {
    auto preflight =
        runScript(prepared, CacheScriptMode::Preflight, executionControl);
    if (!preflight)
      return preflight.takeError();
    if (preflight->exitCode == externalToolExecutionStoppedExitCode)
      return stopped(ExternalToolResultCacheAvailability::Available,
                     waitedForCacheKeyLock, preflight->invoked);
    if (preflight->exitCode != 0)
      return ExternalToolInvocationExecutionObservation{
          prepared.manifestDigest,
          preflight->exitCode,
          reusePolicy,
          ExternalToolResultCacheAvailability::Available,
          ExternalToolResultCacheLookup::Miss,
          discard,
          ExternalToolResultCachePublication::NotAttempted,
          waitedForCacheKeyLock,
          false};
    auto restored = restoreEntry(entry, prepared, loaded->second, *key);
    if (restored && *restored) {
      cacheDiagnostic(DiagnosticVerbosity::Summary, "hit", keySpelling);
      return ExternalToolInvocationExecutionObservation{
          prepared.manifestDigest,
          0,
          reusePolicy,
          ExternalToolResultCacheAvailability::Available,
          ExternalToolResultCacheLookup::Hit,
          discard,
          ExternalToolResultCachePublication::NotAttempted,
          waitedForCacheKeyLock,
          false};
    }
    if (!restored)
      cacheDiagnostic(DiagnosticVerbosity::Summary, "discard",
                      llvm::toString(restored.takeError()));
    std::error_code removeError;
    std::filesystem::remove_all(entry, removeError);
    discard = removeError ? ExternalToolResultCacheDiscard::Failed
                          : ExternalToolResultCacheDiscard::Discarded;
  }
  cacheDiagnostic(DiagnosticVerbosity::Summary, "miss", keySpelling);
  auto execution =
      runScript(prepared, CacheScriptMode::Execute, executionControl);
  if (!execution)
    return execution.takeError();
  if (execution->exitCode == externalToolExecutionStoppedExitCode)
    return ExternalToolInvocationExecutionObservation{
        prepared.manifestDigest,
        execution->exitCode,
        reusePolicy,
        ExternalToolResultCacheAvailability::Available,
        ExternalToolResultCacheLookup::Miss,
        discard,
        ExternalToolResultCachePublication::NotAttempted,
        waitedForCacheKeyLock,
        execution->invoked};
  if (execution->exitCode != 0)
    return ExternalToolInvocationExecutionObservation{
        prepared.manifestDigest,
        execution->exitCode,
        reusePolicy,
        ExternalToolResultCacheAvailability::Available,
        ExternalToolResultCacheLookup::Miss,
        discard,
        ExternalToolResultCachePublication::NotAttempted,
        waitedForCacheKeyLock,
        true};
  if (discard == ExternalToolResultCacheDiscard::Failed)
    return ExternalToolInvocationExecutionObservation{
        prepared.manifestDigest,
        0,
        reusePolicy,
        ExternalToolResultCacheAvailability::Available,
        ExternalToolResultCacheLookup::Miss,
        discard,
        ExternalToolResultCachePublication::Failed,
        waitedForCacheKeyLock,
        true};
  auto postflight =
      runScript(prepared, CacheScriptMode::Postflight, executionControl);
  if (!postflight) {
    cacheDiagnostic(DiagnosticVerbosity::Summary, "publish-unavailable",
                    llvm::toString(postflight.takeError()));
    return ExternalToolInvocationExecutionObservation{
        prepared.manifestDigest,
        0,
        reusePolicy,
        ExternalToolResultCacheAvailability::Available,
        ExternalToolResultCacheLookup::Miss,
        discard,
        ExternalToolResultCachePublication::Failed,
        waitedForCacheKeyLock,
        true};
  }
  if (postflight->exitCode == externalToolExecutionStoppedExitCode)
    return ExternalToolInvocationExecutionObservation{
        prepared.manifestDigest,
        externalToolExecutionStoppedExitCode,
        reusePolicy,
        ExternalToolResultCacheAvailability::Available,
        ExternalToolResultCacheLookup::Miss,
        discard,
        ExternalToolResultCachePublication::NotAttempted,
        waitedForCacheKeyLock,
        true};
  if (postflight->exitCode != 0) {
    cacheDiagnostic(DiagnosticVerbosity::Summary, "publish-unavailable",
                    "invocation inputs or tool changed during execution");
    return ExternalToolInvocationExecutionObservation{
        prepared.manifestDigest,
        0,
        reusePolicy,
        ExternalToolResultCacheAvailability::Available,
        ExternalToolResultCacheLookup::Miss,
        discard,
        ExternalToolResultCachePublication::Failed,
        waitedForCacheKeyLock,
        true};
  }
  auto postflightKey = deriveExternalToolResultCacheKey(prepared);
  if (!postflightKey || *postflightKey != *key) {
    if (!postflightKey)
      cacheDiagnostic(DiagnosticVerbosity::Summary, "publish-unavailable",
                      llvm::toString(postflightKey.takeError()));
    else
      cacheDiagnostic(DiagnosticVerbosity::Summary, "publish-unavailable",
                      "cache-key material changed during execution");
    return ExternalToolInvocationExecutionObservation{
        prepared.manifestDigest,
        0,
        reusePolicy,
        ExternalToolResultCacheAvailability::Available,
        ExternalToolResultCacheLookup::Miss,
        discard,
        ExternalToolResultCachePublication::Failed,
        waitedForCacheKeyLock,
        true};
  }
  ExternalToolResultCachePublication publication =
      ExternalToolResultCachePublication::Published;
  if (llvm::Error error = publishEntry(entry, prepared, loaded->second, *key)) {
    cacheDiagnostic(DiagnosticVerbosity::Summary, "publish-unavailable",
                    llvm::toString(std::move(error)));
    publication = ExternalToolResultCachePublication::Failed;
  } else {
    cacheDiagnostic(DiagnosticVerbosity::Decision, "published", entry.string());
  }
  return ExternalToolInvocationExecutionObservation{
      prepared.manifestDigest,
      0,
      reusePolicy,
      ExternalToolResultCacheAvailability::Available,
      ExternalToolResultCacheLookup::Miss,
      discard,
      publication,
      waitedForCacheKeyLock,
      true};
}

llvm::Expected<int> executeExternalToolInvocationBundle(
    const PreparedExternalToolInvocation &prepared) {
  auto observation = executeExternalToolInvocationBundleObserved(prepared);
  if (!observation)
    return observation.takeError();
  return observation->exitCode;
}

} // namespace loom::external_tool
