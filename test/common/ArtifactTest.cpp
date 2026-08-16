#include "Common/Artifact.h"
#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/ComponentViewDigest.h"
#include "Config/ResolvedConfig.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace loom;

// This test binary is linked with --wrap=fsync so that the real ArtifactStore
// implementation's fsync calls resolve to __wrap_fsync. The wrapper fails
// exactly one scoped class of call, a store-directory flush, to reproduce the
// ambiguous post-insertion failure. Production code carries no injection hook.
static std::atomic<bool> failDirectoryFlush{false};

extern "C" int __wrap_fsync(int fd) {
  struct stat status;
  if (failDirectoryFlush.load(std::memory_order_acquire) &&
      ::fstat(fd, &status) == 0 && S_ISDIR(status.st_mode)) {
    errno = EIO;
    return -1;
  }
  return static_cast<int>(::syscall(SYS_fsync, fd));
}

namespace {

constexpr ArtifactSchemaDescriptor testSchema{"loom.test.artifact",
                                              SchemaVersion{1, 2}};
constexpr ArtifactSchemaDescriptor otherSchema{"loom.test.other",
                                               SchemaVersion{1, 2}};
constexpr ArtifactSchemaDescriptor otherVersion{"loom.test.artifact",
                                                SchemaVersion{1, 3}};

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << "\n";
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T>
T takeExpected(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectErrorContains(const char *test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error containing '" + expected.str() + "'");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected),
          "unexpected error: " + message);
}

CanonicalSemanticBytes semantic(std::initializer_list<std::uint8_t> bytes) {
  return CanonicalSemanticBytes(std::vector<std::uint8_t>(bytes));
}

void appendU32Be(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

std::vector<std::uint8_t>
expectedPreimage(const ArtifactSchemaDescriptor &schema,
                 const CanonicalSemanticBytes &canonicalBytes) {
  static constexpr char domain[] = "loom.artifact.identity.v1\0";
  std::vector<std::uint8_t> bytes(domain, domain + sizeof(domain) - 1);
  appendU32Be(bytes, static_cast<std::uint32_t>(schema.identity.size()));
  bytes.insert(bytes.end(), schema.identity.bytes_begin(),
               schema.identity.bytes_end());
  appendU32Be(bytes, schema.version.major);
  appendU32Be(bytes, schema.version.minor);
  appendU64Be(bytes, canonicalBytes.bytes().size());
  bytes.insert(bytes.end(), canonicalBytes.bytes().begin(),
               canonicalBytes.bytes().end());
  return bytes;
}

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(const char *test) : test_(test) {
    llvm::SmallString<128> path;
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory("loom-artifact-test", path))
      fail(test_, "unable to create temporary directory: " + error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << test_ << ": unable to remove temporary directory: "
                   << error.message() << "\n";
  }

  llvm::StringRef path() const { return path_; }

private:
  const char *test_;
  std::string path_;
};

std::vector<std::string> regularFiles(const char *test, llvm::StringRef root) {
  std::vector<std::string> paths;
  std::error_code error;
  llvm::sys::fs::recursive_directory_iterator iterator(root, error), end;
  if (error)
    fail(test, "unable to inspect artifact store: " + error.message());
  while (iterator != end) {
    if (llvm::sys::fs::is_regular_file(iterator->path()))
      paths.push_back(iterator->path());
    iterator.increment(error);
    if (error)
      fail(test, "unable to inspect artifact store: " + error.message());
  }
  return paths;
}

std::vector<std::uint8_t> readFile(const char *test, llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false);
  if (std::error_code error = buffer.getError())
    fail(test, "unable to read stored artifact: " + error.message());
  llvm::StringRef contents = (*buffer)->getBuffer();
  return std::vector<std::uint8_t>(contents.bytes_begin(),
                                   contents.bytes_end());
}

std::string objectPath(llvm::StringRef root, const ArtifactIdentity &identity) {
  llvm::SmallString<128> path(root);
  llvm::sys::path::append(path, formatArtifactIdentityHex(identity));
  return path.str().str();
}

void writeFile(const char *test, llvm::StringRef path,
               llvm::ArrayRef<std::uint8_t> bytes) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error);
  if (error)
    fail(test, "unable to replace stored artifact: " + error.message());
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.close();
  if (output.has_error()) {
    output.clear_error();
    fail(test, "unable to replace stored artifact");
  }
}

void schemaVersionTextCodecIsCanonical() {
  const std::vector<std::pair<SchemaVersion, std::string>> canonical = {
      {{0, 0}, "0.0"},
      {{1, 0}, "1.0"},
      {{std::numeric_limits<std::uint32_t>::max(),
        std::numeric_limits<std::uint32_t>::max()},
       "4294967295.4294967295"},
  };

  for (const auto &[version, spelling] : canonical) {
    require(__func__, formatSchemaVersion(version) == spelling,
            "schema version formatting changed for " + spelling);
    const SchemaVersion parsed =
        takeExpected(__func__, parseSchemaVersion(spelling));
    require(__func__, parsed == version,
            "schema version parsing changed for " + spelling);
    require(__func__, formatSchemaVersion(parsed) == spelling,
            "schema version round trip was not canonical for " + spelling);
  }

  for (llvm::StringRef spelling :
       {"",     "1",    "1.",   ".1",           "1.0.0",        "+1.0", "-1.0",
        "1.+0", "1.-0", " 1.0", "1.0 ",         "1 .0",         "1. 0", "01.0",
        "1.00", "00.0", "0.01", "4294967296.0", "0.4294967296", "1.a"})
    expectErrorContains(__func__, parseSchemaVersion(spelling),
                        "schema version");
}

void missingStoreRootIsRejected() {
  TemporaryDirectory parent(__func__);
  llvm::SmallString<128> missingParent(parent.path());
  llvm::sys::path::append(missingParent, "missing");
  llvm::SmallString<128> root(missingParent);
  llvm::sys::path::append(root, "store");

  ArtifactStore store(root);
  expectErrorContains(__func__, store.put(testSchema, semantic({0x01})),
                      "artifact_store_io");
  require(__func__, !llvm::sys::fs::exists(missingParent),
          "ArtifactStore created part of its missing root chain");
}

void finalizerMatchesKnownEnvelopeAndDigest() {
  const CanonicalSemanticBytes bytes = semantic({0x00, 0x10, 0xff});
  const std::vector<std::uint8_t> preimage =
      expectedPreimage(testSchema, bytes);
  require(
      __func__,
      llvm::toHex(preimage, true) ==
          "6c6f6f6d2e61727469666163742e6964656e746974792e763100000000126c6f6f6d"
          "2e746573742e6172746966616374000000010000000200000000000000030010ff",
      "known identity preimage changed");
  const ArtifactIdentity identity = finalizeArtifactIdentity(testSchema, bytes);

  require(
      __func__,
      formatArtifactIdentityHex(identity) ==
          "35d5389eca0d279e34fdf21ebd5d51991c8e75aa9312cedf14f452882e0f6db4",
      "known ArtifactIdentity digest changed");
  require(__func__, identity == finalizeArtifactIdentity(testSchema, bytes),
          "identical finalization input was not deterministic");
  require(__func__, identity != finalizeArtifactIdentity(otherSchema, bytes),
          "schema identity did not affect ArtifactIdentity");
  require(__func__, identity != finalizeArtifactIdentity(otherVersion, bytes),
          "schema version did not affect ArtifactIdentity");
  require(__func__,
          identity != finalizeArtifactIdentity(testSchema,
                                               semantic({0x00, 0x10, 0xfe})),
          "canonical semantic bytes did not affect ArtifactIdentity");

  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  require(__func__,
          takeExpected(__func__, store.put(testSchema, bytes)) == identity,
          "ArtifactStore returned a different identity");
  const std::vector<std::string> files =
      regularFiles(__func__, directory.path());
  require(__func__, files.size() == 1,
          "ArtifactStore did not publish exactly one object");
  llvm::SmallString<128> expectedPath(directory.path());
  llvm::sys::path::append(expectedPath, formatArtifactIdentityHex(identity));
  require(__func__, files.front() == expectedPath,
          "ArtifactStore left a temporary object instead of the derived key");
  require(__func__, readFile(__func__, files.front()) == preimage,
          "stored object is not the exact identity preimage");
}

void identityBoundariesRejectInvalidValues() {
  static_assert(!std::is_default_constructible_v<ArtifactIdentity>);
  static_assert(!std::is_default_constructible_v<ArtifactReference<unsigned>>);

  std::vector<std::uint8_t> shortBytes(31, 0);
  std::vector<std::uint8_t> longBytes(33, 0);
  expectErrorContains(__func__, ArtifactIdentity::fromBytes(shortBytes),
                      "exactly 32 bytes");
  expectErrorContains(__func__, ArtifactIdentity::fromBytes(longBytes),
                      "exactly 32 bytes");

  std::vector<std::uint8_t> zeroBytes(32, 0);
  const ArtifactIdentity zero =
      takeExpected(__func__, ArtifactIdentity::fromBytes(zeroBytes));
  require(__func__, formatArtifactIdentityHex(zero) == std::string(64, '0'),
          "all-zero ArtifactIdentity was not preserved as a valid value");

  const std::vector<std::string> invalidSpellings = {
      "", std::string(63, '0'), std::string(65, '0'), std::string(64, 'A'),
      std::string(63, '0') + "g"};
  for (const std::string &spelling : invalidSpellings)
    expectErrorContains(__func__, parseArtifactIdentityHex(spelling),
                        "artifact identity");

  require(__func__,
          takeExpected(__func__,
                       parseArtifactIdentityHex(std::string(64, '0'))) == zero,
          "valid lowercase ArtifactIdentity text did not parse");
}

void concurrentIdenticalPublishDeduplicates() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const CanonicalSemanticBytes bytes = semantic({0x01, 0x02, 0x03});
  const ArtifactIdentity expected = finalizeArtifactIdentity(testSchema, bytes);

  constexpr unsigned threadCount = 8;
  std::atomic<bool> start{false};
  std::vector<std::string> identities(threadCount);
  std::vector<std::string> errors(threadCount);
  std::vector<std::thread> threads;
  threads.reserve(threadCount);
  for (unsigned index = 0; index < threadCount; ++index) {
    threads.emplace_back([&, index] {
      while (!start.load(std::memory_order_acquire))
        std::this_thread::yield();
      auto stored = store.put(testSchema, bytes);
      if (!stored) {
        errors[index] = llvm::toString(stored.takeError());
        return;
      }
      identities[index] = formatArtifactIdentityHex(*stored);
    });
  }
  start.store(true, std::memory_order_release);
  for (std::thread &thread : threads)
    thread.join();

  const std::string expectedText = formatArtifactIdentityHex(expected);
  for (unsigned index = 0; index < threadCount; ++index) {
    require(__func__, errors[index].empty(), errors[index]);
    require(__func__, identities[index] == expectedText,
            "concurrent publisher returned a different identity");
  }
  const std::vector<std::string> files =
      regularFiles(__func__, directory.path());
  require(__func__, files.size() == 1,
          "concurrent identical publication did not deduplicate");
  require(__func__,
          readFile(__func__, files.front()) ==
              expectedPreimage(testSchema, bytes),
          "deduplicated object has unexpected contents");
}

class DirectoryFlushFailureScope {
public:
  DirectoryFlushFailureScope() {
    failDirectoryFlush.store(true, std::memory_order_release);
  }
  ~DirectoryFlushFailureScope() {
    failDirectoryFlush.store(false, std::memory_order_release);
  }
};

void postInsertionIoFailureKeepsCompleteObjectAndAllowsRetry() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const CanonicalSemanticBytes bytes = semantic({0x60, 0x61});
  const ArtifactIdentity identity = finalizeArtifactIdentity(testSchema, bytes);

  {
    DirectoryFlushFailureScope flushFailure;
    expectErrorContains(__func__, store.put(testSchema, bytes),
                        "artifact_store_io");
    const CanonicalSemanticBytes published =
        takeExpected(__func__, store.get(testSchema, identity));
    require(__func__, published.bytes().equals(bytes.bytes()),
            "post-insertion failure did not leave the complete object "
            "readable");
    const std::vector<std::string> files =
        regularFiles(__func__, directory.path());
    require(__func__,
            files.size() == 1 &&
                files.front() == objectPath(directory.path(), identity),
            "post-insertion failure left state other than the one complete "
            "object");
  }

  require(__func__,
          takeExpected(__func__, store.put(testSchema, bytes)) == identity,
          "deterministic retry after a post-insertion failure returned a "
          "different identity");
  const CanonicalSemanticBytes recovered =
      takeExpected(__func__, store.get(testSchema, identity));
  require(__func__, recovered.bytes().equals(bytes.bytes()),
          "deterministic retry did not converge to the complete expected "
          "object");
}

void existingWrongOrCorruptObjectIsRejected() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const CanonicalSemanticBytes bytes = semantic({0x20, 0x21});
  takeExpected(__func__, store.put(testSchema, bytes));
  const std::vector<std::string> files =
      regularFiles(__func__, directory.path());
  require(__func__, files.size() == 1, "expected one stored object");

  const CanonicalSemanticBytes otherBytes = semantic({0x30});
  const std::vector<std::uint8_t> wrongPreimage =
      expectedPreimage(otherSchema, otherBytes);
  writeFile(__func__, files.front(), wrongPreimage);
  expectErrorContains(__func__, store.put(testSchema, bytes),
                      "artifact_store_corruption");
  require(__func__, readFile(__func__, files.front()) == wrongPreimage,
          "ArtifactStore overwrote an existing wrong-key object");

  const std::vector<std::uint8_t> corrupt = {'b', 'a', 'd'};
  writeFile(__func__, files.front(), corrupt);
  expectErrorContains(__func__, store.put(testSchema, bytes),
                      "artifact_store_corruption");
  require(__func__, readFile(__func__, files.front()) == corrupt,
          "ArtifactStore overwrote an existing corrupt object");

  if (std::error_code error = llvm::sys::fs::remove(files.front()))
    fail(__func__, "unable to remove corrupt object: " + error.message());
  if (std::error_code error = llvm::sys::fs::create_directory(files.front()))
    fail(__func__,
         "unable to create corrupt object directory: " + error.message());
  expectErrorContains(__func__, store.put(testSchema, bytes),
                      "artifact_store_corruption");
}

void existingSymlinkObjectIsRejected() {
  TemporaryDirectory directory(__func__);
  const CanonicalSemanticBytes bytes = semantic({0x40, 0x41});
  const ArtifactIdentity identity = finalizeArtifactIdentity(testSchema, bytes);

  llvm::SmallString<128> targetPath(directory.path());
  llvm::sys::path::append(targetPath, ".symlink-target");
  writeFile(__func__, targetPath, expectedPreimage(testSchema, bytes));

  llvm::SmallString<128> objectPath(directory.path());
  llvm::sys::path::append(objectPath, formatArtifactIdentityHex(identity));
  if (std::error_code error =
          llvm::sys::fs::create_symlink(targetPath, objectPath))
    fail(__func__,
         "unable to create stored-object symlink: " + error.message());

  ArtifactStore store(directory.path());
  expectErrorContains(__func__, store.put(testSchema, bytes),
                      "artifact_store_corruption");
  require(__func__, llvm::sys::fs::is_symlink_file(objectPath),
          "ArtifactStore replaced an existing symlink object");
}

void storedCanonicalBytesRoundTrip() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());

  for (const CanonicalSemanticBytes &bytes :
       {semantic({}), semantic({0x00, 0x7f, 0xff})}) {
    const ArtifactIdentity identity =
        takeExpected(__func__, store.put(testSchema, bytes));
    const CanonicalSemanticBytes loaded =
        takeExpected(__func__, store.get(testSchema, identity));
    require(__func__, loaded.bytes().equals(bytes.bytes()),
            "ArtifactStore returned different canonical semantic bytes");
  }
}

void storedSchemaMismatchIsRejected() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const ArtifactIdentity identity =
      takeExpected(__func__, store.put(testSchema, semantic({0x50})));

  expectErrorContains(__func__, store.get(otherSchema, identity),
                      "artifact_schema_mismatch");
  expectErrorContains(__func__, store.get(otherVersion, identity),
                      "artifact_schema_mismatch");
}

void missingStoredObjectIsRejected() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const ArtifactIdentity identity =
      finalizeArtifactIdentity(testSchema, semantic({0x51}));

  expectErrorContains(__func__, store.get(testSchema, identity),
                      "artifact_store_missing");
}

void nonRegularStoredObjectsAreRejectedOnRead() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const CanonicalSemanticBytes bytes = semantic({0x52});
  const ArtifactIdentity identity = finalizeArtifactIdentity(testSchema, bytes);
  const std::string objectName = formatArtifactIdentityHex(identity);
  const std::string path = objectPath(directory.path(), identity);

  llvm::SmallString<128> targetPath(directory.path());
  llvm::sys::path::append(targetPath, ".symlink-target");
  writeFile(__func__, targetPath, expectedPreimage(testSchema, bytes));
  if (std::error_code error = llvm::sys::fs::create_symlink(targetPath, path))
    fail(__func__,
         "unable to create stored-object symlink: " + error.message());
  expectErrorContains(__func__, store.get(testSchema, identity),
                      "artifact_store_corruption");

  if (std::error_code error = llvm::sys::fs::remove(path))
    fail(__func__,
         "unable to remove stored-object symlink: " + error.message());
  if (std::error_code error = llvm::sys::fs::create_directory(path))
    fail(__func__,
         "unable to create stored-object directory: " + error.message());
  expectErrorContains(__func__, store.get(testSchema, identity),
                      "artifact_store_corruption");

  if (std::error_code error = llvm::sys::fs::remove(path))
    fail(__func__,
         "unable to remove stored-object directory: " + error.message());
  const std::string root = directory.path().str();
  int storeDirectory;
  do {
    storeDirectory =
        ::open(root.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
  } while (storeDirectory == -1 && errno == EINTR);
  if (storeDirectory == -1)
    fail(__func__, "unable to open artifact store directory: " +
                       llvm::errnoAsErrorCode().message());
  int socket = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (socket == -1)
    fail(__func__, "unable to create stored-object socket: " +
                       llvm::errnoAsErrorCode().message());
  const std::string socketPath =
      "/proc/self/fd/" + std::to_string(storeDirectory) + "/" + objectName;
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  require(__func__, socketPath.size() < sizeof(address.sun_path),
          "stored-object socket alias is too long");
  std::copy(socketPath.begin(), socketPath.end(), address.sun_path);
  if (::bind(socket, reinterpret_cast<const sockaddr *>(&address),
             sizeof(address)) == -1)
    fail(__func__, "unable to bind stored-object socket: " +
                       llvm::errnoAsErrorCode().message());
  expectErrorContains(__func__, store.get(testSchema, identity),
                      "artifact_store_corruption");
  if (std::error_code error = llvm::sys::fs::closeFile(socket))
    fail(__func__, "unable to close stored-object socket: " + error.message());
  int unlinkResult;
  do {
    unlinkResult = ::unlinkat(storeDirectory, objectName.c_str(), 0);
  } while (unlinkResult == -1 && errno == EINTR);
  if (unlinkResult == -1)
    fail(__func__, "unable to remove stored-object socket: " +
                       llvm::errnoAsErrorCode().message());
  if (std::error_code error = llvm::sys::fs::closeFile(storeDirectory))
    fail(__func__,
         "unable to close artifact store directory: " + error.message());
}

void malformedStoredPreimagesAreRejected() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const CanonicalSemanticBytes bytes = semantic({0x53});
  const ArtifactIdentity identity = finalizeArtifactIdentity(testSchema, bytes);
  const std::string path = objectPath(directory.path(), identity);
  const std::vector<std::uint8_t> valid = expectedPreimage(testSchema, bytes);
  constexpr std::size_t identityDomainSize =
      sizeof("loom.artifact.identity.v1\0") - 1;
  constexpr std::size_t schemaLengthSize = 4;
  constexpr std::size_t schemaVersionSize = 8;
  constexpr std::size_t semanticLengthSize = 8;
  const std::size_t semanticLengthOffset =
      identityDomainSize + schemaLengthSize + testSchema.identity.size() +
      schemaVersionSize;

  std::vector<std::uint8_t> truncatedU32(
      valid.begin(), valid.begin() + identityDomainSize + 3);
  std::vector<std::uint8_t> truncatedU64(valid.begin(),
                                         valid.begin() + semanticLengthOffset +
                                             semanticLengthSize - 1);
  std::vector<std::uint8_t> oversizedSchemaLength = valid;
  std::fill_n(oversizedSchemaLength.begin() + identityDomainSize,
              schemaLengthSize, 0xff);
  std::vector<std::uint8_t> oversizedSemanticLength = valid;
  std::fill_n(oversizedSemanticLength.begin() + semanticLengthOffset,
              semanticLengthSize, 0xff);
  std::vector<std::uint8_t> truncatedSemanticBytes = valid;
  truncatedSemanticBytes.pop_back();

  std::vector<std::vector<std::uint8_t>> malformed = {
      {'b', 'a', 'd'},
      std::move(truncatedU32),
      std::move(truncatedU64),
      std::move(oversizedSchemaLength),
      std::move(oversizedSemanticLength),
      std::move(truncatedSemanticBytes),
  };

  for (const std::vector<std::uint8_t> &preimage : malformed) {
    writeFile(__func__, path, preimage);
    expectErrorContains(__func__, store.get(testSchema, identity),
                        "artifact_store_corruption");
  }
}

void wrongKeyValidPreimageIsRejected() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const CanonicalSemanticBytes expectedBytes = semantic({0x54});
  const ArtifactIdentity identity =
      finalizeArtifactIdentity(testSchema, expectedBytes);
  const std::vector<std::uint8_t> wrongPreimage =
      expectedPreimage(otherSchema, semantic({0x55}));
  writeFile(__func__, objectPath(directory.path(), identity), wrongPreimage);

  expectErrorContains(__func__, store.get(testSchema, identity),
                      "artifact_store_corruption");
}

void tamperedSemanticBytesAreRejected() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const CanonicalSemanticBytes bytes = semantic({0x56, 0x57});
  const ArtifactIdentity identity =
      takeExpected(__func__, store.put(testSchema, bytes));
  const std::string path = objectPath(directory.path(), identity);
  std::vector<std::uint8_t> tampered = readFile(__func__, path);
  tampered.back() ^= 0xff;
  writeFile(__func__, path, tampered);

  expectErrorContains(__func__, store.get(testSchema, identity),
                      "artifact_store_corruption");
}

void resolvedConfigUsesArtifactFinalization() {
  require(__func__,
          ResolvedConfig::artifactSchema.identity ==
              llvm::StringRef("loom.config.resolved"),
          "ResolvedConfig schema identity changed");
  require(__func__,
          ResolvedConfig::artifactSchema.version == SchemaVersion{7, 0},
          "ResolvedConfig schema version changed");

  const ResolvedConfig config = defaultResolvedConfig();
  const CanonicalSemanticBytes bytes = canonicalResolvedConfigBytes(config);
  require(__func__,
          std::string(bytes.bytes().begin(), bytes.bytes().end()) ==
              canonicalResolvedConfigJson(config),
          "ResolvedConfig canonical bytes diverged from canonical JSON");
  require(__func__,
          resolvedConfigIdentity(config) ==
              finalizeArtifactIdentity(ResolvedConfig::artifactSchema, bytes),
          "ResolvedConfig did not use the common finalizer");

  ResolvedConfig changed = config;
  ++changed.dse.structuredOwnership.scopeExpansionLimit;
  require(__func__,
          resolvedConfigIdentity(config) != resolvedConfigIdentity(changed),
          "ResolvedConfig semantic change did not affect identity");
}

// Opaque component-view fixture bytes. The descriptor is never parsed, so it
// deliberately carries an interior NUL and a high byte, and the two inputs have
// different lengths to pin both framed length fields. The digest bytes are the
// independently computed SHA-256 of the framed preimage for those two inputs.
constexpr std::array<std::uint8_t, 6> viewDescriptorBytes = {0x6c, 0x6f, 0x6f,
                                                             0x6d, 0x00, 0xff};
constexpr std::array<std::uint8_t, 5> viewCanonicalBytes = {0x00, 0x7f, 0x80,
                                                            0xff, 0x01};
constexpr std::array<std::uint8_t, ComponentViewDigest::byteSize>
    knownViewDigestBytes = {0x91, 0xc1, 0x26, 0xfb, 0x00, 0x4e, 0xd9, 0x22,
                            0xe7, 0x68, 0x3c, 0x85, 0x73, 0xad, 0x7d, 0x2d,
                            0xe9, 0x10, 0x17, 0x8d, 0xfd, 0x26, 0x81, 0x1e,
                            0xdc, 0x8f, 0xbb, 0x2a, 0x72, 0x0d, 0xc1, 0xc2};

void componentViewDigestMatchesKnownVector() {
  const ComponentViewDigest digest =
      takeExpected(__func__, computeComponentViewDigest(viewDescriptorBytes,
                                                        viewCanonicalBytes));
  require(__func__, digest.bytes() == knownViewDigestBytes,
          "known component view digest changed: " +
              llvm::toHex(digest.bytes(), true));
  require(__func__,
          digest == takeExpected(__func__,
                                 computeComponentViewDigest(
                                     viewDescriptorBytes, viewCanonicalBytes)),
          "identical component view digest input was not deterministic");
  static_assert(!std::is_default_constructible_v<ComponentViewDigest>);
  static_assert(!std::is_same_v<ComponentViewDigest, ArtifactIdentity>);
}

void componentViewDigestFollowsSourceBytes() {
  const ComponentViewDigest digest =
      takeExpected(__func__, computeComponentViewDigest(viewDescriptorBytes,
                                                        viewCanonicalBytes));

  std::array<std::uint8_t, 6> changedDescriptor = viewDescriptorBytes;
  changedDescriptor.back() ^= 0x01;
  require(__func__,
          takeExpected(__func__, computeComponentViewDigest(
                                     changedDescriptor, viewCanonicalBytes)) !=
              digest,
          "schema descriptor bytes did not affect the component view digest");

  std::array<std::uint8_t, 5> changedView = viewCanonicalBytes;
  changedView.back() ^= 0x01;
  require(__func__,
          takeExpected(__func__, computeComponentViewDigest(viewDescriptorBytes,
                                                            changedView)) !=
              digest,
          "canonical view bytes did not affect the component view digest");
}

void componentViewDigestValidatesSuppliedRawDigest() {
  const ComponentViewDigest supplied = takeExpected(
      __func__, ComponentViewDigest::fromBytes(knownViewDigestBytes));
  require(__func__,
          llvm::toString(validateComponentViewDigest(
                             viewDescriptorBytes, viewCanonicalBytes, supplied))
              .empty(),
          "exact supplied component view digest was rejected");

  std::array<std::uint8_t, ComponentViewDigest::byteSize> staleDigestBytes =
      knownViewDigestBytes;
  staleDigestBytes.back() ^= 0x01;
  const std::string message = llvm::toString(validateComponentViewDigest(
      viewDescriptorBytes, viewCanonicalBytes,
      takeExpected(__func__,
                   ComponentViewDigest::fromBytes(staleDigestBytes))));
  require(__func__,
          llvm::StringRef(message).contains("component_view_digest_mismatch"),
          "unexpected component view digest validation result: " + message);

  expectErrorContains(
      __func__,
      ComponentViewDigest::fromBytes(std::vector<std::uint8_t>(31, 0)),
      "component view digest requires exactly 32 bytes");
  expectErrorContains(
      __func__,
      ComponentViewDigest::fromBytes(std::vector<std::uint8_t>(33, 0)),
      "component view digest requires exactly 32 bytes");
}

void componentViewDigestRejectsUnrepresentableDescriptorLength() {
  if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
    // The framed u32 descriptor length is rejected before any descriptor byte
    // is read, so this reference is never dereferenced and nothing is
    // allocated for it.
    std::uint8_t unreadByte = 0;
    const llvm::ArrayRef<std::uint8_t> unrepresentableDescriptor(
        &unreadByte,
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) +
            1);
    expectErrorContains(__func__,
                        computeComponentViewDigest(unrepresentableDescriptor,
                                                   viewCanonicalBytes),
                        "component_view_digest_descriptor_too_large");

    const std::string message = llvm::toString(validateComponentViewDigest(
        unrepresentableDescriptor, viewCanonicalBytes,
        takeExpected(__func__,
                     ComponentViewDigest::fromBytes(knownViewDigestBytes))));
    require(__func__,
            llvm::StringRef(message).contains(
                "component_view_digest_descriptor_too_large"),
            "unrepresentable descriptor length was not propagated: " + message);
  }
}

void componentViewDigestHexSpellingIsCanonical() {
  const ComponentViewDigest digest = takeExpected(
      __func__, ComponentViewDigest::fromBytes(knownViewDigestBytes));
  const std::string spelling = formatComponentViewDigestHex(digest);
  require(__func__, spelling.size() == ComponentViewDigest::byteSize * 2,
          "component view digest hex must be exactly 64 characters");
  require(__func__,
          std::all_of(spelling.begin(), spelling.end(),
                      [](char character) {
                        return (character >= '0' && character <= '9') ||
                               (character >= 'a' && character <= 'f');
                      }),
          "component view digest hex must be lowercase hexadecimal");
  require(__func__,
          spelling == "91c126fb004ed922e7683c8573ad7d2de910178dfd26811edc8fbb"
                      "2a720dc1c2",
          "known component view digest hex spelling changed: " + spelling);
  require(__func__,
          takeExpected(__func__, parseComponentViewDigestHex(spelling)) ==
              digest,
          "component view digest hex did not round-trip");

  expectErrorContains(__func__,
                      parseComponentViewDigestHex(std::string(63, '0')),
                      "exactly 64 lowercase hexadecimal");
  expectErrorContains(__func__,
                      parseComponentViewDigestHex(std::string(65, '0')),
                      "exactly 64 lowercase hexadecimal");
  expectErrorContains(__func__,
                      parseComponentViewDigestHex(std::string(63, '0') + "A"),
                      "lowercase hexadecimal");
  expectErrorContains(__func__,
                      parseComponentViewDigestHex(std::string(63, '0') + "g"),
                      "lowercase hexadecimal");
}

} // namespace

int main() {
  schemaVersionTextCodecIsCanonical();
  missingStoreRootIsRejected();
  finalizerMatchesKnownEnvelopeAndDigest();
  identityBoundariesRejectInvalidValues();
  concurrentIdenticalPublishDeduplicates();
  postInsertionIoFailureKeepsCompleteObjectAndAllowsRetry();
  existingWrongOrCorruptObjectIsRejected();
  existingSymlinkObjectIsRejected();
  storedCanonicalBytesRoundTrip();
  storedSchemaMismatchIsRejected();
  missingStoredObjectIsRejected();
  nonRegularStoredObjectsAreRejectedOnRead();
  malformedStoredPreimagesAreRejected();
  wrongKeyValidPreimageIsRejected();
  tamperedSemanticBytesAreRejected();
  resolvedConfigUsesArtifactFinalization();
  componentViewDigestMatchesKnownVector();
  componentViewDigestFollowsSourceBytes();
  componentViewDigestValidatesSuppliedRawDigest();
  componentViewDigestRejectsUnrepresentableDescriptorLength();
  componentViewDigestHexSpellingIsCanonical();
  return 0;
}
