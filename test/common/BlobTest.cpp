#include "Common/Artifact.h"
#include "Common/BlobDigest.h"
#include "Common/BlobStore.h"
#include "Common/ComponentViewDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace loom;

namespace {

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

std::vector<std::uint8_t> blob(std::initializer_list<std::uint8_t> bytes) {
  return std::vector<std::uint8_t>(bytes);
}

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(const char *test) : test_(test) {
    llvm::SmallString<128> path;
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory("loom-blob-test", path))
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
    fail(test, "unable to inspect blob store: " + error.message());
  while (iterator != end) {
    if (llvm::sys::fs::is_regular_file(iterator->path()))
      paths.push_back(iterator->path());
    iterator.increment(error);
    if (error)
      fail(test, "unable to inspect blob store: " + error.message());
  }
  return paths;
}

std::vector<std::uint8_t> readFile(const char *test, llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false);
  if (std::error_code error = buffer.getError())
    fail(test, "unable to read stored blob: " + error.message());
  llvm::StringRef contents = (*buffer)->getBuffer();
  return std::vector<std::uint8_t>(contents.bytes_begin(),
                                   contents.bytes_end());
}

void writeFile(const char *test, llvm::StringRef path,
               llvm::ArrayRef<std::uint8_t> bytes) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error);
  if (error)
    fail(test, "unable to replace stored blob: " + error.message());
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.close();
  if (output.has_error()) {
    output.clear_error();
    fail(test, "unable to replace stored blob");
  }
}

std::string objectPath(llvm::StringRef root, const BlobDigest &digest) {
  llvm::SmallString<128> path(root);
  llvm::sys::path::append(path, formatBlobDigestHex(digest));
  return path.str().str();
}

// Fixed logical-byte anchors. The nonempty digest is the independently
// computed SHA-256 of the three bytes below; the empty digest is the
// well-known SHA-256 of zero input bytes.
constexpr std::array<std::uint8_t, 3> knownVectorBytes = {0x00, 0x10, 0xff};
constexpr char knownVectorDigestText[] =
    "a7da86bc189688e4070e297a0d03f4e3c4d721044d192b02af7ef0688803425b";
constexpr char emptyDigestText[] =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

void digestMatchesKnownVectors() {
  const BlobDigest digest = computeBlobDigest(knownVectorBytes);
  require(__func__, formatBlobDigestHex(digest) == knownVectorDigestText,
          "known blob digest changed: " + formatBlobDigestHex(digest));
  require(__func__, digest == computeBlobDigest(knownVectorBytes),
          "identical blob digest input was not deterministic");

  const BlobDigest empty = computeBlobDigest({});
  require(__func__, formatBlobDigestHex(empty) == emptyDigestText,
          "zero-length blob digest changed: " + formatBlobDigestHex(empty));

  constexpr std::array<std::uint8_t, 3> changedBytes = {0x00, 0x10, 0xfe};
  require(__func__, computeBlobDigest(changedBytes) != digest,
          "logical bytes did not affect the blob digest");
}

void digestBoundariesRejectInvalidValues() {
  static_assert(!std::is_default_constructible_v<BlobDigest>);
  static_assert(!std::is_same_v<BlobDigest, ArtifactIdentity>);
  static_assert(!std::is_same_v<BlobDigest, ComponentViewDigest>);

  expectErrorContains(__func__,
                      BlobDigest::fromBytes(std::vector<std::uint8_t>(31, 0)),
                      "blob digest requires exactly 32 bytes");
  expectErrorContains(__func__,
                      BlobDigest::fromBytes(std::vector<std::uint8_t>(33, 0)),
                      "blob digest requires exactly 32 bytes");

  const std::vector<std::uint8_t> zeroBytes(BlobDigest::byteSize, 0);
  const BlobDigest zero =
      takeExpected(__func__, BlobDigest::fromBytes(zeroBytes));
  require(__func__, formatBlobDigestHex(zero) == std::string(64, '0'),
          "all-zero BlobDigest was not preserved as an ordinary value");
}

void textCodecIsStrictAndCanonical() {
  const BlobDigest digest = computeBlobDigest(knownVectorBytes);
  const std::string text = formatBlobDigestHex(digest);
  require(__func__, text.size() == BlobDigest::byteSize * 2,
          "blob digest text width changed");
  const BlobDigest parsed = takeExpected(__func__, parseBlobDigestHex(text));
  require(__func__, parsed == digest, "blob digest text did not round trip");
  require(__func__, formatBlobDigestHex(parsed) == text,
          "blob digest text round trip was not canonical");

  const std::vector<std::string> invalidSpellings = {
      "",
      std::string(63, '0'),
      std::string(65, '0'),
      std::string(64, 'A'),
      std::string(64, 'F'),
      std::string(63, '0') + "g",
  };
  for (const std::string &spelling : invalidSpellings)
    expectErrorContains(__func__, parseBlobDigestHex(spelling), "blob digest");

  const BlobDigest zero =
      takeExpected(__func__, parseBlobDigestHex(std::string(64, '0')));
  require(__func__, formatBlobDigestHex(zero) == std::string(64, '0'),
          "all-zero blob digest text did not parse");
}

void missingStoreRootIsRejected() {
  TemporaryDirectory parent(__func__);
  llvm::SmallString<128> missingParent(parent.path());
  llvm::sys::path::append(missingParent, "missing");
  llvm::SmallString<128> root(missingParent);
  llvm::sys::path::append(root, "store");

  BlobStore store(root);
  expectErrorContains(__func__, store.put(blob({0x01})), "blob_store_io");
  expectErrorContains(__func__, store.get(computeBlobDigest(blob({0x01}))),
                      "blob_store_io");
  require(__func__, !llvm::sys::fs::exists(missingParent),
          "BlobStore created part of its missing root chain");
}

void storedLogicalBytesRoundTrip() {
  TemporaryDirectory directory(__func__);
  BlobStore store(directory.path());

  const std::vector<std::uint8_t> bytes = blob({0x00, 0x7f, 0xff});
  const BlobDigest digest = takeExpected(__func__, store.put(bytes));
  require(__func__, digest == computeBlobDigest(bytes),
          "BlobStore returned a different digest");
  const std::vector<std::uint8_t> loaded =
      takeExpected(__func__, store.get(digest));
  require(__func__, loaded == bytes,
          "BlobStore returned different logical bytes");
  require(__func__,
          takeExpected(__func__, store.get(digest, bytes.size())) == bytes,
          "bounded BlobStore read changed logical bytes");
  require(__func__,
          takeExpected(__func__, store.verify(digest, bytes.size())) ==
              bytes.size(),
          "bounded BlobStore verification changed the logical-byte count");
  expectErrorContains(__func__, store.get(digest, bytes.size() - 1),
                      "blob_store_size_limit");
  expectErrorContains(__func__, store.verify(digest, bytes.size() - 1),
                      "blob_store_size_limit");

  const BlobDigest emptyDigest = takeExpected(__func__, store.put({}));
  require(__func__, emptyDigest == computeBlobDigest({}),
          "zero-length blob digest mismatch");
  const std::vector<std::uint8_t> loadedEmpty =
      takeExpected(__func__, store.get(emptyDigest));
  require(__func__, loadedEmpty.empty(), "zero-length blob did not round trip");

  const std::vector<std::string> files =
      regularFiles(__func__, directory.path());
  require(__func__, files.size() == 2,
          "BlobStore did not publish exactly one object per digest");
  require(__func__,
          readFile(__func__, objectPath(directory.path(), digest)) == bytes,
          "stored object is not the exact logical bytes");
}

void missingStoredObjectIsRejected() {
  TemporaryDirectory directory(__func__);
  BlobStore store(directory.path());
  const BlobDigest digest = computeBlobDigest(blob({0x51}));
  expectErrorContains(__func__, store.get(digest), "blob_store_missing");
}

void equalBytesDeduplicate() {
  TemporaryDirectory directory(__func__);
  BlobStore store(directory.path());
  const std::vector<std::uint8_t> bytes = blob({0x01, 0x02, 0x03});
  const BlobDigest expected = computeBlobDigest(bytes);

  require(__func__, takeExpected(__func__, store.put(bytes)) == expected,
          "first publication returned a different digest");
  require(__func__, takeExpected(__func__, store.put(bytes)) == expected,
          "deduplicated publication returned a different digest");
  const std::vector<std::string> files =
      regularFiles(__func__, directory.path());
  require(__func__, files.size() == 1,
          "equal-byte publication did not deduplicate");
  require(__func__, readFile(__func__, files.front()) == bytes,
          "deduplicated object has unexpected contents");
}

void verifiedImportIsBoundedAndContentAddressed() {
  TemporaryDirectory sourceDirectory(__func__);
  TemporaryDirectory destinationDirectory(__func__);
  BlobStore source(sourceDirectory.path());
  BlobStore destination(destinationDirectory.path());

  std::vector<std::uint8_t> bytes(192 * 1024 + 17);
  for (std::size_t index = 0; index < bytes.size(); ++index)
    bytes[index] = static_cast<std::uint8_t>((index * 131) & 0xff);
  const BlobDigest digest = takeExpected(__func__, source.put(bytes));

  expectErrorContains(
      __func__, destination.importVerified(digest, source, bytes.size() - 1),
      "blob_store_size_limit");
  require(__func__, regularFiles(__func__, destinationDirectory.path()).empty(),
          "rejected import published a destination object");

  require(__func__,
          takeExpected(__func__, destination.importVerified(
                                     digest, source, bytes.size())) ==
              bytes.size(),
          "verified import returned a different logical-byte count");
  require(__func__, takeExpected(__func__, destination.get(digest)) == bytes,
          "verified import changed logical bytes");
  require(__func__,
          takeExpected(__func__, destination.importVerified(
                                     digest, source, bytes.size())) ==
              bytes.size(),
          "deduplicated verified import changed the logical-byte count");
  require(__func__, regularFiles(__func__, destinationDirectory.path()).size() ==
                        1,
          "verified import did not deduplicate by exact digest");

  std::vector<std::uint8_t> tampered = bytes;
  tampered.back() ^= 0xff;
  writeFile(__func__, objectPath(sourceDirectory.path(), digest), tampered);
  expectErrorContains(
      __func__, destination.importVerified(digest, source, bytes.size()),
      "blob_store_corruption");
  require(__func__, takeExpected(__func__, destination.get(digest)) == bytes,
          "failed import changed the published destination object");
}

void concurrentIdenticalPublishDeduplicates() {
  TemporaryDirectory directory(__func__);
  BlobStore store(directory.path());
  const std::vector<std::uint8_t> bytes = blob({0x01, 0x02, 0x03});
  const BlobDigest expected = computeBlobDigest(bytes);

  constexpr unsigned threadCount = 8;
  std::atomic<bool> start{false};
  std::vector<std::string> digests(threadCount);
  std::vector<std::string> errors(threadCount);
  std::vector<std::thread> threads;
  threads.reserve(threadCount);
  for (unsigned index = 0; index < threadCount; ++index) {
    threads.emplace_back([&, index] {
      while (!start.load(std::memory_order_acquire))
        std::this_thread::yield();
      auto stored = store.put(bytes);
      if (!stored) {
        errors[index] = llvm::toString(stored.takeError());
        return;
      }
      digests[index] = formatBlobDigestHex(*stored);
    });
  }
  start.store(true, std::memory_order_release);
  for (std::thread &thread : threads)
    thread.join();

  const std::string expectedText = formatBlobDigestHex(expected);
  for (unsigned index = 0; index < threadCount; ++index) {
    require(__func__, errors[index].empty(), errors[index]);
    require(__func__, digests[index] == expectedText,
            "concurrent publisher returned a different digest");
  }
  const std::vector<std::string> files =
      regularFiles(__func__, directory.path());
  require(__func__, files.size() == 1,
          "concurrent identical publication did not deduplicate");
  require(__func__, readFile(__func__, files.front()) == bytes,
          "deduplicated object has unexpected contents");
}

void tamperedStoredObjectIsRejected() {
  TemporaryDirectory directory(__func__);
  BlobStore store(directory.path());
  const std::vector<std::uint8_t> bytes = blob({0x56, 0x57});
  const BlobDigest digest = takeExpected(__func__, store.put(bytes));
  const std::string path = objectPath(directory.path(), digest);

  std::vector<std::uint8_t> tampered = readFile(__func__, path);
  tampered.back() ^= 0xff;
  writeFile(__func__, path, tampered);

  expectErrorContains(__func__, store.get(digest), "blob_store_corruption");
  expectErrorContains(__func__, store.put(bytes), "blob_store_corruption");
  require(__func__, readFile(__func__, path) == tampered,
          "BlobStore overwrote or repaired a tampered object");
}

void wrongKeyObjectIsRejected() {
  TemporaryDirectory directory(__func__);
  BlobStore store(directory.path());
  const std::vector<std::uint8_t> bytes = blob({0x54});
  const BlobDigest digest = computeBlobDigest(bytes);
  const std::vector<std::uint8_t> otherBytes = blob({0x55});
  writeFile(__func__, objectPath(directory.path(), digest), otherBytes);

  expectErrorContains(__func__, store.get(digest), "blob_store_corruption");
  expectErrorContains(__func__, store.put(bytes), "blob_store_corruption");
  require(__func__,
          readFile(__func__, objectPath(directory.path(), digest)) ==
              otherBytes,
          "BlobStore overwrote or repaired a wrong-key object");
}

void nonRegularStoredObjectsAreRejected() {
  TemporaryDirectory directory(__func__);
  BlobStore store(directory.path());
  const std::vector<std::uint8_t> bytes = blob({0x40, 0x41});
  const BlobDigest digest = computeBlobDigest(bytes);
  const std::string path = objectPath(directory.path(), digest);

  llvm::SmallString<128> targetPath(directory.path());
  llvm::sys::path::append(targetPath, ".symlink-target");
  writeFile(__func__, targetPath, bytes);
  if (std::error_code error = llvm::sys::fs::create_symlink(targetPath, path))
    fail(__func__,
         "unable to create stored-object symlink: " + error.message());
  expectErrorContains(__func__, store.get(digest), "blob_store_corruption");
  expectErrorContains(__func__, store.put(bytes), "blob_store_corruption");
  require(__func__, llvm::sys::fs::is_symlink_file(path),
          "BlobStore replaced an existing symlink object");

  if (std::error_code error = llvm::sys::fs::remove(path))
    fail(__func__,
         "unable to remove stored-object symlink: " + error.message());
  if (std::error_code error = llvm::sys::fs::create_directory(path))
    fail(__func__,
         "unable to create stored-object directory: " + error.message());
  expectErrorContains(__func__, store.get(digest), "blob_store_corruption");
  expectErrorContains(__func__, store.put(bytes), "blob_store_corruption");
}

} // namespace

int main() {
  digestMatchesKnownVectors();
  digestBoundariesRejectInvalidValues();
  textCodecIsStrictAndCanonical();
  missingStoreRootIsRejected();
  storedLogicalBytesRoundTrip();
  missingStoredObjectIsRejected();
  equalBytesDeduplicate();
  verifiedImportIsBoundedAndContentAddressed();
  concurrentIdenticalPublishDeduplicates();
  tamperedStoredObjectIsRejected();
  wrongKeyObjectIsRejected();
  nonRegularStoredObjectsAreRejected();
  return 0;
}
