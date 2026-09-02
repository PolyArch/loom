#include "Common/BlobStore.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#if !defined(__linux__)
#error "BlobStore durable publication currently requires Linux"
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace loom {
namespace {

struct OpenedBlobObject {
  llvm::sys::fs::file_status status;
  std::unique_ptr<llvm::MemoryBuffer> contents;

  /// Exact stored bytes, owned by contents; never a second representation.
  llvm::ArrayRef<std::uint8_t> logicalBytes() const {
    const llvm::StringRef bytes = contents->getBuffer();
    return llvm::ArrayRef<std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size());
  }
};

llvm::Error storeError(llvm::StringRef code, const llvm::Twine &detail) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 code + ": " + detail);
}

llvm::Error storeErrno(llvm::StringRef code, const llvm::Twine &detail) {
  const std::error_code error = llvm::errnoAsErrorCode();
  return storeError(code, detail + ": " + error.message());
}

llvm::Expected<llvm::sys::fs::file_status>
regularFileStatus(int file, llvm::StringRef nonRegularErrorCode,
                  llvm::StringRef description) {
  llvm::sys::fs::file_status status;
  if (std::error_code error = llvm::sys::fs::status(file, status))
    return storeError("blob_store_io", llvm::Twine("unable to inspect ") +
                                           description + ": " +
                                           error.message());
  if (!llvm::sys::fs::is_regular_file(status))
    return storeError(nonRegularErrorCode,
                      llvm::Twine(description) + " is not a regular file");
  return status;
}

llvm::Expected<OpenedBlobObject>
readOpenedObject(int file, llvm::StringRef description,
                 llvm::StringRef objectErrorCode,
                 std::uint64_t maximumLogicalBytes =
                     std::numeric_limits<std::uint64_t>::max()) {
  auto status = regularFileStatus(file, objectErrorCode, description);
  if (!status)
    return status.takeError();
  if (status->getSize() > maximumLogicalBytes)
    return storeError("blob_store_size_limit",
                      llvm::Twine(description) +
                          " exceeds the caller-owned logical-byte bound");

  auto buffer = llvm::MemoryBuffer::getOpenFile(file, description,
                                                status->getSize(), false, true);
  if (std::error_code error = buffer.getError())
    return storeError("blob_store_io", llvm::Twine("unable to read ") +
                                           description + ": " +
                                           error.message());
  return OpenedBlobObject{*status, std::move(*buffer)};
}

llvm::Error validateStoredObject(const OpenedBlobObject &object,
                                 llvm::StringRef description,
                                 const BlobDigest &expectedDigest,
                                 llvm::StringRef objectErrorCode) {
  if (computeBlobDigest(object.logicalBytes()) != expectedDigest)
    return storeError(objectErrorCode, llvm::Twine(description) +
                                           " does not match its derived key");
  return llvm::Error::success();
}

llvm::Error closeFile(int &file, llvm::StringRef description) {
  if (std::error_code error = llvm::sys::fs::closeFile(file))
    return storeError("blob_store_io", llvm::Twine("unable to close ") +
                                           description + ": " +
                                           error.message());
  return llvm::Error::success();
}

llvm::Expected<int> openStoredObject(int directory,
                                     llvm::StringRef objectName) {
  const std::string name = objectName.str();
  int handle;
  do {
    handle = ::openat(directory, name.c_str(), O_PATH | O_CLOEXEC | O_NOFOLLOW);
  } while (handle == -1 && errno == EINTR);
  if (handle == -1) {
    if (errno == ENOENT)
      return storeError("blob_store_missing", "stored object is missing");
    return storeErrno("blob_store_io", "unable to open stored object handle");
  }
  llvm::scope_exit closeHandle([&] {
    if (handle != -1)
      llvm::consumeError(closeFile(handle, "stored object handle"));
  });

  auto status =
      regularFileStatus(handle, "blob_store_corruption", "stored object");
  if (!status)
    return status.takeError();

  const std::string handlePath = "/proc/self/fd/" + std::to_string(handle);
  int file;
  do {
    file = ::open(handlePath.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
  } while (file == -1 && errno == EINTR);
  if (file == -1)
    return storeErrno("blob_store_io",
                      "unable to open stored object for reading");
  llvm::scope_exit closeFileOnFailure([&] {
    if (file != -1)
      llvm::consumeError(closeFile(file, "stored object"));
  });

  if (llvm::Error error = closeFile(handle, "stored object handle"))
    return std::move(error);
  closeHandle.release();
  closeFileOnFailure.release();
  return file;
}

llvm::Expected<int> openStoreDirectory(llvm::StringRef root) {
  const std::string path = root.str();
  int directory;
  do {
    directory =
        ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
  } while (directory == -1 && errno == EINTR);
  if (directory == -1)
    return storeErrno("blob_store_io",
                      "unable to open required store root directory");
  return directory;
}

llvm::Expected<int> openAnonymousTemporaryObject(int directory) {
  int file;
  do {
    file = ::openat(directory, ".", O_TMPFILE | O_RDWR | O_CLOEXEC,
                    S_IRUSR | S_IWUSR);
  } while (file == -1 && errno == EINTR);
  if (file == -1)
    return storeErrno("blob_store_io",
                      "unable to create anonymous temporary object");
  return file;
}

llvm::Error verifyStoreDirectoryBinding(int directory, llvm::StringRef root) {
  struct stat expected{};
  if (::fstat(directory, &expected) != 0)
    return storeErrno("blob_store_io", "unable to inspect store directory");
  auto currentOrError = openStoreDirectory(root);
  if (!currentOrError)
    return currentOrError.takeError();
  int current = *currentOrError;
  llvm::scope_exit closeCurrent([&] {
    if (current != -1)
      llvm::consumeError(closeFile(current, "current store directory"));
  });
  struct stat observed{};
  if (::fstat(current, &observed) != 0)
    return storeErrno("blob_store_io",
                      "unable to inspect current store directory");
  if (expected.st_dev != observed.st_dev || expected.st_ino != observed.st_ino)
    return storeError("blob_store_root_changed",
                      "store root changed during publication");
  if (llvm::Error error = closeFile(current, "current store directory"))
    return error;
  closeCurrent.release();
  return llvm::Error::success();
}

llvm::Expected<OpenedBlobObject>
readStoredObject(llvm::StringRef root, const BlobDigest &digest,
                 std::uint64_t maximumLogicalBytes) {
  auto directoryOrError = openStoreDirectory(root);
  if (!directoryOrError)
    return directoryOrError.takeError();
  int directory = *directoryOrError;
  llvm::scope_exit closeDirectory([&] {
    if (directory != -1)
      llvm::consumeError(closeFile(directory, "store directory"));
  });

  const std::string objectName = formatBlobDigestHex(digest);
  auto fileOrError = openStoredObject(directory, objectName);
  if (!fileOrError)
    return fileOrError.takeError();
  int file = *fileOrError;
  llvm::scope_exit closeObject([&] {
    if (file != -1)
      llvm::consumeError(closeFile(file, "stored object"));
  });

  auto object = readOpenedObject(file, "stored object", "blob_store_corruption",
                                 maximumLogicalBytes);
  if (!object)
    return object.takeError();
  if (llvm::Error error = validateStoredObject(*object, "stored object", digest,
                                               "blob_store_corruption"))
    return std::move(error);
  if (llvm::Error error = closeFile(file, "stored object"))
    return std::move(error);
  closeObject.release();
  if (llvm::Error error = closeFile(directory, "store directory"))
    return std::move(error);
  closeDirectory.release();
  return std::move(*object);
}

llvm::Error syncFile(int file, llvm::StringRef description) {
  int result;
  do {
    result = ::fsync(file);
  } while (result == -1 && errno == EINTR);
  if (result == -1)
    return storeErrno("blob_store_io",
                      llvm::Twine("unable to sync ") + description);
  return llvm::Error::success();
}

std::error_code publishNoReplace(int source, int directory,
                                 llvm::StringRef objectName) {
  // Publish the validated inode rather than resolving its temporary path.
  const std::string sourcePath = "/proc/self/fd/" + std::to_string(source);
  const std::string name = objectName.str();
  int result;
  do {
    result = ::linkat(AT_FDCWD, sourcePath.c_str(), directory, name.c_str(),
                      AT_SYMLINK_FOLLOW);
  } while (result == -1 && errno == EINTR);
  if (result == -1)
    return llvm::errnoAsErrorCode();
  return std::error_code();
}

std::error_code publishAnonymousNoReplace(int source, int directory,
                                          llvm::StringRef objectName) {
  const std::string name = objectName.str();
  int result;
  do {
    result = ::linkat(source, "", directory, name.c_str(), AT_EMPTY_PATH);
  } while (result == -1 && errno == EINTR);
  if (result == -1)
    return llvm::errnoAsErrorCode();
  return std::error_code();
}

llvm::Error discardTemporary(llvm::sys::fs::TempFile &temporary) {
  if (llvm::Error error = temporary.discard())
    return storeError("blob_store_io",
                      llvm::Twine("unable to close and remove temporary ") +
                          "object: " + llvm::toString(std::move(error)));
  return llvm::Error::success();
}

bool sameFileState(const struct stat &lhs, const struct stat &rhs) {
  return lhs.st_dev == rhs.st_dev && lhs.st_ino == rhs.st_ino &&
         lhs.st_mode == rhs.st_mode && lhs.st_size == rhs.st_size &&
         lhs.st_mtim.tv_sec == rhs.st_mtim.tv_sec &&
         lhs.st_mtim.tv_nsec == rhs.st_mtim.tv_nsec &&
         lhs.st_ctim.tv_sec == rhs.st_ctim.tv_sec &&
         lhs.st_ctim.tv_nsec == rhs.st_ctim.tv_nsec;
}

llvm::Expected<std::uint64_t> copyAndVerifyOpenedObject(
    int source, int destination, const BlobDigest &expectedDigest,
    std::uint64_t maximumLogicalBytes, llvm::StringRef objectErrorCode,
    llvm::StringRef description) {
  struct stat before{};
  if (::fstat(source, &before) != 0)
    return storeErrno("blob_store_io",
                      llvm::Twine("unable to inspect ") + description);
  if (!S_ISREG(before.st_mode) || before.st_size < 0)
    return storeError(objectErrorCode,
                      llvm::Twine(description) + " is not a regular file");
  const auto logicalSize = static_cast<std::uint64_t>(before.st_size);
  if (logicalSize > maximumLogicalBytes)
    return storeError("blob_store_size_limit",
                      llvm::Twine(description) +
                          " exceeds the caller-owned logical-byte bound");

  auto digest = BlobDigestBuilder::create();
  if (!digest)
    return digest.takeError();
  std::uint64_t observedSize = 0;
  std::array<std::uint8_t, 64 * 1024> buffer{};
  while (true) {
    const ssize_t amount = ::read(source, buffer.data(), buffer.size());
    if (amount == 0)
      break;
    if (amount < 0) {
      if (errno == EINTR)
        continue;
      return storeErrno("blob_store_io",
                        llvm::Twine("unable to read ") + description);
    }
    const auto count = static_cast<std::size_t>(amount);
    if (llvm::Error error = digest->update(
            llvm::ArrayRef<std::uint8_t>(buffer.data(), count)))
      return std::move(error);
    observedSize += count;

    if (destination != -1) {
      std::size_t offset = 0;
      while (offset < count) {
        const ssize_t written =
            ::write(destination, buffer.data() + offset, count - offset);
        if (written < 0) {
          if (errno == EINTR)
            continue;
          return storeErrno("blob_store_io",
                            "unable to write imported temporary object");
        }
        if (written == 0)
          return storeError("blob_store_io",
                            "short write to imported temporary object");
        offset += static_cast<std::size_t>(written);
      }
    }
  }

  struct stat after{};
  if (::fstat(source, &after) != 0)
    return storeErrno("blob_store_io",
                      llvm::Twine("unable to re-inspect ") + description);
  if (!sameFileState(before, after) || observedSize != logicalSize)
    return storeError(objectErrorCode,
                      llvm::Twine(description) + " changed while copied");
  auto actualDigest = digest->finish();
  if (!actualDigest)
    return actualDigest.takeError();
  if (*actualDigest != expectedDigest)
    return storeError(objectErrorCode, llvm::Twine(description) +
                                           " does not match its derived key");
  return observedSize;
}

llvm::Expected<bool> openedFilesEqual(int lhs, int rhs,
                                      std::uint64_t logicalSize) {
  std::array<std::uint8_t, 64 * 1024> lhsBytes{};
  std::array<std::uint8_t, 64 * 1024> rhsBytes{};
  std::uint64_t offset = 0;
  while (offset < logicalSize) {
    const std::size_t count = static_cast<std::size_t>(
        std::min<std::uint64_t>(lhsBytes.size(), logicalSize - offset));
    ssize_t lhsRead;
    do {
      lhsRead =
          ::pread(lhs, lhsBytes.data(), count, static_cast<off_t>(offset));
    } while (lhsRead < 0 && errno == EINTR);
    if (lhsRead < 0)
      return storeErrno("blob_store_io", "unable to compare imported object");
    ssize_t rhsRead;
    do {
      rhsRead =
          ::pread(rhs, rhsBytes.data(), count, static_cast<off_t>(offset));
    } while (rhsRead < 0 && errno == EINTR);
    if (rhsRead < 0)
      return storeErrno("blob_store_io", "unable to compare existing object");
    if (lhsRead != static_cast<ssize_t>(count) ||
        rhsRead != static_cast<ssize_t>(count))
      return storeError("blob_store_corruption",
                        "object size changed while comparing publication");
    if (!std::equal(lhsBytes.begin(), lhsBytes.begin() + count,
                    rhsBytes.begin()))
      return false;
    offset += count;
  }
  return true;
}

struct OpenedBlobDigest final {
  llvm::sys::fs::file_status status;
  BlobDigest digest;
  std::uint64_t logicalSize = 0;
};

llvm::Expected<OpenedBlobDigest>
digestOpenedObject(int file, llvm::StringRef objectErrorCode,
                   llvm::StringRef description) {
  auto status = regularFileStatus(file, objectErrorCode, description);
  if (!status)
    return status.takeError();
  struct stat before{};
  if (::fstat(file, &before) != 0)
    return storeErrno("blob_store_io",
                      llvm::Twine("unable to inspect ") + description);
  if (!S_ISREG(before.st_mode) || before.st_size < 0)
    return storeError(objectErrorCode,
                      llvm::Twine(description) + " is not a regular file");
  const std::uint64_t logicalSize = static_cast<std::uint64_t>(before.st_size);

  auto digest = BlobDigestBuilder::create();
  if (!digest)
    return digest.takeError();
  std::array<std::uint8_t, 64 * 1024> buffer{};
  std::uint64_t offset = 0;
  while (offset < logicalSize) {
    const std::size_t count = static_cast<std::size_t>(
        std::min<std::uint64_t>(buffer.size(), logicalSize - offset));
    ssize_t amount;
    do {
      amount = ::pread(file, buffer.data(), count, static_cast<off_t>(offset));
    } while (amount < 0 && errno == EINTR);
    if (amount < 0)
      return storeErrno("blob_store_io",
                        llvm::Twine("unable to read ") + description);
    if (amount != static_cast<ssize_t>(count))
      return storeError(objectErrorCode,
                        llvm::Twine(description) +
                            " changed while deriving its digest");
    if (llvm::Error error = digest->update(
            llvm::ArrayRef<std::uint8_t>(buffer.data(), count)))
      return std::move(error);
    offset += count;
  }

  struct stat after{};
  if (::fstat(file, &after) != 0)
    return storeErrno("blob_store_io",
                      llvm::Twine("unable to re-inspect ") + description);
  if (!sameFileState(before, after) || offset != logicalSize)
    return storeError(objectErrorCode,
                      llvm::Twine(description) +
                          " changed while deriving its digest");
  auto value = digest->finish();
  if (!value)
    return value.takeError();
  return OpenedBlobDigest{*status, *value, offset};
}

} // namespace

llvm::Expected<BlobDigest>
BlobStore::put(llvm::ArrayRef<std::uint8_t> logicalBytes) const {
  auto publication =
      putGenerated([&](llvm::raw_ostream &output) -> llvm::Error {
        output.write(reinterpret_cast<const char *>(logicalBytes.data()),
                     logicalBytes.size());
        return llvm::Error::success();
      });
  if (!publication)
    return publication.takeError();
  return publication->digest;
}

llvm::Expected<GeneratedBlobPublication> BlobStore::putGenerated(
    llvm::function_ref<llvm::Error(llvm::raw_ostream &)> writer) const {
  if (!writer)
    return storeError("blob_store_io", "generated object has no writer");

  auto directoryOrError = openStoreDirectory(root_);
  if (!directoryOrError)
    return directoryOrError.takeError();
  int directory = *directoryOrError;
  llvm::scope_exit closeDirectoryOnFailure([&] {
    if (directory != -1)
      llvm::consumeError(closeFile(directory, "store directory"));
  });

  auto temporaryOrError = openAnonymousTemporaryObject(directory);
  if (!temporaryOrError)
    return temporaryOrError.takeError();
  int temporary = *temporaryOrError;
  llvm::scope_exit closeTemporaryOnFailure([&] {
    if (temporary != -1)
      llvm::consumeError(closeFile(temporary, "anonymous temporary object"));
  });

  {
    llvm::raw_fd_ostream output(temporary, false);
    if (llvm::Error error = writer(output))
      return std::move(error);
    output.flush();
    if (std::error_code error = output.error()) {
      output.clear_error();
      return storeError("blob_store_io",
                        llvm::Twine("unable to write temporary object: ") +
                            error.message());
    }
  }

  if (llvm::Error error = syncFile(temporary, "temporary object"))
    return std::move(error);
  auto temporaryObject = digestOpenedObject(temporary, "blob_store_io",
                                            "generated temporary object");
  if (!temporaryObject)
    return temporaryObject.takeError();
  const BlobDigest digest = temporaryObject->digest;
  const std::string objectName = formatBlobDigestHex(digest);

  const std::error_code publishError =
      publishAnonymousNoReplace(temporary, directory, objectName);
  if (!publishError) {
    auto published = openStoredObject(directory, objectName);
    if (!published)
      return published.takeError();
    int publishedFile = *published;
    llvm::scope_exit closePublishedOnFailure([&] {
      if (publishedFile != -1)
        llvm::consumeError(closeFile(publishedFile, "published object"));
    });

    auto publishedStatus = regularFileStatus(
        publishedFile, "blob_store_corruption", "published object");
    if (!publishedStatus)
      return publishedStatus.takeError();
    if (publishedStatus->getUniqueID() != temporaryObject->status.getUniqueID())
      return storeError("blob_store_corruption",
                        "published object is not the validated inode");
    if (llvm::Error error = closeFile(publishedFile, "published object"))
      return std::move(error);
    closePublishedOnFailure.release();
  } else if (publishError == std::errc::file_exists) {
    auto existing = openStoredObject(directory, objectName);
    if (!existing)
      return existing.takeError();
    int existingFile = *existing;
    llvm::scope_exit closeExistingOnFailure([&] {
      if (existingFile != -1)
        llvm::consumeError(closeFile(existingFile, "existing object"));
    });

    auto existingObject = digestOpenedObject(
        existingFile, "blob_store_corruption", "existing object");
    if (!existingObject)
      return existingObject.takeError();
    if (existingObject->digest != digest)
      return storeError("blob_store_corruption",
                        "existing object does not match its derived key");
    if (existingObject->logicalSize != temporaryObject->logicalSize)
      return storeError("blob_digest_collision",
                        "different logical bytes share one digest");
    auto equal =
        openedFilesEqual(temporary, existingFile, temporaryObject->logicalSize);
    if (!equal)
      return equal.takeError();
    if (!*equal)
      return storeError("blob_digest_collision",
                        "different logical bytes share one digest");
    if (llvm::Error error = closeFile(existingFile, "existing object"))
      return std::move(error);
    closeExistingOnFailure.release();
  } else {
    return storeError("blob_store_io",
                      llvm::Twine("unable to publish object: ") +
                          publishError.message());
  }

  if (llvm::Error error = syncFile(directory, "store directory"))
    return std::move(error);
  if (llvm::Error error = verifyStoreDirectoryBinding(directory, root_))
    return std::move(error);
  if (llvm::Error error = closeFile(temporary, "anonymous temporary object"))
    return std::move(error);
  closeTemporaryOnFailure.release();
  if (llvm::Error error = closeFile(directory, "store directory"))
    return std::move(error);
  closeDirectoryOnFailure.release();
  return GeneratedBlobPublication{digest, temporaryObject->logicalSize};
}

llvm::Expected<std::vector<std::uint8_t>>
BlobStore::get(const BlobDigest &digest) const {
  return get(digest, std::numeric_limits<std::uint64_t>::max());
}

llvm::Expected<std::vector<std::uint8_t>>
BlobStore::get(const BlobDigest &digest,
               std::uint64_t maximumLogicalBytes) const {
  auto object = readStoredObject(root_, digest, maximumLogicalBytes);
  if (!object)
    return object.takeError();

  const llvm::ArrayRef<std::uint8_t> validatedBytes = object->logicalBytes();
  return std::vector<std::uint8_t>(validatedBytes.begin(),
                                   validatedBytes.end());
}

llvm::Expected<std::uint64_t>
BlobStore::verify(const BlobDigest &digest) const {
  return verify(digest, std::numeric_limits<std::uint64_t>::max());
}

llvm::Expected<std::uint64_t>
BlobStore::verify(const BlobDigest &digest,
                  std::uint64_t maximumLogicalBytes) const {
  auto object = readStoredObject(root_, digest, maximumLogicalBytes);
  if (!object)
    return object.takeError();
  return object->logicalBytes().size();
}

llvm::Expected<std::uint64_t>
BlobStore::importVerified(const BlobDigest &digest, const BlobStore &source,
                          std::uint64_t maximumLogicalBytes) const {
  auto sourceDirectoryOrError = openStoreDirectory(source.root_);
  if (!sourceDirectoryOrError)
    return sourceDirectoryOrError.takeError();
  int sourceDirectory = *sourceDirectoryOrError;
  llvm::scope_exit closeSourceDirectory([&] {
    if (sourceDirectory != -1)
      llvm::consumeError(closeFile(sourceDirectory, "source store directory"));
  });

  const std::string objectName = formatBlobDigestHex(digest);
  auto sourceFileOrError = openStoredObject(sourceDirectory, objectName);
  if (!sourceFileOrError)
    return sourceFileOrError.takeError();
  int sourceFile = *sourceFileOrError;
  llvm::scope_exit closeSourceFile([&] {
    if (sourceFile != -1)
      llvm::consumeError(closeFile(sourceFile, "source stored object"));
  });

  auto destinationDirectoryOrError = openStoreDirectory(root_);
  if (!destinationDirectoryOrError)
    return destinationDirectoryOrError.takeError();
  int destinationDirectory = *destinationDirectoryOrError;
  llvm::scope_exit closeDestinationDirectory([&] {
    if (destinationDirectory != -1)
      llvm::consumeError(
          closeFile(destinationDirectory, "destination store directory"));
  });

  llvm::SmallString<256> temporaryModel(root_);
  llvm::sys::path::append(temporaryModel, ".blob-%%%%%%");
  auto temporaryOrError = llvm::sys::fs::TempFile::create(
      temporaryModel, llvm::sys::fs::owner_read | llvm::sys::fs::owner_write);
  if (!temporaryOrError)
    return storeError("blob_store_io",
                      llvm::Twine("unable to create temporary object: ") +
                          llvm::toString(temporaryOrError.takeError()));
  llvm::sys::fs::TempFile temporary = std::move(*temporaryOrError);
  llvm::scope_exit discardTemporaryOnFailure(
      [&] { llvm::consumeError(temporary.discard()); });

  auto logicalSize = copyAndVerifyOpenedObject(
      sourceFile, temporary.FD, digest, maximumLogicalBytes,
      "blob_store_corruption", "source stored object");
  if (!logicalSize)
    return logicalSize.takeError();
  if (llvm::Error error = syncFile(temporary.FD, "imported temporary object"))
    return std::move(error);

  const std::error_code publishError =
      publishNoReplace(temporary.FD, destinationDirectory, objectName);
  if (!publishError) {
    auto published = openStoredObject(destinationDirectory, objectName);
    if (!published)
      return published.takeError();
    int publishedFile = *published;
    llvm::scope_exit closePublished([&] {
      if (publishedFile != -1)
        llvm::consumeError(closeFile(publishedFile, "published object"));
    });
    auto publishedStatus = regularFileStatus(
        publishedFile, "blob_store_corruption", "published object");
    if (!publishedStatus)
      return publishedStatus.takeError();
    auto temporaryStatus = regularFileStatus(temporary.FD, "blob_store_io",
                                             "imported temporary object");
    if (!temporaryStatus)
      return temporaryStatus.takeError();
    if (publishedStatus->getUniqueID() != temporaryStatus->getUniqueID())
      return storeError("blob_store_corruption",
                        "published object is not the validated inode");
    if (llvm::Error error = closeFile(publishedFile, "published object"))
      return std::move(error);
    closePublished.release();
  } else if (publishError == std::errc::file_exists) {
    auto existing = openStoredObject(destinationDirectory, objectName);
    if (!existing)
      return existing.takeError();
    int existingFile = *existing;
    llvm::scope_exit closeExisting([&] {
      if (existingFile != -1)
        llvm::consumeError(closeFile(existingFile, "existing object"));
    });

    auto existingSize =
        copyAndVerifyOpenedObject(existingFile, -1, digest, maximumLogicalBytes,
                                  "blob_store_corruption", "existing object");
    if (!existingSize)
      return existingSize.takeError();
    if (*existingSize != *logicalSize)
      return storeError("blob_digest_collision",
                        "different logical bytes share one digest");
    auto equal = openedFilesEqual(temporary.FD, existingFile, *logicalSize);
    if (!equal)
      return equal.takeError();
    if (!*equal)
      return storeError("blob_digest_collision",
                        "different logical bytes share one digest");
    if (llvm::Error error = closeFile(existingFile, "existing object"))
      return std::move(error);
    closeExisting.release();
  } else {
    return storeError("blob_store_io",
                      llvm::Twine("unable to publish imported object: ") +
                          publishError.message());
  }

  if (llvm::Error error = discardTemporary(temporary)) {
    discardTemporaryOnFailure.release();
    return std::move(error);
  }
  discardTemporaryOnFailure.release();
  if (llvm::Error error =
          syncFile(destinationDirectory, "destination store directory"))
    return std::move(error);
  if (llvm::Error error =
          closeFile(destinationDirectory, "destination store directory"))
    return std::move(error);
  closeDestinationDirectory.release();
  if (llvm::Error error = closeFile(sourceFile, "source stored object"))
    return std::move(error);
  closeSourceFile.release();
  if (llvm::Error error = closeFile(sourceDirectory, "source store directory"))
    return std::move(error);
  closeSourceDirectory.release();
  return *logicalSize;
}

} // namespace loom
