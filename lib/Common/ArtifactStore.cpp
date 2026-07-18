#include "Common/ArtifactStore.h"

#include "ArtifactFinalizerInternal.h"
#include "Common/ArtifactText.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#if !defined(__linux__)
#error "ArtifactStore durable publication currently requires Linux"
#endif

#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace loom {
namespace {

struct OpenedArtifactObject {
  llvm::sys::fs::file_status status;
  std::unique_ptr<llvm::MemoryBuffer> contents;

  llvm::ArrayRef<std::uint8_t> preimage() const {
    const llvm::StringRef buffer = contents->getBuffer();
    return llvm::ArrayRef<std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(buffer.data()), buffer.size());
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
    return storeError("artifact_store_io", llvm::Twine("unable to inspect ") +
                                               description + ": " +
                                               error.message());
  if (!llvm::sys::fs::is_regular_file(status))
    return storeError(nonRegularErrorCode,
                      llvm::Twine(description) + " is not a regular file");
  return status;
}

llvm::Expected<OpenedArtifactObject>
readOpenedObject(int file, llvm::StringRef description,
                 llvm::StringRef objectErrorCode) {
  auto status = regularFileStatus(file, objectErrorCode, description);
  if (!status)
    return status.takeError();

  auto buffer = llvm::MemoryBuffer::getOpenFile(file, description,
                                                status->getSize(), false, true);
  if (std::error_code error = buffer.getError())
    return storeError("artifact_store_io", llvm::Twine("unable to read ") +
                                               description + ": " +
                                               error.message());
  return OpenedArtifactObject{*status, std::move(*buffer)};
}

llvm::Expected<detail::ParsedArtifactIdentityPreimage> validateStoredObject(
    const OpenedArtifactObject &object, llvm::StringRef description,
    const ArtifactIdentity &expectedIdentity, llvm::StringRef objectErrorCode) {
  auto parsed = detail::parseArtifactIdentityPreimage(object.preimage());
  if (!parsed)
    return storeError(objectErrorCode,
                      llvm::Twine(description) +
                          " has an invalid identity preimage: " +
                          llvm::toString(parsed.takeError()));

  const ArtifactIdentity actualIdentity =
      detail::finalizeArtifactIdentityPreimage(object.preimage());
  if (actualIdentity != expectedIdentity)
    return storeError(objectErrorCode, llvm::Twine(description) +
                                           " does not match its derived key");
  return *parsed;
}

llvm::Expected<int> openStoredObject(int directory,
                                     llvm::StringRef objectName) {
  const std::string name = objectName.str();
  int file;
  do {
    file = ::openat(directory, name.c_str(),
                    O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  } while (file == -1 && errno == EINTR);
  if (file == -1) {
    const int openError = errno;
    struct stat status;
    int statusResult;
    do {
      statusResult =
          ::fstatat(directory, name.c_str(), &status, AT_SYMLINK_NOFOLLOW);
    } while (statusResult == -1 && errno == EINTR);

    if (statusResult == 0 && !S_ISREG(status.st_mode))
      return storeError("artifact_store_corruption",
                        "stored object is not a regular file");
    if (statusResult == -1 && errno == ENOENT)
      return storeError("artifact_store_missing", "stored object is missing");
    if (statusResult == -1)
      return storeErrno("artifact_store_io",
                        "unable to inspect stored object after open failure");
    errno = openError;
    return storeErrno("artifact_store_io", "unable to open stored object");
  }
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
    return storeErrno("artifact_store_io",
                      "unable to open required store root directory");
  return directory;
}

llvm::Error syncFile(int file, llvm::StringRef description) {
  int result;
  do {
    result = ::fsync(file);
  } while (result == -1 && errno == EINTR);
  if (result == -1)
    return storeErrno("artifact_store_io",
                      llvm::Twine("unable to sync ") + description);
  return llvm::Error::success();
}

llvm::Error closeFile(int &file, llvm::StringRef description) {
  if (std::error_code error = llvm::sys::fs::closeFile(file))
    return storeError("artifact_store_io", llvm::Twine("unable to close ") +
                                               description + ": " +
                                               error.message());
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

llvm::Error discardTemporary(llvm::sys::fs::TempFile &temporary) {
  if (llvm::Error error = temporary.discard())
    return storeError("artifact_store_io",
                      llvm::Twine("unable to close and remove temporary ") +
                          "object: " + llvm::toString(std::move(error)));
  return llvm::Error::success();
}

} // namespace

llvm::Expected<ArtifactIdentity>
ArtifactStore::put(const ArtifactSchemaDescriptor &schema,
                   const CanonicalSemanticBytes &canonicalBytes) const {
  const std::vector<std::uint8_t> preimage =
      detail::buildArtifactIdentityPreimage(schema, canonicalBytes);
  const ArtifactIdentity identity =
      detail::finalizeArtifactIdentityPreimage(preimage);
  const std::string objectName = formatArtifactIdentityHex(identity);

  auto directoryOrError = openStoreDirectory(root_);
  if (!directoryOrError)
    return directoryOrError.takeError();
  int directory = *directoryOrError;
  llvm::scope_exit closeDirectoryOnFailure([&] {
    if (directory != -1)
      llvm::consumeError(closeFile(directory, "store directory"));
  });

  llvm::SmallString<256> temporaryModel(root_);
  llvm::sys::path::append(temporaryModel, ".artifact-%%%%%%");
  auto temporaryOrError = llvm::sys::fs::TempFile::create(
      temporaryModel, llvm::sys::fs::owner_read | llvm::sys::fs::owner_write);
  if (!temporaryOrError)
    return storeError("artifact_store_io",
                      llvm::Twine("unable to create temporary object: ") +
                          llvm::toString(temporaryOrError.takeError()));
  llvm::sys::fs::TempFile temporary = std::move(*temporaryOrError);
  llvm::scope_exit discardTemporaryOnFailure(
      [&] { llvm::consumeError(temporary.discard()); });

  {
    llvm::raw_fd_ostream output(temporary.FD, false);
    output.write(reinterpret_cast<const char *>(preimage.data()),
                 preimage.size());
    output.flush();
    if (std::error_code error = output.error()) {
      output.clear_error();
      return storeError("artifact_store_io",
                        llvm::Twine("unable to write temporary object: ") +
                            error.message());
    }
  }

  if (llvm::Error error = syncFile(temporary.FD, "temporary object"))
    return std::move(error);
  auto temporaryObject =
      readOpenedObject(temporary.FD, "temporary object", "artifact_store_io");
  if (!temporaryObject)
    return temporaryObject.takeError();
  auto parsedTemporary = validateStoredObject(
      *temporaryObject, "temporary object", identity, "artifact_store_io");
  if (!parsedTemporary)
    return parsedTemporary.takeError();
  if (!temporaryObject->preimage().equals(preimage))
    return storeError("artifact_identity_collision",
                      "different identity preimages share one digest");

  const std::error_code publishError =
      publishNoReplace(temporary.FD, directory, objectName);
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
        publishedFile, "artifact_store_corruption", "published object");
    if (!publishedStatus)
      return publishedStatus.takeError();
    if (publishedStatus->getUniqueID() != temporaryObject->status.getUniqueID())
      return storeError("artifact_store_corruption",
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

    auto existingObject = readOpenedObject(existingFile, "existing object",
                                           "artifact_store_corruption");
    if (!existingObject)
      return existingObject.takeError();
    auto parsedExisting =
        validateStoredObject(*existingObject, "existing object", identity,
                             "artifact_store_corruption");
    if (!parsedExisting)
      return parsedExisting.takeError();
    if (!existingObject->preimage().equals(preimage))
      return storeError("artifact_identity_collision",
                        "different identity preimages share one digest");
    if (llvm::Error error = closeFile(existingFile, "existing object"))
      return std::move(error);
    closeExistingOnFailure.release();
  } else {
    return storeError("artifact_store_io",
                      llvm::Twine("unable to publish object: ") +
                          publishError.message());
  }

  if (llvm::Error error = discardTemporary(temporary)) {
    discardTemporaryOnFailure.release();
    return std::move(error);
  }
  discardTemporaryOnFailure.release();
  if (llvm::Error error = syncFile(directory, "store directory"))
    return std::move(error);
  if (llvm::Error error = closeFile(directory, "store directory"))
    return std::move(error);
  closeDirectoryOnFailure.release();
  return identity;
}

llvm::Expected<CanonicalSemanticBytes>
ArtifactStore::get(const ArtifactSchemaDescriptor &expectedSchema,
                   const ArtifactIdentity &identity) const {
  auto directoryOrError = openStoreDirectory(root_);
  if (!directoryOrError)
    return directoryOrError.takeError();
  int directory = *directoryOrError;
  llvm::scope_exit closeDirectory([&] {
    if (directory != -1)
      llvm::consumeError(closeFile(directory, "store directory"));
  });

  const std::string objectName = formatArtifactIdentityHex(identity);
  auto fileOrError = openStoredObject(directory, objectName);
  if (!fileOrError)
    return fileOrError.takeError();
  int file = *fileOrError;
  llvm::scope_exit closeObject([&] {
    if (file != -1)
      llvm::consumeError(closeFile(file, "stored object"));
  });

  auto object =
      readOpenedObject(file, "stored object", "artifact_store_corruption");
  if (!object)
    return object.takeError();
  auto parsed = validateStoredObject(*object, "stored object", identity,
                                     "artifact_store_corruption");
  if (!parsed)
    return parsed.takeError();

  if (parsed->schemaIdentity != expectedSchema.identity)
    return storeError("artifact_schema_mismatch",
                      "stored object schema identity does not match expected "
                      "schema identity");
  if (parsed->schemaVersion != expectedSchema.version)
    return storeError("artifact_schema_mismatch",
                      "stored object schema version does not match expected "
                      "schema version");

  std::vector<std::uint8_t> canonicalBytes(
      parsed->canonicalSemanticBytes.begin(),
      parsed->canonicalSemanticBytes.end());
  if (llvm::Error error = closeFile(file, "stored object"))
    return std::move(error);
  closeObject.release();
  if (llvm::Error error = closeFile(directory, "store directory"))
    return std::move(error);
  closeDirectory.release();
  return CanonicalSemanticBytes(std::move(canonicalBytes));
}

} // namespace loom
