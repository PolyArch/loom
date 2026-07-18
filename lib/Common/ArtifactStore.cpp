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
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace loom {
namespace {

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

llvm::Expected<llvm::sys::fs::file_status>
validateOpenedObject(int file, llvm::StringRef description,
                     const ArtifactIdentity &expectedIdentity,
                     llvm::ArrayRef<std::uint8_t> expectedPreimage,
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
  const llvm::StringRef contents = (*buffer)->getBuffer();
  const llvm::ArrayRef<std::uint8_t> object(
      reinterpret_cast<const std::uint8_t *>(contents.data()), contents.size());

  if (object.equals(expectedPreimage))
    return *status;

  if (llvm::Error error = detail::validateArtifactIdentityPreimage(object)) {
    llvm::consumeError(std::move(error));
    return storeError(objectErrorCode,
                      llvm::Twine(description) +
                          " is not a reconstructable identity preimage");
  }

  const ArtifactIdentity actualIdentity =
      detail::finalizeArtifactIdentityPreimage(object);
  if (actualIdentity != expectedIdentity)
    return storeError(objectErrorCode, llvm::Twine(description) +
                                           " does not match its derived key");
  return storeError("artifact_identity_collision",
                    "different identity preimages share one digest");
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
    if (errno == ELOOP)
      return storeError("artifact_store_corruption",
                        "stored object is a symbolic link");
    return storeErrno("artifact_store_io", "unable to open stored object");
  }
  return file;
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

  int directory;
  do {
    directory =
        ::open(root_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
  } while (directory == -1 && errno == EINTR);
  if (directory == -1)
    return storeErrno("artifact_store_io",
                      "unable to open required store root directory");
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
  auto temporaryStatus =
      validateOpenedObject(temporary.FD, "temporary object", identity, preimage,
                           "artifact_store_io");
  if (!temporaryStatus)
    return temporaryStatus.takeError();

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
    if (publishedStatus->getUniqueID() != temporaryStatus->getUniqueID())
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

    auto existingStatus =
        validateOpenedObject(existingFile, "existing object", identity,
                             preimage, "artifact_store_corruption");
    if (!existingStatus)
      return existingStatus.takeError();
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

} // namespace loom
