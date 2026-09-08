#include "Common/ArtifactStore.h"

#include "ArtifactFinalizerInternal.h"
#include "Common/ArtifactFinalizer.h"
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
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace loom {
namespace {

struct OpenedArtifactObject {
  llvm::sys::fs::file_status status;
  std::vector<std::uint8_t> preimage;
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
  const llvm::StringRef contents = (*buffer)->getBuffer();
  return OpenedArtifactObject{
      *status,
      std::vector<std::uint8_t>(contents.bytes_begin(), contents.bytes_end())};
}

llvm::Expected<detail::ParsedArtifactIdentityPreimage> validateStoredObject(
    const OpenedArtifactObject &object, llvm::StringRef description,
    const ArtifactIdentity &expectedIdentity, llvm::StringRef objectErrorCode) {
  auto parsed = detail::parseArtifactIdentityPreimage(object.preimage);
  if (!parsed)
    return storeError(objectErrorCode,
                      llvm::Twine(description) +
                          " has an invalid identity preimage: " +
                          llvm::toString(parsed.takeError()));

  const ArtifactIdentity actualIdentity =
      detail::finalizeArtifactIdentityPreimage(object.preimage);
  if (actualIdentity != expectedIdentity)
    return storeError(objectErrorCode, llvm::Twine(description) +
                                           " does not match its derived key");
  return *parsed;
}

llvm::Error validateStoredObjectAgainstPreimage(
    const OpenedArtifactObject &object, llvm::StringRef description,
    const ArtifactIdentity &identity, llvm::ArrayRef<std::uint8_t> preimage,
    llvm::StringRef objectErrorCode) {
  // put already constructed and hashed this complete identity preimage. Full
  // byte equality proves both its framing and key without hashing the same
  // bytes again; a different object retains the independent error distinction.
  if (llvm::ArrayRef<std::uint8_t>(object.preimage).equals(preimage))
    return llvm::Error::success();
  auto parsed = validateStoredObject(object, description, identity,
                                     objectErrorCode);
  if (!parsed)
    return parsed.takeError();
  return storeError("artifact_identity_collision",
                    "different identity preimages share one digest");
}

llvm::Error closeFile(int &file, llvm::StringRef description) {
  if (std::error_code error = llvm::sys::fs::closeFile(file))
    return storeError("artifact_store_io", llvm::Twine("unable to close ") +
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
      return storeError("artifact_store_missing",
                        "stored object is missing: '" + objectName + "'");
    return storeErrno("artifact_store_io",
                      "unable to open stored object handle");
  }
  llvm::scope_exit closeHandle([&] {
    if (handle != -1)
      llvm::consumeError(closeFile(handle, "stored object handle"));
  });

  auto status =
      regularFileStatus(handle, "artifact_store_corruption", "stored object");
  if (!status)
    return status.takeError();

  const std::string handlePath = "/proc/self/fd/" + std::to_string(handle);
  int file;
  do {
    file = ::open(handlePath.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
  } while (file == -1 && errno == EINTR);
  if (file == -1)
    return storeErrno("artifact_store_io",
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
    return storeErrno("artifact_store_io",
                      "unable to open required store root directory");
  return directory;
}

llvm::Error syncFile(int file, llvm::StringRef description,
                     ArtifactStore::Durability durability) {
  if (durability == ArtifactStore::Durability::Transient)
    return llvm::Error::success();
  int result;
  do {
    result = ::fsync(file);
  } while (result == -1 && errno == EINTR);
  if (result == -1)
    return storeErrno("artifact_store_io",
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

llvm::Error discardTemporary(llvm::sys::fs::TempFile &temporary) {
  if (llvm::Error error = temporary.discard())
    return storeError("artifact_store_io",
                      llvm::Twine("unable to close and remove temporary ") +
                          "object: " + llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Expected<bool>
validateExistingObject(int directory, llvm::StringRef objectName,
                       const ArtifactIdentity &identity,
                       llvm::ArrayRef<std::uint8_t> preimage,
                       ArtifactStore::Durability durability) {
  const std::string name = objectName.str();
  struct stat status;
  int inspected;
  do {
    inspected =
        ::fstatat(directory, name.c_str(), &status, AT_SYMLINK_NOFOLLOW);
  } while (inspected == -1 && errno == EINTR);
  if (inspected == -1) {
    if (errno == ENOENT)
      return false;
    return storeErrno("artifact_store_io", "unable to inspect stored object");
  }
  auto existing = openStoredObject(directory, objectName);
  if (!existing)
    return existing.takeError();
  int file = *existing;
  llvm::scope_exit closeExisting([&] {
    if (file != -1)
      llvm::consumeError(closeFile(file, "existing object"));
  });
  auto object =
      readOpenedObject(file, "existing object", "artifact_store_corruption");
  if (!object)
    return object.takeError();
  if (llvm::Error error = validateStoredObjectAgainstPreimage(
          *object, "existing object", identity, preimage,
          "artifact_store_corruption"))
    return std::move(error);
  if (llvm::Error error = syncFile(file, "existing object", durability))
    return std::move(error);
  if (llvm::Error error = closeFile(file, "existing object"))
    return std::move(error);
  closeExisting.release();
  return true;
}

} // namespace

// Identity admission and retained payload have different lifetimes: closure
// validation needs the former, while a byte consumer needs the latter. Both
// derive only from an independently validated read, never from put arguments.
struct ArtifactStore::VerifiedReadCache final {
  struct IdentityLess final {
    bool operator()(const ArtifactIdentity &lhs,
                    const ArtifactIdentity &rhs) const {
      return lhs.bytes() < rhs.bytes();
    }
  };
  struct Entry final {
    std::string schemaIdentity;
    SchemaVersion schemaVersion;
    std::optional<CanonicalSemanticBytes> bytes;
    std::list<ArtifactIdentity>::iterator payloadPosition;

    llvm::Error validateSchema(llvm::StringRef identity,
                               SchemaVersion version) const {
      if (schemaIdentity != identity || schemaVersion != version)
        return storeError("artifact_schema_mismatch",
                          "stored object schema does not match exact reference");
      return llvm::Error::success();
    }
  };
  static constexpr std::size_t byteBudget = 256u << 20;
  static constexpr std::size_t identityBudget = 256u << 10;
  std::mutex mutex;
  std::map<ArtifactIdentity, Entry, IdentityLess> entries;
  std::list<ArtifactIdentity> identityOrder;
  std::list<ArtifactIdentity> payloadOrder;
  std::size_t retainedBytes = 0;

  void discardPayload(Entry &entry) {
    if (!entry.bytes)
      return;
    retainedBytes -= entry.bytes->bytes().size();
    entry.bytes.reset();
    payloadOrder.erase(entry.payloadPosition);
  }

  void retain(const ArtifactIdentity &identity, llvm::StringRef schemaIdentity,
              SchemaVersion schemaVersion, CanonicalSemanticBytes bytes) {
    std::lock_guard<std::mutex> lock(mutex);
    auto found = entries.find(identity);
    if (found == entries.end()) {
      if (entries.size() == identityBudget) {
        auto evicted = entries.find(identityOrder.front());
        discardPayload(evicted->second);
        entries.erase(evicted);
        identityOrder.pop_front();
      }
      found = entries.emplace(identity, Entry{schemaIdentity.str(),
                                              schemaVersion, std::nullopt, {}})
                  .first;
      identityOrder.push_back(identity);
    }
    Entry &entry = found->second;
    const std::size_t size = bytes.bytes().size();
    if (entry.bytes || size > byteBudget)
      return;
    while (retainedBytes + size > byteBudget)
      discardPayload(entries.find(payloadOrder.front())->second);
    payloadOrder.push_back(identity);
    entry.payloadPosition = std::prev(payloadOrder.end());
    entry.bytes = std::move(bytes);
    retainedBytes += size;
  }
};

ArtifactStore::ArtifactStore(llvm::StringRef root, Durability durability)
    : root_(root.str()), durability_(durability),
      verifiedReads_(std::make_shared<VerifiedReadCache>()) {}

llvm::Error
ArtifactStore::verifyReference(const ArtifactRootReference &reference) const {
  {
    std::lock_guard<std::mutex> lock(verifiedReads_->mutex);
    auto found = verifiedReads_->entries.find(reference.artifact);
    if (found != verifiedReads_->entries.end())
      return found->second.validateSchema(reference.schemaIdentity,
                                          reference.schemaVersion);
  }
  auto bytes = get(reference);
  if (!bytes)
    return bytes.takeError();
  return llvm::Error::success();
}

llvm::Expected<ArtifactIdentity>
ArtifactStore::put(const ArtifactSchemaDescriptor &schema,
                   const CanonicalSemanticBytes &canonicalBytes) const {
  const std::vector<std::uint8_t> preimage =
      detail::buildArtifactIdentityPreimage(schema, canonicalBytes);
  const ArtifactIdentity identity =
      finalizeArtifactIdentity(schema, canonicalBytes);
  const std::string objectName = formatArtifactIdentityHex(identity);

  auto directoryOrError = openStoreDirectory(root_);
  if (!directoryOrError)
    return directoryOrError.takeError();
  int directory = *directoryOrError;
  llvm::scope_exit closeDirectoryOnFailure([&] {
    if (directory != -1)
      llvm::consumeError(closeFile(directory, "store directory"));
  });

  auto existing = validateExistingObject(directory, objectName, identity,
                                         preimage, durability_);
  if (!existing)
    return existing.takeError();
  if (!*existing) {
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

    if (llvm::Error error =
            syncFile(temporary.FD, "temporary object", durability_))
      return std::move(error);
    auto temporaryObject =
        readOpenedObject(temporary.FD, "temporary object", "artifact_store_io");
    if (!temporaryObject)
      return temporaryObject.takeError();
    if (llvm::Error error = validateStoredObjectAgainstPreimage(
            *temporaryObject, "temporary object", identity, preimage,
            "artifact_store_io"))
      return std::move(error);

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
      if (publishedStatus->getUniqueID() !=
          temporaryObject->status.getUniqueID())
        return storeError("artifact_store_corruption",
                          "published object is not the validated inode");
      if (llvm::Error error = closeFile(publishedFile, "published object"))
        return std::move(error);
      closePublishedOnFailure.release();
    } else if (publishError == std::errc::file_exists) {
      auto existing = validateExistingObject(directory, objectName, identity,
                                             preimage, durability_);
      if (!existing)
        return existing.takeError();
      if (!*existing)
        return storeError("artifact_store_missing",
                          "stored object is missing: '" + objectName + "'");
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
  }
  if (llvm::Error error = syncFile(directory, "store directory", durability_))
    return std::move(error);
  if (llvm::Error error = closeFile(directory, "store directory"))
    return std::move(error);
  closeDirectoryOnFailure.release();
  return identity;
}

llvm::Expected<CanonicalSemanticBytes>
ArtifactStore::get(const ArtifactSchemaDescriptor &expectedSchema,
                   const ArtifactIdentity &identity) const {
  return getExact(expectedSchema.identity, expectedSchema.version, identity);
}

llvm::Expected<CanonicalSemanticBytes>
ArtifactStore::get(const ArtifactRootReference &reference) const {
  return getExact(reference.schemaIdentity, reference.schemaVersion,
                  reference.artifact);
}

llvm::Expected<std::vector<std::uint8_t>>
ArtifactStore::getStoredObject(const ArtifactRootReference &reference) const {
  auto directoryOrError = openStoreDirectory(root_);
  if (!directoryOrError)
    return directoryOrError.takeError();
  int directory = *directoryOrError;
  llvm::scope_exit closeDirectory([&] {
    if (directory != -1)
      llvm::consumeError(closeFile(directory, "store directory"));
  });

  const std::string objectName = formatArtifactIdentityHex(reference.artifact);
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
  auto parsed =
      validateStoredObject(*object, "stored object", reference.artifact,
                           "artifact_store_corruption");
  if (!parsed)
    return parsed.takeError();
  if (parsed->schemaIdentity != reference.schemaIdentity ||
      parsed->schemaVersion != reference.schemaVersion)
    return storeError("artifact_schema_mismatch",
                      "stored object schema does not match exact reference");

  if (llvm::Error error = closeFile(file, "stored object"))
    return error;
  closeObject.release();
  if (llvm::Error error = closeFile(directory, "store directory"))
    return error;
  closeDirectory.release();
  return std::move(object->preimage);
}

llvm::Expected<CanonicalSemanticBytes>
ArtifactStore::getExact(llvm::StringRef schemaIdentity,
                        SchemaVersion schemaVersion,
                        const ArtifactIdentity &identity) const {
  {
    std::lock_guard<std::mutex> lock(verifiedReads_->mutex);
    const auto found = verifiedReads_->entries.find(identity);
    if (found != verifiedReads_->entries.end()) {
      if (llvm::Error error =
              found->second.validateSchema(schemaIdentity, schemaVersion))
        return error;
      if (found->second.bytes)
        return *found->second.bytes;
    }
  }
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
    return storeError(
        "artifact_store_lookup",
        "schema='" + schemaIdentity + "' version=" +
            llvm::Twine(schemaVersion.major) + "." +
            llvm::Twine(schemaVersion.minor) + "' identity='" + objectName +
            "': " + llvm::toString(fileOrError.takeError()));
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

  if (parsed->schemaIdentity != schemaIdentity)
    return storeError("artifact_schema_mismatch",
                      "stored object schema identity does not match expected "
                      "schema identity");
  if (parsed->schemaVersion != schemaVersion)
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
  CanonicalSemanticBytes verified(std::move(canonicalBytes));
  verifiedReads_->retain(identity, schemaIdentity, schemaVersion, verified);
  return verified;
}

} // namespace loom
