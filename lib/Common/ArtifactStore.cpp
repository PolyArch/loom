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

#include <cstdint>
#include <string>
#include <system_error>
#include <vector>

namespace loom {
namespace {

llvm::Error storeError(llvm::StringRef code, const llvm::Twine &detail) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 code + ": " + detail);
}

llvm::Expected<std::vector<std::uint8_t>>
readObject(llvm::StringRef path, llvm::StringRef errorCode) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false);
  if (std::error_code error = buffer.getError())
    return storeError(errorCode,
                      llvm::Twine("unable to read object: ") + error.message());
  llvm::StringRef contents = (*buffer)->getBuffer();
  return std::vector<std::uint8_t>(contents.bytes_begin(),
                                   contents.bytes_end());
}

llvm::Error validateObject(llvm::StringRef path,
                           const ArtifactIdentity &expectedIdentity,
                           llvm::ArrayRef<std::uint8_t> expectedPreimage,
                           llvm::StringRef readErrorCode) {
  auto object = readObject(path, readErrorCode);
  if (!object)
    return object.takeError();
  if (llvm::Error error = detail::validateArtifactIdentityPreimage(*object)) {
    llvm::consumeError(std::move(error));
    return storeError(
        "artifact_store_corruption",
        "stored object is not a reconstructable identity preimage");
  }

  const ArtifactIdentity actualIdentity =
      detail::finalizeArtifactIdentityPreimage(*object);
  if (actualIdentity != expectedIdentity)
    return storeError("artifact_store_corruption",
                      "stored object does not match its derived key");
  if (!llvm::ArrayRef<std::uint8_t>(*object).equals(expectedPreimage))
    return storeError("artifact_identity_collision",
                      "different identity preimages share one digest");
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

  if (std::error_code error = llvm::sys::fs::create_directories(root_))
    return storeError("artifact_store_io",
                      llvm::Twine("unable to create store directory: ") +
                          error.message());

  llvm::SmallString<256> objectPath(root_);
  llvm::sys::path::append(objectPath, formatArtifactIdentityHex(identity));

  llvm::SmallString<256> temporaryModel(root_);
  llvm::sys::path::append(temporaryModel, ".artifact-%%%%%%");
  auto temporaryOrError = llvm::sys::fs::TempFile::create(
      temporaryModel, llvm::sys::fs::owner_read | llvm::sys::fs::owner_write);
  if (!temporaryOrError)
    return storeError("artifact_store_io",
                      llvm::Twine("unable to create temporary object: ") +
                          llvm::toString(temporaryOrError.takeError()));
  llvm::sys::fs::TempFile temporary = std::move(*temporaryOrError);
  llvm::scope_exit discardTemporary(
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

  if (llvm::Error error = validateObject(temporary.TmpName, identity, preimage,
                                         "artifact_store_io"))
    return std::move(error);

  const std::error_code publishError =
      llvm::sys::fs::create_hard_link(temporary.TmpName, objectPath);
  if (!publishError)
    return identity;
  if (publishError != std::errc::file_exists)
    return storeError("artifact_store_io",
                      llvm::Twine("unable to publish object: ") +
                          publishError.message());

  if (llvm::Error error = validateObject(objectPath, identity, preimage,
                                         "artifact_store_corruption"))
    return std::move(error);
  return identity;
}

} // namespace loom
