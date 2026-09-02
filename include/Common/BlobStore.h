#ifndef LOOM_COMMON_BLOBSTORE_H
#define LOOM_COMMON_BLOBSTORE_H

#include "Common/BlobDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {

struct GeneratedBlobPublication final {
  BlobDigest digest;
  std::uint64_t logicalByteCount = 0;
};

/// Local filesystem store of complete logical blobs keyed by full BlobDigest.
/// Publication is atomic and never overwrites or repairs an existing object.
/// Equal bytes deduplicate; the full stored byte sequence is verified on
/// deduplication and rehashed on every read. A digest occupied by different
/// bytes is a hard collision, while malformed storage or a digest/bytes
/// mismatch is corruption; neither is resolved by selecting one payload.
/// Blob ownership, media type, and relation to a typed Artifact remain in the
/// referencing owner's manifest; this store owns only content identity.
class BlobStore {
public:
  /// Root must name an existing, durably provisioned non-symlink directory.
  explicit BlobStore(llvm::StringRef root) : root_(root.str()) {}

  llvm::Expected<BlobDigest>
  put(llvm::ArrayRef<std::uint8_t> logicalBytes) const;

  /// Atomically publishes bytes emitted once by the caller without retaining
  /// the complete logical sequence in memory. The writer is invoked on a
  /// temporary object inside this store; publication derives and verifies the
  /// same BlobDigest and collision contract as put(ArrayRef).
  llvm::Expected<GeneratedBlobPublication> putGenerated(
      llvm::function_ref<llvm::Error(llvm::raw_ostream &)> writer) const;

  llvm::Expected<std::vector<std::uint8_t>> get(const BlobDigest &digest) const;

  /// Reads and verifies one object only when its stored size does not exceed
  /// the caller-owned admission bound. The bound is checked before mapping or
  /// copying object contents.
  llvm::Expected<std::vector<std::uint8_t>>
  get(const BlobDigest &digest, std::uint64_t maximumLogicalBytes) const;

  /// Reads and verifies one object without copying its logical bytes into a
  /// returned buffer. The result is the exact validated logical-byte count.
  llvm::Expected<std::uint64_t> verify(const BlobDigest &digest) const;

  llvm::Expected<std::uint64_t> verify(const BlobDigest &digest,
                                       std::uint64_t maximumLogicalBytes) const;

  /// Copies one exact object from another BlobStore without buffering the
  /// complete logical byte sequence. The source is verified against digest,
  /// the bound is checked before copying, and publication retains put()'s
  /// atomic no-replace and collision semantics. Returns the copied byte count.
  llvm::Expected<std::uint64_t>
  importVerified(const BlobDigest &digest, const BlobStore &source,
                 std::uint64_t maximumLogicalBytes) const;

private:
  std::string root_;
};

} // namespace loom

#endif // LOOM_COMMON_BLOBSTORE_H
