#ifndef LOOM_COMMON_BLOBDIGEST_H
#define LOOM_COMMON_BLOBDIGEST_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace loom {

/// Content identity of one logical blob byte sequence: exactly SHA-256 over
/// the logical bytes, with no schema framing, domain separation, lineage, or
/// artifact-local semantics. BlobDigest is a distinct static type from
/// ArtifactIdentity and ComponentViewDigest even though all three contain a
/// SHA-256 result. A value is always exactly 32 bytes; absence is expressed
/// by outer optionality, and an all-zero value remains an ordinary digest
/// rather than a sentinel.
class BlobDigest {
public:
  using Storage = std::array<std::uint8_t, 32>;
  static constexpr std::size_t byteSize = 32;

  static llvm::Expected<BlobDigest>
  fromBytes(llvm::ArrayRef<std::uint8_t> bytes);

  const Storage &bytes() const { return bytes_; }

  friend bool operator==(const BlobDigest &lhs, const BlobDigest &rhs) {
    return lhs.bytes_ == rhs.bytes_;
  }
  friend bool operator!=(const BlobDigest &lhs, const BlobDigest &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit BlobDigest(Storage bytes) : bytes_(bytes) {}

  friend BlobDigest
  computeBlobDigest(llvm::ArrayRef<std::uint8_t> logicalBytes);

  Storage bytes_;
};

/// Incremental implementation of BlobDigest's exact SHA-256 semantic. The
/// byte count is not narrowed at the 512 MiB SHA length-encoding boundary.
class BlobDigestBuilder final {
public:
  static llvm::Expected<BlobDigestBuilder> create();

  BlobDigestBuilder(BlobDigestBuilder &&) noexcept;
  BlobDigestBuilder &operator=(BlobDigestBuilder &&) noexcept;
  ~BlobDigestBuilder();

  llvm::Error update(llvm::ArrayRef<std::uint8_t> bytes);
  llvm::Expected<BlobDigest> finish();

private:
  struct State;
  explicit BlobDigestBuilder(std::unique_ptr<State> state);

  std::unique_ptr<State> state_;
};

/// Digests the exact logical bytes presented to consumers, including the
/// zero-length blob. Transparent storage compression, filesystem paths,
/// chunk placement, indexes, and transport encoding never change those bytes
/// or this digest.
BlobDigest computeBlobDigest(llvm::ArrayRef<std::uint8_t> logicalBytes);

/// External text is exactly 64 lowercase hexadecimal characters.
std::string formatBlobDigestHex(const BlobDigest &digest);
llvm::Expected<BlobDigest> parseBlobDigestHex(llvm::StringRef spelling);

} // namespace loom

#endif // LOOM_COMMON_BLOBDIGEST_H
