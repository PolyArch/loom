#include "RelocatablePayloadRootCodec.h"

#include "Common/ComponentViewDigest.h"
#include "Frontend/Payload/AbiCompatibilityKey.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cstddef>
#include <string>

namespace loom::detail {
namespace {

/// Raw size of the stored normalized-bitcode SHA-256 digest.
constexpr std::size_t bitcodeDigestSize = 32;

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

/// Appends a variable byte sequence as its u64 big-endian length followed by
/// its exact bytes.
void appendFramed(std::vector<std::uint8_t> &bytes,
                  llvm::ArrayRef<std::uint8_t> field) {
  appendU64Be(bytes, static_cast<std::uint64_t>(field.size()));
  bytes.insert(bytes.end(), field.begin(), field.end());
}

/// Appends a fixed digest as its raw bytes, with no length prefix.
void appendFixed(std::vector<std::uint8_t> &bytes,
                 llvm::ArrayRef<std::uint8_t> digest) {
  bytes.insert(bytes.end(), digest.begin(), digest.end());
}

/// Sequential reader over canonical root bytes. Every read is bounded by the
/// remaining input, so a truncated or overlong framed length is a typed
/// rejection rather than an out-of-range access. The first rejection is kept
/// and later reads yield nothing, which lets the root be read as the field list
/// it is and checked once.
class RootReader {
public:
  explicit RootReader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::ArrayRef<std::uint8_t> framed() {
    if (!rejection_.empty())
      return {};
    if (remaining() < 8) {
      reject("relocatable_payload_encoding_truncated: a framed length does not "
             "fit in the canonical bytes");
      return {};
    }
    std::uint64_t length = 0;
    for (unsigned index = 0; index < 8; ++index)
      length = (length << 8) | bytes_[offset_ + index];
    offset_ += 8;
    if (length > remaining()) {
      reject("relocatable_payload_encoding_overflow: a framed length exceeds "
             "the remaining canonical bytes");
      return {};
    }
    return take(static_cast<std::size_t>(length));
  }

  llvm::ArrayRef<std::uint8_t> fixed(std::size_t size) {
    if (!rejection_.empty())
      return {};
    if (remaining() < size) {
      reject("relocatable_payload_encoding_truncated: a fixed digest does not "
             "fit in the canonical bytes");
      return {};
    }
    return take(size);
  }

  /// The first encoding rejection, or the trailing-data rejection when the
  /// complete root was read but bytes remain.
  llvm::Error takeError() {
    if (rejection_.empty() && offset_ != bytes_.size())
      reject("relocatable_payload_encoding_trailing: the canonical bytes carry "
             "data after the canonical root");
    if (rejection_.empty())
      return llvm::Error::success();
    return llvm::createStringError(llvm::inconvertibleErrorCode(), rejection_);
  }

private:
  void reject(llvm::StringRef message) { rejection_ = message.str(); }

  std::uint64_t remaining() const {
    return static_cast<std::uint64_t>(bytes_.size() - offset_);
  }

  llvm::ArrayRef<std::uint8_t> take(std::size_t size) {
    const llvm::ArrayRef<std::uint8_t> field = bytes_.slice(offset_, size);
    offset_ += size;
    return field;
  }

  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
  std::string rejection_;
};

} // namespace

std::vector<std::uint8_t>
encodeRelocatablePayloadRoot(const RelocatablePayloadRoot &root) {
  std::vector<std::uint8_t> bytes;
  appendFramed(bytes, root.repositoryIdentity);
  appendFramed(bytes, root.fullCommitIdentity);
  appendFramed(bytes, root.targetTriple);
  appendFramed(bytes, root.dataLayout);
  appendFixed(bytes, root.abiCompatibilityKey);
  appendFramed(bytes, root.viewSchemaDescriptor);
  appendFramed(bytes, root.viewCanonicalBytes);
  appendFixed(bytes, root.viewDigest);
  appendFixed(bytes, root.normalizedBitcodeDigest);
  appendFramed(bytes, root.normalizedBitcode);
  return bytes;
}

llvm::Expected<RelocatablePayloadRoot>
decodeRelocatablePayloadRoot(llvm::ArrayRef<std::uint8_t> canonicalBytes) {
  RootReader reader(canonicalBytes);
  RelocatablePayloadRoot root;
  root.repositoryIdentity = reader.framed();
  root.fullCommitIdentity = reader.framed();
  root.targetTriple = reader.framed();
  root.dataLayout = reader.framed();
  root.abiCompatibilityKey = reader.fixed(AbiCompatibilityKey::byteSize);
  root.viewSchemaDescriptor = reader.framed();
  root.viewCanonicalBytes = reader.framed();
  root.viewDigest = reader.fixed(ComponentViewDigest::byteSize);
  root.normalizedBitcodeDigest = reader.fixed(bitcodeDigestSize);
  root.normalizedBitcode = reader.framed();
  if (llvm::Error error = reader.takeError())
    return std::move(error);
  return root;
}

} // namespace loom::detail
