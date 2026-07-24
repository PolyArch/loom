#ifndef LOOM_FRONTEND_PAYLOAD_ABICOMPATIBILITYKEY_H
#define LOOM_FRONTEND_PAYLOAD_ABICOMPATIBILITYKEY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace loom {

/// The exact source fields of the frontend ABI compatibility key preimage.
///
/// The key frames these raw authoritative values directly. It deliberately
/// excludes the component-view digest, which is already derived from the view
/// descriptor and bytes below, and excludes the bitcode digest and module
/// contents, because distinct translation units must be able to share one
/// compatibility cohort.
struct AbiCompatibilityKeyInputs {
  llvm::StringRef repositoryIdentity;
  llvm::StringRef fullCommitIdentity;
  llvm::StringRef canonicalTargetTriple;
  llvm::StringRef canonicalDataLayout;
  llvm::ArrayRef<std::uint8_t> viewSchemaDescriptorBytes;
  llvm::ArrayRef<std::uint8_t> viewCanonicalBytes;
};

/// A necessary cohort and preflight value, not the complete LLVM ABI authority.
/// It is a distinct semantic type from ArtifactIdentity and ComponentViewDigest
/// and cannot be separately authored.
class AbiCompatibilityKey {
public:
  using Storage = std::array<std::uint8_t, 32>;
  static constexpr std::size_t byteSize = 32;

  /// Adopts an exact 32-byte key carried alongside the source fields it was
  /// derived from. The adopted value is never authority on its own; a reader
  /// still recomputes through validateAbiCompatibilityKey.
  static llvm::Expected<AbiCompatibilityKey>
  fromBytes(llvm::ArrayRef<std::uint8_t> bytes);

  const Storage &bytes() const { return bytes_; }

  friend bool operator==(const AbiCompatibilityKey &lhs,
                         const AbiCompatibilityKey &rhs) {
    return lhs.bytes_ == rhs.bytes_;
  }
  friend bool operator!=(const AbiCompatibilityKey &lhs,
                         const AbiCompatibilityKey &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit AbiCompatibilityKey(Storage bytes) : bytes_(bytes) {}

  friend AbiCompatibilityKey
  computeAbiCompatibilityKey(const AbiCompatibilityKeyInputs &inputs);

  Storage bytes_;
};

/// The sole production encoder of the key. Every input is framed by its exact
/// length, so no two distinct field assignments share a preimage.
AbiCompatibilityKey
computeAbiCompatibilityKey(const AbiCompatibilityKeyInputs &inputs);

/// Recomputes the key from the authoritative source fields and reports a typed
/// error when the supplied key is not exactly that value.
llvm::Error validateAbiCompatibilityKey(const AbiCompatibilityKeyInputs &inputs,
                                        const AbiCompatibilityKey &suppliedKey);

} // namespace loom

#endif // LOOM_FRONTEND_PAYLOAD_ABICOMPATIBILITYKEY_H
