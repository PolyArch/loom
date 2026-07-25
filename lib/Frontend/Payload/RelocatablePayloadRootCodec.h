#ifndef LOOM_LIB_FRONTEND_PAYLOAD_RELOCATABLEPAYLOADROOTCODEC_H
#define LOOM_LIB_FRONTEND_PAYLOAD_RELOCATABLEPAYLOADROOTCODEC_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::detail {

/// The canonical root of a `loom.relocatable_accelerator_payload` 1.0 Artifact,
/// held as its exact field list in its fixed field order.
///
/// This struct and the two functions below are the one owner of that field
/// order and its framing: fixed digests are raw bytes with no length prefix,
/// and every other field carries an unsigned 64-bit big-endian length followed
/// by its exact bytes. Nothing else may restate either rule.
struct RelocatablePayloadRoot {
  llvm::ArrayRef<std::uint8_t> repositoryIdentity;
  llvm::ArrayRef<std::uint8_t> fullCommitIdentity;
  llvm::ArrayRef<std::uint8_t> targetTriple;
  llvm::ArrayRef<std::uint8_t> dataLayout;
  llvm::ArrayRef<std::uint8_t> abiCompatibilityKey;
  llvm::ArrayRef<std::uint8_t> viewSchemaDescriptor;
  llvm::ArrayRef<std::uint8_t> viewCanonicalBytes;
  llvm::ArrayRef<std::uint8_t> viewDigest;
  llvm::ArrayRef<std::uint8_t> normalizedBitcodeDigest;
  llvm::ArrayRef<std::uint8_t> normalizedBitcode;
};

std::vector<std::uint8_t>
encodeRelocatablePayloadRoot(const RelocatablePayloadRoot &root);

/// Reads the root back. Every field points into `canonicalBytes`, which must
/// outlive the result. A truncated, overlong, or trailing-data encoding is a
/// typed rejection rather than an out-of-range read; the fields themselves are
/// not interpreted here.
llvm::Expected<RelocatablePayloadRoot>
decodeRelocatablePayloadRoot(llvm::ArrayRef<std::uint8_t> canonicalBytes);

} // namespace loom::detail

#endif // LOOM_LIB_FRONTEND_PAYLOAD_RELOCATABLEPAYLOADROOTCODEC_H
