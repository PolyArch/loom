#include "Frontend/Payload/AbiCompatibilityKey.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace loom {
namespace {

constexpr char abiKeyDomain[] = "loom.frontend.abi.compatibility.v1\0";
constexpr std::size_t abiKeyDomainSize = sizeof(abiKeyDomain) - 1;

/// Every framed length is representable in the u64 length field because no
/// supported host has a size_t wider than u64.
static_assert(std::numeric_limits<std::size_t>::max() <=
                  std::numeric_limits<std::uint64_t>::max(),
              "framed field length must be exact in the u64 length");

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendFramed(std::vector<std::uint8_t> &bytes,
                  llvm::ArrayRef<std::uint8_t> field) {
  appendU64Be(bytes, static_cast<std::uint64_t>(field.size()));
  bytes.insert(bytes.end(), field.begin(), field.end());
}

void appendFramed(std::vector<std::uint8_t> &bytes, llvm::StringRef field) {
  appendFramed(bytes, llvm::ArrayRef<std::uint8_t>(
                          reinterpret_cast<const std::uint8_t *>(field.data()),
                          field.size()));
}

std::vector<std::uint8_t>
buildAbiCompatibilityKeyPreimage(const AbiCompatibilityKeyInputs &inputs) {
  std::vector<std::uint8_t> preimage;
  preimage.insert(preimage.end(), abiKeyDomain,
                  abiKeyDomain + abiKeyDomainSize);
  appendFramed(preimage, inputs.repositoryIdentity);
  appendFramed(preimage, inputs.fullCommitIdentity);
  appendFramed(preimage, inputs.canonicalTargetTriple);
  appendFramed(preimage, inputs.canonicalDataLayout);
  appendFramed(preimage, inputs.viewSchemaDescriptorBytes);
  appendFramed(preimage, inputs.viewCanonicalBytes);
  return preimage;
}

} // namespace

llvm::Expected<AbiCompatibilityKey>
AbiCompatibilityKey::fromBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != byteSize)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "abi compatibility key requires exactly 32 bytes");
  Storage storage;
  std::copy(bytes.begin(), bytes.end(), storage.begin());
  return AbiCompatibilityKey(storage);
}

AbiCompatibilityKey
computeAbiCompatibilityKey(const AbiCompatibilityKeyInputs &inputs) {
  return AbiCompatibilityKey(
      llvm::SHA256::hash(buildAbiCompatibilityKeyPreimage(inputs)));
}

llvm::Error
validateAbiCompatibilityKey(const AbiCompatibilityKeyInputs &inputs,
                            const AbiCompatibilityKey &suppliedKey) {
  if (computeAbiCompatibilityKey(inputs) != suppliedKey)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "abi_compatibility_key_mismatch: the supplied key does not match the "
        "provider, target, and frontend view fields it is derived from");
  return llvm::Error::success();
}

} // namespace loom
