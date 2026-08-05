#ifndef LOOM_HARDWARE_IMPLEMENTATION_IMPLEMENTATIONPAYLOAD_H
#define LOOM_HARDWARE_IMPLEMENTATION_IMPLEMENTATIONPAYLOAD_H

#include "Common/BlobDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom::hardware {

enum class PayloadRole : std::uint32_t {
  RtlSource = 0,
  Netlist = 1,
  PhysicalDatabase = 2,
  Parasitics = 3,
  LayoutStream = 4,
  DeviceImage = 5,
  GenerationConstraint = 6,
  BlackBoxContract = 7,
};

struct ImplementationPayload final {
  PayloadRole role;
  std::string canonicalLogicalName;
  BlobDigest blobDigest;

  friend bool operator==(const ImplementationPayload &lhs,
                         const ImplementationPayload &rhs) {
    return lhs.role == rhs.role &&
           lhs.canonicalLogicalName == rhs.canonicalLogicalName &&
           lhs.blobDigest == rhs.blobDigest;
  }
};

/// Encodes one payload as u32be(role), u64be(logical-name length), logical-name
/// bytes, and the exact 32 BlobDigest bytes.
llvm::Expected<std::vector<std::uint8_t>>
encodeImplementationPayload(const ImplementationPayload &payload);

llvm::Expected<ImplementationPayload>
decodeImplementationPayload(llvm::ArrayRef<std::uint8_t> bytes);

/// Uses the schema-2.0 field names and exact displayed payload-role spellings.
llvm::Expected<std::string>
serializeImplementationPayloadJson(const ImplementationPayload &payload);

llvm::Expected<ImplementationPayload>
parseImplementationPayloadJson(llvm::StringRef bytes);

llvm::Error validateImplementationPayload(const ImplementationPayload &payload);

bool implementationPayloadCanonicalLess(const ImplementationPayload &lhs,
                                        const ImplementationPayload &rhs);

/// Produces the sole canonical payload order and rejects duplicate
/// (role, canonical logical name) keys.
llvm::Expected<std::vector<ImplementationPayload>>
canonicalizeImplementationPayloadCatalog(
    llvm::ArrayRef<ImplementationPayload> payloads);

} // namespace loom::hardware

#endif // LOOM_HARDWARE_IMPLEMENTATION_IMPLEMENTATIONPAYLOAD_H
