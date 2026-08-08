#include "Hardware/Implementation/HardwareImplementationLocalReference.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>

namespace loom::hardware {
namespace {

llvm::Error invalidLocalReference(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "hardware_implementation_local_reference_invalid: " + message);
}

llvm::Error validateOrdinal(llvm::StringRef target, std::uint64_t ordinal,
                            std::size_t size) {
  if (ordinal >= static_cast<std::uint64_t>(size))
    return invalidLocalReference(target + " ordinal " + llvm::Twine(ordinal) +
                                 " is out of range for catalog size " +
                                 llvm::Twine(size));
  return llvm::Error::success();
}

} // namespace

namespace detail {

std::array<std::uint8_t, 8>
encodeHardwareImplementationLocalOrdinal(std::uint64_t ordinal) {
  std::array<std::uint8_t, 8> payload{};
  for (unsigned index = 0; index < payload.size(); ++index)
    payload[index] = static_cast<std::uint8_t>(ordinal >> (56 - index * 8));
  return payload;
}

llvm::Expected<std::uint64_t> decodeHardwareImplementationLocalReferenceOrdinal(
    const EncodedArtifactLocalReference &reference,
    HardwareImplementationLocalReferenceKind expectedKind,
    llvm::StringRef typedTarget) {
  if (reference.artifact.schemaIdentity !=
          hardwareImplementationSchema.identity ||
      reference.artifact.schemaVersion != hardwareImplementationSchema.version)
    return invalidLocalReference(
        "local references require the exact " +
        hardwareImplementationSchema.identity + " " +
        llvm::Twine(hardwareImplementationSchema.version.major) + "." +
        llvm::Twine(hardwareImplementationSchema.version.minor) + " schema");
  const std::uint32_t expected =
      hardwareImplementationLocalReferenceKindOrdinal(expectedKind);
  if (reference.ownerLocalKind != expected)
    return invalidLocalReference("owner-local kind " +
                                 llvm::Twine(reference.ownerLocalKind) +
                                 " does not encode " + typedTarget);
  if (reference.payload.size() != 8)
    return invalidLocalReference("a HardwareImplementation local reference "
                                 "payload is exactly eight bytes");
  std::uint64_t ordinal = 0;
  for (std::uint8_t byte : reference.payload)
    ordinal = (ordinal << 8) | byte;
  return ordinal;
}

} // namespace detail

llvm::Error validateHardwareImplementationLocalReference(
    const FinalizedHardwareImplementation &implementation,
    const EncodedArtifactLocalReference &reference) {
  if (reference.artifact != implementation.reference())
    return invalidLocalReference(
        "local reference names a foreign HardwareImplementation");

  switch (static_cast<HardwareImplementationLocalReferenceKind>(
      reference.ownerLocalKind)) {
#define LOOM_HARDWARE_IMPLEMENTATION_LOCAL_REFERENCE_KIND(Ordinal, Kind, Type, \
                                                          CatalogAccessor)     \
  case HardwareImplementationLocalReferenceKind::Kind: {                       \
    auto decoded =                                                             \
        decodeHardwareImplementationLocalReference<Type>(reference);           \
    if (!decoded)                                                              \
      return decoded.takeError();                                              \
    return validateOrdinal(                                                    \
        #Type, decoded->entity.ordinal,                                        \
        implementation.implementation().CatalogAccessor().size());             \
  }
#include "Hardware/Implementation/HardwareImplementationLocalReferences.def"
  }
  return invalidLocalReference("unknown HardwareImplementation owner-local "
                               "reference kind " +
                               llvm::Twine(reference.ownerLocalKind));
}

} // namespace loom::hardware
