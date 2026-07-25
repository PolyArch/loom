#ifndef LOOM_FRONTEND_PAYLOAD_PAYLOADCARRIER_H
#define LOOM_FRONTEND_PAYLOAD_PAYLOADCARRIER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace llvm {
class Module;
} // namespace llvm

namespace loom {

/// The one object section an ordinary relocatable object carries the complete
/// canonical payload bytes in.
///
/// The section name, the carrier symbol, its alignment, and every container or
/// archive layout detail around it are non-semantic projections. They never
/// enter ArtifactIdentity, so a platform adapter may change them without
/// changing the payload a carrier delivers.
llvm::StringRef relocatablePayloadCarrierSection();

/// Embeds the complete canonical payload bytes in `module`, so the object
/// ordinary compilation emits for it stays self-contained.
///
/// The carrier is the compiled translation unit's own baggage: nothing reads it
/// during that compilation, and only the final link collects it back.
void embedRelocatablePayloadCarrier(
    llvm::Module &module, llvm::ArrayRef<std::uint8_t> canonicalBytes);

/// Reads back the complete canonical payload bytes one relocatable object
/// carries.
///
/// An object without the carrier section is a valid payload-free link input and
/// yields no bytes. An object that cannot be read as an object, or that carries
/// the carrier section more than once, is a typed error: a carrier the ordinary
/// linker selected is never silently discarded.
llvm::Expected<std::optional<std::vector<std::uint8_t>>>
readRelocatablePayloadCarrier(llvm::MemoryBufferRef object);

} // namespace loom

#endif // LOOM_FRONTEND_PAYLOAD_PAYLOADCARRIER_H
