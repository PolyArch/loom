#ifndef LOOM_HARDWARE_IMPLEMENTATION_REPRESENTATIONLOCATOR_H
#define LOOM_HARDWARE_IMPLEMENTATION_REPRESENTATIONLOCATOR_H

#include "Hardware/Implementation/RepresentationFormat.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom::hardware {

enum class RepresentationObjectKind : std::uint32_t {
  Module = 0,
  Instance = 1,
  Port = 2,
  Net = 3,
  Register = 4,
  Memory = 5,
  Cell = 6,
  Pin = 7,
  PhysicalObject = 8,
  DeviceResource = 9,
};

struct RepresentationLocator final {
  RepresentationObjectKind kind;
  std::string canonicalName;

  friend bool operator==(const RepresentationLocator &lhs,
                         const RepresentationLocator &rhs) {
    return lhs.kind == rhs.kind && lhs.canonicalName == rhs.canonicalName;
  }
};

/// Encodes one locator as u32be(kind), u64be(name length), and name bytes.
llvm::Expected<std::vector<std::uint8_t>>
encodeRepresentationLocator(const RepresentationLocator &locator);

llvm::Expected<RepresentationLocator>
decodeRepresentationLocator(llvm::ArrayRef<std::uint8_t> bytes);

/// Uses the schema-2.0 field names and exact displayed object-kind spellings.
llvm::Expected<std::string>
serializeRepresentationLocatorJson(const RepresentationLocator &locator);

llvm::Expected<RepresentationLocator>
parseRepresentationLocatorJson(llvm::StringRef bytes);

/// Compares the exact canonical binary encodings without allocating them.
/// Callers validate locators before publishing a sorted catalog.
bool representationLocatorCanonicalLess(const RepresentationLocator &lhs,
                                        const RepresentationLocator &rhs);

/// Applies only the locator grammar and object-kind set owned by the selected
/// format. Object existence, top-prefix resolution, and signal facts belong to
/// the format index and HardwareImplementation finalizer.
llvm::Error
validateRepresentationLocatorSyntax(RepresentationFormatDescriptorRef format,
                                    const RepresentationLocator &locator);

} // namespace loom::hardware

#endif // LOOM_HARDWARE_IMPLEMENTATION_REPRESENTATIONLOCATOR_H
