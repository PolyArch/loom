#ifndef LOOM_HARDWARE_IMPLEMENTATION_REPRESENTATIONFORMAT_H
#define LOOM_HARDWARE_IMPLEMENTATION_REPRESENTATIONFORMAT_H

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom::hardware {

inline constexpr ArtifactSchemaDescriptor hardwareRepresentationFormatRegistry{
    "loom.hardware_representation_format", SchemaVersion{1, 0}};

enum class RepresentationFormatKind : std::uint32_t {
  SystemVerilogRtl = 0,
  StructuralVerilogGateNetlist = 1,
};

class RepresentationFormatDescriptorRef final {
public:
  static llvm::Expected<RepresentationFormatDescriptorRef>
  get(RepresentationFormatKind kind);

  RepresentationFormatKind kind() const { return kind_; }

  friend bool operator==(RepresentationFormatDescriptorRef lhs,
                         RepresentationFormatDescriptorRef rhs) {
    return lhs.kind_ == rhs.kind_;
  }
  friend bool operator!=(RepresentationFormatDescriptorRef lhs,
                         RepresentationFormatDescriptorRef rhs) {
    return !(lhs == rhs);
  }

private:
  explicit RepresentationFormatDescriptorRef(RepresentationFormatKind kind)
      : kind_(kind) {}

  RepresentationFormatKind kind_;
};

std::vector<std::uint8_t> encodeRepresentationFormatDescriptorRef(
    RepresentationFormatDescriptorRef reference);

llvm::Expected<RepresentationFormatDescriptorRef>
decodeRepresentationFormatDescriptorRef(llvm::ArrayRef<std::uint8_t> bytes);

std::string serializeRepresentationFormatDescriptorRefJson(
    RepresentationFormatDescriptorRef reference);

llvm::Expected<RepresentationFormatDescriptorRef>
parseRepresentationFormatDescriptorRefJson(llvm::StringRef bytes);

} // namespace loom::hardware

#endif // LOOM_HARDWARE_IMPLEMENTATION_REPRESENTATIONFORMAT_H
