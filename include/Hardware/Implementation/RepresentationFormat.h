#ifndef LOOM_HARDWARE_IMPLEMENTATION_REPRESENTATIONFORMAT_H
#define LOOM_HARDWARE_IMPLEMENTATION_REPRESENTATIONFORMAT_H

#include "Common/Artifact.h"
#include "Hardware/Implementation/ImplementationPayload.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llvm::json {
class Object;
}

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

inline constexpr ArtifactSchemaDescriptor hardwareRepresentationFormatRegistry{
    "loom.hardware_representation_format", SchemaVersion{2, 1}};

enum class RepresentationFormatKind : std::uint32_t {
  SystemVerilogRtl = 0,
  StructuralVerilogGateNetlist = 1,
  IndexedPhysical = 2,
};

/// Closed root variants of one HardwareImplementation representation, with
/// their stable tags. Owned here because the format descriptor alone admits a
/// variant and stage set.
enum class RepresentationRootVariant : std::uint32_t {
  Rtl = 0,
  GateNetlist = 1,
  AsicPhysical = 2,
  FpgaPhysical = 3,
  FpgaImage = 4,
};

/// Stable stage tags for the physical root variants. `Extracted` is legal
/// only for `AsicPhysical`.
enum class RepresentationPhysicalStage : std::uint32_t {
  Placed = 0,
  Routed = 1,
  Extracted = 2,
};

enum class RepresentationTextPolicy : std::uint32_t {
  Opaque = 0,
  Utf8LfNoNul = 1,
};

enum class RepresentationLanguageProfile : std::uint32_t {
  Ieee1800_2017 = 0,
  Ieee1364_2005 = 1,
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

struct RepresentationPayloadContract final {
  PayloadRole role;
  llvm::StringRef mediaType;
  std::uint64_t minimumCount;
  std::optional<std::uint64_t> maximumCount;
  RepresentationTextPolicy textPolicy;

  friend bool operator==(const RepresentationPayloadContract &lhs,
                         const RepresentationPayloadContract &rhs) {
    return lhs.role == rhs.role && lhs.mediaType == rhs.mediaType &&
           lhs.minimumCount == rhs.minimumCount &&
           lhs.maximumCount == rhs.maximumCount &&
           lhs.textPolicy == rhs.textPolicy;
  }
};

/// One exact admitted root form and all contracts specific to that form.
struct RepresentationRootAdmission final {
  RepresentationRootVariant variant;
  std::optional<RepresentationPhysicalStage> stage;
  RepresentationObjectKind exactRootKind;
  llvm::ArrayRef<RepresentationPayloadContract> payloadContracts;
  llvm::ArrayRef<RepresentationObjectKind> admittedObjectKinds;
};

struct RepresentationFormatDescriptor final {
  RepresentationFormatDescriptorRef formatRef;
  std::optional<PayloadRole> frontendSourceRole;
  std::optional<RepresentationLanguageProfile> languageProfile;
  llvm::ArrayRef<RepresentationRootAdmission> admittedRoots;
};

/// Data-driven admission query: the descriptor alone decides whether one
/// (variant, stage) pair is admitted. No caller-side branch may substitute.
bool admitsRepresentationRoot(const RepresentationFormatDescriptor &descriptor,
                              RepresentationRootVariant variant,
                              std::optional<RepresentationPhysicalStage> stage);

const RepresentationRootAdmission *findRepresentationRootAdmission(
    const RepresentationFormatDescriptor &descriptor,
    RepresentationRootVariant variant,
    std::optional<RepresentationPhysicalStage> stage);

llvm::Error validateRepresentationPayloadCatalog(
    const RepresentationRootAdmission &admission,
    llvm::ArrayRef<ImplementationPayload> canonicalPayloads);

/// Returns immutable metadata owned by the closed static format registry.
const RepresentationFormatDescriptor &
getRepresentationFormatDescriptor(RepresentationFormatDescriptorRef reference);

std::vector<std::uint8_t> encodeRepresentationFormatDescriptorRef(
    RepresentationFormatDescriptorRef reference);

llvm::Expected<RepresentationFormatDescriptorRef>
decodeRepresentationFormatDescriptorRef(llvm::ArrayRef<std::uint8_t> bytes);

std::string serializeRepresentationFormatDescriptorRefJson(
    RepresentationFormatDescriptorRef reference);

/// Field validation and semantic construction from an already parsed object.
/// This is the composition entry point for an enclosing canonical document;
/// the text entry point additionally enforces exact canonical bytes.
llvm::Expected<RepresentationFormatDescriptorRef>
parseRepresentationFormatDescriptorRefJsonValue(
    const llvm::json::Object &object);

llvm::Expected<RepresentationFormatDescriptorRef>
parseRepresentationFormatDescriptorRefJson(llvm::StringRef bytes);

} // namespace loom::hardware

#endif // LOOM_HARDWARE_IMPLEMENTATION_REPRESENTATIONFORMAT_H
