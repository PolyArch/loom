#ifndef LOOM_HARDWARE_IMPLEMENTATION_IMPLEMENTATIONREPRESENTATIONROOT_H
#define LOOM_HARDWARE_IMPLEMENTATION_IMPLEMENTATIONREPRESENTATIONROOT_H

#include "Hardware/Implementation/ImplementationPayload.h"
#include "Hardware/Implementation/RepresentationFormat.h"
#include "Hardware/Implementation/RepresentationLocator.h"

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

/// Canonical typed representation root for HardwareImplementation 3.0.
struct ImplementationRepresentationRoot final {
  RepresentationRootVariant variant;
  /// Present exactly for the physical variants.
  std::optional<RepresentationPhysicalStage> stage;
  RepresentationFormatDescriptorRef formatRef;
  RepresentationLocator top;
  /// Canonical and nonempty, owned by the ImplementationPayload rules.
  std::vector<ImplementationPayload> payloads;

  friend bool operator==(const ImplementationRepresentationRoot &lhs,
                         const ImplementationRepresentationRoot &rhs) {
    return lhs.variant == rhs.variant && lhs.stage == rhs.stage &&
           lhs.formatRef == rhs.formatRef && lhs.top == rhs.top &&
           lhs.payloads == rhs.payloads;
  }
};

/// Validates and canonicalizes one root. The payload catalog is sorted into
/// the sole canonical order, so authoring order never changes the result.
llvm::Expected<ImplementationRepresentationRoot>
createImplementationRepresentationRoot(
    RepresentationRootVariant variant,
    std::optional<RepresentationPhysicalStage> stage,
    RepresentationFormatDescriptorRef formatRef, RepresentationLocator top,
    std::vector<ImplementationPayload> payloads);

llvm::Error validateImplementationRepresentationRoot(
    const ImplementationRepresentationRoot &root);

/// Admission against the exact selected descriptor: the root's own format
/// reference must match the descriptor, and the descriptor must admit the
/// (variant, stage) pair. No physical variant is admitted by an initial HDL
/// descriptor.
llvm::Error validateRepresentationRootAdmission(
    const RepresentationFormatDescriptor &descriptor,
    const ImplementationRepresentationRoot &root);

llvm::Expected<llvm::StringRef>
representationRootVariantSpelling(RepresentationRootVariant variant);

std::optional<RepresentationRootVariant>
parseRepresentationRootVariantSpelling(llvm::StringRef spelling);

llvm::Expected<llvm::StringRef>
representationPhysicalStageSpelling(RepresentationPhysicalStage stage);

std::optional<RepresentationPhysicalStage>
parseRepresentationPhysicalStageSpelling(llvm::StringRef spelling);

/// Binary framing: u32be(variant), u32be(stage) for the physical variants,
/// the exact format-reference bytes, the exact locator bytes, then
/// u64be(payload count) followed by each exact payload record.
llvm::Expected<std::vector<std::uint8_t>>
encodeImplementationRepresentationRoot(
    const ImplementationRepresentationRoot &root);

llvm::Expected<ImplementationRepresentationRoot>
decodeImplementationRepresentationRoot(llvm::ArrayRef<std::uint8_t> bytes);

/// Canonical JSON uses the exact displayed variant and stage spellings.
llvm::Expected<std::string> serializeImplementationRepresentationRootJson(
    const ImplementationRepresentationRoot &root);

llvm::Expected<ImplementationRepresentationRoot>
parseImplementationRepresentationRootJsonValue(
    const llvm::json::Object &object);

llvm::Expected<ImplementationRepresentationRoot>
parseImplementationRepresentationRootJson(llvm::StringRef bytes);

} // namespace loom::hardware

#endif // LOOM_HARDWARE_IMPLEMENTATION_IMPLEMENTATIONREPRESENTATIONROOT_H
