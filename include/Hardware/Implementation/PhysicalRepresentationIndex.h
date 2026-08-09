#ifndef LOOM_HARDWARE_IMPLEMENTATION_PHYSICALREPRESENTATIONINDEX_H
#define LOOM_HARDWARE_IMPLEMENTATION_PHYSICALREPRESENTATIONINDEX_H

#include "Hardware/Implementation/ImplementationRepresentationRoot.h"
#include "Hardware/Implementation/RepresentationIndex.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <vector>

namespace loom::hardware {

struct PhysicalRepresentationObject final {
  RepresentationLocator locator;
  std::optional<RepresentationSignalGeometry> signalGeometry;

  friend bool operator==(const PhysicalRepresentationObject &lhs,
                         const PhysicalRepresentationObject &rhs) {
    return lhs.locator == rhs.locator &&
           lhs.signalGeometry == rhs.signalGeometry;
  }
};

struct PhysicalRepresentationIndexPayload final {
  RepresentationFormatDescriptorRef formatRef;
  RepresentationRootVariant variant;
  std::optional<RepresentationPhysicalStage> stage;
  RepresentationLocator top;
  std::string indexLogicalName;
  std::vector<ImplementationPayload> payloads;
  std::vector<PhysicalRepresentationObject> objects;
  std::vector<RepresentationLocator> unresolvedExternalDefinitions;

  friend bool operator==(const PhysicalRepresentationIndexPayload &lhs,
                         const PhysicalRepresentationIndexPayload &rhs) {
    return lhs.formatRef == rhs.formatRef && lhs.variant == rhs.variant &&
           lhs.stage == rhs.stage && lhs.top == rhs.top &&
           lhs.indexLogicalName == rhs.indexLogicalName &&
           lhs.payloads == rhs.payloads && lhs.objects == rhs.objects &&
           lhs.unresolvedExternalDefinitions ==
               rhs.unresolvedExternalDefinitions;
  }
};

llvm::Expected<PhysicalRepresentationIndexPayload>
createPhysicalRepresentationIndexPayload(
    RepresentationFormatDescriptorRef formatRef,
    RepresentationRootVariant variant,
    std::optional<RepresentationPhysicalStage> stage, RepresentationLocator top,
    std::string indexLogicalName, std::vector<ImplementationPayload> payloads,
    std::vector<PhysicalRepresentationObject> objects,
    std::vector<RepresentationLocator> unresolvedExternalDefinitions);

llvm::Error validatePhysicalRepresentationIndexPayload(
    const PhysicalRepresentationIndexPayload &index);

llvm::Expected<std::string> serializePhysicalRepresentationIndexPayloadJson(
    const PhysicalRepresentationIndexPayload &index);

llvm::Expected<PhysicalRepresentationIndexPayload>
parsePhysicalRepresentationIndexPayloadJson(llvm::StringRef bytes);

} // namespace loom::hardware

#endif // LOOM_HARDWARE_IMPLEMENTATION_PHYSICALREPRESENTATIONINDEX_H
