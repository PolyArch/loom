#include "RepresentationIndexInternal.h"

namespace loom::hardware::detail {

llvm::Expected<RawIndex> indexHdlRepresentation(
    RepresentationFormatDescriptorRef, const RepresentationLocator &,
    llvm::ArrayRef<ImplementationPayload>, const BlobStore &) {
  return unsupportedIndex("HDL representation indexing requires CIRCT");
}

} // namespace loom::hardware::detail
