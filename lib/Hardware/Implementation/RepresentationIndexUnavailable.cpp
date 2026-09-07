#include "RepresentationIndexInternal.h"

namespace loom::hardware::detail {

llvm::Expected<RawIndex> indexHdlRepresentation(
    RepresentationFormatDescriptorRef, const RepresentationLocator &,
    llvm::ArrayRef<ImplementationPayloadBytes>) {
  return unsupportedIndex("HDL representation indexing requires CIRCT");
}

llvm::Expected<RawIndex> indexHdlRepresentation(
    RepresentationFormatDescriptorRef format, const RepresentationLocator &root,
    llvm::ArrayRef<ImplementationPayload>, const BlobStore &) {
  return indexHdlRepresentation(
      format, root, llvm::ArrayRef<ImplementationPayloadBytes>{});
}

} // namespace loom::hardware::detail
