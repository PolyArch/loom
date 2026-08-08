#include "Hardware/Implementation/RepresentationIndex.h"

namespace loom::hardware {

llvm::Expected<RepresentationIndex>
indexRepresentation(RepresentationFormatDescriptorRef,
                    const RepresentationLocator &,
                    llvm::ArrayRef<ImplementationPayload>, const BlobStore &) {
  return llvm::make_error<RepresentationIndexFailure>(
      RepresentationIndexFailureKind::Unsupported,
      "representation indexing requires CIRCT");
}

} // namespace loom::hardware
