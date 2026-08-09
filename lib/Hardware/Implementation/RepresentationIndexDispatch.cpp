#include "Hardware/Implementation/RepresentationIndex.h"

#include "Hardware/Implementation/ImplementationRepresentationRoot.h"

#include "RepresentationIndexInternal.h"

#include "llvm/ADT/Twine.h"

#include <utility>
#include <vector>

namespace loom::hardware {

llvm::Expected<RepresentationIndex>
indexRepresentation(RepresentationFormatDescriptorRef formatRef,
                    const RepresentationLocator &exactRoot,
                    llvm::ArrayRef<ImplementationPayload> canonicalPayloads,
                    const BlobStore &blobs) {
  const detail::StaticRepresentationFormatEntry &format =
      detail::getStaticRepresentationFormatEntry(formatRef);
  llvm::Expected<detail::RawIndex> raw =
      format.indexer == detail::BuiltinRepresentationIndexer::IndexedPhysical
          ? detail::indexPhysicalRepresentation(formatRef, exactRoot,
                                                canonicalPayloads, blobs)
          : detail::indexHdlRepresentation(formatRef, exactRoot,
                                           canonicalPayloads, blobs);
  if (!raw)
    return raw.takeError();
  if (!raw->rootVariant)
    return detail::invalidIndex(
        "representation indexer omitted its exact root variant claim");
  std::vector<RepresentationIndex::Entry> entries;
  entries.reserve(raw->entries.size());
  for (detail::RawIndexEntry &entry : raw->entries)
    entries.push_back(RepresentationIndex::Entry{std::move(entry.locator),
                                                 std::move(entry.facts)});
  return RepresentationIndex(formatRef, *raw->rootVariant, raw->stage,
                             exactRoot, std::move(entries),
                             std::move(raw->unresolved));
}

llvm::Expected<RepresentationIndex>
indexRepresentationRoot(const ImplementationRepresentationRoot &root,
                        const BlobStore &blobs) {
  if (llvm::Error error = validateImplementationRepresentationRoot(root))
    return detail::invalidIndex("representation root is invalid: " +
                                llvm::toString(std::move(error)));
  const RepresentationFormatDescriptor &descriptor =
      getRepresentationFormatDescriptor(root.formatRef);
  if (llvm::Error error = validateRepresentationRootAdmission(descriptor, root))
    return detail::invalidIndex("representation root admission is invalid: " +
                                llvm::toString(std::move(error)));
  auto index =
      indexRepresentation(root.formatRef, root.top, root.payloads, blobs);
  if (!index)
    return index.takeError();
  if (index->rootVariant() != root.variant || index->stage() != root.stage)
    return detail::invalidIndex(
        "RepresentationIndex root claim does not match the outer root");
  return index;
}

} // namespace loom::hardware
