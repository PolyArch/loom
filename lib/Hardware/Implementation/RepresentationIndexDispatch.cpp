#include "Hardware/Implementation/RepresentationIndex.h"

#include "Common/BlobStore.h"
#include "Hardware/Implementation/ImplementationRepresentationRoot.h"

#include "RepresentationIndexInternal.h"

#include "llvm/ADT/Twine.h"

#include <list>
#include <mutex>
#include <utility>
#include <vector>

namespace loom::hardware {
namespace {

constexpr std::size_t kRepresentationIndexCacheCapacity = 8;

struct CachedRepresentationIndex final {
  std::vector<std::uint8_t> rootBytes;
  RepresentationIndex index;
};

std::list<CachedRepresentationIndex> &representationIndexCache() {
  static std::list<CachedRepresentationIndex> cache;
  return cache;
}

std::mutex &representationIndexCacheMutex() {
  static std::mutex mutex;
  return mutex;
}

} // namespace

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

llvm::Expected<RepresentationIndex> indexProspectiveRepresentation(
    RepresentationFormatDescriptorRef formatRef,
    const RepresentationLocator &exactRoot,
    llvm::ArrayRef<ImplementationPayloadBytes> payloads) {
  const detail::StaticRepresentationFormatEntry &format =
      detail::getStaticRepresentationFormatEntry(formatRef);
  if (format.indexer == detail::BuiltinRepresentationIndexer::IndexedPhysical)
    return detail::invalidIndex(
        "prospective payload indexing is available only for HDL formats");
  auto raw = detail::indexHdlRepresentation(formatRef, exactRoot, payloads);
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
  auto rootBytes = encodeImplementationRepresentationRoot(root);
  if (!rootBytes)
    return rootBytes.takeError();
  {
    std::lock_guard<std::mutex> lock(representationIndexCacheMutex());
    auto &cache = representationIndexCache();
    auto found = llvm::find_if(cache, [&](const CachedRepresentationIndex &row) {
      return row.rootBytes == *rootBytes;
    });
    if (found != cache.end()) {
      for (const ImplementationPayload &payload : root.payloads) {
        auto verified = blobs.get(payload.blobDigest);
        if (!verified)
          return detail::invalidIndex(
              "cached representation payload verification failed: " +
              llvm::toString(verified.takeError()));
      }
      RepresentationIndex index = found->index;
      cache.splice(cache.begin(), cache, found);
      return index;
    }
  }
  auto index =
      indexRepresentation(root.formatRef, root.top, root.payloads, blobs);
  if (!index)
    return index.takeError();
  if (index->rootVariant() != root.variant || index->stage() != root.stage)
    return detail::invalidIndex(
        "RepresentationIndex root claim does not match the outer root");
  {
    std::lock_guard<std::mutex> lock(representationIndexCacheMutex());
    auto &cache = representationIndexCache();
    cache.push_front(CachedRepresentationIndex{std::move(*rootBytes), *index});
    if (cache.size() > kRepresentationIndexCacheCapacity)
      cache.pop_back();
  }
  return index;
}

} // namespace loom::hardware
