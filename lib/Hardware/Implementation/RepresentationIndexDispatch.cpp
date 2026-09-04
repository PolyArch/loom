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

llvm::Expected<detail::RawIndex> indexFabricModel(
    RepresentationFormatDescriptorRef formatRef,
    const RepresentationLocator &exactRoot,
    llvm::ArrayRef<ImplementationPayload> canonicalPayloads) {
  if (!canonicalPayloads.empty())
    return detail::invalidIndex("FabricModel representation has payloads");
  if (exactRoot.kind != RepresentationObjectKind::Model ||
      exactRoot.canonicalName != fabricModelRootCanonicalName)
    return detail::invalidIndex(
        "FabricModel representation has an invalid exact root");
  if (llvm::Error error = validateRepresentationLocatorSyntax(formatRef,
                                                               exactRoot))
    return detail::invalidIndex("FabricModel exact root is invalid: " +
                                llvm::toString(std::move(error)));
  detail::RawIndex raw;
  raw.rootVariant = RepresentationRootVariant::FabricModel;
  raw.stage = std::nullopt;
  raw.entries.push_back(
      {exactRoot, RepresentationObjectFacts{RepresentationObjectKind::Model,
                                            std::nullopt}});
  return raw;
}

} // namespace

llvm::Expected<RepresentationIndex>
indexRepresentation(RepresentationFormatDescriptorRef formatRef,
                    const RepresentationLocator &exactRoot,
                    llvm::ArrayRef<ImplementationPayload> canonicalPayloads,
                    const BlobStore &blobs) {
  const detail::StaticRepresentationFormatEntry &format =
      detail::getStaticRepresentationFormatEntry(formatRef);
  llvm::Expected<detail::RawIndex> raw = [&]()
      -> llvm::Expected<detail::RawIndex> {
    switch (format.indexer) {
    case detail::BuiltinRepresentationIndexer::IndexedPhysical:
      return detail::indexPhysicalRepresentation(formatRef, exactRoot,
                                                 canonicalPayloads, blobs);
    case detail::BuiltinRepresentationIndexer::FabricModel:
      return indexFabricModel(formatRef, exactRoot, canonicalPayloads);
    case detail::BuiltinRepresentationIndexer::SystemVerilogRtl:
    case detail::BuiltinRepresentationIndexer::StructuralVerilogGateNetlist:
      return detail::indexHdlRepresentation(formatRef, exactRoot,
                                            canonicalPayloads, blobs);
    }
    llvm_unreachable("closed representation indexer");
  }();
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
                             std::move(raw->unresolved),
                             std::move(raw->definitions),
                             std::move(raw->rootInstances));
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
  if (format.indexer == detail::BuiltinRepresentationIndexer::FabricModel)
    return detail::invalidIndex(
        "prospective payload indexing does not apply to FabricModel");
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
                             std::move(raw->unresolved),
                             std::move(raw->definitions),
                             std::move(raw->rootInstances));
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
