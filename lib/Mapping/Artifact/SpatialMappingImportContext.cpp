#include "Mapping/Artifact/MappingArtifact.h"

#include "Common/ArtifactLocalReference.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <system_error>
#include <vector>

namespace loom::mapping {
namespace {

constexpr llvm::StringLiteral algorithmIdentity =
    "loom.mapping.spatial_import_context.1";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_mapping_import_context_invalid: " + message);
}

void appendU32Be(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendBytes(std::vector<std::uint8_t> &target,
                 llvm::ArrayRef<std::uint8_t> bytes) {
  appendU32Be(target, static_cast<std::uint32_t>(bytes.size()));
  target.insert(target.end(), bytes.begin(), bytes.end());
}

void appendText(std::vector<std::uint8_t> &target, llvm::StringRef text) {
  appendBytes(target, llvm::ArrayRef<std::uint8_t>(
                          reinterpret_cast<const std::uint8_t *>(text.data()),
                          text.size()));
}

std::array<std::uint8_t, 32>
deriveKey(llvm::ArrayRef<ArtifactRootReference> references) {
  std::vector<std::uint8_t> preimage;
  appendText(preimage, algorithmIdentity);
  appendText(preimage, mappingArtifactSchema.identity);
  appendU32Be(preimage, mappingArtifactSchema.version.major);
  appendU32Be(preimage, mappingArtifactSchema.version.minor);
  appendU32Be(preimage, static_cast<std::uint32_t>(references.size()));
  for (const ArtifactRootReference &reference : references)
    appendBytes(preimage, encodeArtifactRootReference(reference));
  return llvm::SHA256::hash(preimage);
}

std::uint64_t elapsedNanoseconds(std::chrono::steady_clock::time_point begin) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - begin)
          .count());
}

std::uint64_t retainedBytes(const FinalizedSpatialMapping &mapping) {
  const SpatialMappingView &view = mapping.view();
  return sizeof(FinalizedSpatialMapping) +
         mapping.canonicalBytes().bytes().size() +
         view.computeBindings().size() * sizeof(SpatialComputeBindingView) +
         view.memoryEngineBindings().size() *
             sizeof(SpatialMemoryEngineBindingView) +
         view.memoryBindings().size() * sizeof(SpatialMemoryBindingView) +
         view.registerFifoTransfers().size() *
             sizeof(SpatialRegisterFifoTransferView) +
         view.routeTrees().size() * sizeof(SpatialRouteTreeView) +
         view.resourceUses().size() * sizeof(SpatialResourceUseView) +
         view.physicalTagSegments().size() *
             sizeof(SpatialPhysicalTagSegmentView);
}

} // namespace

const FinalizedSpatialMapping *SpatialMappingImportContext::find(
    const ArtifactRootReference &reference) const {
  const auto found =
      std::lower_bound(references_.begin(), references_.end(), reference,
                       [](const ArtifactRootReference &left,
                          const ArtifactRootReference &right) {
                         return artifactRootReferenceLess(left, right);
                       });
  if (found == references_.end() ||
      artifactRootReferenceLess(reference, *found) ||
      artifactRootReferenceLess(*found, reference))
    return nullptr;
  return mappings_[static_cast<std::size_t>(found - references_.begin())].get();
}

llvm::Expected<SpatialMappingImportContext> buildSpatialMappingImportContext(
    llvm::ArrayRef<ArtifactRootReference> references,
    const ArtifactStore &store) {
  const auto begin = std::chrono::steady_clock::now();
  std::vector<ArtifactRootReference> canonical(references.begin(),
                                               references.end());
  llvm::sort(canonical, artifactRootReferenceLess);
  if (std::adjacent_find(canonical.begin(), canonical.end()) != canonical.end())
    return invalid("reference set contains a duplicate");

  std::vector<std::shared_ptr<const FinalizedSpatialMapping>> mappings;
  mappings.reserve(canonical.size());
  SpatialMappingImportContextStatistics statistics;
  for (const ArtifactRootReference &reference : canonical) {
    if (reference.schemaIdentity != mappingArtifactSchema.identity ||
        reference.schemaVersion != mappingArtifactSchema.version)
      return invalid("reference set contains a non-Mapping artifact");
    auto mapping = importSpatialMapping(reference, store);
    if (!mapping)
      return mapping.takeError();
    statistics.retainedBytes += retainedBytes(*mapping);
    statistics.deterministicWork += 1 +
                                    mapping->view().computeBindings().size() +
                                    mapping->view().memoryBindings().size() +
                                    mapping->view().routeTrees().size() +
                                    mapping->view().resourceUses().size();
    mappings.push_back(
        std::make_shared<const FinalizedSpatialMapping>(std::move(*mapping)));
  }
  statistics.mappingCount = mappings.size();
  statistics.constructionNanoseconds = elapsedNanoseconds(begin);
  statistics.retainedBytes +=
      canonical.capacity() * sizeof(ArtifactRootReference);
  return SpatialMappingImportContext(deriveKey(canonical), std::move(canonical),
                                     std::move(mappings), statistics);
}

llvm::Expected<const FinalizedSpatialMapping *>
resolveSpatialMappingImport(const SpatialMappingImportContext &context,
                            const ArtifactRootReference &reference) {
  const FinalizedSpatialMapping *mapping = context.find(reference);
  if (!mapping)
    return invalid("reference is outside the exact invocation set");
  return mapping;
}

llvm::Expected<std::shared_ptr<const FinalizedSpatialMapping>>
resolveSpatialMappingImportHandle(const SpatialMappingImportContext &context,
                                  const ArtifactRootReference &reference) {
  const auto found = std::lower_bound(
      context.references_.begin(), context.references_.end(), reference,
      [](const ArtifactRootReference &left,
         const ArtifactRootReference &right) {
        return artifactRootReferenceLess(left, right);
      });
  if (found == context.references_.end() ||
      artifactRootReferenceLess(reference, *found) ||
      artifactRootReferenceLess(*found, reference))
    return invalid("reference is outside the exact invocation set");
  return context
      .mappings_[static_cast<std::size_t>(found - context.references_.begin())];
}

} // namespace loom::mapping
