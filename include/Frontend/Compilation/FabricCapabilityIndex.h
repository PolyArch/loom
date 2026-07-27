#ifndef LOOM_FRONTEND_COMPILATION_FABRICCAPABILITYINDEX_H
#define LOOM_FRONTEND_COMPILATION_FABRICCAPABILITYINDEX_H

#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace loom::frontend {

/// A compact invocation-local index over one exact finalized Fabric. It owns
/// no capability facts: every query resolves the stored typed references back
/// through the immutable FabricArtifactView.
class FabricCapabilityIndex final {
public:
  explicit FabricCapabilityIndex(::loom::fabric::FabricArtifactView fabric);

  const ::loom::fabric::FabricArtifactView &fabric() const { return fabric_; }

  /// Returns every instantiated capability template whose Fabric-owned
  /// relation admits `actor`. An empty result proves only resource-level
  /// unavailability; it says nothing about FU topology, placement, routing,
  /// contention, or performance.
  llvm::SmallVector<
      ::loom::ArtifactReference<::loom::fabric::FabricFuTemplateNodeRef>, 4>
  admittingOperationResources(
      const ::dataflow::CanonicalActorSchemaProjection &actor,
      unsigned indexBitWidth) const;

  /// Counts the concrete operation occurrences represented by the admitted
  /// capability templates. For a System root this expands both module-local
  /// FU occurrences and SpatialCore attachment multiplicity. The count is a
  /// removable Evaluation projection, not a placement or routing proof.
  llvm::Expected<std::uint64_t> admittingOperationResourceCount(
      const ::dataflow::CanonicalActorSchemaProjection &actor,
      unsigned indexBitWidth) const;

  llvm::Expected<llvm::SmallVector<
      ::loom::ArtifactReference<::loom::fabric::FabricFuTemplateNodeRef>, 4>>
  admittingOperationResources(::mlir::Operation *actor) const;

  /// Returns every concrete memory capability alternative whose Fabric-owned
  /// actor, service, access, and resource relations admit `actor`. Malformed
  /// actor semantics are errors; an empty result is ordinary resource-level
  /// unavailability and does not prove Mapping infeasibility.
  llvm::Expected<llvm::SmallVector<
      ::loom::ArtifactReference<
          ::loom::fabric::FabricMemoryCapabilityAlternativeRef>,
      4>>
  admittingMemoryResources(::mlir::Operation *actor) const;

private:
  struct OperationResource final {
    std::size_t ownerOrdinal = 0;
    ::loom::fabric::FabricFuTemplateNodeRef reference;
    std::uint64_t rootOccurrenceCount = 0;
    std::uint64_t localOccurrenceCount = 0;
  };

  struct MemoryResource final {
    std::size_t ownerOrdinal = 0;
    ::loom::fabric::FabricMemoryOperationPortRef reference;
  };

  void index(const ::loom::fabric::FabricArtifactView &fabric,
             std::uint64_t rootOccurrenceCount);

  ::loom::fabric::FabricArtifactView fabric_;
  std::vector<::loom::fabric::FabricArtifactView> owners_;
  std::vector<std::vector<OperationResource>> operationsBySchema_;
  std::vector<std::vector<MemoryResource>> memoryPortsBySchema_;
};

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_FABRICCAPABILITYINDEX_H
