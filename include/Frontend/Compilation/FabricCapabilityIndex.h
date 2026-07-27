#ifndef LOOM_FRONTEND_COMPILATION_FABRICCAPABILITYINDEX_H
#define LOOM_FRONTEND_COMPILATION_FABRICCAPABILITYINDEX_H

#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <vector>

namespace loom::frontend {

/// A compact invocation-local index over one exact finalized Fabric. It owns
/// no capability facts: every query resolves the stored typed references back
/// through the immutable FabricArtifactView.
class FabricCapabilityIndex final {
public:
  explicit FabricCapabilityIndex(::loom::fabric::FabricArtifactView fabric);

  const ::loom::fabric::FabricArtifactView &fabric() const { return fabric_; }

  /// Returns every concrete operation resource whose Fabric-owned relation
  /// admits `actor`. An empty result proves only resource-level
  /// unavailability; it says nothing about FU topology, placement, routing,
  /// contention, or performance.
  llvm::SmallVector<
      ::loom::ArtifactReference<::loom::fabric::FabricFuTemplateNodeRef>, 4>
  admittingOperationResources(
      const ::dataflow::CanonicalActorSchemaProjection &actor) const;

private:
  struct OperationResource final {
    std::size_t ownerOrdinal = 0;
    ::loom::fabric::FabricFuTemplateNodeRef reference;
  };

  void index(const ::loom::fabric::FabricArtifactView &fabric);

  ::loom::fabric::FabricArtifactView fabric_;
  std::vector<::loom::fabric::FabricArtifactView> owners_;
  std::vector<std::vector<OperationResource>> operationsBySchema_;
};

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_FABRICCAPABILITYINDEX_H
