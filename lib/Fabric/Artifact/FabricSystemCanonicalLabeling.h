#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICSYSTEMCANONICALLABELING_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICSYSTEMCANONICALLABELING_H

#include "Common/Artifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace mlir {
class Operation;
}

namespace fabric {
class SystemOp;
}

namespace loom::fabric::detail {

struct FabricSystemEntityCarrier {
  FabricEntityKind kind;
  FabricEntityId id = 0;
  mlir::Operation *op = nullptr;
};

struct FabricSystemCanonicalLabeling {
  CanonicalSemanticBytes relationBytes;
  std::vector<FabricSystemEntityCarrier> carriers;
  std::vector<mlir::Operation *> canonicalOperationOrder;
  std::vector<std::uint64_t> sourceDependencyToCanonical;
  llvm::DenseMap<mlir::Operation *, FabricOrdinal>
      transferPatternOrdinalByOperation;
};

/// Computes the exact semantic labeling of one verified fabric.system
/// authoring root. Source-local EntityIds, symbol names, child order, and
/// dependency-table order are routing handles only and never enter the result.
llvm::Expected<FabricSystemCanonicalLabeling>
computeFabricSystemCanonicalLabeling(
    ::fabric::SystemOp root,
    llvm::ArrayRef<FabricDirectDependency> sourceDependencies);

/// Rewrites every derived entity, owner-relative reference, dependency
/// ordinal, and operation order to the exact canonical form in `labeling`.
llvm::Error materializeFabricSystemCanonicalForm(
    ::fabric::SystemOp root, const FabricSystemCanonicalLabeling &labeling);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICSYSTEMCANONICALLABELING_H
