#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICCANONICALLABELING_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICCANONICALLABELING_H

#include "Common/Artifact.h"
#include "Fabric/Identity/FabricRefs.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace fabric {
class ModuleOp;
}

namespace loom::fabric::detail {

struct FabricEntityCarrier {
  FabricEntityKind kind;
  std::uint64_t id = 0;
  mlir::Operation *op = nullptr;
};

struct FabricFuTemplateCarrier {
  std::uint64_t id = 0;
  mlir::Operation *representative = nullptr;
  std::vector<mlir::Operation *> canonicalNodeOrder;
};

struct FabricCanonicalLabeling {
  CanonicalSemanticBytes relationBytes;
  std::vector<FabricEntityCarrier> carriers;
  std::vector<FabricFuTemplateCarrier> fuTemplates;
  std::vector<mlir::Operation *> canonicalOperationOrder;
  llvm::DenseMap<mlir::Operation *, std::uint64_t> fuTemplateIdByOccurrence;
  llvm::DenseMap<mlir::Operation *, std::vector<std::uint8_t>>
      canonicalFuCapabilityDomainByOccurrence;
};

/// Computes the exact semantic labeling of one already elaborated, declaration-
/// free Fabric Module root. The caller owns structural verification and must
/// reject residual fabric.instantiate operations before calling this function.
llvm::Expected<FabricCanonicalLabeling>
computeFabricModuleCanonicalLabeling(::fabric::ModuleOp root);

/// Replaces every derived ID carrier with the exact assignment in `labeling`.
/// Fabric finalization is the only caller permitted to persist these values;
/// author-supplied values are never preserved.
llvm::Error
materializeFabricCanonicalIds(const FabricCanonicalLabeling &labeling);

/// Rewrites every FU occurrence's owner-local capability domain into the
/// exact canonical FU-node order captured by `labeling`.
llvm::Error materializeFabricCanonicalFuCapabilityDomains(
    const FabricCanonicalLabeling &labeling);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICCANONICALLABELING_H
