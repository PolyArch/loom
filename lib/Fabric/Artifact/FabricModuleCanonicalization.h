#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULECANONICALIZATION_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULECANONICALIZATION_H

#include "Fabric/Artifact/FabricArtifact.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/Support/Error.h"

#include <vector>

namespace loom::fabric::detail {

struct CanonicalFabricModuleCandidate final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<FabricModuleEntityCorrespondence> entities;
};

llvm::Expected<CanonicalFabricModuleCandidate>
buildCanonicalFabricModuleCandidate(
    ::fabric::ModuleOp source,
    const ::fabric::ModuleDomainAuthoringRelation *domainRelation,
    bool captureCorrespondence);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULECANONICALIZATION_H
