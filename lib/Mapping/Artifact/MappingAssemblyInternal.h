#ifndef LOOM_LIB_MAPPING_ARTIFACT_MAPPINGASSEMBLYINTERNAL_H
#define LOOM_LIB_MAPPING_ARTIFACT_MAPPINGASSEMBLYINTERNAL_H

#include "Mapping/Artifact/MappingArtifact.h"

#include "mlir/IR/OwningOpRef.h"

namespace loom::mapping::detail {

struct CanonicalTechMappingAssembly final {
  mlir::OwningOpRef<mlir::Operation *> root;
  CanonicalSemanticBytes bytes;
};

llvm::Expected<CanonicalTechMappingAssembly>
prepareCanonicalTechMappingAssembly(::mapping::TechOp root);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_ARTIFACT_MAPPINGASSEMBLYINTERNAL_H
