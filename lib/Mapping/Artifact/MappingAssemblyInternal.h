#ifndef LOOM_LIB_MAPPING_ARTIFACT_MAPPINGASSEMBLYINTERNAL_H
#define LOOM_LIB_MAPPING_ARTIFACT_MAPPINGASSEMBLYINTERNAL_H

#include "Mapping/Artifact/MappingArtifact.h"

#include "mlir/IR/OwningOpRef.h"

namespace loom::mapping::detail {

struct CanonicalTechMappingAssembly final {
  mlir::OwningOpRef<mlir::Operation *> root;
  CanonicalSemanticBytes bytes;
};

struct CanonicalSpatialMappingAssembly final {
  mlir::OwningOpRef<mlir::Operation *> root;
  CanonicalSemanticBytes bytes;
};

struct CanonicalSystemMappingAssembly final {
  mlir::OwningOpRef<mlir::Operation *> root;
  CanonicalSemanticBytes bytes;
};

llvm::Expected<CanonicalTechMappingAssembly>
prepareCanonicalTechMappingAssembly(::mapping::TechOp root);

llvm::Expected<CanonicalSpatialMappingAssembly>
prepareCanonicalSpatialMappingAssembly(::mapping::SpatialOp root);

llvm::Expected<CanonicalSystemMappingAssembly>
prepareCanonicalSystemMappingAssembly(::mapping::SystemOp root);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_ARTIFACT_MAPPINGASSEMBLYINTERNAL_H
