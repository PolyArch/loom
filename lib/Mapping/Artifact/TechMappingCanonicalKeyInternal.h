#ifndef LOOM_LIB_MAPPING_ARTIFACT_TECHMAPPINGCANONICALKEYINTERNAL_H
#define LOOM_LIB_MAPPING_ARTIFACT_TECHMAPPINGCANONICALKEYINTERNAL_H

#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/IR/MappingOps.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::mapping::detail {

llvm::Expected<std::vector<std::uint8_t>>
canonicalTechMemoryRowKeyFromActorKeys(
    ::loom::fabric::FabricMemoryEngineTemplateRef engine,
    llvm::ArrayRef<llvm::ArrayRef<std::uint8_t>> canonicalActorKeys,
    llvm::ArrayRef<TechMemoryGraphBoundaryView> graphBoundaries,
    llvm::ArrayRef<TechMemoryInternalEdgeView> internalEdges,
    const ArtifactIdentity &dataflowOwner);

std::vector<std::uint8_t> canonicalTechChildKey(mlir::Operation &operation);
std::vector<std::uint8_t>
canonicalTechRealizationPayloadKey(::mapping::ComputeRealizationOp realization);
std::vector<std::uint8_t>
canonicalTechRealizationPayloadKey(::mapping::MemoryRealizationOp realization);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_ARTIFACT_TECHMAPPINGCANONICALKEYINTERNAL_H
