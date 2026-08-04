#ifndef LOOM_TEST_MAPPING_SPATIALMEMORYCONSTRAINTTESTSUPPORT_H
#define LOOM_TEST_MAPPING_SPATIALMEMORYCONSTRAINTTESTSUPPORT_H

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"

namespace mlir {
class MLIRContext;
}

namespace loom::test {

void exerciseSpatialMemoryOperationPortRelations(
    mlir::MLIRContext &context,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric, const ArtifactStore &store);

} // namespace loom::test

#endif // LOOM_TEST_MAPPING_SPATIALMEMORYCONSTRAINTTESTSUPPORT_H
