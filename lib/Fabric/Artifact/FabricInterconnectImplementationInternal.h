#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICINTERCONNECTIMPLEMENTATIONINTERNAL_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICINTERCONNECTIMPLEMENTATIONINTERNAL_H

#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Identity/FabricRefImport.h"

namespace loom::fabric::detail {

llvm::Expected<FabricArtifactView> strictImportInterconnectImplementation(
    const ArtifactRootReference &reference,
    const DecodedFabricArtifact &decoded, const ArtifactStore &store);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICINTERCONNECTIMPLEMENTATIONINTERNAL_H
