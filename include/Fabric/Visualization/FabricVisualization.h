#ifndef LOOM_FABRIC_VISUALIZATION_FABRICVISUALIZATION_H
#define LOOM_FABRIC_VISUALIZATION_FABRICVISUALIZATION_H

#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace loom::fabric {

/// Writes a self-contained, nonsemantic HTML projection of one exact Fabric
/// root and its reachable imported Module roots. All graph geometry is
/// computed before this function writes the document.
llvm::Error writeFabricVisualizationHtml(const FinalizedFabricRoot &root,
                                         const ArtifactStore &store,
                                         llvm::raw_ostream &output);

} // namespace loom::fabric

#endif // LOOM_FABRIC_VISUALIZATION_FABRICVISUALIZATION_H
