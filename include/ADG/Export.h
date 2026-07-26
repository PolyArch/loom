#ifndef LOOM_ADG_EXPORT_H
#define LOOM_ADG_EXPORT_H

#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace loom::adg {

/// Writes `<outputBase>.mlir` and `<outputBase>.html` from one finalized root.
/// The output pair is a removable projection and never participates in Fabric
/// identity or publication.
llvm::Error exportFabricDesign(const loom::fabric::FinalizedFabricRoot &root,
                               const loom::ArtifactStore &store,
                               llvm::StringRef outputBase);

} // namespace loom::adg

#endif // LOOM_ADG_EXPORT_H
