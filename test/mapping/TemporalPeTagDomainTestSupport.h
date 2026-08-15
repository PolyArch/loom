#pragma once

#include "Fabric/Artifact/FabricArtifact.h"

#include "llvm/Support/Error.h"

namespace loom::test {

llvm::Error verifyTemporalPeIngressTagDomains(
    const fabric::FinalizedFabricRoot &fabric);

} // namespace loom::test
