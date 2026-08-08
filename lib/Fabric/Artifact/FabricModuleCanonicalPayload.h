#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULECANONICALPAYLOAD_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULECANONICALPAYLOAD_H

#include "llvm/Support/Error.h"

namespace fabric {
class ModuleOp;
}

namespace loom::fabric::detail {

llvm::Error stripFabricModuleAuthoringState(::fabric::ModuleOp root);
llvm::Error eraseElaboratedFabricModuleDeclarations(::fabric::ModuleOp root);
llvm::Error validateCanonicalFabricModulePayload(::fabric::ModuleOp root);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULECANONICALPAYLOAD_H
