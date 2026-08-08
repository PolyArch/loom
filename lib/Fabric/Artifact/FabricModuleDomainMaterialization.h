#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULEDOMAINMATERIALIZATION_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULEDOMAINMATERIALIZATION_H

#include "llvm/Support/Error.h"

namespace fabric {
class ModuleOp;
} // namespace fabric

namespace loom::fabric::detail {

struct FabricCanonicalLabeling;
struct NormalizedModuleDomainRelation;

llvm::Error materializeFabricModuleDomainRelation(
    ::fabric::ModuleOp root, const NormalizedModuleDomainRelation &relation,
    const FabricCanonicalLabeling &labeling);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICMODULEDOMAINMATERIALIZATION_H
