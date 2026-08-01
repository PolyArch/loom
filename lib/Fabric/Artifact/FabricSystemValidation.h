#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICSYSTEMVALIDATION_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICSYSTEMVALIDATION_H

#include "Fabric/IR/FabricOps.h"

#include "llvm/Support/Error.h"

namespace loom::fabric::detail {

llvm::Error validateInstructionCoreCohort(::fabric::SystemOp root);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICSYSTEMVALIDATION_H
