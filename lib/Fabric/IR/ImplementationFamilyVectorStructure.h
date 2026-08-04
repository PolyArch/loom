#ifndef LOOM_LIB_FABRIC_IR_IMPLEMENTATIONFAMILYVECTORSTRUCTURE_H
#define LOOM_LIB_FABRIC_IR_IMPLEMENTATIONFAMILYVECTORSTRUCTURE_H

#include "Fabric/IR/ImplementationFamily.h"

namespace fabric::detail {

llvm::Error admitFixedVectorSliceAlignMergeAdmission(
    const FamilyCapabilityParams &capability,
    const ::dataflow::CanonicalActorSchemaProjection &actor);

llvm::Error admitFixedVectorShuffleAdmission(
    const FamilyCapabilityParams &capability,
    const ::dataflow::CanonicalActorSchemaProjection &actor);

} // namespace fabric::detail

#endif // LOOM_LIB_FABRIC_IR_IMPLEMENTATIONFAMILYVECTORSTRUCTURE_H
