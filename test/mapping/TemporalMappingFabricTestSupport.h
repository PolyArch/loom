#ifndef LOOM_TEST_MAPPING_TEMPORALMAPPINGFABRICTESTSUPPORT_H
#define LOOM_TEST_MAPPING_TEMPORALMAPPINGFABRICTESTSUPPORT_H

#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/ArrayRef.h"

namespace loom::test {

void addTokenSyncFu(adg::PeBuilder &pe, llvm::ArrayRef<adg::PeValue> inputs,
                    const adg::PortType &type,
                    const ::fabric::ResourceContract &contract);

fabric::FinalizedFabricRoot buildBoundaryTemporalFabric(ArtifactStore &store);

} // namespace loom::test

#endif // LOOM_TEST_MAPPING_TEMPORALMAPPINGFABRICTESTSUPPORT_H
