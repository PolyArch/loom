#ifndef FCC_SIMULATOR_STATICMODELBUILDER_H
#define FCC_SIMULATOR_STATICMODELBUILDER_H

#include "fcc/Mapper/ADGFlattener.h"
#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/MappingState.h"
#include "fcc/Simulator/StaticModel.h"

#include "llvm/ADT/ArrayRef.h"

namespace fcc {
namespace sim {

bool buildStaticMappedModel(const Graph &dfg, const Graph &adg,
                            const MappingState &mapping,
                            llvm::ArrayRef<PEContainment> peContainment,
                            StaticMappedModel &model);

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_STATICMODELBUILDER_H
