#ifndef LOOM_SIMULATOR_STATICMODELBUILDER_H
#define LOOM_SIMULATOR_STATICMODELBUILDER_H

#include "loom/Mapper/ADGFlattener.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Simulator/StaticModel.h"

#include "llvm/ADT/ArrayRef.h"

namespace loom {
namespace sim {

bool buildStaticMappedModel(const Graph &dfg, const Graph &adg,
                            const MappingState &mapping,
                            llvm::ArrayRef<PEContainment> peContainment,
                            StaticMappedModel &model);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_STATICMODELBUILDER_H
