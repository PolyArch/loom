#ifndef LOOM_SIMULATOR_RUNTIMEIMAGEBUILDER_H
#define LOOM_SIMULATOR_RUNTIMEIMAGEBUILDER_H

#include "loom/Mapper/ADGFlattener.h"
#include "loom/Mapper/ConfigGen.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Simulator/RuntimeImage.h"

#include "llvm/ADT/ArrayRef.h"

#include <string>

namespace loom {
namespace sim {

bool buildRuntimeImage(const Graph &dfg, const Graph &adg,
                       const MappingState &mapping,
                       llvm::ArrayRef<PEContainment> peContainment,
                       llvm::ArrayRef<loom::ConfigGen::ConfigSlice> configSlices,
                       llvm::ArrayRef<uint32_t> configWords,
                       RuntimeImage &image, std::string &error);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_RUNTIMEIMAGEBUILDER_H
