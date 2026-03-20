#ifndef FCC_SIMULATOR_RUNTIMEIMAGEBUILDER_H
#define FCC_SIMULATOR_RUNTIMEIMAGEBUILDER_H

#include "fcc/Mapper/ADGFlattener.h"
#include "fcc/Mapper/ConfigGen.h"
#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/MappingState.h"
#include "fcc/Simulator/RuntimeImage.h"

#include "llvm/ADT/ArrayRef.h"

#include <string>

namespace fcc {
namespace sim {

bool buildRuntimeImage(const Graph &dfg, const Graph &adg,
                       const MappingState &mapping,
                       llvm::ArrayRef<PEContainment> peContainment,
                       llvm::ArrayRef<fcc::ConfigGen::ConfigSlice> configSlices,
                       llvm::ArrayRef<uint32_t> configWords,
                       RuntimeImage &image, std::string &error);

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_RUNTIMEIMAGEBUILDER_H
