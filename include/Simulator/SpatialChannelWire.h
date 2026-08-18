#ifndef LOOM_SIMULATOR_SPATIALCHANNELWIRE_H
#define LOOM_SIMULATOR_SPATIALCHANNELWIRE_H

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::sim {

llvm::Expected<std::vector<std::uint8_t>> encodeSpatialChannelStream(
    const CanonicalStreamSequence &stream,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    dataflow::RootedGraphLaunchRef launch,
    std::uint64_t streamOutputOrdinal, std::uint64_t memoryObjectCount);

llvm::Expected<CanonicalStreamSequence> decodeSpatialChannelStream(
    llvm::ArrayRef<std::uint8_t> bytes,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    dataflow::RootedGraphLaunchRef launch,
    std::uint64_t streamInputOrdinal, std::uint64_t memoryObjectCount);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SPATIALCHANNELWIRE_H
