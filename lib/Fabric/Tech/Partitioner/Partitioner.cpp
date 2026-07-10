#include "Fabric/Tech/Partitioner/Partitioner.h"

#include "Fabric/Tech/Partitioner/BeamPartitioner.h"
#include "Fabric/Tech/Partitioner/GreedyPartitioner.h"
#include "Fabric/Tech/Partitioner/ILPPartitioner.h"
#include "Fabric/Tech/Partitioner/ListPartitioner.h"
#include "Fabric/Tech/Partitioner/SAPartitioner.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace fabric {

std::unique_ptr<IPartitioner>
createPartitioner(::llvm::StringRef algorithm) {
  if (algorithm == "list")
    return std::make_unique<ListPartitioner>();
  if (algorithm == "beam")
    return std::make_unique<BeamPartitioner>();
  if (algorithm == "sa")
    return std::make_unique<SAPartitioner>();
  if (algorithm == "ilp")
    return std::make_unique<ILPPartitioner>();
  // Default to "greedy" for unknown / empty values; Config validation
  // should already have rejected invalid algorithm names upstream.
  return std::make_unique<GreedyPartitioner>();
}

} // namespace fabric
