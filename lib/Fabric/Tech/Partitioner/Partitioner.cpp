#include "Fabric/Tech/Partitioner/Partitioner.h"

#include "Fabric/Tech/Partitioner/BeamPartitioner.h"
#include "Fabric/Tech/Partitioner/GreedyPartitioner.h"
#include "Fabric/Tech/Partitioner/ILPPartitioner.h"
#include "Fabric/Tech/Partitioner/ListPartitioner.h"
#include "Fabric/Tech/Partitioner/SAPartitioner.h"
#include "llvm/Support/ErrorHandling.h"

#include <memory>

namespace fabric {

std::unique_ptr<IPartitioner>
createPartitioner(::loom::FabricTechMapAlgorithm algorithm) {
  switch (algorithm) {
  case ::loom::FabricTechMapAlgorithm::Greedy:
    return std::make_unique<GreedyPartitioner>();
  case ::loom::FabricTechMapAlgorithm::List:
    return std::make_unique<ListPartitioner>();
  case ::loom::FabricTechMapAlgorithm::Beam:
    return std::make_unique<BeamPartitioner>();
  case ::loom::FabricTechMapAlgorithm::SimulatedAnnealing:
    return std::make_unique<SAPartitioner>();
  case ::loom::FabricTechMapAlgorithm::ILP:
    return std::make_unique<ILPPartitioner>();
  }
  llvm_unreachable("invalid resolved Fabric TechMapping algorithm");
}

} // namespace fabric
