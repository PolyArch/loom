#ifndef FCC_MAPPER_TECHMAPPER_H
#define FCC_MAPPER_TECHMAPPER_H

#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/MappingState.h"
#include "fcc/Mapper/Types.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
class ModuleOp;
}

namespace fcc {

struct FUConfigField {
  unsigned opIndex = 0;
  std::string opName;
  uint64_t sel = 0;
  bool discard = false;
  bool disconnect = false;
};

struct FUConfigSelection {
  IdIndex hwNodeId = INVALID_ID;
  std::string hwName;
  std::string peName;
  llvm::SmallVector<IdIndex, 4> swNodeIds;
  llvm::SmallVector<FUConfigField, 2> fields;
};

enum class TechMappedEdgeKind : uint8_t {
  Routed = 0,
  IntraFU = 1,
};

class TechMapper {
public:
  struct PortBinding {
    IdIndex swPortId = INVALID_ID;
    unsigned hwPortIndex = 0;
  };

  struct Candidate {
    IdIndex hwNodeId = INVALID_ID;
    llvm::SmallVector<FUConfigField, 2> configFields;
  };

  struct Unit {
    IdIndex contractedNodeId = INVALID_ID;
    llvm::SmallVector<IdIndex, 4> swNodes;
    llvm::SmallVector<IdIndex, 4> internalEdges;
    llvm::SmallVector<IdIndex, 4> contractedInputPorts;
    llvm::SmallVector<IdIndex, 4> contractedOutputPorts;
    llvm::SmallVector<PortBinding, 4> inputBindings;
    llvm::SmallVector<PortBinding, 4> outputBindings;
    llvm::SmallVector<Candidate, 4> candidates;
    bool configurable = false;
  };

  struct Plan {
    Graph contractedDFG;
    llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>>
        contractedCandidates;
    std::vector<Unit> units;
    std::vector<IdIndex> originalNodeToContractedNode;
    std::vector<IdIndex> originalPortToContractedPort;
    std::vector<IdIndex> originalEdgeToContractedEdge;
    std::vector<TechMappedEdgeKind> originalEdgeKinds;
    double coverageScore = 1.0;
    std::string diagnostics;
  };

  bool buildPlan(const Graph &dfg, mlir::ModuleOp adgModule, const Graph &adg,
                 Plan &plan);

  bool expandPlanMapping(const Graph &originalDfg, const Graph &adg,
                         const Plan &plan, const MappingState &contractedState,
                         MappingState &expandedState,
                         llvm::SmallVectorImpl<FUConfigSelection> &fuConfigs);
};

} // namespace fcc

#endif // FCC_MAPPER_TECHMAPPER_H
