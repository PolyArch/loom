#include "Fabric/Tech/SubgraphMatcher.h"

#include "Fabric/IR/ConfiguredFunction.h"
#include "Fabric/Tech/ConfiguredFunctionAdapters.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <utility>

namespace fabric {

FuMatchResult mapPatternToFu(::dataflow::SubgraphOp pattern, FuOp fu) {
  FuMatchResult result;
  ConfiguredFunction softwareFunction;
  std::string error;
  if (::mlir::failed(
          configuredFunctionFromSubgraph(pattern, softwareFunction, error)))
    return result;

  ::llvm::SmallVector<ConfiguredFunction, 8> configuredFunctions;
  if (::mlir::failed(
          projectConfiguredFunctions(fu, configuredFunctions, error)))
    return result;

  for (auto [encodingIndex, candidate] :
       ::llvm::enumerate(configuredFunctions)) {
    ConfiguredFunctionMatch match;
    if (!matchConfiguredFunctions(softwareFunction, candidate,
                                  /*preserveFuBoundaryIdentity=*/false, &match))
      continue;

    result.matched = true;
    result.fu = fu;
    result.encodingIndex = encodingIndex;
    for (unsigned node : match.nodeMap)
      result.actorToFabricOp.push_back(candidate.nodes[node].fabricResource);
    result.inputPorts = std::move(match.inputPorts);
    result.outputPorts = std::move(match.outputPorts);
    return result;
  }
  return result;
}

} // namespace fabric
