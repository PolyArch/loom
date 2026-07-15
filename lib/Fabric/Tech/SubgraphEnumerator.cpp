#include "Fabric/Tech/SubgraphEnumerator.h"

#include "Fabric/IR/ConfiguredFunction.h"
#include "Fabric/Tech/ConfiguredFunctionAdapters.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace fabric {

::llvm::SmallVector<FuSubgraphCandidate>
enumerateFuSubgraphs(FuOp fu, ::mlir::ModuleOp module,
                     ::llvm::StringRef baseName,
                     ::llvm::StringRef *unsupported) {
  ::llvm::SmallVector<FuSubgraphCandidate> candidates;
  auto encodings = fu->getAttrOfType<::mlir::ArrayAttr>("valid_encodings");
  if (!encodings) {
    if (unsupported)
      *unsupported = "fabric.fu.valid_encodings";
    return candidates;
  }

  ::llvm::SmallVector<ConfiguredFunction, 8> functions;
  std::string error;
  if (::mlir::failed(projectConfiguredFunctions(fu, functions, error))) {
    fu.emitError(error);
    return candidates;
  }

  for (auto [index, function] : ::llvm::enumerate(functions)) {
    std::string symbol = (baseName + "_" + std::to_string(index)).str();
    MaterializedSubgraph materialized;
    std::string materializeError;
    if (::mlir::failed(materializeConfiguredFunction(
            function, module, symbol, materialized, materializeError))) {
      fu.emitError(materializeError);
      for (FuSubgraphCandidate &candidate : candidates)
        candidate.wrapper.erase();
      candidates.clear();
      return candidates;
    }

    FuSubgraphCandidate candidate;
    candidate.wrapper = materialized.wrapper;
    candidate.subgraph = materialized.subgraph;
    candidate.encodingIndex = index;
    for (const ConfiguredBoundaryInput &input : function.inputs)
      candidate.inputPortIndices.push_back(input.fuPort);
    for (const ConfiguredBoundaryOutput &output : function.outputs)
      candidate.outputPortIndices.push_back(output.fuPort);
    candidates.push_back(std::move(candidate));
  }
  return candidates;
}

} // namespace fabric
