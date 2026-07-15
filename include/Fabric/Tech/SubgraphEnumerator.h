#ifndef FABRIC_TECH_SUBGRAPHENUMERATOR_H
#define FABRIC_TECH_SUBGRAPHENUMERATOR_H

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace fabric {

struct FuSubgraphCandidate {
  ::mlir::func::FuncOp wrapper;
  ::dataflow::SubgraphOp subgraph;
  unsigned encodingIndex = 0;
  ::llvm::SmallVector<unsigned, 4> inputPortIndices;
  ::llvm::SmallVector<unsigned, 4> outputPortIndices;
};

// Legacy adapter for passes that still consume dataflow.subgraph. The
// semantic source is the FU's explicit valid_encodings attribute and the
// Fabric-owned ConfiguredFunction projector. No configuration discovery or
// deduplication occurs here.
::llvm::SmallVector<FuSubgraphCandidate>
enumerateFuSubgraphs(FuOp fu, ::mlir::ModuleOp module,
                     ::llvm::StringRef baseName,
                     ::llvm::StringRef *unsupported = nullptr);

} // namespace fabric

#endif // FABRIC_TECH_SUBGRAPHENUMERATOR_H
