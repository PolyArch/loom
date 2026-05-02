#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_MCSGRAPH_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_MCSGRAPH_H

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <utility>

namespace loom::fabric::tech {

enum class McsValueKind : uint8_t {
  BlockArgument,
  NodeResult,
};

struct McsValueRef {
  McsValueKind kind = McsValueKind::BlockArgument;
  unsigned argIndex = 0;
  unsigned nodeIndex = 0;
  unsigned resultIndex = 0;

  static McsValueRef blockArgument(unsigned argIndex);
  static McsValueRef nodeResult(unsigned nodeIndex, unsigned resultIndex);

  bool operator==(const McsValueRef &other) const;
  bool operator!=(const McsValueRef &other) const { return !(*this == other); }
  bool operator<(const McsValueRef &other) const;
};

struct McsOperand {
  McsValueRef source;
  std::string sourceLabel;
  bool isBackEdge = false;
  ::mlir::Type type;
  std::string typeKey;
  unsigned width = 0;
};

struct McsNode {
  ::mlir::Operation *op = nullptr;
  unsigned index = 0;
  ::llvm::StringRef opName;
  bool commutative = false;

  ::llvm::SmallVector<McsOperand, 4> operands;
  ::llvm::SmallVector<::mlir::Type, 2> resultTypes;
  ::llvm::SmallVector<std::string, 2> resultTypeKeys;
  ::llvm::SmallVector<unsigned, 2> resultWidths;
  ::llvm::SmallVector<std::pair<std::string, std::string>, 4> attrKeys;
};

struct McsGraph {
  ::dataflow::SubgraphOp subgraph;
  ::mlir::Block *body = nullptr;
  unsigned inputIndex = 0;

  ::llvm::SmallVector<::mlir::Type, 4> blockArgTypes;
  ::llvm::SmallVector<std::string, 4> blockArgTypeKeys;
  ::llvm::SmallVector<McsNode, 8> nodes;
  ::llvm::SmallVector<McsValueRef, 4> yieldSources;
};

struct McsGraphBuildResult {
  ::llvm::SmallVector<McsGraph, 4> graphs;
  ::llvm::SmallVector<std::string, 4> notes;

  bool success() const { return notes.empty(); }
};

McsGraphBuildResult
buildMcsGraphs(::llvm::ArrayRef<::dataflow::SubgraphOp> subgraphs);

unsigned bitWidthOfMcsType(::mlir::Type type);

std::string labelForMcsValue(McsValueRef value);

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_MCSGRAPH_H
