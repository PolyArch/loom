#include "Fabric/Tech/Synthesizer/McsGraph.h"

#include "Common/IndexWidth.h"
#include "Fabric/Tech/SubgraphGraphView.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <tuple>

namespace loom::fabric::tech {

namespace gv = ::loom::fabric::tech::detail;

McsValueRef McsValueRef::blockArgument(unsigned argIndex) {
  McsValueRef ref;
  ref.kind = McsValueKind::BlockArgument;
  ref.argIndex = argIndex;
  return ref;
}

McsValueRef McsValueRef::nodeResult(unsigned nodeIndex, unsigned resultIndex) {
  McsValueRef ref;
  ref.kind = McsValueKind::NodeResult;
  ref.nodeIndex = nodeIndex;
  ref.resultIndex = resultIndex;
  return ref;
}

bool McsValueRef::operator==(const McsValueRef &other) const {
  if (kind != other.kind)
    return false;
  if (kind == McsValueKind::BlockArgument)
    return argIndex == other.argIndex;
  return nodeIndex == other.nodeIndex && resultIndex == other.resultIndex;
}

bool McsValueRef::operator<(const McsValueRef &other) const {
  return std::tie(kind, argIndex, nodeIndex, resultIndex) <
         std::tie(other.kind, other.argIndex, other.nodeIndex,
                  other.resultIndex);
}

unsigned bitWidthOfMcsType(::mlir::Type type) {
  if (auto intType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
    return intType.getWidth();
  if (auto floatType = ::llvm::dyn_cast<::mlir::FloatType>(type))
    return floatType.getWidth();
  if (::llvm::isa<::mlir::IndexType>(type))
    return ::loom::getIndexWidth();
  if (::llvm::isa<::mlir::NoneType>(type))
    return 0;
  return 0;
}

std::string labelForMcsValue(McsValueRef value) {
  std::string out;
  ::llvm::raw_string_ostream os(out);
  if (value.kind == McsValueKind::BlockArgument) {
    os << "arg:" << value.argIndex;
  } else {
    os << "node:" << value.nodeIndex << ":result:" << value.resultIndex;
  }
  return out;
}

namespace {

McsValueRef convertSource(gv::Source source) {
  if (source.kind == gv::Source::BlockArg)
    return McsValueRef::blockArgument(source.idx);
  return McsValueRef::nodeResult(source.idx, source.resultNum);
}

bool isBackEdgeOperand(gv::Source source, unsigned consumerIndex) {
  return source.kind == gv::Source::BodyOp && source.idx >= consumerIndex;
}

} // namespace

McsGraphBuildResult
buildMcsGraphs(::llvm::ArrayRef<::dataflow::SubgraphOp> subgraphs) {
  McsGraphBuildResult result;
  result.graphs.reserve(subgraphs.size());

  for (auto indexed : ::llvm::enumerate(subgraphs)) {
    unsigned inputIndex = static_cast<unsigned>(indexed.index());
    ::dataflow::SubgraphOp subgraph = indexed.value();
    if (!subgraph) {
      std::string note;
      ::llvm::raw_string_ostream os(note);
      os << "mcs-graph: input " << inputIndex << " is null";
      result.notes.push_back(std::move(note));
      continue;
    }

    gv::GraphView view;
    if (!gv::buildGraphView(subgraph, view)) {
      std::string note;
      ::llvm::raw_string_ostream os(note);
      os << "mcs-graph: failed to build graph view for input " << inputIndex;
      result.notes.push_back(std::move(note));
      continue;
    }

    McsGraph graph;
    graph.subgraph = subgraph;
    graph.body = view.body;
    graph.inputIndex = inputIndex;
    graph.blockArgTypeKeys = std::move(view.blockArgTypeKeys);
    graph.blockArgTypes.reserve(view.numBlockArgs);
    for (unsigned i = 0; i < view.numBlockArgs; ++i)
      graph.blockArgTypes.push_back(view.body->getArgument(i).getType());

    graph.nodes.reserve(view.nodes.size());
    for (const gv::NodeInfo &info : view.nodes) {
      McsNode node;
      node.op = info.op;
      node.index = static_cast<unsigned>(graph.nodes.size());
      node.opName = info.opName;
      node.commutative = gv::isCommutativeOp(info.opName);
      node.resultTypeKeys = info.resultTypeKeys;
      node.attrKeys = info.attrKeys;
      node.operands.reserve(info.operands.size());
      for (auto operandIndexed : ::llvm::enumerate(info.operands)) {
        unsigned operandIndex = static_cast<unsigned>(operandIndexed.index());
        gv::Source source = operandIndexed.value();
        McsOperand operand;
        operand.source = convertSource(source);
        operand.sourceLabel = labelForMcsValue(operand.source);
        operand.isBackEdge = isBackEdgeOperand(source, node.index);
        operand.type = info.op->getOperand(operandIndex).getType();
        operand.typeKey = gv::typeKey(operand.type);
        operand.width = bitWidthOfMcsType(operand.type);
        node.operands.push_back(std::move(operand));
      }
      node.resultTypes.reserve(info.op->getNumResults());
      node.resultWidths.reserve(info.op->getNumResults());
      for (::mlir::Type type : info.op->getResultTypes()) {
        node.resultTypes.push_back(type);
        node.resultWidths.push_back(bitWidthOfMcsType(type));
      }
      graph.nodes.push_back(std::move(node));
    }

    graph.yieldSources.reserve(view.yieldSources.size());
    for (gv::Source source : view.yieldSources)
      graph.yieldSources.push_back(convertSource(source));

    result.graphs.push_back(std::move(graph));
  }

  return result;
}

} // namespace loom::fabric::tech
