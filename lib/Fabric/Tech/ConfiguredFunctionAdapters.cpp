#include "Fabric/Tech/ConfiguredFunctionAdapters.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>
#include <utility>

namespace fabric {
namespace {

using ::mlir::Block;
using ::mlir::DictionaryAttr;
using ::mlir::FunctionType;
using ::mlir::NamedAttribute;
using ::mlir::Operation;
using ::mlir::Value;

static DictionaryAttr semanticAttributes(Operation *op) {
  ::llvm::SmallVector<NamedAttribute, 4> attributes;
  for (NamedAttribute named : op->getAttrs()) {
    if (named.getName().getValue().starts_with("loom."))
      continue;
    attributes.push_back(named);
  }
  return DictionaryAttr::get(op->getContext(), attributes);
}

static std::optional<ConfiguredValue>
sourceFor(Value value, Block &body,
          const ::llvm::DenseMap<Operation *, unsigned> &nodeByOp) {
  if (auto argument = ::mlir::dyn_cast<::mlir::BlockArgument>(value)) {
    if (argument.getOwner() != &body)
      return std::nullopt;
    return ConfiguredValue::input(argument.getArgNumber());
  }
  auto result = ::mlir::dyn_cast<::mlir::OpResult>(value);
  if (!result)
    return std::nullopt;
  auto node = nodeByOp.find(result.getOwner());
  if (node == nodeByOp.end())
    return std::nullopt;
  return ConfiguredValue::nodeResult(node->second, result.getResultNumber());
}

} // namespace

::mlir::LogicalResult configuredFunctionFromFunc(::mlir::func::FuncOp source,
                                                 ConfiguredFunction &function,
                                                 std::string &error) {
  function = {};
  if (!source || source.isExternal() || !source.getBody().hasOneBlock()) {
    error = "configured function must have one body block";
    return ::mlir::failure();
  }
  Block &body = source.getBody().front();
  auto returnOp =
      ::mlir::dyn_cast<::mlir::func::ReturnOp>(body.getTerminator());
  if (!returnOp) {
    error = "configured function body has no func.return terminator";
    return ::mlir::failure();
  }

  for (auto [port, argument] : ::llvm::enumerate(body.getArguments()))
    function.inputs.push_back(
        {static_cast<unsigned>(port), argument.getType()});

  ::llvm::DenseMap<Operation *, unsigned> nodeByOp;
  for (Operation &op : body.without_terminator()) {
    if (op.getNumRegions() != 0 || op.getNumSuccessors() != 0) {
      error = "configured function nodes must not contain regions or "
              "successors";
      return ::mlir::failure();
    }
    unsigned node = function.nodes.size();
    nodeByOp[&op] = node;
    ConfiguredFunctionNode configured;
    configured.fabricResource = node;
    configured.operationName = op.getName().getStringRef().str();
    configured.functionType = FunctionType::get(
        op.getContext(), op.getOperandTypes(), op.getResultTypes());
    configured.attributes = semanticAttributes(&op);
    function.nodes.push_back(std::move(configured));
  }

  for (Operation &op : body.without_terminator()) {
    ConfiguredFunctionNode &node = function.nodes[nodeByOp.lookup(&op)];
    for (Value operand : op.getOperands()) {
      auto source = sourceFor(operand, body, nodeByOp);
      if (!source) {
        error = "configured function operand is defined outside the function";
        return ::mlir::failure();
      }
      node.operands.push_back(*source);
    }
  }

  for (auto [port, value] : ::llvm::enumerate(returnOp.getOperands())) {
    auto source = sourceFor(value, body, nodeByOp);
    if (!source) {
      error = "configured function result is defined outside the function";
      return ::mlir::failure();
    }
    function.outputs.push_back(
        {static_cast<unsigned>(port), value.getType(), *source});
  }
  return ::mlir::success();
}

} // namespace fabric
