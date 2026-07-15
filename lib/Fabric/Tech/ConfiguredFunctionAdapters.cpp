#include "Fabric/Tech/ConfiguredFunctionAdapters.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <utility>

namespace fabric {
namespace {

using ::mlir::Block;
using ::mlir::DictionaryAttr;
using ::mlir::FunctionType;
using ::mlir::Location;
using ::mlir::NamedAttribute;
using ::mlir::OpBuilder;
using ::mlir::Operation;
using ::mlir::OperationState;
using ::mlir::Type;
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

::mlir::LogicalResult
configuredFunctionFromSubgraph(::dataflow::SubgraphOp subgraph,
                               ConfiguredFunction &function,
                               std::string &error) {
  function = {};
  if (!subgraph || subgraph.getBody().empty()) {
    error = "missing dataflow.subgraph body";
    return ::mlir::failure();
  }
  Block &body = subgraph.getBody().front();
  auto yield = ::mlir::dyn_cast<::dataflow::YieldOp>(body.getTerminator());
  if (!yield) {
    error = "dataflow.subgraph body has no dataflow.yield terminator";
    return ::mlir::failure();
  }

  for (auto [port, argument] : ::llvm::enumerate(body.getArguments()))
    function.inputs.push_back(
        {static_cast<unsigned>(port), argument.getType()});

  ::llvm::DenseMap<Operation *, unsigned> nodeByOp;
  for (Operation &op : body.without_terminator()) {
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
        error = "dataflow.subgraph operand is defined outside the function";
        return ::mlir::failure();
      }
      node.operands.push_back(*source);
    }
  }

  for (auto [port, value] : ::llvm::enumerate(yield.getValues())) {
    auto source = sourceFor(value, body, nodeByOp);
    if (!source) {
      error = "dataflow.subgraph yield value is defined outside the function";
      return ::mlir::failure();
    }
    function.outputs.push_back(
        {static_cast<unsigned>(port), value.getType(), *source});
  }
  return ::mlir::success();
}

::mlir::LogicalResult materializeConfiguredFunction(
    const ConfiguredFunction &function, ::mlir::ModuleOp module,
    ::llvm::StringRef symbolName, MaterializedSubgraph &materialized,
    std::string &error) {
  if (!module) {
    error = "missing destination module";
    return ::mlir::failure();
  }
  auto *context = module.getContext();
  Location loc = module.getLoc();

  ::llvm::SmallVector<Type, 4> inputTypes;
  for (const ConfiguredBoundaryInput &input : function.inputs)
    inputTypes.push_back(input.type);
  ::llvm::SmallVector<Type, 4> outputTypes;
  for (const ConfiguredBoundaryOutput &output : function.outputs)
    outputTypes.push_back(output.type);

  OpBuilder moduleBuilder(module.getBody(), module.getBody()->end());
  auto wrapper = ::mlir::func::FuncOp::create(
      moduleBuilder, loc, symbolName,
      FunctionType::get(context, inputTypes, outputTypes));
  wrapper.setPrivate();
  Block *wrapperBody = wrapper.addEntryBlock();
  OpBuilder wrapperBuilder(wrapperBody, wrapperBody->end());

  OperationState subgraphState(loc, ::dataflow::SubgraphOp::getOperationName());
  subgraphState.addOperands(wrapperBody->getArguments());
  subgraphState.addTypes(outputTypes);
  ::mlir::Region *region = subgraphState.addRegion();
  Block *body = new Block();
  region->push_back(body);
  ::llvm::SmallVector<Location, 4> argumentLocations(inputTypes.size(), loc);
  body->addArguments(inputTypes, argumentLocations);
  auto subgraph = ::mlir::cast<::dataflow::SubgraphOp>(
      wrapperBuilder.create(subgraphState));
  OpBuilder bodyBuilder(body, body->end());

  ::llvm::DenseMap<unsigned, Value> inputByPort;
  for (auto [position, input] : ::llvm::enumerate(function.inputs))
    inputByPort[input.fuPort] = body->getArgument(position);

  using ResultKey = std::pair<unsigned, unsigned>;
  ::llvm::DenseMap<ResultKey, Value> values;
  ::llvm::DenseMap<ResultKey, Value> placeholders;
  ::llvm::SmallVector<Operation *, 8> placeholderOps;

  auto valueFor = [&](const ConfiguredValue &source, Type expected) -> Value {
    if (source.kind == ConfiguredValue::Kind::InputPort)
      return inputByPort.lookup(source.index);
    ResultKey key{source.index, source.result};
    if (Value value = values.lookup(key))
      return value;
    if (Value placeholder = placeholders.lookup(key))
      return placeholder;
    OperationState state(
        loc, ::mlir::UnrealizedConversionCastOp::getOperationName());
    state.addTypes(expected);
    Operation *placeholder = bodyBuilder.create(state);
    placeholderOps.push_back(placeholder);
    placeholders[key] = placeholder->getResult(0);
    return placeholder->getResult(0);
  };

  for (auto [nodeIndex, node] : ::llvm::enumerate(function.nodes)) {
    ::llvm::SmallVector<Value, 4> operands;
    for (auto [operandIndex, source] : ::llvm::enumerate(node.operands)) {
      Value operand =
          valueFor(source, node.functionType.getInput(operandIndex));
      if (!operand) {
        wrapper.erase();
        error = "configured function references an unknown input port";
        return ::mlir::failure();
      }
      operands.push_back(operand);
    }
    OperationState state(loc, node.operationName);
    state.addOperands(operands);
    state.addTypes(node.functionType.getResults());
    state.addAttributes(node.attributes.getValue());
    Operation *operation = bodyBuilder.create(state);
    for (auto [resultIndex, result] :
         ::llvm::enumerate(operation->getResults()))
      values[{static_cast<unsigned>(nodeIndex),
              static_cast<unsigned>(resultIndex)}] = result;
  }

  for (auto &entry : placeholders) {
    Value real = values.lookup(entry.first);
    if (!real) {
      wrapper.erase();
      error = "configured function contains an unresolved result reference";
      return ::mlir::failure();
    }
    entry.second.replaceAllUsesWith(real);
  }
  for (Operation *placeholder : placeholderOps)
    placeholder->erase();

  ::llvm::SmallVector<Value, 4> yields;
  for (const ConfiguredBoundaryOutput &output : function.outputs) {
    Value value = valueFor(output.value, output.type);
    if (!value) {
      wrapper.erase();
      error = "configured function output cannot be resolved";
      return ::mlir::failure();
    }
    yields.push_back(value);
  }
  ::dataflow::YieldOp::create(bodyBuilder, loc, yields);
  ::mlir::func::ReturnOp::create(wrapperBuilder, loc, subgraph.getResults());

  materialized.wrapper = wrapper;
  materialized.subgraph = subgraph;
  return ::mlir::success();
}

} // namespace fabric
