#include "Fabric/IR/ConfiguredFunction.h"

#include "Common/IndexWidth.h"
#include "Fabric/IR/FabricTypes.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

namespace fabric {
namespace {

using ::mlir::ArrayAttr;
using ::mlir::Attribute;
using ::mlir::Block;
using ::mlir::DictionaryAttr;
using ::mlir::FlatSymbolRefAttr;
using ::mlir::FunctionType;
using ::mlir::IntegerAttr;
using ::mlir::Operation;
using ::mlir::StringRef;
using ::mlir::Type;
using ::mlir::TypeAttr;
using ::mlir::Value;

struct OpMode {
  unsigned resource = 0;
  unsigned modeIndex = 0;
  std::string operationName;
  FunctionType functionType;
  DictionaryAttr attributes;
  ::llvm::SmallVector<unsigned, 4> inputPorts;
  ::llvm::SmallVector<unsigned, 2> outputPorts;
};

struct RouteMode {
  unsigned resource = 0;
  unsigned select = 0;
};

struct ParsedEncoding {
  ::llvm::SmallVector<unsigned, 4> outputPorts;
  ::llvm::DenseMap<unsigned, OpMode> opModes;
  ::llvm::DenseMap<unsigned, RouteMode> routes;
  ::llvm::DenseSet<unsigned> mentionedResources;
};

static std::string printType(Type type) {
  std::string text;
  ::llvm::raw_string_ostream os(text);
  type.print(os);
  return text;
}

static std::string printAttribute(Attribute attr) {
  std::string text;
  ::llvm::raw_string_ostream os(text);
  attr.print(os);
  return text;
}

static bool sameType(Type lhs, Type rhs) {
  return printType(lhs) == printType(rhs);
}

static bool sameAttributes(DictionaryAttr lhs, DictionaryAttr rhs) {
  return printAttribute(lhs) == printAttribute(rhs);
}

static std::optional<unsigned> semanticBitWidth(Type type) {
  if (auto integer = ::mlir::dyn_cast<::mlir::IntegerType>(type))
    return integer.getWidth();
  if (auto floating = ::mlir::dyn_cast<::mlir::FloatType>(type))
    return floating.getWidth();
  if (::mlir::isa<::mlir::IndexType>(type))
    return ::loom::getIndexWidth();
  if (::mlir::isa<::mlir::NoneType>(type))
    return 0u;
  if (auto vector = ::mlir::dyn_cast<::mlir::VectorType>(type)) {
    if (vector.isScalable())
      return std::nullopt;
    auto elementWidth = semanticBitWidth(vector.getElementType());
    if (!elementWidth)
      return std::nullopt;
    return static_cast<unsigned>(vector.getNumElements()) * *elementWidth;
  }
  return std::nullopt;
}

static bool verifyPayloadCapacity(Type physicalType, Type softwareType,
                                  StringRef subject, std::string &error) {
  auto physical = ::mlir::dyn_cast<BitsType>(physicalType);
  if (!physical) {
    error =
        subject.str() + " requires an untagged fabric.bits physical payload";
    return false;
  }
  auto required = semanticBitWidth(softwareType);
  if (!required) {
    error = subject.str() + " has an unsupported software type";
    return false;
  }
  if (*required > physical.getWidth()) {
    error = subject.str() + " type width exceeds the physical payload width";
    return false;
  }
  return true;
}

static Type fuInputBoundaryType(FuOp fu, unsigned port) {
  if (!fu.getSymNameAttr())
    return fu.getInputs()[port].getType();
  return ::mlir::cast<FunctionType>(fu.getFunctionTypeAttr().getValue())
      .getInput(port);
}

static Type fuOutputBoundaryType(FuOp fu, unsigned port) {
  if (!fu.getSymNameAttr())
    return fu.getOutputs()[port].getType();
  return ::mlir::cast<FunctionType>(fu.getFunctionTypeAttr().getValue())
      .getResult(port);
}

static bool hasOnlyKeys(DictionaryAttr dict,
                        ::llvm::ArrayRef<StringRef> allowed,
                        std::string &error) {
  for (::mlir::NamedAttribute named : dict) {
    if (::llvm::is_contained(allowed, named.getName().getValue()))
      continue;
    error = "unexpected key '" + named.getName().getValue().str() + "'";
    return false;
  }
  return true;
}

static std::optional<unsigned> readUnsigned(Attribute attr) {
  auto integer = ::mlir::dyn_cast_or_null<IntegerAttr>(attr);
  if (!integer || integer.getValue().isNegative() ||
      integer.getValue().getActiveBits() > 32)
    return std::nullopt;
  return static_cast<unsigned>(integer.getInt());
}

static bool readIndexArray(ArrayAttr array,
                           ::llvm::SmallVectorImpl<unsigned> &values,
                           std::string &error) {
  if (!array) {
    error = "expected an integer array";
    return false;
  }
  ::llvm::DenseSet<unsigned> seen;
  for (Attribute attr : array) {
    auto value = readUnsigned(attr);
    if (!value) {
      error = "expected a non-negative 32-bit integer";
      return false;
    }
    if (!seen.insert(*value).second) {
      error = "integer array contains a duplicate entry";
      return false;
    }
    values.push_back(*value);
  }
  return true;
}

static bool selectedOperationBelongsTo(::fabric::OpOp op, StringRef selected) {
  for (Attribute attr : op.getOpList()) {
    auto symbol = ::mlir::dyn_cast<FlatSymbolRefAttr>(attr);
    if (symbol && symbol.getValue() == selected)
      return true;
  }
  return false;
}

static bool verifyModePorts(::fabric::OpOp op, const OpMode &mode,
                            std::string &error) {
  if (mode.functionType.getNumInputs() != mode.inputPorts.size() ||
      mode.functionType.getNumResults() != mode.outputPorts.size()) {
    error = "function_type arity does not match the software port maps";
    return false;
  }

  for (auto [semanticPort, physicalPort] : ::llvm::enumerate(mode.inputPorts)) {
    if (physicalPort >= op.getInputs().size()) {
      error = "input_ports contains an out-of-range physical port";
      return false;
    }
    if (!verifyPayloadCapacity(op.getInputs()[physicalPort].getType(),
                               mode.functionType.getInput(semanticPort),
                               "software input", error))
      return false;
  }
  for (auto [semanticPort, physicalPort] :
       ::llvm::enumerate(mode.outputPorts)) {
    if (physicalPort >= op.getOutputs().size()) {
      error = "output_ports contains an out-of-range physical port";
      return false;
    }
    if (!verifyPayloadCapacity(op.getOutputs()[physicalPort].getType(),
                               mode.functionType.getResult(semanticPort),
                               "software result", error))
      return false;
  }
  return true;
}

static bool parseHardwareMode(::fabric::OpOp op, unsigned resource,
                              unsigned modeIndex, OpMode &mode,
                              std::string &error) {
  auto modes = op->getAttrOfType<ArrayAttr>("hw_params");
  if (!modes || modeIndex >= modes.size()) {
    error = "mode index is missing or out of range for fabric.op hw_params";
    return false;
  }
  auto definition = ::mlir::dyn_cast<DictionaryAttr>(modes[modeIndex]);
  if (!definition) {
    error = "hw_params entries must be dictionaries";
    return false;
  }
  if (!hasOnlyKeys(
          definition,
          {"op", "function_type", "input_ports", "output_ports", "attributes"},
          error))
    return false;

  auto selected = definition.getAs<FlatSymbolRefAttr>("op");
  auto functionTypeAttr = definition.getAs<TypeAttr>("function_type");
  auto attributes = definition.getAs<DictionaryAttr>("attributes");
  if (!selected || !functionTypeAttr || !attributes) {
    error = "hw_params mode requires op, function_type, and attributes";
    return false;
  }
  auto functionType =
      ::mlir::dyn_cast<FunctionType>(functionTypeAttr.getValue());
  if (!functionType) {
    error = "function_type must be an MLIR function type";
    return false;
  }
  if (!selectedOperationBelongsTo(op, selected.getValue())) {
    error = "hw_params mode operation is not in fabric.op op_list";
    return false;
  }

  mode.resource = resource;
  mode.modeIndex = modeIndex;
  mode.operationName = selected.getValue().str();
  mode.functionType = functionType;
  mode.attributes = attributes;
  if (!readIndexArray(definition.getAs<ArrayAttr>("input_ports"),
                      mode.inputPorts, error) ||
      !readIndexArray(definition.getAs<ArrayAttr>("output_ports"),
                      mode.outputPorts, error))
    return false;
  return verifyModePorts(op, mode, error);
}

static bool sameHardwareMode(const OpMode &lhs, const OpMode &rhs) {
  return lhs.operationName == rhs.operationName &&
         sameType(lhs.functionType, rhs.functionType) &&
         sameAttributes(lhs.attributes, rhs.attributes) &&
         lhs.inputPorts == rhs.inputPorts && lhs.outputPorts == rhs.outputPorts;
}

static bool verifySoftwareOperationMode(const OpMode &mode,
                                        ::mlir::MLIRContext *context,
                                        std::string &error) {
  StringRef dialectNamespace = StringRef(mode.operationName).split('.').first;
  if (!context->getOrLoadDialect(dialectNamespace)) {
    error = "cannot load the dialect for @" + mode.operationName;
    return false;
  }
  if (!::mlir::RegisteredOperationName::lookup(mode.operationName, context)) {
    error = "operation @" + mode.operationName +
            " is not a registered MLIR operation and cannot be materialized";
    return false;
  }
  ::mlir::Location loc = ::mlir::UnknownLoc::get(context);
  Block block;
  ::llvm::SmallVector<::mlir::Location, 4> argumentLocations(
      mode.functionType.getNumInputs(), loc);
  block.addArguments(mode.functionType.getInputs(), argumentLocations);

  ::mlir::OperationState state(loc, mode.operationName);
  state.addOperands(block.getArguments());
  state.addTypes(mode.functionType.getResults());
  state.addAttributes(mode.attributes.getValue());
  Operation *operation = Operation::create(state);
  block.push_back(operation);

  ::llvm::SmallVector<std::string, 2> diagnostics;
  ::mlir::ScopedDiagnosticHandler capture(
      context, [&](::mlir::Diagnostic &diagnostic) {
        diagnostics.push_back(diagnostic.str());
        return ::mlir::success();
      });
  if (::mlir::succeeded(::mlir::verify(operation)))
    return true;

  error = "does not form a valid @" + mode.operationName + " operation";
  if (!diagnostics.empty())
    error += ": " + diagnostics.front();
  return false;
}

static bool parseEncoding(FuOp fu, DictionaryAttr encoding,
                          ::llvm::ArrayRef<Operation *> bodyOps,
                          ParsedEncoding &parsed, std::string &error) {
  if (!hasOnlyKeys(encoding, {"outputs", "resources"}, error))
    return false;

  if (!readIndexArray(encoding.getAs<ArrayAttr>("outputs"), parsed.outputPorts,
                      error)) {
    error = "outputs: " + error;
    return false;
  }
  auto yield =
      ::mlir::cast<::fabric::YieldOp>(fu.getBody().front().getTerminator());
  for (unsigned output : parsed.outputPorts) {
    if (output >= yield.getValues().size()) {
      error = "outputs contains an out-of-range FU output port";
      return false;
    }
  }

  ArrayAttr resources = encoding.getAs<ArrayAttr>("resources");
  if (!resources) {
    error = "resources must be an array";
    return false;
  }

  unsigned previous = 0;
  bool havePrevious = false;
  for (Attribute attr : resources) {
    auto resource = ::mlir::dyn_cast<DictionaryAttr>(attr);
    if (!resource) {
      error = "resources entries must be dictionaries";
      return false;
    }
    auto resourceIndex = readUnsigned(resource.get("resource"));
    if (!resourceIndex || *resourceIndex >= bodyOps.size()) {
      error = "resource index is missing or out of range";
      return false;
    }
    if (havePrevious && *resourceIndex <= previous) {
      error = "resources must be ordered by strictly increasing index";
      return false;
    }
    previous = *resourceIndex;
    havePrevious = true;
    if (!parsed.mentionedResources.insert(*resourceIndex).second) {
      error = "resource is configured more than once";
      return false;
    }

    Operation *bodyOp = bodyOps[*resourceIndex];
    if (auto op = ::mlir::dyn_cast<::fabric::OpOp>(bodyOp)) {
      if (!hasOnlyKeys(resource, {"resource", "mode"}, error))
        return false;
      auto modeIndex = readUnsigned(resource.get("mode"));
      if (!modeIndex) {
        error = "fabric.op resource requires a mode index";
        return false;
      }
      OpMode mode;
      if (!parseHardwareMode(op, *resourceIndex, *modeIndex, mode, error))
        return false;
      parsed.opModes.try_emplace(*resourceIndex, std::move(mode));
      continue;
    }

    if (::mlir::isa<::fabric::MuxOp, ::fabric::DemuxOp>(bodyOp)) {
      if (!hasOnlyKeys(resource, {"resource", "select"}, error))
        return false;
      auto select = readUnsigned(resource.get("select"));
      if (!select) {
        error = "routing resource requires a non-negative select value";
        return false;
      }
      unsigned bound = 0;
      if (auto mux = ::mlir::dyn_cast<::fabric::MuxOp>(bodyOp))
        bound = mux.getInputs().size();
      else
        bound = ::mlir::cast<::fabric::DemuxOp>(bodyOp).getOutputs().size();
      if (*select >= bound) {
        error = "routing select is out of range";
        return false;
      }
      parsed.routes.try_emplace(*resourceIndex,
                                RouteMode{*resourceIndex, *select});
      continue;
    }

    error = "resource does not identify fabric.op, fabric.mux, or fabric.demux";
    return false;
  }
  return true;
}

struct Projector {
  Projector(FuOp fu, ::llvm::ArrayRef<Operation *> bodyOps,
            const ParsedEncoding &encoding, std::string &error)
      : fu(fu), bodyOps(bodyOps), encoding(encoding), error(error) {}

  FuOp fu;
  ::llvm::ArrayRef<Operation *> bodyOps;
  const ParsedEncoding &encoding;
  std::string &error;
  ::llvm::DenseMap<Operation *, unsigned> resourceByOp;
  ::llvm::DenseSet<unsigned> activeOps;
  ::llvm::DenseSet<unsigned> activeRoutes;
  ::llvm::DenseSet<Value> aliveValues;
  ::llvm::DenseMap<unsigned, Type> inputTypes;

  bool recordInputType(unsigned port, Type type) {
    auto found = inputTypes.find(port);
    if (found == inputTypes.end()) {
      inputTypes[port] = type;
      return true;
    }
    if (sameType(found->second, type))
      return true;
    error = "one FU input port is assigned incompatible software types";
    return false;
  }

  std::optional<Value> traceSelectedRoute(Value value, Type expected = {}) {
    ::llvm::DenseSet<Value> routedValues;
    while (true) {
      aliveValues.insert(value);
      if (expected &&
          !verifyPayloadCapacity(value.getType(), expected,
                                 "selected physical path segment", error))
        return std::nullopt;
      auto result = ::mlir::dyn_cast<::mlir::OpResult>(value);
      if (!result)
        return value;
      Operation *producer = result.getOwner();
      auto resourceIt = resourceByOp.find(producer);

      if (auto mux = ::mlir::dyn_cast<::fabric::MuxOp>(producer)) {
        if (!routedValues.insert(value).second) {
          error = "selected routing topology contains a cycle";
          return std::nullopt;
        }
        if (resourceIt == resourceByOp.end()) {
          error = "configured routing value is produced outside the FU body";
          return std::nullopt;
        }
        auto route = encoding.routes.find(resourceIt->second);
        if (route == encoding.routes.end()) {
          error = "active fabric.mux has no route in the encoding";
          return std::nullopt;
        }
        activeRoutes.insert(resourceIt->second);
        value = mux.getInputs()[route->second.select];
        continue;
      }

      if (auto demux = ::mlir::dyn_cast<::fabric::DemuxOp>(producer)) {
        if (!routedValues.insert(value).second) {
          error = "selected routing topology contains a cycle";
          return std::nullopt;
        }
        if (resourceIt == resourceByOp.end()) {
          error = "configured routing value is produced outside the FU body";
          return std::nullopt;
        }
        auto route = encoding.routes.find(resourceIt->second);
        if (route == encoding.routes.end()) {
          error = "active fabric.demux has no route in the encoding";
          return std::nullopt;
        }
        if (result.getResultNumber() != route->second.select) {
          error = "encoding reaches an unselected fabric.demux output";
          return std::nullopt;
        }
        activeRoutes.insert(resourceIt->second);
        value = demux.getInput();
        continue;
      }
      return value;
    }
  }

  bool markValue(Value value, Type expected = {}) {
    auto routed = traceSelectedRoute(value, expected);
    if (!routed)
      return false;
    value = *routed;
    if (auto argument = ::mlir::dyn_cast<::mlir::BlockArgument>(value)) {
      if (!expected) {
        error =
            "direct FU input-to-output paths require an explicit software type";
        return false;
      }
      unsigned port = argument.getArgNumber();
      if (!verifyPayloadCapacity(fuInputBoundaryType(fu, port), expected,
                                 "FU input boundary", error))
        return false;
      return recordInputType(port, expected);
    }

    Operation *producer = value.getDefiningOp();
    auto resourceIt = resourceByOp.find(producer);
    if (resourceIt == resourceByOp.end()) {
      error = "configured value is produced outside the FU body";
      return false;
    }
    unsigned resource = resourceIt->second;

    if (auto op = ::mlir::dyn_cast<::fabric::OpOp>(producer)) {
      auto modeIt = encoding.opModes.find(resource);
      if (modeIt == encoding.opModes.end()) {
        error = "active fabric.op has no software mode in the encoding";
        return false;
      }
      const OpMode &mode = modeIt->second;
      unsigned physicalResult =
          ::mlir::cast<::mlir::OpResult>(value).getResultNumber();
      auto portIt = ::llvm::find(mode.outputPorts, physicalResult);
      if (portIt == mode.outputPorts.end()) {
        error = "encoding reaches a fabric.op result outside its software mode";
        return false;
      }
      unsigned semanticResult =
          static_cast<unsigned>(portIt - mode.outputPorts.begin());
      Type produced = mode.functionType.getResult(semanticResult);
      if (expected && !sameType(expected, produced)) {
        error = "connected software value types do not match";
        return false;
      }
      if (!activeOps.insert(resource).second)
        return true;
      for (auto [semanticInput, physicalInput] :
           ::llvm::enumerate(mode.inputPorts)) {
        if (!markValue(op.getInputs()[physicalInput],
                       mode.functionType.getInput(semanticInput)))
          return false;
      }
      for (unsigned physicalOutput : mode.outputPorts)
        aliveValues.insert(op.getOutputs()[physicalOutput]);
      return true;
    }

    error = "unsupported producer in FU topology";
    return false;
  }

  bool useIsActive(::mlir::OpOperand &use) const {
    Operation *owner = use.getOwner();
    if (auto yield = ::mlir::dyn_cast<::fabric::YieldOp>(owner)) {
      return ::llvm::is_contained(encoding.outputPorts, use.getOperandNumber());
    }
    auto resourceIt = resourceByOp.find(owner);
    if (resourceIt == resourceByOp.end())
      return false;
    unsigned resource = resourceIt->second;
    if (auto op = ::mlir::dyn_cast<::fabric::OpOp>(owner)) {
      auto mode = encoding.opModes.find(resource);
      return mode != encoding.opModes.end() && activeOps.count(resource) &&
             ::llvm::is_contained(mode->second.inputPorts,
                                  use.getOperandNumber());
    }
    if (::mlir::isa<::fabric::MuxOp>(owner)) {
      auto route = encoding.routes.find(resource);
      return route != encoding.routes.end() && activeRoutes.count(resource) &&
             route->second.select == use.getOperandNumber();
    }
    if (::mlir::isa<::fabric::DemuxOp>(owner))
      return activeRoutes.count(resource) && use.getOperandNumber() == 0;
    return false;
  }

  bool verifyFanout() {
    for (Value value : aliveValues) {
      for (::mlir::OpOperand &use : value.getUses()) {
        if (useIsActive(use))
          continue;
        error = "an active Fabric value has an inactive consumer; explicit "
                "demux routing is required";
        return false;
      }
    }
    return true;
  }

  std::optional<ConfiguredValue>
  sourceOf(Value value, Type expected,
           const ::llvm::DenseMap<unsigned, unsigned> &nodeByResource) {
    auto routed = traceSelectedRoute(value, expected);
    if (!routed)
      return std::nullopt;
    value = *routed;
    if (auto argument = ::mlir::dyn_cast<::mlir::BlockArgument>(value)) {
      unsigned port = argument.getArgNumber();
      if (!verifyPayloadCapacity(fuInputBoundaryType(fu, port), expected,
                                 "FU input boundary", error) ||
          !recordInputType(port, expected))
        return std::nullopt;
      return ConfiguredValue::input(port);
    }
    Operation *producer = value.getDefiningOp();
    unsigned resource = resourceByOp.lookup(producer);
    if (auto op = ::mlir::dyn_cast<::fabric::OpOp>(producer)) {
      const OpMode &mode = encoding.opModes.lookup(resource);
      unsigned physicalResult =
          ::mlir::cast<::mlir::OpResult>(value).getResultNumber();
      auto port = ::llvm::find(mode.outputPorts, physicalResult);
      if (port == mode.outputPorts.end())
        return std::nullopt;
      unsigned semanticResult =
          static_cast<unsigned>(port - mode.outputPorts.begin());
      if (!sameType(mode.functionType.getResult(semanticResult), expected))
        return std::nullopt;
      return ConfiguredValue::nodeResult(nodeByResource.lookup(resource),
                                         semanticResult);
    }
    return std::nullopt;
  }

  bool run(ConfiguredFunction &function) {
    for (auto [index, op] : ::llvm::enumerate(bodyOps))
      resourceByOp[op] = index;

    auto yield =
        ::mlir::cast<::fabric::YieldOp>(fu.getBody().front().getTerminator());
    for (unsigned outputPort : encoding.outputPorts) {
      if (!markValue(yield.getValues()[outputPort]))
        return false;
    }
    if (!verifyFanout())
      return false;
    for (unsigned resource : encoding.mentionedResources) {
      if (activeOps.count(resource) || activeRoutes.count(resource))
        continue;
      error = "encoding contains a configuration that does not affect the "
              "configured function";
      return false;
    }

    ::llvm::SmallVector<unsigned, 8> activeResources(activeOps.begin(),
                                                     activeOps.end());
    ::llvm::sort(activeResources);
    ::llvm::DenseMap<unsigned, unsigned> nodeByResource;
    for (auto [nodeIndex, resource] : ::llvm::enumerate(activeResources))
      nodeByResource[resource] = nodeIndex;

    function.nodes.reserve(activeResources.size());
    for (unsigned resource : activeResources) {
      const OpMode &mode = encoding.opModes.lookup(resource);
      auto op = ::mlir::cast<::fabric::OpOp>(bodyOps[resource]);
      ConfiguredFunctionNode node;
      node.fabricResource = resource;
      node.operationName = mode.operationName;
      node.functionType = mode.functionType;
      node.attributes = mode.attributes;
      for (auto [semanticInput, physicalInput] :
           ::llvm::enumerate(mode.inputPorts)) {
        auto source =
            sourceOf(op.getInputs()[physicalInput],
                     mode.functionType.getInput(semanticInput), nodeByResource);
        if (!source) {
          error = "failed to resolve an active software operand";
          return false;
        }
        node.operands.push_back(*source);
      }
      function.nodes.push_back(std::move(node));
    }

    ::llvm::SmallVector<unsigned, 4> inputPorts;
    inputPorts.reserve(inputTypes.size());
    for (auto &entry : inputTypes)
      inputPorts.push_back(entry.first);
    ::llvm::sort(inputPorts);
    for (unsigned port : inputPorts)
      function.inputs.push_back({port, inputTypes.lookup(port)});

    for (unsigned outputPort : encoding.outputPorts) {
      auto routed = traceSelectedRoute(yield.getValues()[outputPort]);
      if (!routed)
        return false;
      Value value = *routed;
      Type type;
      if (auto result = ::mlir::dyn_cast<::mlir::OpResult>(value)) {
        unsigned resource = resourceByOp.lookup(result.getOwner());
        const OpMode &mode = encoding.opModes.lookup(resource);
        unsigned physicalResult = result.getResultNumber();
        auto port = ::llvm::find(mode.outputPorts, physicalResult);
        if (port == mode.outputPorts.end()) {
          error = "failed to resolve a configured output type";
          return false;
        }
        type = mode.functionType.getResult(port - mode.outputPorts.begin());
      } else if (auto argument =
                     ::mlir::dyn_cast<::mlir::BlockArgument>(value)) {
        type = inputTypes.lookup(argument.getArgNumber());
      }
      if (!type) {
        error = "failed to resolve a configured output type";
        return false;
      }
      if (!verifyPayloadCapacity(fuOutputBoundaryType(fu, outputPort), type,
                                 "FU output boundary", error))
        return false;
      auto source =
          sourceOf(yield.getValues()[outputPort], type, nodeByResource);
      if (!source) {
        error = "failed to resolve a configured output value";
        return false;
      }
      function.outputs.push_back({outputPort, type, *source});
    }
    return true;
  }
};

struct MatchState {
  MatchState(const ConfiguredFunction &pattern,
             const ConfiguredFunction &candidate, bool preserveBoundary)
      : pattern(pattern), candidate(candidate),
        preserveBoundary(preserveBoundary) {}

  const ConfiguredFunction &pattern;
  const ConfiguredFunction &candidate;
  bool preserveBoundary;
  ::llvm::SmallVector<int, 8> nodeMap;
  ::llvm::SmallVector<int, 8> reverseNodeMap;
  ::llvm::DenseMap<unsigned, unsigned> inputMap;
  ::llvm::DenseMap<unsigned, unsigned> reverseInputMap;

  bool matchValue(const ConfiguredValue &lhs, const ConfiguredValue &rhs) {
    if (lhs.kind != rhs.kind)
      return false;
    if (lhs.kind == ConfiguredValue::Kind::InputPort) {
      if (preserveBoundary)
        return lhs.index == rhs.index;
      auto existing = inputMap.find(lhs.index);
      if (existing != inputMap.end())
        return existing->second == rhs.index;
      auto reverse = reverseInputMap.find(rhs.index);
      if (reverse != reverseInputMap.end())
        return false;
      inputMap[lhs.index] = rhs.index;
      reverseInputMap[rhs.index] = lhs.index;
      return true;
    }
    if (lhs.result != rhs.result)
      return false;
    return matchNode(lhs.index, rhs.index);
  }

  bool matchNode(unsigned lhs, unsigned rhs) {
    if (lhs >= pattern.nodes.size() || rhs >= candidate.nodes.size())
      return false;
    if (nodeMap[lhs] >= 0)
      return static_cast<unsigned>(nodeMap[lhs]) == rhs;
    if (reverseNodeMap[rhs] >= 0)
      return false;

    const ConfiguredFunctionNode &a = pattern.nodes[lhs];
    const ConfiguredFunctionNode &b = candidate.nodes[rhs];
    if (a.operationName != b.operationName ||
        !sameType(a.functionType, b.functionType) ||
        !sameAttributes(a.attributes, b.attributes) ||
        a.operands.size() != b.operands.size())
      return false;

    nodeMap[lhs] = rhs;
    reverseNodeMap[rhs] = lhs;
    for (auto [aOperand, bOperand] : ::llvm::zip(a.operands, b.operands)) {
      if (!matchValue(aOperand, bOperand))
        return false;
    }
    return true;
  }
};

} // namespace

::mlir::LogicalResult projectConfiguredFunction(FuOp fu,
                                                DictionaryAttr encoding,
                                                ConfiguredFunction &function,
                                                std::string &error) {
  function = {};
  if (!fu || !encoding) {
    error = "missing FU or semantic encoding";
    return ::mlir::failure();
  }
  ::llvm::SmallVector<Operation *, 16> bodyOps;
  for (Operation &op : fu.getBody().front().without_terminator())
    bodyOps.push_back(&op);

  ParsedEncoding parsed;
  if (!parseEncoding(fu, encoding, bodyOps, parsed, error))
    return ::mlir::failure();
  Projector projector{fu, bodyOps, parsed, error};
  if (!projector.run(function))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult projectConfiguredFunctions(
    FuOp fu, ::llvm::SmallVectorImpl<ConfiguredFunction> &functions,
    std::string &error) {
  functions.clear();
  auto encodings = fu.getValidEncodingsAttr();
  if (!encodings) {
    error = "fabric.fu has no valid_encodings attribute";
    return ::mlir::failure();
  }
  functions.reserve(encodings.size());
  for (auto [index, attr] : ::llvm::enumerate(encodings)) {
    auto encoding = ::mlir::dyn_cast<DictionaryAttr>(attr);
    if (!encoding) {
      error = "valid semantic encoding #" + std::to_string(index) +
              " must be a dictionary";
      return ::mlir::failure();
    }
    ConfiguredFunction function;
    std::string detail;
    if (::mlir::failed(
            projectConfiguredFunction(fu, encoding, function, detail))) {
      error =
          "valid semantic encoding #" + std::to_string(index) + ": " + detail;
      return ::mlir::failure();
    }
    functions.push_back(std::move(function));
  }
  return ::mlir::success();
}

bool matchConfiguredFunctions(const ConfiguredFunction &pattern,
                              const ConfiguredFunction &candidate,
                              bool preserveFuBoundaryIdentity,
                              ConfiguredFunctionMatch *witness) {
  if (pattern.nodes.size() != candidate.nodes.size() ||
      pattern.outputs.size() != candidate.outputs.size() ||
      pattern.inputs.size() != candidate.inputs.size())
    return false;

  MatchState state{pattern, candidate, preserveFuBoundaryIdentity};
  state.nodeMap.assign(pattern.nodes.size(), -1);
  state.reverseNodeMap.assign(candidate.nodes.size(), -1);

  for (auto [patternOutput, candidateOutput] :
       ::llvm::zip(pattern.outputs, candidate.outputs)) {
    if (!sameType(patternOutput.type, candidateOutput.type))
      return false;
    if (preserveFuBoundaryIdentity &&
        patternOutput.fuPort != candidateOutput.fuPort)
      return false;
    if (!state.matchValue(patternOutput.value, candidateOutput.value))
      return false;
  }
  for (int mapped : state.nodeMap)
    if (mapped < 0)
      return false;

  for (const ConfiguredBoundaryInput &input : pattern.inputs) {
    auto mapped = state.inputMap.find(input.fuPort);
    unsigned candidatePort =
        preserveFuBoundaryIdentity
            ? input.fuPort
            : (mapped == state.inputMap.end() ? input.fuPort : mapped->second);
    auto candidateInput = ::llvm::find_if(
        candidate.inputs, [&](const ConfiguredBoundaryInput &other) {
          return other.fuPort == candidatePort;
        });
    if (candidateInput == candidate.inputs.end() ||
        !sameType(input.type, candidateInput->type))
      return false;
  }

  if (witness) {
    witness->nodeMap.clear();
    witness->inputPorts.clear();
    witness->outputPorts.clear();
    for (int mapped : state.nodeMap)
      witness->nodeMap.push_back(static_cast<unsigned>(mapped));
    for (const ConfiguredBoundaryInput &input : pattern.inputs) {
      unsigned candidatePort = preserveFuBoundaryIdentity
                                   ? input.fuPort
                                   : state.inputMap.lookup(input.fuPort);
      witness->inputPorts.emplace_back(input.fuPort, candidatePort);
    }
    for (auto [patternOutput, candidateOutput] :
         ::llvm::zip(pattern.outputs, candidate.outputs))
      witness->outputPorts.emplace_back(patternOutput.fuPort,
                                        candidateOutput.fuPort);
  }
  return true;
}

::mlir::LogicalResult verifyNormalizedHardwareModes(OpOp op) {
  auto modes = op.getHwParamsAttr();
  if (!modes || modes.empty())
    return op.emitOpError("normalized hw_params must not be empty");

  ::llvm::StringSet<> listedOperations;
  for (auto [index, attr] : ::llvm::enumerate(op.getOpList())) {
    auto symbol = ::mlir::dyn_cast<FlatSymbolRefAttr>(attr);
    if (!symbol)
      return op.emitOpError("'op_list' entry #")
             << index << " must be a flat symbol reference";
    if (!listedOperations.insert(symbol.getValue()).second)
      return op.emitOpError("op_list contains duplicate @")
             << symbol.getValue();
  }

  ::llvm::SmallVector<OpMode, 4> parsedModes;
  ::llvm::StringSet<> modeOperations;
  for (unsigned modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
    OpMode mode;
    std::string error;
    if (!parseHardwareMode(op, 0, modeIndex, mode, error))
      return op.emitOpError("hw_params mode #") << modeIndex << ": " << error;
    for (unsigned prior = 0; prior < parsedModes.size(); ++prior) {
      if (!sameHardwareMode(parsedModes[prior], mode))
        continue;
      return op.emitOpError("hw_params modes #")
             << prior << " and #" << modeIndex << " are duplicates";
    }
    if (!verifySoftwareOperationMode(mode, op.getContext(), error))
      return op.emitOpError("hw_params mode #") << modeIndex << " " << error;
    modeOperations.insert(mode.operationName);
    parsedModes.push_back(std::move(mode));
  }

  for (StringRef operation : listedOperations.keys()) {
    if (!modeOperations.count(operation))
      return op.emitOpError("op_list operation @")
             << operation << " has no hw_params mode";
  }
  return ::mlir::success();
}

static bool hasNormalizedHardwareModes(OpOp op) {
  auto modes = op.getHwParamsAttr();
  if (!modes || modes.empty())
    return false;
  return ::llvm::all_of(modes, [](Attribute attr) {
    auto mode = ::mlir::dyn_cast<DictionaryAttr>(attr);
    return mode && mode.get("op") && mode.get("function_type") &&
           mode.get("input_ports") && mode.get("output_ports") &&
           mode.get("attributes");
  });
}

static ::mlir::LogicalResult verifyProgrammedNormalizedFu(FuOp fu) {
  bool hasNormalizedMode = false;
  bool hasSelectedMode = false;
  bool hasSelectedRoute = false;
  for (Operation &operation : fu.getBody().front().without_terminator()) {
    if (auto op = ::mlir::dyn_cast<::fabric::OpOp>(&operation)) {
      if (hasNormalizedHardwareModes(op)) {
        hasNormalizedMode = true;
        hasSelectedMode |= static_cast<bool>(op.getSwConfigsAttr());
      }
      continue;
    }
    if (auto mux = ::mlir::dyn_cast<::fabric::MuxOp>(&operation)) {
      hasSelectedRoute |= static_cast<bool>(mux.getSelAttr());
      continue;
    }
    if (auto demux = ::mlir::dyn_cast<::fabric::DemuxOp>(&operation))
      hasSelectedRoute |= static_cast<bool>(demux.getSelAttr());
  }

  if (!hasNormalizedMode)
    return ::mlir::success();
  if (!hasSelectedMode && !hasSelectedRoute)
    return fu.emitOpError(
        "normalized fabric.fu requires non-empty valid_encodings or "
        "complete programmed selections");

  for (Operation &operation : fu.getBody().front().without_terminator()) {
    if (auto op = ::mlir::dyn_cast<::fabric::OpOp>(&operation)) {
      if (!hasNormalizedHardwareModes(op) || !op.getSwConfigsAttr())
        return op.emitOpError(
            "programmed normalized fabric.fu requires a selected normalized "
            "mode on every fabric.op");
      continue;
    }
    if (auto mux = ::mlir::dyn_cast<::fabric::MuxOp>(&operation)) {
      if (!mux.getSelAttr())
        return mux.emitOpError(
            "programmed normalized fabric.fu requires an explicit selection "
            "for every routing resource");
      continue;
    }
    if (auto demux = ::mlir::dyn_cast<::fabric::DemuxOp>(&operation)) {
      if (!demux.getSelAttr())
        return demux.emitOpError(
            "programmed normalized fabric.fu requires an explicit selection "
            "for every routing resource");
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult verifyValidSemanticEncodings(FuOp fu) {
  auto encodings = fu.getValidEncodingsAttr();
  if (!encodings)
    return verifyProgrammedNormalizedFu(fu);
  if (encodings.empty())
    return fu.emitOpError("valid_encodings must not be empty");

  ::llvm::SmallVector<Operation *, 16> bodyOps;
  ::llvm::DenseMap<unsigned, unsigned> modeCounts;
  for (auto [resource, op] :
       ::llvm::enumerate(fu.getBody().front().without_terminator())) {
    bodyOps.push_back(&op);
    if (auto configurable = ::mlir::dyn_cast<::fabric::OpOp>(&op)) {
      if (configurable.getSwConfigsAttr())
        return configurable.emitOpError(
            "canonical FU capability must not persist selected sw_configs");
      auto modes = configurable->getAttrOfType<ArrayAttr>("hw_params");
      if (!modes || modes.empty())
        return configurable.emitOpError(
            "canonical FU capability requires non-empty hw_params");
      if (::mlir::failed(verifyNormalizedHardwareModes(configurable)))
        return ::mlir::failure();
      modeCounts[resource] = modes.size();
      continue;
    }
    if (auto mux = ::mlir::dyn_cast<::fabric::MuxOp>(&op)) {
      if (mux.getSelAttr() || mux.getDiscardAttr() || mux.getDisconnectAttr())
        return mux.emitOpError(
            "canonical FU capability must not persist selected routing");
      continue;
    }
    if (auto demux = ::mlir::dyn_cast<::fabric::DemuxOp>(&op)) {
      if (demux.getSelAttr() || demux.getDiscardAttr() ||
          demux.getDisconnectAttr())
        return demux.emitOpError(
            "canonical FU capability must not persist selected routing");
    }
  }

  ::llvm::SmallVector<ConfiguredFunction, 8> functions;
  std::string error;
  if (::mlir::failed(projectConfiguredFunctions(fu, functions, error)))
    return fu.emitOpError(error);

  ::llvm::DenseSet<uint64_t> referencedModes;
  for (Attribute encodingAttr : encodings) {
    auto encoding = ::mlir::cast<DictionaryAttr>(encodingAttr);
    for (Attribute resourceAttr : encoding.getAs<ArrayAttr>("resources")) {
      auto resource = ::mlir::cast<DictionaryAttr>(resourceAttr);
      auto resourceIndex = readUnsigned(resource.get("resource"));
      if (!resourceIndex || *resourceIndex >= bodyOps.size() ||
          !::mlir::isa<::fabric::OpOp>(bodyOps[*resourceIndex]))
        continue;
      auto modeIndex = readUnsigned(resource.get("mode"));
      if (!modeIndex)
        continue;
      referencedModes.insert((static_cast<uint64_t>(*resourceIndex) << 32) |
                             *modeIndex);
    }
  }
  ::llvm::SmallVector<unsigned, 8> configurableResources;
  configurableResources.reserve(modeCounts.size());
  for (auto [resource, count] : modeCounts) {
    (void)count;
    configurableResources.push_back(resource);
  }
  ::llvm::sort(configurableResources);
  for (unsigned resource : configurableResources) {
    unsigned count = modeCounts.lookup(resource);
    for (unsigned mode = 0; mode < count; ++mode) {
      uint64_t key = (static_cast<uint64_t>(resource) << 32) | mode;
      if (referencedModes.count(key))
        continue;
      return ::mlir::cast<::fabric::OpOp>(bodyOps[resource])
                 .emitOpError("hw_params mode #")
             << mode << " is not referenced by any valid semantic encoding";
    }
  }

  for (unsigned lhs = 0; lhs < functions.size(); ++lhs) {
    for (unsigned rhs = lhs + 1; rhs < functions.size(); ++rhs) {
      if (!matchConfiguredFunctions(functions[lhs], functions[rhs],
                                    /*preserveFuBoundaryIdentity=*/true))
        continue;
      return fu.emitOpError("valid semantic encodings #")
             << lhs << " and #" << rhs
             << " project to isomorphic configured functions";
    }
  }
  return ::mlir::success();
}

unsigned getValidSemanticEncodingCount(FuOp fu) {
  if (auto encodings = fu.getValidEncodingsAttr())
    return encodings.size();
  return 0;
}

} // namespace fabric
