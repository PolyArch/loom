//===-- FunctionalBackendImpl.h - Impl for FunctionalSimulationBackend ----===//
//
// Internal header shared between FunctionalBackend.cpp and
// FunctionalBackendOps.cpp.  Not part of the public API.
//
//===----------------------------------------------------------------------===//
#ifndef FCC_SIMULATOR_FUNCTIONALBACKENDIMPL_H
#define FCC_SIMULATOR_FUNCTIONALBACKENDIMPL_H

#include "fcc/Simulator/FunctionalBackend.h"

#include "fcc/Mapper/TypeCompat.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fcc {
namespace sim {

// ---------------------------------------------------------------------------
// Shared helper constants and types
// ---------------------------------------------------------------------------

constexpr unsigned kInvalidOrdinal = std::numeric_limits<unsigned>::max();

struct Token {
  uint64_t data = 0;
  uint16_t tag = 0;
  bool hasTag = false;
};

// ---------------------------------------------------------------------------
// Shared free-function helpers (formerly anonymous namespace)
// ---------------------------------------------------------------------------

inline mlir::Attribute getNodeAttr(const Node *node, llvm::StringRef key) {
  if (!node)
    return {};
  for (const auto &attr : node->attributes) {
    if (attr.getName() == key)
      return attr.getValue();
  }
  return {};
}

inline bool getNodeAttrBool(const Node *node, llvm::StringRef key,
                            bool defaultValue = false) {
  if (auto boolAttr = mlir::dyn_cast_or_null<mlir::BoolAttr>(
          getNodeAttr(node, key)))
    return boolAttr.getValue();
  return defaultValue;
}

inline std::string getNodeAttrString(const Node *node, llvm::StringRef key,
                                     llvm::StringRef defaultValue = "") {
  if (auto strAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(
          getNodeAttr(node, key)))
    return strAttr.getValue().str();
  return defaultValue.str();
}

inline bool opMatches(llvm::StringRef opName, llvm::StringRef shortName) {
  if (opName == shortName)
    return true;
  size_t dot = opName.rfind('.');
  return dot != llvm::StringRef::npos &&
         opName.drop_front(dot + 1) == shortName;
}

inline unsigned getTypeBitWidth(mlir::Type type) {
  if (auto width = detail::getScalarWidth(type))
    return *width;
  return 64;
}

inline uint64_t maskToWidth(uint64_t value, unsigned width) {
  if (width == 0)
    return 0;
  if (width >= 64)
    return value;
  return value & ((uint64_t{1} << width) - 1);
}

inline int64_t signExtendToI64(uint64_t value, unsigned width) {
  if (width == 0)
    return 0;
  if (width >= 64)
    return static_cast<int64_t>(value);
  uint64_t masked = maskToWidth(value, width);
  uint64_t signBit = uint64_t{1} << (width - 1);
  if ((masked & signBit) == 0)
    return static_cast<int64_t>(masked);
  uint64_t extendMask = ~((uint64_t{1} << width) - 1);
  return static_cast<int64_t>(masked | extendMask);
}

inline uint64_t coerceValueToType(uint64_t value, mlir::Type type) {
  return maskToWidth(value, getTypeBitWidth(type));
}

inline unsigned getMemoryElemSizeLog2(const Node *swNode, const Node *hwNode,
                                      const Graph &dfg, const Graph &adg) {
  if (swNode) {
    for (IdIndex portId : swNode->inputPorts) {
      const Port *port = dfg.getPort(portId);
      if (!port || !mlir::isa<mlir::MemRefType>(port->type))
        continue;
      if (auto log2 = detail::getMemRefElementByteWidthLog2(port->type))
        return *log2;
    }
  }

  if (hwNode) {
    for (IdIndex portId : hwNode->inputPorts) {
      const Port *port = adg.getPort(portId);
      if (!port || !mlir::isa<mlir::MemRefType>(port->type))
        continue;
      if (auto log2 = detail::getMemRefElementByteWidthLog2(port->type))
        return *log2;
    }
  }

  return 2;
}

inline std::vector<unsigned> buildBoundaryOrdinals(const Graph &graph,
                                                   Node::Kind kind,
                                                   unsigned &count) {
  std::vector<unsigned> ordinals(graph.nodes.size(), kInvalidOrdinal);
  count = 0;
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(graph.nodes.size());
       ++nodeId) {
    const Node *node = graph.getNode(nodeId);
    if (!node || node->kind != kind)
      continue;
    ordinals[nodeId] = count++;
  }
  return ordinals;
}

inline std::optional<mlir::arith::CmpIPredicate>
getCmpPredicate(const Node *node) {
  mlir::Attribute attr = getNodeAttr(node, "predicate");
  if (!attr)
    return std::nullopt;
  if (auto predAttr = mlir::dyn_cast<mlir::arith::CmpIPredicateAttr>(attr))
    return predAttr.getValue();
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
    return static_cast<mlir::arith::CmpIPredicate>(intAttr.getInt());
  return std::nullopt;
}

inline bool evaluateCmpPredicate(mlir::arith::CmpIPredicate predicate,
                                 uint64_t lhs, uint64_t rhs, unsigned width) {
  switch (predicate) {
  case mlir::arith::CmpIPredicate::eq:
    return maskToWidth(lhs, width) == maskToWidth(rhs, width);
  case mlir::arith::CmpIPredicate::ne:
    return maskToWidth(lhs, width) != maskToWidth(rhs, width);
  case mlir::arith::CmpIPredicate::slt:
    return signExtendToI64(lhs, width) < signExtendToI64(rhs, width);
  case mlir::arith::CmpIPredicate::sle:
    return signExtendToI64(lhs, width) <= signExtendToI64(rhs, width);
  case mlir::arith::CmpIPredicate::sgt:
    return signExtendToI64(lhs, width) > signExtendToI64(rhs, width);
  case mlir::arith::CmpIPredicate::sge:
    return signExtendToI64(lhs, width) >= signExtendToI64(rhs, width);
  case mlir::arith::CmpIPredicate::ult:
    return maskToWidth(lhs, width) < maskToWidth(rhs, width);
  case mlir::arith::CmpIPredicate::ule:
    return maskToWidth(lhs, width) <= maskToWidth(rhs, width);
  case mlir::arith::CmpIPredicate::ugt:
    return maskToWidth(lhs, width) > maskToWidth(rhs, width);
  case mlir::arith::CmpIPredicate::uge:
    return maskToWidth(lhs, width) >= maskToWidth(rhs, width);
  }
  return false;
}

// ---------------------------------------------------------------------------
// FunctionalSimulationBackend::Impl class definition
// ---------------------------------------------------------------------------

struct FunctionalSimulationBackend::Impl {
  struct MemoryRegionBinding {
    uint8_t *data = nullptr;
    size_t sizeBytes = 0;
    unsigned elemSizeLog2 = 2;
    IdIndex hwNode = INVALID_ID;
    IdIndex swNode = INVALID_ID;
  };

  struct StreamState {
    bool active = false;
    uint64_t nextIdx = 0;
    uint64_t step = 0;
    uint64_t bound = 0;
    std::string stepOp = "+=";
    std::string contCond = "<";
  };

  struct CarryState {
    enum Phase : uint8_t { NeedInit = 0, NeedCond = 1, NeedLoop = 2 };
    Phase phase = NeedInit;
    uint64_t initValue = 0;
  };

  struct InvariantState {
    enum Phase : uint8_t { NeedInit = 0, NeedCond = 1 };
    Phase phase = NeedInit;
    uint64_t storedValue = 0;
  };

  struct GateState {
    enum Phase : uint8_t { NeedHead = 0, NeedNext = 1 };
    Phase phase = NeedHead;
  };

  struct NodeRuntime {
    bool emittedOnce = false;
    PerfSnapshot perf;
    StreamState stream;
    CarryState carry;
    InvariantState invariant;
    GateState gate;
  };

  struct Action {
    bool progress = false;
    unsigned tokensIn = 0;
    unsigned tokensOut = 0;
    std::string error;
  };

  explicit Impl(const SimConfig &cfg) : config(cfg) {}

  // -- Data members --
  SimConfig config;
  Graph dfg;
  Graph adg;
  MappingState mapping;
  bool built = false;
  uint64_t configWords = 0;
  unsigned numInputPorts = 0;
  unsigned numOutputPorts = 0;

  std::vector<unsigned> hwInputOrdinals;
  std::vector<unsigned> hwOutputOrdinals;
  std::vector<unsigned> swInputToHwOrdinal;
  std::vector<unsigned> swOutputToHwOrdinal;
  std::vector<IdIndex> hwOrdinalToSwInputNode;
  std::vector<IdIndex> hwOrdinalToSwOutputNode;

  std::vector<IdIndex> inputPortEdge;
  std::vector<std::vector<IdIndex>> outputPortEdges;
  std::vector<std::deque<Token>> edgeQueues;
  std::vector<std::vector<Token>> inputBindings;
  std::vector<std::vector<Token>> outputCollectors;
  std::vector<NodeRuntime> nodeRuntime;
  std::vector<MemoryRegionBinding> memoryRegions;
  llvm::DenseMap<IdIndex, unsigned> swMemoryNodeToRegion;

  // -- Public methods (FunctionalBackend.cpp) --
  std::string connect();
  std::string buildFromMappedState(const Graph &dfgGraph, const Graph &adgGraph,
                                   const MappingState &state);
  std::string loadConfig(const std::vector<uint8_t> &configBlob);
  std::string setInput(unsigned portIdx, const std::vector<uint64_t> &data,
                       const std::vector<uint16_t> &tags);
  std::string setExtMemoryBacking(unsigned regionId, uint8_t *data,
                                  size_t sizeBytes);
  unsigned getNumInputPorts() const;
  unsigned getNumOutputPorts() const;
  std::vector<uint64_t> getOutput(unsigned portIdx) const;
  std::vector<uint16_t> getOutputTags(unsigned portIdx) const;
  void resetExecution();
  void resetAll();
  SimResult invoke(uint32_t epochId, uint64_t invocationId);

  // -- Token management (FunctionalBackend.cpp) --
  void seedInputs();
  bool hasInputToken(IdIndex portId) const;
  Token popInputToken(IdIndex portId);
  void pushToken(IdIndex portId, Token token);

  // -- Value helpers (FunctionalBackend.cpp) --
  unsigned getSelectIndex(uint64_t rawValue, mlir::Type type) const;
  uint64_t applyStreamStep(uint64_t current, uint64_t step,
                           llvm::StringRef stepOp) const;
  bool evaluateStreamCond(uint64_t lhs, uint64_t rhs,
                          llvm::StringRef cond) const;
  std::optional<unsigned> getMemoryRegionId(IdIndex swNodeId) const;

  // -- Memory operations (FunctionalBackend.cpp) --
  bool readMemory(unsigned regionId, uint64_t index, uint64_t &value,
                  std::string &error) const;
  bool writeMemory(unsigned regionId, uint64_t index, uint64_t value,
                   std::string &error);

  // -- Node dispatch (FunctionalBackend.cpp) --
  Action tryExecuteNode(IdIndex nodeId);

  // -- Individual node executors (FunctionalBackendOps.cpp) --
  Action executeModuleOutput(IdIndex nodeId, const Node *node);
  Action executeJoin(IdIndex nodeId, const Node *node);
  Action executeForkLike(IdIndex nodeId, const Node *node);
  Action executeSink(IdIndex nodeId, const Node *node);
  Action executeSource(IdIndex nodeId, const Node *node);
  Action executeMerge(IdIndex nodeId, const Node *node, bool withIndex);
  Action executeHandshakeConstant(IdIndex nodeId, const Node *node);
  Action executeArithConstant(IdIndex nodeId, const Node *node);
  Action executeStream(IdIndex nodeId, const Node *node);
  Action executeGate(IdIndex nodeId, const Node *node);
  Action executeCarry(IdIndex nodeId, const Node *node);
  Action executeInvariant(IdIndex nodeId, const Node *node);
  Action executeCondBr(IdIndex nodeId, const Node *node);
  Action executeSelect(IdIndex nodeId, const Node *node);
  Action executeHandshakeMux(IdIndex nodeId, const Node *node);
  Action executeFabricMux(IdIndex nodeId, const Node *node);
  Action executeLoad(IdIndex nodeId, const Node *node);
  Action executeStore(IdIndex nodeId, const Node *node);
  Action executeMemory(IdIndex nodeId, const Node *node);
  Action executeIndexCast(IdIndex nodeId, const Node *node);
  Action executeCmpi(IdIndex nodeId, const Node *node);
  Action executeBinaryArith(IdIndex nodeId, const Node *node,
                            llvm::StringRef opName);
};

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_FUNCTIONALBACKENDIMPL_H
