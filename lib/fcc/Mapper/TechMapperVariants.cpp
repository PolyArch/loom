#include "TechMapperInternal.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <optional>

namespace fcc {
namespace techmapper_detail {

namespace {

mlir::Attribute getNodeAttr(const Node *node, llvm::StringRef name) {
  if (!node)
    return {};
  for (const auto &attr : node->attributes) {
    if (attr.getName() == name)
      return attr.getValue();
  }
  return {};
}

unsigned getTypeBitWidth(mlir::Type type) {
  if (auto width = detail::getScalarWidth(type))
    return *width;
  return 0;
}

std::string printType(mlir::Type type) {
  if (!type)
    return "";
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  return text;
}

llvm::StringRef getCompatibleOp(llvm::StringRef dfgOpName) {
  return "";
}

bool opNamesCompatible(llvm::StringRef dfgOpName, llvm::StringRef fuOpName) {
  if (dfgOpName == fuOpName)
    return true;
  llvm::StringRef compat = getCompatibleOp(dfgOpName);
  return !compat.empty() && compat == fuOpName;
}

// --- Config field helpers ---

llvm::StringRef configFieldKindName(FUConfigFieldKind kind) {
  switch (kind) {
  case FUConfigFieldKind::Mux:
    return "mux";
  case FUConfigFieldKind::ConstantValue:
    return "constant_value";
  case FUConfigFieldKind::CmpIPredicate:
    return "cmpi_predicate";
  case FUConfigFieldKind::CmpFPredicate:
    return "cmpf_predicate";
  case FUConfigFieldKind::StreamContCond:
    return "stream_cont_cond";
  case FUConfigFieldKind::JoinMask:
    return "join_mask";
  }
  return "unknown";
}

std::optional<uint64_t> encodeStreamContCond(llvm::StringRef cond) {
  if (cond == "<")
    return 1u << 0;
  if (cond == "<=")
    return 1u << 1;
  if (cond == ">")
    return 1u << 2;
  if (cond == ">=")
    return 1u << 3;
  if (cond == "!=")
    return 1u << 4;
  return std::nullopt;
}

std::optional<std::pair<FUConfigFieldKind, unsigned>>
getConfigurableOpFieldSpec(mlir::Operation &op) {
  llvm::StringRef opName = op.getName().getStringRef();
  if (opName == "handshake.constant") {
    if (op.getNumResults() == 0)
      return std::nullopt;
    unsigned bitWidth = getTypeBitWidth(op.getResult(0).getType());
    if (bitWidth == 0)
      return std::nullopt;
    return std::make_pair(FUConfigFieldKind::ConstantValue, bitWidth);
  }
  if (opName == "arith.cmpi")
    return std::make_pair(FUConfigFieldKind::CmpIPredicate, 4u);
  if (opName == "arith.cmpf")
    return std::make_pair(FUConfigFieldKind::CmpFPredicate, 4u);
  if (opName == "dataflow.stream")
    return std::make_pair(FUConfigFieldKind::StreamContCond, 5u);
  if (opName == "handshake.join" && op.getNumOperands() > 0) {
    if (op.getNumOperands() > kMaxHardwareJoinFanin)
      return std::nullopt;
    return std::make_pair(FUConfigFieldKind::JoinMask,
                          static_cast<unsigned>(op.getNumOperands()));
  }
  return std::nullopt;
}

std::optional<uint64_t> extractFieldValueFromNode(const Node *swNode,
                                                  const FUConfigField &field) {
  if (!swNode)
    return std::nullopt;

  switch (field.kind) {
  case FUConfigFieldKind::Mux:
    return field.value;
  case FUConfigFieldKind::JoinMask:
    return field.value;
  case FUConfigFieldKind::ConstantValue: {
    mlir::Attribute attr = getNodeAttr(swNode, "value");
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
      return intAttr.getValue().getZExtValue();
    if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr))
      return floatAttr.getValue().bitcastToAPInt().getZExtValue();
    return std::nullopt;
  }
  case FUConfigFieldKind::CmpIPredicate: {
    mlir::Attribute attr = getNodeAttr(swNode, "predicate");
    if (auto predAttr = mlir::dyn_cast<mlir::arith::CmpIPredicateAttr>(attr))
      return static_cast<uint64_t>(predAttr.getValue());
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
      return static_cast<uint64_t>(intAttr.getInt());
    return std::nullopt;
  }
  case FUConfigFieldKind::CmpFPredicate: {
    mlir::Attribute attr = getNodeAttr(swNode, "predicate");
    if (auto predAttr = mlir::dyn_cast<mlir::arith::CmpFPredicateAttr>(attr))
      return static_cast<uint64_t>(predAttr.getValue());
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
      return static_cast<uint64_t>(intAttr.getInt());
    return std::nullopt;
  }
  case FUConfigFieldKind::StreamContCond: {
    mlir::Attribute attr = getNodeAttr(swNode, "cont_cond");
    if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr))
      return encodeStreamContCond(strAttr.getValue());
    return std::nullopt;
  }
  }
  return std::nullopt;
}

// --- Graph queries ---

IdIndex findProducerPort(const Graph &graph, IdIndex inputPortId) {
  const Port *port = graph.getPort(inputPortId);
  if (!port)
    return INVALID_ID;
  for (IdIndex edgeId : port->connectedEdges) {
    const Edge *edge = graph.getEdge(edgeId);
    if (edge && edge->dstPort == inputPortId)
      return edge->srcPort;
  }
  return INVALID_ID;
}

IdIndex findProducerEdge(const Graph &graph, IdIndex inputPortId) {
  const Port *port = graph.getPort(inputPortId);
  if (!port)
    return INVALID_ID;
  for (IdIndex edgeId : port->connectedEdges) {
    const Edge *edge = graph.getEdge(edgeId);
    if (edge && edge->dstPort == inputPortId)
      return edgeId;
  }
  return INVALID_ID;
}

bool outputHasExternalUse(const Graph &graph, IdIndex outputPortId,
                          const llvm::DenseSet<IdIndex> &groupNodes) {
  const Port *port = graph.getPort(outputPortId);
  if (!port)
    return false;
  for (IdIndex edgeId : port->connectedEdges) {
    const Edge *edge = graph.getEdge(edgeId);
    if (!edge || edge->srcPort != outputPortId)
      continue;
    const Port *dstPort = graph.getPort(edge->dstPort);
    if (!dstPort || dstPort->parentNode == INVALID_ID)
      continue;
    if (!groupNodes.count(dstPort->parentNode))
      return true;
  }
  return false;
}

// --- Signature / family building ---

std::string buildFamilySignature(const VariantFamily &family) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "in(";
  for (size_t i = 0; i < family.inputTypes.size(); ++i) {
    if (i)
      os << ",";
    os << printType(family.inputTypes[i]);
  }
  os << ")|out(";
  for (size_t i = 0; i < family.outputTypes.size(); ++i) {
    if (i)
      os << ",";
    os << printType(family.outputTypes[i]);
  }
  os << ")|ops(";
  for (size_t i = 0; i < family.ops.size(); ++i) {
    if (i)
      os << ";";
    os << family.ops[i].opName << ":";
    for (size_t j = 0; j < family.ops[i].operands.size(); ++j) {
      if (j)
        os << ",";
      const auto &ref = family.ops[i].operands[j];
      os << (ref.kind == RefKind::Input ? "i" : "o") << ref.index << "."
         << ref.resultIndex;
    }
  }
  os << ")|yield(";
  for (size_t i = 0; i < family.outputs.size(); ++i) {
    if (i)
      os << ";";
    if (!family.outputs[i].has_value()) {
      os << "x";
      continue;
    }
    const auto &ref = *family.outputs[i];
    os << (ref.kind == RefKind::Input ? "i" : "o") << ref.index << "."
       << ref.resultIndex;
  }
  os << ")|cfg(";
  for (size_t i = 0; i < family.configFields.size(); ++i) {
    if (i)
      os << ";";
    const auto &field = family.configFields[i];
    os << configFieldKindName(field.kind) << ":" << field.opIndex << ":"
       << field.bitWidth;
    if (field.kind == FUConfigFieldKind::Mux)
      os << ":" << field.sel << ":" << field.discard << ":"
         << field.disconnect;
    else if (field.kind == FUConfigFieldKind::JoinMask)
      os << ":" << field.value;
  }
  os << ")";
  return os.str();
}

bool isMuxPassThrough(fcc::fabric::MuxOp muxOp) {
  return muxOp.getInputs().size() == 1 && muxOp.getResults().size() == 1;
}

bool isMuxFanIn(fcc::fabric::MuxOp muxOp) {
  return muxOp.getInputs().size() > 1 && muxOp.getResults().size() == 1;
}

bool isMuxFanOut(fcc::fabric::MuxOp muxOp) {
  return muxOp.getInputs().size() == 1 && muxOp.getResults().size() > 1;
}

unsigned getMuxBranchCount(fcc::fabric::MuxOp muxOp) {
  return isMuxFanOut(muxOp) ? muxOp.getResults().size()
                                  : muxOp.getInputs().size();
}

std::optional<ValueRef>
resolveValueRef(mlir::Value value,
                const llvm::DenseMap<mlir::Operation *, unsigned> &bodyOpToIndex,
                const llvm::DenseMap<mlir::Operation *, unsigned> &muxSelection) {
  mlir::Value cur = value;
  while (auto *defOp = cur.getDefiningOp()) {
    auto muxOp = mlir::dyn_cast<fcc::fabric::MuxOp>(defOp);
    if (!muxOp)
      break;
    if (isMuxPassThrough(muxOp)) {
      cur = muxOp.getInputs().front();
      continue;
    }

    auto it = muxSelection.find(defOp);
    unsigned sel =
        (it != muxSelection.end()) ? it->second : static_cast<unsigned>(muxOp.getSel());
    if (isMuxFanIn(muxOp)) {
      if (sel >= muxOp.getInputs().size())
        return std::nullopt;
      cur = muxOp.getInputs()[sel];
      continue;
    }

    if (isMuxFanOut(muxOp)) {
      auto result = mlir::dyn_cast<mlir::OpResult>(cur);
      if (!result || sel >= muxOp.getResults().size() ||
          static_cast<unsigned>(result.getResultNumber()) != sel)
        return std::nullopt;
      cur = muxOp.getInputs().front();
      continue;
    }

    return std::nullopt;
  }

  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(cur))
    return ValueRef{RefKind::Input, blockArg.getArgNumber(), 0};

  mlir::Operation *defOp = cur.getDefiningOp();
  if (!defOp)
    return std::nullopt;
  auto it = bodyOpToIndex.find(defOp);
  if (it == bodyOpToIndex.end())
    return std::nullopt;
  auto result = mlir::cast<mlir::OpResult>(cur);
  return ValueRef{RefKind::OpResult, it->second,
                  static_cast<unsigned>(result.getResultNumber())};
}

// --- Variant family building ---

std::optional<VariantFamily>
buildVariantFamily(fcc::fabric::FunctionUnitOp fuOp, const Node *hwNode,
                   const llvm::DenseMap<mlir::Operation *, unsigned> &muxSelection,
                   const llvm::DenseMap<mlir::Operation *, uint64_t> &joinSelection) {
  if (!hwNode)
    return std::nullopt;

  VariantFamily family;
  family.hwName = fuOp.getSymName().str();
  family.configurable = false;

  auto fnType = fuOp.getFunctionType();
  for (mlir::Type type : fnType.getInputs())
    family.inputTypes.push_back(type);
  for (mlir::Type type : fnType.getResults())
    family.outputTypes.push_back(type);

  llvm::DenseMap<mlir::Operation *, unsigned> bodyOpToIndex;
  llvm::DenseMap<mlir::Operation *, unsigned> displayOpIndex;
  llvm::SmallVector<mlir::Operation *, 8> bodyOps;
  llvm::SmallVector<mlir::Operation *, 4> staticMuxes;
  unsigned nonMuxOrdinal = 0;
  unsigned anyOpOrdinal = 0;
  for (mlir::Operation &bodyOp : fuOp.getBody().front().getOperations()) {
    if (mlir::isa<fcc::fabric::YieldOp>(bodyOp))
      continue;
    displayOpIndex[&bodyOp] = anyOpOrdinal++;
    if (auto muxOp = mlir::dyn_cast<fcc::fabric::MuxOp>(bodyOp)) {
      if (!isMuxPassThrough(muxOp))
        staticMuxes.push_back(&bodyOp);
      continue;
    }
    bodyOpToIndex[&bodyOp] = nonMuxOrdinal++;
    bodyOps.push_back(&bodyOp);
  }

  llvm::DenseSet<unsigned> reachableOps;
  llvm::DenseSet<unsigned> reachableInputs;
  llvm::DenseSet<mlir::Value> visitedValues;

  auto visitValue = [&](auto &&self, mlir::Value value) -> bool {
    if (visitedValues.contains(value))
      return true;
    visitedValues.insert(value);
    auto ref = resolveValueRef(value, bodyOpToIndex, muxSelection);
    if (!ref)
      return false;
    if (ref->kind == RefKind::Input) {
      reachableInputs.insert(ref->index);
      return true;
    }
    if (ref->index >= bodyOps.size())
      return false;
    if (!reachableOps.insert(ref->index).second)
      return true;
    mlir::Operation *op = bodyOps[ref->index];
    uint64_t joinMask = 0;
    bool useJoinMask = false;
    if (op->getName().getStringRef() == "handshake.join") {
      auto it = joinSelection.find(op);
      if (it != joinSelection.end()) {
        joinMask = it->second;
        useJoinMask = true;
      }
    }
    for (unsigned operandIdx = 0; operandIdx < op->getNumOperands();
         ++operandIdx) {
      if (useJoinMask && ((joinMask >> operandIdx) & 1u) == 0)
        continue;
      mlir::Value operand = op->getOperand(operandIdx);
      if (!self(self, operand))
        return false;
    }
    return true;
  };

  auto yieldOp =
      mlir::cast<fcc::fabric::YieldOp>(fuOp.getBody().front().getTerminator());
  for (mlir::Value operand : yieldOp.getOperands()) {
    if (!visitValue(visitValue, operand))
      return std::nullopt;
  }
  if (reachableOps.empty())
    return std::nullopt;

  llvm::DenseMap<unsigned, unsigned> oldToCompact;
  for (unsigned bodyIndex = 0; bodyIndex < bodyOps.size(); ++bodyIndex) {
    if (!reachableOps.contains(bodyIndex))
      continue;
    oldToCompact[bodyIndex] = family.ops.size();
    TemplateOp op;
    op.bodyOpIndex = bodyIndex;
    op.opName = bodyOps[bodyIndex]->getName().getStringRef().str();
    family.ops.push_back(std::move(op));
  }

  for (unsigned compactIdx = 0; compactIdx < family.ops.size(); ++compactIdx) {
    mlir::Operation *op = bodyOps[family.ops[compactIdx].bodyOpIndex];
    uint64_t joinMask = 0;
    bool useJoinMask = false;
    if (op->getName().getStringRef() == "handshake.join") {
      auto it = joinSelection.find(op);
      if (it != joinSelection.end()) {
        joinMask = it->second;
        useJoinMask = true;
      }
    }
    for (unsigned operandIdx = 0; operandIdx < op->getNumOperands();
         ++operandIdx) {
      if (useJoinMask && ((joinMask >> operandIdx) & 1u) == 0)
        continue;
      mlir::Value operand = op->getOperand(operandIdx);
      auto ref = resolveValueRef(operand, bodyOpToIndex, muxSelection);
      if (!ref)
        return std::nullopt;
      if (ref->kind == RefKind::OpResult) {
        auto mapped = oldToCompact.find(ref->index);
        if (mapped == oldToCompact.end())
          return std::nullopt;
        ref->index = mapped->second;
        family.edges.push_back({ref->index, compactIdx});
      }
      family.ops[compactIdx].operands.push_back(*ref);
    }
  }

  family.outputs.resize(fnType.getNumResults());
  for (unsigned outIdx = 0; outIdx < yieldOp.getNumOperands(); ++outIdx) {
    auto ref =
        resolveValueRef(yieldOp.getOperand(outIdx), bodyOpToIndex, muxSelection);
    if (!ref)
      return std::nullopt;
    if (ref->kind == RefKind::OpResult) {
      auto mapped = oldToCompact.find(ref->index);
      if (mapped == oldToCompact.end())
        return std::nullopt;
      ref->index = mapped->second;
    }
    family.outputs[outIdx] = *ref;
  }

  for (size_t muxOrdinal = 0; muxOrdinal < staticMuxes.size(); ++muxOrdinal) {
    auto muxOp = mlir::cast<fcc::fabric::MuxOp>(staticMuxes[muxOrdinal]);
    auto it = muxSelection.find(muxOp.getOperation());
    FUConfigField field;
    field.kind = FUConfigFieldKind::Mux;
    field.opIndex = displayOpIndex.lookup(muxOp.getOperation());
    field.opName = muxOp->getName().getStringRef().str();
    field.bitWidth =
        (getMuxBranchCount(muxOp) > 1 ? llvm::Log2_32_Ceil(getMuxBranchCount(muxOp))
                                      : 0) +
        2;
    field.sel = (it != muxSelection.end()) ? it->second : 0;
    field.value = field.sel;
    bool selectedOutputUsed = false;
    for (mlir::Value operand : yieldOp.getOperands()) {
      mlir::Value cur = operand;
      while (auto *defOp = cur.getDefiningOp()) {
        if (defOp == muxOp.getOperation()) {
          if (isMuxFanOut(muxOp)) {
            auto result = mlir::dyn_cast<mlir::OpResult>(cur);
            if (result &&
                static_cast<unsigned>(result.getResultNumber()) == field.sel)
              selectedOutputUsed = true;
          } else {
            selectedOutputUsed = true;
          }
          break;
        }
        auto nestedMux = mlir::dyn_cast<fcc::fabric::MuxOp>(defOp);
        if (!nestedMux)
          break;
        auto nestedIt = muxSelection.find(defOp);
        unsigned nestedSel = (nestedIt != muxSelection.end())
                                 ? nestedIt->second
                                 : static_cast<unsigned>(nestedMux.getSel());
        if (isMuxPassThrough(nestedMux)) {
          cur = nestedMux.getInputs().front();
          continue;
        }
        if (isMuxFanIn(nestedMux)) {
          if (nestedSel >= nestedMux.getInputs().size())
            break;
          cur = nestedMux.getInputs()[nestedSel];
          continue;
        }
        if (isMuxFanOut(nestedMux)) {
          auto nestedResult = mlir::dyn_cast<mlir::OpResult>(cur);
          if (!nestedResult ||
              nestedSel >= nestedMux.getResults().size() ||
              static_cast<unsigned>(nestedResult.getResultNumber()) != nestedSel)
            break;
          cur = nestedMux.getInputs().front();
          continue;
        }
        if (nestedIt == muxSelection.end())
          break;
      }
      if (selectedOutputUsed)
        break;
    }
    if (isMuxFanOut(muxOp)) {
      field.disconnect = false;
      field.discard = !selectedOutputUsed;
    } else {
      field.disconnect = !selectedOutputUsed;
      field.discard = false;
    }
    family.configFields.push_back(field);
  }

  for (mlir::Operation &bodyOp : fuOp.getBody().front().getOperations()) {
    if (mlir::isa<fcc::fabric::YieldOp>(bodyOp))
      continue;
    if (auto muxOp = mlir::dyn_cast<fcc::fabric::MuxOp>(bodyOp)) {
      if (!isMuxPassThrough(muxOp))
        continue;
    }

    auto spec = getConfigurableOpFieldSpec(bodyOp);
    if (!spec)
      continue;
    auto compactIt = bodyOpToIndex.find(&bodyOp);
    if (compactIt == bodyOpToIndex.end() ||
        !reachableOps.contains(compactIt->second))
      continue;
    auto mapped = oldToCompact.find(compactIt->second);
    if (mapped == oldToCompact.end())
      return std::nullopt;

    FUConfigField field;
    field.kind = spec->first;
    field.opIndex = displayOpIndex.lookup(&bodyOp);
    field.templateOpIndex = mapped->second;
    field.opName = bodyOp.getName().getStringRef().str();
    field.bitWidth = spec->second;
    if (field.kind == FUConfigFieldKind::JoinMask) {
      auto joinIt = joinSelection.find(&bodyOp);
      if (joinIt == joinSelection.end())
        return std::nullopt;
      field.value = joinIt->second;
      field.sel = field.value;
    }
    family.configFields.push_back(std::move(field));
  }

  family.configurable = !family.configFields.empty();
  family.signature = buildFamilySignature(family);
  return family;
}

// --- Config materialization ---

bool materializeConfigFields(const Graph &dfg, const VariantFamily &family,
                             llvm::ArrayRef<IdIndex> swNodesByOp,
                             llvm::SmallVectorImpl<FUConfigField> &out) {
  out.clear();
  out.reserve(family.configFields.size());
  for (const auto &templField : family.configFields) {
    FUConfigField field = templField;
    if (field.kind != FUConfigFieldKind::Mux &&
        field.kind != FUConfigFieldKind::JoinMask) {
      if (field.templateOpIndex >= swNodesByOp.size())
        return false;
      IdIndex swNodeId = swNodesByOp[field.templateOpIndex];
      const Node *swNode = dfg.getNode(swNodeId);
      auto value = extractFieldValueFromNode(swNode, field);
      if (!value)
        return false;
      field.value = *value;
      field.sel = *value;
    }
    out.push_back(std::move(field));
  }
  return true;
}

// --- Pattern matching ---

bool portsMatchProducer(const Graph &dfg, IdIndex producerNodeId,
                        unsigned producerResult, IdIndex consumerNodeId,
                        unsigned consumerOperand) {
  const Node *producer = dfg.getNode(producerNodeId);
  const Node *consumer = dfg.getNode(consumerNodeId);
  if (!producer || !consumer || producerResult >= producer->outputPorts.size() ||
      consumerOperand >= consumer->inputPorts.size())
    return false;
  IdIndex expectedSrcPort = producer->outputPorts[producerResult];
  IdIndex actualSrcPort = findProducerPort(dfg, consumer->inputPorts[consumerOperand]);
  return expectedSrcPort == actualSrcPort;
}

bool buildMatchBindings(const Graph &dfg, const VariantFamily &family,
                        llvm::ArrayRef<IdIndex> swNodesByOp,
                        Match &match) {
  llvm::DenseSet<IdIndex> swNodeSet;
  for (IdIndex swNodeId : swNodesByOp)
    swNodeSet.insert(swNodeId);

  llvm::DenseMap<unsigned, IdIndex> sourceByHwInput;
  llvm::DenseMap<IdIndex, unsigned> outputHwBySwPort;

  for (unsigned opIdx = 0; opIdx < family.ops.size(); ++opIdx) {
    const TemplateOp &templ = family.ops[opIdx];
    IdIndex swNodeId = swNodesByOp[opIdx];
    const Node *swNode = dfg.getNode(swNodeId);
    if (!swNode || swNode->inputPorts.size() != templ.operands.size())
      return false;

    for (unsigned operandIdx = 0; operandIdx < templ.operands.size();
         ++operandIdx) {
      const ValueRef &ref = templ.operands[operandIdx];
      IdIndex swInputPort = swNode->inputPorts[operandIdx];
      IdIndex producerPort = findProducerPort(dfg, swInputPort);
      const Port *producer = dfg.getPort(producerPort);
      if (!producer || producer->parentNode == INVALID_ID)
        return false;

      if (ref.kind == RefKind::OpResult) {
        IdIndex expectedProducer = swNodesByOp[ref.index];
        if (!swNodeSet.contains(producer->parentNode))
          return false;
        if (!portsMatchProducer(dfg, expectedProducer, ref.resultIndex, swNodeId,
                                operandIdx)) {
          return false;
        }
        IdIndex edgeId = findProducerEdge(dfg, swInputPort);
        if (edgeId != INVALID_ID)
          match.internalEdges.push_back(edgeId);
        continue;
      }

      if (swNodeSet.contains(producer->parentNode))
        return false;
      auto it = sourceByHwInput.find(ref.index);
      if (it != sourceByHwInput.end() && it->second != producerPort)
        return false;
      sourceByHwInput[ref.index] = producerPort;

      const Port *swPort = dfg.getPort(swInputPort);
      if (ref.index >= family.inputTypes.size() || !swPort ||
          !canMapSoftwareTypeToHardware(swPort->type, family.inputTypes[ref.index])) {
        return false;
      }
      match.inputBindings.push_back({swInputPort, ref.index});
    }

    for (unsigned outIdx = 0; outIdx < swNode->outputPorts.size(); ++outIdx) {
      IdIndex swOutputPort = swNode->outputPorts[outIdx];
      if (!outputHasExternalUse(dfg, swOutputPort, swNodeSet))
        continue;
      ValueRef target{RefKind::OpResult, opIdx, outIdx};
      std::optional<unsigned> hwOutputIdx;
      for (unsigned hwOut = 0; hwOut < family.outputs.size(); ++hwOut) {
        if (!family.outputs[hwOut].has_value())
          continue;
        if (*family.outputs[hwOut] == target) {
          hwOutputIdx = hwOut;
          break;
        }
      }
      if (!hwOutputIdx.has_value())
        return false;
      const Port *swPort = dfg.getPort(swOutputPort);
      if (!swPort || *hwOutputIdx >= family.outputTypes.size() ||
          !canMapSoftwareTypeToHardware(swPort->type,
                                        family.outputTypes[*hwOutputIdx])) {
        return false;
      }
      outputHwBySwPort[swOutputPort] = *hwOutputIdx;
    }
  }

  for (const auto &binding : outputHwBySwPort)
    match.outputBindings.push_back({binding.first, binding.second});
  if (!materializeConfigFields(dfg, family, swNodesByOp, match.configFields))
    return false;
  return true;
}

void findMatchesRecursive(
    const Graph &dfg, const VariantFamily &family,
    const llvm::StringMap<llvm::SmallVector<IdIndex, 8>> &nodesByOp,
    unsigned nextOp, llvm::SmallVectorImpl<IdIndex> &swNodesByOp,
    llvm::DenseSet<IdIndex> &usedSwNodes, std::vector<Match> &matches,
    unsigned familyIndex) {
  if (nextOp == family.ops.size()) {
    Match match;
    match.familyIndex = familyIndex;
    match.swNodesByOp.append(swNodesByOp.begin(), swNodesByOp.end());
    if (buildMatchBindings(dfg, family, swNodesByOp, match))
      matches.push_back(std::move(match));
    return;
  }

  const TemplateOp &templ = family.ops[nextOp];
  auto nodeIt = nodesByOp.find(templ.opName);
  if (nodeIt == nodesByOp.end())
    return;

  for (IdIndex swNodeId : nodeIt->second) {
    if (usedSwNodes.contains(swNodeId))
      continue;
    const Node *swNode = dfg.getNode(swNodeId);
    if (!swNode || swNode->kind != Node::OperationNode)
      continue;

    bool compatible = true;
    for (unsigned operandIdx = 0; operandIdx < templ.operands.size();
         ++operandIdx) {
      const ValueRef &ref = templ.operands[operandIdx];
      if (ref.kind != RefKind::OpResult || ref.index >= nextOp)
        continue;
      if (!portsMatchProducer(dfg, swNodesByOp[ref.index], ref.resultIndex,
                              swNodeId, operandIdx)) {
        compatible = false;
        break;
      }
    }
    if (!compatible)
      continue;

    usedSwNodes.insert(swNodeId);
    swNodesByOp[nextOp] = swNodeId;
    findMatchesRecursive(dfg, family, nodesByOp, nextOp + 1, swNodesByOp,
                         usedSwNodes, matches, familyIndex);
    usedSwNodes.erase(swNodeId);
  }
}

} // anonymous namespace

// --- Public functions in techmapper_detail namespace ---

IdIndex findFunctionUnitNode(const Graph &adg, llvm::StringRef peName,
                             llvm::StringRef fuName) {
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size()); ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      continue;
    if (getNodeAttrStr(hwNode, "resource_class") != "functional")
      continue;
    if (getNodeAttrStr(hwNode, "pe_name") == peName &&
        getNodeAttrStr(hwNode, "op_name") == fuName)
      return hwId;
  }
  return INVALID_ID;
}

void collectVariantsForFU(
    fcc::fabric::FunctionUnitOp fuOp, const Node *hwNode,
    llvm::SmallVectorImpl<VariantFamily> &variants) {
  llvm::SmallVector<fcc::fabric::MuxOp, 4> muxes;
  llvm::SmallVector<mlir::Operation *, 4> joins;
  fuOp.walk([&](fcc::fabric::MuxOp muxOp) {
    if (!isMuxPassThrough(muxOp))
      muxes.push_back(muxOp);
  });
  for (mlir::Operation &bodyOp : fuOp.getBody().front().getOperations()) {
    if (mlir::isa<fcc::fabric::YieldOp>(bodyOp))
      continue;
    if (bodyOp.getName().getStringRef() == "handshake.join")
      joins.push_back(&bodyOp);
  }

  llvm::DenseMap<mlir::Operation *, unsigned> muxSelection;
  llvm::DenseMap<mlir::Operation *, uint64_t> joinSelection;

  auto emitFamily = [&]() {
    auto family = buildVariantFamily(fuOp, hwNode, muxSelection, joinSelection);
    if (family)
      variants.push_back(*family);
  };

  std::function<void(size_t)> enumerateJoins = [&](size_t joinIdx) {
    if (joinIdx == joins.size()) {
      emitFamily();
      return;
    }

    mlir::Operation *joinOp = joins[joinIdx];
    unsigned inputCount = joinOp->getNumOperands();
    if (inputCount == 0 || inputCount > kMaxHardwareJoinFanin)
      return;

    std::function<void(unsigned, uint64_t)> enumerateMasks =
        [&](unsigned bit, uint64_t mask) {
          if (bit == inputCount) {
            if (mask != 0) {
              joinSelection[joinOp] = mask;
              enumerateJoins(joinIdx + 1);
            }
            return;
          }
          enumerateMasks(bit + 1, mask);
          enumerateMasks(bit + 1, mask | (uint64_t{1} << bit));
        };
    enumerateMasks(0, 0);
    joinSelection.erase(joinOp);
  };

  std::function<void(size_t)> enumerateMuxes = [&](size_t muxIdx) {
    if (muxIdx == muxes.size()) {
      enumerateJoins(0);
      return;
    }

    fcc::fabric::MuxOp muxOp = muxes[muxIdx];
    unsigned branchCount = getMuxBranchCount(muxOp);
    for (unsigned sel = 0; sel < branchCount; ++sel) {
      muxSelection[muxOp.getOperation()] = sel;
      enumerateMuxes(muxIdx + 1);
    }
    muxSelection.erase(muxOp.getOperation());
  };

  if (muxes.empty() && joins.empty()) {
    emitFamily();
    return;
  }

  enumerateMuxes(0);
}

std::vector<Match>
findMatchesForFamily(const Graph &dfg, const VariantFamily &family,
                     unsigned familyIndex) {
  llvm::StringMap<llvm::SmallVector<IdIndex, 8>> nodesByOp;
  for (IdIndex swId = 0; swId < static_cast<IdIndex>(dfg.nodes.size()); ++swId) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode || swNode->kind != Node::OperationNode)
      continue;
    llvm::StringRef opName = getNodeAttrStr(swNode, "op_name");
    nodesByOp[opName.str()].push_back(swId);
    llvm::StringRef compat = getCompatibleOp(opName);
    if (!compat.empty())
      nodesByOp[compat.str()].push_back(swId);
  }

  std::vector<Match> matches;
  llvm::SmallVector<IdIndex, 4> swNodesByOp(family.ops.size(), INVALID_ID);
  llvm::DenseSet<IdIndex> usedSwNodes;
  findMatchesRecursive(dfg, family, nodesByOp, 0, swNodesByOp, usedSwNodes,
                       matches, familyIndex);
  return matches;
}

} // namespace techmapper_detail
} // namespace fcc
