#include "TechMapperInternal.h"
#include "loom/Mapper/OpCompat.h"
#include "loom/Mapper/TypeCompat.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <tuple>

namespace loom {
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

bool opNamesCompatible(llvm::StringRef dfgOpName, llvm::StringRef fuOpName) {
  if (dfgOpName == fuOpName)
    return true;
  llvm::StringRef compat = opcompat::getCompatibleOp(dfgOpName);
  return !compat.empty() && compat == fuOpName;
}

bool isCommutativeOp(llvm::StringRef opName) {
  return opName == "arith.addi" || opName == "arith.addf" ||
         opName == "arith.muli" || opName == "arith.mulf" ||
         opName == "arith.andi" || opName == "arith.ori" ||
         opName == "arith.xori";
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

std::string serializeConfigFields(llvm::ArrayRef<FUConfigField> fields) {
  llvm::SmallVector<std::string, 4> tokens;
  tokens.reserve(fields.size());
  for (const auto &field : fields) {
    std::string token;
    llvm::raw_string_ostream os(token);
    os << static_cast<unsigned>(field.kind) << ":" << field.opIndex << ":"
       << field.templateOpIndex << ":" << field.opName << ":" << field.bitWidth
       << ":" << field.value << ":" << field.sel << ":" << field.discard << ":"
       << field.disconnect;
    tokens.push_back(os.str());
  }
  std::sort(tokens.begin(), tokens.end());
  std::string text;
  llvm::raw_string_ostream joined(text);
  for (size_t idx = 0; idx < tokens.size(); ++idx) {
    if (idx)
      joined << ";";
    joined << tokens[idx];
  }
  return text;
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
    const auto &op = family.ops[i];
    os << op.opName << ":";
    llvm::SmallVector<std::string, 4> operandTokens;
    operandTokens.reserve(op.operands.size());
    for (const auto &ref : op.operands) {
      std::string token;
      llvm::raw_string_ostream tokenOs(token);
      tokenOs << (ref.kind == RefKind::Input ? "i" : "o") << ref.index << "."
              << ref.resultIndex;
      operandTokens.push_back(tokenOs.str());
    }
    if (op.commutative)
      std::sort(operandTokens.begin(), operandTokens.end());
    for (size_t j = 0; j < operandTokens.size(); ++j) {
      if (j)
        os << ",";
      os << operandTokens[j];
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
  llvm::SmallVector<std::string, 4> configTokens;
  configTokens.reserve(family.configFields.size());
  for (const auto &field : family.configFields) {
    std::string token;
    llvm::raw_string_ostream tokenOs(token);
    tokenOs << configFieldKindName(field.kind) << ":" << field.opIndex << ":"
            << field.bitWidth;
    if (field.kind == FUConfigFieldKind::Mux)
      tokenOs << ":" << field.sel << ":" << field.discard << ":"
              << field.disconnect;
    else if (field.kind == FUConfigFieldKind::JoinMask)
      tokenOs << ":" << field.value;
    configTokens.push_back(tokenOs.str());
  }
  std::sort(configTokens.begin(), configTokens.end());
  for (size_t i = 0; i < configTokens.size(); ++i) {
    if (i)
      os << ";";
    os << configTokens[i];
  }
  os << ")";
  return os.str();
}

bool configFieldLess(const FUConfigField &lhs, const FUConfigField &rhs) {
  return std::tie(lhs.kind, lhs.opIndex, lhs.templateOpIndex, lhs.opName,
                  lhs.bitWidth, lhs.value, lhs.sel, lhs.discard,
                  lhs.disconnect) <
         std::tie(rhs.kind, rhs.opIndex, rhs.templateOpIndex, rhs.opName,
                  rhs.bitWidth, rhs.value, rhs.sel, rhs.discard,
                  rhs.disconnect);
}

bool isMuxPassThrough(loom::fabric::MuxOp muxOp) {
  return muxOp.getInputs().size() == 1 && muxOp.getResults().size() == 1;
}

bool isMuxFanIn(loom::fabric::MuxOp muxOp) {
  return muxOp.getInputs().size() > 1 && muxOp.getResults().size() == 1;
}

bool isMuxFanOut(loom::fabric::MuxOp muxOp) {
  return muxOp.getInputs().size() == 1 && muxOp.getResults().size() > 1;
}

unsigned getMuxBranchCount(loom::fabric::MuxOp muxOp) {
  return isMuxFanOut(muxOp) ? muxOp.getResults().size()
                                  : muxOp.getInputs().size();
}

std::optional<ValueRef>
resolveValueRef(mlir::Value value,
                const llvm::DenseMap<mlir::Operation *, unsigned> &bodyOpToIndex,
                const llvm::DenseMap<mlir::Operation *, unsigned> &muxSelection) {
  mlir::Value cur = value;
  while (auto *defOp = cur.getDefiningOp()) {
    auto muxOp = mlir::dyn_cast<loom::fabric::MuxOp>(defOp);
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
buildVariantFamily(loom::fabric::FunctionUnitOp fuOp, const Node *hwNode,
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
    if (mlir::isa<loom::fabric::YieldOp>(bodyOp))
      continue;
    displayOpIndex[&bodyOp] = anyOpOrdinal++;
    if (auto muxOp = mlir::dyn_cast<loom::fabric::MuxOp>(bodyOp)) {
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
      mlir::cast<loom::fabric::YieldOp>(fuOp.getBody().front().getTerminator());
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
    op.commutative = isCommutativeOp(op.opName);
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
    auto muxOp = mlir::cast<loom::fabric::MuxOp>(staticMuxes[muxOrdinal]);
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
        auto nestedMux = mlir::dyn_cast<loom::fabric::MuxOp>(defOp);
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
    if (mlir::isa<loom::fabric::YieldOp>(bodyOp))
      continue;
    if (auto muxOp = mlir::dyn_cast<loom::fabric::MuxOp>(bodyOp)) {
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
  std::sort(family.configFields.begin(), family.configFields.end(),
            configFieldLess);
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
    llvm::ArrayRef<unsigned> operandOrder =
        opIdx < match.operandOrderByOp.size() ? match.operandOrderByOp[opIdx]
                                              : llvm::ArrayRef<unsigned>();
    if (!operandOrder.empty() && operandOrder.size() != templ.operands.size())
      return false;

    for (unsigned operandIdx = 0; operandIdx < templ.operands.size();
         ++operandIdx) {
      const ValueRef &ref = templ.operands[operandIdx];
      unsigned swOperandIdx =
          operandOrder.empty() ? operandIdx : operandOrder[operandIdx];
      if (swOperandIdx >= swNode->inputPorts.size())
        return false;
      IdIndex swInputPort = swNode->inputPorts[swOperandIdx];
      IdIndex producerPort = findProducerPort(dfg, swInputPort);
      const Port *producer = dfg.getPort(producerPort);
      if (!producer || producer->parentNode == INVALID_ID)
        return false;

      if (ref.kind == RefKind::OpResult) {
        IdIndex expectedProducer = swNodesByOp[ref.index];
        if (!swNodeSet.contains(producer->parentNode))
          return false;
        if (!portsMatchProducer(dfg, expectedProducer, ref.resultIndex, swNodeId,
                                swOperandIdx)) {
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

llvm::SmallVector<llvm::SmallVector<unsigned, 2>, 2>
enumerateOperandOrders(const TemplateOp &templ, const Node *swNode) {
  llvm::SmallVector<llvm::SmallVector<unsigned, 2>, 2> orders;
  if (!swNode || swNode->inputPorts.size() != templ.operands.size())
    return orders;

  llvm::SmallVector<unsigned, 2> identity;
  for (unsigned idx = 0; idx < templ.operands.size(); ++idx)
    identity.push_back(idx);
  orders.push_back(identity);

  if (!templ.commutative || templ.operands.size() != 2)
    return orders;

  llvm::SmallVector<unsigned, 2> swapped{1, 0};
  if (swapped != identity)
    orders.push_back(swapped);
  return orders;
}

void findMatchesRecursive(
    const Graph &dfg, const VariantFamily &family,
    const llvm::StringMap<llvm::SmallVector<IdIndex, 8>> &nodesByOp,
    unsigned nextOp, llvm::SmallVectorImpl<IdIndex> &swNodesByOp,
    llvm::SmallVectorImpl<llvm::SmallVector<unsigned, 4>> &operandOrderByOp,
    llvm::DenseSet<IdIndex> &usedSwNodes, std::vector<Match> &matches,
    unsigned familyIndex) {
  if (nextOp == family.ops.size()) {
    Match match;
    match.familyIndex = familyIndex;
    match.swNodesByOp.append(swNodesByOp.begin(), swNodesByOp.end());
    match.operandOrderByOp.append(operandOrderByOp.begin(),
                                  operandOrderByOp.end());
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
    for (const auto &operandOrder : enumerateOperandOrders(templ, swNode)) {
      bool compatible = true;
      for (unsigned operandIdx = 0; operandIdx < templ.operands.size();
           ++operandIdx) {
        const ValueRef &ref = templ.operands[operandIdx];
        if (ref.kind != RefKind::OpResult || ref.index >= nextOp)
          continue;
        unsigned swOperandIdx = operandOrder[operandIdx];
        if (!portsMatchProducer(dfg, swNodesByOp[ref.index], ref.resultIndex,
                                swNodeId, swOperandIdx)) {
          compatible = false;
          break;
        }
      }
      if (!compatible)
        continue;

      usedSwNodes.insert(swNodeId);
      swNodesByOp[nextOp] = swNodeId;
      operandOrderByOp[nextOp].assign(operandOrder.begin(), operandOrder.end());
      findMatchesRecursive(dfg, family, nodesByOp, nextOp + 1, swNodesByOp,
                           operandOrderByOp, usedSwNodes, matches, familyIndex);
      usedSwNodes.erase(swNodeId);
      operandOrderByOp[nextOp].clear();
    }
  }
}

struct DemandBodyInfo {
  loom::fabric::FunctionUnitOp fuOp;
  const Node *hwNode = nullptr;
  llvm::SmallVector<mlir::Operation *, 8> bodyOps;
  llvm::DenseMap<mlir::Operation *, unsigned> bodyOpToIndex;
  llvm::SmallVector<mlir::Operation *, 8> structuralOps;
  llvm::DenseMap<mlir::Operation *, unsigned> structuralOpToIndex;
  mutable std::map<std::string, std::optional<VariantFamily>>
      structuralFamilyCache;
  DemandMatchStats *stats = nullptr;
};

struct DemandMatchState {
  llvm::DenseMap<mlir::Operation *, IdIndex> swNodeByBodyOp;
  llvm::DenseSet<IdIndex> usedSwNodes;
  llvm::DenseMap<mlir::Operation *, unsigned> muxSelection;
  llvm::DenseMap<mlir::Operation *, uint64_t> joinSelection;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<unsigned, 4>>
      operandOrderByBodyOp;
};

bool bindMuxSelection(llvm::DenseMap<mlir::Operation *, unsigned> &muxSelection,
                      mlir::Operation *muxOp, unsigned sel) {
  auto it = muxSelection.find(muxOp);
  if (it != muxSelection.end())
    return it->second == sel;
  muxSelection[muxOp] = sel;
  return true;
}

bool bindJoinMask(llvm::DenseMap<mlir::Operation *, uint64_t> &joinSelection,
                  mlir::Operation *joinOp, uint64_t mask) {
  auto it = joinSelection.find(joinOp);
  if (it != joinSelection.end())
    return it->second == mask;
  joinSelection[joinOp] = mask;
  return true;
}

void enumerateOperationMatches(const Graph &dfg, mlir::Operation *bodyOp,
                               IdIndex swNodeId, const DemandBodyInfo &info,
                               const DemandMatchState &state,
                               llvm::SmallVectorImpl<DemandMatchState> &results);

void enumerateValueProducerMatches(
    const Graph &dfg, mlir::Value value, IdIndex swNodeId,
    const DemandBodyInfo &info, const DemandMatchState &state,
    llvm::SmallVectorImpl<DemandMatchState> &results) {
  if (mlir::isa<mlir::BlockArgument>(value)) {
    results.push_back(state);
    return;
  }

  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return;

  if (auto muxOp = mlir::dyn_cast<loom::fabric::MuxOp>(defOp)) {
    if (isMuxPassThrough(muxOp)) {
      enumerateValueProducerMatches(dfg, muxOp.getInputs().front(), swNodeId,
                                    info, state, results);
      return;
    }

    if (isMuxFanIn(muxOp)) {
      for (unsigned sel = 0; sel < muxOp.getInputs().size(); ++sel) {
        DemandMatchState trial = state;
        if (!bindMuxSelection(trial.muxSelection, defOp, sel))
          continue;
        enumerateValueProducerMatches(dfg, muxOp.getInputs()[sel], swNodeId,
                                      info, trial, results);
      }
      return;
    }

    if (isMuxFanOut(muxOp)) {
      auto result = mlir::dyn_cast<mlir::OpResult>(value);
      if (!result)
        return;
      unsigned sel = static_cast<unsigned>(result.getResultNumber());
      DemandMatchState trial = state;
      if (!bindMuxSelection(trial.muxSelection, defOp, sel))
        return;
      enumerateValueProducerMatches(dfg, muxOp.getInputs().front(), swNodeId,
                                    info, trial, results);
    }
    return;
  }

  enumerateOperationMatches(dfg, defOp, swNodeId, info, state, results);
}

void enumerateOperandValueToInputPort(
    const Graph &dfg, mlir::Value value, IdIndex swInputPortId,
    const DemandBodyInfo &info, const DemandMatchState &state,
    llvm::SmallVectorImpl<DemandMatchState> &results) {
  const Port *swInputPort = dfg.getPort(swInputPortId);
  if (!swInputPort)
    return;
  IdIndex producerPort = findProducerPort(dfg, swInputPortId);
  const Port *producer = dfg.getPort(producerPort);
  if (!producer || producer->parentNode == INVALID_ID)
    return;
  enumerateValueProducerMatches(dfg, value, producer->parentNode, info, state,
                                results);
}

void enumerateJoinMatches(const Graph &dfg, mlir::Operation *bodyOp,
                          IdIndex swNodeId, const DemandBodyInfo &info,
                          const DemandMatchState &state,
                          llvm::SmallVectorImpl<DemandMatchState> &results) {
  const Node *swNode = dfg.getNode(swNodeId);
  if (!swNode || swNode->inputPorts.size() > bodyOp->getNumOperands() ||
      bodyOp->getNumOperands() > kMaxHardwareJoinFanin)
    return;

  unsigned swArity = swNode->inputPorts.size();
  unsigned hwArity = bodyOp->getNumOperands();
  llvm::SmallVector<unsigned, 8> chosenPositions;

  auto enumerateCombinations = [&](auto &&self, unsigned nextHwIndex,
                                   const DemandMatchState &comboState) -> void {
    if (chosenPositions.size() == swArity) {
      DemandMatchState seededState = comboState;
      uint64_t mask = 0;
      for (unsigned pos : chosenPositions)
        mask |= (uint64_t{1} << pos);
      if (mask == 0)
        return;
      if (!bindJoinMask(seededState.joinSelection, bodyOp, mask))
        return;

      llvm::SmallVector<unsigned, 4> identityOrder;
      for (unsigned idx = 0; idx < swArity; ++idx)
        identityOrder.push_back(idx);
      seededState.operandOrderByBodyOp[bodyOp] = identityOrder;

      llvm::SmallVector<DemandMatchState, 4> frontier;
      frontier.push_back(std::move(seededState));
      for (unsigned idx = 0; idx < swArity; ++idx) {
        llvm::SmallVector<DemandMatchState, 4> nextFrontier;
        for (const DemandMatchState &frontierState : frontier) {
          enumerateOperandValueToInputPort(
              dfg, bodyOp->getOperand(chosenPositions[idx]),
              swNode->inputPorts[idx], info, frontierState, nextFrontier);
        }
        if (nextFrontier.empty())
          return;
        frontier = std::move(nextFrontier);
      }

      results.append(frontier.begin(), frontier.end());
      return;
    }

    unsigned remainingNeed = swArity - chosenPositions.size();
    for (unsigned hwIdx = nextHwIndex; hwIdx + remainingNeed <= hwArity;
         ++hwIdx) {
      chosenPositions.push_back(hwIdx);
      self(self, hwIdx + 1, comboState);
      chosenPositions.pop_back();
    }
  };

  enumerateCombinations(enumerateCombinations, 0, state);
}

void enumerateOperationMatches(const Graph &dfg, mlir::Operation *bodyOp,
                               IdIndex swNodeId, const DemandBodyInfo &info,
                               const DemandMatchState &state,
                               llvm::SmallVectorImpl<DemandMatchState> &results) {
  auto mappedIt = state.swNodeByBodyOp.find(bodyOp);
  if (mappedIt != state.swNodeByBodyOp.end()) {
    if (mappedIt->second == swNodeId)
      results.push_back(state);
    return;
  }

  const Node *swNode = dfg.getNode(swNodeId);
  if (!swNode || swNode->kind != Node::OperationNode ||
      state.usedSwNodes.contains(swNodeId))
    return;

  llvm::StringRef swOpName = getNodeAttrStr(swNode, "op_name");
  llvm::StringRef fuOpName = bodyOp->getName().getStringRef();
  if (!opNamesCompatible(swOpName, fuOpName))
    return;

  DemandMatchState baseState = state;
  baseState.swNodeByBodyOp[bodyOp] = swNodeId;
  baseState.usedSwNodes.insert(swNodeId);

  if (fuOpName == "handshake.join") {
    enumerateJoinMatches(dfg, bodyOp, swNodeId, info, baseState, results);
    return;
  }

  if (swNode->inputPorts.size() != bodyOp->getNumOperands())
    return;

  TemplateOp tempOp;
  tempOp.opName = fuOpName.str();
  tempOp.commutative = isCommutativeOp(fuOpName);
  for (unsigned operandIdx = 0; operandIdx < bodyOp->getNumOperands();
       ++operandIdx)
    tempOp.operands.push_back(ValueRef{RefKind::Input, operandIdx, 0});

  for (const auto &operandOrder : enumerateOperandOrders(tempOp, swNode)) {
    DemandMatchState seededState = baseState;
    llvm::SmallVector<unsigned, 4> order;
    order.append(operandOrder.begin(), operandOrder.end());
    seededState.operandOrderByBodyOp[bodyOp] = order;

    llvm::SmallVector<DemandMatchState, 4> frontier;
    frontier.push_back(std::move(seededState));
    bool operandsMatch = true;
    for (unsigned fuOperandIdx = 0; fuOperandIdx < bodyOp->getNumOperands();
         ++fuOperandIdx) {
      unsigned swOperandIdx = operandOrder[fuOperandIdx];
      llvm::SmallVector<DemandMatchState, 4> nextFrontier;
      for (const DemandMatchState &frontierState : frontier) {
        enumerateOperandValueToInputPort(
            dfg, bodyOp->getOperand(fuOperandIdx), swNode->inputPorts[swOperandIdx],
            info, frontierState, nextFrontier);
      }
      if (nextFrontier.empty()) {
        operandsMatch = false;
        break;
      }
      frontier = std::move(nextFrontier);
    }
    if (!operandsMatch)
      continue;
    results.append(frontier.begin(), frontier.end());
  }
}

std::string buildDemandMatchKey(const VariantFamily &family, const Match &match) {
  std::string key = family.signature;
  key += "|sw(";
  for (size_t idx = 0; idx < match.swNodesByOp.size(); ++idx) {
    if (idx)
      key += ",";
    key += std::to_string(match.swNodesByOp[idx]);
  }
  key += ")|in(";
  for (size_t idx = 0; idx < match.inputBindings.size(); ++idx) {
    if (idx)
      key += ",";
    key += std::to_string(match.inputBindings[idx].swPortId);
    key += "->";
    key += std::to_string(match.inputBindings[idx].hwPortIndex);
  }
  key += ")|out(";
  for (size_t idx = 0; idx < match.outputBindings.size(); ++idx) {
    if (idx)
      key += ",";
    key += std::to_string(match.outputBindings[idx].swPortId);
    key += "->";
    key += std::to_string(match.outputBindings[idx].hwPortIndex);
  }
  key += ")|edge(";
  for (size_t idx = 0; idx < match.internalEdges.size(); ++idx) {
    if (idx)
      key += ",";
    key += std::to_string(match.internalEdges[idx]);
  }
  key += ")|cfg(";
  key += serializeConfigFields(match.configFields);
  key += ")";
  return key;
}

std::string buildDemandStructuralStateKey(const DemandBodyInfo &info,
                                          const DemandMatchState &state) {
  std::string key;
  llvm::raw_string_ostream os(key);
  for (mlir::Operation *bodyOp : info.structuralOps) {
    if (auto muxOp = mlir::dyn_cast<loom::fabric::MuxOp>(bodyOp)) {
      if (isMuxPassThrough(muxOp))
        continue;
      os << "mux#" << info.structuralOpToIndex.lookup(bodyOp) << "=";
      auto it = state.muxSelection.find(bodyOp);
      if (it == state.muxSelection.end())
        os << "unset";
      else
        os << it->second;
      os << ";";
      continue;
    }
    if (bodyOp->getName().getStringRef() == "handshake.join") {
      os << "join#" << info.structuralOpToIndex.lookup(bodyOp) << "=";
      auto it = state.joinSelection.find(bodyOp);
      if (it == state.joinSelection.end())
        os << "unset";
      else
        os << it->second;
      os << ";";
    }
  }
  return key;
}

std::optional<FamilyMatch>
materializeDemandFamilyMatch(const Graph &dfg, const DemandBodyInfo &info,
                             const DemandMatchState &state) {
  std::string structuralKey = buildDemandStructuralStateKey(info, state);
  auto cacheIt = info.structuralFamilyCache.find(structuralKey);
  if (cacheIt == info.structuralFamilyCache.end()) {
    if (info.stats)
      ++info.stats->structuralStateCacheMissCount;
    auto builtFamily = buildVariantFamily(info.fuOp, info.hwNode,
                                          state.muxSelection,
                                          state.joinSelection);
    if (builtFamily && builtFamily->isTechFamily()) {
      cacheIt = info.structuralFamilyCache
                    .emplace(structuralKey, std::move(builtFamily))
                    .first;
    } else {
      cacheIt = info.structuralFamilyCache
                    .emplace(structuralKey, std::nullopt)
                    .first;
    }
  } else if (info.stats) {
    ++info.stats->structuralStateCacheHitCount;
  }
  if (!cacheIt->second)
    return std::nullopt;
  const VariantFamily &family = *cacheIt->second;

  Match match;
  match.operandOrderByOp.reserve(family.ops.size());
  for (const auto &templ : family.ops) {
    if (templ.bodyOpIndex >= info.bodyOps.size())
      return std::nullopt;
    mlir::Operation *bodyOp = info.bodyOps[templ.bodyOpIndex];
    auto swIt = state.swNodeByBodyOp.find(bodyOp);
    if (swIt == state.swNodeByBodyOp.end())
      return std::nullopt;
    match.swNodesByOp.push_back(swIt->second);
    auto orderIt = state.operandOrderByBodyOp.find(bodyOp);
    if (orderIt != state.operandOrderByBodyOp.end())
      match.operandOrderByOp.push_back(orderIt->second);
    else
      match.operandOrderByOp.push_back({});
  }

  if (!buildMatchBindings(dfg, family, match.swNodesByOp, match))
    return std::nullopt;

  FamilyMatch result;
  result.family = family;
  result.match = std::move(match);
  return result;
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
    loom::fabric::FunctionUnitOp fuOp, const Node *hwNode,
    llvm::SmallVectorImpl<VariantFamily> &variants) {
  llvm::SmallVector<loom::fabric::MuxOp, 4> muxes;
  llvm::SmallVector<mlir::Operation *, 4> joins;
  fuOp.walk([&](loom::fabric::MuxOp muxOp) {
    if (!isMuxPassThrough(muxOp))
      muxes.push_back(muxOp);
  });
  for (mlir::Operation &bodyOp : fuOp.getBody().front().getOperations()) {
    if (mlir::isa<loom::fabric::YieldOp>(bodyOp))
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

    loom::fabric::MuxOp muxOp = muxes[muxIdx];
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
    llvm::StringRef compat = opcompat::getCompatibleOp(opName);
    if (!compat.empty())
      nodesByOp[compat.str()].push_back(swId);
  }

  std::vector<Match> matches;
  llvm::SmallVector<IdIndex, 4> swNodesByOp(family.ops.size(), INVALID_ID);
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 4> operandOrderByOp(
      family.ops.size());
  llvm::DenseSet<IdIndex> usedSwNodes;
  findMatchesRecursive(dfg, family, nodesByOp, 0, swNodesByOp, operandOrderByOp,
                       usedSwNodes, matches, familyIndex);
  return matches;
}

std::vector<FamilyMatch>
findDemandDrivenMatchesForFU(const Graph &dfg, loom::fabric::FunctionUnitOp fuOp,
                             const Node *hwNode, DemandMatchStats *stats) {
  std::vector<FamilyMatch> matches;
  if (!hwNode)
    return matches;

  DemandBodyInfo info;
  info.fuOp = fuOp;
  info.hwNode = hwNode;
  info.stats = stats;
  for (mlir::Operation &bodyOp : fuOp.getBody().front().getOperations()) {
    if (mlir::isa<loom::fabric::YieldOp>(bodyOp))
      continue;
    if (auto muxOp = mlir::dyn_cast<loom::fabric::MuxOp>(bodyOp)) {
      if (!isMuxPassThrough(muxOp)) {
        info.structuralOpToIndex[&bodyOp] = info.structuralOps.size();
        info.structuralOps.push_back(&bodyOp);
      }
      continue;
    }
    if (bodyOp.getName().getStringRef() == "handshake.join") {
      info.structuralOpToIndex[&bodyOp] = info.structuralOps.size();
      info.structuralOps.push_back(&bodyOp);
    }
    info.bodyOpToIndex[&bodyOp] = info.bodyOps.size();
    info.bodyOps.push_back(&bodyOp);
  }

  auto yieldOp =
      mlir::cast<loom::fabric::YieldOp>(fuOp.getBody().front().getTerminator());
  llvm::StringSet<> seenKeys;

  for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++swNodeId) {
    const Node *swNode = dfg.getNode(swNodeId);
    if (!swNode || swNode->kind != Node::OperationNode)
      continue;

    for (mlir::Value yieldValue : yieldOp.getOperands()) {
      DemandMatchState seedState;
      llvm::SmallVector<DemandMatchState, 4> resultStates;
      enumerateValueProducerMatches(dfg, yieldValue, swNodeId, info, seedState,
                                    resultStates);
      for (const DemandMatchState &state : resultStates) {
        auto familyMatch = materializeDemandFamilyMatch(dfg, info, state);
        if (!familyMatch)
          continue;
        std::string key =
            buildDemandMatchKey(familyMatch->family, familyMatch->match);
        if (!seenKeys.insert(key).second)
          continue;
        matches.push_back(std::move(*familyMatch));
      }
    }
  }

  if (stats)
    stats->structuralStateCount = info.structuralFamilyCache.size();
  return matches;
}

} // namespace techmapper_detail
} // namespace loom
