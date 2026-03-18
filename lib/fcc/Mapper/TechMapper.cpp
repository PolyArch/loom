#include "fcc/Mapper/TechMapper.h"
#include "fcc/Mapper/TypeCompat.h"

#include "fcc/Dialect/Fabric/FabricOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

namespace fcc {

namespace {

enum class RefKind : uint8_t {
  Input = 0,
  OpResult = 1,
};

struct ValueRef {
  RefKind kind = RefKind::Input;
  unsigned index = 0;
  unsigned resultIndex = 0;

  bool operator==(const ValueRef &other) const {
    return kind == other.kind && index == other.index &&
           resultIndex == other.resultIndex;
  }
};

struct TemplateOp {
  unsigned bodyOpIndex = 0;
  std::string opName;
  llvm::SmallVector<ValueRef, 4> operands;
};

struct VariantFamily {
  std::string signature;
  std::string hwName;
  llvm::SmallVector<IdIndex, 4> hwNodeIds;
  llvm::SmallVector<mlir::Type, 4> inputTypes;
  llvm::SmallVector<mlir::Type, 4> outputTypes;
  llvm::SmallVector<TemplateOp, 4> ops;
  llvm::SmallVector<std::pair<unsigned, unsigned>, 4> edges;
  llvm::SmallVector<std::optional<ValueRef>, 4> outputs;
  llvm::SmallVector<FUConfigField, 2> configFields;
  bool configurable = false;

  bool isTechFamily() const { return configurable || ops.size() > 1; }
};

struct Match {
  unsigned familyIndex = 0;
  llvm::SmallVector<IdIndex, 4> swNodesByOp;
  llvm::SmallVector<TechMapper::PortBinding, 4> inputBindings;
  llvm::SmallVector<TechMapper::PortBinding, 4> outputBindings;
  llvm::SmallVector<IdIndex, 4> internalEdges;
  llvm::SmallVector<FUConfigField, 4> configFields;
};

void addNodeAttr(Node *node, llvm::StringRef key, mlir::Attribute value,
                 mlir::MLIRContext *ctx) {
  node->attributes.push_back(
      mlir::NamedAttribute(mlir::StringAttr::get(ctx, key), value));
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

std::string printType(mlir::Type type) {
  if (!type)
    return "";
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  return text;
}

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
  return std::nullopt;
}

std::optional<uint64_t> extractFieldValueFromNode(const Node *swNode,
                                                  const FUConfigField &field) {
  if (!swNode)
    return std::nullopt;

  switch (field.kind) {
  case FUConfigFieldKind::Mux:
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

std::optional<VariantFamily>
buildVariantFamily(fcc::fabric::FunctionUnitOp fuOp, const Node *hwNode,
                   const llvm::DenseMap<mlir::Operation *, unsigned> &muxSelection) {
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
    for (mlir::Value operand : op->getOperands()) {
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
    for (mlir::Value operand : op->getOperands()) {
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
    family.configFields.push_back(std::move(field));
  }

  family.configurable = !family.configFields.empty();
  family.signature = buildFamilySignature(family);
  return family;
}

bool materializeConfigFields(const Graph &dfg, const VariantFamily &family,
                             llvm::ArrayRef<IdIndex> swNodesByOp,
                             llvm::SmallVectorImpl<FUConfigField> &out) {
  out.clear();
  out.reserve(family.configFields.size());
  for (const auto &templField : family.configFields) {
    FUConfigField field = templField;
    if (field.kind != FUConfigFieldKind::Mux) {
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

void collectVariantsForFU(
    fcc::fabric::FunctionUnitOp fuOp, const Node *hwNode,
    llvm::SmallVectorImpl<VariantFamily> &variants) {
  llvm::SmallVector<fcc::fabric::MuxOp, 4> muxes;
  fuOp.walk([&](fcc::fabric::MuxOp muxOp) {
    if (!isMuxPassThrough(muxOp))
      muxes.push_back(muxOp);
  });

  if (muxes.empty()) {
    llvm::DenseMap<mlir::Operation *, unsigned> empty;
    auto family = buildVariantFamily(fuOp, hwNode, empty);
    if (family)
      variants.push_back(*family);
    return;
  }

  llvm::SmallVector<unsigned, 4> limits;
  limits.reserve(muxes.size());
  for (fcc::fabric::MuxOp muxOp : muxes)
    limits.push_back(getMuxBranchCount(muxOp));

  llvm::SmallVector<unsigned, 4> state(muxes.size(), 0);
  while (true) {
    llvm::DenseMap<mlir::Operation *, unsigned> selection;
    for (size_t i = 0; i < muxes.size(); ++i)
      selection[muxes[i].getOperation()] = state[i];

    auto family = buildVariantFamily(fuOp, hwNode, selection);
    if (family)
      variants.push_back(*family);

    size_t carry = muxes.size();
    while (carry > 0) {
      --carry;
      state[carry] += 1;
      if (state[carry] < limits[carry])
        break;
      state[carry] = 0;
    }
    if (carry == 0 && state[0] == 0)
      break;
  }
}

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

std::unique_ptr<Node> cloneNodeShell(const Node *src, mlir::MLIRContext *ctx) {
  auto node = std::make_unique<Node>();
  node->kind = src->kind;
  node->attributes = src->attributes;
  (void)ctx;
  return node;
}

std::unique_ptr<Port> clonePort(const Port *src) {
  auto port = std::make_unique<Port>();
  port->direction = src->direction;
  port->type = src->type;
  port->attributes = src->attributes;
  return port;
}

} // namespace

bool TechMapper::buildPlan(const Graph &dfg, mlir::ModuleOp adgModule,
                           const Graph &adg, Plan &plan) {
  Plan freshPlan;
  plan = std::move(freshPlan);
  plan.originalNodeToContractedNode.assign(dfg.nodes.size(), INVALID_ID);
  plan.originalPortToContractedPort.assign(dfg.ports.size(), INVALID_ID);
  plan.originalEdgeToContractedEdge.assign(dfg.edges.size(), INVALID_ID);
  plan.originalEdgeKinds.assign(dfg.edges.size(), TechMappedEdgeKind::Routed);
  plan.contractedDFG = Graph(dfg.context);
  plan.coverageScore = 1.0;

  llvm::SmallVector<VariantFamily, 16> familyList;

  llvm::DenseMap<mlir::Block *, llvm::DenseSet<llvm::StringRef>>
      referencedTargetsByBlock;
  adgModule.walk([&](fcc::fabric::InstanceOp instOp) {
    referencedTargetsByBlock[instOp->getBlock()].insert(instOp.getModule());
  });

  auto isDefinitionOp = [&](mlir::Operation *op,
                            llvm::StringRef name) -> bool {
    if (mlir::isa<fcc::fabric::FunctionUnitOp>(op))
      return true;
    if (!mlir::isa<fcc::fabric::SpatialPEOp, fcc::fabric::TemporalPEOp>(op))
      return false;
    return !op->hasAttr("inline_instantiation");
  };

  llvm::StringMap<fcc::fabric::SpatialPEOp> peDefs;
  llvm::StringMap<fcc::fabric::TemporalPEOp> temporalPeDefs;
  llvm::StringMap<fcc::fabric::FunctionUnitOp> functionUnitDefs;
  adgModule->walk([&](fcc::fabric::SpatialPEOp peOp) {
    if (auto symAttr = peOp.getSymNameAttr();
        symAttr && isDefinitionOp(peOp.getOperation(), symAttr.getValue()))
      peDefs[symAttr.getValue()] = peOp;
  });
  adgModule->walk([&](fcc::fabric::TemporalPEOp peOp) {
    if (auto symAttr = peOp.getSymNameAttr();
        symAttr && isDefinitionOp(peOp.getOperation(), symAttr.getValue()))
      temporalPeDefs[symAttr.getValue()] = peOp;
  });
  adgModule->walk([&](fcc::fabric::FunctionUnitOp fuOp) {
    auto symName = fuOp.getSymNameAttr().getValue();
    if (isDefinitionOp(fuOp.getOperation(), symName))
      functionUnitDefs[symName] = fuOp;
  });

  if (auto fabricMod = [&]() -> fcc::fabric::ModuleOp {
        fcc::fabric::ModuleOp found;
        adgModule->walk([&](fcc::fabric::ModuleOp op) {
          if (!found)
            found = op;
        });
        return found;
      }()) {
    auto visitPEFunctionUnits =
        [&](auto peOp, llvm::StringRef peName,
            auto &&visitor) {
          auto &peBody = peOp.getBody().front();
          auto referencedIt = referencedTargetsByBlock.find(&peBody);
          const llvm::DenseSet<llvm::StringRef> *referencedTargets =
              referencedIt != referencedTargetsByBlock.end()
                  ? &referencedIt->second
                  : nullptr;
          for (mlir::Operation &bodyOp : peBody.getOperations()) {
            if (auto fuOp = mlir::dyn_cast<fcc::fabric::FunctionUnitOp>(bodyOp)) {
              llvm::StringRef symName = fuOp.getSymNameAttr().getValue();
              if (!symName.empty() && referencedTargets &&
                  referencedTargets->contains(symName))
                continue;
              visitor(fuOp, symName.str());
              continue;
            }
            auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(bodyOp);
            if (!instOp)
              continue;
            auto fuIt = functionUnitDefs.find(instOp.getModule());
            if (fuIt == functionUnitDefs.end())
              continue;
            visitor(fuIt->second,
                    instOp.getSymName().value_or(instOp.getModule()).str());
          }
        };

    auto recordVariantsForPE = [&](llvm::StringRef peName,
                                   fcc::fabric::FunctionUnitOp fuOp,
                                   const std::string &fuName) {
      IdIndex hwNodeId = findFunctionUnitNode(adg, peName, fuName);
      if (hwNodeId == INVALID_ID)
        return;
      const Node *hwNode = adg.getNode(hwNodeId);
      llvm::SmallVector<VariantFamily, 8> variants;
      collectVariantsForFU(fuOp, hwNode, variants);
      for (auto &variant : variants) {
        bool merged = false;
        for (auto &family : familyList) {
          if (family.signature != variant.signature)
            continue;
          family.hwNodeIds.push_back(hwNodeId);
          merged = true;
          break;
        }
        if (!merged) {
          variant.hwNodeIds.clear();
          variant.hwNodeIds.push_back(hwNodeId);
          familyList.push_back(std::move(variant));
        }
      }
    };

    for (mlir::Operation &op : fabricMod.getBody().front().getOperations()) {
      if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
        std::string peName =
            instOp.getSymName().value_or(instOp.getModule()).str();
        auto peIt = peDefs.find(instOp.getModule());
        if (peIt != peDefs.end()) {
          visitPEFunctionUnits(
              peIt->second, peName,
              [&](fcc::fabric::FunctionUnitOp fuOp, const std::string &fuName) {
                recordVariantsForPE(peName, fuOp, fuName);
              });
          continue;
        }
        auto temporalPeIt = temporalPeDefs.find(instOp.getModule());
        if (temporalPeIt != temporalPeDefs.end()) {
          visitPEFunctionUnits(
              temporalPeIt->second, peName,
              [&](fcc::fabric::FunctionUnitOp fuOp, const std::string &fuName) {
                recordVariantsForPE(peName, fuOp, fuName);
              });
        }
        continue;
      }

      if (auto peOp = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
        llvm::StringRef peName = peOp.getSymName().value_or("");
        visitPEFunctionUnits(
            peOp, peName,
            [&](fcc::fabric::FunctionUnitOp fuOp, const std::string &fuName) {
              recordVariantsForPE(peName, fuOp, fuName);
            });
        continue;
      }

      if (auto peOp = mlir::dyn_cast<fcc::fabric::TemporalPEOp>(op)) {
        llvm::StringRef peName = peOp.getSymName().value_or("");
        visitPEFunctionUnits(
            peOp, peName,
            [&](fcc::fabric::FunctionUnitOp fuOp, const std::string &fuName) {
              recordVariantsForPE(peName, fuOp, fuName);
            });
      }
    }
  }

  for (auto &family : familyList) {
    std::sort(family.hwNodeIds.begin(), family.hwNodeIds.end());
    family.hwNodeIds.erase(
        std::unique(family.hwNodeIds.begin(), family.hwNodeIds.end()),
        family.hwNodeIds.end());
  }

  std::vector<Match> allMatches;
  unsigned techNodeCount = 0;
  unsigned totalOpCount = 0;
  for (const Node *swNode : dfg.nodeRange()) {
    if (swNode && swNode->kind == Node::OperationNode)
      ++totalOpCount;
  }
  for (unsigned familyIndex = 0; familyIndex < familyList.size(); ++familyIndex) {
    if (!familyList[familyIndex].isTechFamily())
      continue;
    auto matches = findMatchesForFamily(dfg, familyList[familyIndex], familyIndex);
    allMatches.insert(allMatches.end(), matches.begin(), matches.end());
  }

  std::sort(allMatches.begin(), allMatches.end(),
            [&](const Match &lhs, const Match &rhs) {
              if (lhs.swNodesByOp.size() != rhs.swNodesByOp.size())
                return lhs.swNodesByOp.size() > rhs.swNodesByOp.size();
              size_t lhsHw = familyList[lhs.familyIndex].hwNodeIds.size();
              size_t rhsHw = familyList[rhs.familyIndex].hwNodeIds.size();
              if (lhsHw != rhsHw)
                return lhsHw > rhsHw;
              return std::lexicographical_compare(
                  lhs.swNodesByOp.begin(), lhs.swNodesByOp.end(),
                  rhs.swNodesByOp.begin(), rhs.swNodesByOp.end());
            });

  std::vector<int> nodeToUnit(dfg.nodes.size(), -1);
  for (const Match &match : allMatches) {
    bool overlaps = false;
    for (IdIndex swNodeId : match.swNodesByOp) {
      if (swNodeId >= nodeToUnit.size() || nodeToUnit[swNodeId] >= 0) {
        overlaps = true;
        break;
      }
    }
    if (overlaps)
      continue;

    TechMapper::Unit unit;
    unit.swNodes = match.swNodesByOp;
    unit.inputBindings = match.inputBindings;
    unit.outputBindings = match.outputBindings;
    unit.internalEdges = match.internalEdges;
    unit.configurable = familyList[match.familyIndex].configurable;
    for (IdIndex hwNodeId : familyList[match.familyIndex].hwNodeIds) {
      TechMapper::Candidate candidate;
      candidate.hwNodeId = hwNodeId;
      candidate.configFields.assign(match.configFields.begin(),
                                    match.configFields.end());
      unit.candidates.push_back(std::move(candidate));
    }
    int unitIndex = static_cast<int>(plan.units.size());
    for (IdIndex swNodeId : unit.swNodes) {
      nodeToUnit[swNodeId] = unitIndex;
      ++techNodeCount;
    }
    plan.units.push_back(std::move(unit));
  }

  if (totalOpCount > 0)
    plan.coverageScore = static_cast<double>(techNodeCount) /
                         static_cast<double>(totalOpCount);

  auto &contracted = plan.contractedDFG;
  contracted.reserve(dfg.countNodes(), dfg.countPorts(), dfg.countEdges());

  std::vector<bool> unitCreated(plan.units.size(), false);
  for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++swNodeId) {
    const Node *swNode = dfg.getNode(swNodeId);
    if (!swNode)
      continue;

    int unitIndex = swNodeId < nodeToUnit.size() ? nodeToUnit[swNodeId] : -1;
    if (unitIndex >= 0) {
      if (unitCreated[unitIndex]) {
        plan.originalNodeToContractedNode[swNodeId] =
            plan.units[unitIndex].contractedNodeId;
        continue;
      }

      const auto &candidate = plan.units[unitIndex].candidates.front();
      const Node *hwNode = adg.getNode(candidate.hwNodeId);
      if (!hwNode)
        return false;

      auto node = std::make_unique<Node>();
      node->kind = Node::OperationNode;
      addNodeAttr(node.get(), "op_name",
                  mlir::StringAttr::get(dfg.context, "techmap_group"),
                  dfg.context);
      addNodeAttr(node.get(), "tech_group_size",
                  mlir::IntegerAttr::get(mlir::IntegerType::get(dfg.context, 32),
                                         plan.units[unitIndex].swNodes.size()),
                  dfg.context);
      IdIndex contractedNodeId = contracted.addNode(std::move(node));
      plan.units[unitIndex].contractedNodeId = contractedNodeId;
      for (IdIndex member : plan.units[unitIndex].swNodes)
        plan.originalNodeToContractedNode[member] = contractedNodeId;

      for (IdIndex hwPortId : hwNode->inputPorts) {
        const Port *hwPort = adg.getPort(hwPortId);
        auto port = std::make_unique<Port>();
        port->direction = Port::Input;
        port->type = hwPort ? hwPort->type : mlir::Type();
        IdIndex portId = contracted.addPort(std::move(port));
        contracted.ports[portId]->parentNode = contractedNodeId;
        contracted.nodes[contractedNodeId]->inputPorts.push_back(portId);
        plan.units[unitIndex].contractedInputPorts.push_back(portId);
      }
      for (IdIndex hwPortId : hwNode->outputPorts) {
        const Port *hwPort = adg.getPort(hwPortId);
        auto port = std::make_unique<Port>();
        port->direction = Port::Output;
        port->type = hwPort ? hwPort->type : mlir::Type();
        IdIndex portId = contracted.addPort(std::move(port));
        contracted.ports[portId]->parentNode = contractedNodeId;
        contracted.nodes[contractedNodeId]->outputPorts.push_back(portId);
        plan.units[unitIndex].contractedOutputPorts.push_back(portId);
      }
      plan.contractedCandidates[contractedNodeId] = {};
      for (const auto &unitCandidate : plan.units[unitIndex].candidates)
        plan.contractedCandidates[contractedNodeId].push_back(
            unitCandidate.hwNodeId);
      unitCreated[unitIndex] = true;
      continue;
    }

    auto node = cloneNodeShell(swNode, dfg.context);
    IdIndex contractedNodeId = contracted.addNode(std::move(node));
    plan.originalNodeToContractedNode[swNodeId] = contractedNodeId;

    for (IdIndex swPortId : swNode->inputPorts) {
      const Port *swPort = dfg.getPort(swPortId);
      auto port = clonePort(swPort);
      IdIndex contractedPortId = contracted.addPort(std::move(port));
      contracted.ports[contractedPortId]->parentNode = contractedNodeId;
      contracted.nodes[contractedNodeId]->inputPorts.push_back(contractedPortId);
      plan.originalPortToContractedPort[swPortId] = contractedPortId;
    }
    for (IdIndex swPortId : swNode->outputPorts) {
      const Port *swPort = dfg.getPort(swPortId);
      auto port = clonePort(swPort);
      IdIndex contractedPortId = contracted.addPort(std::move(port));
      contracted.ports[contractedPortId]->parentNode = contractedNodeId;
      contracted.nodes[contractedNodeId]->outputPorts.push_back(contractedPortId);
      plan.originalPortToContractedPort[swPortId] = contractedPortId;
    }
  }

  for (auto &unit : plan.units) {
    for (const auto &binding : unit.inputBindings) {
      if (binding.swPortId >= plan.originalPortToContractedPort.size() ||
          binding.hwPortIndex >= unit.contractedInputPorts.size())
        return false;
      plan.originalPortToContractedPort[binding.swPortId] =
          unit.contractedInputPorts[binding.hwPortIndex];
    }
    for (const auto &binding : unit.outputBindings) {
      if (binding.swPortId >= plan.originalPortToContractedPort.size() ||
          binding.hwPortIndex >= unit.contractedOutputPorts.size())
        return false;
      plan.originalPortToContractedPort[binding.swPortId] =
          unit.contractedOutputPorts[binding.hwPortIndex];
    }
    for (IdIndex edgeId : unit.internalEdges) {
      if (edgeId < plan.originalEdgeKinds.size())
        plan.originalEdgeKinds[edgeId] = TechMappedEdgeKind::IntraFU;
    }
  }

  std::map<std::string, IdIndex> dedupEdges;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (plan.originalEdgeKinds[edgeId] == TechMappedEdgeKind::IntraFU)
      continue;
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    IdIndex srcPort = plan.originalPortToContractedPort[edge->srcPort];
    IdIndex dstPort = plan.originalPortToContractedPort[edge->dstPort];
    if (srcPort == INVALID_ID || dstPort == INVALID_ID) {
      plan.diagnostics = "tech-mapping lost an external port binding";
      return false;
    }

    std::string key = std::to_string(srcPort) + ":" + std::to_string(dstPort);
    auto found = dedupEdges.find(key);
    if (found != dedupEdges.end()) {
      plan.originalEdgeToContractedEdge[edgeId] = found->second;
      continue;
    }

    auto newEdge = std::make_unique<Edge>();
    newEdge->srcPort = srcPort;
    newEdge->dstPort = dstPort;
    newEdge->attributes = edge->attributes;
    IdIndex contractedEdgeId = contracted.addEdge(std::move(newEdge));
    contracted.ports[srcPort]->connectedEdges.push_back(contractedEdgeId);
    contracted.ports[dstPort]->connectedEdges.push_back(contractedEdgeId);
    dedupEdges[key] = contractedEdgeId;
    plan.originalEdgeToContractedEdge[edgeId] = contractedEdgeId;
  }

  return true;
}

bool TechMapper::expandPlanMapping(
    const Graph &originalDfg, const Graph &adg, const Plan &plan,
    const MappingState &contractedState, MappingState &expandedState,
    llvm::SmallVectorImpl<FUConfigSelection> &fuConfigs) {
  expandedState.init(originalDfg, adg);
  fuConfigs.clear();

  for (IdIndex swNodeId = 0;
       swNodeId < static_cast<IdIndex>(plan.originalNodeToContractedNode.size());
       ++swNodeId) {
    IdIndex contractedNodeId = plan.originalNodeToContractedNode[swNodeId];
    if (contractedNodeId == INVALID_ID ||
        contractedNodeId >= contractedState.swNodeToHwNode.size())
      continue;
    IdIndex hwNodeId = contractedState.swNodeToHwNode[contractedNodeId];
    if (hwNodeId == INVALID_ID)
      continue;
    expandedState.swNodeToHwNode[swNodeId] = hwNodeId;
    expandedState.hwNodeToSwNodes[hwNodeId].push_back(swNodeId);
  }

  for (IdIndex swPortId = 0;
       swPortId < static_cast<IdIndex>(plan.originalPortToContractedPort.size());
       ++swPortId) {
    IdIndex contractedPortId = plan.originalPortToContractedPort[swPortId];
    if (contractedPortId == INVALID_ID ||
        contractedPortId >= contractedState.swPortToHwPort.size())
      continue;
    IdIndex hwPortId = contractedState.swPortToHwPort[contractedPortId];
    if (hwPortId == INVALID_ID)
      continue;
    expandedState.swPortToHwPort[swPortId] = hwPortId;
    expandedState.hwPortToSwPorts[hwPortId].push_back(swPortId);
  }

  for (IdIndex swEdgeId = 0;
       swEdgeId < static_cast<IdIndex>(plan.originalEdgeToContractedEdge.size());
       ++swEdgeId) {
    if (plan.originalEdgeKinds[swEdgeId] == TechMappedEdgeKind::IntraFU)
      continue;
    IdIndex contractedEdgeId = plan.originalEdgeToContractedEdge[swEdgeId];
    if (contractedEdgeId == INVALID_ID ||
        contractedEdgeId >= contractedState.swEdgeToHwPaths.size())
      continue;
    expandedState.swEdgeToHwPaths[swEdgeId] =
        contractedState.swEdgeToHwPaths[contractedEdgeId];
    llvm::ArrayRef<IdIndex> path = expandedState.swEdgeToHwPaths[swEdgeId];
    for (size_t i = 0; i + 1 < path.size(); i += 2) {
      IdIndex outPort = path[i];
      IdIndex inPort = path[i + 1];
      const Port *hwOut = adg.getPort(outPort);
      if (!hwOut)
        continue;
      for (IdIndex hwEdgeId : hwOut->connectedEdges) {
        const Edge *hwEdge = adg.getEdge(hwEdgeId);
        if (!hwEdge)
          continue;
        if (hwEdge->srcPort == outPort && hwEdge->dstPort == inPort) {
          expandedState.hwEdgeToSwEdges[hwEdgeId].push_back(swEdgeId);
          break;
        }
      }
    }
  }

  for (const Unit &unit : plan.units) {
    if (unit.contractedNodeId == INVALID_ID ||
        unit.contractedNodeId >= contractedState.swNodeToHwNode.size())
      continue;
    IdIndex hwNodeId = contractedState.swNodeToHwNode[unit.contractedNodeId];
    if (hwNodeId == INVALID_ID)
      continue;
    for (const Candidate &candidate : unit.candidates) {
      if (candidate.hwNodeId != hwNodeId || candidate.configFields.empty())
        continue;
      FUConfigSelection selection;
      selection.hwNodeId = hwNodeId;
      if (const Node *hwNode = adg.getNode(hwNodeId)) {
        selection.hwName = getNodeAttrStr(hwNode, "op_name").str();
        selection.peName = getNodeAttrStr(hwNode, "pe_name").str();
      }
      selection.swNodeIds.append(unit.swNodes.begin(), unit.swNodes.end());
      selection.fields.append(candidate.configFields.begin(),
                              candidate.configFields.end());
      fuConfigs.push_back(std::move(selection));
      break;
    }
  }

  return true;
}

} // namespace fcc
