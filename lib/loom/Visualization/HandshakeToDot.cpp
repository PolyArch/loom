//===- HandshakeToDot.cpp - Handshake MLIR to DOT Export -----------------===//
//
// Exports handshake MLIR functions to Graphviz DOT format by directly walking
// SSA values and operations — no intermediate graph representation needed.
//
//===----------------------------------------------------------------------===//

#include "loom/Visualization/HandshakeToDot.h"
#include "loom/Visualization/DotStyleConfig.h"

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <sstream>

namespace loom {
namespace visualization {

namespace {

//===----------------------------------------------------------------------===//
// Label Helpers
//===----------------------------------------------------------------------===//

static std::string getTypeString(mlir::Type type) {
  std::string s;
  llvm::raw_string_ostream os(s);
  type.print(os);
  return s;
}

/// Returns true when \p type represents a handshake control token (none type).
static bool isControlType(mlir::Type type) {
  return mlir::isa<mlir::NoneType>(type);
}

/// Generate "dialect\nopName\n(id:N)" label for a regular operation.
static std::string getDialectOpLabel(llvm::StringRef fullName, unsigned id) {
  std::string s;
  llvm::raw_string_ostream os(s);
  size_t dotPos = fullName.find('.');
  if (dotPos != llvm::StringRef::npos) {
    os << fullName.substr(0, dotPos) << "\\n"
       << fullName.substr(dotPos + 1) << "\\n"
       << "(id:" << id << ")";
  } else {
    os << fullName << "\\n(id:" << id << ")";
  }
  return s;
}

/// Generate the display label for an operation node.
static std::string getOpNodeLabel(mlir::Operation *op, unsigned id) {
  llvm::StringRef fullName = op->getName().getStringRef();

  // handshake.fork: compact "f:<id>"
  if (mlir::isa<circt::handshake::ForkOp>(op))
    return "f:" + std::to_string(id);

  // handshake.constant: show the constant value
  if (auto constOp = mlir::dyn_cast<circt::handshake::ConstantOp>(op)) {
    if (auto valueAttr = constOp.getValueAttr()) {
      std::string valStr;
      llvm::raw_string_ostream valOS(valStr);
      valueAttr.print(valOS);
      valOS.flush();
      return valStr + " (const)\\n(id:" + std::to_string(id) + ")";
    }
  }

  // arith.constant: show numeric value on first line, type:id on second
  if (auto arithConst = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
    mlir::Attribute valueAttr = arithConst.getValue();
    std::string s;
    llvm::raw_string_ostream os(s);
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(valueAttr)) {
      os << intAttr.getValue() << "\\n";
    } else if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(valueAttr)) {
      os << floatAttr.getValueAsDouble() << "\\n";
    } else {
      valueAttr.print(os);
      os << "\\n";
    }
    std::string typeStr;
    llvm::raw_string_ostream typeOS(typeStr);
    arithConst.getType().print(typeOS);
    typeOS.flush();
    os << typeStr << ":" << id;
    return s;
  }

  return getDialectOpLabel(fullName, id);
}

//===----------------------------------------------------------------------===//
// Node Emission
//===----------------------------------------------------------------------===//

static void emitNode(llvm::raw_ostream &os, unsigned id,
                     const std::string &label,
                     const DotNodeStyle &style,
                     llvm::StringRef indent = "  ") {
  os << indent << "n" << id << " [label=\"" << label << "\", ";
  os << "shape=" << style.shape << ", ";
  if (style.borderStyle == "dashed") {
    os << "style=\"filled," << style.borderStyle << "\", ";
  } else {
    os << "style=filled, ";
  }
  os << "fillcolor=\"" << style.fillColor << "\", ";
  os << "fontcolor=\"" << style.fontColor << "\"";
  if (!style.extraAttrs.empty())
    os << ", " << style.extraAttrs;
  os << "];\n";
}

static void emitEdge(llvm::raw_ostream &os, unsigned srcId, unsigned dstId,
                     bool isControl, llvm::StringRef indent = "  ") {
  os << indent << "n" << srcId << " -> n" << dstId << " [";
  if (isControl) {
    os << "style=dashed, color=blue, penwidth=1";
  } else {
    os << "style=solid, color=black, penwidth=1";
  }
  os << "];\n";
}

//===----------------------------------------------------------------------===//
// Single-function DOT export
//===----------------------------------------------------------------------===//

static void exportFuncToDot(circt::handshake::FuncOp func,
                             llvm::raw_ostream &os,
                             llvm::StringRef nodePrefix = "",
                             llvm::StringRef indent = "  ") {
  // Map from SSA value → node ID.
  llvm::DenseMap<mlir::Value, unsigned> valueToNode;
  unsigned nextId = 0;

  // Collect input / output node IDs for rank constraints (single-func only).
  llvm::SmallVector<unsigned, 8> inputNodeIds;
  llvm::SmallVector<unsigned, 8> outputNodeIds;

  mlir::Block &entryBlock = func.getBody().front();

  // --- Input nodes (block arguments) ---
  os << indent << "// Input nodes\n";
  for (auto [argIdx, arg] : llvm::enumerate(entryBlock.getArguments())) {
    unsigned id = nextId++;
    valueToNode[arg] = id;
    inputNodeIds.push_back(id);

    std::string label = "input" + std::to_string(argIdx) + "_" +
                        getTypeString(arg.getType());
    DotNodeStyle style =
        DotStyleConfig::getNodeStyle("", /*isMapped=*/true,
                                     /*isFuncArg=*/true, /*isReturn=*/false);
    std::string fullId = nodePrefix.str() + "n" + std::to_string(id);
    os << indent << fullId << " [label=\"" << label << "\", ";
    os << "shape=" << style.shape << ", ";
    os << "style=filled, ";
    os << "fillcolor=\"" << style.fillColor << "\", ";
    os << "fontcolor=\"" << style.fontColor << "\"";
    os << "];\n";
  }

  // Collect operations; identify the return op separately.
  mlir::Operation *returnOp = nullptr;
  llvm::SmallVector<mlir::Operation *, 32> regularOps;

  for (mlir::Operation &op : entryBlock.getOperations()) {
    if (mlir::isa<circt::handshake::ReturnOp>(op)) {
      returnOp = &op;
    } else {
      regularOps.push_back(&op);
    }
  }

  // --- Operation nodes ---
  // Map each operation to a node ID (using its first result or just an ID).
  llvm::DenseMap<mlir::Operation *, unsigned> opToNode;

  os << "\n" << indent << "// Operation nodes\n";
  for (mlir::Operation *op : regularOps) {
    unsigned id = nextId++;
    opToNode[op] = id;

    // All results of this op map to the same node.
    for (mlir::Value result : op->getResults())
      valueToNode[result] = id;

    std::string label = getOpNodeLabel(op, id);
    llvm::StringRef opName = op->getName().getStringRef();
    DotNodeStyle style = DotStyleConfig::getNodeStyle(opName);

    std::string fullId = nodePrefix.str() + "n" + std::to_string(id);
    os << indent << fullId << " [label=\"" << label << "\", ";
    os << "shape=" << style.shape << ", ";
    if (style.borderStyle == "dashed") {
      os << "style=\"filled," << style.borderStyle << "\", ";
    } else {
      os << "style=filled, ";
    }
    os << "fillcolor=\"" << style.fillColor << "\", ";
    os << "fontcolor=\"" << style.fontColor << "\"";
    if (!style.extraAttrs.empty())
      os << ", " << style.extraAttrs;
    os << "];\n";
  }

  // --- Output nodes (from return op) ---
  if (returnOp) {
    os << "\n" << indent << "// Output nodes\n";
    for (auto [retIdx, operand] :
         llvm::enumerate(returnOp->getOperands())) {
      unsigned id = nextId++;
      outputNodeIds.push_back(id);

      std::string label = "output" + std::to_string(retIdx) + "_" +
                          getTypeString(operand.getType());
      DotNodeStyle style =
          DotStyleConfig::getNodeStyle("", /*isMapped=*/true,
                                       /*isFuncArg=*/false, /*isReturn=*/true);

      std::string fullId = nodePrefix.str() + "n" + std::to_string(id);
      os << indent << fullId << " [label=\"" << label << "\", ";
      os << "shape=" << style.shape << ", ";
      os << "style=filled, ";
      os << "fillcolor=\"" << style.fillColor << "\", ";
      os << "fontcolor=\"" << style.fontColor << "\"";
      os << "];\n";

      // Edge: defining value → output node
      auto defIt = valueToNode.find(operand);
      if (defIt != valueToNode.end()) {
        bool ctrl = isControlType(operand.getType());
        os << indent << nodePrefix << "n" << defIt->second
           << " -> " << nodePrefix << "n" << id << " [";
        if (ctrl)
          os << "style=dashed, color=blue, penwidth=1";
        else
          os << "style=solid, color=black, penwidth=1";
        os << "];\n";
      }
    }
  }

  // --- Edges for regular operations ---
  os << "\n" << indent << "// Edges\n";
  for (mlir::Operation *op : regularOps) {
    unsigned dstId = opToNode[op];
    for (mlir::Value operand : op->getOperands()) {
      auto defIt = valueToNode.find(operand);
      if (defIt == valueToNode.end())
        continue;
      unsigned srcId = defIt->second;
      bool ctrl = isControlType(operand.getType());
      os << indent << nodePrefix << "n" << srcId
         << " -> " << nodePrefix << "n" << dstId << " [";
      if (ctrl)
        os << "style=dashed, color=blue, penwidth=1";
      else
        os << "style=solid, color=black, penwidth=1";
      os << "];\n";
    }
  }

  // --- Rank constraints (single-function only, when no prefix) ---
  if (nodePrefix.empty()) {
    if (!inputNodeIds.empty()) {
      os << "\n" << indent << "// Rank constraints\n";
      os << indent << "{rank=min;";
      for (unsigned id : inputNodeIds)
        os << " n" << id << ";";
      os << "}\n";
    }
    if (!outputNodeIds.empty()) {
      os << indent << "{rank=max;";
      for (unsigned id : outputNodeIds)
        os << " n" << id << ";";
      os << "}\n";
    }
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void exportModuleToDot(mlir::ModuleOp module, llvm::raw_ostream &os) {
  llvm::SmallVector<circt::handshake::FuncOp, 4> funcs;
  module.walk([&](circt::handshake::FuncOp func) {
    funcs.push_back(func);
  });

  if (funcs.empty())
    return;

  if (funcs.size() == 1) {
    // Single function: output as the main digraph.
    circt::handshake::FuncOp func = funcs[0];
    os << "digraph HandshakeGraph {\n";
    os << "  rankdir=TB;\n";
    os << "  node [fontname=\"Courier\"];\n";
    os << "  edge [fontname=\"Courier\"];\n\n";
    os << "  label=\"Function: " << func.getName() << "\";\n";
    os << "  labelloc=t;\n\n";
    exportFuncToDot(func, os, /*nodePrefix=*/"", /*indent=*/"  ");
    os << "}\n";
    return;
  }

  // Multiple functions: wrap each in a subgraph cluster.
  os << "digraph HandshakeGraph {\n";
  os << "  rankdir=TB;\n";
  os << "  node [fontname=\"Courier\"];\n";
  os << "  edge [fontname=\"Courier\"];\n\n";
  os << "  label=\"Module with " << funcs.size() << " functions\";\n";
  os << "  labelloc=t;\n\n";

  for (auto [idx, func] : llvm::enumerate(funcs)) {
    os << "  subgraph cluster_" << idx << " {\n";
    os << "    label=\"Function: " << func.getName() << "\";\n";
    os << "    style=filled;\n";
    os << "    color=lightgrey;\n\n";

    std::string prefix = "f" + std::to_string(idx) + "_";
    exportFuncToDot(func, os, prefix, "    ");

    os << "  }\n\n";
  }

  os << "}\n";
}

} // namespace visualization
} // namespace loom
