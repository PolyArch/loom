#include "fcc/Viz/VizExporter.h"

#include "VizAssets.h"

#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace fcc {

// ---- helpers ----

static std::string jsonEsc(llvm::StringRef s) {
  std::string r;
  r.reserve(s.size() + 4);
  for (char c : s) {
    if (c == '"') r += "\\\"";
    else if (c == '\\') r += "\\\\";
    else if (c == '\n') r += "\\n";
    else r += c;
  }
  return r;
}

static std::string htmlEsc(llvm::StringRef s) {
  std::string r;
  for (char c : s) {
    if (c == '<') r += "&lt;";
    else if (c == '>') r += "&gt;";
    else if (c == '&') r += "&amp;";
    else if (c == '"') r += "&quot;";
    else r += c;
  }
  return r;
}

static std::string scriptSafe(const std::string &s) {
  std::string r;
  r.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '<' && i + 1 < s.size() && s[i + 1] == '/') {
      r += "<\\/";
      ++i;
    } else {
      r += s[i];
    }
  }
  return r;
}

static std::string printType(mlir::Type type) {
  if (!type) return "";
  std::string s;
  llvm::raw_string_ostream os(s);
  type.print(os);
  return s;
}

static std::string dfgEdgeType(mlir::Type type) {
  if (!type) return "data";
  if (mlir::isa<mlir::MemRefType>(type)) return "memref";
  if (mlir::isa<mlir::NoneType>(type)) return "control";
  return "data";
}

static std::string dfgOperandName(mlir::Operation *op, unsigned idx) {
  if (auto load = mlir::dyn_cast<circt::handshake::LoadOp>(op))
    return load.getOperandName(idx);
  if (auto store = mlir::dyn_cast<circt::handshake::StoreOp>(op))
    return store.getOperandName(idx);
  if (auto memory = mlir::dyn_cast<circt::handshake::MemoryOp>(op))
    return memory.getOperandName(idx);
  if (auto ext = mlir::dyn_cast<circt::handshake::ExternalMemoryOp>(op))
    return ext.getOperandName(idx);
  if (auto mux = mlir::dyn_cast<circt::handshake::MuxOp>(op))
    return mux.getOperandName(idx);
  if (auto cbr = mlir::dyn_cast<circt::handshake::ConditionalBranchOp>(op))
    return cbr.getOperandName(idx);
  if (auto constant = mlir::dyn_cast<circt::handshake::ConstantOp>(op))
    return constant.getOperandName(idx);
  return ("I" + std::to_string(idx));
}

static std::string dfgResultName(mlir::Operation *op, unsigned idx) {
  if (auto load = mlir::dyn_cast<circt::handshake::LoadOp>(op))
    return load.getResultName(idx);
  if (auto store = mlir::dyn_cast<circt::handshake::StoreOp>(op))
    return store.getResultName(idx);
  if (auto memory = mlir::dyn_cast<circt::handshake::MemoryOp>(op))
    return memory.getResultName(idx);
  if (auto ext = mlir::dyn_cast<circt::handshake::ExternalMemoryOp>(op))
    return ext.getResultName(idx);
  if (auto cbr = mlir::dyn_cast<circt::handshake::ConditionalBranchOp>(op))
    return cbr.getResultName(idx);
  if (auto ctrlMerge =
          mlir::dyn_cast<circt::handshake::ControlMergeOp>(op))
    return ctrlMerge.getResultName(idx);
  return ("O" + std::to_string(idx));
}

static std::string getADGName(mlir::ModuleOp topModule) {
  if (!topModule) return "";
  fcc::fabric::ModuleOp fabricMod;
  topModule->walk([&](fcc::fabric::ModuleOp mod) { fabricMod = mod; });
  if (!fabricMod) return "";
  return fabricMod.getSymName().str();
}

static std::string getDFGName(mlir::ModuleOp topModule) {
  if (!topModule) return "";
  circt::handshake::FuncOp funcOp;
  topModule->walk([&](circt::handshake::FuncOp func) {
    if (!funcOp) funcOp = func;
  });
  if (!funcOp) return "";
  return funcOp.getName().str();
}

static std::string makeVizTitle(mlir::ModuleOp adgModule,
                                mlir::ModuleOp dfgModule,
                                bool hasMapping) {
  std::string adgName = getADGName(adgModule);
  std::string dfgName = getDFGName(dfgModule);
  if (!dfgName.empty() && !adgName.empty())
    return dfgName + (hasMapping ? " on " : " and ") + adgName;
  if (!dfgName.empty()) return dfgName;
  if (!adgName.empty()) return adgName;
  return hasMapping ? "fcc viz (mapped)" : "fcc viz";
}

// ---- Serialize fabric.module to JSON ----

static void writeADGJson(llvm::raw_ostream &os, mlir::ModuleOp topModule,
                          mlir::MLIRContext *ctx) {
  // Find fabric.module
  fcc::fabric::ModuleOp fabricMod;
  topModule->walk([&](fcc::fabric::ModuleOp mod) { fabricMod = mod; });

  os << "{\n";

  if (!fabricMod) {
    os << "  \"module\": null\n}";
    return;
  }

  // Module signature
  auto fnType = fabricMod.getFunctionType();
  os << "  \"name\": \"" << jsonEsc(fabricMod.getSymName().str())
     << "\",\n";
  os << "  \"numInputs\": " << fnType.getNumInputs()
     << ", \"numOutputs\": " << fnType.getNumResults() << ",\n";

  llvm::DenseMap<mlir::Block *, llvm::DenseSet<llvm::StringRef>>
      referencedTargetsByBlock;
  topModule.walk([&](fcc::fabric::InstanceOp instOp) {
    referencedTargetsByBlock[instOp->getBlock()].insert(instOp.getModule());
  });

  auto isDefinitionOp = [&](mlir::Operation *op,
                            llvm::StringRef name) -> bool {
    if (mlir::isa<fcc::fabric::FunctionUnitOp>(op))
      return true;
    if (!mlir::isa<fcc::fabric::SpatialPEOp, fcc::fabric::TemporalPEOp,
                   fcc::fabric::SpatialSwOp, fcc::fabric::TemporalSwOp,
                   fcc::fabric::ExtMemoryOp, fcc::fabric::MemoryOp>(op)) {
      return false;
    }
    return !op->hasAttr("inline_instantiation");
  };

  // Collect definition symbols from both top-level module and fabric.module.
  llvm::StringMap<fcc::fabric::SpatialPEOp> peDefMap;
  llvm::StringMap<fcc::fabric::TemporalPEOp> temporalPeDefMap;
  llvm::StringMap<fcc::fabric::SpatialSwOp> swDefMap;
  llvm::StringMap<fcc::fabric::TemporalSwOp> temporalSwDefMap;
  llvm::StringMap<fcc::fabric::ExtMemoryOp> extMemoryDefMap;
  llvm::StringMap<fcc::fabric::MemoryOp> memoryDefMap;
  llvm::StringMap<fcc::fabric::FunctionUnitOp> functionUnitDefMap;
  topModule->walk([&](fcc::fabric::SpatialPEOp peOp) {
    if (auto nameAttr = peOp.getSymNameAttr();
        nameAttr && isDefinitionOp(peOp.getOperation(), nameAttr.getValue()))
      peDefMap[nameAttr.getValue()] = peOp;
  });
  topModule->walk([&](fcc::fabric::TemporalPEOp peOp) {
    if (auto nameAttr = peOp.getSymNameAttr();
        nameAttr && isDefinitionOp(peOp.getOperation(), nameAttr.getValue()))
      temporalPeDefMap[nameAttr.getValue()] = peOp;
  });
  topModule->walk([&](fcc::fabric::SpatialSwOp swOp) {
    if (auto nameAttr = swOp.getSymNameAttr();
        nameAttr && isDefinitionOp(swOp.getOperation(), nameAttr.getValue()))
      swDefMap[nameAttr.getValue()] = swOp;
  });
  topModule->walk([&](fcc::fabric::TemporalSwOp swOp) {
    if (auto nameAttr = swOp.getSymNameAttr();
        nameAttr && isDefinitionOp(swOp.getOperation(), nameAttr.getValue()))
      temporalSwDefMap[nameAttr.getValue()] = swOp;
  });
  topModule->walk([&](fcc::fabric::ExtMemoryOp extOp) {
    if (auto nameAttr = extOp.getSymNameAttr();
        nameAttr && isDefinitionOp(extOp.getOperation(), nameAttr.getValue()))
      extMemoryDefMap[nameAttr.getValue()] = extOp;
  });
  topModule->walk([&](fcc::fabric::MemoryOp memOp) {
    if (auto nameAttr = memOp.getSymNameAttr();
        nameAttr && isDefinitionOp(memOp.getOperation(), nameAttr.getValue()))
      memoryDefMap[nameAttr.getValue()] = memOp;
  });
  topModule->walk([&](fcc::fabric::FunctionUnitOp fuOp) {
    auto symName = fuOp.getSymNameAttr().getValue();
    if (isDefinitionOp(fuOp.getOperation(), symName))
      functionUnitDefMap[symName] = fuOp;
  });

  llvm::DenseMap<mlir::Operation *, std::string> renderNameMap;
  unsigned extMemCount = 0;
  unsigned memoryCount = 0;
  unsigned temporalSwCount = 0;
  unsigned addTagCount = 0;
  unsigned delTagCount = 0;
  unsigned mapTagCount = 0;
  unsigned fifoCount = 0;

  auto getRenderName = [&](mlir::Operation *op) -> std::string {
    auto it = renderNameMap.find(op);
    if (it != renderNameMap.end())
      return it->second;

    std::string name;
    if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
      name = instOp.getSymName().value_or("inst").str();
    } else if (auto peOp = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
      name = peOp.getSymName().value_or("pe").str();
    } else if (auto swOp = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op)) {
      name = swOp.getSymName().value_or("sw").str();
    } else if (auto tswOp = mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op)) {
      if (auto symAttr = tswOp.getSymNameAttr())
        name = symAttr.getValue().str();
      else
        name = "temporal_sw_" + std::to_string(temporalSwCount++);
    } else if (auto extOp = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op)) {
      if (auto symAttr = extOp.getSymNameAttr())
        name = symAttr.getValue().str();
      else
        name = "extmemory_" + std::to_string(extMemCount++);
    } else if (auto memOp = mlir::dyn_cast<fcc::fabric::MemoryOp>(op)) {
      if (auto symAttr = memOp.getSymNameAttr())
        name = symAttr.getValue().str();
      else
        name = "memory_" + std::to_string(memoryCount++);
    } else if (mlir::isa<fcc::fabric::AddTagOp>(op)) {
      name = "add_tag_" + std::to_string(addTagCount++);
    } else if (mlir::isa<fcc::fabric::DelTagOp>(op)) {
      name = "del_tag_" + std::to_string(delTagCount++);
    } else if (mlir::isa<fcc::fabric::MapTagOp>(op)) {
      name = "map_tag_" + std::to_string(mapTagCount++);
    } else if (auto fifoOp = mlir::dyn_cast<fcc::fabric::FifoOp>(op)) {
      if (auto symAttr = fifoOp.getSymNameAttr())
        name = symAttr.getValue().str();
      else
        name = "fifo_" + std::to_string(fifoCount++);
    } else {
      name = op->getName().getStringRef().str();
    }

    renderNameMap[op] = name;
    return name;
  };

  // Helper: emit FU details for a PE definition.
  // Extracts full SSA connectivity with operand/result order:
  // inputEdges (arg->op), edges (op->op), outputEdges (op/arg->yield output).
  auto emitPEFUs = [&](auto peOp) {
    os << ", \"fus\": [";
    bool firstFU = true;
    auto &peBody = peOp.getBody().front();
    auto referencedIt = referencedTargetsByBlock.find(&peBody);
    const llvm::DenseSet<llvm::StringRef> *referencedTargets =
        referencedIt != referencedTargetsByBlock.end() ? &referencedIt->second
                                                       : nullptr;
    for (auto &innerOp : peBody.getOperations()) {
      fcc::fabric::FunctionUnitOp fuOp;
      std::string fuName;
      if (auto directFu = mlir::dyn_cast<fcc::fabric::FunctionUnitOp>(innerOp)) {
        llvm::StringRef symName = directFu.getSymNameAttr().getValue();
        if (!symName.empty() && referencedTargets &&
            referencedTargets->contains(symName))
          continue;
        fuOp = directFu;
        fuName = directFu.getSymName().str();
      } else if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(innerOp)) {
        auto fuIt = functionUnitDefMap.find(instOp.getModule());
        if (fuIt == functionUnitDefMap.end())
          continue;
        fuOp = fuIt->second;
        fuName = instOp.getSymName().value_or(instOp.getModule()).str();
      } else {
        continue;
      }

      if (!firstFU) os << ", ";
      firstFU = false;

      auto fuFnType = fuOp.getFunctionType();
      os << "{\"name\": \"" << jsonEsc(fuName) << "\"";
      os << ", \"numIn\": " << fuFnType.getNumInputs();
      os << ", \"numOut\": " << fuFnType.getNumResults();

      struct ValueRef {
        int owner = -1;      // block args are negative (-1 - argIdx), ops are 0+
        int resultIdx = 0;   // valid only when owner >= 0
      };
      struct InputEdgeRef {
        int argIdx = -1;
        int dstOp = -1;
        int dstOperand = -1;
      };
      struct DagEdgeRef {
        int srcOp = -1;
        int dstOp = -1;
        int srcResult = 0;
        int dstOperand = -1;
      };
      struct OutputEdgeRef {
        int srcOwner = -1;   // arg or op, same encoding as ValueRef::owner
        int yieldIdx = -1;
        int srcResult = 0;
      };

      llvm::DenseMap<mlir::Value, ValueRef> valToRef;
      for (auto arg : fuOp.getBody().front().getArguments())
        valToRef[arg] = {-1 - static_cast<int>(arg.getArgNumber()), 0};

      llvm::SmallVector<DagEdgeRef, 4> dagEdges;        // op -> op
      llvm::SmallVector<InputEdgeRef, 4> inputEdges;    // argIdx -> opIdx
      llvm::SmallVector<OutputEdgeRef, 4> outputEdges;  // valIdx -> yieldIdx

      os << ", \"ops\": [";
      bool firstOp = true;
      int opIdx = 0;

      for (auto &bodyOp : fuOp.getBody().front().getOperations()) {
        if (auto yieldOp = mlir::dyn_cast<fcc::fabric::YieldOp>(bodyOp)) {
          // Track yield operands -> output connections
          for (unsigned yi = 0; yi < yieldOp.getNumOperands(); ++yi) {
            auto it = valToRef.find(yieldOp.getOperand(yi));
            if (it != valToRef.end()) {
              outputEdges.push_back(
                  {it->second.owner, static_cast<int>(yi), it->second.resultIdx});
            }
          }
          continue;
        }

        if (!firstOp) os << ", ";
        firstOp = false;
        os << "\"" << jsonEsc(bodyOp.getName().getStringRef().str()) << "\"";

        // Track operand sources
        for (unsigned operandIdx = 0; operandIdx < bodyOp.getNumOperands();
             ++operandIdx) {
          auto operand = bodyOp.getOperand(operandIdx);
          auto it = valToRef.find(operand);
          if (it != valToRef.end()) {
            if (it->second.owner >= 0) {
              dagEdges.push_back({it->second.owner, opIdx,
                                  it->second.resultIdx,
                                  static_cast<int>(operandIdx)});  // op -> op
            }
            else
              inputEdges.push_back({-(it->second.owner + 1), opIdx,
                                    static_cast<int>(operandIdx)}); // arg -> op
          }
        }
        for (unsigned resultIdx = 0; resultIdx < bodyOp.getNumResults();
             ++resultIdx)
          valToRef[bodyOp.getResult(resultIdx)] = {opIdx, static_cast<int>(resultIdx)};
        opIdx++;
      }
      os << "]";

      // op-to-op edges
      if (!dagEdges.empty()) {
        os << ", \"edges\": [";
        for (size_t k = 0; k < dagEdges.size(); ++k) {
          if (k > 0) os << ", ";
          os << "[" << dagEdges[k].srcOp << ", " << dagEdges[k].dstOp << ", "
             << dagEdges[k].srcResult << ", " << dagEdges[k].dstOperand << "]";
        }
        os << "]";
      }

      // input arg -> op edges
      if (!inputEdges.empty()) {
        os << ", \"inputEdges\": [";
        for (size_t k = 0; k < inputEdges.size(); ++k) {
          if (k > 0) os << ", ";
          os << "[" << inputEdges[k].argIdx << ", " << inputEdges[k].dstOp
             << ", " << inputEdges[k].dstOperand << "]";
        }
        os << "]";
      }

      // op/arg -> yield output edges
      if (!outputEdges.empty()) {
        os << ", \"outputEdges\": [";
        for (size_t k = 0; k < outputEdges.size(); ++k) {
          if (k > 0) os << ", ";
          os << "[" << outputEdges[k].srcOwner << ", " << outputEdges[k].yieldIdx
             << ", " << outputEdges[k].srcResult << "]";
        }
        os << "]";
      }

      // Generate DOT: full FU internal DAG with I/O port nodes.
      // Graphviz handles ALL layout including port-to-op connections.
      if (opIdx > 0) {
        os << ", \"dot\": \"digraph FU {\\n";
        os << "  rankdir=TB;\\n";
        os << "  bgcolor=\\\"transparent\\\";\\n";
        os << "  node [style=filled, fontsize=9, "
           << "fontname=\\\"monospace\\\", fontcolor=\\\"#c8d6e5\\\"];\\n";
        os << "  edge [color=\\\"#6a8faf\\\", penwidth=1.2, "
           << "arrowsize=0.6];\\n";
        // Input port nodes
        for (unsigned ai = 0; ai < fuFnType.getNumInputs(); ++ai) {
          os << "  in" << ai
             << " [label=\\\"I" << ai << "\\\", shape=square, "
             << "width=0.2, height=0.2, fontsize=7, "
             << "fillcolor=\\\"#2a5f5f\\\", color=\\\"#4ecdc4\\\"];\\n";
        }
        // Op nodes
        {
          int oi2 = 0;
          for (auto &bodyOp2 : fuOp.getBody().front().getOperations()) {
            if (mlir::isa<fcc::fabric::YieldOp>(bodyOp2)) continue;
            std::string on = bodyOp2.getName().getStringRef().str();
            bool isMux = (on.find("mux") != std::string::npos);
            std::string displayName = on;
            auto dotPos = on.find('.');
            if (dotPos != std::string::npos)
              displayName = on.substr(0, dotPos) + "\\n" + on.substr(dotPos + 1);
            os << "  op" << oi2
               << " [label=\\\"" << jsonEsc(displayName) << "\\\", ";
            if (isMux)
              os << "shape=invtrapezium, fillcolor=\\\"#3a3520\\\", "
                 << "color=\\\"#ffd166\\\", fontcolor=\\\"#ffd166\\\"];\\n";
            else
              os << "shape=ellipse, fillcolor=\\\"#1a3050\\\", "
                 << "color=\\\"#5dade2\\\"];\\n";
            oi2++;
          }
        }
        // Output port nodes
        for (unsigned yi = 0; yi < fuFnType.getNumResults(); ++yi) {
          os << "  out" << yi
             << " [label=\\\"O" << yi << "\\\", shape=square, "
             << "width=0.2, height=0.2, fontsize=7, "
             << "fillcolor=\\\"#5f2a1a\\\", color=\\\"#ff6b35\\\"];\\n";
        }
        // Rank constraints
        os << "  { rank=source; ";
        for (unsigned ai = 0; ai < fuFnType.getNumInputs(); ++ai)
          os << "in" << ai << "; ";
        os << "}\\n";
        os << "  { rank=sink; ";
        for (unsigned yi = 0; yi < fuFnType.getNumResults(); ++yi)
          os << "out" << yi << "; ";
        os << "}\\n";
        // Input -> op edges
        for (auto &ie : inputEdges)
          os << "  in" << ie.argIdx << " -> op" << ie.dstOp
             << " [color=\\\"#4ecdc4\\\"];\\n";
        // Op -> op edges
        for (auto &de : dagEdges)
          os << "  op" << de.srcOp << " -> op" << de.dstOp << ";\\n";
        // Op/arg -> output edges
        for (auto &oe : outputEdges) {
          if (oe.srcOwner >= 0)
            os << "  op" << oe.srcOwner << " -> out" << oe.yieldIdx
               << " [color=\\\"#ff6b35\\\"];\\n";
          else
            os << "  in" << (-(oe.srcOwner + 1)) << " -> out" << oe.yieldIdx
               << " [style=dashed, color=\\\"#888888\\\"];\\n";
        }
        os << "}\\n\"";
      }

      os << "}";
    }
    os << "]";
  };

  os << "  \"components\": [\n";
  bool first = true;
  auto &body = fabricMod.getBody().front();

  for (auto &op : body.getOperations()) {
    if (auto peOp = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
      if (!peOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = peOp.getSymNameAttr())
        symName = symNameAttr.getValue();
      if (!first) os << ",\n";
      first = false;
      auto peFnType = peOp.getFunctionType();
      os << "    {\"kind\": \"spatial_pe\", \"name\": \""
         << jsonEsc(getRenderName(peOp.getOperation())) << "\"";
      if (!symName.empty())
        os << ", \"defName\": \"" << jsonEsc(symName.str()) << "\"";
      os << ", \"numInputs\": " << peFnType.getNumInputs();
      os << ", \"numOutputs\": " << peFnType.getNumResults();
      emitPEFUs(peOp);
      os << "}";
      continue;
    }
    if (auto peOp = mlir::dyn_cast<fcc::fabric::TemporalPEOp>(op)) {
      if (!peOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = peOp.getSymNameAttr())
        symName = symNameAttr.getValue();
      if (!first) os << ",\n";
      first = false;
      auto peFnType = peOp.getFunctionType();
      os << "    {\"kind\": \"temporal_pe\", \"name\": \""
         << jsonEsc(getRenderName(peOp.getOperation())) << "\"";
      if (!symName.empty())
        os << ", \"defName\": \"" << jsonEsc(symName.str()) << "\"";
      os << ", \"numInputs\": " << peFnType.getNumInputs();
      os << ", \"numOutputs\": " << peFnType.getNumResults();
      emitPEFUs(peOp);
      os << "}";
      continue;
    }
    if (auto swOp = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op)) {
      if (!swOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = swOp.getSymNameAttr())
        symName = symNameAttr.getValue();
      if (!first) os << ",\n";
      first = false;
      auto swFnType = swOp.getFunctionType();
      os << "    {\"kind\": \"spatial_sw\", \"name\": \""
         << jsonEsc(getRenderName(swOp.getOperation())) << "\"";
      if (!symName.empty())
        os << ", \"defName\": \"" << jsonEsc(symName.str()) << "\"";
      os << ", \"numInputs\": " << swFnType.getNumInputs();
      os << ", \"numOutputs\": " << swFnType.getNumResults();
      os << "}";
      continue;
    }
    if (auto swOp = mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op)) {
      if (!swOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = swOp.getSymNameAttr())
        symName = symNameAttr.getValue();
      if (!first) os << ",\n";
      first = false;
      auto swFnType = swOp.getFunctionType();
      os << "    {\"kind\": \"temporal_sw\", \"name\": \""
         << jsonEsc(getRenderName(swOp.getOperation())) << "\"";
      if (!symName.empty())
        os << ", \"defName\": \"" << jsonEsc(symName.str()) << "\"";
      os << ", \"numInputs\": " << swFnType.getNumInputs();
      os << ", \"numOutputs\": " << swFnType.getNumResults();
      os << "}";
      continue;
    }

    if (auto extOp = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op)) {
      if (!extOp->hasAttr("inline_instantiation"))
        continue;
      if (!first) os << ",\n";
      first = false;
      auto memFnType = extOp.getFunctionType();
      os << "    {\"kind\": \"memory\", \"name\": \""
         << jsonEsc(getRenderName(extOp.getOperation())) << "\"";
      os << ", \"memoryKind\": \"extmemory\"";
      os << ", \"numInputs\": " << memFnType.getNumInputs();
      os << ", \"numOutputs\": " << memFnType.getNumResults();
      os << "}";
    }

    if (auto memOp = mlir::dyn_cast<fcc::fabric::MemoryOp>(op)) {
      if (!memOp->hasAttr("inline_instantiation"))
        continue;
      if (!first) os << ",\n";
      first = false;
      auto memFnType = memOp.getFunctionType();
      os << "    {\"kind\": \"memory\", \"name\": \""
         << jsonEsc(getRenderName(memOp.getOperation())) << "\"";
      os << ", \"memoryKind\": \"memory\"";
      os << ", \"numInputs\": " << memFnType.getNumInputs();
      os << ", \"numOutputs\": " << memFnType.getNumResults();
      os << "}";
    }

    if (auto addTagOp = mlir::dyn_cast<fcc::fabric::AddTagOp>(op)) {
      if (!first) os << ",\n";
      first = false;
      os << "    {\"kind\": \"add_tag\", \"name\": \""
         << jsonEsc(getRenderName(addTagOp.getOperation())) << "\"";
      os << ", \"numInputs\": 1, \"numOutputs\": 1}";
    }

    if (auto delTagOp = mlir::dyn_cast<fcc::fabric::DelTagOp>(op)) {
      if (!first) os << ",\n";
      first = false;
      os << "    {\"kind\": \"del_tag\", \"name\": \""
         << jsonEsc(getRenderName(delTagOp.getOperation())) << "\"";
      os << ", \"numInputs\": 1, \"numOutputs\": 1}";
    }

    if (auto mapTagOp = mlir::dyn_cast<fcc::fabric::MapTagOp>(op)) {
      if (!first) os << ",\n";
      first = false;
      os << "    {\"kind\": \"map_tag\", \"name\": \""
         << jsonEsc(getRenderName(mapTagOp.getOperation())) << "\"";
      os << ", \"numInputs\": 1, \"numOutputs\": 1}";
    }

    if (auto fifoOp = mlir::dyn_cast<fcc::fabric::FifoOp>(op)) {
      if (!fifoOp->hasAttr("inline_instantiation"))
        continue;
      if (!first) os << ",\n";
      first = false;
      auto fifoFnType = fifoOp.getFunctionType();
      os << "    {\"kind\": \"fifo\", \"name\": \""
         << jsonEsc(getRenderName(fifoOp.getOperation())) << "\"";
      os << ", \"numInputs\": " << fifoFnType.getNumInputs();
      os << ", \"numOutputs\": " << fifoFnType.getNumResults();
      os << "}";
    }

    // instances - resolve to PE/SW definitions for FU details
    if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
      if (!first) os << ",\n";
      first = false;

      auto moduleName = instOp.getModule();
      auto peIt = peDefMap.find(moduleName);
      auto temporalPeIt = temporalPeDefMap.find(moduleName);
      auto swIt = swDefMap.find(moduleName);

      if (peIt != peDefMap.end()) {
        // Instance of a spatial_pe - emit as PE with FU details
        auto peOp = peIt->second;
        auto peFnType = peOp.getFunctionType();
        os << "    {\"kind\": \"spatial_pe\", \"name\": \""
           << jsonEsc(instOp.getSymName().value_or("pe").str()) << "\"";
        os << ", \"defName\": \"" << jsonEsc(moduleName.str()) << "\"";
        os << ", \"numInputs\": " << peFnType.getNumInputs();
        os << ", \"numOutputs\": " << peFnType.getNumResults();
        emitPEFUs(peOp);
        os << "}";
      } else if (temporalPeIt != temporalPeDefMap.end()) {
        auto peOp = temporalPeIt->second;
        auto peFnType = peOp.getFunctionType();
        os << "    {\"kind\": \"temporal_pe\", \"name\": \""
           << jsonEsc(instOp.getSymName().value_or("tpe").str()) << "\"";
        os << ", \"defName\": \"" << jsonEsc(moduleName.str()) << "\"";
        os << ", \"numInputs\": " << peFnType.getNumInputs();
        os << ", \"numOutputs\": " << peFnType.getNumResults();
        emitPEFUs(peOp);
        os << "}";
      } else if (swIt != swDefMap.end()) {
        // Instance of a spatial_sw
        auto swOp = swIt->second;
        auto swFnType = swOp.getFunctionType();
        os << "    {\"kind\": \"spatial_sw\", \"name\": \""
           << jsonEsc(instOp.getSymName().value_or("sw").str()) << "\"";
        os << ", \"numInputs\": " << swFnType.getNumInputs();
        os << ", \"numOutputs\": " << swFnType.getNumResults();
        os << "}";
      } else if (auto tswIt = temporalSwDefMap.find(moduleName);
                 tswIt != temporalSwDefMap.end()) {
        auto tswOp = tswIt->second;
        auto tswFnType = tswOp.getFunctionType();
        os << "    {\"kind\": \"temporal_sw\", \"name\": \""
           << jsonEsc(getRenderName(instOp.getOperation())) << "\"";
        os << ", \"numInputs\": " << tswFnType.getNumInputs();
        os << ", \"numOutputs\": " << tswFnType.getNumResults();
        os << "}";
      } else {
        // Generic instance
        os << "    {\"kind\": \"instance\", \"name\": \""
           << jsonEsc(instOp.getSymName().value_or("inst").str()) << "\"";
        os << ", \"module\": \"" << jsonEsc(moduleName.str()) << "\"";
        os << ", \"numInputs\": " << instOp.getNumOperands();
        os << ", \"numOutputs\": " << instOp.getNumResults();
        os << "}";
      }
    }
  }

  os << "\n  ],\n";

  // Connections: trace SSA (two-pass for graph-region circular references).
  os << "  \"connections\": [\n";
  bool firstConn = true;

  llvm::DenseMap<mlir::Value, int> blockArgIdx;
  for (auto arg : body.getArguments())
    blockArgIdx[arg] = static_cast<int>(arg.getArgNumber());

  // Pass 1: collect all producer results first (handles forward references).
  struct ResultProducer {
    std::string name;
    unsigned idx;
  };
  llvm::DenseMap<mlir::Value, ResultProducer> resultProducerMap;
  for (auto &op : body.getOperations()) {
    if (mlir::isa<fcc::fabric::YieldOp>(op))
      continue;
    if (mlir::isa<fcc::fabric::SpatialPEOp, fcc::fabric::TemporalPEOp,
                  fcc::fabric::SpatialSwOp, fcc::fabric::TemporalSwOp>(op))
      continue;
    std::string renderName = getRenderName(&op);
    for (unsigned i = 0; i < op.getNumResults(); ++i)
      resultProducerMap[op.getResult(i)] = {renderName, i};
  }

  // Pass 2: trace all operand connections for renderable operations.
  for (auto &op : body.getOperations()) {
    if (mlir::isa<fcc::fabric::YieldOp>(op))
      continue;
    if (mlir::isa<fcc::fabric::SpatialPEOp, fcc::fabric::TemporalPEOp,
                  fcc::fabric::SpatialSwOp, fcc::fabric::TemporalSwOp>(op))
      continue;
    std::string opName = getRenderName(&op);

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto operand = op.getOperand(i);
      // Module input -> instance
      auto argIt = blockArgIdx.find(operand);
      if (argIt != blockArgIdx.end()) {
        if (!firstConn) os << ",\n";
        firstConn = false;
        os << "    {\"from\": \"module_in\", \"fromIdx\": " << argIt->second
           << ", \"to\": \"" << jsonEsc(opName) << "\", \"toIdx\": " << i
           << "}";
      }
      // Instance -> instance (now works for circular refs too)
      auto irIt = resultProducerMap.find(operand);
      if (irIt != resultProducerMap.end()) {
        if (!firstConn) os << ",\n";
        firstConn = false;
        os << "    {\"from\": \"" << jsonEsc(irIt->second.name)
           << "\", \"fromIdx\": " << irIt->second.idx
           << ", \"to\": \"" << jsonEsc(opName) << "\", \"toIdx\": " << i
           << "}";
      }
    }
  }

  // Module memref bindings and SW->ExtMem reverse links come from ExtMemory
  // metadata because the Fabric inline syntax does not spell them out as SSA
  // operands.
  for (auto &op : body.getOperations()) {
    auto extOp = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op);
    if (!extOp)
      continue;
    if (extOp.getNumOperands() > 0)
      continue;

    std::string memName = getRenderName(extOp.getOperation());
    if (auto argIdxAttr =
            extOp->getAttrOfType<mlir::IntegerAttr>("memref_arg_index")) {
      if (!firstConn)
        os << ",\n";
      firstConn = false;
      os << "    {\"from\": \"module_in\", \"fromIdx\": "
         << argIdxAttr.getInt() << ", \"to\": \"" << jsonEsc(memName)
         << "\", \"toIdx\": 0}";
    }

    auto emitExtMemBackEdges = [&](mlir::ArrayAttr detailAttr,
                                   bool detailed) {
      for (auto elem : detailAttr) {
        llvm::StringRef swName;
        int64_t outputBase = detailed ? 0 : 4;
        if (detailed) {
          auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(elem);
          if (!dictAttr)
            continue;
          auto nameAttr = dictAttr.getAs<mlir::StringAttr>("name");
          if (!nameAttr)
            continue;
          swName = nameAttr.getValue();
          if (auto outBaseAttr =
                  dictAttr.getAs<mlir::IntegerAttr>("output_port_base")) {
            outputBase = outBaseAttr.getInt();
          }
        } else {
          auto strAttr = mlir::dyn_cast<mlir::StringAttr>(elem);
          if (!strAttr)
            continue;
          swName = strAttr.getValue();
        }

        unsigned numDataInputs =
            extOp.getFunctionType().getNumInputs() > 0
                ? extOp.getFunctionType().getNumInputs() - 1
                : 0;
        for (unsigned p = 0; p < numDataInputs; ++p) {
          if (!firstConn)
            os << ",\n";
          firstConn = false;
          os << "    {\"from\": \"" << jsonEsc(swName.str())
             << "\", \"fromIdx\": " << (outputBase + static_cast<int64_t>(p))
             << ", \"to\": \"" << jsonEsc(memName) << "\", \"toIdx\": "
             << (1 + p) << "}";
        }
      }
    };

    if (auto detailAttr =
            extOp->getAttrOfType<mlir::ArrayAttr>("connected_sw_detail")) {
      emitExtMemBackEdges(detailAttr, /*detailed=*/true);
    } else if (auto connAttr =
                   extOp->getAttrOfType<mlir::ArrayAttr>("connected_sw")) {
      emitExtMemBackEdges(connAttr, /*detailed=*/false);
    }
  }

  // Yield: instance results -> module outputs
  auto yieldOp = mlir::dyn_cast<fcc::fabric::YieldOp>(body.getTerminator());
  if (yieldOp) {
    for (unsigned i = 0; i < yieldOp->getNumOperands(); ++i) {
      auto ir = resultProducerMap.find(yieldOp->getOperand(i));
      if (ir != resultProducerMap.end()) {
        if (!firstConn) os << ",\n";
        firstConn = false;
        os << "    {\"from\": \"" << jsonEsc(ir->second.name)
           << "\", \"fromIdx\": " << ir->second.idx
           << ", \"to\": \"module_out\", \"toIdx\": " << i << "}";
      }
    }
  }

  os << "\n  ]\n}";
}

// ---- Serialize handshake.func DFG to DOT + JSON ----

static void writeDFGJson(llvm::raw_ostream &os, mlir::ModuleOp topModule) {
  circt::handshake::FuncOp funcOp;
  topModule->walk([&](circt::handshake::FuncOp func) {
    if (!funcOp) funcOp = func;
  });

  os << "{\n";
  if (!funcOp) {
    os << "  \"dot\": null, \"nodes\": [], \"edges\": []\n}";
    return;
  }

  auto &body = funcOp.getBody().front();
  auto argNames = funcOp->getAttrOfType<mlir::ArrayAttr>("argNames");
  auto resNames = funcOp->getAttrOfType<mlir::ArrayAttr>("resNames");

  llvm::SmallVector<mlir::Operation *, 16> ops;
  circt::handshake::ReturnOp returnOp;
  for (auto &op : body.getOperations()) {
    if (auto ret = mlir::dyn_cast<circt::handshake::ReturnOp>(op)) {
      returnOp = ret;
      continue;
    }
    ops.push_back(&op);
  }

  llvm::DenseMap<mlir::Value, std::pair<int, unsigned>> valueToNodePort;

  os << "  \"func\": \"" << jsonEsc(funcOp.getName().str()) << "\",\n";
  os << "  \"nodes\": [\n";
  bool firstNode = true;
  int nodeId = 0;

  auto emitPortArray = [&](llvm::StringRef key, unsigned count, auto nameFn,
                           auto typeFn) {
    os << ", \"" << key << "\": [";
    for (unsigned i = 0; i < count; ++i) {
      if (i > 0) os << ", ";
      os << "{\"index\": " << i << ", \"name\": \""
         << jsonEsc(nameFn(i)) << "\", \"type\": \""
         << jsonEsc(typeFn(i)) << "\"}";
    }
    os << "]";
  };

  for (auto arg : body.getArguments()) {
    if (!firstNode) os << ",\n";
    firstNode = false;

    std::string argName;
    if (argNames && arg.getArgNumber() < argNames.size()) {
      if (auto str =
              mlir::dyn_cast<mlir::StringAttr>(argNames[arg.getArgNumber()]))
        argName = str.getValue().str();
    }
    std::string typeStr = printType(arg.getType());
    os << "    {\"id\": " << nodeId << ", \"kind\": \"input\""
       << ", \"label\": \"arg" << arg.getArgNumber() << "\""
       << ", \"arg_index\": " << arg.getArgNumber()
       << ", \"name\": \"" << jsonEsc(argName) << "\""
       << ", \"type\": \"" << jsonEsc(typeStr) << "\"";
    emitPortArray("inputs", 0,
                  [&](unsigned) { return std::string(); },
                  [&](unsigned) { return std::string(); });
    emitPortArray("outputs", 1,
                  [&](unsigned) {
                    return !argName.empty() ? argName : std::string("value");
                  },
                  [&](unsigned) { return typeStr; });
    os << "}";
    valueToNodePort[arg] = {nodeId, 0};
    nodeId++;
  }

  for (auto *op : ops) {
    if (!firstNode) os << ",\n";
    firstNode = false;

    std::string opName = op->getName().getStringRef().str();
    std::string displayName = opName;
    size_t dotPos = displayName.find('.');
    if (dotPos != std::string::npos)
      displayName =
          displayName.substr(0, dotPos) + "\n" + displayName.substr(dotPos + 1);

    os << "    {\"id\": " << nodeId << ", \"kind\": \"op\""
       << ", \"label\": \"" << jsonEsc(opName) << "\""
       << ", \"display\": \"" << jsonEsc(displayName) << "\""
       << ", \"op\": \"" << jsonEsc(opName) << "\"";
    emitPortArray(
        "inputs", op->getNumOperands(),
        [&](unsigned i) { return dfgOperandName(op, i); },
        [&](unsigned i) { return printType(op->getOperand(i).getType()); });
    emitPortArray(
        "outputs", op->getNumResults(),
        [&](unsigned i) { return dfgResultName(op, i); },
        [&](unsigned i) { return printType(op->getResult(i).getType()); });
    os << "}";

    for (unsigned i = 0; i < op->getNumResults(); ++i)
      valueToNodePort[op->getResult(i)] = {nodeId, i};
    nodeId++;
  }

  if (returnOp) {
    for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
      if (!firstNode) os << ",\n";
      firstNode = false;

      std::string resName;
      if (resNames && i < resNames.size()) {
        if (auto str = mlir::dyn_cast<mlir::StringAttr>(resNames[i]))
          resName = str.getValue().str();
      }
      std::string typeStr = printType(returnOp.getOperand(i).getType());
      os << "    {\"id\": " << nodeId << ", \"kind\": \"output\""
         << ", \"label\": \"return\""
         << ", \"result_index\": " << i
         << ", \"name\": \"" << jsonEsc(resName) << "\""
         << ", \"type\": \"" << jsonEsc(typeStr) << "\"";
      emitPortArray("inputs", 1,
                    [&](unsigned) {
                      return !resName.empty() ? resName : std::string("value");
                    },
                    [&](unsigned) { return typeStr; });
      emitPortArray("outputs", 0,
                    [&](unsigned) { return std::string(); },
                    [&](unsigned) { return std::string(); });
      os << "}";
      nodeId++;
    }
  }
  os << "\n  ],\n";

  llvm::DenseMap<mlir::Operation *, int> opToNodeId;
  int firstOpNodeId = static_cast<int>(body.getNumArguments());
  for (unsigned i = 0; i < ops.size(); ++i)
    opToNodeId[ops[i]] = firstOpNodeId + static_cast<int>(i);

  int firstOutputNodeId = firstOpNodeId + static_cast<int>(ops.size());
  os << "  \"edges\": [\n";
  bool firstEdge = true;
  int edgeId = 0;

  for (auto *op : ops) {
    int dstNodeId = opToNodeId[op];
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      auto it = valueToNodePort.find(op->getOperand(i));
      if (it == valueToNodePort.end())
        continue;
      if (!firstEdge) os << ",\n";
      firstEdge = false;
      std::string typeStr = printType(op->getOperand(i).getType());
      os << "    {\"id\": " << edgeId
         << ", \"from\": " << it->second.first
         << ", \"from_port\": " << it->second.second
         << ", \"to\": " << dstNodeId
         << ", \"to_port\": " << i
         << ", \"edge_type\": \"" << dfgEdgeType(op->getOperand(i).getType())
         << "\", \"value_type\": \"" << jsonEsc(typeStr) << "\"}";
      edgeId++;
    }
  }

  if (returnOp) {
    for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
      auto it = valueToNodePort.find(returnOp.getOperand(i));
      if (it == valueToNodePort.end())
        continue;
      if (!firstEdge) os << ",\n";
      firstEdge = false;
      std::string typeStr = printType(returnOp.getOperand(i).getType());
      os << "    {\"id\": " << edgeId
         << ", \"from\": " << it->second.first
         << ", \"from_port\": " << it->second.second
         << ", \"to\": " << (firstOutputNodeId + static_cast<int>(i))
         << ", \"to_port\": 0"
         << ", \"edge_type\": \""
         << dfgEdgeType(returnOp.getOperand(i).getType())
         << "\", \"value_type\": \"" << jsonEsc(typeStr) << "\"}";
      edgeId++;
    }
  }
  os << "\n  ],\n";

  os << "  \"dot\": null\n}";
}

// ---- Public API ----

mlir::LogicalResult exportVizOnly(const std::string &outputPath,
                                  mlir::ModuleOp adgModule,
                                  mlir::ModuleOp dfgModule,
                                  mlir::MLIRContext *ctx) {
  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "fcc viz: cannot open " << outputPath << "\n";
    return mlir::failure();
  }

  std::string title = makeVizTitle(adgModule, dfgModule, false);

  // Serialize data
  std::string adgJson;
  if (adgModule) {
    llvm::raw_string_ostream ss(adgJson);
    writeADGJson(ss, adgModule, ctx);
  } else {
    adgJson = "null";
  }

  std::string dfgJson;
  if (dfgModule) {
    llvm::raw_string_ostream ss(dfgJson);
    writeDFGJson(ss, dfgModule);
  } else {
    dfgJson = "null";
  }

  // Emit HTML
  out << "<!DOCTYPE html>\n<html>\n<head>\n"
      << "  <meta charset=\"UTF-8\">\n"
      << "  <title>" << htmlEsc(title) << "</title>\n"
      << "  <style>\n" << viz::RENDERER_CSS << "\n  </style>\n"
      << "</head>\n<body>\n\n";

  out << "<div id=\"toolbar\">\n"
      << "  <span id=\"title\">" << htmlEsc(title) << "</span>\n"
      << "  <button id=\"btn-fit\">Fit</button>\n"
      << "  <span id=\"status-bar\">Loading...</span>\n"
      << "</div>\n\n";

  out << "<div id=\"graph-area\">\n"
      << "  <div id=\"panel-adg\">\n"
      << "    <div class=\"panel-header\">Hardware (ADG)</div>\n"
      << "    <svg id=\"svg-adg\"></svg>\n"
      << "  </div>\n"
      << "  <div id=\"panel-divider\"></div>\n"
      << "  <div id=\"panel-dfg\">\n"
      << "    <div class=\"panel-header\">Software (DFG)</div>\n"
      << "    <svg id=\"svg-dfg\"></svg>\n"
      << "  </div>\n"
      << "</div>\n\n";

  // Embedded data
  out << "<script>\n"
      << "const ADG_DATA = " << scriptSafe(adgJson) << ";\n\n"
      << "const DFG_DATA = " << scriptSafe(dfgJson) << ";\n"
      << "const MAPPING_DATA = null;\n"
      << "</script>\n\n";

  // Bundled D3 for self-contained local viewing.
  out << "<script>\n" << viz::D3_MIN_JS << "\n</script>\n\n";

  // Renderer JS
  out << "<script>\n" << viz::RENDERER_JS << "\n</script>\n\n";

  out << "</body>\n</html>\n";

  return mlir::success();
}

mlir::LogicalResult exportVizWithMapping(const std::string &outputPath,
                                         mlir::ModuleOp adgModule,
                                         mlir::ModuleOp dfgModule,
                                         const std::string &mapJsonPath,
                                         mlir::MLIRContext *ctx) {
  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "fcc viz: cannot open " << outputPath << "\n";
    return mlir::failure();
  }

  // Read mapping JSON
  std::string mapJson = "null";
  auto mapBuf = llvm::MemoryBuffer::getFile(mapJsonPath);
  if (mapBuf)
    mapJson = (*mapBuf)->getBuffer().str();

  std::string adgJson;
  if (adgModule) {
    llvm::raw_string_ostream ss(adgJson);
    writeADGJson(ss, adgModule, ctx);
  } else {
    adgJson = "null";
  }

  std::string dfgJson;
  if (dfgModule) {
    llvm::raw_string_ostream ss(dfgJson);
    writeDFGJson(ss, dfgModule);
  } else {
    dfgJson = "null";
  }

  std::string title = makeVizTitle(adgModule, dfgModule, true);

  out << "<!DOCTYPE html>\n<html>\n<head>\n"
      << "  <meta charset=\"UTF-8\">\n"
      << "  <title>" << htmlEsc(title) << "</title>\n"
      << "  <style>\n" << viz::RENDERER_CSS << "\n  </style>\n"
      << "</head>\n<body>\n\n";

  out << "<div id=\"toolbar\">\n"
      << "  <span id=\"title\">" << htmlEsc(title) << "</span>\n"
      << "  <button id=\"btn-fit\">Fit</button>\n"
      << "  <span id=\"status-bar\">Loading...</span>\n"
      << "</div>\n\n";

  out << "<div id=\"graph-area\">\n"
      << "  <div id=\"panel-adg\">\n"
      << "    <div class=\"panel-header\">Hardware (ADG)</div>\n"
      << "    <svg id=\"svg-adg\"></svg>\n"
      << "  </div>\n"
      << "  <div id=\"panel-divider\"></div>\n"
      << "  <div id=\"panel-dfg\">\n"
      << "    <div class=\"panel-header\">Software (DFG)</div>\n"
      << "    <svg id=\"svg-dfg\"></svg>\n"
      << "  </div>\n"
      << "</div>\n\n";

  out << "<script>\n"
      << "const ADG_DATA = " << scriptSafe(adgJson) << ";\n\n"
      << "const DFG_DATA = " << scriptSafe(dfgJson) << ";\n\n"
      << "const MAPPING_DATA = " << scriptSafe(mapJson) << ";\n"
      << "</script>\n\n";

  out << "<script>\n" << viz::D3_MIN_JS << "\n</script>\n\n";
  out << "<script>\n" << viz::RENDERER_JS << "\n</script>\n\n";
  out << "</body>\n</html>\n";

  return mlir::success();
}

} // namespace fcc
