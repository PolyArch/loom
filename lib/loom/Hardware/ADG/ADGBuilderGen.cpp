//===-- ADGBuilderGen.cpp - MLIR text generation -----------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Generates Fabric MLIR text from ADGBuilder internal state.
//
//===----------------------------------------------------------------------===//

#include "loom/Hardware/ADG/ADGBuilderImpl.h"

#include <algorithm>
#include <queue>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Generation Helpers
//===----------------------------------------------------------------------===//

/// Emit latency and interval attributes to the output stream.
static void emitLatencyInterval(std::ostringstream &os, int16_t latMin,
                                int16_t latTyp, int16_t latMax, int16_t intMin,
                                int16_t intTyp, int16_t intMax) {
  os << "    [latency = [" << latMin << " : i16, " << latTyp << " : i16, "
     << latMax << " : i16]";
  os << ", interval = [" << intMin << " : i16, " << intTyp << " : i16, "
     << intMax << " : i16]";
}

/// Emit a comma-separated list of Type values as MLIR strings.
static void emitTypeList(std::ostringstream &os,
                         const std::vector<Type> &types) {
  for (size_t i = 0; i < types.size(); ++i) {
    if (i > 0) os << ", ";
    os << types[i].toMLIR();
  }
}

/// Flatten a 2D connectivity table into a 1D vector of 0/1 ints.
/// If the table is empty, returns a full-crossbar (all 1s) of size numOut*numIn.
static std::vector<int>
flattenConnectivity(const std::vector<std::vector<bool>> &table,
                    unsigned numOut, unsigned numIn) {
  if (table.empty())
    return std::vector<int>(numOut * numIn, 1);
  std::vector<int> flat;
  for (const auto &row : table)
    for (bool v : row)
      flat.push_back(v ? 1 : 0);
  return flat;
}

/// Extract the tag type string from a tagged MLIR type string.
/// For "!dataflow.tagged<i32, i4>", returns "i4".
static std::string extractTagType(const std::string &typeStr) {
  auto pos = typeStr.find(", ");
  if (pos != std::string::npos)
    return typeStr.substr(pos + 2, typeStr.size() - pos - 3);
  return "";
}

/// Map arith.cmpi predicate integer to MLIR keyword.
static const char *cmpiPredicateToStr(int pred) {
  static const char *names[] = {"eq",  "ne",  "slt", "sle", "sgt",
                                "sge", "ult", "ule", "ugt", "uge"};
  if (pred >= 0 && pred < 10)
    return names[pred];
  return "eq";
}

/// Map arith.cmpf predicate integer to MLIR keyword.
static const char *cmpfPredicateToStr(int pred) {
  static const char *names[] = {"false", "oeq", "ogt", "oge", "olt", "ole",
                                "one",   "ord", "ueq", "ugt", "uge", "ult",
                                "ule",   "une", "uno", "true"};
  if (pred >= 0 && pred < 16)
    return names[pred];
  return "false";
}

/// Emit the MLIR compare predicate keyword for a single-op compare PE.
/// Returns "" if the op is not a compare.
static std::string getComparePredicateStr(const PEDef &pe) {
  if (pe.singleOp == "arith.cmpi")
    return std::string(" ") + cmpiPredicateToStr(pe.comparePredicate) + ",";
  if (pe.singleOp == "arith.cmpf")
    return std::string(" ") + cmpfPredicateToStr(pe.comparePredicate) + ",";
  return "";
}

/// Determine the constant literal for a given value type.
/// Floating point types use "0.0", integers use "0".
static std::string getConstLiteral(Type valueType) {
  auto vk = valueType.getKind();
  if (vk == Type::F16 || vk == Type::BF16 || vk == Type::F32 ||
      vk == Type::F64)
    return "0.0";
  return "0";
}

//===----------------------------------------------------------------------===//
// PE definition MLIR generation
//===----------------------------------------------------------------------===//

/// Transform bodyMLIR: if it starts with ^bb0(%name: type, ...), strip the
/// block header and replace the user arg names with %arg0, %arg1, etc.
static std::string
transformBodyMLIR(const std::string &body,
                  const std::vector<Type> &inputPorts) {
  std::string b = body;

  // Find ^bb0( prefix.
  auto bbPos = b.find("^bb0(");
  if (bbPos == std::string::npos)
    return b;

  // Find the matching '):'
  auto closePos = b.find("):", bbPos);
  if (closePos == std::string::npos)
    return b;

  // Extract the arg list: "%name: type, %name2: type2"
  std::string argList = b.substr(bbPos + 5, closePos - bbPos - 5);

  // Parse arg names from the list.
  std::vector<std::string> argNames;
  std::istringstream argStream(argList);
  std::string token;
  while (std::getline(argStream, token, ',')) {
    // Trim whitespace.
    size_t start = token.find('%');
    if (start == std::string::npos) continue;
    size_t end = token.find(':', start);
    if (end == std::string::npos) end = token.size();
    argNames.push_back(token.substr(start, end - start));
  }

  // Get the body after "):" (skip optional newline).
  std::string remainder = b.substr(closePos + 2);
  if (!remainder.empty() && remainder[0] == '\n')
    remainder = remainder.substr(1);

  // Word-boundary-aware replacement helper.
  auto replaceAllIdent = [](std::string &text, const std::string &from,
                            const std::string &to) {
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
      size_t afterEnd = pos + from.size();
      if (afterEnd < text.size()) {
        char next = text[afterEnd];
        if (std::isalnum(next) || next == '_') {
          pos = afterEnd;
          continue;
        }
      }
      text.replace(pos, from.size(), to);
      pos += to.size();
    }
  };

  // Two-pass rename to avoid collisions (e.g. user body contains %arg0).
  // Pass 1: rename user args to unique temporaries.
  for (size_t i = 0; i < argNames.size() && i < inputPorts.size(); ++i) {
    std::string tmp = "%__loom_rename_" + std::to_string(i);
    if (argNames[i] == tmp) continue;
    replaceAllIdent(remainder, argNames[i], tmp);
  }
  // Pass 2: rename temporaries to final %argN names.
  for (size_t i = 0; i < argNames.size() && i < inputPorts.size(); ++i) {
    std::string tmp = "%__loom_rename_" + std::to_string(i);
    std::string target = "%arg" + std::to_string(i);
    if (tmp == target) continue;
    replaceAllIdent(remainder, tmp, target);
  }

  return remainder;
}

std::string ADGBuilder::Impl::generatePEBody(const PEDef &pe) const {
  if (!pe.bodyMLIR.empty())
    return transformBodyMLIR(pe.bodyMLIR, pe.inputPorts);

  assert(!pe.singleOp.empty() && "PE must have either bodyMLIR or singleOp");
  assert(!pe.inputPorts.empty() && "PE must have input ports");
  assert(!pe.outputPorts.empty() && "PE must have output ports");

  std::ostringstream os;
  os << "  %0 = " << pe.singleOp;
  os << getComparePredicateStr(pe);
  for (size_t i = 0; i < pe.inputPorts.size(); ++i) {
    os << (i == 0 ? " " : ", ");
    os << "%arg" << i;
  }
  // Compare ops require the operand type, not the result type (i1).
  bool isCmp = (pe.singleOp == "arith.cmpi" || pe.singleOp == "arith.cmpf");
  const auto &trailingType = isCmp ? pe.inputPorts[0] : pe.outputPorts[0];
  os << " : " << trailingType.toMLIR() << "\n";
  os << "  fabric.yield %0 : " << pe.outputPorts[0].toMLIR() << "\n";
  return os.str();
}

std::string ADGBuilder::Impl::generatePEDef(const PEDef &pe) const {
  std::ostringstream os;
  bool isTagged = pe.interface == InterfaceCategory::Tagged;

  // Emit signature (shared between tagged and native).
  os << "fabric.pe @" << pe.name << "(";
  for (size_t i = 0; i < pe.inputPorts.size(); ++i) {
    if (i > 0) os << ", ";
    os << "%arg" << i << ": " << pe.inputPorts[i].toMLIR();
  }
  os << ")\n";
  emitLatencyInterval(os, pe.latMin, pe.latTyp, pe.latMax, pe.intMin,
                      pe.intTyp, pe.intMax);
  os << "]\n";

  if (isTagged) {
    // Emit output_tag in {runtime_config} section.
    os << "    {output_tag = [";
    for (size_t i = 0; i < pe.outputPorts.size(); ++i) {
      if (i > 0) os << ", ";
      if (pe.outputPorts[i].isTagged())
        os << "0 : " << pe.outputPorts[i].getTagType().toMLIR();
      else
        os << "0 : i4"; // fallback
    }
    os << "]}\n";
  }

  os << "    -> (";
  emitTypeList(os, pe.outputPorts);
  os << ") {\n";

  if (isTagged) {
    // Tagged PE: body operates on value types.
    std::vector<Type> bodyInputTypes, bodyOutputTypes;
    for (const auto &t : pe.inputPorts)
      bodyInputTypes.push_back(t.isTagged() ? t.getValueType() : t);
    for (const auto &t : pe.outputPorts)
      bodyOutputTypes.push_back(t.isTagged() ? t.getValueType() : t);

    if (!pe.bodyMLIR.empty()) {
      os << pe.bodyMLIR;
    } else {
      os << "^bb0(";
      for (size_t i = 0; i < bodyInputTypes.size(); ++i) {
        if (i > 0) os << ", ";
        os << "%x" << i << ": " << bodyInputTypes[i].toMLIR();
      }
      os << "):\n";
      os << "  %0 = " << pe.singleOp;
      for (size_t i = 0; i < bodyInputTypes.size(); ++i) {
        os << (i == 0 ? " " : ", ");
        os << "%x" << i;
      }
      os << " : " << bodyOutputTypes[0].toMLIR() << "\n";
      os << "  fabric.yield %0 : " << bodyOutputTypes[0].toMLIR() << "\n";
    }
  } else {
    os << generatePEBody(pe);
  }
  os << "}\n\n";

  return os.str();
}

std::string
ADGBuilder::Impl::generateConstantPEDef(const ConstantPEDef &def) const {
  std::ostringstream os;
  Type outType = def.outputType;
  bool isTagged = outType.isTagged();
  Type valueType = isTagged ? outType.getValueType() : outType;

  std::string ctrlTypeStr =
      isTagged ? Type::tagged(Type::none(), outType.getTagType()).toMLIR()
               : "none";

  os << "fabric.pe @" << def.name << "(%ctrl: " << ctrlTypeStr << ")\n";
  emitLatencyInterval(os, def.latMin, def.latTyp, def.latMax, def.intMin,
                      def.intTyp, def.intMax);
  os << "]\n";
  if (isTagged)
    os << "    {output_tag = [0 : " << outType.getTagType().toMLIR() << "]}\n";
  os << "    -> (" << outType.toMLIR() << ") {\n";

  std::string constLiteral = getConstLiteral(valueType);
  std::string vStr = valueType.toMLIR();

  if (isTagged) {
    os << "^bb0(%c_native: none):\n";
    os << "  %c = handshake.constant %c_native {value = " << constLiteral
       << " : " << vStr << "} : " << vStr << "\n";
  } else {
    os << "  %c = handshake.constant %ctrl {value = " << constLiteral << " : "
       << vStr << "} : " << vStr << "\n";
  }
  os << "  fabric.yield %c : " << vStr << "\n";
  os << "}\n\n";
  return os.str();
}

std::string
ADGBuilder::Impl::generateLoadPEDef(const LoadPEDef &def) const {
  std::ostringstream os;
  bool isTagged = def.interface == InterfaceCategory::Tagged;
  Type dataType = def.dataType;
  std::string dTypeStr = dataType.toMLIR();

  if (isTagged) {
    Type tagType = Type::iN(def.tagWidth);
    Type taggedData = Type::tagged(dataType, tagType);
    Type taggedIndex = Type::tagged(Type::index(), tagType);
    Type taggedNone = Type::tagged(Type::none(), tagType);
    std::string tdStr = taggedData.toMLIR();
    std::string tiStr = taggedIndex.toMLIR();
    std::string tnStr = taggedNone.toMLIR();

    os << "fabric.pe @" << def.name
       << "(%addr: " << tiStr << ", %data_in: " << tdStr
       << ", %ctrl: " << tnStr << ")\n";
    os << "    [latency = [1 : i16, 1 : i16, 1 : i16]";
    os << ", interval = [1 : i16, 1 : i16, 1 : i16]";
    if (def.hwType == HardwareType::TagTransparent)
      os << ", lqDepth = " << def.queueDepth;
    os << "]\n";
    os << "    {output_tag = [0 : " << tagType.toMLIR()
       << ", 0 : " << tagType.toMLIR() << "]}\n";
    os << "    -> (" << tdStr << ", " << tiStr << ") {\n";
    os << "^bb0(%x: index, %y: " << dTypeStr << ", %c: none):\n";
    os << "  %ld_d, %ld_a = handshake.load [%x] %y, %c : index, "
       << dTypeStr << "\n";
    os << "  fabric.yield %ld_d, %ld_a : " << dTypeStr << ", index\n";
    os << "}\n\n";
  } else {
    os << "fabric.pe @" << def.name
       << "(%addr: index, %data_in: " << dTypeStr << ", %ctrl: none)\n";
    os << "    [latency = [1 : i16, 1 : i16, 1 : i16]";
    os << ", interval = [1 : i16, 1 : i16, 1 : i16]";
    os << "]\n";
    os << "    -> (" << dTypeStr << ", index) {\n";
    os << "  %ld_d, %ld_a = handshake.load [%addr] %data_in, %ctrl : index, "
       << dTypeStr << "\n";
    os << "  fabric.yield %ld_d, %ld_a : " << dTypeStr << ", index\n";
    os << "}\n\n";
  }
  return os.str();
}

std::string
ADGBuilder::Impl::generateStorePEDef(const StorePEDef &def) const {
  std::ostringstream os;
  bool isTagged = def.interface == InterfaceCategory::Tagged;
  Type dataType = def.dataType;
  std::string dTypeStr = dataType.toMLIR();

  if (isTagged) {
    Type tagType = Type::iN(def.tagWidth);
    Type taggedData = Type::tagged(dataType, tagType);
    Type taggedIndex = Type::tagged(Type::index(), tagType);
    Type taggedNone = Type::tagged(Type::none(), tagType);
    std::string tdStr = taggedData.toMLIR();
    std::string tiStr = taggedIndex.toMLIR();
    std::string tnStr = taggedNone.toMLIR();

    os << "fabric.pe @" << def.name
       << "(%addr: " << tiStr << ", %data: " << tdStr
       << ", %ctrl: " << tnStr << ")\n";
    os << "    [latency = [1 : i16, 1 : i16, 1 : i16]";
    os << ", interval = [1 : i16, 1 : i16, 1 : i16]";
    if (def.hwType == HardwareType::TagTransparent)
      os << ", sqDepth = " << def.queueDepth;
    os << "]\n";
    os << "    {output_tag = [0 : " << tagType.toMLIR()
       << ", 0 : " << tagType.toMLIR() << "]}\n";
    os << "    -> (" << tiStr << ", " << tnStr << ") {\n";
    os << "^bb0(%x: index, %y: " << dTypeStr << ", %c: none):\n";
    os << "  handshake.store [%x] %y, %c : index, " << dTypeStr << "\n";
    os << "  fabric.yield %x, %c : index, none\n";
    os << "}\n\n";
  } else {
    os << "fabric.pe @" << def.name
       << "(%addr: index, %data: " << dTypeStr << ", %ctrl: none)\n";
    os << "    [latency = [1 : i16, 1 : i16, 1 : i16]";
    os << ", interval = [1 : i16, 1 : i16, 1 : i16]";
    os << "]\n";
    os << "    -> (index, none) {\n";
    os << "  handshake.store [%addr] %data, %ctrl : index, " << dTypeStr
       << "\n";
    os << "  fabric.yield %addr, %ctrl : index, none\n";
    os << "}\n\n";
  }
  return os.str();
}

std::string
ADGBuilder::Impl::generateTemporalPEDef(const TemporalPEDef &def) const {
  std::ostringstream os;
  Type ifType = def.interfaceType;
  std::string ifStr = ifType.toMLIR();

  // Determine number of input/output ports from first FU.
  unsigned numIn = 1, numOut = 1;
  if (!def.fuPEDefIndices.empty()) {
    numIn = peDefs[def.fuPEDefIndices[0]].inputPorts.size();
    numOut = peDefs[def.fuPEDefIndices[0]].outputPorts.size();
  }

  os << "fabric.temporal_pe @" << def.name << "(";
  for (unsigned i = 0; i < numIn; ++i) {
    if (i > 0) os << ", ";
    os << "%in" << i << ": " << ifStr;
  }
  os << ")\n";
  os << "    [num_register = " << def.numRegisters
     << ", num_instruction = " << def.numInstructions
     << ", reg_fifo_depth = " << def.regFifoDepth;
  if (def.shareModeB)
    os << ", enable_share_operand_buffer = true"
       << ", operand_buffer_size = " << def.shareBufferSize;
  os << "]\n";
  os << "    -> (";
  for (unsigned i = 0; i < numOut; ++i) {
    if (i > 0) os << ", ";
    os << ifStr;
  }
  os << ") {\n";

  // Emit FU definitions (simpler format without latency/interval).
  for (unsigned fuIdx : def.fuPEDefIndices) {
    const auto &fu = peDefs[fuIdx];
    os << "  fabric.pe @" << fu.name << "(";
    for (size_t i = 0; i < fu.inputPorts.size(); ++i) {
      if (i > 0) os << ", ";
      os << "%arg" << i << ": " << fu.inputPorts[i].toMLIR();
    }
    os << ") -> (";
    for (size_t i = 0; i < fu.outputPorts.size(); ++i) {
      if (i > 0) os << ", ";
      os << fu.outputPorts[i].toMLIR();
    }
    os << ") {\n";
    os << "  " << generatePEBody(fu);
    os << "  }\n";
  }

  os << "  fabric.yield\n";
  os << "}\n\n";
  return os.str();
}

//===----------------------------------------------------------------------===//
// Full MLIR generation
//===----------------------------------------------------------------------===//

std::string ADGBuilder::Impl::generateMLIR() const {
  std::ostringstream os;

  // Collect which definitions are actually referenced by instances.
  std::set<unsigned> usedPEDefs, usedConstPEDefs, usedLoadPEDefs,
      usedStorePEDefs, usedTemporalPEDefs;

  for (const auto &inst : instances) {
    switch (inst.kind) {
    case ModuleKind::PE:
      usedPEDefs.insert(inst.defIdx);
      break;
    case ModuleKind::ConstantPE:
      usedConstPEDefs.insert(inst.defIdx);
      break;
    case ModuleKind::LoadPE:
      usedLoadPEDefs.insert(inst.defIdx);
      break;
    case ModuleKind::StorePE:
      usedStorePEDefs.insert(inst.defIdx);
      break;
    case ModuleKind::TemporalPE:
      usedTemporalPEDefs.insert(inst.defIdx);
      break;
    default:
      break;
    }
  }

  // Emit named PE definitions (Pattern A).
  // Skip tagged PEs -- they use inline form (Pattern B) instead.
  for (unsigned idx : usedPEDefs) {
    if (peDefs[idx].interface != InterfaceCategory::Tagged)
      os << generatePEDef(peDefs[idx]);
  }
  for (unsigned idx : usedConstPEDefs) {
    if (!constantPEDefs[idx].outputType.isTagged())
      os << generateConstantPEDef(constantPEDefs[idx]);
  }
  for (unsigned idx : usedLoadPEDefs) {
    if (loadPEDefs[idx].interface != InterfaceCategory::Tagged)
      os << generateLoadPEDef(loadPEDefs[idx]);
  }
  for (unsigned idx : usedStorePEDefs) {
    if (storePEDefs[idx].interface != InterfaceCategory::Tagged)
      os << generateStorePEDef(storePEDefs[idx]);
  }
  for (unsigned idx : usedTemporalPEDefs)
    os << generateTemporalPEDef(temporalPEDefs[idx]);

  // Collect module input and output ports with proper ordering.
  // Port ordering: memref*, native*, tagged*
  std::vector<unsigned> inputPortIndices, outputPortIndices;
  for (unsigned i = 0; i < ports.size(); ++i) {
    if (ports[i].isInput)
      inputPortIndices.push_back(i);
    else
      outputPortIndices.push_back(i);
  }

  auto portOrder = [&](unsigned a, unsigned b) -> bool {
    const auto &pa = ports[a];
    const auto &pb = ports[b];
    int orderA = pa.isMemref ? 0 : (pa.type.isTagged() ? 2 : 1);
    int orderB = pb.isMemref ? 0 : (pb.type.isTagged() ? 2 : 1);
    if (orderA != orderB) return orderA < orderB;
    return a < b; // stable order
  };
  std::sort(inputPortIndices.begin(), inputPortIndices.end(), portOrder);
  std::sort(outputPortIndices.begin(), outputPortIndices.end(), portOrder);

  // Build maps for port resolution.
  std::map<unsigned, unsigned> inputPortToArgIdx;
  for (size_t i = 0; i < inputPortIndices.size(); ++i)
    inputPortToArgIdx[inputPortIndices[i]] = i;

  std::map<unsigned, unsigned> outputPortToResultIdx;
  for (size_t i = 0; i < outputPortIndices.size(); ++i)
    outputPortToResultIdx[outputPortIndices[i]] = i;

  // Emit fabric.module signature.
  os << "fabric.module @" << moduleName << "(";
  for (size_t i = 0; i < inputPortIndices.size(); ++i) {
    if (i > 0) os << ", ";
    const auto &p = ports[inputPortIndices[i]];
    os << "%" << p.name << ": ";
    os << (p.isMemref ? p.memrefType.toMLIR() : p.type.toMLIR());
  }
  os << ") -> (";
  for (size_t i = 0; i < outputPortIndices.size(); ++i) {
    if (i > 0) os << ", ";
    const auto &p = ports[outputPortIndices[i]];
    os << (p.isMemref ? p.memrefType.toMLIR() : p.type.toMLIR());
  }
  os << ") {\n";

  // Topological sort of instances.
  // The connection graph may contain cycles (e.g. torus wraparound with fifos).
  // Any remaining instances not reachable via Kahn's algorithm are appended
  // in original order (the Graph region semantics allow forward references).
  unsigned numInst = instances.size();
  std::vector<std::vector<unsigned>> adjList(numInst);
  std::vector<unsigned> inDeg(numInst, 0);

  for (const auto &conn : internalConns) {
    adjList[conn.srcInst].push_back(conn.dstInst);
    inDeg[conn.dstInst]++;
  }

  std::queue<unsigned> readyQueue;
  for (unsigned i = 0; i < numInst; ++i) {
    if (inDeg[i] == 0)
      readyQueue.push(i);
  }

  std::vector<unsigned> topoOrder;
  while (!readyQueue.empty()) {
    unsigned cur = readyQueue.front();
    readyQueue.pop();
    topoOrder.push_back(cur);
    for (unsigned next : adjList[cur]) {
      if (--inDeg[next] == 0)
        readyQueue.push(next);
    }
  }
  // Add any remaining instances (cycles or disconnected) in original order.
  if (topoOrder.size() < numInst) {
    std::set<unsigned> ordered(topoOrder.begin(), topoOrder.end());
    for (unsigned i = 0; i < numInst; ++i) {
      if (ordered.find(i) == ordered.end())
        topoOrder.push_back(i);
    }
  }

  // Pre-allocate SSA names for all instances (enables forward references in
  // bidirectional connections like mesh switch-to-switch).
  unsigned ssaCounter = 0;
  std::map<std::pair<unsigned, int>, std::string> instResultSSA;
  std::map<std::pair<unsigned, int>, std::string> instResultType;

  for (unsigned ii : topoOrder) {
    unsigned numOutputs = getInstanceOutputCount(ii);
    for (unsigned r = 0; r < numOutputs; ++r) {
      std::string name = "%" + std::to_string(ssaCounter + r);
      instResultSSA[{ii, (int)r}] = name;
      // Use PortType to correctly handle memref outputs (e.g. memory port 0).
      instResultType[{ii, (int)r}] =
          getInstanceOutputPortType(ii, r).toMLIR();
    }
    ssaCounter += numOutputs;
  }

  // Emit instances in topological order.
  ssaCounter = 0;
  for (unsigned ii : topoOrder) {
    const auto &inst = instances[ii];

    // Gather operand SSA names and types.
    unsigned numInputs = getInstanceInputCount(ii);
    std::vector<std::string> operands(numInputs, "");
    std::vector<std::string> operandTypes(numInputs, "");

    // From module input connections.
    for (const auto &conn : inputConns) {
      if (conn.instIdx == ii) {
        auto it = inputPortToArgIdx.find(conn.portIdx);
        if (it != inputPortToArgIdx.end()) {
          const auto &p = ports[inputPortIndices[it->second]];
          operands[conn.dstPort] = "%" + p.name;
          operandTypes[conn.dstPort] =
              p.isMemref ? p.memrefType.toMLIR() : p.type.toMLIR();
        }
      }
    }

    // From internal connections (SSA names pre-allocated, so forward refs work).
    for (const auto &conn : internalConns) {
      if (conn.dstInst == ii) {
        auto it = instResultSSA.find({conn.srcInst, conn.srcPort});
        if (it != instResultSSA.end())
          operands[conn.dstPort] = it->second;
        auto tit = instResultType.find({conn.srcInst, conn.srcPort});
        if (tit != instResultType.end())
          operandTypes[conn.dstPort] = tit->second;
      }
    }

    // Collect pre-allocated result SSA names.
    unsigned numOutputs = getInstanceOutputCount(ii);
    std::vector<std::string> resultNames;
    for (unsigned r = 0; r < numOutputs; ++r)
      resultNames.push_back(instResultSSA[{ii, (int)r}]);

    // Emit the instance/operation.
    os << "  ";
    for (size_t r = 0; r < resultNames.size(); ++r) {
      if (r > 0) os << ", ";
      os << resultNames[r];
    }
    if (!resultNames.empty())
      os << " = ";

    // Check if this PE should use inline form (for tagged PEs).
    bool useInlinePE = false;
    if (inst.kind == ModuleKind::PE) {
      auto &pd = peDefs[inst.defIdx];
      useInlinePE = pd.interface == InterfaceCategory::Tagged;
    } else if (inst.kind == ModuleKind::ConstantPE) {
      useInlinePE = constantPEDefs[inst.defIdx].outputType.isTagged();
    } else if (inst.kind == ModuleKind::LoadPE) {
      useInlinePE = loadPEDefs[inst.defIdx].interface == InterfaceCategory::Tagged;
    } else if (inst.kind == ModuleKind::StorePE) {
      useInlinePE = storePEDefs[inst.defIdx].interface == InterfaceCategory::Tagged;
    }

    switch (inst.kind) {
    case ModuleKind::PE:
    case ModuleKind::ConstantPE:
    case ModuleKind::LoadPE:
    case ModuleKind::StorePE:
    case ModuleKind::TemporalPE: {
      std::string defName;
      std::vector<Type> inTypes, outTypes;

      switch (inst.kind) {
      case ModuleKind::PE:
        defName = peDefs[inst.defIdx].name;
        inTypes = peDefs[inst.defIdx].inputPorts;
        outTypes = peDefs[inst.defIdx].outputPorts;
        break;
      case ModuleKind::ConstantPE: {
        auto &cpDef = constantPEDefs[inst.defIdx];
        defName = cpDef.name;
        if (cpDef.outputType.isTagged()) {
          Type tagType = cpDef.outputType.getTagType();
          inTypes = {Type::tagged(Type::none(), tagType)};
        } else {
          inTypes = {Type::none()};
        }
        outTypes = {cpDef.outputType};
        break;
      }
      case ModuleKind::LoadPE: {
        auto &lpDef = loadPEDefs[inst.defIdx];
        defName = lpDef.name;
        if (lpDef.interface == InterfaceCategory::Tagged) {
          Type tagType = Type::iN(lpDef.tagWidth);
          inTypes = {Type::tagged(Type::index(), tagType),
                     Type::tagged(lpDef.dataType, tagType),
                     Type::tagged(Type::none(), tagType)};
          outTypes = {Type::tagged(lpDef.dataType, tagType),
                      Type::tagged(Type::index(), tagType)};
        } else {
          inTypes = {Type::index(), lpDef.dataType, Type::none()};
          outTypes = {lpDef.dataType, Type::index()};
        }
        break;
      }
      case ModuleKind::StorePE: {
        auto &spDef = storePEDefs[inst.defIdx];
        defName = spDef.name;
        if (spDef.interface == InterfaceCategory::Tagged) {
          Type tagType = Type::iN(spDef.tagWidth);
          inTypes = {Type::tagged(Type::index(), tagType),
                     Type::tagged(spDef.dataType, tagType),
                     Type::tagged(Type::none(), tagType)};
          outTypes = {Type::tagged(Type::index(), tagType),
                      Type::tagged(Type::none(), tagType)};
        } else {
          inTypes = {Type::index(), spDef.dataType, Type::none()};
          outTypes = {Type::index(), Type::none()};
        }
        break;
      }
      case ModuleKind::TemporalPE: {
        auto &tpeDef = temporalPEDefs[inst.defIdx];
        defName = tpeDef.name;
        unsigned nIn = 1, nOut = 1;
        if (!tpeDef.fuPEDefIndices.empty()) {
          nIn = peDefs[tpeDef.fuPEDefIndices[0]].inputPorts.size();
          nOut = peDefs[tpeDef.fuPEDefIndices[0]].outputPorts.size();
        }
        inTypes.resize(nIn, tpeDef.interfaceType);
        outTypes.resize(nOut, tpeDef.interfaceType);
        break;
      }
      default:
        break;
      }

      if (useInlinePE) {
        // Inline PE form (Pattern B) for tagged PEs.
        os << "fabric.pe ";
        for (size_t o = 0; o < operands.size(); ++o) {
          if (o > 0) os << ", ";
          os << operands[o];
        }
        os << "\n";

        // Emit hw_params [latency, interval, lqDepth, sqDepth].
        {
          int16_t latMin = 1, latTyp = 1, latMax = 1;
          int16_t intMin = 1, intTyp = 1, intMax = 1;
          if (inst.kind == ModuleKind::PE) {
            auto &pd = peDefs[inst.defIdx];
            latMin = pd.latMin; latTyp = pd.latTyp; latMax = pd.latMax;
            intMin = pd.intMin; intTyp = pd.intTyp; intMax = pd.intMax;
          } else if (inst.kind == ModuleKind::ConstantPE) {
            auto &cd = constantPEDefs[inst.defIdx];
            latMin = cd.latMin; latTyp = cd.latTyp; latMax = cd.latMax;
            intMin = cd.intMin; intTyp = cd.intTyp; intMax = cd.intMax;
          }
          os << "      [latency = ["
             << latMin << " : i16, " << latTyp << " : i16, "
             << latMax << " : i16]"
             << ", interval = ["
             << intMin << " : i16, " << intTyp << " : i16, "
             << intMax << " : i16]";
          if (inst.kind == ModuleKind::LoadPE) {
            auto &lpDef = loadPEDefs[inst.defIdx];
            if (lpDef.hwType == HardwareType::TagTransparent)
              os << ", lqDepth = " << lpDef.queueDepth;
          } else if (inst.kind == ModuleKind::StorePE) {
            auto &spDef = storePEDefs[inst.defIdx];
            if (spDef.hwType == HardwareType::TagTransparent)
              os << ", sqDepth = " << spDef.queueDepth;
          }
          os << "]\n";
        }

        // Emit runtime_config {output_tag} if applicable (TagOverwrite mode).
        bool hasOutputTag = false;
        if (inst.kind == ModuleKind::PE) {
          hasOutputTag = true; // tagged PE always has output_tag
        } else if (inst.kind == ModuleKind::ConstantPE) {
          hasOutputTag = true; // tagged constant PE always has output_tag
        } else if (inst.kind == ModuleKind::LoadPE) {
          hasOutputTag =
              loadPEDefs[inst.defIdx].hwType == HardwareType::TagOverwrite;
        } else if (inst.kind == ModuleKind::StorePE) {
          hasOutputTag =
              storePEDefs[inst.defIdx].hwType == HardwareType::TagOverwrite;
        }
        if (hasOutputTag) {
          os << "      {output_tag = [";
          for (size_t i = 0; i < outTypes.size(); ++i) {
            if (i > 0) os << ", ";
            if (outTypes[i].isTagged())
              os << "0 : " << outTypes[i].getTagType().toMLIR();
            else
              os << "0 : i4"; // fallback
          }
          os << "]}\n";
        }

        // Emit type signature.
        os << "      : (";
        for (size_t p = 0; p < inTypes.size(); ++p) {
          if (p > 0) os << ", ";
          os << inTypes[p].toMLIR();
        }
        os << ")\n      -> (";
        for (size_t p = 0; p < outTypes.size(); ++p) {
          if (p > 0) os << ", ";
          os << outTypes[p].toMLIR();
        }
        os << ") {\n";

        // Emit body with value types in ^bb0.
        // Compute value types from tagged types.
        std::vector<Type> bodyInTypes;
        for (const auto &t : inTypes)
          bodyInTypes.push_back(t.isTagged() ? t.getValueType() : t);
        std::vector<Type> bodyOutTypes;
        for (const auto &t : outTypes)
          bodyOutTypes.push_back(t.isTagged() ? t.getValueType() : t);

        switch (inst.kind) {
        case ModuleKind::PE: {
          auto &pd = peDefs[inst.defIdx];
          if (!pd.bodyMLIR.empty()) {
            // Inline PE regions need an explicit ^bb0 header with value types.
            // Emit a fresh header, then append the transformed body statements
            // (which renames user args to %argN).
            os << "  ^bb0(";
            for (size_t i = 0; i < bodyInTypes.size(); ++i) {
              if (i > 0) os << ", ";
              os << "%arg" << i << ": " << bodyInTypes[i].toMLIR();
            }
            os << "):\n";
            os << transformBodyMLIR(pd.bodyMLIR, bodyInTypes);
          } else {
            os << "  ^bb0(";
            for (size_t i = 0; i < bodyInTypes.size(); ++i) {
              if (i > 0) os << ", ";
              os << "%x" << i << ": " << bodyInTypes[i].toMLIR();
            }
            os << "):\n";
            os << "    %r = " << pd.singleOp;
            os << getComparePredicateStr(pd);
            for (size_t i = 0; i < bodyInTypes.size(); ++i) {
              os << (i == 0 ? " " : ", ");
              os << "%x" << i;
            }
            bool isFuCmp = (pd.singleOp == "arith.cmpi" ||
                            pd.singleOp == "arith.cmpf");
            const auto &fuTrailingType =
                isFuCmp ? bodyInTypes[0] : bodyOutTypes[0];
            os << " : " << fuTrailingType.toMLIR() << "\n";
            os << "    fabric.yield %r : " << bodyOutTypes[0].toMLIR() << "\n";
          }
          break;
        }
        case ModuleKind::ConstantPE: {
          auto &cpDef = constantPEDefs[inst.defIdx];
          Type valueType = cpDef.outputType.isTagged()
                               ? cpDef.outputType.getValueType()
                               : cpDef.outputType;
          std::string constLiteral = getConstLiteral(valueType);
          std::string vStr = valueType.toMLIR();
          os << "  ^bb0(%c_native: none):\n";
          os << "    %c = handshake.constant %c_native {value = "
             << constLiteral << " : " << vStr << "} : " << vStr << "\n";
          os << "    fabric.yield %c : " << vStr << "\n";
          break;
        }
        case ModuleKind::LoadPE: {
          auto &lpDef = loadPEDefs[inst.defIdx];
          std::string dTypeStr = lpDef.dataType.toMLIR();
          os << "  ^bb0(%x: index, %y: " << dTypeStr << ", %c: none):\n";
          os << "    %ld_d, %ld_a = handshake.load [%x] %y, %c : index, "
             << dTypeStr << "\n";
          os << "    fabric.yield %ld_d, %ld_a : " << dTypeStr
             << ", index\n";
          break;
        }
        case ModuleKind::StorePE: {
          auto &spDef = storePEDefs[inst.defIdx];
          std::string dTypeStr = spDef.dataType.toMLIR();
          os << "  ^bb0(%x: index, %y: " << dTypeStr << ", %c: none):\n";
          os << "    handshake.store [%x] %y, %c : index, " << dTypeStr
             << "\n";
          os << "    fabric.yield %x, %c : index, none\n";
          break;
        }
        default:
          break;
        }
        os << "  }\n";
      } else {
        // Named PE form (Pattern A) via fabric.instance.
        os << "fabric.instance @" << defName << "(";
        for (size_t o = 0; o < operands.size(); ++o) {
          if (o > 0) os << ", ";
          os << operands[o];
        }
        os << ")";
        if (!inst.name.empty())
          os << " {sym_name = \"" << inst.name << "\"}";
        os << " : (";
        for (size_t p = 0; p < inTypes.size(); ++p) {
          if (p > 0) os << ", ";
          os << inTypes[p].toMLIR();
        }
        os << ") -> (";
        for (size_t p = 0; p < outTypes.size(); ++p) {
          if (p > 0) os << ", ";
          os << outTypes[p].toMLIR();
        }
        os << ")\n";
      }
      break;
    }

    case ModuleKind::Switch: {
      auto &swDef = switchDefs[inst.defIdx];
      std::string typeStr = swDef.portType.toMLIR();
      auto flatConn = flattenConnectivity(swDef.connectivity, swDef.numOut,
                                          swDef.numIn);

      os << "fabric.switch [connectivity_table = [";
      for (size_t i = 0; i < flatConn.size(); ++i) {
        if (i > 0) os << ", ";
        os << flatConn[i];
      }
      os << "]] ";
      for (size_t o = 0; o < operands.size(); ++o) {
        if (o > 0) os << ", ";
        os << operands[o];
      }
      os << " : " << typeStr << " -> ";
      for (unsigned o = 0; o < swDef.numOut; ++o) {
        if (o > 0) os << ", ";
        os << typeStr;
      }
      os << "\n";
      break;
    }

    case ModuleKind::TemporalSwitch: {
      auto &tsDef = temporalSwitchDefs[inst.defIdx];
      std::string typeStr = tsDef.interfaceType.toMLIR();
      auto flatConn = flattenConnectivity(tsDef.connectivity, tsDef.numOut,
                                          tsDef.numIn);

      os << "fabric.temporal_sw [num_route_table = " << tsDef.numRouteTable
         << ", connectivity_table = [";
      for (size_t i = 0; i < flatConn.size(); ++i) {
        if (i > 0) os << ", ";
        os << flatConn[i];
      }
      os << "]] ";
      for (size_t o = 0; o < operands.size(); ++o) {
        if (o > 0) os << ", ";
        os << operands[o];
      }
      os << " : " << typeStr << " -> ";
      for (unsigned o = 0; o < tsDef.numOut; ++o) {
        if (o > 0) os << ", ";
        os << typeStr;
      }
      os << "\n";
      break;
    }

    case ModuleKind::Memory: {
      auto &memDef = memoryDefs[inst.defIdx];
      std::string memrefStr = memDef.shape.toMLIR();
      Type elemType = memDef.shape.getElemType();
      std::string elemStr = elemType.toMLIR();

      // Determine per-side tagging from operand types.
      bool isTaggedLd = memDef.ldCount > 1;
      bool isTaggedSt = memDef.stCount > 1;

      // Derive tag type strings from operand types at the boundary of each group.
      // Input layout: [ld_addr * ldCount, st_addr * stCount, st_data * stCount]
      std::string ldTagTypeStr, stTagTypeStr;
      if (isTaggedLd && !operandTypes.empty())
        ldTagTypeStr = extractTagType(operandTypes[0]);
      if (isTaggedSt && memDef.ldCount < operandTypes.size())
        stTagTypeStr = extractTagType(operandTypes[memDef.ldCount]);

      os << "fabric.memory\n      [ldCount = " << memDef.ldCount
         << ", stCount = " << memDef.stCount;
      if (memDef.lsqDepth > 0)
        os << ", lsqDepth = " << memDef.lsqDepth;
      os << ", is_private = " << (memDef.isPrivate ? "true" : "false");
      os << "]\n      (";
      for (size_t o = 0; o < operands.size(); ++o) {
        if (o > 0) os << ", ";
        os << operands[o];
      }
      os << ")\n      : " << memrefStr << ", (";
      // Input types from actual operand types.
      for (size_t o = 0; o < operandTypes.size(); ++o) {
        if (o > 0) os << ", ";
        os << operandTypes[o];
      }
      os << ") -> (";
      // Output types: [memref?] [lddata * ldCount] [lddone] [stdone?]
      bool first = true;
      if (!memDef.isPrivate) {
        os << memrefStr;
        first = false;
      }
      for (unsigned i = 0; i < memDef.ldCount; ++i) {
        if (!first) os << ", ";
        first = false;
        if (isTaggedLd)
          os << "!dataflow.tagged<" << elemStr << ", " << ldTagTypeStr << ">";
        else
          os << elemStr;
      }
      // lddone (always present)
      if (!first) os << ", ";
      if (isTaggedLd)
        os << "!dataflow.tagged<none, " << ldTagTypeStr << ">";
      else
        os << "none";
      if (memDef.stCount > 0) {
        os << ", ";
        if (isTaggedSt)
          os << "!dataflow.tagged<none, " << stTagTypeStr << ">";
        else
          os << "none";
      }
      os << ")\n";
      break;
    }

    case ModuleKind::ExtMemory: {
      auto &emDef = extMemoryDefs[inst.defIdx];
      std::string memrefStr = emDef.shape.toMLIR();
      Type elemType = emDef.shape.getElemType();
      std::string elemStr = elemType.toMLIR();

      // Determine per-side tagging from operand types after memref.
      bool isTaggedLd = emDef.ldCount > 1;
      bool isTaggedSt = emDef.stCount > 1;

      // ExtMemory input layout: [memref, ld_addr * ldCount, st_addr * stCount, st_data * stCount]
      // operandTypes[0] is memref, operandTypes[1] is first ld_addr.
      std::string ldTagTypeStr, stTagTypeStr;
      if (isTaggedLd && operandTypes.size() > 1)
        ldTagTypeStr = extractTagType(operandTypes[1]);
      if (isTaggedSt && (1 + emDef.ldCount) < operandTypes.size())
        stTagTypeStr = extractTagType(operandTypes[1 + emDef.ldCount]);

      os << "fabric.extmemory\n      [ldCount = " << emDef.ldCount
         << ", stCount = " << emDef.stCount;
      if (emDef.lsqDepth > 0)
        os << ", lsqDepth = " << emDef.lsqDepth;
      os << "]\n      (";
      for (size_t o = 0; o < operands.size(); ++o) {
        if (o > 0) os << ", ";
        os << operands[o];
      }
      os << ")\n      : " << memrefStr << ", (";
      // Input types from actual operand types.
      for (size_t o = 0; o < operandTypes.size(); ++o) {
        if (o > 0) os << ", ";
        os << operandTypes[o];
      }
      os << ") -> (";
      // Output types: [lddata * ldCount] [lddone] [stdone?]
      bool first = true;
      for (unsigned i = 0; i < emDef.ldCount; ++i) {
        if (!first) os << ", ";
        first = false;
        if (isTaggedLd)
          os << "!dataflow.tagged<" << elemStr << ", " << ldTagTypeStr << ">";
        else
          os << elemStr;
      }
      if (!first) os << ", ";
      if (isTaggedLd)
        os << "!dataflow.tagged<none, " << ldTagTypeStr << ">";
      else
        os << "none";
      if (emDef.stCount > 0) {
        os << ", ";
        if (isTaggedSt)
          os << "!dataflow.tagged<none, " << stTagTypeStr << ">";
        else
          os << "none";
      }
      os << ")\n";
      break;
    }

    case ModuleKind::AddTag: {
      auto &atDef = addTagDefs[inst.defIdx];
      os << "fabric.add_tag " << operands[0]
         << " {tag = 0 : " << atDef.tagType.toMLIR() << "} : "
         << atDef.valueType.toMLIR() << " -> "
         << Type::tagged(atDef.valueType, atDef.tagType).toMLIR() << "\n";
      break;
    }

    case ModuleKind::MapTag: {
      auto &mtDef = mapTagDefs[inst.defIdx];
      Type inTagged = Type::tagged(mtDef.valueType, mtDef.inputTagType);
      Type outTagged = Type::tagged(mtDef.valueType, mtDef.outputTagType);
      os << "fabric.map_tag " << operands[0]
         << " {table_size = " << mtDef.tableSize << "} : "
         << inTagged.toMLIR() << " -> " << outTagged.toMLIR() << "\n";
      break;
    }

    case ModuleKind::DelTag: {
      auto &dtDef = delTagDefs[inst.defIdx];
      Type outType = dtDef.inputType.getValueType();
      os << "fabric.del_tag " << operands[0]
         << " : " << dtDef.inputType.toMLIR() << " -> "
         << outType.toMLIR() << "\n";
      break;
    }

    case ModuleKind::Fifo: {
      auto &fifoDef = fifoDefs[inst.defIdx];
      os << "fabric.fifo [depth = " << fifoDef.depth;
      if (fifoDef.bypassable)
        os << ", bypassable";
      os << "] ";
      if (fifoDef.bypassable)
        os << "{bypassed = false} ";
      os << operands[0] << " : " << fifoDef.elementType.toMLIR() << "\n";
      break;
    }

    } // switch inst.kind
  } // for each instance

  // Emit fabric.yield with output connections.
  os << "  fabric.yield";
  if (!outputPortIndices.empty()) {
    os << " ";
    std::vector<std::pair<std::string, std::string>> yieldArgs;
    for (size_t oi = 0; oi < outputPortIndices.size(); ++oi) {
      unsigned outPortIdx = outputPortIndices[oi];
      for (const auto &conn : outputConns) {
        if (conn.portIdx == outPortIdx) {
          auto it = instResultSSA.find({conn.instIdx, conn.srcPort});
          if (it != instResultSSA.end()) {
            const auto &p = ports[outPortIdx];
            std::string typeStr =
                p.isMemref ? p.memrefType.toMLIR() : p.type.toMLIR();
            yieldArgs.push_back({it->second, typeStr});
          }
          break;
        }
      }
    }
    for (size_t i = 0; i < yieldArgs.size(); ++i) {
      if (i > 0) os << ", ";
      os << yieldArgs[i].first;
    }
    if (!yieldArgs.empty()) {
      os << " : ";
      for (size_t i = 0; i < yieldArgs.size(); ++i) {
        if (i > 0) os << ", ";
        os << yieldArgs[i].second;
      }
    }
  }
  os << "\n";
  os << "}\n";

  return os.str();
}

} // namespace adg
} // namespace loom
