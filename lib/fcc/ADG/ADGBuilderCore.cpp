//===-- ADGBuilderCore.cpp - ADG Builder core API ----------------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//

#include "fcc/ADG/ADGBuilderDetail.h"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include <sstream>

namespace fcc {
namespace adg {

namespace detail {

std::string bitsType(unsigned width) {
  return "!fabric.bits<" + std::to_string(width) + ">";
}

std::optional<unsigned> tryParseBitsWidth(llvm::StringRef typeStr) {
  if (!typeStr.consume_front("!fabric.bits<") || !typeStr.consume_back(">"))
    return std::nullopt;
  unsigned width = 0;
  if (typeStr.getAsInteger(10, width))
    return std::nullopt;
  return width;
}

std::optional<unsigned> inferUniformBitsWidth(
    const std::vector<std::string> &types, unsigned prefixCount) {
  if (prefixCount == 0 || types.size() < prefixCount)
    return std::nullopt;
  auto firstWidth = tryParseBitsWidth(types.front());
  if (!firstWidth)
    return std::nullopt;
  for (unsigned idx = 1; idx < prefixCount; ++idx) {
    auto width = tryParseBitsWidth(types[idx]);
    if (!width || *width != *firstWidth)
      return std::nullopt;
  }
  return firstWidth;
}

void emitFUBody(std::ostringstream &os, const FUDef &fu,
                const std::string &indent) {
  if (!fu.rawBody.empty()) {
    std::string body = fu.rawBody;
    size_t start = 0;
    while (start < body.size()) {
      size_t end = body.find('\n', start);
      std::string line = end == std::string::npos
                             ? body.substr(start)
                             : body.substr(start, end - start);
      os << indent << "  " << line << "\n";
      if (end == std::string::npos)
        break;
      start = end + 1;
    }
    return;
  }
  if (fu.ops.empty() || fu.outputTypes.empty()) {
    os << indent << "  fabric.yield\n";
    return;
  }

  if (fu.ops.size() >= 2) {
    const std::string &op0 = fu.ops[0];
    os << indent << "  %d = " << op0 << " %arg0, %arg1 : "
       << fu.outputTypes[0] << "\n";

    const std::string &op1 = fu.ops[1];
    unsigned nextArg = 2;
    if (nextArg < fu.inputTypes.size()) {
      os << indent << "  %e = " << op1 << " %d, %arg" << nextArg << " : "
         << fu.outputTypes[0] << "\n";
    } else {
      os << indent << "  %e = " << op1 << " %d, %arg1 : "
         << fu.outputTypes[0] << "\n";
    }

    os << indent << "  %g = fabric.mux"
       << " %d, %e"
       << " {sel = 0 : i64, discard = false, disconnect = false}"
       << " : " << fu.outputTypes[0] << ", " << fu.outputTypes[0]
       << " -> " << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %g : " << fu.outputTypes[0] << "\n";
    return;
  }

  const std::string &op = fu.ops[0];

  if (op == "dataflow.stream") {
    os << indent << "  %0, %1 = dataflow.stream %arg0, %arg1, %arg2"
       << " {step_op = \"" << "+=" << "\", cont_cond = \"" << "<" << "\"}"
       << " : (index, index, index) -> (index, i1)\n";
    os << indent << "  fabric.yield %0, %1 : index, i1\n";
    return;
  }

  if (op == "dataflow.gate") {
    os << indent << "  %0, %1 = dataflow.gate %arg0, %arg1 : "
       << fu.inputTypes[0] << ", i1 -> " << fu.outputTypes[0] << ", i1\n";
    os << indent << "  fabric.yield %0, %1 : " << fu.outputTypes[0]
       << ", i1\n";
    return;
  }

  if (op == "dataflow.carry") {
    os << indent << "  %0 = dataflow.carry %arg0, %arg1, %arg2 : "
       << "i1, " << fu.inputTypes[1] << ", " << fu.inputTypes[2] << " -> "
       << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  if (op == "dataflow.invariant") {
    os << indent << "  %0 = dataflow.invariant %arg0, %arg1 : "
       << "i1, " << fu.inputTypes[1] << " -> " << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  if (op == "handshake.load") {
    os << indent << "  %0, %1 = handshake.load [%arg0] %arg1, %arg2 : "
       << fu.inputTypes[0] << ", " << fu.inputTypes[1] << "\n";
    os << indent << "  fabric.yield %0, %1 : " << fu.outputTypes[0] << ", "
       << fu.outputTypes[1] << "\n";
    return;
  }

  if (op == "handshake.store") {
    os << indent << "  %0, %1 = handshake.store [%arg0] %arg1, %arg2 : "
       << fu.inputTypes[0] << ", " << fu.inputTypes[1] << "\n";
    os << indent << "  fabric.yield %0, %1 : " << fu.outputTypes[0] << ", "
       << fu.outputTypes[1] << "\n";
    return;
  }

  if (op == "handshake.constant") {
    os << indent << "  %0 = handshake.constant %arg0 {value = 0 : "
       << fu.outputTypes[0] << "} : " << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  if (op == "handshake.cond_br") {
    os << indent << "  %0, %1 = handshake.cond_br %arg0, %arg1 : "
       << fu.inputTypes[1] << "\n";
    os << indent << "  fabric.yield %0, %1 : " << fu.outputTypes[0] << ", "
       << fu.outputTypes[1] << "\n";
    return;
  }

  if (op == "handshake.mux") {
    os << indent << "  %0 = handshake.mux %arg0 [%arg1, %arg2] : "
       << fu.inputTypes[0] << ", " << fu.inputTypes[1] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  if (op == "handshake.join") {
    os << indent << "  %0 = handshake.join %arg0";
    for (size_t j = 1; j < fu.inputTypes.size(); ++j)
      os << ", %arg" << j;
    os << " : " << fu.inputTypes[0];
    for (size_t j = 1; j < fu.inputTypes.size(); ++j)
      os << ", " << fu.inputTypes[j];
    os << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  if (op == "arith.cmpi") {
    os << indent << "  %0 = arith.cmpi eq, %arg0, %arg1 : " << fu.inputTypes[0]
       << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }
  if (op == "arith.cmpf") {
    os << indent << "  %0 = arith.cmpf oeq, %arg0, %arg1 : "
       << fu.inputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  if (op == "arith.select") {
    os << indent << "  %0 = arith.select %arg0, %arg1, %arg2 : "
       << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  if (op == "arith.extui" || op == "arith.extsi" || op == "arith.trunci" ||
      op == "arith.fptosi" || op == "arith.fptoui" || op == "arith.sitofp" ||
      op == "arith.uitofp") {
    os << indent << "  %0 = " << op << " %arg0 : " << fu.inputTypes[0]
       << " to " << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  if (op == "arith.index_cast" || op == "arith.index_castui") {
    os << indent << "  %0 = " << op << " %arg0 : " << fu.inputTypes[0]
       << " to " << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  if (op == "arith.negf" || op == "math.absf" || op == "math.cos" ||
      op == "math.sin" || op == "math.exp" || op == "math.log2" ||
      op == "math.sqrt") {
    os << indent << "  %0 = " << op << " %arg0 : " << fu.outputTypes[0]
       << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  if (op == "math.fma") {
    os << indent << "  %0 = math.fma %arg0, %arg1, %arg2 : "
       << fu.outputTypes[0] << "\n";
    os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
    return;
  }

  os << indent << "  %0 = " << op << " %arg0, %arg1 : " << fu.outputTypes[0]
     << "\n";
  os << indent << "  fabric.yield %0 : " << fu.outputTypes[0] << "\n";
}

unsigned getInstanceOutputCount(const std::vector<InstanceDef> &instances,
                                const std::vector<PEDef> &peDefs,
                                const std::vector<SWDef> &swDefs,
                                const std::vector<MemoryDef> &memoryDefs,
                                const std::vector<ExtMemDef> &extMemDefs,
                                unsigned instIdx) {
  const auto &inst = instances[instIdx];
  switch (inst.kind) {
  case InstanceKind::PE:
    return peDefs[inst.defIdx].outputTypes.size();
  case InstanceKind::SW:
    return swDefs[inst.defIdx].outputTypes.size();
  case InstanceKind::Memory: {
    const auto &mem = memoryDefs[inst.defIdx];
    unsigned count = mem.ldPorts + mem.stPorts + mem.ldPorts;
    if (!mem.isPrivate)
      count += 1;
    return count;
  }
  case InstanceKind::ExtMem: {
    const auto &mem = extMemDefs[inst.defIdx];
    return mem.ldPorts + mem.stPorts + mem.ldPorts;
  }
  case InstanceKind::FIFO:
    return 1;
  case InstanceKind::AddTag:
  case InstanceKind::MapTag:
  case InstanceKind::DelTag:
    return 1;
  }
  return 0;
}

std::string getInstanceInputType(const std::vector<InstanceDef> &instances,
                                 const std::vector<PEDef> &peDefs,
                                 const std::vector<SWDef> &swDefs,
                                 const std::vector<MemoryDef> &memoryDefs,
                                 const std::vector<AddTagNodeDef> &addTagDefs,
                                 const std::vector<MapTagNodeDef> &mapTagDefs,
                                 const std::vector<DelTagNodeDef> &delTagDefs,
                                 const std::vector<FIFODef> &fifoDefs,
                                 unsigned instIdx, unsigned portIdx) {
  const auto &inst = instances[instIdx];
  switch (inst.kind) {
  case InstanceKind::PE:
    return peDefs[inst.defIdx].inputTypes[portIdx];
  case InstanceKind::SW:
    return swDefs[inst.defIdx].inputTypes[portIdx];
  case InstanceKind::Memory: {
    const auto &mem = memoryDefs[inst.defIdx];
    if (portIdx < mem.ldPorts + mem.stPorts * 2)
      return bitsType(64);
    return mem.memrefType;
  }
  case InstanceKind::FIFO:
    return bitsType(fifoDefs[inst.defIdx].bitsWidth);
  case InstanceKind::ExtMem:
    return bitsType(64);
  case InstanceKind::AddTag:
    return addTagDefs[inst.defIdx].inputType;
  case InstanceKind::MapTag:
    return mapTagDefs[inst.defIdx].inputType;
  case InstanceKind::DelTag:
    return delTagDefs[inst.defIdx].inputType;
  }
  return bitsType(64);
}

std::string getInstanceOutputType(const std::vector<InstanceDef> &instances,
                                  const std::vector<PEDef> &peDefs,
                                  const std::vector<SWDef> &swDefs,
                                  const std::vector<MemoryDef> &memoryDefs,
                                  const std::vector<ExtMemDef> &extMemDefs,
                                  const std::vector<AddTagNodeDef> &addTagDefs,
                                  const std::vector<MapTagNodeDef> &mapTagDefs,
                                  const std::vector<DelTagNodeDef> &delTagDefs,
                                  const std::vector<FIFODef> &fifoDefs,
                                  unsigned instIdx, unsigned portIdx) {
  const auto &inst = instances[instIdx];
  switch (inst.kind) {
  case InstanceKind::PE:
    return peDefs[inst.defIdx].outputTypes[portIdx];
  case InstanceKind::SW:
    return swDefs[inst.defIdx].outputTypes[portIdx];
  case InstanceKind::Memory: {
    const auto &mem = memoryDefs[inst.defIdx];
    if (portIdx < mem.ldPorts + mem.stPorts + mem.ldPorts)
      return bitsType(64);
    return mem.memrefType;
  }
  case InstanceKind::FIFO:
    return bitsType(fifoDefs[inst.defIdx].bitsWidth);
  case InstanceKind::ExtMem: {
    const auto &mem = extMemDefs[inst.defIdx];
    if (portIdx < mem.ldPorts + mem.stPorts + mem.ldPorts)
      return bitsType(64);
    return bitsType(64);
  }
  case InstanceKind::AddTag:
    return addTagDefs[inst.defIdx].outputType;
  case InstanceKind::MapTag:
    return mapTagDefs[inst.defIdx].outputType;
  case InstanceKind::DelTag:
    return delTagDefs[inst.defIdx].outputType;
  }
  return bitsType(64);
}

} // namespace detail

using namespace detail;

ADGBuilder::ADGBuilder(const std::string &moduleName)
    : impl_(std::make_unique<Impl>()) {
  impl_->moduleName = moduleName;
}

ADGBuilder::~ADGBuilder() = default;

FUHandle ADGBuilder::defineFU(const std::string &name,
                              const std::vector<std::string> &inputTypes,
                              const std::vector<std::string> &outputTypes,
                              const std::vector<std::string> &ops,
                              unsigned latency, unsigned interval) {
  unsigned id = impl_->fuDefs.size();
  impl_->fuDefs.push_back(
      {name, inputTypes, outputTypes, ops, "", latency, interval});
  return {id};
}

FUHandle ADGBuilder::defineFU(const FunctionUnitSpec &spec) {
  if (!spec.rawBody.empty())
    return defineFUWithBody(spec.name, spec.inputTypes, spec.outputTypes,
                            spec.rawBody, spec.latency, spec.interval);
  return defineFU(spec.name, spec.inputTypes, spec.outputTypes, spec.ops,
                  spec.latency, spec.interval);
}

FUHandle ADGBuilder::defineFUWithBody(const std::string &name,
                                      const std::vector<std::string> &inputTypes,
                                      const std::vector<std::string> &outputTypes,
                                      const std::string &rawBody,
                                      unsigned latency,
                                      unsigned interval) {
  unsigned id = impl_->fuDefs.size();
  impl_->fuDefs.push_back(
      {name, inputTypes, outputTypes, {}, rawBody, latency, interval});
  return {id};
}

FUHandle ADGBuilder::defineUnaryFU(const std::string &name,
                                   const std::string &opName,
                                   const std::string &inputType,
                                   const std::string &resultType,
                                   unsigned latency,
                                   unsigned interval) {
  return defineFU(name, {inputType}, {resultType}, {opName}, latency,
                  interval);
}

FUHandle ADGBuilder::defineBinaryFU(const std::string &name,
                                    const std::string &opName,
                                    const std::string &operandType,
                                    const std::string &resultType,
                                    unsigned latency,
                                    unsigned interval) {
  return defineBinaryFU(name, opName, operandType, operandType, resultType,
                        latency, interval);
}

FUHandle ADGBuilder::defineBinaryFU(const std::string &name,
                                    const std::string &opName,
                                    const std::string &lhsType,
                                    const std::string &rhsType,
                                    const std::string &resultType,
                                    unsigned latency,
                                    unsigned interval) {
  return defineFU(name, {lhsType, rhsType}, {resultType}, {opName}, latency,
                  interval);
}

FUHandle ADGBuilder::defineConstantFU(const std::string &name,
                                      const std::string &resultType,
                                      const std::string &valueLiteral,
                                      unsigned latency,
                                      unsigned interval) {
  std::string rawBody;
  rawBody += "%0 = handshake.constant %arg0 {value = " + valueLiteral + "} : " +
             resultType + "\n";
  rawBody += "fabric.yield %0 : " + resultType;
  return defineFUWithBody(name, {"none"}, {resultType}, rawBody, latency,
                          interval);
}

FUHandle ADGBuilder::defineCmpiFU(const std::string &name,
                                  const std::string &operandType,
                                  const std::string &predicate,
                                  unsigned latency,
                                  unsigned interval) {
  std::string rawBody;
  rawBody += "%0 = arith.cmpi " + predicate + ", %arg0, %arg1 : " +
             operandType + "\n";
  rawBody += "fabric.yield %0 : i1";
  return defineFUWithBody(name, {operandType, operandType}, {"i1"}, rawBody,
                          latency, interval);
}

FUHandle ADGBuilder::defineCmpfFU(const std::string &name,
                                  const std::string &operandType,
                                  const std::string &predicate,
                                  unsigned latency,
                                  unsigned interval) {
  std::string rawBody;
  rawBody += "%0 = arith.cmpf " + predicate + ", %arg0, %arg1 : " +
             operandType + "\n";
  rawBody += "fabric.yield %0 : i1";
  return defineFUWithBody(name, {operandType, operandType}, {"i1"}, rawBody,
                          latency, interval);
}

FUHandle ADGBuilder::defineStreamFU(const std::string &name,
                                    const std::string &indexType,
                                    const std::string &stepOp,
                                    const std::string &contCond,
                                    unsigned latency,
                                    unsigned interval) {
  std::string rawBody;
  rawBody += "%0, %1 = dataflow.stream %arg0, %arg1, %arg2 {step_op = \"" +
             stepOp + "\", cont_cond = \"" + contCond + "\"} : (" + indexType +
             ", " + indexType + ", " + indexType + ") -> (" + indexType +
             ", i1)\n";
  rawBody += "fabric.yield %0, %1 : " + indexType + ", i1";
  return defineFUWithBody(name, {indexType, indexType, indexType},
                          {indexType, "i1"}, rawBody, latency, interval);
}

FUHandle ADGBuilder::defineIndexCastFU(const std::string &name,
                                       const std::string &inputType,
                                       const std::string &resultType,
                                       unsigned latency,
                                       unsigned interval) {
  return defineFU(name, {inputType}, {resultType}, {"arith.index_cast"},
                  latency, interval);
}

FUHandle ADGBuilder::defineSelectFU(const std::string &name,
                                    const std::string &valueType,
                                    unsigned latency,
                                    unsigned interval) {
  return defineFU(name, {"i1", valueType, valueType}, {valueType},
                  {"arith.select"}, latency, interval);
}

FUHandle ADGBuilder::defineGateFU(const std::string &name,
                                  const std::string &valueType,
                                  unsigned latency,
                                  unsigned interval) {
  return defineFU(name, {valueType, "i1"}, {valueType, "i1"},
                  {"dataflow.gate"}, latency, interval);
}

FUHandle ADGBuilder::defineCarryFU(const std::string &name,
                                   const std::string &valueType,
                                   unsigned latency,
                                   unsigned interval) {
  return defineFU(name, {"i1", valueType, valueType}, {valueType},
                  {"dataflow.carry"}, latency, interval);
}

FUHandle ADGBuilder::defineInvariantFU(const std::string &name,
                                       const std::string &valueType,
                                       unsigned latency,
                                       unsigned interval) {
  return defineFU(name, {"i1", valueType}, {valueType},
                  {"dataflow.invariant"}, latency, interval);
}

FUHandle ADGBuilder::defineCondBrFU(const std::string &name,
                                    const std::string &valueType,
                                    unsigned latency,
                                    unsigned interval) {
  return defineFU(name, {"i1", valueType}, {valueType, valueType},
                  {"handshake.cond_br"}, latency, interval);
}

FUHandle ADGBuilder::defineMuxFU(const std::string &name,
                                 const std::string &valueType,
                                 const std::string &indexType,
                                 unsigned latency,
                                 unsigned interval) {
  return defineFU(name, {indexType, valueType, valueType}, {valueType},
                  {"handshake.mux"}, latency, interval);
}

FUHandle ADGBuilder::defineJoinFU(const std::string &name,
                                  unsigned inputCount,
                                  const std::string &inputType,
                                  unsigned latency,
                                  unsigned interval) {
  std::vector<std::string> inputTypes(inputCount, inputType);
  return defineFU(name, inputTypes, {"none"}, {"handshake.join"}, latency,
                  interval);
}

FUHandle ADGBuilder::defineLoadFU(const std::string &name,
                                  const std::string &addrType,
                                  const std::string &dataType,
                                  unsigned latency,
                                  unsigned interval) {
  return defineFU(name, {addrType, dataType, "none"}, {dataType, addrType},
                  {"handshake.load"}, latency, interval);
}

FUHandle ADGBuilder::defineStoreFU(const std::string &name,
                                   const std::string &addrType,
                                   const std::string &dataType,
                                   unsigned latency,
                                   unsigned interval) {
  return defineFU(name, {addrType, dataType, "none"}, {dataType, addrType},
                  {"handshake.store"}, latency, interval);
}

PEHandle ADGBuilder::defineSpatialPE(const std::string &name, unsigned numInputs,
                                     unsigned numOutputs, unsigned bitsWidth,
                                     const std::vector<FUHandle> &fus) {
  std::vector<std::string> inputTypes(numInputs, bitsType(bitsWidth));
  std::vector<std::string> outputTypes(numOutputs, bitsType(bitsWidth));
  return defineSpatialPE(name, inputTypes, outputTypes, fus);
}

PEHandle ADGBuilder::defineSingleFUSpatialPE(const std::string &name,
                                             unsigned numInputs,
                                             unsigned numOutputs,
                                             unsigned bitsWidth,
                                             FUHandle fu) {
  return defineSpatialPE(name, numInputs, numOutputs, bitsWidth, {fu});
}

PEHandle ADGBuilder::defineSpatialPE(const std::string &name,
                                     const std::vector<std::string> &inputTypes,
                                     const std::vector<std::string> &outputTypes,
                                     const std::vector<FUHandle> &fus) {
  unsigned id = impl_->peDefs.size();
  PEDef pe;
  pe.name = name;
  pe.inputTypes = inputTypes;
  pe.outputTypes = outputTypes;
  for (const auto &fu : fus)
    pe.fuIndices.push_back(fu.id);
  impl_->peDefs.push_back(std::move(pe));
  return {id};
}

PEHandle ADGBuilder::defineSingleFUSpatialPE(
    const std::string &name, const std::vector<std::string> &inputTypes,
    const std::vector<std::string> &outputTypes, FUHandle fu) {
  return defineSpatialPE(name, inputTypes, outputTypes, {fu});
}

PEHandle ADGBuilder::defineSpatialPE(const SpatialPESpec &spec) {
  if (!spec.inputTypes.empty() || !spec.outputTypes.empty())
    return defineSpatialPE(spec.name, spec.inputTypes, spec.outputTypes,
                           spec.functionUnits);
  return defineSpatialPE(spec.name, spec.numInputs, spec.numOutputs,
                         spec.bitsWidth, spec.functionUnits);
}

PEHandle ADGBuilder::defineTemporalPE(
    const std::string &name, const std::vector<std::string> &inputTypes,
    const std::vector<std::string> &outputTypes,
    const std::vector<FUHandle> &fus, unsigned numRegister,
    unsigned numInstruction, unsigned regFifoDepth,
    bool enableShareOperandBuffer,
    std::optional<unsigned> operandBufferSize) {
  unsigned id = impl_->peDefs.size();
  PEDef pe;
  pe.name = name;
  pe.inputTypes = inputTypes;
  pe.outputTypes = outputTypes;
  pe.temporal = true;
  pe.numRegister = numRegister;
  pe.numInstruction = numInstruction;
  pe.regFifoDepth = regFifoDepth;
  pe.enableShareOperandBuffer = enableShareOperandBuffer;
  pe.operandBufferSize = operandBufferSize;
  for (const auto &fu : fus)
    pe.fuIndices.push_back(fu.id);
  impl_->peDefs.push_back(std::move(pe));
  return {id};
}

PEHandle ADGBuilder::defineSingleFUTemporalPE(
    const std::string &name, const std::vector<std::string> &inputTypes,
    const std::vector<std::string> &outputTypes, FUHandle fu,
    unsigned numRegister, unsigned numInstruction, unsigned regFifoDepth,
    bool enableShareOperandBuffer,
    std::optional<unsigned> operandBufferSize) {
  return defineTemporalPE(name, inputTypes, outputTypes, {fu}, numRegister,
                          numInstruction, regFifoDepth,
                          enableShareOperandBuffer, operandBufferSize);
}

PEHandle ADGBuilder::defineTemporalPE(const TemporalPESpec &spec) {
  return defineTemporalPE(
      spec.name, spec.inputTypes, spec.outputTypes, spec.functionUnits,
      spec.numRegister, spec.numInstruction, spec.regFifoDepth,
      spec.enableShareOperandBuffer, spec.operandBufferSize);
}

SWHandle ADGBuilder::defineSpatialSW(const std::string &name,
                                     const std::vector<unsigned> &inputWidths,
                                     const std::vector<unsigned> &outputWidths,
                                     const std::vector<std::vector<bool>> &conn,
                                     int decomposableBits) {
  std::vector<std::string> inputTypes;
  std::vector<std::string> outputTypes;
  inputTypes.reserve(inputWidths.size());
  outputTypes.reserve(outputWidths.size());
  for (unsigned width : inputWidths)
    inputTypes.push_back(bitsType(width));
  for (unsigned width : outputWidths)
    outputTypes.push_back(bitsType(width));
  return defineSpatialSW(name, inputTypes, outputTypes, conn,
                         decomposableBits);
}

SWHandle ADGBuilder::defineSpatialSW(const std::string &name,
                                     const std::vector<std::string> &inputTypes,
                                     const std::vector<std::string> &outputTypes,
                                     const std::vector<std::vector<bool>> &conn,
                                     int decomposableBits) {
  unsigned id = impl_->swDefs.size();
  SWDef sw;
  sw.name = name;
  sw.inputTypes = inputTypes;
  sw.outputTypes = outputTypes;
  sw.connectivity = conn;
  sw.decomposableBits = decomposableBits;
  impl_->swDefs.push_back(std::move(sw));
  return {id};
}

SWHandle ADGBuilder::defineSpatialSW(const SpatialSWSpec &spec) {
  if (!spec.inputTypes.empty() || !spec.outputTypes.empty())
    return defineSpatialSW(spec.name, spec.inputTypes, spec.outputTypes,
                           spec.connectivity, spec.decomposableBits);
  return defineSpatialSW(spec.name, spec.inputWidths, spec.outputWidths,
                         spec.connectivity, spec.decomposableBits);
}

SWHandle ADGBuilder::defineFullCrossbarSpatialSW(const std::string &name,
                                                 unsigned numInputs,
                                                 unsigned numOutputs,
                                                 unsigned bitsWidth,
                                                 int decomposableBits) {
  std::vector<unsigned> inputWidths(numInputs, bitsWidth);
  std::vector<unsigned> outputWidths(numOutputs, bitsWidth);
  std::vector<std::vector<bool>> connectivity(
      numOutputs, std::vector<bool>(numInputs, true));
  return defineSpatialSW(name, inputWidths, outputWidths, connectivity,
                         decomposableBits);
}

SWHandle ADGBuilder::defineTemporalSW(
    const std::string &name, const std::vector<std::string> &inputTypes,
    const std::vector<std::string> &outputTypes,
    const std::vector<std::vector<bool>> &connectivity,
    unsigned numRouteTable) {
  unsigned id = impl_->swDefs.size();
  SWDef sw;
  sw.name = name;
  sw.inputTypes = inputTypes;
  sw.outputTypes = outputTypes;
  sw.connectivity = connectivity;
  sw.temporal = true;
  sw.numRouteTable = numRouteTable;
  impl_->swDefs.push_back(std::move(sw));
  return {id};
}

SWHandle ADGBuilder::defineTemporalSW(const TemporalSWSpec &spec) {
  return defineTemporalSW(spec.name, spec.inputTypes, spec.outputTypes,
                          spec.connectivity, spec.numRouteTable);
}

SWHandle ADGBuilder::defineFullCrossbarTemporalSW(const std::string &name,
                                                  unsigned numInputs,
                                                  unsigned numOutputs,
                                                  const std::string &portType,
                                                  unsigned numRouteTable) {
  std::vector<std::string> inputTypes(numInputs, portType);
  std::vector<std::string> outputTypes(numOutputs, portType);
  std::vector<std::vector<bool>> connectivity(
      numOutputs, std::vector<bool>(numInputs, true));
  return defineTemporalSW(name, inputTypes, outputTypes, connectivity,
                          numRouteTable);
}

MemoryHandle ADGBuilder::defineMemory(const std::string &name,
                                      unsigned ldPorts, unsigned stPorts,
                                      unsigned lsqDepth,
                                      const std::string &memrefType,
                                      bool isPrivate) {
  unsigned id = impl_->memoryDefs.size();
  impl_->memoryDefs.push_back(
      {name, ldPorts, stPorts, lsqDepth, memrefType, isPrivate});
  return {id};
}

MemoryHandle ADGBuilder::defineMemory(const MemorySpec &spec) {
  return defineMemory(spec.name, spec.ldPorts, spec.stPorts, spec.lsqDepth,
                      spec.memrefType, spec.isPrivate);
}

ExtMemHandle ADGBuilder::defineExtMemory(const std::string &name,
                                         unsigned ldPorts, unsigned stPorts,
                                         unsigned lsqDepth) {
  unsigned id = impl_->extMemDefs.size();
  impl_->extMemDefs.push_back(
      {name, ldPorts, stPorts, lsqDepth, "memref<?xi32>"});
  return {id};
}

ExtMemHandle ADGBuilder::defineExtMemory(const ExtMemorySpec &spec) {
  unsigned id = impl_->extMemDefs.size();
  impl_->extMemDefs.push_back(
      {spec.name, spec.ldPorts, spec.stPorts, spec.lsqDepth, spec.memrefType});
  return {id};
}

FIFOHandle ADGBuilder::defineFIFO(const std::string &name, unsigned depth,
                                  unsigned bitsWidth) {
  unsigned id = impl_->fifoDefs.size();
  impl_->fifoDefs.push_back({name, depth, bitsWidth});
  return {id};
}

InstanceHandle ADGBuilder::instantiatePE(PEHandle pe,
                                         const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({InstanceKind::PE, pe.id, instanceName});
  return {id};
}

std::vector<InstanceHandle> ADGBuilder::instantiatePEArray(
    unsigned count, PEHandle pe, const std::string &prefix) {
  return instantiatePEArray(
      count, [pe](unsigned) { return pe; }, prefix);
}

std::vector<InstanceHandle> ADGBuilder::instantiatePEArray(
    unsigned count, const std::function<PEHandle(unsigned)> &peSelector,
    const std::string &prefix) {
  std::vector<InstanceHandle> instances;
  instances.reserve(count);
  for (unsigned idx = 0; idx < count; ++idx)
    instances.push_back(
        instantiatePE(peSelector(idx), prefix + "_" + std::to_string(idx)));
  return instances;
}

std::vector<std::vector<InstanceHandle>> ADGBuilder::instantiatePEGrid(
    unsigned rows, unsigned cols, PEHandle pe, const std::string &prefix) {
  return instantiatePEGrid(
      rows, cols, [pe](unsigned, unsigned) { return pe; }, prefix);
}

std::vector<std::vector<InstanceHandle>> ADGBuilder::instantiatePEGrid(
    unsigned rows, unsigned cols,
    const std::function<PEHandle(unsigned, unsigned)> &peSelector,
    const std::string &prefix) {
  std::vector<std::vector<InstanceHandle>> grid(
      rows, std::vector<InstanceHandle>(cols));
  for (unsigned row = 0; row < rows; ++row) {
    for (unsigned col = 0; col < cols; ++col) {
      grid[row][col] = instantiatePE(peSelector(row, col),
                                     prefix + "_" + std::to_string(row) + "_" +
                                         std::to_string(col));
    }
  }
  return grid;
}

InstanceHandle ADGBuilder::instantiateSW(SWHandle sw,
                                         const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({InstanceKind::SW, sw.id, instanceName});
  return {id};
}

std::vector<InstanceHandle> ADGBuilder::instantiateSWArray(
    unsigned count, SWHandle sw, const std::string &prefix) {
  return instantiateSWArray(
      count, [sw](unsigned) { return sw; }, prefix);
}

std::vector<InstanceHandle> ADGBuilder::instantiateSWArray(
    unsigned count, const std::function<SWHandle(unsigned)> &swSelector,
    const std::string &prefix) {
  std::vector<InstanceHandle> instances;
  instances.reserve(count);
  for (unsigned idx = 0; idx < count; ++idx)
    instances.push_back(
        instantiateSW(swSelector(idx), prefix + "_" + std::to_string(idx)));
  return instances;
}

std::vector<std::vector<InstanceHandle>> ADGBuilder::instantiateSWGrid(
    unsigned rows, unsigned cols, SWHandle sw, const std::string &prefix) {
  return instantiateSWGrid(
      rows, cols, [sw](unsigned, unsigned) { return sw; }, prefix);
}

std::vector<std::vector<InstanceHandle>> ADGBuilder::instantiateSWGrid(
    unsigned rows, unsigned cols,
    const std::function<SWHandle(unsigned, unsigned)> &swSelector,
    const std::string &prefix) {
  std::vector<std::vector<InstanceHandle>> grid(
      rows, std::vector<InstanceHandle>(cols));
  for (unsigned row = 0; row < rows; ++row) {
    for (unsigned col = 0; col < cols; ++col) {
      grid[row][col] = instantiateSW(swSelector(row, col),
                                     prefix + "_" + std::to_string(row) + "_" +
                                         std::to_string(col));
    }
  }
  return grid;
}

InstanceHandle ADGBuilder::instantiateMemory(MemoryHandle mem,
                                             const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({InstanceKind::Memory, mem.id, instanceName});
  return {id};
}

std::vector<InstanceHandle> ADGBuilder::instantiateMemoryArray(
    unsigned count, MemoryHandle mem, const std::string &prefix) {
  return instantiateMemoryArray(
      count, [mem](unsigned) { return mem; }, prefix);
}

std::vector<InstanceHandle> ADGBuilder::instantiateMemoryArray(
    unsigned count, const std::function<MemoryHandle(unsigned)> &memSelector,
    const std::string &prefix) {
  std::vector<InstanceHandle> instances;
  instances.reserve(count);
  for (unsigned idx = 0; idx < count; ++idx)
    instances.push_back(instantiateMemory(
        memSelector(idx), prefix + "_" + std::to_string(idx)));
  return instances;
}

InstanceHandle ADGBuilder::instantiateExtMem(ExtMemHandle mem,
                                             const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({InstanceKind::ExtMem, mem.id, instanceName});
  return {id};
}

std::vector<InstanceHandle> ADGBuilder::instantiateExtMemArray(
    unsigned count, ExtMemHandle mem, const std::string &prefix) {
  return instantiateExtMemArray(
      count, [mem](unsigned) { return mem; }, prefix);
}

std::vector<InstanceHandle> ADGBuilder::instantiateExtMemArray(
    unsigned count, const std::function<ExtMemHandle(unsigned)> &memSelector,
    const std::string &prefix) {
  std::vector<InstanceHandle> instances;
  instances.reserve(count);
  for (unsigned idx = 0; idx < count; ++idx)
    instances.push_back(instantiateExtMem(
        memSelector(idx), prefix + "_" + std::to_string(idx)));
  return instances;
}

InstanceHandle ADGBuilder::instantiateFIFO(FIFOHandle fifo,
                                           const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({InstanceKind::FIFO, fifo.id, instanceName});
  return {id};
}

InstanceHandle ADGBuilder::createAddTag(const std::string &inputType,
                                        const std::string &outputType,
                                        std::uint64_t tag) {
  unsigned defId = impl_->addTagDefs.size();
  impl_->addTagDefs.push_back({inputType, outputType, tag});
  unsigned id = impl_->instances.size();
  impl_->instances.push_back(
      {InstanceKind::AddTag, defId, "__add_tag_" + std::to_string(defId)});
  return {id};
}

std::vector<InstanceHandle> ADGBuilder::createAddTagBank(
    const std::string &inputType, const std::string &outputType,
    const std::vector<std::uint64_t> &tags) {
  std::vector<InstanceHandle> instances;
  instances.reserve(tags.size());
  for (std::uint64_t tag : tags)
    instances.push_back(createAddTag(inputType, outputType, tag));
  return instances;
}

InstanceHandle ADGBuilder::createMapTag(
    const std::string &inputType, const std::string &outputType,
    const std::vector<MapTagEntrySpec> &table) {
  unsigned defId = impl_->mapTagDefs.size();
  impl_->mapTagDefs.push_back({inputType, outputType, table});
  unsigned id = impl_->instances.size();
  impl_->instances.push_back(
      {InstanceKind::MapTag, defId, "__map_tag_" + std::to_string(defId)});
  return {id};
}

InstanceHandle ADGBuilder::createDelTag(const std::string &inputType,
                                        const std::string &outputType) {
  unsigned defId = impl_->delTagDefs.size();
  impl_->delTagDefs.push_back({inputType, outputType});
  unsigned id = impl_->instances.size();
  impl_->instances.push_back(
      {InstanceKind::DelTag, defId, "__del_tag_" + std::to_string(defId)});
  return {id};
}

std::vector<InstanceHandle> ADGBuilder::createDelTagBank(
    const std::string &inputType, const std::string &outputType,
    unsigned count) {
  std::vector<InstanceHandle> instances;
  instances.reserve(count);
  for (unsigned idx = 0; idx < count; ++idx)
    instances.push_back(createDelTag(inputType, outputType));
  return instances;
}

void ADGBuilder::connect(InstanceHandle src, unsigned srcPort,
                         InstanceHandle dst, unsigned dstPort) {
  impl_->connections.push_back({src.id, srcPort, dst.id, dstPort});
}

void ADGBuilder::connect(PortRef src, PortRef dst) {
  connect(src.instance, src.port, dst.instance, dst.port);
}

void ADGBuilder::connectRange(InstanceHandle src, unsigned srcPortBase,
                              InstanceHandle dst, unsigned dstPortBase,
                              unsigned count) {
  for (unsigned idx = 0; idx < count; ++idx)
    connect(src, srcPortBase + idx, dst, dstPortBase + idx);
}

unsigned ADGBuilder::addMemrefInput(const std::string &name,
                                    const std::string &memrefTypeStr) {
  unsigned idx = impl_->memrefInputs.size();
  impl_->memrefInputs.push_back({name, memrefTypeStr});
  return idx;
}

std::vector<unsigned> ADGBuilder::addMemrefInputs(
    const std::string &prefix, unsigned count,
    const std::string &memrefTypeStr) {
  std::vector<unsigned> inputs;
  inputs.reserve(count);
  for (unsigned idx = 0; idx < count; ++idx)
    inputs.push_back(
        addMemrefInput(prefix + "_" + std::to_string(idx), memrefTypeStr));
  return inputs;
}

void ADGBuilder::connectMemrefToExtMem(unsigned memrefIdx,
                                       InstanceHandle extMemInst) {
  impl_->memrefConnections.push_back({memrefIdx, extMemInst.id});
}

unsigned ADGBuilder::addScalarInput(const std::string &name,
                                    unsigned bitsWidth) {
  return addInput(name, bitsType(bitsWidth));
}

std::vector<unsigned> ADGBuilder::addScalarInputs(const std::string &prefix,
                                                  unsigned count,
                                                  unsigned bitsWidth) {
  std::vector<unsigned> inputs;
  inputs.reserve(count);
  for (unsigned idx = 0; idx < count; ++idx)
    inputs.push_back(
        addScalarInput(prefix + "_" + std::to_string(idx), bitsWidth));
  return inputs;
}

unsigned ADGBuilder::addInput(const std::string &name,
                              const std::string &typeStr) {
  unsigned idx = impl_->scalarInputs.size();
  impl_->scalarInputs.push_back({name, typeStr});
  return idx;
}

std::vector<unsigned>
ADGBuilder::addInputs(const std::string &prefix,
                      const std::vector<std::string> &typeStrs) {
  std::vector<unsigned> inputs;
  inputs.reserve(typeStrs.size());
  for (unsigned idx = 0; idx < typeStrs.size(); ++idx)
    inputs.push_back(
        addInput(prefix + "_" + std::to_string(idx), typeStrs[idx]));
  return inputs;
}

unsigned ADGBuilder::addScalarOutput(const std::string &name,
                                     unsigned bitsWidth) {
  return addOutput(name, bitsType(bitsWidth));
}

std::vector<unsigned> ADGBuilder::addScalarOutputs(const std::string &prefix,
                                                   unsigned count,
                                                   unsigned bitsWidth) {
  std::vector<unsigned> outputs;
  outputs.reserve(count);
  for (unsigned idx = 0; idx < count; ++idx)
    outputs.push_back(
        addScalarOutput(prefix + "_" + std::to_string(idx), bitsWidth));
  return outputs;
}

unsigned ADGBuilder::addOutput(const std::string &name,
                               const std::string &typeStr) {
  unsigned idx = impl_->scalarOutputs.size();
  impl_->scalarOutputs.push_back({name, typeStr});
  return idx;
}

std::vector<unsigned>
ADGBuilder::addOutputs(const std::string &prefix,
                       const std::vector<std::string> &typeStrs) {
  std::vector<unsigned> outputs;
  outputs.reserve(typeStrs.size());
  for (unsigned idx = 0; idx < typeStrs.size(); ++idx)
    outputs.push_back(
        addOutput(prefix + "_" + std::to_string(idx), typeStrs[idx]));
  return outputs;
}

void ADGBuilder::connectScalarInputToInstance(unsigned scalarIdx,
                                              InstanceHandle dst,
                                              unsigned dstPort) {
  connectInputToInstance(scalarIdx, dst, dstPort);
}

void ADGBuilder::connectInputVectorToInstance(
    const std::vector<unsigned> &inputIdxs, InstanceHandle dst,
    unsigned dstPortBase) {
  for (unsigned idx = 0; idx < inputIdxs.size(); ++idx)
    connectInputToInstance(inputIdxs[idx], dst, dstPortBase + idx);
}

void ADGBuilder::connectInputToPort(unsigned inputIdx, PortRef dst) {
  connectInputToInstance(inputIdx, dst.instance, dst.port);
}

void ADGBuilder::connectInputToInstance(unsigned inputIdx, InstanceHandle dst,
                                        unsigned dstPort) {
  impl_->scalarToInstConns.push_back({inputIdx, dst.id, dstPort});
}

void ADGBuilder::connectInstanceToScalarOutput(InstanceHandle src,
                                               unsigned srcPort,
                                               unsigned scalarOutputIdx) {
  connectInstanceToOutput(src, srcPort, scalarOutputIdx);
}

void ADGBuilder::connectInstanceToOutputVector(
    InstanceHandle src, unsigned srcPortBase,
    const std::vector<unsigned> &outputIdxs) {
  for (unsigned idx = 0; idx < outputIdxs.size(); ++idx)
    connectInstanceToOutput(src, srcPortBase + idx, outputIdxs[idx]);
}

void ADGBuilder::connectPortToOutput(PortRef src, unsigned outputIdx) {
  connectInstanceToOutput(src.instance, src.port, outputIdx);
}

void ADGBuilder::connectInstanceToOutput(InstanceHandle src, unsigned srcPort,
                                         unsigned outputIdx) {
  impl_->instToScalarConns.push_back({src.id, srcPort, outputIdx});
}

SwitchPortCursor ADGBuilder::connectPEBankToSwitch(
    InstanceHandle sw, const std::vector<InstanceHandle> &peInstances,
    unsigned peInputCount, unsigned peOutputCount, SwitchPortCursor cursor) {
  for (InstanceHandle peInst : peInstances) {
    for (unsigned port = 0; port < peOutputCount; ++port)
      connect(peInst, port, sw, cursor.nextInputPort++);
    for (unsigned port = 0; port < peInputCount; ++port)
      connect(sw, cursor.nextOutputPort++, peInst, port);
  }
  return cursor;
}

void ADGBuilder::associateExtMemWithSW(InstanceHandle extMem, InstanceHandle sw,
                                       unsigned swInputPortBase,
                                       unsigned swOutputPortBase) {
  const auto &extMemInst = impl_->instances[extMem.id];
  assert(extMemInst.kind == InstanceKind::ExtMem);
  const auto &mem = impl_->extMemDefs[extMemInst.defIdx];

  unsigned numExtMemOutputs = mem.ldPorts + mem.stPorts + mem.ldPorts;
  for (unsigned p = 0; p < numExtMemOutputs; ++p)
    impl_->connections.push_back({extMem.id, p, sw.id, swInputPortBase + p});

  unsigned numExtMemDataInputs = mem.stPorts * 2 + mem.ldPorts;
  for (unsigned p = 0; p < numExtMemDataInputs; ++p)
    impl_->connections.push_back({sw.id, swOutputPortBase + p, extMem.id, 1 + p});
}

SwitchPortCursor ADGBuilder::associateExtMemBankWithSW(
    const std::vector<InstanceHandle> &extMems, InstanceHandle sw,
    unsigned swInputPortsPerExtMem, unsigned swOutputPortsPerExtMem,
    SwitchPortCursor cursor) {
  for (InstanceHandle extMem : extMems) {
    associateExtMemWithSW(extMem, sw, cursor.nextInputPort,
                          cursor.nextOutputPort);
    cursor.nextInputPort += swInputPortsPerExtMem;
    cursor.nextOutputPort += swOutputPortsPerExtMem;
  }
  return cursor;
}

void ADGBuilder::associateMemoryWithSW(InstanceHandle memory, InstanceHandle sw,
                                       unsigned swInputPortBase,
                                       unsigned swOutputPortBase) {
  const auto &memInst = impl_->instances[memory.id];
  assert(memInst.kind == InstanceKind::Memory);
  const auto &mem = impl_->memoryDefs[memInst.defIdx];

  unsigned numMemoryOutputs = mem.ldPorts + mem.stPorts + mem.ldPorts;
  for (unsigned p = 0; p < numMemoryOutputs; ++p)
    impl_->connections.push_back({memory.id, p, sw.id, swInputPortBase + p});

  unsigned numMemoryInputs = mem.ldPorts + mem.stPorts * 2;
  for (unsigned p = 0; p < numMemoryInputs; ++p)
    impl_->connections.push_back({sw.id, swOutputPortBase + p, memory.id, p});
}

SwitchPortCursor ADGBuilder::associateMemoryBankWithSW(
    const std::vector<InstanceHandle> &memories, InstanceHandle sw,
    unsigned swInputPortsPerMemory, unsigned swOutputPortsPerMemory,
    SwitchPortCursor cursor) {
  for (InstanceHandle memory : memories) {
    associateMemoryWithSW(memory, sw, cursor.nextInputPort,
                          cursor.nextOutputPort);
    cursor.nextInputPort += swInputPortsPerMemory;
    cursor.nextOutputPort += swOutputPortsPerMemory;
  }
  return cursor;
}

SwitchBankDomainResult
ADGBuilder::buildSwitchBankDomain(const SwitchBankDomainSpec &spec) {
  assert(spec.pe.has_value() &&
         "buildSwitchBankDomain requires a PE template or selector");
  return buildSwitchBankDomain(
      spec, [pe = *spec.pe](unsigned) { return pe; });
}

SwitchBankDomainResult ADGBuilder::buildSwitchBankDomain(
    const SwitchBankDomainSpec &spec,
    const std::function<PEHandle(unsigned)> &peSelector) {
  SwitchBankDomainResult result{};
  result.sw = instantiateSW(spec.sw, spec.switchInstanceName);

  if (spec.numPEs > 0) {
    result.peInstances.reserve(spec.numPEs);
    for (unsigned idx = 0; idx < spec.numPEs; ++idx) {
      auto pe = peSelector(idx);
      result.peInstances.push_back(
          instantiatePE(pe, spec.pePrefix + "_" + std::to_string(idx)));
    }
    result.cursor = connectPEBankToSwitch(result.sw, result.peInstances,
                                          spec.peInputCount,
                                          spec.peOutputCount, result.cursor);
  }

  if (spec.extMem.has_value() && spec.numExtMems > 0) {
    result.extMemInstances = instantiateExtMemArray(
        spec.numExtMems, *spec.extMem, spec.extMemPrefix);
    if (spec.addExtMemrefInputs) {
      result.extMemrefInputs = addMemrefInputs(spec.extMemrefPrefix,
                                               spec.numExtMems,
                                               spec.extMemrefType);
      for (unsigned idx = 0; idx < result.extMemInstances.size(); ++idx)
        connectMemrefToExtMem(result.extMemrefInputs[idx],
                              result.extMemInstances[idx]);
    }
    result.cursor = associateExtMemBankWithSW(
        result.extMemInstances, result.sw, spec.swInputPortsPerExtMem,
        spec.swOutputPortsPerExtMem, result.cursor);
  }

  if (spec.memory.has_value() && spec.numMemories > 0) {
    result.memoryInstances = instantiateMemoryArray(
        spec.numMemories, *spec.memory, spec.memoryPrefix);
    result.cursor = associateMemoryBankWithSW(
        result.memoryInstances, result.sw, spec.swInputPortsPerMemory,
        spec.swOutputPortsPerMemory, result.cursor);
  }

  result.scalarInputs =
      addInputs(spec.scalarInputPrefix, spec.scalarInputTypes);
  result.scalarOutputs =
      addOutputs(spec.scalarOutputPrefix, spec.scalarOutputTypes);
  connectInputVectorToInstance(result.scalarInputs, result.sw,
                               result.cursor.nextInputPort);
  connectInstanceToOutputVector(result.sw, result.cursor.nextOutputPort,
                                result.scalarOutputs);
  result.cursor.nextInputPort += result.scalarInputs.size();
  result.cursor.nextOutputPort += result.scalarOutputs.size();

  if (spec.memory.has_value() && spec.numMemories > 0) {
    const auto &memoryDef = impl_->memoryDefs[spec.memory->id];
    if (!memoryDef.isPrivate && spec.addMemoryMemrefOutputs) {
      std::vector<std::string> memTypes(result.memoryInstances.size(),
                                        memoryDef.memrefType);
      result.memoryMemrefOutputs =
          addOutputs(spec.memoryMemrefOutputPrefix, memTypes);
      const unsigned memrefPort = memoryDef.ldPorts + memoryDef.stPorts +
                                  memoryDef.ldPorts;
      for (unsigned idx = 0; idx < result.memoryInstances.size(); ++idx)
        connectInstanceToOutput(result.memoryInstances[idx], memrefPort,
                                result.memoryMemrefOutputs[idx]);
    }
  }

  return result;
}

void ADGBuilder::setInstanceVizPosition(InstanceHandle inst, double centerX,
                                        double centerY, int gridRow,
                                        int gridCol) {
  impl_->vizPlacements[inst.id] = {centerX, centerY, gridRow, gridCol};
}

bool ADGBuilder::Impl::validate(std::string &errMsg) const {
  bool valid = true;
  std::ostringstream errs;

  std::map<unsigned, std::set<unsigned>> connectedInputs;
  std::map<unsigned, std::set<unsigned>> connectedOutputs;
  for (const auto &conn : connections) {
    connectedInputs[conn.dstInst].insert(conn.dstPort);
    connectedOutputs[conn.srcInst].insert(conn.srcPort);
  }

  for (const auto &sc : scalarToInstConns)
    connectedInputs[sc.dstInst].insert(sc.dstPort);
  for (const auto &ic : instToScalarConns)
    connectedOutputs[ic.srcInst].insert(ic.srcPort);
  for (const auto &mc : memrefConnections)
    connectedInputs[mc.instIdx].insert(0);

  for (size_t i = 0; i < instances.size(); ++i) {
    const auto &inst = instances[i];
    unsigned numIn = 0;
    unsigned numOut = 0;

    switch (inst.kind) {
    case InstanceKind::PE: {
      const auto &pe = peDefs[inst.defIdx];
      numIn = pe.inputTypes.size();
      numOut = pe.outputTypes.size();
      break;
    }
    case InstanceKind::SW: {
      const auto &sw = swDefs[inst.defIdx];
      numIn = sw.inputTypes.size();
      numOut = sw.outputTypes.size();
      break;
    }
    case InstanceKind::Memory: {
      const auto &mem = memoryDefs[inst.defIdx];
      numIn = mem.ldPorts + mem.stPorts * 2;
      numOut = mem.ldPorts + mem.stPorts + mem.ldPorts +
               (mem.isPrivate ? 0 : 1);
      break;
    }
    case InstanceKind::ExtMem: {
      const auto &mem = extMemDefs[inst.defIdx];
      numIn = 1;
      for (unsigned s = 0; s < mem.stPorts; ++s)
        numIn += 2;
      for (unsigned l = 0; l < mem.ldPorts; ++l)
        numIn += 1;
      numOut = mem.ldPorts + mem.stPorts + mem.ldPorts;
      break;
    }
    case InstanceKind::FIFO:
      numIn = 1;
      numOut = 1;
      break;
    case InstanceKind::AddTag:
    case InstanceKind::MapTag:
    case InstanceKind::DelTag:
      numIn = 1;
      numOut = 1;
      break;
    }

    for (unsigned p = 0; p < numIn; ++p) {
      if (connectedInputs[i].find(p) == connectedInputs[i].end()) {
        errs << "  dangling input: " << inst.name << " port " << p << "\n";
        valid = false;
      }
    }

    for (unsigned p = 0; p < numOut; ++p) {
      if (connectedOutputs[i].find(p) == connectedOutputs[i].end()) {
        errs << "  dangling output: " << inst.name << " port " << p << "\n";
        valid = false;
      }
    }
  }

  errMsg = errs.str();
  return valid;
}

} // namespace adg
} // namespace fcc
