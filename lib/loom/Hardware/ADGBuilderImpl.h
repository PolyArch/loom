//===-- ADGBuilderImpl.h - ADG Builder internal structures --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Internal implementation details for ADGBuilder. Not part of the public API.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_HARDWARE_ADGBUILDERIMPL_H
#define LOOM_HARDWARE_ADGBUILDERIMPL_H

#include "loom/Hardware/adg.h"

#include <cassert>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Module Definition Kinds
//===----------------------------------------------------------------------===//

enum class ModuleKind {
  PE,
  Switch,
  TemporalPE,
  TemporalSwitch,
  Memory,
  ExtMemory,
  AddTag,
  MapTag,
  DelTag,
  ConstantPE,
  LoadPE,
  StorePE,
};

//===----------------------------------------------------------------------===//
// Internal Definition Structs
//===----------------------------------------------------------------------===//

struct PEDef {
  std::string name;
  int16_t latMin = 1, latTyp = 1, latMax = 1;
  int16_t intMin = 1, intTyp = 1, intMax = 1;
  std::vector<Type> inputPorts;
  std::vector<Type> outputPorts;
  std::string bodyMLIR;
  std::string singleOp;
  InterfaceCategory interface = InterfaceCategory::Native;
};

struct ConstantPEDef {
  std::string name;
  int16_t latMin = 0, latTyp = 0, latMax = 0;
  int16_t intMin = 1, intTyp = 1, intMax = 1;
  Type outputType = Type::i32();
};

struct LoadPEDef {
  std::string name;
  Type dataType = Type::i32();
  InterfaceCategory interface = InterfaceCategory::Native;
  unsigned tagWidth = 4;
  unsigned queueDepth = 0;
  HardwareType hwType = HardwareType::TagOverwrite;
};

struct StorePEDef {
  std::string name;
  Type dataType = Type::i32();
  InterfaceCategory interface = InterfaceCategory::Native;
  unsigned tagWidth = 4;
  unsigned queueDepth = 0;
  HardwareType hwType = HardwareType::TagOverwrite;
};

struct SwitchDef {
  std::string name;
  unsigned numIn = 2;
  unsigned numOut = 2;
  Type portType = Type::i32();
  std::vector<std::vector<bool>> connectivity; // [numOut][numIn], empty = full
};

struct TemporalPEDef {
  std::string name;
  unsigned numRegisters = 0;
  unsigned numInstructions = 2;
  unsigned regFifoDepth = 0;
  Type interfaceType = Type::tagged(Type::i32(), Type::iN(4));
  std::vector<unsigned> fuPEDefIndices; // indices into peDefs
  bool shareModeB = false;
  unsigned shareBufferSize = 0;
};

struct TemporalSwitchDef {
  std::string name;
  unsigned numRouteTable = 2;
  unsigned numIn = 2;
  unsigned numOut = 2;
  Type interfaceType = Type::tagged(Type::i32(), Type::iN(4));
  std::vector<std::vector<bool>> connectivity; // empty = full
};

struct MemoryDef {
  std::string name;
  unsigned ldCount = 1;
  unsigned stCount = 0;
  unsigned lsqDepth = 0;
  bool isPrivate = true;
  MemrefType shape = MemrefType::static1D(64, Type::i32());
};

struct ExtMemoryDef {
  std::string name;
  unsigned ldCount = 1;
  unsigned stCount = 0;
  unsigned lsqDepth = 0;
  MemrefType shape = MemrefType::dynamic1D(Type::i32());
};

struct AddTagDef {
  std::string name;
  Type valueType = Type::i32();
  Type tagType = Type::iN(4);
};

struct MapTagDef {
  std::string name;
  Type valueType = Type::i32();
  Type inputTagType = Type::iN(4);
  Type outputTagType = Type::iN(2);
  unsigned tableSize = 4;
};

struct DelTagDef {
  std::string name;
  Type inputType = Type::tagged(Type::i32(), Type::iN(4));
};

//===----------------------------------------------------------------------===//
// Instance and Connection Tracking
//===----------------------------------------------------------------------===//

struct InstanceDef {
  ModuleKind kind;
  unsigned defIdx; // index into the appropriate *Def vector
  std::string name;
};

struct ModulePort {
  std::string name;
  Type type;
  bool isMemref = false;
  MemrefType memrefType = MemrefType::dynamic1D(Type::i32());
  bool isInput;
};

struct InputConn {
  unsigned portIdx;
  unsigned instIdx;
  int dstPort;
};

struct OutputConn {
  unsigned instIdx;
  int srcPort;
  unsigned portIdx;
};

struct InternalConn {
  unsigned srcInst;
  int srcPort;
  unsigned dstInst;
  int dstPort;
};

//===----------------------------------------------------------------------===//
// ADGBuilder::Impl
//===----------------------------------------------------------------------===//

struct ADGBuilder::Impl {
  std::string moduleName;

  // Module definitions by kind.
  std::vector<PEDef> peDefs;
  std::vector<ConstantPEDef> constantPEDefs;
  std::vector<LoadPEDef> loadPEDefs;
  std::vector<StorePEDef> storePEDefs;
  std::vector<SwitchDef> switchDefs;
  std::vector<TemporalPEDef> temporalPEDefs;
  std::vector<TemporalSwitchDef> temporalSwitchDefs;
  std::vector<MemoryDef> memoryDefs;
  std::vector<ExtMemoryDef> extMemoryDefs;
  std::vector<AddTagDef> addTagDefs;
  std::vector<MapTagDef> mapTagDefs;
  std::vector<DelTagDef> delTagDefs;

  // Instances (uniform tracking).
  std::vector<InstanceDef> instances;

  // Module-level ports.
  std::vector<ModulePort> ports;

  // Connections: module-input-to-instance, instance-to-module-output, internal.
  std::vector<InputConn> inputConns;
  std::vector<OutputConn> outputConns;
  std::vector<InternalConn> internalConns;

  /// Get the number of input ports for an instance.
  unsigned getInstanceInputCount(unsigned instIdx) const;
  /// Get the number of output ports for an instance.
  unsigned getInstanceOutputCount(unsigned instIdx) const;
  /// Get the input port type for an instance.
  Type getInstanceInputType(unsigned instIdx, int port) const;
  /// Get the output port type for an instance.
  Type getInstanceOutputType(unsigned instIdx, int port) const;

  /// Generate the PE body MLIR text for a PEDef.
  std::string generatePEBody(const PEDef &pe) const;
  /// Generate a named PE definition MLIR text.
  std::string generatePEDef(const PEDef &pe) const;
  /// Generate a named constant PE definition MLIR text.
  std::string generateConstantPEDef(const ConstantPEDef &def) const;
  /// Generate a named load PE definition MLIR text.
  std::string generateLoadPEDef(const LoadPEDef &def) const;
  /// Generate a named store PE definition MLIR text.
  std::string generateStorePEDef(const StorePEDef &def) const;
  /// Generate a named temporal PE definition MLIR text.
  std::string generateTemporalPEDef(const TemporalPEDef &def) const;

  /// Generate full MLIR text from the internal state.
  std::string generateMLIR() const;
};

} // namespace adg
} // namespace loom

#endif // LOOM_HARDWARE_ADGBUILDERIMPL_H
