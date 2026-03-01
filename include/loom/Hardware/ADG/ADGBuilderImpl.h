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

#include "loom/adg.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace loom {
namespace adg {

/// Default address width in bits for index type ports (load/store addresses).
/// Centralized here so that all code-gen paths share the same value.
static constexpr unsigned DEFAULT_ADDR_WIDTH = 64;

//===----------------------------------------------------------------------===//
// Helper: get data width in bits for a Type (accessible from multiple TUs)
//===----------------------------------------------------------------------===//

inline unsigned getTypeDataWidth(const Type &t) {
  switch (t.getKind()) {
  case Type::I1:    return 1;
  case Type::I8:    return 8;
  case Type::I16:   return 16;
  case Type::I32:   return 32;
  case Type::I64:   return 64;
  case Type::IN:    return t.getWidth();
  case Type::BF16:  return 16;
  case Type::F16:   return 16;
  case Type::F32:   return 32;
  case Type::F64:   return 64;
  case Type::Index: return 64;
  case Type::None:  return 0;
  case Type::Bits:  return t.getWidth();
  case Type::Tagged:
    return getTypeDataWidth(t.getValueType());
  }
  return 32;
}

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
  Fifo,
};

/// Check whether a module kind is a pure routing node (pass-through, no
/// computation). Used to decide whether routingCompatible() should be applied
/// instead of exact type matching for connected ports.
inline bool isRoutingKind(ModuleKind k) {
  return k == ModuleKind::Switch || k == ModuleKind::TemporalSwitch ||
         k == ModuleKind::Fifo;
}

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
  int comparePredicate = 0; // MLIR predicate encoding for cmpi/cmpf
  int streamStepOp = 0;    // step operation for dataflow.stream (0=+=, 1=-=, ...)
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
  bool sharedOperandBuffer = false;
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
  unsigned numRegion = 1;
  MemrefType shape = MemrefType::static1D(64, Type::i32());
};

struct ExtMemoryDef {
  std::string name;
  unsigned ldCount = 1;
  unsigned stCount = 0;
  unsigned lsqDepth = 0;
  unsigned numRegion = 1;
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

struct FifoDef {
  std::string name;
  unsigned depth = 2;
  bool bypassable = false;
  Type elementType = Type::i32();
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

/// Internal port type representation that can encode either a scalar Type
/// or a MemrefType. Used for type checking in validation and generation.
struct PortType {
  Type scalarType = Type::i32();
  bool isMemref = false;
  MemrefType memrefType = MemrefType::dynamic1D(Type::i32());

  static PortType scalar(Type t) { return {t, false, MemrefType::dynamic1D(Type::i32())}; }
  static PortType memref(MemrefType m) { return {Type::index(), true, m}; }

  std::string toMLIR() const {
    return isMemref ? memrefType.toMLIR() : scalarType.toMLIR();
  }

  bool matches(const PortType &other) const {
    if (isMemref != other.isMemref) return false;
    if (isMemref)
      return memrefType.toMLIR() == other.memrefType.toMLIR();
    return scalarType == other.scalarType;
  }

  /// Width-compatible check: allows tagged types with the same tag type but
  /// different value widths. Used for module-port-to-instance connections
  /// where the SV generator handles width adaptation via emitDataAssign.
  bool widthCompatible(const PortType &other) const {
    if (matches(other)) return true;
    if (isMemref || other.isMemref) return false;
    if (scalarType.getKind() == Type::Tagged &&
        other.scalarType.getKind() == Type::Tagged) {
      return scalarType.getTagType() == other.scalarType.getTagType();
    }
    return false;
  }

  /// Routing-compatible check: allows types with the same bit width to
  /// connect through routing nodes (switches, FIFOs). For example, f32 and
  /// i32 both have 32-bit width and can share a width-merged switch plane.
  /// Index and None types require exact match (never merged with others).
  /// Tagged types: value bit widths AND tag types must both match.
  /// Bits types: bits<N> is compatible with any native type of width N
  /// and with other bits<N>. For tagged: tagged<bits<N>,tagT> is
  /// compatible with tagged<native(N),tagT> when tag types match.
  bool routingCompatible(const PortType &other) const {
    if (matches(other)) return true;
    if (isMemref || other.isMemref) return false;
    // Tagged types: both must be tagged with matching tag types and
    // routing-compatible value widths.
    bool tagA = scalarType.getKind() == Type::Tagged;
    bool tagB = other.scalarType.getKind() == Type::Tagged;
    if (tagA != tagB) return false; // mixed tagged/native: never compatible
    if (tagA) {
      Type valA = scalarType.getValueType();
      Type valB = other.scalarType.getValueType();
      // Index and None value types: exact match only (unless paired with Bits).
      bool bitsA = valA.getKind() == Type::Bits;
      bool bitsB = valB.getKind() == Type::Bits;
      if (!bitsA && !bitsB) {
        if (valA.getKind() == Type::Index || valB.getKind() == Type::Index)
          return false;
        if (valA.getKind() == Type::None || valB.getKind() == Type::None)
          return false;
      }
      return getTypeDataWidth(valA) == getTypeDataWidth(valB) &&
             scalarType.getTagType() == other.scalarType.getTagType();
    }
    // Bits type: compatible with any type of matching width.
    bool bitsA = scalarType.getKind() == Type::Bits;
    bool bitsB = other.scalarType.getKind() == Type::Bits;
    if (bitsA || bitsB) {
      // Bits never matches None (width 0) or Index (semantically distinct).
      if (!bitsA && (scalarType.getKind() == Type::Index ||
                     scalarType.getKind() == Type::None))
        return false;
      if (!bitsB && (other.scalarType.getKind() == Type::Index ||
                     other.scalarType.getKind() == Type::None))
        return false;
      return getTypeDataWidth(scalarType) == getTypeDataWidth(other.scalarType);
    }
    // Index: exact match only (semantically distinct from i64).
    if (scalarType.getKind() == Type::Index ||
        other.scalarType.getKind() == Type::Index)
      return false;
    // None: exact match only.
    if (scalarType.getKind() == Type::None ||
        other.scalarType.getKind() == Type::None)
      return false;
    // Other scalars: bit width must match (e.g. i32 == f32 == 32).
    return getTypeDataWidth(scalarType) == getTypeDataWidth(other.scalarType);
  }
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
  std::vector<FifoDef> fifoDefs;

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
  /// Get the output port type including memref distinction.
  PortType getInstanceOutputPortType(unsigned instIdx, int port) const;
  /// Get the input port type including memref distinction.
  PortType getInstanceInputPortType(unsigned instIdx, int port) const;

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

  /// Generate SystemVerilog output into the given directory.
  void generateSV(const std::string &directory) const;
};

} // namespace adg
} // namespace loom

#endif // LOOM_HARDWARE_ADGBUILDERIMPL_H
