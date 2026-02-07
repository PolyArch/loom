//===-- adg.h - ADG Builder API ----------------------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Pure C++ API for constructing Architecture Description Graphs (ADGs) and
// exporting them as Fabric MLIR. Uses the pimpl pattern to avoid exposing
// any MLIR/LLVM headers to user code.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_HARDWARE_ADG_H
#define LOOM_HARDWARE_ADG_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

class ADGBuilder;

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

enum class Topology { Mesh, Torus, DiagonalMesh, DiagonalTorus };

enum class HardwareType { TagOverwrite, TagTransparent };

enum class InterfaceCategory { Native, Tagged };

//===----------------------------------------------------------------------===//
// Type System
//===----------------------------------------------------------------------===//

/// Represents a data type for module ports.
class Type {
public:
  enum Kind {
    I1, I8, I16, I32, I64, IN, BF16, F16, F32, F64, Index, None, Tagged
  };

  static Type i1() { return Type(I1); }
  static Type i8() { return Type(I8); }
  static Type i16() { return Type(I16); }
  static Type i32() { return Type(I32); }
  static Type i64() { return Type(I64); }
  static Type iN(unsigned w) { return Type(IN, w); }
  static Type bf16() { return Type(BF16); }
  static Type f16() { return Type(F16); }
  static Type f32() { return Type(F32); }
  static Type f64() { return Type(F64); }
  static Type index() { return Type(Index); }
  static Type none() { return Type(None); }
  static Type tagged(Type value, Type tag);

  Kind getKind() const { return kind_; }
  unsigned getWidth() const { return width_; }
  bool isTagged() const { return kind_ == Tagged; }
  Type getValueType() const;
  Type getTagType() const;

  /// Returns the MLIR type string (e.g. "i32", "!dataflow.tagged<i32, i4>").
  std::string toMLIR() const;

  bool operator==(const Type &other) const;
  bool operator!=(const Type &other) const { return !(*this == other); }

private:
  explicit Type(Kind k, unsigned w = 0) : kind_(k), width_(w) {}
  Kind kind_;
  unsigned width_ = 0;
  // For tagged types, store value and tag types via heap allocation.
  struct TaggedData;
  std::shared_ptr<TaggedData> taggedData_;
};

struct Type::TaggedData {
  Type valueType;
  Type tagType;
};

/// Represents a memref type for memory shapes.
class MemrefType {
public:
  static MemrefType static1D(unsigned size, Type elemType);
  static MemrefType dynamic1D(Type elemType);

  bool isDynamic() const { return isDynamic_; }
  unsigned getSize() const { return size_; }
  Type getElemType() const { return elemType_; }

  /// Returns the MLIR type string (e.g. "memref<1024xi32>", "memref<?xi32>").
  std::string toMLIR() const;

private:
  MemrefType(bool isDynamic, unsigned size, Type elemType)
      : isDynamic_(isDynamic), size_(size), elemType_(elemType) {}
  bool isDynamic_;
  unsigned size_;
  Type elemType_;
};

//===----------------------------------------------------------------------===//
// Handle Types
//===----------------------------------------------------------------------===//

/// Opaque handle to a named PE definition.
struct PEHandle { unsigned id; };

/// Opaque handle to a component instance.
struct InstanceHandle { unsigned id; };

/// Opaque handle to a module-level port.
struct PortHandle { unsigned id; };

/// Opaque handle to a switch definition.
struct SwitchHandle { unsigned id; };

/// Opaque handle to a temporal PE definition.
struct TemporalPEHandle { unsigned id; };

/// Opaque handle to a temporal switch definition.
struct TemporalSwitchHandle { unsigned id; };

/// Opaque handle to a memory definition.
struct MemoryHandle { unsigned id; };

/// Opaque handle to an external memory definition.
struct ExtMemoryHandle { unsigned id; };

/// Opaque handle to a constant PE definition.
struct ConstantPEHandle { unsigned id; };

/// Opaque handle to a load PE definition.
struct LoadPEHandle { unsigned id; };

/// Opaque handle to a store PE definition.
struct StorePEHandle { unsigned id; };

/// Opaque handle to an add_tag definition.
struct AddTagHandle { unsigned id; };

/// Opaque handle to a map_tag definition.
struct MapTagHandle { unsigned id; };

/// Opaque handle to a del_tag definition.
struct DelTagHandle { unsigned id; };

/// Generic handle to any module definition. Can be implicitly constructed
/// from any typed handle.
struct ModuleHandle {
  enum Kind { PE, Switch, TemporalPE, TemporalSwitch, Memory, ExtMemory,
              ConstantPE, LoadPE, StorePE, AddTag, MapTag, DelTag };
  Kind kind;
  unsigned id;

  ModuleHandle(PEHandle h) : kind(PE), id(h.id) {}
  ModuleHandle(SwitchHandle h) : kind(Switch), id(h.id) {}
  ModuleHandle(TemporalPEHandle h) : kind(TemporalPE), id(h.id) {}
  ModuleHandle(TemporalSwitchHandle h) : kind(TemporalSwitch), id(h.id) {}
  ModuleHandle(MemoryHandle h) : kind(Memory), id(h.id) {}
  ModuleHandle(ExtMemoryHandle h) : kind(ExtMemory), id(h.id) {}
  ModuleHandle(ConstantPEHandle h) : kind(ConstantPE), id(h.id) {}
  ModuleHandle(LoadPEHandle h) : kind(LoadPE), id(h.id) {}
  ModuleHandle(StorePEHandle h) : kind(StorePE), id(h.id) {}
  ModuleHandle(AddTagHandle h) : kind(AddTag), id(h.id) {}
  ModuleHandle(MapTagHandle h) : kind(MapTag), id(h.id) {}
  ModuleHandle(DelTagHandle h) : kind(DelTag), id(h.id) {}
};

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

struct ValidationError {
  std::string code;
  std::string message;
  std::string location;
};

struct ValidationResult {
  bool success = true;
  std::vector<ValidationError> errors;
};

//===----------------------------------------------------------------------===//
// Topology Result
//===----------------------------------------------------------------------===//

struct MeshResult {
  std::vector<std::vector<InstanceHandle>> peGrid;
  std::vector<std::vector<InstanceHandle>> swGrid;
};

//===----------------------------------------------------------------------===//
// Builder Classes
//===----------------------------------------------------------------------===//

/// Builder for configuring a processing element definition.
class PEBuilder {
public:
  PEBuilder &setLatency(int16_t min, int16_t typical, int16_t max);
  PEBuilder &setInterval(int16_t min, int16_t typical, int16_t max);
  PEBuilder &setInputPorts(std::vector<Type> types);
  PEBuilder &setOutputPorts(std::vector<Type> types);
  PEBuilder &addOp(const std::string &opName);
  PEBuilder &setBodyMLIR(const std::string &mlirString);
  PEBuilder &setInterfaceCategory(InterfaceCategory category);

  operator PEHandle() const;

private:
  friend class ADGBuilder;
  PEBuilder(ADGBuilder *builder, unsigned peId);
  ADGBuilder *builder_;
  unsigned peId_;
};

/// Builder for configuring a constant PE definition.
class ConstantPEBuilder {
public:
  ConstantPEBuilder &setLatency(int16_t min, int16_t typical, int16_t max);
  ConstantPEBuilder &setInterval(int16_t min, int16_t typical, int16_t max);
  ConstantPEBuilder &setOutputType(Type type);

  operator ConstantPEHandle() const;

private:
  friend class ADGBuilder;
  ConstantPEBuilder(ADGBuilder *builder, unsigned defId);
  ADGBuilder *builder_;
  unsigned defId_;
};

/// Builder for configuring a load PE definition.
class LoadPEBuilder {
public:
  LoadPEBuilder &setDataType(Type type);
  LoadPEBuilder &setInterfaceCategory(InterfaceCategory category);
  LoadPEBuilder &setTagWidth(unsigned width);
  LoadPEBuilder &setQueueDepth(unsigned depth);
  LoadPEBuilder &setHardwareType(HardwareType type);

  operator LoadPEHandle() const;

private:
  friend class ADGBuilder;
  LoadPEBuilder(ADGBuilder *builder, unsigned defId);
  ADGBuilder *builder_;
  unsigned defId_;
};

/// Builder for configuring a store PE definition.
class StorePEBuilder {
public:
  StorePEBuilder &setDataType(Type type);
  StorePEBuilder &setInterfaceCategory(InterfaceCategory category);
  StorePEBuilder &setTagWidth(unsigned width);
  StorePEBuilder &setQueueDepth(unsigned depth);
  StorePEBuilder &setHardwareType(HardwareType type);

  operator StorePEHandle() const;

private:
  friend class ADGBuilder;
  StorePEBuilder(ADGBuilder *builder, unsigned defId);
  ADGBuilder *builder_;
  unsigned defId_;
};

/// Builder for configuring a switch definition.
class SwitchBuilder {
public:
  SwitchBuilder &setPortCount(unsigned inputs, unsigned outputs);
  SwitchBuilder &setConnectivity(std::vector<std::vector<bool>> table);
  SwitchBuilder &setType(Type type);

  operator SwitchHandle() const;

private:
  friend class ADGBuilder;
  SwitchBuilder(ADGBuilder *builder, unsigned defId);
  ADGBuilder *builder_;
  unsigned defId_;
};

/// Builder for configuring a temporal PE definition.
class TemporalPEBuilder {
public:
  TemporalPEBuilder &setNumRegisters(unsigned n);
  TemporalPEBuilder &setNumInstructions(unsigned n);
  TemporalPEBuilder &setRegFifoDepth(unsigned n);
  TemporalPEBuilder &setInterface(Type taggedType);
  TemporalPEBuilder &addFU(PEHandle pe);
  TemporalPEBuilder &enableShareOperandBuffer(unsigned size);

  operator TemporalPEHandle() const;

private:
  friend class ADGBuilder;
  TemporalPEBuilder(ADGBuilder *builder, unsigned defId);
  ADGBuilder *builder_;
  unsigned defId_;
};

/// Builder for configuring a temporal switch definition.
class TemporalSwitchBuilder {
public:
  TemporalSwitchBuilder &setNumRouteTable(unsigned n);
  TemporalSwitchBuilder &setPortCount(unsigned inputs, unsigned outputs);
  TemporalSwitchBuilder &setConnectivity(std::vector<std::vector<bool>> table);
  TemporalSwitchBuilder &setInterface(Type taggedType);

  operator TemporalSwitchHandle() const;

private:
  friend class ADGBuilder;
  TemporalSwitchBuilder(ADGBuilder *builder, unsigned defId);
  ADGBuilder *builder_;
  unsigned defId_;
};

/// Builder for configuring a memory definition.
class MemoryBuilder {
public:
  MemoryBuilder &setLoadPorts(unsigned count);
  MemoryBuilder &setStorePorts(unsigned count);
  MemoryBuilder &setQueueDepth(unsigned depth);
  MemoryBuilder &setPrivate(bool isPrivate);
  MemoryBuilder &setShape(MemrefType shape);

  operator MemoryHandle() const;

private:
  friend class ADGBuilder;
  MemoryBuilder(ADGBuilder *builder, unsigned defId);
  ADGBuilder *builder_;
  unsigned defId_;
};

/// Builder for configuring an external memory definition.
class ExtMemoryBuilder {
public:
  ExtMemoryBuilder &setLoadPorts(unsigned count);
  ExtMemoryBuilder &setStorePorts(unsigned count);
  ExtMemoryBuilder &setQueueDepth(unsigned depth);
  ExtMemoryBuilder &setShape(MemrefType shape);

  operator ExtMemoryHandle() const;

private:
  friend class ADGBuilder;
  ExtMemoryBuilder(ADGBuilder *builder, unsigned defId);
  ADGBuilder *builder_;
  unsigned defId_;
};

/// Builder for configuring an add_tag operation (auto-instantiated).
class AddTagBuilder {
public:
  AddTagBuilder &setValueType(Type type);
  AddTagBuilder &setTagType(Type type);

  /// Implicitly converts to InstanceHandle (auto-instantiation).
  /// Creates the instance on first conversion; subsequent conversions return
  /// the same handle. Copies share the same cached instance ID.
  operator InstanceHandle() const;

private:
  friend class ADGBuilder;
  AddTagBuilder(ADGBuilder *builder, unsigned defId);
  ADGBuilder *builder_;
  unsigned defId_;
  std::shared_ptr<int> instanceId_;
};

/// Builder for configuring a map_tag operation (auto-instantiated).
class MapTagBuilder {
public:
  MapTagBuilder &setValueType(Type type);
  MapTagBuilder &setInputTagType(Type type);
  MapTagBuilder &setOutputTagType(Type type);
  MapTagBuilder &setTableSize(unsigned size);

  /// Implicitly converts to InstanceHandle (auto-instantiation).
  /// Creates the instance on first conversion; subsequent conversions return
  /// the same handle. Copies share the same cached instance ID.
  operator InstanceHandle() const;

private:
  friend class ADGBuilder;
  MapTagBuilder(ADGBuilder *builder, unsigned defId);
  ADGBuilder *builder_;
  unsigned defId_;
  std::shared_ptr<int> instanceId_;
};

/// Builder for configuring a del_tag operation (auto-instantiated).
class DelTagBuilder {
public:
  DelTagBuilder &setInputType(Type type);

  /// Implicitly converts to InstanceHandle (auto-instantiation).
  /// Creates the instance on first conversion; subsequent conversions return
  /// the same handle. Copies share the same cached instance ID.
  operator InstanceHandle() const;

private:
  friend class ADGBuilder;
  DelTagBuilder(ADGBuilder *builder, unsigned defId);
  ADGBuilder *builder_;
  unsigned defId_;
  std::shared_ptr<int> instanceId_;
};

//===----------------------------------------------------------------------===//
// ADGBuilder
//===----------------------------------------------------------------------===//

/// Main builder for constructing an ADG and exporting it as Fabric MLIR.
class ADGBuilder {
public:
  explicit ADGBuilder(const std::string &moduleName);
  ~ADGBuilder();

  ADGBuilder(const ADGBuilder &) = delete;
  ADGBuilder &operator=(const ADGBuilder &) = delete;

  // --- Module creation methods ---

  /// Create a new named PE definition.
  PEBuilder newPE(const std::string &name);

  /// Create a new constant PE definition.
  ConstantPEBuilder newConstantPE(const std::string &name);

  /// Create a new load PE definition.
  LoadPEBuilder newLoadPE(const std::string &name);

  /// Create a new store PE definition.
  StorePEBuilder newStorePE(const std::string &name);

  /// Create a new switch definition.
  SwitchBuilder newSwitch(const std::string &name);

  /// Create a new temporal PE definition.
  TemporalPEBuilder newTemporalPE(const std::string &name);

  /// Create a new temporal switch definition.
  TemporalSwitchBuilder newTemporalSwitch(const std::string &name);

  /// Create a new memory definition.
  MemoryBuilder newMemory(const std::string &name);

  /// Create a new external memory definition.
  ExtMemoryBuilder newExtMemory(const std::string &name);

  /// Create a new add_tag operation.
  AddTagBuilder newAddTag(const std::string &name);

  /// Create a new map_tag operation.
  MapTagBuilder newMapTag(const std::string &name);

  /// Create a new del_tag operation.
  DelTagBuilder newDelTag(const std::string &name);

  // --- Instantiation ---

  /// Create an instance of an existing PE definition.
  InstanceHandle clone(PEHandle source, const std::string &instanceName);

  /// Create an instance of an existing switch definition.
  InstanceHandle clone(SwitchHandle source, const std::string &instanceName);

  /// Create an instance of an existing temporal PE definition.
  InstanceHandle clone(TemporalPEHandle source,
                       const std::string &instanceName);

  /// Create an instance of an existing temporal switch definition.
  InstanceHandle clone(TemporalSwitchHandle source,
                       const std::string &instanceName);

  /// Create an instance of an existing memory definition.
  InstanceHandle clone(MemoryHandle source, const std::string &instanceName);

  /// Create an instance of an existing external memory definition.
  InstanceHandle clone(ExtMemoryHandle source, const std::string &instanceName);

  /// Create an instance of a constant PE definition.
  InstanceHandle clone(ConstantPEHandle source,
                       const std::string &instanceName);

  /// Create an instance of a load PE definition.
  InstanceHandle clone(LoadPEHandle source, const std::string &instanceName);

  /// Create an instance of a store PE definition.
  InstanceHandle clone(StorePEHandle source, const std::string &instanceName);

  /// Create an instance of any module definition using a generic handle.
  /// Tag-operation handles are not valid clone sources (they auto-instantiate).
  InstanceHandle clone(ModuleHandle source, const std::string &instanceName);

  // --- Internal connections ---

  /// Connect port 0 of src to port 0 of dst.
  void connect(InstanceHandle src, InstanceHandle dst);

  /// Connect specific ports between two instances.
  void connectPorts(InstanceHandle src, int srcPort, InstanceHandle dst,
                    int dstPort);

  // --- Module I/O ---

  /// Add a module-level input port (streaming).
  PortHandle addModuleInput(const std::string &name, Type type);

  /// Add a module-level input port (memory-mapped).
  PortHandle addModuleInput(const std::string &name, MemrefType memrefType);

  /// Add a module-level output port (streaming).
  PortHandle addModuleOutput(const std::string &name, Type type);

  /// Add a module-level output port (memory-mapped).
  PortHandle addModuleOutput(const std::string &name, MemrefType memrefType);

  /// Connect a module input port to an instance's input port.
  void connectToModuleInput(PortHandle port, InstanceHandle dst, int dstPort);

  /// Connect an instance's output port to a module output port.
  void connectToModuleOutput(InstanceHandle src, int srcPort, PortHandle port);

  // --- Topology ---

  /// Build a regular mesh/torus of PEs and switches.
  MeshResult buildMesh(int rows, int cols, PEHandle peTemplate,
                       SwitchHandle swTemplate, Topology topology);

  // --- Query ---

  /// Return all module-level input port names (in creation order).
  std::vector<std::string> getModuleInputNames() const;

  /// Return all module-level output port names (in creation order).
  std::vector<std::string> getModuleOutputNames() const;

  // --- Validation and export ---

  /// Validate the constructed ADG.
  ValidationResult validateADG();

  /// Export the ADG as Fabric MLIR to the given file path.
  void exportMLIR(const std::string &path);

private:
  friend class PEBuilder;
  friend class ConstantPEBuilder;
  friend class LoadPEBuilder;
  friend class StorePEBuilder;
  friend class SwitchBuilder;
  friend class TemporalPEBuilder;
  friend class TemporalSwitchBuilder;
  friend class MemoryBuilder;
  friend class ExtMemoryBuilder;
  friend class AddTagBuilder;
  friend class MapTagBuilder;
  friend class DelTagBuilder;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace adg
} // namespace loom

#endif // LOOM_HARDWARE_ADG_H
