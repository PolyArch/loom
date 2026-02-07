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

/// Represents a data type for PE ports.
class Type {
public:
  enum Kind { I1, I8, I16, I32, I64, IN, BF16, F16, F32, F64, Index, None };

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

  Kind getKind() const { return kind_; }
  unsigned getWidth() const { return width_; }

  /// Returns the MLIR type string (e.g. "i32", "f64", "index").
  std::string toMLIR() const;

private:
  explicit Type(Kind k, unsigned w = 0) : kind_(k), width_(w) {}
  Kind kind_;
  unsigned width_ = 0;
};

/// Opaque handle to a named PE definition.
struct PEHandle {
  unsigned id;
};

/// Opaque handle to a component instance.
struct InstanceHandle {
  unsigned id;
};

/// Opaque handle to a module-level port.
struct PortHandle {
  unsigned id;
};

class ADGBuilder; // forward

/// Builder for configuring a processing element definition.
class PEBuilder {
public:
  PEBuilder &setLatency(int16_t min, int16_t typical, int16_t max);
  PEBuilder &setInterval(int16_t min, int16_t typical, int16_t max);
  PEBuilder &setInputPorts(std::vector<Type> types);
  PEBuilder &setOutputPorts(std::vector<Type> types);
  PEBuilder &addOp(const std::string &opName);
  PEBuilder &setBodyMLIR(const std::string &mlirString);

  /// Implicit conversion to PEHandle for use with clone().
  operator PEHandle() const;

private:
  friend class ADGBuilder;
  PEBuilder(ADGBuilder *builder, unsigned peId);
  ADGBuilder *builder_;
  unsigned peId_;
};

/// Main builder for constructing an ADG and exporting it as Fabric MLIR.
class ADGBuilder {
public:
  explicit ADGBuilder(const std::string &moduleName);
  ~ADGBuilder();

  ADGBuilder(const ADGBuilder &) = delete;
  ADGBuilder &operator=(const ADGBuilder &) = delete;

  /// Create a new named PE definition.
  PEBuilder newPE(const std::string &name);

  /// Create an instance of an existing PE definition.
  InstanceHandle clone(PEHandle source, const std::string &instanceName);

  /// Add a module-level input port.
  PortHandle addModuleInput(const std::string &name, Type type);

  /// Add a module-level output port.
  PortHandle addModuleOutput(const std::string &name, Type type);

  /// Connect a module input port to an instance's input port.
  void connectToModuleInput(PortHandle port, InstanceHandle dst, int dstPort);

  /// Connect an instance's output port to a module output port.
  void connectToModuleOutput(InstanceHandle src, int srcPort, PortHandle port);

  /// Validate the constructed ADG (currently a no-op placeholder).
  void validateADG();

  /// Export the ADG as Fabric MLIR to the given file path.
  void exportMLIR(const std::string &path);

private:
  friend class PEBuilder;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace adg
} // namespace loom

#endif // LOOM_HARDWARE_ADG_H
