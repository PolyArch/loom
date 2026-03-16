//===-- ADGBuilder.h - ADG Builder API ----------------------------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//
//
// C++ API for constructing Architecture Description Graphs (ADGs) and
// exporting them as Fabric MLIR. Uses internal data structures to accumulate
// definitions and connections, then generates MLIR text for parsing/export.
//
//===----------------------------------------------------------------------===//

#ifndef FCC_ADG_ADGBUILDER_H
#define FCC_ADG_ADGBUILDER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fcc {
namespace adg {

//===----------------------------------------------------------------------===//
// Handle Types
//===----------------------------------------------------------------------===//

struct FUHandle { unsigned id; };
struct PEHandle { unsigned id; };
struct SWHandle { unsigned id; };
struct ExtMemHandle { unsigned id; };
struct FIFOHandle { unsigned id; };
struct InstanceHandle { unsigned id; };

//===----------------------------------------------------------------------===//
// Mesh Topology Result
//===----------------------------------------------------------------------===//

struct MeshResult {
  std::vector<std::vector<InstanceHandle>> peGrid;  // [rows][cols]
  std::vector<std::vector<InstanceHandle>> swGrid;   // [rows][cols]
};

//===----------------------------------------------------------------------===//
// ADGBuilder
//===----------------------------------------------------------------------===//

/// Main builder for constructing an ADG and exporting it as Fabric MLIR.
///
/// Usage pattern:
///   1. Create builder with module name
///   2. Define function units (FU templates)
///   3. Define spatial PEs (containing FU references)
///   4. Define spatial switches with connectivity
///   5. Define external memories, FIFOs
///   6. Build mesh topology or manually instantiate + wire
///   7. Export to MLIR file
class ADGBuilder {
public:
  explicit ADGBuilder(const std::string &moduleName);
  ~ADGBuilder();

  ADGBuilder(const ADGBuilder &) = delete;
  ADGBuilder &operator=(const ADGBuilder &) = delete;

  // --- Function Unit definitions ---

  /// Define a function unit template. ops lists the MLIR op names this FU
  /// can execute (e.g. "arith.addi"). For single-op FUs like dataflow ops,
  /// provide exactly one op.
  FUHandle defineFU(const std::string &name,
                    const std::vector<std::string> &inputTypes,
                    const std::vector<std::string> &outputTypes,
                    const std::vector<std::string> &ops,
                    unsigned latency = 1, unsigned interval = 1);

  // --- Spatial PE definitions ---

  /// Define a spatial PE containing the given function units.
  /// numInputs/numOutputs are the PE-level port counts.
  /// bitsWidth is the data width for !fabric.bits<N> ports.
  PEHandle defineSpatialPE(const std::string &name,
                           unsigned numInputs, unsigned numOutputs,
                           unsigned bitsWidth,
                           const std::vector<FUHandle> &fus);

  // --- Spatial Switch definitions ---

  /// Define a spatial switch. inputWidths/outputWidths specify per-port
  /// bit widths. connectivity is [numOutputs][numInputs] boolean matrix.
  /// If connectivity is empty, full crossbar is assumed.
  SWHandle defineSpatialSW(const std::string &name,
                           const std::vector<unsigned> &inputWidths,
                           const std::vector<unsigned> &outputWidths,
                           const std::vector<std::vector<bool>> &connectivity,
                           int decomposableBits = -1);

  // --- External Memory definitions ---

  /// Define an external memory interface with load/store port counts.
  ExtMemHandle defineExtMemory(const std::string &name,
                               unsigned ldPorts, unsigned stPorts,
                               unsigned lsqDepth = 0);

  // --- FIFO definitions ---

  /// Define a FIFO buffer.
  FIFOHandle defineFIFO(const std::string &name,
                        unsigned depth, unsigned bitsWidth);

  // --- Instantiation ---

  /// Create a named instance of a PE template.
  InstanceHandle instantiatePE(PEHandle pe, const std::string &instanceName);

  /// Create a named instance of a switch template.
  InstanceHandle instantiateSW(SWHandle sw, const std::string &instanceName);

  /// Create a named instance of an external memory template.
  InstanceHandle instantiateExtMem(ExtMemHandle mem,
                                   const std::string &instanceName);

  /// Create a named instance of a FIFO template.
  InstanceHandle instantiateFIFO(FIFOHandle fifo,
                                 const std::string &instanceName);

  // --- Wiring ---

  /// Connect a specific output port of src instance to a specific input
  /// port of dst instance.
  void connect(InstanceHandle src, unsigned srcPort,
               InstanceHandle dst, unsigned dstPort);

  // --- Topology helpers ---

  /// Build a rows x cols mesh of PEs and switches. PEs and switches
  /// alternate in a grid pattern. Switches are connected to their NSEW
  /// neighbors. Each PE connects to the switch at its grid position.
  /// Returns the grid of PE and switch instance handles.
  MeshResult buildMesh(unsigned rows, unsigned cols,
                       PEHandle pe, SWHandle sw);

  // --- Module I/O ---

  /// Add a module-level memref input port (for external memory binding).
  unsigned addMemrefInput(const std::string &name,
                          const std::string &memrefTypeStr);

  /// Connect a module memref input to an ext memory instance.
  void connectMemrefToExtMem(unsigned memrefIdx, InstanceHandle extMemInst);

  /// Add a module-level scalar input port (e.g. i32, none).
  /// bitsWidth is the fabric port width for this scalar.
  unsigned addScalarInput(const std::string &name, unsigned bitsWidth);

  /// Add a module-level scalar output port (e.g. none/done token).
  /// bitsWidth is the fabric port width for this output.
  unsigned addScalarOutput(const std::string &name, unsigned bitsWidth);

  /// Connect a scalar input to an instance's input port.
  /// The scalar input value (%scalarN) will be used as the SSA operand.
  void connectScalarInputToInstance(unsigned scalarIdx,
                                   InstanceHandle dst, unsigned dstPort);

  /// Connect an instance's output port to a scalar output.
  /// The instance output will be yielded as the module result.
  void connectInstanceToScalarOutput(InstanceHandle src, unsigned srcPort,
                                    unsigned scalarOutputIdx);

  /// Associate an external memory instance with a nearby switch and create
  /// real SSA connections between them. swInputPortBase is the first SW
  /// input port used for extmem outputs. swOutputPortBase is the first SW
  /// output port used to feed extmem inputs. The SW definition must have
  /// enough ports allocated for these connections.
  void associateExtMemWithSW(InstanceHandle extMem, InstanceHandle sw,
                             unsigned swInputPortBase,
                             unsigned swOutputPortBase);

  // --- Export ---

  /// Export the ADG as Fabric MLIR to the given file path.
  void exportMLIR(const std::string &path);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace adg
} // namespace fcc

#endif // FCC_ADG_ADGBUILDER_H
