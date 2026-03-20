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
#include <functional>
#include <memory>
#include <optional>
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
struct MemoryHandle { unsigned id; };
struct ExtMemHandle { unsigned id; };
struct FIFOHandle { unsigned id; };
struct InstanceHandle { unsigned id; };
struct PortRef {
  InstanceHandle instance;
  unsigned port = 0;
};
struct SwitchPortCursor {
  unsigned nextInputPort = 0;
  unsigned nextOutputPort = 0;
};

struct SwitchBankDomainSpec {
  SWHandle sw;
  std::string switchInstanceName = "sw_0";

  std::optional<PEHandle> pe;
  unsigned numPEs = 0;
  unsigned peInputCount = 0;
  unsigned peOutputCount = 0;
  std::string pePrefix = "pe";

  std::optional<ExtMemHandle> extMem;
  unsigned numExtMems = 0;
  unsigned swInputPortsPerExtMem = 0;
  unsigned swOutputPortsPerExtMem = 0;
  std::string extMemPrefix = "extmem";
  std::string extMemrefPrefix = "mem";
  bool addExtMemrefInputs = true;
  std::string extMemrefType = "memref<?xi32>";

  std::optional<MemoryHandle> memory;
  unsigned numMemories = 0;
  unsigned swInputPortsPerMemory = 0;
  unsigned swOutputPortsPerMemory = 0;
  std::string memoryPrefix = "memory";
  std::string memoryMemrefOutputPrefix = "memory_mem";
  bool addMemoryMemrefOutputs = true;

  std::vector<std::string> scalarInputTypes;
  std::string scalarInputPrefix = "scalar";
  std::vector<std::string> scalarOutputTypes;
  std::string scalarOutputPrefix = "scalar_out";
};

struct SwitchBankDomainResult {
  InstanceHandle sw;
  std::vector<InstanceHandle> peInstances;
  std::vector<InstanceHandle> extMemInstances;
  std::vector<InstanceHandle> memoryInstances;
  std::vector<unsigned> extMemrefInputs;
  std::vector<unsigned> memoryMemrefOutputs;
  std::vector<unsigned> scalarInputs;
  std::vector<unsigned> scalarOutputs;
  SwitchPortCursor cursor;
};

//===----------------------------------------------------------------------===//
// Mesh Topology Result
//===----------------------------------------------------------------------===//

struct MeshResult {
  std::vector<std::vector<InstanceHandle>> peGrid;  // [rows][cols]
  std::vector<std::vector<InstanceHandle>> swGrid;  // topology-specific
  std::vector<PortRef> ingressPorts;
  std::vector<PortRef> egressPorts;
};

struct CubeResult {
  std::vector<std::vector<std::vector<InstanceHandle>>> peGrid; // [depth][row][col]
  std::vector<std::vector<std::vector<InstanceHandle>>> swGrid; // [depth+1][row+1][col+1]
  std::vector<PortRef> ingressPorts;
  std::vector<PortRef> egressPorts;
};

//===----------------------------------------------------------------------===//
// High-level specification structs
//===----------------------------------------------------------------------===//

struct FunctionUnitSpec {
  std::string name;
  std::vector<std::string> inputTypes;
  std::vector<std::string> outputTypes;
  std::vector<std::string> ops;
  std::string rawBody;
  std::int64_t latency = 1;
  std::int64_t interval = 1;
};

struct SpatialPESpec {
  std::string name;
  unsigned numInputs = 0;
  unsigned numOutputs = 0;
  unsigned bitsWidth = 32;
  std::vector<std::string> inputTypes;
  std::vector<std::string> outputTypes;
  std::vector<FUHandle> functionUnits;
};

struct SpatialSWSpec {
  std::string name;
  std::vector<unsigned> inputWidths;
  std::vector<unsigned> outputWidths;
  std::vector<std::string> inputTypes;
  std::vector<std::string> outputTypes;
  std::vector<std::vector<bool>> connectivity;
  int decomposableBits = -1;
};

struct TemporalPESpec {
  std::string name;
  std::vector<std::string> inputTypes;
  std::vector<std::string> outputTypes;
  std::vector<FUHandle> functionUnits;
  unsigned numRegister = 0;
  unsigned numInstruction = 1;
  unsigned regFifoDepth = 0;
  bool enableShareOperandBuffer = false;
  std::optional<unsigned> operandBufferSize;
};

struct TemporalSWSpec {
  std::string name;
  std::vector<std::string> inputTypes;
  std::vector<std::string> outputTypes;
  std::vector<std::vector<bool>> connectivity;
  unsigned numRouteTable = 1;
};

struct MemorySpec {
  std::string name;
  unsigned ldPorts = 1;
  unsigned stPorts = 1;
  unsigned lsqDepth = 0;
  std::string memrefType = "memref<256xi32>";
  unsigned numRegion = 1;
  bool isPrivate = true;
};

struct ExtMemorySpec {
  std::string name;
  unsigned ldPorts = 1;
  unsigned stPorts = 1;
  unsigned lsqDepth = 0;
  std::string memrefType = "memref<?xi32>";
  unsigned numRegion = 1;
};

struct MapTagEntrySpec {
  bool valid = true;
  std::uint64_t srcTag = 0;
  std::uint64_t dstTag = 0;
};

struct ChessMeshOptions {
  int decomposableBits = -1;
  unsigned topLeftExtraInputs = 0;
  unsigned topRightExtraInputs = 0;
  unsigned topRightExtraOutputs = 0;
  unsigned bottomLeftExtraOutputs = 0;
  unsigned bottomRightExtraOutputs = 0;
};

struct LatticeMeshOptions {
  int decomposableBits = -1;
  unsigned topLeftExtraInputs = 0;
  unsigned bottomRightExtraOutputs = 0;
};

struct CubeOptions {
  int decomposableBits = -1;
  unsigned originExtraInputs = 0;
  unsigned farCornerExtraOutputs = 0;
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
                    std::int64_t latency = 1, std::int64_t interval = 1);
  FUHandle defineFU(const FunctionUnitSpec &spec);

  /// Define a function unit whose body is provided as raw MLIR text.
  /// The body should contain the operations inside the function_unit region,
  /// including the terminating fabric.yield.
  FUHandle defineFUWithBody(const std::string &name,
                            const std::vector<std::string> &inputTypes,
                            const std::vector<std::string> &outputTypes,
                            const std::string &rawBody,
                            std::int64_t latency = 1,
                            std::int64_t interval = 1);

  /// Define a single-input single-output function unit for one software op.
  FUHandle defineUnaryFU(const std::string &name, const std::string &opName,
                         const std::string &inputType,
                         const std::string &resultType,
                         std::int64_t latency = 1,
                         std::int64_t interval = 1);

  /// Define a two-input function unit whose operands share one type.
  FUHandle defineBinaryFU(const std::string &name, const std::string &opName,
                          const std::string &operandType,
                          const std::string &resultType,
                          std::int64_t latency = 1,
                          std::int64_t interval = 1);

  /// Define a two-input function unit whose operands may use distinct types.
  FUHandle defineBinaryFU(const std::string &name, const std::string &opName,
                          const std::string &lhsType,
                          const std::string &rhsType,
                          const std::string &resultType,
                          std::int64_t latency = 1,
                          std::int64_t interval = 1);

  /// Define a handshake.constant function unit with a structured literal.
  /// valueLiteral must include both value text and result type, for example
  /// "42 : i32" or "1 : index".
  FUHandle defineConstantFU(const std::string &name,
                            const std::string &resultType,
                            const std::string &valueLiteral,
                            std::int64_t latency = 1,
                            std::int64_t interval = 1);

  /// Define an arith.cmpi function unit with the given predicate mnemonic,
  /// for example "eq", "sgt", or "ult".
  FUHandle defineCmpiFU(const std::string &name,
                        const std::string &operandType,
                        const std::string &predicate,
                        std::int64_t latency = 1,
                        std::int64_t interval = 1);

  /// Define an arith.cmpf function unit with the given predicate mnemonic,
  /// for example "oeq", "ogt", or "une".
  FUHandle defineCmpfFU(const std::string &name,
                        const std::string &operandType,
                        const std::string &predicate,
                        std::int64_t latency = 1,
                        std::int64_t interval = 1);

  /// Define a dataflow.stream function unit with structured stream controls.
  FUHandle defineStreamFU(const std::string &name,
                          const std::string &indexType = "index",
                          const std::string &stepOp = "+=",
                          const std::string &contCond = "<",
                          std::int64_t latency = -1,
                          std::int64_t interval = -1);

  /// Define an arith.index_cast-style function unit.
  FUHandle defineIndexCastFU(const std::string &name,
                             const std::string &inputType,
                             const std::string &resultType = "index",
                             std::int64_t latency = 1,
                             std::int64_t interval = 1);

  /// Define an arith.select function unit.
  FUHandle defineSelectFU(const std::string &name,
                          const std::string &valueType,
                          std::int64_t latency = 1,
                          std::int64_t interval = 1);

  /// Define a dataflow.gate function unit.
  FUHandle defineGateFU(const std::string &name,
                        const std::string &valueType,
                        std::int64_t latency = -1,
                        std::int64_t interval = -1);

  /// Define a dataflow.carry function unit.
  FUHandle defineCarryFU(const std::string &name,
                         const std::string &valueType,
                         std::int64_t latency = -1,
                         std::int64_t interval = -1);

  /// Define a dataflow.invariant function unit.
  FUHandle defineInvariantFU(const std::string &name,
                             const std::string &valueType,
                             std::int64_t latency = -1,
                             std::int64_t interval = -1);

  /// Define a handshake.cond_br function unit.
  FUHandle defineCondBrFU(const std::string &name,
                          const std::string &valueType,
                          std::int64_t latency = 1,
                          std::int64_t interval = 1);

  /// Define a two-way handshake.mux function unit.
  FUHandle defineMuxFU(const std::string &name,
                       const std::string &valueType,
                       const std::string &indexType = "index",
                       std::int64_t latency = 1,
                       std::int64_t interval = 1);

  /// Define a handshake.join function unit.
  FUHandle defineJoinFU(const std::string &name,
                        unsigned inputCount,
                        const std::string &inputType = "none",
                        std::int64_t latency = 1,
                        std::int64_t interval = 1);

  /// Define a handshake.load function unit.
  FUHandle defineLoadFU(const std::string &name,
                        const std::string &addrType,
                        const std::string &dataType,
                        std::int64_t latency = 1,
                        std::int64_t interval = 1);

  /// Define a handshake.store function unit.
  FUHandle defineStoreFU(const std::string &name,
                         const std::string &addrType,
                         const std::string &dataType,
                         std::int64_t latency = 1,
                         std::int64_t interval = 1);

  // --- Spatial PE definitions ---

  /// Define a spatial PE containing the given function units.
  /// numInputs/numOutputs are the PE-level port counts.
  /// bitsWidth is the data width for !fabric.bits<N> ports.
  PEHandle defineSpatialPE(const std::string &name,
                           unsigned numInputs, unsigned numOutputs,
                           unsigned bitsWidth,
                           const std::vector<FUHandle> &fus);
  PEHandle defineSingleFUSpatialPE(const std::string &name,
                                   unsigned numInputs, unsigned numOutputs,
                                   unsigned bitsWidth, FUHandle fu);
  PEHandle defineSpatialPE(const std::string &name,
                           const std::vector<std::string> &inputTypes,
                           const std::vector<std::string> &outputTypes,
                           const std::vector<FUHandle> &fus);
  PEHandle defineSingleFUSpatialPE(const std::string &name,
                                   const std::vector<std::string> &inputTypes,
                                   const std::vector<std::string> &outputTypes,
                                   FUHandle fu);
  PEHandle defineSpatialPE(const SpatialPESpec &spec);

  // --- Temporal PE definitions ---

  /// Define a temporal PE with fully typed boundary ports.
  PEHandle defineTemporalPE(const std::string &name,
                            const std::vector<std::string> &inputTypes,
                            const std::vector<std::string> &outputTypes,
                            const std::vector<FUHandle> &fus,
                            unsigned numRegister = 0,
                            unsigned numInstruction = 1,
                            unsigned regFifoDepth = 0,
                            bool enableShareOperandBuffer = false,
                            std::optional<unsigned> operandBufferSize =
                                std::nullopt);
  PEHandle defineSingleFUTemporalPE(
      const std::string &name, const std::vector<std::string> &inputTypes,
      const std::vector<std::string> &outputTypes, FUHandle fu,
      unsigned numRegister = 0, unsigned numInstruction = 1,
      unsigned regFifoDepth = 0, bool enableShareOperandBuffer = false,
      std::optional<unsigned> operandBufferSize = std::nullopt);
  PEHandle defineTemporalPE(const TemporalPESpec &spec);

  // --- Spatial Switch definitions ---

  /// Define a spatial switch. inputWidths/outputWidths specify per-port
  /// bit widths. connectivity is [numOutputs][numInputs] boolean matrix.
  /// If connectivity is empty, full crossbar is assumed.
  SWHandle defineSpatialSW(const std::string &name,
                           const std::vector<unsigned> &inputWidths,
                           const std::vector<unsigned> &outputWidths,
                           const std::vector<std::vector<bool>> &connectivity,
                           int decomposableBits = -1);
  SWHandle defineSpatialSW(const std::string &name,
                           const std::vector<std::string> &inputTypes,
                           const std::vector<std::string> &outputTypes,
                           const std::vector<std::vector<bool>> &connectivity,
                           int decomposableBits = -1);
  SWHandle defineSpatialSW(const SpatialSWSpec &spec);

  /// Define a full-crossbar spatial switch with uniform per-port bit width.
  SWHandle defineFullCrossbarSpatialSW(const std::string &name,
                                       unsigned numInputs,
                                       unsigned numOutputs,
                                       unsigned bitsWidth,
                                       int decomposableBits = -1);

  // --- Temporal Switch definitions ---

  /// Define a temporal switch with fully typed tagged ports.
  SWHandle defineTemporalSW(const std::string &name,
                            const std::vector<std::string> &inputTypes,
                            const std::vector<std::string> &outputTypes,
                            const std::vector<std::vector<bool>> &connectivity,
                            unsigned numRouteTable = 1);
  SWHandle defineTemporalSW(const TemporalSWSpec &spec);

  /// Define a full-crossbar temporal switch whose ports share one tagged type.
  SWHandle defineFullCrossbarTemporalSW(const std::string &name,
                                        unsigned numInputs,
                                        unsigned numOutputs,
                                        const std::string &portType,
                                        unsigned numRouteTable = 1);

  // --- On-chip memory definitions ---

  /// Define an on-chip memory interface with load/store port counts.
  MemoryHandle defineMemory(const std::string &name,
                            unsigned ldPorts, unsigned stPorts,
                            unsigned lsqDepth = 0,
                            const std::string &memrefType = "memref<256xi32>",
                            unsigned numRegion = 1,
                            bool isPrivate = true);
  MemoryHandle defineMemory(const MemorySpec &spec);

  // --- External Memory definitions ---

  /// Define an external memory interface with load/store port counts.
  ExtMemHandle defineExtMemory(const std::string &name,
                               unsigned ldPorts, unsigned stPorts,
                               unsigned lsqDepth = 0,
                               unsigned numRegion = 1);
  ExtMemHandle defineExtMemory(const ExtMemorySpec &spec);

  // --- FIFO definitions ---

  /// Define a FIFO buffer.
  FIFOHandle defineFIFO(const std::string &name,
                        unsigned depth, unsigned bitsWidth);

  // --- Instantiation ---

  /// Create a named instance of a PE template. The handle may refer to either
  /// a spatial or temporal PE definition.
  InstanceHandle instantiatePE(PEHandle pe, const std::string &instanceName);
  std::vector<InstanceHandle> instantiatePEArray(unsigned count, PEHandle pe,
                                                 const std::string &prefix);
  std::vector<InstanceHandle> instantiatePEArray(
      unsigned count,
      const std::function<PEHandle(unsigned)> &peSelector,
      const std::string &prefix);
  std::vector<std::vector<InstanceHandle>>
  instantiatePEGrid(unsigned rows, unsigned cols, PEHandle pe,
                    const std::string &prefix);
  std::vector<std::vector<InstanceHandle>> instantiatePEGrid(
      unsigned rows, unsigned cols,
      const std::function<PEHandle(unsigned, unsigned)> &peSelector,
      const std::string &prefix);

  /// Create a named instance of a switch template. The handle may refer to
  /// either a spatial or temporal switch definition.
  InstanceHandle instantiateSW(SWHandle sw, const std::string &instanceName);
  std::vector<InstanceHandle> instantiateSWArray(unsigned count, SWHandle sw,
                                                 const std::string &prefix);
  std::vector<InstanceHandle> instantiateSWArray(
      unsigned count,
      const std::function<SWHandle(unsigned)> &swSelector,
      const std::string &prefix);
  std::vector<std::vector<InstanceHandle>> instantiateSWGrid(
      unsigned rows, unsigned cols, SWHandle sw, const std::string &prefix);
  std::vector<std::vector<InstanceHandle>> instantiateSWGrid(
      unsigned rows, unsigned cols,
      const std::function<SWHandle(unsigned, unsigned)> &swSelector,
      const std::string &prefix);

  /// Create a named instance of an on-chip memory template.
  InstanceHandle instantiateMemory(MemoryHandle mem,
                                   const std::string &instanceName);
  std::vector<InstanceHandle> instantiateMemoryArray(unsigned count,
                                                     MemoryHandle mem,
                                                     const std::string &prefix);
  std::vector<InstanceHandle> instantiateMemoryArray(
      unsigned count,
      const std::function<MemoryHandle(unsigned)> &memSelector,
      const std::string &prefix);

  /// Create a named instance of an external memory template.
  InstanceHandle instantiateExtMem(ExtMemHandle mem,
                                   const std::string &instanceName);
  std::vector<InstanceHandle> instantiateExtMemArray(unsigned count,
                                                     ExtMemHandle mem,
                                                     const std::string &prefix);
  std::vector<InstanceHandle> instantiateExtMemArray(
      unsigned count,
      const std::function<ExtMemHandle(unsigned)> &memSelector,
      const std::string &prefix);

  /// Create a named instance of a FIFO template.
  InstanceHandle instantiateFIFO(FIFOHandle fifo,
                                 const std::string &instanceName);

  // --- Wiring ---

  /// Connect a specific output port of src instance to a specific input
  /// port of dst instance.
  void connect(InstanceHandle src, unsigned srcPort,
               InstanceHandle dst, unsigned dstPort);
  void connect(PortRef src, PortRef dst);

  /// Connect count consecutive output ports to consecutive input ports.
  void connectRange(InstanceHandle src, unsigned srcPortBase,
                    InstanceHandle dst, unsigned dstPortBase,
                    unsigned count);

  // --- Inline tag operations inside fabric.module ---

  InstanceHandle createAddTag(const std::string &inputType,
                              const std::string &outputType,
                              std::uint64_t tag);
  std::vector<InstanceHandle> createAddTagBank(
      const std::string &inputType, const std::string &outputType,
      const std::vector<std::uint64_t> &tags);
  InstanceHandle createMapTag(const std::string &inputType,
                              const std::string &outputType,
                              const std::vector<MapTagEntrySpec> &table);
  InstanceHandle createDelTag(const std::string &inputType,
                              const std::string &outputType);
  std::vector<InstanceHandle> createDelTagBank(const std::string &inputType,
                                               const std::string &outputType,
                                               unsigned count);

  // --- Topology helpers ---

  /// Build a rows x cols mesh of PEs and switches. PEs and switches
  /// alternate in a grid pattern. Switches are connected to their NSEW
  /// neighbors. Each PE connects to the switch at its grid position.
  /// Returns the grid of PE and switch instance handles.
  ///
  /// This is the legacy torus-style helper that expects one reusable
  /// switch template and therefore wraps at boundaries.
  MeshResult buildMesh(unsigned rows, unsigned cols,
                       PEHandle pe, SWHandle sw);
  MeshResult buildMesh(
      unsigned rows, unsigned cols,
      const std::function<PEHandle(unsigned, unsigned)> &peSelector,
      SWHandle sw);

  /// Build a boundary-aware lattice mesh with one local spatial_sw per cell
  /// and no wrap-around connections. The helper synthesizes spatial_sw
  /// templates whose degrees match corner, edge, and interior positions.
  MeshResult buildLatticeMesh(unsigned rows, unsigned cols, PEHandle pe,
                              int decomposableBits = -1);
  MeshResult buildLatticeMesh(
      unsigned rows, unsigned cols,
      const std::function<PEHandle(unsigned, unsigned)> &peSelector,
      int decomposableBits = -1);
  MeshResult buildLatticeMesh(unsigned rows, unsigned cols, PEHandle pe,
                              const LatticeMeshOptions &options);
  MeshResult buildLatticeMesh(
      unsigned rows, unsigned cols,
      const std::function<PEHandle(unsigned, unsigned)> &peSelector,
      const LatticeMeshOptions &options);

  /// Alias for the current torus-style buildMesh helper.
  MeshResult buildTorusMesh(unsigned rows, unsigned cols,
                            PEHandle pe, SWHandle sw);
  MeshResult buildTorusMesh(
      unsigned rows, unsigned cols,
      const std::function<PEHandle(unsigned, unsigned)> &peSelector,
      SWHandle sw);

  /// Build a chessboard-style mesh: spatial_pe instances occupy cell centers
  /// and spatial_sw instances occupy cell corners.
  ///
  /// The returned switch grid has shape [rows + 1][cols + 1]. Each PE is
  /// connected to its four surrounding corner switches using the first four
  /// PE inputs/outputs in NW, NE, SW, SE order.
  ///
  /// This helper expects the PE template to provide at least four inputs and
  /// four outputs for the network-facing ports.
  MeshResult buildChessMesh(unsigned rows, unsigned cols, PEHandle pe,
                            int decomposableBits = -1,
                            unsigned topLeftExtraInputs = 0,
                            unsigned bottomRightExtraOutputs = 0);
  MeshResult buildChessMesh(unsigned rows, unsigned cols,
                            const std::function<PEHandle(unsigned, unsigned)> &peSelector,
                            const ChessMeshOptions &options = {});

  /// Build a 1-D ring of PEs around a single spatial switch template.
  /// This helper is intended for quick topology sketches and uses one switch
  /// per PE location.
  MeshResult buildRing(unsigned count, PEHandle pe, SWHandle sw);
  MeshResult buildRing(unsigned count,
                       const std::function<PEHandle(unsigned)> &peSelector,
                       SWHandle sw);

  /// Build a 3-D cube-style topology: spatial_pe instances occupy cell centers
  /// and spatial_sw instances occupy cell vertices. The returned switch grid has
  /// shape [depths + 1][rows + 1][cols + 1]. Each PE is connected to its eight
  /// surrounding corner switches using the first eight PE inputs/outputs.
  CubeResult buildCube(unsigned depths, unsigned rows, unsigned cols,
                       PEHandle pe, int decomposableBits = -1,
                       unsigned originExtraInputs = 0,
                       unsigned farCornerExtraOutputs = 0);
  CubeResult buildCube(unsigned depths, unsigned rows, unsigned cols,
                       PEHandle pe, const CubeOptions &options);
  CubeResult buildCube(
      unsigned depths, unsigned rows, unsigned cols,
      const std::function<PEHandle(unsigned, unsigned, unsigned)> &peSelector,
      int decomposableBits = -1);
  CubeResult buildCube(
      unsigned depths, unsigned rows, unsigned cols,
      const std::function<PEHandle(unsigned, unsigned, unsigned)> &peSelector,
      const CubeOptions &options = {});

  /// Attach explicit visualization coordinates to an instantiated component.
  /// The exported fabric.mlir will reference a sidecar *.viz.json file, and
  /// the visualization renderer will prefer these coordinates over inferred
  /// layout when available.
  void setInstanceVizPosition(InstanceHandle inst, double centerX,
                              double centerY, int gridRow = -1,
                              int gridCol = -1);

  // --- Module I/O ---

  /// Add a module-level memref input port (for external memory binding).
  unsigned addMemrefInput(const std::string &name,
                          const std::string &memrefTypeStr);
  std::vector<unsigned> addMemrefInputs(const std::string &prefix,
                                        unsigned count,
                                        const std::string &memrefTypeStr);

  /// Connect a module memref input to an ext memory instance.
  void connectMemrefToExtMem(unsigned memrefIdx, InstanceHandle extMemInst);

  /// Add a module-level scalar input port (e.g. i32, none).
  /// bitsWidth is the fabric port width for this scalar.
  unsigned addScalarInput(const std::string &name, unsigned bitsWidth);
  std::vector<unsigned> addScalarInputs(const std::string &prefix,
                                        unsigned count, unsigned bitsWidth);
  unsigned addInput(const std::string &name, const std::string &typeStr);
  std::vector<unsigned> addInputs(const std::string &prefix,
                                  const std::vector<std::string> &typeStrs);

  /// Add a module-level scalar output port (e.g. none/done token).
  /// bitsWidth is the fabric port width for this output.
  unsigned addScalarOutput(const std::string &name, unsigned bitsWidth);
  std::vector<unsigned> addScalarOutputs(const std::string &prefix,
                                         unsigned count, unsigned bitsWidth);
  unsigned addOutput(const std::string &name, const std::string &typeStr);
  std::vector<unsigned> addOutputs(const std::string &prefix,
                                   const std::vector<std::string> &typeStrs);

  /// Connect a scalar input to an instance's input port.
  /// The scalar input value (%scalarN) will be used as the SSA operand.
  void connectScalarInputToInstance(unsigned scalarIdx,
                                   InstanceHandle dst, unsigned dstPort);
  void connectInputVectorToInstance(const std::vector<unsigned> &inputIdxs,
                                    InstanceHandle dst,
                                    unsigned dstPortBase = 0);
  void connectInputToPort(unsigned inputIdx, PortRef dst);
  void connectInputToInstance(unsigned inputIdx,
                              InstanceHandle dst, unsigned dstPort);

  /// Connect an instance's output port to a scalar output.
  /// The instance output will be yielded as the module result.
  void connectInstanceToScalarOutput(InstanceHandle src, unsigned srcPort,
                                    unsigned scalarOutputIdx);
  void connectInstanceToOutputVector(InstanceHandle src, unsigned srcPortBase,
                                     const std::vector<unsigned> &outputIdxs);
  void connectPortToOutput(PortRef src, unsigned outputIdx);
  void connectInstanceToOutput(InstanceHandle src, unsigned srcPort,
                               unsigned outputIdx);

  /// Connect a bank of PE instances to consecutive ports of one switch.
  /// PE outputs consume switch input ports; switch outputs feed PE inputs.
  SwitchPortCursor connectPEBankToSwitch(
      InstanceHandle sw, const std::vector<InstanceHandle> &peInstances,
      unsigned peInputCount, unsigned peOutputCount,
      SwitchPortCursor cursor = {});

  /// Associate an external memory instance with a nearby switch and create
  /// real SSA connections between them. swInputPortBase is the first SW
  /// input port used for extmem outputs. swOutputPortBase is the first SW
  /// output port used to feed extmem inputs. The SW definition must have
  /// enough ports allocated for these connections.
  void associateExtMemWithSW(InstanceHandle extMem, InstanceHandle sw,
                             unsigned swInputPortBase,
                             unsigned swOutputPortBase);
  SwitchPortCursor associateExtMemBankWithSW(
      const std::vector<InstanceHandle> &extMems, InstanceHandle sw,
      unsigned swInputPortsPerExtMem, unsigned swOutputPortsPerExtMem,
      SwitchPortCursor cursor = {});
  void associateMemoryWithSW(InstanceHandle memory, InstanceHandle sw,
                             unsigned swInputPortBase,
                             unsigned swOutputPortBase);
  SwitchPortCursor associateMemoryBankWithSW(
      const std::vector<InstanceHandle> &memories, InstanceHandle sw,
      unsigned swInputPortsPerMemory, unsigned swOutputPortsPerMemory,
      SwitchPortCursor cursor = {});

  /// Build a common "central switch + PE/memory banks + scalar boundaries"
  /// domain skeleton and wire the banks consecutively onto one switch.
  SwitchBankDomainResult buildSwitchBankDomain(
      const SwitchBankDomainSpec &spec);
  SwitchBankDomainResult buildSwitchBankDomain(
      const SwitchBankDomainSpec &spec,
      const std::function<PEHandle(unsigned)> &peSelector);

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
