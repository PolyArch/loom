//===-- ADGGen.h - ADG Generation from DFG Requirements ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Pure C++ API for generating Architecture Description Graphs from DFG
// requirements. Analyzes one or more handshake DFGs to determine PE and
// I/O requirements, then builds a lattice-based ADG using the ADGBuilder API.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_HARDWARE_ADG_ADGGEN_H
#define LOOM_HARDWARE_ADG_ADGGEN_H

#include <map>
#include <string>
#include <vector>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// PE Specification
//===----------------------------------------------------------------------===//

/// Describes a unique PE type needed in the generated ADG.
struct PESpec {
  std::string opName; // MLIR operation name, e.g. "arith.addi"
  std::vector<unsigned> inWidths;  // input port bit widths
  std::vector<unsigned> outWidths; // output port bit widths

  /// The width plane this PE belongs to (max input width).
  unsigned primaryWidth() const;

  /// True if any output width differs from the primary input width.
  bool isCrossWidth() const;

  /// Generate a canonical PE definition name.
  std::string peName() const;

  bool operator<(const PESpec &o) const;
  bool operator==(const PESpec &o) const;
};

//===----------------------------------------------------------------------===//
// Memory Specification
//===----------------------------------------------------------------------===//

/// Distinguishes on-chip (fabric.memory) from external (fabric.extmemory).
enum class MemKind { External, OnChip };

/// Describes a memory group needed in the generated ADG.
struct MemorySpec {
  unsigned ldCount = 0;    // number of load ports
  unsigned stCount = 0;    // number of store ports
  unsigned dataWidth = 32; // data bit width (e.g. 32 for f32)
  MemKind kind = MemKind::External; // on-chip SRAM vs external interface
  unsigned memCapacity = 0; // static element count for OnChip (e.g. 256)

  bool operator<(const MemorySpec &o) const;
  bool operator==(const MemorySpec &o) const;
};

//===----------------------------------------------------------------------===//
// DFG Analysis Results
//===----------------------------------------------------------------------===//

/// Analysis result for a single handshake DFG.
struct SingleDFGAnalysis {
  /// PE type -> instance count needed by this DFG.
  std::map<PESpec, unsigned> peCounts;
  /// Bit width -> number of module input ports of that width.
  std::map<unsigned, unsigned> inputsByWidth;
  /// Bit width -> number of module output ports of that width.
  std::map<unsigned, unsigned> outputsByWidth;
  /// Memory group -> instance count needed by this DFG.
  std::map<MemorySpec, unsigned> memoryCounts;
};

/// Merged requirements from multiple DFGs. The generated ADG must support
/// any single DFG mapping (not all simultaneously).
struct MergedRequirements {
  /// PE type -> max count across all DFGs.
  std::map<PESpec, unsigned> peMaxCounts;
  /// Bit width -> max input port count across all DFGs.
  std::map<unsigned, unsigned> maxInputsByWidth;
  /// Bit width -> max output port count across all DFGs.
  std::map<unsigned, unsigned> maxOutputsByWidth;
  /// Memory group -> max count across all DFGs.
  std::map<MemorySpec, unsigned> maxMemoryCounts;

  /// Merge a single DFG analysis into the aggregate requirements.
  void mergeFrom(const SingleDFGAnalysis &analysis);

  /// Compute a hash string from the merged requirements.
  std::string computeHash() const;
};

//===----------------------------------------------------------------------===//
// Generation Configuration
//===----------------------------------------------------------------------===//

struct GenConfig {
  enum Topology { Mesh2D, Cube3D };
  Topology topology = Mesh2D;
  unsigned numSwitchTrack = 1;

  /// Controls FIFO insertion on inter-switch edges.
  /// none:   no FIFOs on any direction (both direct).
  /// single: FIFOs on reverse direction only (W, N). Default.
  /// dual:   FIFOs on both forward (E, S) and reverse (W, N) directions.
  enum FifoMode { FifoNone, FifoSingle, FifoDual };
  FifoMode fifoMode = FifoNone;
  unsigned fifoDepth = 2;
  bool fifoBypassable = false;

  /// Enable temporal domain generation (dual-mesh with bridges).
  bool genTemporal = false;
  unsigned temporalTagWidth = 4;
  unsigned temporalNumInstruction = 4;

  /// Analysis-driven spatial/temporal PE partition.
  /// When non-empty, temporal generation uses these instead of duplicating
  /// all PEs. If empty, falls back to duplicating all PEs (original behavior).
  std::map<PESpec, unsigned> spatialPECounts;
  std::map<PESpec, unsigned> temporalPECounts;
};

//===----------------------------------------------------------------------===//
// ADG Generator
//===----------------------------------------------------------------------===//

/// Generates an ADG (fabric MLIR) from merged DFG requirements.
class ADGGen {
public:
  /// Generate an ADG and export to outputPath.
  /// \param reqs      Merged DFG requirements.
  /// \param config    Topology and track configuration.
  /// \param outputPath Path to write the .fabric.mlir file.
  /// \param moduleName Name for the fabric.module.
  void generate(const MergedRequirements &reqs, const GenConfig &config,
                const std::string &outputPath,
                const std::string &moduleName);
};

} // namespace adg
} // namespace loom

#endif // LOOM_HARDWARE_ADG_ADGGEN_H
