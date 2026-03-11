//===-- ADGGenHelpers.h - Shared helpers for ADG generation ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Shared utility functions and data structures used by the ADG generation
// implementations (Mesh2D, Cube3D, Temporal, Memory).
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_HARDWARE_ADG_ADGGENHELPERS_H
#define LOOM_HARDWARE_ADG_ADGGENHELPERS_H

#include "loom/Hardware/ADG/ADGGen.h"
#include "loom/Hardware/Common/FabricConstants.h"
#include "loom/adg.h"

#include <cassert>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Type helpers
//===----------------------------------------------------------------------===//

/// Map a bit width to its native MLIR type representation.
/// Width 0 maps to none type (control tokens).
inline Type widthToNativeType(unsigned w) {
  if (w == 0)
    return Type::none();
  if (w == ADDR_BIT_WIDTH)
    return Type::index();
  switch (w) {
  case 1:  return Type::i1();
  case 8:  return Type::i8();
  case 16: return Type::i16();
  case 32: return Type::i32();
  case 64: return Type::i64();
  default: return Type::iN(w);
  }
}

/// True if the operation requires floating-point body types.
inline bool isFloatOp(const std::string &opName) {
  return opName == "arith.addf" || opName == "arith.subf" ||
         opName == "arith.mulf" || opName == "arith.divf" ||
         opName == "arith.negf" || opName == "arith.cmpf" ||
         opName == "math.absf" || opName == "math.cos" ||
         opName == "math.sin" || opName == "math.exp" ||
         opName == "math.log2" || opName == "math.sqrt" ||
         opName == "math.fma";
}

/// Map a bit width to its native float MLIR type representation.
inline Type widthToFloatType(unsigned w) {
  switch (w) {
  case 16: return Type::f16();
  case 32: return Type::f32();
  case 64: return Type::f64();
  default: return widthToNativeType(w); // fallback (e.g. i1 for cmp output)
  }
}

/// True if the operation is a conversion/cast operation.
inline bool isConversionOp(const std::string &opName) {
  return opName == "arith.extsi" || opName == "arith.extui" ||
         opName == "arith.extf" || opName == "arith.trunci" ||
         opName == "arith.truncf" || opName == "arith.sitofp" ||
         opName == "arith.uitofp" || opName == "arith.fptosi" ||
         opName == "arith.fptoui" || opName == "arith.bitcast" ||
         opName == "arith.index_cast" || opName == "arith.index_castui";
}

/// Format a width as an MLIR type string (57 -> "index", others -> "iN").
inline std::string widthToTypeStr(unsigned w) {
  if (w == ADDR_BIT_WIDTH) return "index";
  return "i" + std::to_string(w);
}

/// Format a width as a float MLIR type string (16 -> "f16", 32 -> "f32", etc).
inline std::string widthToFloatTypeStr(unsigned w) {
  return "f" + std::to_string(w);
}

/// True if conversion op has a float input (fptosi, fptoui, extf, truncf).
inline bool conversionHasFloatInput(const std::string &opName) {
  return opName == "arith.fptosi" || opName == "arith.fptoui" ||
         opName == "arith.extf" || opName == "arith.truncf";
}

/// True if conversion op has a float output (sitofp, uitofp, extf, truncf).
inline bool conversionHasFloatOutput(const std::string &opName) {
  return opName == "arith.sitofp" || opName == "arith.uitofp" ||
         opName == "arith.extf" || opName == "arith.truncf";
}

/// Build MLIR body string for a conversion PE.
inline std::string buildConversionBody(const PESpec &spec) {
  assert(spec.inWidths.size() == 1 && spec.outWidths.size() == 1);
  std::string inTy = conversionHasFloatInput(spec.opName)
                          ? widthToFloatTypeStr(spec.inWidths[0])
                          : widthToTypeStr(spec.inWidths[0]);
  std::string outTy = conversionHasFloatOutput(spec.opName)
                           ? widthToFloatTypeStr(spec.outWidths[0])
                           : widthToTypeStr(spec.outWidths[0]);
  std::ostringstream os;
  os << "^bb0(%arg0: " << inTy << "):\n";
  os << "  %0 = " << spec.opName << " %arg0 : " << inTy
     << " to " << outTy << "\n";
  os << "  fabric.yield %0 : " << outTy;
  return os.str();
}

//===----------------------------------------------------------------------===//
// Dimension computation helpers
//===----------------------------------------------------------------------===//

/// Compute 2D mesh dimensions for a given PE count.
inline std::pair<unsigned, unsigned> computeMesh2DDims(unsigned peCount) {
  if (peCount == 0)
    return {1, 1};
  unsigned dim = static_cast<unsigned>(std::ceil(std::sqrt(peCount)));
  if (dim == 0)
    dim = 1;
  return {dim, dim};
}

//===----------------------------------------------------------------------===//
// Anti-diagonal zigzag sweep
//===----------------------------------------------------------------------===//

/// Generate anti-diagonal zigzag sweep for a grid starting from (0,0).
/// Even diagonals (d=r+c): descending row. Odd diagonals: ascending row.
inline std::vector<std::pair<int, int>> antiDiagSweep(int rows, int cols) {
  std::vector<std::pair<int, int>> result;
  int maxDiag = rows + cols - 2;
  for (int d = 0; d <= maxDiag; ++d) {
    int rMin = std::max(0, d - cols + 1);
    int rMax = std::min(d, rows - 1);
    if (d % 2 == 0) {
      for (int r = rMax; r >= rMin; --r)
        result.push_back({r, d - r});
    } else {
      for (int r = rMin; r <= rMax; ++r)
        result.push_back({r, d - r});
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Per-switch port map (2D lattice)
//===----------------------------------------------------------------------===//

/// Tracks port index assignments for a switch at a given 2D grid position.
struct SwPortMap {
  unsigned numIn = 0;
  unsigned numOut = 0;

  // Directional port indices: dirIn[dir * T + track], -1 if absent.
  // dir: 0=N, 1=E, 2=S, 3=W
  std::vector<int> dirIn;
  std::vector<int> dirOut;

  // PE port indices by corner role: 0=UL, 1=LL, 2=UR, 3=LR.
  int peOut[4] = {-1, -1, -1, -1};
  int peIn[4] = {-1, -1, -1, -1};

  // Module I/O port indices on this switch (in assignment order).
  std::vector<int> modIn;
  std::vector<int> modOut;
};

//===----------------------------------------------------------------------===//
// Cell info for PE placement
//===----------------------------------------------------------------------===//

/// Describes a PE cell in the lattice grid.
struct CellInfo {
  enum CellType { Empty, Regular, LoadPE, StorePE };
  CellType type = Empty;
  const PESpec *spec = nullptr;
  unsigned numIn = 0;
  unsigned numOut = 0;
};

//===----------------------------------------------------------------------===//
// PE definition/instantiation helpers
//===----------------------------------------------------------------------===//

/// Create a PE definition from a PESpec and return its handle.
inline PEHandle createPEDef(ADGBuilder &builder, const PESpec &spec) {
  std::vector<Type> inTypes, outTypes;
  bool useFloat = isFloatOp(spec.opName);
  for (unsigned w : spec.inWidths)
    inTypes.push_back(useFloat ? widthToFloatType(w) : widthToNativeType(w));
  for (unsigned w : spec.outWidths)
    outTypes.push_back(useFloat ? widthToFloatType(w) : widthToNativeType(w));

  auto peBuilder = builder.newPE(spec.peName())
                       .setLatency(1, 1, 1)
                       .setInterval(1, 1, 1)
                       .setInputPorts(inTypes)
                       .setOutputPorts(outTypes);
  if (isConversionOp(spec.opName))
    peBuilder.setBodyMLIR(buildConversionBody(spec));
  else
    peBuilder.addOp(spec.opName);
  return peBuilder;
}

//===----------------------------------------------------------------------===//
// Multi-width lattice data structures
//===----------------------------------------------------------------------===//

/// A cell's port requirements in one width plane.
struct WidthCellSpec {
  unsigned numIn = 0;
  unsigned numOut = 0;
};

/// Tracks a boundary I/O slot assignment on a lattice switch.
struct LatticeIOSlot {
  int swRow, swCol;
  int switchPort;
};

/// Result of building a lattice for one width plane.
struct WidthLattice {
  unsigned width = 0;
  unsigned peRows = 0, peCols = 0;
  int swRows = 0, swCols = 0;
  std::vector<std::vector<SwPortMap>> swPorts;
  std::vector<std::vector<InstanceHandle>> swGrid;

  // Boundary I/O slot assignments.
  // inputSlots: data entering lattice (switch INPUT ports).
  // outputSlots: data exiting lattice (switch OUTPUT ports).
  std::vector<LatticeIOSlot> inputSlots;
  std::vector<LatticeIOSlot> outputSlots;
};

//===----------------------------------------------------------------------===//
// Generation sub-functions (defined in separate .cpp files)
//===----------------------------------------------------------------------===//

/// Generate a Cube3D lattice topology.
void generateCube3D(ADGBuilder &builder, const MergedRequirements &reqs,
                    const GenConfig &config);

/// Generate temporal domain (dual-mesh with add_tag/del_tag bridges).
/// The lattices map provides native mesh data for direct bridge connections.
/// bridgeOutStartByWidth/bridgeInStartByWidth give the starting I/O slot
/// indices in each native lattice for temporal bridge connections.
void generateTemporal(ADGBuilder &builder, const MergedRequirements &reqs,
                      const GenConfig &config,
                      std::map<unsigned, WidthLattice> &lattices,
                      const std::map<unsigned, unsigned> &bridgeOutStartByWidth,
                      const std::map<unsigned, unsigned> &bridgeInStartByWidth);

} // namespace adg
} // namespace loom

#endif // LOOM_HARDWARE_ADG_ADGGENHELPERS_H
