//===-- ADGGen.cpp - ADG Generation from DFG Requirements -------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Multi-width lattice placement: each PE occupies one cell in each width
// plane that its ports touch. A PE with N unique port widths occupies N cells,
// one in each of the N different width lattice networks.
//
//===----------------------------------------------------------------------===//

#include "loom/Hardware/ADG/ADGGen.h"
#include "loom/Hardware/ADG/ADGBuilderImpl.h"
#include "loom/Hardware/ADG/ADGGenHelpers.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// PESpec helpers
//===----------------------------------------------------------------------===//

unsigned PESpec::primaryWidth() const {
  unsigned maxW = 0;
  for (unsigned w : inWidths)
    maxW = std::max(maxW, w);
  if (maxW == 0) {
    for (unsigned w : outWidths)
      maxW = std::max(maxW, w);
  }
  return maxW;
}

bool PESpec::isCrossWidth() const {
  unsigned pw = primaryWidth();
  // If primaryWidth was derived from outputs (no non-zero input matches it),
  // the outputs differ from the inputs and the name must reflect that.
  if (pw > 0) {
    bool pwFromInput = false;
    for (unsigned w : inWidths)
      if (w == pw) { pwFromInput = true; break; }
    if (!pwFromInput)
      return true;
  }
  for (unsigned w : outWidths) {
    if (w != pw)
      return true;
  }
  return false;
}

std::string PESpec::peName() const {
  std::string shortOp = opName;
  auto dotPos = shortOp.rfind('.');
  if (dotPos != std::string::npos)
    shortOp = shortOp.substr(dotPos + 1);

  std::ostringstream os;
  os << "pe_" << shortOp;
  for (unsigned w : inWidths)
    os << "_i" << w;
  if (isCrossWidth()) {
    os << "_to";
    for (unsigned w : outWidths)
      os << "_o" << w;
  }
  return os.str();
}

bool PESpec::operator<(const PESpec &o) const {
  if (opName != o.opName)
    return opName < o.opName;
  if (inWidths != o.inWidths)
    return inWidths < o.inWidths;
  return outWidths < o.outWidths;
}

bool PESpec::operator==(const PESpec &o) const {
  return opName == o.opName && inWidths == o.inWidths &&
         outWidths == o.outWidths;
}

//===----------------------------------------------------------------------===//
// MemorySpec helpers
//===----------------------------------------------------------------------===//

bool MemorySpec::operator<(const MemorySpec &o) const {
  if (kind != o.kind)
    return kind < o.kind;
  if (ldCount != o.ldCount)
    return ldCount < o.ldCount;
  if (stCount != o.stCount)
    return stCount < o.stCount;
  if (dataWidth != o.dataWidth)
    return dataWidth < o.dataWidth;
  if (isFloat != o.isFloat)
    return isFloat < o.isFloat;
  return memCapacity < o.memCapacity;
}

bool MemorySpec::operator==(const MemorySpec &o) const {
  return kind == o.kind && ldCount == o.ldCount && stCount == o.stCount &&
         dataWidth == o.dataWidth && isFloat == o.isFloat &&
         memCapacity == o.memCapacity;
}

//===----------------------------------------------------------------------===//
// MergedRequirements
//===----------------------------------------------------------------------===//

void MergedRequirements::mergeFrom(const SingleDFGAnalysis &analysis) {
  for (const auto &[spec, count] : analysis.peCounts) {
    auto &cur = peMaxCounts[spec];
    cur = std::max(cur, count);
  }
  for (const auto &[w, c] : analysis.inputsByWidth) {
    auto &cur = maxInputsByWidth[w];
    cur = std::max(cur, c);
  }
  for (const auto &[w, c] : analysis.outputsByWidth) {
    auto &cur = maxOutputsByWidth[w];
    cur = std::max(cur, c);
  }
  for (const auto &[spec, count] : analysis.memoryCounts) {
    auto &cur = maxMemoryCounts[spec];
    cur = std::max(cur, count);
  }
}

std::string MergedRequirements::computeHash() const {
  std::hash<std::string> hasher;
  std::ostringstream os;
  for (const auto &[spec, count] : peMaxCounts) {
    os << spec.opName;
    for (unsigned w : spec.inWidths)
      os << "i" << w;
    for (unsigned w : spec.outWidths)
      os << "o" << w;
    os << "x" << count;
  }
  for (const auto &[w, c] : maxInputsByWidth)
    os << "in" << w << "x" << c;
  for (const auto &[w, c] : maxOutputsByWidth)
    os << "out" << w << "x" << c;
  for (const auto &[spec, count] : maxMemoryCounts)
    os << (spec.kind == MemKind::OnChip ? "omem" : "mem")
       << spec.ldCount << "l" << spec.stCount << "s"
       << spec.dataWidth << "w" << "cap" << spec.memCapacity
       << "x" << count;
  size_t h = hasher(os.str());
  std::ostringstream hex;
  hex << std::hex << (h & 0xFFFFFFFF);
  return hex.str();
}

/// Tracks a PE placement across multiple width lattices.
struct PEPlacement {
  enum Kind { RegularPE, MemLoadPE, MemStorePE };
  Kind kind;
  const PESpec *spec = nullptr; // For RegularPE
  unsigned dataWidth = 0;       // For MemLoadPE/MemStorePE
  bool isFloat = false;         // For MemLoadPE/MemStorePE: true if float data

  // Per-width port mapping: which global PE ports connect to each width.
  struct WidthPorts {
    std::vector<unsigned> inPorts;  // Global input port indices
    std::vector<unsigned> outPorts; // Global output port indices
  };
  std::map<unsigned, WidthPorts> portsByWidth;

  // Filled during placement: cell (row, col) in each width's lattice.
  std::map<unsigned, std::pair<int, int>> cellAllocs;
};

//===----------------------------------------------------------------------===//
// buildWidthLattice: construct one width plane's lattice infrastructure
//===----------------------------------------------------------------------===//

/// Build lattice infrastructure (switches, FIFOs, inter-switch connections)
/// for a single width plane. Does NOT create PE instances.
/// Module I/O slots are allocated but not connected to module-level ports;
/// the caller connects them to module ports or internal modules.
static WidthLattice buildWidthLattice(
    ADGBuilder &builder, unsigned width,
    const std::vector<WidthCellSpec> &cells,
    unsigned numIOIn, unsigned numIOOut,
    const GenConfig &config) {

  WidthLattice lattice;
  lattice.width = width;

  unsigned totalCells = cells.size();
  if (totalCells == 0 && numIOIn == 0 && numIOOut == 0)
    return lattice;

  auto peDims = computeMesh2DDims(totalCells > 0 ? totalCells : 1);
  unsigned peRows = peDims.first;
  unsigned peCols = peDims.second;
  lattice.peRows = peRows;
  lattice.peCols = peCols;
  lattice.swRows = static_cast<int>(peRows) + 1;
  lattice.swCols = static_cast<int>(peCols) + 1;
  int swRows = lattice.swRows, swCols = lattice.swCols;
  unsigned T = config.numSwitchTrack;

  // Populate cell grid from the flat cell list.
  std::vector<std::vector<WidthCellSpec>> cellGrid(
      peRows, std::vector<WidthCellSpec>(peCols));
  for (unsigned i = 0; i < totalCells && i < peRows * peCols; ++i)
    cellGrid[i / peCols][i % peCols] = cells[i];

  //=================================================================
  // I/O assignment via anti-diagonal sweep
  //=================================================================

  auto sweep = antiDiagSweep(swRows, swCols);

  struct IOAssign { int swR, swC; };

  // Input I/O slots (data entering lattice) - from top-left corner.
  std::vector<IOAssign> inAssigns;
  {
    unsigned assigned = 0;
    for (auto [r, c] : sweep) {
      if (assigned >= numIOIn) break;
      bool hasDir[4] = {r > 0, c < swCols - 1, r < swRows - 1, c > 0};
      for (int dir = 0; dir < 4 && assigned < numIOIn; ++dir) {
        if (!hasDir[dir]) {
          for (unsigned t = 0; t < T && assigned < numIOIn; ++t) {
            inAssigns.push_back({r, c});
            ++assigned;
          }
        }
      }
    }
  }

  // Output I/O slots (data exiting lattice) - from bottom-right corner.
  std::vector<IOAssign> outAssigns;
  {
    unsigned assigned = 0;
    for (int i = static_cast<int>(sweep.size()) - 1; i >= 0; --i) {
      if (assigned >= numIOOut) break;
      auto [r, c] = sweep[i];
      bool hasDir[4] = {r > 0, c < swCols - 1, r < swRows - 1, c > 0};
      for (int dir = 3; dir >= 0 && assigned < numIOOut; --dir) {
        if (!hasDir[dir]) {
          for (int t = static_cast<int>(T) - 1;
               t >= 0 && assigned < numIOOut; --t) {
            outAssigns.push_back({r, c});
            ++assigned;
          }
        }
      }
    }
  }

  // Count I/O per switch for port map computation.
  std::map<std::pair<int, int>, unsigned> swIOInCount, swIOOutCount;
  for (auto &a : inAssigns) swIOInCount[{a.swR, a.swC}]++;
  for (auto &a : outAssigns) swIOOutCount[{a.swR, a.swC}]++;

  //=================================================================
  // Compute per-switch port maps
  //=================================================================

  lattice.swPorts.assign(swRows, std::vector<SwPortMap>(swCols));

  for (int r = 0; r < swRows; ++r) {
    for (int c = 0; c < swCols; ++c) {
      auto &pm = lattice.swPorts[r][c];
      pm.dirIn.assign(4 * T, -1);
      pm.dirOut.assign(4 * T, -1);
      unsigned inIdx = 0, outIdx = 0;

      // Directional ports.
      bool hasDir[4] = {r > 0, c < swCols - 1, r < swRows - 1, c > 0};
      for (int dir = 0; dir < 4; ++dir) {
        if (hasDir[dir]) {
          for (unsigned t = 0; t < T; ++t) {
            pm.dirIn[dir * T + t] = static_cast<int>(inIdx++);
            pm.dirOut[dir * T + t] = static_cast<int>(outIdx++);
          }
        }
      }

      // PE output ports (switch out -> PE input).
      // Corner roles: 0=UL, 1=LL, 2=UR, 3=LR.
      // For cells needing >4 PE inputs, overflow wraps around corners.
      // Input port localIdx goes to corner localIdx%4.
      auto peInPortsForCorner = [&](int corner) -> unsigned {
        int cr, cc;
        switch (corner) {
        case 0: cr = r;     cc = c;     break; // UL: cell (r, c)
        case 1: cr = r - 1; cc = c;     break; // LL: cell (r-1, c)
        case 2: cr = r;     cc = c - 1; break; // UR: cell (r, c-1)
        case 3: cr = r - 1; cc = c - 1; break; // LR: cell (r-1, c-1)
        default: return 0;
        }
        if (cr < 0 || cc < 0 || cr >= (int)peRows || cc >= (int)peCols)
          return 0;
        unsigned total = cellGrid[cr][cc].numIn;
        unsigned count = 0;
        for (unsigned i = corner; i < total; i += 4)
          count++;
        return count;
      };
      unsigned peOutOff = 0;
      for (int corner = 0; corner < 4; ++corner) {
        unsigned n = peInPortsForCorner(corner);
        pm.peOutCornerOff[corner] = peOutOff;
        pm.peOutCornerCnt[corner] = n;
        for (unsigned k = 0; k < n; ++k)
          pm.peOutVec.push_back(static_cast<int>(outIdx++));
        peOutOff += n;
      }

      // PE input ports (PE output -> switch in).
      // Output corners reversed: localIdx%4 0=LR, 1=UR, 2=LL, 3=UL.
      // Allocation corner numbering: 0=UL, 1=LL, 2=UR, 3=LR.
      // Cell at alloc corner c sends output ports with localIdx%4 == (3-c).
      auto peOutPortsForCorner = [&](int corner) -> unsigned {
        int cr, cc;
        switch (corner) {
        case 0: cr = r;     cc = c;     break; // UL: cell (r, c)
        case 1: cr = r - 1; cc = c;     break; // LL: cell (r-1, c)
        case 2: cr = r;     cc = c - 1; break; // UR: cell (r, c-1)
        case 3: cr = r - 1; cc = c - 1; break; // LR: cell (r-1, c-1)
        default: return 0;
        }
        if (cr < 0 || cc < 0 || cr >= (int)peRows || cc >= (int)peCols)
          return 0;
        unsigned total = cellGrid[cr][cc].numOut;
        unsigned count = 0;
        unsigned start = 3 - corner; // reversed output corner mapping
        for (unsigned i = start; i < total; i += 4)
          count++;
        return count;
      };
      // Allocate in reverse corner order (LR=3, UR=2, LL=1, UL=0)
      unsigned peInOff = 0;
      for (int corner = 3; corner >= 0; --corner) {
        unsigned n = peOutPortsForCorner(corner);
        pm.peInCornerOff[corner] = peInOff;
        pm.peInCornerCnt[corner] = n;
        for (unsigned k = 0; k < n; ++k)
          pm.peInVec.push_back(static_cast<int>(inIdx++));
        peInOff += n;
      }

      // I/O ports on this switch.
      auto inIt = swIOInCount.find({r, c});
      unsigned nIn = inIt != swIOInCount.end() ? inIt->second : 0;
      for (unsigned i = 0; i < nIn; ++i)
        pm.modIn.push_back(static_cast<int>(inIdx++));

      auto outIt = swIOOutCount.find({r, c});
      unsigned nOut = outIt != swIOOutCount.end() ? outIt->second : 0;
      for (unsigned i = 0; i < nOut; ++i)
        pm.modOut.push_back(static_cast<int>(outIdx++));

      pm.numIn = inIdx;
      pm.numOut = outIdx;
    }
  }

  //=================================================================
  // Create switches
  //=================================================================

  std::string wp = std::to_string(width);
  Type switchType = widthToNativeType(width);

  std::map<std::pair<unsigned, unsigned>, SwitchHandle> swTemplates;
  for (int r = 0; r < swRows; ++r) {
    for (int c = 0; c < swCols; ++c) {
      auto &pm = lattice.swPorts[r][c];
      auto key = std::make_pair(pm.numIn, pm.numOut);
      if (swTemplates.find(key) == swTemplates.end()) {
        std::string name = "sw_w" + wp + "_" + std::to_string(pm.numIn) +
                           "x" + std::to_string(pm.numOut);
        swTemplates[key] = builder.newSwitch(name)
                               .setPortCount(pm.numIn, pm.numOut)
                               .setType(switchType);
      }
    }
  }

  lattice.swGrid.assign(swRows, std::vector<InstanceHandle>(swCols));
  for (int r = 0; r < swRows; ++r) {
    for (int c = 0; c < swCols; ++c) {
      auto &pm = lattice.swPorts[r][c];
      auto key = std::make_pair(pm.numIn, pm.numOut);
      std::string instName =
          "sw_w" + wp + "_" + std::to_string(r) + "_" + std::to_string(c);
      lattice.swGrid[r][c] = builder.clone(swTemplates[key], instName);
    }
  }

  //=================================================================
  // Create FIFOs and wire inter-switch connections
  //=================================================================

  bool useFwd = (config.fifoMode == GenConfig::FifoDual);
  bool useRev = (config.fifoMode != GenConfig::FifoNone);

  FifoHandle fwdFifo{0}, revFifo{0};
  if (useFwd) {
    auto fb = builder.newFifo("fwd_fifo_w" + wp)
                  .setDepth(config.fifoDepth)
                  .setType(switchType);
    if (config.fifoBypassable) fb.setBypassable(true);
    fwdFifo = fb;
  }
  if (useRev) {
    auto fb = builder.newFifo("rev_fifo_w" + wp)
                  .setDepth(config.fifoDepth)
                  .setType(switchType);
    if (config.fifoBypassable) fb.setBypassable(true);
    revFifo = fb;
  }

  auto connectDir = [&](InstanceHandle srcSw, int srcOut,
                        InstanceHandle dstSw, int dstIn, bool viaFifo,
                        FifoHandle fifoTmpl, const std::string &fifoName) {
    if (viaFifo) {
      auto f = builder.clone(fifoTmpl, fifoName);
      builder.connectPorts(srcSw, srcOut, f, 0);
      builder.connectPorts(f, 0, dstSw, dstIn);
    } else {
      builder.connectPorts(srcSw, srcOut, dstSw, dstIn);
    }
  };

  for (int r = 0; r < swRows; ++r) {
    for (int c = 0; c < swCols; ++c) {
      auto &pm = lattice.swPorts[r][c];

      // East
      if (c + 1 < swCols) {
        auto &pmE = lattice.swPorts[r][c + 1];
        for (unsigned t = 0; t < T; ++t) {
          int so = pm.dirOut[1 * T + t], di = pmE.dirIn[3 * T + t];
          assert(so >= 0 && di >= 0);
          connectDir(lattice.swGrid[r][c], so, lattice.swGrid[r][c + 1], di,
                     useFwd, fwdFifo,
                     "fifo_w" + wp + "_e_t" + std::to_string(t) + "_" +
                         std::to_string(r) + "_" + std::to_string(c));

          int rso = pmE.dirOut[3 * T + t], rdi = pm.dirIn[1 * T + t];
          assert(rso >= 0 && rdi >= 0);
          connectDir(lattice.swGrid[r][c + 1], rso, lattice.swGrid[r][c], rdi,
                     useRev, revFifo,
                     "fifo_w" + wp + "_w_t" + std::to_string(t) + "_" +
                         std::to_string(r) + "_" + std::to_string(c));
        }
      }

      // South
      if (r + 1 < swRows) {
        auto &pmS = lattice.swPorts[r + 1][c];
        for (unsigned t = 0; t < T; ++t) {
          int so = pm.dirOut[2 * T + t], di = pmS.dirIn[0 * T + t];
          assert(so >= 0 && di >= 0);
          connectDir(lattice.swGrid[r][c], so, lattice.swGrid[r + 1][c], di,
                     useFwd, fwdFifo,
                     "fifo_w" + wp + "_s_t" + std::to_string(t) + "_" +
                         std::to_string(r) + "_" + std::to_string(c));

          int rso = pmS.dirOut[0 * T + t], rdi = pm.dirIn[2 * T + t];
          assert(rso >= 0 && rdi >= 0);
          connectDir(lattice.swGrid[r + 1][c], rso, lattice.swGrid[r][c], rdi,
                     useRev, revFifo,
                     "fifo_w" + wp + "_n_t" + std::to_string(t) + "_" +
                         std::to_string(r) + "_" + std::to_string(c));
        }
      }
    }
  }

  //=================================================================
  // Record I/O slot assignments (do NOT create module ports here)
  //=================================================================

  {
    std::map<std::pair<int, int>, unsigned> swIOInIdx;
    for (auto &a : inAssigns) {
      auto &pm = lattice.swPorts[a.swR][a.swC];
      unsigned localIdx = swIOInIdx[{a.swR, a.swC}]++;
      lattice.inputSlots.push_back(
          {a.swR, a.swC, pm.modIn[localIdx]});
    }
  }
  {
    std::map<std::pair<int, int>, unsigned> swIOOutIdx;
    for (auto &a : outAssigns) {
      auto &pm = lattice.swPorts[a.swR][a.swC];
      unsigned localIdx = swIOOutIdx[{a.swR, a.swC}]++;
      lattice.outputSlots.push_back(
          {a.swR, a.swC, pm.modOut[localIdx]});
    }
  }

  return lattice;
}

//===----------------------------------------------------------------------===//
// wirePEPortToCell: connect one PE port to its cell's corner switch
//===----------------------------------------------------------------------===//

/// Connect PE port `globalPort` to its cell's corner switch in a width lattice.
/// `localIdx` is the port's index among ports of the same width in this cell
/// (determines which corner: inputs UL=0,LL=1,UR=2,LR=3; outputs reversed).
static void wirePEPortToCell(
    ADGBuilder &builder, const WidthLattice &lattice,
    int cellRow, int cellCol,
    InstanceHandle peInst, unsigned globalPort,
    bool isInput, unsigned localIdx) {

  unsigned corner = localIdx % 4;
  unsigned wrapIdx = localIdx / 4;

  if (isInput) {
    // Input corner assignment: 0=UL, 1=LL, 2=UR, 3=LR.
    int swR, swC;
    switch (corner) {
    case 0: swR = cellRow;     swC = cellCol;     break; // UL
    case 1: swR = cellRow + 1; swC = cellCol;     break; // LL
    case 2: swR = cellRow;     swC = cellCol + 1; break; // UR
    case 3: swR = cellRow + 1; swC = cellCol + 1; break; // LR
    default: return;
    }
    auto &pm = lattice.swPorts[swR][swC];
    // Cell is at allocation corner 'corner' on this switch.
    assert(wrapIdx < pm.peOutCornerCnt[corner]);
    unsigned idx = pm.peOutCornerOff[corner] + wrapIdx;
    int swOut = pm.peOutVec[idx];
    assert(swOut >= 0);
    builder.connectPorts(lattice.swGrid[swR][swC], swOut, peInst, globalPort);
  } else {
    // Output corner assignment (reversed): 0=LR, 1=UR, 2=LL, 3=UL.
    int swR, swC;
    switch (corner) {
    case 0: swR = cellRow + 1; swC = cellCol + 1; break; // LR
    case 1: swR = cellRow;     swC = cellCol + 1; break; // UR
    case 2: swR = cellRow + 1; swC = cellCol;     break; // LL
    case 3: swR = cellRow;     swC = cellCol;     break; // UL
    default: return;
    }
    auto &pm = lattice.swPorts[swR][swC];
    // Wiring corner N maps to allocation corner (3-N).
    unsigned allocCorner = 3 - corner;
    assert(wrapIdx < pm.peInCornerCnt[allocCorner]);
    unsigned idx = pm.peInCornerOff[allocCorner] + wrapIdx;
    int swIn = pm.peInVec[idx];
    assert(swIn >= 0);
    builder.connectPorts(peInst, globalPort, lattice.swGrid[swR][swC], swIn);
  }
}

//===----------------------------------------------------------------------===//
// ADGGen::generate - multi-width lattice placement
//===----------------------------------------------------------------------===//

void ADGGen::generate(const MergedRequirements &reqs, const GenConfig &config,
                      const std::string &outputPath,
                      const std::string &moduleName) {
  ADGBuilder builder(moduleName);

  if (config.topology == GenConfig::Cube3D) {
    generateCube3D(builder, reqs, config);
    if (config.genTemporal) {
      // Cube3D has its own lattice structure; pass empty maps so temporal
      // generation falls back to module I/O bridges.
      std::map<unsigned, WidthLattice> emptyLattices;
      std::map<unsigned, unsigned> emptyStarts;
      generateTemporal(builder, reqs, config, emptyLattices,
                       emptyStarts, emptyStarts);
    }
    builder.exportMLIR(outputPath);
    return;
  }

  //=================================================================
  // Phase 1: Build PE placement records and compute cell requirements
  //=================================================================

  std::vector<PEPlacement> placements;
  std::map<unsigned, std::vector<WidthCellSpec>> cellsByWidth;

  // Regular PEs: each PE occupies one cell per unique port width.
  for (const auto &[spec, count] : reqs.peMaxCounts) {
    for (unsigned i = 0; i < count; ++i) {
      PEPlacement p;
      p.kind = PEPlacement::RegularPE;
      p.spec = &spec;

      // Group ports by width.
      for (unsigned k = 0; k < spec.inWidths.size(); ++k)
        p.portsByWidth[spec.inWidths[k]].inPorts.push_back(k);
      for (unsigned k = 0; k < spec.outWidths.size(); ++k)
        p.portsByWidth[spec.outWidths[k]].outPorts.push_back(k);

      // Add cells for each width this PE touches.
      for (auto &[w, wp] : p.portsByWidth)
        cellsByWidth[w].push_back(
            {(unsigned)wp.inPorts.size(), (unsigned)wp.outPorts.size()});

      placements.push_back(std::move(p));
    }
  }

  // LoadPEs: addr(57-bit) + data(W-bit) + ctrl(none/0-bit).
  // DFG handshake.load outputs (data, addr), so PE out[0]=data, out[1]=addr.
  for (const auto &[memSpec, count] : reqs.maxMemoryCounts) {
    for (unsigned i = 0; i < memSpec.ldCount * count; ++i) {
      PEPlacement p;
      p.kind = PEPlacement::MemLoadPE;
      p.dataWidth = memSpec.dataWidth;
      p.isFloat = memSpec.isFloat;
      // LoadPE ports: in[0]=addr(57), in[1]=data(W), in[2]=ctrl(none)
      //              out[0]=data(W), out[1]=addr(57)
      p.portsByWidth[ADDR_BIT_WIDTH] = {{0}, {1}};
      p.portsByWidth[memSpec.dataWidth] = {{1}, {0}};
      p.portsByWidth[0] = {{2}, {}};

      for (auto &[w, wp] : p.portsByWidth)
        cellsByWidth[w].push_back(
            {(unsigned)wp.inPorts.size(), (unsigned)wp.outPorts.size()});

      placements.push_back(std::move(p));
    }
  }

  // StorePEs: same port layout as LoadPEs.
  // DFG handshake.store outputs (data, addr), so PE out[0]=data, out[1]=addr.
  for (const auto &[memSpec, count] : reqs.maxMemoryCounts) {
    for (unsigned i = 0; i < memSpec.stCount * count; ++i) {
      PEPlacement p;
      p.kind = PEPlacement::MemStorePE;
      p.dataWidth = memSpec.dataWidth;
      p.isFloat = memSpec.isFloat;
      p.portsByWidth[ADDR_BIT_WIDTH] = {{0}, {1}};
      p.portsByWidth[memSpec.dataWidth] = {{1}, {0}};
      p.portsByWidth[0] = {{2}, {}};

      for (auto &[w, wp] : p.portsByWidth)
        cellsByWidth[w].push_back(
            {(unsigned)wp.inPorts.size(), (unsigned)wp.outPorts.size()});

      placements.push_back(std::move(p));
    }
  }

  //=================================================================
  // Phase 2: Compute I/O requirements per width plane
  //=================================================================

  // DFG module I/O.
  std::map<unsigned, unsigned> dfgInByWidth, dfgOutByWidth;
  for (const auto &[w, c] : reqs.maxInputsByWidth) dfgInByWidth[w] = c;
  for (const auto &[w, c] : reqs.maxOutputsByWidth) dfgOutByWidth[w] = c;

  // ExtMem contributions to lattice boundary I/O.
  // Data that flows between ExtMem and LoadPE/StorePE passes through the
  // lattice boundary switches.
  std::map<unsigned, unsigned> extMemInByWidth, extMemOutByWidth;
  for (const auto &[memSpec, count] : reqs.maxMemoryCounts) {
    unsigned n = count;
    // LoadPE addr exits lattice → ExtMem addr input
    extMemOutByWidth[ADDR_BIT_WIDTH] += memSpec.ldCount * n;
    // StorePE addr exits lattice → ExtMem addr input
    extMemOutByWidth[ADDR_BIT_WIDTH] += memSpec.stCount * n;
    // ExtMem ld_data output → enters lattice → LoadPE data input
    extMemInByWidth[memSpec.dataWidth] += memSpec.ldCount * n;
    // StorePE data exits lattice → ExtMem st_data input
    extMemOutByWidth[memSpec.dataWidth] += memSpec.stCount * n;
    // ExtMem done outputs → enter lattice
    extMemInByWidth[0] += (memSpec.ldCount + memSpec.stCount) * n;
  }

  // Temporal bridge I/O: the temporal mesh creates its own module I/O ports
  // for add_tag and del_tag bridges; native mesh does not reserve bridge slots.
  std::map<unsigned, unsigned> bridgeOutByWidth, bridgeInByWidth;

  // Total I/O per width = DFG I/O + ExtMem I/O + temporal bridge I/O.
  std::map<unsigned, unsigned> totalInByWidth, totalOutByWidth;
  for (auto &[w, _] : cellsByWidth) {
    totalInByWidth[w] = dfgInByWidth[w] + extMemInByWidth[w] +
                        bridgeInByWidth[w];
    totalOutByWidth[w] = dfgOutByWidth[w] + extMemOutByWidth[w] +
                         bridgeOutByWidth[w];
  }
  // Also ensure widths from DFG I/O have entries.
  for (auto &[w, c] : dfgInByWidth) totalInByWidth[w] += 0;
  for (auto &[w, c] : dfgOutByWidth) totalOutByWidth[w] += 0;
  // Ensure widths from bridge I/O have entries.
  for (auto &[w, c] : bridgeInByWidth) {
    totalInByWidth[w] += 0;
    totalOutByWidth[w] += 0;
  }

  //=================================================================
  // Phase 3: Build lattice per width plane
  //=================================================================

  std::map<unsigned, WidthLattice> lattices;
  for (auto &[width, cells] : cellsByWidth) {
    lattices[width] = buildWidthLattice(
        builder, width, cells,
        totalInByWidth[width], totalOutByWidth[width], config);
  }
  // Build lattices for widths that have I/O but no cells.
  for (auto &[w, c] : totalInByWidth) {
    if (lattices.find(w) == lattices.end() && (c > 0 || totalOutByWidth[w] > 0))
      lattices[w] = buildWidthLattice(builder, w, {}, c, totalOutByWidth[w],
                                      config);
  }
  for (auto &[w, c] : totalOutByWidth) {
    if (lattices.find(w) == lattices.end() && c > 0)
      lattices[w] = buildWidthLattice(builder, w, {}, totalInByWidth[w], c,
                                      config);
  }

  //=================================================================
  // Phase 4: Allocate cells and place PEs
  //=================================================================

  std::map<unsigned, unsigned> nextCellByWidth;

  // Cache PE definitions to avoid duplicate symbol names.
  std::map<std::string, PEHandle> regularPEDefs;
  std::map<unsigned, LoadPEHandle> loadPEDefs;
  std::map<unsigned, StorePEHandle> storePEDefs;

  for (auto &p : placements) {
    // Allocate cells in each width's lattice.
    for (auto &[w, wp] : p.portsByWidth) {
      auto &lat = lattices[w];
      unsigned idx = nextCellByWidth[w]++;
      int r = static_cast<int>(idx / lat.peCols);
      int c = static_cast<int>(idx % lat.peCols);
      p.cellAllocs[w] = {r, c};
    }

    // Create PE instance (named after its primary-width cell position).
    InstanceHandle peInst{0};
    unsigned nameWidth = (p.kind == PEPlacement::RegularPE)
                             ? p.spec->primaryWidth()
                             : p.dataWidth;
    auto [nr, nc] = p.cellAllocs.count(nameWidth)
                        ? p.cellAllocs[nameWidth]
                        : p.cellAllocs.begin()->second;

    switch (p.kind) {
    case PEPlacement::RegularPE: {
      std::string defName = p.spec->peName();
      auto it = regularPEDefs.find(defName);
      if (it == regularPEDefs.end()) {
        PEHandle peHandle = createPEDef(builder, *p.spec);
        it = regularPEDefs.emplace(defName, peHandle).first;
      }
      std::string instName = defName + "_r" + std::to_string(nr) +
                             "_c" + std::to_string(nc);
      peInst = builder.clone(it->second, instName);
      break;
    }
    case PEPlacement::MemLoadPE: {
      // Key includes isFloat to distinguish i32 vs f32 LoadPEs.
      unsigned key = p.dataWidth | (p.isFloat ? 0x80000000u : 0);
      auto it = loadPEDefs.find(key);
      if (it == loadPEDefs.end()) {
        Type dataType = p.isFloat ? widthToFloatType(p.dataWidth)
                                  : widthToNativeType(p.dataWidth);
        std::string suffix = p.isFloat ? "f" : "";
        LoadPEHandle loadDef =
            builder.newLoadPE("load_pe_w" + std::to_string(p.dataWidth) + suffix)
                .setDataType(dataType);
        it = loadPEDefs.emplace(key, loadDef).first;
      }
      std::string instName =
          "load_pe_r" + std::to_string(nr) + "_c" + std::to_string(nc);
      peInst = builder.clone(it->second, instName);
      break;
    }
    case PEPlacement::MemStorePE: {
      unsigned key = p.dataWidth | (p.isFloat ? 0x80000000u : 0);
      auto it = storePEDefs.find(key);
      if (it == storePEDefs.end()) {
        Type dataType = p.isFloat ? widthToFloatType(p.dataWidth)
                                  : widthToNativeType(p.dataWidth);
        std::string suffix = p.isFloat ? "f" : "";
        StorePEHandle storeDef =
            builder.newStorePE("store_pe_w" + std::to_string(p.dataWidth) + suffix)
                .setDataType(dataType);
        it = storePEDefs.emplace(key, storeDef).first;
      }
      std::string instName =
          "store_pe_r" + std::to_string(nr) + "_c" + std::to_string(nc);
      peInst = builder.clone(it->second, instName);
      break;
    }
    }

    // Wire each port to its cell's corner switch in the correct lattice.
    for (auto &[w, wp] : p.portsByWidth) {
      auto [cr, cc] = p.cellAllocs[w];
      auto &lat = lattices[w];

      for (unsigned i = 0; i < wp.inPorts.size(); ++i)
        wirePEPortToCell(builder, lat, cr, cc, peInst, wp.inPorts[i], true, i);
      for (unsigned i = 0; i < wp.outPorts.size(); ++i)
        wirePEPortToCell(builder, lat, cr, cc, peInst, wp.outPorts[i], false,
                         i);
    }
  }

  //=================================================================
  // Phase 5: Connect DFG module I/O to lattice boundary
  //=================================================================

  for (auto &[width, lat] : lattices) {
    Type ioType = widthToNativeType(width);
    unsigned numDFGIn = dfgInByWidth[width];
    unsigned numDFGOut = dfgOutByWidth[width];

    // DFG module inputs → lattice input slots.
    for (unsigned k = 0; k < numDFGIn; ++k) {
      auto &slot = lat.inputSlots[k];
      std::string name =
          "in_w" + std::to_string(width) + "_" + std::to_string(k);
      auto port = builder.addModuleInput(name, ioType);
      builder.connectToModuleInput(
          port, lat.swGrid[slot.swRow][slot.swCol], slot.switchPort);
    }

    // DFG module outputs ← lattice output slots.
    for (unsigned k = 0; k < numDFGOut; ++k) {
      auto &slot = lat.outputSlots[k];
      std::string name =
          "out_w" + std::to_string(width) + "_" + std::to_string(k);
      auto port = builder.addModuleOutput(name, ioType);
      builder.connectToModuleOutput(
          lat.swGrid[slot.swRow][slot.swCol], slot.switchPort, port);
    }
  }

  //=================================================================
  // Phase 6: Create ExtMemory modules and connect to lattice boundary
  //=================================================================

  // Track which I/O slot to use next per width (after DFG I/O slots).
  std::map<unsigned, unsigned> nextExtInSlot, nextExtOutSlot;
  for (auto &[w, _] : lattices) {
    nextExtInSlot[w] = dfgInByWidth[w];
    nextExtOutSlot[w] = dfgOutByWidth[w];
  }

  unsigned memIdx = 0;
  for (const auto &[memSpec, count] : reqs.maxMemoryCounts) {
    Type elemType = memSpec.isFloat ? widthToFloatType(memSpec.dataWidth)
                                    : widthToNativeType(memSpec.dataWidth);
    bool isOnChip = (memSpec.kind == MemKind::OnChip);
    unsigned lsqDepth = (memSpec.stCount > 0)
                            ? std::max(1u, memSpec.stCount)
                            : 0;

    // Create memory definition: on-chip (fabric.memory) or external
    // (fabric.extmemory). The builder has separate handle types for each,
    // so we store both and use the appropriate one.
    MemoryHandle onChipDef{0};
    ExtMemoryHandle extMemDef{0};
    if (isOnChip) {
      unsigned capacity =
          memSpec.memCapacity > 0 ? memSpec.memCapacity : 64;
      MemrefType memrefTy = MemrefType::static1D(capacity, elemType);
      auto memBuilder =
          builder.newMemory("mem_" + std::to_string(memIdx));
      memBuilder.setLoadPorts(memSpec.ldCount)
          .setStorePorts(memSpec.stCount)
          .setPrivate(true)
          .setShape(memrefTy);
      if (lsqDepth > 0)
        memBuilder.setQueueDepth(lsqDepth);
      onChipDef = memBuilder;
    } else {
      MemrefType memrefTy = MemrefType::dynamic1D(elemType);
      auto extBuilder =
          builder.newExtMemory("extmem_" + std::to_string(memIdx));
      extBuilder.setLoadPorts(memSpec.ldCount)
          .setStorePorts(memSpec.stCount)
          .setShape(memrefTy);
      if (lsqDepth > 0)
        extBuilder.setQueueDepth(lsqDepth);
      extMemDef = extBuilder;
    }

    // Determine whether this memory needs tagged bridge networks.
    // Multi-port memory (ldCount>1 or stCount>1) uses single tagged ports
    // per category. TAG_WIDTH = clog2(max(ldCount, stCount)).
    unsigned tagWidth =
        computeMemoryTagWidth(memSpec.ldCount, memSpec.stCount);

    // Pre-build temporal_sw definitions for tagged bridges (shared across
    // instances of this memSpec). Only needed when tagWidth > 0.
    Type tagType = Type::none();
    Type taggedAddrType = Type::none();
    Type taggedDataType = Type::none();
    Type taggedDoneType = Type::none();
    TemporalSwitchHandle stDataMuxDef{0}, stAddrMuxDef{0}, ldAddrMuxDef{0};
    TemporalSwitchHandle ldDataDemuxDef{0}, ldDoneDemuxDef{0};
    TemporalSwitchHandle stDoneDemuxDef{0};

    if (tagWidth > 0) {
      std::string mp = std::to_string(memIdx);
      tagType = Type::iN(tagWidth);
      taggedAddrType =
          Type::tagged(Type::bits(ADDR_BIT_WIDTH), tagType);
      taggedDataType = Type::tagged(elemType, tagType);
      taggedDoneType = Type::tagged(Type::none(), tagType);

      if (memSpec.stCount > 1) {
        stDataMuxDef =
            builder.newTemporalSwitch("tsw_st_data_mux_m" + mp)
                .setPortCount(memSpec.stCount, 1)
                .setNumRouteTable(memSpec.stCount)
                .setInterface(taggedDataType);
        stAddrMuxDef =
            builder.newTemporalSwitch("tsw_st_addr_mux_m" + mp)
                .setPortCount(memSpec.stCount, 1)
                .setNumRouteTable(memSpec.stCount)
                .setInterface(taggedAddrType);
        stDoneDemuxDef =
            builder.newTemporalSwitch("tsw_st_done_demux_m" + mp)
                .setPortCount(1, memSpec.stCount)
                .setNumRouteTable(memSpec.stCount)
                .setInterface(taggedDoneType);
      }
      if (memSpec.ldCount > 1) {
        ldAddrMuxDef =
            builder.newTemporalSwitch("tsw_ld_addr_mux_m" + mp)
                .setPortCount(memSpec.ldCount, 1)
                .setNumRouteTable(memSpec.ldCount)
                .setInterface(taggedAddrType);
        ldDataDemuxDef =
            builder.newTemporalSwitch("tsw_ld_data_demux_m" + mp)
                .setPortCount(1, memSpec.ldCount)
                .setNumRouteTable(memSpec.ldCount)
                .setInterface(taggedDataType);
        ldDoneDemuxDef =
            builder.newTemporalSwitch("tsw_ld_done_demux_m" + mp)
                .setPortCount(1, memSpec.ldCount)
                .setNumRouteTable(memSpec.ldCount)
                .setInterface(taggedDoneType);
      }
    }

    for (unsigned i = 0; i < count; ++i) {
      std::string instName =
          (isOnChip ? "mem_" : "extmem_") + std::to_string(memIdx) +
          "_" + std::to_string(i);
      InstanceHandle memInst = isOnChip
                                   ? builder.clone(onChipDef, instName)
                                   : builder.clone(extMemDef, instName);

      // For extmemory: connect memref module input to port 0.
      // For on-chip memory: no memref input port.
      unsigned memInPort = 0;
      if (!isOnChip) {
        Type extElemType = memSpec.isFloat
                               ? widthToFloatType(memSpec.dataWidth)
                               : widthToNativeType(memSpec.dataWidth);
        MemrefType extMemrefTy = MemrefType::dynamic1D(extElemType);
        std::string memPortName =
            "mem_" + std::to_string(memIdx) + "_" + std::to_string(i);
        PortHandle memPort =
            builder.addModuleInput(memPortName, extMemrefTy);
        builder.connectToModuleInput(memPort, memInst, 0);
        memInPort = 1; // Skip memref port
      }

      if (tagWidth == 0) {
        // Single-port memory: direct wiring (untagged).
        // Memory port layout (after optional memref):
        //   Inputs:  [st_data, st_addr] (if stCount>0), [ld_addr]
        //   Outputs: [ld_data, ld_done] (if ldCount>0), [st_done]

        // Store data + addr
        for (unsigned st = 0; st < memSpec.stCount; ++st) {
          {
            auto &lat = lattices[memSpec.dataWidth];
            unsigned slotIdx = nextExtOutSlot[memSpec.dataWidth]++;
            auto &slot = lat.outputSlots[slotIdx];
            builder.connectPorts(lat.swGrid[slot.swRow][slot.swCol],
                                 slot.switchPort, memInst, memInPort++);
          }
          {
            auto &lat = lattices[ADDR_BIT_WIDTH];
            unsigned slotIdx = nextExtOutSlot[ADDR_BIT_WIDTH]++;
            auto &slot = lat.outputSlots[slotIdx];
            builder.connectPorts(lat.swGrid[slot.swRow][slot.swCol],
                                 slot.switchPort, memInst, memInPort++);
          }
        }

        // Load addresses
        for (unsigned ld = 0; ld < memSpec.ldCount; ++ld) {
          auto &lat = lattices[ADDR_BIT_WIDTH];
          unsigned slotIdx = nextExtOutSlot[ADDR_BIT_WIDTH]++;
          auto &slot = lat.outputSlots[slotIdx];
          builder.connectPorts(lat.swGrid[slot.swRow][slot.swCol],
                               slot.switchPort, memInst, memInPort++);
        }

        unsigned memOutPort = 0;

        for (unsigned ld = 0; ld < memSpec.ldCount; ++ld) {
          auto &lat = lattices[memSpec.dataWidth];
          unsigned slotIdx = nextExtInSlot[memSpec.dataWidth]++;
          auto &slot = lat.inputSlots[slotIdx];
          builder.connectPorts(memInst, memOutPort++,
                               lat.swGrid[slot.swRow][slot.swCol],
                               slot.switchPort);
        }

        for (unsigned ld = 0; ld < memSpec.ldCount; ++ld) {
          auto &lat = lattices[0];
          unsigned slotIdx = nextExtInSlot[0]++;
          auto &slot = lat.inputSlots[slotIdx];
          builder.connectPorts(memInst, memOutPort++,
                               lat.swGrid[slot.swRow][slot.swCol],
                               slot.switchPort);
        }

        for (unsigned st = 0; st < memSpec.stCount; ++st) {
          auto &lat = lattices[0];
          unsigned slotIdx = nextExtInSlot[0]++;
          auto &slot = lat.inputSlots[slotIdx];
          builder.connectPorts(memInst, memOutPort++,
                               lat.swGrid[slot.swRow][slot.swCol],
                               slot.switchPort);
        }
      } else {
        // Multi-port memory: tagged bridge networks.
        // Memory has single tagged port per category. Per-lane lattice
        // slots connect through add_tag -> [temporal_sw mux] -> memory
        // for inputs, and memory -> [temporal_sw demux] -> del_tag ->
        // lattice for outputs.

        // --- INPUT: st_data ---
        if (memSpec.stCount > 0) {
          std::vector<InstanceHandle> stDataATs;
          for (unsigned st = 0; st < memSpec.stCount; ++st) {
            auto &lat = lattices[memSpec.dataWidth];
            unsigned slotIdx = nextExtOutSlot[memSpec.dataWidth]++;
            auto &slot = lat.outputSlots[slotIdx];
            InstanceHandle at =
                builder
                    .newAddTag(instName + "_st_data_at" +
                               std::to_string(st))
                    .setValueType(elemType)
                    .setTagType(tagType);
            builder.connectPorts(lat.swGrid[slot.swRow][slot.swCol],
                                 slot.switchPort, at, 0);
            stDataATs.push_back(at);
          }
          if (memSpec.stCount == 1) {
            builder.connectPorts(stDataATs[0], 0, memInst, memInPort);
          } else {
            InstanceHandle mux = builder.clone(
                stDataMuxDef, instName + "_st_data_mux");
            for (unsigned st = 0; st < memSpec.stCount; ++st)
              builder.connectPorts(stDataATs[st], 0, mux, st);
            builder.connectPorts(mux, 0, memInst, memInPort);
          }
          memInPort++;

          // --- INPUT: st_addr ---
          std::vector<InstanceHandle> stAddrATs;
          for (unsigned st = 0; st < memSpec.stCount; ++st) {
            auto &lat = lattices[ADDR_BIT_WIDTH];
            unsigned slotIdx = nextExtOutSlot[ADDR_BIT_WIDTH]++;
            auto &slot = lat.outputSlots[slotIdx];
            InstanceHandle at =
                builder
                    .newAddTag(instName + "_st_addr_at" +
                               std::to_string(st))
                    .setValueType(Type::bits(ADDR_BIT_WIDTH))
                    .setTagType(tagType);
            builder.connectPorts(lat.swGrid[slot.swRow][slot.swCol],
                                 slot.switchPort, at, 0);
            stAddrATs.push_back(at);
          }
          if (memSpec.stCount == 1) {
            builder.connectPorts(stAddrATs[0], 0, memInst, memInPort);
          } else {
            InstanceHandle mux = builder.clone(
                stAddrMuxDef, instName + "_st_addr_mux");
            for (unsigned st = 0; st < memSpec.stCount; ++st)
              builder.connectPorts(stAddrATs[st], 0, mux, st);
            builder.connectPorts(mux, 0, memInst, memInPort);
          }
          memInPort++;
        }

        // --- INPUT: ld_addr ---
        if (memSpec.ldCount > 0) {
          std::vector<InstanceHandle> ldAddrATs;
          for (unsigned ld = 0; ld < memSpec.ldCount; ++ld) {
            auto &lat = lattices[ADDR_BIT_WIDTH];
            unsigned slotIdx = nextExtOutSlot[ADDR_BIT_WIDTH]++;
            auto &slot = lat.outputSlots[slotIdx];
            InstanceHandle at =
                builder
                    .newAddTag(instName + "_ld_addr_at" +
                               std::to_string(ld))
                    .setValueType(Type::bits(ADDR_BIT_WIDTH))
                    .setTagType(tagType);
            builder.connectPorts(lat.swGrid[slot.swRow][slot.swCol],
                                 slot.switchPort, at, 0);
            ldAddrATs.push_back(at);
          }
          if (memSpec.ldCount == 1) {
            builder.connectPorts(ldAddrATs[0], 0, memInst, memInPort);
          } else {
            InstanceHandle mux = builder.clone(
                ldAddrMuxDef, instName + "_ld_addr_mux");
            for (unsigned ld = 0; ld < memSpec.ldCount; ++ld)
              builder.connectPorts(ldAddrATs[ld], 0, mux, ld);
            builder.connectPorts(mux, 0, memInst, memInPort);
          }
          memInPort++;
        }

        unsigned memOutPort = 0;

        // --- OUTPUT: ld_data ---
        if (memSpec.ldCount > 0) {
          if (memSpec.ldCount == 1) {
            InstanceHandle dt =
                builder
                    .newDelTag(instName + "_ld_data_dt0")
                    .setInputType(taggedDataType);
            builder.connectPorts(memInst, memOutPort, dt, 0);
            auto &lat = lattices[memSpec.dataWidth];
            unsigned slotIdx = nextExtInSlot[memSpec.dataWidth]++;
            auto &slot = lat.inputSlots[slotIdx];
            builder.connectPorts(
                dt, 0, lat.swGrid[slot.swRow][slot.swCol],
                slot.switchPort);
          } else {
            InstanceHandle demux = builder.clone(
                ldDataDemuxDef, instName + "_ld_data_demux");
            builder.connectPorts(memInst, memOutPort, demux, 0);
            for (unsigned ld = 0; ld < memSpec.ldCount; ++ld) {
              InstanceHandle dt =
                  builder
                      .newDelTag(instName + "_ld_data_dt" +
                                 std::to_string(ld))
                      .setInputType(taggedDataType);
              builder.connectPorts(demux, ld, dt, 0);
              auto &lat = lattices[memSpec.dataWidth];
              unsigned slotIdx = nextExtInSlot[memSpec.dataWidth]++;
              auto &slot = lat.inputSlots[slotIdx];
              builder.connectPorts(
                  dt, 0, lat.swGrid[slot.swRow][slot.swCol],
                  slot.switchPort);
            }
          }
          memOutPort++;

          // --- OUTPUT: ld_done ---
          if (memSpec.ldCount == 1) {
            InstanceHandle dt =
                builder
                    .newDelTag(instName + "_ld_done_dt0")
                    .setInputType(taggedDoneType);
            builder.connectPorts(memInst, memOutPort, dt, 0);
            auto &lat = lattices[0];
            unsigned slotIdx = nextExtInSlot[0]++;
            auto &slot = lat.inputSlots[slotIdx];
            builder.connectPorts(
                dt, 0, lat.swGrid[slot.swRow][slot.swCol],
                slot.switchPort);
          } else {
            InstanceHandle demux = builder.clone(
                ldDoneDemuxDef, instName + "_ld_done_demux");
            builder.connectPorts(memInst, memOutPort, demux, 0);
            for (unsigned ld = 0; ld < memSpec.ldCount; ++ld) {
              InstanceHandle dt =
                  builder
                      .newDelTag(instName + "_ld_done_dt" +
                                 std::to_string(ld))
                      .setInputType(taggedDoneType);
              builder.connectPorts(demux, ld, dt, 0);
              auto &lat = lattices[0];
              unsigned slotIdx = nextExtInSlot[0]++;
              auto &slot = lat.inputSlots[slotIdx];
              builder.connectPorts(
                  dt, 0, lat.swGrid[slot.swRow][slot.swCol],
                  slot.switchPort);
            }
          }
          memOutPort++;
        }

        // --- OUTPUT: st_done ---
        if (memSpec.stCount > 0) {
          if (memSpec.stCount == 1) {
            InstanceHandle dt =
                builder
                    .newDelTag(instName + "_st_done_dt0")
                    .setInputType(taggedDoneType);
            builder.connectPorts(memInst, memOutPort, dt, 0);
            auto &lat = lattices[0];
            unsigned slotIdx = nextExtInSlot[0]++;
            auto &slot = lat.inputSlots[slotIdx];
            builder.connectPorts(
                dt, 0, lat.swGrid[slot.swRow][slot.swCol],
                slot.switchPort);
          } else {
            InstanceHandle demux = builder.clone(
                stDoneDemuxDef, instName + "_st_done_demux");
            builder.connectPorts(memInst, memOutPort, demux, 0);
            for (unsigned st = 0; st < memSpec.stCount; ++st) {
              InstanceHandle dt =
                  builder
                      .newDelTag(instName + "_st_done_dt" +
                                 std::to_string(st))
                      .setInputType(taggedDoneType);
              builder.connectPorts(demux, st, dt, 0);
              auto &lat = lattices[0];
              unsigned slotIdx = nextExtInSlot[0]++;
              auto &slot = lat.inputSlots[slotIdx];
              builder.connectPorts(
                  dt, 0, lat.swGrid[slot.swRow][slot.swCol],
                  slot.switchPort);
            }
          }
          memOutPort++;
        }
      }
    }
    ++memIdx;
  }

  //=================================================================
  // Phase 7: Temporal domain (if requested)
  //=================================================================

  if (config.genTemporal) {
    // Compute starting slot indices for bridge connections.
    // Slots are ordered: [DFG I/O | ExtMem I/O | Bridge I/O].
    std::map<unsigned, unsigned> bridgeOutStart, bridgeInStart;
    for (auto &[w, _] : bridgeOutByWidth)
      bridgeOutStart[w] = dfgOutByWidth[w] + extMemOutByWidth[w];
    for (auto &[w, _] : bridgeInByWidth)
      bridgeInStart[w] = dfgInByWidth[w] + extMemInByWidth[w];
    generateTemporal(builder, reqs, config, lattices,
                     bridgeOutStart, bridgeInStart);
  }

  builder.exportMLIR(outputPath);
}

} // namespace adg
} // namespace loom
