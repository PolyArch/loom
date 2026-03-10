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
  if (ldCount != o.ldCount)
    return ldCount < o.ldCount;
  if (stCount != o.stCount)
    return stCount < o.stCount;
  return dataWidth < o.dataWidth;
}

bool MemorySpec::operator==(const MemorySpec &o) const {
  return ldCount == o.ldCount && stCount == o.stCount &&
         dataWidth == o.dataWidth;
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
    os << "mem" << spec.ldCount << "l" << spec.stCount << "s"
       << spec.dataWidth << "w" << "x" << count;
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

  auto [peRows, peCols] = computeMesh2DDims(
      totalCells > 0 ? totalCells : 1);
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
      // UL corner of cell (r, c): feeds PE input[0].
      if (r < (int)peRows && c < (int)peCols && cellGrid[r][c].numIn > 0)
        pm.peOut[0] = static_cast<int>(outIdx++);
      // LL corner of cell (r-1, c): feeds PE input[1].
      if (r > 0 && c < (int)peCols && cellGrid[r - 1][c].numIn > 1)
        pm.peOut[1] = static_cast<int>(outIdx++);
      // UR corner of cell (r, c-1): feeds PE input[2].
      if (r < (int)peRows && c > 0 && cellGrid[r][c - 1].numIn > 2)
        pm.peOut[2] = static_cast<int>(outIdx++);
      // LR corner of cell (r-1, c-1): feeds PE input[3].
      if (r > 0 && c > 0 && cellGrid[r - 1][c - 1].numIn > 3)
        pm.peOut[3] = static_cast<int>(outIdx++);

      // PE input ports (PE output -> switch in).
      // LR of cell (r-1, c-1): receives PE output[0].
      if (r > 0 && c > 0 && cellGrid[r - 1][c - 1].numOut > 0)
        pm.peIn[3] = static_cast<int>(inIdx++);
      // UR of cell (r, c-1): receives PE output[1].
      if (r < (int)peRows && c > 0 && cellGrid[r][c - 1].numOut > 1)
        pm.peIn[2] = static_cast<int>(inIdx++);
      // LL of cell (r-1, c): receives PE output[2].
      if (r > 0 && c < (int)peCols && cellGrid[r - 1][c].numOut > 2)
        pm.peIn[1] = static_cast<int>(inIdx++);
      // UL of cell (r, c): receives PE output[3].
      if (r < (int)peRows && c < (int)peCols && cellGrid[r][c].numOut > 3)
        pm.peIn[0] = static_cast<int>(inIdx++);

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

  if (isInput) {
    int swR, swC, slot;
    switch (localIdx) {
    case 0: swR = cellRow;     swC = cellCol;     slot = 0; break; // UL
    case 1: swR = cellRow + 1; swC = cellCol;     slot = 1; break; // LL
    case 2: swR = cellRow;     swC = cellCol + 1; slot = 2; break; // UR
    case 3: swR = cellRow + 1; swC = cellCol + 1; slot = 3; break; // LR
    default: return;
    }
    int swOut = lattice.swPorts[swR][swC].peOut[slot];
    assert(swOut >= 0);
    builder.connectPorts(lattice.swGrid[swR][swC], swOut, peInst, globalPort);
  } else {
    // Output corners reversed: LR=0, UR=1, LL=2, UL=3
    int swR, swC, slot;
    switch (localIdx) {
    case 0: swR = cellRow + 1; swC = cellCol + 1; slot = 3; break; // LR
    case 1: swR = cellRow;     swC = cellCol + 1; slot = 2; break; // UR
    case 2: swR = cellRow + 1; swC = cellCol;     slot = 1; break; // LL
    case 3: swR = cellRow;     swC = cellCol;     slot = 0; break; // UL
    default: return;
    }
    int swIn = lattice.swPorts[swR][swC].peIn[slot];
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
  for (const auto &[memSpec, count] : reqs.maxMemoryCounts) {
    for (unsigned i = 0; i < memSpec.ldCount * count; ++i) {
      PEPlacement p;
      p.kind = PEPlacement::MemLoadPE;
      p.dataWidth = memSpec.dataWidth;
      // LoadPE ports: in[0]=addr(57), in[1]=data(W), in[2]=ctrl(none)
      //              out[0]=addr(57), out[1]=data(W)
      p.portsByWidth[ADDR_BIT_WIDTH] = {{0}, {0}};
      p.portsByWidth[memSpec.dataWidth] = {{1}, {1}};
      p.portsByWidth[0] = {{2}, {}};

      for (auto &[w, wp] : p.portsByWidth)
        cellsByWidth[w].push_back(
            {(unsigned)wp.inPorts.size(), (unsigned)wp.outPorts.size()});

      placements.push_back(std::move(p));
    }
  }

  // StorePEs: same port layout as LoadPEs.
  for (const auto &[memSpec, count] : reqs.maxMemoryCounts) {
    for (unsigned i = 0; i < memSpec.stCount * count; ++i) {
      PEPlacement p;
      p.kind = PEPlacement::MemStorePE;
      p.dataWidth = memSpec.dataWidth;
      p.portsByWidth[ADDR_BIT_WIDTH] = {{0}, {0}};
      p.portsByWidth[memSpec.dataWidth] = {{1}, {1}};
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

  // Temporal bridge I/O: for each width with temporal PEs, add_tag needs
  // native output slots and del_tag needs native input slots.
  std::map<unsigned, unsigned> bridgeOutByWidth, bridgeInByWidth;
  if (config.genTemporal) {
    std::map<unsigned, unsigned> tempPECountByWidth;
    for (const auto &[spec, count] : reqs.peMaxCounts)
      tempPECountByWidth[spec.primaryWidth()] += count;
    for (auto &[w, peCount] : tempPECountByWidth) {
      auto [tpeRows, tpeCols] = computeMesh2DDims(peCount);
      unsigned tswRows = tpeRows + 1;
      bridgeOutByWidth[w] = tswRows;
      bridgeInByWidth[w] = tswRows;
    }
  }

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
      PEHandle peHandle = createPEDef(builder, *p.spec);
      std::string instName = p.spec->peName() + "_r" + std::to_string(nr) +
                             "_c" + std::to_string(nc);
      peInst = builder.clone(peHandle, instName);
      break;
    }
    case PEPlacement::MemLoadPE: {
      LoadPEHandle loadDef =
          builder.newLoadPE("load_pe_w" + std::to_string(p.dataWidth))
              .setDataType(widthToNativeType(p.dataWidth));
      std::string instName =
          "load_pe_r" + std::to_string(nr) + "_c" + std::to_string(nc);
      peInst = builder.clone(loadDef, instName);
      break;
    }
    case PEPlacement::MemStorePE: {
      StorePEHandle storeDef =
          builder.newStorePE("store_pe_w" + std::to_string(p.dataWidth))
              .setDataType(widthToNativeType(p.dataWidth));
      std::string instName =
          "store_pe_r" + std::to_string(nr) + "_c" + std::to_string(nc);
      peInst = builder.clone(storeDef, instName);
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
    Type elemType = widthToNativeType(memSpec.dataWidth);
    MemrefType memrefTy = MemrefType::dynamic1D(elemType);

    ExtMemoryHandle extMemDef =
        builder.newExtMemory("extmem_" + std::to_string(memIdx))
            .setLoadPorts(memSpec.ldCount)
            .setStorePorts(memSpec.stCount)
            .setShape(memrefTy);

    for (unsigned i = 0; i < count; ++i) {
      std::string instName =
          "extmem_" + std::to_string(memIdx) + "_" + std::to_string(i);
      InstanceHandle extInst = builder.clone(extMemDef, instName);

      // Connect memref module input to ExtMem port 0.
      std::string memPortName =
          "mem_" + std::to_string(memIdx) + "_" + std::to_string(i);
      PortHandle memPort = builder.addModuleInput(memPortName, memrefTy);
      builder.connectToModuleInput(memPort, extInst, 0);

      // ExtMem port layout (after memref):
      //   inputs:  [ld_addr_0..ld_addr_N, st_addr_0..st_addr_M, st_data_0..st_data_M]
      //   outputs: [ld_data_0..ld_data_N, ld_done_0..ld_done_N, st_done_0..st_done_M]
      unsigned extInPort = 1; // Start after memref

      // Load addr: lattice 57-bit output → ExtMem input
      for (unsigned ld = 0; ld < memSpec.ldCount; ++ld) {
        auto &lat = lattices[ADDR_BIT_WIDTH];
        unsigned slotIdx = nextExtOutSlot[ADDR_BIT_WIDTH]++;
        auto &slot = lat.outputSlots[slotIdx];
        builder.connectPorts(lat.swGrid[slot.swRow][slot.swCol],
                             slot.switchPort, extInst, extInPort++);
      }

      // Store addr: lattice 57-bit output → ExtMem input
      for (unsigned st = 0; st < memSpec.stCount; ++st) {
        auto &lat = lattices[ADDR_BIT_WIDTH];
        unsigned slotIdx = nextExtOutSlot[ADDR_BIT_WIDTH]++;
        auto &slot = lat.outputSlots[slotIdx];
        builder.connectPorts(lat.swGrid[slot.swRow][slot.swCol],
                             slot.switchPort, extInst, extInPort++);
      }

      // Store data: lattice W-bit output → ExtMem input
      for (unsigned st = 0; st < memSpec.stCount; ++st) {
        auto &lat = lattices[memSpec.dataWidth];
        unsigned slotIdx = nextExtOutSlot[memSpec.dataWidth]++;
        auto &slot = lat.outputSlots[slotIdx];
        builder.connectPorts(lat.swGrid[slot.swRow][slot.swCol],
                             slot.switchPort, extInst, extInPort++);
      }

      unsigned extOutPort = 0;

      // Load data: ExtMem output → lattice W-bit input
      for (unsigned ld = 0; ld < memSpec.ldCount; ++ld) {
        auto &lat = lattices[memSpec.dataWidth];
        unsigned slotIdx = nextExtInSlot[memSpec.dataWidth]++;
        auto &slot = lat.inputSlots[slotIdx];
        builder.connectPorts(extInst, extOutPort++,
                             lat.swGrid[slot.swRow][slot.swCol],
                             slot.switchPort);
      }

      // Load done: ExtMem output → lattice none input
      for (unsigned ld = 0; ld < memSpec.ldCount; ++ld) {
        auto &lat = lattices[0];
        unsigned slotIdx = nextExtInSlot[0]++;
        auto &slot = lat.inputSlots[slotIdx];
        builder.connectPorts(extInst, extOutPort++,
                             lat.swGrid[slot.swRow][slot.swCol],
                             slot.switchPort);
      }

      // Store done: ExtMem output → lattice none input
      for (unsigned st = 0; st < memSpec.stCount; ++st) {
        auto &lat = lattices[0];
        unsigned slotIdx = nextExtInSlot[0]++;
        auto &slot = lat.inputSlots[slotIdx];
        builder.connectPorts(extInst, extOutPort++,
                             lat.swGrid[slot.swRow][slot.swCol],
                             slot.switchPort);
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
