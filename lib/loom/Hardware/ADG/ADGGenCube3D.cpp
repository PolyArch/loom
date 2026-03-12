//===-- ADGGenCube3D.cpp - Cube3D topology for ADG generation ---*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Hardware/ADG/ADGGenHelpers.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// 3D dimension computation
//===----------------------------------------------------------------------===//

/// Compute Cube3D grid dimensions: ceil(cbrt(N))^3, returns (rows, cols, depths).
static std::tuple<unsigned, unsigned, unsigned>
computeCube3DDims(unsigned peCount) {
  if (peCount == 0)
    return {1, 1, 1};
  unsigned dim = static_cast<unsigned>(std::ceil(std::cbrt(peCount)));
  if (dim == 0)
    dim = 1;
  return {dim, dim, dim};
}

//===----------------------------------------------------------------------===//
// 3D port map
//===----------------------------------------------------------------------===//

/// Per-switch port map for 3D lattice.
/// 6 directions: 0=N, 1=E, 2=S, 3=W, 4=Up, 5=Down.
/// 8 PE corners per cell (vertices of the cube cell).
struct SwPortMap3D {
  unsigned numIn = 0;
  unsigned numOut = 0;
  std::vector<int> dirIn;   // [6 * T], -1 if absent
  std::vector<int> dirOut;  // [6 * T], -1 if absent
  int peOut[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int peIn[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  std::vector<int> modIn;
  std::vector<int> modOut;
};

//===----------------------------------------------------------------------===//
// 3D cell info
//===----------------------------------------------------------------------===//

struct CellInfo3D {
  unsigned numIn = 0;
  unsigned numOut = 0;
};

//===----------------------------------------------------------------------===//
// 3D plane-cut sweep for I/O assignment
//===----------------------------------------------------------------------===//

/// Enumerate boundary switches in a 3D grid via cutting planes from (0,0,0).
static std::vector<std::tuple<int, int, int>>
planeCutSweep3D(int swDepths, int swRows, int swCols) {
  std::vector<std::tuple<int, int, int>> result;
  int maxDist = (swDepths - 1) + (swRows - 1) + (swCols - 1);
  for (int k = 0; k <= maxDist; ++k) {
    for (int d = 0; d < swDepths; ++d) {
      for (int r = 0; r < swRows; ++r) {
        int c = k - d - r;
        if (c < 0 || c >= swCols)
          continue;
        bool onBoundary = (d == 0 || d == swDepths - 1 ||
                           r == 0 || r == swRows - 1 ||
                           c == 0 || c == swCols - 1);
        if (onBoundary)
          result.push_back({d, r, c});
      }
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// PE placement with per-width port mapping
//===----------------------------------------------------------------------===//

struct WidthPorts3D {
  std::vector<unsigned> inPorts;
  std::vector<unsigned> outPorts;
};

struct PEPlacement3D {
  enum Kind { Regular, Load, Store };
  Kind kind;
  const PESpec *spec = nullptr;
  unsigned dataWidth = 0;
  bool isFloat = false;
  std::map<unsigned, WidthPorts3D> portsByWidth;
};

//===----------------------------------------------------------------------===//
// Per-width cube grid data
//===----------------------------------------------------------------------===//

struct CubeGrid {
  unsigned peRows = 0, peCols = 0, peDepths = 0;
  int swRows = 0, swCols = 0, swDepths = 0;
  unsigned totalCells = 0;
  std::vector<std::vector<std::vector<InstanceHandle>>> swGrid;
  std::vector<std::vector<std::vector<SwPortMap3D>>> swPorts;
  std::vector<std::vector<std::vector<bool>>> pruned;
  std::vector<std::vector<std::vector<CellInfo3D>>> cells;
};

//===----------------------------------------------------------------------===//
// 8-corner topology definitions
//===----------------------------------------------------------------------===//

struct Corner3D {
  int dd, dr, dc;
  unsigned corner;
};

static const Corner3D kInCorners[8] = {
    {0, 0, 0, 0}, {1, 0, 0, 1}, {0, 1, 0, 2}, {0, 0, 1, 3},
    {0, 1, 1, 4}, {1, 0, 1, 5}, {1, 1, 0, 6}, {1, 1, 1, 7}};

static const Corner3D kOutCorners[8] = {
    {1, 1, 1, 7}, {1, 1, 0, 6}, {1, 0, 1, 5}, {0, 1, 1, 4},
    {0, 0, 1, 3}, {0, 1, 0, 2}, {1, 0, 0, 1}, {0, 0, 0, 0}};

//===----------------------------------------------------------------------===//
// Cube3D generation
//===----------------------------------------------------------------------===//

void generateCube3D(ADGBuilder &builder, const MergedRequirements &reqs,
                    const GenConfig &config) {

  //=================================================================
  // Build placement list with per-width port mapping
  //=================================================================

  std::vector<PEPlacement3D> placements;
  std::map<unsigned, unsigned> cellsPerWidth;

  // Regular PEs: group ports by width (mixed-width PEs span multiple planes).
  for (const auto &[spec, count] : reqs.peMaxCounts) {
    for (unsigned i = 0; i < count; ++i) {
      PEPlacement3D p;
      p.kind = PEPlacement3D::Regular;
      p.spec = &spec;
      for (unsigned k = 0; k < spec.inWidths.size(); ++k)
        p.portsByWidth[spec.inWidths[k]].inPorts.push_back(k);
      for (unsigned k = 0; k < spec.outWidths.size(); ++k)
        p.portsByWidth[spec.outWidths[k]].outPorts.push_back(k);
      for (auto &[w, _] : p.portsByWidth)
        cellsPerWidth[w]++;
      placements.push_back(std::move(p));
    }
  }

  // Load PEs: in[0]=addr(57), in[1]=data(W), in[2]=ctrl(none/0)
  //          out[0]=data(W), out[1]=addr(57)
  for (const auto &[memSpec, memCount] : reqs.maxMemoryCounts) {
    for (unsigned i = 0; i < memSpec.ldCount * memCount; ++i) {
      PEPlacement3D p;
      p.kind = PEPlacement3D::Load;
      p.dataWidth = memSpec.dataWidth;
      p.isFloat = memSpec.isFloat;
      p.portsByWidth[ADDR_BIT_WIDTH] = {{0}, {1}};
      p.portsByWidth[memSpec.dataWidth] = {{1}, {0}};
      p.portsByWidth[0] = {{2}, {}};
      for (auto &[w, _] : p.portsByWidth)
        cellsPerWidth[w]++;
      placements.push_back(std::move(p));
    }
    for (unsigned i = 0; i < memSpec.stCount * memCount; ++i) {
      PEPlacement3D p;
      p.kind = PEPlacement3D::Store;
      p.dataWidth = memSpec.dataWidth;
      p.isFloat = memSpec.isFloat;
      p.portsByWidth[ADDR_BIT_WIDTH] = {{0}, {1}};
      p.portsByWidth[memSpec.dataWidth] = {{1}, {0}};
      p.portsByWidth[0] = {{2}, {}};
      for (auto &[w, _] : p.portsByWidth)
        cellsPerWidth[w]++;
      placements.push_back(std::move(p));
    }
  }

  // Ensure I/O width planes exist.
  for (const auto &[w, _] : reqs.maxInputsByWidth)
    cellsPerWidth[w] += 0;
  for (const auto &[w, _] : reqs.maxOutputsByWidth)
    cellsPerWidth[w] += 0;

  unsigned T = config.numSwitchTrack;

  //=================================================================
  // Build cube grids per width plane
  //=================================================================

  std::map<unsigned, CubeGrid> cubeMap;

  // Cell assignment: [placementIdx][width] = (d, r, c)
  std::vector<std::map<unsigned, std::tuple<unsigned, unsigned, unsigned>>>
      placementCells(placements.size());

  unsigned globalModInIdx = 0;
  unsigned globalModOutIdx = 0;
  for (auto &[width, numCells] : cellsPerWidth) {
    CubeGrid &cube = cubeMap[width];
    if (numCells == 0)
      numCells = 1;
    // Apply PE margin to provide extra routing cells, matching mesh path.
    unsigned dimCells = numCells;
    if (config.peMargin > 0.0 && dimCells > 1)
      dimCells = static_cast<unsigned>(
          std::ceil(dimCells * (1.0 + config.peMargin)));

    auto [peRows, peCols, peDepths] = computeCube3DDims(dimCells);
    cube.peRows = peRows;
    cube.peCols = peCols;
    cube.peDepths = peDepths;
    cube.swRows = static_cast<int>(peRows) + 1;
    cube.swCols = static_cast<int>(peCols) + 1;
    cube.swDepths = static_cast<int>(peDepths) + 1;
    cube.totalCells = peRows * peCols * peDepths;
    std::string wp = std::to_string(width);

    // Prune excess cells from the far corner.
    unsigned excess =
        cube.totalCells > numCells ? cube.totalCells - numCells : 0;
    cube.pruned.assign(
        peDepths, std::vector<std::vector<bool>>(
                      peRows, std::vector<bool>(peCols, false)));
    {
      unsigned rem = excess;
      for (int d = (int)peDepths - 1; d >= 0 && rem > 0; --d)
        for (int c = (int)peCols - 1; c >= 0 && rem > 0; --c)
          for (int r = (int)peRows - 1; r >= 0 && rem > 0; --r) {
            cube.pruned[d][r][c] = true;
            --rem;
          }
    }

    // Assign placements to cells in this width plane.
    cube.cells.assign(
        peDepths, std::vector<std::vector<CellInfo3D>>(
                      peRows, std::vector<CellInfo3D>(peCols)));
    unsigned cellIdx = 0;
    for (unsigned pi = 0; pi < placements.size(); ++pi) {
      auto it = placements[pi].portsByWidth.find(width);
      if (it == placements[pi].portsByWidth.end())
        continue;
      auto &wp_data = it->second;
      while (cellIdx < cube.totalCells) {
        unsigned d = cellIdx / (peRows * peCols);
        unsigned r = (cellIdx % (peRows * peCols)) / peCols;
        unsigned c = cellIdx % peCols;
        if (!cube.pruned[d][r][c]) {
          cube.cells[d][r][c] = {(unsigned)wp_data.inPorts.size(),
                                 (unsigned)wp_data.outPorts.size()};
          placementCells[pi][width] = {d, r, c};
          ++cellIdx;
          break;
        }
        ++cellIdx;
      }
    }

    // Module I/O counts for this width.
    unsigned numModIn = 0, numModOut = 0;
    {
      auto it2 = reqs.maxInputsByWidth.find(width);
      if (it2 != reqs.maxInputsByWidth.end())
        numModIn = it2->second;
    }
    {
      auto it2 = reqs.maxOutputsByWidth.find(width);
      if (it2 != reqs.maxOutputsByWidth.end())
        numModOut = it2->second;
    }

    // 3D plane-cut sweep for I/O assignment.
    auto sweep3D = planeCutSweep3D(cube.swDepths, cube.swRows, cube.swCols);

    struct ModIOAssign3D {
      int swD, swR, swC;
    };

    std::vector<ModIOAssign3D> modInAssigns;
    {
      unsigned assigned = 0;
      for (auto [d, r, c] : sweep3D) {
        if (assigned >= numModIn)
          break;
        bool hasDir[6] = {r > 0,
                          c < cube.swCols - 1,
                          r < cube.swRows - 1,
                          c > 0,
                          d > 0,
                          d < cube.swDepths - 1};
        for (int dir = 0; dir < 6 && assigned < numModIn; ++dir) {
          if (!hasDir[dir]) {
            for (unsigned t = 0; t < T && assigned < numModIn; ++t) {
              modInAssigns.push_back({d, r, c});
              ++assigned;
            }
          }
        }
      }
    }

    std::vector<ModIOAssign3D> modOutAssigns;
    {
      unsigned assigned = 0;
      for (int i = (int)sweep3D.size() - 1; i >= 0; --i) {
        if (assigned >= numModOut)
          break;
        auto [d, r, c] = sweep3D[i];
        bool hasDir[6] = {r > 0,
                          c < cube.swCols - 1,
                          r < cube.swRows - 1,
                          c > 0,
                          d > 0,
                          d < cube.swDepths - 1};
        for (int dir = 5; dir >= 0 && assigned < numModOut; --dir) {
          if (!hasDir[dir]) {
            for (int t = (int)T - 1;
                 t >= 0 && assigned < numModOut; --t) {
              modOutAssigns.push_back({d, r, c});
              ++assigned;
            }
          }
        }
      }
    }

    std::map<std::tuple<int, int, int>, unsigned> swModInCnt, swModOutCnt;
    for (auto &a : modInAssigns)
      swModInCnt[{a.swD, a.swR, a.swC}]++;
    for (auto &a : modOutAssigns)
      swModOutCnt[{a.swD, a.swR, a.swC}]++;

    //=================================================================
    // Compute per-switch 3D port maps
    //=================================================================

    cube.swPorts.assign(
        cube.swDepths,
        std::vector<std::vector<SwPortMap3D>>(
            cube.swRows, std::vector<SwPortMap3D>(cube.swCols)));

    for (int d = 0; d < cube.swDepths; ++d) {
      for (int r = 0; r < cube.swRows; ++r) {
        for (int c = 0; c < cube.swCols; ++c) {
          auto &pm = cube.swPorts[d][r][c];
          pm.dirIn.assign(6 * T, -1);
          pm.dirOut.assign(6 * T, -1);
          unsigned inIdx = 0, outIdx = 0;

          bool hasDir[6] = {r > 0,
                            c < cube.swCols - 1,
                            r < cube.swRows - 1,
                            c > 0,
                            d > 0,
                            d < cube.swDepths - 1};
          for (int dir = 0; dir < 6; ++dir) {
            if (hasDir[dir]) {
              for (unsigned t = 0; t < T; ++t) {
                pm.dirIn[dir * T + t] = static_cast<int>(inIdx++);
                pm.dirOut[dir * T + t] = static_cast<int>(outIdx++);
              }
            }
          }

          // PE output ports (switch out -> PE input) for 8 corners.
          if (d < (int)peDepths && r < (int)peRows && c < (int)peCols &&
              cube.cells[d][r][c].numIn > 0)
            pm.peOut[0] = static_cast<int>(outIdx++);
          if (d > 0 && r < (int)peRows && c < (int)peCols &&
              cube.cells[d - 1][r][c].numIn > 1)
            pm.peOut[1] = static_cast<int>(outIdx++);
          if (d < (int)peDepths && r > 0 && c < (int)peCols &&
              cube.cells[d][r - 1][c].numIn > 2)
            pm.peOut[2] = static_cast<int>(outIdx++);
          if (d < (int)peDepths && r < (int)peRows && c > 0 &&
              cube.cells[d][r][c - 1].numIn > 3)
            pm.peOut[3] = static_cast<int>(outIdx++);
          if (d < (int)peDepths && r > 0 && c > 0 &&
              cube.cells[d][r - 1][c - 1].numIn > 4)
            pm.peOut[4] = static_cast<int>(outIdx++);
          if (d > 0 && r < (int)peRows && c > 0 &&
              cube.cells[d - 1][r][c - 1].numIn > 5)
            pm.peOut[5] = static_cast<int>(outIdx++);
          if (d > 0 && r > 0 && c < (int)peCols &&
              cube.cells[d - 1][r - 1][c].numIn > 6)
            pm.peOut[6] = static_cast<int>(outIdx++);
          if (d > 0 && r > 0 && c > 0 &&
              cube.cells[d - 1][r - 1][c - 1].numIn > 7)
            pm.peOut[7] = static_cast<int>(outIdx++);

          // PE input ports (PE output -> switch in) in reverse corner order.
          if (d > 0 && r > 0 && c > 0 &&
              cube.cells[d - 1][r - 1][c - 1].numOut > 0)
            pm.peIn[7] = static_cast<int>(inIdx++);
          if (d > 0 && r > 0 && c < (int)peCols &&
              cube.cells[d - 1][r - 1][c].numOut > 1)
            pm.peIn[6] = static_cast<int>(inIdx++);
          if (d > 0 && r < (int)peRows && c > 0 &&
              cube.cells[d - 1][r][c - 1].numOut > 2)
            pm.peIn[5] = static_cast<int>(inIdx++);
          if (d < (int)peDepths && r > 0 && c > 0 &&
              cube.cells[d][r - 1][c - 1].numOut > 3)
            pm.peIn[4] = static_cast<int>(inIdx++);
          if (d < (int)peDepths && r < (int)peRows && c > 0 &&
              cube.cells[d][r][c - 1].numOut > 4)
            pm.peIn[3] = static_cast<int>(inIdx++);
          if (d < (int)peDepths && r > 0 && c < (int)peCols &&
              cube.cells[d][r - 1][c].numOut > 5)
            pm.peIn[2] = static_cast<int>(inIdx++);
          if (d > 0 && r < (int)peRows && c < (int)peCols &&
              cube.cells[d - 1][r][c].numOut > 6)
            pm.peIn[1] = static_cast<int>(inIdx++);
          if (d < (int)peDepths && r < (int)peRows && c < (int)peCols &&
              cube.cells[d][r][c].numOut > 7)
            pm.peIn[0] = static_cast<int>(inIdx++);

          // Module I/O ports.
          auto inIt = swModInCnt.find({d, r, c});
          unsigned nModIn = inIt != swModInCnt.end() ? inIt->second : 0;
          for (unsigned i = 0; i < nModIn; ++i)
            pm.modIn.push_back(static_cast<int>(inIdx++));

          auto outIt = swModOutCnt.find({d, r, c});
          unsigned nModOut = outIt != swModOutCnt.end() ? outIt->second : 0;
          for (unsigned i = 0; i < nModOut; ++i)
            pm.modOut.push_back(static_cast<int>(outIdx++));

          pm.numIn = inIdx;
          pm.numOut = outIdx;
        }
      }
    }

    //=================================================================
    // Create switch templates and instances
    //=================================================================

    std::map<std::pair<unsigned, unsigned>, SwitchHandle> swTemplates;
    for (int d = 0; d < cube.swDepths; ++d) {
      for (int r = 0; r < cube.swRows; ++r) {
        for (int c = 0; c < cube.swCols; ++c) {
          auto &pm = cube.swPorts[d][r][c];
          auto key = std::make_pair(pm.numIn, pm.numOut);
          if (swTemplates.find(key) == swTemplates.end()) {
            std::string name = "sw_w" + wp + "_" +
                               std::to_string(pm.numIn) + "x" +
                               std::to_string(pm.numOut);
            swTemplates[key] = builder.newSwitch(name)
                                   .setPortCount(pm.numIn, pm.numOut)
                                   .setType(widthToNativeType(width));
          }
        }
      }
    }

    cube.swGrid.assign(
        cube.swDepths,
        std::vector<std::vector<InstanceHandle>>(
            cube.swRows, std::vector<InstanceHandle>(cube.swCols)));
    for (int d = 0; d < cube.swDepths; ++d) {
      for (int r = 0; r < cube.swRows; ++r) {
        for (int c = 0; c < cube.swCols; ++c) {
          auto &pm = cube.swPorts[d][r][c];
          auto key = std::make_pair(pm.numIn, pm.numOut);
          std::string instName = "sw_w" + wp + "_" + std::to_string(d) +
                                 "_" + std::to_string(r) + "_" +
                                 std::to_string(c);
          cube.swGrid[d][r][c] =
              builder.clone(swTemplates[key], instName);
        }
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
                    .setType(widthToNativeType(width));
      if (config.fifoBypassable)
        fb.setBypassable(true);
      fwdFifo = fb;
    }
    if (useRev) {
      auto fb = builder.newFifo("rev_fifo_w" + wp)
                    .setDepth(config.fifoDepth)
                    .setType(widthToNativeType(width));
      if (config.fifoBypassable)
        fb.setBypassable(true);
      revFifo = fb;
    }

    auto connectDir3D = [&](InstanceHandle srcSw, int srcOut,
                            InstanceHandle dstSw, int dstIn, bool viaFifo,
                            FifoHandle fifoTmpl, const std::string &fName) {
      if (viaFifo) {
        auto f = builder.clone(fifoTmpl, fName);
        builder.connectPorts(srcSw, srcOut, f, 0);
        builder.connectPorts(f, 0, dstSw, dstIn);
      } else {
        builder.connectPorts(srcSw, srcOut, dstSw, dstIn);
      }
    };

    for (int d = 0; d < cube.swDepths; ++d) {
      for (int r = 0; r < cube.swRows; ++r) {
        for (int c = 0; c < cube.swCols; ++c) {
          auto &pm = cube.swPorts[d][r][c];
          std::string pos = std::to_string(d) + "_" +
                            std::to_string(r) + "_" + std::to_string(c);

          // East: (d,r,c) -> (d,r,c+1)
          if (c + 1 < cube.swCols) {
            auto &pmE = cube.swPorts[d][r][c + 1];
            for (unsigned t = 0; t < T; ++t) {
              int so = pm.dirOut[1 * T + t], di = pmE.dirIn[3 * T + t];
              assert(so >= 0 && di >= 0);
              connectDir3D(cube.swGrid[d][r][c], so,
                           cube.swGrid[d][r][c + 1], di, useFwd, fwdFifo,
                           "fifo_e_t" + std::to_string(t) + "_" + pos);
              int rso = pmE.dirOut[3 * T + t], rdi = pm.dirIn[1 * T + t];
              assert(rso >= 0 && rdi >= 0);
              connectDir3D(cube.swGrid[d][r][c + 1], rso,
                           cube.swGrid[d][r][c], rdi, useRev, revFifo,
                           "fifo_w_t" + std::to_string(t) + "_" + pos);
            }
          }

          // South: (d,r,c) -> (d,r+1,c)
          if (r + 1 < cube.swRows) {
            auto &pmS = cube.swPorts[d][r + 1][c];
            for (unsigned t = 0; t < T; ++t) {
              int so = pm.dirOut[2 * T + t], di = pmS.dirIn[0 * T + t];
              assert(so >= 0 && di >= 0);
              connectDir3D(cube.swGrid[d][r][c], so,
                           cube.swGrid[d][r + 1][c], di, useFwd, fwdFifo,
                           "fifo_s_t" + std::to_string(t) + "_" + pos);
              int rso = pmS.dirOut[0 * T + t], rdi = pm.dirIn[2 * T + t];
              assert(rso >= 0 && rdi >= 0);
              connectDir3D(cube.swGrid[d][r + 1][c], rso,
                           cube.swGrid[d][r][c], rdi, useRev, revFifo,
                           "fifo_n_t" + std::to_string(t) + "_" + pos);
            }
          }

          // Down: (d,r,c) -> (d+1,r,c)
          if (d + 1 < cube.swDepths) {
            auto &pmD = cube.swPorts[d + 1][r][c];
            for (unsigned t = 0; t < T; ++t) {
              int so = pm.dirOut[5 * T + t], di = pmD.dirIn[4 * T + t];
              assert(so >= 0 && di >= 0);
              connectDir3D(cube.swGrid[d][r][c], so,
                           cube.swGrid[d + 1][r][c], di, useFwd, fwdFifo,
                           "fifo_d_t" + std::to_string(t) + "_" + pos);
              int rso = pmD.dirOut[4 * T + t], rdi = pm.dirIn[5 * T + t];
              assert(rso >= 0 && rdi >= 0);
              connectDir3D(cube.swGrid[d + 1][r][c], rso,
                           cube.swGrid[d][r][c], rdi, useRev, revFifo,
                           "fifo_u_t" + std::to_string(t) + "_" + pos);
            }
          }
        }
      }
    }

    //=================================================================
    // Module I/O
    //=================================================================

    {
      std::map<std::tuple<int, int, int>, unsigned> swModInIdx;
      for (unsigned k = 0; k < modInAssigns.size(); ++k) {
        auto &a = modInAssigns[k];
        auto port = builder.addModuleInput(
            "in_" + std::to_string(globalModInIdx++),
            widthToNativeType(width));
        unsigned localIdx = swModInIdx[{a.swD, a.swR, a.swC}]++;
        auto &pm = cube.swPorts[a.swD][a.swR][a.swC];
        builder.connectToModuleInput(
            port, cube.swGrid[a.swD][a.swR][a.swC], pm.modIn[localIdx]);
      }
    }
    {
      std::map<std::tuple<int, int, int>, unsigned> swModOutIdx;
      for (unsigned k = 0; k < modOutAssigns.size(); ++k) {
        auto &a = modOutAssigns[k];
        auto port = builder.addModuleOutput(
            "out_" + std::to_string(globalModOutIdx++),
            widthToNativeType(width));
        unsigned localIdx = swModOutIdx[{a.swD, a.swR, a.swC}]++;
        auto &pm = cube.swPorts[a.swD][a.swR][a.swC];
        builder.connectToModuleOutput(
            cube.swGrid[a.swD][a.swR][a.swC], pm.modOut[localIdx], port);
      }
    }
  } // end per-width cube loop

  //=================================================================
  // Create PE instances and wire to cube grids
  //=================================================================

  // Multi-width wiring: connect PE ports to the correct width plane's
  // cube switches based on per-port width assignment.
  auto wirePE3DPorts = [&](InstanceHandle peInst, int pd, int pr, int pc,
                           const std::vector<unsigned> &inPortIndices,
                           const std::vector<unsigned> &outPortIndices,
                           CubeGrid &cube) {
    for (unsigned k = 0; k < inPortIndices.size() && k < 8; ++k) {
      auto &ic = kInCorners[k];
      int sd = pd + ic.dd, sr = pr + ic.dr, sc = pc + ic.dc;
      int swOut = cube.swPorts[sd][sr][sc].peOut[ic.corner];
      if (swOut >= 0) {
        builder.connectPorts(cube.swGrid[sd][sr][sc], swOut, peInst,
                             static_cast<int>(inPortIndices[k]));
      }
    }
    for (unsigned k = 0; k < outPortIndices.size() && k < 8; ++k) {
      auto &oc = kOutCorners[k];
      int sd = pd + oc.dd, sr = pr + oc.dr, sc = pc + oc.dc;
      int swIn = cube.swPorts[sd][sr][sc].peIn[oc.corner];
      if (swIn >= 0) {
        builder.connectPorts(peInst,
                             static_cast<int>(outPortIndices[k]),
                             cube.swGrid[sd][sr][sc], swIn);
      }
    }
  };

  // PE template caches.
  std::map<std::string, PEHandle> peTemplateCache;
  std::map<unsigned, LoadPEHandle> loadTemplateCache;
  std::map<unsigned, StorePEHandle> storeTemplateCache;

  for (unsigned pi = 0; pi < placements.size(); ++pi) {
    auto &p = placements[pi];

    // Determine the primary width for naming (first available cell position).
    unsigned primaryW = 0;
    if (p.kind == PEPlacement3D::Regular)
      primaryW = p.spec->primaryWidth();
    else
      primaryW = p.dataWidth;
    auto cellIt = placementCells[pi].find(primaryW);
    if (cellIt == placementCells[pi].end()) {
      // Fallback: use the first available width.
      if (placementCells[pi].empty())
        continue;
      cellIt = placementCells[pi].begin();
      primaryW = cellIt->first;
    }
    auto [pd, pr, pc] = cellIt->second;

    // Create PE instance.
    InstanceHandle peInst{0};
    if (p.kind == PEPlacement3D::Regular) {
      std::string peKey = p.spec->peName();
      if (peTemplateCache.find(peKey) == peTemplateCache.end())
        peTemplateCache[peKey] = createPEDef(builder, *p.spec);
      std::string instName = peKey + "_d" + std::to_string(pd) + "_r" +
                             std::to_string(pr) + "_c" +
                             std::to_string(pc);
      peInst = builder.clone(peTemplateCache[peKey], instName);
    } else if (p.kind == PEPlacement3D::Load) {
      unsigned key = p.dataWidth | (p.isFloat ? 0x80000000u : 0);
      if (loadTemplateCache.find(key) == loadTemplateCache.end()) {
        Type dt = p.isFloat ? widthToFloatType(p.dataWidth)
                            : widthToNativeType(p.dataWidth);
        std::string suffix = (p.isFloat ? "f" : "w") +
                             std::to_string(p.dataWidth);
        loadTemplateCache[key] =
            builder.newLoadPE("load_pe_" + suffix).setDataType(dt);
      }
      std::string instName = "load_pe_d" + std::to_string(pd) + "_r" +
                             std::to_string(pr) + "_c" +
                             std::to_string(pc);
      peInst = builder.clone(loadTemplateCache[key], instName);
    } else {
      unsigned key = p.dataWidth | (p.isFloat ? 0x80000000u : 0);
      if (storeTemplateCache.find(key) == storeTemplateCache.end()) {
        Type dt = p.isFloat ? widthToFloatType(p.dataWidth)
                            : widthToNativeType(p.dataWidth);
        std::string suffix = (p.isFloat ? "f" : "w") +
                             std::to_string(p.dataWidth);
        storeTemplateCache[key] =
            builder.newStorePE("store_pe_" + suffix).setDataType(dt);
      }
      std::string instName = "store_pe_d" + std::to_string(pd) + "_r" +
                             std::to_string(pr) + "_c" +
                             std::to_string(pc);
      peInst = builder.clone(storeTemplateCache[key], instName);
    }

    // Wire PE ports to each width plane's cube.
    for (auto &[width, wp_data] : p.portsByWidth) {
      auto pcIt = placementCells[pi].find(width);
      if (pcIt == placementCells[pi].end())
        continue;
      auto cubeIt = cubeMap.find(width);
      if (cubeIt == cubeMap.end())
        continue;
      auto [wd, wr, wc] = pcIt->second;
      wirePE3DPorts(peInst, wd, wr, wc, wp_data.inPorts,
                    wp_data.outPorts, cubeIt->second);
    }
  }
}

} // namespace adg
} // namespace loom
