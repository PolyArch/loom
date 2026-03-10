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
  const PESpec *spec = nullptr;
  unsigned numIn = 0;
  unsigned numOut = 0;
};

//===----------------------------------------------------------------------===//
// 3D plane-cut sweep for I/O assignment
//===----------------------------------------------------------------------===//

/// Enumerate boundary switches in a 3D grid via cutting planes from (0,0,0).
/// A switch at (d, r, c) is "boundary" if at least one of its 6 directions
/// has no neighbor (i.e., it's on the surface of the grid).
/// Planes sweep outward: distance k = d + r + c, from 0 to maxDist.
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
        // Check if this switch is on the boundary.
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
// Cube3D generation
//===----------------------------------------------------------------------===//

void generateCube3D(ADGBuilder &builder, const MergedRequirements &reqs,
                    const GenConfig &config) {

  // Group PEs by width plane.
  std::map<unsigned, std::vector<std::pair<PESpec, unsigned>>> pesByWidth;
  for (const auto &[spec, count] : reqs.peMaxCounts) {
    unsigned w = spec.primaryWidth();
    pesByWidth[w].push_back({spec, count});
  }
  for (const auto &[w, _] : reqs.maxInputsByWidth)
    pesByWidth[w];
  for (const auto &[w, _] : reqs.maxOutputsByWidth)
    pesByWidth[w];

  // Count memory PEs per width plane.
  std::map<unsigned, unsigned> loadPEsByWidth;
  std::map<unsigned, unsigned> storePEsByWidth;
  for (const auto &[memSpec, count] : reqs.maxMemoryCounts) {
    loadPEsByWidth[memSpec.dataWidth] += memSpec.ldCount * count;
    storePEsByWidth[memSpec.dataWidth] += memSpec.stCount * count;
    pesByWidth[memSpec.dataWidth];
  }

  unsigned T = config.numSwitchTrack;

  for (auto &[width, peList] : pesByWidth) {
    unsigned totalPEs = 0;
    for (const auto &[spec, count] : peList)
      totalPEs += count;
    unsigned nLoad = 0, nStore = 0;
    {
      auto it = loadPEsByWidth.find(width);
      if (it != loadPEsByWidth.end())
        nLoad = it->second;
    }
    {
      auto it = storePEsByWidth.find(width);
      if (it != storePEsByWidth.end())
        nStore = it->second;
    }
    totalPEs += nLoad + nStore;

    auto [peRows, peCols, peDepths] = computeCube3DDims(totalPEs);
    int swRows = static_cast<int>(peRows) + 1;
    int swCols = static_cast<int>(peCols) + 1;
    int swDepths = static_cast<int>(peDepths) + 1;
    std::string wp = std::to_string(width);

    // Prune excess cells: iterate (d, c, r) removing from last.
    unsigned totalCells = peRows * peCols * peDepths;
    unsigned excessCells = totalCells > totalPEs ? totalCells - totalPEs : 0;

    // Mark pruned cells.
    std::vector<std::vector<std::vector<bool>>> pruned(
        peDepths,
        std::vector<std::vector<bool>>(peRows, std::vector<bool>(peCols, false)));
    {
      unsigned remaining = excessCells;
      for (int d = static_cast<int>(peDepths) - 1; d >= 0 && remaining > 0; --d) {
        for (int c = static_cast<int>(peCols) - 1; c >= 0 && remaining > 0; --c) {
          for (int r = static_cast<int>(peRows) - 1; r >= 0 && remaining > 0; --r) {
            pruned[d][r][c] = true;
            --remaining;
          }
        }
      }
    }

    // Populate 3D PE cell grid.
    std::vector<std::vector<std::vector<CellInfo3D>>> cells(
        peDepths,
        std::vector<std::vector<CellInfo3D>>(
            peRows, std::vector<CellInfo3D>(peCols)));
    {
      unsigned cellIdx = 0;
      for (const auto &[spec, count] : peList) {
        for (unsigned i = 0; i < count; ++i) {
          // Find next non-pruned cell.
          while (cellIdx < totalCells) {
            unsigned d = cellIdx / (peRows * peCols);
            unsigned rem = cellIdx % (peRows * peCols);
            unsigned r = rem / peCols;
            unsigned c = rem % peCols;
            if (!pruned[d][r][c]) {
              cells[d][r][c] = {&spec, (unsigned)spec.inWidths.size(),
                                (unsigned)spec.outWidths.size()};
              ++cellIdx;
              break;
            }
            ++cellIdx;
          }
        }
      }
      // LoadPEs (3 inputs, 2 outputs).
      for (unsigned i = 0; i < nLoad; ++i) {
        while (cellIdx < totalCells) {
          unsigned d = cellIdx / (peRows * peCols);
          unsigned rem = cellIdx % (peRows * peCols);
          unsigned r = rem / peCols;
          unsigned c = rem % peCols;
          if (!pruned[d][r][c]) {
            cells[d][r][c] = {nullptr, 3, 2};
            ++cellIdx;
            break;
          }
          ++cellIdx;
        }
      }
      // StorePEs (3 inputs, 2 outputs).
      for (unsigned i = 0; i < nStore; ++i) {
        while (cellIdx < totalCells) {
          unsigned d = cellIdx / (peRows * peCols);
          unsigned rem = cellIdx % (peRows * peCols);
          unsigned r = rem / peCols;
          unsigned c = rem % peCols;
          if (!pruned[d][r][c]) {
            cells[d][r][c] = {nullptr, 3, 2};
            ++cellIdx;
            break;
          }
          ++cellIdx;
        }
      }
    }

    // Module I/O counts.
    unsigned numModIn = 0, numModOut = 0;
    {
      auto it = reqs.maxInputsByWidth.find(width);
      if (it != reqs.maxInputsByWidth.end())
        numModIn = it->second;
    }
    {
      auto it = reqs.maxOutputsByWidth.find(width);
      if (it != reqs.maxOutputsByWidth.end())
        numModOut = it->second;
    }

    // 3D plane-cut sweep for I/O assignment.
    auto sweep3D = planeCutSweep3D(swDepths, swRows, swCols);

    struct ModIOAssign3D { int swD, swR, swC; };

    // Assign module inputs from (0,0,0) corner outward.
    std::vector<ModIOAssign3D> modInAssigns;
    {
      unsigned assigned = 0;
      for (auto [d, r, c] : sweep3D) {
        if (assigned >= numModIn)
          break;
        bool hasDir[6] = {r > 0, c < swCols - 1, r < swRows - 1, c > 0,
                          d > 0, d < swDepths - 1};
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

    // Assign module outputs from far corner inward.
    std::vector<ModIOAssign3D> modOutAssigns;
    {
      unsigned assigned = 0;
      for (int i = static_cast<int>(sweep3D.size()) - 1; i >= 0; --i) {
        if (assigned >= numModOut)
          break;
        auto [d, r, c] = sweep3D[i];
        bool hasDir[6] = {r > 0, c < swCols - 1, r < swRows - 1, c > 0,
                          d > 0, d < swDepths - 1};
        for (int dir = 5; dir >= 0 && assigned < numModOut; --dir) {
          if (!hasDir[dir]) {
            for (int t = static_cast<int>(T) - 1;
                 t >= 0 && assigned < numModOut; --t) {
              modOutAssigns.push_back({d, r, c});
              ++assigned;
            }
          }
        }
      }
    }

    // Count module I/O per switch.
    std::map<std::tuple<int, int, int>, unsigned> swModInCnt, swModOutCnt;
    for (auto &a : modInAssigns)
      swModInCnt[{a.swD, a.swR, a.swC}]++;
    for (auto &a : modOutAssigns)
      swModOutCnt[{a.swD, a.swR, a.swC}]++;

    //=================================================================
    // Compute per-switch 3D port maps
    //=================================================================

    std::vector<std::vector<std::vector<SwPortMap3D>>> swPorts(
        swDepths,
        std::vector<std::vector<SwPortMap3D>>(
            swRows, std::vector<SwPortMap3D>(swCols)));

    // 8-corner mapping: for cell (d, r, c), corner index maps to switch offset.
    // corner 0: (d, r, c)        corner 1: (d+1, r, c)
    // corner 2: (d, r+1, c)      corner 3: (d, r, c+1)
    // corner 4: (d, r+1, c+1)    corner 5: (d+1, r, c+1)
    // corner 6: (d+1, r+1, c)    corner 7: (d+1, r+1, c+1)
    // Input order: 0,1,2,3,4,5,6,7. Output order: 7,6,5,4,3,2,1,0 (reversed).

    for (int d = 0; d < swDepths; ++d) {
      for (int r = 0; r < swRows; ++r) {
        for (int c = 0; c < swCols; ++c) {
          auto &pm = swPorts[d][r][c];
          pm.dirIn.assign(6 * T, -1);
          pm.dirOut.assign(6 * T, -1);
          unsigned inIdx = 0, outIdx = 0;

          // 6 directional ports.
          bool hasDir[6] = {
              r > 0,              // N
              c < swCols - 1,     // E
              r < swRows - 1,     // S
              c > 0,              // W
              d > 0,              // Up
              d < swDepths - 1    // Down
          };
          for (int dir = 0; dir < 6; ++dir) {
            if (hasDir[dir]) {
              for (unsigned t = 0; t < T; ++t) {
                pm.dirIn[dir * T + t] = static_cast<int>(inIdx++);
                pm.dirOut[dir * T + t] = static_cast<int>(outIdx++);
              }
            }
          }

          // PE output ports (switch out -> PE input) for 8 corners.
          // Corner 0: cell(d, r, c) input[0]
          if (d < (int)peDepths && r < (int)peRows && c < (int)peCols &&
              cells[d][r][c].numIn > 0)
            pm.peOut[0] = static_cast<int>(outIdx++);
          // Corner 1: cell(d-1, r, c) input[1]
          if (d > 0 && r < (int)peRows && c < (int)peCols &&
              cells[d - 1][r][c].numIn > 1)
            pm.peOut[1] = static_cast<int>(outIdx++);
          // Corner 2: cell(d, r-1, c) input[2]
          if (d < (int)peDepths && r > 0 && c < (int)peCols &&
              cells[d][r - 1][c].numIn > 2)
            pm.peOut[2] = static_cast<int>(outIdx++);
          // Corner 3: cell(d, r, c-1) input[3]
          if (d < (int)peDepths && r < (int)peRows && c > 0 &&
              cells[d][r][c - 1].numIn > 3)
            pm.peOut[3] = static_cast<int>(outIdx++);
          // Corner 4: cell(d, r-1, c-1) input[4]
          if (d < (int)peDepths && r > 0 && c > 0 &&
              cells[d][r - 1][c - 1].numIn > 4)
            pm.peOut[4] = static_cast<int>(outIdx++);
          // Corner 5: cell(d-1, r, c-1) input[5]
          if (d > 0 && r < (int)peRows && c > 0 &&
              cells[d - 1][r][c - 1].numIn > 5)
            pm.peOut[5] = static_cast<int>(outIdx++);
          // Corner 6: cell(d-1, r-1, c) input[6]
          if (d > 0 && r > 0 && c < (int)peCols &&
              cells[d - 1][r - 1][c].numIn > 6)
            pm.peOut[6] = static_cast<int>(outIdx++);
          // Corner 7: cell(d-1, r-1, c-1) input[7]
          if (d > 0 && r > 0 && c > 0 &&
              cells[d - 1][r - 1][c - 1].numIn > 7)
            pm.peOut[7] = static_cast<int>(outIdx++);

          // PE input ports (PE output -> switch in) in reverse corner order.
          // Corner 7: cell(d-1, r-1, c-1) output[0]
          if (d > 0 && r > 0 && c > 0 &&
              cells[d - 1][r - 1][c - 1].numOut > 0)
            pm.peIn[7] = static_cast<int>(inIdx++);
          // Corner 6: cell(d-1, r-1, c) output[1]
          if (d > 0 && r > 0 && c < (int)peCols &&
              cells[d - 1][r - 1][c].numOut > 1)
            pm.peIn[6] = static_cast<int>(inIdx++);
          // Corner 5: cell(d-1, r, c-1) output[2]
          if (d > 0 && r < (int)peRows && c > 0 &&
              cells[d - 1][r][c - 1].numOut > 2)
            pm.peIn[5] = static_cast<int>(inIdx++);
          // Corner 4: cell(d, r-1, c-1) output[3]
          if (d < (int)peDepths && r > 0 && c > 0 &&
              cells[d][r - 1][c - 1].numOut > 3)
            pm.peIn[4] = static_cast<int>(inIdx++);
          // Corner 3: cell(d, r, c-1) output[4]
          if (d < (int)peDepths && r < (int)peRows && c > 0 &&
              cells[d][r][c - 1].numOut > 4)
            pm.peIn[3] = static_cast<int>(inIdx++);
          // Corner 2: cell(d, r-1, c) output[5]
          if (d < (int)peDepths && r > 0 && c < (int)peCols &&
              cells[d][r - 1][c].numOut > 5)
            pm.peIn[2] = static_cast<int>(inIdx++);
          // Corner 1: cell(d-1, r, c) output[6]
          if (d > 0 && r < (int)peRows && c < (int)peCols &&
              cells[d - 1][r][c].numOut > 6)
            pm.peIn[1] = static_cast<int>(inIdx++);
          // Corner 0: cell(d, r, c) output[7]
          if (d < (int)peDepths && r < (int)peRows && c < (int)peCols &&
              cells[d][r][c].numOut > 7)
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
    // Build 3D ADG
    //=================================================================

    // Create switch templates.
    std::map<std::pair<unsigned, unsigned>, SwitchHandle> swTemplates;
    for (int d = 0; d < swDepths; ++d) {
      for (int r = 0; r < swRows; ++r) {
        for (int c = 0; c < swCols; ++c) {
          auto &pm = swPorts[d][r][c];
          auto key = std::make_pair(pm.numIn, pm.numOut);
          if (swTemplates.find(key) == swTemplates.end()) {
            std::string name = "sw_w" + wp + "_" + std::to_string(pm.numIn) +
                               "x" + std::to_string(pm.numOut);
            swTemplates[key] = builder.newSwitch(name)
                                   .setPortCount(pm.numIn, pm.numOut)
                                   .setType(widthToNativeType(width));
          }
        }
      }
    }

    // Create switch instances.
    std::vector<std::vector<std::vector<InstanceHandle>>> swGrid(
        swDepths,
        std::vector<std::vector<InstanceHandle>>(
            swRows, std::vector<InstanceHandle>(swCols)));
    for (int d = 0; d < swDepths; ++d) {
      for (int r = 0; r < swRows; ++r) {
        for (int c = 0; c < swCols; ++c) {
          auto &pm = swPorts[d][r][c];
          auto key = std::make_pair(pm.numIn, pm.numOut);
          std::string instName = "sw_" + std::to_string(d) + "_" +
                                 std::to_string(r) + "_" + std::to_string(c);
          swGrid[d][r][c] = builder.clone(swTemplates[key], instName);
        }
      }
    }

    // Create FIFO templates.
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

    // Wire 6-direction inter-switch connections.
    for (int d = 0; d < swDepths; ++d) {
      for (int r = 0; r < swRows; ++r) {
        for (int c = 0; c < swCols; ++c) {
          auto &pm = swPorts[d][r][c];
          std::string pos = std::to_string(d) + "_" + std::to_string(r) +
                            "_" + std::to_string(c);

          // East: (d,r,c) -> (d,r,c+1)
          if (c + 1 < swCols) {
            auto &pmE = swPorts[d][r][c + 1];
            for (unsigned t = 0; t < T; ++t) {
              int so = pm.dirOut[1 * T + t], di = pmE.dirIn[3 * T + t];
              assert(so >= 0 && di >= 0);
              connectDir3D(swGrid[d][r][c], so, swGrid[d][r][c + 1], di,
                           useFwd, fwdFifo, "fifo_e_t" + std::to_string(t) + "_" + pos);
              int rso = pmE.dirOut[3 * T + t], rdi = pm.dirIn[1 * T + t];
              assert(rso >= 0 && rdi >= 0);
              connectDir3D(swGrid[d][r][c + 1], rso, swGrid[d][r][c], rdi,
                           useRev, revFifo, "fifo_w_t" + std::to_string(t) + "_" + pos);
            }
          }

          // South: (d,r,c) -> (d,r+1,c)
          if (r + 1 < swRows) {
            auto &pmS = swPorts[d][r + 1][c];
            for (unsigned t = 0; t < T; ++t) {
              int so = pm.dirOut[2 * T + t], di = pmS.dirIn[0 * T + t];
              assert(so >= 0 && di >= 0);
              connectDir3D(swGrid[d][r][c], so, swGrid[d][r + 1][c], di,
                           useFwd, fwdFifo, "fifo_s_t" + std::to_string(t) + "_" + pos);
              int rso = pmS.dirOut[0 * T + t], rdi = pm.dirIn[2 * T + t];
              assert(rso >= 0 && rdi >= 0);
              connectDir3D(swGrid[d][r + 1][c], rso, swGrid[d][r][c], rdi,
                           useRev, revFifo, "fifo_n_t" + std::to_string(t) + "_" + pos);
            }
          }

          // Down: (d,r,c) -> (d+1,r,c)
          if (d + 1 < swDepths) {
            auto &pmD = swPorts[d + 1][r][c];
            for (unsigned t = 0; t < T; ++t) {
              int so = pm.dirOut[5 * T + t], di = pmD.dirIn[4 * T + t];
              assert(so >= 0 && di >= 0);
              connectDir3D(swGrid[d][r][c], so, swGrid[d + 1][r][c], di,
                           useFwd, fwdFifo, "fifo_d_t" + std::to_string(t) + "_" + pos);
              int rso = pmD.dirOut[4 * T + t], rdi = pm.dirIn[5 * T + t];
              assert(rso >= 0 && rdi >= 0);
              connectDir3D(swGrid[d + 1][r][c], rso, swGrid[d][r][c], rdi,
                           useRev, revFifo, "fifo_u_t" + std::to_string(t) + "_" + pos);
            }
          }
        }
      }
    }

    // Helper: wire PE to 8-corner switches.
    auto wirePE3D = [&](InstanceHandle peInst, int pd, int pr, int pc,
                        unsigned nIn, unsigned nOut) {
      // Input connections in corner order 0..7.
      struct { int dd, dr, dc; unsigned corner; } inCorners[] = {
        {0, 0, 0, 0}, {1, 0, 0, 1}, {0, 1, 0, 2}, {0, 0, 1, 3},
        {0, 1, 1, 4}, {1, 0, 1, 5}, {1, 1, 0, 6}, {1, 1, 1, 7}
      };
      for (unsigned k = 0; k < nIn && k < 8; ++k) {
        auto &ic = inCorners[k];
        int sd = pd + ic.dd, sr = pr + ic.dr, sc = pc + ic.dc;
        int swOut = swPorts[sd][sr][sc].peOut[ic.corner];
        assert(swOut >= 0);
        builder.connectPorts(swGrid[sd][sr][sc], swOut, peInst,
                             static_cast<int>(k));
      }
      // Output connections in reverse corner order 7..0.
      struct { int dd, dr, dc; unsigned corner; } outCorners[] = {
        {1, 1, 1, 7}, {1, 1, 0, 6}, {1, 0, 1, 5}, {0, 1, 1, 4},
        {0, 0, 1, 3}, {0, 1, 0, 2}, {1, 0, 0, 1}, {0, 0, 0, 0}
      };
      for (unsigned k = 0; k < nOut && k < 8; ++k) {
        auto &oc = outCorners[k];
        int sd = pd + oc.dd, sr = pr + oc.dr, sc = pc + oc.dc;
        int swIn = swPorts[sd][sr][sc].peIn[oc.corner];
        assert(swIn >= 0);
        builder.connectPorts(peInst, static_cast<int>(k),
                             swGrid[sd][sr][sc], swIn);
      }
    };

    // Create PE instances and wire to corner switches.
    {
      unsigned cellIdx = 0;
      // Regular PEs.
      for (const auto &[spec, count] : peList) {
        PEHandle peHandle = createPEDef(builder, spec);
        for (unsigned i = 0; i < count; ++i) {
          while (cellIdx < totalCells) {
            unsigned cd = cellIdx / (peRows * peCols);
            unsigned rem = cellIdx % (peRows * peCols);
            unsigned cr = rem / peCols;
            unsigned cc = rem % peCols;
            if (!pruned[cd][cr][cc]) {
              std::string instName = spec.peName() + "_d" + std::to_string(cd) +
                                     "_r" + std::to_string(cr) +
                                     "_c" + std::to_string(cc);
              auto peInst = builder.clone(peHandle, instName);
              wirePE3D(peInst, cd, cr, cc, spec.inWidths.size(),
                       spec.outWidths.size());
              ++cellIdx;
              break;
            }
            ++cellIdx;
          }
        }
      }
      // LoadPEs.
      if (nLoad > 0) {
        LoadPEHandle loadDef =
            builder.newLoadPE("load_pe_w" + wp)
                .setDataType(widthToNativeType(width));
        for (unsigned i = 0; i < nLoad; ++i) {
          while (cellIdx < totalCells) {
            unsigned cd = cellIdx / (peRows * peCols);
            unsigned rem = cellIdx % (peRows * peCols);
            unsigned cr = rem / peCols;
            unsigned cc = rem % peCols;
            if (!pruned[cd][cr][cc]) {
              std::string instName = "load_pe_d" + std::to_string(cd) +
                                     "_r" + std::to_string(cr) +
                                     "_c" + std::to_string(cc);
              auto peInst = builder.clone(loadDef, instName);
              wirePE3D(peInst, cd, cr, cc, 3, 2);
              ++cellIdx;
              break;
            }
            ++cellIdx;
          }
        }
      }
      // StorePEs.
      if (nStore > 0) {
        StorePEHandle storeDef =
            builder.newStorePE("store_pe_w" + wp)
                .setDataType(widthToNativeType(width));
        for (unsigned i = 0; i < nStore; ++i) {
          while (cellIdx < totalCells) {
            unsigned cd = cellIdx / (peRows * peCols);
            unsigned rem = cellIdx % (peRows * peCols);
            unsigned cr = rem / peCols;
            unsigned cc = rem % peCols;
            if (!pruned[cd][cr][cc]) {
              std::string instName = "store_pe_d" + std::to_string(cd) +
                                     "_r" + std::to_string(cr) +
                                     "_c" + std::to_string(cc);
              auto peInst = builder.clone(storeDef, instName);
              wirePE3D(peInst, cd, cr, cc, 3, 2);
              ++cellIdx;
              break;
            }
            ++cellIdx;
          }
        }
      }
    }

    // Module I/O.
    {
      std::map<std::tuple<int, int, int>, unsigned> swModInIdx;
      for (unsigned k = 0; k < modInAssigns.size(); ++k) {
        auto &a = modInAssigns[k];
        auto port = builder.addModuleInput("in_" + std::to_string(k),
                                           widthToNativeType(width));
        unsigned localIdx = swModInIdx[{a.swD, a.swR, a.swC}]++;
        auto &pm = swPorts[a.swD][a.swR][a.swC];
        builder.connectToModuleInput(port, swGrid[a.swD][a.swR][a.swC],
                                     pm.modIn[localIdx]);
      }
    }
    {
      std::map<std::tuple<int, int, int>, unsigned> swModOutIdx;
      for (unsigned k = 0; k < modOutAssigns.size(); ++k) {
        auto &a = modOutAssigns[k];
        auto port = builder.addModuleOutput("out_" + std::to_string(k),
                                            widthToNativeType(width));
        unsigned localIdx = swModOutIdx[{a.swD, a.swR, a.swC}]++;
        auto &pm = swPorts[a.swD][a.swR][a.swC];
        builder.connectToModuleOutput(swGrid[a.swD][a.swR][a.swC],
                                      pm.modOut[localIdx], port);
      }
    }
  }
}

} // namespace adg
} // namespace loom
