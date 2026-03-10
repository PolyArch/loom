//===-- ADGGen.cpp - ADG Generation from DFG Requirements -------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Hardware/ADG/ADGGen.h"
#include "loom/Hardware/Common/FabricConstants.h"
#include "loom/adg.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <map>
#include <sstream>
#include <vector>

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
  size_t h = hasher(os.str());
  std::ostringstream hex;
  hex << std::hex << (h & 0xFFFFFFFF);
  return hex.str();
}

//===----------------------------------------------------------------------===//
// Type helpers
//===----------------------------------------------------------------------===//

static Type widthToNativeType(unsigned w) {
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

static bool isConversionOp(const std::string &opName) {
  return opName == "arith.extsi" || opName == "arith.extui" ||
         opName == "arith.extf" || opName == "arith.trunci" ||
         opName == "arith.truncf" || opName == "arith.sitofp" ||
         opName == "arith.uitofp" || opName == "arith.fptosi" ||
         opName == "arith.fptoui" || opName == "arith.bitcast" ||
         opName == "arith.index_cast" || opName == "arith.index_castui";
}

static std::string buildConversionBody(const PESpec &spec) {
  assert(spec.inWidths.size() == 1 && spec.outWidths.size() == 1);
  std::ostringstream os;
  os << "^bb0(%arg0: i" << spec.inWidths[0] << "):\n";
  os << "  %0 = " << spec.opName << " %arg0 : i" << spec.inWidths[0]
     << " to i" << spec.outWidths[0] << "\n";
  os << "  fabric.yield %0 : i" << spec.outWidths[0];
  return os.str();
}

//===----------------------------------------------------------------------===//
// Lattice mesh dimension computation
//===----------------------------------------------------------------------===//

static std::pair<unsigned, unsigned> computeMesh2DDims(unsigned peCount) {
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
/// Example 3x3: (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(2,1),(1,2),(2,2)
static std::vector<std::pair<int, int>> antiDiagSweep(int rows, int cols) {
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
// Per-switch port map (Phase 1 data structure)
//===----------------------------------------------------------------------===//

/// Tracks port index assignments for a switch at a given grid position.
/// Port indices are assigned consecutively: directional, PE, module I/O.
struct SwPortMap {
  unsigned numIn = 0;
  unsigned numOut = 0;

  // Directional port indices: dirIn[dir * T + track], -1 if absent.
  // dir: 0=N, 1=E, 2=S, 3=W
  std::vector<int> dirIn;
  std::vector<int> dirOut;

  // PE port indices by corner role: 0=UL, 1=LL, 2=UR, 3=LR.
  // peOut[k]: switch output port sending to PE input (at corner k).
  // peIn[k]: switch input port receiving PE output (at corner k).
  int peOut[4] = {-1, -1, -1, -1};
  int peIn[4] = {-1, -1, -1, -1};

  // Module I/O port indices on this switch (in assignment order).
  std::vector<int> modIn;
  std::vector<int> modOut;
};

//===----------------------------------------------------------------------===//
// ADGGen::generate  (collect-then-build approach)
//===----------------------------------------------------------------------===//

void ADGGen::generate(const MergedRequirements &reqs, const GenConfig &config,
                      const std::string &outputPath,
                      const std::string &moduleName) {
  ADGBuilder builder(moduleName);

  // Group PE specs by their primary width plane.
  std::map<unsigned, std::vector<std::pair<PESpec, unsigned>>> pesByWidth;
  for (const auto &[spec, count] : reqs.peMaxCounts) {
    unsigned w = spec.primaryWidth();
    pesByWidth[w].push_back({spec, count});
  }
  for (const auto &[w, _] : reqs.maxInputsByWidth)
    pesByWidth[w];
  for (const auto &[w, _] : reqs.maxOutputsByWidth)
    pesByWidth[w];

  for (auto &[width, peList] : pesByWidth) {
    unsigned totalPEs = 0;
    for (const auto &[spec, count] : peList)
      totalPEs += count;

    auto [peRows, peCols] = computeMesh2DDims(totalPEs);
    int swRows = static_cast<int>(peRows) + 1;
    int swCols = static_cast<int>(peCols) + 1;
    unsigned T = config.numSwitchTrack;

    //=================================================================
    // Phase 1: Collect topology specification
    //=================================================================

    // PE placement grid.
    struct CellInfo {
      const PESpec *spec = nullptr;
      unsigned numIn = 0, numOut = 0;
    };
    std::vector<std::vector<CellInfo>> cells(
        peRows, std::vector<CellInfo>(peCols));
    {
      unsigned cellIdx = 0;
      for (const auto &[spec, count] : peList) {
        for (unsigned i = 0; i < count; ++i) {
          unsigned r = cellIdx / peCols;
          unsigned c = cellIdx % peCols;
          if (r >= peRows)
            break;
          cells[r][c] = {&spec, (unsigned)spec.inWidths.size(),
                         (unsigned)spec.outWidths.size()};
          ++cellIdx;
        }
      }
    }

    // Module I/O counts for this width plane.
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

    // Anti-diagonal sweep for module I/O assignment.
    auto sweep = antiDiagSweep(swRows, swCols);

    struct ModIOAssign {
      int swR, swC;
    };

    // Assign module inputs to boundary switch ports (from top-left).
    std::vector<ModIOAssign> modInAssigns;
    {
      unsigned assigned = 0;
      for (auto [r, c] : sweep) {
        if (assigned >= numModIn)
          break;
        bool hasDir[4] = {r > 0, c < swCols - 1, r < swRows - 1, c > 0};
        for (int dir = 0; dir < 4 && assigned < numModIn; ++dir) {
          if (!hasDir[dir]) {
            for (unsigned t = 0; t < T && assigned < numModIn; ++t) {
              modInAssigns.push_back({r, c});
              ++assigned;
            }
          }
        }
      }
    }

    // Assign module outputs to boundary switch ports (from bottom-right).
    std::vector<ModIOAssign> modOutAssigns;
    {
      unsigned assigned = 0;
      for (int i = static_cast<int>(sweep.size()) - 1; i >= 0; --i) {
        if (assigned >= numModOut)
          break;
        auto [r, c] = sweep[i];
        bool hasDir[4] = {r > 0, c < swCols - 1, r < swRows - 1, c > 0};
        for (int dir = 3; dir >= 0 && assigned < numModOut; --dir) {
          if (!hasDir[dir]) {
            for (int t = static_cast<int>(T) - 1;
                 t >= 0 && assigned < numModOut; --t) {
              modOutAssigns.push_back({r, c});
              ++assigned;
            }
          }
        }
      }
    }

    // Count module I/O per switch for port map computation.
    std::map<std::pair<int, int>, unsigned> swModInCount, swModOutCount;
    for (auto &a : modInAssigns)
      swModInCount[{a.swR, a.swC}]++;
    for (auto &a : modOutAssigns)
      swModOutCount[{a.swR, a.swC}]++;

    // Compute per-switch port maps.
    std::vector<std::vector<SwPortMap>> swPorts(
        swRows, std::vector<SwPortMap>(swCols));

    for (int r = 0; r < swRows; ++r) {
      for (int c = 0; c < swCols; ++c) {
        auto &pm = swPorts[r][c];
        pm.dirIn.assign(4 * T, -1);
        pm.dirOut.assign(4 * T, -1);
        unsigned inIdx = 0, outIdx = 0;

        // Directional ports (only for existing neighbors).
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
        // Corner role UL of cell (r, c): feeds PE input[0].
        if (r < static_cast<int>(peRows) && c < static_cast<int>(peCols) &&
            cells[r][c].spec && cells[r][c].numIn > 0)
          pm.peOut[0] = static_cast<int>(outIdx++);
        // Corner role LL of cell (r-1, c): feeds PE input[1].
        if (r > 0 && c < static_cast<int>(peCols) && cells[r - 1][c].spec &&
            cells[r - 1][c].numIn > 1)
          pm.peOut[1] = static_cast<int>(outIdx++);
        // Corner role UR of cell (r, c-1): feeds PE input[2].
        if (r < static_cast<int>(peRows) && c > 0 && cells[r][c - 1].spec &&
            cells[r][c - 1].numIn > 2)
          pm.peOut[2] = static_cast<int>(outIdx++);
        // Corner role LR of cell (r-1, c-1): feeds PE input[3].
        if (r > 0 && c > 0 && cells[r - 1][c - 1].spec &&
            cells[r - 1][c - 1].numIn > 3)
          pm.peOut[3] = static_cast<int>(outIdx++);

        // PE input ports (PE output -> switch in).
        // wirePEToLattice connects PE outputs in reverse diagonal order:
        //   PE out[0] -> LR, out[1] -> UR, out[2] -> LL, out[3] -> UL
        // LR of cell (r-1, c-1): receives PE output[0].
        if (r > 0 && c > 0 && cells[r - 1][c - 1].spec &&
            cells[r - 1][c - 1].numOut > 0)
          pm.peIn[3] = static_cast<int>(inIdx++);
        // UR of cell (r, c-1): receives PE output[1].
        if (r < static_cast<int>(peRows) && c > 0 && cells[r][c - 1].spec &&
            cells[r][c - 1].numOut > 1)
          pm.peIn[2] = static_cast<int>(inIdx++);
        // LL of cell (r-1, c): receives PE output[2].
        if (r > 0 && c < static_cast<int>(peCols) && cells[r - 1][c].spec &&
            cells[r - 1][c].numOut > 2)
          pm.peIn[1] = static_cast<int>(inIdx++);
        // UL of cell (r, c): receives PE output[3].
        if (r < static_cast<int>(peRows) && c < static_cast<int>(peCols) &&
            cells[r][c].spec && cells[r][c].numOut > 3)
          pm.peIn[0] = static_cast<int>(inIdx++);

        // Module I/O ports.
        auto inIt = swModInCount.find({r, c});
        unsigned nModIn = inIt != swModInCount.end() ? inIt->second : 0;
        for (unsigned i = 0; i < nModIn; ++i)
          pm.modIn.push_back(static_cast<int>(inIdx++));

        auto outIt = swModOutCount.find({r, c});
        unsigned nModOut = outIt != swModOutCount.end() ? outIt->second : 0;
        for (unsigned i = 0; i < nModOut; ++i)
          pm.modOut.push_back(static_cast<int>(outIdx++));

        pm.numIn = inIdx;
        pm.numOut = outIdx;
      }
    }

    //=================================================================
    // Phase 2: Build ADG from collected specification
    //=================================================================

    // Create switch templates (one per unique port configuration).
    std::map<std::pair<unsigned, unsigned>, SwitchHandle> swTemplates;
    for (int r = 0; r < swRows; ++r) {
      for (int c = 0; c < swCols; ++c) {
        auto &pm = swPorts[r][c];
        auto key = std::make_pair(pm.numIn, pm.numOut);
        if (swTemplates.find(key) == swTemplates.end()) {
          std::string name = "sw_w" + std::to_string(width) + "_" +
                             std::to_string(pm.numIn) + "x" +
                             std::to_string(pm.numOut);
          swTemplates[key] = builder.newSwitch(name)
                                 .setPortCount(pm.numIn, pm.numOut)
                                 .setType(Type::bits(width));
        }
      }
    }

    // Create switch instances.
    std::vector<std::vector<InstanceHandle>> swGrid(
        swRows, std::vector<InstanceHandle>(swCols));
    for (int r = 0; r < swRows; ++r) {
      for (int c = 0; c < swCols; ++c) {
        auto &pm = swPorts[r][c];
        auto key = std::make_pair(pm.numIn, pm.numOut);
        std::string instName =
            "sw_" + std::to_string(r) + "_" + std::to_string(c);
        swGrid[r][c] = builder.clone(swTemplates[key], instName);
      }
    }

    // Create FIFO templates based on fifo mode.
    bool useFwd = (config.fifoMode == GenConfig::FifoDual);
    bool useRev = (config.fifoMode != GenConfig::FifoNone);

    FifoHandle fwdFifo{0}, revFifo{0};
    if (useFwd) {
      auto fb = builder.newFifo("fwd_fifo_w" + std::to_string(width))
                    .setDepth(config.fifoDepth)
                    .setType(Type::bits(width));
      if (config.fifoBypassable)
        fb.setBypassable(true);
      fwdFifo = fb;
    }
    if (useRev) {
      auto fb = builder.newFifo("rev_fifo_w" + std::to_string(width))
                    .setDepth(config.fifoDepth)
                    .setType(Type::bits(width));
      if (config.fifoBypassable)
        fb.setBypassable(true);
      revFifo = fb;
    }

    // Helper: connect src switch output to dst switch input,
    // optionally through a FIFO.
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

    // Wire inter-switch connections.
    for (int r = 0; r < swRows; ++r) {
      for (int c = 0; c < swCols; ++c) {
        auto &pm = swPorts[r][c];

        // East: sw[r][c] -> sw[r][c+1]
        if (c + 1 < swCols) {
          auto &pmE = swPorts[r][c + 1];
          for (unsigned t = 0; t < T; ++t) {
            int srcOut = pm.dirOut[1 * T + t]; // E out
            int dstIn = pmE.dirIn[3 * T + t];  // W in
            assert(srcOut >= 0 && dstIn >= 0);
            connectDir(swGrid[r][c], srcOut, swGrid[r][c + 1], dstIn, useFwd,
                       fwdFifo,
                       "fifo_e_t" + std::to_string(t) + "_" +
                           std::to_string(r) + "_" + std::to_string(c));

            // Reverse: sw[r][c+1] W out -> sw[r][c] E in
            int revSrcOut = pmE.dirOut[3 * T + t];
            int revDstIn = pm.dirIn[1 * T + t];
            assert(revSrcOut >= 0 && revDstIn >= 0);
            connectDir(swGrid[r][c + 1], revSrcOut, swGrid[r][c], revDstIn,
                       useRev, revFifo,
                       "fifo_w_t" + std::to_string(t) + "_" +
                           std::to_string(r) + "_" + std::to_string(c));
          }
        }

        // South: sw[r][c] -> sw[r+1][c]
        if (r + 1 < swRows) {
          auto &pmS = swPorts[r + 1][c];
          for (unsigned t = 0; t < T; ++t) {
            int srcOut = pm.dirOut[2 * T + t]; // S out
            int dstIn = pmS.dirIn[0 * T + t];  // N in
            assert(srcOut >= 0 && dstIn >= 0);
            connectDir(swGrid[r][c], srcOut, swGrid[r + 1][c], dstIn, useFwd,
                       fwdFifo,
                       "fifo_s_t" + std::to_string(t) + "_" +
                           std::to_string(r) + "_" + std::to_string(c));

            // Reverse: sw[r+1][c] N out -> sw[r][c] S in
            int revSrcOut = pmS.dirOut[0 * T + t];
            int revDstIn = pm.dirIn[2 * T + t];
            assert(revSrcOut >= 0 && revDstIn >= 0);
            connectDir(swGrid[r + 1][c], revSrcOut, swGrid[r][c], revDstIn,
                       useRev, revFifo,
                       "fifo_n_t" + std::to_string(t) + "_" +
                           std::to_string(r) + "_" + std::to_string(c));
          }
        }
      }
    }

    // Create PE definitions, instances, and wire to corner switches.
    {
      unsigned cellIdx = 0;
      for (const auto &[spec, count] : peList) {
        std::vector<Type> inTypes, outTypes;
        for (unsigned w : spec.inWidths)
          inTypes.push_back(widthToNativeType(w));
        for (unsigned w : spec.outWidths)
          outTypes.push_back(widthToNativeType(w));

        auto peBuilder = builder.newPE(spec.peName())
                             .setLatency(1, 1, 1)
                             .setInterval(1, 1, 1)
                             .setInputPorts(inTypes)
                             .setOutputPorts(outTypes);
        if (isConversionOp(spec.opName))
          peBuilder.setBodyMLIR(buildConversionBody(spec));
        else
          peBuilder.addOp(spec.opName);
        PEHandle peHandle = peBuilder;

        for (unsigned i = 0; i < count; ++i) {
          int pr = static_cast<int>(cellIdx / peCols);
          int pc = static_cast<int>(cellIdx % peCols);
          if (pr >= static_cast<int>(peRows))
            break;

          std::string instName = spec.peName() + "_r" + std::to_string(pr) +
                                 "_c" + std::to_string(pc);
          InstanceHandle peInst = builder.clone(peHandle, instName);

          unsigned nIn = spec.inWidths.size();
          unsigned nOut = spec.outWidths.size();

          // PE input[0] <- UL switch = sw[pr][pc]
          if (nIn > 0) {
            int swOut = swPorts[pr][pc].peOut[0];
            assert(swOut >= 0);
            builder.connectPorts(swGrid[pr][pc], swOut, peInst, 0);
          }
          // PE input[1] <- LL switch = sw[pr+1][pc]
          if (nIn > 1) {
            int swOut = swPorts[pr + 1][pc].peOut[1];
            assert(swOut >= 0);
            builder.connectPorts(swGrid[pr + 1][pc], swOut, peInst, 1);
          }
          // PE input[2] <- UR switch = sw[pr][pc+1]
          if (nIn > 2) {
            int swOut = swPorts[pr][pc + 1].peOut[2];
            assert(swOut >= 0);
            builder.connectPorts(swGrid[pr][pc + 1], swOut, peInst, 2);
          }
          // PE input[3] <- LR switch = sw[pr+1][pc+1]
          if (nIn > 3) {
            int swOut = swPorts[pr + 1][pc + 1].peOut[3];
            assert(swOut >= 0);
            builder.connectPorts(swGrid[pr + 1][pc + 1], swOut, peInst, 3);
          }

          // PE output[0] -> LR switch = sw[pr+1][pc+1]
          if (nOut > 0) {
            int swIn = swPorts[pr + 1][pc + 1].peIn[3];
            assert(swIn >= 0);
            builder.connectPorts(peInst, 0, swGrid[pr + 1][pc + 1], swIn);
          }
          // PE output[1] -> UR switch = sw[pr][pc+1]
          if (nOut > 1) {
            int swIn = swPorts[pr][pc + 1].peIn[2];
            assert(swIn >= 0);
            builder.connectPorts(peInst, 1, swGrid[pr][pc + 1], swIn);
          }
          // PE output[2] -> LL switch = sw[pr+1][pc]
          if (nOut > 2) {
            int swIn = swPorts[pr + 1][pc].peIn[1];
            assert(swIn >= 0);
            builder.connectPorts(peInst, 2, swGrid[pr + 1][pc], swIn);
          }
          // PE output[3] -> UL switch = sw[pr][pc]
          if (nOut > 3) {
            int swIn = swPorts[pr][pc].peIn[0];
            assert(swIn >= 0);
            builder.connectPorts(peInst, 3, swGrid[pr][pc], swIn);
          }

          ++cellIdx;
        }
      }
    }

    // Create module I/O ports and connect to switches.
    {
      std::map<std::pair<int, int>, unsigned> swModInIdx;
      for (unsigned k = 0; k < modInAssigns.size(); ++k) {
        auto &a = modInAssigns[k];
        std::string name = "in_" + std::to_string(k);
        auto port = builder.addModuleInput(name, Type::bits(width));
        auto &pm = swPorts[a.swR][a.swC];
        unsigned localIdx = swModInIdx[{a.swR, a.swC}]++;
        builder.connectToModuleInput(port, swGrid[a.swR][a.swC],
                                     pm.modIn[localIdx]);
      }
    }
    {
      std::map<std::pair<int, int>, unsigned> swModOutIdx;
      for (unsigned k = 0; k < modOutAssigns.size(); ++k) {
        auto &a = modOutAssigns[k];
        std::string name = "out_" + std::to_string(k);
        auto port = builder.addModuleOutput(name, Type::bits(width));
        auto &pm = swPorts[a.swR][a.swC];
        unsigned localIdx = swModOutIdx[{a.swR, a.swC}]++;
        builder.connectToModuleOutput(swGrid[a.swR][a.swC],
                                      pm.modOut[localIdx], port);
      }
    }
  }

  builder.exportMLIR(outputPath);
}

} // namespace adg
} // namespace loom
