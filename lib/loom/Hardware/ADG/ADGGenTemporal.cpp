//===-- ADGGenTemporal.cpp - Temporal domain for ADG generation --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Generates a dual-mesh temporal domain: a second lattice of temporal_sw
// with temporal_pe instances, bridged to the native mesh via add_tag and
// del_tag operations. Follows the pattern in mixed-temporal.fabric.mlir.
//
//===----------------------------------------------------------------------===//

#include "loom/Hardware/ADG/ADGGenHelpers.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace loom {
namespace adg {

void generateTemporal(ADGBuilder &builder, const MergedRequirements &reqs,
                      const GenConfig &config,
                      std::map<unsigned, WidthLattice> &lattices,
                      const std::map<unsigned, unsigned> &bridgeOutStartByWidth,
                      const std::map<unsigned, unsigned> &bridgeInStartByWidth) {
  if (reqs.peMaxCounts.empty())
    return;

  unsigned tagWidth = config.temporalTagWidth;
  unsigned numInstr = config.temporalNumInstruction;

  // When analysis-driven partitioning is available, use temporalPECounts
  // for temporal mesh sizing. Otherwise, fall back to duplicating all PEs.
  const auto &temporalCounts = config.temporalPECounts.empty()
                                   ? reqs.peMaxCounts
                                   : config.temporalPECounts;

  if (temporalCounts.empty())
    return;

  // Collect unique PE specs grouped by width plane.
  // Each width plane gets one temporal PE definition containing all
  // unique ops as FU sub-PEs.
  // Filter out cross-width ops (e.g. extui i32->i64) whose output exceeds
  // the width plane's interface type. These stay on the native mesh.
  std::map<unsigned, std::vector<PESpec>> specsByWidth;
  for (const auto &[spec, count] : temporalCounts) {
    if (spec.isCrossWidth())
      continue;
    unsigned w = spec.primaryWidth();
    specsByWidth[w].push_back(spec);
  }

  for (auto &[width, specs] : specsByWidth) {
    // Determine temporal PE count from partitioned requirements.
    unsigned temporalPECount = 0;
    for (const auto &[spec, count] : temporalCounts) {
      if (spec.primaryWidth() == width)
        temporalPECount += count;
    }
    if (temporalPECount == 0)
      continue;
    unsigned nativePECount = temporalPECount;

    std::string wp = std::to_string(width);
    Type tagType = Type::iN(tagWidth);
    Type valueType = Type::bits(width);
    Type taggedType = Type::tagged(valueType, tagType);

    // Build temporal PE definition with FU sub-PEs from DFG ops.
    // Deduplicate FU definitions by PESpec.
    std::set<std::string> fuNames;
    auto tpeBuilder = builder.newTemporalPE("tpe_w" + wp)
                          .setNumRegisters(0)
                          .setNumInstructions(numInstr)
                          .setRegFifoDepth(0)
                          .setInterface(taggedType);

    for (const auto &spec : specs) {
      std::string fuName = "fu_" + spec.peName();
      if (fuNames.count(fuName))
        continue;
      fuNames.insert(fuName);
      PEHandle fu = createPEDef(builder, spec);
      tpeBuilder.addFU(fu);
    }
    TemporalPEHandle tpeDef = tpeBuilder;

    // Compute temporal mesh dimensions (same as native mesh).
    auto tpeDims = computeMesh2DDims(nativePECount);
    unsigned tpeRows = tpeDims.first;
    unsigned tpeCols = tpeDims.second;
    int tswRows = static_cast<int>(tpeRows) + 1;
    int tswCols = static_cast<int>(tpeCols) + 1;

    // Create temporal switch templates.
    // Temporal switches use the tagged type.
    // Port counts mirror native mesh: each cell has 2-in 1-out TPEs.
    // Use the same port assignment scheme as native mesh.

    // For simplicity, build the temporal mesh using the latticeMesh approach:
    // first create switches, then wire inter-switch connections, then place TPEs.

    // Compute per-switch port maps (reuse 2D scheme).
    // Derive actual TPE port counts from max across all FU sub-PEs.
    unsigned tpeInCount = 1, tpeOutCount = 1;
    for (const auto &spec : specs) {
      tpeInCount = std::max(tpeInCount, (unsigned)spec.inWidths.size());
      tpeOutCount = std::max(tpeOutCount, (unsigned)spec.outWidths.size());
    }

    struct TCellInfo {
      unsigned numIn = 0, numOut = 0;
    };
    std::vector<std::vector<TCellInfo>> tCells(
        tpeRows, std::vector<TCellInfo>(tpeCols));
    {
      unsigned placed = 0;
      for (unsigned r = 0; r < tpeRows && placed < nativePECount; ++r) {
        for (unsigned c = 0; c < tpeCols && placed < nativePECount; ++c) {
          tCells[r][c] = {tpeInCount, tpeOutCount};
          ++placed;
        }
      }
    }

    // Compute temporal switch port maps.
    // Also reserve one cross-mesh input on leftmost column (from add_tag)
    // and one cross-mesh output on rightmost column (to del_tag).
    std::vector<std::vector<SwPortMap>> tswPorts(
        tswRows, std::vector<SwPortMap>(tswCols));

    for (int r = 0; r < tswRows; ++r) {
      for (int c = 0; c < tswCols; ++c) {
        auto &pm = tswPorts[r][c];
        pm.dirIn.assign(4, -1);
        pm.dirOut.assign(4, -1);
        unsigned inIdx = 0, outIdx = 0;

        bool hasDir[4] = {r > 0, c < tswCols - 1, r < tswRows - 1, c > 0};

        // Cross-mesh input: add_tag feeds leftmost column (c == 0).
        if (c == 0 && r < tswRows)
          pm.modIn.push_back(static_cast<int>(inIdx++));

        for (int dir = 0; dir < 4; ++dir) {
          if (hasDir[dir]) {
            pm.dirIn[dir] = static_cast<int>(inIdx++);
            pm.dirOut[dir] = static_cast<int>(outIdx++);
          }
        }

        // PE output ports (switch out -> TPE input).
        // Same corner scheme as native mesh: UL=0, LL=1, UR=2, LR=3.
        // Port localIdx maps to corner localIdx%4, with wrap-around.
        auto tpeInForCorner = [&](int corner) -> unsigned {
          int cr, cc;
          switch (corner) {
          case 0: cr = r;     cc = c;     break;
          case 1: cr = r - 1; cc = c;     break;
          case 2: cr = r;     cc = c - 1; break;
          case 3: cr = r - 1; cc = c - 1; break;
          default: return 0;
          }
          if (cr < 0 || cc < 0 || cr >= (int)tpeRows || cc >= (int)tpeCols)
            return 0;
          unsigned total = tCells[cr][cc].numIn;
          unsigned count = 0;
          for (unsigned i = corner; i < total; i += 4)
            count++;
          return count;
        };
        for (int corner = 0; corner < 4; ++corner) {
          unsigned n = tpeInForCorner(corner);
          if (n > 0) {
            pm.peOutCornerOff[corner] = pm.peOutVec.size();
            pm.peOutCornerCnt[corner] = n;
            for (unsigned k = 0; k < n; ++k)
              pm.peOutVec.push_back(static_cast<int>(outIdx++));
          }
        }

        // PE input ports (TPE output -> switch in).
        // Output corner reversal: wiring corner 0=LR, 1=UR, 2=LL, 3=UL.
        auto tpeOutForCorner = [&](int corner) -> unsigned {
          int cr, cc;
          switch (corner) {
          case 0: cr = r;     cc = c;     break;
          case 1: cr = r - 1; cc = c;     break;
          case 2: cr = r;     cc = c - 1; break;
          case 3: cr = r - 1; cc = c - 1; break;
          default: return 0;
          }
          if (cr < 0 || cc < 0 || cr >= (int)tpeRows || cc >= (int)tpeCols)
            return 0;
          unsigned total = tCells[cr][cc].numOut;
          unsigned count = 0;
          unsigned start = 3 - corner;
          for (unsigned i = start; i < total; i += 4)
            count++;
          return count;
        };
        for (int corner = 0; corner < 4; ++corner) {
          unsigned n = tpeOutForCorner(corner);
          if (n > 0) {
            pm.peInCornerOff[corner] = pm.peInVec.size();
            pm.peInCornerCnt[corner] = n;
            for (unsigned k = 0; k < n; ++k)
              pm.peInVec.push_back(static_cast<int>(inIdx++));
          }
        }

        // Cross-mesh output: rightmost column (c == tswCols - 1) -> del_tag.
        if (c == tswCols - 1 && r < tswRows)
          pm.modOut.push_back(static_cast<int>(outIdx++));

        pm.numIn = inIdx;
        pm.numOut = outIdx;
      }
    }

    // Create temporal switch templates.
    std::map<std::pair<unsigned, unsigned>, TemporalSwitchHandle> tswTemplates;
    for (int r = 0; r < tswRows; ++r) {
      for (int c = 0; c < tswCols; ++c) {
        auto &pm = tswPorts[r][c];
        auto key = std::make_pair(pm.numIn, pm.numOut);
        if (tswTemplates.find(key) == tswTemplates.end()) {
          std::string name = "tsw_w" + wp + "_" + std::to_string(pm.numIn) +
                             "x" + std::to_string(pm.numOut);
          tswTemplates[key] = builder.newTemporalSwitch(name)
                                  .setPortCount(pm.numIn, pm.numOut)
                                  .setInterface(taggedType);
        }
      }
    }

    // Create temporal switch instances.
    std::vector<std::vector<InstanceHandle>> tswGrid(
        tswRows, std::vector<InstanceHandle>(tswCols));
    for (int r = 0; r < tswRows; ++r) {
      for (int c = 0; c < tswCols; ++c) {
        auto &pm = tswPorts[r][c];
        auto key = std::make_pair(pm.numIn, pm.numOut);
        std::string instName =
            "tsw_" + std::to_string(r) + "_" + std::to_string(c);
        tswGrid[r][c] = builder.clone(tswTemplates[key], instName);
      }
    }

    // Create FIFOs for temporal mesh (reverse directions: W and N).
    FifoHandle tRevFifo{0};
    {
      auto fb = builder.newFifo("trev_fifo_w" + wp)
                    .setDepth(config.fifoDepth)
                    .setType(taggedType);
      if (config.fifoBypassable)
        fb.setBypassable(true);
      tRevFifo = fb;
    }

    // Wire temporal inter-switch connections.
    for (int r = 0; r < tswRows; ++r) {
      for (int c = 0; c < tswCols; ++c) {
        auto &pm = tswPorts[r][c];

        // East: direct
        if (c + 1 < tswCols) {
          auto &pmE = tswPorts[r][c + 1];
          int so = pm.dirOut[1], di = pmE.dirIn[3];
          if (so >= 0 && di >= 0)
            builder.connectPorts(tswGrid[r][c], so, tswGrid[r][c + 1], di);
          // West: via FIFO
          int rso = pmE.dirOut[3], rdi = pm.dirIn[1];
          if (rso >= 0 && rdi >= 0) {
            auto f = builder.clone(tRevFifo,
                "tfifo_w_" + std::to_string(r) + "_" + std::to_string(c));
            builder.connectPorts(tswGrid[r][c + 1], rso, f, 0);
            builder.connectPorts(f, 0, tswGrid[r][c], rdi);
          }
        }

        // South: direct
        if (r + 1 < tswRows) {
          auto &pmS = tswPorts[r + 1][c];
          int so = pm.dirOut[2], di = pmS.dirIn[0];
          if (so >= 0 && di >= 0)
            builder.connectPorts(tswGrid[r][c], so, tswGrid[r + 1][c], di);
          // North: via FIFO
          int rso = pmS.dirOut[0], rdi = pm.dirIn[2];
          if (rso >= 0 && rdi >= 0) {
            auto f = builder.clone(tRevFifo,
                "tfifo_n_" + std::to_string(r) + "_" + std::to_string(c));
            builder.connectPorts(tswGrid[r + 1][c], rso, f, 0);
            builder.connectPorts(f, 0, tswGrid[r][c], rdi);
          }
        }
      }
    }

    // Place temporal PE instances using same corner scheme as native mesh.
    {
      unsigned placed = 0;
      for (unsigned r = 0; r < tpeRows && placed < nativePECount; ++r) {
        for (unsigned c = 0; c < tpeCols && placed < nativePECount; ++c) {
          std::string instName =
              "tpe_r" + std::to_string(r) + "_c" + std::to_string(c);
          auto tpeInst = builder.clone(tpeDef, instName);

          // Wire inputs: localIdx -> corner (localIdx%4), wrapIdx (localIdx/4)
          // Input corners: UL=0, LL=1, UR=2, LR=3.
          for (unsigned p = 0; p < tpeInCount; ++p) {
            unsigned corner = p % 4;
            unsigned wrapIdx = p / 4;
            int swR, swC;
            switch (corner) {
            case 0: swR = r;     swC = c;     break;
            case 1: swR = r + 1; swC = c;     break;
            case 2: swR = r;     swC = c + 1; break;
            case 3: swR = r + 1; swC = c + 1; break;
            default: continue;
            }
            auto &pm = tswPorts[swR][swC];
            if (wrapIdx < pm.peOutCornerCnt[corner]) {
              unsigned idx = pm.peOutCornerOff[corner] + wrapIdx;
              int swOut = pm.peOutVec[idx];
              builder.connectPorts(tswGrid[swR][swC], swOut, tpeInst, p);
            }
          }

          // Wire outputs: localIdx -> reversed corner, wrapIdx
          // Output corners (reversed): 0=LR, 1=UR, 2=LL, 3=UL.
          for (unsigned p = 0; p < tpeOutCount; ++p) {
            unsigned corner = p % 4;
            unsigned wrapIdx = p / 4;
            int swR, swC;
            switch (corner) {
            case 0: swR = r + 1; swC = c + 1; break;
            case 1: swR = r;     swC = c + 1; break;
            case 2: swR = r + 1; swC = c;     break;
            case 3: swR = r;     swC = c;     break;
            default: continue;
            }
            auto &pm = tswPorts[swR][swC];
            unsigned allocCorner = 3 - corner;
            if (wrapIdx < pm.peInCornerCnt[allocCorner]) {
              unsigned idx = pm.peInCornerOff[allocCorner] + wrapIdx;
              int swIn = pm.peInVec[idx];
              builder.connectPorts(tpeInst, p, tswGrid[swR][swC], swIn);
            }
          }

          ++placed;
        }
      }
    }

    // Cross-mesh bridges: native mesh -> temporal mesh via add_tag.
    // add_tag on rightmost column of native mesh -> leftmost column of temporal.
    // Since we don't have access to the native mesh switch grid here, we create
    // add_tag instances connected to module I/O ports that the native mesh
    // can connect to.
    //
    // Bridge pattern (matching mixed-temporal reference):
    //   Native rightmost column -> add_tag -> temporal leftmost column
    //   Temporal rightmost column -> del_tag + FIFO -> native rightmost column

    // Create bridge FIFO for del_tag output.
    FifoHandle bridgeFifo{0};
    {
      auto fb = builder.newFifo("bridge_fifo_w" + wp)
                    .setDepth(config.fifoDepth)
                    .setType(valueType);
      if (config.fifoBypassable)
        fb.setBypassable(true);
      bridgeFifo = fb;
    }

    // add_tag bridges: feed temporal mesh leftmost column.
    for (int r = 0; r < tswRows; ++r) {
      auto &pm = tswPorts[r][0];
      if (pm.modIn.empty())
        continue;

      // Create add_tag: native bits -> tagged.
      std::string atName = "add_tag_r" + std::to_string(r) + "_w" + wp;
      InstanceHandle addTagInst =
          builder.newAddTag(atName)
              .setValueType(valueType)
              .setTagType(tagType);

      // Connect add_tag output to temporal switch leftmost column input.
      builder.connectPorts(addTagInst, 0, tswGrid[r][0], pm.modIn[0]);

      // Create a module input for the add_tag's native-side input.
      std::string portName = "bridge_in_r" + std::to_string(r) + "_w" + wp;
      auto bridgePort = builder.addModuleInput(portName, valueType);
      builder.connectToModuleInput(bridgePort, addTagInst, 0);
    }

    // del_tag bridges: temporal mesh rightmost column -> FIFO -> module output.
    for (int r = 0; r < tswRows; ++r) {
      auto &pm = tswPorts[r][tswCols - 1];
      if (pm.modOut.empty())
        continue;

      // Create del_tag: tagged -> native bits.
      std::string dtName = "del_tag_r" + std::to_string(r) + "_w" + wp;
      InstanceHandle delTagInst =
          builder.newDelTag(dtName)
              .setInputType(taggedType);

      // Connect temporal switch output to del_tag input.
      builder.connectPorts(tswGrid[r][tswCols - 1], pm.modOut[0],
                           delTagInst, 0);

      // del_tag output -> FIFO -> module output.
      std::string fifoName =
          "bridge_fifo_r" + std::to_string(r) + "_w" + wp;
      auto fifoInst = builder.clone(bridgeFifo, fifoName);
      builder.connectPorts(delTagInst, 0, fifoInst, 0);

      std::string portName = "bridge_out_r" + std::to_string(r) + "_w" + wp;
      auto bridgePort = builder.addModuleOutput(portName, valueType);
      builder.connectToModuleOutput(fifoInst, 0, bridgePort);
    }
  }
}

} // namespace adg
} // namespace loom
